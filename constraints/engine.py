# elusight/constraints/engine.py
"""Constraint evaluation engine for AQbD-style constraints."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

from elusight.schemas.base import MethodResult, ConstraintType


@dataclass
class ConstraintResult:
    """Result of a single constraint evaluation."""
    name: str
    passed: bool
    actual_value: float
    constraint_value: float
    constraint_type: ConstraintType
    margin: float
    margin_percentage: float
    risk_level: str  # 'low', 'medium', 'high'


@dataclass
class ConstraintReport:
    """Complete constraint evaluation report."""
    overall_pass: bool
    constraints: List[ConstraintResult]
    failed_constraints: List[ConstraintResult]
    passed_constraints: List[ConstraintResult]
    constraint_satisfaction_rate: float
    worst_margin: float
    worst_margin_percentage: float
    critical_constraints: List[str]
    aqbd_design_space_status: str  # 'within', 'edge', 'outside'


class ConstraintEngine:
    """
    Evaluates chromatographic constraints following AQbD principles.
    Implements margin analysis and risk-based constraint assessment.
    """
    
    def __init__(self, constraints: Dict[str, Dict[str, Any]]):
        """
        Initialize constraint engine.
        
        Args:
            constraints: Dictionary of constraints in format:
                {
                    "resolution": {"type": "minimum", "value": 2.0, "critical": True},
                    "runtime": {"type": "maximum", "value": 20, "critical": False},
                    "pressure": {"type": "range", "min": 0, "max": 400}
                }
        """
        self.constraints = constraints
    
    def evaluate(self, method: MethodResult) -> ConstraintReport:
        """Evaluate all constraints for a given method."""
        constraint_results = []
        
        for name, constraint in self.constraints.items():
            actual_value = self._get_objective_value(method, name)
            if actual_value is None:
                continue
            
            result = self._evaluate_single_constraint(
                name, actual_value, constraint
            )
            constraint_results.append(result)
        
        failed = [r for r in constraint_results if not r.passed]
        passed = [r for r in constraint_results if r.passed]
        
        margins = [r.margin_percentage for r in constraint_results]
        worst_margin = min(margins) if margins else float('inf')
        
        # Determine AQbD design space status
        design_space_status = self._determine_design_space_status(
            constraint_results, margins
        )
        
        return ConstraintReport(
            overall_pass=len(failed) == 0,
            constraints=constraint_results,
            failed_constraints=failed,
            passed_constraints=passed,
            constraint_satisfaction_rate=len(passed) / len(constraint_results) if constraint_results else 1.0,
            worst_margin=min([r.margin for r in constraint_results]) if constraint_results else float('inf'),
            worst_margin_percentage=worst_margin,
            critical_constraints=[r.name for r in failed if self._is_critical(r.name)],
            aqbd_design_space_status=design_space_status
        )
    
    def _get_objective_value(self, method: MethodResult, name: str) -> Optional[float]:
        """Extract objective value from method result."""
        # Try direct attribute
        if hasattr(method.objectives, name):
            return getattr(method.objectives, name)
        
        # Try dictionary access
        obj_dict = method.objectives.dict()
        if name in obj_dict:
            return obj_dict[name]
        
        # Try case-insensitive
        for key in obj_dict:
            if key.lower() == name.lower():
                return obj_dict[key]
        
        return None
    
    def _evaluate_single_constraint(
        self, name: str, actual: float, constraint: Dict[str, Any]
    ) -> ConstraintResult:
        """Evaluate a single constraint."""
        constraint_type = ConstraintType(constraint['type'])
        
        if constraint_type == ConstraintType.MINIMUM:
            target = constraint['value']
            passed = actual >= target
            margin = actual - target
            margin_pct = (margin / target) * 100 if target != 0 else float('inf')
            constraint_value = target
            
        elif constraint_type == ConstraintType.MAXIMUM:
            target = constraint['value']
            passed = actual <= target
            margin = target - actual
            margin_pct = (margin / target) * 100 if target != 0 else float('inf')
            constraint_value = target
            
        elif constraint_type == ConstraintType.RANGE:
            min_val = constraint['min']
            max_val = constraint['max']
            passed = min_val <= actual <= max_val
            margin = min(actual - min_val, max_val - actual)
            margin_pct = (margin / ((max_val - min_val) / 2)) * 100 if (max_val - min_val) > 0 else float('inf')
            constraint_value = f"{min_val}-{max_val}"
            
        else:  # EQUALITY
            target = constraint['value']
            tolerance = constraint.get('tolerance', 0.01)
            passed = abs(actual - target) <= tolerance
            margin = tolerance - abs(actual - target)
            margin_pct = (margin / tolerance) * 100 if tolerance > 0 else float('inf')
            constraint_value = target
        
        # Determine risk level
        if margin_pct >= 20:
            risk_level = 'low'
        elif margin_pct >= 10:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return ConstraintResult(
            name=name,
            passed=passed,
            actual_value=actual,
            constraint_value=constraint_value,
            constraint_type=constraint_type,
            margin=margin,
            margin_percentage=margin_pct,
            risk_level=risk_level
        )
    
    def _is_critical(self, constraint_name: str) -> bool:
        """Check if a constraint is critical."""
        if constraint_name in self.constraints:
            return self.constraints[constraint_name].get('critical', False)
        return False
    
    def _determine_design_space_status(
        self, results: List[ConstraintResult], margins: List[float]
    ) -> str:
        """Determine AQbD design space status."""
        if not results:
            return 'unknown'
        
        # All constraints passed with >10% margin
        if all(r.passed and r.margin_percentage >= 10 for r in results):
            return 'within'
        # All constraints passed but some margins <10%
        elif all(r.passed for r in results):
            return 'edge'
        else:
            return 'outside'