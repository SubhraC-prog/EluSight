from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class RiskMetrics:
    """Individual risk metrics."""
    coelution_probability: float
    sst_failure_probability: float
    constraint_violation_probability: float
    robustness_failure_probability: float
    overall_risk_score: float


@dataclass
class RiskReport:
    """Complete risk assessment report."""
    risk_metrics: RiskMetrics
    risk_breakdown: Dict[str, float]
    high_risk_factors: List[str]
    medium_risk_factors: List[str]
    low_risk_factors: List[str]
    mitigation_strategies: Dict[str, List[str]]
    risk_acceptability: str


class RiskEngine:
    """Quantifies scientific and operational risks."""
    
    def __init__(self, risk_tolerance: float = 0.05):
        self.risk_tolerance = risk_tolerance
    
    def assess_risk(
        self,
        method: Dict[str, Any],
        uncertainty_data: Optional[Dict[str, Any]] = None
    ) -> RiskReport:
        """Comprehensive risk assessment for a chromatographic method."""
        
        # Calculate individual risks
        resolution = method.get('objectives', {}).get('resolution', 2.0)
        coelution_risk = self._calculate_coelution_risk(resolution)
        sst_risk = self._calculate_sst_risk(method)
        constraint_risk = self._calculate_constraint_risk(method, uncertainty_data)
        robustness_risk = self._calculate_robustness_risk(method)
        
        # Aggregate risks
        risk_metrics = RiskMetrics(
            coelution_probability=coelution_risk,
            sst_failure_probability=sst_risk,
            constraint_violation_probability=constraint_risk,
            robustness_failure_probability=robustness_risk,
            overall_risk_score=np.mean([
                coelution_risk, sst_risk, constraint_risk, robustness_risk
            ])
        )
        
        # Break down risks
        risk_breakdown = {
            'coelution': coelution_risk,
            'sst_failure': sst_risk,
            'constraint_violation': constraint_risk,
            'robustness_failure': robustness_risk
        }
        
        # Categorize risk factors
        high_risks = [k for k, v in risk_breakdown.items() if v > 0.2]
        medium_risks = [k for k, v in risk_breakdown.items() if 0.05 < v <= 0.2]
        low_risks = [k for k, v in risk_breakdown.items() if v <= 0.05]
        
        # Generate mitigation strategies
        mitigation = self._generate_mitigation_strategies(high_risks)
        
        # Determine acceptability
        if risk_metrics.overall_risk_score <= self.risk_tolerance:
            acceptability = "acceptable"
        elif risk_metrics.overall_risk_score <= 3 * self.risk_tolerance:
            acceptability = "tolerable"
        else:
            acceptability = "unacceptable"
        
        return RiskReport(
            risk_metrics=risk_metrics,
            risk_breakdown=risk_breakdown,
            high_risk_factors=high_risks,
            medium_risk_factors=medium_risks,
            low_risk_factors=low_risks,
            mitigation_strategies=mitigation,
            risk_acceptability=acceptability
        )
    
    def _calculate_coelution_risk(self, resolution: float) -> float:
        """Calculate probability of co-elution based on resolution."""
        if resolution >= 2.0:
            return 0.01
        elif resolution >= 1.5:
            return 0.05
        elif resolution >= 1.0:
            return 0.15
        else:
            return 0.50
    
    def _calculate_sst_risk(self, method: Dict[str, Any]) -> float:
        """Calculate probability of system suitability test failure."""
        risk_factors = []
        
        resolution = method.get('objectives', {}).get('resolution', 2.0)
        if resolution < 1.5:
            risk_factors.append(0.3)
        elif resolution < 2.0:
            risk_factors.append(0.1)
        
        tailing = method.get('metadata', {}).get('tailing_factor', 1.2)
        if tailing > 2.0:
            risk_factors.append(0.2)
        elif tailing > 1.5:
            risk_factors.append(0.1)
        
        return min(0.5, sum(risk_factors)) if risk_factors else 0.03
    
    def _calculate_constraint_risk(
        self,
        method: Dict[str, Any],
        uncertainty_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate probability of constraint violation."""
        if uncertainty_data is None:
            return 0.08
        
        violation_probs = []
        for constraint, value in uncertainty_data.items():
            if isinstance(value, dict) and 'margin' in value:
                if value['margin'] < 0.05:
                    violation_probs.append(0.3)
                elif value['margin'] < 0.1:
                    violation_probs.append(0.15)
        
        return np.mean(violation_probs) if violation_probs else 0.05
    
    def _calculate_robustness_risk(self, method: Dict[str, Any]) -> float:
        """Calculate robustness failure probability."""
        robustness = method.get('objectives', {}).get('robustness', 90.0)
        
        if robustness >= 98:
            return 0.02
        elif robustness >= 95:
            return 0.05
        elif robustness >= 90:
            return 0.10
        else:
            return 0.20
    
    def _generate_mitigation_strategies(
        self, high_risks: List[str]
    ) -> Dict[str, List[str]]:
        """Generate risk mitigation strategies."""
        strategies = {
            'coelution': [
                "Increase gradient slope to improve resolution",
                "Change stationary phase selectivity",
                "Optimize mobile phase pH"
            ],
            'sst_failure': [
                "Add system suitability test with acceptance criteria",
                "Validate column efficiency before use",
                "Include reference standard injections"
            ],
            'constraint_violation': [
                "Add safety margins to critical constraints",
                "Implement real-time monitoring",
                "Use design space approach"
            ],
            'robustness_failure': [
                "Perform design of experiments for robustness",
                "Identify critical process parameters",
                "Implement multivariate control"
            ]
        }
        
        return {risk: strategies.get(risk, ["Review method parameters"]) for risk in high_risks}
EOF
