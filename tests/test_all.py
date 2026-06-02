"""
Comprehensive tests for EluSight.
"""

import pytest
import json
from pathlib import Path

# Import all modules directly (no 'elusight.' prefix)
from schemas.base import MethodResult, MethodVariables, MethodObjectives
from ingest import Ingestor
from constraints.engine import ConstraintEngine
from trust.engine import TrustEngine
from reasoning.engine import ReasoningEngine
from uncertainty.engine import UncertaintyEngine
from robustness.engine import RobustnessEngine
from risk.engine import RiskEngine
from tradeoffs.engine import TradeoffEngine
from preferences.engine import PreferenceLearningEngine
from explainability.engine import ExplainabilityEngine


class TestEluSight:
    """Test suite for EluSight library."""
    
    @pytest.fixture
    def sample_methods(self):
        """Create sample method data for testing."""
        return [
            {
                'method_id': 'M001',
                'variables': {'pH': 3.2, 'gradient_time': 18.0, 'temperature': 40.0},
                'objectives': {'resolution': 2.5, 'runtime': 12.3, 'robustness': 95.0}
            },
            {
                'method_id': 'M002',
                'variables': {'pH': 3.5, 'gradient_time': 15.0, 'temperature': 45.0},
                'objectives': {'resolution': 2.8, 'runtime': 10.5, 'robustness': 92.0}
            },
            {
                'method_id': 'M003',
                'variables': {'pH': 3.0, 'gradient_time': 22.0, 'temperature': 35.0},
                'objectives': {'resolution': 2.2, 'runtime': 15.0, 'robustness': 98.0}
            }
        ]
    
    @pytest.fixture
    def sample_constraints(self):
        """Create sample constraints for testing."""
        return {
            'resolution': {'type': 'minimum', 'value': 2.0, 'critical': True},
            'runtime': {'type': 'maximum', 'value': 20, 'critical': False},
            'robustness': {'type': 'minimum', 'value': 90, 'critical': True}
        }
    
    def test_imports(self):
        """Test that all modules import correctly."""
        assert Ingestor is not None
        assert ConstraintEngine is not None
        assert TrustEngine is not None
        assert ReasoningEngine is not None
        print("✅ All imports successful!")
    
    def test_method_creation(self):
        """Test creating a MethodResult object."""
        method = MethodResult(
            method_id="TEST_001",
            variables=MethodVariables(pH=3.2, gradient_time=18.0),
            objectives=MethodObjectives(resolution=2.5, runtime=12.3)
        )
        assert method.method_id == "TEST_001"
        assert method.variables.pH == 3.2
        assert method.objectives.resolution == 2.5
        print("✅ Method creation successful!")
    
    def test_constraint_evaluation(self, sample_methods, sample_constraints):
        """Test constraint engine."""
        engine = ConstraintEngine(sample_constraints)
        
        method = MethodResult(
            method_id=sample_methods[0]['method_id'],
            variables=MethodVariables(**sample_methods[0]['variables']),
            objectives=MethodObjectives(**sample_methods[0]['objectives'])
        )
        
        report = engine.evaluate(method)
        
        assert report.overall_pass == True
        assert len(report.constraints) == 3
        assert report.constraint_satisfaction_rate == 1.0
        print("✅ Constraint evaluation successful!")
    
    def test_trust_engine(self):
        """Test trust engine."""
        class MockReport:
            overall_pass = True
            constraints = []
            constraint_satisfaction_rate = 1.0
            overall_robustness_score = 0.95
            overall_confidence_score = 0.92
            
            class method_robustness:
                robustness_score = 0.95
                pass_probability = 0.97
            
            class risk_metrics:
                overall_risk_score = 0.04
        
        engine = TrustEngine()
        trust = engine.compute_trust(
            method_id='TEST_001',
            constraint_report=MockReport(),
            robustness_report=MockReport(),
            confidence_report=MockReport(),
            risk_report=MockReport(),
            preference_scores=[]
        )
        
        assert 0 <= trust.overall_score <= 100
        assert trust.recommendation in ["Strongly Recommend", "Recommend", "Consider", "Avoid"]
        print("✅ Trust engine successful!")
    
    def test_reasoning_engine(self, sample_methods):
        """Test reasoning engine."""
        engine = ReasoningEngine()
        
        reasoning = engine.generate_reasoning(
            method_id='TEST_001',
            method_data=sample_methods[0],
            constraint_report=None,
            robustness_report=None,
            confidence_report=None,
            risk_report=None
        )
        
        assert reasoning.conclusion is not None
        assert len(reasoning.reasoning_steps) > 0
        print("✅ Reasoning engine successful!")


def run_all_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
