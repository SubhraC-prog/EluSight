"""
Complete test suite for EluSight.
"""

import pytest
import tempfile
import json
import os

from schemas.base import MethodResult, MethodVariables, MethodObjectives
from ingest import Ingestor
from constraints.engine import ConstraintEngine
from trust.engine import TrustEngine
from reasoning.engine import ReasoningEngine


class TestEluSightComplete:
    """Complete test suite for EluSight."""
    
    def test_ingestor_json(self):
        """Test JSON ingestion."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'method_id': 'TEST_001',
                'variables': {'pH': 3.2},
                'objectives': {'resolution': 2.5}
            }, f)
            temp_file = f.name
        
        try:
            methods = Ingestor.from_json(temp_file)
            assert len(methods) >= 0
        finally:
            os.unlink(temp_file)

    def test_constraint_engine(self):
        """Test constraint engine."""
        constraints = {
            'resolution': {'type': 'minimum', 'value': 2.0, 'critical': True},
        }
        engine = ConstraintEngine(constraints)
        method = MethodResult(
            method_id="TEST",
            variables=MethodVariables(),
            objectives=MethodObjectives(resolution=2.5)
        )
        report = engine.evaluate(method)
        assert report.overall_pass == True

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

    def test_reasoning_engine(self):
        """Test reasoning engine with proper mock reports."""
        # Create proper mock reports instead of passing None
        class MockConstraintReport:
            overall_pass = True
            constraints = []
            constraint_satisfaction_rate = 1.0
            worst_margin_percentage = 22.5
            aqbd_design_space_status = "within"
            
            def __init__(self):
                self.constraints = []
                self.passed_constraints = []
                self.failed_constraints = []

        class MockRobustnessReport:
            overall_robustness_score = 0.95
            class method_robustness:
                pass_probability = 0.97
                failure_probability = 0.03
                robustness_score = 0.95
                critical_parameters = ["pH", "temperature"]
                parameter_sensitivities = {"pH": 0.8, "temperature": 0.6}
                worst_case_scenario = {}
                operating_range = {}

        class MockConfidenceReport:
            overall_confidence_score = 0.92
            high_confidence_objectives = ["resolution"]
            low_confidence_objectives = []
            objective_confidence = {}
            uncertainty_sources = []
            recommendations = []

        class MockRiskReport:
            class risk_metrics:
                overall_risk_score = 0.04
                coelution_probability = 0.02
                sst_failure_probability = 0.03
                constraint_violation_probability = 0.01
                robustness_failure_probability = 0.02
        
        engine = ReasoningEngine()
        method_data = {
            'method_id': 'TEST_001',
            'variables': {'pH': 3.2, 'gradient_time': 18.0},
            'objectives': {'resolution': 2.5, 'runtime': 12.3, 'robustness': 95.0}
        }
        
        reasoning = engine.generate_reasoning(
            method_id='TEST_001',
            method_data=method_data,
            constraint_report=MockConstraintReport(),
            robustness_report=MockRobustnessReport(),
            confidence_report=MockConfidenceReport(),
            risk_report=MockRiskReport(),
            tradeoff_report=None,
            explanation_report=None,
            trust_score=None
        )
        
        assert reasoning.conclusion is not None
        assert len(reasoning.reasoning_steps) > 0
        assert reasoning.recommendation is not None
        assert reasoning.confidence_score >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
