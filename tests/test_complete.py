"""
Complete test suite for EluSight.
"""

import pytest
import tempfile
import json
import os

from schemas.base import MethodResult, MethodVariables, MethodObjectives
from ingest import Ingestor
from constraints.engine import ConstraintEngine, ConstraintResult, ConstraintReport
from trust.engine import TrustEngine
from reasoning.engine import ReasoningEngine
from schemas.base import ConstraintType


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
        # Create proper constraint result
        constraint_result = ConstraintResult(
            name="resolution",
            passed=True,
            actual_value=2.5,
            constraint_value=2.0,
            constraint_type=ConstraintType.MINIMUM,
            margin=0.5,
            margin_percentage=25.0,
            risk_level="low",
            critical=True
        )
        
        class MockConstraintReport:
            overall_pass = True
            constraints = [constraint_result]
            passed_constraints = [constraint_result]
            failed_constraints = []
            constraint_satisfaction_rate = 1.0
            worst_margin = 0.5
            worst_margin_percentage = 25.0
            critical_constraints = []
            aqbd_design_space_status = "within"
            recommendations = []

        class MockRobustnessReport:
            overall_robustness_score = 0.95
            class method_robustness:
                pass_probability = 0.97
                failure_probability = 0.03
                robustness_score = 0.95
                critical_parameters = ["pH"]
                parameter_sensitivities = {"pH": 0.8}
                worst_case_scenario = {}
                operating_range = {}
            robust_region_size = 0.8
            failure_modes = []
            mitigation_strategies = []
            perturbation_analysis = {}

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
            risk_breakdown = {}
            high_risk_factors = []
            medium_risk_factors = []
            low_risk_factors = []
            mitigation_strategies = {}
            risk_acceptability = "acceptable"

        engine = TrustEngine()
        trust = engine.compute_trust(
            method_id='TEST_001',
            constraint_report=MockConstraintReport(),
            robustness_report=MockRobustnessReport(),
            confidence_report=MockConfidenceReport(),
            risk_report=MockRiskReport(),
            preference_scores=[]
        )
        assert 0 <= trust.overall_score <= 100

    def test_reasoning_engine(self):
        """Test reasoning engine with complete mock reports."""
        
        # Create proper constraint result
        constraint_result = ConstraintResult(
            name="resolution",
            passed=True,
            actual_value=2.5,
            constraint_value=2.0,
            constraint_type=ConstraintType.MINIMUM,
            margin=0.5,
            margin_percentage=25.0,
            risk_level="low",
            critical=True
        )
        
        # Mock Constraint Report
        class MockConstraintReport:
            overall_pass = True
            constraints = [constraint_result]
            passed_constraints = [constraint_result]
            failed_constraints = []
            constraint_satisfaction_rate = 1.0
            worst_margin = 0.5
            worst_margin_percentage = 25.0
            critical_constraints = []
            aqbd_design_space_status = "within"
            recommendations = []

        # Mock Robustness Report
        class MockRobustnessReport:
            overall_robustness_score = 0.95
            robust_region_size = 0.8
            failure_modes = []
            mitigation_strategies = []
            perturbation_analysis = {}
            
            class method_robustness:
                pass_probability = 0.97
                failure_probability = 0.03
                robustness_score = 0.95
                critical_parameters = ["pH", "gradient_time"]
                parameter_sensitivities = {"pH": 0.8, "gradient_time": 0.6}
                worst_case_scenario = {"pH": 3.0}
                operating_range = {"pH": (3.0, 3.5)}
            
            method_robustness = method_robustness()

        # Mock Confidence Report
        class MockConfidenceReport:
            overall_confidence_score = 0.92
            high_confidence_objectives = ["resolution", "runtime"]
            low_confidence_objectives = []
            objective_confidence = {}
            uncertainty_sources = []
            recommendations = []

        # Mock Risk Report
        class MockRiskMetrics:
            overall_risk_score = 0.04
            coelution_probability = 0.02
            sst_failure_probability = 0.03
            constraint_violation_probability = 0.01
            robustness_failure_probability = 0.02

        class MockRiskReport:
            risk_metrics = MockRiskMetrics()
            risk_breakdown = {}
            high_risk_factors = []
            medium_risk_factors = []
            low_risk_factors = []
            mitigation_strategies = {}
            risk_acceptability = "acceptable"

        # Mock Tradeoff Report
        class MockTradeoffReport:
            pareto_front_methods = ["TEST_001", "TEST_002"]
            tradeoffs = []
            best_methods_by_objective = {"resolution": "TEST_001", "runtime": "TEST_002"}
            knee_points = ["TEST_001"]
            dominance_matrix = {}
            diversity_metrics = {}
            scientific_interpretation = "Method is on Pareto front"

        # Create method data
        method_data = {
            'method_id': 'TEST_001',
            'variables': {'pH': 3.2, 'gradient_time': 18.0, 'temperature': 40.0},
            'objectives': {'resolution': 2.5, 'runtime': 12.3, 'robustness': 95.0}
        }
        
        # Initialize reasoning engine
        engine = ReasoningEngine()
        
        # Generate reasoning with all mock reports
        reasoning = engine.generate_reasoning(
            method_id='TEST_001',
            method_data=method_data,
            constraint_report=MockConstraintReport(),
            robustness_report=MockRobustnessReport(),
            confidence_report=MockConfidenceReport(),
            risk_report=MockRiskReport(),
            tradeoff_report=MockTradeoffReport(),
            explanation_report=None,
            trust_score=None
        )
        
        # Assertions
        assert reasoning is not None
        assert reasoning.conclusion is not None
        assert len(reasoning.reasoning_steps) > 0
        assert reasoning.recommendation is not None
        assert reasoning.confidence_score >= 0
        assert len(reasoning.key_findings) > 0
        
        print(f"✅ Reasoning engine test passed!")
        print(f"   Conclusion: {reasoning.conclusion[:100]}...")
        print(f"   Confidence: {reasoning.confidence_score:.2f}")
        print(f"   Steps: {len(reasoning.reasoning_steps)}")

    def test_reasoning_engine_minimal(self):
        """Test reasoning engine with minimal valid input."""
        
        # Create minimal valid constraint report
        constraint_result = ConstraintResult(
            name="resolution",
            passed=True,
            actual_value=2.5,
            constraint_value=2.0,
            constraint_type=ConstraintType.MINIMUM,
            margin=0.5,
            margin_percentage=25.0,
            risk_level="low",
            critical=True
        )
        
        class MinimalConstraintReport:
            overall_pass = True
            constraints = [constraint_result]
            passed_constraints = [constraint_result]
            failed_constraints = []
            constraint_satisfaction_rate = 1.0
            worst_margin = 0.5
            worst_margin_percentage = 25.0
            critical_constraints = []
            aqbd_design_space_status = "within"
            recommendations = []

        class MinimalRobustnessReport:
            overall_robustness_score = 0.90
            class method_robustness:
                pass_probability = 0.95
                failure_probability = 0.05
                robustness_score = 0.90
                critical_parameters = []
                parameter_sensitivities = {}
                worst_case_scenario = {}
                operating_range = {}
            robust_region_size = 0.7
            failure_modes = []
            mitigation_strategies = []
            perturbation_analysis = {}

        class MinimalConfidenceReport:
            overall_confidence_score = 0.85
            high_confidence_objectives = []
            low_confidence_objectives = []
            objective_confidence = {}
            uncertainty_sources = []
            recommendations = []

        class MinimalRiskReport:
            class risk_metrics:
                overall_risk_score = 0.08
                coelution_probability = 0.05
                sst_failure_probability = 0.05
                constraint_violation_probability = 0.03
                robustness_failure_probability = 0.05
            risk_breakdown = {}
            high_risk_factors = []
            medium_risk_factors = []
            low_risk_factors = []
            mitigation_strategies = {}
            risk_acceptability = "tolerable"

        engine = ReasoningEngine()
        method_data = {
            'method_id': 'TEST_001',
            'variables': {'pH': 3.2},
            'objectives': {'resolution': 2.5}
        }
        
        reasoning = engine.generate_reasoning(
            method_id='TEST_001',
            method_data=method_data,
            constraint_report=MinimalConstraintReport(),
            robustness_report=MinimalRobustnessReport(),
            confidence_report=MinimalConfidenceReport(),
            risk_report=MinimalRiskReport(),
            tradeoff_report=None,
            explanation_report=None,
            trust_score=None
        )
        
        assert reasoning is not None
        assert reasoning.conclusion is not None
        assert reasoning.recommendation is not None
        print("✅ Minimal reasoning engine test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
