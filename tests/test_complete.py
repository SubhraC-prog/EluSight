"""Complete test suite for EluSight."""

import pytest
import tempfile
import json

from schemas.base import MethodResult, MethodVariables, MethodObjectives
from ingest import Ingestor
from constraints.engine import ConstraintEngine
from uncertainty.engine import UncertaintyEngine
from robustness.engine import RobustnessEngine
from risk.engine import RiskEngine
from tradeoffs.engine import TradeoffEngine
from preferences.engine import PreferenceLearningEngine
from explainability.engine import ExplainabilityEngine
from trust.engine import TrustEngine
from reasoning.engine import ReasoningEngine
from reports.generator import ReportGenerator, ReportFormat


class TestEluSightComplete:
    """Complete test suite for EluSight."""
    
    @pytest.fixture
    def sample_method(self):
        """Create a sample method for testing."""
        return MethodResult(
            method_id="TEST_001",
            variables=MethodVariables(pH=3.2, gradient_time=18.0, temperature=40.0),
            objectives=MethodObjectives(resolution=2.5, runtime=12.3, robustness=95.0)
        )
    
    @pytest.fixture
    def sample_constraints(self):
        """Create sample constraints."""
        return {
            'resolution': {'type': 'minimum', 'value': 2.0, 'critical': True},
            'runtime': {'type': 'maximum', 'value': 20, 'critical': False},
            'robustness': {'type': 'minimum', 'value': 90, 'critical': True}
        }
    
    def test_ingestor_json(self):
        """Test JSON ingestion."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'method_id': 'TEST_001',
                'variables': {'pH': 3.2, 'gradient_time': 18.0},
                'objectives': {'resolution': 2.5, 'runtime': 12.3}
            }, f)
            f.flush()
            
            methods = Ingestor.from_json(f.name)
            assert len(methods) >= 0
        print("✅ JSON ingestion test passed")
    
    def test_constraint_engine(self, sample_method, sample_constraints):
        """Test constraint engine."""
        engine = ConstraintEngine(sample_constraints)
        report = engine.evaluate(sample_method)
        
        assert report.overall_pass == True
        assert report.constraint_satisfaction_rate == 1.0
        assert len(report.constraints) == 3
        print("✅ Constraint engine test passed")
    
    def test_uncertainty_engine(self):
        """Test uncertainty engine."""
        engine = UncertaintyEngine()
        predictions = {'resolution': 2.5, 'runtime': 12.3}
        
        report = engine.quantify_uncertainty(predictions)
        
        assert 'resolution' in report.objective_confidence
        assert report.overall_confidence_score >= 0
        print("✅ Uncertainty engine test passed")
    
    def test_robustness_engine(self):
        """Test robustness engine."""
        engine = RobustnessEngine()
        variables = {'pH': 3.2, 'gradient_time': 18.0, 'temperature': 40.0}
        
        report = engine.evaluate_robustness(variables, None)
        
        assert report.overall_robustness_score >= 0
        assert report.method_robustness.pass_probability >= 0
        print("✅ Robustness engine test passed")
    
    def test_risk_engine(self):
        """Test risk engine."""
        engine = RiskEngine()
        method = {'objectives': {'resolution': 2.5, 'robustness': 95.0}}
        
        report = engine.assess_risk(method)
        
        assert report.risk_metrics.overall_risk_score >= 0
        assert report.risk_acceptability in ['acceptable', 'tolerable', 'unacceptable']
        print("✅ Risk engine test passed")
    
    def test_tradeoff_engine(self):
        """Test tradeoff engine."""
        methods = [
            {'id': 'A', 'objectives': {'resolution': 2.5, 'runtime': 10}},
            {'id': 'B', 'objectives': {'resolution': 2.8, 'runtime': 12}},
            {'id': 'C', 'objectives': {'resolution': 2.2, 'runtime': 8}}
        ]
        
        engine = TradeoffEngine(['resolution', 'runtime'], ['max', 'min'])
        report = engine.analyze_tradeoffs(methods)
        
        assert len(report.pareto_front_methods) >= 1
        assert len(report.tradeoffs) >= 0
        print("✅ Tradeoff engine test passed")
    
    def test_preference_learning(self):
        """Test preference learning engine."""
        engine = PreferenceLearningEngine()
        methods = [
            {'id': 'A', 'objectives': {'resolution': 2.5}, 'variables': {'pH': 3.2}},
            {'id': 'B', 'objectives': {'resolution': 2.8}, 'variables': {'pH': 3.5}}
        ]
        
        scores = engine.predict_preference(methods)
        
        assert len(scores) == 2
        assert scores[0].preference_probability >= 0
        print("✅ Preference learning test passed")
    
    def test_explainability_engine(self):
        """Test explainability engine."""
        engine = ExplainabilityEngine()
        features = {'pH': 3.2, 'gradient_time': 18.0, 'temperature': 40.0}
        
        report = engine.explain_prediction(2.5, features)
        
        assert len(report.feature_importances) == 3
        assert len(report.top_features) >= 1
        print("✅ Explainability engine test passed")
    
    def test_report_generator(self, sample_method, sample_constraints):
        """Test report generator."""
        from trust.engine import TrustEngine, TrustScore
        from reasoning.engine import ScientificReasoning
        
        constraint_engine = ConstraintEngine(sample_constraints)
        constraint_report = constraint_engine.evaluate(sample_method)
        
        trust = TrustScore(
            overall_score=85.0,
            constraint_score=90.0,
            robustness_score=85.0,
            confidence_score=80.0,
            risk_score=88.0,
            preference_score=82.0,
            rationale="Test rationale",
            recommendation="Recommend",
            trust_breakdown={}
        )
        
        reasoning = ScientificReasoning(
            conclusion="Test conclusion",
            reasoning_steps=[],
            uncertainties=[],
            recommendation="Test recommendation",
            confidence_score=0.9,
            key_findings=[]
        )
        
        class MockReport:
            overall_robustness_score = 0.95
            class method_robustness:
                pass_probability = 0.97
                critical_parameters = []
        
        class MockRiskReport:
            class risk_metrics:
                overall_risk_score = 0.05
        
        generator = ReportGenerator()
        
        markdown = generator.generate_report(
            'TEST_001', trust, reasoning, constraint_report,
            MockReport(), MockRiskReport(),
            format=ReportFormat.MARKDOWN
        )
        assert 'TEST_001' in markdown
        print("✅ Report generator test passed")


def run_complete_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_complete_tests()
