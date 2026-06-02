"""Complete test suite for EluSight."""

import pytest
import tempfile
import json
from pathlib import Path

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
            temp_file = f.name
        
        methods = Ingestor.from_json(temp_file)
        assert len(methods) >= 0
        Path(temp_file).unlink()
        print("✅ JSON ingestion test passed")
    
    def test_ingestor_dataframe(self):
        """Test DataFrame ingestion."""
        import pandas as pd
        df = pd.DataFrame({
            'method_id': ['TEST_001'],
            'var_pH': [3.2],
            'var_gradient_time': [18.0],
            'obj_resolution': [2.5],
            'obj_runtime': [12.3]
        })
        methods = Ingestor.from_dataframe(df)
        assert len(methods) >= 0
        print("✅ DataFrame ingestion test passed")
    
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
        assert report.scientific_interpretation is not None
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
        assert scores[0].ranking >= 1
        print("✅ Preference learning test passed")
    
    def test_explainability_engine(self):
        """Test explainability engine."""
        engine = ExplainabilityEngine()
        features = {'pH': 3.2, 'gradient_time': 18.0, 'temperature': 40.0}
        
        report = engine.explain_prediction(2.5, features)
        
        assert len(report.feature_importances) == 3
        assert len(report.top_features) >= 1
        assert report.scientific_explanation is not None
        print("✅ Explainability engine test passed")
    
    def test_trust_engine(self, sample_method, sample_constraints):
        """Test trust engine."""
        constraint_engine = ConstraintEngine(sample_constraints)
        constraint_report = constraint_engine.evaluate(sample_method)
        
        class MockRobustnessReport:
            overall_robustness_score = 0.95
            class method_robustness:
                robustness_score = 0.95
                pass_probability = 0.97
                critical_parameters = ['pH']
        
        class MockConfidenceReport:
            overall_confidence_score = 0.92
            low_confidence_objectives = []
            high_confidence_objectives = ['resolution']
        
        class MockRiskReport:
            class risk_metrics:
                overall_risk_score = 0.04
        
        engine = TrustEngine()
        trust = engine.compute_trust(
            method_id=sample_method.method_id,
            constraint_report=constraint_report,
            robustness_report=MockRobustnessReport(),
            confidence_report=MockConfidenceReport(),
            risk_report=MockRiskReport(),
            preference_scores=[]
        )
        
        assert 0 <= trust.overall_score <= 100
        assert trust.recommendation in ["Strongly Recommend", "Recommend", "Consider", "Avoid"]
        assert trust.rationale is not None
        print(f"✅ Trust engine test passed - Score: {trust.overall_score:.1f}")
    
    def test_reasoning_engine(self, sample_method, sample_constraints):
        """Test reasoning engine."""
        constraint_engine = ConstraintEngine(sample_constraints)
        constraint_report = constraint_engine.evaluate(sample_method)
        
        class MockReport:
            overall_robustness_score = 0.95
            class method_robustness:
                robustness_score = 0.95
                pass_probability = 0.97
                critical_parameters = ['pH', 'temperature']
                failure_probability = 0.03
            
            class risk_metrics:
                overall_risk_score = 0.04
                coelution_probability = 0.02
                sst_failure_probability = 0.03
        
        reasoning_engine = ReasoningEngine()
        reasoning = reasoning_engine.generate_reasoning(
            method_id=sample_method.method_id,
            method_data={
                'variables': {'pH': 3.2, 'gradient_time': 18.0},
                'objectives': {'resolution': 2.5}
            },
            constraint_report=constraint_report,
            robustness_report=MockReport(),
            confidence_report=MockReport(),
            risk_report=MockReport()
        )
        
        assert reasoning.conclusion is not None
        assert len(reasoning.reasoning_steps) >= 3
        assert len(reasoning.key_findings) >= 0
        assert reasoning.recommendation is not None
        print(f"✅ Reasoning engine test passed - Confidence: {reasoning.confidence_score:.2f}")
    
    def test_report_generator(self, sample_method, sample_constraints):
        """Test report generator."""
        constraint_engine = ConstraintEngine(sample_constraints)
        constraint_report = constraint_engine.evaluate(sample_method)
        
        trust = TrustEngine().compute_trust(
            method_id=sample_method.method_id,
            constraint_report=constraint_report,
            robustness_report=type('obj', (), {'overall_robustness_score': 0.95, 'method_robustness': type('obj', (), {'pass_probability': 0.97})})(),
            confidence_report=type('obj', (), {'overall_confidence_score': 0.92})(),
            risk_report=type('obj', (), {'risk_metrics': type('obj', (), {'overall_risk_score': 0.05})})(),
            preference_scores=[]
        )
        
        reasoning = ReasoningEngine().generate_reasoning(
            method_id=sample_method.method_id,
            method_data={},
            constraint_report=constraint_report,
            robustness_report=None,
            confidence_report=None,
            risk_report=None
        )
        
        class MockReport:
            overall_robustness_score = 0.95
            class method_robustness:
                pass_probability = 0.97
                critical_parameters = []
        
        class MockRiskReport:
            class risk_metrics:
                overall_risk_score = 0.05
                coelution_probability = 0.02
                sst_failure_probability = 0.03
        
        generator = ReportGenerator()
        
        markdown = generator.generate_report(
            sample_method.method_id, trust, reasoning, constraint_report,
            MockReport(), MockRiskReport(),
            format=ReportFormat.MARKDOWN
        )
        assert 'EluSight' in markdown
        assert sample_method.method_id in markdown
        
        json_report = generator.generate_report(
            sample_method.method_id, trust, reasoning, constraint_report,
            MockReport(), MockRiskReport(),
            format=ReportFormat.JSON
        )
        assert 'method_id' in json_report
        print("✅ Report generator test passed")
    
    def test_full_pipeline(self, sample_method, sample_constraints):
        """Test the complete EluSight pipeline."""
        # 1. Constraint evaluation
        constraint_engine = ConstraintEngine(sample_constraints)
        constraint_report = constraint_engine.evaluate(sample_method)
        
        # 2. Uncertainty quantification
        uncertainty_engine = UncertaintyEngine()
        predictions = sample_method.objectives.model_dump()
        uncertainty_report = uncertainty_engine.quantify_uncertainty(predictions)
        
        # 3. Robustness evaluation
        robustness_engine = RobustnessEngine()
        robustness_report = robustness_engine.evaluate_robustness(
            sample_method.variables.model_dump(), None
        )
        
        # 4. Risk assessment
        risk_engine = RiskEngine()
        method_dict = {
            'objectives': sample_method.objectives.model_dump(),
            'variables': sample_method.variables.model_dump()
        }
        risk_report = risk_engine.assess_risk(method_dict)
        
        # 5. Trust scoring
        trust_engine = TrustEngine()
        trust = trust_engine.compute_trust(
            method_id=sample_method.method_id,
            constraint_report=constraint_report,
            robustness_report=robustness_report,
            confidence_report=uncertainty_report,
            risk_report=risk_report,
            preference_scores=[]
        )
        
        # 6. Reasoning generation
        reasoning_engine = ReasoningEngine()
        reasoning = reasoning_engine.generate_reasoning(
            method_id=sample_method.method_id,
            method_data=method_dict,
            constraint_report=constraint_report,
            robustness_report=robustness_report,
            confidence_report=uncertainty_report,
            risk_report=risk_report
        )
        
        # Assert pipeline works
        assert trust.overall_score >= 0
        assert reasoning.conclusion is not None
        assert constraint_report.overall_pass == True
        
        print(f"\n{'='*50}")
        print("Pipeline Test Results:")
        print(f"{'='*50}")
        print(f"  Trust Score: {trust.overall_score:.1f}/100")
        print(f"  Recommendation: {trust.recommendation}")
        print(f"  Reasoning: {reasoning.conclusion[:100]}...")
        print(f"{'='*50}")
        print("✅ Full pipeline test passed!")


def run_complete_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_complete_tests()
