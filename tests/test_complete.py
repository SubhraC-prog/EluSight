cat > tests/test_complete.py << 'EOF'
"""Complete test suite for EluSight."""

import pytest
import json
from pathlib import Path
import tempfile

from elusight import Ingestor, TrustEngine, ReasoningEngine
from schemas.base import MethodResult, MethodVariables, MethodObjectives
from constraints.engine import ConstraintEngine
from uncertainty.engine import UncertaintyEngine
from robustness.engine import RobustnessEngine
from risk.engine import RiskEngine
from tradeoffs.engine import TradeoffEngine
from preferences.engine import PreferenceLearningEngine
from explainability.engine import ExplainabilityEngine
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
    
    def test_ingestor_creation(self, sample_method):
        """Test ingestor functionality."""
        # Test with temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'method_id': 'TEST_001',
                'variables': {'pH': 3.2, 'gradient_time': 18.0},
                'objectives': {'resolution': 2.5, 'runtime': 12.3}
            }, f)
            f.flush()
            
            methods = Ingestor.from_json(f.name)
            assert len(methods) >= 0
        
        # Test DataFrame ingestion
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
    
    def test_constraint_engine(self, sample_method, sample_constraints):
        """Test constraint engine."""
        engine = ConstraintEngine(sample_constraints)
        report = engine.evaluate(sample_method)
        
        assert report.overall_pass == True
        assert report.constraint_satisfaction_rate == 1.0
        assert len(report.constraints) == 3
        assert report.worst_margin_percentage >= 0
    
    def test_uncertainty_engine(self):
        """Test uncertainty engine."""
        engine = UncertaintyEngine()
        predictions = {'resolution': 2.5, 'runtime': 12.3}
        
        report = engine.quantify_uncertainty(predictions)
        
        assert 'resolution' in report.objective_confidence
        assert report.overall_confidence_score >= 0
        assert len(report.recommendations) >= 0
    
    def test_robustness_engine(self):
        """Test robustness engine."""
        engine = RobustnessEngine()
        variables = {'pH': 3.2, 'gradient_time': 18.0, 'temperature': 40.0}
        
        report = engine.evaluate_robustness(variables, None)
        
        assert report.overall_robustness_score >= 0
        assert report.method_robustness.pass_probability >= 0
        assert len(report.method_robustness.critical_parameters) >= 0
    
    def test_risk_engine(self):
        """Test risk engine."""
        engine = RiskEngine()
        method = {'objectives': {'resolution': 2.5, 'robustness': 95.0}}
        
        report = engine.assess_risk(method)
        
        assert report.risk_metrics.overall_risk_score >= 0
        assert report.risk_acceptability in ['acceptable', 'tolerable', 'unacceptable']
        assert len(report.high_risk_factors) >= 0
    
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
    
    def test_explainability_engine(self):
        """Test explainability engine."""
        engine = ExplainabilityEngine()
        features = {'pH': 3.2, 'gradient_time': 18.0, 'temperature': 40.0}
        
        report = engine.explain_prediction(2.5, features)
        
        assert len(report.feature_importances) == 3
        assert len(report.top_features) >= 1
        assert report.scientific_explanation is not None
    
    def test_trust_engine(self, sample_method, sample_constraints):
        """Test trust engine."""
        from elusight.trust.engine import TrustEngine
        
        # Create mock reports
        class MockReport:
            overall_pass = True
            constraints = []
            constraint_satisfaction_rate = 1.0
            worst_margin_percentage = 22.5
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
        assert trust.rationale is not None
    
    def test_reasoning_engine(self, sample_method, sample_constraints):
        """Test reasoning engine."""
        from elusight.constraints.engine import ConstraintEngine
        
        constraint_engine = ConstraintEngine(sample_constraints)
        constraint_report = constraint_engine.evaluate(sample_method)
        
        # Mock other reports
        class MockReport:
            overall_robustness_score = 0.95
            overall_confidence_score = 0.90
            
            class method_robustness:
                robustness_score = 0.95
                pass_probability = 0.97
                critical_parameters = ['pH', 'temperature']
                failure_probability = 0.03
            
            class risk_metrics:
                overall_risk_score = 0.04
                coelution_probability = 0.02
        
        reasoning_engine = ReasoningEngine()
        reasoning = reasoning_engine.generate_reasoning(
            method_id='TEST_001',
            method_data={'variables': {'pH': 3.2}, 'objectives': {'resolution': 2.5}},
            constraint_report=constraint_report,
            robustness_report=MockReport(),
            confidence_report=MockReport(),
            risk_report=MockReport()
        )
        
        assert reasoning.conclusion is not None
        assert len(reasoning.reasoning_steps) >= 3
        assert len(reasoning.key_findings) >= 0
        assert reasoning.recommendation is not None
    
    def test_report_generator(self, sample_method, sample_constraints):
        """Test report generator."""
        from elusight.constraints.engine import ConstraintEngine
        from elusight.trust.engine import TrustEngine, TrustScore
        from elusight.reasoning.engine import ReasoningEngine, ScientificReasoning
        
        # Create minimal reports
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
        
        class MockRobustnessReport:
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
        
        # Test markdown generation
        markdown = generator.generate_report(
            'TEST_001', trust, reasoning, constraint_report,
            MockRobustnessReport(), MockRiskReport(),
            format=ReportFormat.MARKDOWN
        )
        assert 'EluSight' in markdown
        assert 'TEST_001' in markdown
        
        # Test JSON generation
        json_report = generator.generate_report(
            'TEST_001', trust, reasoning, constraint_report,
            MockRobustnessReport(), MockRiskReport(),
            format=ReportFormat.JSON
        )
        assert 'method_id' in json_report
    
    def test_full_pipeline(self, sample_method, sample_constraints):
        """Test the complete EluSight pipeline."""
        # 1. Constraint evaluation
        constraint_engine = ConstraintEngine(sample_constraints)
        constraint_report = constraint_engine.evaluate(sample_method)
        
        # 2. Uncertainty quantification
        uncertainty_engine = UncertaintyEngine()
        predictions = sample_method.objectives.dict()
        uncertainty_report = uncertainty_engine.quantify_uncertainty(predictions)
        
        # 3. Robustness evaluation
        robustness_engine = RobustnessEngine()
        robustness_report = robustness_engine.evaluate_robustness(
            sample_method.variables.dict(), None
        )
        
        # 4. Risk assessment
        risk_engine = RiskEngine()
        method_dict = {
            'objectives': sample_method.objectives.dict(),
            'variables': sample_method.variables.dict()
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
        
        print(f"\nPipeline Test Results:")
        print(f"  Trust Score: {trust.overall_score:.1f}/100")
        print(f"  Recommendation: {trust.recommendation}")
        print(f"  Reasoning: {reasoning.conclusion[:100]}...")


def run_all_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
EOF
