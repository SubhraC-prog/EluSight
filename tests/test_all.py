# tests/test_all.py
"""Comprehensive tests for EluSight."""

import pytest
import json
from pathlib import Path

from elusight import (
    MethodResult, MethodVariables, MethodObjectives,
    Ingestor, TrustEngine, ReasoningEngine
)
from constraints.engine import ConstraintEngine
from tradeoffs.engine import TradeoffEngine


class TestEluSight:
    """Test suite for EluSight library."""
    
    @pytest.fixture
    def sample_methods(self):
        """Create sample method data for testing."""
        return [
            {
                'id': 'M001',
                'variables': {'pH': 3.2, 'gradient_time': 18.0, 'temperature': 40.0},
                'objectives': {'resolution': 2.5, 'runtime': 12.3, 'robustness': 95.0}
            },
            {
                'id': 'M002',
                'variables': {'pH': 3.5, 'gradient_time': 15.0, 'temperature': 45.0},
                'objectives': {'resolution': 2.8, 'runtime': 10.5, 'robustness': 92.0}
            },
            {
                'id': 'M003',
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
    
    def test_ingest_json_creation(self, sample_methods, tmp_path):
        """Test JSON ingestion."""
        json_path = tmp_path / "methods.json"
        with open(json_path, 'w') as f:
            json.dump({'methods': sample_methods}, f)
        
        results = Ingestor.from_json(json_path)
        assert len(results) == 3
        assert isinstance(results[0], MethodResult)
    
    def test_constraint_evaluation(self, sample_methods, sample_constraints):
        """Test constraint engine."""
        engine = ConstraintEngine(sample_constraints)
        
        # Convert dict to MethodResult
        method = MethodResult(
            method_id=sample_methods[0]['id'],
            variables=MethodVariables(**sample_methods[0]['variables']),
            objectives=MethodObjectives(**sample_methods[0]['objectives'])
        )
        
        report = engine.evaluate(method)
        
        assert report.overall_pass == True
        assert len(report.constraints) == 3
        assert report.constraint_satisfaction_rate == 1.0
    
    def test_tradeoff_analysis(self, sample_methods):
        """Test tradeoff engine."""
        objectives = ['resolution', 'runtime', 'robustness']
        directions = ['max', 'min', 'max']
        
        engine = TradeoffEngine(objectives, directions)
        
        methods = [
            {'id': m['id'], 'objectives': m['objectives']}
            for m in sample_methods
        ]
        
        report = engine.analyze_tradeoffs(methods)
        
        assert len(report.pareto_front_methods) >= 1
        assert len(report.tradeoffs) > 0
    
    def test_trust_score_calculation(self, sample_methods, sample_constraints):
        """Test trust engine."""
        from elusight.trust.engine import TrustEngine
        
        # Mock reports for testing
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
        
        class MockRiskReport:
            class risk_metrics:
                overall_risk_score = 0.04
        
        class MockPreference:
            preference_probability = 0.85
        
        engine = TrustEngine()
        trust = engine.compute_trust(
            method_id='M001',
            constraint_report=MockReport(),
            robustness_report=MockReport(),
            confidence_report=MockReport(),
            risk_report=MockRiskReport(),
            preference_scores=[MockPreference()]
        )
        
        assert 0 <= trust.overall_score <= 100
        assert trust.recommendation in ["Strongly Recommend", "Recommend", "Consider", "Avoid"]
        assert len(trust.rationale) > 0
    
    def test_reasoning_generation(self, sample_methods, sample_constraints):
        """Test reasoning engine."""
        from elusight.constraints.engine import ConstraintEngine
        from elusight.trust.engine import TrustEngine
        
        # Setup
        constraint_engine = ConstraintEngine(sample_constraints)
        method = MethodResult(
            method_id=sample_methods[0]['id'],
            variables=MethodVariables(**sample_methods[0]['variables']),
            objectives=MethodObjectives(**sample_methods[0]['objectives'])
        )
        
        constraint_report = constraint_engine.evaluate(method)
        
        # Mock other reports
        class MockReport:
            overall_pass = True
            constraints = []
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
                sst_failure_probability = 0.03
                constraint_violation_probability = 0.01
                robustness_failure_probability = 0.02
        
        reasoning_engine = ReasoningEngine()
        reasoning = reasoning_engine.generate_reasoning(
            method_id='M001',
            method_data=sample_methods[0],
            constraint_report=constraint_report,
            robustness_report=MockReport(),
            confidence_report=MockReport(),
            risk_report=MockReport(),
            tradeoff_report=None,
            explanation_report=None,
            trust_score=None
        )
        
        assert reasoning.conclusion is not None
        assert len(reasoning.reasoning_steps) >= 3
        assert len(reasoning.key_findings) > 0
        assert reasoning.recommendation is not None
    
    def test_end_to_end_pipeline(self, sample_methods, sample_constraints):
        """Test complete pipeline from ingestion to reasoning."""
        from elusight.constraints.engine import ConstraintEngine
        from elusight.trust.engine import TrustEngine
        from elusight.reasoning.engine import ReasoningEngine
        from elusight.tradeoffs.engine import TradeoffEngine
        
        # 1. Tradeoff analysis
        objectives = ['resolution', 'runtime', 'robustness']
        directions = ['max', 'min', 'max']
        tradeoff_engine = TradeoffEngine(objectives, directions)
        
        methods_for_tradeoff = [
            {'id': m['id'], 'objectives': m['objectives']}
            for m in sample_methods
        ]
        tradeoff_report = tradeoff_engine.analyze_tradeoffs(methods_for_tradeoff)
        
        # 2. Constraint evaluation
        constraint_engine = ConstraintEngine(sample_constraints)
        
        # 3. Process each method
        results = []
        for method_data in sample_methods:
            method = MethodResult(
                method_id=method_data['id'],
                variables=MethodVariables(**method_data['variables']),
                objectives=MethodObjectives(**method_data['objectives'])
            )
            
            constraint_report = constraint_engine.evaluate(method)
            
            # Mock reports for simplicity
            class MockReport:
                overall_pass = constraint_report.overall_pass
                constraints = constraint_report.constraints
                overall_robustness_score = 0.95
                overall_confidence_score = 0.90
                
                class method_robustness:
                    robustness_score = 0.95
                    pass_probability = 0.97
                    critical_parameters = []
                    failure_probability = 0.03
                
                class risk_metrics:
                    overall_risk_score = 0.05
            
            # Trust score
            trust_engine = TrustEngine()
            trust = trust_engine.compute_trust(
                method_id=method_data['id'],
                constraint_report=constraint_report,
                robustness_report=MockReport(),
                confidence_report=MockReport(),
                risk_report=MockReport(),
                preference_scores=[]
            )
            
            # Reasoning
            reasoning_engine = ReasoningEngine()
            reasoning = reasoning_engine.generate_reasoning(
                method_id=method_data['id'],
                method_data=method_data,
                constraint_report=constraint_report,
                robustness_report=MockReport(),
                confidence_report=MockReport(),
                risk_report=MockReport(),
                tradeoff_report=tradeoff_report
            )
            
            results.append({
                'method_id': method_data['id'],
                'trust_score': trust.overall_score,
                'recommendation': trust.recommendation,
                'reasoning': reasoning.conclusion
            })
        
        assert len(results) == 3
        # At least one method should be recommended
        assert any(r['recommendation'] in ['Recommend', 'Strongly Recommend'] for r in results)


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
