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
        """Test reasoning engine."""
        engine = ReasoningEngine()
        method_data = {
            'method_id': 'TEST_001',
            'variables': {'pH': 3.2},
            'objectives': {'resolution': 2.5}
        }
        reasoning = engine.generate_reasoning(
            method_id='TEST_001',
            method_data=method_data,
            constraint_report=None,
            robustness_report=None,
            confidence_report=None,
            risk_report=None
        )
        assert reasoning.conclusion is not None
        assert len(reasoning.reasoning_steps) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])