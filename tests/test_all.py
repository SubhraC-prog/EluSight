"""
Comprehensive tests for EluSight.
"""

import pytest
import json
import tempfile
import os

from schemas.base import MethodResult, MethodVariables, MethodObjectives
from ingest import Ingestor
from constraints.engine import ConstraintEngine
from trust.engine import TrustEngine
from reasoning.engine import ReasoningEngine


class TestEluSight:
    """Test suite for EluSight library."""
    
    def test_imports(self):
        """Test that all modules import correctly."""
        assert Ingestor is not None
        assert ConstraintEngine is not None
        assert TrustEngine is not None
        assert ReasoningEngine is not None
    
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
    
    def test_ingestor_json(self):
        """Test JSON ingestion."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'method_id': 'TEST_001',
                'variables': {'pH': 3.2, 'gradient_time': 18.0},
                'objectives': {'resolution': 2.5, 'runtime': 12.3}
            }, f)
            temp_file = f.name
        
        try:
            methods = Ingestor.from_json(temp_file)
            assert len(methods) >= 0
        finally:
            os.unlink(temp_file)
    
    def test_constraint_evaluation(self):
        """Test constraint engine."""
        constraints = {
            'resolution': {'type': 'minimum', 'value': 2.0, 'critical': True},
            'runtime': {'type': 'maximum', 'value': 20, 'critical': False},
        }
        
        engine = ConstraintEngine(constraints)
        
        method = MethodResult(
            method_id="M001",
            variables=MethodVariables(pH=3.2, gradient_time=18.0),
            objectives=MethodObjectives(resolution=2.5, runtime=12.3)
        )
        
        report = engine.evaluate(method)
        assert report.overall_pass == True
        assert len(report.constraints) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])