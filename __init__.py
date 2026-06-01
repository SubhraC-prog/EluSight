# elusight/__init__.py
"""
EluSight - Chromatographic Decision Intelligence Framework

Transforms optimization outputs into scientifically explainable recommendations.
"""

__version__ = "1.0.0"
__author__ = "EluSight Team"

from elusight.schemas.base import (
    MethodResult, MethodVariables, MethodObjectives,
    ConstraintType, ObjectiveType, ChromatographicPlatform
)
from elusight.ingest import Ingestor
from elusight.trust import TrustEngine, TrustScore
from elusight.reasoning import ReasoningEngine, ScientificReasoning

__all__ = [
    "MethodResult",
    "MethodVariables", 
    "MethodObjectives",
    "ConstraintType",
    "ObjectiveType",
    "ChromatographicPlatform",
    "Ingestor",
    "TrustEngine",
    "TrustScore",
    "ReasoningEngine",
    "ScientificReasoning",
]