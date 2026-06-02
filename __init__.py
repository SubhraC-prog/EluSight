"""
EluSight - Chromatographic Decision Intelligence Framework
"""

__version__ = "1.0.0"
__author__ = "SubhraC-prog"

# CHANGE THESE LINES - Remove "elusight." prefix
from schemas.base import (
    MethodResult,
    MethodVariables,
    MethodObjectives,
    ChromatographicPlatform,
    ConstraintType,
)

from ingest import Ingestor
from trust.engine import TrustEngine, TrustScore
from reasoning.engine import ReasoningEngine, ScientificReasoning
from constraints.engine import ConstraintEngine, ConstraintReport
from uncertainty.engine import UncertaintyEngine, ConfidenceReport
from robustness.engine import RobustnessEngine, RobustnessReport
from risk.engine import RiskEngine, RiskReport
from tradeoffs.engine import TradeoffEngine, TradeoffReport
from preferences.engine import PreferenceLearningEngine, PreferenceScore
from explainability.engine import ExplainabilityEngine, ExplanationReport

__all__ = [
    "__version__",
    "MethodResult",
    "MethodVariables",
    "MethodObjectives",
    "ChromatographicPlatform",
    "ConstraintType",
    "Ingestor",
    "TrustEngine",
    "TrustScore",
    "ReasoningEngine",
    "ScientificReasoning",
    "ConstraintEngine",
    "ConstraintReport",
    "UncertaintyEngine",
    "ConfidenceReport",
    "RobustnessEngine",
    "RobustnessReport",
    "RiskEngine",
    "RiskReport",
    "TradeoffEngine",
    "TradeoffReport",
    "PreferenceLearningEngine",
    "PreferenceScore",
    "ExplainabilityEngine",
    "ExplanationReport",
]
