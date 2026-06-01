# elusight/schemas/base.py
"""Base Pydantic schemas for EluSight."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum


class ChromatographicPlatform(str, Enum):
    """Supported chromatographic platforms."""
    HPLC = "hplc"
    UPLC = "uplc"
    UHPLC = "uhplc"
    LC_MS = "lc_ms"
    GC = "gc"
    GC_MS = "gc_ms"
    SFC = "sfc"
    CE = "ce"
    TWO_D_LC = "2d_lc"


class ObjectiveType(str, Enum):
    """Types of optimization objectives."""
    RESOLUTION = "resolution"
    RUNTIME = "runtime"
    ROBUSTNESS = "robustness"
    SENSITIVITY = "sensitivity"
    PEAK_CAPACITY = "peak_capacity"
    PRESSURE = "pressure"
    SELECTIVITY = "selectivity"
    EFFICIENCY = "efficiency"
    CUSTOM = "custom"


class ConstraintType(str, Enum):
    """Types of constraints."""
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    EQUALITY = "equality"
    RANGE = "range"


class MethodVariables(BaseModel):
    """Chromatographic method variables."""
    model_config = ConfigDict(extra="allow")
    
    ph: Optional[float] = Field(None, alias="pH")
    gradient_time: Optional[float] = None
    temperature: Optional[float] = None
    flow_rate: Optional[float] = None
    organic_modifier: Optional[float] = None
    buffer_concentration: Optional[float] = None
    column_type: Optional[str] = None
    column_length: Optional[float] = None
    particle_size: Optional[float] = None
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {k: v for k, v in super().dict(**kwargs).items() if v is not None}
        if 'ph' in result:
            result['pH'] = result.pop('ph')
        return result
    
    class Config:
        populate_by_name = True


class MethodObjectives(BaseModel):
    """Chromatographic method objectives."""
    model_config = ConfigDict(extra="allow")
    
    resolution: Optional[float] = None
    runtime: Optional[float] = None
    robustness: Optional[float] = None
    peak_capacity: Optional[float] = None
    pressure: Optional[float] = None
    selectivity: Optional[float] = None
    tailing_factor: Optional[float] = None
    theoretical_plates: Optional[float] = None
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in super().dict(**kwargs).items() if v is not None}


class UncertaintyMetrics(BaseModel):
    """Uncertainty quantification metrics."""
    mean: float
    std: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.95
    coefficient_of_variation: Optional[float] = None


class MethodResult(BaseModel):
    """Standardized method result from any optimization framework."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    method_id: Union[str, int]
    variables: MethodVariables
    objectives: MethodObjectives
    constraints: Optional[Dict[str, Any]] = None
    uncertainties: Optional[Dict[str, UncertaintyMetrics]] = None
    pareto_rank: Optional[int] = None
    dominance_count: Optional[int] = None
    source_framework: Optional[str] = None
    platform: Optional[ChromatographicPlatform] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }