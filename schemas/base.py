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


# ============================================
# NEW: Categorical Variable Enums
# ============================================

class ColumnType(str, Enum):
    """Supported column types for chromatography."""
    C18 = "c18"
    C8 = "c8"
    C4 = "c4"
    Phenyl = "phenyl"
    HILIC = "hilic"
    IonExchange = "ion_exchange"
    Chiral = "chiral"
    C30 = "c30"
    MixedMode = "mixed_mode"
    Custom = "custom"


class OrganicModifier(str, Enum):
    """Supported organic modifiers."""
    METHANOL = "methanol"
    ACETONITRILE = "acetonitrile"
    ETHANOL = "ethanol"
    IPA = "isopropanol"
    THF = "thf"
    ACETONE = "acetone"


class BufferType(str, Enum):
    """Supported buffer types."""
    PHOSPHATE = "phosphate"
    ACETATE = "acetate"
    FORMATE = "formate"
    TRIS = "tris"
    AMMONIUM = "ammonium"
    CITRATE = "citrate"
    BICARBONATE = "bicarbonate"


class DetectorType(str, Enum):
    """Supported detector types."""
    UV = "uv"
    PDA = "pda"
    DAD = "dad"
    MS = "ms"
    MSMS = "msms"
    CAD = "cad"
    ELSD = "elsd"
    RI = "ri"
    FLD = "fld"


class IonizationMode(str, Enum):
    """Supported ionization modes for MS."""
    ESI_POSITIVE = "esi_positive"
    ESI_NEGATIVE = "esi_negative"
    APCI_POSITIVE = "apci_positive"
    APCI_NEGATIVE = "apci_negative"
    MALDI = "maldi"


# ============================================
# UPDATED: MethodVariables with Categorical Support
# ============================================

class MethodVariables(BaseModel):
    """Chromatographic method variables with categorical support."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    
    # Numerical variables
    pH: Optional[float] = Field(None, alias="pH")
    gradient_time: Optional[float] = None
    temperature: Optional[float] = None
    flow_rate: Optional[float] = None
    organic_modifier_percentage: Optional[float] = None
    buffer_concentration: Optional[float] = None
    column_length: Optional[float] = None
    particle_size: Optional[float] = None
    injection_volume: Optional[float] = None
    wavelength: Optional[float] = None
    
    # Categorical variables (NEW)
    column_type: Optional[Union[str, ColumnType]] = None
    organic_modifier: Optional[Union[str, OrganicModifier]] = None
    buffer_type: Optional[Union[str, BufferType]] = None
    detector_type: Optional[Union[str, DetectorType]] = None
    ionization_mode: Optional[Union[str, IonizationMode]] = None
    
    # Additional custom categorical variables
    custom_categorical: Optional[Dict[str, str]] = Field(default_factory=dict)
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = super().model_dump(**kwargs)
        
        # Convert enums to strings for serialization
        if isinstance(result.get('column_type'), Enum):
            result['column_type'] = result['column_type'].value
        if isinstance(result.get('organic_modifier'), Enum):
            result['organic_modifier'] = result['organic_modifier'].value
        if isinstance(result.get('buffer_type'), Enum):
            result['buffer_type'] = result['buffer_type'].value
        if isinstance(result.get('detector_type'), Enum):
            result['detector_type'] = result['detector_type'].value
        if isinstance(result.get('ionization_mode'), Enum):
            result['ionization_mode'] = result['ionization_mode'].value
        
        return {k: v for k, v in result.items() if v is not None}
    
    def get_one_hot_encoded(self) -> Dict[str, float]:
        """
        Convert categorical variables to one-hot encoded features.
        Useful for machine learning models.
        
        Returns:
            Dictionary with one-hot encoded feature names and values
        """
        encoded = {}
        
        # Handle column type one-hot encoding
        col_type = self.column_type
        if col_type:
            if isinstance(col_type, str):
                col_type = col_type.lower()
            for ct in ColumnType:
                encoded[f'column_type_{ct.value}'] = 1.0 if ct.value == col_type else 0.0
        
        # Handle organic modifier one-hot encoding
        org_mod = self.organic_modifier
        if org_mod:
            if isinstance(org_mod, str):
                org_mod = org_mod.lower()
            for om in OrganicModifier:
                encoded[f'organic_modifier_{om.value}'] = 1.0 if om.value == org_mod else 0.0
        
        # Handle buffer type one-hot encoding
        buf_type = self.buffer_type
        if buf_type:
            if isinstance(buf_type, str):
                buf_type = buf_type.lower()
            for bt in BufferType:
                encoded[f'buffer_type_{bt.value}'] = 1.0 if bt.value == buf_type else 0.0
        
        # Handle detector type one-hot encoding
        det_type = self.detector_type
        if det_type:
            if isinstance(det_type, str):
                det_type = det_type.lower()
            for dt in DetectorType:
                encoded[f'detector_type_{dt.value}'] = 1.0 if dt.value == det_type else 0.0
        
        # Handle ionization mode one-hot encoding
        ion_mode = self.ionization_mode
        if ion_mode:
            if isinstance(ion_mode, str):
                ion_mode = ion_mode.lower()
            for im in IonizationMode:
                encoded[f'ionization_mode_{im.value}'] = 1.0 if im.value == ion_mode else 0.0
        
        # Handle custom categorical variables
        for key, value in self.custom_categorical.items():
            encoded[f'custom_{key}_{value}'] = 1.0
        
        # Add all numerical variables
        for key, value in self.dict().items():
            if isinstance(value, (int, float)) and value is not None:
                encoded[key] = float(value)
        
        return encoded
    
    def get_categorical_summary(self) -> Dict[str, str]:
        """
        Get summary of categorical variables.
        
        Returns:
            Dictionary with categorical variable names and their values
        """
        summary = {}
        
        if self.column_type:
            summary['column_type'] = str(self.column_type)
        if self.organic_modifier:
            summary['organic_modifier'] = str(self.organic_modifier)
        if self.buffer_type:
            summary['buffer_type'] = str(self.buffer_type)
        if self.detector_type:
            summary['detector_type'] = str(self.detector_type)
        if self.ionization_mode:
            summary['ionization_mode'] = str(self.ionization_mode)
        if self.custom_categorical:
            summary.update(self.custom_categorical)
        
        return summary


# ============================================
# UPDATED: MethodObjectives (Enhanced)
# ============================================

class MethodObjectives(BaseModel):
    """Chromatographic method objectives."""
    model_config = ConfigDict(extra="allow")
    
    # Standard CQAs
    resolution: Optional[float] = None
    runtime: Optional[float] = None
    robustness: Optional[float] = None
    peak_capacity: Optional[float] = None
    pressure: Optional[float] = None
    selectivity: Optional[float] = None
    tailing_factor: Optional[float] = None
    theoretical_plates: Optional[float] = None
    
    # Additional CQAs
    retention_factor: Optional[float] = None
    peak_symmetry: Optional[float] = None
    signal_to_noise: Optional[float] = None
    limit_of_detection: Optional[float] = None
    limit_of_quantification: Optional[float] = None
    recovery: Optional[float] = None
    precision: Optional[float] = None
    accuracy: Optional[float] = None
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = super().model_dump(**kwargs)
        return {k: v for k, v in result.items() if v is not None}


# ============================================
# UncertaintyMetrics (Unchanged)
# ============================================

class UncertaintyMetrics(BaseModel):
    """Uncertainty quantification metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    mean: float
    std: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.95
    coefficient_of_variation: Optional[float] = None


# ============================================
# UPDATED: MethodResult with Categorical Support
# ============================================

class MethodResult(BaseModel):
    """Standardized method result from any optimization framework."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
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
    
    def get_one_hot_encoded_features(self) -> Dict[str, float]:
        """
        Get complete one-hot encoded feature vector including variables and objectives.
        Useful for machine learning model input.
        
        Returns:
            Dictionary of all features (categorical one-hot + numerical)
        """
        features = {}
        
        # Add one-hot encoded categorical variables
        features.update(self.variables.get_one_hot_encoded())
        
        # Add numerical variables
        for key, value in self.variables.dict().items():
            if isinstance(value, (int, float)) and value is not None:
                features[key] = float(value)
        
        # Add objectives as features (if needed for multi-task learning)
        for key, value in self.objectives.dict().items():
            if isinstance(value, (int, float)) and value is not None:
                features[f'objective_{key}'] = float(value)
        
        return features
    
    def get_categorical_info(self) -> Dict[str, str]:
        """
        Get categorical variable information for this method.
        
        Returns:
            Dictionary with categorical variable names and values
        """
        return self.variables.get_categorical_summary()


# ============================================
# Helper Functions for Categorical Variables
# ============================================

def get_all_categorical_columns() -> List[str]:
    """
    Get list of all possible categorical column names.
    Useful for DataFrame column creation.
    
    Returns:
        List of categorical column names
    """
    columns = []
    
    # Column type one-hot columns
    for ct in ColumnType:
        columns.append(f'column_type_{ct.value}')
    
    # Organic modifier one-hot columns
    for om in OrganicModifier:
        columns.append(f'organic_modifier_{om.value}')
    
    # Buffer type one-hot columns
    for bt in BufferType:
        columns.append(f'buffer_type_{bt.value}')
    
    # Detector type one-hot columns
    for dt in DetectorType:
        columns.append(f'detector_type_{dt.value}')
    
    # Ionization mode one-hot columns
    for im in IonizationMode:
        columns.append(f'ionization_mode_{im.value}')
    
    return columns


def get_categorical_encoding_map() -> Dict[str, Dict[str, int]]:
    """
    Get mapping of categorical values to indices for custom encoding.
    
    Returns:
        Dictionary mapping category names to value-index maps
    """
    return {
        'column_type': {ct.value: idx for idx, ct in enumerate(ColumnType)},
        'organic_modifier': {om.value: idx for idx, om in enumerate(OrganicModifier)},
        'buffer_type': {bt.value: idx for idx, bt in enumerate(BufferType)},
        'detector_type': {dt.value: idx for idx, dt in enumerate(DetectorType)},
        'ionization_mode': {im.value: idx for idx, im in enumerate(IonizationMode)},
    }
