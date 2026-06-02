from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class ConfidenceMetrics:
    """Confidence metrics for a prediction."""
    mean: float
    std: float
    lower_bound_95: float
    upper_bound_95: float
    lower_bound_90: float
    upper_bound_90: float
    coefficient_of_variation: float
    prediction_interval_ratio: float


@dataclass
class ConfidenceReport:
    """Complete uncertainty quantification report."""
    objective_confidence: Dict[str, ConfidenceMetrics]
    overall_confidence_score: float
    high_confidence_objectives: List[str]
    low_confidence_objectives: List[str]
    uncertainty_sources: List[str]
    recommendations: List[str]


class UncertaintyEngine:
    """Quantifies confidence in chromatographic predictions."""
    
    def __init__(
        self,
        method: str = "bootstrap",
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95
    ):
        self.method = method
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
    
    def quantify_uncertainty(
        self,
        predictions: Dict[str, float],
        uncertainties: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> ConfidenceReport:
        """Quantify uncertainty for predictions."""
        confidence_metrics = {}
        
        for objective, value in predictions.items():
            if uncertainties and objective in uncertainties:
                mean, std = uncertainties[objective]
                metrics = self._compute_metrics_from_uncertainty(mean, std)
            else:
                metrics = self._bootstrap_uncertainty(value)
            
            confidence_metrics[objective] = metrics
        
        # Compute overall confidence score
        cv_scores = [m.coefficient_of_variation for m in confidence_metrics.values()]
        overall_confidence = 1.0 - np.mean([min(1.0, cv) for cv in cv_scores])
        
        high_conf = [obj for obj, m in confidence_metrics.items() 
                     if m.coefficient_of_variation < 0.1]
        low_conf = [obj for obj, m in confidence_metrics.items() 
                    if m.coefficient_of_variation > 0.3]
        
        return ConfidenceReport(
            objective_confidence=confidence_metrics,
            overall_confidence_score=overall_confidence,
            high_confidence_objectives=high_conf,
            low_confidence_objectives=low_conf,
            uncertainty_sources=self._identify_uncertainty_sources(confidence_metrics),
            recommendations=self._generate_recommendations(confidence_metrics)
        )
    
    def _compute_metrics_from_uncertainty(self, mean: float, std: float) -> ConfidenceMetrics:
        """Compute confidence metrics from mean and standard deviation."""
        z_95 = stats.norm.ppf(0.975)
        z_90 = stats.norm.ppf(0.95)
        
        return ConfidenceMetrics(
            mean=mean,
            std=std,
            lower_bound_95=mean - z_95 * std,
            upper_bound_95=mean + z_95 * std,
            lower_bound_90=mean - z_90 * std,
            upper_bound_90=mean + z_90 * std,
            coefficient_of_variation=std / abs(mean) if mean != 0 else float('inf'),
            prediction_interval_ratio=(2 * z_95 * std) / abs(mean) if mean != 0 else float('inf')
        )
    
    def _bootstrap_uncertainty(self, value: float) -> ConfidenceMetrics:
        """Use bootstrap for uncertainty estimation."""
        # Generate bootstrap samples around the predicted value
        bootstrap_samples = np.random.normal(value, value * 0.1, self.bootstrap_samples)
        mean = np.mean(bootstrap_samples)
        std = np.std(bootstrap_samples)
        
        return self._compute_metrics_from_uncertainty(mean, std)
    
    def _identify_uncertainty_sources(self, metrics: Dict[str, ConfidenceMetrics]) -> List[str]:
        """Identify sources of uncertainty."""
        sources = []
        for obj, m in metrics.items():
            if m.coefficient_of_variation > 0.3:
                sources.append(f"High uncertainty in {obj} (CV={m.coefficient_of_variation:.2f})")
            if m.prediction_interval_ratio > 1.0:
                sources.append(f"Wide prediction intervals for {obj}")
        return sources
    
    def _generate_recommendations(self, metrics: Dict[str, ConfidenceMetrics]) -> List[str]:
        """Generate recommendations based on uncertainty analysis."""
        recommendations = []
        for obj, m in metrics.items():
            if m.coefficient_of_variation > 0.2:
                recommendations.append(
                    f"Consider additional experiments to reduce uncertainty in {obj}"
                )
        return recommendations
