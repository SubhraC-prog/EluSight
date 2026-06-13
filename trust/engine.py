from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class TrustScore:
    """
    Universal trust score for chromatographic methods.
    Scale: 0-100, where higher is more trustworthy.
    """
    overall_score: float
    constraint_score: float
    robustness_score: float
    confidence_score: float
    risk_score: float
    preference_score: float
    rationale: str
    confidence_interval: Tuple[float, float]
    recommendation: str
    trust_breakdown: Dict[str, float]


@dataclass
class TrustReport:
    """Complete trust assessment report."""
    method_id: str
    trust_score: TrustScore
    score_distribution: Dict[str, float]
    validation_status: str
    improvement_suggestions: List[str]


class TrustEngine:
    """
    Computes comprehensive trust scores for chromatographic methods.
    Implements weighted scoring with scientific rationale.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize trust engine with customizable weights.
        
        Default weights prioritize constraints and robustness as per AQbD.
        """
        self.weights = weights or {
            'constraints': 0.30,
            'robustness': 0.25,
            'confidence': 0.15,
            'risk': 0.20,
            'preference': 0.10
        }
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def compute_trust(
        self,
        method_id: str,
        constraint_report: Any,
        robustness_report: Any,
        confidence_report: Any,
        risk_report: Any,
        preference_scores: List[Any]
    ) -> TrustScore:
        """Compute comprehensive trust score for a method."""
        
        # Extract individual scores with NaN protection
        constraint_score = self._safe_extract(constraint_report, '_extract_constraint_score', 0.5) * 100
        robustness_score = self._safe_extract(robustness_report, '_extract_robustness_score', 0.5) * 100
        confidence_score = self._safe_extract(confidence_report, '_extract_confidence_score', 0.5) * 100
        risk_raw = self._safe_extract(risk_report, '_extract_risk_score', 0.5)
        risk_score = (1 - risk_raw) * 100
        preference_score = self._extract_preference_score(preference_scores) * 100
        
        # Handle NaN values - replace with default
        for score_name, score in [('constraint', constraint_score), ('robustness', robustness_score),
                                   ('confidence', confidence_score), ('risk', risk_score),
                                   ('preference', preference_score)]:
            if np.isnan(score):
                locals()[f"{score_name}_score"] = 50.0
        
        # Clamp scores to 0-100 range
        constraint_score = max(0, min(100, constraint_score))
        robustness_score = max(0, min(100, robustness_score))
        confidence_score = max(0, min(100, confidence_score))
        risk_score = max(0, min(100, risk_score))
        preference_score = max(0, min(100, preference_score))
        
        # Weighted combination
        overall_score = (
            self.weights['constraints'] * constraint_score +
            self.weights['robustness'] * robustness_score +
            self.weights['confidence'] * confidence_score +
            self.weights['risk'] * risk_score +
            self.weights['preference'] * preference_score
        )
        
        # Handle NaN in overall score
        if np.isnan(overall_score):
            overall_score = 50.0
        
        # Calculate confidence interval (95% CI)
        scores = [constraint_score, robustness_score, confidence_score, risk_score, preference_score]
        valid_scores = [s for s in scores if not np.isnan(s)]
        if valid_scores:
            std_dev = np.std(valid_scores) if len(valid_scores) > 1 else 10.0
        else:
            std_dev = 10.0
        
        ci_lower = max(0, overall_score - 1.96 * std_dev)
        ci_upper = min(100, overall_score + 1.96 * std_dev)
        
        # Generate rationale
        rationale = self._generate_rationale(
            overall_score, constraint_score, robustness_score,
            confidence_score, risk_score, preference_score,
            constraint_report
        )
        
        # Generate recommendation
        if overall_score >= 80:
            recommendation = "Strongly Recommend"
        elif overall_score >= 65:
            recommendation = "Recommend"
        elif overall_score >= 50:
            recommendation = "Consider"
        else:
            recommendation = "Avoid"
        
        trust_breakdown = {
            'constraints': constraint_score,
            'robustness': robustness_score,
            'confidence': confidence_score,
            'risk_management': risk_score,
            'expert_alignment': preference_score
        }
        
        return TrustScore(
            overall_score=overall_score,
            constraint_score=constraint_score,
            robustness_score=robustness_score,
            confidence_score=confidence_score,
            risk_score=risk_score,
            preference_score=preference_score,
            rationale=rationale,
            confidence_interval=(ci_lower, ci_upper),
            recommendation=recommendation,
            trust_breakdown=trust_breakdown
        )
    
    def _safe_extract(self, report, method_name, default=0.5) -> float:
        """Safely extract a value with None and NaN handling."""
        if report is None:
            return default
        
        try:
            if method_name.startswith('_'):
                # Call internal method
                if hasattr(self, method_name):
                    value = getattr(self, method_name)(report)
                else:
                    value = default
            else:
                # Direct attribute access
                value = getattr(report, method_name, default)
            
            # Handle NaN
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return default
            
            # Ensure float
            return float(value)
        except Exception:
            return default
    
    def _extract_constraint_score(self, report) -> float:
        """Extract score from constraint report (0-1 scale)."""
        if report is None:
            return 0.5
        
        # Check for overall_pass
        overall_pass = getattr(report, 'overall_pass', None)
        if overall_pass is None:
            return 0.5
        
        if not overall_pass:
            return 0.3
        
        # Score based on margin percentages
        constraints = getattr(report, 'constraints', [])
        if constraints:
            margins = []
            for c in constraints:
                if hasattr(c, 'margin_percentage') and c.margin_percentage is not None:
                    if not np.isnan(c.margin_percentage):
                        margins.append(min(100, c.margin_percentage))
            
            if margins:
                avg_margin = np.mean(margins)
                # Base 0.7 + up to 0.3 based on margins
                return min(1.0, 0.7 + avg_margin / 100)
        
        # Check satisfaction rate
        sat_rate = getattr(report, 'constraint_satisfaction_rate', None)
        if sat_rate is not None and not np.isnan(sat_rate):
            return sat_rate
        
        return 0.7
    
    def _extract_robustness_score(self, report) -> float:
        """Extract score from robustness report (0-1 scale)."""
        if report is None:
            return 0.5
        
        # Try direct attribute
        score = getattr(report, 'overall_robustness_score', None)
        if score is not None and not np.isnan(score):
            return score
        
        # Try nested attribute
        if hasattr(report, 'method_robustness'):
            mr = report.method_robustness
            score = getattr(mr, 'robustness_score', None)
            if score is not None and not np.isnan(score):
                return score
            score = getattr(mr, 'pass_probability', None)
            if score is not None and not np.isnan(score):
                return score
        
        return 0.5
    
    def _extract_confidence_score(self, report) -> float:
        """Extract score from confidence report (0-1 scale)."""
        if report is None:
            return 0.5
        
        score = getattr(report, 'overall_confidence_score', None)
        if score is not None and not np.isnan(score):
            return score
        
        return 0.5
    
    def _extract_risk_score(self, report) -> float:
        """Extract risk score (0-1 scale, lower risk is better)."""
        if report is None:
            return 0.5
        
        # Try direct risk_metrics
        if hasattr(report, 'risk_metrics'):
            rm = report.risk_metrics
            score = getattr(rm, 'overall_risk_score', None)
            if score is not None and not np.isnan(score):
                return score
        
        # Try direct attribute
        score = getattr(report, 'overall_risk_score', None)
        if score is not None and not np.isnan(score):
            return score
        
        return 0.5
    
    def _extract_preference_score(self, scores: List) -> float:
        """Extract preference score (0-1 scale)."""
        if not scores or len(scores) == 0:
            return 0.5
        
        first = scores[0]
        if hasattr(first, 'preference_probability'):
            val = first.preference_probability
            if val is not None and not np.isnan(val):
                return val
        
        return 0.5
    
    def _generate_rationale(
        self,
        overall: float,
        constraint: float,
        robustness: float,
        confidence: float,
        risk: float,
        preference: float,
        constraint_report
    ) -> str:
        """Generate human-readable rationale for trust score."""
        # Handle NaN values
        overall = 50.0 if np.isnan(overall) else overall
        constraint = 50.0 if np.isnan(constraint) else constraint
        robustness = 50.0 if np.isnan(robustness) else robustness
        confidence = 50.0 if np.isnan(confidence) else confidence
        risk = 50.0 if np.isnan(risk) else risk
        preference = 50.0 if np.isnan(preference) else preference
        
        rationale_parts = []
        
        # Overall assessment
        if overall >= 80:
            rationale_parts.append("This method demonstrates excellent overall trustworthiness.")
        elif overall >= 65:
            rationale_parts.append("This method shows good overall trustworthiness.")
        elif overall >= 50:
            rationale_parts.append("This method has acceptable trustworthiness with areas for improvement.")
        else:
            rationale_parts.append("This method shows low trustworthiness and requires optimization.")
        
        # Constraint assessment with specific margins
        margin_info = self._get_margin_info(constraint_report)
        if margin_info:
            margin, text = margin_info
            if margin >= 20:
                rationale_parts.append(f"All constraints satisfied with substantial margins ({text}).")
            elif margin >= 10:
                rationale_parts.append(f"All constraints satisfied with adequate margins ({text}).")
            elif margin >= 0:
                rationale_parts.append(f"Constraints satisfied but with narrow margins ({text}).")
            else:
                rationale_parts.append("Critical constraints violated.")
        else:
            if constraint >= 70:
                rationale_parts.append(f"Constraints satisfied ({constraint:.0f}%).")
            elif constraint >= 50:
                rationale_parts.append(f"Constraints marginally satisfied ({constraint:.0f}%).")
            else:
                rationale_parts.append(f"Constraint violations detected ({constraint:.0f}%).")
        
        # Robustness assessment
        if robustness >= 85:
            rationale_parts.append(f"Exceptional robustness predicted ({robustness:.0f}%).")
        elif robustness >= 70:
            rationale_parts.append(f"Good robustness predicted ({robustness:.0f}%).")
        elif robustness >= 50:
            rationale_parts.append(f"Adequate robustness predicted ({robustness:.0f}%).")
        else:
            rationale_parts.append(f"Robustness concerns identified ({robustness:.0f}%).")
        
        # Risk assessment
        risk_managed = risk
        if risk_managed >= 85:
            rationale_parts.append(f"Very low risk profile ({100-risk_managed:.0f}% risk).")
        elif risk_managed >= 70:
            rationale_parts.append(f"Low risk profile ({100-risk_managed:.0f}% risk).")
        elif risk_managed >= 50:
            rationale_parts.append(f"Moderate risk profile ({100-risk_managed:.0f}% risk).")
        else:
            rationale_parts.append(f"High risk profile ({100-risk_managed:.0f}% risk).")
        
        # Confidence
        if confidence >= 85:
            rationale_parts.append(f"High confidence in predictions ({confidence:.0f}%).")
        elif confidence >= 70:
            rationale_parts.append(f"Good confidence in predictions ({confidence:.0f}%).")
        else:
            rationale_parts.append(f"Limited confidence in predictions ({confidence:.0f}%).")
        
        # Preference alignment
        if preference >= 75:
            rationale_parts.append(f"Strong alignment with expert preferences ({preference:.0f}%).")
        elif preference >= 50:
            rationale_parts.append(f"Moderate alignment with expert preferences ({preference:.0f}%).")
        
        # Final synthesis
        if overall >= 65:
            rationale_parts.append("Therefore, this method represents a scientifically sound choice for chromatographic separation.")
        elif overall >= 50:
            rationale_parts.append("Therefore, this method is acceptable with recommended risk mitigation.")
        else:
            rationale_parts.append("Therefore, this method requires further optimization before implementation.")
        
        return " ".join(rationale_parts)
    
    def _get_margin_info(self, constraint_report) -> Optional[Tuple[float, str]]:
        """Extract margin information from constraint report."""
        if constraint_report is None:
            return None
        
        # Try to get worst margin percentage
        margin = getattr(constraint_report, 'worst_margin_percentage', None)
        if margin is not None and not np.isnan(margin):
            return (margin, f"worst: {margin:.1f}%")
        
        # Try to get margins from individual constraints
        constraints = getattr(constraint_report, 'constraints', [])
        if constraints:
            margins = []
            for c in constraints:
                if hasattr(c, 'margin_percentage') and c.margin_percentage is not None:
                    if not np.isnan(c.margin_percentage):
                        margins.append(c.margin_percentage)
            if margins:
                worst = min(margins)
                return (worst, f"worst: {worst:.1f}%")
        
        return None
