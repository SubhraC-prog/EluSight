from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


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
    risk_score: float  # Inverted: higher means lower risk
    preference_score: float
    rationale: str
    confidence_interval: Tuple[float, float]
    recommendation: str  # "Strongly Recommend", "Recommend", "Consider", "Avoid"
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
        
        # Extract individual scores (convert to 0-100 scale)
        constraint_score = self._extract_constraint_score(constraint_report) * 100
        robustness_score = self._extract_robustness_score(robustness_report) * 100
        confidence_score = self._extract_confidence_score(confidence_report) * 100
        risk_score = (1 - self._extract_risk_score(risk_report)) * 100
        preference_score = self._extract_preference_score(preference_scores) * 100
        
        # Weighted combination
        overall_score = (
            self.weights['constraints'] * constraint_score +
            self.weights['robustness'] * robustness_score +
            self.weights['confidence'] * confidence_score +
            self.weights['risk'] * risk_score +
            self.weights['preference'] * preference_score
        )
        
        # Calculate confidence interval (95% CI)
        scores = [constraint_score, robustness_score, confidence_score, risk_score, preference_score]
        ci_lower = max(0, overall_score - 1.96 * np.std(scores))
        ci_upper = min(100, overall_score + 1.96 * np.std(scores))
        
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
    
    def _extract_constraint_score(self, report) -> float:
        """Extract score from constraint report (0-1 scale)."""
        if not hasattr(report, 'overall_pass'):
            return 0.5
        
        if not report.overall_pass:
            return 0.3
        
        # Score based on margin percentages
        if hasattr(report, 'constraints') and report.constraints:
            margins = [min(100, c.margin_percentage) for c in report.constraints]
            avg_margin = np.mean(margins)
            # Base 0.7 + up to 0.3 based on margins
            return min(1.0, 0.7 + avg_margin / 100)
        
        return 0.7
    
    def _extract_robustness_score(self, report) -> float:
        """Extract score from robustness report (0-1 scale)."""
        if hasattr(report, 'overall_robustness_score'):
            return report.overall_robustness_score
        elif hasattr(report, 'method_robustness') and hasattr(report.method_robustness, 'robustness_score'):
            return report.method_robustness.robustness_score
        return 0.5
    
    def _extract_confidence_score(self, report) -> float:
        """Extract score from confidence report (0-1 scale)."""
        if hasattr(report, 'overall_confidence_score'):
            return report.overall_confidence_score
        return 0.5
    
    def _extract_risk_score(self, report) -> float:
        """Extract risk score (0-1 scale, lower risk is better)."""
        if hasattr(report, 'risk_metrics') and hasattr(report.risk_metrics, 'overall_risk_score'):
            return report.risk_metrics.overall_risk_score
        return 0.5
    
    def _extract_preference_score(self, scores: List) -> float:
        """Extract preference score (0-1 scale)."""
        if scores and len(scores) > 0:
            if hasattr(scores[0], 'preference_probability'):
                return scores[0].preference_probability
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
        if hasattr(constraint_report, 'worst_margin_percentage'):
            margin = constraint_report.worst_margin_percentage
            if margin >= 20:
                rationale_parts.append(f"All constraints satisfied with substantial margins (worst: {margin:.1f}%).")
            elif margin >= 10:
                rationale_parts.append(f"All constraints satisfied with adequate margins (worst: {margin:.1f}%).")
            elif margin >= 0:
                rationale_parts.append(f"Constraints satisfied but with narrow margins (worst: {margin:.1f}%).")
            else:
                rationale_parts.append("Critical constraints violated.")
        
        # Robustness assessment
        if robustness >= 85:
            rationale_parts.append(f"Exceptional robustness predicted ({robustness:.1f}%).")
        elif robustness >= 70:
            rationale_parts.append(f"Good robustness predicted ({robustness:.1f}%).")
        elif robustness >= 50:
            rationale_parts.append(f"Adequate robustness predicted ({robustness:.1f}%).")
        else:
            rationale_parts.append(f"Robustness concerns identified ({robustness:.1f}%).")
        
        # Risk assessment
        risk_managed = risk
        if risk_managed >= 85:
            rationale_parts.append(f"Very low risk profile (risk score: {100-risk_managed:.1f}%).")
        elif risk_managed >= 70:
            rationale_parts.append(f"Low risk profile (risk score: {100-risk_managed:.1f}%).")
        elif risk_managed >= 50:
            rationale_parts.append(f"Moderate risk profile (risk score: {100-risk_managed:.1f}%).")
        else:
            rationale_parts.append(f"High risk profile (risk score: {100-risk_managed:.1f}%).")
        
        # Confidence
        if confidence >= 85:
            rationale_parts.append(f"High confidence in predictions ({confidence:.1f}%).")
        elif confidence >= 70:
            rationale_parts.append(f"Good confidence in predictions ({confidence:.1f}%).")
        else:
            rationale_parts.append(f"Limited confidence in predictions ({confidence:.1f}%).")
        
        # Preference alignment
        if preference >= 75:
            rationale_parts.append(f"Strong alignment with expert preferences ({preference:.1f}%).")
        elif preference >= 50:
            rationale_parts.append(f"Moderate alignment with expert preferences ({preference:.1f}%).")
        
        # Final synthesis
        if overall >= 65:
            rationale_parts.append("Therefore, this method represents a scientifically sound choice for chromatographic separation.")
        elif overall >= 50:
            rationale_parts.append("Therefore, this method is acceptable with recommended risk mitigation.")
        else:
            rationale_parts.append("Therefore, this method requires further optimization before implementation.")
        
        return " ".join(rationale_parts)
