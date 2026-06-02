from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class ReasoningStep:
    """Individual step in scientific reasoning."""
    claim: str
    evidence: List[str]
    confidence: float
    citations: List[str]
    quantitative_support: Dict[str, float]


@dataclass
class ScientificReasoning:
    """
    Complete scientific reasoning output.
    Converts numerical optimization results into human-readable scientific arguments.
    """
    conclusion: str
    reasoning_steps: List[ReasoningStep]
    supporting_evidence: Dict[str, Any]
    uncertainties: List[str]
    recommendation: str
    confidence_score: float
    key_findings: List[str]
    expert_commentary: str


class ReasoningEngine:
    """
    Converts numerical outputs into scientific reasoning.
    This is the most important module in EluSight - it answers "WHY?"
    """
    
    def __init__(self, domain_knowledge: Optional[Dict[str, Any]] = None):
        self.domain_knowledge = domain_knowledge or self._load_chromatography_knowledge()
    
    def generate_reasoning(
        self,
        method_id: str,
        method_data: Dict[str, Any],
        constraint_report: Any,
        robustness_report: Any,
        confidence_report: Any,
        risk_report: Any,
        tradeoff_report: Optional[Any] = None,
        explanation_report: Optional[Any] = None,
        trust_score: Optional[Any] = None
    ) -> ScientificReasoning:
        """
        Generate scientific reasoning for a method recommendation.
        This is the core method that produces human-readable explanations.
        """
        
        reasoning_steps = []
        key_findings = []
        
        # Step 1: Constraint satisfaction (AQbD compliance)
        constraint_reasoning, constraint_findings = self._reason_constraints(
            constraint_report, method_data
        )
        reasoning_steps.append(constraint_reasoning)
        key_findings.extend(constraint_findings)
        
        # Step 2: Robustness assessment
        robustness_reasoning, robustness_findings = self._reason_robustness(
            robustness_report, method_data
        )
        reasoning_steps.append(robustness_reasoning)
        key_findings.extend(robustness_findings)
        
        # Step 3: Uncertainty and confidence
        confidence_reasoning, confidence_findings = self._reason_confidence(
            confidence_report
        )
        reasoning_steps.append(confidence_reasoning)
        key_findings.extend(confidence_findings)
        
        # Step 4: Risk assessment
        risk_reasoning, risk_findings = self._reason_risk(risk_report)
        reasoning_steps.append(risk_reasoning)
        key_findings.extend(risk_findings)
        
        # Step 5: Trade-off analysis (if available)
        if tradeoff_report:
            tradeoff_reasoning, tradeoff_findings = self._reason_tradeoffs(
                tradeoff_report, method_id
            )
            reasoning_steps.append(tradeoff_reasoning)
            key_findings.extend(tradeoff_findings)
        
        # Step 6: Feature importance (if available)
        if explanation_report:
            feature_reasoning, feature_findings = self._reason_features(
                explanation_report, method_data
            )
            reasoning_steps.append(feature_reasoning)
            key_findings.extend(feature_findings)
        
        # Synthesize conclusion
        conclusion = self._synthesize_conclusion(
            method_id, reasoning_steps, trust_score, key_findings
        )
        
        # Identify remaining uncertainties
        uncertainties = self._identify_uncertainties(
            constraint_report, robustness_report, confidence_report, risk_report
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            trust_score, risk_report, robustness_report
        )
        
        # Generate expert commentary
        expert_commentary = self._generate_expert_commentary(
            constraint_report, robustness_report, risk_report
        )
        
        # Calculate confidence in reasoning
        confidence_score = self._calculate_reasoning_confidence(reasoning_steps)
        
        return ScientificReasoning(
            conclusion=conclusion,
            reasoning_steps=reasoning_steps,
            supporting_evidence={
                'constraints': constraint_report,
                'robustness': robustness_report,
                'risk': risk_report,
                'confidence': confidence_report
            },
            uncertainties=uncertainties,
            recommendation=recommendation,
            confidence_score=confidence_score,
            key_findings=key_findings[:10],  # Top 10 findings
            expert_commentary=expert_commentary
        )
    
    def _reason_constraints(self, report, method_data) -> Tuple[ReasoningStep, List[str]]:
        """Generate reasoning about constraint satisfaction."""
        evidence = []
        findings = []
        claim = ""
        
        if hasattr(report, 'overall_pass'):
            if report.overall_pass:
                margins = [c.margin_percentage for c in report.constraints]
                avg_margin = np.mean(margins)
                best_margin = max(margins)
                worst_margin = min(margins)
                
                # Find the constraint with worst margin
                worst_constraint = min(report.constraints, key=lambda x: x.margin_percentage)
                
                claim = f"Method meets or exceeds all chromatographic constraints with an average margin of {avg_margin:.1f}%."
                
                evidence.append(f"All {len(report.constraints)} constraints are satisfied according to AQbD principles")
                evidence.append(f"Minimum resolution exceeds acceptance criteria by {best_margin:.1f}%")
                
                if worst_margin >= 20:
                    evidence.append(f"Even the tightest constraint ({worst_constraint.name}) has a substantial {worst_margin:.1f}% safety margin")
                    findings.append(f"✓ Wide safety margins (lowest: {worst_margin:.1f}%)")
                elif worst_margin >= 10:
                    evidence.append(f"The most critical constraint ({worst_constraint.name}) has a {worst_margin:.1f}% margin")
                    findings.append(f"✓ Adequate safety margins (lowest: {worst_margin:.1f}%)")
                else:
                    evidence.append(f"Constraint '{worst_constraint.name}' has a narrow margin of {worst_margin:.1f}%")
                    findings.append(f"⚠ Narrow margin for {worst_constraint.name}")
                
                # AQbD design space status
                if hasattr(report, 'aqbd_design_space_status'):
                    if report.aqbd_design_space_status == 'within':
                        evidence.append("Method operates well within the AQbD design space")
                        findings.append("✓ Within AQbD design space")
                    elif report.aqbd_design_space_status == 'edge':
                        evidence.append("Method operates at the edge of the design space")
                        findings.append("⚠ At design space edge")
            else:
                failed = report.failed_constraints[0] if report.failed_constraints else None
                claim = f"Method fails critical constraint: {failed.name if failed else 'constraint violation'}."
                evidence.append(f"{failed.name if failed else 'Constraint'} value: {failed.actual_value:.2f} vs requirement {failed.constraint_value:.2f}")
                evidence.append(f"Margin to failure: {failed.margin:.2f} ({failed.margin_percentage:.1f}%)")
                findings.append(f"✗ Constraint violation: {failed.name if failed else 'unknown'}")
        
        step = ReasoningStep(
            claim=claim,
            evidence=evidence,
            confidence=0.95 if report.overall_pass else 0.99,
            citations=["ICH Q14 (Analytical Procedure Development)", "USP <1225> (Validation of Compendial Procedures)"],
            quantitative_support={
                'satisfaction_rate': getattr(report, 'constraint_satisfaction_rate', 0),
                'worst_margin': getattr(report, 'worst_margin_percentage', 0)
            }
        )
        
        return step, findings
    
    def _reason_robustness(self, report, method_data) -> Tuple[ReasoningStep, List[str]]:
        """Generate reasoning about method robustness."""
        evidence = []
        findings = []
        claim = ""
        
        if hasattr(report, 'method_robustness'):
            mr = report.method_robustness
            
            if mr.pass_probability >= 0.99:
                claim = f"Method demonstrates exceptional robustness with {mr.pass_probability*100:.1f}% probability of success under typical operating variations."
                findings.append("✓ Exceptional robustness (>99% success rate)")
            elif mr.pass_probability >= 0.95:
                claim = f"Method shows excellent robustness with {mr.pass_probability*100:.1f}% probability of success."
                findings.append(f"✓ Excellent robustness ({mr.pass_probability*100:.1f}% success)")
            elif mr.pass_probability >= 0.90:
                claim = f"Method demonstrates good robustness with {mr.pass_probability*100:.1f}% probability of success."
                findings.append(f"✓ Good robustness ({mr.pass_probability*100:.1f}% success)")
            elif mr.pass_probability >= 0.85:
                claim = f"Method shows adequate robustness with {mr.pass_probability*100:.1f}% probability of success."
                findings.append(f"~ Adequate robustness ({mr.pass_probability*100:.1f}% success)")
            else:
                claim = f"Method has marginal robustness with only {mr.pass_probability*100:.1f}% probability of success."
                findings.append(f"⚠ Marginal robustness ({mr.pass_probability*100:.1f}% success)")
            
            evidence.append(f"Predicted robustness score: {mr.robustness_score*100:.1f}%")
            evidence.append(f"Probability of failure under normal operating conditions: {mr.failure_probability*100:.2f}%")
            
            if mr.critical_parameters:
                evidence.append(f"Critical parameters requiring tight control: {', '.join(mr.critical_parameters[:3])}")
                findings.append(f"⚠ Critical parameters: {', '.join(mr.critical_parameters[:2])}")
            
            # Robust region analysis
            if hasattr(report, 'robust_region_size'):
                region_size = report.robust_region_size * 100
                if region_size > 80:
                    evidence.append(f"Large robust design space covering {region_size:.1f}% of parameter range")
                    findings.append(f"✓ Wide robust region ({region_size:.1f}%)")
                elif region_size > 50:
                    evidence.append(f"Moderate robust design space covering {region_size:.1f}% of parameter range")
                else:
                    evidence.append(f"Limited robust design space covering only {region_size:.1f}% of parameter range")
                    findings.append(f"⚠ Limited robust region ({region_size:.1f}%)")
        
        step = ReasoningStep(
            claim=claim,
            evidence=evidence,
            confidence=0.85,
            citations=["ICH Q2(R2) (Validation of Analytical Procedures)", "USP <1210> (Statistical Tools for Procedure Validation)"],
            quantitative_support={
                'pass_probability': getattr(mr, 'pass_probability', 0) if hasattr(report, 'method_robustness') else 0,
                'robustness_score': getattr(mr, 'robustness_score', 0) if hasattr(report, 'method_robustness') else 0
            }
        )
        
        return step, findings
    
    def _reason_confidence(self, report) -> Tuple[ReasoningStep, List[str]]:
        """Generate reasoning about prediction confidence."""
        evidence = []
        findings = []
        claim = ""
        
        if hasattr(report, 'overall_confidence_score'):
            conf = report.overall_confidence_score
            
            if conf >= 0.95:
                claim = f"Predictions have very high confidence ({conf*100:.1f}%). The model demonstrates exceptional predictive accuracy."
                findings.append("✓ Very high prediction confidence")
            elif conf >= 0.90:
                claim = f"Predictions have high confidence ({conf*100:.1f}%)."
                findings.append("✓ High prediction confidence")
            elif conf >= 0.80:
                claim = f"Predictions have good confidence ({conf*100:.1f}%)."
                findings.append(f"~ Good prediction confidence ({conf*100:.1f}%)")
            elif conf >= 0.70:
                claim = f"Predictions have adequate confidence ({conf*100:.1f}%)."
                findings.append(f"~ Adequate confidence ({conf*100:.1f}%)")
            else:
                claim = f"Predictions have limited confidence ({conf*100:.1f}%). Additional validation recommended."
                findings.append(f"⚠ Limited confidence ({conf*100:.1f}%)")
            
            evidence.append(f"Overall confidence score: {conf*100:.1f}%")
            
            if report.high_confidence_objectives:
                evidence.append(f"High confidence predictions for: {', '.join(report.high_confidence_objectives[:3])}")
            
            if report.low_confidence_objectives:
                evidence.append(f"Lower confidence predictions for: {', '.join(report.low_confidence_objectives[:3])}")
                findings.append(f"⚠ Uncertainty in: {', '.join(report.low_confidence_objectives[:2])}")
            
            for rec in report.recommendations[:2]:
                evidence.append(rec)
        
        step = ReasoningStep(
            claim=claim,
            evidence=evidence,
            confidence=0.90,
            citations=["Prediction Interval Theory", "Uncertainty Quantification in Analytical Chemistry"],
            quantitative_support={'confidence_score': getattr(report, 'overall_confidence_score', 0)}
        )
        
        return step, findings
    
    def _reason_risk(self, report) -> Tuple[ReasoningStep, List[str]]:
        """Generate reasoning about risks."""
        evidence = []
        findings = []
        claim = ""
        
        if hasattr(report, 'risk_metrics'):
            rm = report.risk_metrics
            
            if rm.overall_risk_score < 0.03:
                claim = f"Overall risk is very low ({rm.overall_risk_score*100:.1f}%). Method is suitable for routine use."
                findings.append("✓ Very low overall risk")
            elif rm.overall_risk_score < 0.08:
                claim = f"Overall risk is low ({rm.overall_risk_score*100:.1f}%)."
                findings.append(f"✓ Low overall risk ({rm.overall_risk_score*100:.1f}%)")
            elif rm.overall_risk_score < 0.15:
                claim = f"Overall risk is moderate ({rm.overall_risk_score*100:.1f}%). Mitigation strategies recommended."
                findings.append(f"~ Moderate risk ({rm.overall_risk_score*100:.1f}%)")
            else:
                claim = f"Overall risk is high ({rm.overall_risk_score*100:.1f}%). Significant mitigation required."
                findings.append(f"⚠ High risk ({rm.overall_risk_score*100:.1f}%)")
            
            # Individual risk factors
            if rm.coelution_probability > 0.1:
                evidence.append(f"Probability of co-elution: {rm.coelution_probability*100:.1f}%")
                if rm.coelution_probability > 0.2:
                    findings.append("⚠ Co-elution risk")
            
            if rm.sst_failure_probability > 0.1:
                evidence.append(f"Probability of SST failure: {rm.sst_failure_probability*100:.1f}%")
                if rm.sst_failure_probability > 0.15:
                    findings.append("⚠ SST failure risk")
            
            if rm.constraint_violation_probability > 0.05:
                evidence.append(f"Probability of constraint violation: {rm.constraint_violation_probability*100:.1f}%")
            
            evidence.append(f"Overall risk score: {rm.overall_risk_score*100:.1f}%")
            
            # Mitigation strategies
            if hasattr(report, 'mitigation_strategies') and report.mitigation_strategies:
                evidence.append("Recommended risk mitigation strategies:")
                for risk, strategies in list(report.mitigation_strategies.items())[:2]:
                    evidence.append(f"  - For {risk}: {strategies[0]}")
        
        step = ReasoningStep(
            claim=claim,
            evidence=evidence,
            confidence=0.85,
            citations=["ICH Q9 (Quality Risk Management)", "FMEA Guidelines for Analytical Procedures"],
            quantitative_support={
                'overall_risk': getattr(rm, 'overall_risk_score', 0) if hasattr(report, 'risk_metrics') else 0,
                'coelution_risk': getattr(rm, 'coelution_probability', 0) if hasattr(report, 'risk_metrics') else 0
            }
        )
        
        return step, findings
    
    def _reason_tradeoffs(self, report, method_id: str) -> Tuple[ReasoningStep, List[str]]:
        """Generate reasoning about trade-offs."""
        evidence = []
        findings = []
        claim = ""
        
        if hasattr(report, 'tradeoffs'):
            # Find trade-offs involving this method
            relevant_tradeoffs = [t for t in report.tradeoffs 
                                 if t.method_a_id == method_id or t.method_b_id == method_id]
            
            if relevant_tradeoffs:
                top_tradeoff = relevant_tradeoffs[0]
                
                if top_tradeoff.method_a_id == method_id:
                    other = top_tradeoff.method_b_id
                    gains = top_tradeoff.gain
                    costs = top_tradeoff.cost
                else:
                    other = top_tradeoff.method_a_id
                    gains = {k: top_tradeoff.cost.get(k, 0) for k in top_tradeoff.cost}
                    costs = {k: top_tradeoff.gain.get(k, 0) for k in top_tradeoff.gain}
                
                if gains and costs:
                    gain_str = ", ".join([f"+{v:.2f} in {k}" for k, v in gains.items() if v > 0])
                    cost_str = ", ".join([f"-{v:.2f} in {k}" for k, v in costs.items() if v > 0])
                    claim = f"Method {method_id} represents a balanced trade-off: {gain_str} at the cost of {cost_str}."
                    findings.append(f"~ Trade-off: {gain_str.split()[0] if gain_str else 'gains'} vs {cost_str.split()[0] if cost_str else 'losses'}")
                else:
                    claim = f"Method {method_id} is non-dominated and lies on the Pareto optimal frontier."
                    findings.append("✓ On Pareto optimal frontier")
            else:
                if hasattr(report, 'pareto_front_methods') and method_id in report.pareto_front_methods:
                    claim = f"Method {method_id} is a Pareto-optimal solution representing the best possible trade-offs."
                    findings.append("✓ Pareto-optimal solution")
            
            # Check if knee point
            if hasattr(report, 'knee_points') and method_id in report.knee_points:
                evidence.append("This method is identified as a 'knee point' representing the best balance of conflicting objectives")
                findings.append("✓ Best trade-off point (knee point)")
            
            # Best in class
            if hasattr(report, 'best_methods_by_objective'):
                for obj, best_id in report.best_methods_by_objective.items():
                    if best_id == method_id:
                        evidence.append(f"Achieves best-in-class performance for {obj}")
                        findings.append(f"✓ Best {obj}")
        
        step = ReasoningStep(
            claim=claim or f"Method {method_id} shows competitive performance in the multi-objective space.",
            evidence=evidence,
            confidence=0.90,
            citations=["Multi-objective Optimization in Chromatography", "Pareto Optimality Theory"],
            quantitative_support={}
        )
        
        return step, findings
    
    def _reason_features(self, report, method_data) -> Tuple[ReasoningStep, List[str]]:
        """Generate reasoning about feature importance."""
        evidence = []
        findings = []
        claim = ""
        
        if hasattr(report, 'feature_importances') and report.feature_importances:
            top_features = report.feature_importances[:3]
            
            claim = f"The primary drivers of separation performance are: {', '.join([f.feature_name for f in top_features])}."
            
            for feature in top_features:
                current_val = method_data.get('variables', {}).get(feature.feature_name, 0)
                evidence.append(
                    f"{feature.feature_name} (current: {current_val:.2f}) contributes {feature.shap_value:.3f} to the prediction"
                )
                
                if feature.shap_value > 0.3:
                    findings.append(f"★ {feature.feature_name} is a key driver")
            
            # Counterfactual analysis
            if hasattr(report, 'counterfactuals') and report.counterfactuals:
                cf = report.counterfactuals[0]
                if cf.feasibility_score > 0.7:
                    evidence.append(
                        f"A feasible alternative exists: adjusting {', '.join(cf.changes_made.keys())} "
                        f"could improve performance by {cf.predicted_improvement:.2f}"
                    )
                    findings.append(f"~ Improvement possible via {', '.join(list(cf.changes_made.keys())[:2])}")
        
        step = ReasoningStep(
            claim=claim or "Multiple factors contribute to method performance.",
            evidence=evidence,
            confidence=0.85,
            citations=["SHAP Values in Analytical Chemistry", "Interpretable Machine Learning"],
            quantitative_support={}
        )
        
        return step, findings
    
    def _synthesize_conclusion(
        self,
        method_id: str,
        reasoning_steps: List[ReasoningStep],
        trust_score: Any,
        key_findings: List[str]
    ) -> str:
        """Synthesize overall conclusion from reasoning steps."""
        
        # Count positive and warning findings
        positive = [f for f in key_findings if f.startswith('✓') or f.startswith('★')]
        warnings = [f for f in key_findings if f.startswith('⚠') or f.startswith('~')]
        negatives = [f for f in key_findings if f.startswith('✗')]
        
        # Determine overall status
        if negatives:
            status = "requires optimization"
            severity = "critical issues"
        elif warnings and len(warnings) > 2:
            status = "has several considerations"
            severity = "moderate concerns"
        elif warnings:
            status = "has minor considerations"
            severity = "some concerns"
        else:
            status = "is scientifically sound"
            severity = "no major issues"
        
        # Get trust level
        trust_level = ""
        if trust_score:
            if trust_score.overall_score >= 80:
                trust_level = "highly trustworthy"
            elif trust_score.overall_score >= 65:
                trust_level = "trustworthy"
            elif trust_score.overall_score >= 50:
                trust_level = "moderately trustworthy"
            else:
                trust_level = "has low trustworthiness"
        
        # Build conclusion
        conclusion = (
            f"Method {method_id} {trust_level} and {status}. "
            f"Key strengths include: {', '.join(positive[:3]) if positive else 'meeting basic requirements'}. "
        )
        
        if warnings:
            conclusion += f"Considerations include: {', '.join(warnings[:2])}. "
        
        if negatives:
            conclusion += f"Critical issues: {', '.join(negatives[:2])}. "
        
        conclusion += (
            f"Therefore, this method represents "
            f"{'a recommended' if trust_score and trust_score.overall_score >= 65 else 'an acceptable' if trust_score and trust_score.overall_score >= 50 else 'a non-recommended'} "
            f"choice for chromatographic separation."
        )
        
        return conclusion
    
    def _identify_uncertainties(self, constraint_report, robustness_report, confidence_report, risk_report) -> List[str]:
        """Identify remaining uncertainties."""
        uncertainties = []
        
        # From confidence report
        if hasattr(confidence_report, 'low_confidence_objectives') and confidence_report.low_confidence_objectives:
            uncertainties.append(f"Predictions for {', '.join(confidence_report.low_confidence_objectives)} have higher uncertainty")
        
        # From robustness report
        if hasattr(robustness_report, 'method_robustness'):
            mr = robustness_report.method_robustness
            if mr.pass_probability < 0.95:
                uncertainties.append(f"Method success probability is {mr.pass_probability*100:.1f}%, indicating some operational risk")
        
        # From constraint report
        if hasattr(constraint_report, 'worst_margin_percentage'):
            if constraint_report.worst_margin_percentage < 15:
                uncertainties.append(f"Constraint '{constraint_report.failed_constraints[0].name if constraint_report.failed_constraints else 'critical'}' has narrow margin requiring careful control")
        
        # From risk report
        if hasattr(risk_report, 'risk_metrics'):
            if risk_report.risk_metrics.coelution_probability > 0.1:
                uncertainties.append(f"Co-elution probability of {risk_report.risk_metrics.coelution_probability*100:.1f}% requires verification")
        
        return uncertainties if uncertainties else ["No major uncertainties identified"]
    
    def _generate_recommendation(self, trust_score, risk_report, robustness_report) -> str:
        """Generate final recommendation."""
        if trust_score:
            if trust_score.recommendation == "Strongly Recommend":
                return (
                    "This method is strongly recommended for immediate implementation. "
                    "All quality attributes are well-controlled, robustness is exceptional, "
                    "and risks are minimal. Proceed with validation according to ICH Q2(R2)."
                )
            elif trust_score.recommendation == "Recommend":
                return (
                    "This method is recommended for implementation with routine monitoring. "
                    "Implement the suggested risk mitigation strategies and monitor critical "
                    "parameters during initial validation runs. The method is suitable for "
                    "intended use with proper controls."
                )
            elif trust_score.recommendation == "Consider":
                return (
                    "This method may be considered with additional validation. "
                    "Conduct supplemental robustness testing focusing on identified critical "
                    "parameters. Consider implementing enhanced quality controls before "
                    "routine use. Alternative methods on the Pareto front may offer better trade-offs."
                )
            else:
                return (
                    "This method is not recommended in its current form. "
                    "Further optimization is required to address constraint violations "
                    "and robustness concerns. Consider adjusting the most influential "
                    "parameters identified in the analysis or evaluating alternative "
                    "methods on the Pareto frontier."
                )
        
        return "Insufficient data to generate recommendation."
    
    def _generate_expert_commentary(self, constraint_report, robustness_report, risk_report) -> str:
        """Generate expert-style commentary."""
        commentary_parts = []
        
        # Scientific assessment
        if hasattr(constraint_report, 'aqbd_design_space_status'):
            if constraint_report.aqbd_design_space_status == 'within':
                commentary_parts.append("From an AQbD perspective, this method operates well within the established design space.")
            elif constraint_report.aqbd_design_space_status == 'edge':
                commentary_parts.append("AQbD analysis indicates the method operates at the edge of the design space, requiring careful control.")
        
        # Robustness expert opinion
        if hasattr(robustness_report, 'method_robustness'):
            mr = robustness_report.method_robustness
            if mr.pass_probability > 0.95:
                commentary_parts.append("The robustness assessment suggests the method will perform reliably across typical laboratory variations.")
            elif mr.pass_probability > 0.85:
                commentary_parts.append("While generally robust, the method shows sensitivity to certain parameters that should be tightly controlled.")
        
        # Risk-based recommendation
        if hasattr(risk_report, 'risk_metrics'):
            if risk_report.risk_metrics.overall_risk_score < 0.05:
                commentary_parts.append("Risk assessment indicates the method has an excellent safety profile for routine use.")
            elif risk_report.risk_metrics.overall_risk_score < 0.10:
                commentary_parts.append("The risk profile is acceptable with standard quality controls in place.")
        
        if not commentary_parts:
            commentary_parts.append("The method requires further characterization to fully assess its suitability.")
        
        return " ".join(commentary_parts)
    
    def _calculate_reasoning_confidence(self, reasoning_steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence in the reasoning."""
        if not reasoning_steps:
            return 0.5
        
        confidences = [step.confidence for step in reasoning_steps]
        return np.mean(confidences)
    
    def _load_chromatography_knowledge(self) -> Dict[str, Any]:
        """Load domain knowledge about chromatography."""
        return {
            "critical_parameters": {
                "pH": {"range": (2, 8), "impact": "high", "typical_control": "±0.05"},
                "temperature": {"range": (25, 60), "impact": "medium", "typical_control": "±2°C"},
                "gradient_time": {"range": (5, 60), "impact": "high", "typical_control": "±2%"},
                "flow_rate": {"range": (0.1, 2.0), "impact": "medium", "typical_control": "±5%"},
                "organic_modifier": {"range": (5, 95), "impact": "high", "typical_control": "±1%"},
                "buffer_concentration": {"range": (5, 50), "impact": "medium", "typical_control": "±5%"}
            },
            "typical_constraints": {
                "resolution": {"minimum": 1.5, "target": 2.0, "critical": True},
                "runtime": {"maximum": 30, "target": 15, "critical": False},
                "pressure": {"maximum": 400, "critical": True},
                "tailing_factor": {"maximum": 2.0, "target": 1.2, "critical": False},
                "theoretical_plates": {"minimum": 2000, "target": 5000, "critical": False}
            },
            "aqbd_principles": [
                "Design space should be established through multivariate experiments",
                "Critical method parameters must be identified and controlled",
                "Method operable design region should be defined with guard bands",
                "Control strategy should address all sources of variability"
            ],
            "regulatory_references": {
                "ICH_Q2_R2": "Validation of Analytical Procedures",
                "ICH_Q14": "Analytical Procedure Development",
                "ICH_Q9": "Quality Risk Management",
                "USP_1225": "Validation of Compendial Procedures",
                "USP_1210": "Statistical Tools for Procedure Validation"
            }
        }
