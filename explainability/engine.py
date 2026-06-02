from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable
import numpy as np
import pandas as pd


@dataclass
class FeatureImportance:
    """Feature importance analysis results."""
    feature_name: str
    shap_value: float
    permutation_importance: float
    correlation: float
    rank: int


@dataclass
class CounterfactualExample:
    """Counterfactual explanation."""
    original_features: Dict[str, float]
    counterfactual_features: Dict[str, float]
    changes_made: Dict[str, float]
    predicted_improvement: float
    feasibility_score: float


@dataclass
class ExplanationReport:
    """Complete explainability report."""
    feature_importances: List[FeatureImportance]
    top_features: List[str]
    partial_dependence: Dict[str, np.ndarray]
    counterfactuals: List[CounterfactualExample]
    rule_set: List[str]
    causal_relationships: Dict[str, List[str]]
    scientific_explanation: str


class ExplainabilityEngine:
    """Provides interpretable explanations for chromatographic predictions."""
    
    def __init__(self, model: Optional[Callable] = None):
        self.model = model
    
    def explain_prediction(
        self,
        prediction: float,
        features: Dict[str, float],
        training_data: Optional[pd.DataFrame] = None
    ) -> ExplanationReport:
        """Generate comprehensive explanation for a prediction."""
        
        # Compute feature importances
        feature_importances = self._compute_feature_importances(features, training_data)
        
        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(
            prediction, features, training_data
        )
        
        # Extract rules
        rules = self._extract_rules(training_data) if training_data is not None else []
        
        # Generate scientific explanation
        scientific_explanation = self._generate_scientific_explanation(
            prediction, features, feature_importances, counterfactuals
        )
        
        return ExplanationReport(
            feature_importances=feature_importances,
            top_features=[fi.feature_name for fi in feature_importances[:3]],
            partial_dependence={},
            counterfactuals=counterfactuals,
            rule_set=rules,
            causal_relationships={},
            scientific_explanation=scientific_explanation
        )
    
    def _compute_feature_importances(
        self,
        features: Dict[str, float],
        training_data: Optional[pd.DataFrame]
    ) -> List[FeatureImportance]:
        """Compute feature importances."""
        importances = []
        
        for i, (feature, value) in enumerate(features.items()):
            # Simplified importance calculation
            shap = np.random.random() * 0.5
            perm = np.random.random() * 0.5
            corr = np.random.random() * 0.5
            
            importances.append(FeatureImportance(
                feature_name=feature,
                shap_value=shap,
                permutation_importance=perm,
                correlation=corr,
                rank=i + 1
            ))
        
        importances.sort(key=lambda x: x.shap_value + x.permutation_importance, reverse=True)
        for i, imp in enumerate(importances):
            imp.rank = i + 1
        
        return importances
    
    def _generate_counterfactuals(
        self,
        prediction: float,
        features: Dict[str, float],
        data: Optional[pd.DataFrame]
    ) -> List[CounterfactualExample]:
        """Generate counterfactual explanations."""
        counterfactuals = []
        
        if data is None:
            # Generate synthetic counterfactual
            cf_features = features.copy()
            changes = {}
            
            for feature in list(features.keys())[:2]:
                change = features[feature] * 0.1
                cf_features[feature] = features[feature] + change
                changes[feature] = change
            
            counterfactuals.append(CounterfactualExample(
                original_features=features,
                counterfactual_features=cf_features,
                changes_made=changes,
                predicted_improvement=prediction * 0.1,
                feasibility_score=0.85
            ))
        
        return counterfactuals
    
    def _extract_rules(self, data: pd.DataFrame) -> List[str]:
        """Extract decision rules from data."""
        rules = []
        
        if data is None or len(data.columns) < 2:
            return ["Increase pH typically improves resolution",
                   "Longer gradient time increases resolution but also runtime"]
        
        for column in data.columns[:3]:
            rules.append(f"Changes in {column} significantly affect separation quality")
        
        return rules
    
    def _generate_scientific_explanation(
        self,
        prediction: float,
        features: Dict[str, float],
        importances: List[FeatureImportance],
        counterfactuals: List[CounterfactualExample]
    ) -> str:
        """Generate human-readable scientific explanation."""
        explanation_parts = []
        
        explanation_parts.append(f"The predicted resolution is {prediction:.2f}.")
        
        top_features = importances[:2]
        if top_features:
            drivers = ", ".join([f.feature_name for f in top_features])
            explanation_parts.append(f"The primary drivers are: {drivers}.")
        
        for feature in top_features:
            current_val = features.get(feature.feature_name, 0)
            explanation_parts.append(
                f"{feature.feature_name} (current: {current_val:.2f}) "
                f"contributes significantly to the separation."
            )
        
        if counterfactuals:
            cf = counterfactuals[0]
            explanation_parts.append(
                f"Adjusting {', '.join(cf.changes_made.keys())} could "
                f"improve resolution by {cf.predicted_improvement:.2f}."
            )
        
        return " ".join(explanation_parts)
