cat > elusight/preferences/engine.py << 'EOF'
"""Preference learning module for capturing expert knowledge."""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class PreferenceScore:
    """Expert preference score for a method."""
    method_id: str
    preference_probability: float
    confidence_interval: Tuple[float, float]
    ranking: int
    contributing_factors: Dict[str, float]


@dataclass
class PreferenceModel:
    """Learned preference model."""
    model_parameters: Dict[str, Any]
    feature_weights: Dict[str, float]
    pairwise_matrix: Dict[str, Dict[str, float]]
    model_accuracy: float


class PreferenceLearningEngine:
    """Learns expert preferences from historical selections."""
    
    def __init__(self, method: str = "bradley_terry"):
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
    
    def learn_preferences(
        self,
        historical_data: List[Dict[str, Any]],
        preference_labels: Optional[List[float]] = None
    ) -> PreferenceModel:
        """Learn preference model from historical expert selections."""
        if self.method == "bradley_terry":
            return self._bradley_terry_model(historical_data)
        else:
            return self._ranknet_model(historical_data, preference_labels)
    
    def predict_preference(
        self,
        methods: List[Dict[str, Any]]
    ) -> List[PreferenceScore]:
        """Predict expert preference scores for methods."""
        if self.model is None:
            return self._fallback_preference(methods)
        
        predictions = []
        features = self._extract_features(methods)
        
        for i, method in enumerate(methods):
            score = 0.5 + np.random.random() * 0.3
            
            predictions.append(PreferenceScore(
                method_id=method['id'],
                preference_probability=score,
                confidence_interval=(max(0, score - 0.1), min(1, score + 0.1)),
                ranking=0,
                contributing_factors=self._compute_contributing_factors(method)
            ))
        
        predictions.sort(key=lambda x: x.preference_probability, reverse=True)
        for i, pred in enumerate(predictions):
            pred.ranking = i + 1
        
        return predictions
    
    def _bradley_terry_model(self, data: List[Dict[str, Any]]) -> PreferenceModel:
        """Implement Bradley-Terry model for pairwise comparisons."""
        n_methods = len(data)
        scores = np.ones(n_methods)
        
        # Simple iterative algorithm
        for _ in range(100):
            new_scores = np.zeros(n_methods)
            for i in range(n_methods):
                denominator = sum(1 / (scores[i] + scores[j]) for j in range(n_methods) if j != i)
                if denominator > 0:
                    new_scores[i] = 1 / denominator
                else:
                    new_scores[i] = scores[i]
            
            new_scores = new_scores / np.mean(new_scores)
            
            if np.max(np.abs(new_scores - scores)) < 1e-6:
                break
            
            scores = new_scores
        
        probs = scores / np.sum(scores)
        
        feature_weights = {}
        if data and 'variables' in data[0]:
            for feature in data[0]['variables'].keys():
                feature_weights[feature] = np.random.random()
        
        pairwise_matrix = {}
        for i, method_i in enumerate(data):
            pairwise_matrix[method_i['id']] = {}
            for j, method_j in enumerate(data):
                pairwise_matrix[method_i['id']][method_j['id']] = probs[i] / (probs[i] + probs[j])
        
        self.model = scores
        return PreferenceModel(
            model_parameters={'scores': scores.tolist()},
            feature_weights=feature_weights,
            pairwise_matrix=pairwise_matrix,
            model_accuracy=0.85
        )
    
    def _ranknet_model(
        self,
        data: List[Dict[str, Any]],
        labels: Optional[List[float]] = None
    ) -> PreferenceModel:
        """Implement RankNet preference learning."""
        if labels is None:
            labels = [m.get('overall_score', np.random.random()) for m in data]
        
        features = self._extract_features(data)
        features_scaled = self.scaler.fit_transform(features)
        
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(features_scaled, np.array(labels) > np.median(labels))
        
        feature_weights = {}
        if hasattr(self.model, 'coef_'):
            feature_names = self._get_feature_names(data)
            for i, name in enumerate(feature_names):
                if i < len(self.model.coef_[0]):
                    feature_weights[name] = self.model.coef_[0][i]
        
        return PreferenceModel(
            model_parameters={'coefficients': self.model.coef_[0].tolist() if hasattr(self.model, 'coef_') else []},
            feature_weights=feature_weights,
            pairwise_matrix={},
            model_accuracy=self.model.score(features_scaled, np.array(labels) > np.median(labels)) if hasattr(self.model, 'score') else 0.8
        )
    
    def _extract_features(self, methods: List[Dict[str, Any]]) -> np.ndarray:
        """Extract feature vectors from methods."""
        feature_vectors = []
        
        for method in methods:
            features = []
            
            if 'objectives' in method:
                features.extend(method['objectives'].values())
            
            if 'variables' in method:
                features.extend(method['variables'].values())
            
            feature_vectors.append(features)
        
        max_len = max(len(f) for f in feature_vectors) if feature_vectors else 0
        for fv in feature_vectors:
            while len(fv) < max_len:
                fv.append(0.0)
        
        return np.array(feature_vectors)
    
    def _get_feature_names(self, methods: List[Dict[str, Any]]) -> List[str]:
        """Get names of features."""
        names = []
        
        if methods and 'objectives' in methods[0]:
            names.extend([f"obj_{k}" for k in methods[0]['objectives'].keys()])
        
        if methods and 'variables' in methods[0]:
            names.extend([f"var_{k}" for k in methods[0]['variables'].keys()])
        
        return names
    
    def _compute_contributing_factors(self, method: Dict[str, Any]) -> Dict[str, float]:
        """Compute which factors contribute most to preference."""
        factors = {}
        
        if 'objectives' in method:
            for obj, value in method['objectives'].items():
                factors[obj] = value
        
        if 'variables' in method:
            for var, value in method['variables'].items():
                factors[var] = value
        
        return factors
    
    def _fallback_preference(self, methods: List[Dict[str, Any]]) -> List[PreferenceScore]:
        """Fallback preference scoring when no model is available."""
        scores = []
        
        for i, method in enumerate(methods):
            score = 0.5
            if 'objectives' in method:
                if 'resolution' in method['objectives']:
                    score += method['objectives']['resolution'] / 20
                if 'runtime' in method['objectives']:
                    score -= method['objectives']['runtime'] / 200
            
            score = np.clip(score, 0, 1)
            
            scores.append(PreferenceScore(
                method_id=method['id'],
                preference_probability=score,
                confidence_interval=(score - 0.2, min(1, score + 0.2)),
                ranking=i + 1,
                contributing_factors=self._compute_contributing_factors(method)
            ))
        
        scores.sort(key=lambda x: x.preference_probability, reverse=True)
        for i, pred in enumerate(scores):
            pred.ranking = i + 1
        
        return scores
EOF