from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class Tradeoff:
    """Single trade-off between two methods."""
    method_a_id: str
    method_b_id: str
    gain: Dict[str, float]
    cost: Dict[str, float]
    net_benefit: float
    dominance_relationship: str


@dataclass
class TradeoffReport:
    """Complete trade-off analysis report."""
    pareto_front_methods: List[str]
    dominance_matrix: Dict[str, Dict[str, bool]]
    tradeoffs: List[Tradeoff]
    best_methods_by_objective: Dict[str, str]
    knee_points: List[str]
    diversity_metrics: Dict[str, float]
    scientific_interpretation: str


class TradeoffEngine:
    """Interprets trade-offs in multi-objective chromatographic optimization."""
    
    def __init__(self, objectives: List[str], directions: List[str]):
        self.objectives = objectives
        self.directions = directions
        self.n_objectives = len(objectives)
    
    def analyze_tradeoffs(self, methods: List[Dict[str, Any]]) -> TradeoffReport:
        """Analyze trade-offs among multiple methods."""
        
        # Compute Pareto front
        pareto_methods = self._compute_pareto_front(methods)
        
        # Compute dominance matrix
        dominance_matrix = self._compute_dominance_matrix(methods)
        
        # Identify trade-offs
        tradeoffs = self._identify_tradeoffs(methods, pareto_methods)
        
        # Find best methods per objective
        best_methods = self._best_methods_per_objective(methods)
        
        # Find knee points
        knee_points = self._find_knee_points(pareto_methods, methods)
        
        # Compute diversity metrics
        diversity = self._compute_diversity_metrics(pareto_methods, methods)
        
        # Generate scientific interpretation
        interpretation = self._generate_interpretation(
            pareto_methods, tradeoffs, best_methods
        )
        
        return TradeoffReport(
            pareto_front_methods=[m['id'] for m in pareto_methods],
            dominance_matrix=dominance_matrix,
            tradeoffs=tradeoffs[:10],
            best_methods_by_objective=best_methods,
            knee_points=knee_points,
            diversity_metrics=diversity,
            scientific_interpretation=interpretation
        )
    
    def _compute_pareto_front(self, methods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute Pareto front using non-dominated sorting."""
        pareto_front = []
        
        for i, method in enumerate(methods):
            is_dominated = False
            
            for j, other in enumerate(methods):
                if i == j:
                    continue
                
                better_in_all = True
                strictly_better_in_one = False
                
                for obj, direction in zip(self.objectives, self.directions):
                    val_i = method['objectives'].get(obj, 0)
                    val_j = other['objectives'].get(obj, 0)
                    
                    if direction == 'max':
                        better = val_j > val_i
                        worse = val_j < val_i
                    else:
                        better = val_j < val_i
                        worse = val_j > val_i
                    
                    if worse:
                        better_in_all = False
                        break
                    if better:
                        strictly_better_in_one = True
                
                if better_in_all and strictly_better_in_one:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(method)
        
        return pareto_front
    
    def _compute_dominance_matrix(
        self, methods: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, bool]]:
        """Compute pairwise dominance relationships."""
        matrix = {}
        
        for i, method_i in enumerate(methods):
            matrix[method_i['id']] = {}
            for j, method_j in enumerate(methods):
                if i == j:
                    matrix[method_i['id']][method_j['id']] = False
                    continue
                
                dominates = True
                for obj, direction in zip(self.objectives, self.directions):
                    val_i = method_i['objectives'].get(obj, 0)
                    val_j = method_j['objectives'].get(obj, 0)
                    
                    if direction == 'max':
                        if val_i <= val_j:
                            dominates = False
                            break
                    else:
                        if val_i >= val_j:
                            dominates = False
                            break
                
                matrix[method_i['id']][method_j['id']] = dominates
        
        return matrix
    
    def _identify_tradeoffs(
        self, methods: List[Dict[str, Any]], pareto_front: List[Dict[str, Any]]
    ) -> List[Tradeoff]:
        """Identify significant trade-offs between methods."""
        tradeoffs = []
        
        for i, method_a in enumerate(pareto_front):
            for j, method_b in enumerate(pareto_front):
                if i >= j:
                    continue
                
                gains = {}
                costs = {}
                
                for obj, direction in zip(self.objectives, self.directions):
                    val_a = method_a['objectives'].get(obj, 0)
                    val_b = method_b['objectives'].get(obj, 0)
                    
                    if direction == 'max':
                        if val_b > val_a:
                            gains[obj] = val_b - val_a
                        elif val_a > val_b:
                            costs[obj] = val_a - val_b
                    else:
                        if val_b < val_a:
                            gains[obj] = val_a - val_b
                        elif val_a < val_b:
                            costs[obj] = val_b - val_a
                
                total_gain = sum(gains.values())
                total_cost = sum(costs.values())
                net_benefit = total_gain - total_cost
                
                if total_gain > 0 and total_cost == 0:
                    relationship = "dominant"
                elif total_cost > 0 and total_gain == 0:
                    relationship = "dominated"
                else:
                    relationship = "non-dominated"
                
                tradeoffs.append(Tradeoff(
                    method_a_id=method_a['id'],
                    method_b_id=method_b['id'],
                    gain=gains,
                    cost=costs,
                    net_benefit=net_benefit,
                    dominance_relationship=relationship
                ))
        
        tradeoffs.sort(key=lambda x: abs(x.net_benefit), reverse=True)
        return tradeoffs
    
    def _best_methods_per_objective(self, methods: List[Dict[str, Any]]) -> Dict[str, str]:
        """Find the best method for each objective."""
        best_methods = {}
        
        for obj, direction in zip(self.objectives, self.directions):
            best_value = -np.inf if direction == 'max' else np.inf
            best_method = None
            
            for method in methods:
                value = method['objectives'].get(obj, 0)
                
                if direction == 'max':
                    if value > best_value:
                        best_value = value
                        best_method = method['id']
                else:
                    if value < best_value:
                        best_value = value
                        best_method = method['id']
            
            best_methods[obj] = best_method
        
        return best_methods
    
    def _find_knee_points(
        self, pareto_front: List[Dict[str, Any]], all_methods: List[Dict[str, Any]]
    ) -> List[str]:
        """Find knee points on the Pareto front."""
        if len(pareto_front) < 3:
            return [m['id'] for m in pareto_front]
        
        # Normalize objectives
        normalized_points = []
        for method in pareto_front:
            point = []
            for obj, direction in zip(self.objectives, self.directions):
                values = [m['objectives'].get(obj, 0) for m in all_methods]
                val = method['objectives'].get(obj, 0)
                
                if direction == 'max':
                    norm_val = (val - min(values)) / (max(values) - min(values))
                else:
                    norm_val = (max(values) - val) / (max(values) - min(values))
                
                point.append(norm_val)
            
            normalized_points.append(point)
        
        # Compute utility (distance to ideal point - distance to worst point)
        ideal_point = [1.0] * self.n_objectives
        worst_point = [0.0] * self.n_objectives
        
        utilities = []
        for point in normalized_points:
            dist_to_ideal = np.linalg.norm(np.array(point) - np.array(ideal_point))
            dist_to_worst = np.linalg.norm(np.array(point) - np.array(worst_point))
            utility = dist_to_worst - dist_to_ideal
            utilities.append(utility)
        
        threshold = np.percentile(utilities, 70)
        knee_indices = [i for i, u in enumerate(utilities) if u >= threshold]
        
        return [pareto_front[i]['id'] for i in knee_indices]
    
    def _compute_diversity_metrics(
        self, pareto_front: List[Dict[str, Any]], all_methods: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute diversity metrics of the Pareto front."""
        if len(pareto_front) < 2:
            return {'spread': 0.0, 'uniformity': 0.0}
        
        spreads = []
        for obj in self.objectives:
            values = [m['objectives'].get(obj, 0) for m in pareto_front]
            spreads.append(max(values) - min(values))
        
        avg_spread = np.mean(spreads)
        
        sorted_front = sorted(pareto_front, key=lambda x: x['objectives'].get(self.objectives[0], 0))
        
        distances = []
        for i in range(len(sorted_front) - 1):
            point1 = [sorted_front[i]['objectives'].get(obj, 0) for obj in self.objectives]
            point2 = [sorted_front[i+1]['objectives'].get(obj, 0) for obj in self.objectives]
            distances.append(np.linalg.norm(np.array(point1) - np.array(point2)))
        
        uniformity = np.std(distances) / np.mean(distances) if distances else 0
        
        return {
            'spread': avg_spread,
            'uniformity': 1.0 - min(1.0, uniformity)
        }
    
    def _generate_interpretation(
        self,
        pareto_front: List[Dict[str, Any]],
        tradeoffs: List[Tradeoff],
        best_methods: Dict[str, str]
    ) -> str:
        """Generate scientific interpretation of trade-offs."""
        interpretation_parts = []
        
        interpretation_parts.append(
            f"The Pareto front contains {len(pareto_front)} non-dominated solutions."
        )
        
        for obj, method_id in best_methods.items():
            interpretation_parts.append(
                f"Method {method_id} achieves the best {obj} performance."
            )
        
        if tradeoffs:
            significant_tradeoffs = [t for t in tradeoffs if abs(t.net_benefit) > 0.1]
            if significant_tradeoffs:
                top_tradeoff = significant_tradeoffs[0]
                gain_str = ", ".join([f"{k}+{v:.2f}" for k, v in top_tradeoff.gain.items() if v > 0])
                cost_str = ", ".join([f"{k}-{v:.2f}" for k, v in top_tradeoff.cost.items() if v > 0])
                
                interpretation_parts.append(
                    f"The most significant trade-off exists between methods "
                    f"{top_tradeoff.method_a_id} and {top_tradeoff.method_b_id}: "
                    f"{gain_str} at the cost of {cost_str}."
                )
        
        return " ".join(interpretation_parts)
EOF
