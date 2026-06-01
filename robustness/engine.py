cat > elusight/robustness/engine.py << 'EOF'
"""Robustness evaluation engine for chromatographic methods."""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Callable, Optional
import numpy as np


@dataclass
class RobustnessMetrics:
    """Robustness metrics for a method."""
    pass_probability: float
    failure_probability: float
    robustness_score: float
    critical_parameters: List[str]
    parameter_sensitivities: Dict[str, float]
    worst_case_scenario: Dict[str, float]
    operating_range: Dict[str, Tuple[float, float]]


@dataclass
class RobustnessReport:
    """Complete robustness evaluation report."""
    overall_robustness_score: float
    method_robustness: RobustnessMetrics
    robust_region_size: float
    failure_modes: List[str]
    mitigation_strategies: List[str]
    perturbation_analysis: Dict[str, Dict[str, Any]]


class RobustnessEngine:
    """Evaluates chromatographic method robustness."""
    
    def __init__(
        self,
        perturbation_magnitude: float = 0.1,
        monte_carlo_samples: int = 5000
    ):
        self.perturbation_magnitude = perturbation_magnitude
        self.monte_carlo_samples = monte_carlo_samples
    
    def evaluate_robustness(
        self,
        method_variables: Dict[str, float],
        predictive_model: Optional[Callable] = None,
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> RobustnessReport:
        """Evaluate robustness of a chromatographic method."""
        
        if parameter_bounds is None:
            parameter_bounds = {k: (v*0.8, v*1.2) for k, v in method_variables.items()}
        
        # Monte Carlo simulation
        mc_results = self._monte_carlo_simulation(
            method_variables, predictive_model, parameter_bounds
        )
        
        # Sensitivity analysis
        sensitivities = self._sensitivity_analysis(
            method_variables, predictive_model, parameter_bounds
        )
        
        # Identify critical parameters
        critical_params = self._identify_critical_parameters(sensitivities)
        
        # Compute robustness metrics
        pass_prob = mc_results.get('pass_probability', 0.95)
        robustness_score = pass_prob * (1 - np.std(mc_results.get('performance', [0.95])))
        
        robustness_metrics = RobustnessMetrics(
            pass_probability=pass_prob,
            failure_probability=1 - pass_prob,
            robustness_score=robustness_score,
            critical_parameters=critical_params,
            parameter_sensitivities=sensitivities,
            worst_case_scenario=self._find_worst_case(method_variables, sensitivities),
            operating_range=self._compute_operating_range(
                method_variables, parameter_bounds, sensitivities
            )
        )
        
        return RobustnessReport(
            overall_robustness_score=robustness_score,
            method_robustness=robustness_metrics,
            robust_region_size=self._compute_robust_region_size(sensitivities),
            failure_modes=self._identify_failure_modes(mc_results),
            mitigation_strategies=self._generate_mitigation_strategies(critical_params),
            perturbation_analysis=mc_results.get('perturbations', {})
        )
    
    def _monte_carlo_simulation(
        self,
        variables: Dict[str, float],
        model: Optional[Callable],
        bounds: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Perform Monte Carlo simulation."""
        if model is None:
            # Mock model for demonstration
            return {
                'performance': [0.95] * self.monte_carlo_samples,
                'pass_probability': 0.95,
                'perturbations': {}
            }
        
        results = []
        perturbations = {}
        
        for _ in range(self.monte_carlo_samples):
            perturbed = {}
            for param, value in variables.items():
                if param in bounds:
                    noise = np.random.normal(0, self.perturbation_magnitude * value)
                    perturbed[param] = np.clip(
                        value + noise, bounds[param][0], bounds[param][1]
                    )
                else:
                    perturbed[param] = value
            
            performance = model(perturbed) if model else 0.95
            results.append(performance)
        
        pass_probability = np.mean([1 if p >= 0.9 else 0 for p in results])
        
        return {
            'performance': results,
            'pass_probability': pass_probability,
            'perturbations': perturbations
        }
    
    def _sensitivity_analysis(
        self,
        variables: Dict[str, float],
        model: Optional[Callable],
        bounds: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """Perform sensitivity analysis."""
        if model is None:
            return {k: np.random.random() for k in variables.keys()}
        
        sensitivities = {}
        base_perf = model(variables)
        
        for param, value in variables.items():
            if param not in bounds:
                sensitivities[param] = 0.0
                continue
            
            delta = self.perturbation_magnitude * value
            
            perturbed_up = variables.copy()
            perturbed_up[param] = min(value + delta, bounds[param][1])
            perturbed_down = variables.copy()
            perturbed_down[param] = max(value - delta, bounds[param][0])
            
            perf_up = model(perturbed_up)
            perf_down = model(perturbed_down)
            
            sensitivity = (abs(perf_up - base_perf) + abs(perf_down - base_perf)) / 2
            sensitivities[param] = sensitivity
        
        # Normalize
        max_sens = max(sensitivities.values()) if sensitivities else 1
        return {k: v / max_sens for k, v in sensitivities.items()}
    
    def _identify_critical_parameters(self, sensitivities: Dict[str, float]) -> List[str]:
        """Identify critical parameters based on sensitivity."""
        if not sensitivities:
            return []
        threshold = np.percentile(list(sensitivities.values()), 75)
        return [p for p, s in sensitivities.items() if s >= threshold]
    
    def _find_worst_case(
        self, variables: Dict[str, float], sensitivities: Dict[str, float]
    ) -> Dict[str, float]:
        """Find worst-case scenario."""
        worst_case = variables.copy()
        for param in sensitivities.keys():
            if sensitivities[param] > 0.7:
                worst_case[param] = variables[param] * (1 - self.perturbation_magnitude)
        return worst_case
    
    def _compute_operating_range(
        self,
        variables: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
        sensitivities: Dict[str, float]
    ) -> Dict[str, Tuple[float, float]]:
        """Compute acceptable operating range for each parameter."""
        operating_ranges = {}
        
        for param, value in variables.items():
            if param not in bounds:
                continue
            
            sensitivity = sensitivities.get(param, 0.5)
            allowed_deviation = self.perturbation_magnitude * value * (1 - sensitivity)
            
            operating_ranges[param] = (
                max(bounds[param][0], value - allowed_deviation),
                min(bounds[param][1], value + allowed_deviation)
            )
        
        return operating_ranges
    
    def _compute_robust_region_size(self, sensitivities: Dict[str, float]) -> float:
        """Compute the size of the robust region."""
        if not sensitivities:
            return 0.5
        return np.mean([1 - min(1.0, s) for s in sensitivities.values()])
    
    def _identify_failure_modes(self, mc_results: Dict[str, Any]) -> List[str]:
        """Identify potential failure modes."""
        failure_modes = []
        failure_rate = 1 - mc_results.get('pass_probability', 0.95)
        
        if failure_rate > 0.1:
            failure_modes.append("High probability of performance failure")
        
        return failure_modes
    
    def _generate_mitigation_strategies(self, critical_params: List[str]) -> List[str]:
        """Generate mitigation strategies for critical parameters."""
        strategies = []
        for param in critical_params:
            strategies.append(
                f"Tighten control limits for {param} and consider higher-precision equipment"
            )
        return strategies
EOF