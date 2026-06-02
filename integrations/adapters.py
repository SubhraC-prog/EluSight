from typing import Dict, Any, List


class OptimizationFrameworkAdapter:
    """Base adapter for optimization frameworks."""
    
    def adapt_output(self, raw_output: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError


class BayesianOptimizationAdapter(OptimizationFrameworkAdapter):
    """Adapter for Bayesian Optimization outputs."""
    
    def adapt_output(self, raw_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        methods = []
        
        for i, params in enumerate(raw_output.get('parameters', [])):
            method = {
                'id': f"BO_{i}",
                'variables': params,
                'objectives': {
                    'resolution': raw_output.get('objectives', {}).get('resolution', [None])[i] if i < len(raw_output.get('objectives', {}).get('resolution', [])) else None,
                    'runtime': raw_output.get('objectives', {}).get('runtime', [None])[i] if i < len(raw_output.get('objectives', {}).get('runtime', [])) else None
                },
                'source_framework': 'bayesian_optimization'
            }
            methods.append(method)
        
        return methods


class NSGAIIAdapter(OptimizationFrameworkAdapter):
    """Adapter for NSGA-II outputs."""
    
    def adapt_output(self, raw_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        methods = []
        
        for i, solution in enumerate(raw_output.get('solutions', [])):
            method = {
                'id': f"NSGA2_{i}",
                'variables': solution.get('variables', {}),
                'objectives': solution.get('objectives', {}),
                'pareto_rank': solution.get('rank'),
                'source_framework': 'nsga_ii'
            }
            methods.append(method)
        
        return methods


class AQbDDoEAdapter(OptimizationFrameworkAdapter):
    """Adapter for AQbD DoE outputs."""
    
    def adapt_output(self, raw_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        methods = []
        
        design_space = raw_output.get('design_space', {})
        operating_point = design_space.get('operating_point', {})
        
        method = {
            'id': 'AQbD_nominal',
            'variables': operating_point.get('variables', {}),
            'objectives': operating_point.get('responses', {}),
            'constraints': raw_output.get('constraints', {}),
            'source_framework': 'aqbd_doe'
        }
        methods.append(method)
        
        return methods


class DryLabAdapter(OptimizationFrameworkAdapter):
    """Adapter for DryLab outputs."""
    
    def adapt_output(self, raw_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        methods = []
        
        for method in raw_output.get('methods', []):
            adapted = {
                'id': method.get('name', f"DryLab_{len(methods)}"),
                'variables': {
                    'pH': method.get('pH'),
                    'gradient_time': method.get('gradient_time'),
                    'temperature': method.get('temperature')
                },
                'objectives': {
                    'resolution': method.get('resolution'),
                    'runtime': method.get('runtime')
                },
                'source_framework': 'drylab'
            }
            methods.append(adapted)
        
        return methods


class GenericOptimizationAdapter(OptimizationFrameworkAdapter):
    """Generic adapter for any optimization framework."""
    
    def adapt_output(self, raw_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        methods = []
        
        if 'methods' in raw_output:
            methods_data = raw_output['methods']
        elif 'solutions' in raw_output:
            methods_data = raw_output['solutions']
        elif 'results' in raw_output:
            methods_data = raw_output['results']
        else:
            methods_data = [raw_output] if isinstance(raw_output, dict) else []
        
        for i, method_data in enumerate(methods_data):
            method = {
                'id': method_data.get('id', method_data.get('method_id', f"Method_{i}")),
                'variables': method_data.get('variables', {}),
                'objectives': method_data.get('objectives', {}),
                'constraints': method_data.get('constraints'),
                'source_framework': method_data.get('source_framework', 'generic')
            }
            methods.append(method)
        
        return methods


def get_adapter(framework: str) -> OptimizationFrameworkAdapter:
    """Get the appropriate adapter for a framework."""
    adapters = {
        'bayesian': BayesianOptimizationAdapter(),
        'bo': BayesianOptimizationAdapter(),
        'nsga2': NSGAIIAdapter(),
        'nsga-ii': NSGAIIAdapter(),
        'aqbd': AQbDDoEAdapter(),
        'doe': AQbDDoEAdapter(),
        'drylab': DryLabAdapter(),
        'drylab': DryLabAdapter(),
    }
    
    return adapters.get(framework.lower(), GenericOptimizationAdapter())
