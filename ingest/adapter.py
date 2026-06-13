"""
Enhanced data ingestion adapter with flexible CSV structure support
"""

import json
import csv
import io
import os
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Iterator
from datetime import datetime
import pandas as pd
import numpy as np

# FIXED: Correct import path for schemas
try:
    from elusight.schemas.base import MethodResult, MethodVariables, MethodObjectives, ChromatographicPlatform
except ImportError:
    # Fallback for direct execution
    from schemas.base import MethodResult, MethodVariables, MethodObjectives, ChromatographicPlatform


class DataAdapter:
    """
    Base adapter for data ingestion.
    Handles conversion from various formats to MethodResult objects.
    """
    
    @staticmethod
    def detect_format(file_path: Union[str, Path]) -> str:
        """Detect file format based on extension."""
        ext = Path(file_path).suffix.lower()
        format_map = {
            '.json': 'json',
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.parquet': 'parquet',
            '.pkl': 'pickle',
            '.pkl.gz': 'pickle'
        }
        return format_map.get(ext, 'unknown')
    
    @staticmethod
    def from_json(
        file_path: Union[str, Path],
        method_key: Optional[str] = None,
        variables_mapping: Optional[Dict[str, str]] = None,
        objectives_mapping: Optional[Dict[str, str]] = None
    ) -> List[MethodResult]:
        """Load methods from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if method_key and method_key in data:
            data = data[method_key]
        elif 'methods' in data:
            data = data['methods']
        elif 'results' in data:
            data = data['results']
        elif isinstance(data, dict):
            data = [data]
        
        return DataAdapter._parse_methods(
            data, 
            variables_mapping=variables_mapping,
            objectives_mapping=objectives_mapping
        )
    
    @staticmethod
    def from_json_string(
        json_string: str,
        variables_mapping: Optional[Dict[str, str]] = None,
        objectives_mapping: Optional[Dict[str, str]] = None
    ) -> List[MethodResult]:
        """Load methods from JSON string."""
        data = json.loads(json_string)
        
        if 'methods' in data:
            data = data['methods']
        elif 'results' in data:
            data = data['results']
        elif isinstance(data, dict):
            data = [data]
        
        return DataAdapter._parse_methods(
            data,
            variables_mapping=variables_mapping,
            objectives_mapping=objectives_mapping
        )
    
    @staticmethod
    def from_csv(
        file_path: Union[str, Path],
        method_id_col: Optional[str] = None,
        variable_prefix: str = 'var_',
        objective_prefix: str = 'obj_',
        **kwargs
    ) -> List[MethodResult]:
        """Load methods from CSV file."""
        df = pd.read_csv(file_path, **kwargs)
        return DataAdapter._parse_dataframe(
            df,
            method_id_col=method_id_col,
            variable_prefix=variable_prefix,
            objective_prefix=objective_prefix
        )
    
    @staticmethod
    def from_excel(
        file_path: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        method_id_col: Optional[str] = None,
        variable_prefix: str = 'var_',
        objective_prefix: str = 'obj_',
        **kwargs
    ) -> List[MethodResult]:
        """Load methods from Excel file."""
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        return DataAdapter._parse_dataframe(
            df,
            method_id_col=method_id_col,
            variable_prefix=variable_prefix,
            objective_prefix=objective_prefix
        )
    
    @staticmethod
    def from_parquet(
        file_path: Union[str, Path],
        method_id_col: Optional[str] = None,
        variable_prefix: str = 'var_',
        objective_prefix: str = 'obj_'
    ) -> List[MethodResult]:
        """Load methods from Parquet file."""
        df = pd.read_parquet(file_path)
        return DataAdapter._parse_dataframe(
            df,
            method_id_col=method_id_col,
            variable_prefix=variable_prefix,
            objective_prefix=objective_prefix
        )
    
    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        method_id_col: Optional[str] = None,
        variable_prefix: str = 'var_',
        objective_prefix: str = 'obj_'
    ) -> List[MethodResult]:
        """Load methods from pandas DataFrame."""
        return DataAdapter._parse_dataframe(
            df,
            method_id_col=method_id_col,
            variable_prefix=variable_prefix,
            objective_prefix=objective_prefix
        )
    
    @staticmethod
    def from_sql(
        connection_string: str,
        query: str,
        method_id_col: Optional[str] = None,
        variable_prefix: str = 'var_',
        objective_prefix: str = 'obj_'
    ) -> List[MethodResult]:
        """Load methods from SQL database."""
        import sqlalchemy as sa
        engine = sa.create_engine(connection_string)
        df = pd.read_sql(query, engine)
        return DataAdapter._parse_dataframe(
            df,
            method_id_col=method_id_col,
            variable_prefix=variable_prefix,
            objective_prefix=objective_prefix
        )
    
    @staticmethod
    def from_api(
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        method_key: Optional[str] = None,
        timeout: int = 30
    ) -> List[MethodResult]:
        """Load methods from REST API endpoint."""
        import requests
        
        response = requests.get(
            endpoint,
            params=params,
            headers=headers or {},
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        
        if method_key and method_key in data:
            data = data[method_key]
        elif 'data' in data:
            data = data['data']
        elif 'results' in data:
            data = data['results']
        elif isinstance(data, dict):
            data = [data]
        
        return DataAdapter._parse_methods(data)
    
    @staticmethod
    def from_pickle(
        file_path: Union[str, Path],
        variables_mapping: Optional[Dict[str, str]] = None,
        objectives_mapping: Optional[Dict[str, str]] = None
    ) -> List[MethodResult]:
        """Load methods from pickle file."""
        import pickle
        import gzip
        
        file_path = Path(file_path)
        
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        
        if isinstance(data, dict):
            if 'methods' in data:
                data = data['methods']
            elif 'results' in data:
                data = data['results']
            else:
                data = [data]
        
        return DataAdapter._parse_methods(
            data,
            variables_mapping=variables_mapping,
            objectives_mapping=objectives_mapping
        )
    
    @staticmethod
    def from_directory(
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False,
        **kwargs
    ) -> Iterator[MethodResult]:
        """Iterate through files in directory and yield methods."""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        glob_pattern = f"**/{pattern}" if recursive else pattern
        
        for file_path in directory.glob(glob_pattern):
            if not file_path.is_file():
                continue
            
            format_type = DataAdapter.detect_format(file_path)
            
            if format_type == 'json':
                methods = DataAdapter.from_json(file_path, **kwargs)
            elif format_type == 'csv':
                methods = DataAdapter.from_csv(file_path, **kwargs)
            elif format_type == 'excel':
                methods = DataAdapter.from_excel(file_path, **kwargs)
            elif format_type == 'parquet':
                methods = DataAdapter.from_parquet(file_path, **kwargs)
            elif format_type == 'pickle':
                methods = DataAdapter.from_pickle(file_path, **kwargs)
            else:
                continue
            
            for method in methods:
                yield method
    
    @staticmethod
    def _parse_methods(
        data: List[Dict[str, Any]],
        variables_mapping: Optional[Dict[str, str]] = None,
        objectives_mapping: Optional[Dict[str, str]] = None
    ) -> List[MethodResult]:
        """Parse raw method dictionaries into MethodResult objects."""
        results = []
        
        for item in data:
            variables_raw = item.get('variables', {})
            if not variables_raw:
                variables_raw = {}
                for key, value in item.items():
                    if key not in ['method_id', 'id', 'objectives', 'constraints', 'metadata', 'uncertainties']:
                        if variables_mapping and key in variables_mapping:
                            key = variables_mapping[key]
                        # Only add numeric or string values
                        if isinstance(value, (int, float, str)):
                            variables_raw[key] = value
            
            if variables_mapping:
                variables_raw = {
                    variables_mapping.get(k, k): v 
                    for k, v in variables_raw.items()
                }
            
            # Handle pH naming
            if 'ph' in variables_raw and 'pH' not in variables_raw:
                variables_raw['pH'] = variables_raw.pop('ph')
            
            # Filter only valid fields for MethodVariables
            valid_var_fields = set(MethodVariables.model_fields.keys())
            variables = MethodVariables(**{
                k: v for k, v in variables_raw.items()
                if k in valid_var_fields and v is not None
            })
            
            # Extract objectives
            objectives_raw = item.get('objectives', {})
            
            if objectives_mapping:
                objectives_raw = {
                    objectives_mapping.get(k, k): v 
                    for k, v in objectives_raw.items()
                }
            
            # Filter only valid fields for MethodObjectives
            valid_obj_fields = set(MethodObjectives.model_fields.keys())
            objectives = MethodObjectives(**{
                k: v for k, v in objectives_raw.items()
                if k in valid_obj_fields and v is not None
            })
            
            # Parse platform
            platform = None
            if 'platform' in item:
                try:
                    platform = ChromatographicPlatform(item['platform'].lower())
                except ValueError:
                    pass
            
            # Create MethodResult (skip if no objectives)
            if objectives and any(v is not None for v in objectives.dict().values()):
                result = MethodResult(
                    method_id=item.get('method_id', item.get('id', f"method_{len(results)}")),
                    variables=variables,
                    objectives=objectives,
                    constraints=item.get('constraints'),
                    uncertainties=item.get('uncertainties'),
                    pareto_rank=item.get('pareto_rank'),
                    dominance_count=item.get('dominance_count'),
                    source_framework=item.get('source_framework', item.get('optimizer')),
                    platform=platform,
                    metadata=item.get('metadata', {})
                )
                results.append(result)
        
        return results
    
    @staticmethod
    def _parse_dataframe(
        df: pd.DataFrame,
        method_id_col: Optional[str] = None,
        variable_prefix: str = 'var_',
        objective_prefix: str = 'obj_'
    ) -> List[MethodResult]:
        """Parse DataFrame into MethodResult objects."""
        results = []
        
        # Handle NaN values
        df = df.replace({np.nan: None, pd.NA: None})
        
        # Determine method ID column
        if method_id_col is None:
            for col in ['method_id', 'id', 'Method_ID', 'ID', 'run_id']:
                if col in df.columns:
                    method_id_col = col
                    break
        
        # Determine if using prefix or direct mapping
        has_prefix_columns = any(
            col.lower().startswith(variable_prefix.lower()) or 
            col.lower().startswith(objective_prefix.lower()) 
            for col in df.columns
        )
        
        for idx, row in df.iterrows():
            variables = {}
            objectives = {}
            
            for col in df.columns:
                value = row[col]
                if value is None or pd.isna(value):
                    continue
                
                col_lower = col.lower()
                
                if has_prefix_columns:
                    # Prefix-based detection
                    if col_lower.startswith(variable_prefix.lower()):
                        var_name = col[len(variable_prefix):]
                        if var_name.lower() == 'ph':
                            var_name = 'pH'
                        try:
                            variables[var_name] = float(value) if isinstance(value, (int, float)) else value
                        except (ValueError, TypeError):
                            variables[var_name] = value
                    
                    elif col_lower.startswith(objective_prefix.lower()):
                        obj_name = col[len(objective_prefix):]
                        try:
                            objectives[obj_name] = float(value) if isinstance(value, (int, float)) else value
                        except (ValueError, TypeError):
                            objectives[obj_name] = value
                
                else:
                    # Keyword-based detection
                    objective_keywords = [
                        'resolution', 'res', 'runtime', 'run_time', 'time', 'rt',
                        'robustness', 'robust', 'tailing', 'tail', 'asymmetry',
                        'peak', 'area', 'plate', 'efficiency', 'capacity',
                        'selectivity', 'retention', 'symmetry', 'signal', 'noise', 'snr'
                    ]
                    
                    if col == method_id_col:
                        continue
                    elif any(keyword in col_lower for keyword in objective_keywords):
                        try:
                            objectives[col] = float(value) if isinstance(value, (int, float)) else value
                        except (ValueError, TypeError):
                            objectives[col] = value
                    else:
                        # Assume it's a variable
                        if isinstance(value, (int, float)):
                            variables[col] = value
                        elif isinstance(value, str):
                            variables[col] = value
            
            # Determine method ID
            if method_id_col and method_id_col in row and row[method_id_col] is not None:
                method_id = str(row[method_id_col])
            else:
                method_id = f"Method_{idx+1:04d}"
            
            # Create MethodResult (skip if no valid objectives)
            if objectives:
                try:
                    result = MethodResult(
                        method_id=method_id,
                        variables=MethodVariables(**variables),
                        objectives=MethodObjectives(**objectives)
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Warning: Could not create method {method_id}: {e}")
                    continue
        
        return results


class StreamingAdapter:
    """Adapter for streaming large datasets."""
    
    @staticmethod
    def stream_json(
        file_path: Union[str, Path],
        chunk_size: int = 1000
    ) -> Iterator[List[MethodResult]]:
        """Stream JSON file in chunks."""
        import ijson
        
        chunk = []
        
        with open(file_path, 'rb') as f:
            parser = ijson.items(f, 'methods.item')
            
            for method_data in parser:
                parsed_methods = DataAdapter._parse_methods([method_data])
                if parsed_methods:
                    chunk.extend(parsed_methods)
                
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
        
        if chunk:
            yield chunk
    
    @staticmethod
    def stream_csv(
        file_path: Union[str, Path],
        chunk_size: int = 1000,
        **kwargs
    ) -> Iterator[List[MethodResult]]:
        """Stream CSV file in chunks."""
        for chunk_df in pd.read_csv(file_path, chunksize=chunk_size, **kwargs):
            methods = DataAdapter._parse_dataframe(chunk_df)
            if methods:
                yield methods


class OptimizerOutputAdapter:
    """Adapter for specific optimizer output formats."""
    
    @staticmethod
    def from_botorch(
        experiment_data: Dict[str, Any],
        parameter_names: List[str],
        objective_names: List[str]
    ) -> List[MethodResult]:
        """Convert BoTorch optimization output."""
        methods = []
        
        candidates = experiment_data.get('candidates', [])
        observations = experiment_data.get('observations', [])
        
        for i, candidate in enumerate(candidates):
            variables = {}
            for j, param_name in enumerate(parameter_names):
                if j < len(candidate):
                    variables[param_name] = float(candidate[j])
            
            objectives = {}
            if i < len(observations):
                for j, obj_name in enumerate(objective_names):
                    if j < len(observations[i]):
                        objectives[obj_name] = float(observations[i][j])
            
            method = MethodResult(
                method_id=f"botorch_{i}",
                variables=MethodVariables(**variables),
                objectives=MethodObjectives(**objectives),
                source_framework="botorch"
            )
            methods.append(method)
        
        return methods
    
    @staticmethod
    def from_optuna(
        study_data: Dict[str, Any],
        variable_names: Optional[List[str]] = None
    ) -> List[MethodResult]:
        """Convert Optuna optimization output."""
        methods = []
        
        trials = study_data.get('trials', [])
        
        for i, trial in enumerate(trials):
            params = trial.get('params', {})
            
            if variable_names is None:
                variable_names = list(params.keys())
            
            variables = {}
            for var_name in variable_names:
                if var_name in params:
                    variables[var_name] = params[var_name]
            
            value = trial.get('value')
            objectives = {}
            if value is not None:
                if isinstance(value, (list, tuple)):
                    for j, v in enumerate(value):
                        objectives[f"objective_{j}"] = float(v)
                else:
                    objectives['value'] = float(value)
            
            method = MethodResult(
                method_id=f"optuna_{i}",
                variables=MethodVariables(**variables),
                objectives=MethodObjectives(**objectives),
                source_framework="optuna"
            )
            methods.append(method)
        
        return methods
    
    @staticmethod
    def from_pymoo(
        result_data: Dict[str, Any],
        variable_names: List[str],
        objective_names: List[str]
    ) -> List[MethodResult]:
        """Convert pymoo optimization output."""
        methods = []
        
        X = result_data.get('X', [])
        F = result_data.get('F', [])
        
        for i, solution in enumerate(X):
            variables = {}
            for j, var_name in enumerate(variable_names):
                if j < len(solution):
                    variables[var_name] = float(solution[j])
            
            objectives = {}
            if i < len(F):
                for j, obj_name in enumerate(objective_names):
                    if j < len(F[i]):
                        objectives[obj_name] = float(F[i][j])
            
            method = MethodResult(
                method_id=f"pymoo_{i}",
                variables=MethodVariables(**variables),
                objectives=MethodObjectives(**objectives),
                source_framework="pymoo",
                pareto_rank=result_data.get('rank', [None])[i] if 'rank' in result_data else None
            )
            methods.append(method)
        
        return methods
    
    @staticmethod
    def from_drylab(
        export_data: Dict[str, Any]
    ) -> List[MethodResult]:
        """Convert DryLab export output."""
        methods = []
        
        methods_data = export_data.get('methods', export_data.get('scouting_runs', []))
        
        for i, method_data in enumerate(methods_data):
            variables = {}
            
            if 'pH' in method_data:
                variables['pH'] = method_data['pH']
            if 'gradient_time' in method_data:
                variables['gradient_time'] = method_data['gradient_time']
            if 'temperature' in method_data:
                variables['temperature'] = method_data['temperature']
            if 'flow_rate' in method_data:
                variables['flow_rate'] = method_data['flow_rate']
            
            objectives = {}
            if 'resolution' in method_data:
                objectives['resolution'] = method_data['resolution']
            if 'runtime' in method_data:
                objectives['runtime'] = method_data['runtime']
            
            if objectives:
                method = MethodResult(
                    method_id=method_data.get('name', f"drylab_{i}"),
                    variables=MethodVariables(**variables),
                    objectives=MethodObjectives(**objectives),
                    source_framework="drylab"
                )
                methods.append(method)
        
        return methods


class BatchAdapter:
    """Adapter for batch processing of multiple data sources."""
    
    def __init__(self):
        self.sources = []
        self.results = []
    
    def add_json(self, file_path: Union[str, Path], **kwargs):
        self.sources.append(('json', file_path, kwargs))
        return self
    
    def add_csv(self, file_path: Union[str, Path], **kwargs):
        self.sources.append(('csv', file_path, kwargs))
        return self
    
    def add_excel(self, file_path: Union[str, Path], **kwargs):
        self.sources.append(('excel', file_path, kwargs))
        return self
    
    def add_parquet(self, file_path: Union[str, Path], **kwargs):
        self.sources.append(('parquet', file_path, kwargs))
        return self
    
    def add_dataframe(self, df: pd.DataFrame, **kwargs):
        self.sources.append(('dataframe', df, kwargs))
        return self
    
    def process(self) -> List[MethodResult]:
        all_methods = []
        
        for source_type, source_data, kwargs in self.sources:
            if source_type == 'json':
                methods = DataAdapter.from_json(source_data, **kwargs)
            elif source_type == 'csv':
                methods = DataAdapter.from_csv(source_data, **kwargs)
            elif source_type == 'excel':
                methods = DataAdapter.from_excel(source_data, **kwargs)
            elif source_type == 'parquet':
                methods = DataAdapter.from_parquet(source_data, **kwargs)
            elif source_type == 'dataframe':
                methods = DataAdapter.from_dataframe(source_data, **kwargs)
            else:
                continue
            
            all_methods.extend(methods)
        
        self.results = all_methods
        return all_methods
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_sources': len(self.sources),
            'total_methods': len(self.results),
            'sources': [s[0] for s in self.sources]
        }
