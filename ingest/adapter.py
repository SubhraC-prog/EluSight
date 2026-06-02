# elusight/ingest/adapter.py

"""
Data ingestion adapters for various data sources and formats.
Provides flexible data loading from JSON, CSV, Excel, Parquet, SQL, and APIs.
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

from schemas.base import MethodResult, MethodVariables, MethodObjectives, ChromatographicPlatform


class DataAdapter:
    """
    Base adapter for data ingestion.
    Handles conversion from various formats to MethodResult objects.
    """
    
    @staticmethod
    def detect_format(file_path: Union[str, Path]) -> str:
        """
        Detect file format based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Format string: 'json', 'csv', 'excel', 'parquet', 'unknown'
        """
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
        """
        Load methods from JSON file.
        
        Args:
            file_path: Path to JSON file
            method_key: Key containing methods list (if nested)
            variables_mapping: Mapping of variable names
            objectives_mapping: Mapping of objective names
            
        Returns:
            List of MethodResult objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract methods from nested structure
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
        """
        Load methods from JSON string.
        
        Args:
            json_string: JSON string containing method data
            variables_mapping: Mapping of variable names
            objectives_mapping: Mapping of objective names
            
        Returns:
            List of MethodResult objects
        """
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
        """
        Load methods from CSV file.
        
        Args:
            file_path: Path to CSV file
            method_id_col: Column name for method ID
            variable_prefix: Prefix for variable columns
            objective_prefix: Prefix for objective columns
            **kwargs: Additional arguments to pd.read_csv
            
        Returns:
            List of MethodResult objects
        """
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
        """
        Load methods from Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name or index
            method_id_col: Column name for method ID
            variable_prefix: Prefix for variable columns
            objective_prefix: Prefix for objective columns
            **kwargs: Additional arguments to pd.read_excel
            
        Returns:
            List of MethodResult objects
        """
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
        """
        Load methods from Parquet file.
        
        Args:
            file_path: Path to Parquet file
            method_id_col: Column name for method ID
            variable_prefix: Prefix for variable columns
            objective_prefix: Prefix for objective columns
            
        Returns:
            List of MethodResult objects
        """
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
        """
        Load methods from pandas DataFrame.
        
        Args:
            df: pandas DataFrame
            method_id_col: Column name for method ID
            variable_prefix: Prefix for variable columns
            objective_prefix: Prefix for objective columns
            
        Returns:
            List of MethodResult objects
        """
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
        """
        Load methods from SQL database.
        
        Args:
            connection_string: SQLAlchemy connection string
            query: SQL query to execute
            method_id_col: Column name for method ID
            variable_prefix: Prefix for variable columns
            objective_prefix: Prefix for objective columns
            
        Returns:
            List of MethodResult objects
        """
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
        """
        Load methods from REST API endpoint.
        
        Args:
            endpoint: API endpoint URL
            params: Query parameters
            headers: Request headers
            method_key: Key containing methods in response
            timeout: Request timeout in seconds
            
        Returns:
            List of MethodResult objects
        """
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
        """
        Load methods from pickle file.
        
        Args:
            file_path: Path to pickle file
            variables_mapping: Mapping of variable names
            objectives_mapping: Mapping of objective names
            
        Returns:
            List of MethodResult objects
        """
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
        """
        Iterate through files in directory and yield methods.
        
        Args:
            directory: Directory path
            pattern: File pattern to match
            recursive: Search recursively
            **kwargs: Additional arguments for specific loaders
            
        Yields:
            MethodResult objects
        """
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
        """
        Parse raw method dictionaries into MethodResult objects.
        
        Args:
            data: List of method dictionaries
            variables_mapping: Mapping for variable names
            objectives_mapping: Mapping for objective names
            
        Returns:
            List of MethodResult objects
        """
        results = []
        
        for item in data:
            # Extract variables
            variables_raw = item.get('variables', {})
            if not variables_raw:
                # Try to extract from top-level keys with variable mapping
                variables_raw = {}
                for key, value in item.items():
                    if key not in ['method_id', 'id', 'objectives', 'constraints', 'metadata']:
                        if variables_mapping and key in variables_mapping:
                            key = variables_mapping[key]
                        variables_raw[key] = value
            
            # Apply variable mapping
            if variables_mapping:
                variables_raw = {
                    variables_mapping.get(k, k): v 
                    for k, v in variables_raw.items()
                }
            
            # Handle pH/pH naming
            if 'ph' in variables_raw and 'pH' not in variables_raw:
                variables_raw['pH'] = variables_raw.pop('ph')
            
            variables = MethodVariables(**{
                k: v for k, v in variables_raw.items()
                if k in MethodVariables.model_fields
            })
            
            # Extract objectives
            objectives_raw = item.get('objectives', {})
            
            # Apply objective mapping
            if objectives_mapping:
                objectives_raw = {
                    objectives_mapping.get(k, k): v 
                    for k, v in objectives_raw.items()
                }
            
            objectives = MethodObjectives(**{
                k: v for k, v in objectives_raw.items()
                if k in MethodObjectives.model_fields
            })
            
            # Parse platform
            platform = None
            if 'platform' in item:
                try:
                    platform = ChromatographicPlatform(item['platform'].lower())
                except ValueError:
                    pass
            
            # Create MethodResult
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
        """
        Parse DataFrame into MethodResult objects.
        
        Args:
            df: pandas DataFrame
            method_id_col: Column name for method ID
            variable_prefix: Prefix for variable columns
            objective_prefix: Prefix for objective columns
            
        Returns:
            List of MethodResult objects
        """
        results = []
        
        # Determine method ID column
        if method_id_col is None:
            if 'method_id' in df.columns:
                method_id_col = 'method_id'
            elif 'id' in df.columns:
                method_id_col = 'id'
            else:
                method_id_col = None
        
        for idx, row in df.iterrows():
            variables = {}
            objectives = {}
            
            for col, value in row.items():
                if pd.isna(value):
                    continue
                
                col_lower = col.lower()
                
                # Check for variable columns
                if col_lower.startswith(variable_prefix.lower()):
                    var_name = col[len(variable_prefix):]
                    if var_name.lower() == 'ph':
                        var_name = 'pH'
                    variables[var_name] = float(value) if isinstance(value, (int, float)) else value
                
                # Check for objective columns
                elif col_lower.startswith(objective_prefix.lower()):
                    obj_name = col[len(objective_prefix):]
                    objectives[obj_name] = float(value) if isinstance(value, (int, float)) else value
                
                # Direct column mapping without prefix (if no prefix columns exist)
                elif not variables and not objectives:
                    # Assume all numeric columns are variables/objectives
                    if isinstance(value, (int, float)):
                        if col_lower in ['resolution', 'runtime', 'robustness']:
                            objectives[col] = value
                        else:
                            variables[col] = value
            
            # Determine method ID
            if method_id_col and method_id_col in row:
                method_id = str(row[method_id_col])
            else:
                method_id = f"method_{idx}"
            
            # Extract platform from metadata if available
            platform = None
            if 'platform' in row and not pd.isna(row['platform']):
                try:
                    platform = ChromatographicPlatform(str(row['platform']).lower())
                except ValueError:
                    pass
            
            result = MethodResult(
                method_id=method_id,
                variables=MethodVariables(**variables),
                objectives=MethodObjectives(**objectives),
                source_framework=row.get('source_framework') if 'source_framework' in row else None,
                platform=platform,
                metadata={'row_index': idx}
            )
            results.append(result)
        
        return results


class StreamingAdapter:
    """
    Adapter for streaming large datasets.
    Handles incremental loading of methods from large files.
    """
    
    @staticmethod
    def stream_json(
        file_path: Union[str, Path],
        chunk_size: int = 1000
    ) -> Iterator[List[MethodResult]]:
        """
        Stream JSON file in chunks for large datasets.
        
        Args:
            file_path: Path to JSON file
            chunk_size: Number of methods per chunk
            
        Yields:
            Chunks of MethodResult objects
        """
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
        """
        Stream CSV file in chunks.
        
        Args:
            file_path: Path to CSV file
            chunk_size: Number of rows per chunk
            **kwargs: Additional arguments to pd.read_csv
            
        Yields:
            Chunks of MethodResult objects
        """
        for chunk_df in pd.read_csv(file_path, chunksize=chunk_size, **kwargs):
            methods = DataAdapter._parse_dataframe(chunk_df)
            if methods:
                yield methods


class OptimizerOutputAdapter:
    """
    Adapter for specific optimizer output formats.
    Converts optimizer-specific outputs to MethodResult objects.
    """
    
    @staticmethod
    def from_botorch(
        experiment_data: Dict[str, Any],
        parameter_names: List[str],
        objective_names: List[str]
    ) -> List[MethodResult]:
        """
        Convert BoTorch optimization output.
        
        Args:
            experiment_data: BoTorch experiment data
            parameter_names: Names of parameters/variables
            objective_names: Names of objectives
            
        Returns:
            List of MethodResult objects
        """
        methods = []
        
        # Extract candidates and observations
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
        """
        Convert Optuna optimization output.
        
        Args:
            study_data: Optuna study data
            variable_names: Names of variables (auto-detected if None)
            
        Returns:
            List of MethodResult objects
        """
        methods = []
        
        trials = study_data.get('trials', [])
        
        for i, trial in enumerate(trials):
            params = trial.get('params', {})
            
            # Auto-detect variable names if not provided
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
        """
        Convert pymoo optimization output.
        
        Args:
            result_data: pymoo result data
            variable_names: Names of variables
            objective_names: Names of objectives
            
        Returns:
            List of MethodResult objects
        """
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
        """
        Convert DryLab export output.
        
        Args:
            export_data: DryLab export dictionary
            
        Returns:
            List of MethodResult objects
        """
        methods = []
        
        # DryLab typical structure
        methods_data = export_data.get('methods', export_data.get('scouting_runs', []))
        
        for i, method_data in enumerate(methods_data):
            variables = {}
            
            # Extract common DryLab parameters
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
            
            method = MethodResult(
                method_id=method_data.get('name', f"drylab_{i}"),
                variables=MethodVariables(**variables),
                objectives=MethodObjectives(**objectives),
                source_framework="drylab"
            )
            methods.append(method)
        
        return methods


class BatchAdapter:
    """
    Adapter for batch processing of multiple data sources.
    """
    
    def __init__(self):
        self.sources = []
        self.results = []
    
    def add_json(self, file_path: Union[str, Path], **kwargs):
        """Add JSON source to batch."""
        self.sources.append(('json', file_path, kwargs))
        return self
    
    def add_csv(self, file_path: Union[str, Path], **kwargs):
        """Add CSV source to batch."""
        self.sources.append(('csv', file_path, kwargs))
        return self
    
    def add_excel(self, file_path: Union[str, Path], **kwargs):
        """Add Excel source to batch."""
        self.sources.append(('excel', file_path, kwargs))
        return self
    
    def add_parquet(self, file_path: Union[str, Path], **kwargs):
        """Add Parquet source to batch."""
        self.sources.append(('parquet', file_path, kwargs))
        return self
    
    def add_dataframe(self, df: pd.DataFrame, **kwargs):
        """Add DataFrame source to batch."""
        self.sources.append(('dataframe', df, kwargs))
        return self
    
    def process(self) -> List[MethodResult]:
        """
        Process all added sources and return combined results.
        
        Returns:
            Combined list of MethodResult objects
        """
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
        """
        Get summary of batch processing.
        
        Returns:
            Summary dictionary
        """
        return {
            'total_sources': len(self.sources),
            'total_methods': len(self.results),
            'sources': [s[0] for s in self.sources]
        }
