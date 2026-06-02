# elusight/ingest/__init__.py
"""Ingestion module for standardizing optimization outputs."""

import json
from pathlib import Path
from typing import Union, Dict, Any, List, Optional
import pandas as pd
import numpy as np

from schemas.base import (
    MethodResult, MethodVariables, MethodObjectives, ChromatographicPlatform
)


class Ingestor:
    """
    Universal ingestor for chromatographic optimization outputs.
    Supports JSON, CSV, DataFrame, SQL, Parquet, and API sources.
    """
    
    @staticmethod
    def from_json(file_path: Union[str, Path]) -> List[MethodResult]:
        """Ingest from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            if 'methods' in data:
                data = data['methods']
            elif 'results' in data:
                data = data['results']
            else:
                data = [data]
        
        return [Ingestor._parse_method(m) for m in data]
    
    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> List[MethodResult]:
        """Ingest from pandas DataFrame."""
        results = []
        for idx, row in df.iterrows():
            result = Ingestor._parse_dataframe_row(row, idx)
            results.append(result)
        return results
    
    @staticmethod
    def from_csv(file_path: Union[str, Path]) -> List[MethodResult]:
        """Ingest from CSV file."""
        df = pd.read_csv(file_path)
        return Ingestor.from_dataframe(df)
    
    @staticmethod
    def from_parquet(file_path: Union[str, Path]) -> List[MethodResult]:
        """Ingest from Parquet file."""
        df = pd.read_parquet(file_path)
        return Ingestor.from_dataframe(df)
    
    @staticmethod
    def from_sql(connection_string: str, query: str) -> List[MethodResult]:
        """Ingest from SQL database."""
        import sqlalchemy as sa
        engine = sa.create_engine(connection_string)
        df = pd.read_sql(query, engine)
        return Ingestor.from_dataframe(df)
    
    @staticmethod
    def from_api(endpoint: str, params: Optional[Dict[str, Any]] = None) -> List[MethodResult]:
        """Ingest from REST API endpoint."""
        import requests
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
            elif 'results' in data:
                data = data['results']
            else:
                data = [data]
        
        return [Ingestor._parse_method(m) for m in data]
    
    @staticmethod
    def from_optimizer_output(optimizer: str, data: Dict[str, Any]) -> List[MethodResult]:
        """Direct ingestion from optimizer-specific format."""
        from elusight.integrations.adapters import get_adapter
        
        adapter = get_adapter(optimizer)
        adapted_data = adapter.adapt_output(data)
        return [Ingestor._parse_method(m) for m in adapted_data]
    
    @staticmethod
    def _parse_method(data: Dict[str, Any]) -> MethodResult:
        """Parse raw dictionary into MethodResult."""
        # Handle variable naming variations
        variables_raw = data.get('variables', {})
        if 'pH' not in variables_raw and 'ph' in variables_raw:
            variables_raw['pH'] = variables_raw.pop('ph')
        
        variables = MethodVariables(**{
            k: v for k, v in variables_raw.items() 
            if k in MethodVariables.model_fields
        })
        
        objectives = MethodObjectives(**{
            k: v for k, v in data.get('objectives', {}).items()
            if k in MethodObjectives.model_fields
        })
        
        # Parse platform if provided
        platform = None
        if 'platform' in data:
            try:
                platform = ChromatographicPlatform(data['platform'].lower())
            except ValueError:
                platform = None
        
        return MethodResult(
            method_id=data.get('method_id', data.get('id', 'unknown')),
            variables=variables,
            objectives=objectives,
            constraints=data.get('constraints'),
            uncertainties=data.get('uncertainties'),
            pareto_rank=data.get('pareto_rank'),
            dominance_count=data.get('dominance_count'),
            source_framework=data.get('source_framework', data.get('optimizer')),
            platform=platform,
            metadata=data.get('metadata', {})
        )
    
    @staticmethod
    def _parse_dataframe_row(row: pd.Series, idx: int) -> MethodResult:
        """Parse pandas DataFrame row into MethodResult."""
        variables = {}
        objectives = {}
        
        for col, value in row.items():
            if pd.isna(value):
                continue
                
            col_lower = col.lower()
            if col_lower.startswith('var_'):
                var_name = col[4:]
                if var_name.lower() == 'ph':
                    var_name = 'pH'
                variables[var_name] = float(value) if isinstance(value, (int, float)) else value
            elif col_lower.startswith('obj_'):
                objectives[col[4:]] = float(value) if isinstance(value, (int, float)) else value
            elif col_lower == 'method_id':
                method_id = str(value)
            elif col_lower == 'source_framework':
                source_framework = str(value)
        
        return MethodResult(
            method_id=row.get('method_id', idx),
            variables=MethodVariables(**variables),
            objectives=MethodObjectives(**objectives),
            source_framework=row.get('source_framework', 'unknown')
        )
