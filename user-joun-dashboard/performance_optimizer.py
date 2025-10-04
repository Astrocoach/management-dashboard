"""High-performance data processing utilities using simdjson, aiohttp, and optional Polars.

This module is part of the core architecture. It provides:
- optimized_data_processing: fast JSON/CSV parsing
- async_fetcher: asynchronous HTTP fetcher
- benchmark_function: simple benchmarking decorator
"""
from typing import Any, List, Dict, Union, Optional
import time
import pandas as pd

# Optional high-performance libraries
try:
    import simdjson  # pysimdjson
    _SIMDJSON_AVAILABLE = True
    _parser = simdjson.Parser()
except Exception:
    simdjson = None
    _SIMDJSON_AVAILABLE = False
    _parser = None

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except Exception:
    pl = None
    _POLARS_AVAILABLE = False

# Optional async HTTP client
try:
    import asyncio
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except Exception:
    asyncio = None
    aiohttp = None
    _AIOHTTP_AVAILABLE = False


def benchmark_function(func):
    """Decorator to measure execution time of a function.
    Returns (result, elapsed_seconds) if used as a wrapper; otherwise logs internally.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper


class AsyncFetcher:
    """Asynchronous JSON fetcher. Returns list payload from {'data': [...]} or raw JSON."""
    async def _get(self, url: str, timeout: int = 30) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        if not _AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available")
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                # Prefer bytes for simdjson
                content_bytes = await resp.read()
                if _SIMDJSON_AVAILABLE and isinstance(content_bytes, (bytes, bytearray)):
                    doc = _parser.parse(content_bytes)
                    # doc behaves like dict; access 'data' if present
                    try:
                        payload = doc.get('data', None)
                    except Exception:
                        payload = None
                    if payload is None:
                        # Fallback: convert whole document
                        try:
                            payload = simdjson.loads(content_bytes.decode('utf-8'))
                        except Exception:
                            payload = {}
                    return payload if isinstance(payload, list) else payload
                else:
                    # Fallback to text-based json
                    text = content_bytes.decode('utf-8', errors='replace')
                    try:
                        import json as _json
                        return _json.loads(text).get('data', [])
                    except Exception:
                        return []

    def fetch_data(self, url: str, retries: int = 3, timeout: int = 30) -> List[Dict[str, Any]]:
        """Synchronous wrapper around async fetch. Returns list of dicts."""
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                if _AIOHTTP_AVAILABLE and asyncio is not None:
                    return asyncio.run(self._get(url, timeout))  # type: ignore[arg-type]
                else:
                    raise RuntimeError("Async HTTP not available")
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        # Final fallback to requests
        try:
            import requests
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get('data', []) if isinstance(data, dict) else []
        except Exception:
            # As a last resort, return empty list
            return []


# Provide a module-level instance for convenience (matches main.py import pattern)
async_fetcher = AsyncFetcher()


def _to_pandas(records: Union[List[Dict[str, Any]], Dict[str, Any]]) -> pd.DataFrame:
    if isinstance(records, list):
        return pd.DataFrame(records)
    if isinstance(records, dict):
        # If dict contains 'data', prefer it
        if 'data' in records and isinstance(records['data'], list):
            return pd.DataFrame(records['data'])
        return pd.DataFrame([records])
    return pd.DataFrame()


def optimized_data_processing(input_data: Any, file_type: str = 'json') -> pd.DataFrame:
    """High-performance data processing.
    - JSON: Uses simdjson where available. Accepts list[dict], str, bytes.
    - CSV: Uses Polars if available, otherwise optimized pandas.
    Returns pandas DataFrame for compatibility with the app.
    """
    if file_type.lower() == 'json':
        # Already Python-native list/dict
        if isinstance(input_data, (list, dict)):
            return _to_pandas(input_data)
        # Bytes: prefer simdjson Parser
        if isinstance(input_data, (bytes, bytearray)) and _SIMDJSON_AVAILABLE:
            doc = _parser.parse(input_data)
            try:
                # Try to get 'data' list first
                payload = doc.get('data', None)
            except Exception:
                payload = None
            if payload is None:
                # Fallback to loads on decoded string
                try:
                    payload = simdjson.loads(input_data.decode('utf-8'))
                except Exception:
                    payload = []
            return _to_pandas(payload)
        # String JSON
        if isinstance(input_data, str):
            try:
                if _SIMDJSON_AVAILABLE:
                    payload = simdjson.loads(input_data)
                else:
                    import json as _json
                    payload = _json.loads(input_data)
            except Exception:
                payload = []
            return _to_pandas(payload)
        # Unknown format
        return pd.DataFrame()

    # CSV path or file-like
    if file_type.lower() == 'csv':
        # Polars fast path
        if _POLARS_AVAILABLE:
            try:
                df_pl = pl.read_csv(input_data)
                return df_pl.to_pandas()
            except Exception:
                pass
        # Optimized pandas fallback
        try:
            return pd.read_csv(input_data, low_memory=False)
        except Exception:
            return pd.DataFrame()

    # Default fallback
    return pd.DataFrame()


# Simple placeholder to match imported name in main.py
class performance_optimizer:  # noqa: N801 (keep name to match import)
    optimized_data_processing = staticmethod(optimized_data_processing)
    async_fetcher = async_fetcher
    benchmark_function = staticmethod(benchmark_function)


__all__ = [
    'optimized_data_processing',
    'async_fetcher',
    'benchmark_function',
    'performance_optimizer',
]