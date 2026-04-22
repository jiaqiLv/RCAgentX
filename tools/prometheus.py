"""
Prometheus Tool - Metric Query and Analysis

LangChain tool for querying Prometheus metrics.
Provides methods for instant queries, range queries, and metadata retrieval.
"""

import requests
from typing import Any, Dict, List, Optional
from datetime import datetime
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class PrometheusInput(BaseModel):
    """Input schema for Prometheus tool operations."""
    query: str = Field(..., description="PromQL query string")
    time: Optional[datetime] = Field(None, description="Evaluation timestamp (for range queries)")


class PrometheusTool(BaseTool):
    """
    Tool for querying Prometheus metrics.

    Provides methods for executing PromQL queries against a Prometheus
    server, including instant queries, range queries, and label value
    discovery.

    Attributes:
        name (str): Tool name for LangChain
        description (str): Tool description
        url (str): Prometheus server URL
        timeout (int): Request timeout in seconds

    Example:
        ```python
        prometheus = PrometheusTool(url="http://prometheus:9090")

        # Instant query
        result = prometheus.query("up")

        # Range query
        result = prometheus.query_range(
            "rate(http_requests_total[5m])",
            start=datetime.now() - timedelta(minutes=30),
            end=datetime.now(),
            step="1m"
        )
        ```
    """

    name: str = "prometheus_query"
    description: str = "Query Prometheus metrics using PromQL. Use for CPU, memory, latency, and error rate analysis."
    url: str
    timeout: int = 30

    args_schema: type[BaseModel] = PrometheusInput

    def _run(
        self,
        query: str,
        time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute an instant query against Prometheus.

        Args:
            query (str): PromQL query string
            time (Optional[datetime]): Evaluation timestamp. If None,
                uses current time.

        Returns:
            List[Dict[str, Any]]: Query results as list of metric
                dictionaries with labels and values

        Raises:
            requests.RequestException: If HTTP request fails
            ValueError: If response format is unexpected

        Example:
            ```python
            result = prometheus.query("container_cpu_usage_seconds_total")
            for metric in result:
                print(f"Pod: {metric['metric']['pod']}, Value: {metric['value']}")
            ```
        """
        endpoint = f"{self.url}/api/v1/query"
        params = {
            "query": query,
        }

        if time:
            params["time"] = int(time.timestamp())

        response = requests.get(endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        if data.get("status") != "success":
            raise ValueError(f"Prometheus query failed: {data.get('error', 'unknown error')}")

        result = data.get("data", {}).get("result", [])
        return result

    def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "1m"
    ) -> List[Dict[str, Any]]:
        """
        Execute a range query against Prometheus.

        Args:
            query (str): PromQL query string
            start (datetime): Start of time range
            end (datetime): End of time range
            step (str): Query resolution step (e.g., "1m", "30s", "1h")

        Returns:
            List[Dict[str, Any]]: Query results with time series data.
                Each result contains 'metric' (labels) and 'values'
                (list of [timestamp, value] pairs).

        Raises:
            requests.RequestException: If HTTP request fails
            ValueError: If response format is unexpected

        Example:
            ```python
            result = prometheus.query_range(
                "container_cpu_usage_seconds_total",
                start=datetime.now() - timedelta(hours=1),
                end=datetime.now(),
                step="5m"
            )
            for ts in result:
                print(f"Metric: {ts['metric']}")
                print(f"Values: {ts['values'][:5]}...")  # First 5 points
            ```
        """
        endpoint = f"{self.url}/api/v1/query_range"
        params = {
            "query": query,
            "start": int(start.timestamp()),
            "end": int(end.timestamp()),
            "step": step
        }

        response = requests.get(endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        if data.get("status") != "success":
            raise ValueError(f"Prometheus range query failed: {data.get('error', 'unknown error')}")

        result = data.get("data", {}).get("result", [])
        return result

    def get_labels(self, label_name: str) -> List[str]:
        """
        Get all values for a specific label.

        Args:
            label_name (str): Name of the label to query

        Returns:
            List[str]: List of label values

        Example:
            ```python
            namespaces = prometheus.get_labels("namespace")
            services = prometheus.get_labels("service")
            ```
        """
        endpoint = f"{self.url}/api/v1/label/{label_name}/values"
        response = requests.get(endpoint, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        if data.get("status") != "success":
            raise ValueError(f"Failed to get label values: {data.get('error', 'unknown error')}")

        return data.get("data", [])

    def get_metrics(self, metric_prefix: Optional[str] = None) -> List[str]:
        """
        Get list of available metric names.

        Args:
            metric_prefix (Optional[str]): Optional prefix to filter metrics

        Returns:
            List[str]: List of metric names

        Example:
            ```python
            # Get all metrics
            all_metrics = prometheus.get_metrics()

            # Get only HTTP-related metrics
            http_metrics = prometheus.get_metrics("http_")
            ```
        """
        endpoint = f"{self.url}/api/v1/label/__name__/values"
        response = requests.get(endpoint, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        if data.get("status") != "success":
            raise ValueError(f"Failed to get metrics: {data.get('error', 'unknown error')}")

        metrics = data.get("data", [])

        if metric_prefix:
            metrics = [m for m in metrics if m.startswith(metric_prefix)]

        return metrics

    def health_check(self) -> bool:
        """
        Check if Prometheus server is healthy.

        Returns:
            bool: True if server is healthy, False otherwise

        Example:
            ```python
            if prometheus.health_check():
                print("Prometheus is available")
            else:
                print("Prometheus is unreachable")
            ```
        """
        try:
            endpoint = f"{self.url}/-/healthy"
            response = requests.get(endpoint, timeout=self.timeout)
            return response.status_code == 200
        except Exception:
            return False
