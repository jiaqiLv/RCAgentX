"""
Loki Tool - Log Query and Analysis

LangChain tool for querying Loki logs.
Provides methods for logql queries, label discovery, and log streaming.
"""

import requests
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class LokiInput(BaseModel):
    """Input schema for Loki tool operations."""
    query: str = Field(..., description="LogQL query string")
    start: Optional[datetime] = Field(None, description="Start time for query")
    end: Optional[datetime] = Field(None, description="End time for query")
    limit: Optional[int] = Field(1000, description="Maximum number of log entries")


class LokiTool(BaseTool):
    """
    Tool for querying Loki logs.

    Provides methods for executing LogQL queries against a Loki
    server, including instant queries, range queries, and label
    value discovery.

    Attributes:
        name (str): Tool name for LangChain
        description (str): Tool description
        url (str): Loki server URL
        timeout (int): Request timeout in seconds

    Example:
        ```python
        loki = LokiTool(url="http://loki:3100")

        # Query error logs
        logs = loki.query('{app="api-service"} |= "error"')

        # Query with time range
        logs = loki.query(
            '{namespace="production"} |~ "(?i)exception"',
            start=datetime.now() - timedelta(hours=1),
            limit=500
        )
        ```
    """

    name: str = "loki_query"
    description: str = "Query Loki logs using LogQL. Use for error analysis, pattern detection, and log correlation."
    url: str
    timeout: int = 30

    args_schema: type[BaseModel] = LokiInput

    def _run(
        self,
        query: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Execute a log query against Loki.

        Args:
            query (str): LogQL query string
            start (Optional[datetime]): Start of time range. If None,
                defaults to 1 hour ago.
            end (Optional[datetime]): End of time range. If None,
                defaults to current time.
            limit (int): Maximum number of log entries to return

        Returns:
            List[Dict[str, Any]]: List of log entries. Each entry
                contains 'labels' (dict) and 'line' (str) keys.

        Raises:
            requests.RequestException: If HTTP request fails
            ValueError: If response format is unexpected

        Example:
            ```python
            logs = loki.query('{app="api"} |= "error"')
            for log in logs:
                print(f"[{log['timestamp']}] {log['line']}")
            ```
        """
        # Set default time range
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(hours=1)

        endpoint = f"{self.url}/loki/api/v1/query_range"
        params = {
            "query": query,
            "start": int(start.timestamp() * 1e9),  # Loki uses nanoseconds
            "end": int(end.timestamp() * 1e9),
            "limit": limit,
            "direction": "BACKWARD"
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.get(endpoint, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        if data.get("status") != "success":
            raise ValueError(f"Loki query failed: {data.get('error', 'unknown error')}")

        # Parse response
        logs = []
        result = data.get("data", {}).get("result", [])

        for stream in result:
            labels = stream.get("stream", {})
            values = stream.get("values", [])

            for entry in values:
                timestamp_ns = int(entry[0])
                line = entry[1]

                logs.append({
                    "timestamp": datetime.fromtimestamp(timestamp_ns / 1e9),
                    "labels": labels,
                    "line": line
                })

        # Sort by timestamp
        logs.sort(key=lambda x: x["timestamp"], reverse=True)

        return logs

    def query_instant(
        self,
        query: str,
        at: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute an instant log query (logs at a specific point in time).

        Args:
            query (str): LogQL query string
            at (Optional[datetime]): Evaluation time. If None, uses
                current time.

        Returns:
            List[Dict[str, Any]]: List of matching log entries

        Example:
            ```python
            # Get logs from a specific incident time
            incident_logs = loki.query_instant(
                '{app="api"} |= "timeout"',
                at=incident_time
            )
            ```
        """
        if at is None:
            at = datetime.now()

        endpoint = f"{self.url}/loki/api/v1/query"
        params = {
            "query": query,
            "time": int(at.timestamp() * 1e9)
        }

        response = requests.get(endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        if data.get("status") != "success":
            raise ValueError(f"Loki instant query failed: {data.get('error', 'unknown error')}")

        logs = []
        result = data.get("data", {}).get("result", [])

        for stream in result:
            labels = stream.get("stream", {})
            values = stream.get("values", [])

            for entry in values:
                timestamp_ns = int(entry[0])
                line = entry[1]

                logs.append({
                    "timestamp": datetime.fromtimestamp(timestamp_ns / 1e9),
                    "labels": labels,
                    "line": line
                })

        return logs

    def get_labels(self) -> List[str]:
        """
        Get all available label names in Loki.

        Returns:
            List[str]: List of label names

        Example:
            ```python
            labels = loki.get_labels()
            print(f"Available labels: {labels}")
            ```
        """
        endpoint = f"{self.url}/loki/api/v1/labels"
        response = requests.get(endpoint, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        return data.get("data", [])

    def get_label_values(self, label_name: str) -> List[str]:
        """
        Get all values for a specific label.

        Args:
            label_name (str): Name of the label

        Returns:
            List[str]: List of label values

        Example:
            ```python
            apps = loki.get_label_values("app")
            namespaces = loki.get_label_values("namespace")
            ```
        """
        endpoint = f"{self.url}/loki/api/v1/label/{label_name}/values"
        response = requests.get(endpoint, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        return data.get("data", [])

    def health_check(self) -> bool:
        """
        Check if Loki server is healthy.

        Returns:
            bool: True if server is healthy, False otherwise

        Example:
            ```python
            if loki.health_check():
                print("Loki is available")
            else:
                print("Loki is unreachable")
            ```
        """
        try:
            endpoint = f"{self.url}/loki/api/v1/status"
            response = requests.get(endpoint, timeout=self.timeout)
            return response.status_code == 200
        except Exception:
            return False
