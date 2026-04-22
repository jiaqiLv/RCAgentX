"""
Mock Monitoring Tool - Simulated Metrics and Logs

Provides mock data for testing without real Prometheus/Loki connections.
Generates realistic-looking metrics and logs for AIOps agent testing.
"""

import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class MockMetricsGenerator:
    """
    Generates realistic mock metrics for testing.

    Simulates various metric patterns including:
    - Normal operation
    - CPU spikes
    - Memory leaks
    - Latency increases
    - Error rate anomalies
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the mock metrics generator.

        Args:
            seed (Optional[int]): Random seed for reproducible data
        """
        if seed:
            random.seed(seed)
        self.base_values = {
            "cpu_usage": 0.3,
            "memory_usage": 0.5,
            "request_latency": 0.05,
            "error_rate": 0.01,
            "requests_per_second": 100,
        }

    def generate_cpu_metrics(
        self,
        anomaly: bool = False,
        duration_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Generate CPU usage metrics.

        Args:
            anomaly (bool): If True, include a CPU spike
            duration_minutes (int): Duration of metrics in minutes

        Returns:
            List[Dict[str, Any]]: List of metric data points
        """
        metrics = []
        base_cpu = 0.3 if not anomaly else 0.4

        for i in range(duration_minutes):
            # Add some noise
            value = base_cpu + random.gauss(0, 0.05)

            # Create spike in the middle
            if anomaly and 10 <= i <= 20:
                value = 0.85 + random.gauss(0, 0.05)

            value = max(0.05, min(0.99, value))

            metrics.append({
                "metric": {
                    "__name__": "container_cpu_usage_seconds_total",
                    "pod": "api-service-pod-1",
                    "namespace": "default",
                    "service": "api-service"
                },
                "values": [
                    [
                        int((datetime.now() - timedelta(minutes=duration_minutes-i)).timestamp()),
                        str(value)
                    ]
                ]
            })

        return [{"query": "container_cpu_usage_seconds_total", "data": metrics}]

    def generate_memory_metrics(
        self,
        anomaly: bool = False,
        duration_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Generate memory usage metrics.

        Args:
            anomaly (bool): If True, show memory leak pattern
            duration_minutes (int): Duration of metrics

        Returns:
            List[Dict[str, Any]]: Memory metric data points
        """
        metrics = []
        base_memory = 0.5  # 50% usage as ratio

        for i in range(duration_minutes):
            value = base_memory + (i * 0.01 if anomaly else 0)
            value += random.gauss(0, 0.02)
            value = max(0.1, min(0.99, value))

            metrics.append({
                "metric": {
                    "__name__": "container_memory_usage_ratio",
                    "pod": "api-service-pod-1",
                    "namespace": "default",
                    "service": "api-service"
                },
                "values": [
                    [
                        int((datetime.now() - timedelta(minutes=duration_minutes-i)).timestamp()),
                        str(value)  # Keep as ratio (0-1)
                    ]
                ]
            })

        return [{"query": "container_memory_usage_ratio", "data": metrics}]

    def generate_latency_metrics(
        self,
        anomaly: bool = False,
        duration_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Generate request latency metrics.

        Args:
            anomaly (bool): If True, show latency spike
            duration_minutes (int): Duration of metrics

        Returns:
            List[Dict[str, Any]]: Latency metric data points
        """
        metrics = []
        base_latency = 0.05  # 50ms

        for i in range(duration_minutes):
            value = base_latency + random.gauss(0, 0.01)

            if anomaly and 10 <= i <= 20:
                value = 0.5 + random.gauss(0, 0.1)  # 500ms spike

            value = max(0.01, value)

            metrics.append({
                "metric": {
                    "__name__": "http_request_duration_seconds",
                    "pod": "api-service-pod-1",
                    "namespace": "default",
                    "service": "api-service",
                    "le": "0.5"
                },
                "values": [
                    [
                        int((datetime.now() - timedelta(minutes=duration_minutes-i)).timestamp()),
                        str(value)
                    ]
                ]
            })

        return [{"query": "http_request_duration_seconds", "data": metrics}]

    def generate_error_rate_metrics(
        self,
        anomaly: bool = False,
        duration_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Generate error rate metrics.

        Args:
            anomaly (bool): If True, show elevated error rate
            duration_minutes (int): Duration of metrics

        Returns:
            List[Dict[str, Any]]: Error rate metric data points
        """
        metrics = []
        base_error_rate = 0.01  # 1%

        for i in range(duration_minutes):
            value = base_error_rate + random.gauss(0, 0.005)

            if anomaly and 10 <= i <= 20:
                value = 0.15 + random.gauss(0, 0.03)  # 15% error rate

            value = max(0.001, min(0.5, value))

            metrics.append({
                "metric": {
                    "__name__": "http_requests_total",
                    "pod": "api-service-pod-1",
                    "namespace": "default",
                    "service": "api-service",
                    "status": "500"
                },
                "values": [
                    [
                        int((datetime.now() - timedelta(minutes=duration_minutes-i)).timestamp()),
                        str(value * 100)  # Scale to reasonable count
                    ]
                ]
            })

        return [{"query": "http_requests_total{status=~'5..'}", "data": metrics}]

    def generate_all_metrics(
        self,
        anomaly_type: Optional[str] = None,
        duration_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Generate a complete set of metrics.

        Args:
            anomaly_type (Optional[str]): Type of anomaly to simulate.
                Options: "cpu", "memory", "latency", "error", or None
            duration_minutes (int): Duration of metrics

        Returns:
            List[Dict[str, Any]]: All metric data combined in Prometheus format
        """
        all_metrics = []

        # CPU
        cpu_metrics = self.generate_cpu_metrics(
            anomaly=(anomaly_type == "cpu"),
            duration_minutes=duration_minutes
        )
        if cpu_metrics:
            all_metrics.extend(cpu_metrics[0]["data"])

        # Memory
        mem_metrics = self.generate_memory_metrics(
            anomaly=(anomaly_type == "memory"),
            duration_minutes=duration_minutes
        )
        if mem_metrics:
            all_metrics.extend(mem_metrics[0]["data"])

        # Latency
        latency_metrics = self.generate_latency_metrics(
            anomaly=(anomaly_type == "latency"),
            duration_minutes=duration_minutes
        )
        if latency_metrics:
            all_metrics.extend(latency_metrics[0]["data"])

        # Error Rate
        error_metrics = self.generate_error_rate_metrics(
            anomaly=(anomaly_type == "error"),
            duration_minutes=duration_minutes
        )
        if error_metrics:
            all_metrics.extend(error_metrics[0]["data"])

        return all_metrics


class MockLogsGenerator:
    """
    Generates realistic mock log entries for testing.

    Simulates various log patterns including:
    - Normal operation logs
    - Error messages
    - Exception stack traces
    - Warning messages
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the mock logs generator.

        Args:
            seed (Optional[int]): Random seed for reproducible data
        """
        if seed:
            random.seed(seed)

        self.normal_messages = [
            "Request processed successfully in {ms}ms",
            "Database query completed in {ms}ms",
            "Cache hit for key: user:{id}",
            "Connection established to upstream service",
            "Health check passed",
            "Configuration reloaded",
            "Session created for user:{id}",
            "Request completed with status 200",
        ]

        self.error_messages = [
            "Connection timeout to database after {ms}ms",
            "Failed to process request: {error}",
            "Upstream service returned 503",
            "Out of memory error in worker thread",
            "Connection refused to redis://localhost:6379",
            "Request timeout exceeded {ms}ms",
            "NullPointerException in RequestHandler.process()",
            "Failed to parse JSON response: Unexpected token",
        ]

        self.warning_messages = [
            "High memory usage detected: {pct}%",
            "Slow query detected: {ms}ms",
            "Connection pool near capacity: {pct}%",
            "Retry attempt {n} of 3",
            "Rate limit approaching for API key",
        ]

    def generate_logs(
        self,
        anomaly_type: Optional[str] = None,
        count: int = 100,
        time_range_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Generate mock log entries.

        Args:
            anomaly_type (Optional[str]): Type of anomaly to simulate
            count (int): Number of log entries to generate
            time_range_minutes (int): Time range for logs

        Returns:
            List[Dict[str, Any]]: List of log entries
        """
        logs = []
        base_time = datetime.now() - timedelta(minutes=time_range_minutes)

        error_probability = 0.02  # 2% error rate normally

        if anomaly_type == "cpu":
            error_probability = 0.1
        elif anomaly_type == "error":
            error_probability = 0.3
        elif anomaly_type == "memory":
            error_probability = 0.15

        for i in range(count):
            # Calculate timestamp
            offset = timedelta(minutes=(time_range_minutes * i) / count)
            timestamp = base_time + offset

            # Determine log level
            rand = random.random()
            if rand < error_probability:
                level = "ERROR"
                message = random.choice(self.error_messages)
            elif rand < error_probability + 0.1:
                level = "WARN"
                message = random.choice(self.warning_messages)
            else:
                level = "INFO"
                message = random.choice(self.normal_messages)

            # Replace placeholders
            message = message.format(
                ms=random.randint(10, 5000),
                id=random.randint(1000, 9999),
                error=random.choice(["ConnectionError", "TimeoutError", "ValueError"]),
                pct=random.randint(70, 99),
                n=random.randint(1, 3)
            )

            logs.append({
                "timestamp": timestamp.isoformat(),
                "level": level,
                "service": "api-service",
                "pod": "api-service-pod-1",
                "namespace": "default",
                "line": f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}"
            })

        # Sort by timestamp
        logs.sort(key=lambda x: x["timestamp"])

        return logs


class MockPrometheusInput(BaseModel):
    """Input schema for MockPrometheusTool."""
    query: str = "container_cpu_usage_seconds_total"


class MockPrometheusTool(BaseTool):
    """
    Mock Prometheus tool for testing without real Prometheus.

    Drop-in replacement for PrometheusTool that returns generated data.
    """

    name: str = "prometheus_query"
    description: str = "Query Prometheus metrics using PromQL. Use for CPU, memory, latency, and error rate analysis."
    args_schema: type[BaseModel] = MockPrometheusInput
    url: str = "http://mock:9090"
    seed: Optional[int] = None
    generator: Optional[MockMetricsGenerator] = None

    def __init__(self, url: str = "http://mock:9090", seed: Optional[int] = None):
        """
        Initialize mock Prometheus tool.

        Args:
            url (str): Mock URL (for compatibility)
            seed (Optional[int]): Random seed for reproducible data
        """
        super().__init__(url=url, seed=seed, generator=MockMetricsGenerator(seed))

    def _run(
        self,
        query: str = "container_cpu_usage_seconds_total"
    ) -> List[Dict[str, Any]]:
        """
        Mock instant query.

        Args:
            query (str): PromQL query

        Returns:
            List[Dict[str, Any]]: Mock query results
        """
        # Return a single data point
        return self.generator.generate_all_metrics(duration_minutes=1)

    def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "1m"
    ) -> List[Dict[str, Any]]:
        """
        Mock range query.

        Args:
            query (str): PromQL query
            start (datetime): Start time
            end (datetime): End time
            step (str): Step size

        Returns:
            List[Dict[str, Any]]: Mock range results
        """
        duration = int((end - start).total_seconds() / 60)
        return self.generator.generate_all_metrics(duration_minutes=max(5, duration))

    def health_check(self) -> bool:
        """Mock health check always returns True."""
        return True


class MockLokiInput(BaseModel):
    """Input schema for MockLokiTool."""
    query: str = "{service=\"api-service\"}"


class MockLokiTool(BaseTool):
    """
    Mock Loki tool for testing without real Loki.

    Drop-in replacement for LokiTool that returns generated logs.
    """

    name: str = "loki_query"
    description: str = "Query Loki logs using LogQL. Use for log analysis and error investigation."
    args_schema: type[BaseModel] = MockLokiInput
    url: str = "http://mock:3100"
    seed: Optional[int] = None
    generator: Optional[MockLogsGenerator] = None

    def __init__(self, url: str = "http://mock:3100", seed: Optional[int] = None):
        """
        Initialize mock Loki tool.

        Args:
            url (str): Mock URL (for compatibility)
            seed (Optional[int]): Random seed for reproducible data
        """
        super().__init__(url=url, seed=seed, generator=MockLogsGenerator(seed))

    def _run(
        self,
        query: str = "{service=\"api-service\"}"
    ) -> List[Dict[str, Any]]:
        """
        Mock log query.

        Args:
            query (str): LogQL query

        Returns:
            List[Dict[str, Any]]: Mock log results
        """
        end = datetime.now()
        start = end - timedelta(hours=1)
        logs = self.generator.generate_logs(
            count=100,
            time_range_minutes=60
        )

        # Convert to Loki format
        return [
            {
                "labels": {
                    "pod": log.get("pod", "unknown"),
                    "service": log.get("service", "unknown"),
                    "level": log.get("level", "INFO")
                },
                "line": log["line"],
                "timestamp": log["timestamp"]
            }
            for log in logs
        ]

    def query(
        self,
        query: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Mock log query (compatibility method).

        Args:
            query (str): LogQL query (ignored in mock)
            start (Optional[datetime]): Start time
            end (Optional[datetime]): End time
            limit (int): Maximum number of results

        Returns:
            List[Dict[str, Any]]: Mock log results
        """
        # Calculate time range
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(hours=1)

        duration = int((end - start).total_seconds() / 60)
        logs = self.generator.generate_logs(
            count=min(limit, 100),
            time_range_minutes=duration
        )

        # Convert to Loki format
        return [
            {
                "labels": {
                    "pod": log.get("pod", "unknown"),
                    "service": log.get("service", "unknown"),
                    "level": log.get("level", "INFO")
                },
                "line": log["line"],
                "timestamp": log["timestamp"]
            }
            for log in logs
        ]

    def health_check(self) -> bool:
        """Mock health check always returns True."""
        return True


def create_mock_system(
    anomaly_type: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a complete mock monitoring system.

    Args:
        anomaly_type (Optional[str]): Type of anomaly to simulate
        seed (Optional[int]): Random seed

    Returns:
        Dict[str, Any]: Mock tools dictionary
    """
    return {
        "prometheus": MockPrometheusTool(seed=seed),
        "loki": MockLokiTool(seed=seed),
        "anomaly_type": anomaly_type,
    }
