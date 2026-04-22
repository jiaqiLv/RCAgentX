"""
Detection Agent - Anomaly Detection and Classification

The Detection Agent is responsible for identifying anomalies in the
observability data collected by the Observability Agent. It uses a
combination of statistical methods and LLM-based analysis to detect
performance degradation, errors, and unusual patterns.

Key responsibilities:
1. Multi-modal real-time anomaly detection (metrics, logs, traces)
2. Advanced anomaly detection algorithms
3. Human-in-the-loop confirmation and feedback
4. Event aggregation and noise filtering
"""

from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from pydantic import ConfigDict

from agents.base import BaseAgent
from memory.shared_state import AnomalyEvent, IncidentStatus


class DetectionAgent(BaseAgent):
    """
    Anomaly detection and classification agent.

    This agent analyzes observability data to detect anomalies using
    statistical methods, threshold-based rules, and LLM-powered analysis.

    Attributes:
        name (str): Always "detection"
        description (str): Description of detection capabilities
        llm (ChatOpenAI): Language model for pattern analysis
        thresholds (Dict[str, float]): Custom threshold configurations
        verbose (bool): Enable verbose logging

    Example:
        ```python
        detection = DetectionAgent(
            llm=ChatOpenAI(model="gpt-4"),
            thresholds={
                "cpu_warning": 0.8,
                "cpu_critical": 0.95,
                "error_rate": 0.05
            }
        )

        result = detection.execute(state)
        anomaly = result["anomaly"]
        ```
    """

    name: str = "detection"
    description: str = "Detects anomalies in metrics, logs, and traces"
    llm: Optional[ChatOpenAI] = None
    thresholds: Dict[str, float] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute anomaly detection on observability data.

        Analyzes metrics and logs to detect anomalies, assigns severity
        levels, and generates confidence scores.

        Args:
            state (Dict[str, Any]): Current workflow state containing
                observability data from the Observability Agent

        Returns:
            Dict[str, Any]: Updated state with detected anomaly

        Example:
            ```python
            state = {
                "observability": obs_data,
                "alert_labels": {"service": "api"}
            }
            result = detection.execute(state)
            if result.get("anomaly"):
                print(f"Detected: {result['anomaly'].type}")
            ```
        """
        self.log("Starting anomaly detection")

        obs_data = state.get("observability")
        if not obs_data:
            self.log("No observability data found")
            state["errors"] = state.get("errors", [])
            state["errors"].append("No observability data for detection")
            return state

        # Detect anomalies in metrics
        metric_anomalies = self._detect_metric_anomalies(obs_data.metrics)

        # Detect anomalies in logs
        log_anomalies = self._detect_log_anomalies(obs_data.logs)

        # Combine and score anomalies
        if metric_anomalies or log_anomalies:
            anomaly = self._combine_anomalies(metric_anomalies, log_anomalies, obs_data)
            state["anomaly"] = anomaly
            state["status"] = IncidentStatus.DETECTING.value
            self.log(f"Anomaly detected: {anomaly.type}, severity={anomaly.severity}")
        else:
            self.log("No anomalies detected")
            # Create a low-confidence "no issue" event
            state["anomaly"] = AnomalyEvent(
                type="no_issue",
                severity="LOW",
                confidence=0.9,
                evidence={"metric_anomalies": [], "log_anomalies": []}
            )

        return state

    def _detect_metric_anomalies(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in metric data.

        Uses threshold-based detection and statistical analysis to
        identify unusual metric values.

        Args:
            metrics (List[Dict[str, Any]]): List of metric data points

        Returns:
            List[Dict[str, Any]]: List of detected metric anomalies
        """
        anomalies = []

        for metric in metrics:
            query = metric.get("query", "")
            data = metric.get("data", [])

            # CPU usage detection
            if "cpu" in query.lower():
                anomalies.extend(self._check_cpu_anomaly(data))

            # Memory usage detection
            if "memory" in query.lower():
                anomalies.extend(self._check_memory_anomaly(data))

            # Error rate detection
            if "error" in query.lower() or "5.." in query:
                anomalies.extend(self._check_error_rate_anomaly(data, query))

            # Latency detection
            if "latency" in query.lower() or "duration" in query.lower():
                anomalies.extend(self._check_latency_anomaly(data))

        return anomalies

    def _check_cpu_anomaly(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for CPU usage anomalies"""
        anomalies = []
        warning_threshold = self.thresholds.get("cpu_warning", 0.8)
        critical_threshold = self.thresholds.get("cpu_critical", 0.95)

        for result in data:
            values = result.get("values", [])
            if not values:
                continue

            # Check latest value
            latest_value = float(values[-1][1]) if values else 0

            if latest_value > critical_threshold:
                anomalies.append({
                    "type": "cpu_critical",
                    "metric": "cpu_usage",
                    "value": latest_value,
                    "threshold": critical_threshold,
                    "severity": "CRITICAL",
                    "labels": result.get("metric", {})
                })
            elif latest_value > warning_threshold:
                anomalies.append({
                    "type": "cpu_warning",
                    "metric": "cpu_usage",
                    "value": latest_value,
                    "threshold": warning_threshold,
                    "severity": "HIGH",
                    "labels": result.get("metric", {})
                })

        return anomalies

    def _check_memory_anomaly(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for memory usage anomalies"""
        anomalies = []
        threshold = self.thresholds.get("memory_warning", 0.9)

        for result in data:
            values = result.get("values", [])
            if not values:
                continue

            latest_value = float(values[-1][1]) if values else 0

            if latest_value > threshold:
                anomalies.append({
                    "type": "memory_high",
                    "metric": "memory_usage",
                    "value": latest_value,
                    "threshold": threshold,
                    "severity": "HIGH",
                    "labels": result.get("metric", {})
                })

        return anomalies

    def _check_error_rate_anomaly(
        self,
        data: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Check for error rate anomalies"""
        anomalies = []
        threshold = self.thresholds.get("error_rate", 0.05)

        for result in data:
            values = result.get("values", [])
            if not values:
                continue

            # Check if error rate exceeds threshold
            latest_value = float(values[-1][1]) if values else 0

            if latest_value > threshold:
                anomalies.append({
                    "type": "error_rate_high",
                    "metric": "error_rate",
                    "value": latest_value,
                    "threshold": threshold,
                    "severity": "HIGH" if latest_value > threshold * 2 else "MEDIUM",
                    "labels": result.get("metric", {}),
                    "query": query
                })

        return anomalies

    def _check_latency_anomaly(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for latency anomalies"""
        anomalies = []
        # Default p99 latency threshold: 1 second
        threshold = self.thresholds.get("latency_p99", 1.0)

        for result in data:
            values = result.get("values", [])
            if not values:
                continue

            latest_value = float(values[-1][1]) if values else 0

            if latest_value > threshold:
                anomalies.append({
                    "type": "latency_high",
                    "metric": "latency_p99",
                    "value": latest_value,
                    "threshold": threshold,
                    "severity": "MEDIUM",
                    "labels": result.get("metric", {})
                })

        return anomalies

    def _detect_log_anomalies(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in log data.

        Identifies error patterns, exception bursts, and unusual
        log volumes.

        Args:
            logs (List[Dict[str, Any]]): List of log entries

        Returns:
            List[Dict[str, Any]]: List of detected log anomalies
        """
        anomalies = []

        # Count error patterns
        error_patterns = {
            "exception": 0,
            "timeout": 0,
            "connection_refused": 0,
            "oom": 0,
            "panic": 0
        }

        for log in logs:
            line = log.get("line", "").lower()
            for pattern in error_patterns:
                if pattern in line:
                    error_patterns[pattern] += 1

        # Detect anomaly if significant errors found
        total_logs = len(logs)
        if total_logs > 0:
            for pattern, count in error_patterns.items():
                if count > 10:  # Absolute threshold
                    anomalies.append({
                        "type": f"log_{pattern}",
                        "pattern": pattern,
                        "count": count,
                        "severity": "HIGH" if count > 50 else "MEDIUM"
                    })

        return anomalies

    def _combine_anomalies(
        self,
        metric_anomalies: List[Dict[str, Any]],
        log_anomalies: List[Dict[str, Any]],
        obs_data: Any
    ) -> AnomalyEvent:
        """
        Combine multiple anomaly signals into a single event.

        Aggregates related anomalies, determines overall severity,
        and calculates confidence score.

        Args:
            metric_anomalies: Detected metric anomalies
            log_anomalies: Detected log anomalies
            obs_data: Original observability data

        Returns:
            AnomalyEvent: Combined anomaly event
        """
        all_anomalies = metric_anomalies + log_anomalies

        # Determine primary anomaly type
        if metric_anomalies:
            primary = max(metric_anomalies, key=lambda x: self._severity_score(x.get("severity", "LOW")))
            anomaly_type = primary.get("type", "unknown")
        else:
            primary = max(log_anomalies, key=lambda x: self._severity_score(x.get("severity", "LOW")))
            anomaly_type = primary.get("type", "unknown")

        # Calculate overall severity
        max_severity = max(
            (self._severity_score(a.get("severity", "LOW")) for a in all_anomalies),
            default=0
        )
        severity_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
        severity = severity_map.get(max_severity, "LOW")

        # Calculate confidence based on number of correlated signals
        confidence = min(0.5 + (len(all_anomalies) * 0.1), 0.95)

        # Collect affected entities
        affected = set()
        for a in metric_anomalies:
            labels = a.get("labels", {})
            if "pod" in labels:
                affected.add(labels["pod"])
            if "service" in labels:
                affected.add(labels["service"])

        return AnomalyEvent(
            type=anomaly_type,
            severity=severity,
            confidence=confidence,
            affected_entities=list(affected),
            evidence={
                "metric_anomalies": metric_anomalies,
                "log_anomalies": log_anomalies,
                "total_signals": len(all_anomalies)
            }
        )

    def _severity_score(self, severity: str) -> int:
        """Convert severity to numeric score"""
        return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(severity, 0)
