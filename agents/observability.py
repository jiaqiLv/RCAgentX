"""
Observability Agent - Multi-modal Data Collection and Processing

The Observability Agent is the perception layer of the AIOps system.
It collects, processes, and summarizes multi-modal observability data
including Metrics, Logs, and Traces from various data sources.

Key responsibilities:
1. Multi-modal data collection from Prometheus, Loki, Jaeger, etc.
2. Data compression and summarization for LLM consumption
3. Multi-modal fusion and correlation
4. Real-time streaming data processing

The output of this agent serves as input for the Detection Agent.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field

from agents.base import BaseAgent
from memory.shared_state import ObservabilityData


class ObservabilityAgent(BaseAgent):
    """
    Observability data collection and processing agent.

    This agent collects metrics, logs, and traces from monitoring systems,
    processes them into compressed summaries, and correlates events
    across modalities using trace IDs, timestamps, and entity labels.

    Attributes:
        name (str): Always "observability"
        description (str): Description of observability capabilities
        prometheus_tool (PrometheusTool): Tool for metric queries
        loki_tool (LokiTool): Tool for log queries
        llm (ChatOpenAI): Language model for summarization
        time_range_minutes (int): Default time range for data collection
        verbose (bool): Enable verbose logging

    Example:
        ```python
        obs_agent = ObservabilityAgent(
            prometheus_tool=PrometheusTool(url="http://prometheus:9090"),
            loki_tool=LokiTool(url="http://loki:3100"),
            llm=ChatOpenAI(model="gpt-4"),
            time_range_minutes=30
        )

        state = {
            "incident_id": "inc-001",
            "status": "pending"
        }
        result = obs_agent.execute(state)
        print(result["observability"].summary)
        ```
    """

    name: str = "observability"
    description: str = "Collects and processes multi-modal observability data (metrics, logs, traces)"
    prometheus_tool: Optional[BaseTool] = None
    loki_tool: Optional[BaseTool] = None
    time_range_minutes: int = 30
    llm: Optional[ChatOpenAI] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute observability data collection and processing.

        Collects metrics, logs, and traces for the specified time range
        and entities, then generates a unified summary.

        Args:
            state (Dict[str, Any]): Current workflow state. May contain:
                - incident_id: Unique incident identifier
                - entities: List of specific entities to investigate
                - time_range: Custom time range override
                - alert_labels: Labels from the triggering alert

        Returns:
            Dict[str, Any]: Updated state with observability data

        Example:
            ```python
            state = {
                "incident_id": "inc-001",
                "entities": ["api-service-pod-1"],
                "alert_labels": {"service": "api-service", "severity": "critical"}
            }
            result = obs_agent.execute(state)
            obs_data = result["observability"]
            ```
        """
        self.log("Starting observability data collection")

        # Extract parameters from state
        entities = state.get("entities", [])
        alert_labels = state.get("alert_labels", {})
        time_range = state.get("time_range")

        # Calculate time range
        if time_range:
            end_time = time_range.get("end", datetime.now())
            start_time = time_range.get("start", end_time - timedelta(minutes=self.time_range_minutes))
        else:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=self.time_range_minutes)

        self.log(f"Collecting data from {start_time} to {end_time}")

        # Initialize observability data container
        obs_data = ObservabilityData(
            time_range={"start": start_time, "end": end_time}
        )

        # Collect metrics if Prometheus tool is available
        if self.prometheus_tool:
            try:
                metrics = self._collect_metrics(alert_labels, entities, start_time, end_time)
                obs_data.metrics = metrics
                self.log(f"Collected {len(metrics)} metric data points")
            except Exception as e:
                self.log(f"Failed to collect metrics: {str(e)}")
                state["errors"] = state.get("errors", [])
                state["errors"].append(f"Metrics collection error: {str(e)}")

        # Collect logs if Loki tool is available
        if self.loki_tool:
            try:
                logs = self._collect_logs(alert_labels, entities, start_time, end_time)
                obs_data.logs = logs
                self.log(f"Collected {len(logs)} log entries")
            except Exception as e:
                self.log(f"Failed to collect logs: {str(e)}")
                state["errors"] = state.get("errors", [])
                state["errors"].append(f"Logs collection error: {str(e)}")

        # Generate summary using LLM if available
        if self.llm and (obs_data.metrics or obs_data.logs):
            try:
                summary = self._generate_summary(obs_data)
                obs_data.summary = summary
                self.log("Generated observability summary")
            except Exception as e:
                self.log(f"Failed to generate summary: {str(e)}")
                obs_data.summary = self._generate_basic_summary(obs_data)

        # Update state
        state["observability"] = obs_data
        state["logs"] = state.get("logs", []) + obs_data.logs[:10]  # Keep first 10 logs in state

        self.log("Observability data collection completed")
        return state

    def _collect_metrics(
        self,
        alert_labels: Dict[str, str],
        entities: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Collect relevant metrics from Prometheus.

        Queries metrics based on alert labels and affected entities.
        Focuses on CPU, memory, latency, and error rate metrics.

        Args:
            alert_labels (Dict[str, str]): Labels from the triggering alert
            entities (List[str]): List of affected entity names
            start_time (datetime): Start of time range
            end_time (datetime): End of time range

        Returns:
            List[Dict[str, Any]]: List of metric data points
        """
        metrics = []

        if not self.prometheus_tool:
            return metrics

        # Build label selectors from alert
        label_selectors = []
        if alert_labels:
            for key, value in alert_labels.items():
                label_selectors.append(f'{key}="{value}"')

        # Default selector if none provided
        if not label_selectors:
            label_selectors = ['namespace="default"']

        selector = ",".join(label_selectors)

        # Query key metrics
        metric_queries = [
            f"container_cpu_usage_seconds_total{{{selector}}}",
            f"container_memory_usage_bytes{{{selector}}}",
            f"rate(http_requests_total{{{selector}}}[5m])",
            f"rate(http_requests_total{{{selector},status=~'5..'}}[5m])",
            f"histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{{{selector}}}[5m]))",
        ]

        for query in metric_queries:
            try:
                result = self.prometheus_tool.query_range(
                    query=query,
                    start=start_time,
                    end=end_time,
                    step="1m"
                )
                if result:
                    metrics.append({
                        "query": query,
                        "data": result
                    })
            except Exception as e:
                self.log(f"Metric query failed: {query}, error: {str(e)}")

        return metrics

    def _collect_logs(
        self,
        alert_labels: Dict[str, str],
        entities: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Collect relevant logs from Loki.

        Queries logs based on alert labels and filters for error
        patterns and anomalies.

        Args:
            alert_labels (Dict[str, str]): Labels from the triggering alert
            entities (List[str]): List of affected entity names
            start_time (datetime): Start of time range
            end_time (datetime): End of time range

        Returns:
            List[Dict[str, Any]]: List of log entries
        """
        logs = []

        if not self.loki_tool:
            return logs

        # Build logql label selectors
        label_selectors = []
        if alert_labels:
            if "service" in alert_labels:
                label_selectors.append(f'app="{alert_labels["service"]}"')
            if "namespace" in alert_labels:
                label_selectors.append(f'namespace="{alert_labels["namespace"]}"')

        # Default selector
        if not label_selectors:
            label_selectors = ['namespace="default"']

        selector = "{%s}" % ",".join(label_selectors)

        # Query logs with error filter
        logql = f'{selector} |=~ "(?i)error|exception|fail|timeout|critical"'

        try:
            result = self.loki_tool.query(
                query=logql,
                start=start_time,
                end=end_time,
                limit=1000
            )
            if result:
                logs = result
        except Exception as e:
            self.log(f"Log query failed: {logql}, error: {str(e)}")

        # Also query recent logs without filter for context
        try:
            recent_logql = f'{selector}'
            result = self.loki_tool.query(
                query=recent_logql,
                start=end_time - timedelta(minutes=5),
                end=end_time,
                limit=200
            )
            if result:
                logs.extend(result)
        except Exception as e:
            self.log(f"Recent log query failed: {str(e)}")

        return logs

    def _generate_summary(self, obs_data: ObservabilityData) -> str:
        """
        Generate a comprehensive summary using LLM.

        Uses the language model to analyze metrics and logs and
        produce a human-readable summary highlighting anomalies
        and correlations.

        Args:
            obs_data (ObservabilityData): Collected observability data

        Returns:
            str: Human-readable summary of the observability data
        """
        # Prepare context for LLM
        metrics_summary = []
        for m in obs_data.metrics[:10]:  # Limit to avoid token limits
            metrics_summary.append(f"- Query: {m['query']}\n  Data points: {len(m.get('data', []))}")

        logs_summary = []
        error_logs = [l for l in obs_data.logs if "error" in str(l).lower()][:20]
        for log in error_logs:
            logs_summary.append(f"- {log.get('line', str(log))}")

        prompt = f"""
        Analyze the following observability data and provide a concise summary:

        ## Metrics Data
        {chr(10).join(metrics_summary)}

        ## Error Logs
        {chr(10).join(logs_summary[:20])}

        Please provide:
        1. Key anomalies detected in metrics
        2. Error patterns observed in logs
        3. Potential correlations between metrics and logs
        4. Most affected services/components

        Keep the summary concise and actionable.
        """

        response = self.llm.invoke(prompt)
        return response.content

    def _generate_basic_summary(self, obs_data: ObservabilityData) -> str:
        """
        Generate a basic summary without LLM.

        Fallback method when LLM is not available or fails.
        Provides basic statistics about collected data.

        Args:
            obs_data (ObservabilityData): Collected observability data

        Returns:
            str: Basic summary with statistics
        """
        lines = ["Observability Data Summary", ""]

        # Metrics summary
        lines.append(f"Metrics collected: {len(obs_data.metrics)}")
        for m in obs_data.metrics[:5]:
            lines.append(f"  - {m.get('query', 'unknown')}")

        # Logs summary
        error_count = sum(1 for l in obs_data.logs if "error" in str(l).lower())
        lines.append(f"\nLogs collected: {len(obs_data.logs)}")
        lines.append(f"  Error logs: {error_count}")

        return "\n".join(lines)
