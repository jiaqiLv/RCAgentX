"""
Report Agent - Incident Report Generation

The Report Agent generates comprehensive incident reports including
executive summaries, technical details, timelines, and recommendations.
It supports multiple output formats and distribution channels.

Key responsibilities:
1. Multi-modal report generation
2. LLM-driven narrative summarization
3. Visualization and dashboard integration
4. Post-incident knowledge capture
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from pydantic import ConfigDict

from agents.base import BaseAgent
from memory.shared_state import IncidentStatus


class ReportAgent(BaseAgent):
    """
    Incident report generation agent.

    This agent generates comprehensive incident reports combining
    data from all previous stages of the incident lifecycle.

    Attributes:
        name (str): Always "report"
        description (str): Description of reporting capabilities
        llm (ChatOpenAI): Language model for narrative generation
        include_technical_details (bool): Include detailed technical info
        include_recommendations (bool): Generate improvement recommendations
        verbose (bool): Enable verbose logging

    Example:
        ```python
        report = ReportAgent(
            llm=ChatOpenAI(model="gpt-4"),
            include_technical_details=True,
            include_recommendations=True
        )

        result = report.execute(state)
        print(result["report"]["summary"])
        ```
    """

    name: str = "report"
    description: str = "Generates comprehensive incident reports with summaries and recommendations"
    llm: Optional[ChatOpenAI] = None
    include_technical_details: bool = True
    include_recommendations: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate incident report.

        Creates a comprehensive report combining data from all stages
        of the incident lifecycle.

        Args:
            state (Dict[str, Any]): Current workflow state containing
                all incident-related data

        Returns:
            Dict[str, Any]: Updated state with generated report

        Example:
            ```python
            result = report.execute(state)
            report_data = result["report"]
            ```
        """
        self.log("Generating incident report")

        # Generate report sections
        report = {
            "incident_id": state.get("incident_id", "unknown"),
            "generated_at": datetime.now().isoformat(),
            "status": state.get("status", "unknown"),
            "summary": self._generate_summary(state),
            "timeline": self._generate_timeline(state),
            "root_cause": self._generate_root_cause_section(state),
            "actions_taken": self._generate_actions_section(state),
            "metrics": self._generate_metrics_section(state),
            "recommendations": self._generate_recommendations(state) if self.include_recommendations else [],
        }

        # Generate full report text
        report["full_text"] = self._format_report(report)

        # Generate LLM-enhanced summary if available
        if self.llm:
            report["executive_summary"] = self._generate_executive_summary(report)

        state["report"] = report
        state["status"] = IncidentStatus.RESOLVED.value

        self.log("Incident report generated")
        return state

    def _generate_summary(self, state: Dict[str, Any]) -> str:
        """
        Generate incident summary.

        Args:
            state: Current workflow state

        Returns:
            str: Brief incident summary
        """
        anomaly = state.get("anomaly")
        diagnosis = state.get("diagnosis")
        repair = state.get("repair")

        parts = []

        if anomaly:
            parts.append(f"Detected {anomaly.type} anomaly with {anomaly.severity} severity.")

        if diagnosis and diagnosis.root_causes:
            cause = diagnosis.root_causes[0]
            parts.append(f"Root cause: {cause.get('cause', 'unknown')} on {cause.get('entity', 'unknown')}.")

        if repair:
            parts.append(f"Resolution status: {repair.status}.")

        return " ".join(parts) or "No summary available."

    def _generate_timeline(self, state: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate incident timeline.

        Args:
            state: Current workflow state

        Returns:
            List[Dict[str, str]]: Timeline events
        """
        timeline = []
        logs = state.get("logs", [])

        # Add log entries to timeline
        for log_entry in logs[:20]:  # Limit to 20 entries
            timeline.append({
                "event": log_entry,
                "timestamp": "N/A"
            })

        # Add key milestones
        if state.get("anomaly"):
            timeline.append({
                "event": "Anomaly detected",
                "timestamp": state.get("anomaly").time_window.get("start", "N/A") if hasattr(state.get("anomaly").time_window, "get") else "N/A"
            })

        if state.get("diagnosis"):
            timeline.append({
                "event": "Root cause identified",
                "timestamp": "N/A"
            })

        if state.get("repair"):
            timeline.append({
                "event": f"Repair completed: {state.get('repair').status}",
                "timestamp": "N/A"
            })

        return timeline

    def _generate_root_cause_section(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate root cause section.

        Args:
            state: Current workflow state

        Returns:
            Dict[str, Any]: Root cause information
        """
        diagnosis = state.get("diagnosis")

        if not diagnosis:
            return {"cause": "Unknown", "confidence": 0.0}

        root_causes = []
        for cause in diagnosis.root_causes:
            root_causes.append({
                "entity": cause.get("entity", "unknown"),
                "cause": cause.get("cause", "unknown"),
                "category": cause.get("category", "unknown"),
                "confidence": f"{cause.get('confidence', 0):.1%}"
            })

        return {
            "root_causes": root_causes,
            "propagation_path": diagnosis.propagation_path,
            "confidence": f"{diagnosis.confidence:.1%}"
        }

    def _generate_actions_section(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate actions taken section.

        Args:
            state: Current workflow state

        Returns:
            List[Dict[str, Any]]: Actions taken
        """
        decision = state.get("decision")
        repair = state.get("repair")

        actions = []

        if decision:
            for action in decision.actions:
                actions.append({
                    "type": action.get("type", "unknown"),
                    "description": action.get("description", "unknown"),
                    "target": action.get("target", "unknown")
                })

        if repair and repair.executed_actions:
            for executed in repair.executed_actions:
                action_info = executed.get("action", {})
                actions.append({
                    "type": action_info.get("type", "unknown"),
                    "status": executed.get("status", "unknown"),
                    "result": "Executed"
                })

        return actions

    def _generate_metrics_section(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate metrics section.

        Args:
            state: Current workflow state

        Returns:
            Dict[str, Any]: Metrics information
        """
        obs_data = state.get("observability")
        anomaly = state.get("anomaly")

        metrics = {
            "detection_confidence": anomaly.confidence if anomaly else 0.0,
            "severity": anomaly.severity if anomaly else "UNKNOWN",
            "affected_entities": anomaly.affected_entities if anomaly else [],
            "data_points_collected": len(obs_data.metrics) + len(obs_data.logs) if obs_data else 0
        }

        return metrics

    def _generate_recommendations(
        self,
        state: Dict[str, Any]
    ) -> List[str]:
        """
        Generate improvement recommendations.

        Args:
            state: Current workflow state

        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        diagnosis = state.get("diagnosis")
        anomaly = state.get("anomaly")

        if not diagnosis:
            return ["Conduct thorough investigation to identify root cause."]

        # Generate recommendations based on cause category
        for cause in diagnosis.root_causes:
            category = cause.get("category", "")
            entity = cause.get("entity", "")

            if category == "resource_exhaustion":
                recommendations.append(f"Increase resource limits for {entity}")
                recommendations.append("Implement proactive resource monitoring alerts")
                recommendations.append("Consider auto-scaling policies")

            elif category == "application_error":
                recommendations.append(f"Review recent deployments to {entity}")
                recommendations.append("Improve application error handling")
                recommendations.append("Add integration tests for edge cases")

            elif category == "network_issue":
                recommendations.append("Review network policies and configurations")
                recommendations.append("Implement connection pooling")
                recommendations.append("Add retry logic with exponential backoff")

            elif category == "performance_degradation":
                recommendations.append(f"Profile {entity} for performance bottlenecks")
                recommendations.append("Consider caching strategies")
                recommendations.append("Review database query performance")

        # Add general recommendations
        if anomaly and anomaly.severity in ["HIGH", "CRITICAL"]:
            recommendations.append("Conduct post-incident review within 48 hours")
            recommendations.append("Update runbook with lessons learned")

        return list(set(recommendations))  # Remove duplicates

    def _format_report(self, report: Dict[str, Any]) -> str:
        """
        Format report as readable text.

        Args:
            report: Report data dictionary

        Returns:
            str: Formatted report text
        """
        lines = [
            "=" * 60,
            "INCIDENT REPORT",
            "=" * 60,
            "",
            f"Incident ID: {report['incident_id']}",
            f"Generated: {report['generated_at']}",
            f"Status: {report['status']}",
            "",
            "-" * 60,
            "SUMMARY",
            "-" * 60,
            report['summary'],
            "",
            "-" * 60,
            "ROOT CAUSE",
            "-" * 60,
        ]

        rc = report['root_cause']
        if isinstance(rc, dict):
            for cause in rc.get('root_causes', []):
                lines.append(f"- {cause.get('cause')} on {cause.get('entity')}")

        lines.extend([
            "",
            "-" * 60,
            "ACTIONS TAKEN",
            "-" * 60,
        ])

        for action in report['actions_taken']:
            lines.append(f"- {action.get('description', action.get('type'))}")

        if report['recommendations']:
            lines.extend([
                "",
                "-" * 60,
                "RECOMMENDATIONS",
                "-" * 60,
            ])
            for rec in report['recommendations']:
                lines.append(f"- {rec}")

        lines.extend([
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ])

        return "\n".join(lines)

    def _generate_executive_summary(self, report: Dict[str, Any]) -> str:
        """
        Generate LLM-powered executive summary.

        Args:
            report: Report data dictionary

        Returns:
            str: Executive summary
        """
        prompt = f"""
        Generate a concise executive summary for this incident report:

        Incident ID: {report['incident_id']}
        Status: {report['status']}
        Summary: {report['summary']}
        Root Cause: {report['root_cause']}
        Recommendations: {report['recommendations'][:3]}

        Provide a 2-3 sentence executive summary suitable for leadership.
        """

        try:
            response = self.llm.invoke(prompt)
            self.log_llm(prompt, response.content)
            return response.content
        except Exception:
            return report['summary']

    def send_notification(
        self,
        report: Dict[str, Any],
        channel: str,
        recipients: List[str]
    ) -> bool:
        """
        Send report notification via specified channel.

        Args:
            report: Generated report
            channel: Notification channel (wechat, slack, email)
            recipients: List of recipients

        Returns:
            bool: True if notification sent successfully

        Note:
            This would integrate with the integrations module
            for actual notification sending.
        """
        self.log(f"Sending report via {channel} to {recipients}")

        # Placeholder for notification integration
        # In production, integrate with:
        # - WeChatTool for Enterprise WeChat
        # - Slack SDK for Slack
        # - SMTP for email

        return True
