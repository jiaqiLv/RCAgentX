"""
Prompt Templates for AIOps Agents

Centralized prompt template management for all agents.
Provides consistent, well-tested prompts for LLM interactions.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """
    Prompt template structure.

    Attributes:
        name (str): Template identifier
        template (str): Prompt template string
        variables (List[str]): List of template variables
    """
    name: str
    template: str
    variables: List[str]


class PromptTemplates:
    """
    Centralized prompt template registry.

    Provides access to all prompt templates used across the
    AIOps agent system.

    Example:
        ```python
        templates = PromptTemplates()

        # Get observability summary template
        summary_template = templates.get("observability_summary")

        # Format with variables
        prompt = summary_template.format(
            metrics=metrics_data,
            logs=log_data
        )
        ```
    """

    def __init__(self):
        """Initialize prompt templates."""
        self.templates: Dict[str, PromptTemplate] = {}
        self._init_templates()

    def _init_templates(self):
        """Initialize all prompt templates."""

        # Observability summary template
        self.templates["observability_summary"] = PromptTemplate(
            name="observability_summary",
            template="""
Analyze the following observability data and provide a concise summary:

## Metrics Data
{metrics}

## Error Logs
{logs}

Please provide:
1. Key anomalies detected in metrics
2. Error patterns observed in logs
3. Potential correlations between metrics and logs
4. Most affected services/components

Keep the summary concise and actionable.
""",
            variables=["metrics", "logs"]
        )

        # Root cause analysis template
        self.templates["root_cause_analysis"] = PromptTemplate(
            name="root_cause_analysis",
            template="""
Analyze the following incident data and identify the root cause:

## Anomaly Information
Type: {anomaly_type}
Severity: {severity}
Confidence: {confidence}
Affected Entities: {entities}
Evidence: {evidence}

## Recent Error Logs
{logs}

Please identify:
1. The most likely root cause entity
2. The underlying cause category
3. Supporting evidence

Respond in JSON format:
{{
    "root_causes": [
        {{
            "entity": "...",
            "cause": "...",
            "category": "...",
            "confidence": 0.0-1.0,
            "evidence": "..."
        }}
    ]
}}
""",
            variables=["anomaly_type", "severity", "confidence", "entities", "evidence", "logs"]
        )

        # Remediation strategy template
        self.templates["remediation_strategy"] = PromptTemplate(
            name="remediation_strategy",
            template="""
Generate a remediation strategy based on the following diagnosis:

## Root Cause
{root_cause}

## Diagnosis Details
Category: {category}
Confidence: {confidence}
Affected Entity: {entity}

## Historical Strategies
{historical_strategies}

Please provide:
1. Recommended actions in order
2. Risk assessment (LOW/MEDIUM/HIGH)
3. Rollback plan
4. Estimated recovery time

Respond in JSON format:
{{
    "actions": [
        {{
            "type": "...",
            "target": "...",
            "parameters": {{}},
            "description": "..."
        }}
    ],
    "risk_level": "...",
    "rollback_plan": "...",
    "estimated_recovery_time": "..."
}}
""",
            variables=["root_cause", "category", "confidence", "entity", "historical_strategies"]
        )

        # Executive summary template
        self.templates["executive_summary"] = PromptTemplate(
            name="executive_summary",
            template="""
Generate a concise executive summary for this incident report:

## Incident Details
Incident ID: {incident_id}
Status: {status}
Severity: {severity}

## Summary
{summary}

## Root Cause
{root_cause}

## Recommendations
{recommendations}

Provide a 2-3 sentence executive summary suitable for leadership.
Focus on business impact and resolution status.
""",
            variables=["incident_id", "status", "severity", "summary", "root_cause", "recommendations"]
        )

        # Anomaly classification template
        self.templates["anomaly_classification"] = PromptTemplate(
            name="anomaly_classification",
            template="""
Classify the following metrics and logs to detect anomalies:

## Metrics
{metrics}

## Logs
{logs}

## Known Patterns
- CPU spike: >80% utilization
- Memory leak: Continuous growth pattern
- Error burst: Sudden increase in error rate
- Latency issue: P99 > 1 second

Identify any anomalies and classify them by:
1. Type (cpu, memory, error, latency)
2. Severity (LOW, MEDIUM, HIGH, CRITICAL)
3. Confidence score (0.0-1.0)

Respond in JSON format:
{{
    "anomalies": [
        {{
            "type": "...",
            "severity": "...",
            "confidence": 0.0,
            "description": "..."
        }}
    ]
}}
""",
            variables=["metrics", "logs"]
        )

        # Impact assessment template
        self.templates["impact_assessment"] = PromptTemplate(
            name="impact_assessment",
            template="""
Assess the business impact of this incident:

## Affected Components
{affected_entities}

## Error Patterns
{error_patterns}

## Duration
{duration}

Please assess:
1. User impact (number of affected users if available)
2. Business impact (revenue, SLA, reputation)
3. Data integrity impact

Respond in JSON format:
{{
    "user_impact": "...",
    "business_impact": "...",
    "data_impact": "...",
    "sla_impact": true/false
}}
""",
            variables=["affected_entities", "error_patterns", "duration"]
        )

    def get(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a prompt template by name.

        Args:
            name (str): Template name

        Returns:
            Optional[PromptTemplate]: Template if found, None otherwise

        Example:
            ```python
            template = templates.get("observability_summary")
            ```
        """
        return self.templates.get(name)

    def render(self, name: str, **kwargs) -> str:
        """
        Render a template with provided variables.

        Args:
            name (str): Template name
            **kwargs: Template variables as keyword arguments

        Returns:
            str: Rendered template string

        Raises:
            KeyError: If template not found

        Example:
            ```python
            prompt = templates.render(
                "observability_summary",
                metrics=metrics_data,
                logs=log_data
            )
            ```
        """
        template = self.templates.get(name)
        if not template:
            raise KeyError(f"Template '{name}' not found")

        return template.template.format(**kwargs)

    def list_templates(self) -> List[str]:
        """
        List all available template names.

        Returns:
            List[str]: List of template names

        Example:
            ```python
            names = templates.list_templates()
            ```
        """
        return list(self.templates.keys())

    def add_template(
        self,
        name: str,
        template: str,
        variables: List[str]
    ):
        """
        Add a custom template.

        Args:
            name (str): Template name
            template (str): Template string
            variables (List[str]): List of variable names

        Example:
            ```python
            templates.add_template(
                "custom_analysis",
                "Analyze: {data}",
                ["data"]
            )
            ```
        """
        self.templates[name] = PromptTemplate(
            name=name,
            template=template,
            variables=variables
        )
