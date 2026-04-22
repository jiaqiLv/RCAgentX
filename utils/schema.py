"""
Unified Data Schema Utilities

Provides utilities for creating and validating data schemas
used across the AIOps system.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


def create_schema(
    name: str,
    fields: Dict[str, type]
) -> type:
    """
    Dynamically create a dataclass schema.

    Creates a dataclass with the specified fields for use as
    a data schema.

    Args:
        name (str): Schema name
        fields (Dict[str, type]): Field definitions

    Returns:
        type: Created dataclass

    Example:
        ```python
        MetricSchema = create_schema(
            "MetricSchema",
            {"name": str, "value": float, "timestamp": datetime}
        )
        metric = MetricSchema(name="cpu", value=0.95, timestamp=datetime.now())
        ```
    """
    return dataclass(
        type(name, (), {"__annotations__": fields})
    )


def schema_to_dict(schema_instance: Any) -> Dict[str, Any]:
    """
    Convert a schema instance to dictionary.

    Args:
        schema_instance: Dataclass instance

    Returns:
        Dict[str, Any]: Dictionary representation
    """
    return asdict(schema_instance)


def validate_schema(
    data: Dict[str, Any],
    required_fields: List[str],
    optional_fields: Optional[List[str]] = None
) -> tuple[bool, List[str]]:
    """
    Validate data against schema requirements.

    Args:
        data (Dict[str, Any]): Data to validate
        required_fields (List[str]): List of required field names
        optional_fields (Optional[List[str]]): List of optional field names

    Returns:
        tuple[bool, List[str]]: (is_valid, list of missing fields)

    Example:
        ```python
        is_valid, missing = validate_schema(
            data={"name": "cpu", "value": 0.95},
            required_fields=["name", "value"],
            optional_fields=["timestamp"]
        )
        ```
    """
    missing = []
    for field in required_fields:
        if field not in data:
            missing.append(field)

    return len(missing) == 0, missing


@dataclass
class IncidentSchema:
    """Standard incident data schema."""
    incident_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    severity: str
    affected_services: List[str]


@dataclass
class MetricSchema:
    """Standard metric data schema."""
    name: str
    value: float
    unit: str
    labels: Dict[str, str]
    timestamp: datetime


@dataclass
class LogEntrySchema:
    """Standard log entry schema."""
    timestamp: datetime
    level: str
    service: str
    message: str
    labels: Dict[str, str]


@dataclass
class TraceSchema:
    """Standard trace data schema."""
    trace_id: str
    span_id: str
    service: str
    operation: str
    duration_ms: float
    status: str
    tags: Dict[str, str]


class SchemaValidator:
    """
    Schema validation utility class.

    Provides methods for validating data against standard schemas.

    Example:
        ```python
        validator = SchemaValidator()

        # Validate incident data
        is_valid = validator.validate_incident(incident_data)

        # Validate metric data
        is_valid = validator.validate_metric(metric_data)
        ```
    """

    def __init__(self):
        """Initialize schema validator."""
        self.required_fields = {
            "incident": ["incident_id", "status", "severity"],
            "metric": ["name", "value", "timestamp"],
            "log": ["timestamp", "level", "message"],
            "trace": ["trace_id", "span_id", "service", "operation"],
        }

    def validate_incident(self, data: Dict[str, Any]) -> bool:
        """Validate incident data."""
        is_valid, _ = validate_schema(
            data,
            self.required_fields["incident"]
        )
        return is_valid

    def validate_metric(self, data: Dict[str, Any]) -> bool:
        """Validate metric data."""
        is_valid, _ = validate_schema(
            data,
            self.required_fields["metric"]
        )
        return is_valid

    def validate_log(self, data: Dict[str, Any]) -> bool:
        """Validate log data."""
        is_valid, _ = validate_schema(
            data,
            self.required_fields["log"]
        )
        return is_valid

    def validate_trace(self, data: Dict[str, Any]) -> bool:
        """Validate trace data."""
        is_valid, _ = validate_schema(
            data,
            self.required_fields["trace"]
        )
        return is_valid
