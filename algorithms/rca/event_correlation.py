"""
Event Correlation Analysis

Identifies patterns and correlations between multiple events to find
the root cause. Uses statistical methods and temporal analysis to
determine which events are related and which is the primary cause.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from langchain_core.tools import BaseTool
from pydantic import BaseModel


@dataclass
class Event:
    """Represents a system event."""
    id: str
    timestamp: datetime
    source: str
    event_type: str
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    description: str
    affected_component: str
    correlation_id: Optional[str] = None


@dataclass
class EventCluster:
    """A cluster of related events."""
    events: List[Event] = field(default_factory=list)
    primary_event: Optional[Event] = None
    time_span_seconds: float = 0.0
    affected_components: List[str] = field(default_factory=list)
    cluster_confidence: float = 0.0


@dataclass
class CorrelationResult:
    """Result of event correlation analysis."""
    total_events: int
    clusters: List[EventCluster] = field(default_factory=list)
    root_cause_event: Optional[Event] = None
    cascade_chain: List[Event] = field(default_factory=list)
    correlation_confidence: float = 0.0


class EventCorrelationInput(BaseModel):
    """Input schema for Event Correlation."""
    events: Optional[List[Dict[str, Any]]] = None
    time_window_minutes: int = 30
    correlation_threshold: float = 0.5


class EventCorrelator:
    """
    Implements Event Correlation Analysis.

    This algorithm:
    1. Groups events by temporal proximity
    2. Identifies cascade patterns
    3. Finds the primary (root cause) event
    4. Builds the failure cascade chain

    Attributes:
        llm: Language model for pattern analysis
        verbose (bool): Enable verbose output
    """

    def __init__(self, llm=None, verbose: bool = False):
        """
        Initialize the Event Correlator.

        Args:
            llm: Language model instance
            verbose (bool): Enable detailed logging
        """
        self.llm = llm
        self.verbose = verbose

    def correlate(
        self,
        events: List[Event],
        time_window_minutes: int = 30,
        correlation_threshold: float = 0.5
    ) -> CorrelationResult:
        """
        Perform event correlation analysis.

        Args:
            events (List[Event]): List of events to analyze
            time_window_minutes (int): Time window for clustering
            correlation_threshold (float): Minimum correlation score

        Returns:
            CorrelationResult: Correlation analysis result
        """
        if not events:
            return CorrelationResult(total_events=0)

        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        # Cluster events by time proximity
        clusters = self._cluster_events(sorted_events, time_window_minutes)

        # Score and rank clusters
        for cluster in clusters:
            self._score_cluster(cluster)

        # Find root cause event (earliest high-severity event in largest cluster)
        root_cause = self._find_root_cause(clusters)

        # Build cascade chain
        cascade_chain = self._build_cascade_chain(root_cause, clusters)

        # Build result
        result = CorrelationResult(
            total_events=len(events),
            clusters=clusters,
            root_cause_event=root_cause,
            cascade_chain=cascade_chain
        )

        if root_cause and clusters:
            result.correlation_confidence = max(c.cluster_confidence for c in clusters)

        return result

    def _cluster_events(
        self,
        events: List[Event],
        time_window_minutes: int
    ) -> List[EventCluster]:
        """Cluster events by temporal proximity."""
        if not events:
            return []

        clusters = []
        current_cluster = EventCluster(events=[events[0]])

        time_delta = timedelta(minutes=time_window_minutes)

        for event in events[1:]:
            # Check if event belongs to current cluster
            if event.timestamp - current_cluster.events[-1].timestamp <= time_delta:
                current_cluster.events.append(event)
            else:
                # Finalize current cluster and start new one
                if len(current_cluster.events) >= 2:  # Only keep clusters with 2+ events
                    self._finalize_cluster(current_cluster)
                    clusters.append(current_cluster)
                current_cluster = EventCluster(events=[event])

        # Don't forget the last cluster
        if len(current_cluster.events) >= 2:
            self._finalize_cluster(current_cluster)
            clusters.append(current_cluster)

        # Sort clusters by size (largest first)
        clusters.sort(key=lambda c: len(c.events), reverse=True)

        return clusters

    def _finalize_cluster(self, cluster: EventCluster):
        """Finalize cluster calculations."""
        if len(cluster.events) < 2:
            return

        # Calculate time span
        times = [e.timestamp for e in cluster.events]
        cluster.time_span_seconds = (max(times) - min(times)).total_seconds()

        # Collect affected components
        cluster.affected_components = list(set(
            e.affected_component for e in cluster.events
        ))

    def _score_cluster(self, cluster: EventCluster):
        """Calculate a confidence score for the cluster."""
        score = 0.0

        # More events = higher confidence (up to a point)
        event_score = min(len(cluster.events) / 10, 0.3)
        score += event_score

        # Shorter time span = tighter correlation
        if cluster.time_span_seconds < 60:  # Less than 1 minute
            score += 0.3
        elif cluster.time_span_seconds < 300:  # Less than 5 minutes
            score += 0.2
        elif cluster.time_span_seconds < 900:  # Less than 15 minutes
            score += 0.1

        # More affected components = broader impact
        component_score = min(len(cluster.affected_components) / 5, 0.2)
        score += component_score

        # Severity distribution
        severities = [e.severity for e in cluster.events]
        critical_count = severities.count("CRITICAL")
        error_count = severities.count("ERROR")

        if critical_count > 0:
            score += 0.2
        elif error_count > len(severities) / 2:
            score += 0.15

        cluster.cluster_confidence = min(score, 1.0)

    def _find_root_cause(self, clusters: List[EventCluster]) -> Optional[Event]:
        """Find the most likely root cause event."""
        if not clusters:
            return None

        # Look at the largest cluster
        primary_cluster = clusters[0]

        # Find the earliest high-severity event
        high_severity_events = [
            e for e in primary_cluster.events
            if e.severity in ["CRITICAL", "ERROR"]
        ]

        if high_severity_events:
            # Sort by timestamp and return earliest
            high_severity_events.sort(key=lambda e: e.timestamp)
            return high_severity_events[0]

        # If no high-severity events, return earliest event
        primary_cluster.events.sort(key=lambda e: e.timestamp)
        return primary_cluster.events[0]

    def _build_cascade_chain(
        self,
        root_cause: Optional[Event],
        clusters: List[EventCluster]
    ) -> List[Event]:
        """Build the cascade chain from root cause."""
        if not root_cause or not clusters:
            return []

        # Find the cluster containing root cause
        root_cluster = None
        for cluster in clusters:
            if root_cause in cluster.events:
                root_cluster = cluster
                break

        if not root_cluster:
            return [root_cause]

        # Build chain based on timing and component relationships
        chain = [root_cause]

        # Get remaining events sorted by time
        remaining = [e for e in root_cluster.events if e != root_cause]
        remaining.sort(key=lambda e: e.timestamp)

        # Add events that could be downstream effects
        for event in remaining:
            # Check if this could be a downstream effect
            if self._is_downstream_effect(root_cause, event):
                chain.append(event)

        return chain

    def _is_downstream_effect(self, cause: Event, effect: Event) -> bool:
        """Determine if an event could be a downstream effect of another."""
        # Must happen after the cause
        if effect.timestamp <= cause.timestamp:
            return False

        # Check for component relationships
        related_components = {
            "database": ["api-service", "backend", "cache"],
            "api-service": ["frontend", "gateway", "ingress"],
            "cache": ["api-service", "backend"],
            "network": ["api-service", "database", "cache", "frontend"],
        }

        cause_related = related_components.get(cause.affected_component, [])
        effect_related = related_components.get(effect.affected_component, [])

        # Direct relationship
        if effect.affected_component in cause_related:
            return True

        # Shared relationship
        if effect.affected_component in effect_related and cause.affected_component in effect_related:
            return True

        # Same component
        if cause.affected_component == effect.affected_component:
            return True

        return False


class EventCorrelatorTool(BaseTool):
    """
    LangChain tool for Event Correlation Analysis.
    """

    name: str = "event_correlation"
    description: str = "Correlate multiple events to identify patterns and find the root cause event"
    args_schema: type[BaseModel] = EventCorrelationInput

    correlator: Optional[EventCorrelator] = None

    def __init__(self, llm=None, **kwargs):
        super().__init__(**kwargs)
        self.correlator = EventCorrelator(llm=llm)

    def _run(
        self,
        events: Optional[List[Dict[str, Any]]] = None,
        time_window_minutes: int = 30,
        correlation_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Execute Event Correlation Analysis.

        Args:
            events (Optional[List[Dict]]): List of events (or use mock)
            time_window_minutes (int): Time window for clustering
            correlation_threshold (float): Minimum correlation score

        Returns:
            Dict[str, Any]: Analysis results
        """
        # Convert or generate events
        if events:
            event_objects = self._parse_events(events)
        else:
            # Generate mock events
            event_objects = self._generate_mock_events()

        # Run correlation
        result = self.correlator.correlate(
            event_objects,
            time_window_minutes,
            correlation_threshold
        )

        # Convert to JSON-serializable format
        def event_to_dict(e: Event) -> Dict:
            return {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "source": e.source,
                "event_type": e.event_type,
                "severity": e.severity,
                "description": e.description,
                "affected_component": e.affected_component
            }

        clusters_data = []
        for cluster in result.clusters:
            clusters_data.append({
                "event_count": len(cluster.events),
                "time_span_seconds": cluster.time_span_seconds,
                "affected_components": cluster.affected_components,
                "confidence": cluster.cluster_confidence,
                "events": [event_to_dict(e) for e in cluster.events[:5]]  # Limit to 5
            })

        return {
            "total_events": result.total_events,
            "cluster_count": len(result.clusters),
            "clusters": clusters_data,
            "root_cause_event": event_to_dict(result.root_cause_event) if result.root_cause_event else None,
            "cascade_chain": [event_to_dict(e) for e in result.cascade_chain[:10]],
            "correlation_confidence": result.correlation_confidence
        }

    def _parse_events(self, events: List[Dict[str, Any]]) -> List[Event]:
        """Parse event dictionaries into Event objects."""
        result = []
        for e in events:
            try:
                ts = datetime.fromisoformat(e.get("timestamp", datetime.now().isoformat()))
            except ValueError:
                ts = datetime.now()

            result.append(Event(
                id=e.get("id", "unknown"),
                timestamp=ts,
                source=e.get("source", "unknown"),
                event_type=e.get("event_type", "unknown"),
                severity=e.get("severity", "INFO"),
                description=e.get("description", ""),
                affected_component=e.get("affected_component", "unknown"),
                correlation_id=e.get("correlation_id")
            ))
        return result

    def _generate_mock_events(self) -> List[Event]:
        """Generate mock events for demonstration."""
        base_time = datetime.now() - timedelta(minutes=30)

        return [
            Event(
                id="evt-001",
                timestamp=base_time,
                source="database",
                event_type="CONNECTION_EXHAUSTED",
                severity="CRITICAL",
                description="Database connection pool exhausted - all 100 connections in use",
                affected_component="database"
            ),
            Event(
                id="evt-002",
                timestamp=base_time + timedelta(seconds=30),
                source="api-service",
                event_type="TIMEOUT",
                severity="ERROR",
                description="Request timeout waiting for database connection",
                affected_component="api-service"
            ),
            Event(
                id="evt-003",
                timestamp=base_time + timedelta(seconds=45),
                source="api-service",
                event_type="ERROR_RATE_HIGH",
                severity="ERROR",
                description="Error rate exceeded 10% threshold",
                affected_component="api-service"
            ),
            Event(
                id="evt-004",
                timestamp=base_time + timedelta(seconds=60),
                source="gateway",
                event_type="UPSTREAM_ERROR",
                severity="WARNING",
                description="Upstream api-service returning 5xx errors",
                affected_component="gateway"
            ),
            Event(
                id="evt-005",
                timestamp=base_time + timedelta(seconds=90),
                source="frontend",
                event_type="LATENCY_HIGH",
                severity="WARNING",
                description="Page load time exceeded 5 seconds",
                affected_component="frontend"
            ),
            Event(
                id="evt-006",
                timestamp=base_time + timedelta(seconds=120),
                source="alertmanager",
                event_type="ALERT_FIRED",
                severity="CRITICAL",
                description="Multiple services in degraded state",
                affected_component="alertmanager"
            ),
        ]
