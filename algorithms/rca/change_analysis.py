"""
Change Analysis Algorithm

Root cause analysis by examining recent changes to the system.
Based on the principle that most incidents are caused by recent changes.

This implementation:
1. Collects recent changes (deployments, config updates, etc.)
2. Correlates changes with incident timing
3. Scores changes by risk and temporal proximity
4. Identifies the most likely change-related cause
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from langchain_core.tools import BaseTool
from pydantic import BaseModel


@dataclass
class Change:
    """Represents a system change."""
    id: str
    change_type: str  # deployment, config, data, infrastructure, process
    description: str
    timestamp: datetime
    author: str = "unknown"
    risk_level: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    affected_components: List[str] = field(default_factory=list)
    rollback_available: bool = True
    change_score: float = 0.0  # Calculated risk score


@dataclass
class ChangeAnalysisResult:
    """Result of change analysis."""
    incident_time: datetime
    changes_analyzed: int
    suspicious_changes: List[Change] = field(default_factory=list)
    most_likely_cause: Optional[Change] = None
    correlation_confidence: float = 0.0
    recommendation: str = ""


class ChangeAnalysisInput(BaseModel):
    """Input schema for Change Analysis."""
    incident_time: str  # ISO format datetime
    affected_services: Optional[List[str]] = None
    time_window_hours: int = 24
    changes: Optional[List[Dict[str, Any]]] = None


class ChangeAnalyzer:
    """
    Implements Change Analysis for root cause identification.

    The key insight: most incidents are caused by recent changes.
    This analyzer:
    1. Collects changes within a time window
    2. Scores changes by risk, timing, and affected components
    3. Correlates changes with the incident
    4. Identifies the most suspicious change

    Attributes:
        llm: Language model for analysis
        verbose (bool): Enable verbose output
    """

    def __init__(self, llm=None, verbose: bool = False):
        """
        Initialize the Change analyzer.

        Args:
            llm: Language model instance
            verbose (bool): Enable detailed logging
        """
        self.llm = llm
        self.verbose = verbose

    def analyze(
        self,
        incident_time: datetime,
        changes: List[Change],
        affected_services: Optional[List[str]] = None,
        time_window_hours: int = 24
    ) -> ChangeAnalysisResult:
        """
        Analyze changes to find potential root causes.

        Args:
            incident_time (datetime): When the incident occurred
            changes (List[Change]): List of recent changes
            affected_services (Optional[List[str]]): Affected service names
            time_window_hours (int): Hours before incident to consider

        Returns:
            ChangeAnalysisResult: Analysis results
        """
        # Filter changes within time window
        window_start = incident_time - timedelta(hours=time_window_hours)
        relevant_changes = [
            c for c in changes
            if window_start <= c.timestamp <= incident_time
        ]

        # Score each change
        for change in relevant_changes:
            change.change_score = self._score_change(
                change, incident_time, affected_services
            )

        # Sort by score (most suspicious first)
        relevant_changes.sort(key=lambda c: c.change_score, reverse=True)

        # Build result
        result = ChangeAnalysisResult(
            incident_time=incident_time,
            changes_analyzed=len(relevant_changes),
            suspicious_changes=relevant_changes[:10]  # Top 10
        )

        if relevant_changes:
            result.most_likely_cause = relevant_changes[0]
            result.correlation_confidence = min(relevant_changes[0].change_score, 0.95)
            result.recommendation = self._generate_recommendation(result)

        return result

    def _score_change(
        self,
        change: Change,
        incident_time: datetime,
        affected_services: Optional[List[str]]
    ) -> float:
        """
        Calculate a risk score for a change.

        Scoring factors:
        - Temporal proximity (closer = higher risk)
        - Change type (deployments riskier than docs)
        - Risk level declared
        - Affected component overlap
        - Rollback availability
        """
        score = 0.0

        # 1. Temporal proximity (0-0.3 points)
        time_delta = incident_time - change.timestamp
        hours_before = time_delta.total_seconds() / 3600

        if hours_before < 1:
            score += 0.3  # Within 1 hour - highest risk
        elif hours_before < 4:
            score += 0.25
        elif hours_before < 12:
            score += 0.15
        elif hours_before < 24:
            score += 0.1
        else:
            score += 0.05

        # 2. Change type risk (0-0.25 points)
        type_risk = {
            "deployment": 0.25,
            "infrastructure": 0.2,
            "config": 0.15,
            "data": 0.15,
            "process": 0.1,
            "documentation": 0.05,
        }
        score += type_risk.get(change.change_type.lower(), 0.1)

        # 3. Declared risk level (0-0.25 points)
        risk_scores = {
            "HIGH": 0.25,
            "MEDIUM": 0.15,
            "LOW": 0.05,
        }
        score += risk_scores.get(change.risk_level, 0.1)

        # 4. Affected component overlap (0-0.2 points)
        if affected_services:
            overlap = set(change.affected_components) & set(affected_services)
            if overlap:
                score += 0.2  # Direct overlap
            elif change.affected_components:
                score += 0.1  # Some components mentioned
        else:
            score += 0.1  # Unknown

        # 5. Rollback penalty (0-0.1 points)
        if not change.rollback_available:
            score += 0.1  # Can't roll back = riskier

        return min(score, 1.0)

    def _generate_recommendation(self, result: ChangeAnalysisResult) -> str:
        """Generate actionable recommendation."""
        if not result.most_likely_cause:
            return "No suspicious changes found in the time window."

        change = result.most_likely_cause

        rec = f"Most suspicious change: {change.description} "
        rec += f"(Change Score: {change.change_score:.2f}). "

        if change.rollback_available:
            rec += "Recommendation: Consider rolling back this change to verify causation."
        else:
            rec += "Recommendation: This change cannot be rolled back. "
            rec += "Investigate the change details and consider mitigation strategies."

        return rec


class ChangeTool(BaseTool):
    """
    LangChain tool for Change Analysis.
    """

    name: str = "change_analysis"
    description: str = "Analyze recent changes to identify the change most likely causing an incident"
    args_schema: type[BaseModel] = ChangeAnalysisInput

    analyzer: Optional[ChangeAnalyzer] = None

    def __init__(self, llm=None, **kwargs):
        super().__init__(**kwargs)
        self.analyzer = ChangeAnalyzer(llm=llm)

    def _run(
        self,
        incident_time: str,
        affected_services: Optional[List[str]] = None,
        time_window_hours: int = 24,
        changes: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute Change Analysis.

        Args:
            incident_time (str): ISO format datetime of incident
            affected_services (Optional[List[str]]): Affected services
            time_window_hours (int): Hours to look back
            changes (Optional[List[Dict]]]): List of changes (or use mock)

        Returns:
            Dict[str, Any]: Analysis results
        """
        # Parse incident time
        try:
            inc_time = datetime.fromisoformat(incident_time)
        except ValueError:
            inc_time = datetime.now()

        # Convert or generate changes
        if changes:
            change_objects = self._parse_changes(changes)
        else:
            # Generate mock changes for demonstration
            change_objects = self._generate_mock_changes(inc_time)

        # Run analysis
        result = self.analyzer.analyze(
            inc_time,
            change_objects,
            affected_services,
            time_window_hours
        )

        # Convert to JSON-serializable format
        suspicious = []
        for c in result.suspicious_changes:
            suspicious.append({
                "id": c.id,
                "change_type": c.change_type,
                "description": c.description,
                "timestamp": c.timestamp.isoformat(),
                "risk_level": c.risk_level,
                "affected_components": c.affected_components,
                "change_score": c.change_score
            })

        return {
            "incident_time": result.incident_time.isoformat(),
            "changes_analyzed": result.changes_analyzed,
            "suspicious_changes": suspicious,
            "most_likely_cause": {
                "id": result.most_likely_cause.id if result.most_likely_cause else None,
                "description": result.most_likely_cause.description if result.most_likely_cause else None,
                "change_score": result.most_likely_cause.change_score if result.most_likely_cause else 0,
            } if result.most_likely_cause else None,
            "correlation_confidence": result.correlation_confidence,
            "recommendation": result.recommendation
        }

    def _parse_changes(self, changes: List[Dict[str, Any]]) -> List[Change]:
        """Parse change dictionaries into Change objects."""
        result = []
        for c in changes:
            try:
                ts = datetime.fromisoformat(c.get("timestamp", datetime.now().isoformat()))
            except ValueError:
                ts = datetime.now()

            result.append(Change(
                id=c.get("id", "unknown"),
                change_type=c.get("change_type", "deployment"),
                description=c.get("description", "Unknown change"),
                timestamp=ts,
                author=c.get("author", "unknown"),
                risk_level=c.get("risk_level", "MEDIUM"),
                affected_components=c.get("affected_components", []),
                rollback_available=c.get("rollback_available", True)
            ))
        return result

    def _generate_mock_changes(self, incident_time: datetime) -> List[Change]:
        """Generate mock changes for demonstration."""
        return [
            Change(
                id="chg-001",
                change_type="deployment",
                description="Deploy api-service v2.3.1 with new caching layer",
                timestamp=incident_time - timedelta(hours=2),
                author="CI/CD Pipeline",
                risk_level="HIGH",
                affected_components=["api-service", "cache"],
                rollback_available=True,
                change_score=0.0
            ),
            Change(
                id="chg-002",
                change_type="config",
                description="Update database connection pool size from 50 to 100",
                timestamp=incident_time - timedelta(hours=4),
                author="admin",
                risk_level="MEDIUM",
                affected_components=["database", "api-service"],
                rollback_available=True,
                change_score=0.0
            ),
            Change(
                id="chg-003",
                change_type="infrastructure",
                description="Scale down worker nodes from 10 to 8",
                timestamp=incident_time - timedelta(hours=6),
                author="ops-team",
                risk_level="LOW",
                affected_components=["worker-service"],
                rollback_available=True,
                change_score=0.0
            ),
            Change(
                id="chg-004",
                change_type="data",
                description="Run data migration for user profile schema",
                timestamp=incident_time - timedelta(hours=12),
                author="data-team",
                risk_level="HIGH",
                affected_components=["user-service", "database"],
                rollback_available=False,
                change_score=0.0
            ),
            Change(
                id="chg-005",
                change_type="deployment",
                description="Update nginx ingress controller configuration",
                timestamp=incident_time - timedelta(hours=1),
                author="platform-team",
                risk_level="MEDIUM",
                affected_components=["ingress", "api-service"],
                rollback_available=True,
                change_score=0.0
            ),
        ]
