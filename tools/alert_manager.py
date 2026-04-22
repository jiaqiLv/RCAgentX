"""
Alert Manager Tool - Alert Management and Silencing

LangChain tool for interacting with Alertmanager.
Provides methods for querying alerts, silencing, and managing receivers.
"""

import requests
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class AlertManagerInput(BaseModel):
    """Input schema for AlertManager tool operations."""
    filter: Optional[str] = Field(None, description="Alert filter expression")
    receiver: Optional[str] = Field(None, description="Receiver name")


class AlertManagerTool(BaseTool):
    """
    Tool for managing Prometheus Alertmanager.

    Provides methods for querying active alerts, creating silences,
    and managing alert receivers.

    Attributes:
        name (str): Tool name for LangChain
        description (str): Tool description
        url (str): Alertmanager server URL
        timeout (int): Request timeout in seconds

    Example:
        ```python
        am = AlertManagerTool(url="http://alertmanager:9093")

        # Get active alerts
        alerts = am.get_alerts(filter="severity=critical")

        # Create silence
        am.create_silence(
            matchers={"service": "api-service"},
            duration=timedelta(hours=1),
            comment="Planned maintenance"
        )
        ```
    """

    name: str = "alertmanager"
    description: str = "Manage Prometheus Alertmanager. Query alerts, create silences, and manage receivers."
    url: str
    timeout: int = 30

    args_schema: type[BaseModel] = AlertManagerInput

    def _run(
        self,
        filter: Optional[str] = None,
        receiver: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get active alerts from Alertmanager.

        Args:
            filter (Optional[str]): Filter expression (e.g., "severity=critical")
            receiver (Optional[str]): Filter by receiver name

        Returns:
            List[Dict[str, Any]]: List of active alerts

        Example:
            ```python
            critical_alerts = am.get_alerts(filter="severity=critical")
            for alert in critical_alerts:
                print(f"Alert: {alert['labels']['alertname']}")
                print(f"Severity: {alert['labels']['severity']}")
            ```
        """
        endpoint = f"{self.url}/api/v2/alerts"
        params = {}

        if filter:
            params["filter"] = filter

        response = requests.get(endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()

        alerts = response.json()

        # Filter by receiver if specified
        if receiver:
            alerts = [a for a in alerts if receiver in str(a.get("receivers", []))]

        return alerts

    def get_alert_count(self) -> Dict[str, int]:
        """
        Get count of alerts by status.

        Returns:
            Dict[str, int]: Count of alerts by status (active, suppressed)

        Example:
            ```python
            counts = am.get_alert_count()
            print(f"Active: {counts['active']}, Suppressed: {counts['suppressed']}")
            ```
        """
        alerts = self._run()

        counts = {"active": 0, "suppressed": 0}
        for alert in alerts:
            if alert.get("status", {}).get("state") == "suppressed":
                counts["suppressed"] += 1
            else:
                counts["active"] += 1

        return counts

    def create_silence(
        self,
        matchers: Dict[str, str],
        duration: timedelta,
        comment: str,
        created_by: str = "AIOps Agent"
    ) -> str:
        """
        Create a silence in Alertmanager.

        Args:
            matchers (Dict[str, str]): Label matchers for the silence
            duration (timedelta): How long the silence should last
            comment (str): Reason for the silence
            created_by (str): Creator identifier

        Returns:
            str: Silence ID

        Example:
            ```python
            silence_id = am.create_silence(
                matchers={"service": "api-service", "severity": "critical"},
                duration=timedelta(hours=1),
                comment="Planned maintenance window"
            )
            print(f"Created silence: {silence_id}")
            ```
        """
        endpoint = f"{self.url}/api/v2/silences"

        # Calculate time range
        starts_at = datetime.now()
        ends_at = starts_at + duration

        silence = {
            "matchers": [
                {"name": k, "value": v, "isRegex": False}
                for k, v in matchers.items()
            ],
            "startsAt": starts_at.isoformat(),
            "endsAt": ends_at.isoformat(),
            "createdBy": created_by,
            "comment": comment
        }

        response = requests.post(
            endpoint,
            json=silence,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout
        )
        response.raise_for_status()

        result = response.json()
        return result.get("silenceId", "")

    def delete_silence(self, silence_id: str) -> bool:
        """
        Delete a silence.

        Args:
            silence_id (str): ID of the silence to delete

        Returns:
            bool: True if deleted successfully

        Example:
            ```python
            am.delete_silence(silence_id)
            ```
        """
        endpoint = f"{self.url}/api/v2/silence/{silence_id}"
        response = requests.delete(endpoint, timeout=self.timeout)
        return response.status_code == 200

    def get_receivers(self) -> List[str]:
        """
        Get list of configured receivers.

        Returns:
            List[str]: List of receiver names

        Example:
            ```python
            receivers = am.get_receivers()
            print(f"Receivers: {receivers}")
            ```
        """
        endpoint = f"{self.url}/api/v2/receivers"
        response = requests.get(endpoint, timeout=self.timeout)
        response.raise_for_status()

        receivers = response.json()
        return [r.get("name", "") for r in receivers]

    def health_check(self) -> bool:
        """
        Check if Alertmanager is healthy.

        Returns:
            bool: True if healthy, False otherwise

        Example:
            ```python
            if am.health_check():
                print("Alertmanager is available")
            ```
        """
        try:
            endpoint = f"{self.url}/-/healthy"
            response = requests.get(endpoint, timeout=self.timeout)
            return response.status_code == 200
        except Exception:
            return False
