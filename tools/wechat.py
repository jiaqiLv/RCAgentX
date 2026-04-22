"""
WeChat Tool - Enterprise WeChat Integration

LangChain tool for sending messages via Enterprise WeChat (WeCom).
Provides methods for sending text, markdown, and card messages.
"""

import requests
import hashlib
import hmac
import time
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class WeChatInput(BaseModel):
    """Input schema for WeChat tool operations."""
    content: str = Field(..., description="Message content")
    mentioned_list: Optional[List[str]] = Field(None, description="List of user IDs to mention")


class WeChatTool(BaseTool):
    """
    Tool for sending messages via Enterprise WeChat (WeCom).

    Supports sending text, markdown, and card messages to users
    or groups via webhooks.

    Attributes:
        name (str): Tool name for LangChain
        description (str): Tool description
        webhook_url (str): WeChat webhook URL for the bot
        secret (Optional[str]): API secret for signature verification

    Example:
        ```python
        wechat = WeChatTool(webhook_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/...")

        # Send text message
        wechat.send_text("Alert: High CPU usage detected")

        # Send markdown message with mentions
        wechat.send_markdown(
            content="## CPU Alert\\n- Service: api-service\\n- Usage: 95%",
            mentioned_list=["user1", "user2"]
        )
        ```
    """

    name: str = "wechat_message"
    description: str = "Send messages via Enterprise WeChat. Use for alerts, notifications, and incident updates."
    webhook_url: str
    secret: Optional[str] = None

    args_schema: type[BaseModel] = WeChatInput

    def _run(
        self,
        content: str,
        mentioned_list: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Send a text message via WeChat.

        Args:
            content (str): Message content
            mentioned_list (Optional[List[str]]): List of user IDs to mention

        Returns:
            Dict[str, Any]: Response from WeChat API

        Example:
            ```python
            result = wechat.send_text(
                "Server alert: High memory usage",
                mentioned_list=["@all"]
            )
            ```
        """
        payload = {
            "msgtype": "text",
            "text": {
                "content": content,
                "mentioned_list": mentioned_list or []
            }
        }

        return self._send_message(payload)

    def send_markdown(
        self,
        content: str,
        mentioned_list: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Send a markdown-formatted message via WeChat.

        Args:
            content (str): Markdown content (escape newlines as \\n)
            mentioned_list (Optional[List[str]]): List of user IDs to mention

        Returns:
            Dict[str, Any]: Response from WeChat API

        Example:
            ```python
            wechat.send_markdown(
                "## Incident Report\\n"
                "- **Service**: api-service\\n"
                "- **Severity**: HIGH\\n"
                "- **Status**: Investigating"
            )
            ```
        """
        payload = {
            "msgtype": "markdown",
            "markdown": {
                "content": content,
                "mentioned_list": mentioned_list or []
            }
        }

        return self._send_message(payload)

    def send_card(
        self,
        title: str,
        description: str,
        url: Optional[str] = None,
        btns: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Send an interactive card message via WeChat.

        Args:
            title (str): Card title
            description (str): Card description
            url (Optional[str]): URL to open on card click
            btns (Optional[List[Dict[str, str]]]): List of buttons with
                'text' and 'key' keys

        Returns:
            Dict[str, Any]: Response from WeChat API

        Example:
            ```python
            wechat.send_card(
                title="Incident Detected",
                description="High CPU usage on api-service",
                url="https://grafana.example.com/dashboard",
                btns=[
                    {"text": "View Dashboard", "key": "view"},
                    {"text": "Acknowledge", "key": "ack"}
                ]
            )
            ```
        """
        card = {
            "msgtype": "template_card",
            "template_card": {
                "source": {
                    "desc": "AIOps Alert"
                },
                "main_title": {
                    "title": title,
                    "desc": description
                },
                "card_action": {
                    "type": 2,
                    "url": url or ""
                }
            }
        }

        if btns:
            card["template_card"]["horizontal_content_list"] = [
                {"item_type": "button", "button": btn}
                for btn in btns
            ]

        return self._send_message(card)

    def send_incident_alert(
        self,
        incident_id: str,
        severity: str,
        service: str,
        description: str,
        status: str = "Investigating"
    ) -> Dict[str, Any]:
        """
        Send a formatted incident alert message.

        Args:
            incident_id (str): Incident identifier
            severity (str): Severity level (CRITICAL, HIGH, MEDIUM, LOW)
            service (str): Affected service name
            description (str): Incident description
            status (str): Current status

        Returns:
            Dict[str, Any]: Response from WeChat API

        Example:
            ```python
            wechat.send_incident_alert(
                incident_id="INC-001",
                severity="HIGH",
                service="api-service",
                description="CPU usage at 95% for 5 minutes"
            )
            ```
        """
        # Color based on severity
        colors = {
            "CRITICAL": "red",
            "HIGH": "orange",
            "MEDIUM": "yellow",
            "LOW": "green"
        }
        color = colors.get(severity.upper(), "gray")

        # Map severity to Chinese for WeChat
        severity_cn = {
            "CRITICAL": "紧急",
            "HIGH": "高",
            "MEDIUM": "中",
            "LOW": "低"
        }
        severity_display = severity_cn.get(severity.upper(), severity)

        content = (
            f"## Incident Alert\\n\\n"
            f"**ID**: {incident_id}\\n"
            f"**Severity**: {severity_display} ({color})\\n"
            f"**Service**: {service}\\n"
            f"**Status**: {status}\\n\\n"
            f"**Description**:\\n{description}"
        )

        return self.send_markdown(content)

    def _send_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message payload to WeChat.

        Args:
            payload (Dict[str, Any]): Message payload

        Returns:
            Dict[str, Any]: Response from WeChat API

        Raises:
            requests.RequestException: If HTTP request fails
        """
        url = self.webhook_url

        # Add signature if secret is configured
        if self.secret:
            timestamp = str(int(time.time()))
            signature = self._generate_signature(timestamp)
            url = f"{url}&timestamp={timestamp}&msg_signature={signature}"

        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()

        return response.json()

    def _generate_signature(self, timestamp: str) -> str:
        """
        Generate HMAC-SHA256 signature for WeChat webhook.

        Args:
            timestamp (str): Current timestamp

        Returns:
            str: Base64-encoded signature
        """
        message = f"{timestamp}\n{self.secret}"
        signature = hmac.new(
            self.secret.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()

        import base64
        return base64.b64encode(signature).decode()

    def health_check(self) -> bool:
        """
        Check if WeChat webhook is accessible.

        Returns:
            bool: True if webhook is reachable

        Note:
            This only checks network connectivity, not webhook validity.
        """
        try:
            # Send a test message
            response = requests.post(
                self.webhook_url,
                json={"msgtype": "text", "text": {"content": "Health check"}},
                timeout=10
            )
            # WeChat returns errcode 0 for success
            data = response.json()
            return data.get("errcode", -1) == 0
        except Exception:
            return False
