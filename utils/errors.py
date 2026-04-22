"""
Custom Exceptions for AIOps System

Defines specific exception types for error handling
across the AIOps agent system.
"""

from typing import Any, Dict, Optional


class AgentError(Exception):
    """
    Base exception for agent-related errors.

    Raised when an agent encounters an error during execution.

    Attributes:
        message (str): Error message
        agent_name (str): Name of the agent that raised the error
        context (Dict[str, Any]): Additional context information

    Example:
        ```python
        raise AgentError(
            "Failed to analyze metrics",
            agent_name="observability",
            context={"metric": "cpu_usage"}
        )
        ```
    """

    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.agent_name = agent_name
        self.context = context or {}

        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with context."""
        msg = f"[{self.agent_name or 'Agent'}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            msg = f"{msg} ({context_str})"
        return msg


class ToolError(Exception):
    """
    Exception for tool execution errors.

    Raised when a tool fails to execute or returns an error response.

    Attributes:
        message (str): Error message
        tool_name (str): Name of the tool that failed
        original_error (Optional[Exception]): Original exception if any

    Example:
        ```python
        try:
            prometheus.query("invalid_query")
        except Exception as e:
            raise ToolError("Query failed", tool_name="prometheus", original_error=e)
        ```
    """

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.tool_name = tool_name
        self.original_error = original_error

        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with tool name."""
        msg = f"[{self.tool_name or 'Tool'}] {self.message}"
        if self.original_error:
            msg = f"{msg}: {str(self.original_error)}"
        return msg


class WorkflowError(Exception):
    """
    Exception for workflow execution errors.

    Raised when a LangGraph workflow encounters an error.

    Attributes:
        message (str): Error message
        workflow_name (str): Name of the workflow
        state (Optional[Dict[str, Any]]): State at time of error

    Example:
        ```python
        raise WorkflowError(
            "Failed to transition from detection to diagnosis",
            workflow_name="incident_closure",
            state=current_state
        )
        ```
    """

    def __init__(
        self,
        message: str,
        workflow_name: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.workflow_name = workflow_name
        self.state = state

        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with workflow name."""
        msg = f"[{self.workflow_name or 'Workflow'}] {self.message}"
        return msg


class ConfigurationError(Exception):
    """
    Exception for configuration errors.

    Raised when there's an issue with the system configuration.

    Attributes:
        message (str): Error message
        config_key (Optional[str]): Configuration key that caused the error

    Example:
        ```python
        raise ConfigurationError(
            "Missing required API key",
            config_key="OPENAI_API_KEY"
        )
        ```
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None
    ):
        self.message = message
        self.config_key = config_key

        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with config key."""
        if self.config_key:
            return f"[{self.config_key}] {self.message}"
        return self.message


class IntegrationError(Exception):
    """
    Exception for external integration errors.

    Raised when communication with external systems fails.

    Attributes:
        message (str): Error message
        integration_name (str): Name of the integration
        status_code (Optional[int]): HTTP status code if applicable

    Example:
        ```python
        raise IntegrationError(
            "Failed to send WeChat notification",
            integration_name="wechat",
            status_code=401
        )
        ```
    """

    def __init__(
        self,
        message: str,
        integration_name: Optional[str] = None,
        status_code: Optional[int] = None
    ):
        self.message = message
        self.integration_name = integration_name
        self.status_code = status_code

        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with integration name."""
        msg = f"[{self.integration_name or 'Integration'}] {self.message}"
        if self.status_code:
            msg = f"{msg} (status: {self.status_code})"
        return msg
