"""
Base Agent Abstract Base Class

All agents must inherit from this class and implement the execute method.
This provides a unified interface for agent execution and tool management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict


class BaseAgent(ABC, BaseModel):
    """
    Abstract base class for all AIOps agents.

    This class defines the standard interface that all agents must follow,
    including execution method, tool management, and logging capabilities.

    Attributes:
        name (str): Unique identifier for this agent
        description (str): Human-readable description of agent capabilities
        llm (Optional[Any]): Language model instance for reasoning tasks
        tools (List[BaseTool]): List of LangChain tools available to this agent
        verbose (bool): Enable verbose logging output

    Example:
        ```python
        class MyCustomAgent(BaseAgent):
            name: str = "my_agent"
            description: str = "Performs custom analysis"

            def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
                # Implementation here
                return state
        ```
    """

    name: str
    description: str
    llm: Optional[Any] = None
    tools: List[BaseTool] = []
    verbose: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's core logic.

        This method is called by the Supervisor to process the current
        shared state and produce updated state information. Each agent
        should focus on its specific responsibility and leave other
        state components unchanged.

        Args:
            state (Dict[str, Any]): Shared state dictionary containing
                all incident-related data. Common keys include:
                - incident_id: Unique identifier for the incident
                - status: Current processing status
                - observability: ObservabilityData object
                - anomaly: AnomalyEvent object
                - diagnosis: RootCauseAnalysis object
                - decision: DecisionPackage object
                - repair: RepairState object
                - human_feedback: Optional feedback from operators
                - errors: List of error messages
                - logs: List of log entries

        Returns:
            Dict[str, Any]: Updated state dictionary with new information
                added by this agent. The agent should only modify the
                state components relevant to its responsibility.

        Raises:
            NotImplementedError: If subclass doesn't implement this method
            AgentError: If agent execution fails

        Example:
            ```python
            def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    # Extract relevant data from state
                    anomaly = state.get("anomaly")

                    # Perform analysis
                    result = self._analyze(anomaly)

                    # Update state with results
                    state["diagnosis"] = result
                    state["status"] = "diagnosed"

                    self.log(f"Analysis completed: {result.root_cause}")
                    return state
                except Exception as e:
                    self.log(f"Error: {str(e)}")
                    raise
            ```
        """
        pass

    def get_tools(self) -> List[BaseTool]:
        """
        Get the list of tools available to this agent.

        Returns:
            List[BaseTool]: List of LangChain BaseTool instances that
                can be invoked by this agent during execution.
        """
        return self.tools

    def log(self, message: str):
        """
        Write a log message with agent name prefix.

        Messages are only output when verbose mode is enabled.
        Use this method for all agent logging to maintain consistency.

        Args:
            message (str): The message to log. Will be prefixed with
                the agent name in square brackets.

        Example:
            ```python
            self.log("Starting analysis phase")
            self.log(f"Found {len(results)} potential root causes")
            ```
        """
        if self.verbose:
            print(f"[{self.name}] {message}")
