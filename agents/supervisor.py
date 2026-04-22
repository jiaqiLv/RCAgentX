"""
Supervisor Agent - Global Orchestration and Coordination

The Supervisor Agent is the central control component of the AIOps system.
It orchestrates all other agents (Observability, Detection, Diagnosis,
Decision, Repair, Report) and manages the complete incident lifecycle.

Key responsibilities:
1. Dynamic workflow orchestration based on incident severity
2. Multi-agent routing and scheduling
3. Human-in-the-loop escalation handling
4. Global monitoring and fault tolerance
5. Feedback collection for GRPO optimization
"""

from typing import Any, Dict, List, Optional, Callable
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import ConfigDict

from agents.base import BaseAgent
from memory.shared_state import SharedState, IncidentStatus, RepairMode
from memory.knowledge_base import GRPOKnowledgeBase


class SupervisorAgent(BaseAgent):
    """
    Central orchestration agent for the AIOps multi-agent system.

    The Supervisor manages the complete incident resolution workflow by:
    - Coordinating execution of all sub-agents
    - Making routing decisions based on confidence and risk levels
    - Handling human escalation when needed
    - Collecting experience records for GRPO learning

    The workflow follows this general pattern:
    Observability -> Detection -> Diagnosis -> Decision -> Repair -> Report

    However, the Supervisor can dynamically reroute based on:
    - Anomaly confidence scores
    - Risk assessment
    - Human feedback
    - Agent health status

    Attributes:
        name (str): Always "supervisor"
        description (str): Description of supervisor capabilities
        sub_agents (Dict[str, BaseAgent]): Map of agent name to instance
        llm (ChatOpenAI): Language model for decision making
        knowledge_base (GRPOKnowledgeBase): Experience storage
        verbose (bool): Enable verbose logging
        workflow_graph (StateGraph): Compiled LangGraph workflow

    Example:
        ```python
        # Create sub-agents
        obs_agent = ObservabilityAgent(...)
        detection_agent = DetectionAgent(...)
        diagnosis_agent = DiagnosisAgent(...)

        # Create supervisor
        supervisor = SupervisorAgent(
            sub_agents={
                "observability": obs_agent,
                "detection": detection_agent,
                "diagnosis": diagnosis_agent,
            },
            llm=ChatOpenAI(model="gpt-4")
        )

        # Execute workflow
        result = supervisor.execute(initial_state)
        ```
    """

    name: str = "supervisor"
    description: str = "Orchestrates all AIOps agents and manages incident lifecycle"
    sub_agents: Dict[str, BaseAgent] = {}
    knowledge_base: Optional[GRPOKnowledgeBase] = None
    workflow_graph: Optional[Any] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """
        Initialize the Supervisor Agent.

        Sets up the LangGraph workflow with all sub-agents and configures
        routing logic based on confidence thresholds and risk levels.
        """
        super().__init__(**data)

        # Build the workflow graph
        self.workflow_graph = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph state machine for incident workflow.

        Creates a directed graph with nodes for each agent and edges
        defining the possible transitions. Supports conditional routing
        and human-in-the-loop interrupts.

        Returns:
            StateGraph: Compiled LangGraph workflow ready for execution

        Workflow structure:
            START -> observability -> detection -> [conditional] -> diagnosis
                    -> decision -> [conditional: auto/manual] -> repair
                    -> report -> END
        """
        # Create state graph
        graph = StateGraph(dict)

        # Add nodes for each agent
        graph.add_node("observability", self._run_observability)
        graph.add_node("detection", self._run_detection)
        graph.add_node("diagnosis", self._run_diagnosis)
        graph.add_node("decision", self._run_decision)
        graph.add_node("repair", self._run_repair)
        graph.add_node("report", self._run_report)
        graph.add_node("escalate", self._run_escalation)

        # Define edges
        graph.set_entry_point("observability")

        # Observability -> Detection (always)
        graph.add_edge("observability", "detection")

        # Detection -> [Diagnosis or Escalate] based on confidence
        graph.add_conditional_edges(
            "detection",
            self._route_after_detection,
            {
                "diagnosis": "diagnosis",
                "escalate": "escalate",
            }
        )

        # Diagnosis -> [Decision or Escalate] based on confidence
        graph.add_conditional_edges(
            "diagnosis",
            self._route_after_diagnosis,
            {
                "decision": "decision",
                "escalate": "escalate",
            }
        )

        # Decision -> [Auto Repair or Manual Approval or Escalate]
        graph.add_conditional_edges(
            "decision",
            self._route_after_decision,
            {
                "auto_repair": "repair",
                "manual_approval": "repair",
                "escalate": "escalate",
            }
        )

        # Repair -> Report
        graph.add_edge("repair", "report")

        # Escalate -> Report (after human handles)
        graph.add_edge("escalate", "report")

        # Report -> END
        graph.add_edge("report", END)

        # Compile with interrupt before repair for manual approval
        return graph.compile(
            interrupt_before=["repair"],
            debug=self.verbose
        )

    def _route_after_detection(self, state: Dict[str, Any]) -> str:
        """
        Determine next step after detection phase.

        Routes to diagnosis if anomaly confidence is high enough,
        otherwise escalates for human review.

        Args:
            state (Dict[str, Any]): Current workflow state

        Returns:
            str: Next node name - "diagnosis" or "escalate"
        """
        anomaly = state.get("anomaly")

        if anomaly is None:
            self.log("No anomaly detected, skipping diagnosis")
            return "escalate"

        # Check confidence threshold
        if anomaly.confidence < 0.5:
            self.log(f"Low confidence ({anomaly.confidence}), escalating")
            return "escalate"

        # Check if human confirmed
        if anomaly.is_confirmed:
            self.log("Human confirmed anomaly, proceeding to diagnosis")
            return "diagnosis"

        # Default: proceed if confidence is adequate
        if anomaly.confidence >= 0.7:
            return "diagnosis"

        # Medium confidence - check severity
        if anomaly.severity in ["HIGH", "CRITICAL"]:
            return "diagnosis"

        self.log(f"Uncertain detection (conf={anomaly.confidence}, sev={anomaly.severity}), escalating")
        return "escalate"

    def _route_after_diagnosis(self, state: Dict[str, Any]) -> str:
        """
        Determine next step after diagnosis phase.

        Routes to decision if root cause confidence is adequate,
        otherwise escalates for human analysis.

        Args:
            state (Dict[str, Any]): Current workflow state

        Returns:
            str: Next node name - "decision" or "escalate"
        """
        diagnosis = state.get("diagnosis")

        if diagnosis is None or not diagnosis.root_causes:
            self.log("No root cause identified, escalating")
            return "escalate"

        # Check confidence threshold
        if diagnosis.confidence < 0.5:
            self.log(f"Low diagnosis confidence ({diagnosis.confidence}), escalating")
            return "escalate"

        # Proceed to decision
        return "decision"

    def _route_after_decision(self, state: Dict[str, Any]) -> str:
        """
        Determine repair execution mode after decision phase.

        Routes to auto repair for low-risk actions, manual approval
        for medium/high risk, or escalation for complex scenarios.

        Args:
            state (Dict[str, Any]): Current workflow state

        Returns:
            str: Next node name - "auto_repair", "manual_approval", or "escalate"
        """
        decision = state.get("decision")

        if decision is None or not decision.actions:
            self.log("No actionable decision, escalating")
            return "escalate"

        # Check if decision explicitly requires approval
        if decision.requires_approval:
            self.log(f"Decision requires approval (risk={decision.risk_level})")
            return "manual_approval"

        # Route based on risk level
        if decision.risk_level == "LOW":
            self.log("Low risk, proceeding with auto repair")
            return "auto_repair"

        # Medium/High risk requires approval
        if decision.risk_level in ["MEDIUM", "HIGH"]:
            self.log(f"{decision.risk_level} risk, requiring manual approval")
            return "manual_approval"

        # Unknown risk level - escalate
        self.log(f"Unknown risk level '{decision.risk_level}', escalating")
        return "escalate"

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete supervised workflow.

        Runs the incident through all phases with dynamic routing
        based on confidence scores and risk assessments.

        Args:
            state (Dict[str, Any]): Initial state containing incident data.
                Must include at minimum:
                - observability: ObservabilityData with metrics/logs/traces

        Returns:
            Dict[str, Any]: Final state after workflow completion

        Raises:
            Exception: If workflow execution fails

        Example:
            ```python
            initial_state = {
                "incident_id": "inc-001",
                "observability": obs_data,
                "status": "pending"
            }

            result = supervisor.execute(initial_state)
            print(f"Final status: {result['status']}")
            ```
        """
        self.log(f"Starting supervised workflow for incident: {state.get('incident_id', 'unknown')}")

        try:
            # Execute the compiled workflow
            final_state = self.workflow_graph.invoke(state)

            self.log("Workflow completed successfully")

            # Record experience for GRPO learning
            self._record_experience(final_state)

            return final_state

        except Exception as e:
            self.log(f"Workflow execution failed: {str(e)}")

            # Add error to state
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Supervisor error: {str(e)}")
            state["status"] = IncidentStatus.ESCALATED.value

            return state

    def _run_observability(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Observability Agent"""
        self.log("Running Observability Agent")

        if "observability" not in self.sub_agents:
            state["errors"] = state.get("errors", [])
            state["errors"].append("Observability agent not configured")
            return state

        agent = self.sub_agents["observability"]
        state["status"] = IncidentStatus.DETECTING.value
        return agent.execute(state)

    def _run_detection(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Detection Agent"""
        self.log("Running Detection Agent")

        if "detection" not in self.sub_agents:
            state["errors"] = state.get("errors", [])
            state["errors"].append("Detection agent not configured")
            return state

        agent = self.sub_agents["detection"]
        state["status"] = IncidentStatus.DIAGNOSING.value
        return agent.execute(state)

    def _run_diagnosis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Diagnosis Agent"""
        self.log("Running Diagnosis Agent")

        if "diagnosis" not in self.sub_agents:
            state["errors"] = state.get("errors", [])
            state["errors"].append("Diagnosis agent not configured")
            return state

        agent = self.sub_agents["diagnosis"]
        return agent.execute(state)

    def _run_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Decision Agent"""
        self.log("Running Decision Agent")

        if "decision" not in self.sub_agents:
            state["errors"] = state.get("errors", [])
            state["errors"].append("Decision agent not configured")
            return state

        agent = self.sub_agents["decision"]
        state["status"] = IncidentStatus.REPAIRING.value
        return agent.execute(state)

    def _run_repair(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Repair Agent"""
        self.log("Running Repair Agent")

        if "repair" not in self.sub_agents:
            state["errors"] = state.get("errors", [])
            state["errors"].append("Repair agent not configured")
            return state

        agent = self.sub_agents["repair"]
        state["status"] = IncidentStatus.VERIFYING.value
        return agent.execute(state)

    def _run_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Report Agent"""
        self.log("Running Report Agent")

        if "report" not in self.sub_agents:
            state["errors"] = state.get("errors", [])
            state["errors"].append("Report agent not configured")
            state["status"] = IncidentStatus.RESOLVED.value
            return state

        agent = self.sub_agents["report"]
        result = agent.execute(state)
        result["status"] = IncidentStatus.RESOLVED.value
        return result

    def _run_escalation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle escalation to human operators"""
        self.log("Escalating to human operators")

        state["status"] = IncidentStatus.ESCALATED.value
        state["intervention_requested"] = True

        # TODO: Send notification to human operators via WeChat/Slack
        # This would integrate with the integrations module

        return state

    def _record_experience(self, state: Dict[str, Any]):
        """
        Record the incident resolution as an experience for GRPO.

        Extracts key information from the final state and stores it
        in the knowledge base for future retrieval.

        Args:
            state (Dict[str, Any]): Final state after workflow completion
        """
        if self.knowledge_base is None:
            return

        try:
            from memory.shared_state import ExperienceRecord

            anomaly = state.get("anomaly")
            diagnosis = state.get("diagnosis")
            repair = state.get("repair")

            if not all([anomaly, diagnosis, repair]):
                self.log("Missing data for experience record, skipping")
                return

            # Calculate reward based on outcome
            reward = 1.0 if repair.status == "success" else 0.0

            # Extract root cause
            root_cause = diagnosis.root_causes[0].get("cause", "unknown") if diagnosis.root_causes else "unknown"

            # Create experience record
            record = ExperienceRecord(
                incident_id=state.get("incident_id", "unknown"),
                anomaly_type=anomaly.type,
                root_cause=root_cause,
                action_taken=str(repair.executed_actions),
                outcome=repair.status,
                reward=reward,
            )

            # Add to knowledge base
            self.knowledge_base.add_experience(record)
            self.log(f"Recorded experience: {record.incident_id}")

        except Exception as e:
            self.log(f"Failed to record experience: {str(e)}")

    def approve_repair(self, state: Dict[str, Any], approved: bool) -> Dict[str, Any]:
        """
        Handle human approval for repair actions.

        Called when a human operator reviews and approves/rejects
        the proposed repair plan.

        Args:
            state (Dict[str, Any]): Current workflow state
            approved (bool): Whether the repair was approved

        Returns:
            Dict[str, Any]: Updated state
        """
        if approved:
            self.log("Repair approved by human operator")
            state["repair"].pending_approval = None
            # Continue workflow
            return self._run_repair(state)
        else:
            self.log("Repair rejected by human operator")
            state["repair"].status = "rejected"
            state["intervention_requested"] = True
            return state
