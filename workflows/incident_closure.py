"""
Incident Closure Workflow

Standard workflow for processing incidents from detection to resolution.
This is the primary workflow used by the Supervisor agent.

Workflow steps:
1. Observability data collection
2. Anomaly detection
3. Root cause diagnosis
4. Remediation decision
5. Repair execution (auto or manual)
6. Report generation
"""

from typing import Any, Dict
from langgraph.graph import StateGraph, END


def create_incident_closure_workflow(
    observability_node: callable,
    detection_node: callable,
    diagnosis_node: callable,
    decision_node: callable,
    repair_node: callable,
    report_node: callable,
    escalation_node: callable
) -> StateGraph:
    """
    Create the incident closure workflow graph.

    Args:
        observability_node: Observability agent execution function
        detection_node: Detection agent execution function
        diagnosis_node: Diagnosis agent execution function
        decision_node: Decision agent execution function
        repair_node: Repair agent execution function
        report_node: Report agent execution function
        escalation_node: Escalation handler function

    Returns:
        StateGraph: Compiled workflow graph

    Example:
        ```python
        workflow = create_incident_closure_workflow(
            observability_node=run_observability,
            detection_node=run_detection,
            # ... other nodes
        )
        result = workflow.invoke(initial_state)
        ```
    """
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("observability", observability_node)
    graph.add_node("detection", detection_node)
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("decision", decision_node)
    graph.add_node("repair", repair_node)
    graph.add_node("report", report_node)
    graph.add_node("escalate", escalation_node)

    # Set entry point
    graph.set_entry_point("observability")

    # Define edges
    graph.add_edge("observability", "detection")

    # Conditional edge after detection
    graph.add_conditional_edges(
        "detection",
        route_after_detection,
        {
            "diagnosis": "diagnosis",
            "escalate": "escalate",
        }
    )

    # Conditional edge after diagnosis
    graph.add_conditional_edges(
        "diagnosis",
        route_after_diagnosis,
        {
            "decision": "decision",
            "escalate": "escalate",
        }
    )

    # Conditional edge after decision
    graph.add_conditional_edges(
        "decision",
        route_after_decision,
        {
            "auto_repair": "repair",
            "manual_approval": "repair",
            "escalate": "escalate",
        }
    )

    # Repair to report
    graph.add_edge("repair", "report")

    # Escalation to report
    graph.add_edge("escalate", "report")

    # Report to end
    graph.add_edge("report", END)

    return graph


def route_after_detection(state: Dict[str, Any]) -> str:
    """
    Determine routing after detection phase.

    Args:
        state (Dict[str, Any]): Current workflow state

    Returns:
        str: Next node name - "diagnosis" or "escalate"
    """
    anomaly = state.get("anomaly")

    if anomaly is None:
        return "escalate"

    # Check confidence
    if anomaly.confidence < 0.5:
        return "escalate"

    # Check human confirmation
    if anomaly.is_confirmed:
        return "diagnosis"

    # Route based on confidence and severity
    if anomaly.confidence >= 0.7:
        return "diagnosis"

    if anomaly.severity in ["HIGH", "CRITICAL"]:
        return "diagnosis"

    return "escalate"


def route_after_diagnosis(state: Dict[str, Any]) -> str:
    """
    Determine routing after diagnosis phase.

    Args:
        state (Dict[str, Any]): Current workflow state

    Returns:
        str: Next node name - "decision" or "escalate"
    """
    diagnosis = state.get("diagnosis")

    if diagnosis is None or not diagnosis.root_causes:
        return "escalate"

    if diagnosis.confidence < 0.5:
        return "escalate"

    return "decision"


def route_after_decision(state: Dict[str, Any]) -> str:
    """
    Determine routing after decision phase.

    Args:
        state (Dict[str, Any]): Current workflow state

    Returns:
        str: Next node name - "auto_repair", "manual_approval", or "escalate"
    """
    decision = state.get("decision")

    if decision is None or not decision.actions:
        return "escalate"

    if decision.requires_approval:
        return "manual_approval"

    if decision.risk_level == "LOW":
        return "auto_repair"

    if decision.risk_level in ["MEDIUM", "HIGH"]:
        return "manual_approval"

    return "escalate"
