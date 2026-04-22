"""
Manual Approval Workflow

Workflow for repairs requiring human approval.
Used when risk level is MEDIUM/HIGH or confidence is below auto-repair threshold.

This workflow:
1. Pauses execution and waits for human review
2. Sends notification with repair plan details
3. Waits for approval/rejection
4. Continues or escalates based on response
"""

from typing import Any, Dict, Optional
from memory.shared_state import RepairState, RepairMode


def create_manual_approval_workflow() -> Dict[str, Any]:
    """
    Create configuration for manual approval workflow.

    Returns:
        Dict[str, Any]: Workflow configuration

    Example:
        ```python
        config = create_manual_approval_workflow()
        print(f"Timeout: {config['timeout_minutes']} minutes")
        ```
    """
    return {
        "name": "manual_approval",
        "description": "Repair execution requiring human approval",
        "trigger_conditions": {
            "risk_level": ["MEDIUM", "HIGH"],
            "requires_approval": True,
        },
        "notification_channels": ["wechat", "slack", "email"],
        "timeout_minutes": 30,
        "escalation_on_timeout": True,
        "approval_template": """
## Repair Approval Required

**Incident ID**: {incident_id}
**Risk Level**: {risk_level}
**Confidence**: {confidence:.1%}

**Proposed Actions**:
{actions}

**Rollback Plan**:
{rollback_plan}

Reply with APPROVE or REJECT.
        """,
    }


def request_approval(
    state: Dict[str, Any],
    notification_service: Any
) -> Dict[str, Any]:
    """
    Request human approval for repair.

    Args:
        state (Dict[str, Any]): Current workflow state
        notification_service (Any): Notification service instance

    Returns:
        Dict[str, Any]: Updated state with pending approval

    Example:
        ```python
        state = request_approval(state, wechat_service)
        # State now has repair.pending_approval set
        ```
    """
    decision = state.get("decision")
    repair = state.get("repair", RepairState())

    # Set pending approval state
    repair.mode = RepairMode.MANUAL
    repair.status = "waiting_approval"
    repair.pending_approval = {
        "actions": decision.actions if decision else [],
        "risk_level": decision.risk_level if decision else "UNKNOWN",
        "rollback_plan": decision.rollback_plan if decision else "",
    }

    state["repair"] = repair

    # Send notification
    if notification_service and decision:
        approval_message = format_approval_message(state)
        notification_service.send(approval_message)

    return state


def format_approval_message(state: Dict[str, Any]) -> str:
    """
    Format approval request message.

    Args:
        state (Dict[str, Any]): Current workflow state

    Returns:
        str: Formatted approval message
    """
    decision = state.get("decision")
    incident_id = state.get("incident_id", "unknown")

    if not decision:
        return f"Incident {incident_id}: No decision available"

    actions_str = "\n".join(
        f"- {a.get('description', a.get('type', 'unknown'))}"
        for a in decision.actions
    )

    return f"""
## Repair Approval Required

**Incident ID**: {incident_id}
**Risk Level**: {decision.risk_level}
**Confidence**: {decision.confidence:.1%}

**Proposed Actions**:
{actions_str}

**Rollback Plan**:
{decision.rollback_plan}

Reply with APPROVE or REJECT.
    """.strip()


def handle_approval_response(
    state: Dict[str, Any],
    approved: bool,
    repair_agent: Any
) -> Dict[str, Any]:
    """
    Handle human approval response.

    Args:
        state (Dict[str, Any]): Current workflow state
        approved (bool): Whether repair was approved
        repair_agent (Any): Repair agent instance

    Returns:
        Dict[str, Any]: Updated state after approval handling

    Example:
        ```python
        # Called when human responds to approval request
        state = handle_approval_response(state, approved=True, repair_agent=repair)
        ```
    """
    return repair_agent.approve_and_execute(state, approved)


def check_approval_timeout(
    state: Dict[str, Any],
    created_at: float,
    timeout_minutes: int = 30
) -> bool:
    """
    Check if approval has timed out.

    Args:
        state (Dict[str, Any]): Current workflow state
        created_at (float): When approval was requested (timestamp)
        timeout_minutes (int): Timeout in minutes

    Returns:
        bool: True if timed out
    """
    import time
    current_time = time.time()
    elapsed_minutes = (current_time - created_at) / 60

    return elapsed_minutes > timeout_minutes
