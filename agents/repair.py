"""
Repair Agent - Automated Remediation Execution

The Repair Agent executes remediation actions determined by the Decision
Agent. It supports dual-mode operation: automatic execution for low-risk
actions and manual approval mode for high-risk operations.

Key responsibilities:
1. Dual-mode repair execution (auto/manual)
2. Action orchestration and execution
3. Real-time status tracking
4. Effect verification and rollback on failure
"""

from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from pydantic import ConfigDict

from agents.base import BaseAgent
from memory.shared_state import RepairState, RepairMode, IncidentStatus


class RepairAgent(BaseAgent):
    """
    Remediation execution agent with dual-mode support.

    This agent executes remediation actions with support for both
    automatic execution and manual approval workflows.

    Attributes:
        name (str): Always "repair"
        description (str): Description of repair capabilities
        llm (ChatOpenAI): Language model for dynamic execution
        mode (RepairMode): Default execution mode
        dry_run (bool): Enable dry-run mode for testing
        verbose (bool): Enable verbose logging

    Example:
        ```python
        repair = RepairAgent(
            mode=RepairMode.HYBRID,
            dry_run=False,
            verbose=True
        )

        result = repair.execute(state)
        print(f"Repair status: {result['repair'].status}")
        ```
    """

    name: str = "repair"
    description: str = "Executes remediation actions with auto/manual mode support"
    llm: Optional[ChatOpenAI] = None
    mode: RepairMode = RepairMode.HYBRID
    dry_run: bool = False
    max_iterations: int = 3

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute remediation actions.

        Executes the actions defined in the decision package, supporting
        both automatic and manual approval modes.

        Args:
            state (Dict[str, Any]): Current workflow state containing
                decision package with actions to execute

        Returns:
            Dict[str, Any]: Updated state with repair execution results

        Example:
            ```python
            state = {
                "decision": decision_package,
                "repair": RepairState(mode=RepairMode.AUTO)
            }
            result = repair.execute(state)
            ```
        """
        self.log("Starting repair execution")

        decision = state.get("decision")
        repair_state = state.get("repair", RepairState())

        if not decision or not decision.actions:
            self.log("No actions to execute")
            state["errors"] = state.get("errors", [])
            state["errors"].append("No remediation actions defined")
            repair_state.status = "failed"
            state["repair"] = repair_state
            return state

        # Determine execution mode
        if decision.requires_approval:
            repair_state.mode = RepairMode.MANUAL
        else:
            repair_state.mode = self.mode

        self.log(f"Execution mode: {repair_state.mode}")

        # Handle based on mode
        if repair_state.mode == RepairMode.MANUAL:
            # Set pending approval state
            repair_state.status = "waiting_approval"
            repair_state.pending_approval = {
                "actions": decision.actions,
                "risk_level": decision.risk_level,
                "rollback_plan": decision.rollback_plan
            }
            state["repair"] = repair_state
            self.log("Waiting for human approval")
            return state

        # Automatic execution
        repair_state.status = "running"
        state["repair"] = repair_state

        # Execute actions
        success = self._execute_actions(decision.actions, repair_state)

        if success:
            repair_state.status = "success"
            repair_state.result = "All actions executed successfully"
            self.log("Repair completed successfully")
        else:
            repair_state.status = "failed"
            repair_state.result = "Action execution failed"
            self.log("Repair execution failed")

        state["repair"] = repair_state
        state["status"] = IncidentStatus.VERIFYING.value

        return state

    def _execute_actions(
        self,
        actions: List[Dict[str, Any]],
        repair_state: RepairState
    ) -> bool:
        """
        Execute a list of remediation actions.

        Args:
            actions: List of actions to execute
            repair_state: Current repair state for tracking

        Returns:
            bool: True if all actions succeeded
        """
        all_success = True

        for i, action in enumerate(actions):
            self.log(f"Executing action {i + 1}/{len(actions)}: {action.get('description', 'unknown')}")

            success = self._execute_single_action(action, repair_state)

            if not success:
                self.log(f"Action {i + 1} failed")
                all_success = False

                # Attempt rollback if available
                if repair_state.executed_actions:
                    self.log("Attempting rollback")
                    self._execute_rollback(repair_state)
                break

            repair_state.executed_actions.append({
                "action": action,
                "status": "success",
                "timestamp": str(__import__("datetime").datetime.now())
            })

        return all_success

    def _execute_single_action(
        self,
        action: Dict[str, Any],
        repair_state: RepairState
    ) -> bool:
        """
        Execute a single remediation action.

        Args:
            action: Action to execute
            repair_state: Current repair state

        Returns:
            bool: True if action succeeded
        """
        action_type = action.get("type", "")
        target = action.get("target", "")
        parameters = action.get("parameters", {})

        self.log(f"Executing {action_type} on {target}")

        if self.dry_run:
            self.log(f"[DRY RUN] Would execute: {action_type} on {target}")
            return True

        try:
            if action_type == "restart":
                return self._execute_restart(target, parameters)
            elif action_type == "scale":
                return self._execute_scale(target, parameters)
            elif action_type == "rollback":
                return self._execute_rollback_version(target, parameters)
            elif action_type == "check":
                return self._execute_check(target, parameters)
            elif action_type == "custom":
                return self._execute_custom(target, parameters)
            else:
                self.log(f"Unknown action type: {action_type}")
                return False

        except Exception as e:
            self.log(f"Action execution failed: {str(e)}")
            return False

    def _execute_restart(
        self,
        target: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """
        Execute restart action.

        In a real implementation, this would call Kubernetes API
        or other orchestration systems.

        Args:
            target: Resource to restart
            parameters: Restart parameters (graceful, timeout, etc.)

        Returns:
            bool: True if restart succeeded
        """
        self.log(f"Restarting {target} (graceful={parameters.get('graceful', False)})")

        # Simulated success for now
        # In production, integrate with:
        # - Kubernetes: kubectl rollout restart deployment/<target>
        # - Docker: docker restart <container>
        # - Cloud: AWS ECS UpdateService, etc.

        return True

    def _execute_scale(
        self,
        target: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """
        Execute scale action.

        Args:
            target: Resource to scale
            parameters: Scale parameters (replicas, etc.)

        Returns:
            bool: True if scale succeeded
        """
        replicas = parameters.get("replicas", 2)
        self.log(f"Scaling {target} to {replicas} replicas")

        # Simulated success
        # In production, integrate with:
        # - Kubernetes: kubectl scale deployment/<target> --replicas=<n>
        # - Cloud: AWS AutoScaling, etc.

        return True

    def _execute_rollback_version(
        self,
        target: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """
        Execute rollback to previous version.

        Args:
            target: Resource to rollback
            parameters: Rollback parameters (version, etc.)

        Returns:
            bool: True if rollback succeeded
        """
        version = parameters.get("version", "previous")
        self.log(f"Rolling back {target} to version: {version}")

        # Simulated success
        # In production, integrate with:
        # - Kubernetes: kubectl rollout undo deployment/<target>
        # - Cloud: AWS ECS UpdateService with previous task definition

        return True

    def _execute_check(
        self,
        target: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """
        Execute check/verification action.

        Args:
            target: Resource to check
            parameters: Check parameters

        Returns:
            bool: True if check passed
        """
        self.log(f"Checking {target}")

        # Simulated success
        return True

    def _execute_custom(
        self,
        target: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """
        Execute custom action.

        Args:
            target: Target resource
            parameters: Custom action parameters

        Returns:
            bool: True if custom action succeeded
        """
        action = parameters.get("action", "")
        self.log(f"Executing custom action on {target}: {action}")

        # Could use LLM to generate and execute scripts
        if self.llm and action:
            self.log(f"LLM-assisted custom execution")

        return True

    def _execute_rollback(
        self,
        repair_state: RepairState
    ) -> bool:
        """
        Execute rollback of previously executed actions.

        Args:
            repair_state: Current repair state

        Returns:
            bool: True if rollback succeeded
        """
        executed = repair_state.executed_actions

        # Rollback in reverse order
        for executed_action in reversed(executed):
            action = executed_action.get("action", {})
            action_type = action.get("type", "")
            target = action.get("target", "")

            self.log(f"Rolling back {action_type} on {target}")

            try:
                if action_type == "restart":
                    # Rollback: restore from backup
                    self.log(f"Restoring {target} from backup")
                elif action_type == "scale":
                    # Rollback: scale back
                    self.log(f"Scaling {target} back to original")
            except Exception as e:
                self.log(f"Rollback failed: {str(e)}")
                return False

        return True

    def approve_and_execute(
        self,
        state: Dict[str, Any],
        approved: bool
    ) -> Dict[str, Any]:
        """
        Handle human approval and continue execution.

        Called when a human operator approves pending actions.

        Args:
            state: Current workflow state
            approved: Whether actions were approved

        Returns:
            Dict[str, Any]: Updated state after approval handling
        """
        repair_state = state.get("repair", RepairState())

        if not approved:
            self.log("Repair actions rejected by human operator")
            repair_state.status = "rejected"
            repair_state.result = "Actions rejected by operator"
            state["repair"] = repair_state
            return state

        self.log("Repair actions approved by human operator")

        # Clear pending approval
        repair_state.pending_approval = None
        repair_state.status = "running"
        state["repair"] = repair_state

        # Execute approved actions
        decision = state.get("decision")
        if decision and decision.actions:
            success = self._execute_actions(decision.actions, repair_state)

            if success:
                repair_state.status = "success"
                repair_state.result = "Actions executed successfully after approval"
            else:
                repair_state.status = "failed"
                repair_state.result = "Actions failed after approval"

        state["repair"] = repair_state
        return state
