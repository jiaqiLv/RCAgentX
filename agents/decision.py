"""
Decision Agent - Remediation Strategy Decision

The Decision Agent determines the optimal remediation strategy based on
the root cause analysis. It combines GRPO-retrieved historical strategies,
risk assessment, and LLM-powered planning to generate actionable
remediation plans.

Key responsibilities:
1. Multi-modal decision input and strategy evaluation
2. Advanced decision optimization algorithms
3. GRPO-based historical strategy retrieval
4. Risk assessment and approval routing
"""

from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from pydantic import ConfigDict

from agents.base import BaseAgent
from memory.shared_state import DecisionPackage, RepairMode
from memory.knowledge_base import GRPOKnowledgeBase


class DecisionAgent(BaseAgent):
    """
    Remediation strategy decision agent.

    This agent analyzes root cause diagnosis and retrieves historical
    strategies using GRPO to generate optimal remediation plans.

    Attributes:
        name (str): Always "decision"
        description (str): Description of decision capabilities
        llm (ChatOpenAI): Language model for strategy planning
        knowledge_base (GRPOKnowledgeBase): Experience retrieval
        auto_repair_threshold (float): Confidence threshold for auto-repair
        verbose (bool): Enable verbose logging

    Example:
        ```python
        decision = DecisionAgent(
            llm=ChatOpenAI(model="gpt-4"),
            knowledge_base=kb,
            auto_repair_threshold=0.8
        )

        result = decision.execute(state)
        plan = result["decision"]
        print(f"Strategy: {plan.strategy}")
        ```
    """

    name: str = "decision"
    description: str = "Determines optimal remediation strategy based on root cause analysis"
    llm: Optional[ChatOpenAI] = None
    knowledge_base: Optional[GRPOKnowledgeBase] = None
    auto_repair_threshold: float = 0.8

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute remediation strategy decision.

        Analyzes the diagnosis result and retrieves historical strategies
        to generate an optimal remediation plan with risk assessment.

        Args:
            state (Dict[str, Any]): Current workflow state containing
                diagnosis result and anomaly information

        Returns:
            Dict[str, Any]: Updated state with decision package

        Example:
            ```python
            state = {
                "diagnosis": diagnosis_result,
                "anomaly": anomaly_event
            }
            result = decision.execute(state)
            decision_plan = result["decision"]
            ```
        """
        self.log("Starting remediation strategy decision")

        diagnosis = state.get("diagnosis")
        anomaly = state.get("anomaly")

        if not diagnosis or not diagnosis.root_causes:
            self.log("No diagnosis available for decision")
            state["errors"] = state.get("errors", [])
            state["errors"].append("No diagnosis available")
            return state

        # Retrieve historical strategies using GRPO
        historical_strategies = self._retrieve_strategies(anomaly)

        # Generate remediation plan
        decision_package = self._generate_decision(
            diagnosis, anomaly, historical_strategies
        )

        state["decision"] = decision_package
        state["status"] = "deciding"

        self.log(f"Decision made: {decision_package.strategy}")
        self.log(f"Risk level: {decision_package.risk_level}")
        self.log(f"Requires approval: {decision_package.requires_approval}")

        return state

    def _retrieve_strategies(
        self,
        anomaly: Any
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical strategies using GRPO.

        Queries the knowledge base for similar historical incidents
        and their successful remediation strategies.

        Args:
            anomaly: Detected anomaly event

        Returns:
            List[Dict[str, Any]]: List of historical strategies
        """
        if not self.knowledge_base or not anomaly:
            return []

        try:
            strategies = self.knowledge_base.get_successful_strategies(
                anomaly_type=anomaly.type,
                k=3
            )
            self.log(f"Retrieved {len(strategies)} historical strategies")
            return strategies
        except Exception as e:
            self.log(f"Failed to retrieve strategies: {str(e)}")
            return []

    def _generate_decision(
        self,
        diagnosis: Any,
        anomaly: Any,
        historical_strategies: List[Dict[str, Any]]
    ) -> DecisionPackage:
        """
        Generate remediation decision package.

        Combines diagnosis information with historical strategies to
        create an actionable remediation plan.

        Args:
            diagnosis: Root cause analysis result
            anomaly: Detected anomaly event
            historical_strategies: Retrieved historical strategies

        Returns:
            DecisionPackage: Complete remediation plan
        """
        # Extract root cause
        root_cause = diagnosis.root_causes[0] if diagnosis.root_causes else {}
        cause_category = root_cause.get("category", "unknown")
        entity = root_cause.get("entity", "unknown")

        # Generate actions based on cause category
        actions = self._generate_actions(cause_category, entity, historical_strategies)

        # Assess risk level
        risk_level = self._assess_risk(actions, anomaly, diagnosis)

        # Determine if approval is required
        requires_approval = self._requires_approval(risk_level, diagnosis)

        # Generate strategy description
        strategy = self._generate_strategy_description(cause_category, actions)

        # Estimate recovery time
        estimated_recovery_time = self._estimate_recovery_time(actions)

        # Generate rollback plan
        rollback_plan = self._generate_rollback_plan(actions)

        # Calculate confidence
        confidence = self._calculate_decision_confidence(diagnosis, historical_strategies)

        return DecisionPackage(
            strategy=strategy,
            actions=actions,
            risk_level=risk_level,
            estimated_recovery_time=estimated_recovery_time,
            rollback_plan=rollback_plan,
            confidence=confidence,
            requires_approval=requires_approval
        )

    def _generate_actions(
        self,
        cause_category: str,
        entity: str,
        historical_strategies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate remediation actions based on cause category.

        Args:
            cause_category: Category of root cause
            entity: Affected entity
            historical_strategies: Historical strategies for reference

        Returns:
            List[Dict[str, Any]]: Ordered list of actions to execute
        """
        actions = []

        # Resource exhaustion (CPU/Memory)
        if cause_category == "resource_exhaustion":
            actions = [
                {
                    "type": "restart",
                    "target": entity,
                    "parameters": {"graceful": True},
                    "expected_outcome": "Release accumulated resources",
                    "description": f"Restart {entity} to release resources"
                },
                {
                    "type": "scale",
                    "target": entity,
                    "parameters": {"replicas": 2},
                    "expected_outcome": "Distribute load",
                    "description": f"Scale {entity} to handle increased load"
                }
            ]

        # Application errors
        elif cause_category == "application_error":
            actions = [
                {
                    "type": "rollback",
                    "target": entity,
                    "parameters": {"version": "previous"},
                    "expected_outcome": "Revert to stable version",
                    "description": f"Rollback {entity} to previous version"
                }
            ]

        # Network issues
        elif cause_category == "network_issue":
            actions = [
                {
                    "type": "restart",
                    "target": entity,
                    "parameters": {},
                    "expected_outcome": "Reset network connections",
                    "description": f"Restart {entity} to reset connections"
                },
                {
                    "type": "check",
                    "target": "network_policy",
                    "parameters": {},
                    "expected_outcome": "Verify network configuration",
                    "description": "Check network policies"
                }
            ]

        # Performance degradation
        elif cause_category == "performance_degradation":
            actions = [
                {
                    "type": "scale",
                    "target": entity,
                    "parameters": {"replicas": 3},
                    "expected_outcome": "Improve response time",
                    "description": f"Scale {entity} horizontally"
                }
            ]

        # Default: use historical strategies
        if not actions and historical_strategies:
            # Use the best historical action
            best_strategy = historical_strategies[0] if historical_strategies else {}
            action_str = best_strategy.get("action", "")

            if action_str:
                actions = [{
                    "type": "custom",
                    "target": entity,
                    "parameters": {"action": action_str},
                    "expected_outcome": "Based on historical success",
                    "description": f"Apply historical strategy: {action_str}"
                }]

        # Fallback action
        if not actions:
            actions = [{
                "type": "restart",
                "target": entity,
                "parameters": {},
                "expected_outcome": "Restore service",
                "description": f"Restart {entity}"
            }]

        return actions

    def _assess_risk(
        self,
        actions: List[Dict[str, Any]],
        anomaly: Any,
        diagnosis: Any
    ) -> str:
        """
        Assess risk level of proposed actions.

        Args:
            actions: Proposed remediation actions
            anomaly: Detected anomaly
            diagnosis: Root cause analysis

        Returns:
            str: Risk level - LOW, MEDIUM, or HIGH
        """
        risk_score = 0

        # Base risk from action types
        for action in actions:
            action_type = action.get("type", "")
            if action_type == "rollback":
                risk_score += 2
            elif action_type == "restart":
                risk_score += 1
            elif action_type == "scale":
                risk_score += 0.5

        # Adjust based on severity
        if anomaly and anomaly.severity == "CRITICAL":
            risk_score += 1

        # Adjust based on diagnosis confidence
        if diagnosis and diagnosis.confidence < 0.5:
            risk_score += 2

        # Determine risk level
        if risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    def _requires_approval(
        self,
        risk_level: str,
        diagnosis: Any
    ) -> bool:
        """
        Determine if human approval is required.

        Args:
            risk_level: Assessed risk level
            diagnosis: Root cause analysis

        Returns:
            bool: True if approval required
        """
        # High risk always requires approval
        if risk_level == "HIGH":
            return True

        # Low confidence diagnosis requires approval
        if diagnosis and diagnosis.confidence < self.auto_repair_threshold:
            return True

        # MEDIUM risk may require approval based on threshold
        if risk_level == "MEDIUM":
            return True

        return False

    def _generate_strategy_description(
        self,
        cause_category: str,
        actions: List[Dict[str, Any]]
    ) -> str:
        """
        Generate human-readable strategy description.

        Args:
            cause_category: Category of root cause
            actions: Remediation actions

        Returns:
            str: Strategy description
        """
        action_descriptions = [a.get("description", "") for a in actions]
        return " -> ".join(action_descriptions) or "Execute remediation actions"

    def _estimate_recovery_time(
        self,
        actions: List[Dict[str, Any]]
    ) -> str:
        """
        Estimate time to recover.

        Args:
            actions: Remediation actions

        Returns:
            str: Estimated recovery time
        """
        # Base estimate on action count and type
        base_time = len(actions) * 2  # 2 minutes per action

        for action in actions:
            if action.get("type") == "rollback":
                base_time += 5
            elif action.get("type") == "scale":
                base_time += 3

        return f"{base_time} minutes"

    def _generate_rollback_plan(
        self,
        actions: List[Dict[str, Any]]
    ) -> str:
        """
        Generate rollback plan in case remediation fails.

        Args:
            actions: Remediation actions

        Returns:
            str: Rollback plan description
        """
        rollback_steps = []

        for action in reversed(actions):
            action_type = action.get("type", "")
            target = action.get("target", "unknown")

            if action_type == "restart":
                rollback_steps.append(f"Restore {target} from backup")
            elif action_type == "scale":
                rollback_steps.append(f"Scale {target} back to original replicas")
            elif action_type == "rollback":
                rollback_steps.append(f"Re-apply current version to {target}")

        return "; ".join(rollback_steps) if rollback_steps else "Manual intervention required"

    def _calculate_decision_confidence(
        self,
        diagnosis: Any,
        historical_strategies: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence in the decision.

        Args:
            diagnosis: Root cause analysis
            historical_strategies: Retrieved historical strategies

        Returns:
            float: Confidence score [0.0, 1.0]
        """
        # Base confidence on diagnosis confidence
        base_confidence = diagnosis.confidence if diagnosis else 0.5

        # Boost confidence if historical strategies exist
        if historical_strategies:
            historical_boost = min(len(historical_strategies) * 0.1, 0.2)
            return min(base_confidence + historical_boost, 0.95)

        return base_confidence * 0.9  # Slight penalty without historical data
