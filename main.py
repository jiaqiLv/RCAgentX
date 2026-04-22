"""
Main Entry Point for AIOps Agent System

Provides the primary interface for running the AIOps multi-agent system.
Supports both programmatic usage and CLI execution.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import asyncio
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI

from agents.supervisor import SupervisorAgent
from agents.observability import ObservabilityAgent
from agents.detection import DetectionAgent
from agents.diagnosis import DiagnosisAgent
from agents.decision import DecisionAgent
from agents.repair import RepairAgent
from agents.report import ReportAgent
from memory.shared_state import SharedState, RepairMode
from memory.knowledge_base import GRPOKnowledgeBase
from tools.prometheus import PrometheusTool
from tools.loki import LokiTool
from tools.mock_monitoring import MockPrometheusTool, MockLokiTool
from config.settings import Settings


class AIOpsSystem:
    """
    Main AIOps multi-agent system orchestrator.

    Initializes and configures all agents, tools, and shared resources
    for incident management.

    Attributes:
        settings (Settings): System configuration
        supervisor (SupervisorAgent): Main orchestration agent
        knowledge_base (GRPOKnowledgeBase): Experience storage

    Example:
        ```python
        # Initialize system
        aiops = AIOpsSystem.from_settings(settings)

        # Run incident processing
        result = aiops.process_incident(
            alert_labels={"service": "api", "severity": "critical"},
            entities=["api-service-pod-1"]
        )

        # Access results
        print(f"Status: {result['status']}")
        print(f"Report: {result['report']['summary']}")
        ```
    """

    def __init__(
        self,
        settings: Settings,
        llm: Optional[ChatOpenAI] = None
    ):
        """
        Initialize the AIOps system.

        Args:
            settings (Settings): System configuration
            llm (Optional[ChatOpenAI]): Pre-configured LLM instance
        """
        self.settings = settings

        # Initialize LLM
        self.llm = llm or ChatOpenAI(
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_base,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )

        # Initialize tools (use mock if enabled)
        if settings.mock.enabled:
            print(f"[Mock Mode] Using mock monitoring data (anomaly_type={settings.mock.anomaly_type})")
            self.prometheus = MockPrometheusTool(seed=settings.mock.seed)
            self.loki = MockLokiTool(seed=settings.mock.seed)
        else:
            self.prometheus = PrometheusTool(url=settings.prometheus.url)
            self.loki = LokiTool(url=settings.loki.url)

        # Initialize knowledge base
        self.knowledge_base = GRPOKnowledgeBase(
            persist_dir=settings.knowledge_base.persist_dir
        )

        # Initialize agents
        self.observability_agent = ObservabilityAgent(
            llm=self.llm,
            prometheus_tool=self.prometheus,
            loki_tool=self.loki,
            time_range_minutes=settings.prometheus.default_time_range,
            verbose=settings.agent.verbose
        )

        self.detection_agent = DetectionAgent(
            llm=self.llm,
            verbose=settings.agent.verbose
        )

        self.diagnosis_agent = DiagnosisAgent(
            llm=self.llm,
            verbose=settings.agent.verbose
        )

        self.decision_agent = DecisionAgent(
            llm=self.llm,
            knowledge_base=self.knowledge_base,
            auto_repair_threshold=settings.agent.auto_repair_threshold,
            verbose=settings.agent.verbose
        )

        self.repair_agent = RepairAgent(
            mode=RepairMode.HYBRID if settings.agent.dry_run else RepairMode.AUTO,
            dry_run=settings.agent.dry_run,
            verbose=settings.agent.verbose
        )

        self.report_agent = ReportAgent(
            llm=self.llm,
            verbose=settings.agent.verbose
        )

        # Initialize supervisor with all sub-agents
        self.supervisor = SupervisorAgent(
            sub_agents={
                "observability": self.observability_agent,
                "detection": self.detection_agent,
                "diagnosis": self.diagnosis_agent,
                "decision": self.decision_agent,
                "repair": self.repair_agent,
                "report": self.report_agent,
            },
            knowledge_base=self.knowledge_base,
            llm=self.llm,
            verbose=settings.agent.verbose
        )

    @classmethod
    def from_env(cls) -> "AIOpsSystem":
        """
        Create AIOps system from environment variables.

        Returns:
            AIOpsSystem: Configured system instance

        Example:
            ```python
            aiops = AIOpsSystem.from_env()
            ```
        """
        settings = Settings.from_env()
        return cls(settings)

    def process_incident(
        self,
        incident_id: Optional[str] = None,
        alert_labels: Optional[Dict[str, str]] = None,
        entities: Optional[List[str]] = None,
        time_range: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an incident through the complete workflow.

        Args:
            incident_id (Optional[str]): Custom incident ID
            alert_labels (Optional[Dict[str, str]]): Alert labels
            entities (Optional[List[str]]): Affected entities
            time_range (Optional[Dict[str, Any]]): Custom time range

        Returns:
            Dict[str, Any]: Final workflow state

        Example:
            ```python
            result = aiops.process_incident(
                alert_labels={"service": "api", "severity": "critical"},
                entities=["api-service-pod-1"]
            )
            ```
        """
        # Create initial state
        initial_state = {
            "incident_id": incident_id,
            "alert_labels": alert_labels or {},
            "entities": entities or [],
            "time_range": time_range,
            "logs": [],
            "errors": [],
        }

        # Execute workflow
        final_state = self.supervisor.execute(initial_state)

        return final_state

    def approve_repair(
        self,
        state: Dict[str, Any],
        approved: bool
    ) -> Dict[str, Any]:
        """
        Approve or reject pending repair actions.

        Called when a human operator reviews and approves/rejects
        the proposed repair plan.

        Args:
            state (Dict[str, Any]): Current workflow state
            approved (bool): Whether repair was approved

        Returns:
            Dict[str, Any]: Updated workflow state

        Example:
            ```python
            # After human approval via UI/Slack/WeChat
            result = aiops.approve_repair(state, approved=True)
            ```
        """
        return self.repair_agent.approve_and_execute(state, approved)

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the GRPO knowledge base.

        Returns:
            Dict[str, Any]: Knowledge base statistics

        Example:
            ```python
            stats = aiops.get_knowledge_base_stats()
            print(f"Total experiences: {stats['total_records']}")
            print(f"Success rate: {stats['success_rate']:.1%}")
            ```
        """
        return self.knowledge_base.get_stats()


def main():
    """
    CLI entry point for the AIOps system.

    Example usage:
        ```bash
        # Run with environment configuration
        OPENAI_API_KEY=xxx PROMETHEUS_URL=xxx python -m main

        # Process a test incident
        python -m main --test-incident
        ```
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="AIOps Multi-Agent System"
    )
    parser.add_argument(
        "--test-incident",
        action="store_true",
        help="Run a test incident"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no actual changes)"
    )
    args = parser.parse_args()

    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")

    # Initialize system
    print("Initializing AIOps system...")
    aiops = AIOpsSystem.from_env()

    if args.test_incident:
        print("Processing test incident...")

        # Simulate a test incident
        result = aiops.process_incident(
            alert_labels={"service": "api-service", "severity": "high"},
            entities=["api-service-pod-1"]
        )

        print("\n" + "=" * 60)
        print("INCIDENT RESULT")
        print("=" * 60)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Incident ID: {result.get('incident_id', 'unknown')}")

        # Show all state keys for debugging
        print(f"\nWorkflow state keys: {list(result.keys())}")

        # Show anomaly if detected
        if result.get("anomaly"):
            anomaly = result["anomaly"]
            print(f"\nAnomaly detected:")
            print(f"  Type: {getattr(anomaly, 'type', 'unknown')}")
            print(f"  Confidence: {getattr(anomaly, 'confidence', 0):.1%}")
            print(f"  Severity: {getattr(anomaly, 'severity', 'unknown')}")

        # Show diagnosis if available
        if result.get("diagnosis"):
            diagnosis = result["diagnosis"]
            print(f"\nDiagnosis:")
            print(f"  Root causes: {getattr(diagnosis, 'root_causes', [])}")

        # Show decision if available
        if result.get("decision"):
            decision = result["decision"]
            print(f"\nDecision:")
            print(f"  Risk level: {getattr(decision, 'risk_level', 'unknown')}")
            print(f"  Actions: {getattr(decision, 'actions', [])}")
            print(f"  Requires approval: {getattr(decision, 'requires_approval', False)}")

        # Show report if available
        if result.get("report"):
            print(f"\nSummary: {result['report'].get('summary', 'N/A')}")

        if result.get("errors"):
            print(f"\nErrors: {result['errors']}")

    else:
        print("AIOps system initialized successfully!")
        print("Use the aiops object to process incidents programmatically.")
        print("\nExample:")
        print("  result = aiops.process_incident(")
        print("      alert_labels={'service': 'api', 'severity': 'critical'}")
        print("  )")


if __name__ == "__main__":
    main()
