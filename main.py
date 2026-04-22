"""
Main Entry Point for AIOps Agent System

Provides the primary interface for running the AIOps multi-agent system.
Supports both programmatic usage and CLI execution with interactive mode.
"""

import sys
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from enum import Enum

# Add project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import asyncio
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


# ANSI Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


# RCA Method definitions
class RCAMethod:
    """Available RCA methods for user selection."""

    def __init__(self, value: str, label: str, description: str):
        self._value = value
        self._label = label
        self._description = description

    @property
    def value(self) -> str:
        return self._value

    @property
    def label(self) -> str:
        return self._label

    @property
    def description(self) -> str:
        return self._description

    @classmethod
    def all_methods(cls) -> List["RCAMethod"]:
        """Get all RCA method instances."""
        return [
            cls("auto", "🤖 Auto Select", "Automatically select the best method based on problem characteristics"),
            cls("5_whys", "❓ 5 Whys", "Ask 'why' repeatedly (5 times) to drill down to root cause"),
            cls("ishikawa", "🐟 Ishikawa (Fishbone)", "Categorize causes in 6 dimensions: Infrastructure, Software, Process, People, Data, Environment"),
            cls("fault_tree", "🌳 Fault Tree Analysis", "Top-down Boolean logic analysis with AND/OR gates"),
            cls("change_analysis", "🔄 Change Analysis", "Analyze recent changes to find the cause"),
            cls("event_correlation", "🔗 Event Correlation", "Correlate multiple events to find primary cause and cascade patterns"),
            cls("ensemble", "🎯 Ensemble (All Methods)", "Run multiple methods and synthesize results for comprehensive analysis"),
        ]


def colorize(text: str, color: str) -> str:
    """Add ANSI color codes to text."""
    return f"{color}{text}{Colors.RESET}"


def print_header(text: str, char: str = "=", width: int = 60):
    """Print a formatted header."""
    print(f"\n{colorize(char * width, Colors.GRAY)}")
    print(colorize(f"  {text}", Colors.BOLD))
    print(colorize(char * width, Colors.GRAY))


def print_step(step_num: int, title: str, status: str = "pending"):
    """Print a workflow step with status indicator."""
    icons = {
        "pending": colorize("○", Colors.GRAY),
        "running": colorize("◐", Colors.YELLOW),
        "done": colorize("✓", Colors.GREEN),
        "error": colorize("✗", Colors.RED),
        "waiting": colorize("⏳", Colors.YELLOW),
    }
    icon = icons.get(status, "○")
    print(f"\n{icon} {colorize(f'Step {step_num}', Colors.DIM)}: {colorize(title, Colors.BOLD)}")


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate long text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def truncate_list(items: List[Any], max_items: int = 3) -> str:
    """Truncate list display with count."""
    if len(items) <= max_items:
        return str(items)
    return f"[{', '.join(str(x) for x in items[:max_items])}, ... ({len(items) - max_items} more)]"


def print_collapsible(section_title: str, content: str, max_lines: int = 5):
    """Print content that can be collapsed if too long."""
    lines = content.split('\n')
    print(f"\n{colorize(f'▼ {section_title}', Colors.CYAN)}")
    if len(lines) > max_lines:
        for line in lines[:max_lines]:
            print(colorize(f"  {line}", Colors.GRAY))
        print(colorize(f"  ... ({len(lines) - max_lines} more lines hidden)", Colors.DIM))
    else:
        for line in lines:
            print(colorize(f"  {line}", Colors.WHITE))


def print_progress_bar(current: int, total: int, width: int = 40) -> str:
    """Generate a progress bar string."""
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percent*100:.0f}%"


def print_rca_method_selection() -> Optional[str]:
    """Print available RCA methods and let user select one."""
    print(colorize("\n" + "═" * 70, Colors.GRAY))
    print(colorize("  🔍 Select Root Cause Analysis Algorithm", Colors.BOLD + Colors.CYAN))
    print(colorize("═" * 70, Colors.GRAY))
    print()

    methods = RCAMethod.all_methods()

    # Print all available methods
    for i, method in enumerate(methods, 1):
        print(colorize(f"  [{i}] {method.label}", Colors.WHITE))
        print(colorize(f"      {method.description}", Colors.GRAY))
        print()

    print(colorize("─" * 70, Colors.DIM))

    # Get user selection
    while True:
        try:
            choice = input(colorize("\nSelect method [1-7, default=1]: ", Colors.CYAN)).strip()

            if not choice:
                choice = "1"

            choice_num = int(choice)

            if 1 <= choice_num <= len(methods):
                selected = methods[choice_num - 1]
                print(colorize(f"\n✓ Selected: {selected.label}", Colors.GREEN))
                return selected.value
            else:
                print(colorize(f"  Please enter a number between 1 and {len(methods)}", Colors.RED))
        except ValueError:
            print(colorize("  Invalid input. Please enter a number.", Colors.RED))
        except EOFError:
            print(colorize("\n  Using default (Auto Select)", Colors.YELLOW))
            return RCAMethod.AUTO.value


class AIOpsSystem:
    """
    Main AIOps multi-agent system orchestrator.

    Initializes and configures all agents, tools, and shared resources
    for incident management.

    Attributes:
        settings (Settings): System configuration
        supervisor (SupervisorAgent): Main orchestration agent
        knowledge_base (GRPOKnowledgeBase): Experience storage
    """

    def __init__(
        self,
        settings: Settings,
        llm: Optional[ChatOpenAI] = None,
        interactive: bool = False
    ):
        """
        Initialize the AIOps system.

        Args:
            settings (Settings): System configuration
            llm (Optional[ChatOpenAI]): Pre-configured LLM instance
            interactive (bool): Enable interactive mode with user prompts
        """
        self.settings = settings
        self.interactive = interactive

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
            if interactive:
                print(colorize(f"📊 [Mock Mode] Using mock monitoring data (anomaly_type={settings.mock.anomaly_type})", Colors.CYAN))
            else:
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

        # Initialize agents with interactive mode
        agent_kwargs = {"verbose": False} if not interactive else {"verbose": settings.agent.verbose}

        self.observability_agent = ObservabilityAgent(
            llm=self.llm,
            prometheus_tool=self.prometheus,
            loki_tool=self.loki,
            time_range_minutes=settings.prometheus.default_time_range,
            **agent_kwargs
        )

        self.detection_agent = DetectionAgent(
            llm=self.llm,
            **agent_kwargs
        )

        self.diagnosis_agent = DiagnosisAgent(
            llm=self.llm,
            **agent_kwargs
        )

        self.decision_agent = DecisionAgent(
            llm=self.llm,
            knowledge_base=self.knowledge_base,
            auto_repair_threshold=settings.agent.auto_repair_threshold,
            **agent_kwargs
        )

        self.repair_agent = RepairAgent(
            mode=RepairMode.HYBRID if settings.agent.dry_run else RepairMode.AUTO,
            dry_run=settings.agent.dry_run,
            **agent_kwargs
        )

        self.report_agent = ReportAgent(
            llm=self.llm,
            **agent_kwargs
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
            **agent_kwargs
        )

    @classmethod
    def from_env(cls, interactive: bool = False, select_rca: bool = False) -> "AIOpsSystem":
        """
        Create AIOps system from environment variables.

        Args:
            interactive (bool): Enable interactive mode
            select_rca (bool): Enable RCA method selection

        Returns:
            AIOpsSystem: Configured system instance
        """
        settings = Settings.from_env()
        instance = cls(settings, interactive=interactive)
        instance.select_rca = select_rca
        return instance

    def process_incident_interactive(
        self,
        incident_id: Optional[str] = None,
        alert_labels: Optional[Dict[str, str]] = None,
        entities: Optional[List[str]] = None,
        time_range: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an incident with interactive progress display and user interaction.

        Args:
            incident_id (Optional[str]): Custom incident ID
            alert_labels (Optional[Dict[str, str]]): Alert labels
            entities (Optional[List[str]]): Affected entities
            time_range (Optional[Dict[str, Any]]): Custom time range

        Returns:
            Dict[str, Any]: Final workflow state
        """
        workflow_steps = [
            (1, "📊 Data Collection", "observability"),
            (2, "🔍 Anomaly Detection", "detection"),
            (3, "🧠 Root Cause Analysis", "diagnosis"),
            (4, "📋 Decision Making", "decision"),
            (5, "🔧 Repair Execution", "repair"),
            (6, "📝 Report Generation", "report"),
        ]

        # Create initial state
        state = {
            "incident_id": incident_id or f"inc-{int(time.time())}",
            "alert_labels": alert_labels or {},
            "entities": entities or [],
            "time_range": time_range,
            "logs": [],
            "errors": [],
            "rca_method": None,  # Will be set by user selection if enabled
        }

        print_header(f"🚀 Processing Incident: {state['incident_id']}", "═")

        # Show incident context
        print(f"\n{colorize('Alert Labels:', Colors.CYAN)} {alert_labels}")
        print(f"{colorize('Affected Entities:', Colors.CYAN)} {entities}")
        print(f"{colorize('Mode:', Colors.CYAN)} {'🎭 Mock' if self.settings.mock.enabled else '🔌 Live'}")

        # Let user select RCA method if requested
        if hasattr(self, 'select_rca') and self.select_rca:
            state["rca_method"] = print_rca_method_selection()

        current_step = 0
        agent_map = {
            "observability": self.observability_agent,
            "detection": self.detection_agent,
            "diagnosis": self.diagnosis_agent,
            "decision": self.decision_agent,
            "repair": self.repair_agent,
            "report": self.report_agent,
        }

        for step_num, title, agent_name in workflow_steps:
            current_step = step_num
            print_step(step_num, title, "running")

            if agent_name in agent_map:
                agent = agent_map[agent_name]

                # Special handling for repair step (requires approval)
                if agent_name == "repair" and state.get("decision"):
                    decision = state["decision"]
                    if getattr(decision, 'requires_approval', False):
                        approved = self._ask_repair_approval(decision)
                        if not approved:
                            print(colorize("\n⚠️  Repair rejected by operator", Colors.YELLOW))
                            state["repair_rejected"] = True
                            state["status"] = "escalated"
                            break

                # Execute agent with timing
                start_time = time.time()
                try:
                    state = agent.execute(state)
                    elapsed = time.time() - start_time
                    print_step(step_num, title, "done")

                    # Show summary for each step
                    self._show_step_summary(agent_name, state, elapsed)

                except Exception as e:
                    print_step(step_num, title, "error")
                    print(colorize(f"  Error: {str(e)}", Colors.RED))
                    state["errors"] = state.get("errors", [])
                    state["errors"].append(f"{agent_name} error: {str(e)}")

                    if self.interactive:
                        choice = input(colorize("\n⚠️  Continue anyway? (y/n): ", Colors.YELLOW))
                        if choice.lower() != 'y':
                            state["status"] = "failed"
                            break

            # Show progress bar
            progress = print_progress_bar(current_step, len(workflow_steps))
            print(colorize(f"\n  {progress}", Colors.DIM))

        # Show final status
        self._show_final_result(state)

        # Save report to file
        report_path = self._save_report(state)
        if report_path:
            print(colorize(f"📁 Report saved to: {colorize(report_path, Colors.CYAN)}", Colors.GREEN))

        return state

    def _ask_repair_approval(self, decision) -> bool:
        """Ask user to approve repair actions."""
        print_header("🔧 Repair Plan Requires Approval", "─")

        actions = getattr(decision, 'actions', [])
        risk_level = getattr(decision, 'risk_level', 'UNKNOWN')

        print(f"\n{colorize('Risk Level:', Colors.YELLOW)} {risk_level}")
        print(f"\n{colorize('Proposed Actions:', Colors.CYAN)}")

        for i, action in enumerate(actions, 1):
            if isinstance(action, dict):
                action_type = action.get("type", "unknown")
                print(f"\n  {colorize(f'{i}. {action_type}', Colors.GREEN)}")
                print(f"     Target: {action.get('target', 'N/A')}")
                print(f"     Description: {truncate_text(action.get('description', ''), 60)}")
                print(f"     Expected: {action.get('expected_outcome', 'N/A')}")

        print(f"\n{colorize('─' * 40, Colors.GRAY)}")

        while True:
            choice = input(colorize("\n✅ Approve repair plan? (yes/no/view): ", Colors.CYAN)).lower().strip()

            if choice in ['yes', 'y']:
                print(colorize("✓ Repair approved", Colors.GREEN))
                return True
            elif choice in ['no', 'n']:
                print(colorize("✗ Repair rejected", Colors.RED))
                return False
            elif choice == 'view':
                print(f"\n{colorize('Full Action Details:', Colors.CYAN)}")
                for action in actions:
                    print(f"  {action}")
            else:
                print(colorize("  Please enter 'yes', 'no', or 'view'", Colors.YELLOW))

    def _show_step_summary(self, agent_name: str, state: Dict[str, Any], elapsed: float):
        """Show summary after each agent step."""
        if agent_name == "observability":
            obs = state.get("observability")
            if obs:
                metrics_count = len(getattr(obs, 'metrics', []))
                logs_count = len(getattr(obs, 'logs', []))
                print(colorize(f"  📈 Collected {metrics_count} metrics, {logs_count} logs in {elapsed:.1f}s", Colors.GREEN))

        elif agent_name == "detection":
            anomaly = state.get("anomaly")
            if anomaly:
                atype = getattr(anomaly, 'type', 'unknown')
                confidence = getattr(anomaly, 'confidence', 0)
                severity = getattr(anomaly, 'severity', 'unknown')
                emoji = "🚨" if severity in ['HIGH', 'CRITICAL'] else "⚠️"
                print(colorize(f"  {emoji} Detected: {atype} (confidence: {confidence:.0%}, severity: {severity})", Colors.GREEN if confidence > 0.7 else Colors.YELLOW))

        elif agent_name == "diagnosis":
            diagnosis = state.get("diagnosis")
            if diagnosis:
                root_causes = getattr(diagnosis, 'root_causes', [])
                print(colorize(f"  🎯 Identified {len(root_causes)} potential root cause(s) in {elapsed:.1f}s", Colors.GREEN))
                if root_causes and isinstance(root_causes[0], dict):
                    print(colorize(f"     Primary: {truncate_text(root_causes[0].get('cause', ''), 50)}", Colors.CYAN))

        elif agent_name == "decision":
            decision = state.get("decision")
            if decision:
                actions = getattr(decision, 'actions', [])
                risk = getattr(decision, 'risk_level', 'unknown')
                risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk, "⚪")
                print(colorize(f"  {risk_emoji} Decision: {len(actions)} action(s), risk: {risk}", Colors.GREEN))

        elif agent_name == "repair":
            repair = state.get("repair")
            if repair:
                status = getattr(repair, 'status', 'unknown')
                status_emoji = {"success": "✅", "pending": "⏳", "failed": "❌"}.get(status, "⚪")
                print(colorize(f"  {status_emoji} Repair status: {status}", Colors.GREEN if status == 'success' else Colors.YELLOW))

        elif agent_name == "report":
            report = state.get("report")
            if report:
                summary = report.get('summary', '')
                print(colorize(f"  📄 Report generated in {elapsed:.1f}s", Colors.GREEN))
                if summary:
                    print(colorize(f"     Summary: {truncate_text(summary, 60)}", Colors.CYAN))

    def _show_final_result(self, state: Dict[str, Any]):
        """Show final incident result."""
        status = state.get("status", "unknown")
        status_colors = {
            "resolved": Colors.GREEN,
            "deciding": Colors.YELLOW,
            "escalated": Colors.RED,
            "failed": Colors.RED,
        }
        status_emoji = {
            "resolved": "✅",
            "deciding": "⏸️",
            "escalated": "🔔",
            "failed": "❌",
        }

        print_header(f"{status_emoji.get(status, '⚪')} Final Status: {colorize(status.upper(), status_colors.get(status, Colors.WHITE))}", "═")

        # Show key findings
        if state.get("anomaly"):
            anomaly = state["anomaly"]
            print(f"\n{colorize('Anomaly:', Colors.CYAN)}")
            print(f"  Type: {getattr(anomaly, 'type', 'unknown')}")
            print(f"  Confidence: {getattr(anomaly, 'confidence', 0):.1%}")
            print(f"  Severity: {getattr(anomaly, 'severity', 'unknown')}")

        if state.get("diagnosis"):
            diagnosis = state["diagnosis"]
            root_causes = getattr(diagnosis, 'root_causes', [])
            if root_causes:
                print(f"\n{colorize('Root Cause:', Colors.CYAN)}")
                if isinstance(root_causes[0], dict):
                    print(f"  {root_causes[0].get('cause', 'unknown')}")

        if state.get("decision"):
            decision = state["decision"]
            actions = getattr(decision, 'actions', [])
            if actions:
                print(f"\n{colorize('Actions:', Colors.CYAN)}")
                for action in actions[:3]:
                    if isinstance(action, dict):
                        print(f"  • {action.get('type', 'unknown')}: {action.get('target', 'N/A')}")
                if len(actions) > 3:
                    print(f"  ... and {len(actions) - 3} more")

        if state.get("errors"):
            print(f"\n{colorize('Errors:', Colors.RED)}")
            for error in state["errors"]:
                print(f"  ⚠️  {error}")

        print()

    def _save_report(self, state: Dict[str, Any], report_dir: str = "./reports") -> Optional[str]:
        """
        Save the incident report to a file.

        Args:
            state (Dict[str, Any]): Final workflow state
            report_dir (str): Directory to save reports

        Returns:
            Optional[str]: Path to saved report file, or None if no report
        """
        import json
        from datetime import datetime

        report = state.get("report")
        if not report:
            return None

        # Ensure reports directory exists
        os.makedirs(report_dir, exist_ok=True)

        # Generate filename with timestamp
        incident_id = state.get("incident_id", f"inc-{int(time.time())}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{incident_id}_{timestamp}_report.md"
        filepath = os.path.join(report_dir, filename)

        # Build markdown report
        lines = []
        lines.append(f"# AIOps Incident Report")
        lines.append(f"")
        lines.append(f"**Incident ID:** {incident_id}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Status:** {state.get('status', 'unknown').upper()}")
        lines.append(f"")

        # Anomaly section
        if state.get("anomaly"):
            anomaly = state["anomaly"]
            lines.append(f"## 🚨 Anomaly Detected")
            lines.append(f"")
            lines.append(f"- **Type:** {getattr(anomaly, 'type', 'unknown')}")
            lines.append(f"- **Confidence:** {getattr(anomaly, 'confidence', 0):.1%}")
            lines.append(f"- **Severity:** {getattr(anomaly, 'severity', 'unknown')}")
            lines.append(f"- **Affected Entities:** {getattr(anomaly, 'affected_entities', [])}")
            lines.append(f"")

        # Diagnosis section
        if state.get("diagnosis"):
            diagnosis = state["diagnosis"]
            lines.append(f"## 🧠 Root Cause Analysis")
            lines.append(f"")
            root_causes = getattr(diagnosis, 'root_causes', [])
            for i, rc in enumerate(root_causes[:5], 1):
                if isinstance(rc, dict):
                    lines.append(f"{i}. **{rc.get('cause', 'unknown')}**")
                    lines.append(f"   - Category: {rc.get('category', 'unknown')}")
                    lines.append(f"   - Entity: {rc.get('entity', 'unknown')}")
                    lines.append(f"   - Confidence: {rc.get('confidence', 0):.0%}")
                    lines.append(f"   - Evidence: {rc.get('evidence', 'N/A')}")
            lines.append(f"")

        # Decision section
        if state.get("decision"):
            decision = state["decision"]
            lines.append(f"## 📋 Decision")
            lines.append(f"")
            lines.append(f"- **Risk Level:** {getattr(decision, 'risk_level', 'unknown')}")
            lines.append(f"- **Requires Approval:** {getattr(decision, 'requires_approval', False)}")
            lines.append(f"")
            lines.append(f"### Actions")
            actions = getattr(decision, 'actions', [])
            for i, action in enumerate(actions, 1):
                if isinstance(action, dict):
                    lines.append(f"{i}. **{action.get('type', 'unknown')}**")
                    lines.append(f"   - Target: {action.get('target', 'N/A')}")
                    lines.append(f"   - Description: {action.get('description', 'N/A')}")
                    lines.append(f"   - Expected Outcome: {action.get('expected_outcome', 'N/A')}")
                    params = action.get('parameters', {})
                    if params:
                        lines.append(f"   - Parameters: {params}")
            lines.append(f"")

        # Report summary
        if report:
            lines.append(f"## 📄 Summary")
            lines.append(f"")
            lines.append(f"{report.get('summary', 'N/A')}")
            lines.append(f"")

        # Errors section
        if state.get("errors"):
            lines.append(f"## ⚠️ Errors")
            lines.append(f"")
            for error in state["errors"]:
                lines.append(f"- {error}")
            lines.append(f"")

        # Raw JSON data
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## Raw Data (JSON)")
        lines.append(f"")
        lines.append(f"```json")
        lines.append(json.dumps({
            "incident_id": incident_id,
            "status": state.get("status"),
            "alert_labels": state.get("alert_labels"),
            "entities": state.get("entities"),
        }, indent=2))
        lines.append(f"```")
        lines.append(f"")

        # Write to file
        content = "\n".join(lines)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath

    def process_incident(
        self,
        incident_id: Optional[str] = None,
        alert_labels: Optional[Dict[str, str]] = None,
        entities: Optional[List[str]] = None,
        time_range: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an incident through the complete workflow (non-interactive).

        Args:
            incident_id (Optional[str]): Custom incident ID
            alert_labels (Optional[Dict[str, str]]): Alert labels
            entities (Optional[List[str]]): Affected entities
            time_range (Optional[Dict[str, Any]]): Custom time range

        Returns:
            Dict[str, Any]: Final workflow state
        """
        initial_state = {
            "incident_id": incident_id,
            "alert_labels": alert_labels or {},
            "entities": entities or [],
            "time_range": time_range,
            "logs": [],
            "errors": [],
        }

        final_state = self.supervisor.execute(initial_state)
        return final_state

    def approve_repair(
        self,
        state: Dict[str, Any],
        approved: bool
    ) -> Dict[str, Any]:
        """
        Approve or reject pending repair actions.

        Args:
            state (Dict[str, Any]): Current workflow state
            approved (bool): Whether repair was approved

        Returns:
            Dict[str, Any]: Updated workflow state
        """
        return self.repair_agent.approve_and_execute(state, approved)

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the GRPO knowledge base.

        Returns:
            Dict[str, Any]: Knowledge base statistics
        """
        return self.knowledge_base.get_stats()


def main():
    """
    CLI entry point for the AIOps system.

    Example usage:
        ```bash
        # Run with environment configuration
        OPENAI_API_KEY=xxx PROMETHEUS_URL=xxx python -m main

        # Process a test incident with interactive mode
        python -m main --interactive

        # Quick test (non-interactive)
        python -m main --test-incident

        # Show detailed RCA algorithm execution logs
        python -m main --interactive --debug-rca
        ```
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="AIOps Multi-Agent System"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode with progress and user prompts"
    )
    parser.add_argument(
        "--test-incident",
        action="store_true",
        help="Run a test incident (non-interactive)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no actual changes)"
    )
    parser.add_argument(
        "--debug-rca",
        action="store_true",
        help="Show detailed RCA algorithm execution logs"
    )
    parser.add_argument(
        "--select-rca",
        action="store_true",
        help="Interactively select RCA algorithm before analysis"
    )
    parser.add_argument(
        "--list-rca",
        action="store_true",
        help="List available RCA algorithms and exit"
    )
    args = parser.parse_args()

    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")

    # Handle --list-rca flag
    if args.list_rca:
        # Just print the list without waiting for input
        print(colorize("\n" + "═" * 70, Colors.GRAY))
        print(colorize("  🔍 Available Root Cause Analysis Algorithms", Colors.BOLD + Colors.CYAN))
        print(colorize("═" * 70, Colors.GRAY))
        print()

        methods = RCAMethod.all_methods()
        for i, method in enumerate(methods, 1):
            print(colorize(f"  [{i}] {method.label}", Colors.WHITE))
            print(colorize(f"      {method.description}", Colors.GRAY))
            print()

        print(colorize("─" * 70, Colors.DIM))
        print(colorize("\nUsage: python main.py --select-rca --test-incident", Colors.CYAN))
        print(colorize("       python main.py --interactive --select-rca", Colors.CYAN))
        sys.exit(0)

    # Initialize system
    if args.interactive:
        print(colorize("\n🤖 AIOps Multi-Agent System - Interactive Mode", Colors.CYAN))
        print(colorize("━" * 50, Colors.GRAY))
        if args.debug_rca:
            print(colorize("🔍 RCA Debug Mode: Enabled", Colors.YELLOW))
        if args.select_rca:
            print(colorize("🎯 RCA Method Selection: Enabled", Colors.YELLOW))
    else:
        print("Initializing AIOps system...")
        if args.debug_rca:
            # In non-interactive debug mode, show full LLM interactions
            print(colorize("🔍 RCA Debug Mode: Showing LLM interactions", Colors.YELLOW))
            print(colorize("─" * 50, Colors.GRAY))

    # Set verbose based on debug flag
    os.environ["VERBOSE"] = "true" if args.debug_rca else os.getenv("VERBOSE", "false")

    aiops = AIOpsSystem.from_env(interactive=args.interactive, select_rca=args.select_rca)

    # Force verbose mode for debug output
    if args.debug_rca:
        aiops.observability_agent.verbose = True
        aiops.detection_agent.verbose = True
        aiops.diagnosis_agent.verbose = True
        aiops.decision_agent.verbose = True
        aiops.repair_agent.verbose = True
        aiops.report_agent.verbose = True

    if args.test_incident or args.interactive or args.select_rca:
        # Use interactive mode if requested
        if args.interactive or args.select_rca or args.debug_rca:
            # In debug mode, also use interactive path to show LLM logs
            result = aiops.process_incident_interactive(
                alert_labels={"service": "api-service", "severity": "high"},
                entities=["api-service-pod-1"]
            )
        else:
            print("Processing test incident...")
            result = aiops.process_incident(
                alert_labels={"service": "api-service", "severity": "high"},
                entities=["api-service-pod-1"]
            )

            # Simple output for non-interactive mode
            print("\n" + "=" * 60)
            print("INCIDENT RESULT")
            print("=" * 60)
            print(f"Status: {result.get('status', 'unknown')}")

            if result.get("anomaly"):
                anomaly = result["anomaly"]
                print(f"Anomaly: {getattr(anomaly, 'type', 'unknown')} ({getattr(anomaly, 'confidence', 0):.1%})")

            if result.get("decision"):
                decision = result["decision"]
                actions = getattr(decision, 'actions', [])
                print(f"Actions: {len(actions)} proposed")
    else:
        print("AIOps system initialized successfully!")
        print("\nUsage:")
        print("  python main.py --interactive  # Interactive mode")
        print("  python main.py --test-incident  # Quick test")
        print("\nOr use programmatically:")
        print("  aiops = AIOpsSystem.from_env(interactive=True)")
        print("  result = aiops.process_incident_interactive(...)")


if __name__ == "__main__":
    main()
