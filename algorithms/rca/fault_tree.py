"""
Fault Tree Analysis (FTA)

A top-down, deductive analysis method that starts with an undesired
state (the "top event") and works backward to identify all the ways
it could occur. Uses Boolean logic (AND/OR gates) to combine events.

This implementation builds a fault tree and calculates failure probabilities.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class GateType(Enum):
    """Boolean gate types for fault trees."""
    AND = "AND"  # All inputs must occur
    OR = "OR"    # Any input can cause output
    VOTING = "VOTING"  # K of N inputs must occur


@dataclass
class BasicEvent:
    """A basic (leaf) event in the fault tree."""
    id: str
    description: str
    probability: float  # 0-1 probability of occurrence
    category: str = "unknown"


@dataclass
class IntermediateEvent:
    """An intermediate event with a logic gate."""
    id: str
    description: str
    gate_type: GateType
    inputs: List[Any]  # Can be BasicEvent or IntermediateEvent
    vote_threshold: Optional[int] = None  # For VOTING gates


@dataclass
class FaultTree:
    """Complete fault tree structure."""
    top_event: str
    root: IntermediateEvent
    all_events: List[BasicEvent] = field(default_factory=list)
    calculated_probability: float = 0.0
    minimal_cut_sets: List[List[str]] = field(default_factory=list)


class FaultTreeInput(BaseModel):
    """Input schema for Fault Tree Analysis."""
    top_event: str
    system_context: Optional[str] = None
    known_failures: Optional[List[str]] = None


class FaultTreeAnalyzer:
    """
    Implements Fault Tree Analysis (FTA).

    FTA is a deductive, top-down method that:
    1. Starts with an undesired top event (system failure)
    2. Identifies immediate causes using logic gates
    3. Continues decomposing until basic events
    4. Calculates failure probability
    5. Identifies minimal cut sets

    Attributes:
        llm: Language model for tree construction
        verbose (bool): Enable verbose output
    """

    def __init__(self, llm=None, verbose: bool = False):
        """
        Initialize the Fault Tree analyzer.

        Args:
            llm: Language model instance
            verbose (bool): Enable detailed logging
        """
        self.llm = llm
        self.verbose = verbose

    def analyze(self, top_event: str, system_context: Optional[str] = None) -> FaultTree:
        """
        Build and analyze a fault tree for the top event.

        Args:
            top_event (str): The undesired event to analyze
            system_context (Optional[str]): System description

        Returns:
            FaultTree: Complete fault tree structure
        """
        # Build the fault tree structure
        root = self._build_fault_tree(top_event, system_context)

        # Create fault tree object
        tree = FaultTree(
            top_event=top_event,
            root=root,
            all_events=self._collect_basic_events(root)
        )

        # Calculate top event probability
        tree.calculated_probability = self._calculate_probability(root)

        # Find minimal cut sets
        tree.minimal_cut_sets = self._find_minimal_cut_sets(root)

        return tree

    def _build_fault_tree(self, top_event: str, context: Optional[str]) -> IntermediateEvent:
        """Build fault tree using LLM or template."""
        if not self.llm:
            return self._build_template_tree(top_event)

        prompt = self._build_tree_prompt(top_event, context)

        try:
            response = self.llm.invoke(prompt)
            return self._parse_tree_response(response.content, top_event)
        except Exception:
            return self._build_template_tree(top_event)

    def _build_template_tree(self, top_event: str) -> IntermediateEvent:
        """Build a template fault tree for common scenarios."""
        # Common patterns for technical incidents
        if "cpu" in top_event.lower() or "performance" in top_event.lower():
            return self._cpu_fault_tree(top_event)
        elif "memory" in top_event.lower():
            return self._memory_fault_tree(top_event)
        elif "latency" in top_event.lower() or "slow" in top_event.lower():
            return self._latency_fault_tree(top_event)
        elif "error" in top_event.lower() or "failure" in top_event.lower():
            return self._error_fault_tree(top_event)
        else:
            return self._generic_fault_tree(top_event)

    def _cpu_fault_tree(self, top_event: str) -> IntermediateEvent:
        """Template fault tree for CPU-related issues."""
        # Top level: OR gate
        root = IntermediateEvent(
            id="root",
            description=top_event,
            gate_type=GateType.OR,
            inputs=[]
        )

        # Branch 1: Resource exhaustion
        resource_exhaustion = IntermediateEvent(
            id="cpu_exhaustion",
            description="CPU Resource Exhaustion",
            gate_type=GateType.AND,
            inputs=[
                BasicEvent("cpu_workload", "High workload demand", 0.3, "workload"),
                BasicEvent("cpu_capacity", "Insufficient CPU capacity", 0.2, "capacity"),
            ]
        )

        # Branch 2: Runaway process
        runaway = IntermediateEvent(
            id="runaway",
            description="Runaway Process",
            gate_type=GateType.OR,
            inputs=[
                BasicEvent("infinite_loop", "Infinite loop in code", 0.1, "software"),
                BasicEvent("memory_leak", "Memory leak causing GC pressure", 0.15, "software"),
            ]
        )

        # Branch 3: External factors
        external = IntermediateEvent(
            id="external",
            description="External Factors",
            gate_type=GateType.OR,
            inputs=[
                BasicEvent("ddos", "DDoS attack", 0.05, "security"),
                BasicEvent("batch_job", "Unexpected batch job", 0.2, "operations"),
            ]
        )

        root.inputs = [resource_exhaustion, runaway, external]
        return root

    def _memory_fault_tree(self, top_event: str) -> IntermediateEvent:
        """Template fault tree for memory-related issues."""
        root = IntermediateEvent(
            id="root",
            description=top_event,
            gate_type=GateType.OR,
            inputs=[]
        )

        memory_leak = IntermediateEvent(
            id="mem_leak",
            description="Memory Leak",
            gate_type=GateType.AND,
            inputs=[
                BasicEvent("alloc_bug", "Allocation without free", 0.3, "software"),
                BasicEvent("time", "Sufficient time elapsed", 0.8, "temporal"),
            ]
        )

        high_usage = IntermediateEvent(
            id="high_usage",
            description="High Memory Usage",
            gate_type=GateType.OR,
            inputs=[
                BasicEvent("data_growth", "Data volume growth", 0.4, "data"),
                BasicEvent("cache_unbounded", "Unbounded cache growth", 0.3, "software"),
            ]
        )

        root.inputs = [memory_leak, high_usage]
        return root

    def _latency_fault_tree(self, top_event: str) -> IntermediateEvent:
        """Template fault tree for latency issues."""
        root = IntermediateEvent(
            id="root",
            description=top_event,
            gate_type=GateType.OR,
            inputs=[]
        )

        # Application layer
        app_layer = IntermediateEvent(
            id="app",
            description="Application Layer Issues",
            gate_type=GateType.OR,
            inputs=[
                BasicEvent("slow_query", "Slow database query", 0.4, "database"),
                BasicEvent("inefficient_algo", "Inefficient algorithm", 0.3, "software"),
                BasicEvent("sync_call", "Blocking synchronous call", 0.3, "software"),
            ]
        )

        # Infrastructure layer
        infra_layer = IntermediateEvent(
            id="infra",
            description="Infrastructure Layer Issues",
            gate_type=GateType.OR,
            inputs=[
                BasicEvent("network_latency", "Network latency", 0.3, "network"),
                BasicEvent("disk_io", "Disk I/O bottleneck", 0.2, "storage"),
                BasicEvent("contention", "Resource contention", 0.25, "resource"),
            ]
        )

        # External dependencies
        external = IntermediateEvent(
            id="external",
            description="External Dependency Issues",
            gate_type=GateType.OR,
            inputs=[
                BasicEvent("api_slow", "Slow external API", 0.3, "external"),
                BasicEvent("dns", "DNS resolution delay", 0.1, "network"),
            ]
        )

        root.inputs = [app_layer, infra_layer, external]
        return root

    def _error_fault_tree(self, top_event: str) -> IntermediateEvent:
        """Template fault tree for error/failure scenarios."""
        root = IntermediateEvent(
            id="root",
            description=top_event,
            gate_type=GateType.OR,
            inputs=[]
        )

        code_issue = IntermediateEvent(
            id="code",
            description="Code Issue",
            gate_type=GateType.OR,
            inputs=[
                BasicEvent("bug", "Software bug", 0.4, "software"),
                BasicEvent("null_ref", "Null reference", 0.3, "software"),
                BasicEvent("exception", "Unhandled exception", 0.3, "software"),
            ]
        )

        config_issue = IntermediateEvent(
            id="config",
            description="Configuration Issue",
            gate_type=GateType.OR,
            inputs=[
                BasicEvent("wrong_config", "Wrong configuration value", 0.3, "config"),
                BasicEvent("missing_config", "Missing configuration", 0.2, "config"),
            ]
        )

        dependency_issue = IntermediateEvent(
            id="deps",
            description="Dependency Issue",
            gate_type=GateType.OR,
            inputs=[
                BasicEvent("service_down", "Upstream service down", 0.2, "external"),
                BasicEvent("timeout", "Connection timeout", 0.3, "network"),
            ]
        )

        root.inputs = [code_issue, config_issue, dependency_issue]
        return root

    def _generic_fault_tree(self, top_event: str) -> IntermediateEvent:
        """Generic fault tree for unknown scenarios."""
        return IntermediateEvent(
            id="root",
            description=top_event,
            gate_type=GateType.OR,
            inputs=[
                BasicEvent("human_error", "Human error", 0.3, "people"),
                BasicEvent("system_failure", "System failure", 0.3, "technology"),
                BasicEvent("process_gap", "Process gap", 0.2, "process"),
                BasicEvent("external", "External factor", 0.2, "environment"),
            ]
        )

    def _calculate_probability(self, event: Any) -> float:
        """Calculate failure probability using fault tree logic."""
        if isinstance(event, BasicEvent):
            return event.probability

        if isinstance(event, IntermediateEvent):
            input_probs = [self._calculate_probability(inp) for inp in event.inputs]

            if event.gate_type == GateType.OR:
                # P(A OR B) = 1 - P(not A) * P(not B)
                return 1 - prod([1 - p for p in input_probs])

            elif event.gate_type == GateType.AND:
                # P(A AND B) = P(A) * P(B)
                return prod(input_probs)

            elif event.gate_type == GateType.VOTING:
                # K-of-N voting - simplified approximation
                k = event.vote_threshold or len(input_probs)
                return sum(input_probs) / len(input_probs) * k

        return 0.0

    def _find_minimal_cut_sets(self, event: Any, path: List[str] = None) -> List[List[str]]:
        """Find minimal cut sets (combinations that cause top event)."""
        if path is None:
            path = []

        if isinstance(event, BasicEvent):
            return [path + [event.id]]

        if isinstance(event, IntermediateEvent):
            if event.gate_type == GateType.OR:
                # Any input can cause failure - union of cut sets
                cut_sets = []
                for inp in event.inputs:
                    cut_sets.extend(self._find_minimal_cut_sets(inp, path))
                return cut_sets

            elif event.gate_type == GateType.AND:
                # All inputs must occur - cross product
                all_input_cuts = [self._find_minimal_cut_sets(inp, path) for inp in event.inputs]
                if not all_input_cuts:
                    return []

                # Start with first
                result = all_input_cuts[0]

                # Cross with remaining
                for cuts in all_input_cuts[1:]:
                    new_result = []
                    for r in result:
                        for c in cuts:
                            new_result.append(r + c)
                    result = new_result

                return result

        return []

    def _collect_basic_events(self, event: Any) -> List[BasicEvent]:
        """Collect all basic events from the tree."""
        events = []

        if isinstance(event, BasicEvent):
            events.append(event)
        elif isinstance(event, IntermediateEvent):
            for inp in event.inputs:
                events.extend(self._collect_basic_events(inp))

        return events

    def _build_tree_prompt(self, top_event: str, context: Optional[str]) -> str:
        """Build prompt for LLM-based tree construction."""
        context_str = f"\nSystem Context: {context}" if context else ""

        return f"""You are an expert in Fault Tree Analysis (FTA).

**Top Event (Undesired State):** {top_event}{context_str}

Build a fault tree that decomposes this top event into basic causes.
Use AND gates (all inputs required) and OR gates (any input sufficient).

Structure your response as a hierarchical tree:
```
TOP: {top_event}
  OR:
    - Intermediate: [cause category 1]
      AND:
        - Basic: [specific cause] (probability: 0.X)
        - Basic: [specific cause] (probability: 0.X)
    - Intermediate: [cause category 2]
      OR:
        - Basic: [specific cause] (probability: 0.X)
```

Include at least 2-3 levels of decomposition."""

    def _parse_tree_response(self, response: str, top_event: str) -> IntermediateEvent:
        """Parse LLM response into fault tree structure."""
        # Simplified parsing - would need more robust implementation
        return self._generic_fault_tree(top_event)


def prod(numbers: List[float]) -> float:
    """Calculate product of a list of numbers."""
    result = 1.0
    for n in numbers:
        result *= n
    return result


class FaultTreeTool(BaseTool):
    """
    LangChain tool for Fault Tree Analysis.
    """

    name: str = "fault_tree_analysis"
    description: str = "Perform Fault Tree Analysis (FTA) to decompose failures into root causes using Boolean logic"
    args_schema: type[BaseModel] = FaultTreeInput

    analyzer: Optional[FaultTreeAnalyzer] = None

    def __init__(self, llm=None, **kwargs):
        super().__init__(**kwargs)
        self.analyzer = FaultTreeAnalyzer(llm=llm)

    def _run(
        self,
        top_event: str,
        system_context: Optional[str] = None,
        known_failures: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute Fault Tree Analysis.

        Args:
            top_event (str): The undesired event to analyze
            system_context (Optional[str]): System description
            known_failures (Optional[List[str]]): Known failure modes

        Returns:
            Dict[str, Any]: Analysis results
        """
        tree = self.analyzer.analyze(top_event, system_context)

        # Convert to JSON-serializable format
        def event_to_dict(event) -> Dict:
            if isinstance(event, BasicEvent):
                return {
                    "type": "basic",
                    "id": event.id,
                    "description": event.description,
                    "probability": event.probability,
                    "category": event.category
                }
            elif isinstance(event, IntermediateEvent):
                return {
                    "type": "intermediate",
                    "id": event.id,
                    "description": event.description,
                    "gate_type": event.gate_type.value,
                    "inputs": [event_to_dict(inp) for inp in event.inputs],
                    "vote_threshold": event.vote_threshold
                }
            return {}

        return {
            "top_event": tree.top_event,
            "root_event": event_to_dict(tree.root),
            "basic_events": [event_to_dict(e) for e in tree.all_events],
            "calculated_probability": tree.calculated_probability,
            "minimal_cut_sets": tree.minimal_cut_sets,
            "cut_set_count": len(tree.minimal_cut_sets)
        }
