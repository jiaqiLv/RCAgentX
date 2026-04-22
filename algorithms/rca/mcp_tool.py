"""
RCA MCP Tool - Unified Root Cause Analysis Tool

Provides a unified interface to multiple classic RCA algorithms:
- 5 Whys Analysis
- Ishikawa (Fishbone) Diagram
- Fault Tree Analysis (FTA)
- Change Analysis
- Event Correlation

This tool can be used as an MCP (Model Context Protocol) tool for
LLM-based agents to perform structured root cause analysis.
"""

from typing import Any, Dict, List, Optional
from enum import Enum

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from algorithms.rca.five_whys import FiveWhysAnalyzer, FiveWhysTool
from algorithms.rca.ishikawa import IshikawaAnalyzer, IshikawaTool
from algorithms.rca.fault_tree import FaultTreeAnalyzer, FaultTreeTool
from algorithms.rca.change_analysis import ChangeAnalyzer, ChangeTool
from algorithms.rca.event_correlation import EventCorrelator, EventCorrelatorTool


class RCAMethod(Enum):
    """Available RCA methods."""
    FIVE_WHYS = "5_whys"
    ISHIKAWA = "ishikawa"
    FAULT_TREE = "fault_tree"
    CHANGE_ANALYSIS = "change_analysis"
    EVENT_CORRELATION = "event_correlation"
    AUTO = "auto"  # Automatically select method


class RCAInput(BaseModel):
    """Input schema for RCA analysis."""
    problem: str = Field(..., description="The problem or incident to analyze")
    method: str = Field(default="auto", description="RCA method to use")
    context: Optional[str] = Field(None, description="Additional context about the incident")
    symptoms: Optional[str] = Field(None, description="Observed symptoms")
    affected_services: Optional[List[str]] = Field(None, description="List of affected services")
    timeline: Optional[str] = Field(None, description="Timeline of events")


class RCAMCPTool(BaseTool):
    """
    Unified Root Cause Analysis MCP Tool.

    This tool provides access to multiple classic RCA algorithms
    through a single interface. It can automatically select the
    appropriate method based on the incident characteristics.

    Attributes:
        llm: Language model instance
        verbose (bool): Enable verbose output

    Example:
        ```python
        rca_tool = RCAMCPTool(llm=llm)

        result = rca_tool.run(
            problem="API service experiencing high latency",
            method="auto",
            context="CPU usage spiked after deployment"
        )
        ```
    """

    name: str = "root_cause_analysis"
    description: str = """
    Perform root cause analysis using classic RCA methods.

    Available methods:
    - 5_whys: Ask 'why' repeatedly to drill down to root cause
    - ishikawa: Fishbone diagram categorizing causes (Infrastructure, Software, Process, People, Data, Environment)
    - fault_tree: Top-down Boolean logic analysis
    - change_analysis: Analyze recent changes to find the cause
    - event_correlation: Correlate multiple events to find the primary cause
    - auto: Automatically select the best method

    Use this when you need to systematically identify the root cause of an incident.
    """
    args_schema: type[BaseModel] = RCAInput

    llm: Optional[Any] = None
    verbose: bool = False

    _tools: Dict[str, BaseTool] = {}

    def __init__(self, llm=None, verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.verbose = verbose

        # Initialize all RCA tools
        self._tools = {
            RCAMethod.FIVE_WHYS.value: FiveWhysTool(llm=llm, max_whys=5),
            RCAMethod.ISHIKAWA.value: IshikawaTool(llm=llm),
            RCAMethod.FAULT_TREE.value: FaultTreeTool(llm=llm),
            RCAMethod.CHANGE_ANALYSIS.value: ChangeTool(llm=llm),
            RCAMethod.EVENT_CORRELATION.value: EventCorrelatorTool(llm=llm),
        }

    def _run(
        self,
        problem: str,
        method: str = "auto",
        context: Optional[str] = None,
        symptoms: Optional[str] = None,
        affected_services: Optional[List[str]] = None,
        timeline: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute root cause analysis.

        Args:
            problem (str): The problem to analyze
            method (str): RCA method to use (default: auto)
            context (Optional[str]): Additional context
            symptoms (Optional[str]): Observed symptoms
            affected_services (Optional[List[str]]): Affected services
            timeline (Optional[str]): Timeline of events

        Returns:
            Dict[str, Any]: Analysis results
        """
        # Select method
        if method == RCAMethod.AUTO.value or method == "auto":
            method = self._select_method(problem, context, symptoms)

        # Get the appropriate tool
        tool = self._tools.get(method)

        if not tool:
            return {
                "error": f"Unknown RCA method: {method}",
                "available_methods": list(self._tools.keys())
            }

        # Execute the analysis
        try:
            result = self._execute_tool(tool, method, problem, context, symptoms, affected_services, timeline)
            result["method_used"] = method
            return result
        except Exception as e:
            return {
                "error": str(e),
                "method_attempted": method,
                "problem": problem
            }

    def _select_method(
        self,
        problem: str,
        context: Optional[str],
        symptoms: Optional[str]
    ) -> str:
        """
        Automatically select the best RCA method based on problem characteristics.

        Selection logic:
        - If problem involves timing/sequence -> event_correlation
        - If problem mentions deployment/change -> change_analysis
        - If problem is complex with multiple symptoms -> fault_tree
        - If problem is specific and focused -> 5_whys
        - Default -> ishikawa (comprehensive)
        """
        problem_lower = problem.lower()
        context_lower = (context or "").lower()

        # Check for timing/sequence indicators
        timing_keywords = ["after", "then", "sequence", "cascade", "chain", "correlation"]
        if any(kw in problem_lower or kw in context_lower for kw in timing_keywords):
            return RCAMethod.EVENT_CORRELATION.value

        # Check for change/deployment indicators
        change_keywords = ["deploy", "release", "update", "change", "modify", "upgrade", "migration"]
        if any(kw in problem_lower or kw in context_lower for kw in change_keywords):
            return RCAMethod.CHANGE_ANALYSIS.value

        # Check for complex/multi-symptom indicators
        complex_keywords = ["multiple", "complex", "system", "interconnected", "cascade"]
        if any(kw in problem_lower or kw in context_lower for kw in complex_keywords):
            return RCAMethod.FAULT_TREE.value

        # Check for focused/specific problem
        if len(problem.split()) < 15 and "and" not in problem_lower:
            return RCAMethod.FIVE_WHYS.value

        # Default to comprehensive Ishikawa analysis
        return RCAMethod.ISHIKAWA.value

    def _execute_tool(
        self,
        tool: BaseTool,
        method: str,
        problem: str,
        context: Optional[str],
        symptoms: Optional[str],
        affected_services: Optional[List[str]],
        timeline: Optional[str]
    ) -> Dict[str, Any]:
        """Execute the selected RCA tool with appropriate parameters."""

        if method == RCAMethod.FIVE_WHYS.value:
            return tool._run(problem=problem, context=context)

        elif method == RCAMethod.ISHIKAWA.value:
            return tool._run(
                problem=problem,
                symptoms=symptoms,
                affected_services=affected_services,
                timeline=timeline
            )

        elif method == RCAMethod.FAULT_TREE.value:
            return tool._run(
                top_event=problem,
                system_context=context
            )

        elif method == RCAMethod.CHANGE_ANALYSIS.value:
            # Parse timeline if available
            from datetime import datetime
            incident_time = datetime.now()
            if timeline:
                try:
                    incident_time = datetime.fromisoformat(timeline.split("incident:")[1].strip())
                except (ValueError, IndexError):
                    pass

            return tool._run(
                incident_time=incident_time.isoformat(),
                affected_services=affected_services
            )

        elif method == RCAMethod.EVENT_CORRELATION.value:
            return tool._run()

        return {}


class RCAEnsembleTool(BaseTool):
    """
    Ensemble RCA Tool that runs multiple methods and combines results.

    This provides more comprehensive analysis by using multiple
    RCA techniques and synthesizing their findings.
    """

    name: str = "ensemble_root_cause_analysis"
    description: str = """
    Run multiple RCA methods and combine results for comprehensive analysis.

    This tool executes 5_whys, ishikawa, and fault_tree analysis,
    then synthesizes the findings into a unified root cause assessment.
    """
    args_schema: type[BaseModel] = RCAInput

    llm: Optional[Any] = None
    verbose: bool = False

    def __init__(self, llm=None, verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.verbose = verbose

        self.tools = {
            "5_whys": FiveWhysTool(llm=llm),
            "ishikawa": IshikawaTool(llm=llm),
            "fault_tree": FaultTreeTool(llm=llm),
        }

    def _run(
        self,
        problem: str,
        method: str = "auto",
        context: Optional[str] = None,
        symptoms: Optional[str] = None,
        affected_services: Optional[List[str]] = None,
        timeline: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute ensemble RCA analysis.

        Args:
            problem (str): The problem to analyze
            method (str): Ignored for ensemble (runs all methods)
            context (Optional[str]): Additional context
            symptoms (Optional[str]): Observed symptoms
            affected_services (Optional[List[str]]): Affected services
            timeline (Optional[str]): Timeline of events

        Returns:
            Dict[str, Any]: Combined analysis results
        """
        results = {}

        # Run each RCA method
        for name, tool in self.tools.items():
            try:
                if name == "5_whys":
                    results[name] = tool._run(problem=problem, context=context)
                elif name == "ishikawa":
                    results[name] = tool._run(
                        problem=problem,
                        symptoms=symptoms,
                        affected_services=affected_services
                    )
                elif name == "fault_tree":
                    results[name] = tool._run(
                        top_event=problem,
                        system_context=context
                    )
            except Exception as e:
                results[name] = {"error": str(e)}

        # Extract root causes from each method
        root_causes = []

        if "5_whys" in results and "root_cause" in results["5_whys"]:
            root_causes.append({
                "source": "5_whys",
                "cause": results["5_whys"]["root_cause"],
                "confidence": results["5_whys"].get("confidence", 0)
            })

        if "ishikawa" in results and results["ishikawa"].get("most_likely_cause"):
            mlc = results["ishikawa"]["most_likely_cause"]
            root_causes.append({
                "source": "ishikawa",
                "cause": mlc.get("description"),
                "confidence": mlc.get("likelihood", 0)
            })

        if "fault_tree" in results and results["fault_tree"].get("basic_events"):
            # Get highest probability basic event
            events = sorted(
                results["fault_tree"]["basic_events"],
                key=lambda x: x.get("probability", 0),
                reverse=True
            )
            if events:
                root_causes.append({
                    "source": "fault_tree",
                    "cause": events[0].get("description"),
                    "confidence": events[0].get("probability", 0)
                })

        # Synthesize results
        synthesis = self._synthesize_results(root_causes, problem, context)

        return {
            "individual_results": results,
            "root_causes_by_method": root_causes,
            "synthesis": synthesis,
            "consensus_root_cause": self._find_consensus(root_causes)
        }

    def _synthesize_results(
        self,
        root_causes: List[Dict],
        problem: str,
        context: Optional[str]
    ) -> str:
        """Synthesize findings from multiple RCA methods."""
        if not root_causes:
            return "Unable to determine root cause."

        # If LLM available, use it for synthesis
        if self.llm:
            prompt = self._build_synthesis_prompt(root_causes, problem, context)
            try:
                response = self.llm.invoke(prompt)
                return response.content
            except Exception:
                pass

        # Fallback: simple concatenation
        causes = [rc.get("cause") for rc in root_causes if rc.get("cause")]
        return f"Analysis identified {len(causes)} potential root causes: {'; '.join(causes)}"

    def _build_synthesis_prompt(
        self,
        root_causes: List[Dict],
        problem: str,
        context: Optional[str]
    ) -> str:
        """Build prompt for LLM-based synthesis."""
        causes_str = "\n".join([
            f"- {rc['source']}: {rc.get('cause', 'N/A')} (confidence: {rc.get('confidence', 0):.2f})"
            for rc in root_causes
        ])

        return f"""You are analyzing root cause analysis results from multiple methods.

**Problem:** {problem}
**Context:** {context or 'N/A'}

**Results from different RCA methods:**
{causes_str}

Synthesize these findings into a single, coherent root cause statement.
If the methods agree, state the consensus. If they differ, explain the
most likely cause and any alternative hypotheses.

Provide a concise root cause summary:"""

    def _find_consensus(self, root_causes: List[Dict]) -> Optional[Dict]:
        """Find the consensus root cause if methods agree."""
        if not root_causes:
            return None

        # Simple heuristic: highest confidence cause
        valid_causes = [rc for rc in root_causes if rc.get("cause")]
        if not valid_causes:
            return None

        return max(valid_causes, key=lambda x: x.get("confidence", 0))
