"""
Ishikawa (Fishbone) Diagram Analysis

Also known as the Cause-and-Effect Diagram or Fishbone Diagram,
this method was developed by Kaoru Ishikawa in 1968. It categorizes
potential causes into major categories to identify the root cause.

For technical/incident analysis, we use adapted categories:
- Infrastructure (硬件/基础设施)
- Software/Application (软件/应用)
- Process/Procedure (流程/规范)
- People/Skills (人员/技能)
- Data/Information (数据/信息)
- Environment (环境)
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from langchain_core.tools import BaseTool
from pydantic import BaseModel


# Standard Ishikawa categories for technical incidents
ISHIKAWA_CATEGORIES = [
    "Infrastructure",  # Hardware, network, servers, cloud services
    "Software",        # Application code, dependencies, configuration
    "Process",         # Procedures, runbooks, change management
    "People",          # Training, skills, human error, staffing
    "Data",            # Data quality, volume, corruption, schema changes
    "Environment",     # External dependencies, third-party services
]


@dataclass
class CauseCandidate:
    """Represents a potential cause in a category."""
    category: str
    description: str
    likelihood: float  # 0-1 score
    evidence: List[str] = field(default_factory=list)
    is_root_cause: bool = False


@dataclass
class FishboneResult:
    """Result of Ishikawa analysis."""
    problem: str
    causes_by_category: Dict[str, List[CauseCandidate]] = field(default_factory=dict)
    most_likely_cause: Optional[CauseCandidate] = None
    root_cause_confidence: float = 0.0
    analysis_notes: str = ""


class IshikawaInput(BaseModel):
    """Input schema for Ishikawa analysis."""
    problem: str
    symptoms: Optional[str] = None
    affected_systems: Optional[List[str]] = None
    timeline: Optional[str] = None


class IshikawaAnalyzer:
    """
    Implements the Ishikawa (Fishbone) Diagram analysis technique.

    This method systematically explores potential causes across different
    categories to identify the most likely root cause.

    Attributes:
        llm: Language model for cause identification and analysis
        categories (List[str]): Categories to analyze
        verbose (bool): Enable verbose output
    """

    def __init__(self, llm=None, categories: Optional[List[str]] = None, verbose: bool = False):
        """
        Initialize the Ishikawa analyzer.

        Args:
            llm: Language model instance
            categories (Optional[List[str]]): Categories to use (default: technical incident categories)
            verbose (bool): Enable detailed logging
        """
        self.llm = llm
        self.categories = categories or ISHIKAWA_CATEGORIES
        self.verbose = verbose

    def analyze(
        self,
        problem: str,
        symptoms: Optional[str] = None,
        affected_systems: Optional[List[str]] = None,
        timeline: Optional[str] = None
    ) -> FishboneResult:
        """
        Perform Ishikawa analysis on a problem.

        Args:
            problem (str): The problem statement
            symptoms (Optional[str]): Observed symptoms
            affected_systems (Optional[List[str]]): List of affected systems
            timeline (Optional[str]): Timeline of events

        Returns:
            FishboneResult: Analysis result with causes by category
        """
        result = FishboneResult(problem=problem)

        # Analyze each category
        for category in self.categories:
            causes = self._analyze_category(
                category, problem, symptoms, affected_systems, timeline
            )
            result.causes_by_category[category] = causes

        # Find most likely cause
        all_causes = []
        for causes in result.causes_by_category.values():
            all_causes.extend(causes)

        if all_causes:
            result.most_likely_cause = max(all_causes, key=lambda c: c.likelihood)
            result.most_likely_cause.is_root_cause = True
            result.root_cause_confidence = result.most_likely_cause.likelihood

        # Generate analysis notes
        result.analysis_notes = self._generate_analysis_notes(result)

        return result

    def _analyze_category(
        self,
        category: str,
        problem: str,
        symptoms: Optional[str],
        affected_systems: Optional[List[str]],
        timeline: Optional[str]
    ) -> List[CauseCandidate]:
        """Analyze potential causes within a category."""
        if not self.llm:
            return self._analyze_category_simple(category, problem)

        prompt = self._build_category_prompt(
            category, problem, symptoms, affected_systems, timeline
        )

        try:
            response = self.llm.invoke(prompt)
            return self._parse_category_response(category, response.content)
        except Exception:
            return self._analyze_category_simple(category, problem)

    def _analyze_category_simple(self, category: str, problem: str) -> List[CauseCandidate]:
        """Simple analysis without LLM - returns template causes."""
        template_causes = {
            "Infrastructure": [
                ("Server/VM resource exhaustion", 0.5),
                ("Network connectivity issues", 0.4),
                ("Load balancer misconfiguration", 0.3),
            ],
            "Software": [
                ("Bug in application code", 0.4),
                ("Memory leak", 0.5),
                ("Dependency version conflict", 0.3),
            ],
            "Process": [
                ("Insufficient monitoring/alerting", 0.4),
                ("Missing runbook for incident", 0.3),
                ("Change management gap", 0.5),
            ],
            "People": [
                ("Insufficient training", 0.3),
                ("Human error during operation", 0.4),
                ("Staff shortage", 0.2),
            ],
            "Data": [
                ("Data volume spike", 0.4),
                ("Corrupted data", 0.3),
                ("Schema change issue", 0.3),
            ],
            "Environment": [
                ("Third-party API failure", 0.4),
                ("Cloud provider issue", 0.3),
                ("External dependency change", 0.3),
            ],
        }

        causes = []
        for desc, likelihood in template_causes.get(category, []):
            causes.append(CauseCandidate(
                category=category,
                description=desc,
                likelihood=likelihood
            ))

        return causes

    def _build_category_prompt(
        self,
        category: str,
        problem: str,
        symptoms: Optional[str],
        affected_systems: Optional[List[str]],
        timeline: Optional[str]
    ) -> str:
        """Build prompt for category analysis."""
        prompt = f"""You are conducting an Ishikawa (Fishbone) root cause analysis.

**Problem:** {problem}

**Category to analyze:** {category}

"""
        if symptoms:
            prompt += f"**Symptoms:** {symptoms}\n\n"

        if affected_systems:
            prompt += f"**Affected Systems:** {', '.join(affected_systems)}\n\n"

        if timeline:
            prompt += f"**Timeline:** {timeline}\n\n"

        prompt += f"""Based on the problem and available evidence, identify 3-5 potential causes
in the '{category}' category that could explain this incident.

For each potential cause, provide:
1. A brief description
2. Likelihood score (0.0-1.0)
3. Supporting evidence or reasoning

Format your response as:
```
CAUSE: <description>
LIKELIHOOD: <0.0-1.0>
EVIDENCE: <supporting evidence>

CAUSE: <description>
...
```
"""
        return prompt

    def _parse_category_response(self, category: str, response: str) -> List[CauseCandidate]:
        """Parse LLM response into CauseCandidate objects."""
        causes = []

        # Simple parsing - split by CAUSE:
        parts = response.split("CAUSE:")
        for part in parts[1:]:  # Skip first empty part
            lines = part.strip().split("\n")

            description = lines[0].strip() if lines else ""
            likelihood = 0.5
            evidence = []

            for line in lines:
                if "LIKELIHOOD:" in line.upper():
                    try:
                        likelihood = float(line.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass
                if "EVIDENCE:" in line.upper():
                    evidence.append(line.split(":")[1].strip())

            causes.append(CauseCandidate(
                category=category,
                description=description,
                likelihood=min(max(likelihood, 0), 1),
                evidence=evidence
            ))

        return causes

    def _generate_analysis_notes(self, result: FishboneResult) -> str:
        """Generate summary analysis notes."""
        if not result.most_likely_cause:
            return "Unable to determine most likely cause."

        notes = f"Analyzed {len(self.categories)} categories. "
        notes += f"Most likely root cause: {result.most_likely_cause.description} "
        notes += f"in category '{result.most_likely_cause.category}' "
        notes += f"with {result.most_likely_cause.likelihood:.0%} confidence."

        return notes


class IshikawaTool(BaseTool):
    """
    LangChain tool for Ishikawa (Fishbone) analysis.
    """

    name: str = "ishikawa_analysis"
    description: str = "Perform Ishikawa/Fishbone diagram analysis to categorize and identify root causes"
    args_schema: type[BaseModel] = IshikawaInput

    analyzer: Optional[IshikawaAnalyzer] = None

    def __init__(self, llm=None, **kwargs):
        super().__init__(**kwargs)
        self.analyzer = IshikawaAnalyzer(llm=llm)

    def _run(
        self,
        problem: str,
        symptoms: Optional[str] = None,
        affected_systems: Optional[List[str]] = None,
        timeline: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Ishikawa analysis.

        Args:
            problem (str): The problem to analyze
            symptoms (Optional[str]): Observed symptoms
            affected_systems (Optional[List[str]]): Affected systems
            timeline (Optional[str]): Timeline of events

        Returns:
            Dict[str, Any]: Analysis results
        """
        result = self.analyzer.analyze(problem, symptoms, affected_systems, timeline)

        # Convert to JSON-serializable format
        causes_by_category = {}
        for category, causes in result.causes_by_category.items():
            causes_by_category[category] = [
                {
                    "description": c.description,
                    "likelihood": c.likelihood,
                    "evidence": c.evidence,
                    "is_root_cause": c.is_root_cause
                }
                for c in causes
            ]

        return {
            "problem": result.problem,
            "causes_by_category": causes_by_category,
            "most_likely_cause": {
                "category": result.most_likely_cause.category if result.most_likely_cause else None,
                "description": result.most_likely_cause.description if result.most_likely_cause else None,
                "likelihood": result.most_likely_cause.likelihood if result.most_likely_cause else 0,
            } if result.most_likely_cause else None,
            "root_cause_confidence": result.root_cause_confidence,
            "analysis_notes": result.analysis_notes
        }
