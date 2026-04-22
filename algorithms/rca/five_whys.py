"""
5 Whys Analysis Algorithm

A classic root cause analysis technique that involves asking "why"
repeatedively (typically 5 times) to drill down from the symptom
to the underlying root cause.

This implementation uses an LLM to simulate the 5 Whys questioning process
and identify the chain of causation.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from langchain_core.tools import BaseTool
from pydantic import BaseModel


@dataclass
class WhyChain:
    """Represents a chain of why questions and answers."""
    symptom: str
    why_questions: List[str] = field(default_factory=list)
    why_answers: List[str] = field(default_factory=list)
    root_cause: str = ""
    confidence: float = 0.0
    category: str = "unknown"


class FiveWhysInput(BaseModel):
    """Input schema for Five Whys analysis."""
    problem: str
    context: Optional[str] = None


class FiveWhysAnalyzer:
    """
    Implements the 5 Whys root cause analysis technique.

    The 5 Whys method was developed by Sakichi Toyoda for Toyota Motor Corporation.
    It's used to determine the root cause of a problem by repeatedly asking
    "Why?" (typically 5 times) to drill down through layers of symptoms.

    Attributes:
        llm: Language model for generating why questions and analyzing answers
        max_whys (int): Maximum depth of why questioning (default: 5)
        verbose (bool): Enable verbose output

    Example:
        ```python
        analyzer = FiveWhysAnalyzer(llm=llm)

        result = analyzer.analyze(
            problem="API service is experiencing high latency",
            context="CPU usage spiked to 95%, memory at 80%"
        )

        print(f"Root cause: {result.root_cause}")
        print(f"Confidence: {result.confidence}")
        ```
    """

    def __init__(self, llm=None, max_whys: int = 5, verbose: bool = False):
        """
        Initialize the Five Whys analyzer.

        Args:
            llm: Language model instance for analysis
            max_whys (int): Maximum number of why iterations
            verbose (bool): Enable detailed logging
        """
        self.llm = llm
        self.max_whys = max_whys
        self.verbose = verbose

    def analyze(self, problem: str, context: Optional[str] = None) -> WhyChain:
        """
        Perform Five Whys analysis on a problem.

        Args:
            problem (str): The problem or symptom to analyze
            context (Optional[str]): Additional context about the incident

        Returns:
            WhyChain: Chain of why questions leading to root cause
        """
        chain = WhyChain(symptom=problem)

        current_question = problem
        if context:
            current_question = f"{problem} (Context: {context})"

        for i in range(self.max_whys):
            # Generate the next why question
            why_question = self._generate_why_question(chain, i)
            chain.why_questions.append(why_question)

            # Generate answer to the why question
            why_answer = self._generate_why_answer(why_question, context)
            chain.why_answers.append(why_answer)

            # Check if we've reached a fundamental cause
            if self._is_root_cause(why_answer, i):
                chain.root_cause = why_answer
                chain.confidence = self._calculate_confidence(i + 1)
                chain.category = self._categorize_cause(why_answer)
                break

            # Continue with the answer as the new problem
            current_question = why_answer

        # If we haven't found a root cause, use the last answer
        if not chain.root_cause and chain.why_answers:
            chain.root_cause = chain.why_answers[-1]
            chain.confidence = self._calculate_confidence(len(chain.why_questions))
            chain.category = self._categorize_cause(chain.root_cause)

        return chain

    def _generate_why_question(self, chain: WhyChain, depth: int) -> str:
        """Generate the next why question based on current understanding."""
        if not self.llm:
            # Fallback without LLM
            if depth == 0:
                return f"Why did this happen: {chain.symptom}?"
            return f"Why did '{chain.why_answers[-1]}' occur?"

        prompt = self._build_why_question_prompt(chain, depth)
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            return f"Why did this happen? (Depth {depth + 1})"

    def _generate_why_answer(self, question: str, context: Optional[str]) -> str:
        """Generate an answer to the why question."""
        if not self.llm:
            return "Unknown - requires human analysis"

        prompt = self._build_why_answer_prompt(question, context)
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            return "Unknown"

    def _is_root_cause(self, answer: str, depth: int) -> bool:
        """Determine if we've reached a fundamental root cause."""
        # Root cause indicators
        root_indicators = [
            "because", "due to", "caused by", "result of",
            "lack of", "missing", "failed to", "unable to",
            "design", "process", "policy", "system",
            "human error", "training", "communication"
        ]

        answer_lower = answer.lower()

        # Check if answer contains root cause indicators
        has_indicator = any(ind in answer_lower for ind in root_indicators)

        # Deeper why questions more likely to be root causes
        min_depth = 3

        return has_indicator and depth >= min_depth

    def _calculate_confidence(self, depth: int) -> float:
        """Calculate confidence based on analysis depth."""
        # More why iterations typically mean deeper analysis
        base_confidence = min(0.5 + (depth * 0.1), 0.95)
        return base_confidence

    def _categorize_cause(self, cause: str) -> str:
        """Categorize the root cause into standard categories."""
        categories = {
            "people": ["training", "skill", "knowledge", "human error", "staffing"],
            "process": ["procedure", "policy", "workflow", "process", "documentation"],
            "technology": ["system", "software", "hardware", "tool", "automation", "bug"],
            "environment": ["environment", "temperature", "humidity", "workspace"],
            "management": ["management", "leadership", "communication", "oversight"],
            "resource": ["budget", "time", "resource", "capacity", "load"]
        }

        cause_lower = cause.lower()
        for category, keywords in categories.items():
            if any(kw in cause_lower for kw in keywords):
                return category

        return "unknown"

    def _build_why_question_prompt(self, chain: WhyChain, depth: int) -> str:
        """Build prompt for generating why question."""
        return f"""You are conducting a Five Whys root cause analysis.

Problem/Symptom: {chain.symptom}

Previous Analysis:
{self._format_chain(chain)}

Generate the next "Why" question (question #{depth + 1}) to drill deeper
into the root cause. Be specific and focus on understanding causation.

Why Question #""" + str(depth + 1) + ":"

    def _build_why_answer_prompt(self, question: str, context: Optional[str]) -> str:
        """Build prompt for generating why answer."""
        context_str = f"\nAdditional Context: {context}" if context else ""

        return f"""You are an expert investigating a technical incident.

Question: {question}{context_str}

Provide a factual, specific answer based on typical incident patterns.
Focus on technical causes and be concise.

Answer:"""

    def _format_chain(self, chain: WhyChain) -> str:
        """Format the why chain for display."""
        if not chain.why_questions:
            return "  (No previous analysis)"

        lines = []
        for i, (q, a) in enumerate(zip(chain.why_questions, chain.why_answers), 1):
            lines.append(f"  Why {i}: {q}")
            lines.append(f"  Answer: {a}")

        return "\n".join(lines)


class FiveWhysTool(BaseTool):
    """
    LangChain tool for Five Whys root cause analysis.

    This tool wraps the FiveWhysAnalyzer for use with LangChain agents.
    """

    name: str = "five_whys_analysis"
    description: str = "Perform Five Whys root cause analysis on a problem or incident"
    args_schema: type[BaseModel] = FiveWhysInput

    analyzer: Optional[FiveWhysAnalyzer] = None

    def __init__(self, llm=None, max_whys: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.analyzer = FiveWhysAnalyzer(llm=llm, max_whys=max_whys)

    def _run(self, problem: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute Five Whys analysis.

        Args:
            problem (str): The problem to analyze
            context (Optional[str]): Additional context

        Returns:
            Dict[str, Any]: Analysis results
        """
        result = self.analyzer.analyze(problem, context)
        return {
            "symptom": result.symptom,
            "why_chain": list(zip(result.why_questions, result.why_answers)),
            "root_cause": result.root_cause,
            "confidence": result.confidence,
            "category": result.category,
            "depth": len(result.why_questions)
        }
