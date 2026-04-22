"""
Diagnosis Agent - Root Cause Analysis

The Diagnosis Agent performs deep root cause analysis on detected
anomalies. It combines causal inference algorithms with LLM-powered
reasoning to identify the underlying cause of failures and generate
actionable diagnostic reports.

Key responsibilities:
1. Multi-modal root cause localization
2. Advanced causal inference algorithms
3. Fault propagation path construction
4. Explainable diagnostic reports with evidence chains
"""

from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from pydantic import ConfigDict

from agents.base import BaseAgent
from memory.shared_state import RootCauseAnalysis, IncidentStatus

# Import classic RCA algorithms
try:
    from algorithms.rca import (
        RCAMCPTool,
        RCAEnsembleTool,
        FiveWhysAnalyzer,
        IshikawaAnalyzer,
    )
    RCA_AVAILABLE = True
except ImportError:
    RCA_AVAILABLE = False


class DiagnosisAgent(BaseAgent):
    """
    Root cause analysis and diagnosis agent.

    This agent analyzes detected anomalies to identify the underlying
    root cause, construct fault propagation paths, and generate
    explainable diagnostic reports.

    Attributes:
        name (str): Always "diagnosis"
        description (str): Description of diagnosis capabilities
        llm (ChatOpenAI): Language model for causal reasoning
        verbose (bool): Enable verbose logging

    Example:
        ```python
        diagnosis = DiagnosisAgent(
            llm=ChatOpenAI(model="gpt-4"),
            verbose=True
        )

        result = diagnosis.execute(state)
        rca = result["diagnosis"]
        print(f"Root cause: {rca.root_causes[0]['cause']}")
        ```
    """

    name: str = "diagnosis"
    description: str = "Performs root cause analysis using classic RCA algorithms and LLM reasoning"
    llm: Optional[ChatOpenAI] = None
    rca_tool: Optional[Any] = None  # RCAMCPTool instance
    rca_analyzers: Dict[str, Any] = {}  # Individual analyzers

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        """Initialize DiagnosisAgent with RCA tools."""
        super().__init__(**kwargs)

        # Initialize classic RCA tools if LLM is available
        if RCA_AVAILABLE and self.llm:
            self.rca_tool = RCAMCPTool(llm=self.llm, verbose=False)
            self.rca_analyzers = {
                "5_whys": FiveWhysAnalyzer(llm=self.llm, max_whys=5),
                "ishikawa": IshikawaAnalyzer(llm=self.llm),
            }

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute root cause analysis on detected anomaly.

        Analyzes the anomaly event and observability data to identify
        the root cause, construct propagation paths, and generate
        evidence chains.

        Args:
            state (Dict[str, Any]): Current workflow state containing
                anomaly event and observability data

        Returns:
            Dict[str, Any]: Updated state with root cause analysis

        Example:
            ```python
            state = {
                "anomaly": anomaly_event,
                "observability": obs_data
            }
            result = diagnosis.execute(state)
            diagnosis_result = result["diagnosis"]
            ```
        """
        self.log("Starting root cause analysis")

        anomaly = state.get("anomaly")
        obs_data = state.get("observability")

        if not anomaly or anomaly.type == "no_issue":
            self.log("No anomaly to diagnose")
            state["diagnosis"] = RootCauseAnalysis(
                root_causes=[{"cause": "No anomaly detected", "confidence": 1.0}],
                confidence=1.0
            )
            return state

        if not obs_data:
            self.log("No observability data for diagnosis")
            state["errors"] = state.get("errors", [])
            state["errors"].append("No observability data for diagnosis")
            return state

        # Perform root cause analysis
        root_causes = self._analyze_root_causes(anomaly, obs_data, state)

        # Build propagation path
        propagation_path = self._build_propagation_path(root_causes, obs_data)

        # Generate evidence chain
        evidence_chain = self._generate_evidence_chain(root_causes, anomaly)

        # Calculate confidence
        confidence = self._calculate_confidence(root_causes, anomaly)

        # Determine impact scope
        impact_scope = self._assess_impact(anomaly, obs_data)

        # Create diagnosis result
        diagnosis_result = RootCauseAnalysis(
            root_causes=root_causes,
            propagation_path=propagation_path,
            confidence=confidence,
            evidence_chain=evidence_chain,
            impact_scope=impact_scope
        )

        state["diagnosis"] = diagnosis_result
        state["status"] = IncidentStatus.DIAGNOSING.value

        self.log(f"Diagnosis complete: {len(root_causes)} root causes identified")
        return state

    def _analyze_root_causes(
        self,
        anomaly: Any,
        obs_data: Any,
        state: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze and identify potential root causes.

        Priority order:
        1. Classic RCA algorithms (5 Whys, Ishikawa) with LLM
        2. Rule-based analysis for known patterns
        3. Fallback LLM reasoning

        Args:
            anomaly: Detected anomaly event
            obs_data: Observability data
            state: Current workflow state (may contain user-selected RCA method)

        Returns:
            List[Dict[str, Any]]: List of identified root causes
        """
        root_causes = []

        # Priority 1: Run classic RCA algorithms with LLM (most sophisticated)
        if self.rca_tool and hasattr(anomaly, 'type') and anomaly.type != "no_issue":
            self.log("Running classic RCA algorithms...")
            rca_method = state.get("rca_method") if state else None
            rca_results = self._run_classic_rca(anomaly, obs_data, rca_method)
            if rca_results:
                self.log(f"RCA algorithms identified {len(rca_results)} root cause(s)")
                root_causes.extend(rca_results)

        # Priority 2: Rule-based analysis for specific anomaly types
        anomaly_type = anomaly.type if hasattr(anomaly, 'type') else ""
        evidence = anomaly.evidence if hasattr(anomaly, 'evidence') else {}

        rule_based_causes = []

        # CPU-related anomalies
        if "cpu" in anomaly_type.lower():
            rule_based_causes.extend(self._analyze_cpu_causes(evidence, obs_data))

        # Memory-related anomalies
        elif "memory" in anomaly_type.lower():
            rule_based_causes.extend(self._analyze_memory_causes(evidence, obs_data))

        # Error rate anomalies
        elif "error" in anomaly_type.lower():
            rule_based_causes.extend(self._analyze_error_causes(evidence, obs_data))

        # Latency anomalies
        elif "latency" in anomaly_type.lower():
            rule_based_causes.extend(self._analyze_latency_causes(evidence, obs_data))

        # Log pattern anomalies
        elif "log" in anomaly_type.lower():
            rule_based_causes.extend(self._analyze_log_pattern_causes(evidence, obs_data))

        # Only add rule-based causes if RCA didn't find anything
        if not root_causes and rule_based_causes:
            self.log(f"Using rule-based analysis: {len(rule_based_causes)} cause(s)")
            root_causes.extend(rule_based_causes)

        # Priority 3: Fallback LLM analysis if no causes found
        if not root_causes and self.llm:
            self.log("Using LLM-based analysis as fallback...")
            root_causes = self._llm_analyze(anomaly, obs_data)

        # Deduplicate and merge root causes
        merged = self._merge_root_causes(root_causes)
        self.log(f"Final root causes after merge: {len(merged)}")
        return merged

    def _analyze_cpu_causes(
        self,
        evidence: Dict[str, Any],
        obs_data: Any
    ) -> List[Dict[str, Any]]:
        """Analyze CPU-related root causes"""
        causes = []
        metric_anomalies = evidence.get("metric_anomalies", [])

        for anomaly in metric_anomalies:
            if "cpu" in anomaly.get("type", ""):
                labels = anomaly.get("labels", {})
                pod = labels.get("pod", "unknown")
                service = labels.get("service", "unknown")

                causes.append({
                    "entity": pod,
                    "service": service,
                    "cause": "High CPU utilization",
                    "category": "resource_exhaustion",
                    "confidence": 0.8,
                    "evidence": f"CPU usage at {anomaly.get('value', 0):.1%}"
                })

        return causes

    def _analyze_memory_causes(
        self,
        evidence: Dict[str, Any],
        obs_data: Any
    ) -> List[Dict[str, Any]]:
        """Analyze memory-related root causes"""
        causes = []
        metric_anomalies = evidence.get("metric_anomalies", [])

        for anomaly in metric_anomalies:
            if "memory" in anomaly.get("type", ""):
                labels = anomaly.get("labels", {})
                pod = labels.get("pod", "unknown")

                causes.append({
                    "entity": pod,
                    "cause": "High memory utilization",
                    "category": "resource_exhaustion",
                    "confidence": 0.75,
                    "evidence": f"Memory usage exceeds threshold"
                })

        return causes

    def _analyze_error_causes(
        self,
        evidence: Dict[str, Any],
        obs_data: Any
    ) -> List[Dict[str, Any]]:
        """Analyze error rate root causes"""
        causes = []
        log_anomalies = evidence.get("log_anomalies", [])
        metric_anomalies = evidence.get("metric_anomalies", [])

        # Check for specific error patterns
        for log_anomaly in log_anomalies:
            pattern = log_anomaly.get("pattern", "")

            if "exception" in pattern:
                causes.append({
                    "entity": "application",
                    "cause": "Application exceptions detected",
                    "category": "application_error",
                    "confidence": 0.7,
                    "evidence": f"{log_anomaly.get('count', 0)} exceptions logged"
                })

            if "timeout" in pattern:
                causes.append({
                    "entity": "network/dependencies",
                    "cause": "Connection timeouts",
                    "category": "network_issue",
                    "confidence": 0.65,
                    "evidence": f"{log_anomaly.get('count', 0)} timeout errors"
                })

            if "oom" in pattern:
                causes.append({
                    "entity": "application",
                    "cause": "Out of memory errors",
                    "category": "resource_exhaustion",
                    "confidence": 0.85,
                    "evidence": "OOM killer invoked"
                })

        return causes

    def _analyze_latency_causes(
        self,
        evidence: Dict[str, Any],
        obs_data: Any
    ) -> List[Dict[str, Any]]:
        """Analyze latency-related root causes"""
        causes = []
        metric_anomalies = evidence.get("metric_anomalies", [])

        for anomaly in metric_anomalies:
            if "latency" in anomaly.get("type", ""):
                causes.append({
                    "entity": "service",
                    "cause": "Elevated response latency",
                    "category": "performance_degradation",
                    "confidence": 0.6,
                    "evidence": f"P99 latency at {anomaly.get('value', 0):.2f}s"
                })

        return causes

    def _analyze_log_pattern_causes(
        self,
        evidence: Dict[str, Any],
        obs_data: Any
    ) -> List[Dict[str, Any]]:
        """Analyze log pattern root causes"""
        causes = []
        log_anomalies = evidence.get("log_anomalies", [])

        for log_anomaly in log_anomalies:
            pattern = log_anomaly.get("pattern", "")
            count = log_anomaly.get("count", 0)

            causes.append({
                "entity": "application",
                "cause": f"Abnormal log pattern: {pattern}",
                "category": "application_error",
                "confidence": min(0.5 + (count / 100), 0.8),
                "evidence": f"Pattern '{pattern}' appeared {count} times"
            })

        return causes

    def _llm_analyze(
        self,
        anomaly: Any,
        obs_data: Any
    ) -> List[Dict[str, Any]]:
        """
        Use LLM for complex root cause analysis.

        When rule-based analysis is insufficient, uses the language
        model to perform causal reasoning.

        Args:
            anomaly: Detected anomaly
            obs_data: Observability data

        Returns:
            List[Dict[str, Any]]: LLM-derived root causes
        """
        # Prepare context for LLM
        anomaly_summary = f"""
        Anomaly Type: {anomaly.type}
        Severity: {anomaly.severity}
        Confidence: {anomaly.confidence}
        Affected Entities: {anomaly.affected_entities}
        Evidence: {anomaly.evidence}
        """

        log_summary = "\n".join([
            f"- {log.get('line', '')[:200]}"
            for log in obs_data.logs[:10]
        ])

        prompt = f"""
        Analyze the following incident data and identify the root cause:

        ## Anomaly Information
        {anomaly_summary}

        ## Recent Error Logs
        {log_summary}

        Please identify:
        1. The most likely root cause entity
        2. The underlying cause category
        3. Supporting evidence

        Respond in JSON format:
        {{
            "root_causes": [
                {{
                    "entity": "...",
                    "cause": "...",
                    "category": "...",
                    "confidence": 0.0-1.0,
                    "evidence": "..."
                }}
            ]
        }}
        """

        try:
            response = self.llm.invoke(prompt)
            self.log_llm(prompt, response.content)
            # Parse LLM response (simplified - would need proper JSON parsing)
            return [{
                "entity": "inferred_by_llm",
                "cause": response.content[:200],
                "category": "inferred",
                "confidence": 0.5,
                "evidence": "LLM analysis"
            }]
        except Exception as e:
            self.log(f"LLM analysis failed: {str(e)}")
            return []

    def _run_classic_rca(
        self,
        anomaly: Any,
        obs_data: Any,
        rca_method: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run classic RCA algorithms on the anomaly.

        Uses 5 Whys and Ishikawa analysis to identify root causes.

        Args:
            anomaly: Detected anomaly event
            obs_data: Observability data
            rca_method: User-selected RCA method (None = run all)

        Returns:
            List[Dict[str, Any]]: Root causes from RCA algorithms
        """
        root_causes = []

        if not self.rca_tool:
            self.log("RCA tools not available, skipping classic RCA")
            return root_causes

        # Build problem description
        anomaly_type = getattr(anomaly, 'type', 'unknown')
        severity = getattr(anomaly, 'severity', 'unknown')
        affected = getattr(anomaly, 'affected_entities', [])

        problem = f"{anomaly_type} anomaly with {severity} severity"
        if affected:
            problem += f" affecting {', '.join(affected[:3])}"

        # Build context from observability data
        context = ""
        if hasattr(obs_data, 'summary') and obs_data.summary:
            context = obs_data.summary
            self.log(f"Using observability summary as context: {context[:100]}...")

        self.log(f"Running RCA with problem: {problem}")
        if rca_method:
            self.log(f"User-selected RCA method: {rca_method}")

        # Run 5 Whys analysis
        run_five_whys = rca_method in [None, "auto", "5_whys", "ensemble"]
        run_ishikawa = rca_method in [None, "auto", "ishikawa", "ensemble"]

        if run_five_whys and "5_whys" in self.rca_analyzers:
            try:
                self.log("Starting 5 Whys analysis...")
                five_whys_result = self.rca_analyzers["5_whys"].analyze(
                    problem=problem,
                    context=context[:500] if context else None
                )
                self.log(f"5 Whys completed. Root cause: {five_whys_result.root_cause[:80] if five_whys_result.root_cause else 'None'}")

                if five_whys_result.root_cause:
                    root_causes.append({
                        "entity": affected[0] if affected else "unknown",
                        "cause": five_whys_result.root_cause,
                        "category": five_whys_result.category,
                        "confidence": five_whys_result.confidence,
                        "evidence": f"5 Whys analysis: {' -> '.join(five_whys_result.why_answers[:3])}",
                        "method": "5_whys"
                    })
            except Exception as e:
                self.log(f"5 Whys analysis failed: {str(e)}")

        # Run Ishikawa analysis
        if run_ishikawa and "ishikawa" in self.rca_analyzers:
            try:
                self.log("Starting Ishikawa (Fishbone) analysis...")
                symptoms = f"Type: {anomaly_type}, Severity: {severity}"
                ishikawa_result = self.rca_analyzers["ishikawa"].analyze(
                    problem=problem,
                    symptoms=symptoms,
                    affected_systems=affected[:5] if affected else None
                )
                if ishikawa_result.most_likely_cause:
                    mlc = ishikawa_result.most_likely_cause
                    self.log(f"Ishikawa completed. Most likely: {mlc.description[:60]} ({mlc.category})")

                    root_causes.append({
                        "entity": affected[0] if affected else "unknown",
                        "cause": mlc.description,
                        "category": mlc.category,
                        "confidence": mlc.likelihood,
                        "evidence": f"Ishikawa analysis: {ishikawa_result.analysis_notes}",
                        "method": "ishikawa"
                    })
            except Exception as e:
                self.log(f"Ishikawa analysis failed: {str(e)}")

        self.log(f"Classic RCA identified {len(root_causes)} root cause(s)")
        return root_causes

    def _merge_root_causes(self, root_causes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate root causes from multiple sources.

        Args:
            root_causes: List of root causes from different analyses

        Returns:
            List[Dict[str, Any]]: Merged and deduplicated root causes
        """
        if not root_causes:
            return []

        # Group by similar causes
        seen_causes = {}
        for rc in root_causes:
            cause_key = rc.get("cause", "")[:50].lower()
            if cause_key not in seen_causes:
                seen_causes[cause_key] = rc
            else:
                # Merge: keep higher confidence
                existing = seen_causes[cause_key]
                if rc.get("confidence", 0) > existing.get("confidence", 0):
                    seen_causes[cause_key] = rc

        return list(seen_causes.values())

    def _build_propagation_path(
        self,
        root_causes: List[Dict[str, Any]],
        obs_data: Any
    ) -> List[str]:
        """
        Build the fault propagation path.

        Constructs a sequence showing how the fault spread from the
        root cause to affected components.

        Args:
            root_causes: Identified root causes
            obs_data: Observability data

        Returns:
            List[str]: Ordered list of entities in propagation order
        """
        if not root_causes:
            return []

        # Simple propagation path - starts from root cause entity
        path = [root_causes[0].get("entity", "unknown")]

        # Add affected services/components
        for cause in root_causes[1:]:
            entity = cause.get("entity", "")
            if entity and entity not in path:
                path.append(entity)

        return path

    def _generate_evidence_chain(
        self,
        root_causes: List[Dict[str, Any]],
        anomaly: Any
    ) -> List[str]:
        """
        Generate human-readable evidence chain.

        Creates a list of evidence statements supporting the diagnosis.

        Args:
            root_causes: Identified root causes
            anomaly: Detected anomaly

        Returns:
            List[str]: List of evidence statements
        """
        evidence = []

        # Add anomaly evidence
        evidence.append(f"Anomaly detected: {anomaly.type} (severity: {anomaly.severity})")

        # Add root cause evidence
        for cause in root_causes:
            evidence.append(f"Root cause: {cause.get('cause')} on {cause.get('entity')}")
            if cause.get("evidence"):
                evidence.append(f"  Evidence: {cause.get('evidence')}")

        return evidence

    def _calculate_confidence(
        self,
        root_causes: List[Dict[str, Any]],
        anomaly: Any
    ) -> float:
        """
        Calculate overall diagnosis confidence.

        Combines individual cause confidences with anomaly confidence
        to produce an overall score.

        Args:
            root_causes: Identified root causes
            anomaly: Detected anomaly

        Returns:
            float: Overall confidence score [0.0, 1.0]
        """
        if not root_causes:
            return 0.0

        # Average confidence of root causes
        cause_confidence = sum(c.get("confidence", 0) for c in root_causes) / len(root_causes)

        # Combine with anomaly confidence
        combined = (cause_confidence + anomaly.confidence) / 2

        return min(combined, 0.95)

    def _assess_impact(
        self,
        anomaly: Any,
        obs_data: Any
    ) -> Dict[str, Any]:
        """
        Assess the impact scope of the incident.

        Args:
            anomaly: Detected anomaly
            obs_data: Observability data

        Returns:
            Dict[str, Any]: Impact assessment including affected
                services, users, and business metrics
        """
        return {
            "affected_entities": anomaly.affected_entities,
            "severity": anomaly.severity,
            "potential_user_impact": anomaly.severity in ["HIGH", "CRITICAL"],
            "affected_services": list(set(
                e.split("-")[0] if "-" in e else e
                for e in anomaly.affected_entities
            ))
        }
