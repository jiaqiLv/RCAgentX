"""
GRPO Experience Knowledge Base

Training-Free GRPO: Optimizes decision strategies through RAG retrieval
of historical experiences. No parameter updates required - achieves
strategy optimization through similarity-based case retrieval.
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from memory.shared_state import ExperienceRecord, RootCauseAnalysis


class GRPOKnowledgeBase:
    """
    Training-Free GRPO Knowledge Base.

    Retrieves similar historical cases through vector search to extract
    successful strategies. Uses ChromaDB for persistent vector storage
    and semantic retrieval.

    The GRPO (Generalized Reward-based Policy Optimization) approach
    stores complete incident resolution experiences and retrieves the
    most relevant cases when similar anomalies are detected. This enables
    continuous improvement without model fine-tuning.

    Attributes:
        persist_dir (Path): Directory for persistent storage
        client (chromadb.PersistentClient): ChromaDB client instance
        collection (chromadb.Collection): Collection for experience records

    Example:
        ```python
        # Initialize knowledge base
        kb = GRPOKnowledgeBase(persist_dir="./data/chroma_db")

        # Add experience record
        record = ExperienceRecord(
            incident_id="inc-001",
            anomaly_type="cpu_spike",
            root_cause="memory_leak",
            action_taken="restarted_pod",
            outcome="success",
            reward=0.9
        )
        kb.add_experience(record)

        # Query similar cases
        strategies = kb.get_successful_strategies("cpu_spike", k=3)
        ```
    """

    def __init__(self, persist_dir: str = "./.chroma_db"):
        """
        Initialize the GRPO knowledge base.

        Args:
            persist_dir (str): Directory path for persistent storage
                of ChromaDB vectors. Defaults to local .chroma_db folder.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Create or get the experiences collection
        self.collection = self.client.get_or_create_collection(
            name="grpo_experiences",
            metadata={"description": "GRPO experience records for strategy optimization"}
        )

    def add_experience(
        self,
        record: ExperienceRecord,
        context: Optional[str] = None
    ):
        """
        Add an experience record to the knowledge base.

        Stores the experience as a vector embedding for future retrieval.
        The anomaly_type is used as the primary indexing dimension for
        similarity search.

        Args:
            record (ExperienceRecord): The experience record to store
            context (Optional[str]): Additional context information that
                might help with future retrieval (e.g., environmental
                conditions, recent deployments, known issues)

        Returns:
            None

        Example:
            ```python
            record = ExperienceRecord(
                incident_id="inc-001",
                anomaly_type="latency_spike",
                root_cause="database_connection_pool_exhaustion",
                action_taken="increased_pool_size_and_restarted",
                outcome="success",
                reward=0.85
            )
            kb.add_experience(record, context="After v2.3 deployment")
            ```
        """
        # Generate unique document ID
        doc_id = f"{record.incident_id}_{record.timestamp.timestamp()}"

        # Build document content as JSON
        doc_content = json.dumps({
            "incident_id": record.incident_id,
            "anomaly_type": record.anomaly_type,
            "root_cause": record.root_cause,
            "action_taken": record.action_taken,
            "outcome": record.outcome,
            "reward": record.reward,
            "context": context or "",
        })

        # Upsert into ChromaDB (insert or update)
        # ChromaDB automatically generates embeddings using default model
        self.collection.upsert(
            ids=[doc_id],
            documents=[doc_content],
            metadatas=[{
                "anomaly_type": record.anomaly_type,
                "outcome": record.outcome,
                "reward": record.reward,
                "timestamp": record.timestamp.isoformat(),
            }]
        )

    def query_similar(
        self,
        anomaly_type: str,
        k: int = 5,
        min_reward: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Query similar experience records by anomaly type.

        Performs semantic similarity search to find historically
        similar incidents. Results are filtered by minimum reward
        threshold and sorted by reward score.

        Args:
            anomaly_type (str): Type of anomaly to search for. This is
                embedded and compared against stored experiences.
            k (int): Maximum number of results to return. Defaults to 5.
            min_reward (Optional[float]): Minimum reward threshold for
                filtering results. Only experiences with reward >= this
                value will be returned.

        Returns:
            List[Dict[str, Any]]: List of similar experience records,
                sorted by reward (descending). Each dict contains:
                - incident_id: Original incident identifier
                - anomaly_type: Type of anomaly
                - root_cause: Identified root cause
                - action_taken: Remediation action performed
                - outcome: 'success' or 'failure'
                - reward: Reward score
                - context: Additional context if provided
                - distance: Vector distance from query (lower is closer)

        Example:
            ```python
            # Find similar CPU spike incidents
            similar = kb.query_similar("cpu_spike", k=5, min_reward=0.5)

            for case in similar:
                print(f"Case {case['incident_id']}: {case['action_taken']}")
                print(f"  Reward: {case['reward']}, Distance: {case['distance']}")
            ```
        """
        # Query ChromaDB for similar documents
        # Retrieve 2x requested to allow filtering headroom
        query_results = self.collection.query(
            query_texts=[anomaly_type],
            n_results=k * 2,
        )

        # Filter and process results
        results = []
        for i, doc in enumerate(query_results["documents"][0]):
            metadata = query_results["metadatas"][0][i]

            # Filter by minimum reward threshold
            if min_reward and metadata.get("reward", 0) < min_reward:
                continue

            try:
                # Parse JSON document
                data = json.loads(doc)
                data["distance"] = query_results["distances"][0][i]
                results.append(data)
            except (json.JSONDecodeError, KeyError) as e:
                # Skip malformed documents
                continue

        # Sort by reward (highest first)
        results.sort(key=lambda x: x.get("reward", 0), reverse=True)

        # Return top k results
        return results[:k]

    def get_successful_strategies(
        self,
        anomaly_type: str,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get successful strategies for a given anomaly type.

        Retrieves historical cases that resulted in successful outcomes
        and extracts the key strategy information. This is the primary
        method used by the Decision Agent for GRPO-based recommendations.

        Args:
            anomaly_type (str): Type of anomaly to find strategies for
            k (int): Maximum number of strategies to return. Defaults to 3.

        Returns:
            List[Dict[str, Any]]: List of successful strategies, each
                containing:
                - action: The action that was taken
                - root_cause: The identified root cause
                - reward: The reward score for this outcome
                - incident_id: Reference to original incident

        Example:
            ```python
            strategies = kb.get_successful_strategies("memory_leak")

            for strategy in strategies:
                print(f"Action: {strategy['action']}")
                print(f"Root cause: {strategy['root_cause']}")
                print(f"Success score: {strategy['reward']}")
            ```
        """
        # Query with higher k to ensure enough successful cases
        similar = self.query_similar(anomaly_type, k=k * 2, min_reward=0.5)

        # Extract only successful outcomes
        strategies = []
        for case in similar:
            if case.get("outcome") == "success":
                strategies.append({
                    "action": case.get("action_taken"),
                    "root_cause": case.get("root_cause"),
                    "reward": case.get("reward"),
                    "incident_id": case.get("incident_id"),
                })

        # Return top k successful strategies
        return strategies[:k]

    def update_reward(
        self,
        incident_id: str,
        new_reward: float
    ):
        """
        Update the reward value for an existing experience.

        Used to update experience records after the outcome is known.
        This enables the GRPO loop to learn from both successes and
        failures.

        Args:
            incident_id (str): The incident ID to update
            new_reward (float): New reward value [0.0, 1.0]

        Returns:
            bool: True if updated, False if incident not found

        Example:
            ```python
            # After successful repair
            kb.update_reward("inc-001", new_reward=0.9)

            # After failed repair
            kb.update_reward("inc-002", new_reward=0.1)
            ```
        """
        # Find existing record by incident_id
        existing = self.collection.get(
            where={"incident_id": incident_id}
        )

        # Update if found
        if existing["ids"]:
            doc_id = existing["ids"][0]
            self.collection.update(
                ids=[doc_id],
                metadatas=[{"reward": new_reward}]
            )
            return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.

        Returns aggregate metrics including total records, success
        counts, and overall success rate. Useful for monitoring
        the growth and quality of accumulated experience.

        Returns:
            Dict[str, Any]: Statistics dictionary containing:
                - total_records: Total number of stored experiences
                - success_records: Number of successful outcomes
                - failure_records: Number of failed outcomes
                - success_rate: Ratio of successful outcomes
                - avg_reward: Average reward across all records

        Example:
            ```python
            stats = kb.get_stats()
            print(f"Total experiences: {stats['total_records']}")
            print(f"Success rate: {stats['success_rate']:.2%}")
            print(f"Average reward: {stats['avg_reward']:.2f}")
            ```
        """
        count = self.collection.count()

        # Get success count
        success_result = self.collection.get(
            where={"outcome": "success"}
        )
        success_count = len(success_result["ids"])

        # Get failure count
        failure_result = self.collection.get(
            where={"outcome": "failure"}
        )
        failure_count = len(failure_result["ids"])

        # Calculate average reward
        all_records = self.collection.get(include=["metadatas"])
        rewards = [m.get("reward", 0) for m in all_records["metadatas"]]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        return {
            "total_records": count,
            "success_records": success_count,
            "failure_records": failure_count,
            "success_rate": success_count / count if count > 0 else 0.0,
            "avg_reward": avg_reward,
        }

    def clear(self):
        """
        Clear all records from the knowledge base.

        WARNING: This permanently deletes all stored experiences.
        Use with caution - typically only for testing or reset scenarios.

        Returns:
            None
        """
        self.client.delete_collection("grpo_experiences")
        self.collection = self.client.create_collection(
            name="grpo_experiences",
            metadata={"description": "GRPO experience records"}
        )
