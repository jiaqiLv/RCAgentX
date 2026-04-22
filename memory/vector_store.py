"""
Vector Store - ChromaDB Vector Storage

Wrapper for ChromaDB vector storage operations.
Used for semantic search and RAG retrieval.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    """
    Vector storage and retrieval using ChromaDB.

    Provides methods for storing embeddings and performing
    semantic similarity search.

    Attributes:
        persist_dir (Path): Directory for persistent storage
        client (chromadb.PersistentClient): ChromaDB client
        collections (Dict[str, chromadb.Collection]): Named collections

    Example:
        ```python
        store = VectorStore(persist_dir="./chroma_db")

        # Add documents
        store.add_documents(
            collection="incidents",
            documents=["CPU spike incident", "Memory leak"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"type": "cpu"}, {"type": "memory"}]
        )

        # Search
        results = store.search("CPU spike", collection="incidents", k=5)
        ```
    """

    def __init__(self, persist_dir: str = "./.chroma_db"):
        """
        Initialize the vector store.

        Args:
            persist_dir (str): Directory for persistent storage
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        self._collections: Dict[str, chromadb.Collection] = {}

    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> chromadb.Collection:
        """
        Get or create a named collection.

        Args:
            name (str): Collection name
            metadata (Optional[Dict[str, Any]]): Collection metadata

        Returns:
            chromadb.Collection: The collection

        Example:
            ```python
            collection = store.get_or_create_collection("incidents")
            ```
        """
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata=metadata or {}
            )
        return self._collections[name]

    def add_documents(
        self,
        collection: str,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents to a collection.

        Args:
            collection (str): Collection name
            documents (List[str]): Documents to add
            embeddings (Optional[List[List[float]]]): Optional embeddings.
                If None, ChromaDB will generate them automatically.
            metadatas (Optional[List[Dict[str, Any]]]): Optional metadata
            ids (Optional[List[str]]): Optional IDs. If None, auto-generated.

        Returns:
            bool: True if successful

        Example:
            ```python
            store.add_documents(
                collection="incidents",
                documents=["CPU spike on api-service"],
                metadatas=[{"severity": "high"}],
                ids=["inc-001"]
            )
            ```
        """
        coll = self.get_or_create_collection(collection)

        # Auto-generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        # Auto-generate embeddings if not provided (ChromaDB default)
        coll.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return True

    def search(
        self,
        query: str,
        collection: str = "default",
        k: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query (str): Query text for semantic search
            collection (str): Collection to search
            k (int): Number of results
            where (Optional[Dict[str, Any]]): Optional filter

        Returns:
            List[Dict[str, Any]]: Search results with documents,
                metadatas, and distances

        Example:
            ```python
            results = store.search("CPU issue", collection="incidents", k=3)
            for r in results:
                print(f"Document: {r['document']}")
                print(f"Distance: {r['distance']}")
            ```
        """
        coll = self.get_or_create_collection(collection)

        results = coll.query(
            query_texts=[query],
            n_results=k,
            where=where
        )

        # Format results
        formatted = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "document": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })

        return formatted

    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.

        Args:
            name (str): Collection name

        Returns:
            bool: True if deleted
        """
        if name in self._collections:
            del self._collections[name]
        self.client.delete_collection(name)
        return True

    def count(self, collection: str) -> int:
        """
        Get document count in a collection.

        Args:
            collection (str): Collection name

        Returns:
            int: Number of documents
        """
        coll = self.get_or_create_collection(collection)
        return coll.count()

    def list_collections(self) -> List[str]:
        """
        List all collections.

        Returns:
            List[str]: Collection names
        """
        return [c.name for c in self.client.list_collections()]
