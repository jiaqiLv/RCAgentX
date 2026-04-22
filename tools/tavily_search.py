"""
Tavily Search Tool - AI-Powered Search Integration

LangChain tool for searching the web using Tavily API.
Provides real-time information retrieval for RAG and research tasks.
"""

from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import os


class TavilyInput(BaseModel):
    """Input schema for Tavily tool operations."""
    query: str = Field(..., description="Search query string")
    search_depth: Optional[str] = Field("basic", description="Search depth: 'basic' or 'advanced'")
    max_results: Optional[int] = Field(5, description="Maximum number of results to return")


class TavilySearchTool(BaseTool):
    """
    Tool for searching the web using Tavily AI search engine.

    Tavily provides AI-powered search results with relevant excerpts
    and can be used for real-time information retrieval in AIOps
    scenarios like:
    - Looking up error messages and known issues
    - Researching incident patterns
    - Finding documentation and runbooks

    Attributes:
        name (str): Tool name for LangChain
        description (str): Tool description
        api_key (str): Tavily API key
        search_depth (str): Default search depth

    Example:
        ```python
        tavily = TavilySearchTool(api_key="tvly-xxx")

        # Basic search
        results = tavily.search("Kubernetes pod crashloopbackoff")

        # Advanced search with more results
        results = tavily.search(
            query="Prometheus high memory usage solutions",
            search_depth="advanced",
            max_results=10
        )
        ```
    """

    name: str = "tavily_search"
    description: str = "Search the web using Tavily AI. Use for researching errors, finding documentation, and incident patterns."
    api_key: str
    search_depth: str = "basic"
    max_results: int = 5

    args_schema: type[BaseModel] = TavilyInput

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the Tavily search tool.

        Args:
            api_key (Optional[str]): Tavily API key. If not provided,
                reads from TAVILY_API_KEY environment variable.
        """
        api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        super().__init__(api_key=api_key, **kwargs)

    def _run(
        self,
        query: str,
        search_depth: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a search query.

        Args:
            query (str): Search query string
            search_depth (Optional[str]): 'basic' or 'advanced'
            max_results (Optional[int]): Maximum results to return

        Returns:
            List[Dict[str, Any]]: List of search results, each containing:
                - title: Result title
                - url: Source URL
                - content: Relevant excerpt
                - score: Relevance score

        Raises:
            ImportError: If tavily-python is not installed
            Exception: If API request fails

        Example:
            ```python
            results = tavily.search("Kubernetes OOMKilled troubleshooting")
            for r in results:
                print(f"Title: {r['title']}")
                print(f"URL: {r['url']}")
                print(f"Content: {r['content']}")
            ```
        """
        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError(
                "tavily-python is not installed. "
                "Install it with: pip install tavily-python"
            )

        client = TavilyClient(self.api_key)
        depth = search_depth or self.search_depth
        limit = max_results or self.max_results

        try:
            response = client.search(
                query=query,
                search_depth=depth,
                max_results=limit
            )

            # Parse and format results
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", "No title"),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                })

            return results

        except Exception as e:
            raise Exception(f"Tavily search failed: {str(e)}")

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Convenience method for searching.

        Args:
            query (str): Search query
            **kwargs: Additional arguments (search_depth, max_results)

        Returns:
            List[Dict[str, Any]]: Search results
        """
        return self._run(query, **kwargs)

    def get_answer(self, query: str) -> Dict[str, Any]:
        """
        Get a direct answer to a query using Tavily's answer endpoint.

        Args:
            query (str): Question to answer

        Returns:
            Dict[str, Any]: Answer with sources

        Example:
            ```python
            answer = tavily.get_answer("What causes Kubernetes OOMKilled?")
            print(f"Answer: {answer['answer']}")
            print(f"Sources: {answer['sources']}")
            ```
        """
        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError("tavily-python is not installed")

        client = TavilyClient(self.api_key)

        response = client.qna_search(query=query)

        return {
            "answer": response.get("answer", ""),
            "sources": response.get("search_results", []),
        }
