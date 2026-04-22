"""
Global Settings Configuration

Centralized configuration management for the AIOps agent system.
Supports environment-based configuration and secrets management.
"""

import os
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class LLMSettings:
    """
    Language model configuration settings.

    Attributes:
        provider (str): LLM provider (openai, azure, local)
        model (str): Model name to use
        api_key (str): API key for authentication
        api_base (str): Custom API base URL
        temperature (float): Sampling temperature
        max_tokens (int): Maximum tokens in response
        timeout (int): Request timeout in seconds
    """
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    api_base: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 60


@dataclass
class PrometheusSettings:
    """
    Prometheus monitoring configuration.

    Attributes:
        url (str): Prometheus server URL
        timeout (int): Query timeout in seconds
        default_time_range (int): Default query time range in minutes
    """
    url: str = "http://localhost:9090"
    timeout: int = 30
    default_time_range: int = 30


@dataclass
class LokiSettings:
    """
    Loki logging configuration.

    Attributes:
        url (str): Loki server URL
        timeout (int): Query timeout in seconds
        default_limit (int): Default log entry limit
    """
    url: str = "http://localhost:3100"
    timeout: int = 30
    default_limit: int = 1000


@dataclass
class AlertManagerSettings:
    """
    Alertmanager configuration.

    Attributes:
        url (str): Alertmanager server URL
        timeout (int): Request timeout in seconds
    """
    url: str = "http://localhost:9093"
    timeout: int = 30


@dataclass
class WeChatSettings:
    """
    Enterprise WeChat configuration.

    Attributes:
        webhook_url (str): WeChat bot webhook URL
        secret (str): API secret for signature verification
        default_mentioned_list (List[str]): Default users to mention
    """
    webhook_url: str = ""
    secret: Optional[str] = None
    default_mentioned_list: list = field(default_factory=list)


@dataclass
class AgentSettings:
    """
    Agent behavior configuration.

    Attributes:
        auto_repair_threshold (float): Confidence threshold for auto-repair
        max_repair_iterations (int): Maximum repair retry attempts
        verbose (bool): Enable verbose logging
        dry_run (bool): Enable dry-run mode (no actual changes)
    """
    auto_repair_threshold: float = 0.8
    max_repair_iterations: int = 3
    verbose: bool = True
    dry_run: bool = False


@dataclass
class KnowledgeBaseSettings:
    """
    GRPO knowledge base configuration.

    Attributes:
        persist_dir (str): Directory for persistent storage
        min_reward_threshold (float): Minimum reward for retrieval
        max_retrieval_results (int): Maximum results to retrieve
    """
    persist_dir: str = "./.chroma_db"
    min_reward_threshold: float = 0.5
    max_retrieval_results: int = 5


class Settings:
    """
    Centralized settings manager for the AIOps system.

    Provides unified access to all configuration options with
    environment variable override support.

    Example:
        ```python
        settings = Settings()

        # Access LLM settings
        llm_config = settings.llm

        # Access monitoring settings
        prometheus_config = settings.prometheus
        loki_config = settings.loki

        # Access agent settings
        agent_config = settings.agent
        ```
    """

    def __init__(
        self,
        llm: Optional[LLMSettings] = None,
        prometheus: Optional[PrometheusSettings] = None,
        loki: Optional[LokiSettings] = None,
        alertmanager: Optional[AlertManagerSettings] = None,
        wechat: Optional[WeChatSettings] = None,
        agent: Optional[AgentSettings] = None,
        knowledge_base: Optional[KnowledgeBaseSettings] = None
    ):
        """
        Initialize settings with optional overrides.

        Args:
            llm: LLM configuration override
            prometheus: Prometheus configuration override
            loki: Loki configuration override
            alertmanager: Alertmanager configuration override
            wechat: WeChat configuration override
            agent: Agent configuration override
            knowledge_base: Knowledge base configuration override
        """
        self.llm = llm or LLMSettings()
        self.prometheus = prometheus or PrometheusSettings()
        self.loki = loki or LokiSettings()
        self.alertmanager = alertmanager or AlertManagerSettings()
        self.wechat = wechat or WeChatSettings()
        self.agent = agent or AgentSettings()
        self.knowledge_base = knowledge_base or KnowledgeBaseSettings()

    @classmethod
    def from_env(cls) -> "Settings":
        """
        Create settings from environment variables.

        Reads configuration from environment variables with
        sensible defaults for unspecified values.

        Environment variables:
            OPENAI_API_KEY: OpenAI API key
            OPENAI_API_BASE: Custom OpenAI API base URL
            PROMETHEUS_URL: Prometheus server URL
            LOKI_URL: Loki server URL
            ALERTMANAGER_URL: Alertmanager server URL
            WECHAT_WEBHOOK_URL: WeChat webhook URL
            VERBOSE: Enable verbose mode (true/false)

        Returns:
            Settings: Configured settings instance

        Example:
            ```python
            settings = Settings.from_env()
            ```
        """
        # LLM settings from environment
        llm = LLMSettings(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            api_base=os.getenv("OPENAI_API_BASE"),
            model=os.getenv("LLM_MODEL", "gpt-4o"),
        )

        # Monitoring settings
        prometheus = PrometheusSettings(
            url=os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
        )

        loki = LokiSettings(
            url=os.getenv("LOKI_URL", "http://localhost:3100"),
        )

        alertmanager = AlertManagerSettings(
            url=os.getenv("ALERTMANAGER_URL", "http://localhost:9093"),
        )

        # WeChat settings
        wechat = WeChatSettings(
            webhook_url=os.getenv("WECHAT_WEBHOOK_URL", ""),
            secret=os.getenv("WECHAT_SECRET"),
        )

        # Agent settings
        agent = AgentSettings(
            verbose=os.getenv("VERBOSE", "false").lower() == "true",
            dry_run=os.getenv("DRY_RUN", "false").lower() == "true",
        )

        return cls(
            llm=llm,
            prometheus=prometheus,
            loki=loki,
            alertmanager=alertmanager,
            wechat=wechat,
            agent=agent
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary.

        Returns:
            Dict[str, Any]: Settings as dictionary

        Note:
            Sensitive values (API keys, secrets) are masked.
        """
        return {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "api_key": "***" if self.llm.api_key else None,
                "temperature": self.llm.temperature,
            },
            "prometheus": {
                "url": self.prometheus.url,
                "timeout": self.prometheus.timeout,
            },
            "loki": {
                "url": self.loki.url,
                "timeout": self.loki.timeout,
            },
            "agent": {
                "auto_repair_threshold": self.agent.auto_repair_threshold,
                "verbose": self.agent.verbose,
                "dry_run": self.agent.dry_run,
            },
        }
