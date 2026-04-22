# RCAgentX - AIOps Multi-Agent System

An intelligent operations (AIOps) multi-agent system built on LangChain and LangGraph.
This system provides automated incident detection, root cause analysis, remediation decision-making,
and repair execution with human-in-the-loop support.

## Architecture

```
                    ┌─────────────────┐
                    │   Supervisor    │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌─────────────────┐   ┌───────────────┐
│ Observability │──►│   Detection     │──►│   Diagnosis   │
└───────────────┘   └─────────────────┘   └───────┬───────┘
                                                   │
                    ┌──────────────────────────────┼──────────┐
                    │                              │          │
                    ▼                              ▼          ▼
            ┌───────────────┐            ┌───────────┐  ┌─────────┐
            │    Decision   │───────────►│  Repair   │  │ Report  │
            └───────────────┘            └───────────┘  └─────────┘
```

## Core Components

### Agents

| Agent | Responsibility |
|-------|---------------|
| **Supervisor** | Global orchestration and multi-agent coordination |
| **Observability** | Multi-modal data collection (Metrics, Logs, Traces) |
| **Detection** | Anomaly detection and classification |
| **Diagnosis** | Root cause analysis with causal inference |
| **Decision** | Remediation strategy with GRPO-based retrieval |
| **Repair** | Dual-mode execution (auto/manual approval) |
| **Report** | Incident report generation |

### Tools

| Tool | Integration |
|------|-------------|
| `PrometheusTool` | Metrics query and analysis |
| `LokiTool` | Log query and pattern detection |
| `AlertManagerTool` | Alert management and silencing |
| `WeChatTool` | Enterprise WeChat notifications |

### Key Features

1. **Multi-Agent Orchestration**: Supervisor-based dynamic workflow routing
2. **Training-Free GRPO**: Experience-based strategy optimization via RAG retrieval
3. **Human-in-the-Loop**: Support for both automatic and manual approval modes
4. **Observable**: Full audit trail with logs and state tracking

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd RCAgentX

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export PROMETHEUS_URL="http://prometheus:9090"
export LOKI_URL="http://loki:3100"
export WECHAT_WEBHOOK_URL="your-webhook-url"
```

## Quick Start

```python
from main import AIOpsSystem
from config.settings import Settings

# Initialize from environment
aiops = AIOpsSystem.from_env()

# Process an incident
result = aiops.process_incident(
    alert_labels={"service": "api-service", "severity": "critical"},
    entities=["api-service-pod-1"]
)

# Access results
print(f"Status: {result['status']}")
print(f"Root Cause: {result['diagnosis'].root_causes}")
print(f"Report: {result['report']['summary']}")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_API_BASE` | Custom API base URL | - |
| `LLM_MODEL` | Model name | `gpt-4o` |
| `PROMETHEUS_URL` | Prometheus server URL | `http://localhost:9090` |
| `LOKI_URL` | Loki server URL | `http://localhost:3100` |
| `ALERTMANAGER_URL` | Alertmanager URL | `http://localhost:9093` |
| `WECHAT_WEBHOOK_URL` | WeChat webhook URL | - |
| `VERBOSE` | Enable verbose logging | `false` |
| `DRY_RUN` | Dry-run mode (no changes) | `false` |

### Programmatic Configuration

```python
from config.settings import Settings, LLMSettings, PrometheusSettings

settings = Settings(
    llm=LLMSettings(
        model="gpt-4o",
        temperature=0.0,
    ),
    prometheus=PrometheusSettings(
        url="http://prometheus:9090",
        timeout=30,
    )
)

aiops = AIOpsSystem(settings)
```

## Workflow

### Incident Processing Flow

1. **Observability**: Collect metrics from Prometheus, logs from Loki
2. **Detection**: Analyze for anomalies using thresholds and patterns
3. **Diagnosis**: Identify root cause with causal inference
4. **Decision**: Generate remediation plan using GRPO retrieval
5. **Repair**: Execute repair (auto or with approval)
6. **Report**: Generate incident report

### Human-in-the-Loop

Repairs are automatically routed for human approval when:
- Risk level is MEDIUM or HIGH
- Diagnosis confidence is below threshold
- Action involves rollback or destructive operations

## GRPO Knowledge Base

The system uses Training-Free GRPO for strategy optimization:

```python
# Access knowledge base
kb = aiops.knowledge_base

# Get statistics
stats = kb.get_stats()
print(f"Total experiences: {stats['total_records']}")
print(f"Success rate: {stats['success_rate']:.1%}")

# Query similar cases
strategies = kb.get_successful_strategies("cpu_spike", k=3)
```

## Project Structure

```
RCAgentX/
├── agents/                 # Agent implementations
│   ├── base.py            # Base agent class
│   ├── supervisor.py      # Orchestrator
│   ├── observability.py   # Data collection
│   ├── detection.py       # Anomaly detection
│   ├── diagnosis.py       # Root cause analysis
│   ├── decision.py        # Strategy decision
│   ├── repair.py          # Repair execution
│   └── report.py          # Report generation
├── tools/                  # LangChain tools
│   ├── prometheus.py      # Prometheus integration
│   ├── loki.py            # Loki integration
│   ├── alert_manager.py   # Alertmanager integration
│   └── wechat.py          # WeChat integration
├── workflows/              # Workflow definitions
│   ├── incident_closure.py
│   ├── auto_repair.py
│   └── manual_approval.py
├── memory/                 # State and knowledge
│   ├── shared_state.py    # Shared state management
│   ├── knowledge_base.py  # GRPO knowledge base
│   └── vector_store.py    # Vector storage
├── config/                 # Configuration
│   ├── settings.py        # Settings management
│   └── prompts.py         # Prompt templates
├── utils/                  # Utilities
│   ├── logger.py          # Logging setup
│   ├── errors.py          # Custom exceptions
│   └── schema.py          # Schema utilities
└── main.py                 # Entry point
```

## License

MIT License
