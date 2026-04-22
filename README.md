# RCAgentX - AIOps Multi-Agent System

An intelligent operations (AIOps) multi-agent system built on LangChain and LangGraph.
This system provides automated incident detection, root cause analysis, remediation decision-making,
and repair execution with human-in-the-loop support.

## ✨ Features

- **Multi-Agent Orchestration** - 7 specialized agents coordinated by a Supervisor
- **Training-Free GRPO** - Experience-based strategy optimization via RAG retrieval
- **Human-in-the-Loop** - Dual-mode execution (automatic or manual approval)
- **LLM Integration** - Support for OpenAI, DeepSeek, and other OpenAI-compatible APIs
- **Interactive CLI** - Built-in chat application for natural language interaction

## 🏗️ Architecture

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

## 📦 Project Structure

```
RCAgentX/
├── agents/                 # Multi-agent implementations
│   ├── base.py            # Abstract base class for all agents
│   ├── supervisor.py      # Global orchestration and workflow routing
│   ├── observability.py   # Multi-modal data collection
│   ├── detection.py       # Anomaly detection and classification
│   ├── diagnosis.py       # Root cause analysis
│   ├── decision.py        # Remediation strategy with GRPO
│   ├── repair.py          # Dual-mode repair execution
│   └── report.py          # Incident report generation
├── tools/                  # LangChain tools
│   ├── prometheus.py      # Prometheus metrics query
│   ├── loki.py            # Loki log query
│   ├── alert_manager.py   # AlertManager management
│   ├── wechat.py          # Enterprise WeChat notifications
│   └── tavily_search.py   # Tavily AI web search
├── workflows/              # LangGraph workflow definitions
│   ├── incident_closure.py    # Standard incident closure flow
│   ├── auto_repair.py         # Automatic repair workflow
│   └── manual_approval.py     # Human approval workflow
├── memory/                 # State and knowledge management
│   ├── shared_state.py    # Shared state container
│   ├── knowledge_base.py  # GRPO experience storage (ChromaDB)
│   └── vector_store.py    # Vector storage wrapper
├── algorithms/             # Core algorithms
│   └── __init__.py        # Algorithm exports
├── config/                 # Configuration
│   ├── settings.py        # Settings management
│   └── prompts.py         # Prompt templates
├── utils/                  # Utilities
│   ├── logger.py          # Logging configuration
│   ├── errors.py          # Custom exceptions
│   └── schema.py          # Schema utilities
├── integrations/           # External integrations
│   └── __init__.py        # Integration exports
├── app.py                  # Interactive CLI chat application
├── main.py                 # Main entry point
├── run.sh                  # Quick start script
├── requirements.txt        # Python dependencies
└── .env.example            # Environment variables template
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd RCAgentX

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
vi .env
```

**Required settings:**
```bash
# LLM Configuration (DeepSeek example)
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# Monitoring (optional)
PROMETHEUS_URL=http://localhost:9090
LOKI_URL=http://localhost:3100
```

### 3. Run

**Interactive Chat:**
```bash
./run.sh app
# or
python app.py
```

**Programmatic Usage:**
```bash
./run.sh --test-incident --dry-run
```

## 💬 Interactive CLI

The built-in chat application (`app.py`) provides natural language interaction:

```bash
python app.py
```

**Available Commands:**
| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/clear` | Clear conversation history |
| `/history` | Show recent conversation |
| `/status` | Show system status |
| `/quit` | Exit application |

**Example Questions:**
- "What is AIOps?"
- "How do I troubleshoot high CPU usage?"
- "What are the best practices for alerting?"
- "Explain root cause analysis"

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | LLM API key | - |
| `OPENAI_API_BASE` | Custom API base URL | - |
| `LLM_MODEL` | Model name | `deepseek-chat` |
| `PROMETHEUS_URL` | Prometheus server URL | `http://localhost:9090` |
| `LOKI_URL` | Loki server URL | `http://localhost:3100` |
| `ALERTMANAGER_URL` | Alertmanager URL | `http://localhost:9093` |
| `WECHAT_WEBHOOK_URL` | WeChat webhook URL | - |
| `VERBOSE` | Enable verbose logging | `false` |
| `DRY_RUN` | Dry-run mode (no actual changes) | `false` |

### Supported LLM Providers

| Provider | Base URL | Example Model |
|----------|----------|---------------|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| Azure OpenAI | Your Azure endpoint | `gpt-4` |
| Local (Ollama) | `http://localhost:11434/v1` | `llama2` |

## 📊 Workflow

### Incident Processing Flow

1. **Observability** → Collect metrics, logs, traces
2. **Detection** → Identify anomalies using rules + ML
3. **Diagnosis** → Root cause analysis with causal inference
4. **Decision** → Generate remediation plan using GRPO retrieval
5. **Repair** → Execute repair (auto or manual approval)
6. **Report** → Generate incident report

### Human-in-the-Loop

Repairs require human approval when:
- Risk level is MEDIUM or HIGH
- Diagnosis confidence < threshold (default: 0.8)
- Action involves rollback or destructive operations

## 🧠 GRPO Knowledge Base

Training-Free GRPO (Generalized Reward-based Policy Optimization) enables
continuous improvement through experience retrieval:

```python
from main import AIOpsSystem

aiops = AIOpsSystem.from_env()

# Get knowledge base stats
stats = aiops.get_knowledge_base_stats()
print(f"Total experiences: {stats['total_records']}")
print(f"Success rate: {stats['success_rate']:.1%}")

# Query successful strategies
strategies = aiops.knowledge_base.get_successful_strategies(
    anomaly_type="cpu_spike",
    k=3
)
```

## 📚 API Reference

### AIOpsSystem

```python
from main import AIOpsSystem

# Initialize
aiops = AIOpsSystem.from_env()

# Process incident
result = aiops.process_incident(
    alert_labels={"service": "api", "severity": "critical"},
    entities=["api-pod-1"]
)

# Access results
print(f"Status: {result['status']}")
print(f"Root Cause: {result['diagnosis'].root_causes}")
print(f"Report: {result['report']['summary']}")
```

### Agents

Each agent implements a standard interface:

```python
from agents.detection import DetectionAgent
from memory.shared_state import ObservabilityData

agent = DetectionAgent(llm=llm, verbose=True)
result = agent.execute({"observability": obs_data})
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Links

- [GitHub Repository](https://github.com/jiaqiLv/RCAgentX)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
