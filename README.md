# RCAgentX - AIOps Multi-Agent System

基于 LangChain 和 LangGraph 的智能运维多 Agent 系统，提供自动化事件检测、根因分析、修复决策和执行的完整闭环。

## ✨ 核心特性

- **多 Agent 编排** - 7 个专用 Agent + Supervisor 统一编排
- **经典 RCA 算法** - 5 Whys、鱼骨图、故障树、变更分析、事件相关性
- **无训练 GRPO** - 基于 RAG 检索的经验策略优化
- **人机协同** - 支持自动修复与人工审批双模式
- **LLM 集成** - 支持 OpenAI、DeepSeek 等模型

## 🏗️ 架构

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

## 📦 项目结构

```
RCAgentX/
├── agents/                 # Agent 实现
│   ├── base.py            # 基础抽象类
│   ├── supervisor.py      # 全局编排
│   ├── observability.py   # 多模态数据收集
│   ├── detection.py       # 异常检测
│   ├── diagnosis.py       # 根因分析
│   ├── decision.py        # 修复决策
│   ├── repair.py          # 修复执行
│   └── report.py          # 报告生成
├── algorithms/rca/         # 经典 RCA 算法
│   ├── five_whys.py       # 5 Whys 分析
│   ├── ishikawa.py        # 鱼骨图分析
│   ├── fault_tree.py      # 故障树分析
│   ├── change_analysis.py # 变更分析
│   ├── event_correlation.py# 事件相关性
│   └── mcp_tool.py        # 统一 MCP 工具
├── tools/                  # LangChain Tools
├── memory/                 # 状态与知识库
├── workflows/              # LangGraph 工作流
├── config/                 # 配置管理
├── main.py                 # 入口
└── .env.example            # 环境变量模板
```

## 🚀 快速开始

### 1. 安装

```bash
git clone <repo-url>
cd RCAgentX
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env 配置 LLM 和监控服务
```

**必要配置:**
```bash
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.deepseek.com/v1  # 或 OpenAI
LLM_MODEL=deepseek-chat
```

### 3. 运行

```bash
# 测试事件
python main.py --test-incident

# 交互式模式
python main.py --interactive

# 显示 LLM 交互日志
python main.py --test-incident --debug-rca
```

## 🔍 RCA 算法选择

### 查看可用算法

```bash
python main.py --list-rca
```

```
══════════════════════════════════════════════════════════════
  🔍 Available Root Cause Analysis Algorithms
══════════════════════════════════════════════════════════════

  [1] 🤖 Auto Select         - 自动选择最佳方法
  [2] ❓ 5 Whys              - 5 Why 分析法
  [3] 🐟 Ishikawa (Fishbone) - 鱼骨图分析 (6 维度)
  [4] 🌳 Fault Tree Analysis - 故障树分析
  [5] 🔄 Change Analysis     - 变更分析
  [6] 🔗 Event Correlation   - 事件相关性分析
  [7] 🎯 Ensemble            - 运行所有方法并综合
```

### 选择特定算法

```bash
# 交互式选择算法
python main.py --select-rca

# 选择 5 Whys 运行
echo "2" | python main.py --select-rca --test-incident

# 选择 Ishikawa 运行
echo "3" | python main.py --select-rca --test-incident
```

## 📊 工作流

### 事件处理流程

| 步骤 | Agent | LLM 交互 | 说明 |
|------|-------|---------|------|
| 1 | Observability | ✅ #1 | 多模态数据收集与摘要生成 |
| 2 | Detection | ❌ | 基于阈值的异常检测 |
| 3 | Diagnosis | ✅ #2 | 根因分析 (RCA 算法+LLM) |
| 4 | Decision | ❌ | 修复策略生成 (GRPO 检索) |
| 5 | Repair | ❌ | 修复执行 (自动/人工) |
| 6 | Report | ✅ #3 | 报告生成与执行摘要 |

### LLM 交互点

启用 `--debug-rca` 显示所有 LLM 交互 (绿色输出):

```bash
python main.py --test-incident --debug-rca
```

输出示例:
```
[LLM] Prompt to observability:
  Analyze the following observability data...

[LLM] Response from observability:
  **Summary of Observability Data**
  1. Key Anomalies in Metrics...
```

## 🔧 配置选项

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | LLM API 密钥 | - |
| `OPENAI_API_BASE` | API 基础 URL | - |
| `LLM_MODEL` | 模型名称 | `deepseek-chat` |
| `VERBOSE` | 详细日志 | `false` |
| `MOCK_ENABLED` | 使用模拟数据 | `true` |

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--interactive` | 交互式模式 |
| `--test-incident` | 运行测试事件 |
| `--debug-rca` | 显示 RCA 调试日志 |
| `--select-rca` | 交互式选择 RCA 算法 |
| `--list-rca` | 列出所有 RCA 算法 |

## 🧠 GRPO 知识库

无训练的策略优化，通过向量检索相似历史案例:

```python
from main import AIOpsSystem

aiops = AIOpsSystem.from_env()

# 获取知识库统计
stats = aiops.get_knowledge_base_stats()

# 检索成功策略
strategies = aiops.knowledge_base.get_successful_strategies(
    anomaly_type="cpu_spike", k=3
)
```

## 📁 报告生成

报告自动保存到 `./reports/` 目录:

```bash
ls -la reports/
-rw-r--r--  1 user  staff  8789 Apr 22 16:26 inc-1776846454_20260422_162734_report.md
```

## 🔗 支持的服务商

| 服务商 | Base URL | 模型 |
|--------|----------|------|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| Azure OpenAI | 自定义端点 | `gpt-4` |

## 📚 文档

- [WORKFLOW.md](WORKFLOW.md) - 详细工作流程图
- [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) - 完整使用指南
