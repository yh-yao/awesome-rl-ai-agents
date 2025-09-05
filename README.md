# Awesome RL AI Agents

---

## 🔎 Quick Navigation

- 🧠✨ [Agentic Workflow without Training](#agentic-workflow-without-training)
- 🧪📊 [Agent Evaluation and Benchmarks](#agent-evaluation-and-benchmarks)
- 🧰⚙️ [Agent Training Frameworks](#agent-training-frameworks)
- 👤🧭 [RL for Single Agent](#rl-for-single-agent)
  - 🔁🧪 [Self-Evolution & Test-Time RL](#self-evolution--test-time-rl)
  - 🛠️🧠 [RL for Tool Use & Agent Training](#rl-for-tool-use--agent-training)
  - 💾🧠 [Memory & Knowledge Management](#memory--knowledge-management)
  - 🔁📈 [Fine-Grained RL & Trajectory Calibration](#fine-grained-rl--trajectory-calibration)
  - 🎛️🎯 [Alignment & Preference Optimization](#alignment--preference-optimization)
- 💸🧠 [Cost-Aware Reasoning & Budget-Constrained RL](#cost-aware-reasoning--budget-constrained-rl)
- 👥🤝 [RL for Multi-Agent Systems](#rl-for-multi-agent-systems)
  - 🗺️📅 [Planning](#planning)
  - 🤝🧩 [Collaboration](#collaboration)
- 🤖🌍 [Embodied Agents & World Models](#embodied-agents--world-models)
- 📚💡 [Surveys & Position Papers](#surveys--position-papers)
- 🧾✅ [Concluding Remarks](#concluding-remarks)

---

## Agentic Workflow without Training

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| Tree of Thoughts | ToT | ICML | 2023 | [Paper](https://arxiv.org/abs/2305.10601) | Explores multiple reasoning paths via search trees. |
| Reflexion | Reflexion | NeurIPS | 2023 | [Paper](https://arxiv.org/abs/2303.11366) | Self-verbal reinforcement for reducing hallucinations. |
| Self-Refine | Self-Refine | NeurIPS | 2023 | [Paper](https://arxiv.org/abs/2303.17651) | Iterative refinement with self-feedback. |
| ReAct | ReAct | ICLR | 2023 | [Paper](https://arxiv.org/abs/2210.03629) | Interleaves reasoning with acting. |
| SwiftSage | SwiftSage | ACL | 2023 | [Paper](https://arxiv.org/abs/2305.17390) | Fast/slow thinking agents for complex tasks. |
| DynaSaur | DynaSaur | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2411.01747) | Dynamic action spaces beyond predefined tools. |

---

## Agent Evaluation and Benchmarks

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| GAIA | GAIA | Arxiv | 2023 | [Paper](https://arxiv.org/abs/2311.12983) | 466 diverse tasks for general AI assistants. |
| TaskBench | TaskBench | EMNLP | 2023 | [Paper](https://arxiv.org/abs/2311.18760) | Benchmarks LLM automation. |
| AgentBench | AgentBench | Arxiv | 2023 | [Paper](https://arxiv.org/abs/2308.03688) | 51 scenarios for evaluating agents. |
| ACEBench | ACEBench | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2501.12851) | Tool usage benchmark. |
| Agent Leaderboard | Galileo LB | HF | 2024 | [Dataset](https://huggingface.co/datasets/galileo-ai/agent-leaderboard) | Tracks GAIA leaderboard agents. |
| Agentic Predictor | Agentic Predictor | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2505.19764) | Predicts workflow performance. |

---

## Agent Training Frameworks

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| Agent Lightning | Agent Lightning | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2508.03680) | Decouples execution from training. |
| SkyRL-v0 🔁 | SkyRL-v0 | ArXiv | 2025 | [Blog](https://novasky-ai.notion.site/skyrl-v0) \| [Code](https://github.com/NovaSky-AI/SkyRL) | Online RL for long-horizon tasks. |
| OpenManus-RL 🔁 | OpenManus-RL | GitHub | 2025 | [Code](https://github.com/OpenManus/OpenManus-RL) | Live-streamed RL tuning. |
| MASLab | MASLab | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2505.16988) | Unified multi-agent RL codebase. |
| VerlTool | VerlTool | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2509.01055) \| [Code](https://github.com/TIGER-AI-Lab/verl-tool) | Unified ARLT framework with async rollouts. |

---

## RL for Single Agent

### Self-Evolution & Test-Time RL

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| TTRL | TTRL | ICLR | 2025 | [Paper](https://arxiv.org/abs/2504.16084) | Test-time RL via majority-vote. |
| ProRL | ProRL | ICLR | 2025 | [Paper](https://arxiv.org/abs/2505.24864) | Expands reasoning boundaries. |
| Self-Evolving Survey 📚 | Self-Evolving Survey | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2507.21046) | Survey of evolving agents. |
| RAGEN / StarPO | RAGEN | ICLR | 2025 | [Paper](https://arxiv.org/abs/2504.20073) | Multi-turn RL with trajectory filtering. |
| Alita | Alita | GAIA LB | 2025 | [Paper](https://arxiv.org/abs/2505.20286) | Generalist evolving agent. |
| Gödel Agent | Gödel Agent | ACL/Arxiv | 2024–25 | [Paper](https://arxiv.org/abs/2410.04444) | Recursive self-improvement. |
| Darwin GM | Darwin GM | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.22954) | Darwinian exploration. |
| SkyRL-v0 🔁 | SkyRL-v0 | Arxiv/GitHub | 2025 | [Blog](https://novasky-ai.notion.site/skyrl-v0) \| [Code](https://github.com/NovaSky-AI/SkyRL) | Long-horizon RL. |

---

### RL for Tool Use & Agent Training

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| AGILE | AGILE | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2405.14751) | RL with memory, tool use, and expert consultation. |
| AgentOptimizer | AgentOptimizer | ICML | 2024 | [Paper](https://arxiv.org/abs/2402.11359) | Functions as learnable weights with rollback/early stop. |
| FireAct | FireAct | Arxiv | 2023 | [Paper](https://arxiv.org/abs/2310.05915) | Multi-task fine-tuning for agents. |
| ToRL | ToRL | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2503.23383) | RL framework for tool strategies. |
| ToolRL | ToolRL | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2504.13958) | Reward design for tool use. |
| ARTIST | ARTIST | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.01441) | Integrates reasoning with tool RL. |
| ZeroTIR | ZeroTIR | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.07773) | Scaling law for tool RL. |
| OTC | OTC | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2504.14870) | Teaches efficient tool call policy. |
| WebAgent-R1 | WebAgent-R1 | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.16421) | Multi-turn RL for web tasks. |
| GiGPO | GiGPO | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.10978) | Hierarchical RL with step-level rewards. |
| Nemotron-Tool-N1 | Nemotron | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.00024) | RL for tool calls rivaling SFT+RL. |
| CATP-LLM | CATP-LLM | ICCV/Arxiv | 2024–25 | [Paper](https://arxiv.org/abs/2411.16313) \| [Code](https://github.com/duowuyms/OpenCATP-LLM) | Cost-aware non-sequential tool planning. |
| Tool-Star | Tool-Star | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.16410) | Multi-tool RL with hierarchical rewards. |

---

### Memory & Knowledge Management

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| Memory-R1 | Memory-R1 | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2508.19828) | RL-trained memory manager + answer agent. |
| A-MEM | A-MEM | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2502.12110) | Zettelkasten-style dynamic memory. |
| KnowAgent | KnowAgent | NAACL Findings | 2025 | [Paper](https://arxiv.org/abs/2403.03101) | Knowledge-augmented planning. |

---

### Fine-Grained RL & Trajectory Calibration

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| StepTool | StepTool | CIKM | 2025 | [Paper](https://arxiv.org/abs/2410.07745) | Step-grained reward shaping. |
| RLTR | RLTR | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2508.19598) | Planning completeness as reward. |
| SPA-RL | SPA-RL | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.20732) | Stepwise reward decomposition. |
| STeCa | STeCa | ACL Findings | 2025 | [Paper](https://arxiv.org/abs/2502.14276) | Step-level trajectory calibration. |
| SWEET-RL | SWEET-RL | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2503.15478) | ColBench + step-wise critic. |
| ATLaS | ATLaS | ACL | 2025 | [Paper](https://arxiv.org/abs/2503.02197) | Critical step selection for fine-tuning. |

---

### Alignment & Preference Optimization

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| MODPO | MODPO | Arxiv | 2023–24 | [Paper](https://arxiv.org/abs/2310.03708) | Multi-objective DPO. |

---

## Cost-Aware Reasoning & Budget-Constrained RL

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| CATS | CATS | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.14656) | Cost-augmented MCTS. |
| TALE | TALE | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2412.18547) | Token-budget-aware reasoning. |
| FrugalGPT | FrugalGPT | Arxiv | 2023 | [Paper](https://arxiv.org/abs/2305.05176) | Minimizes API costs. |
| TREACLE | TREACLE | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2404.13082) | Budget-constrained cascades. |
| BudgetMLAgent | BudgetMLAgent | AIMLSystems | 2025 | — | Cost-effective ML automation. |
| Cost of Dynamic Reasoning | Cost Reasoning | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2506.04301) | Systems perspective on reasoning cost. |
| Budget-Aware Evaluation | BudgetEval | EMNLP | 2024 | [Paper](https://aclanthology.org/2024.emnlp-main.1112/) | Budget-aware evaluation study. |
| LLM MoT Cascade | LLM Cascade | ICLR | 2024 | [Paper](https://arxiv.org/abs/2310.03094) \| [Code](https://github.com/MurongYue/LLM_MoT_cascade) | Cascades with mixed thought reps. |
| BudgetThinker | BudgetThinker | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2508.17196) | Control tokens for budget adherence. |

---
## RL for Multi-Agent Systems

### Planning

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| OWL | OWL | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.23885) | Hierarchical planner + worker agents for automation. |
| AWorld | AWorld | NeurIPS | 2024 | [Paper](https://arxiv.org/abs/2508.09889) | Profile-aware maneuvering with guard agents. |
| Plan-over-Graph | PlanGraph | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2502.14563) | Graph-based parallel scheduling for agents. |
| MARL Survey 📚 | MARL Survey | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2405.11106) | Survey of multi-agent RL coordination/communication. |
| Self-Resource Allocation | Self-ResAlloc | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2504.02051) | Planner vs orchestrator allocation strategies. |
| MASLab 🔁 | MASLab | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.16988) | Unified MAS benchmarks & APIs. |

---

### Collaboration

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| ACC-Collab | ACC-Collab | ICLR | 2025 | [Paper](https://openreview.net/forum?id=nfKfAzkiez) | Actor-critic agents for collaboration. |
| Chain of Agents | ChainAgents | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2406.02818) | Long-context collaboration via agent chains. |
| Scaling LLM MAC | ScalingMAC | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2406.07155) | Scaling studies of multi-agent collab. |
| MMAC-Copilot | MMAC | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2404.18074) | Multi-modal collaboration. |
| CORY | CORY | NeurIPS | 2024 | [Paper](https://arxiv.org/abs/2410.06101) \| [Code](https://github.com/Harry67Hu/CORY) | Role-swapping PPO co-training. |
| OpenManus-RL 🔁 | OpenManus-RL | GitHub | 2025 | [Code](https://github.com/OpenManus/OpenManus-RL) | Live-streamed RL tuning. |
| MAPoRL | MAPoRL | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2502.18439) | Co-refinement via verifier reward. |
| ACC-Collab (preprint) | ACC-Collab | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2411.00053) | Guided collaboration trajectories. |

---

## Embodied Agents & World Models

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| LEO | LEO | ICML | 2024 | [Paper](https://arxiv.org/abs/2311.12871) | Embodied multimodal generalist agent. |
| DreamerV3 | DreamerV3 | Arxiv | 2023 | [Paper](https://arxiv.org/abs/2301.04104) | World-model RL across 150+ tasks. |
| WMA Web Agent | WMA | ICLR | 2025 | [Paper](https://arxiv.org/abs/2410.13232) | Web outcome simulation. |
| WorldCoder | WorldCoder | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2402.12275) | Code-based world model construction. |
| WALL-E 2.0 | WALL-E 2.0 | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2504.15785) | Neuro-symbolic world alignment. |
| WorldLLM | WorldLLM | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2506.06725) | Curiosity-driven world modeling. |
| SimuRA | SimuRA | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2507.23773) | Simulative reasoning for planning. |

---

## 📚 Surveys & 💡 Position Papers

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| SLMs as Agentic AI 💡 | SLMs Survey | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2506.02153) | Argues for small LMs in agent systems. |
| LLM Agent Eval Survey 📚 | Eval Survey | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2503.16416) | Survey of agent evaluation methodologies. |

---

## Concluding Remarks

Reinforcement learning for AI agents is rapidly evolving. From self-evolving agents like **Alita**, to unified frameworks like **VerlTool**, to fine-grained trajectory calibration approaches such as **STeCa** and **SPA-RL**, the field is moving toward robust, adaptive systems.  

Multi-agent collaboration, budget-aware reasoning, and embodied agents are expanding the horizons of what RL-driven AI can achieve. This curated list highlights the latest methods, frameworks, and benchmarks for agentic AI.  

💡 *Pull requests welcome to keep this list up to date!*
