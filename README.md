# Awesome RL AI Agents

---

## 🔎 Quick Navigation

- [Agentic Workflow without Training](#agentic-workflow-without-training)
- [Agent Evaluation and Benchmarks](#agent-evaluation-and-benchmarks)
- [RL for Single Agent](#rl-for-single-agent)
  - [Self-Evolution & Test-Time RL](#self-evolution--test-time-rl)
  - [RL for Tool Use & Agent Training](#rl-for-tool-use--agent-training)
- [Cost-Aware Reasoning & Budget-Constrained RL](#cost-aware-reasoning--budget-constrained-rl)
- [RL for Multi-Agent Systems](#rl-for-multi-agent-systems)
  - [Planning](#planning)
  - [Collaboration](#collaboration)
- [Concluding Remarks](#concluding-remarks)

---

Reinforcement learning (RL) is rapidly becoming a driving force behind AI agents that can reason, act and adapt in the real world. Large language models (LLMs) provide a powerful prior for reasoning, but without feedback they remain static and brittle. RL enables agents to learn from interaction – whether it’s via self-reflection, outcome-based rewards or interacting with humans and tools.  

The goal of this repository is to curate up-to-date resources on RL for AI agents, focusing on three axes:

- **Agentic workflows without training** – prompting strategies that improve reasoning without fine-tuning.  
- **Evaluation and benchmarks** – systematic tests for reasoning, tool use, and task automation.  
- **RL for single and multi-agent systems** – enabling self-evolution, efficient tool use, and collaboration.

Tables give a quick overview; detailed descriptions follow in the text.

---

## Agentic Workflow without Training

These methods improve planning and reasoning using clever prompting or algorithmic structures rather than additional training. They often serve as initial policies for RL fine-tuning.

| Title | Venue | Year | Materials | Description |
|:-----:|:-----:|:----:|:---------:|:-----------|
| Tree of Thoughts (ToT) | ICML | 2023 | [Paper](https://arxiv.org/abs/2305.10601) | Explores multiple reasoning paths via search trees, significantly boosting problem-solving accuracy. |
| Reflexion / Self-Refine | NeurIPS | 2023 | [Paper](https://arxiv.org/abs/2303.11366) | Agents critique and refine their own outputs, reducing hallucinations. |
| ReAct | ICLR | 2023 | [Paper](https://arxiv.org/abs/2210.03629) | Interleaves reasoning and acting; foundation for tool-augmented agents. |
| SwiftSage | ACL | 2023 | [Paper](https://arxiv.org/pdf/2305.17390) | Separates planning and solving models for complex interactive tasks. |
| DynaSaur | Arxiv | 2024 | [Paper](https://arxiv.org/pdf/2411.01747) | Dynamic action spaces for LLM agents, beyond predefined tool calls. |

**Explanations**  
- **Tree-of-Thought (ToT):** Generalises chain-of-thought prompting into a search process. Demonstrates large gains on math puzzles (74 % vs 4 %).  
- **Reflexion:** Adds self-critique cycles, widely used in tool-augmented reasoning.  
- **ReAct:** Combined thinking and acting steps, critical for later RL work.  
- **SwiftSage:** Introduces modular planner/solver separation.  
- **DynaSaur:** Aims for flexibility by letting agents dynamically generate new actions.

---

## Agent Evaluation and Benchmarks

Choosing the right benchmark is crucial for measuring progress.

| Title | Venue | Year | Materials | Description |
|:-----:|:-----:|:----:|:---------:|:-----------|
| GAIA | Arxiv | 2023 | [Paper](https://arxiv.org/pdf/2311.12983) | 466 diverse tasks requiring reasoning, multimodal understanding, and web interaction. |
| TaskBench | EMNLP | 2023 | [Paper](https://arxiv.org/pdf/2311.18760) | Evaluates task automation: decomposition, tool selection, parameter prediction. |
| AgentBench | Arxiv | 2023 | [Paper](https://arxiv.org/pdf/2308.03688) | Covers 51 diverse scenarios from information seeking to interactive tasks. |
| ACEBench | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2501.12851) | Fine-grained evaluation of tool usage and error analysis. |
| Galileo Leaderboard | HuggingFace | 2024 | [Dataset](https://huggingface.co/datasets/galileo-ai/agent-leaderboard) | Tracks progress of top agents (e.g., Alita, AWorld) on GAIA tasks. |

**Explanations**  
- **GAIA:** Generalist benchmark; human ≈ 92 % vs GPT-4 plugins ≈ 15 %.  
- **TaskBench:** Graph-based evaluation of automation workflows.  
- **AgentBench:** Widely used, multi-domain coverage.  
- **ACEBench:** Analyzes different categories of tool use.  
- **Leaderboards:** Provide live comparisons and ranking.

---

## RL for Single Agent

### Self-Evolution & Test-Time RL

| Title | Venue | Year | Materials | Description |
|:-----:|:-----:|:----:|:---------:|:-----------|
| TTRL | ICLR | 2025 | [Paper](https://arxiv.org/pdf/2504.16084) | Trains LLMs at inference time using majority-vote rewards. |
| ProRL | ICLR | 2025 | [Paper](https://arxiv.org/pdf/2505.24864) | Introduces KL-control, reference resets; expands reasoning boundaries. |
| Self-Evolving Agents Survey | Arxiv | 2025 | [Paper](https://arxiv.org/pdf/2507.21046) | Reviews methods for evolving models, memories, and tools. |
| RAGEN / StarPO | ICLR | 2025 | [Paper](https://arxiv.org/abs/2504.20073) | Multi-turn RL with trajectory filtering and critic incorporation. |
| Alita | GAIA LB | 2025 | [Paper](https://arxiv.org/pdf/2505.20286) | Self-evolution with minimal predefinition via MCP modules. |

**Explanations**  
- **TTRL:** Enables continual self-improvement using unlabeled data.  
- **ProRL:** Pushes agents to explore beyond baseline reasoning with KL control.  
- **Survey:** Categorises *what*, *when*, and *how* to evolve.  
- **RAGEN:** Tackles instability; StarPO variant improves reward reliability.  
- **Alita:** GAIA leader; simple modules generate context protocols.

---

### RL for Tool Use & Agent Training

| Title | Venue | Year | Materials | Description |
|:-----:|:-----:|:----:|:---------:|:-----------|
| AGILE | Arxiv | 2024 | [Paper](https://arxiv.org/pdf/2405.14751) | RL with memory, tool use, and expert consultation. |
| AgentOptimizer | ICML | 2024 | [Paper](https://arxiv.org/abs/2402.11359) | Functions treated as learnable weights; supports rollback & early stop. |
| FireAct | Arxiv | 2023 | [Paper](https://arxiv.org/abs/2310.05915) | Multi-task fine-tuning on agent trajectories. |
| ToRL | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2503.23383) | RL framework for exploring tool-use strategies. |
| ToolRL | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2504.13958) | Reward-design study for tool use. |
| ARTIST | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.01441) | Unified reasoning + tool integration via RL. |
| Agent RL Scaling Law / ZeroTIR | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.07773) | Examines scaling of tool-integrated RL. |
| OTC | Arxiv | 2025 | [Paper](https://arxiv.org/pdf/2504.14870) | Reduces tool calls with Optimal Tool Call policy. |
| WebAgent-R1 | Arxiv | 2025 | [Paper](https://arxiv.org/html/2505.16421v1) | End-to-end multi-turn RL for web environments. |
| GiGPO | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.10978) | Hierarchical RL with fine-grained step-level rewards. |
| Nemotron-Tool-N1 | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.00024) | Pure RL for tool calls; rivals SFT+RL pipelines. |

**Explanations**  
- **AGILE:** Integrates memory, tools, and experts under PPO; beats GPT-4 on QA tasks.  
- **AgentOptimizer:** Trains functions as weights; keeps base LLM frozen.  
- **FireAct:** Shows big gains from trajectory fine-tuning.  
- **ToRL / ToolRL:** Focus on how RL agents learn effective tool use.  
- **ARTIST:** RL agents learn tool use without step supervision.  
- **OTC:** Jointly optimises correctness + tool efficiency.  
- **WebAgent-R1:** Raises success on WebArena-Lite from 6.1 % → 33.9 %.  
- **GiGPO:** Improves ALFWorld/WebShop performance significantly.  
- **Nemotron-N1:** Pure RL rivaling fine-tuned pipelines.

---

## Cost-Aware Reasoning & Budget-Constrained RL

| Title | Venue | Year | Materials | Description |
|:-----:|:-----:|:----:|:---------:|:-----------|
| CATS | Arxiv | 2025 | [Paper](https://arxiv.org/html/2505.14656v1) | Cost-augmented MCTS for LLM planning under budgets. |
| TALE | Arxiv | 2024 | [Paper](https://arxiv.org/pdf/2412.18547) | Token-budget-aware reasoning policies. |
| FrugalGPT | Arxiv | 2023 | [Paper](https://arxiv.org/pdf/2305.05176) | Minimises API costs while maintaining accuracy. |
| TREACLE | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2404.13082) | Contextual cascades with budget constraints. |

**Explanations**  
- **CATS:** Introduces explicit cost constraints; balances feasibility vs cost.  
- **TALE / FrugalGPT:** Baselines for budget-aware evaluation.  
- **TREACLE:** Optimises reasoning cascades with constrained budgets.

---

## RL for Multi-Agent Systems

### Planning

| Title | Venue | Year | Materials | Description |
|:-----:|:-----:|:----:|:---------:|:-----------|
| OWL (Optimized Workforce Learning) | Arxiv | 2025 | [Paper](https://arxiv.org/pdf/2505.23885) | Hierarchical planner + worker agents for task automation. |
| AWorld | NeurIPS | 2024 | [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/1c2b1c8f7d317719a9ce32dd7386ba35-Paper-Conference.pdf) | Dynamic multi-agent system with guard agent for stable reasoning. |
| Plan-over-Graph | Arxiv | 2025 | [Paper](https://arxiv.org/pdf/2502.14563) | Parallel sub-task generation via graph scheduling. |
| Multi-Agent RL Survey | Arxiv | 2024 | [Paper](https://arxiv.org/pdf/2405.11106) | Reviews coordination/communication challenges in multi-agent RL. |

**Explanations**  
- **OWL:** Outperforms baselines on GAIA tasks using hierarchical planning.  
- **AWorld:** Stability through guard agents; GAIA benchmark leader.  
- **Plan-over-Graph:** Accelerates workflows by parallel scheduling.  
- **Survey:** Highlights complexity of multi-agent communication.

---

### Collaboration

| Title | Venue | Year | Materials | Description |
|:-----:|:-----:|:----:|:---------:|:-----------|
| ACC-Collab | OpenReview | 2024 | [Paper](https://openreview.net/pdf/8a3b2842348cfd4559cafe483c35e5cc89ca6da8.pdf) | Actor-critic framework for collaborative LLM agents. |
| Chain of Agents | Arxiv | 2024 | [Paper](https://arxiv.org/pdf/2406.02818) | Long-context tasks solved via agent communication. |
| Scaling LLM Multi-Agent Collaboration | Arxiv | 2024 | [Paper](https://arxiv.org/pdf/2406.07155) | Scaling analysis for multi-agent collaborations. |
| MMAC-Copilot | Arxiv | 2024 | [Paper](https://arxiv.org/pdf/2404.18074) | Multi-modal copilot collaboration among agents. |

**Explanations**  
- **ACC-Collab:** Critic guides actor agents; outperforms baselines.  
- **Chain of Agents / Scaling:** Structured communication matters more than sheer number of agents.  
- **MMAC-Copilot:** Explores multimodal cooperation.

---

## Concluding Remarks

RL for AI agents is a rapidly evolving area. Agents must reason, plan and act in open-ended environments while balancing costs and leveraging external tools. This curated list highlights recent progress across self-evolution, tool-augmented RL, multi-agent collaboration and cost-aware planning.  

As new benchmarks and algorithms appear, contributions such as **Alita**, **AGILE**, and **ToRL** demonstrate that reinforcement learning can unlock powerful new behaviours.
