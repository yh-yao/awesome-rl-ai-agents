# Awesome RL AI Agents

---

## 🔎 Quick Navigation

- 🧠✨ [Agentic Workflow without Training](#agentic-workflow-without-training)
- 🧪📊 [Agent Evaluation and Benchmarks](#agent-evaluation-and-benchmarks)
- 🧰⚙️ [Agent Training Frameworks](#agent-training-frameworks)
- 👤🧭 [RL for Single Agent](#rl-for-single-agent)
  - 🔁🧪 [Self-Evolution & Test-Time RL](#self-evolution--test-time-rl)
  - 🛠️🧠 [RL for Tool Use & Agent Training](#rl-for-tool-use--agent-training)
  - 🎛️🎯 [Alignment & Preference Optimization](#alignment--preference-optimization)
- 💸🧠 [Cost-Aware Reasoning & Budget-Constrained RL](#cost-aware-reasoning--budget-constrained-rl)
- 👥🤝 [RL for Multi-Agent Systems](#rl-for-multi-agent-systems)
  - 🗺️📅 [Planning](#planning)
  - 🤝🧩 [Collaboration](#collaboration)
- 🤖🌍 [Embodied Agents & World Models](#embodied-agents--world-models)
- 📚💡 [Surveys & Position Papers](#emerging-perspectives--position-papers)
- 🧾✅ [Concluding Remarks](#concluding-remarks)

---

Reinforcement learning (RL) is rapidly becoming a driving force behind AI agents that can reason, act and adapt in the real world. Large language models (LLMs) provide a powerful prior for reasoning, but without feedback they remain static and brittle. RL enables agents to learn from interaction – whether it’s via self-reflection, outcome-based rewards or interacting with humans and tools.  

The goal of this repository is to curate up-to-date resources on RL for AI agents, focusing on three axes:

- **Agentic workflows without training** – prompting strategies that improve reasoning without fine-tuning.  
- **Evaluation and benchmarks** – systematic tests for reasoning, tool use, and task automation.  
- **RL for single and multi-agent systems** – enabling self-evolution, efficient tool use, and collaboration.

Tables give a quick overview; detailed descriptions follow in the text.

---

## Agentic Workflow without Training

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| Tree of Thoughts: Deliberate Problem Solving with Large Language Models | ToT | ICML | 2023 | [Paper](https://arxiv.org/abs/2305.10601) | Explores multiple reasoning paths via search trees, significantly boosting problem-solving accuracy. |
| Reflexion: Language Agents with Verbal Reinforcement Learning | Reflexion | NeurIPS | 2023 | [Paper](https://arxiv.org/abs/2303.11366) | Agents critique and refine their own outputs, reducing hallucinations. |
| Self-Refine: Iterative Refinement with Self-Feedback | Self-Refine | NeurIPS | 2023 | [Paper](https://arxiv.org/abs/2303.17651) | Iteratively refines LLM outputs through self-feedback without extra training. |
| ReAct: Synergizing Reasoning and Acting in Language Models | ReAct | ICLR | 2023 | [Paper](https://arxiv.org/abs/2210.03629) | Interleaves reasoning and acting; foundation for tool-augmented agents. |
| SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks | SwiftSage | ACL | 2023 | [Paper](https://arxiv.org/abs/2305.17390) | Separates planning and solving models for complex interactive tasks. |
| DynaSaur: Large Language Agents Beyond Predefined Actions | DynaSaur | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2411.01747) | Dynamic action spaces for LLM agents, beyond predefined tool calls. |

---

## Agent Evaluation and Benchmarks

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| GAIA: a benchmark for General AI Assistants | GAIA | Arxiv | 2023 | [Paper](https://arxiv.org/abs/2311.12983) | 466 diverse tasks requiring reasoning, multimodal understanding, and web interaction. |
| TaskBench: Benchmarking Large Language Models for Task Automation | TaskBench | EMNLP | 2023 | [Paper](https://arxiv.org/abs/2311.18760) | Evaluates task automation: decomposition, tool selection, parameter prediction. |
| AgentBench: Evaluating LLMs as Agents | AgentBench | Arxiv | 2023 | [Paper](https://arxiv.org/abs/2308.03688) | Covers 51 diverse scenarios from information seeking to interactive tasks. |
| ACEBench: Who Wins the Match Point in Tool Usage? | ACEBench | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2501.12851) | Fine-grained evaluation of tool usage and error analysis. |
| Agent Leaderboard | Galileo LB | HuggingFace | 2024 | [Dataset](https://huggingface.co/datasets/galileo-ai/agent-leaderboard) | Tracks progress of top agents (e.g., Alita, AWorld) on GAIA tasks. |
| Agentic Predictor: Performance Prediction for Agentic Workflows via Multi-View Encoding | Agentic Predictor | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2505.19764) | Lightweight predictor of workflow performance using multi-view encoding. |

---

## Agent Training Frameworks

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| Agent Lightning: Train ANY AI Agents with Reinforcement Learning | Agent Lightning | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2508.03680) | Flexible RL framework decoupling execution from training via a unified MDP interface. |
| SkyRL-v0: Train Real-World Long-Horizon Agents via Reinforcement Learning 🔁 | SkyRL-v0 | ArXiv / GitHub | 2025 | [Blog](https://novasky-ai.notion.site/skyrl-v0) \| [Code](https://github.com/NovaSky-AI/SkyRL) | Modular online RL pipeline for long-horizon multi-turn tool-use tasks. *(Also listed under [Self-Evolution & Test-Time RL](#self-evolution--test-time-rl))* |
| OpenManus-RL: Live-Streamed RL Tuning Framework for LLM Agents 🔁 | OpenManus-RL | GitHub | 2025 | [Code](https://github.com/OpenManus/OpenManus-RL) \| [Dataset](https://huggingface.co/datasets/CharlieDreemur/OpenManus-RL) | Live-streamed RL tuning framework collaboratively developed by UIUC-Ulab and MetaGPT. *(Also listed under [Collaboration](#collaboration))* |
| MASLab: A Unified and Comprehensive Codebase for LLM-based Multi-Agent Systems | MASLab | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2505.16988) | Open-source codebase integrating 20+ MAS methods with unified benchmarks and APIs. |

---

## RL for Single Agent

### Self-Evolution & Test-Time RL

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| Test-Time Reinforcement Learning | TTRL | ICLR | 2025 | [Paper](https://arxiv.org/abs/2504.16084) | Trains LLMs at inference time using majority-vote rewards. |
| Prolonged Reinforcement Learning Expands Reasoning Boundaries | ProRL | ICLR | 2025 | [Paper](https://arxiv.org/abs/2505.24864) | Introduces KL-control, reference resets; expands reasoning boundaries. |
| A Survey of Self-Evolving Agents 📚 | Self-Evolving Survey | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2507.21046) | Reviews methods for evolving models, memories, and tools. |
| RAGEN: Understanding Self-Evolution via Multi-Turn RL | RAGEN / StarPO | ICLR | 2025 | [Paper](https://arxiv.org/abs/2504.20073) | Multi-turn RL with trajectory filtering and critic incorporation. |
| Alita: Generalist Self-Evolving Agent | Alita | GAIA LB | 2025 | [Paper](https://arxiv.org/abs/2505.20286) | Scalable agent reasoning with minimal predefinition via MCP modules. |
| Gödel Agent: Recursive Self-Improvement | Gödel Agent | ACL / ArXiv | 2024–2025 | [Paper](https://arxiv.org/abs/2410.04444) | Self-evolving agent inspired by Gödel machine; modifies own code recursively. |
| Darwin Gödel Machine | Darwin GM | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.22954) | Open-ended self-improvement via Darwinian exploration; validates on coding benchmarks. |
| SkyRL-v0 🔁 | SkyRL-v0 | ArXiv / GitHub | 2025 | [Blog](https://novasky-ai.notion.site/skyrl-v0) \| [Code](https://github.com/NovaSky-AI/SkyRL) | RL pipeline for long-horizon tool-use. *(Also listed under [Agent Training Frameworks](#agent-training-frameworks))* |

---

### RL for Tool Use & Agent Training

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| AGILE: A Novel Reinforcement Learning Framework of LLM Agents | AGILE | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2405.14751) | RL with memory, tool use, and expert consultation. |
| Offline Training of Language Model Agents with Functions as Learnable Weights | AgentOptimizer | ICML | 2024 | [Paper](https://arxiv.org/abs/2402.11359) | Functions as learnable weights; supports rollback & early stop. |
| FireAct: Toward Language Agent Fine-tuning | FireAct | Arxiv | 2023 | [Paper](https://arxiv.org/abs/2310.05915) | Multi-task fine-tuning on agent trajectories. |
| ToRL: Scaling Tool-Integrated RL | ToRL | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2503.23383) | RL framework for exploring tool-use strategies. |
| ToolRL: Reward is All Tool Learning Needs | ToolRL | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2504.13958) | Reward-design study for tool use. |
| Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning | ARTIST | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.01441) | Unified reasoning + tool integration via RL. |
| Agent RL Scaling Law: Agent RL with Spontaneous Code Execution | ZeroTIR | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.07773) | Scaling analysis of tool-integrated RL. |
| Acting Less is Reasoning More! Teaching Model to Act Efficiently | OTC | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2504.14870) | Reduces tool calls with optimal tool-call policy. |
| WebAgent-R1: Training Web Agents via End-to-End Multi-Turn RL | WebAgent-R1 | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.16421) | End-to-end multi-turn RL for web environments. |
| Group-in-Group Policy Optimization for LLM Agent Training | GiGPO | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.10978) | Hierarchical RL with step-level rewards. |
| Nemotron-Research-Tool-N1: Exploring Tool-Using LMs with Reinforced Reasoning | Nemotron-Tool-N1 | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.00024) | Pure RL for tool calls; rivals SFT+RL pipelines. |
| CATP-LLM: Cost-Aware Tool Planning | CATP-LLM | ICCV / arXiv | 2024–2025 | [Paper](https://arxiv.org/abs/2411.16313) \| [Code & Dataset](https://github.com/duowuyms/OpenCATP-LLM) | Non-sequential tool planning with cost via TPL & CAORL; reduces cost up to ~46%. |
| Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via RL | Tool-Star | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2505.16410) \| [Code](https://github.com/<repourl>) | Multi-tool trajectories + self-critic RL with hierarchical rewards. |

---

### Alignment & Preference Optimization

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| Beyond One-Preference-Fits-All Alignment: Multi-Objective DPO | MODPO | ArXiv | 2023 (v4: 2024) | [Paper](https://arxiv.org/abs/2310.03708) | Extends DPO to multi-objective alignment; Pareto-optimal and stable. |

---

## Cost-Aware Reasoning & Budget-Constrained RL

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| Cost-Augmented Monte Carlo Tree Search for LLM-Assisted Planning | CATS | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.14656) | Cost-augmented MCTS for planning under budgets. |
| Token-Budget-Aware LLM Reasoning | TALE | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2412.18547) | Token-budget-aware reasoning policies. |
| FrugalGPT: Use LLMs While Reducing Cost | FrugalGPT | Arxiv | 2023 | [Paper](https://arxiv.org/abs/2305.05176) | Minimizes API costs while maintaining accuracy. |
| Efficient Contextual LLM Cascades via Budget-Constrained Policy Learning | TREACLE | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2404.13082) | Budget-constrained cascades for context usage. |
| BudgetMLAgent: Cost-Effective Multi-Agent System for ML Automation | BudgetMLAgent | AIMLSystems | 2025 | — | Multi-agent ML automation with cost efficiency. |
| The Cost of Dynamic Reasoning: Systems View | Cost of Dynamic Reasoning | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2506.04301) | System-level evaluation of multi-step agents; cost/latency/energy tradeoffs. |
| Reasoning in Token Economies: Budget-Aware Evaluation | Budget-Aware Evaluation | EMNLP | 2024 | [Paper](https://aclanthology.org/2024.emnlp-main.1112/) | Budget-aware evaluation; self-consistency often beats complex strategies. |
| LLM Cascades with Mixture of Thoughts for Cost-Efficient Reasoning | LLM MoT Cascade | ICLR / ArXiv | 2024 | [Paper](https://arxiv.org/abs/2310.03094) \| [Code](https://github.com/MurongYue/LLM_MoT_cascade) | Cascaded models with mixed thought reps; 40–60% lower cost. |

---

## RL for Multi-Agent Systems

### Planning

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| OWL: Optimized Workforce Learning for Real-World Automation | OWL | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.23885) | Hierarchical planner + workers for task automation. |
| Profile-Aware Maneuvering for GAIA by AWorld | AWorld | NeurIPS | 2024 | [Paper](https://arxiv.org/abs/2508.09889) | Dynamic multi-agent with guard agent for stable reasoning. |
| Plan-over-Graph: Towards Parallelable Agent Schedule | Plan-over-Graph | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2502.14563) | Parallel sub-task generation via graph scheduling. |
| LLM-based Multi-Agent Reinforcement Learning: Directions | MARL Survey 📚 | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2405.11106) | Survey on coordination/communication in multi-agent RL. |
| Self-Resource Allocation in Multi-Agent LLM Systems | Self-Resource Allocation | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2504.02051) | Planner vs orchestrator strategies; transparency helps. |
| MASLab: Unified Codebase for LLM-based MAS 🔁 | MASLab | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2505.16988) | Unified benchmarks & APIs. *(Also listed under [Agent Training Frameworks](#agent-training-frameworks))* |

---

### Collaboration

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| ACC-Collab: Actor-Critic for Multi-Agent Collaboration | ACC-Collab | ICLR | 2025 | [Paper](https://openreview.net/forum?id=nfKfAzkiez) | Actor + critic agents trained jointly; improves collaboration. |
| Chain of Agents: Collaborating on Long-Context Tasks | Chain of Agents | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2406.02818) | Long-context tasks via agent communication. |
| Scaling LLM-based Multi-Agent Collaboration | Scaling LLM MAC | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2406.07155) | Scaling analysis for multi-agent collaborations. |
| MMAC-Copilot: Multi-modal Agent Collaboration | MMAC-Copilot | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2404.18074) | Multi-modal copilot collaboration. |
| CORY: Sequential Cooperative Multi-Agent RL | CORY | NeurIPS | 2024 | [Paper](https://arxiv.org/abs/2410.06101) \| [OpenReview](https://openreview.net/forum?id=OoOCoZFVK3) \| [Code](https://github.com/Harry67Hu/CORY) | Two instances of the same LLM co-train via role swapping; stable vs PPO. |
| OpenManus-RL 🔁 | OpenManus-RL | GitHub | 2025 | [Code](https://github.com/OpenManus/OpenManus-RL) \| [Dataset](https://huggingface.co/datasets/CharlieDreemur/OpenManus-RL) | Live-streamed RL tuning framework. *(Also listed under [Agent Training Frameworks](#agent-training-frameworks))* |
| MAPoRL: Multi-Agent Post-Co-Training with RL | MAPoRL | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2502.18439) | Agents generate, discuss, co-refine; verifier reward guides training. |
| ACC-Collab (ArXiv preprint) | ACC-Collab | ArXiv / ICLR 2025 | 2024–2025 | [Paper](https://arxiv.org/abs/2411.00053) \| [Code](https://github.com/LlenRotse/ACC-Collab) | Guided collaboration trajectories; SOTA across benchmarks. |

---

## Embodied Agents & World Models

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| An Embodied Generalist Agent in 3D World | LEO | ICML | 2024 | [Paper](https://arxiv.org/abs/2311.12871) | Multimodal embodied agent for 3D tasks (captioning, QA, nav, manipulation). |
| DreamerV3: Mastering Diverse Domains through World Models | DreamerV3 | arXiv | 2023 | [Paper](https://arxiv.org/abs/2301.04104) | World-model RL with imagined trajectories; robust across 150+ tasks. |
| World-model-augmented Web Agent | WMA Web Agent | ICLR / arXiv | 2025 | [Paper](https://arxiv.org/abs/2410.13232) | Simulates outcomes before acting; more efficient on WebArena + Mind2Web. |
| WorldCoder: Model-Based LLM Agent | WorldCoder | arXiv | 2024 | [Paper](https://arxiv.org/abs/2402.12275) | Builds Python world models from interactions; transfers via code editing. |
| WALL-E 2.0: World Alignment via NeuroSymbolic Learning | WALL-E 2.0 | arXiv | 2025 | [Paper](https://arxiv.org/abs/2504.15785) | Neuro-symbolic alignment for MPC planning (Minecraft, ALFWorld). |
| WorldLLM: Curiosity-Driven World Modeling | WorldLLM | arXiv | 2025 | [Paper](https://arxiv.org/abs/2506.06725) | Bayesian curiosity to refine world models; textual games focus. |
| SimuRA: Simulative Reasoning Architecture with World Model | SimuRA | arXiv | 2025 | [Paper](https://arxiv.org/abs/2507.23773) | Goal-oriented agent with mental simulation; big gains in flight search. |

---

## 📚 Surveys & 💡 Position Papers

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| Small Language Models are the Future of Agentic AI 💡 | SLMs as Agentic AI | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2506.02153) | Advocates for SLMs in agentic systems for efficiency and cost savings. |
| Survey on Evaluation of LLM-based Agents 📚 | LLM Agent Eval Survey | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2503.16416) | Comprehensive survey of evaluation methodologies for LLM-based agents. |

---

## Concluding Remarks

RL for AI agents is a rapidly evolving area. Agents must reason, plan and act in open-ended environments while balancing costs and leveraging external tools. This curated list highlights recent progress across self-evolution, tool-augmented RL, multi-agent collaboration and cost-aware planning.  

As new benchmarks and algorithms appear, contributions such as **Alita**, **AGILE**, and **ToRL** demonstrate that reinforcement learning can unlock powerful new behaviours.  

💡 *Pull requests are welcome to keep this list updated with the latest papers, codebases, and benchmarks.*
