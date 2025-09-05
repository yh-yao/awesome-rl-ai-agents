# Awesome RL AI Agents

---

## 🔎 Quick Navigation



- [Agentic Workflow without Training](#agentic-workflow-without-training)
- [Agent Evaluation and Benchmarks](#agent-evaluation-and-benchmarks)
- [Agent Training Frameworks](#agent-training-frameworks)
- [RL for Single Agent](#rl-for-single-agent)
  - [Self-Evolution & Test-Time RL](#self-evolution--test-time-rl)
  - [RL for Tool Use & Agent Training](#rl-for-tool-use--agent-training)
  - [Alignment & Preference Optimization](#alignment--preference-optimization)
- [Cost-Aware Reasoning & Budget-Constrained RL](#cost-aware-reasoning--budget-constrained-rl)
- [RL for Multi-Agent Systems](#rl-for-multi-agent-systems)
  - [Planning](#planning)
  - [Collaboration](#collaboration)
- [Embodied Agents & World Models](#embodied-agents--world-models)
- [Emerging Perspectives & Position Papers](#emerging-perspectives--position-papers)
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
| Agent Leaderboard | Galileo Leaderboard | HuggingFace | 2024 | [Dataset](https://huggingface.co/datasets/galileo-ai/agent-leaderboard) | Tracks progress of top agents (e.g., Alita, AWorld) on GAIA tasks. |
| Agentic Predictor: Performance Prediction for Agentic Workflows via Multi-View Encoding | Agentic Predictor | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2505.19764) | Lightweight predictor of agentic workflow performance using multi-view encoding (code, prompts, interaction graph)—enables efficient evaluation and optimization. |


---

## Agent Training Frameworks

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| Agent Lightning: Train ANY AI Agents with Reinforcement Learning | Agent Lightning | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2508.03680) | A flexible RL framework ("LightningRL") that decouples agent execution from training via a unified MDP interface—integrates with any agent (e.g., LangChain, OpenAI Agents SDK) with minimal code changes. Supports hierarchical credit assignment and enables stable RL fine-tuning across diverse tasks like text-to-SQL, RAG, and math tool use. |
| SkyRL-v0: Train Real-World Long-Horizon Agents via Reinforcement Learning | SkyRL-v0 | ArXiv / GitHub | 2025 | [Blog & Notion](https://novasky-ai.notion.site/skyrl-v0) \| [Codebase](https://github.com/NovaSky-AI/SkyRL) | Modular online RL pipeline for LLM agents tackling long-horizon, multi-turn tool-use tasks (e.g., SWE-Bench). Built on VeRL and OpenHands, supports asynchronous rollouts and real-environment interactions, delivering 4-5× speedups and improved accuracy across 7B–14B models. |
| OpenManus-RL | OpenManus-RL | GitHub | 2025 | [Code & Docs](https://github.com/OpenManus/OpenManus-RL) \| [Dataset](https://huggingface.co/datasets/CharlieDreemur/OpenManus-RL) | Live-streamed RL tuning framework for LLM agents, collaboratively developed by UIUC-Ulab and MetaGPT. Incorporates benchmarks like GAIA and AgentBench, supports trajectory tuning and GRPO-based reinforcement strategies. |
| MASLab: A Unified and Comprehensive Codebase for LLM-based Multi-Agent Systems | MASLab | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2505.16988) | Open-source codebase integrating 20+ MAS methods with unified benchmarks and APIs—enables fair comparison and streamlined experimentation across models and tasks. |


---

## RL for Single Agent

### Self-Evolution & Test-Time RL

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| TTRL: Test-Time Reinforcement Learning | TTRL | ICLR | 2025 | [Paper](https://arxiv.org/abs/2504.16084) | Trains LLMs at inference time using majority-vote rewards. |
| ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models | ProRL | ICLR | 2025 | [Paper](https://arxiv.org/abs/2505.24864) | Introduces KL-control, reference resets; expands reasoning boundaries. |
| A Survey of Self-Evolving Agents: On Path to Artificial Super Intelligence | Self-Evolving Agents Survey | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2507.21046) | Reviews methods for evolving models, memories, and tools. |
| RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning | RAGEN / StarPO | ICLR | 2025 | [Paper](https://arxiv.org/abs/2504.20073) | Multi-turn RL with trajectory filtering and critic incorporation. |
| Alita: Generalist Agent Enabling Scalable Agentic Reasoning with Minimal Predefinition and Maximal Self-Evolution | Alita | GAIA LB | 2025 | [Paper](https://arxiv.org/abs/2505.20286) | Self-evolution with minimal predefinition via MCP modules. |
| Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement | Gödel Agent | ACL / arXiv | 2024–2025 | [Paper](https://arxiv.org/abs/2410.04444) | Self-evolving LLM agent inspired by the Gödel machine, modifies its own code and logic recursively via prompts—yields continuous self-improvement. |
| Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents | Darwin Godel Machine | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.22954) | Self-improving LLM system that rewrites its own code via Darwinian open-ended exploration and validates improvements on coding benchmarks—continual innovation and improved performance. |
| SkyRL-v0: Train Real-World Long-Horizon Agents via Reinforcement Learning | SkyRL-v0 | ArXiv Blog / GitHub | 2025 | [Blog & Notion](https://novasky-ai.notion.site/skyrl-v0) \| [Codebase](https://github.com/NovaSky-AI/SkyRL) | Modular online RL pipeline for LLM agents tackling long-horizon, multi-turn tool-use tasks (e.g., SWE-Bench). Built on VeRL and OpenHands, it supports asynchronous rollouts and real-environment interactions, delivering 4-5× speedups and improved accuracy across 7B–14B models (e.g., 11.0 % → 14.6 %) :contentReference[oaicite:0]{index=0}. |




---

### RL for Tool Use & Agent Training

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| AGILE: A Novel Reinforcement Learning Framework of LLM Agents | AGILE | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2405.14751) | RL with memory, tool use, and expert consultation. |
| Offline Training of Language Model Agents with Functions as Learnable Weights | AgentOptimizer | ICML | 2024 | [Paper](https://arxiv.org/abs/2402.11359) | Functions treated as learnable weights; supports rollback & early stop. |
| FireAct: Toward Language Agent Fine-tuning | FireAct | Arxiv | 2023 | [Paper](https://arxiv.org/abs/2310.05915) | Multi-task fine-tuning on agent trajectories. |
| ToRL: Scaling Tool-Integrated RL | ToRL | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2503.23383) | RL framework for exploring tool-use strategies. |
| ToolRL: Reward is All Tool Learning Needs | ToolRL | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2504.13958) | Reward-design study for tool use. |
| Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning | ARTIST | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.01441) | Unified reasoning + tool integration via RL. |
| Agent RL Scaling Law: Agent RL with Spontaneous Code Execution for Mathematical Problem Solving | Agent RL Scaling Law / ZeroTIR | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.07773) | Examines scaling of tool-integrated RL. |
| Acting Less is Reasoning More! Teaching Model to Act Efficiently | OTC | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2504.14870) | Reduces tool calls with Optimal Tool Call policy. |
| WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning | WebAgent-R1 | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.16421) | End-to-end multi-turn RL for web environments. |
| Group-in-Group Policy Optimization for LLM Agent Training | GiGPO | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.10978) | Hierarchical RL with fine-grained step-level rewards. |
| Nemotron-Research-Tool-N1: Exploring Tool-Using Language Models with Reinforced Reasoning | Nemotron-Tool-N1 | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.00024) | Pure RL for tool calls; rivals SFT+RL pipelines. |
| CATP-LLM: Empowering Large Language Models for Cost-Aware Tool Planning | CATP-LLM | ICCV / arXiv | 2024–2025 | [Paper](https://arxiv.org/abs/2411.16313) \| [Code & Dataset](https://github.com/duowuyms/OpenCATP-LLM) | Introduces CATP-LLM, enabling LLMs to perform non-sequential tool planning with cost consideration via TPL and CAORL; evaluated on OpenCATP dataset using the QoP metric—improves plan quality by ~30% while reducing cost by up to ~46%. |
| Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning | Tool-Star | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2505.16410) \| [Code](https://github.com/<repourl>) | RL-based framework enabling LLMs to autonomously invoke and collaborate across multiple external tools; builds trajectories via tool-integrated prompting and employs self-critic RL with hierarchical rewards. Validated on 10+ reasoning benchmarks with notable improvements. |


### Alignment & Preference Optimization

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization | MODPO | ArXiv | 2023 (v4: Aug 2024) | [Paper](https://arxiv.org/abs/2310.03708) | Extends Direct Preference Optimization (DPO) to multi-objective alignment without RL. Produces Pareto-optimal models, matching MORLHF performance at ~3× lower compute with improved stability. |



---

## Cost-Aware Reasoning & Budget-Constrained RL

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| Cost-Augmented Monte Carlo Tree Search for LLM-Assisted Planning | CATS | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.14656) | Cost-augmented MCTS for LLM planning under budgets. |
| Token-Budget-Aware LLM Reasoning | TALE | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2412.18547) | Token-budget-aware reasoning policies. |
| FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance | FrugalGPT | Arxiv | 2023 | [Paper](https://arxiv.org/abs/2305.05176) | Minimises API costs while maintaining accuracy. |
| Efficient Contextual LLM Cascades through Budget-Constrained Policy Learning | TREACLE | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2404.13082) | Contextual cascades with budget constraints. |
| BudgetMLAgent: A Cost-Effective LLM Multi-Agent System for Automating Machine Learning Tasks | BudgetMLAgent | AIMLSystems | 2025 | — | Multi-agent LLM architecture for automating ML workflows with an emphasis on cost efficiency. |
| The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective | Cost of Dynamic Reasoning | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2506.04301) | First system-level evaluation of dynamic, multi-step reasoning agents—measures compute cost, latency, energy, and infrastructure impact; finds diminishing accuracy returns and scalability concerns, prompting the need for cost-aware designs. |
| Reasoning in Token Economies: Budget-Aware Evaluation of LLM Reasoning Strategies | Budget-Aware Evaluation | EMNLP | 2024 | [Paper](https://aclanthology.org/2024.emnlp-main.1112/) | Proposes a budget-aware evaluation framework that incorporates queries, tokens, and cost; finds that self-consistency often outperforms complex reasoning strategies like Reflexion or multi-agent debate when compute is matched. |
| Large Language Model Cascades with Mixture of Thoughts Representations for Cost-Efficient Reasoning | LLM MoT Cascade | ICLR / ArXiv | 2024 | [Paper](https://arxiv.org/abs/2310.03094) \| [Code](https://github.com/MurongYue/LLM_MoT_cascade) | Uses a cascade of LLMs—cheaper model handles easy tasks, defers harder ones to stronger LLM—leveraging answer consistency and mixed thought representations (CoT + PoT). Achieves similar reasoning performance at 40–60 % lower cost. |





---

## RL for Multi-Agent Systems

### Planning

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation | OWL (Optimized Workforce Learning) | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.23885) | Hierarchical planner + worker agents for task automation. |
| Profile-Aware Maneuvering: A Dynamic Multi-Agent System for Robust GAIA Problem Solving by AWorld | AWorld | NeurIPS | 2024 | [Paper](https://arxiv.org/abs/2508.09889) | Dynamic multi-agent system with guard agent for stable reasoning. |
| Plan-over-Graph: Towards Parallelable LLM Agent Schedule | Plan-over-Graph | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2502.14563) | Parallel sub-task generation via graph scheduling. |
| LLM-based Multi-Agent Reinforcement Learning: Current and Future Directions | Multi-Agent RL Survey | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2405.11106) | Reviews coordination/communication challenges in multi-agent RL. |
| Self-Resource Allocation in Multi-Agent LLM Systems | Self-Resource Allocation | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2504.02051) | Compares orchestrator vs planner strategies for task allocation among LLM agents—planners outperform orchestrators in efficiency; worker capability transparency further improves performance. |
| MASLab: A Unified and Comprehensive Codebase for LLM-based Multi-Agent Systems | MASLab | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2505.16988) | Open-source codebase integrating 20+ MAS methods with unified benchmarks and APIs—enables fair comparison and streamlined experimentation across models and tasks. |



---

### Collaboration

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| ACC-Collab: An Actor-Critic Approach to Multi-Agent LLM Collaboration | ACC-Collab | ICLR | 2025 | [Paper](https://openreview.net/forum?id=nfKfAzkiez) | Trains an actor-agent and critic-agent (instantiate from the same LLM) jointly via actor-critic RL to improve collaborative reasoning — outperforms state-of-the-art multi-agent techniques. |
| Chain of Agents: Large Language Models Collaborating on Long-Context Tasks | Chain of Agents | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2406.02818) | Long-context tasks solved via agent communication. |
| Scaling Large Language Model-based Multi-Agent Collaboration | Scaling LLM Multi-Agent Collaboration | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2406.07155) | Scaling analysis for multi-agent collaborations. |
| MMAC-Copilot: Multi-modal Agent Collaboration Operating Copilot | MMAC-Copilot | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2404.18074) | Multi-modal copilot collaboration among agents. |
| Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning | CORY | NeurIPS | 2024 | [Paper](https://arxiv.org/abs/2410.06101) \| [OpenReview](https://openreview.net/forum?id=OoOCoZFVK3) \| [Code](https://github.com/Harry67Hu/CORY) | Introduces *CORY*, where two instances of the same LLM (pioneer and observer) cooperate via sequential reinforcement learning, periodically swapping roles. Improves stability and robustness over PPO baselines. |
| OpenManus-RL: Live-Streamed RL Tuning Framework for LLM Agents | OpenManus-RL | GitHub | 2025 | [Code & Docs](https://github.com/OpenManus/OpenManus-RL) \| [Dataset](https://huggingface.co/datasets/CharlieDreemur/OpenManus-RL) | Live-streamed RL tuning framework for LLM agents, collaboratively developed by UIUC-Ulab and MetaGPT. Incorporates benchmarks like GAIA and AgentBench, supports trajectory tuning and GRPO-based reinforcement strategies. |
| MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning | MAPoRL | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2502.18439) | Trains LLMs jointly via multi-agent RL post-training. Agents generate, discuss, and co-refine responses, guided by a verifier reward—boosts collaborative behavior and generalization compared to solo training. |
| ACC-Collab: An Actor-Critic Approach to Multi-Agent LLM Collaboration | ACC-Collab | ArXiv / ICLR 2025 | 2024–2025 | [Paper](https://arxiv.org/abs/2411.00053) \| [Code](https://github.com/LlenRotse/ACC-Collab) | Trains two LLM agents—actor and critic—in an actor-critic framework via guided collaboration trajectories to explicitly learn multi-agent collaboration. Outperforms state-of-the-art multi-agent methods across diverse benchmarks. |


---

## Embodied Agents & World Models

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| An Embodied Generalist Agent in 3D World | LEO | ICML | 2024 | [Paper](https://arxiv.org/abs/2311.12871) | Introduces **LEO**, a multimodal, embodied agent trained via 3D vision-language alignment and instruction tuning; performs captioning, QA, navigation, reasoning, and manipulation in 3D environments. |
| DreamerV3: Mastering Diverse Domains through World Models | DreamerV3 | arXiv | 2023 | [Paper](https://arxiv.org/abs/2301.04104) | A general world-model-based RL algorithm that imagines future trajectories; excels across 150+ tasks without hyperparameter tuning, including collecting diamonds in Minecraft—demonstrating universal imagined planning. |
| World-model-augmented Web Agent | WMA Web Agent | ICLR / arXiv | 2025 | [Paper](https://arxiv.org/abs/2410.13232) | Introduces a web agent that simulates outcomes (world model) before acting; uses transition-focused observation abstraction for better decision-making, improving cost and time efficiency on WebArena + Mind2Web. |
| WorldCoder: Model-Based LLM Agent | WorldCoder | arXiv | 2024 | [Paper](https://arxiv.org/abs/2402.12275) | Builds a Python-based world model from interactions; achieves higher sample- and compute-efficiency than deep RL or ReAct agents, and enables transfer across environments via code editing. |
| WALL-E 2.0: World Alignment via NeuroSymbolic Learning | WALL-E 2.0 | arXiv | 2025 | [Paper](https://arxiv.org/abs/2504.15785) | Aligns LLM world models with environment via symbolic rule learning; supports efficient MPC planning for LLM agents, boosting success in open-world tasks like Minecraft and ALFWorld. |
| WorldLLM: Curiosity-Driven World Modeling | WorldLLM | arXiv | 2025 | [Paper](https://arxiv.org/abs/2506.06725) | Enhances LLM-based world models using Bayesian inference and curiosity-driven RL; iteratively generates and refines hypotheses to improve predictions and interpretability in textual game environments. |
| SimuRA: Simulative Reasoning Architecture with World Model | SimuRA | arXiv | 2025 | [Paper](https://arxiv.org/abs/2507.23773) | A goal-oriented agent using a world model built on LLM for planning via mental simulation; shows substantial performance gains (e.g., flight search success rises from 0% to 32.2%) and vastly outperforms autoregressive planning. |


---

## Emerging Perspectives & Position Papers

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| Small Language Models are the Future of Agentic AI | SLMs as Agentic AI | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2506.02153) | Advocates using small language models (SLMs) in agentic systems for efficiency and cost savings; proposes heterogeneous setups and an LLM-to-SLM conversion pipeline. |
| Survey on Evaluation of LLM-based Agents | LLM Agent Eval Survey | ArXiv | 2025 | [Paper](https://arxiv.org/abs/2503.16416) | First comprehensive survey of evaluation methodologies for LLM-based agents—covers core capabilities, diverse benchmarks, and evaluation frameworks; highlights gaps in cost-efficiency, safety, and robustness. |

---


---

## Concluding Remarks

RL for AI agents is a rapidly evolving area. Agents must reason, plan and act in open-ended environments while balancing costs and leveraging external tools. This curated list highlights recent progress across self-evolution, tool-augmented RL, multi-agent collaboration and cost-aware planning.  

As new benchmarks and algorithms appear, contributions such as **Alita**, **AGILE**, and **ToRL** demonstrate that reinforcement learning can unlock powerful new behaviours.
