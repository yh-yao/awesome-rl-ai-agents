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

---

## Cost-Aware Reasoning & Budget-Constrained RL

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| Cost-Augmented Monte Carlo Tree Search for LLM-Assisted Planning | CATS | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.14656) | Cost-augmented MCTS for LLM planning under budgets. |
| Token-Budget-Aware LLM Reasoning | TALE | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2412.18547) | Token-budget-aware reasoning policies. |
| FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance | FrugalGPT | Arxiv | 2023 | [Paper](https://arxiv.org/abs/2305.05176) | Minimises API costs while maintaining accuracy. |
| Efficient Contextual LLM Cascades through Budget-Constrained Policy Learning | TREACLE | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2404.13082) | Contextual cascades with budget constraints. |

---

## RL for Multi-Agent Systems

### Planning

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation | OWL (Optimized Workforce Learning) | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2505.23885) | Hierarchical planner + worker agents for task automation. |
| Profile-Aware Maneuvering: A Dynamic Multi-Agent System for Robust GAIA Problem Solving by AWorld | AWorld | NeurIPS | 2024 | [Paper](https://arxiv.org/abs/2508.09889) | Dynamic multi-agent system with guard agent for stable reasoning. |
| Plan-over-Graph: Towards Parallelable LLM Agent Schedule | Plan-over-Graph | Arxiv | 2025 | [Paper](https://arxiv.org/abs/2502.14563) | Parallel sub-task generation via graph scheduling. |
| LLM-based Multi-Agent Reinforcement Learning: Current and Future Directions | Multi-Agent RL Survey | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2405.11106) | Reviews coordination/communication challenges in multi-agent RL. |

---

### Collaboration

| Title | Short title | Venue | Year | Materials | Description |
|:-----:|:-----------:|:-----:|:----:|:---------:|:-----------|
| ACC-Collab: An Actor-Critic Approach to Multi-Agent LLM Collaboration | ACC-Collab | OpenReview | 2024 | [Paper](https://openreview.net/forum?id=8a3b2842348cfd4559cafe483c35e5cc89ca6da8) | Actor-critic framework for collaborative LLM agents. |
| Chain of Agents: Large Language Models Collaborating on Long-Context Tasks | Chain of Agents | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2406.02818) | Long-context tasks solved via agent communication. |
| Scaling Large Language Model-based Multi-Agent Collaboration | Scaling LLM Multi-Agent Collaboration | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2406.07155) | Scaling analysis for multi-agent collaborations. |
| MMAC-Copilot: Multi-modal Agent Collaboration Operating Copilot | MMAC-Copilot | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2404.18074) | Multi-modal copilot collaboration among agents. |

---

## Concluding Remarks

RL for AI agents is a rapidly evolving area. Agents must reason, plan and act in open-ended environments while balancing costs and leveraging external tools. This curated list highlights recent progress across self-evolution, tool-augmented RL, multi-agent collaboration and cost-aware planning.  

As new benchmarks and algorithms appear, contributions such as **Alita**, **AGILE**, and **ToRL** demonstrate that reinforcement learning can unlock powerful new behaviours.
