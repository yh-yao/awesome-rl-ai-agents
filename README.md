# Awesome RL AI Agents

---

## 🔎 Quick Navigation

* 🧠✨ [Agentic Workflow without Training](#agentic-workflow-without-training)
* 🧪📊 [Agent Evaluation and Benchmarks](#agent-evaluation-and-benchmarks)
* 🧰⚙️ [Agent Training Frameworks](#agent-training-frameworks)
* 👤🧭 [RL for Single Agent](#rl-for-single-agent)

  * 🔁🧪 [Self-Evolution & Test-Time RL](#self-evolution--test-time-rl)
  * 🛠️🧠 [RL for Tool Use & Agent Training](#rl-for-tool-use--agent-training)
  * 💾🧠 [Memory & Knowledge Management](#memory--knowledge-management)
  * 🔁📈 [Fine-Grained RL & Trajectory Calibration](#fine-grained-rl--trajectory-calibration)
  * 🎛️🎯 [Alignment & Preference Optimization](#alignment--preference-optimization)
* 💸🧠 [Cost-Aware Reasoning & Budget-Constrained RL](#cost-aware-reasoning--budget-constrained-rl)
* 👥🤝 [RL for Multi-Agent Systems](#rl-for-multi-agent-systems)

  * 🗺️📅 [Planning](#planning)
  * 🤝🧩 [Collaboration](#collaboration)
* 🤖🌍 [Embodied Agents & World Models](#embodied-agents--world-models)
* 📚💡 [Surveys & Position Papers](#surveys--position-papers)
* 🧾✅ [Concluding Remarks](#concluding-remarks)

---

Reinforcement learning (RL) is rapidly becoming a driving force behind AI agents that can reason, act and adapt in the real world. Large language models (LLMs) provide a powerful prior for reasoning, but without feedback they remain static and brittle. RL enables agents to learn from interaction – whether it’s via self-reflection, outcome-based rewards or interacting with humans and tools.

The goal of this repository is to curate up-to-date resources on RL for AI agents, focusing on three axes:

* **Agentic workflows without training** – prompting strategies that improve reasoning without fine-tuning.
* **Evaluation and benchmarks** – systematic tests for reasoning, tool use, and task automation.
* **RL for single and multi-agent systems** – enabling self-evolution, efficient tool use, and collaboration.

Tables give a quick overview; detailed descriptions follow in the text.

---

## Agentic Workflow without Training

| Title                                                                                   | Short title |  Venue  | Year |                 Materials                 | Description                   |
| :-------------------------------------------------------------------------------------- | :---------: | :-----: | :--: | :---------------------------------------: | :---------------------------- |
| Tree of Thoughts: Deliberate Problem Solving with Large Language Models                 |     ToT     |   ICML  | 2023 | [Paper](https://arxiv.org/abs/2305.10601) | Search over reasoning trees   |
| Reflexion: Language Agents with Verbal Reinforcement Learning                           |  Reflexion  | NeurIPS | 2023 | [Paper](https://arxiv.org/abs/2303.11366) | Self-critique & retry loop    |
| Self-Refine: Iterative Refinement with Self-Feedback                                    | Self-Refine | NeurIPS | 2023 | [Paper](https://arxiv.org/abs/2303.17651) | Iterative self-improvement    |
| ReAct: Synergizing Reasoning and Acting in Language Models                              |    ReAct    |   ICLR  | 2023 | [Paper](https://arxiv.org/abs/2210.03629) | Interleave thoughts & actions |
| SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks |  SwiftSage  |   ACL   | 2023 | [Paper](https://arxiv.org/abs/2305.17390) | Fast/slow planning split      |
| DynaSaur: Large Language Agents Beyond Predefined Actions                               |   DynaSaur  |  arXiv  | 2024 | [Paper](https://arxiv.org/abs/2411.01747) | Dynamic action spaces         |

---

## Agent Evaluation and Benchmarks

| Title                                                                                   |    Short title    | Venue | Year |                                Materials                                | Description                    |
| :-------------------------------------------------------------------------------------- | :---------------: | :---: | :--: | :---------------------------------------------------------------------: | :----------------------------- |
| GAIA: A Benchmark for General AI Assistants                                             |        GAIA       | arXiv | 2023 |                [Paper](https://arxiv.org/abs/2311.12983)                | 466 real-world tasks           |
| TaskBench: Benchmarking Large Language Models for Task Automation                       |     TaskBench     | EMNLP | 2023 |                [Paper](https://arxiv.org/abs/2311.18760)                | Automation & tool use          |
| AgentBench: Evaluating LLMs as Agents                                                   |     AgentBench    | arXiv | 2023 |                [Paper](https://arxiv.org/abs/2308.03688)                | 51 scenarios                   |
| ACEBench: Who Wins the Match Point in Tool Usage?                                       |      ACEBench     | arXiv | 2025 |                [Paper](https://arxiv.org/abs/2501.12851)                | Fine-grained tool eval         |
| Agent Leaderboard                                                                       |     Galileo LB    |   HF  | 2024 | [Dataset](https://huggingface.co/datasets/galileo-ai/agent-leaderboard) | GAIA leaderboard               |
| Agentic Predictor: Performance Prediction for Agentic Workflows via Multi-View Encoding | Agentic Predictor | arXiv | 2025 |                [Paper](https://arxiv.org/abs/2505.19764)                | Workflow performance predictor |

---

## Agent Training Frameworks

| Title                                                                          |   Short title   |     Venue    | Year |                                                          Materials                                                          | Description                      |
| :----------------------------------------------------------------------------- | :-------------: | :----------: | :--: | :-------------------------------------------------------------------------------------------------------------------------: | :------------------------------- |
| Agent Lightning: Train ANY AI Agents with Reinforcement Learning               | Agent Lightning |     arXiv    | 2025 |                                          [Paper](https://arxiv.org/abs/2508.03680)                                          | Unified MDP; decouple exec/train |
| SkyRL-v0: Train Real-World Long-Horizon Agents via Reinforcement Learning      |     SkyRL-v0    | arXiv/GitHub | 2025 |                [Blog](https://novasky-ai.notion.site/skyrl-v0) \| [Code](https://github.com/NovaSky-AI/SkyRL)               | Online RL pipeline               |
| OpenManus-RL: Live-Streamed RL Tuning Framework for LLM Agents                 |   OpenManus-RL  |    GitHub    | 2025 | [Code](https://github.com/OpenManus/OpenManus-RL) \| [Dataset](https://huggingface.co/datasets/CharlieDreemur/OpenManus-RL) | Live-streamed tuning             |
| MASLab: A Unified and Comprehensive Codebase for LLM-based Multi-Agent Systems |      MASLab     |     arXiv    | 2025 |                                          [Paper](https://arxiv.org/abs/2505.16988)                                          | Unified MAS codebase             |
| VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use        |     VerlTool    |     arXiv    | 2025 |                [Paper](https://arxiv.org/abs/2509.01055) \| [Code](https://github.com/TIGER-AI-Lab/verl-tool)               | Modular ARLT; async rollouts     |

---

## RL for Single Agent

### Self-Evolution & Test-Time RL

| Title                                                                     |  Short title |     Venue    |    Year   |                                            Materials                                           | Description            |
| :------------------------------------------------------------------------ | :----------: | :----------: | :-------: | :--------------------------------------------------------------------------------------------: | :--------------------- |
| Test-Time Reinforcement Learning                                          |     TTRL     |     ICLR     |    2025   |                            [Paper](https://arxiv.org/abs/2504.16084)                           | Inference-time RL      |
| Prolonged Reinforcement Learning Expands Reasoning Boundaries             |     ProRL    |     ICLR     |    2025   |                            [Paper](https://arxiv.org/abs/2505.24864)                           | KL-control; ref resets |
| A Survey of Self-Evolving Agents                                          |   SE Survey  |     arXiv    |    2025   |                            [Paper](https://arxiv.org/abs/2507.21046)                           | Methods & taxonomy     |
| RAGEN: Understanding Self-Evolution via Multi-Turn Reinforcement Learning | RAGEN/StarPO |     ICLR     |    2025   |                            [Paper](https://arxiv.org/abs/2504.20073)                           | Multi-turn RL; critic  |
| Alita: Generalist Self-Evolving Agent                                     |     Alita    |    GAIA LB   |    2025   |                            [Paper](https://arxiv.org/abs/2505.20286)                           | Modular self-evolution |
| Gödel Agent: Towards Recursive Self-Improvement in LLM-Based Agents       |  Gödel Agent |   ACL/arXiv  | 2024–2025 |                            [Paper](https://arxiv.org/abs/2410.04444)                           | Self-modifying agents  |
| Darwin Gödel Machine: A Framework for Open-Ended Self-Improvement         |   Darwin GM  |     arXiv    |    2025   |                            [Paper](https://arxiv.org/abs/2505.22954)                           | Darwinian exploration  |
| SkyRL-v0 (duplicate listing)                                              |   SkyRL-v0   | arXiv/GitHub |    2025   | [Blog](https://novasky-ai.notion.site/skyrl-v0) \| [Code](https://github.com/NovaSky-AI/SkyRL) | Long-horizon RL        |

---

### RL for Tool Use & Agent Training

| Title                                                                             |    Short title   |    Venue   |    Year   |                                           Materials                                           | Description                |
| :-------------------------------------------------------------------------------- | :--------------: | :--------: | :-------: | :-------------------------------------------------------------------------------------------: | :------------------------- |
| AGILE: A Novel Reinforcement Learning Framework of LLM Agents                     |       AGILE      |    arXiv   |    2024   |                           [Paper](https://arxiv.org/abs/2405.14751)                           | RL + memory + tools        |
| Offline Training of Language Model Agents with Functions as Learnable Weights     |  AgentOptimizer  |    ICML    |    2024   |                           [Paper](https://arxiv.org/abs/2402.11359)                           | Learnable function weights |
| FireAct: Toward Language Agent Fine-tuning                                        |      FireAct     |    arXiv   |    2023   |                           [Paper](https://arxiv.org/abs/2310.05915)                           | Multi-task agent SFT       |
| ToRL: Scaling Tool-Integrated Reinforcement Learning for LLM Agents               |       ToRL       |    arXiv   |    2025   |                           [Paper](https://arxiv.org/abs/2503.23383)                           | Tool-integrated RL         |
| ToolRL: Reward is All Tool Learning Needs                                         |      ToolRL      |    arXiv   |    2025   |                           [Paper](https://arxiv.org/abs/2504.13958)                           | Reward design study        |
| Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning        |      ARTIST      |    arXiv   |    2025   |                           [Paper](https://arxiv.org/abs/2505.01441)                           | Unified reasoning + tools  |
| Agent RL Scaling Law: Agent RL with Spontaneous Code Execution                    |      ZeroTIR     |    arXiv   |    2025   |                           [Paper](https://arxiv.org/abs/2505.07773)                           | Scaling analysis           |
| Acting Less is Reasoning More! Teaching Models to Act Efficiently                 |        OTC       |    arXiv   |    2025   |                           [Paper](https://arxiv.org/abs/2504.14870)                           | Fewer tool calls           |
| WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning |    WebAgent-R1   |    arXiv   |    2025   |                           [Paper](https://arxiv.org/abs/2505.16421)                           | Web multiturn RL           |
| Group-in-Group Policy Optimization for LLM Agent Training                         |       GiGPO      |    arXiv   |    2025   |                           [Paper](https://arxiv.org/abs/2505.10978)                           | Hierarchical PPO           |
| Nemotron-Research-Tool-N1: Exploring Tool-Using LMs with Reinforced Reasoning     | Nemotron-Tool-N1 |    arXiv   |    2025   |                           [Paper](https://arxiv.org/abs/2505.00024)                           | Pure RL for tools          |
| CATP-LLM: Cost-Aware Tool Planning for Large Language Models                      |     CATP-LLM     | ICCV/arXiv | 2024–2025 | [Paper](https://arxiv.org/abs/2411.16313) \| [Code](https://github.com/duowuyms/OpenCATP-LLM) | Cost-aware planning        |
| Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning  |     Tool-Star    |    arXiv   |    2025   |                           [Paper](https://arxiv.org/abs/2505.16410)                           | Hierarchical rewards       |

---

### Memory & Knowledge Management

| Title                                                                                     | Short title |      Venue     | Year |                 Materials                 | Description                       |
| :---------------------------------------------------------------------------------------- | :---------: | :------------: | :--: | :---------------------------------------: | :-------------------------------- |
| Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning |  Memory-R1  |      arXiv     | 2025 | [Paper](https://arxiv.org/abs/2508.19828) | RL memory manager + answer agent  |
| A-MEM: Agentic Memory for LLM Agents                                                      |    A-MEM    |      arXiv     | 2025 | [Paper](https://arxiv.org/abs/2502.12110) | Zettelkasten-style dynamic memory |
| KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents                              |  KnowAgent  | NAACL Findings | 2025 | [Paper](https://arxiv.org/abs/2403.03101) | KB-augmented planning             |

---

### Fine-Grained RL & Trajectory Calibration

| Title                                                                                                       | Short title |     Venue    | Year |                 Materials                 | Description                |
| :---------------------------------------------------------------------------------------------------------- | :---------: | :----------: | :--: | :---------------------------------------: | :------------------------- |
| StepTool: Enhancing Multi-Step Tool Usage in LLMs via Step-Grained Reinforcement Learning                   |   StepTool  |     CIKM     | 2025 | [Paper](https://arxiv.org/abs/2410.07745) | Step-grained rewards       |
| Encouraging Good Processes Without the Need for Good Answers: Reinforcement Learning for LLM Agent Planning |     RLTR    |     arXiv    | 2025 | [Paper](https://arxiv.org/abs/2508.19598) | Process-centric rewards    |
| SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution                                            |    SPA-RL   |     arXiv    | 2025 | [Paper](https://arxiv.org/abs/2505.20732) | Stepwise attribution       |
| STeCa: Step-Level Trajectory Calibration for LLM Agent Learning                                             |    STeCa    | ACL Findings | 2025 | [Paper](https://arxiv.org/abs/2502.14276) | Calibrate suboptimal steps |
| SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks                                   |   SWEET-RL  |     arXiv    | 2025 | [Paper](https://arxiv.org/abs/2503.15478) | ColBench; stepwise critic  |
| ATLaS: Agent Tuning via Learning Critical Steps                                                             |    ATLaS    |      ACL     | 2025 | [Paper](https://arxiv.org/abs/2503.02197) | Critical-step selection    |

---

### Alignment & Preference Optimization

| Title                                                         | Short title | Venue |    Year   |                 Materials                 | Description         |
| :------------------------------------------------------------ | :---------: | :---: | :-------: | :---------------------------------------: | :------------------ |
| Beyond One-Preference-Fits-All Alignment: Multi-Objective DPO |    MODPO    | arXiv | 2023–2024 | [Paper](https://arxiv.org/abs/2310.03708) | Multi-objective DPO |

---

## Cost-Aware Reasoning & Budget-Constrained RL

| Title                                                                    |  Short title  |    Venue    | Year |                                             Materials                                             | Description             |
| :----------------------------------------------------------------------- | :-----------: | :---------: | :--: | :-----------------------------------------------------------------------------------------------: | :---------------------- |
| Cost-Augmented Monte Carlo Tree Search for LLM-Assisted Planning         |      CATS     |    arXiv    | 2025 |                             [Paper](https://arxiv.org/abs/2505.14656)                             | Cost-aware MCTS         |
| Token-Budget-Aware LLM Reasoning                                         |      TALE     |    arXiv    | 2024 |                             [Paper](https://arxiv.org/abs/2412.18547)                             | Token budget policy     |
| FrugalGPT: How to Use Large Language Models While Reducing Cost          |   FrugalGPT   |    arXiv    | 2023 |                             [Paper](https://arxiv.org/abs/2305.05176)                             | Cost minimization       |
| Efficient Contextual LLM Cascades via Budget-Constrained Policy Learning |    TREACLE    |    arXiv    | 2024 |                             [Paper](https://arxiv.org/abs/2404.13082)                             | Budgeted cascades       |
| BudgetMLAgent: Cost-Effective Multi-Agent System for ML Automation       | BudgetMLAgent | AIMLSystems | 2025 |                                                 —                                                 | Cost-effective MAS      |
| The Cost of Dynamic Reasoning: A Systems View                            |  Systems Cost |    arXiv    | 2025 |                             [Paper](https://arxiv.org/abs/2506.04301)                             | Latency/energy/cost     |
| Budget-Aware Evaluation of LLM Reasoning Strategies                      |   BudgetEval  |    EMNLP    | 2024 |                      [Paper](https://aclanthology.org/2024.emnlp-main.1112/)                      | Budget-aware eval       |
| LLM Cascades with Mixture of Thoughts for Cost-Efficient Reasoning       |  MoT Cascade  |  ICLR/arXiv | 2024 | [Paper](https://arxiv.org/abs/2310.03094) \| [Code](https://github.com/MurongYue/LLM_MoT_cascade) | Mixed thoughts cascades |
| BudgetThinker: Empowering Budget-Aware LLM Reasoning with Control Tokens | BudgetThinker |    arXiv    | 2025 |                             [Paper](https://arxiv.org/abs/2508.17196)                             | Control-token budgeting |

---

## RL for Multi-Agent Systems

### Planning

| Title                                                       |   Short title   |  Venue  | Year |                 Materials                 | Description             |
| :---------------------------------------------------------- | :-------------: | :-----: | :--: | :---------------------------------------: | :---------------------- |
| OWL: Optimized Workforce Learning for Real-World Automation |       OWL       |  arXiv  | 2025 | [Paper](https://arxiv.org/abs/2505.23885) | Planner + workers       |
| Profile-Aware Maneuvering for GAIA by AWorld                |      AWorld     | NeurIPS | 2024 | [Paper](https://arxiv.org/abs/2508.09889) | Guard agents            |
| Plan-over-Graph: Towards Parallelable Agent Schedule        | Plan-over-Graph |  arXiv  | 2025 | [Paper](https://arxiv.org/abs/2502.14563) | Graph scheduling        |
| LLM-Based Multi-Agent Reinforcement Learning: Directions    |   MARL Survey   |  arXiv  | 2024 | [Paper](https://arxiv.org/abs/2405.11106) | Survey                  |
| Self-Resource Allocation in Multi-Agent LLM Systems         |  Self-ResAlloc  |  arXiv  | 2025 | [Paper](https://arxiv.org/abs/2504.02051) | Planner vs orchestrator |
| MASLab (duplicate listing)                                  |      MASLab     |  arXiv  | 2025 | [Paper](https://arxiv.org/abs/2505.16988) | Unified MAS APIs        |

---

### Collaboration

| Title                                                            |   Short title   |  Venue  | Year |                                        Materials                                       | Description          |
| :--------------------------------------------------------------- | :-------------: | :-----: | :--: | :------------------------------------------------------------------------------------: | :------------------- |
| ACC-Collab: Actor-Critic for Multi-Agent Collaboration           |    ACC-Collab   |   ICLR  | 2025 |                   [Paper](https://openreview.net/forum?id=nfKfAzkiez)                  | Joint actor-critic   |
| Chain of Agents: Collaborating on Long-Context Tasks             | Chain of Agents |  arXiv  | 2024 |                        [Paper](https://arxiv.org/abs/2406.02818)                       | Long-context chains  |
| Scaling LLM-Based Multi-Agent Collaboration                      |   Scaling MAC   |  arXiv  | 2024 |                        [Paper](https://arxiv.org/abs/2406.07155)                       | Scaling study        |
| MMAC-Copilot: Multi-Modal Agent Collaboration                    |   MMAC-Copilot  |  arXiv  | 2024 |                        [Paper](https://arxiv.org/abs/2404.18074)                       | Multi-modal collab   |
| CORY: Sequential Cooperative Multi-Agent Reinforcement Learning  |       CORY      | NeurIPS | 2024 | [Paper](https://arxiv.org/abs/2410.06101) \| [Code](https://github.com/Harry67Hu/CORY) | Role-swapping PPO    |
| OpenManus-RL (duplicate listing)                                 |   OpenManus-RL  |  GitHub | 2025 |                    [Code](https://github.com/OpenManus/OpenManus-RL)                   | Live-streamed tuning |
| MAPoRL: Multi-Agent Post-Co-Training with Reinforcement Learning |      MAPoRL     |  arXiv  | 2025 |                        [Paper](https://arxiv.org/abs/2502.18439)                       | Co-refine + verifier |

---

## Embodied Agents & World Models

| Title                                                      |  Short title  |    Venue   | Year |                 Materials                 | Description              |
| :--------------------------------------------------------- | :-----------: | :--------: | :--: | :---------------------------------------: | :----------------------- |
| An Embodied Generalist Agent in 3D World                   |      LEO      |    ICML    | 2024 | [Paper](https://arxiv.org/abs/2311.12871) | 3D embodied agent        |
| DreamerV3: Mastering Diverse Domains through World Models  |   DreamerV3   |    arXiv   | 2023 | [Paper](https://arxiv.org/abs/2301.04104) | World-model RL           |
| World-Model-Augmented Web Agent                            | WMA Web Agent | ICLR/arXiv | 2025 | [Paper](https://arxiv.org/abs/2410.13232) | Simulative web agent     |
| WorldCoder: Model-Based LLM Agent                          |   WorldCoder  |    arXiv   | 2024 | [Paper](https://arxiv.org/abs/2402.12275) | Code-based world model   |
| WALL-E 2.0: World Alignment via Neuro-Symbolic Learning    |   WALL-E 2.0  |    arXiv   | 2025 | [Paper](https://arxiv.org/abs/2504.15785) | Neuro-symbolic alignment |
| WorldLLM: Curiosity-Driven World Modeling                  |    WorldLLM   |    arXiv   | 2025 | [Paper](https://arxiv.org/abs/2506.06725) | Curiosity + world model  |
| SimuRA: Simulative Reasoning Architecture with World Model |     SimuRA    |    arXiv   | 2025 | [Paper](https://arxiv.org/abs/2507.23773) | Mental simulation        |

---

## Surveys & Position Papers

| Title                                              | Short title | Venue | Year |                 Materials                 | Description            |
| :------------------------------------------------- | :---------: | :---: | :--: | :---------------------------------------: | :--------------------- |
| Small Language Models are the Future of Agentic AI | SLMs Survey | arXiv | 2025 | [Paper](https://arxiv.org/abs/2506.02153) | SLMs for agent systems |
| Survey on Evaluation of LLM-Based Agents           | Eval Survey | arXiv | 2025 | [Paper](https://arxiv.org/abs/2503.16416) | Eval methodologies     |

---

## Concluding Remarks

Reinforcement learning for AI agents is rapidly evolving. From self-evolving agents like **Alita**, to unified frameworks like **VerlTool**, to fine-grained trajectory calibration approaches such as **STeCa** and **SPA-RL**, the field is moving toward robust, adaptive systems.

Multi-agent collaboration, budget-aware reasoning, and embodied agents are expanding the horizons of what RL-driven AI can achieve. This curated list highlights the latest methods, frameworks, and benchmarks for agentic AI.

💡 *Pull requests welcome to keep this list up to date!*
