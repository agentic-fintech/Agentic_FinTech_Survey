# Agentic FinTech Survey

## Agentic FinTech: A Comprehensive Survey on AI Agents in Finance in the Era of LLMs

<p align="center">
  <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6136529">📄 Paper</a> |
  <a href="https://github.com/agentic-fintech/Agentic_FinTech_Survey">🔗 GitHub</a> |
  <a href="https://agentic-fintech.github.io/Agentic_FinTech_Survey/">🌐 Project</a>
</p>

As large language models (LLMs) evolve from passive text generators into autonomous financial agents with perception, planning, memory, and tool use, a new paradigm—Agentic FinTech—is emerging. This survey formalizes the paradigm by addressing three core questions: what Agentic FinTech is, why it matters, and how such systems can be built and evaluated. We characterize Agentic FinTech through key agentic capabilities, including market-grounded perception, risk-aware planning under constraints, tool use over heterogeneous financial data, long-horizon memory of prior experience, and self-improvement driven by feedback and objectives. From the perspectives of automation and autonomy, we explain why the transition toward agentic systems is essential for next-generation financial intelligence. To support practical implementation and evaluation, we review major financial application domains (e.g., trading, risk management, and regulatory compliance), summarize advances in financial agents in terms of workflows, architectures, and optimizations, and discuss evaluation methodologies, open challenges, and future research directions toward scalable, trustworthy, and efficient agentic financial systems. To keep up with the latest advances in this field, we maintain this repository.

## Agentic FinTech

To establish a clear conceptual foundation, we formally define **Agentic FinTech** as follows.

**Definition: Agentic FinTech**

> Agentic FinTech represents a new paradigm of financial technology systems that leverage LLM-powered agents to automate financial workflows, enabling them to autonomously perceive financial information, plan under complex market contexts, invoke prior experience, make multi-step decisions, interact with users, and continuously self-improve through feedback from their operating environments to achieve specific financial objectives.

![Agentic FinTech](/imgs/agentic_fintech.png)

## Implementation of Agentic FinTech

| Method | Category | Framework | Key Feature | Link |
| --- | --- | --- | --- | --- |
| [TradingAgents](https://arxiv.org/pdf/2412.20138?) | Trading | Multi-Agent | Role Specialization | [GitHub](https://github.com/TauricResearch/TradingAgents) |
| [NOFX](https://github.com/NoFxAiOS/nofx) | Trading | Multi-Agent | Role Specialization | [GitHub](https://github.com/NoFxAiOS/nofx) |
| [AlphaQuanter](https://arxiv.org/pdf/2510.14264) | Trading | Single-Agent | Tool-Orchestrated Agentic RL | [GitHub](https://github.com/AlphaQuanter/AlphaQuanter) |
| [R&D-Agent(Q)](https://arxiv.org/pdf/2505.15155?) | Trading | Multi-Agent | Factor-Model Co-Evolving | [GitHub](https://github.com/microsoft/RD-Agent) |
| [QuantEvolve](https://arxiv.org/pdf/2510.18569) | Trading | Multi-Agent | Evolutionary Optimization | [GitHub](https://github.com/tarsyang/quantevolve) |
| [AlphaAgent](https://dl.acm.org/doi/pdf/10.1145/3711896.3736838) | Trading | Multi-Agent | Decay-Resistant Alpha Mining | [GitHub](https://github.com/RndmVariableQ/AlphaAgent) |
| [CryptoTrade](https://aclanthology.org/2024.emnlp-main.63.pdf) | Trading | Multi-Agent | Reflective Decision Refinement | [GitHub](https://github.com/Xtra-Computing/CryptoTrade) |
| [StockAgent](https://arxiv.org/pdf/2407.18957?) | Trading | Multi-Agent | Stock Trading Simualtion | [GitHub](https://github.com/MingyuJ666/Stockagent) |
| [ContestTrade](https://arxiv.org/pdf/2508.00554) | Trading | Multi-Agent | Internal Contest Mechanisms | [GitHub](https://github.com/FinStep-AI/ContestTrade) |
| [FLAG-TRADER](https://arxiv.org/pdf/2502.11433?) | Trading | Single-Agent | Gradient-based RL | [Site](https://huggingface.co/TheFinAI) |
| [AI Hedge Fund](https://github.com/virattt/ai-hedge-fund) | Trading | Multi-Agent | Role Specialization | [GitHub](https://github.com/virattt/ai-hedge-fund) |
| [MM-DREX](https://arxiv.org/pdf/2509.05080) | Trading | Multi-Agent | SFT-RL for Router & Experts | - |
| [DeepFund](https://arxiv.org/pdf/2503.18313) | Trading | Multi-Agent | Role Specialization | [GitHub](https://github.com/HKUSTDial/DeepFund) |
| [TradingGroup](https://arxiv.org/pdf/2508.17565?) | Trading | Multi-Agent | Self-Reflection | - |
| [ATLAS](https://arxiv.org/pdf/2510.15949) | Trading | Multi-Agent | Prompt Optimization | - |
| [CGA-Agent](https://arxiv.org/pdf/2510.07943?) | Trading | Multi-Agent | Genetic Algorithms | - |
| [AICrypto-Assistant](https://www.set-science.com/manage/uploads/ICOF2025_00101/SETSCI_ICOF2025_00101_002.pdf) | Trading | Multi-Agent | Router & Specialists | - |
| [TiMi](https://arxiv.org/pdf/2510.04787?) | Trading | Multi-Agent | Hierarchical Optimization | - |
| [P1GPT](https://arxiv.org/pdf/2510.23032) | Trading | Multi-Agent | Structured Reasoning Pipeline | - |
| [MarketSenseAI 2.0](https://arxiv.org/pdf/2502.00415?) | Trading | Multi-Agent | Chain-of-Agents | [Site](https://marketsense-ai.com/) |
| [FinMem](https://ojs.aaai.org/index.php/AAAI-SS/article/download/31290/33450) | Trading | Multi-Agent | Adjustable Cognitive Memory | [GitHub](https://github.com/pipiku915/FinMem-LLM-StockTrading) |
| [FinWorld](https://arxiv.org/pdf/2508.02292?) | Multiple | Multi-Agent | All-in-One Platform | [GitHub](https://github.com/DVampire/FinWorld) |
| [QuantMind](https://arxiv.org/pdf/2509.21507) | IR | Multi-Agent | Extraction & Retrieval | [GitHub](https://github.com/LLMQuant/quant-mind) |
| [FinSearch](https://dl.acm.org/doi/pdf/10.1145/3768292.3770382) | IR | Multi-Agent | Planner & Executor | [GitHub](https://github.com/eeeshushusang/FinSearch) |
| [MASCA](https://arxiv.org/pdf/2507.22758?) | Credit | Multi-Agent | Multidimensional Assessment | - |
| [Ryt AI](https://aclanthology.org/2025.emnlp-industry.45.pdf) | Service | Multi-Agent | Guardrails, Intent, & Action | [Site](https://www.rytbank.my/ryt-ai) |
| [TS-Agent](https://dl.acm.org/doi/pdf/10.1145/3768292.3771251) | Forecasting | Multi-Agent | Code Base & Refinement | - |
| [FinRpt-Gen](https://arxiv.org/pdf/2511.07322) | Reporting | Multi-Agent | Extraction & Analysis | [GitHub](https://github.com/jinsong8/FinRpt) |
| [ValueCell](https://github.com/ValueCell-ai/valuecell) | Reporting | Multi-Agent | Deep Research | [GitHub](https://github.com/ValueCell-ai/valuecell) |
| [FinDebate](https://aclanthology.org/2025.finnlp-2.20.pdf) | Reporting | Multi-Agent | Safe Debate Protocol | - |
| [FinRobot](https://arxiv.org/pdf/2405.14767?) | Reporting | Multi-Agent | Financial Chain-of-Thought | [GitHub](https://github.com/AI4Finance-Foundation/FinRobot) |
| [FinTeam](https://arxiv.org/pdf/2507.10448) | Reporting | Multi-Agent | Role Specialization | [GitHub](https://github.com/FudanDISC/DISC-FinLLM/blob/main/README-en.md) |
| [MS-Agent FinResearch](https://github.com/modelscope/ms-agent) | Reporting | Multi-Agent | Deep Research | [GitHub](https://github.com/modelscope/ms-agent) |
| [FinSight](https://arxiv.org/pdf/2510.16844?) | Reporting | Multi-Agent | Deep Research | - |
| [Coinvisor](https://arxiv.org/pdf/2510.17235) | Reporting | Multi-Agent | RL-based Tool Selection | [GitHub](https://github.com/BugmakerCC/Coinvisor) |
| [FinPos](https://arxiv.org/pdf/2510.27251) | Risk | Multi-Agent | Position-Aware Trading | - |
| [Risk Analyst](https://arxiv.org/pdf/2507.08584) | Risk | Multi-Agent | Agentic Model Discovery Loop | - |
| [FinHEAR](https://arxiv.org/pdf/2506.09080?) | Risk | Multi-Agent | Adaptive Risk Modeling | [GitHub](https://github.com/pilgrim00/FinHEAR) |
| [FinRS](https://arxiv.org/pdf/2511.12599) | Risk | Multi-Agent | Risk-Aware Reasoning | - |
| [M-SAEA](https://openreview.net/pdf?id=frPFuji3Hz) | Risk | Multi-Agent | Multi Safety-Aware Evaluation | - |
| [AuditAgent](https://dl.acm.org/doi/pdf/10.1145/3768292.3770383) | Fraud | Multi-Agent | Cross-Doc Fraud Discovery | - |
| [RCA](https://arxiv.org/pdf/2509.10546) | Regulatory | Multi-Agent | Risk-Concealment Attacks | [GitHub](https://github.com/gcheng128/RCA/) |
| [AlphaAgents](https://arxiv.org/pdf/2508.11152?) | Portfolio | Multi-Agent | Role Specialization | - |
| [FinCon](https://proceedings.neurips.cc/paper_files/paper/2024/file/f7ae4fe91d96f50abc2211f09b6a7e49-Paper-Conference.pdf) | Trading & Portfolio | Multi-Agent | Manager-Analyst Hierarchy & Dual-Level Risk Control | [GitHub](https://github.com/The-FinAI/FinCon) |
| [Wealth-Voyager](https://dl.acm.org/doi/pdf/10.1145/3766918.3766944) | Portfolio | Multi-Agent | Strategic Asset Allocation | - |
| [MACI](https://arxiv.org/pdf/2501.00826?) | Portfolio | Multi-Agent | Inter-/Intra-Team Collaboration | - |
| [MASS](https://arxiv.org/pdf/2505.10278) | Portfolio | Multi-Agent | Optimal Investor Distribution | [GitHub](https://github.com/gta0804/MASS) |

## Datasets & Benchmarks of Agentic FinTech

| Method | Category | Type | Key Feature | Link |
| --- | --- | --- | --- | --- |
| [Qlib](https://arxiv.org/pdf/2009.11189) | Trading | Platform | Quantitative Investment | [GitHub](https://github.com/microsoft/qlib) |
| [FinRL](https://arxiv.org/pdf/2011.09607) | Trading | Framework | Financial RL Framework | [GitHub](https://github.com/AI4Finance-Foundation/FinRL) |
| [InvestorBench](https://aclanthology.org/2025.acl-long.126.pdf) | Trading | Benchmark | Financial Decision Making | [GitHub](https://github.com/felis33/INVESTOR-BENCH) |
| [TradeTrap](https://arxiv.org/pdf/2512.02261) | Trading | Framework | Stress-Testing Trading | [GitHub](https://github.com/Yanlewen/TradeTrap) |
| [TwinMarket](https://arxiv.org/pdf/2502.01506) | Trading | Simulation | Stock Market Simulation | [GitHub](https://github.com/FreedomIntelligence/TwinMarket) |
| [Agent Trading Arena](https://aclanthology.org/anthology-files/anthology-files/pdf/findings/2025.findings-emnlp.294.pdf) | Trading | Simulation | Virtual Zero-Sum Stock Market | [GitHub](https://github.com/wekjsdvnm/Agent-Trading-Arena) |
| [DeepFund](https://arxiv.org/pdf/2503.18313) | Trading | Benchmark | Real-Time Fund Investment | [GitHub](https://github.com/HKUSTDial/DeepFund) |
| [Agent Market Arena](https://arxiv.org/pdf/2510.11695) | Trading | Benchmark | Live Multi-Market Trading | [Site](https://ama.thefin.ai/live) |
| [LiveTradeBench](https://arxiv.org/pdf/2511.03628) | Trading | Benchmark | Live Trading Environment | [GitHub](https://github.com/ulab-uiuc/live-trade-bench) |
| [FinRL-Meta](https://proceedings.neurips.cc/paper_files/paper/2022/file/0bf54b80686d2c4dc0808c2e98d430f7-Paper-Datasets_and_Benchmarks.pdf) | Trading | Benchmark | Data-Driven Financial RL | [GitHub](https://github.com/AI4Finance-Foundation/FinRL-Meta) |
| [FinLake-Bench](https://arxiv.org/pdf/2510.07920?) | Trading | Benchmark | Leakage-Robust Evaluation | - |
| [FINSABER](https://arxiv.org/pdf/2505.07078?) | Trading | Framework | Evaluating Trading Strategies | [GitHub](https://github.com/waylonli/FINSABER) |
| [StockBench](https://arxiv.org/pdf/2510.02209?) | Trading | Benchmark | Stock Trading Decision-Making | [GitHub](https://github.com/ChenYXxxx/stockbench) |
| [FinTSB](https://arxiv.org/pdf/2502.18834?) | Forecasting | Benchmark | Time Series Forecasting | [GitHub](https://github.com/TongjiFinLab/FinTSB) |
| [FinDeepForecastBench](https://arxiv.org/pdf/2601.05039) | Forecasting | Benchmark | Forecasting at Corporate & Macro Levels | [Site](https://openfinarena.com/fin-deep-forecast) |
| [FinQA](https://aclanthology.org/2021.emnlp-main.300.pdf) | QA | Dataset | Numerical Reasoning | [GitHub](https://github.com/czyssrs/FinQA) |
| [ConvFinQA](https://arxiv.org/pdf/2210.03849) | QA | Dataset | Conversational Finance QA | [GitHub](https://github.com/czyssrs/ConvFinQA) |
| [FinEval 1.0](https://aclanthology.org/2025.naacl-long.318.pdf) | QA | Benchmark | Financial Domain Knowledge & Practical Abilities of LLMs | [GitHub](https://github.com/SUFE-AIFLM-Lab/FinEval) |
| [FinanceIQ](https://github.com/Duxiaoman-DI/XuanYuan/tree/main/FinanceIQ) | QA | Dataset | Chinese Financial Knowledge | [GitHub](https://github.com/Duxiaoman-DI/XuanYuan/tree/main/FinanceIQ) |
| [BizFinBench](https://openreview.net/pdf?id=86DunY8cY5) | QA | Benchmark | Business-Driven Real-World Financial Benchmark | [GitHub](https://github.com/HiThink-Research/BizFinBench) |
| [OmniEval](https://aclanthology.org/2025.emnlp-main.292.pdf) | QA | Benchmark | Omnidirectional & Automatic RAG in Finance Domain | [GitHub](https://github.com/RUC-NLPIR/OmniEval) |
| [FinAgentBench](https://dl.acm.org/doi/pdf/10.1145/3768292.3770362) | QA | Benchmark | Agentic Retrieval with Multi-Step Reasoning in Financial QA | - |
| [FinSearchComp](https://arxiv.org/pdf/2509.13160?) | QA | Benchmark | Expert-Level Evaluation of Financial Search and Reasoning | [Site](https://randomtutu.github.io/FinSearchComp/) |
| [FinRAGBench-V](https://arxiv.org/pdf/2505.17471) | QA | Benchmark | Visual RAG in Finance | [GitHub](https://github.com/zhaosuifeng/FinRAGBench-V) |
| [VisFinEval](https://aclanthology.org/2025.emnlp-main.1229.pdf) | QA | Benchmark | Chinese Financial Knowledge | [GitHub](https://github.com/SUFE-AIFLM-Lab/VisFinEval) |
| [MME-Finance](https://dl.acm.org/doi/pdf/10.1145/3746027.3758230) | QA | Benchmark | Bilingual Multimodal Financial QA for Expert-level Reasoning | [GitHub](https://github.com/HiThink-Research/MME-Finance) |
| [FinRpt](https://arxiv.org/pdf/2511.07322) | Reporting | Benchmark | Equity Research Report (ERR) Generation | [Site](https://huggingface.co/datasets/jinsong8/FinRpt) |
| [FinDocResearch](https://www.preprints.org/frontend/manuscript/bacdf98d0b691e23b27008acd8360efe/download_pub) | Reporting | Benchmark | Analyzing a Listed Company's Multi-Year Annual Reports | [Site](https://openfinarena.com/fin-doc-research) |
| [FinDeepResearch](https://arxiv.org/pdf/2510.13936?) | Reporting | Benchmark | Corporate Financial Analysis | [Site](https://openfinarena.com/fin-deep-research) |
| [FinResearchBench](https://dl.acm.org/doi/pdf/10.1145/3768292.3770364) | Reporting | Framework | Logic Tree based Agent-as-a-Judge Evaluation Framework | - |
| [Fraud-R1](https://arxiv.org/pdf/2502.12904) | Fraud | Benchmark | LLMs' Resistance to Fraud | [GitHub](https://github.com/kaustpradalab/Fraud-R1) |
| [MultiAgentFraudBench](https://arxiv.org/pdf/2511.06448) | Fraud | Simulation | Financial-Fraud Simulation | [GitHub](https://github.com/zheng977/MutiAgent4Fraud) |
| [FIN-Bench](https://arxiv.org/pdf/2509.10546) | Regulatory | Benchmark | LLM Jailbreaks under Regulatory Criteria | [GitHub](https://github.com/gcheng128/RCA/) |
| [Finova](https://arxiv.org/pdf/2507.16802) | Multiple | Benchmark | Financial Operable and Verifiable Agent Benchmark | [GitHub](https://github.com/antgroup/Finova) |
| [FinMaster](https://arxiv.org/pdf/2505.13533) | Multiple | Benchmark | Full-Pipeline Financial Workflows | [Site](https://jianginsight.github.io/Finmaster_leaderboard/) |

## Citation

```latex
@article{wu2026agenticfintech,
  title   = {Agentic FinTech: A Comprehensive Survey on AI Agents in Finance in the Era of LLMs},
  author  = {Wu, Yaxiong and Li, Yixuan},
  journal = {SSRN},
  year    = {2026},
  month   = jan,
  note    = {Available at SSRN: https://ssrn.com/abstract=6136529},
  doi     = {10.2139/ssrn.6136529}
}
```
