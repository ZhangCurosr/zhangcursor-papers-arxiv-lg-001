# AT-AT: Automatic Tool Optimization from Agent Traces

Maya Ravichandran Netflix

Pragya Tripathi Netflix

Dharamendra Kumar Netflix

Adith Swaminathan Netflix

Nathan Kallus Netflix

{mravichandran,pragyat,dharamendrak,aswaminathan,nkallus}@netflix.com

## Abstract

Efforts to improve LLM-based agents typically focus on modifying the agent itself, yet success also depends on the environment in which an agent acts. Tools mediate this interaction: a tool’s description shapes how an agent calls it, and its implementation determines the results and errors the agent receives afterwards. Many tools, such as APIs and MCP servers, are shared by many agents and maintained by providers who cannot change those agents. We introduce AT-AT (Automatic Tool Optimization from Agent Traces), an agentic tool optimizer that improves a shared toolset offline from historical agent executions while holding the agents fixed. A lightweight coding agent reads agent trajectories and their evaluation results, and selectively edits tool descriptions and, when permitted, implementations. On τ<sup>2</sup>- bench-verified, AT-AT achieves the highest task completion rates in the Airline and Telecom domains at lower cost than strong prompt- and tool-optimizer baselines, and its Telecom gain is statistically significant under paired permutation tests. Optimizing the tools within a coding agent harness also improves held-out performance on TerminalBench-2 and TBLite. When applied to three production MCP servers at a large enterprise, AT-AT lowered cost or tool error rates on every server and increased held-out task passes on one. By turning accumulated execution traces into updates to shared tool environments, AT-AT provides a path toward continual improvement of agent systems through environment-side learning.

## 1 Introduction

Efforts to improve LLM-based agents typically modify the agent itself: training the underlying model, optimizing its prompts [Agrawal et al., 2026, Kassianik et al., 2026], or engineering the harness that manages its execution [Lee et al., 2026b, Zhang et al., 2026, Weng, 2026]. However, an agent’s success is a property not only of the agent, but also of its interaction with the external environment. Tools mediate this interaction, as they define the actions available to the agent and the information returned in response. We study a complementary path to better agent performance: improving the tool environment from accumulated agent experience based on historical execution traces, while holding the agent fixed.

The opportunity for such optimization is large and growing. In an enterprise, an agent’s tool environment may include Model Context Protocol (MCP) servers spanning many internal systems. More broadly, nearly any software API can be exposed as an agent-callable tool. Most APIs, however, were designed for human developers or conventional software rather than for the capabilities and limitations of LLM agents. Cloud services can now automatically translate existing APIs into agent callable tools [Microsoft, 2026, Wadhwa et al., 2025], but callability does not imply usability by LLMs. An API’s agent-facing description, schema, outputs, and errors remain consequential design choices that we aim to automatically optimize. These tools are typically maintained by a provider who cannot change the many agents that call them, but any change the provider makes to a tool reaches all of those agents. This is our primary setting.

![](images/c42ff1ee7d5ae52dcf0e9951400cdf25f3c58ccd1f892e61e28ef187eaabae42.jpg)  
Figure 1: AT-AT optimizes tools from historical agent traces. (A) An agent completes a task by calling tools, implemented in its harness or the external environment, producing a trajectory. (B) In a sandbox holding evaluated traces and the current tool source, a Pi coding agent driven by an optimizer LLM selectively edits tool descriptions and implementations.

A tool has two parts. Its description, comprising its name, parameter schema, and usage notes, shapes the agent’s expectations before a call, while its implementation determines the results and errors the agent receives after it. Because tool results enter the agent’s context, actionable errors can enable recovery and relevant outputs can support better subsequent decisions. We call this mechanism feedback shaping: improving the information returned by tools without changing the agent or its evaluation objective. For example, a τ<sup>2</sup>-bench-verified Telecom tool that reports a line’s data usage and plan limit leaves the agent to work out whether the line has run out of data, while an implementation that also returns the remaining data gives the agent this information directly.

Anthropic reported a 40% decrease in task completion time after redesigning the tools available to an agent [Anthropic, 2025b], suggesting that tool design materially affects agent performance. Existing automatic methods primarily optimize the information presented before a tool is invoked: DRAFT [Qu et al., 2025], Play2Prompt [Fang et al., 2025], EasyTool [Yuan et al., 2025], and ToolObserver [Hallinan et al., 2026] use exploration traces or reflection to refine tool descriptions or usage examples. Harness optimizers such as Meta-Harness [Lee et al., 2026b] can also edit tool code, but they optimize the harness of a single agent rather than a toolset shared by many agents. In this work, we aim to automatically improve both the descriptions and implementations of shared tools from evaluated agent traces.

We introduce AT-AT (Automatic Tool Optimization from Agent Traces), which converts historical agent experience into selective edits to a toolset. A lightweight coding agent searches evaluated agent traces for recurring failure modes, inspects the tool source, and edits tool descriptions and, when permitted, implementations. A coding agent can search many long traces as needed, and it can read the tool’s code to decide whether a failure calls for a description or an implementation fix. AT-AT performs a single offline pass over previously collected executions, requiring no new agent rollouts.

Trace-driven optimization raises two methodological challenges. First, repeated agent executions can produce materially different outcomes even when the task, model, harness, environment, and decoding configuration are held constant. We evaluate the original and optimized toolsets with 30 trials per held-out task and use paired permutation tests to determine whether observed changes exceed stochastic variation. Second, an optimizer can overfit to a small set of trajectories or tasks, which is a particularly consequential risk for shared tools. We expose AT-AT to traces spanning multiple tasks and repeated trials, direct it toward generalizable edits, and evaluate the resulting toolsets on held-out executions.

We evaluate AT-AT in three settings. On $\tau ^ { 2 } .$ -bench-verified, AT-AT achieves the highest held-out task completion rate in Airline and Telecom at a lower cost per test simulation than prior prompt and tool optimizers, and its gain of 7.25 percentage points in Telecom is statistically significant. On TerminalBench-2 and TBLite, optimizing the tools within the OpenCode harness improves held-out pass@1 and pass@3, demonstrating applicability beyond customer service tools. On three production MCP servers at Netflix, description edits learned from historical logs from production traffic increased held-out task passes on one server and reduced tool errors and cost while maintaining performance on the other two. Together, these results show that the traces used to diagnose agent failures can also be converted into improvements to the environment itself. Because AT-AT can be rerun as traces accumulate, it provides a mechanism for continual improvement, although we evaluate a single update rather than a continual learning loop.

Our contributions are as follows:

• We formulate tool optimization for fixed agents from previously collected traces, where a tool’s description guides the agent before a call and its implementation shapes the feedback the agent receives after the call (feedback shaping).

• We introduce AT-AT, a lightweight coding agent optimizer that translates evaluated historical trajectories into selective edits of tool descriptions and, when permitted, implementations, without new agent executions during optimization.

• We evaluate AT-AT on τ<sup>2</sup>-bench-verified, TerminalBench-2, TBLite, and three production MCP servers, using repeated held-out trials and paired permutation tests on $\scriptstyle { \tau ^ { 2 } } ,$ -benchverified to separate tool improvements from run-to-run noise.

## 2 Related Work

Prompt and harness optimization. Prompt optimizers search over natural-language or structured prompt programs, using execution outcomes or reflective feedback to guide candidate generation [Schnabel and Neville, 2024, Agrawal et al., 2026]. Coding agent optimizers expand this search space: FAPO can modify prompts and multi-step pipeline structure, while Meta-Harness, Self-Harness, and AutoSaddler optimize harness code using execution traces and scores [Kassianik et al., 2026, Lee et al., 2026b, Zhang et al., 2026, Park et al., 2026]. Meta-Harness is especially similar in giving an agentic proposer access to an agent’s source code, scores, and full traces via a filesystem, but it iteratively searches over the agent harness, which includes the harness’s internal tools, rather than a toolset alone [Lee et al., 2026b]. When tools are implemented inside the harness, as in our OpenCode experiments, AT-AT performs a restricted form of harness optimization that edits only the tools. Our primary setting instead updates a shared environmental toolset without changing the agents that consume it, since a tool provider typically does not own those agents.

Manual tool design. Anthropic recommends designing and testing tools for LLMs rather than exposing conventional developer-facing APIs and describes an evaluation-driven workflow in which developers inspect agent transcripts and use a coding agent to revise descriptions and implementations, checking generalization on held-out tasks [Anthropic, 2025b,a]. AT-AT automates such a workflow.

Documentation and schema optimization. Most prior methods optimize the prompt-visible surface of a tool. EasyTool condenses heterogeneous documentation into standardized instructions, while PA-Tool renames schema components to align with a model’s pretrained naming conventions [Yuan et al., 2025, Lee et al., 2026a]. DRAFT and Play2Prompt interact with tools to refine documentation or synthesize usage examples without labeled tasks [Qu et al., 2025, Fang et al., 2025]. A study of 856 tools from 103 MCP servers found at least one quality defect in 97.1% of tool descriptions, and augmenting the descriptions raised median task success while also increasing execution steps [Hasan et al., 2026]. This still leaves executable tool behavior fixed.

Trace-driven tool optimization. Joint context optimization generates training-task rollouts and uses feedback on final answers and tool-call histories to jointly refine agent instructions and tool descriptions, while JTPRO uses rollout-driven reflection to co-optimize global agent instructions and per-tool schemas [Wu et al., 2025, Ghoshal et al., 2026]. ToolObserver and DocsChisel repeatedly rerun the agent being optimized and use fixed LLM pipelines to revise only tool documentation from its trajectories. ToolObserver distills patterns through mini-batching and consensus, whereas DocsChisel uses field-level edits with validation and regression checks [Hallinan et al., 2026, Lu et al., 2026]. AT-AT instead uses a coding-agent optimizer to update shared tools from previously evaluated trajectories without optimization-time agent reruns, and can make contract-preserving implementation edits that change execution-time results, errors, and recovery cues. ContDa uses self-generated probing traces to continually adapt documentation as tools are added, changed, or deprecated [Wu et al., 2026]. Although AT-AT can be rerun as production logs accumulate, the present work evaluates one update to an existing toolset.

Environment optimization and tool synthesis. Other work also improves the environment rather than the agent. Song et al. [2025] derive a taxonomy of agent–environment failures from traces and show that hand-designed environment changes, such as better observability and offloading computation, improve fixed agents; AT-AT instead derives edits automatically from evaluated traces. Tool-synthesis methods expand the action space by creating new tools from API documentation [Ni et al., 2025], composing atomic actions into reusable procedures [Ding et al., 2026], or compiling procedures repeated in execution traces [Kujanpää et al., 2026]. AT-AT instead improves tools that are already deployed while preserving the interfaces their callers depend on.

## 3 Optimization Method

## 3.1 Problem Formulation

Environmental tool optimization. We consider a population of agents $A \sim A .$ Each agent $A = ( H , M )$ consists of a fixed harness H and a fixed large language model M. The harness includes components such as the system prompt, code that manages the agent loop and multiple turns, and internal tools such as file reading and file writing utilities. We also consider a population of tasks $x \sim \mathcal { X }$ , which can be drawn from a variety of domains.

Given an agent A, a task $x ,$ and an environmental toolset $T _ { E } ,$ , executing the agent generates a trajectory τ consisting of the LLM outputs, user messages, tool invocations, and tool results that occur during execution. We denote the induced trajectory distribution by $p _ { A } ( \tau \mid x , T _ { E } )$ . Equivalently, for ${ \cal A } = ( \bar { H } , M ) , p _ { A } ( \tau \mid x , T _ { E } ) = p _ { M } ( \tau \mid H , x , \bar { T } _ { E } )$ . This distribution captures stochasticity from both the LLM and the environment. The reward $r ( \tau , x )$ measures task success.

We optimize a shared environmental toolset $T _ { E } = ( T _ { E _ { D } } , T _ { E _ { C } } )$ , where $T _ { E _ { D } }$ denotes the tool descrip tions and $T _ { E _ { C } }$ denotes the executable tool code. Environmental tools may include external functions, APIs, or MCP servers that are not implemented within the agent harness. Our objective is to select the toolset that maximizes expected performance across a population of agents and tasks:

$$
\begin{array} { r } { T _ { E _ { D } } ^ { * } , T _ { E _ { C } } ^ { * } = \arg \operatorname* { m a x } _ { T _ { E _ { D } } , T _ { E _ { C } } } \mathbb { E } _ { A \sim A , { x } \sim \mathcal { X } , \tau \sim p _ { A } ( \cdot \vert { x } , T _ { E _ { D } } , T _ { E _ { C } } ) } \left[ r ( \tau , x ) \right] . } \end{array}\tag{1}
$$

Optimization of tools within a harness. For tools implemented within a harness, tool optimization is a restricted form of harness optimization: only the tool descriptions and implementations are editable, while all other harness components remain fixed. Let $T _ { H } = \bar { ( T _ { H _ { D } } , T _ { H _ { C } } ) }$ denote the internal tools being optimized, and let $H _ { - T _ { H } }$ denote the remaining harness components. The corresponding objective is

$$
\begin{array} { r } { T _ { H _ { D } } ^ { * } , T _ { H _ { C } } ^ { * } = \arg \operatorname* { m a x } _ { T _ { H _ { D } } , T _ { H _ { C } } } \mathbb { E } _ { x \sim \mathcal { X } , \tau \sim p _ { M } ( \cdot \vert H _ { - T _ { H } } , x , T _ { H _ { D } } , T _ { H _ { C } } ) } \left[ r ( \tau , x ) \right] . } \end{array}\tag{2}
$$

We evaluate AT-AT in both tool optimization settings. Our $\tau ^ { 2 }$ -bench-verified experiments (Sec. 4.1) and production MCP server experiments (Sec. 4.3) optimize environmental tools as defined in Eq. 1. Our coding agent experiments (Sec. 4.2) instead optimize tools within the OpenCode harness.

## 3.2 AT-AT Method

Agent traces on a subset of tasks (either the train split on a benchmark, or historical tasks in a production setting) are constructed. Each agent trace consists of the agent trajectory on a certain task (agent outputs, user messages, tool invocations and results) along with the final evaluation result on the task. The agent traces and tools to edit are provided to the agentic tool optimizer as files in a file system. The optimizer is a lightweight, open-source coding agent, the Pi agent [Earendil Inc. and Contributors, 2026]. The optimization objective is provided as a system prompt to the Pi agent. An instruction to the optimizer to make generalizable tool edits is also provided in the system prompt. Agent traces are also provided in the user prompt to the optimizer; the first 4,000 characters of each failing run’s transcript plus up to three passing runs are included in the first user message to the optimizer LLM. The prompts to AT-AT are provided in Appendix G. The Pi agent makes selective edits to the tool files in the codebase based on what it sees in the agent traces. Once edits are finished, they must pass a validation check. The edited tool files must pass a syntax/compilation/interpreter check, and no files may be added, deleted, or edited outside the editable tool files. If validation fails, the Pi agent’s session is resumed with a prompt stating that validation failed, and the agent may revise its edits. The optimizer is given up to five attempts in total to produce a proposal that passes validation. The implementation of AT-AT is available at https: $/ / { \tt g i }$ thub.com/Netflix-Skunkworks/agent-tool-opt-core.

## 4 Experiments & Results

## 4.1 $\tau ^ { 2 }$ -bench-verified

$\tau ^ { 2 } .$ -bench [Barres et al., 2026] is a popular benchmark for evaluating LLMs on customer service agent tasks. $\tau ^ { 2 } .$ -bench-verified [Cuadron et al., 2025] is an updated version of $\tau ^ { 2 }$ -bench in which the tasks were manually reviewed and corrected. In our setup, the user agent and customer service agent LLMs were GPT-5.1 and GPT-4.1, respectively, which is the $\tau ^ { 2 }$ -bench-verified default. The default train/test task indices and evaluators from the $\dot { \tau } ^ { 2 }$ -bench-verified repository were used. First, a baseline run of $\tau ^ { 2 }$ -bench-verified with the default settings was conducted, with $n = 5$ trials run on each task. The agent trajectories and evaluations were combined to form the baseline agent traces. The same baseline agent traces (train split only) were inputted into all tool optimization methods used, and the tool optimizers were run to generate updated toolsets. The LLM used for all tool optimizers was Claude Opus $4 . 7 . \ \tau ^ { 2 }$ -bench-verified was rerun using the updated toolsets. Additionally, the average cost of running a task in $\tau ^ { 2 } .$ -bench-verified was reported, both before and after tool optimizers were applied. This cost was computed using the built-in cost tracker in $\tau ^ { 2 }$ -bench-verified. The cost of running the optimizer is also reported and was computed by multiplying the number of tokens used by the cost per token from the LLM APIs.

Paired permutation testing. Agent outcomes vary across repeated executions even when the model, harness, tasks, environment, and decoding configuration are held fixed. A difference between two aggregate benchmark scores therefore does not by itself establish that one toolset reliably outperforms another. To account for this evaluation noise, we evaluate the original and optimized toolsets on the same held-out tasks, with 30 trials per task, and compare their outcomes using a paired permutation test. Pairing outcomes within tasks accounts for differences in task difficulty, while repeated trials capture run-to-run variation in agent execution. Under the null hypothesis that the toolset has no effect, the original and optimized condition labels are exchangeable within each pair; permuting these labels produces the null distribution of the difference in task completion rate. We then test whether the observed improvement is larger than would be expected from run-to-run variation. At small sample sizes, exact computation of the paired permutation test p value is possible, as described in Zmigrod et al. [2022]. However, at our larger sample size of $n = 3 0$ , we estimate the p-value using Monte Carlo permutation sampling. Each value we compute is the comparison of the optimized toolset result compared to the original, unoptimized toolset.

Baseline optimizers. We compare to DRAFT [Qu et al., 2025], ToolObserver [Hallinan et al., 2026], and GEPA [Agrawal et al., 2026]. DRAFT is an online method requiring interaction with live tools, while our $\tau ^ { 2 }$ -bench-verified experiment setup provides pre-collected agent traces. Hence, our implementation contained the offline portions of DRAFT, applying DRAFT’s Analyzer and Rewriter stages once to the past agent traces. Similarly, we adapt ToolObserver to a fixed offline setting by applying its minibatch reflection and consensus merging once to pre-collected trajectories, omitting iterative exploration and trajectory recollection. We run GEPA [Agrawal et al., 2026] by treating the toolset as a prompt to be optimized. GEPA was used with 1 exploration/reflection step to provide a fair comparison to the other methods, which also used 1 exploration/reflection step.

Overall results. We report the average reward (pass@1) scores as percentages for each of the three domains: Airline, Retail, and Telecom, both for the baseline (unoptimized) toolset and optimized toolsets produced by AT-AT, our ablations of AT-AT, and the prior baseline optimizers. The results of these experiments can be viewed in Tab. 1. AT-AT has the highest reward score on the Airline and Telecom domains, and its improvement in Telecom is the only improvement that was statistically significant. In both domains, AT-AT also costs less per test simulation than ToolObserver, the nextbest prior method. DRAFT resulted in a statistically significant decrease in performance on Telecom. We contextualize the improvements AT-AT was able to achieve by comparing them to submissions on the official τ<sup>2</sup>-bench leaderboard [Sierra Research, 2026]. The 7-point improvement AT-AT achieved on Airline is comparable to the improvement from Claude-3.7-Sonnet (released Feb. 24, 2025; 64%) to Claude Sonnet 4.5 (released Sep. 29, 2025; 72%). The 7-point improvement AT-AT achieved on Telecom is comparable to the improvement from Claude-3.7-Sonnet (released Feb. 24, 2025; 49%) to GPT-5.2 (released Dec. 11, 2025; 57%). However, the tool edits AT-AT makes can be applied to a variety of agents and potentially improve performance of many agents (discussed in Sec. 4.1.1).

Costs. Two types of costs are reported: the cost to run a task in τ<sup>2</sup>-bench-verified, and the cost to run the optimizer. Optimization increased the cost to run the benchmark by a statistically significant amount in many cases. We attribute this partially to the increased tool file length, leading to a higher number of tokens that the executor agent needs to process. In addition, variance in costs may be due to prompt cache hit rates. Optimizer costs were similar across optimizers, except for ToolObserver, which incurred substantially higher optimization costs. We attribute this to its minibatch-analysis and consensus-merge structure, which required many more LLM calls than the other optimizers.

## 4.1.1 Ablations

Effect of feedback shaping: implementation edits vs description-only edits. To test our hypothesis that tools’ feedback provided to an agent can meaningfully shape its trajectory and improve task success, we ran a version of AT-AT that was constrained to editing tool descriptions only and could not edit tool code. This ablation also omitted agent traces from the user prompt, though they were available in the filesystem. Results are shown in the “Edit descriptions only” rows of Tab. 1. The descriptions-only method performed substantially worse than AT-AT in all domains but Retail (in Retail, all optimized tools performed worse than unoptimized tools, but no difference was statistically significant). A further qualitative analysis comparing tool edits from this ablation is in App. B.

Importance of past agent traces. To test if historical agent traces are a useful learning signal for AT-AT, we ran AT-AT without any historical agent traces available to it. Results are shown in the “Without past traces” rows of Tab. 1. The method without past agent traces performed substantially worse than AT-AT in all domains but Retail (but in the Retail domain, all optimized tools performed worse than unoptimized tools). This evidence supports that past agent traces are a useful learning signal for a tool optimizer. In addition to enabling a tool optimizer to learn about tool failure patterns, historical agent traces may also enable an optimizer to learn about a new domain from data that is automatically gathered, rather than being curated or labeled by humans.

Importance of coding agent harness. To investigate the importance of the coding agent setup to the tool optimizer, we ran an experiment with a single-shot LLM-based tool optimizer. Since the LLM-based optimizer did not have the ability to search through a file system itself, the agent traces were only provided in their truncated form in the optimizer user prompt. The optimizer LLM was instructed to output a new tools.py file. Results are shown in the “Without coding agent harness” rows of Tab. 1. The LLM optimizer performed substantially worse than AT-AT in Airline and Telecom.

Generalization across agents. To investigate our hypothesis that edits to tools made by AT-AT can be useful across a variety of agents and tasks, we generated new baseline transcripts using a variety of customer service agent LLMs: GPT-4.1, GPT-4.1-mini, Inkling, Claude Sonnet 5, and GLM-5.2. AT-AT was run using the train split of these trials. The same 5 LLMs were used when running the test split before and after optimization. This ablation also omitted agent traces from the user prompt, though they were available in the file system. Results are shown in Tab. 2. AT-AT is able to improve reward in Airline and Retail, though the gain is not statistically significant. This is consistent with the hypothesis that AT-AT can improve an environment’s overall quality and help a variety of agents.

Table 1: AT-AT achieves the highest test reward in Airline and Telecom. Held-out results over $n = 3 0$ trials per task. Test cost is agent plus user-simulator cost per simulation; optimizer cost is the one-time cost of producing the toolset. $p \mathrm { : }$ two-sided task-paired permutation test against the domain baseline; ∗ marks $p < 0 . 0 5$ . Bold: best value per domain (optimizer cost excludes baseline).
<table><tr><td colspan="2">Optimizer</td><td>Test reward (%)</td><td>Reward p</td><td>Test cost (USD/sim.)</td><td>Cost p</td><td>Optimizer cost (USD)</td></tr><tr><td rowspan="7">Airine</td><td>Baseline</td><td>52.17%</td><td></td><td>0.0618</td><td></td><td></td></tr><tr><td>AT-AT</td><td>59.17%</td><td>0.2305</td><td>0.0675*</td><td>0.002155</td><td>2.7800</td></tr><tr><td>Without past traces</td><td>50.33%</td><td>0.3575</td><td>0.1205*</td><td>0.000002</td><td>0.2330</td></tr><tr><td>Edit descriptions only</td><td>51.33%</td><td>0.7609</td><td>0.1176*</td><td>0.000002</td><td>0.3839</td></tr><tr><td>Without coding agent harness</td><td>52.50%</td><td>0.9446</td><td>0.0825*</td><td>0.000002</td><td>2.2481</td></tr><tr><td>DRAFT</td><td>53.50%</td><td>0.5988</td><td>0.0822*</td><td>0.000002</td><td>2.4367</td></tr><tr><td>ToolObserver</td><td>55.83%</td><td>0.1650</td><td>0.0870*</td><td>0.000002</td><td>16.2171</td></tr><tr><td rowspan="8">Retaiı</td><td>GEPA</td><td>51.50%</td><td>0.7539</td><td>0.0748*</td><td>0.000008</td><td>0.8042</td></tr><tr><td>Baseline</td><td>77.00%</td><td></td><td>0.0707</td><td></td><td></td></tr><tr><td>AT-AT</td><td>74.67%</td><td>0.2313</td><td>0.0724</td><td>0.070629</td><td>1.9300</td></tr><tr><td>Without past traces Edit descriptions only</td><td>75.42%</td><td>0.3129</td><td>0.1383*</td><td>0.000010</td><td>0.2174</td></tr><tr><td>Without coding agent harness</td><td>74.08% 74.92%</td><td>0.0992 0.1979</td><td>0.1453*</td><td>0.000010 0.000010</td><td>0.5613</td></tr><tr><td></td><td></td><td></td><td>0.0802*</td><td></td><td>3.8496</td></tr><tr><td>DRAFT ToolObserver</td><td>75.50%</td><td>0.5296</td><td>0.0871*</td><td>0.000010</td><td>3.9785</td></tr><tr><td>GEPA</td><td>75.92%</td><td>0.6196</td><td>0.0926*</td><td>0.000010</td><td>42.1400</td></tr><tr><td rowspan="8">Telcom</td><td></td><td>76.08%</td><td>0.5731</td><td>0.0798*</td><td>0.000010</td><td>0.8765</td></tr><tr><td>Baseline</td><td>29.50%</td><td></td><td>0.0996</td><td></td><td></td></tr><tr><td>AT-AT</td><td>36.75%*</td><td>0.0033</td><td>0.0986</td><td>0.508165</td><td>10.6900</td></tr><tr><td>Without past traces</td><td>30.42%</td><td>0.6281</td><td>0.2850*</td><td>0.000010</td><td>0.2888</td></tr><tr><td>Edit descriptions only</td><td>31.17%</td><td>0.4547</td><td>0.2857*</td><td>0.000010</td><td>1.1342</td></tr><tr><td>Without coding agent harness</td><td>30.50%</td><td>0.6172</td><td>0.1146*</td><td>0.000010</td><td>4.4205</td></tr><tr><td>DRAFT</td><td>25.58%*</td><td>0.0026</td><td>0.1259*</td><td>0.000010</td><td>4.5630</td></tr><tr><td>ToolObserver GEPA</td><td>31.67% 29.25%</td><td>0.4686 0.9102</td><td>0.1276* 0.1132*</td><td>0.000010 0.000010</td><td>49.1224 0.8846</td></tr></table>

Table 2: AT-AT’s tool edits improve held-out reward in 2 of 3 τ<sup>2</sup>-bench-verified domains when averaged across five customer service agent LLMs. Results are averaged across models within each task, then across tasks, with 30 trials per task. Columns and ∗ are as in Tab. 1.
<table><tr><td>Domain</td><td>Optimizer</td><td>Test reward (%)</td><td>Reward p</td><td>Test cost (USD/sim.)</td><td>Cost p</td><td>Optimizer cost (USD)</td></tr><tr><td>Airline</td><td>Baseline AT-AT</td><td>58.00% 59.67%</td><td>0.3584</td><td>0.1135 0.1239*</td><td>0.000557</td><td>2.4029</td></tr><tr><td>Retail</td><td>Baseline AT-AT</td><td>73.83% 76.42%</td><td>0.1662</td><td>0.1221 0.1315*</td><td>0.000010</td><td>2.3330</td></tr><tr><td>Telecom</td><td>Baseline AT-AT</td><td>41.42% 40.67%</td><td>0.7320</td><td>0.2457 0.2602*</td><td>0.000100</td><td>1.8308</td></tr></table>

## 4.2 TerminalBench-2 & TBLite

To assess whether AT-AT can generalize to a different domain than the customer service agent tasks assessed in $\tau ^ { 2 }$ -bench-verified, we ran AT-AT on TerminalBench-2 and TBLite. TerminalBench-2 [Merrill et al., 2026] is a popular computer use and agentic coding benchmark with long-horizon, multi-turn tasks. TBLite [OpenThoughts-Agent team, Snorkel AI, and Bespoke Labs, 2026] is a lightweight alternative to TerminalBench-2, designed to track results on TerminalBench-2 closely while enabling a faster iteration speed and runtime. The toolset optimized was that within OpenCode [Anomaly, 2026], an open-source coding agent harness. OpenCode contains tools with implementations and descriptions written in TypeScript code, and these tools were the target of the tool optimizer. OpenCode was run on TerminalBench-2 and TBLite, and the resulting agent traces were split into a train and test set. The train set was then provided to AT-AT to optimize the tools within the OpenCode repository. In these experiments, past traces were omitted from the optimizer user prompt, although they were available in the file system. Finally, the tool-optimized version of OpenCode was rerun on each benchmark to calculate final performance on the train and test splits. The LLM used by AT-AT was Claude Opus 4.7, while the LLM used by the OpenCode agent was Claude Opus 4.5. We report pass@k as computed by the benchmarks’ native evaluators, before and after tool optimization. Results on TerminalBench-2 and TBLite can be viewed in Tab. 3. On TerminalBench-2, AT-AT improves held-out pass@1, pass@3, and pass@10, while on TBLite, AT-AT improves held-out pass@1 and pass@3, while held-out pass@10 is unchanged.

Table 3: AT-AT improves held-out pass@k on TerminalBench-2 and TBLite. Pass@k from each benchmark’s native evaluator before (Baseline) and after (Optimized) optimizing OpenCode’s tools, on the train split used for optimization and the held-out test split.
<table><tr><td rowspan="3"></td><td colspan="4">Train</td><td colspan="4">Test</td></tr><tr><td colspan="2">TerminalBench-2</td><td colspan="2">TBLite</td><td colspan="2">TerminalBench-2</td><td colspan="2">TBLite</td></tr><tr><td>Baseline</td><td>Optimized</td><td>Baseline</td><td>Optimized</td><td>Baseline</td><td>Optimized</td><td>Baseline</td><td>Optimized</td></tr><tr><td>1</td><td>45.78%</td><td>48.44%</td><td>72.80%</td><td>72.80%</td><td>49.32%</td><td>51.36%</td><td>75.60%</td><td>77.20%</td></tr><tr><td>3</td><td>58.43%</td><td>58.69%</td><td>81.38%</td><td>81.95%</td><td>59.43%</td><td>61.34%</td><td>88.65%</td><td>89.63%</td></tr><tr><td>10</td><td>68.89%</td><td>68.89%</td><td>84.00%</td><td>86.00%</td><td>68.18%</td><td>70.45%</td><td>94.00%</td><td>94.00%</td></tr></table>

## 4.3 Production MCP Servers

MCP and task selection. We applied AT-AT to production MCP servers at Netflix. Three production MCP servers at the company were chosen for optimization due to (1) having a high usage rate, (2) having a sufficient number of unique single-user prompts with actionable failures, and (3) the ability within the company infrastructure to expose and modify the tool descriptions in a sandbox environment. The MCP servers used were the following: Core Developer Tools (Dev Tools), which provides broad engineering retrieval across internal documentation, collaboration platforms, and enterprise knowledge sources; Observability Collaboration Toolbox (Collab Toolbox), which searches, summarizes, and exports channel and thread content from an enterprise collaboration platform; and Enterprise Engineering Context and Best Practices (Eng. Context), which provides enterprise-specific engineering guidance from indexed documentation and code. Tasks were defined as an initial agent prompt provided by a user, and tasks were randomly selected from each server’s prior two weeks of production traffic. Tab. 12 summarizes representative tasks.

Experiment procedure. The list of tasks was randomly partitioned into train/test splits of 10/10, 5/5, and 11/11 tasks for Dev Tools, Collab Toolbox, and Eng. Context and Best Practices, respectively. Across both splits and in the baseline and optimized runs, system prompt, four-user-turn limit, agent model, user simulator, and all other environment settings were held fixed. The agent executing the task used the open-weight DeepSeek V4 Flash model, and the user simulator used Claude Sonnet 4.6. For each task, a baseline run with the original MCP servers was executed, with the task being the initial prompt provided to the executor agent. The baseline run was evaluated as described below. Traces of agent trajectories and evaluation results from the train split of tasks were constructed and provided to AT-AT, which used Claude Opus 4.7 as its optimizer LLM. Past traces were omitted from the optimizer user prompt, although they were available in the file system. After tool optimization, evaluation was repeated on the train and test splits using the same surrounding setup, but now with the optimized tool descriptions.

Evaluation. Three evaluation metrics were computed for each task. Firstly, a binary pass/fail value for a task was computed via an LLM judge. Secondly, a reward score was computed for each task.

$$
r ( \tau , x ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ t h e ~ t a s k - s u c c e s s ~ c r i t e r i a ~ w e r e ~ m e t } , } \\ { \frac { n _ { \mathrm { a c c e p t e d } } } { 2 } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.
$$

Here, $n _ { \mathrm { e l i g i b l e } }$ is the number of tool calls remaining after harness-blocked and environmental calls are excluded, and $n _ { \mathrm { a c c e p t e d } }$ is the subset of eligible calls whose result was not an error. This framing provided some partial credit to traces that had fewer tool call failures, while capping unsuccessful traces at a reward of 0.5. Thirdly, an LLM judge-based content score was reported to measure answer quality separately from tool execution. An LLM converted each task’s success criteria into a fixed rubric: a checklist of 3–7 independently checkable clauses. For K clauses, the content score was

Table 4: AT-AT improves task pass rate on one out of three production MCP servers, while reducing tool call error rates and/or cost on all three MCP servers. Held-out results with original (Orig.) and AT-AT-optimized descriptions; n is the number of scored test tasks. Mean reward and content score are shown as percentages. Session cost covers the agent and user simulator; optimizer cost is one-time.
<table><tr><td></td><td colspan="2">Dev Tools (n=10)</td><td colspan="2">Collab Toolbox (n=5)</td><td colspan="2">Eng. Context (n=11)</td></tr><tr><td>Metric</td><td>Orig.</td><td>AT-AT</td><td>Orig.</td><td>AT-AT</td><td>Orig.</td><td>AT-AT</td></tr><tr><td>Task pass rate (%)</td><td>80.0</td><td>80.0</td><td>0.0</td><td>0.0</td><td>54.5</td><td>72.7</td></tr><tr><td>Mean reward (%)</td><td>90.0</td><td>90.0</td><td>22.3</td><td>35.1</td><td>77.3</td><td>86.4</td></tr><tr><td>Content score (%)</td><td>88.3</td><td>80.0</td><td>37.7</td><td>36.7</td><td>64.7</td><td>69.0</td></tr><tr><td>Tool calls</td><td>176</td><td>133</td><td>121</td><td>86</td><td>114</td><td>76</td></tr><tr><td>Tool errors</td><td>15</td><td>10</td><td>73</td><td>37</td><td>0</td><td>0</td></tr><tr><td>Session cost</td><td>$0.956</td><td>$0.737</td><td>$0.442</td><td>$0.276</td><td>$0.288</td><td>$0.235</td></tr><tr><td>Optimizer cost</td><td></td><td>$1.34</td><td></td><td>$6.935</td><td></td><td>$7.684</td></tr></table>

$$
\begin{array} { r } { s _ { \mathrm { c o n t e n t } } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } q _ { j } , \qquad q _ { j } \in \{ 1 , 0 . 5 , 0 \} } \end{array}
$$

for clause verdicts met, partially met, and not met, respectively. All evaluation generation and judging used GPT-5.6 Luna. In addition, the total number of tool calls, the number of failed tool calls, the mean cost of agent sessions in each domain, and the cost to run AT-AT were reported. Session cost includes the agent and user simulator but excludes evaluation costs, while optimizer cost is a one-time cost that includes all optimization attempts. Task pairs with a session error either before or after optimization were excluded from efficacy, tool-use, and session-cost metrics. One task in the Eng. Context server was omitted due to a session error and was not used in the optimization or evaluation.

Results. Results are shown in Tab. 4. On Dev Tools, AT-AT reduced cost, number of tool calls, and number of tool errors, while task completion rate remained unchanged, though content score fell. On Collab Toolbox, AT-AT reduced tool calls, tool errors, and execution cost. The mean task reward increased since a larger fraction of eligible tool calls succeeded, but neither arm completed a held-out task, and mean content score declined slightly. On Eng. Context, AT-AT improved task completion and content score while reducing tool calls and execution cost. Thus, AT-AT improved efficiency on all three servers and task completion on one. App. D gives paraphrased examples of the resulting tool-description changes and one task’s evaluation specification.

## 5 Conclusion

AT-AT optimizes tools to improve agent performance, holding agents fixed and leveraging logged agent traces. This improved performance on multiple benchmarks, including three production MCP servers at Netflix. Because AT-AT learns from logs that deployed systems already produce, it can be rerun as traces accumulate, storing experience in the environment rather than in the agent.

App. B qualitatively analyzes feedback shaping in selected transcripts, and App. C discusses how to handle evaluation stochasticity.

Our results suggest several possible avenues for future work. First, a more explicit model of feedback shaping would be of interest: how changing what a tool returns steers an agent’s trajectory. Second, attributing failures to tools, agents, or tasks would bound the headroom for any environment-side intervention and let an optimizer focus on where tools are at fault. Third, a shared tool serves a heterogeneous and evolving population of agents, so tool optimization is fundamentally a problem of generalization across agents and of co-adaptation with agent- and harness-side optimization (App. E), whose interaction is not yet understood. Finally, treating the environment as a site of continual learning calls for guarantees against regression and drift as edits accumulate, and raises the possibility of using tool design to steer agents away from reward hacking and unsafe actions.

## References

Lakshya A. Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J. Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab. GEPA: Reflective prompt evolution can outperform reinforcement learning. In ICLR, 2026. URL https: //openreview.net/forum?id=RQm2KQTM5r.

Anomaly. OpenCode: The open source AI coding agent, 2026. URL https://github.com/ano malyco/opencode. GitHub repository. Accessed August 26, 2026.

Anthropic. Writing effective tools for agents—with agents, September 2025a. URL https: //www.anthropic.com/engineering/writing-tools-for-agents. Engineering at Anthropic, September 11, 2025.

Anthropic. Raising the bar on SWE-bench Verified with Claude 3.5 Sonnet, January 2025b. URL ht tps://www.anthropic.com/engineering/swe-bench-sonnet. Engineering at Anthropic, January 6, 2025.

Anthropic. Quantifying infrastructure noise in agentic coding evals, February 2026. URL https: //www.anthropic.com/engineering/infrastructure-noise. Engineering at Anthropic, February 5, 2026.

Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ<sup>2</sup>-Bench: Evaluating conversational agents in a dual-control environment. In ICML, 2026. URL https://icml.cc/ virtual/2026/poster/64377.

Alejandro Cuadron, Pengfei Yu, Yang Liu, and Arpit Gupta. SABER: Small actions, big errors – safeguarding mutating steps in LLM agents, 2025. URL https://arxiv.org/abs/2512.078 50.

Haipeng Ding, Yuexiang Xie, Zhewei Wei, Yaliang Li, and Bolin Ding. From atomic actions to standard operating procedures: Iterative tool optimization for self-evolving LLM agents, 2026. URL https://arxiv.org/abs/2607.07321.

Earendil Inc. and Contributors. Pi agent harness, 2026. URL https://github.com/earendil-w orks/pi. GitHub repository. Accessed August 27, 2026.

Wei Fang, Yang Zhang, Kaizhi Qian, James R. Glass, and Yada Zhu. PLAY2PROMPT: Zero-shot tool instruction optimization for LLM agents via tool play. In ACL Findings, pages 26274–26290, 2025. doi: 10.18653/v1/2025.findings-acl.1347. URL https://aclanthology.org/2025.fi ndings-acl.1347/.

Sandip Ghoshal, Anshul Mittal, Jyotika Singh, Miguel Ballesteros, Weiyi Sun, Fang Tu, Shailender Singh, Yassine Benajiba, Fahad Shah, Sujeeth Bharadwaj, Sujith Ravi, and Dan Roth. JTPRO: A joint tool–prompt reflective optimization framework for language agents. In ACL Findings, pages 40573–40595, 2026. doi: 10.18653/v1/2026.findings-acl.2017. URL https://aclanthology .org/2026.findings-acl.2017/.

Skyler Hallinan, Thejas Venkatesh, Xiang Ren, Sai Praneeth Karimireddy, Ashwin Paranjape, Yuhao Zhang, and Jack Hessel. OpaqueToolsBench: Learning nuances of tool behavior through interaction. In CAIS, 2026. doi: 10.1145/3786335.3813182. URL https://doi.org/10.1145/3786335. 3813182.

Mohammed Mehedi Hasan, Hao Li, Gopi Krishnan Rajbahadur, Bram Adams, and Ahmed E. Hassan. Model Context Protocol (MCP) tool descriptions are smelly! towards improving AI agent efficiency with augmented MCP tool descriptions, 2026. URL https://arxiv.org/abs/2602.14878.

Paul Kassianik, Baturay Saglam, Huaibo Zhao, Blaine Nelson, Supriti Vijay, Aman Priyanshu, and Amin Karbasi. FAPO: Fully automated prompt optimization of multi-step LLM pipelines, 2026. URL https://arxiv.org/abs/2606.19605.

Kalle Kujanpää, Ning Liu, Shahnawaz Alam, Yeshwanth Reddy Sura, Tianyu Yang, Kristina Klinkner, and Shervin Malmasi. Tool-making and self-evolving LLM agents in low-latency systems, 2026. URL https://arxiv.org/abs/2607.08010. To appear in the EMNLP 2026 Industry Track.

Jonggeun Lee, Woojung Song, Jongwook Han, Haesung Pyun, and Yohan Jo. Don’t adapt small language models for tools; adapt tool schemas to the models. In ACL, pages 20695–20719, 2026a. doi: 10.18653/v1/2026.acl-long.948. URL https://aclanthology.org/2026.acl-long.94 8/.

Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, and Chelsea Finn. Meta-Harness: End-to-end optimization of model harnesses. In COLM, 2026b. URL https: //arxiv.org/abs/2603.28052.

You Lu, Kun Zhang, Bihuan Chen, and Xin Peng. DocsChisel: Adaptive tool documentation optimization framework for LLM agents, 2026. URL https://arxiv.org/abs/2608.10037.

Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, Boxuan Li, et al. Terminal-Bench: Benchmarking agents on hard, realistic tasks in command line interfaces. In ICLR, 2026. URL https://openreview.net/forum?id=a7Qa4CcHak.

Microsoft. Expose REST API in API Management as an MCP server, April 2026. URL https: //learn.microsoft.com/en-us/azure/api-management/export-rest-mcp-server. Microsoft Learn. Last updated April 30, 2026. Accessed August 28, 2026.

Zairah Mustahsan, Abel Lim, Megna Anand, Saahil Jain, and Bryan McCann. Stochasticity in agentic evaluations: Quantifying inconsistency with intraclass correlation, 2025. URL https: //arxiv.org/abs/2512.06710.

Xinyi Ni, Qiuyang Wang, Yukun Zhang, and Pengyu Hong. ToolFactory: Automating tool generation by leveraging LLM to understand REST API documentations, 2025. URL https://arxiv.org/ abs/2501.16945.

OpenThoughts-Agent team, Snorkel AI, and Bespoke Labs. OpenThoughts-TBLite: A high-signal benchmark for iterating on terminal agents, February 2026. URL https://www.openthoughts .ai/blog/openthoughts-tblite. OpenThoughts blog, February 18, 2026.

Sungho Park, Wonjoong Kim, Rongyuan Tan, Jue Zhang, Wook-Shin Han, Pengfei Gao, Chanyoung Park, Yongqiang Yao, Rao Fu, Elsie Nallipogu, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. AutoSaddler: Automatic harness optimization with durable updates from agent execution traces, 2026. URL https://arxiv.org/abs/2608.23041.

Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. From exploration to mastery: Enabling LLMs to master tools via self-driven interactions. In ICLR, 2025. URL https://openreview.net/forum?id=QKBu1BOAwd.

Tobias Schnabel and Jennifer Neville. Symbolic prompt program search: A structure-aware approach to efficient compile-time prompt optimization. In EMNLP Findings, pages 670–686, 2024. doi: 10.18653/v1/2024.findings-emnlp.37. URL https://aclanthology.org/2024.findings-e mnlp.37/.

Sierra Research. τ<sup>2</sup>-bench Leaderboard. https://taubench.com/leaderboard/?benchmark= core, 2026. Accessed: 2026-09-25.

Kevin Song, Anand Jayarajan, Yaoyao Ding, Qidong Su, Zhanda Zhu, Sihang Liu, and Gennady Pekhimenko. Aegis: Taxonomy and optimizations for overcoming agent-environment failures in LLM agents, 2025. URL https://arxiv.org/abs/2508.19504.

Sparsh Wadhwa, Dhawalkumar Patel, and Heeki Park. Streamline AI agent tool interactions: Connect API Gateway to AgentCore Gateway with MCP, December 2025. URL https://aws.amazon .com/blogs/machine-learning/streamline-ai-agent-tool-interactions-conne ct-api-gateway-to-agentcore-gateway-with-mcp/. AWS Artificial Intelligence Blog, December 8, 2025. Accessed August 28, 2026.

Lilian Weng. Harness engineering for self-improvement, July 2026. URL https://lilianweng.g ithub.io/posts/2026-07-04-harness/. Lil’Log, July 4, 2026.

Bin Wu, Edgar Meij, and Emine Yilmaz. A joint optimization framework for enhancing efficiency of tool utilization in LLM agents. In ACL Findings, pages 22361–22373, 2025. doi: 10.18653/v1/20 25.findings-acl.1149. URL https://aclanthology.org/2025.findings-acl.1149/.

Bin Wu, Edgar Meij, and Emine Yilmaz. Beyond static toolsets: Self-evolving LLM tool agents via continual documentation adaptation. In ACL Findings, pages 21519–21539, 2026. doi: 10.18653/v 1/2026.findings-acl.1082. URL https://aclanthology.org/2026.findings-acl.1082/.

Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Yongliang Shen, Kan Ren, Dongsheng Li, and Deqing Yang. EASYTOOL: Enhancing LLM-based agents with concise tool instruction. In NAACL, pages 951–972, 2025. URL https://aclanthology.org/2025.naacl-long.44/.

Hangfan Zhang, Shao Zhang, Kangcong Li, Chen Zhang, Yang Chen, Yiqun Zhang, Lei Bai, and Shuyue Hu. Self-Harness: Harnesses that improve themselves, 2026. URL https://arxiv.or g/abs/2606.09498.

Ran Zmigrod, Tim Vieira, and Ryan Cotterell. Exact paired-permutation testing for structured test statistics. In NAACL, pages 4894–4902, 2022. doi: 10.18653/v1/2022.naacl-main.360. URL https://aclanthology.org/2022.naacl-main.360/.

## A Detailed $\tau ^ { 2 } .$ -bench-verified Results

## A.1 Additional Ablations

Specificity of method to optimizer LLM used. To assess AT-AT’s efficacy with different optimizer LLMs, AT-AT, and ToolObserver for comparison, were run with GPT-5.6 Sol and the open-weight model Kimi K3 with the same $\tau ^ { 2 } .$ -bench-verified experiment setup. This ablation also omitted agent traces from the user prompt, though they were available in the file system. Results are shown in Tab. 5. AT-AT was able to achieve a statistically significant improvement in Telecom using GPT-5.6 Sol and Kimi K3, although regressions were seen when AT-AT was run with these two models on Airline, though the regressions were not statistically significant. This experiment supports that gains from AT-AT are possible with other models besides Claude Opus 4.7 and that the method is not overfit to a particular optimizer LLM.

Importance of generalization prompt and full traces in file system. To test the contribution of these two additions, we ran the Pi optimizer with only the trace digest in its first user message, without the generalization instruction and without the full per-task traces in its file system. Results are shown in Tab. 6. This variant underperforms AT-AT and does not achieve its gains on Airline and Telecom. Because both components were removed together, this ablation does not separate their individual contributions.

## A.2 Detailed τ<sup>2</sup>-bench-verified results

Tab. 7 reports the reward and recurring-cost deltas, 95% task-cluster bootstrap confidence intervals, and raw paired-permutation p-values corresponding to the main-paper summary in Tab. 1.

Table 5: AT-AT’s improvements are robust to the choice of optimizer LLM in $\tau ^ { 2 } .$ -bench-verified’s Telecom, where both GPT-5.6 Sol and Kimi K3 produce statistically significant gains. The customer service agent and user simulator are GPT-4.1 and GPT-5.1, respectively; only the optimizer LLM varies. Results are computed over $n = 3 0$ trials per held-out test task. Test cost is mean agent plus user-simulator inference cost in USD per simulation, and optimizer cost is the one-time cost of producing the candidate toolset. ∗ marks a statistically significant change from the domain baseline at $p < 0 . 0 5$ under a two-sided task-paired permutation test. Best reward (highest), test cost (lowest), and optimizer cost (lowest non-baseline value) in each domain are bold.
<table><tr><td>Domain</td><td>Optimizer</td><td>Optimizer LLM</td><td>Test reward (%)</td><td>Reward p</td><td>Test cost (USD/sim.)</td><td>Cost p</td><td>Optimizer cost (USD)</td></tr><tr><td rowspan="6">Airline</td><td>Baseline</td><td></td><td>52.17%</td><td></td><td>0.0618</td><td></td><td></td></tr><tr><td></td><td>Opus 4.7</td><td>59.17%</td><td>0.2305</td><td>0.0675*</td><td>0.002155</td><td>2.7800</td></tr><tr><td>AT-AT</td><td>GPT-5.6 Sol</td><td>47.33%</td><td>0.3743</td><td>0.1194*</td><td>0.000002</td><td>1.2866</td></tr><tr><td></td><td>Kimi K3</td><td>54.33%</td><td>0.3135</td><td>0.1246*</td><td>0.000002</td><td>4.7419</td></tr><tr><td rowspan="2">ToolObserver</td><td>Opus 4.7</td><td>55.83%</td><td>0.1650</td><td>0.0870*</td><td>0.000002</td><td>16.2171</td></tr><tr><td>GPT-5.6 Sol</td><td>52.83%</td><td>0.7461</td><td>0.1498*</td><td>0.000002</td><td>10.1014</td></tr><tr><td rowspan="6">Retail</td><td>Baseline</td><td>Kimi K3</td><td>53.33%</td><td>0.7576</td><td>0.1619*</td><td>0.000002</td><td>8.5355</td></tr><tr><td></td><td></td><td>77.00%</td><td></td><td>0.0707</td><td></td><td></td></tr><tr><td>AT-AT</td><td>Opus 4.7</td><td>74.67%</td><td>0.2313</td><td>0.0724</td><td>0.070629</td><td>1.9300</td></tr><tr><td rowspan="3"></td><td>GPT-5.6 Sol</td><td>74.75%</td><td>0.2487</td><td>0.1444*</td><td>0.000010</td><td>0.8445</td></tr><tr><td>Kimi K3</td><td>75.92%</td><td>0.5848</td><td>0.1395*</td><td>0.000010</td><td>1.4375</td></tr><tr><td>Opus 4.7 GPT-5.6 Sol</td><td>75.92%</td><td>0.6196</td><td>0.0926*</td><td>0.000010</td><td>42.1400</td></tr><tr><td rowspan="6">Telecom</td><td>ToolObserver</td><td>Kimi K3</td><td>75.83% 77.50%</td><td>0.6272 0.8646</td><td>0.1490* 0.1774*</td><td>0.000010 0.000010</td><td>27.5875 23.4634</td></tr><tr><td>Baseline</td><td></td><td>29.50%</td><td></td><td>0.0996</td><td></td><td></td></tr><tr><td rowspan="3">AT-AT</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Opus 4.7 GPT-5.6 Sol</td><td>36.75%* 47.58%*</td><td>0.0033 &lt;0.0001</td><td>0.0986 0.2633*</td><td>0.508165</td><td>10.6900</td></tr><tr><td>Kimi K3</td><td>38.42%*</td><td>0.0056</td><td>0.2854*</td><td>0.000010 0.000010</td><td>1.1033 1.4343</td></tr><tr><td rowspan="3">ToolObserver</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2"></td><td>Opus 4.7</td><td>31.67%</td><td>0.4686 0.0001</td><td>0.1276* 0.3083*</td><td>0.000010 0.000010</td><td>49.1224</td></tr><tr><td>GPT-5.6 Sol Kimi K3</td><td>40.50%* 37.67%*</td><td>0.0182</td><td>0.3219*</td><td>0.000010</td><td>32.8944 25.2612</td></tr></table>

Table 6: Pi optimizer without the generalization prompt or full file system traces underperforms AT-AT’s reward gains in Airline and Telecom. Held-out results over n = 30 trials per task; baseline and AT-AT rows are repeated from Tab. 1 for comparison. Test cost is agent plus user-simulator cost per simulation; optimizer cost is the one-time cost of producing the toolset. p: two-sided task-paired permutation test against the domain baseline; ∗ marks $p < 0 . 0 5$ . Bold: best test reward and test cost per domain.
<table><tr><td rowspan="2">Domain</td><td rowspan="2">Optimizer</td><td rowspan="2">Test reward (%)</td><td rowspan="2">Reward p</td><td rowspan="2">Test cost (USD/sim.)</td><td rowspan="2">Cost p</td><td rowspan="2">Optimizer cost (USD)</td></tr><tr><td></td></tr><tr><td rowspan="3">Airline</td><td>Baseline</td><td>52.17%</td><td></td><td>0.0618</td><td></td><td></td></tr><tr><td>AT-AT</td><td>59.17%</td><td>0.2305</td><td>0.0675*</td><td>0.002155</td><td>2.7800</td></tr><tr><td>Without generalization prompt or full traces</td><td>51.33%</td><td>0.8099</td><td>0.0605</td><td>0.363811</td><td>2.7800</td></tr><tr><td rowspan="3">Retail</td><td>Baseline</td><td>77.00%</td><td>一</td><td>0.0707</td><td>一</td><td></td></tr><tr><td>AT-AT</td><td>74.67%</td><td>0.2313</td><td>0.0724</td><td>0.070629</td><td>1.9300</td></tr><tr><td>Without generalization prompt or full traces</td><td>75.75%</td><td>0.3186</td><td>0.0710</td><td>0.683283</td><td>1.9300</td></tr><tr><td rowspan="3">Telecom</td><td>Baseline</td><td>29.50%</td><td></td><td>0.0996</td><td></td><td></td></tr><tr><td>AT-AT</td><td>36.75%*</td><td>0.0033</td><td>0.0986</td><td>0.508165</td><td>10.6900</td></tr><tr><td>Without generalization prompt or full traces</td><td>26.92%</td><td>0.1373</td><td>0.0993</td><td>0.750562</td><td>10.6900</td></tr></table>

<table><tr><td colspan="9">Table 7: Detailed τ2-bench-verified reward and cost results. Test reward and recurring inference cost are evaluated over 30 trials per held-out task. Each domain&#x27;s baseline is reported as a separate row. Deltas are optimized minus baseline and are shown with 95% task-cluster bootstrap confidence intervals. Reported p-values are from two-sided task-paired permutation tests; * marks p &lt; 0.05. Recurring cost is mean agent plus user-simulator cost in USD per test simulation. Optimizer cost is</td></tr><tr><td rowspan="2">the one-time cost of producing the candidate toolset. Domain Optimizer</td><td rowspan="2"></td><td rowspan="2">Test reward (%)</td><td rowspan="2">Reward delta [95% CI]</td><td rowspan="2">Reward p</td><td rowspan="2">Test cost (USD/sim.)</td><td rowspan="2">Cost delta [95% CI]</td><td rowspan="2">Cost</td><td rowspan="2">Optimizer cost</td></tr><tr><td>p</td></tr><tr><td rowspan="8">Airline</td><td>Baseline</td><td>52.17%</td><td></td><td></td><td>0.0618</td><td></td><td></td><td></td></tr><tr><td>AT-AT</td><td>59.17%</td><td>+0.0700 [-0.0117, +0.1750]</td><td>0.2305</td><td>0.0675*</td><td>+0.0057 [+0.0025, +0.0095]</td><td>0.002155</td><td>2.7800</td></tr><tr><td>Without past traces</td><td>50.33%</td><td>-0.0183 [-0.0533, +0.0133]</td><td>0.3575</td><td>0.1205*</td><td>+0.0587 [+0.0497, +0.0709]</td><td>0.000002</td><td>0.2330</td></tr><tr><td>Edit descriptions only</td><td>51.33%</td><td>-0.0083 [-0.0483, +0.0350]</td><td>0.7609</td><td>0.1176*</td><td>+0.0558 [+0.0471, +0.0655]</td><td>0.000002</td><td>0.3839</td></tr><tr><td>Without coding agent harness</td><td>52.50%</td><td>+0.0033 [-0.0417, +0.0500]</td><td>0.9446</td><td>0.0825*</td><td>+0.0207 [+0.0161, +0.0258]</td><td>0.000002</td><td>2.2481</td></tr><tr><td>DRAFT</td><td>53.50%</td><td>+0.0133 [-0.0250, +0.0567]</td><td>0.5988</td><td>0.0822*</td><td>+0.0204 [+0.0150, +0.0269]</td><td>0.000002</td><td>2.4367</td></tr><tr><td>ToolObserver</td><td>55.83%</td><td>+0.0367 [-0.0067, +0.0850]</td><td>0.1650</td><td>0.0870*</td><td>+0.0252 [+0.0162, +0.0360]</td><td>0.000002</td><td>16.2171</td></tr><tr><td>GEPA</td><td>51.50%</td><td>-0.0067 [-0.0317, +0.0233]</td><td>0.7539</td><td>0.0748*</td><td>+0.0130 [+0.0079, +0.0202]</td><td>0.000008</td><td>0.8042</td></tr><tr><td rowspan="8">Retail</td><td>Baseline</td><td>77.00%</td><td></td><td></td><td>0.0707</td><td></td><td></td><td></td></tr><tr><td>AT-AT</td><td>74.67%</td><td>-0.0233 [-0.0600, +0.0092]</td><td>0.2313</td><td>0.0724</td><td>+0.0017 [-0.0001, +0.0034]</td><td>0.070629</td><td>1.9300</td></tr><tr><td>Without past traces</td><td>75.42%</td><td>-0.0158 [-0.0433, +0.0133]</td><td>0.3129</td><td>0.1383*</td><td>+0.0675 [+0.0610, +0.0746]</td><td>0.000010</td><td>0.2174</td></tr><tr><td>Edit descriptions only</td><td>74.08%</td><td>-0.0292 [-0.0617, +0.0033]</td><td>0.0992</td><td>0.1453*</td><td>+0.0745 [+0.0668, +0.0832]</td><td>0.000010</td><td>0.5613</td></tr><tr><td>Without coding agent harness</td><td>74.92%</td><td>-0.0208 [-0.0517, +0.0067]</td><td>0.1979</td><td>0.0802*</td><td>+0.0094 [+0.0074, +0.0115]</td><td>0.000010</td><td>3.8496</td></tr><tr><td>DRAFT</td><td>75.50%</td><td>-0.0150 [-0.0567, +0.0292]</td><td>0.5296</td><td>0.0871*</td><td>+0.0163 [+0.0132, +0.0195]</td><td>0.000010</td><td>3.9785</td></tr><tr><td>ToolObserver</td><td>75.92%</td><td>-0.0108 [-0.0500, +0.0275]</td><td>0.6196</td><td>0.0926*</td><td>+0.0219 [+0.0179, +0.0259]</td><td>0.000010</td><td>42.1400</td></tr><tr><td>GEPA</td><td>76.08%</td><td>-0.0092 [-0.0375, +0.0183]</td><td>0.5731</td><td>0.0798*</td><td>+0.0091 [+0.0072, +0.0111]</td><td>0.000010</td><td>0.8765</td></tr><tr><td rowspan="8">Telecom</td><td>Baseline</td><td>29.50%</td><td></td><td></td><td>0.0996</td><td></td><td></td><td></td></tr><tr><td>AT-AT</td><td></td><td>36.75%*+0.0725 [+0.0292, +0.1242]</td><td>0.0033</td><td>0.0986</td><td>-0.0010 [-0.0040, +0.0017]</td><td>0.508165</td><td>10.6900</td></tr><tr><td>Without past traces</td><td>30.42%</td><td>+0.0092 [-0.0225, +0.0425]</td><td>0.6281</td><td>0.2850*</td><td>+0.1854 [+0.1772, +0.1932]</td><td>0.000010</td><td>0.2888</td></tr><tr><td>Edit descriptions only</td><td>31.17%</td><td>+0.0167 [-0.0217, +0.0592]</td><td>0.4547</td><td>0.2857*</td><td>+0.1861 [+0.1755, +0.1959]</td><td>0.000010</td><td>1.1342</td></tr><tr><td>Without coding agent harness</td><td>30.50%</td><td>+0.0100 [-0.0250, +0.0467]</td><td>0.6172</td><td>0.1146*</td><td>+0.0150 [+0.0125, +0.0175]</td><td>0.000010</td><td>4.4205</td></tr><tr><td>DRAFT</td><td>25.58%*</td><td>-0.0392 [-0.0642, -0.0167]</td><td>0.0026</td><td>0.1259*</td><td>+0.0263 [+0.0238, +0.0287]</td><td>0.000010</td><td>4.5630</td></tr><tr><td>ToolObserver</td><td>31.67%</td><td>+0.0217 [-0.0283, +0.0792]</td><td>0.4686</td><td>0.1276*</td><td>+0.0280 [+0.0249, +0.0311]</td><td>0.000010</td><td>49.1224</td></tr><tr><td>GEPA</td><td>29.25%</td><td>-0.0025 [-0.0308, +0.0267]</td><td>0.9102</td><td>0.1132*</td><td>+0.0136 [+0.0117, +0.0154]</td><td>0.000010</td><td>0.8846</td></tr></table>

## B Tool implementation vs. description-only optimizer edits: a qualitative analysis

This section contains a qualitative analysis on an ablation reported in Tab. 1 comparing the main AT-AT condition, which is permitted to edit both tool descriptions and implementations, with a variant of AT-AT restricted to editing existing tool descriptions only. Implementation edits can help an agent in two distinct ways: by changing what an existing tool returns (feedback shaping), and by adding a new tool the agent can call (expanding the action space). We illustrate each with paired transcripts. Tab. 8 summarizes the tools edited by each condition.

Table 8: Tools edited by full-scope AT-AT and by the descriptions-only ablation. Tags mark which part of each tool changed: D = description, C = code, C+D = both, new = new tool.
<table><tr><td>Domain</td><td></td><td>Full AT-AT</td><td></td><td>Descriptions only</td><td></td></tr><tr><td>Airline</td><td>Tools edited</td><td>book_reservation cancel_reservation get_flight_status get_reservation_details get_user_details search_direct_flight search_onestop_flight C</td><td>D D D C+D D C+D</td><td>update_reservation_baggages update_reservation_flights</td><td>D D</td></tr><tr><td></td><td>Qualitative changes</td><td>update_reservation_flights update_reservation_passengers age, 24-hour eligibility, flight-date bounds, and past/upcoming status. • Direct and one-stop search results now carry their flight dates.</td><td>D D • Added timestamp parsing and reservation-time helpers. • Reservation lookup now returns current time, booking</td><td>• Clarified that baggage totals are absolute. • Documented paid-bag charges and certificate restrictions. • Added flight-change guidance covering basic-economy restrictions, downgrade refunds.</td><td></td></tr><tr><td></td><td>Tools edited</td><td>booking, cancellation, lookup, search, certificate. baggage, flight-change, and passenger-change operations. exchange_delivered_order_items D get_item_details modify_pending_order_items C+D</td><td>C</td><td>cancel_pending_order exchange_delivered_order_items D find_user_id_by_name_zip</td><td>D D</td></tr><tr><td></td><td>Qualitative changes</td><td>Corrected item lookup for object-backed products (get_item_details). Corrected pending-order modification to apply the requested replacement variant&#x27;s price and options. Clarified that exchanges and pending-order modifications</td><td></td><td>return_delivered_order_items D • Added explicit confirmation and irreversibility guidance. • Distinguished whole-order cancellation from item modification. Required batching one-time return or exchange actions.</td><td></td></tr><tr><td></td><td>Tools edited</td><td>or remove items. get_line_by-phone get_customer_by_phone</td><td>new C+D</td><td>Clarified same-product variant constraints and preferred customer lookup. get_details_by_id send_payment_request</td><td>D</td></tr><tr><td></td><td></td><td>get_data_usage enable_roaming disable_roaming refuel_data</td><td>C+D C+D D C+D</td><td>enable_roaming disable_roaming</td><td>D D D</td></tr></table>

## B.1 Comparisons of agent trajectories

All highlighted pairs use the same task, trial, and seed.

## B.1.1 Feedback shaping: richer outputs from existing tools

Airline task example: computed state prevents an unsupported refund. In Airline task 48, the user claimed that reservation 3RK2T9 had been booked 10 hours earlier and demanded a full refund (Tab. 9). Both agents called the same reservation lookup, and both had access to the current time, which is stated in the domain policy. Only the form of the tool output differed: the full-scope implementation returned the elapsed time and an explicit 24-hour eligibility flag, which the agent used to reject the user’s claim, whereas the descriptions-only agent received only the booking timestamp and accepted the claim.

Table 9: Airline task: computed state returned by an existing tool prevents an unsupported refund.
<table><tr><td>Trajectory step</td><td>Full AT-AT</td><td>Descriptions only</td></tr><tr><td>Tool feedback</td><td>created_at=2024-05-02, current_datetime=2024-05-15, hours_since_booking=320.95,</td><td>created_at=2024-05-02; no elapsed time or eligibility flag</td></tr><tr><td>Agent response</td><td>within_24h_of_booking=False hours ago)&quot;; refuses refund</td><td>&quot;Booked more than 24 hours ago (about 13 days ago, not 10 “Booking made about 10 hours ago (within 24 hours)&quot;; offers and executes cancellation</td></tr><tr><td>Outcome</td><td>Correctly preserves reservation; reward 1</td><td>Incorrectly cancels and refunds; reward 0</td></tr></table>

## B.1.2 Expanding the action space: adding a new tool

Telecom task example: adding a new tool enables a direct line lookup that avoids misdiagnosing the problem. In Telecom task [mobile\_data\_issue]bad\_vpn|data\_mode\_off|data\_usa ge\_exceeded|user\_abroad\_roaming\_disabled\_off[PERSONA:None], the customer, whose phone number is on line L1002, had used up that line’s data allowance (Tab. 10). The full-scope optimizer added a new tool, get\_line\_by\_phone, that maps a phone number directly to its line. The descriptions-only agent had no such tool: the customer lookup returned three line IDs, and the agent inspected the first, L1001, without noticing that its phone number did not match the caller’s. It then enabled roaming on and diagnosed the wrong line. The failure did not stem from missing derived fields: the existing usage tool reports usage and limit, and descriptions-only trajectories that queried L1002 recognized the overage. Across trials of this task, the full-scope agent resolved L1002 in all 5 trials, while the descriptions-only agent stopped at L1001 in 7 of 30.

Table 10: Telecom task: adding a new tool enables a direct line lookup that avoids misdiagnosing the problem.
<table><tr><td>Trajectory step</td><td>Full AT-AT</td><td>Descriptions only</td></tr><tr><td>Line lookup</td><td>New tool get_line_by-phone(555-123-2002) returns customer id=C1001. line L1002</td><td>get_customer_by-phone(555-123-2002) returns 1ine_ids=[L1001,L1002,L1003]; agent queries the first, L1001, which returns</td></tr><tr><td>Agent action</td><td>Enables roaming on L1002</td><td>phone_number=555-123-2001 Overlooks the mismatched number; enables roaming on L1001</td></tr><tr><td>Diagnosis</td><td>Usage on L1002: 15.1/15.0 GB; attributes outage to exhausted data and refuels 2 GB</td><td>Usage on L1001: 3.2/5.0 GB; concludes the cap was not reached and transfers to a human agent</td></tr><tr><td>Outcome</td><td>Reward 1</td><td>Reward 0</td></tr></table>

The same tool also made successful trajectories more efficient. In Telecom task [mobile\_data\_is sue]airplane\_mode\_on|data\_saver\_mode\_on|user\_abroad\_roaming\_disabled\_on[PE RSONA:None], both agents passed, but where the full-AT-AT agent went directly from the phone number to the line, the descriptions-only agent needed a clarification turn and two exploratory line lookups (Tab. 11, which shows only the steps where the line is identified).

## B.2 Interpretation

The analysis is consistent with the hypothesis that editing tool implementations provides a greater capability for a tool optimizer to improve agent outcomes. Description-only editing can change how an agent interprets an existing interface, but it cannot add a missing lookup, repair a mutation, or cause a tool to expose latent state. Airline and Telecom contained failures of this form. In Airline, refund eligibility depended on comparing timestamps, and an existing tool that returned the comparison directly changed the agent’s response (feedback shaping). In Telecom, diagnosis depended on resolving the correct line, and a new lookup tool removed the error-prone step of iterating over a customer’s lines. Retail was already high-performing and showed almost no difference between conditions, consistent with fewer failures being limited by the tool implementation itself.

Table 11: Telecom task: the new lookup tool removes exploratory line searches.
<table><tr><td>Trajectory step</td><td>Full AT-AT</td><td>Descriptions only</td></tr><tr><td>Tool feedback</td><td>get_line_by_phone directly returns customer_id=C1001,line_id=L1002,</td><td>Customer lookup returns 1ine_ids=[L1001,L1002,L1003] without identifying</td></tr><tr><td>Agent action</td><td>roaming_enabled=False Immediately calls enable_roaming(C1001, L1002)</td><td>the relevant line Asks the user to identify the line, then queries L1001 and L1002 separately before enabling roaming</td></tr><tr><td>Outcome</td><td>Reward 1</td><td>Reward 1</td></tr></table>

## C Additional Analysis

Effectiveness of coding agents as text optimizers. Our results here, as well as in prior work, suggest that using a coding agent as an optimizer is helpful: coding agents’ ability to reason over multiple files and make selective edits in a multi-turn fashion may be useful in text optimization settings. Here, AT-AT outperformed the ablation method without the coding agent harness, as well as the prior tool and prompt optimizers that do not use a coding agent harness, DRAFT, ToolObserver, and GEPA [Qu et al., 2025, Hallinan et al., 2026, Agrawal et al., 2026]. This echoes the outperformance of FAPO [Kassianik et al., 2026], a recent prompt optimization method using a Claude Code coding agent as an optimizer, compared to GEPA [Agrawal et al., 2026], the previous state-of-the-art prompt optimizer. Meta-Harness [Lee et al., 2026b] also uses the Claude Code coding agent as a harness optimizer. LLM capabilities have advanced steadily, and coding agents utilize the ability of LLMs to create and execute upon long-horizon plans. Given the promise of coding agents, we encourage future text optimization work to experiment with coding agents as part of their methodology. The Pi agent [Earendil Inc. and Contributors, 2026] that we use in AT-AT, and similar other coding agents that are open-source, lightweight, and extensible, may be good candidates for further experiments and worthwhile to compare to closed coding agents like Claude Code.

Evaluation stochasticity. Agent optimization involves two distinct sources of stochasticity: evaluation noise and optimizer noise. Evaluation noise arises because repeated agent executions can produce different trajectories and outcomes even when the model, harness, tasks, environment, and decoding configuration are held constant [Mustahsan et al., 2025]. Differences in evaluation infrastructure configuration or infrastructure crashes can also lead to task failures and contribute to evaluation noise [Anthropic, 2026]. Consequently, rankings based on single benchmark point estimates, particularly when the reported differences are small relative to run-to-run variation, should not be interpreted as reliable evidence that one system outperforms another. Our paired permutation statistical testing protocol accounts for evaluation noise and provides statistical evidence for AT-AT’s improvements on τ<sup>2</sup>-bench-verified. However, it does not address optimizer noise: repeated runs of an LLM-based optimizer may produce different tool updates and, consequently, different downstream effects. Because we evaluate one toolset from each optimizer, our reported results are conditional on those particular optimization runs. Future work should repeat the complete optimization-and-evaluation pipeline to characterize both sources of variation, reporting uncertainty across optimizer runs as well as repeated downstream evaluations. Given the imperfect nature of benchmarks and evaluations, the ultimate test of an ML method is if it works in real-world scenarios; our production MCP server experiments demonstrate that AT-AT may provide value in a deployment setting, although replication across deployments and time remains necessary to establish robustness.

Failure to achieve gains on Retail. No optimizer significantly changed Retail reward (Tab. 1); all optimized toolsets scored slightly below the 77.00% baseline (74.08–76.08%; all $p \geq 0 . 0 9 9 )$ We attribute this pattern to two factors. First, Retail offers limited optimization headroom. The baseline reward is already quite high at 77%, and an LLM audit of 24 failed baseline trajectories per domain found that most Retail failures were agent reasoning errors that occurred even though the tools had returned the necessary information. Failures classified by an LLM as addressable via tool implementations accounted for an estimated 3.1% of Retail simulations, compared with 8.9% in Telecom, where AT-AT achieved its only significant gain. Second, the small declines are within evaluation noise. GEPA’s selected toolset is byte-identical to the unoptimized tools, yet its evaluation scored 0.92 points below the baseline (76.08%), and the edited toolsets differ from this control by only −2.00 to −0.17 points (all $p \geq 0 . 2 0 )$ .

## D Enterprise MCP Edits: Further Details

## D.1 Representative tasks

Table 12 gives a high-level description of representative tasks for each of the three enterprise MCP servers evaluated in Table 4.

Table 12: Representative tasks for each enterprise MCP server.
<table><tr><td>MCP server</td><td>Representative tasks</td></tr><tr><td></td><td>Core Developer Tools • Interpret an application performance metric • Investigate VPN-related host reachability</td></tr><tr><td>Observability</td><td>• Diagnose repeated software installation failures • Classify date-scoped support threads</td></tr><tr><td>Collaboration</td><td>• Evaluate automated support answer quality and recommend fixes</td></tr><tr><td>Toolbox</td><td>• Compare activity across support channels</td></tr><tr><td>Enterprise</td><td>• Assess whether dependency changes caused canary latency</td></tr><tr><td>Engineering Context</td><td>• Diagnose a linked CI/CD failure</td></tr><tr><td>and Best Practices</td><td>• Reconcile data compaction settings with an underlying messaging topic</td></tr><tr><td></td><td></td></tr></table>

## D.2 Examples of updates to MCP tool descriptions

Tab. 13 summarizes representative AT-AT updates from the enterprise MCP evaluation. The examples are paraphrased, with organization-specific tool names and infrastructure details omitted. The accepted candidates changed natural-language descriptions only; tool names, parameter contracts, and backend implementations were unchanged.

Table 13: Paraphrased examples of AT-AT description updates for enterprise MCP tools.
<table><tr><td>Tool category</td><td>Paraphrased guidance added by AT-AT</td></tr><tr><td>Dev Tools</td><td>Use namespace-specific retrieval only when the exact registered namespace is known. Avoid guessing namespaces from product or team names; otherwise, prefer cross-source or dedicated search tools.</td></tr><tr><td>Collab Toolbox</td><td>Normalize channel references to the identifier format expected by each tool and resolve display names before calling tools that require channel IDs. Distinguish access-denied or unindexed channels from genuine zero-result searches, and treat an unresolved workspace or channel identifier as an identity or access problem</td></tr><tr><td>Eng. Context</td><td>rather than an empty result. Clarify that the tool searches indexed documentation, code, and discussion content rather than live executions, incidents, dashboards, metrics, or logs. Avoid near-duplicate retries after no relevant context is found, and use focused calls for</td></tr></table>

## D.3 Example task evaluation

The following example replaces organization-specific metric and service names while preserving the evaluated request and checklist structure.

Task prompt. What does application.ui\_delay\_by\_label measurefor client-device telemetry, and how is ui\_label attributed?

Success criteria. Before either the baseline or optimized run was executed, GPT-5.6 Luna generated the following user-visible criteria from the task prompt alone:

1. Clearly define the metric’s measured quantity and scope for client-device telemetry.

2. Explain how ui\_label is assigned or attributed, including the relevant source or attribution rule.

3. Distinguish the metric measurement from the label-attribution process without unsupported speculation.

Fixed content rubric. Luna converted those criteria, before seeing either trajectory, into three independently scored clauses: two content clauses corresponding to the first two criteria and one correction clause corresponding to the third.

LLM-as-a-judge procedure. We used GPT-5.6 Luna in two separate absolute-scoring calls. The task-success judge received the success criteria and one rendered transcript, then returned a binary met\_criteria decision with a one-sentence reason. Separately, the content-rubric judge received the fixed checklist and the rendered transcript. For every clause it returned met, partial, or not\_met, a short reason, and an exact transcript quote for positive credit. Code verified that each quote occurred in the transcript and downgraded unsupported verdicts. Neither judge saw the other transcript or a label indicating whether the transcript was from before or after optimization.

## E Composition with harness optimization

Eq. 1 treats the harness as a fixed component of each agent. However, the harness can also be optimized. Let $\mathcal { X } _ { H }$ and $\mathcal { X } _ { E }$ denote separate task distributions used for harness and environmental tool optimization, respectively, with corresponding tasks $x _ { H }$ and $x _ { E }$ and trajectories $\tau _ { H }$ and $\tau _ { E }$ . Let $T _ { E } ^ { 0 } = \dot { ( } T _ { E _ { D } } ^ { 0 } , T _ { E _ { C } } ^ { 0 } )$ denote the fixed initial environmental toolset. Harness engineering optimizes H to maximize expected performance over tasks and trajectories [Lee et al., 2026b]:

$$
H ^ { * } = \arg \operatorname* { m a x } _ { H } \mathbb { E } _ { \boldsymbol { x } _ { H } \sim \mathcal { X } _ { H } , \boldsymbol { \tau } _ { H } \sim p _ { M } ( \cdot \vert H , \boldsymbol { x } _ { H } , T _ { E _ { D } } ^ { 0 } , T _ { E _ { C } } ^ { 0 } ) } \left[ r ( \boldsymbol { \tau } _ { H } , \boldsymbol { x } _ { H } ) \right] .\tag{3}
$$

Harness optimization can be composed with environmental tool optimization. A harness optimized on $\mathcal { X } _ { H }$ can be held fixed and used to generate trajectories on $\mathcal { X } _ { E }$ for optimizing the environmental tools. For a population of models $M \bar { \sim } \mathcal M$ , this two-stage objective is

$$
\begin{array} { r l } & { \qquad T _ { E _ { D } } ^ { * } , T _ { E _ { C } } ^ { * } = \arg \underset { T _ { E _ { D } } , T _ { E _ { C } } } { \operatorname* { m a x } } \mathbb { E } _ { M \sim \mathcal { M } , \boldsymbol { x } _ { E } \sim \mathcal { X } _ { E } , \tau _ { E } \sim p _ { M } ( \cdot \vert H ^ { * } , \boldsymbol { x } _ { E } , T _ { E _ { D } } , T _ { E _ { C } } ) } \left[ r ( \tau _ { E } , \boldsymbol { x } _ { E } ) \right] } \\ & { \mathrm { s u b j e c t ~ t o } \quad \quad H ^ { * } = \arg \underset { H } { \operatorname* { m a x } } \mathbb { E } _ { M \sim \mathcal { M } , \boldsymbol { x } _ { H } \sim \mathcal { X } _ { H } , \tau _ { H } \sim p _ { M } ( \cdot \vert H , \boldsymbol { x } _ { H } , T _ { E _ { D } } ^ { 0 } , T _ { E _ { C } } ^ { 0 } ) } \left[ r ( \tau _ { H } , \boldsymbol { x } _ { H } ) \right] . } \end{array}\tag{4}
$$

## F OpenCode tool edits on TerminalBench-2 and TBLite

This appendix lists the exact edits AT-AT made to OpenCode’s tools in the experiments of Sec. 4.2. In both benchmarks, the accepted candidate changed tool descriptions only (two .txt files and zero .ts files). Tool names, parameter schemas, implementations, and return shapes were unchanged. Lines beginning with - were removed from the original description and lines beginning with + were added. Unchanged lines are omitted.

## F.1 TerminalBench-2

edit.txt

-- The edit will FAIL if \`oldString\` is not found in the file with an error "oldString not found in content".

-- The edit will FAIL if \`oldString\` is found multiple times in the file with an error "Found multiple matches for oldString. Provide more surrounding lines in oldString to identify the correct match." Either provide a larger string with more surrounding context to make it unique or use \`replaceAll\` to change every instance of \`oldString\`.

-- Use \`replaceAll\` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.

+- The edit will FAIL if \`oldString\` is not found in the file. Re-read the file and copy the exact bytes (including whitespace, indentation, and line endings) into \`oldString\`.

+- The edit will FAIL if \`oldString\` matches multiple places in the file. Either expand \`oldString\` with more surrounding context so it becomes unique, or set \`replaceAll: true\` to change every instance.

+- Use \`replaceAll\` for renaming an identifier across a file or applying the same change to every occurrence.

+- \`oldString\` may be empty ONLY when the file does not yet exist; in that case the tool creates it with \`newString\` as its full contents. To create a new file that already exists, use the Write tool instead.

+- Prefer one larger, well-scoped edit over many tiny consecutive edits to the same region; each edit re-reads and re-writes the whole file.

write.txt

-- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first. +- If this is an existing file you intend to modify (not fully replace), prefer the Edit tool. Only use Write when you intend to replace the whole file contents. +- The file's parent directories are created automatically if they don't exist.

## F.2 TBLite

grep.txt

-- Returns file paths and line numbers with at least one match sorted by modification time

+- Returns file paths and line numbers with at least one match sorted by modification time (up to 100 matches; if truncated, narrow the search with \`path or \`include\`, or make the pattern more specific)

read.txt

-- By default, this tool returns up to 2000 lines from the start of the file.

-- The offset parameter is the line number to start from (1-indexed).

-- To read later sections, call this tool again with a larger offset.

+- By default, this tool returns up to 2000 lines from the start of the file, and additionally caps the output at \~50 KB of decoded text (whichever limit is hit first). If the byte cap fires, the response includes the last line shown and a suggested \`offset=\` to resume from — call the tool again with that offset to continue reading.

+- The offset parameter is the line number to start from (1-indexed). The limit parameter sets the maximum number of lines returned (still subject to the byte cap).

+- To read later sections, call this tool again with a larger offset. Prefer a single larger window (e.g. limit=2000) over many small windows; the byte cap alone will trim it if the content is dense.

## G AT-AT prompts

This appendix lists the prompts given to the Pi agent optimizer in AT-AT. Prompt text is reproduced verbatim, except that line breaks within paragraphs are reflowed.

System prompt. The system prompt is the concatenation, separated by blank lines, of (1) the optimization objective, (2) benchmark-specific editing rules, and (3) the generalization instruction.

Optimization objective   
You are given an agent's tools (source you may edit) and transcripts of the agent   
attempting tasks with them, each labelled with its reward. Edit the tools so the   
agent succeeds on more tasks.   
Read the transcripts for places where the tools — not the agent's judgment — got   
in the way. Look for: the agent picking the wrong tool or wrong arguments because   
the docs don't make a tool's purpose or usage clear; results it can't act on   
(unclear, incomplete, or silent) that force it to guess or retry; and repeated   
calls or workarounds for something the tools should let it do in one step.   
Effective tools make the right action easy to choose, behave as their docs   
describe, and hand back what the agent needs to take the next step. Edit the   
tools to remove those obstacles, and read the successful transcripts too, so a   
change doesn't break what already works.   
Constraints: keep every tool's name and signature unchanged and the source valid;   
do not encode task identifiers, specific inputs, or expected answers into the   
tools.   
Apply edits with the edit tool. The harness validates the result; if it fails,   
fix the underlying issue rather than working around the check.

The editing rules state how to edit the tools of a given benchmark without breaking them. For τ<sup>2</sup>-bench-verified, where all tools are defined in a single tools.py file:

Editing rules: τ<sup>2</sup>-bench-verified   
The tools live in a single tools.py. Keep it importable: preserve all imports,   
class names, and @is\_tool method signatures (names, types, order).

For the τ<sup>2</sup>-bench-verified ablation that edits tool descriptions only (Sec. 4.1.1), the tool descriptions are extracted into an editable descriptions.md file:

Editing rules: τ<sup>2</sup>-bench-verified, descriptions only   
Edit ONLY the tool descriptions in descriptions.md (each \`## <tool>\` heading   
names a tool). The full tools.py is provided read-only for context; you cannot   
change code, only the description text under each heading.

For TerminalBench-2 and TBLite, where the optimized tools are TypeScript modules in the OpenCode harness:

Editing rules: OpenCode   
Each tool is a .ts module exporting \`export const <Name>Tool =   
Tool.define("<id>", {...})\`, optionally with a companion .txt description.   
Preserve every Tool.define id and the exported constant name; do not add imports   
or reference runtime-context symbols at module top level; preserve each execute()   
return shape { title, output, metadata }.

## Generalization instruction

The tools are scored on held-out tasks, not the ones in these transcripts. Prefer edits that address the general defect a transcript reveals over edits that only fit the specific observed case.

User prompt. In the main τ<sup>2</sup>-bench-verified experiments (Sec. 4.1), the first user message contains a digest of the agent traces followed by a short instruction. The digest includes every failing task’s trace, followed by the traces of the first three passing tasks. Each trace is truncated to 4000 characters and preceded by a header giving its task identifier, outcome, and reward:

```markdown
User prompt
### task <task id> [FAIL] (reward=<reward>)
<agent trace, truncated to 4000 characters>
### task <task id> [PASS] (reward=<reward>)
<agent trace, truncated to 4000 characters>
Above are transcripts of an agent attempting tasks with the current tools, each
labelled with its reward. Edit the tools as instructed.
```

The descriptions-only ablation and the cross-LLM generalization experiment (Sec. 4.1), and the TerminalBench-2, TBLite (Sec. 4.2), and MCP (Sec. 4.3) experiments omit the digest from the user prompt. Instead, the full agent traces are written to the optimizer’s workspace as read-only files, one per task, together with an index file listing each trace’s task identifier, outcome, and reward. The user prompt directs the optimizer to these files:

## User prompt without trace digest

The complete baseline transcripts are available as read-only workspace files. Start with BASELINE\_TRANSCRIPTS\_INDEX.md, then use the read tool to inspect the relevant per-task files (in pages when they are large). Ground your edits in both failed and successful runs.

If a proposal fails validation, the optimizer’s session is resumed with the following user prompt instead:

## User prompt on retry

Your previous attempt did not pass validation. Re-read the files you edited, reconsider, and try again; fix the underlying issue rather than working around the check.