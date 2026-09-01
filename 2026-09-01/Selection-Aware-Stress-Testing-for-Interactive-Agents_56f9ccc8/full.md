# Selection-Aware Stress Testing for Interactive Agents

Yang Xu<sup>∗</sup> Purdue University xu1720@purdue.edu

Haixiang Sun Purdue University sun1321@purdue.edu

Chenang Li<sup>∗</sup> University of California, Irvine chenangl@uci.edu

Zhou Li University of California, Irvine zhou.li@uci.edu

Jiefu Zhang Purdue University zhan4018@purdue.edu

Vaneet Aggarwal Purdue University vaneet@purdue.edu

## Abstract

Agent evaluations often use one benchmark to choose a workflow and then search for task types where its advantage weakens, so both conclusions are selected from the same data. We introduce Selection-Aware Semantic Stress Testing (SASST), which learns a task reweighting from pre-execution features on discovery tasks and evaluates the same paired comparison on separate confirmation tasks. The protocol checks support and stability, uses joint bounds for all planned claims, and can return no claim. We prove conditional asymptotic validity under stated cluster assumptions. A forty-cluster audit finds Gaussian undercoverage and conservative Bonferroni t bounds. In one 480-episode τ-bench study, a 3.75 point discovery gain vanished on confirmation. A second-model study likewise confirmed neither a workflow benefit nor a stable stress rule.

## 1 Introduction

Interactive-agent evaluations increasingly inform workflow selection [1, 13]. A team may compare workflows that interleave reasoning and acting, add reflection or search-based planning, or compile retrieved skills [11, 17, 22, 24], then select the strongest on a heterogeneous benchmark. However, that winner may lose its advantage on a coherent mixture of subset tasks. If confirmed, such a reversal could support reconsidering the workflow choice for that task regime. Teams may therefore search many task types for where the selected workflow loses its overall advantage. Because the subgroup is chosen after inspecting these same outcomes, an extreme reversal may reflect sampling noise rather than a pattern that persists on new tasks, making independent confirmation necessary before changing the deployment decision. In existing works, [20] protects workflow comparison, [4, 7, 18, 21] find coherent error groups, [2] protects fixed group families, and [14] confirms frozen descriptors for a fixed model. These methods do not by themselves protect the full process of choosing an average winner and then searching for task types where its advantage weakens or reverses. We propose Selection-Aware Semantic Stress Testing (SASST). It keeps repeated runs of each task together and divides tasks into two sets: a discovery set for choosing a reweighting rule and a confirmation set for evaluating that frozen rule on tasks not used to choose it. Discovery-set workflow scores are used to choose among candidate rules based only on pre-execution attributes, such as goal length, required tools, or permission flags. Before any results from the confirmation set are inspected, the chosen rule is fixed. It then assigns weights to those tasks, and the same workflows are compared again. SASST therefore protects two data-dependent choices: the workflow winner and the rule that most weakens its advantage. It rejects rules supported by too few tasks or unstable under small changes to the discovery data, and may return no claim.

Contributions. SASST (i) asks whether a workflow that wins on average still wins when a coherent task type receives more emphasis, with all workflows evaluated on the same task mix; (ii) learns from discovery results a reusable rule based on task attributes known before execution, rejecting rules that rely on too few tasks or exceed a declared shift; (iii) freezes the rule and retests the same comparison on separate confirmation tasks, with uncertainty bounds for all planned claims; and (iv) returns confirmed, not confirmed, failed discovery, failed feasibility, or abstained, distinguishing supported, inconclusive, infeasible, and unstable findings. Theory, controlled studies, and agent studies evaluate the procedure.

## 2 Method

Tasks and the fixed comparison. Let $W \sim P$ denote one benchmark task before execution. It contains the goal, initial state, available tools, permissions, and other fixed task information. Scores produced by running workflows under different simulator seeds are observed outcomes, not being a part of $\dot { W }$ . All outcomes from the same task remain together as one task-level cluster. Before choosing a task-weighting rule, we fix the comparison plan denoted as A for both the discovery and the confirmation. Given the observed workflow scores and a set of task weights, A returns one performance estimate for each workflow. It also specifies how repeated scores are combined and, when applicable, how workflows are selected or weighted. Let ${ \bar { \pmb { \theta } } } ( Q ; A )$ be these estimates when Q provides the task weights, and let c specify the comparison target of interest. We define ${ \cal T } _ { c } ( Q ; \dot { \mathcal { A } } ) = c ^ { \top } \pmb { \theta } ( Q ; \mathcal { A } )$ . For example, when c aims at comparing Plan + Verifier with ReAct [22], $T _ { c } ( Q ; A )$ is their difference in weighted average success under the same task weights. Throughout the setting, SASST aims at changing task emphasis, not the workflows or the comparison plan. The stress target defined next asks whether this same comparison weakens when coherent task types receive more weight.

Stress target. A rule g uses only pre-execution task features to assign a bounded nonnegative score $s _ { g } ( W )$ . Larger scores give a task more emphasis. With $\begin{array} { r } { P s _ { g } = \mathbb { E } _ { P } [ \bar { s _ { g } } ( W ) ] } \end{array}$ ], define

$$
\omega _ { g } ( W ) = { \frac { s _ { g } ( W ) } { P s _ { g } } } , \qquad d P ^ { g } = \omega _ { g } d P , \qquad \Delta ( g ) = T _ { c } ( P ^ { g } ; A ) .\tag{1}
$$

Here $P ^ { g }$ is the benchmark distribution reweighted by $^ { g , }$ and $\Delta ( g )$ is the workflow contrast under that emphasis. It covers represented task types and allowed shifts, not absent tasks or arbitrary deployment traffic. We next explain how $g$ is chosen on discovery data and evaluated independently.

Discovery and confirmation. On discovery clusters D, the equal-weight comparison gives $\widehat { \Delta } _ { \mathcal { D } }$ The contribution $\widehat { \psi } _ { i }$ approximates how changing task i’s weight changes the final comparison, including any workflow-selection step inside ${ \mathcal { A } } .$ With $\begin{array} { r } { w _ { i } ( g ; \bar { \mathcal { D } } ) = s _ { g } ( \bar { W _ { i } } ) / \sum _ { \ell \in \mathcal { D } } s _ { g } ( \bar { W _ { \ell } } ) } \end{array}$ , rank candidate rules by

$$
\widehat { \Delta } _ { \mathcal { D } } ^ { \mathrm { l o c } } ( g ) = \widehat { \Delta } _ { \mathcal { D } } + \displaystyle \sum _ { i \in \mathcal { D } } \left\{ w _ { i } ( g ; \mathcal { D } ) - \frac { 1 } { | \mathcal { D } | } \right\} \widehat { \psi } _ { i } .\tag{2}
$$

Lower values predict a weaker contrast. Discovery outcomes choose among candidate functions, but every candidate uses only pre-execution features. The selected function is frozen before confirmation. Equation (2) ranks rules only during discovery. Confirmation later recomputes the exact contrast.

Density and effective sample size (ESS) checks prevent a few tasks from dominating. A radius limits the distance between the original and reweighted distributions of fixed task features. We use a finite-witness Kernel Max-Sliced Wasserstein (KMS) approximation [19], with maximum mean discrepancy (MMD) and linear max-sliced alternatives [8]. Resampling the discovery clusters checks rule stability. On confirmation clusters I, SASST applies the frozen rule, repeats the support checks, recomputes the exact paired comparison, and forms a joint uncertainty bound over all planned claims. Each planned claim fixes its contrast, direction, candidate rule family, allowed radius, thresholds, and inference rule. Algorithm 1 combines these choices, support checks, and confirmation into one decision.

Algorithm 1 SASST discovery and confirmation.   
Input: Fixed comparison plan A, independent task clusters, pre-execution features, planned claims ${ \mathcal { I } } _ { 0 } .$   
1: Split whole clusters into discovery D and confirmation I. Freeze workflows, candidate rule families, thresholds, and th   
inference rule.   
2: On D, select for each claim the feasible rule minimizing Equation (2). If none is feasible, recordfailed discovery.   
3: Freeze the rule and estimate its stability using only D. Apply it unchanged to I. If support fails, recordfailedfeasibility.   
4: For every support-feasible rule, compute the exact weighted comparison on I and its joint uncertainty bound.   
5: Record confirmed or not confirmed when stability passes. Otherwise report the estimate diagnostically and record   
abstained.   
Output: A complete status and diagnostic report for every planned claim.

The algorithm above defines the operational report. The theorem below states when its confirmation bounds are jointly valid.

Theorem 1 (Validity after discovery). Condition on the discovery data ${ \mathcal { F } } _ { { \mathcal { D } } } .$ For a finite predeclared family, suppose the confirmation cluster vectors $H _ { k } = ( H _ { k , m } : m \in \mathcal { T } _ { 0 } )$ are conditionally independent and identically distributed. Let $\begin{array} { r } { \widehat { \Delta } _ { m } = F _ { m } ( N ^ { - 1 } \sum _ { k = 1 } ^ { N } H _ { k , m } ) , \mu _ { m } = \mathbb { E } ( H _ { 1 , m } \mid \mathcal { F } _ { \mathcal { D } } ) } \end{array}$ , and $\bar { \Delta } _ { m } = F _ { m } ( \mu _ { m } )$ , where $H _ { k , m }$ contains every weighted numerator and denominator, every selector input, and both sides ofthe paired contrast. Assume bounded density ratios, denominators bounded away from zero, continuously differentiable $F _ { m }$ near $\mu _ { m } ,$ , finite cluster second moments, a finitedimensional Lindeberg condition, conditional covariance converging to $\Sigma ,$ and positive limiting variancesfor reported coordinates. Then $\sqrt { N } ( \widehat { \Delta } _ { m } - \Delta _ { m } : m \in \mathcal { I } _ { 0 } )$ is asymptotically Gaussian, and the cluster Gaussian multiplier consistently approximates its joint law. Pointwise intervals and signed simultaneous bands have asymptotic conditional coverage, includingfor a coordinate highlighted after inspecting the protectedfamily.

In plain language, once discovery fixes the rules, confirmation supports joint uncertainty statements for all planned claims. Appendix B gives the proof [3]. The theorem assumes identically distributed confirmation clusters, so the stratified designs in Studies F and G require an extension not supplied here. With hard workflow selection, the protected objects are the candidate contrasts, not the argmax. Section 3 audits the forty-cluster regime.

## 3 Validation

We test whether confirmation preserves coverage after rule discovery, what fails with forty clusters, and whether a supported signal remains detectable. For $M \geq 1 0 0 0$ , pointwise coverage is 94.75– 95.30% and six-slot simultaneous coverage is 94.95–95.50%. At M = 1000, same-data worst-tercile coverage is 91.6%, versus 93.6% with discovery and confirmation separated $( p = 0 . 0 3 2 )$ .

At N = 40, an initial audit found 92.2–92.8% Gaussian coverage; a later fresh-seed diagnostic found 90.55–93.50% Gaussian and 96.40–97.55% Bonferroni-t coverage across four null settings. With 200 clusters and adequate stressed ESS, a planted-fragility control selects the target rule in 92.1% of trials, confirms it conditionally in 70.1%, detects it end-to-end in 64.6%, and has 1.7–1.8% null familywise error.

Table 1: Inferential checks and the small-cluster safeguard.
<table><tr><td>Setting</td><td>Result</td><td>Reading</td></tr><tr><td>Pointwise,  $M \geq 1 0 0 0$ </td><td>94.75–95.30%</td><td>near 95%</td></tr><tr><td>Six-slot simultaneous</td><td>94.95 / 95.50%</td><td>near 95%</td></tr><tr><td>Ten-slot Gaussian,  $N = 4 0$ </td><td>92.2–92.8%</td><td>undercoverage</td></tr><tr><td>Ten-slot Bonferroni  $t , N = 4 0$ </td><td>96.4–97.55%</td><td>conservative</td></tr><tr><td>Positive control,  $N = 2 0 0$ </td><td>64.6% end-to-end</td><td>signal found</td></tr></table>

Additional audits test the rule-search safeguards: no geometry dominates, a declared radius can exclude a planted rule, support checks reject concentrated rules, and a detectable real-feature effect is reported as abstained when its description is unstable.

## 4 Agent studies

We apply the full protocol to two models. ReAct [22] is the baseline tool-use loop; Plan + Verifier adds planning and model-based verification; Confirmation Gate requests approval before declared irreversible actions. For each model, two slots test average success against ReAct, six test the same comparisons under three allowed task shifts, and two compare the enhanced workflows. Each study uses the same 80 τ-bench tasks [23], two simulator seeds, and three workflows (480 episodes). Task ID is the independent cluster, whole tasks are split 40/40 for discovery and confirmation, and rules below the predeclared 0.60 stability threshold are abstained.

Table 2: Across-model results. Confirmation is the discovery-selected workflow minus ReAct; counts are confirmed / not confirmed / abstained / failed feasibility.
<table><tr><td>Model</td><td>Discovery result</td><td></td><td>Confirmation Final counts</td></tr><tr><td>Qwen3-8B</td><td>Plan+Verifier (+3.75 pp)</td><td>0.0 pp</td><td> $0 / 2 / 8 / 0$ </td></tr><tr><td>Qwen2.5-7B</td><td>Confirmation Gate selected</td><td>+2.5 pp</td><td>0/2/6/2</td></tr></table>

The Qwen3 discovery advantage disappeared on confirmation. Qwen2.5 had lower overall success, selected a different workflow, and produced different stress rules. Maximum rule stability was only 0.37 and 0.27, respectively, so neither model yielded a reusable stress description or confirmed workflow benefit.

These studies do not establish equivalence or safety. Forty confirmation clusters and stressed ESS near 20 provide little power for effects of practical size, many tasks fail under every workflow, and the Qwen3 kernel bandwidth made most task embeddings nearly orthogonal. The second study reuses the same task pool, so it is a model replication rather than an independent benchmark replication.

## 5 Decision interpretation and scope

Table 3: Decision status returned for every predeclared slot.
<table><tr><td>Status</td><td>Meaning</td></tr><tr><td>Confirmed</td><td>Support and stability pass, and the joint uncertainty bound supports the planned claim.</td></tr><tr><td>Not confirmed</td><td>Checks pass, but the bound does not support the claim; this is not equivalence.</td></tr><tr><td>Failed discovery Failed feasibility</td><td>No candidate rule satisfies the discovery constraints. The selected task weighting lacks confirmation support or violates a predeclared</td></tr><tr><td></td><td>check.</td></tr><tr><td>Abstained</td><td>The effect can be estimated, but the rule is too unstable to reuse.</td></tr></table>

The positive control shows that the protocol can confirm a planted fragility when the rule is eligible and confirmation has adequate support. The Qwen3 study shows the opposite case: a +3.75 pp discovery gain and an apparently weak segment could have motivated a verifier, but the average gain vanishes on confirmation and the segment description is unstable. The protected report supports neither claim and records what is missing rather than searching again on the same confirmation-task results.

Reported evidence. For every planned claim, SASST reports the frozen rule, estimated comparison, uncertainty bound, support, stability, geometry diagnostics, and final status. These fields explain a missing claim: low support calls for more independent task clusters; instability calls for better features or rule geometry; and a non-confirmed bound calls for more power or a redesigned intervention.

Limitations. The theorem requires conditionally i.i.d. confirmation clusters, bounded task weights, stable denominators, honest discovery, and smooth comparison functions. The Bonferroni t safeguard is supported by four audited null settings rather than a universal finite-sample theorem. Empirically, the studies use one τ-bench task pool, low-success open models, the same model as actor and user simulator, and forty confirmation tasks; the second model is not an independent benchmark replication. The next protected study needs fresh task clusters, score-blind pilot calibration of geometry and radius, an outcome aligned with the intended policy or state transition, an independent checker where relevant, and a power plan based on stressed ESS; more seeds on the same tasks do not replace independent clusters.

Operational scope. SASST supports decisions only for the benchmark and allowed task shifts; it is not a deployment-security certificate. It distinguishes supported, underpowered, out-of-scope, and unstable claims before a workflow or control is released.

## References

[1] Anthropic. Demystifying evals for ai agents. Anthropic Engineering, 2026. URL https:// www.anthropic.com/engineering/demystifying-evals-for-ai-agents. Published January 9, 2026.

[2] John J Cherian and Emmanuel J Candès. Statistical inference for fairness auditing. Journal of machine learning research, 25(149):1–49, 2024.

[3] Victor Chernozhukov, Denis Chetverikov, and Kengo Kato. Gaussian approximations and multiplier bootstrap for maxima of sums of high-dimensional random vectors. The Annals of Statistics, pages 2786–2819, 2013.

[4] Greg d’Eon, Jason d’Eon, James R Wright, and Kevin Leyton-Brown. The spotlight: A general method for discovering systematic errors in deep learning models. In Proceedings ofthe 2022 ACM Conference on Fairness, Accountability, and Transparency, pages 1962–1981, 2022.

[5] Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H. Laradji, Manuel Del Verme, Tom Marty, David Vazquez, Nicolas Chapados, and Alexandre Lacoste. WorkArena: How capable are web agents at solving common knowledge work tasks? In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 11642–11662. PMLR, 2024.

[6] John C Duchi and Hongseok Namkoong. Learning models with uniform performance via distributionally robust optimization. The Annals ofStatistics, 49(3):1378–1406, 2021.

[7] Sabri Eyuboglu, Maya Varma, Khaled Kamal Saab, Jean-Benoit Delbrouck, Christopher Lee-Messer, Jared Dunnmon, James Zou, and Christopher Re. Domino: Discovering systematic errors with cross-modal embeddings. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=FPCMqjI0jXN.

[8] Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel two-sample test. The journal ofmachine learning research, 13:723–773, 2012.

[9] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. In International Conference on Learning Representations, volume 2024, pages 52989–53046, 2024.

[10] Jiarui Lu, Thomas Holleis, Yizhe Zhang, Bernhard Aumayer, Feng Nan, Haoping Bai, Shuang Ma, Shen Ma, Mengyu Li, Guoli Yin, et al. Toolsandbox: A stateful, conversational, interactive evaluation benchmark for llm tool use capabilities. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 1160–1183, 2025.

[11] Xiangcheng Meng, Shu Wang, and Yixiang Fang. Skillrae: Agent skill-based context compilation for retrieval-augmented execution. arXiv preprint arXiv:2605.10114, 2026.

[12] Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general ai assistants. In International Conference on Learning Representations, volume 2024, pages 9025–9049, 2024.

[13] National Institute of Standards and Technology. Ai agent standards initiative. NIST, 2026. URL https://www.nist.gov/artificial-intelligence/ ai-agent-standards-initiative. Announced February 17, 2026; accessed August 2026.

[14] Vyzantinos Repantis, Ameya Gawde, and Harshvardhan Singh. Decoy-calibrated failure audits for language models. arXiv preprint arXiv:2606.09046, 2026.

[15] Shiori Sagawa\*, Pang Wei Koh\*, Tatsunori B. Hashimoto, and Percy Liang. Distributionally robust neural networks. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=ryxGuJrFvS.

[16] Yongliang Shen, Kaitao Song, Xu Tan, Wenqi Zhang, Kan Ren, Siyu Yuan, Weiming Lu, Dongsheng Li, and Yueting Zhuang. Taskbench: Benchmarking large language models for task automation. Advances in Neural Information Processing Systems, 37:4540–4574, 2024.

[17] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023.

[18] Nimit Sohoni, Jared Dunnmon, Geoffrey Angus, Albert Gu, and Christopher Ré. No subclass left behind: Fine-grained robustness in coarse-grained classification problems. Advances in Neural Information Processing Systems, 33:19339–19352, 2020.

[19] Jie Wang, March Boedihardjo, and Yao Xie. Statistical and computational guarantees of kernel max-sliced Wasserstein distances. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 62373– 62400. PMLR, 2025.

[20] Yang Xu, Jiefu Zhang, Haixiang Sun, Zihan Zhou, Tianyu Cao, and Vaneet Aggarwal. Towards reliable llm evaluation: Correcting the winner’s curse in adaptive benchmarking. arXiv preprint arXiv:2605.05973, 2026.

[21] Yuzhe Yang, Haoran Zhang, Dina Katabi, and Marzyeh Ghassemi. Change is hard: A closer look at subpopulation shift. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 39584–39622. PMLR, 2023.

[22] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023.

[23] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for tool-agent-user interaction in real-world domains. In 13th International Conference on Learning Representations, ICLR 2025, 13th International Conference on Learning Representations, ICLR 2025, pages 74824–74876. International Conference on Learning Representations, ICLR, 2025.

[24] Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning, acting, and planning in language models. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview. net/forum?id=njwv9BsGHF.

## A Related work and industry context

Agent benchmarks now test tool use, persistent state, web work, and general task completion. Examples include τ-bench, ToolSandbox, WorkArena, AgentBench, GAIA, and TaskBench [5, 9, 10, 12, 16, 23]. The benchmarks make workflow evaluation concrete, but they do not provide after discovery confidence bounds for a task mixture selected after workflow search.

Several methods discover weak groups or protect group comparisons. GEORGE, Domino, Spotlight, and SubpopBench study hidden groups, systematic errors, and subpopulation shift [4, 7, 18, 21]. Fairness auditing gives simultaneous inference over rich group families [2], while Janus uses decoys and held out replication for a frozen descriptor list and a fixed model [14]. Distributionally robust optimization trains models for worst group or nearby distribution performance [6, 15]. SASST instead audits a frozen adaptive report. It discovers a common task reweighting, recomputes the complete paired report on independent clusters, and allows non-confirmation or abstention.

KMS and MMD provide two choices of geometry for comparing feature distributions [8, 19]. The geometry limits the task mixtures considered by discovery. Coverage comes from the frozen rule cluster inference, which follows finite dimensional multiplier bootstrap arguments [3]. Industry guidance also treats agent evaluation as a release and governance process. Anthropic discusses the difficulty of evaluating agents that use tools and change state, Microsoft provides repeated evaluation for business agents, and NIST is developing agent security evaluations and standards [1, 13].

## B Proofs and technical details

## B.1 Normalized weighted-ratio expansion

Lemma 1 (Weighted-ratio expansion). Let $a _ { g } = P s _ { g } > 0 , P ^ { g } h = P ( s _ { g } h ) / a _ { g } ,$ , and $P _ { n } ^ { g } h \ =$ $P _ { n } ( s _ { g } h ) / P _ { n } s _ { g }$ . Whenever $P _ { n } s _ { g } > 0$

$$
P _ { n } ^ { g } h - P ^ { g } h = \frac { a _ { g } } { P _ { n } s _ { g } } ( P _ { n } - P ) \left\{ \omega _ { g } ( h - P ^ { g } h ) \right\} , \qquad \omega _ { g } = s _ { g } / a _ { g } .\tag{3}
$$

If ${ \cal P } \{ \omega _ { g } ^ { 2 } ( h - { P ^ { g } h } ) ^ { 2 } \} <$ ∞ and $P _ { n } s _ { g }  _ { p } a _ { g } ,$ then

$$
\sqrt { n } ( P _ { n } ^ { g } h - P ^ { g } h ) = \frac { 1 } { \sqrt { n } } \sum _ { i = 1 } ^ { n } \omega _ { g } ( W _ { i } ) \{ h ( W _ { i } ) - P ^ { g } h \} + o _ { p } ( 1 ) .\tag{4}
$$

The same statement holds coordinatewisefor afixed-dimensional vector h.

Proof. Put the empirical and population tilted means over the common denominator $P _ { n } s _ { g } \mathbf { . }$

$$
P _ { n } ^ { g } h - P ^ { g } h = \frac { P _ { n } \{ s _ { g } ( h - P ^ { g } h ) \} } { P _ { n } s _ { g } } = \frac { a _ { g } } { P _ { n } s _ { g } } P _ { n } \{ \omega _ { g } ( h - P ^ { g } h ) \} .
$$

Because ${ \cal P } \{ \omega _ { g } ( h - { P ^ { g } h } ) \} = 0$ , this is exactly (3). The centered empirical average is $O _ { p } ( n ^ { - 1 / 2 } )$ under the stated second-moment condition, while $a _ { g } / ( P _ { n } s _ { g } ) = 1 + o _ { p } ( 1 )$ . Slutsky’s theorem gives (4). The vector statement follows coordinatewise because the dimension is fixed. □

## B.2 Cluster linearization and plug-in contributions

Lemma 2 (Weighted-report cluster linearization). Conditional on $\mathcal { F } _ { \mathcal { D } ; }$ , suppose the conditions of Theorem 1 hold. Let $\mu _ { m } = \operatorname { \mathbb { E } } ( H _ { 1 , m } \mid { \mathcal { F } } _ { \mathcal { D } } )$ and $\Phi _ { k , m } = \nabla F _ { m } ( \mu _ { m } ) ^ { \top } ( H _ { k , m } - \mu _ { m } )$ . Uniformly over the finite family $\mathcal { I } _ { 0 }$

$$
\sqrt { N } ( \widehat { \Delta } _ { m } - \Delta _ { m } ) = N ^ { - 1 / 2 } \sum _ { k = 1 } ^ { N } \Phi _ { k , m } + o _ { p } ( 1 ) ,\tag{5}
$$

and the canonical plug-in contributions satisfy

$$
\operatorname* { m a x } _ { m \in \mathcal { T } _ { 0 } } N ^ { - 1 } \sum _ { k = 1 } ^ { N } ( \widehat { \Phi } _ { k , m } - \Phi _ { k , m } ) ^ { 2 } \overset { p } {  } 0 .\tag{6}
$$

Proof. Every population weighted scoring and held out denominator is at least a fixed $b _ { 0 } > 0 .$ . The cluster law of large numbers and finiteness of $\mathcal { I } _ { 0 }$ imply that, with probability tending to one, all empirical denominators are at least $b _ { 0 } / 2$ . Hence every $\bar { F _ { m } }$ is evaluated inside a neighborhood where its gradient is bounded and uniformly continuous.

Finite second moments and the cluster Lindeberg condition give $\widehat { \mu } _ { m } - \mu _ { m } = O _ { p } ( N ^ { - 1 / 2 } )$ . Taylor’s theorem with integral remainder yields

$$
\begin{array} { r l } & { F _ { m } ( \widehat { \mu } _ { m } ) - F _ { m } ( \mu _ { m } ) = \nabla F _ { m } ( \mu _ { m } ) ^ { \top } ( \widehat { \mu } _ { m } - \mu _ { m } ) + R _ { N , m } , } \\ & { \qquad | R _ { N , m } | \leq \| \widehat { \mu } _ { m } - \mu _ { m } \| \underset { 0 \leq t \leq 1 } { \operatorname* { s u p } } \| \nabla F _ { m } ( \mu _ { m } + t ( \widehat { \mu } _ { m } - \mu _ { m } ) ) - \nabla F _ { m } ( \mu _ { m } ) \| . } \end{array}
$$

The first factor is $O _ { p } ( N ^ { - 1 / 2 } )$ and the second is $o _ { p } ( 1 )$ , uniformly over the finite family. The Taylor expansion proves (5). The coordinates of $H _ { k , m }$ include the weighted scoring and held out numerators and denominators, along with the selector and aggregation inputs. Differentiating $q ( S ) ^ { \intercal } T$ gives

$$
d \{ q ( S ) ^ { \top } T \} = q ( S ) ^ { \top } d T + \{ D q ( S ) d S \} ^ { \top } T ,
$$

so $\Phi _ { k , m }$ contains both held out evaluation and splitwise selection effects. Held out evaluation and splitwise selection are the two channels through which an evaluation unit changes an adaptive report. For the plug-in result, write

$$
\begin{array} { c } { \widehat { \Phi } _ { k , m } - \Phi _ { k , m } = \xi \nabla F _ { m } ( \widehat { \mu } _ { m } ) - \nabla F _ { m } ( \mu _ { m } ) \} ^ { \top } ( H _ { k , m } - \mu _ { m } ) } \\ { - \nabla F _ { m } ( \widehat { \mu } _ { m } ) ^ { \top } ( \widehat { \mu } _ { m } - \mu _ { m } ) . } \end{array}
$$

The average squared first term is $o _ { p } ( 1 )$ by gradient continuity, Cauchy Schwarz, and bounded empirical second moments. The second term is common across k and is $o _ { p } ( 1 )$ because the gradient is bounded and $\widehat { \mu } _ { m } - \mu _ { m } = o _ { p } ( 1 )$ . Taking the maximum over the finite family proves (6). □

## B.3 Proof of Theorem 1

Proof. Condition throughout on $\mathcal { F } _ { \mathcal { D } }$ . The discovered rules and their random targets are then fixed. Let

$$
\Phi _ { k } = ( \Phi _ { k , m } : m \in \mathcal { I } _ { 0 } ) ^ { \top } , \qquad S _ { N } = N ^ { - 1 / 2 } \sum _ { k = 1 } ^ { N } \Phi _ { k } .
$$

By Lemma 2, the centered estimator vector equals $S _ { N } + o _ { p } ( 1 )$ . For every fixed $t \in \mathbb { R } ^ { | \mathcal { I } _ { 0 } | }$ , the scalar triangular array $N ^ { - 1 / 2 } \sum _ { k } t ^ { \top } \Phi _ { k }$ satisfies the Lindeberg Feller conditions and has variance converging to $t ^ { \top } \Sigma t$ . The scalar CLT and Cramér–Wold device therefore give

$$
S _ { N } \stackrel { d } { \to } N _ { | \mathcal { T } _ { 0 } | } ( 0 , \Sigma ) .
$$

Slutsky’s theorem proves the asserted joint limit for the estimator.

It remains to establish the bootstrap approximation. Let

$$
\widehat { \boldsymbol { \Sigma } } = { \cal { N } } ^ { - 1 } \sum _ { k = 1 } ^ { N } ( \widehat { \boldsymbol { \Phi } } _ { k } - \overline { { \widehat { \boldsymbol { \Phi } } } } ) ( \widehat { \boldsymbol { \Phi } } _ { k } - \overline { { \widehat { \boldsymbol { \Phi } } } } ) ^ { \top } .
$$

Let $D _ { k } = \widehat \Phi _ { k } - \Phi _ { k }$ . Lemma 2 gives $\begin{array} { r } { N ^ { - 1 } \sum _ { k } \| D _ { k } \| ^ { 2 } = o _ { p } ( 1 ) } \end{array}$ . Cauchy Schwarz then shows that every cross term $N ^ { - 1 } \sum _ { k } D _ { k } \Phi _ { k } ^ { \top }$ is $o _ { p } ( 1 )$ , and the $D _ { k } D _ { k } ^ { \top }$ terms are also $o _ { p } ( 1 )$ . The cluster law of large numbers and covariance assumption imply that the sample covariance of the true $\Phi _ { k }$ converges to Σ. Hence

$$
\widehat { \Sigma } \ { \overset { p } { \Longrightarrow } } \ \Sigma .
$$

Conditional on the inference data, the multiplier vector

$$
G ^ { * } = N ^ { - 1 / 2 } \sum _ { k = 1 } ^ { N } \zeta _ { k } ( \widehat { \Phi } _ { k } - \overline { { \widehat { \Phi } } } ) , \qquad \zeta _ { k } \overset { \mathrm { i i d } } { \sim } N ( 0 , 1 ) ,
$$

is exactly Gaussian with mean zero and covariance $\widehat { \Sigma }$ . Continuity of centered Gaussian laws in their covariance matrices therefore yields conditional weak convergence to $N ( 0 , \Sigma )$ in bounded Lipschitz distance. Coordinatewise bootstrap quantiles give pointwise coverage.

For signed simultaneous claims, fix $\eta _ { m } \in \{ - 1 , + 1 \}$ and let $\sigma _ { m } ^ { 2 } = \Sigma _ { m m }$ . On the reported nondegenerate coordinates assume min $\iota _ { m } \sigma _ { m } ^ { 2 } > 0$ . The map

$$
x \longmapsto \operatorname* { m a x } _ { m \in \mathcal { T } _ { 0 } } \frac { \eta _ { m } x _ { m } } { \sigma _ { m } }
$$

is continuous, and its finite dimensional nondegenerate Gaussian distribution is continuous. Conditional bootstrap quantiles of the corresponding studentized maximum are therefore consistent. The resulting lower bands $B _ { m }$ for $\eta _ { m } \Delta _ { m }$ satisfy

$$
\begin{array} { r } { { \mathbb P } ( B _ { m } \leq \eta _ { m } \Delta _ { m } \mathrm { ~ f o r ~ a l l ~ } m \in \mathcal { I } _ { 0 } \mid \mathcal { F } _ { \mathcal { D } } ) \geq 1 - \alpha - o _ { p } ( 1 ) . } \end{array}
$$

On this event the inequality holds for every coordinate, so it also holds for any coordinate selected after inspecting the estimates and bands. The simultaneous event proves the pointwise, simultaneous, and selected report conclusions. □

## C Expanded operational algorithm

The following details make Algorithm 1 fully executable. The algorithm is common to static LLM evaluations and agentic workflows; only the evaluation-unit representation, cluster definition, score tensor, and allowed feature map change.

## C.1 Frozen inputs and reporting slots

The inputs are: independent clusters $C _ { 1 } , \dots , C _ { N } ;$ ; the completed reporting layer A; a score blind feature map $\phi ;$ and a finite family $\mathcal { I } _ { 0 } . \mathrm { ~ \textbf ~ { ~ A ~ } ~ }$ slot $m \in \mathcal { I } _ { 0 }$ fixes the contrast $c _ { m } .$ , one sided sign $\eta _ { m } \in \{ - 1 , + 1 \}$ , feature/kernel protocol, candidate family $\mathcal { G } _ { m }$ , KMS radius $\rho _ { m }$ , density cap $\kappa _ { m }$ minimum effective sample size $n _ { \mathrm { m i n } , m }$ , denominator thresholds, stability threshold, optimization tolerance, and deterministic tie rule. A candidate must be a reusable object g that maps any allowed feature vector to a nonnegative score $s _ { g } ( W )$ ; a transductive vector of discovery weights is not an admissible output.

Split whole clusters into D and I. All encoders, standardization statistics, random Fourier features, KMS witness directions, dictionaries, split schedules, and baseline systems are frozen according to the slot before confirmatory outcomes are read.

## C.2 Discovery on D

For each contrast needed by a slot, run the unweighted fixed reporting layer on $\mathcal { D }$ and obtain $\widehat { \Delta } _ { \mathcal { D } , \mathfrak { c } }$ and first-order contributions $\widehat { \psi } _ { c , i }$ . For every $g \in \mathcal { G } _ { m }$ , compute

$$
w _ { i } ( g ; \mathcal { D } ) = \frac { s _ { g } ( W _ { i } ) } { \sum _ { \ell \in \mathcal { D } } s _ { g } ( W _ { \ell } ) } , ~ \widehat { \mathrm { E S S } } _ { g } = \left\{ \sum _ { i \in \mathcal { D } } w _ { i } ( g ; \mathcal { D } ) ^ { 2 } \right\} ^ { - 1 } , ~ \widehat { \kappa } _ { g } = | \mathcal { D } | \operatorname* { m a x } _ { i } w _ { i } ( g ; \mathcal { D } ) .
$$

Let $\widehat { P } _ { \mathcal { D } }$ be uniform on the discovery features and let $\widehat { P } _ { \mathcal { D } } ^ { g }$ use the weights above. With a frozen nonlinear feature map z and finite witness set $\boldsymbol { B } _ { L }$ , the implemented radius is

$$
\widehat { \mathrm { K M S } } _ { 1 , L } ( \widehat { P } _ { \mathcal { D } } , \widehat { P } _ { \mathcal { D } } ^ { g } ) = \operatorname* { m a x } _ { \beta \in \mathcal { B } _ { L } } \mathbf { W } _ { 1 } \Big ( ( \beta ^ { \top } z ) _ { \# } \widehat { P } _ { \mathcal { D } } , ( \beta ^ { \top } z ) _ { \# } \widehat { P } _ { \mathcal { D } } ^ { g } \Big ) .
$$

A candidate is discovery-feasible when its density, ESS, KMS, source/template, and other predeclared score blind constraints pass. Among feasible candidates, select

$$
\widehat { g } _ { m } \in \arg \operatorname* { m i n } _ { g \in \mathcal { G } _ { m } } \left[ \widehat { \Delta } _ { \mathcal { D } , c _ { m } } + \sum _ { i \in \mathcal { D } } \left\{ w _ { i } ( g ; \mathcal { D } ) - | \mathcal { D } | ^ { - 1 } \right\} \widehat { \psi } _ { c _ { m } , i } \right] ,
$$

up to the declared optimization tolerance and fixed tie rule. If no candidate is feasible, record failed discovery. Serialize every fitted threshold, basis, center, bandwidth, preprocessing statistic, and diagnostic needed to apply ${ \widehat { g } } _ { m }$ to new units.

Discovery stability is evaluated by resampling discovery clusters, rerunning the same slot, and comparing the resulting weight functions on a frozen reference feature set. Discrete rules may use exact agreement; continuous rules use a predeclared functional metric such as weight correlation and top-weighted-set overlap. Stability never uses inference outcomes.

## C.3 Frozen-rule confirmation on $\mathcal { T }$

Apply $s _ { \widehat { g } _ { m } }$ to inference features. Recompute the score-blind density ratio, ESS, source/template caps, and every scoring and held-out denominator. If any predeclared inference feasibility check fails, record failed feasibility and stop that slot. If support passes but the discovery description fails its predeclared stability criterion, set an abstention flag and continue so that the frozen-rule estimate and bound can be reported diagnostically.

For every support-feasible slot, let $S _ { r }$ and $E _ { r }$ be the fixed scoring and held-out subsets in repeated split r. For system $j$ and retained artifact a, compute the exact ratios

$$
\widehat { S } _ { r , j , a } ^ { g } = \frac { \sum _ { i \in S _ { r } } s _ { g } ( W _ { i } ) Z _ { j } ( W _ { i } , a ) } { \sum _ { i \in S _ { r } } s _ { g } ( W _ { i } ) } , \qquad \widehat { T } _ { r , j , a } ^ { g } = \frac { \sum _ { i \in E _ { r } } s _ { g } ( W _ { i } ) Z _ { j } ( W _ { i } , a ) } { \sum _ { i \in E _ { r } } s _ { g } ( W _ { i } ) } .
$$

Apply the frozen selector and aggregation without modification:

$$
\widehat { q } _ { r , j } ^ { g } = \pi _ { j } ( \widehat { S } _ { r , j } ^ { g } ) , \qquad \widehat { Y } _ { r , j } ^ { g } = ( \widehat { q } _ { r , j } ^ { g } ) ^ { \top } \widehat { T } _ { r , j } ^ { g } , \qquad \widehat { \theta } _ { j } ^ { g } = \sum _ { r } \lambda _ { r } \widehat { Y } _ { r , j } ^ { g } , \qquad \widehat { \Delta } _ { m } = c _ { m } ^ { \top } \widehat { \theta } ^ { g } .
$$

Both sides of a paired comparison are included in this same contrast. Construct the cluster tensor $H _ { k , m }$ from all weighted numerators and denominators, selector inputs, aggregation coordinates, and systems in the contrast, and compute

$$
\widehat { \Phi } _ { k , m } = \nabla F _ { m } ( \widehat { \mu } _ { m } ) ^ { \top } ( H _ { k , m } - \widehat { \mu } _ { m } ) .
$$

The exact derivative contains both the held out evaluation term and the splitwise selection term displayed in Lemma 2.

## C.4 Inference rule and small cluster safeguard

The Gaussian multiplier maximum is the default asymptotic procedure. The inference rule must be frozen before confirmation outcomes are read. When a score-blind pilot or an independent simulation places the study in a small-cluster regime that has not passed the multiplier calibration check, the protocol may instead use an independently audited conservative family. Our audited safeguard is the one-sided Bonferroni t band

$$
B _ { m } ^ { \mathrm { B o n f } } = \eta _ { m } \widehat { \Delta } _ { m } - t _ { 1 - \alpha / | \mathcal { I } _ { 0 } | , N - 1 } \frac { \widehat { \sigma } _ { m } } { \sqrt { N } } .
$$

Theorem 1 does not imply this finite-sample safeguard. It is an operating rule supported by the small-cluster simulations in Appendix D. The Gaussian analysis was the original Study F analysis; Bonferroni t was a later conservative sensitivity, and both give the same status for every Study F slot. Future small-cluster studies should freeze the safeguarded rule before confirmation outcomes are read.

## C.5 Simultaneous reporting and statuses

Use one multiplier $\zeta _ { k }$ per inference cluster and share it across all slots. For predeclared signs $\eta _ { m }$ apply the multiplier maximum to the signed contributions $\eta _ { m } \hat { \Phi } _ { k , m } .$ A fragility claim uses $\eta _ { m } = - 1$ so a lower band for $- \Delta _ { m }$ is an upper bound for $\Delta _ { m }$ . Report the complete attempted family, including nonreported slots:

<table><tr><td>Status</td><td>Meaning</td></tr><tr><td>confirmed</td><td>Feasibility and stability pass, and the predeclared signed band supports the stated claim.</td></tr><tr><td>not confirmed</td><td>Feasibility and stability pass, but the signed band does not support the predeclared effect threshold.</td></tr><tr><td>failed discovery failed feasibility</td><td>No candidate in the slot satisfies the discovery constraints.</td></tr><tr><td></td><td>The frozen rule lacks inference support, violates a cap, or has an unstable scoring or held out denominator.</td></tr><tr><td>abstained</td><td>The frozen rule effect may be estimable, but its semantic or workflow description fails the declared replicability or auditability criterion.</td></tr></table>

The output row contains the frozen rule and description, ordinary and stressed contrasts, signed bound, ESS, maximum density ratio, minimum split denominator, stability, geometry diagnostics, runtime, and audit/control recommendation. Any external retrieval, generation, human curation, or mitigation experiment is a separate generalization study and is not part of the benchmark relative coverage certificate.

## D Corrected experiments and audits

Artifact policy. Only corrected and versioned artifacts from the independent code and artifact audit support the results below. Superseded first pass analyses are excluded from every paper table and claim. The original full paper plan reserved Study E for a larger static evaluation. The workshop paper does not claim a completed Study E.

## D.1 Study A and A4: calibration after discovery

Study A uses eight Gaussian score blind features, five artifacts with different stress responses, a fixed paired baseline, nineteen bounded candidate rules, and a 50/50 discovery and confirmation split. Each configuration uses 2,000 trials and 2,000 multiplier draws.

Table 4: Study A pointwise coverage in percent. Fixed means a fixed oracle rule, and learned means a rule chosen on discovery data.
<table><tr><td>M</td><td>Fixed, R = 5</td><td>Learned,  $R = 5$ </td><td>Fixed, R = 20</td><td>Learned,  $R = 2 0$ </td></tr><tr><td>500</td><td>95.00</td><td>93.85</td><td>94.60</td><td>94.05</td></tr><tr><td>1000</td><td>94.75</td><td>95.10</td><td>95.10</td><td>94.75</td></tr><tr><td>2000</td><td>95.10</td><td>95.30</td><td>94.95</td><td>95.05</td></tr><tr><td>4000</td><td>95.30</td><td>95.15</td><td>95.75</td><td>95.10</td></tr></table>

Table 5: Matched same data comparison and Study A4 signed family results.
<table><tr><td>Setting</td><td>Split procedure</td><td>Same data procedure</td><td>Protected family result</td></tr><tr><td>M = 500</td><td>94.4% coverage</td><td>91.3% coverage</td><td>not run</td></tr><tr><td>M = 1000</td><td>93.6% coverage</td><td>91.6% coverage</td><td>94.95% familywise, 0.85% false fragility</td></tr><tr><td> $M = 2 0 0 0$ </td><td>94.4% coverage</td><td>93.3% coverage</td><td>95.50% familywise, 0.55% false fragility</td></tr></table>

At $M = 1 0 0 0 .$ the paired split minus same data coverage gap is 2.0 percentage points, with one sided lower bound 0.31 percentage points and McNemar $p = 0 . 0 3 2$ . Fixed rule coverage ranges from 94.60% to 95.75%. For $M \geq 1 0 0 0$ , learned rule coverage ranges from 94.75% to 95.30%, with negligible bias and estimated to empirical standard error ratios from 0.991 to 1.018.

## D.2 Small cluster audit and safeguard

Study F has only $N = 4 0$ inference clusters, placing it in a regime where the Gaussian multiplier bootstrap may undercover. This subsection quantifies the gap and identifies a conservative alternative. The corrected small sample audit uses the exact Study F estimator and the complete ten slot family.

Table 6: Small cluster familywise coverage across four audited null settings (FSV-remedy, post-hoc fresh-seed diagnostic). Values are ranges over the settings.
<table><tr><td>Inference rule</td><td>Familywise coverage</td><td>Reading</td></tr><tr><td>Gaussian multiplier maximum</td><td>90.55 to 93.50%</td><td>below the declared threshold</td></tr><tr><td>HC1 variance inflation</td><td>91.95 to 93.85%</td><td>not adequate</td></tr><tr><td>Common t scaling</td><td>93.75 to 95.20%</td><td>not uniformly adequate</td></tr><tr><td>Bonferroni one sided t</td><td>96.40 to 97.55%</td><td>conservative in all four settings</td></tr></table>

Table 7: Power for a negative ten point stressed effect under density-feasible designs (FSV-remedy, fresh seeds).
<table><tr><td>Clusters</td><td>Stressed ESS</td><td>Gaussian power</td><td>Bonferroni t power</td><td>Reading</td></tr><tr><td>40</td><td>20</td><td>10.95%</td><td>3.90%</td><td>low support</td></tr><tr><td>80</td><td>20</td><td>9.70%</td><td>4.35%</td><td>more clusters do not help at fixed ESS</td></tr><tr><td>200</td><td>40</td><td>18.90%</td><td>14.45%</td><td>moderate support</td></tr><tr><td>200</td><td>100</td><td>48.10%</td><td>40.25%</td><td>support improves power</td></tr></table>

Gaussian familywise coverage falls 1.5–4.5 points below the 95% target at N = 40. The nominal 200-cluster/ESS-20 design is excluded because its maximum density ratio is 10.26, exceeding κ = 5. The theorem is asymptotic, while the Bonferroni rule is a tested operating safeguard rather than a theorem consequence.

## D.3 Corrected positive control

The small cluster audit shows the method is conservative; this subsection shows it is not vacuous by demonstrating that SASST can confirm a real fragility when one exists at sufficient scale. The positive control is a same-seed diagnostic (not fresh validation) using frozen τ-bench features, the Study F estimator, independent confirmation clusters, a planted fragility in the claim direction, and Bonferroni t inference. The conditional confirmation rate is 0.701 with MC95 interval [0.671, 0.730], barely clearing its 0.700 gate.

Table 8: Post-audit same-seed positive-control diagnostic.
<table><tr><td>Gate</td><td>Value</td><td>Required value</td><td>Result</td></tr><tr><td>Null familywise error, N = 40</td><td>0.017</td><td>at most 0.075</td><td>pass</td></tr><tr><td>Null familywise error, N = 200</td><td>0.018</td><td>at most 0.075</td><td>pass</td></tr><tr><td>Target rule discovery, N = 200</td><td>0.921</td><td>at least 0.600</td><td>pass</td></tr><tr><td>Conditional confirmation, N = 200</td><td>0.701</td><td>at least 0.700</td><td>pass</td></tr><tr><td>End to end detection, N = 200</td><td>0.646</td><td>at least 0.500</td><td>pass</td></tr></table>

## D.4 Study B and the geometry audit

Study B compares KMS, MMD, linear max-sliced, no radius, and a restricted linear registry. A corrected Wasserstein routine changes 2.125% of KMS selections and 0.625% of max-sliced selections, but it does not change the qualitative result.

Table 9: Corrected Study B held-out AUC after KMS replay (mean over 200 trials). Bold marks the best method per mechanism.
<table><tr><td>Mechanism</td><td>KMS</td><td>MMD</td><td>Max-sliced</td><td>No-radius</td><td>Linear-only</td></tr><tr><td>Linear</td><td>0.602</td><td>0.631</td><td>0.623</td><td>0.597</td><td>0.594</td></tr><tr><td>XOR</td><td>0.639</td><td>0.747</td><td>0.610</td><td>0.649</td><td>0.631</td></tr><tr><td>Circular</td><td>0.695</td><td>0.716</td><td>0.696</td><td>0.558</td><td>0.500</td></tr><tr><td>Policy by tool</td><td>0.698</td><td>0.679</td><td>0.707</td><td>0.700</td><td>0.576</td></tr></table>

MMD is strongest on three of four mechanisms; max-sliced wins on policy by tool. No geometry wins in every mechanism. Study B supports nonlinear rule recovery for some mechanisms; it does not show that KMS is universally best or provide a confirmed natural fragility result.

A separate geometry audit uses frozen τ -bench features and a planted short goal mechanism.

Table 10: Geometry audit under one frozen absolute radius budget (fixed registry axis, mean over six kernel seeds).
<table><tr><td>Geometry</td><td>Accepted rules</td><td>Correct rule recovery</td><td>AUC</td></tr><tr><td>Standardized features</td><td>53.0</td><td>0.000</td><td>0.579</td></tr><tr><td>Normalized plus structured features</td><td>50.7</td><td>0.000</td><td>0.526</td></tr><tr><td>PCA plus structured features</td><td>50.7</td><td>0.000</td><td>0.522</td></tr><tr><td>Structured features only</td><td>52.5</td><td>0.000</td><td>0.527</td></tr><tr><td>All geometries without radius constraint</td><td>71</td><td>0.610</td><td>0.814</td></tr></table>

The target rule is outside the common 75% radius budget for every geometry. Median unit bandwidth RBF similarity in the standardized high dimensional space is about $4 \times 1 0 ^ { - 1 7 7 }$ . The radius and bandwidth must therefore be checked on score blind pilot data.

## D.5 Study C: feasibility and abstention

Study C verifies that the support gates (density, ESS, split denominators, stability) correctly block infeasible or unstable rules across five synthetic settings.

Table 11: Study C operating checks over 500 trials per setting.
<table><tr><td>Setting</td><td>Abstention or failure</td><td>Nonabstention</td><td>Recorded check</td></tr><tr><td>Severe concentration</td><td>100.0%</td><td>0.0%</td><td>ESS and density</td></tr><tr><td>Density only</td><td>100.0%</td><td>0.0%</td><td>density, while ESS passes</td></tr><tr><td>Split support only</td><td>100.0%</td><td>0.0%</td><td>split support, while global checks pass</td></tr><tr><td>Near tied rules</td><td>95.8%</td><td>4.2%</td><td>stability</td></tr><tr><td>Stable positive rule</td><td>0.0%</td><td>100.0%</td><td>none</td></tr></table>

The 4.2% near tie nonabstention rate is not a false fragility rate. Study A4 measures false fragility separately.

## D.6 Study D: partly synthetic real feature study

Study D uses 1,001 MMLU-Pro Law items, Qwen3-8B score tensors, 384 dimensional text features, and an injected degradation that flips 70% of correct answers within a 40-item jurisprudence subgroup.

Table 12: Study D result and explanation diagnostics.
<table><tr><td>Quantity</td><td>Value</td></tr><tr><td>Weighted confirmation contrast</td><td>-0.0554</td></tr><tr><td>Pointwise 95% interval</td><td>[-0.0886, -0.0221]</td></tr><tr><td>Unweighted confirmation contrast</td><td>-0.0294</td></tr><tr><td>Stress amplification</td><td>-0.0260</td></tr><tr><td>Recovery AUC</td><td>0.974</td></tr><tr><td>Top weighted precision and Jaccard</td><td>0.444</td></tr><tr><td>Top weighted recall</td><td>1.000</td></tr><tr><td>Inference ESS</td><td>405.7</td></tr><tr><td>Functional stability, 100 resamples</td><td>0.18</td></tr><tr><td>Formal status</td><td>abstained</td></tr></table>

The corrected radius is 0.003052. The frozen-rule effect is detectable, but functional stability is 0.18, so the formal status remains abstained and the description is not reusable. The high AUC is a ranking result and does not imply exact subgroup recovery.

## D.7 Controller mechanical audit

Before interpreting Study F and G workflow differences, the routing code must be verified: does ReAct pass everything through, does the verifier block on rejection, and does the confirmation gate respect consent tokens? A fixed deck sends sixteen state changing calls to the three workflows. Oracle verifier signals isolate controller mechanics from model judgment quality.

Table 13: Controller replay. The test measures mechanics rather than semantic verifier quality.
<table><tr><td>Workflow</td><td>Invalid block recall</td><td>Valid approval</td><td>Confirmed executed</td><td>Declined blocked</td></tr><tr><td>ReAct pass through</td><td>0.0%</td><td>100.0%</td><td>100.0%</td><td>0.0%</td></tr><tr><td>Planner with oracle verifier</td><td>100.0%</td><td>100.0%</td><td>50.0%</td><td>50.0%</td></tr><tr><td>Confirmation gate</td><td>50.0%</td><td>50.0%</td><td>100.0%</td><td>100.0%</td></tr></table>

All twenty declared mechanical gates and ten boundary and regression tests pass. The prompt contains the exact serialized mutation and full arguments, and malformed verifier output fails closed. This audit applies to the revised f2\_v2 controller; Studies F and G used the historical study\_f\_v1 controller, whose routing mechanics are consistent but whose confirmation prompt disclosed only the tool name. The audit does not validate informed consent or model based semantic verification.

## D.8 Studies F and G: interactive agent evaluations

Study F uses Qwen3-8B. Study G uses Qwen2.5-7B-Instruct. Both use the same eighty task IDs, two simulator seeds, three workflows, 40/40 task cluster split, frozen features, three radii, and ten reporting slots. Each study contains 480 episodes.

Table 14: Across-model agent summary. Study F full-study rates combine the exact 80-observation discovery and confirmation halves; Study G rates are reported over its full 160 observations per workflow.
<table><tr><td>Quantity</td><td>Study F, Qwen3-8B</td><td>Study G, Qwen2.5-7B-Instruct</td></tr><tr><td>Full-study ReAct success</td><td>18.75%</td><td>11.9%</td></tr><tr><td>Full-study Plan + verifier success</td><td>20.625%</td><td>11.2%</td></tr><tr><td>Full-study confirmation-gate success</td><td>19.375%</td><td>14.4%</td></tr><tr><td>Discovery choice</td><td>Plan + verifier</td><td>Confirmation gate</td></tr><tr><td>Plan + verifier minus ReAct on confirmation</td><td>0.000</td><td>+0.0125</td></tr><tr><td>Confirmation gate minus ReAct on confirmation</td><td>-0.0250</td><td>+0.0250</td></tr><tr><td>Maximum rule stability</td><td>0.37</td><td>0.27</td></tr><tr><td>Confirmed</td><td>0</td><td>0</td></tr><tr><td>Not-confirmed</td><td>2</td><td>2</td></tr><tr><td>Abstained</td><td>8</td><td>6</td></tr><tr><td>Failed-feasibility</td><td>0</td><td>2</td></tr></table>

Table 15: Study F simultaneous ten-slot report under the original Gaussian procedure (critical value 2.4632). Bonferroni t (critical 2.7424) widens bands by factor 1.113 and leaves every status unchanged.
<table><tr><td>Slot</td><td>Estimate</td><td>SE</td><td>Simultaneous bound</td><td>ESS</td><td>Density</td><td>Stability</td><td>Status</td></tr><tr><td>Uniform: Plan + verifier — ReAct</td><td>0.0000</td><td>0.0791</td><td>lower -0.195</td><td>40.0</td><td>1.00</td><td>1.00</td><td>not confirmed</td></tr><tr><td>Uniform: confirmation gate — ReAct</td><td>-0.0250</td><td>0.0498</td><td>lower -0.148</td><td>40.0</td><td>1.00</td><td>1.00</td><td>not confirmed</td></tr><tr><td>r1 = 0.01017: Plan + verifier</td><td>-0.0227</td><td>0.0782</td><td>upper 0.170</td><td>34.6</td><td>2.73</td><td>0.05</td><td>abstained</td></tr><tr><td>r1 = 0.01017: confirmation gate</td><td>-0.0227</td><td>0.0454</td><td>upper 0.089</td><td>34.6</td><td>2.73</td><td>0.05</td><td>abstained</td></tr><tr><td>r2 = 0.01799: Plan + verifier</td><td>+0.0400</td><td>0.0836</td><td>upper 0.246</td><td>31.3</td><td>2.40</td><td>0.07</td><td>abstained</td></tr><tr><td>r2 = 0.01799: confirmation gate</td><td>0.0000</td><td>0.0490</td><td>upper 0.121</td><td>31.3</td><td>2.40</td><td>0.07</td><td>abstained</td></tr><tr><td>r3 = 0.04003: Plan + verifier</td><td>+0.1298</td><td>0.1398</td><td>upper 0.474</td><td>20.6</td><td>2.40</td><td>0.37</td><td>abstained</td></tr><tr><td>r3 = 0.04003: confirmation gate</td><td>+0.0178</td><td>0.0801</td><td>upper 0.215</td><td>20.6</td><td>2.40</td><td>0.37</td><td>abstained</td></tr><tr><td>Control r2: Plan — gate</td><td>+0.0400</td><td>0.0574</td><td>lower -0.101</td><td>31.3</td><td>2.40</td><td>0.07</td><td>abstained</td></tr><tr><td>Control r2: gate — Plan</td><td>-0.0400</td><td>0.0574</td><td>lower -0.181</td><td>31.3</td><td>2.40</td><td>0.07</td><td>abstained</td></tr></table>

Study F’s maximum density ratio is 2.73, well within κ = 5. Study G’s third radius fails feasibility because ESS = 18.7 falls below the predeclared minimum of 20.

Table 16: Study G ten-slot report under the preregistered Gaussian simultaneous procedure (critical value 2.347). The Gaussian band is not well calibrated at N = 40 (Section D); Bonferroni t widens all intervals without changing any status. NA means unavailable after feasibility failure.
<table><tr><td>Slot</td><td>Estimate</td><td>SE</td><td>Simultaneous bound</td><td>ESS</td><td>Density</td><td>Stability</td><td>Status</td></tr><tr><td>Uniform: Plan + verifier — ReAct</td><td>+0.0125</td><td>0.0414</td><td>lower -0.085</td><td>40.0</td><td>1.00</td><td>1.00</td><td>not confirmed</td></tr><tr><td>Uniform: confirmation gate — ReAct</td><td>+0.0250</td><td>0.0393</td><td>lower -0.067</td><td>40.0</td><td>1.00</td><td>1.00</td><td>not confirmed</td></tr><tr><td>Stress r1: Plan + verifier</td><td>+0.0125</td><td>0.0414</td><td>upper 0.110</td><td>40.0</td><td>1.00</td><td>0.10</td><td>abstained</td></tr><tr><td>Stress r₁: confirmation gate</td><td>+0.0250</td><td>0.0393</td><td>upper 0.117</td><td>40.0</td><td>1.00</td><td>0.10</td><td>abstained</td></tr><tr><td>Stress r2: Plan + verifier</td><td>+0.0145</td><td>0.0435</td><td>upper 0.117</td><td>34.8</td><td>1.16</td><td>0.27</td><td>abstained</td></tr><tr><td>Stress r2: confirmation gate</td><td>+0.0021</td><td>0.0411</td><td>upper 0.099</td><td>34.8</td><td>1.16</td><td>0.27</td><td>abstained</td></tr><tr><td>Stress r3: Plan + verifier</td><td>+0.0356</td><td>NA</td><td>NA</td><td>18.7</td><td>2.85</td><td>0.14</td><td>failed feasibility</td></tr><tr><td>Stress r3: confirmation gate</td><td>+0.0711</td><td>NA</td><td>NA</td><td>18.7</td><td>2.85</td><td>0.14</td><td>failed feasibility</td></tr><tr><td>Control r2: Plan — gate</td><td>+0.0124</td><td>0.0384</td><td>lower -0.078</td><td>34.8</td><td>1.16</td><td>0.27</td><td>abstained</td></tr><tr><td>Control r2: gate — Plan</td><td>-0.0124</td><td>0.0384</td><td>lower -0.103</td><td>34.8</td><td>1.16</td><td>0.27</td><td>abstained</td></tr></table>

Study G uses the same tasks and is therefore an across model replication of the reporting decision rather than an independent benchmark replication. It selects a different workflow and different stress rules, but it also produces zero confirmed claims.

Table 17: Study F descriptive execution counts. The counts are not causal safety effects.
<table><tr><td>Workflow</td><td>Tokens</td><td>Tool calls</td><td>Failed calls</td><td>Irreversible calls</td></tr><tr><td>ReAct</td><td>19,932,313</td><td>1,091</td><td>217</td><td>269</td></tr><tr><td>Plan + verifier</td><td>17,535,902</td><td>1,058</td><td>157</td><td>176</td></tr><tr><td>Confirmation gate</td><td>19,728,565</td><td>1,021</td><td>200</td><td>233</td></tr></table>

Study F used Qwen3-8B for 480 episodes, totaling 57,196,780 model tokens and 18,280 summed episode seconds. Study G used Qwen2.5-7B-Instruct for 480 episodes and required about 2.5 hours of collection at three concurrent workers plus 30 seconds of inference. The recorded execution environment used an NVIDIA RTX 6000 Ada Generation (49,140 MiB); a separate Study G environment manifest was not preserved.

Study F sensitivity analyses answer different questions. Treating three parser errors as missing leaves the two confirmation contrasts and all statuses unchanged. A stronger balanced deletion lowers stressed ESS for two slots below the threshold, changing those slots from abstained to failed feasibility while keeping zero confirmed claims.

Post-hoc descriptive decomposition. Across all task-seed blocks, 104/160 in Study F and 118/160 in Study G scored zero under all three workflows. These are observed all-workflow-zero blocks, not estimates of a structural floor. In Study G, Plan + verifier reached environment completion in 76/160 episodes versus 118/160 for ReAct. Trace termination codes show that the 84 Plan + verifier non-completions comprise 32 max-step terminations and 52 verifier-rejection terminations, so the completion difference primarily reflects controller-induced stopping. Conditional success among completed episodes was 23.7% for Plan + verifier and 16.1% for ReAct. Because completion is affected by workflow assignment, these completed sets are selected post-treatment populations; the difference should not be interpreted as a causal quality improvement.

## D.9 Reproducibility and superseded analyses

The paper tables use only corrected artifacts. The arithmetic audit verifies the discovery gain, confirmation contrasts, protected bounds, critical values, and status counts. The claim source matrix maps each paper claim to a corrected artifact and records allowed and forbidden wording. An exclusion audit checks that no paper script reads a superseded first pass result.

Table 18: Artifact provenance summary.
<table><tr><td>Analysis</td><td>Provenance</td><td>Seeds</td></tr><tr><td>Study A, A4, B, C, D</td><td>Preregistered simulation</td><td>dedicated</td></tr><tr><td>Study F, G</td><td>Preregistered interactive</td><td>dedicated</td></tr><tr><td>KMS correction</td><td>Corrective overlay (frozen Study F)</td><td>same</td></tr><tr><td>FSV-v2</td><td>Independent calibration audit</td><td>fresh</td></tr><tr><td>FSV-remedy</td><td>Post-hoc fresh-seed diagnostic</td><td>fresh</td></tr><tr><td>FGEO-v2</td><td>Corrective geometry audit</td><td>fresh</td></tr><tr><td>FPOS-v2</td><td>Same-seed diagnostic</td><td>same</td></tr><tr><td>FCTL2</td><td>Mechanical audit (f2_v2 controller)</td><td>deterministic</td></tr></table>

A prospective natural follow up would require fresh tasks, a valid outcome aligned with the policy claim, a predeclared small sample inference rule, corrected feature geometry, a nonredundant intervention, and a power plan based on stressed effective sample size. We leave that experiment to later work.

## E Limitations and broader impact

The theorem is asymptotic and assumes bounded density ratios, stable denominators, honest discovery, and plausibly independent clusters. The forty-cluster audit shows that the generic Gaussian family can undercover in a small sample, while the independently audited Bonferroni t safeguard is conservative in the tested settings. The safeguard has been audited in four null settings rather than proved for every finite sample.

The stress rule search is limited by the declared registry, feature map, radius, and support thresholds. A rule outside the radius is outside the audit scope. High-dimensional kernel features can also become nearly orthogonal when the bandwidth is poorly scaled. The agent studies use one benchmark task pool, low-success open models, the same model as actor and user simulator, and forty confirmation task clusters. The second-model study is a model replication on the same tasks rather than an independent benchmark replication.

The intended use is evaluation and release control. A team can use the protocol to distinguish a supported workflow claim from a development-set gain, an unstable segment, or a low-support analysis. Misuse would include treating a benchmark relative result as proof of deployment safety, treating non-confirmation as equivalence, or selecting a new analysis after seeing confirmation outcomes. The paper rejects those interpretations.