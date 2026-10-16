# CAN CAPABILITY ENHANCEMENT BE LOCALIZED?

Anh Tuan Nguyen, Anna Bair, Zico Kolter

Carnegie Mellon University

{atnguyen,abair,zkolter}@cs.cmu.edu

## ABSTRACT

Transformer-based language models have shown strong capabilities across many tasks and domains. However, prior work shows that, for a given task, not all attention heads matter equally. Specifically, they suggest that many redundant heads can be safely removed with little-to-no effect, while a small set of important heads, when ablated, severely damages the model’s performance on the given task. In this work, we investigate a complementary aspect, called Capability Enhancement Localization (CEL): given a model and a task, does there exist a small set of heads whose removal improves the task performance? We first investigate this question empirically across models from different families. Using a simple two-stage routine that screens candidate heads and evaluates their joint ablations, we identify sparse head combinations whose removal improves task-specific performance without further fine-tuning. Motivated by these observations, we ask theoretically whether such a phenomenon can naturally arise from training dynamics, and under what conditions. To this end, we construct a simplified multihead softmax attention model and two tasks with conflicting features, and show that gradient descent from small random initialization provably concentrates the conflicting features into particular heads. We further show that ablating such heads improves performance on one task while harming the other, even though training converges to a global optimum of the joint objective. This establishes a possible training-induced theoretical explanation for CEL.

## 1 INTRODUCTION

Large language models (LLMs) (Vaswani et al., 2017; Hoffmann et al., 2022) have demonstrated strong capabilities across diverse tasks, including language understanding, text generation, question answering, code generation, and mathematical reasoning (Brown et al., 2020; Raffel et al., 2020; Rae et al., 2021), among others. A notable feature of LLMs is that a single set of parameters supports many different tasks, which can often be specified through natural-language instructions or a few demonstrations without task-specific parameter updates (Brown et al., 2020; Wei et al., 2021). This breadth of functionality motivates a line of work studying how a shared network supports diverse capabilities and how its internal components (e.g., attention heads/layers) contribute to LLM performance on individual tasks.

Early works on LLM pruning demonstrated that the attention heads exhibit substantial redundancy and contribute unequally across different tasks. For example, Clark et al. (2019b) observed that attention heads within the same BERT layers (Devlin et al., 2019) often exhibit similar attention patterns, leading to overlapping behavior. Moreover, Michel et al. (2019) and Voita et al. (2019) showed that many attention heads can be pruned after training without substantially affecting the overall LLM performance. Together, those findings emphasize that the full complement of attention heads is generally not required at inference time for preserving task capabilities, motivating a closer look at which heads are dispensable and which are critical for particular tasks.

Besides the evidence of redundancy, recent works have identified attention heads with specialized roles. Examples include induction mechanism corresponding to in-context learning (Olsson et al., 2022), information retrieval from long contexts (Wu et al., 2025), and processing operands and operators for arithmetic tasks (Zhang et al., 2024), to name a few. Other work identified distinct heads that favor internal knowledge encoded in the model’s weights or external context when these information sources conflict (Jin et al., 2024). These analyses revealed how specific heads support or route computations under various scenarios. More recently, broader capability-localization studies complement these findings. In particular, Huang et al. (2025) identified a small group of neurons associated with shared capabilities, while Bair et al. (2026) leveraged a cheap compressed sensing method (i.e., LASSO (Tibshirani, 1996)) to identify task-critical attention heads corresponding to separate capabilities. Ablating just a few of these task-critical heads can dramatically degrade the LLM performance on the corresponding task.

The above findings provided evidence of redundant and task-critical attention heads. In this work, we complement these findings by asking the following counter-intuitive question: given an arbitrary task and an LLM, is there a sparse set of attention heads such that ablating them improves the performance of that LLM on that task, without any further fine-tuning? For convenience, we term this question Capability Enhancement Localization (CEL).

We note that beneficial head removal has already been observed in some specific scenarios. Held & Yang (2023) identify heads that interfere with performance in multilingual models and show that pruning them can improve performance for a target language. Jin et al. (2024) prune heads involved in conflicts between a model’s internal knowledge and external context. More recently, Nguyen et al. (2025) observe that pruning certain heads can improve mathematical reasoning and use different pruned configurations to generate diverse answers for best-of-N selection. In this work, we ask whether sparse beneficial head ablations can be found across a broader range of tasks and models, without presupposing a particular source of interference. More importantly, we ask how successful training can produce a head whose removal improves a task. We answer this theoretical question in a simplified attention model, where task conflict and learned feature allocation give rise to beneficial head ablation.

Contribution. In this work, we study the Capability Enhancement Localization (CEL) question, which asks whether there exists a sparse set of attention heads whose ablation improves task performance without further fine-tuning, given a task. Our contributions are:

1. In Section 3, we consider multiple model families and task configurations. Using a simple twostage approach, we found evidence of the CEL phenomenon: given a task, ablating a sparse set of heads can significantly improve models’ task-specific performance.

2. Motivated by empirical evidence of CEL, we seek a potential theoretical explanation for whether and how this phenomenon naturally arises in LLM training. To this end, in Section 4, we construct a simplified softmax attention model, trained on two tasks with a feature conflict. We then show in Theorem 4.2 that, under standard GD training, such conflicting features can be encoded into a specific attention head, leading to the localization phenomenon.

3. We then further show in Theorem 4.3 that ablating such a head can improve the model’s performance on a single task while harming the other, even though training converges to an optimum of the joint objective. Together with the localization phenomenon, this theoretically establishes one possible mechanism for CEL.

## 2 RELATED WORK

Attention-head functionality. Studies of Transformer components reveal both redundancy and functional specialization. Many attention heads can be removed with little loss in task performance (Michel et al., 2019; Voita et al., 2019), while others participate in identifiable computations, including induction and information retrieval (Olsson et al., 2022; Wu et al., 2025). Beyond individ ual head functions, localization studies investigate components associated with factual associations, task-specific skills, and capabilities shared across examples (Meng et al., 2022; Panigrahi et al., 2023; Huang et al., 2025). At the attention-head level, Bair et al. (2026) identify sparse sets of task-critical heads whose joint ablation substantially degrades target-task performance while largely preserving performance on unrelated tasks. Together, these findings motivate examining how sparse interventions affect task performance, including whether their effects can be beneficial. In this work, we instead focus on the enhancement regime: identifying small sets of attention heads whose joint removal improves task performance without further fine-tuning. Furthermore, we also seek a theo retical explanation for how this phenomenon can arise in standard LLM training.

Performance enhancement through head pruning. Prior work has found benefits from head removal in several settings. Held & Yang (2023) show that pruning heads associated with interference across languages can improve performance for a target language. Jin et al. (2024) study conflicts between internal knowledge and external context, and show that pruning selected heads can improve question answering. In mathematical reasoning, Nguyen et al. (2025) observe improvements from pruning particular heads and use different pruned configurations to produce diverse answers for best-of-N selection. In this work, we consider a more general setting by examining sparse beneficial interventions across a broader range of tasks and models, and provide a potential theoretical explanation of how this phenomenon can arise in LLM training dynamics, an aspect that was ignored by prior works.

Mechanistic interpretability and circuit discovery. Mechanistic interpretability studies how internal components implement model behavior. Wang et al. (2022) identify an attention-head circuit for indirect object identification, while Conmy et al. (2023) use activation patching to automate the discovery of computational subgraphs. Such localization need not predict the effect of changing a model: Hase et al. (2023) find that causal-tracing results do not identify the best MLP layers for factual editing. In our experiments, head ablation measures the effect of a specified intervention but does not reveal the computation performed by the removed heads. Our theory instead shows, in a simplified model, how training can produce a head whose removal benefits one task.

## 3 EMPIRICAL MOTIVATION: SPARSE BENEFICIAL HEAD ABLATION

In this section, we introduce our notation, define task-specific performance gain under head ablations, and formulate the Capability Enhancement Localization (CEL) problem as discovering small head subsets whose removal improves task-specific performance of the considered model.

## 3.1 HEAD ABLATION AND EVALUATION

Notation. We consider a transformer-based LLM M with L layers and H attention heads per layer, with $\begin{array} { r c l } { N } & { = } & { L H } \end{array}$ heads in total. We denote the head at layer $\ell ~ \in ~ \{ 1 , \ldots , L \}$ and position $\begin{array} { r l r } { i } & { { } \in } & { \{ 1 , \ldots , H \} } \end{array}$ by $h _ { \ell , i } ,$ , and the set of all attention heads by H = $\{ h _ { \ell , i } : \ell \in \{ 1 , \dots , L \} , i \in \{ 1 , \dots , H \} \}$ . Our intervention targets a sparse subset of attention heads $S \subset \mathcal { H }$ , with the number of elements $| { \bar { S } } | \leq k$ for $k \ll N$ . In this work, we typically choose $k = 3$

Head ablation, performance gain, and beneficial head subset. Given a set $S \subsetneq \mathcal { H }$ of attention heads, let $M _ { \backslash S }$ denote the model obtained by setting the output of heads in S to zero at every token position before the attention output projection. We note that all remaining parameters remain unchanged, and no fine-tuning is performed. For a task $T$ with a test dataset $D _ { \mathrm { t e s t } }$ , we denote the performance of the model M under the task $T$ as

$$
\mathrm { A c c } _ { T , D _ { \mathrm { t e s } } } ( M ) = \frac { 1 } { | D _ { \mathrm { t e s t } } | } \sum _ { ( x , y ) \in D _ { \mathrm { t e s t } } } \mathbf { 1 } [ M \mathrm { a n s w e r s ~ } x \mathrm { ~ c o r r e c t l y ~ u n d e r ~ t h e ~ t a s k ~ s c o r e r } ]
$$

Under the same inference and scoring protocol for the intact and ablated models, we denote the empirical ablation gain as $\Delta _ { T , D _ { \mathrm { t e s t } } } ( S ) \dot { = } \mathrm { A c c } _ { T , D _ { \mathrm { t e s t } } } ( M _ { \backslash S } ) - \mathrm { A c c } _ { T , D _ { \mathrm { t e s t } } } ( M )$ . We then call the sparse subset of attention heads S beneficial on the test set $D _ { \mathrm { t e s t } } \operatorname { i f } \Delta _ { T , D _ { \mathrm { t e s t } } } ( S ) > 0$

Objective. Given a model M, a task T with a test set $D _ { \mathrm { t e s t } } .$ , and a head budget $k ,$ we use a disjoint discovery set $D _ { \mathrm { d i s t } } ( \mathrm { i . e . , } D _ { \mathrm { d i s t } } \cap D _ { \mathrm { t e s t } } = \emptyset )$ from the task T to select a head subset through a search routine A:

$$
\widehat { S } = A ( M , T , D _ { \mathrm { d i s t } } , k ) , \qquad 1 \leq | \widehat { S } | \leq k .
$$

The goal of CEL is to identify subsets $\hat { S }$ whose joint ablation yields a large positive test gain $\Delta _ { T , D _ { \mathrm { t e s t } } } ( { \widehat { S } } )$ , while using only discovery data $D _ { \mathrm { d i s t } }$ to guide selection. The selected subset $\hat { S }$ is fixed before test evaluation. By using this setup, our hope is that the identified subset $\hat { S }$ truly improves the task-specific performance on task $T .$ , rather than only overfitting the discovery set $D _ { \mathrm { d i s t } }$

## 3.2 EXPERIMENTAL SETUP

Models and tasks. We evaluate three models of different families: Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, and SmolLM3-3B-Instruct (Grattafiori et al., 2024; Qwen et al., 2025; Bakouch et al., 2025). The number of layers and heads of considered models can be found in Table 2.

Algorithm 1 Two-stage intervention selection and held-out evaluation   
Require: Model M with head set $\overline { { \mathcal { H } ; } }$ task T; discovery set $D _ { \mathrm { d i s t } } ;$ test set $D _ { \mathrm { t e s t } }$   
Require: Candidate count $p = 1 0 ;$ head budget $k = 3$   
Ensure: Selected subset $\widehat { S }$ and its test-set gain   
1: For $D \in \{ D _ { \mathrm { d i s t } } , D _ { \mathrm { t e s t } } \}$ , use $\mathrm { A c c } _ { T , D } ( \tilde { M _ { \backslash } } _ { S } )$ and $\Delta _ { T , D } ( S )$ as defined above   
2: Let $d _ { T , D } ( S )$ count examples in D answered correctly by M but incorrectly by $M _ { \backslash s }$   
3: Rank subsets on $D _ { \mathrm { d i s t } }$ by higher gain, lower damage, higher accuracy, fewer heads, then lexico  
graphic head order   
Stage 1: Screen individual heads   
4: for $h \in \mathcal H$ do   
5: Evaluate $\{ h \}$ on $D _ { \mathrm { d i s t } }$   
6: end for   
7: P ← heads in the top p singleton interventions   
Stage 2: Select a joint intervention   
8: ${ \mathcal { C } }  \{ S \subseteq P : 1 \leq | S | \leq k \}$   
9: for $\bar { S ^ { \prime } } \in \mathcal { C }$ with $| S | \ge 2$ do   
10: Evaluate S on $\dot { D } _ { \mathrm { d i s t } }$ ▷ Reuse singleton results   
11: end for   
12: Sb ← highest-ranked subset in C on $D _ { \mathrm { d i s t } }$   
13: Freeze $\succnapprox$   
Held-out evaluation   
14: Evaluate $\widehat { S }$ on $D _ { \mathrm { t e s t } }$ with a matched intact-model run   
15: return $\widehat { S } , \operatorname { A c c } _ { T , D _ { \mathrm { t e s t } } } ( M _ { \setminus } \widehat { S } ) , \Delta _ { T , D _ { \mathrm { t e s t } } } ( \widehat { S } )$

For the tasks, we consider three capabilities using various datasets: (1) math capabilities with Arithmetic (Brown et al., 2020), (2) code generation with MBPP (Austin et al., 2021), and (3) language skills via HellaSwag (Zellers et al., 2019) and BoolQ (Clark et al., 2019a).

Intervention selection. For each model–task pair, we use a two-stage procedure on the discovery set $D _ { \mathrm { d i s t } }$ . We first score individual heads to form a small candidate pool, then evaluate joint ablations of subsets drawn from that pool and select one subset with the best discovery-set performance. We fix this subset before measuring its gain on the held-out test set $D _ { \mathrm { t e s t } }$ . The test examples play no role in scoring heads or choosing the intervention. Details of the procedure are given in Algorithm 1. For the dataset MBPP, we use the official test set as $D _ { t e s t }$ and subsample 374 examples for the training set as $D _ { \mathrm { d i s t } }$ . For the datasets HellaSwag and BoolQ, we use the official validation datasets as the test set, and subsample 2,000 examples from the official training sets as discovery set $D _ { d i s t } .$ For the dataset Arithmetic, since it does not have a separate test set with official labels, we select from its official validation split of 20,000 examples 2,000 examples for $D _ { \mathrm { d i s t } }$ and another disjoint 1,000 examples as $D _ { \mathrm { t e s t } }$ . For convenience, we summarize the number of examples of discovery and test sets for each task in Table 3.

Hardware configuration. We ran the experiments as Slurm jobs, each allocated one NVIDIA L40S GPU, two CPUs, 24 GB of system memory, and with a batch size of 8.

## 3.3 RESULTS

Table 1 reports the held-out accuracy changes for the model–task pairs studied here. Each gain compares an ablated model with an intact model evaluated under the same task protocol. The reported interventions improve held-out accuracy across Arithmetic, BoolQ, HellaSwag, and MBPP. For example, the SmolLM3 Arithmetic intervention raises accuracy from 70.40% to 83.50%, a gain of 13.10 percentage points. The gains vary across model–task pairs, so the evidence supports beneficial sparse ablations in these tested settings, rather than a claim that ablation will help every model on every task.

Our two-stage search is deliberately simple. It searches small subsets within a pool selected using individual-head effects, so it need not find the best subset among all heads. These results establish the empirical phenomenon but do not explain why training can produce components whose removal improves a particular task. We next study this question in a simplified attention model. The theory provides one possible mechanism for beneficial head ablation; it does not identify the mechanism responsible for the gains in the language models above.

Table 1: Held-out performance of head ablations. SmolLM3 is evaluated in non-thinking mode.
<table><tr><td>Model</td><td>Task</td><td>Intact acc.</td><td>Ablated acc.</td><td>Gain (pp)</td><td>Rel. error reduction</td></tr><tr><td>SmolLM3</td><td>Arithmetic</td><td>70.40%</td><td>83.50%</td><td>+13.10</td><td>44.26%</td></tr><tr><td></td><td>BoolQ</td><td>82.57%</td><td>84.92%</td><td>+2.35</td><td>13.51%</td></tr><tr><td></td><td>HellaSwag</td><td>75.75%</td><td>77.05%</td><td>+1.29</td><td>5.34%</td></tr><tr><td></td><td>MBPP</td><td>50.40%</td><td>52.00%</td><td>+1.60</td><td>3.23%</td></tr><tr><td>Qwen2.5-3B-Instruct</td><td>Arithmetic</td><td>89.50%</td><td>90.90%</td><td>+1.40</td><td>13.33%</td></tr><tr><td></td><td>BoolQ</td><td>80.21</td><td>82.35</td><td>+2.14</td><td>10.82%</td></tr><tr><td></td><td>HellaSwag</td><td>75.02</td><td>76.66</td><td>+1.63</td><td>6.54%</td></tr><tr><td></td><td>MBPP</td><td>60.40%</td><td>62.20</td><td>+1.80</td><td>4.55%</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>Arithmetic</td><td>68.00%</td><td>74.30%</td><td>+6.30</td><td>19.69%</td></tr><tr><td></td><td>BoolQ</td><td>78.72%</td><td>81.25%</td><td>+2.54</td><td>11.93%</td></tr><tr><td></td><td>HellaSwag</td><td>71.64%</td><td>73.78%</td><td>+2.14</td><td>7.55%</td></tr><tr><td></td><td>MBPP</td><td>47.80</td><td>53.00%</td><td>+5.20</td><td>9.96%</td></tr></table>

## 4 A THEORETICAL EXPLANATION OF THE CAPABILITY ENHANCEMENT LOCALIZATION PHENOMENON

As we saw previously, there is strong evidence suggesting the existence of the CEL phenomenon. But why does it happen? In this section, we will investigate this question via a simple model with diagonal weights and small initialization scale, inspired by prior works (Boix-Adsera et al., 2023; Pesme & Flammarion, 2023; Berthier, 2023). We then show that the CEL phenomenon can potentially arise naturally in the training trajectory if there is initially some conflict between tasks.

## 4.1 TASKS, ATTENTION MODEL, AND TRAINING

In this section, we will first describe our simple model within which we study the CEL problem.

Data and task. Let U, V, W be independent Rademacher variables which take the value in {−1, 1} with equal chance. Fix real values $a , b , c > 0$ , and set $\begin{array} { r } { t = \frac { a } { 2 } } \end{array}$ . We define two equally weighted training tasks

$$
A : \quad X = ( U , V , 0 ) , \quad Y = b V , \quad { \mathrm { a n d } } \quad B : \quad X = ( U , 0 , W ) , \quad Y = a U + c W .
$$

In this construction, the important feature is coordinate 0. On task A, the variable U appears in the input, but does not appear in the output, meaning that the ideal predictor should ignore coordinate 0. In contrast, on task B, the same variable U is useful, and should receive coefficient a. Therefore, in this case, the two tasks disagree about the desired use of the same input coordinate (a simple scenario of disagreements between tasks).

The situation with other coordinates is simpler. Coordinate 1, with variable V, carries the useful signal for task A, while coordinate 2 carries the useful W-signal for task B. In this case, for simplicity, we assume that the observations are noiseless, and we study population squared risk rather than finite-sample estimation error (which is a proxy to evaluate the population risk). In particular, this ensures that any beneficial ablation in the analysis comesfrom task conflict and training dynamics, rather thanfrom overfitting to afinite sample.

Attention architecture and input encoding. We next consider a simple one-layer Transformerstyle attention model with three attention heads. In this case, our goal is to understand how training allocates already available features across heads, rather than how features themselves are learned. We use a fixed coordinate encoding as follows. Let $e _ { 0 } , e _ { 1 } , e _ { 2 } \ \in \ \mathbb { R } ^ { 3 }$ denote the standard basis vectors. For an input $x = ( x _ { 0 } , x _ { 1 } , \overset { \cdot } { x } _ { 2 } ) \in \{ - 1 , 0 , 1 \} ^ { 3 }$ , we associate coordinate j with the query $z _ { j } ( x ) = \sqrt { 3 } x _ { j } e _ { j }$ . This means that the original input x will be encoded as the series of tokens before

being fed to the transformer

$$
z ( x ) = \left( \begin{array} { c c c } { { \sqrt { 3 } x _ { 0 } } } & { { 0 } } & { { 0 } } \\ { { 0 } } & { { \sqrt { 3 } x _ { 1 } } } & { { 0 } } \\ { { 0 } } & { { 0 } } & { { \sqrt { 3 } x _ { 2 } . } } \end{array} \right)
$$

Therefore, the three coordinates of x are represented by three fixed feature directions. This is a deliberate modeling choice, as we want that the feature basis is supplied to the model, while how these features are associated to the attention heads is learned during the training process.

Fixed memory bank . For every coordinate $j \in \{ 0 , 1 , 2 \}$ , we introduce a fixed pair of memory tokens $m _ { j , + } = e _ { j } , \quad m _ { j , - } = - e _ { j }$ . The query $z _ { j } ( x )$ attends only to the corresponding pair $( m _ { j , + } , m _ { j , - } )$ , using a fixed attention mask shared by all three heads. This paired construction serves two purposes. First, it gives every head exactly the same access to every feature, so no feature-tohead assignment is deterministically imposed in advance. Second, the symmetric pair $( + e _ { j } , - e _ { j } )$ yields a particular signed softmax response (which we will see below), allowing us to analyze the training dynamics without replacing softmax attention by an artificial surrogate mechanism.

Attention output. Given an input $\boldsymbol { x } = ( x _ { 0 } , x _ { 1 } , x _ { 2 } )$ , we form three queries $( z _ { 0 } ( x ) , z _ { 1 } ( x ) , z _ { 2 } ( x ) )$ For each head $h \in \{ 1 , 2 , 3 \}$ and query $z _ { j } ( x )$ , we use the standard scaled dot-product attention over the corresponding memory pair $( m _ { j , + } , m _ { j , - } )$ . Specifically, for $\sigma \in \{ - 1 , 1 \}$ , the attention weight $A _ { h , j , \sigma }$ can be calculated as

$$
A _ { h , j , \sigma } ( x ) = \frac { \exp \left( \left. Q _ { h } z _ { j } ( x ) , K _ { h } m _ { j , \sigma } \right. / \sqrt { 3 } \right) } { \Sigma _ { \tau \in \{ \pm 1 \} } \exp \left( \left. Q _ { h } z _ { j } ( x ) , K _ { h } m _ { j , \tau } \right. / \sqrt { 3 } \right) } .
$$

Roughly speaking $A _ { h , j , + } ( x )$ and $A _ { h , j , - } ( x )$ are simply the two softmax weights assigned by head h to the positive and negative memory token associated with feature $j .$ The corresponding value vectors are $V _ { h } m _ { j , + }$ and $V _ { h } m _ { j , - }$ , and the output $g _ { h , j } ( x )$ of head h for the query $j$ can be written as

$$
g _ { h , j } ( x ) = O _ { h } \left( \sum _ { \sigma \in \{ - 1 , 1 \} } A _ { h , j , \sigma } ( x ) V _ { h } m _ { j , \sigma } \right) \in \mathbb { R } ^ { 3 } .
$$

We then sum over all feature queries and all three attention heads: $\begin{array} { r } { g ( x ) = \sum _ { h = 1 } ^ { 3 } \sum _ { j = 0 } ^ { 2 } g _ { h , j } ( x ) \in } \end{array}$ $\mathbb { R } ^ { 3 }$ . Because our prediction target $Y$ is scalar, we finally apply the fixed readout $r ( z ) = \mathbf { 1 } ^ { \top } z ,$ and the scalar prediction of the model is $f _ { \theta } ( x ) = \mathbf { 1 } ^ { \top } g ( x )$ , where $\theta$ denotes the tuple containing all the trainable parameters $Q _ { h } , K _ { h } , V _ { h } , O _ { h }$ , for $h \in \{ 1 , 2 , 3 \}$

Diagonal parameterization. Following prior works of Transformer training dynamics $( \mathrm { e . g . }$ (Boix-Adsera et al., 2023)), for simplicity and tractability of the analysis, we restrict the query, key, value, and output projection matrices of each head to be diagonal, that is

$$
\begin{array} { r l } & { Q _ { h } = \mathrm { d i a g } ( q _ { h 0 } , q _ { h 1 } , q _ { h 2 } ) , K _ { h } = \mathrm { d i a g } ( k _ { h 0 } , k _ { h , 1 } , k _ { h 2 } ) , } \\ & { V _ { h } = \mathrm { d i a g } ( v _ { h 0 } , v _ { h 1 } , v _ { h 2 } ) , O _ { h } = \mathrm { d i a g } ( o _ { h 0 } , o _ { h 1 } , o _ { h 2 } ) . } \end{array}
$$

This restriction removes cross-coordinate interactions and allows us to focus on analyzing the competition among heads for each feature. Importantly, we note that this assumption does NOT make the optimization dynamics linear, as the query, key, value, and output parameters are still trained separately and interact multiplicatively through the softmax attention mechanism.

Training objective. Since our goal is to understand whether joint training can create the capability enhancement localization phenomenon, we train the same attention model jointly on tasks A and $B .$ For task $T \in \{ A , B \}$ , let $R _ { T } ( \theta ) \ = \ \mathbb { E } _ { T } [ ( f _ { \theta } ( X ) - Y ) ^ { 2 } ]$ denote the population squared risk of the attention model on task $T _ { \ast }$ , where $f _ { \theta }$ is the scalar predictor defined by the attention block above. Since the two considered tasks are equally weighted, we train the predictor $f _ { \theta }$ using the half-squared population loss, that is $\begin{array} { r } { \dot { \mathcal { L } } ( \theta ) = \frac { 1 } { 4 } ( R _ { A } ^ { \bullet } ( \theta ) ^ { \bullet } + R _ { B } ^ { \bullet } ( \theta ) } \end{array}$

Weight initialization and training algorithm. We train all query, key, value, and output parameters using fixed step-size full-batch gradient descent. Concretely, for each head $h \in \{ 1 , 2 , 3 \}$ and feature j, we collect the four trainable scalar parameters into $\theta _ { h j } = ( q _ { h j } , k _ { h j } , v _ { h j } , o _ { h j } )$ . We initialize the weights $\theta _ { h j }$ as $\theta _ { h j } ^ { 0 }$ , where

$$
\theta _ { h j } ^ { 0 } = \alpha G _ { h j } , \quad G _ { h j } \stackrel { i . i . d } { \sim } N ( 0 , I _ { 4 } ) ,
$$

where $\alpha > 0$ is a small positive constant controlling the initialization scale. In other words, the weights $\theta _ { h j }$ at initialization have covariance $\alpha ^ { 2 } I _ { 4 }$ . Starting from this initialization, all parameters are updated using full-batch GD

$$
\theta ^ { n + 1 } = \theta ^ { n } - \eta \nabla \mathcal { L } ( \theta ^ { n } ) ,
$$

where $\eta > 0$ is a fixed learning rate. Importantly, no feature is assigned to any head beforehand, and we do NOT perform sign conditioning, parameter tying, rebalancing, or any other intervention mechanism during training. This means that ifa particularfeature eventually becomes concentrated in one attention head, that localization must arise naturally from the training dynamics and the random initialization rather than from predetermined head assignment.

## 4.2 JOINT OPTIMALITY DOES NOT DETERMINE HEAD ALLOCATION

We first express the attention model in terms of its effective feature coefficients. This representation identifies the predictor that minimizes the joint population loss and separates this aggregate optimum from the allocation of features across individual heads.

Proposition 4.1 (Exact softmax reduction and coefficient compromise). Consider the three-head attention model described as in Section 4.1, with the fixed coordinate encoding, paired memory bank, and diagonal query, key, value and output projection matrices. For each head $h \in \{ 1 , 2 , 3 \}$ and feature $j \in \{ 0 , \bar { 1 } , 2 \}$ , define the effective head-feature coefficient $c _ { h j } = o _ { h j } v _ { h j }$ tanh $\mathsf { \Omega } _ { \mathsf { l } } ( q _ { h j } k _ { h j } )$ and let $\begin{array} { r } { w _ { j } = \sum _ { h = 1 } ^ { 3 } c _ { h j } } \end{array}$ denote the total coefficient assigned to feature j after summing over three heads. Then we have

1. Head-feature response: For each input $\boldsymbol { x } = ( x _ { 0 } , x _ { 1 } , x _ { 2 } )$ , the contribution of head h associated with feature j is exactly $x _ { j } c _ { h j } e _ { j }$

2. Predictor representation: The model prediction can be written as $f _ { \theta } ( x ) = w ^ { \top } x ,$ where $w =$ $( w _ { 0 } , w _ { 1 } , w _ { 2 } )$ and $w _ { i }$ is defined as above. In other words, the resulting predictor is linear in the input features. However, the effective coefficients w<sub>j</sub> are highly non-linear functions of the trainable weights.

3. Population risks: Under the two data distributions defined as in Section 4.1, the population risks reduce to

$$
R _ { A } ( w ) = w _ { 0 } ^ { 2 } + ( w _ { 1 } - b ) ^ { 2 } , \quad R _ { B } ( w ) = ( w _ { 0 } - a ) ^ { 2 } + ( w _ { 2 } - c ) ^ { 2 } .
$$

4. Joint training objective and optimum: As a result, the population objective $\begin{array} { r l } { { \mathcal { L } } ( \theta ) } & { { } = } \end{array}$ $\begin{array} { r } { \frac { 1 } { 4 } \left( R _ { A } ( w ( \theta ) ) + \dot { R _ { B } } ( w ( \theta ) ) \right) } \end{array}$ can be rewritten as

$$
\mathcal { L } ( \theta ) = \frac { 1 } { 2 } \left( w _ { 0 } - \frac { a } { 2 } \right) ^ { 2 } + \frac { 1 } { 4 } ( w _ { 1 } - b ) ^ { 2 } + \frac { 1 } { 4 } ( w _ { 2 } - c ) ^ { 2 } + \frac { a ^ { 2 } } { 8 } .
$$

This implies the optimal aggregate coefficient vector $\begin{array} { r } { w ^ { \star } = \beta = \left( \frac { a } { 2 } , b , c \right) } \end{array}$

Proof sketch. The proof of Proposition 4.1 is straightforward from the original problem setting (Section 4.1). The detailed proof can be found in Appendix B. 口

Task conflict produces a compromise. Within this model, the coefficient on (U) is shared across tasks and cannot depend on which task generated the input. Task (A) requires coefficient zero, whereas task (B) requires coefficient (a). Joint optimization therefore selects $\bar { w } _ { 0 } ^ { \star } = a / 2$ , while fitting the task-specific coefficients (b) and (c) exactly. Both tasks have risk $a ^ { 2 } / 4$ at this joint optimum.

Head allocation remains undetermined. Although the optimal aggregate vector is unique, its decomposition across heads is not: the objective constrains only the sums $\begin{array} { r } { w _ { j } \ = \ \sum _ { h } c _ { h j } } \end{array}$ . This distinction matters because ablating head h changes the predictor’s coefficient vector from w to $w - c _ { h }$ , where $c _ { h } = ( c _ { h 0 } , c _ { h 1 } , c _ { h 2 } )$ . Removing a head that carries the conflicting coordinate may therefore also remove useful task-specific signal. Establishing beneficial head ablation requires understanding which decomposition training selects. We next show that gradient descent from small random initialization concentrates each feature in a single head.

## 4.3 GRADIENT DESCENT LOCALIZES FEATURES

We now show that gradient descent selects a particular structure among the many head decompositions compatible with the optimal aggregate predictor. Under small Gaussian initialization, each feature becomes concentrated in one head, while its contributions from the remaining heads stay small throughout training.

Theorem 4.2 (Training-induced feature localization). For $h \in \{ 1 , 2 , 3 \} , j \in \{ 0 , 1 , 2 \}$ , let

$$
\theta _ { h j } = ( q _ { h j } , k _ { h j } , v _ { h j } , o _ { h j } ) , \quad c _ { h j } = o _ { h j } v _ { h j } \operatorname { t a n h } ( q _ { h j } k _ { h j } ) , \quad w _ { j } = \sum _ { h = 1 } ^ { 3 } c _ { h j } .
$$

Fix $a , b , c > 0 ,$ , set $\beta = \left( { \textstyle { \frac { a } { 2 } } } , b , c \right)$ and $p = \left( 1 , \frac { 1 } { 2 } , \frac { 1 } { 2 } \right)$ , and train $\begin{array} { r } { \mathcal { L } ( \theta ) = \frac { 1 } { 2 } \sum _ { j = 0 } ^ { 2 } p _ { j } ( w _ { j } - \beta _ { j } ) ^ { 2 } + \frac { a ^ { 2 } } { 8 } } \end{array}$ by full-batch GD from the initialization $\theta _ { h j } ^ { 0 } = \alpha G _ { h j }$ , where $G _ { h j } \stackrel { i . i . d . } { \sim } N ( 0 , I _ { 4 } )$ . Then there exists $\eta _ { \star } > 0$ , depending only on $a , b , c ,$ such that the following holds for every fixed step size $0 < \eta < \eta _ { \star }$ For almost every draw ofG, andfor all sufficiently small $\alpha > 0 ;$ , we have:

1. Convergence: the parameter trajectory is bounded and converges to a finite limit with $w ^ { \infty } = \beta .$ Each residual $r _ { j } ^ { n } \overset { \cdot } { = } \beta _ { j } - w _ { j } ^ { n }$ is positive and non-increasing.

2. Localization: there are labels $h _ { j } = h _ { j } ( G )$ , independent and uniform on $\{ 1 , 2 , 3 \}$ , and a finite constant $K _ { G }$ , independent ofα, such that

$$
\operatorname* { s u p } _ { n \geq 0 } \| \theta _ { h j } ^ { n } \| _ { \infty } \leq K _ { G } \alpha , \quad \operatorname* { s u p } _ { n \geq 0 } | c _ { h j } ^ { n } | \leq K _ { G } \alpha ^ { 4 } { \mathrm { ~ f o r ~ } } h \neq h _ { j } .
$$

Consequently, $c _ { h _ { j } j } ^ { \infty } = \beta _ { j } + \mathcal { O } _ { G } ( \alpha ^ { 4 } )$

3. Finite checkpoint: define $\begin{array} { r } { N _ { \alpha } = \lceil \frac { \log ( 1 / \alpha ) } { \eta \alpha ^ { 2 } } \rceil } \end{array}$ , for every fixed $M > 0$ and sufficiently small α, we have $\begin{array} { r } { \operatorname* { s u p } _ { n \geq N _ { \alpha } } \operatorname* { m a x } _ { j } r _ { j } ^ { n } \leq \alpha ^ { M } } \end{array}$

The proof details of Theorem 4.2 can be found in Appendix B. Intuitively, Theorem 4.2 establishes localization through training without prescribing a feature-to-head assignment. Every head initially has access to every feature, but random initialization and gradient descent select a winning head for each feature. This is a feature-wise statement: different features may share the same winning head. Thus, localization alone does not yet guarantee beneficial head ablation. The next subsection identifies an allocation event under which removing the head carrying the conflicting feature preserves the task-specific features and improves task A.

## 4.4 LOCALIZATION ENABLES BENEFICIAL HEAD ABLATION

We now connect feature localization to task-specific improvement. If the conflicting feature is concentrated in a head that carries neither task-specific feature, ablating that head removes the unwanted contribution to task A while approximately preserving its useful signal. The independent winninghead assignments in Theorem 4.2 give this event a positive probability. The detailed proofs of Theorem 4.3 can be found in Appendix B.

Theorem 4.3 (Training-induced beneficial head ablation). Under the assumptions of Theorem 4.2, let $t = a / 2 ,$ , and define the event

$$
E = \{ h _ { 0 } \notin \{ h _ { 1 } , h _ { 2 } \} \} ,
$$

where $h _ { j }$ is the head that wins the feature $j \in \{ 0 , 1 , 2 \}$ . Then $\textstyle \operatorname* { P r } ( E ) = { \frac { 4 } { 9 } }$ . Under such event, set $h _ { \star } = h _ { 0 } ,$ , and delete this head without changing any other parameters

$$
f _ { - h _ { \star } } ^ { n } = f ^ { n } - f _ { h _ { \star } } ^ { n } .
$$

For almost every initialization draw G in E, all sufficiently small $\alpha > 0 ,$ , and every $n \geq N _ { \alpha } ,$

$$
G _ { A } ^ { n } \triangleq R _ { A } ( f ^ { n } ) - R _ { A } ( f _ { - h _ { \star } } ^ { n } ) = t ^ { 2 } + \mathcal { O } _ { G } ( \alpha ^ { 4 } ) > 0 ,
$$

$$
D _ { B } ^ { n } \triangleq R _ { B } ( f _ { - h _ { \star } } ^ { n } ) - R _ { B } ( f ^ { n } ) = 3 t ^ { 2 } + \mathcal { O } _ { G } ( \alpha ^ { 4 } ) > 0 ,
$$

where the error bounds are uniform over $n \geq N _ { \alpha }$ . Consequently

$$
\operatorname* { l i m i n f } _ { \alpha \to 0 ^ { + } } \operatorname* { P r } _ { G } \left( \exists h : R _ { A } ( f ^ { N _ { \alpha } } ) - R _ { A } ( f _ { - h } ^ { N _ { \alpha } } ) \frac { t ^ { 2 } } { 2 } , R _ { B } ( f _ { - h } ^ { N _ { \alpha } } ) - R _ { B } ( f ^ { N _ { \alpha } } ) \geq \frac { 3 t ^ { 2 } } { 2 } \right) \geq \frac { 4 } { 9 } .
$$

Interpretation of Theorem 4.3. In the small-initialization limit, both intact task risks approach $t ^ { 2 }$ . Removing the selected head reduces task $\mathrm { A } \mathbf { \bar { s } }$ risk to zero but increases task B’s risk to $4 t ^ { 2 }$ Consequently, the joint objective increases by $t ^ { 2 } / 2$ . The intervention therefore improves one task by moving away from the jointly optimal compromise. By parameter convergence, the same ablation conclusions also hold at the limiting trained model. This therefore theoretically describes the CEL phenomenon.

## 5 DISCUSSION AND LIMITATIONS

Empirical scope. Our experiments provide evidence of beneficial sparse head ablations across three model families and four benchmarks. These findings concern the evaluated models, datasets, and inference protocols; they do not establish that beneficial ablations exist for every model–task pair. Although selection and evaluation use disjoint examples, the stability of the selected subsets and their gains across different discovery samples remains insufficiently characterized.

Search and head interactions. Our two-stage procedure identifies beneficial interventions but does not guarantee an optimal subset. Individual-head screening can exclude combinations whose benefits emerge only through joint ablation. Consequently, the gains depend on both the discovery data and the candidate pool, and an unsuccessful search would not establish the absence of beneficial sparse interventions.

Theoretical scope. We note that our analysis establishes a possible training-induced mechanism for CEL under a fixed feature encoding, diagonal projections, small Gaussian initialization, and full-batch gradient descent on a noiseless population objective. The architecture supplies the feature directions and restricts cross-coordinate interactions, while training determines their allocation across heads. The resulting task conflict arises within the shared predictor class. Therefore, the theory establishes one possible mechanism through which beneficial head ablation can arise despite joint optimality. It does not establish that this mechanism explains the empirical gains in the evaluated language models, nor that it accounts for all instances of CEL.

## 6 CONCLUSION AND FUTURE WORK

In this work, we study Capability Enhancement Localization (CEL): whether removing a small set of attention heads can improve task-specific performance without further fine-tuning. Across three model families and four benchmarks, a simple two-stage search identifies sparse head subsets whose ablation improves held-out accuracy. Motivated by these observations, we establish a potential training-induced mechanism for CEL in a simplified softmax attention model. Gradient descent from small Gaussian initialization concentrates features in individual heads, and, with positive probability, removing the head carrying a conflicting feature improves one task while harming another. This can occur even at a global minimum of the joint training objective, showing that successful joint optimization is compatible with beneficial task-specific head ablation.

We note that this work only provides an initial empirical and theoretical investigation of CEL. On the empirical side, future work could develop more efficient search procedures that account for interactions among heads, including beneficial combinations that individual-head screening may overlook. On the theoretical side, an important direction is to characterize when training produces beneficial head ablations beyond the fixed feature encoding and diagonal parameterization studied here. In particular, allowing learned representations and interactions among features could clarify when conflicting contributions become sufficiently separated from useful ones for sparse ablation to improve task performance.

## AI USE STATEMENT

Experiment implementation. For the codes of all experiments described in this work, the authors use OpenAI Codex (with models GPT-5.6 Sol, GPT-6 Astra) for implementations.

Theoretical results. For the theoretical models, claims, and construction provided in this work, the authors use GPT-6.0 Astra for brainstorming ideas, theorem construction, and theorem proving, though the authors already had clue about the constructions at the first place. After that, the authors organize the statements, the proofs, and the elaboration in this paper with the help of LLMs. Finally, when incorporating the statements/proofs/elaborations into the main body and appendix, the authors rewrite the contents based on the authors’ understanding. The authors verified the soundness and correctness of the theoretical results presented in this draft.

Paper writing. For paper writing, the authors use GPT-5.6 Sol and Grammarly. For every paragraph in this paper, the authors use the following workflow: (1) the authors draft the paragraph without the use of LLMs, (2) the authors then use GPT-5.6 Sol to improve/correct the wording and grammars, (3) the authors again modify the paragraph (fixed by GPT-5.6 Sol) again to improve the flow and personalize the deliver, and (4) the authors finally use Grammarly to fix the typos and grammars once again. The authors provide the citations and references by hand in the first step using Google Scholar, and no LLM is used for generating citation.

## REFERENCES

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Anna Bair, Yixuan Even Xu, Mingjie Sun, and J Zico Kolter. Compressed sensing for capability localization in large language models. arXiv preprint arXiv:2603.03335, 2026.

Elie Bakouch, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Lewis Tunstall, Carlos Miguel Patino, Edward Beeching, Aymeric Roucher, Aksel Joonas Reedi, Quentin Gallou˜ edec, et al.´ Smollm3: smol, multilingual, long-context reasoner. Hugging Face Blog, 2025.

Raphael Berthier. Incremental learning in diagonal linear networks. ¨ Journal of Machine Learning Research, 24(171):1–26, 2023.

Enric Boix-Adsera, Etai Littwin, Emmanuel Abbe, Samy Bengio, and Joshua Susskind. Transformers learn through gradual rank increase. Advances in Neural Information Processing Systems, 36: 24519–24551, 2023.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 2924–2936, 2019a.

Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher D Manning. What does bert look at? an analysis of bert’s attention. In Proceedings of the 2019 ACL workshop BlackboxNLP: analyzing and interpreting neural networksfor NLP, pp. 276–286, 2019b.

Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adria Garriga-\` Alonso. Towards automated circuit discovery for mechanistic interpretability. Advances in Neural Information Processing Systems, 36:16318–16352, 2023.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Peter Hase, Mohit Bansal, Been Kim, and Asma Ghandeharioun. Does localization inform editing? surprising differences in causality-based localization vs. knowledge editing in language models. Advances in Neural Information Processing Systems, 36:17643–17668, 2023.

William Held and Diyi Yang. Shapley head pruning: Identifying and removing interference in multilingual transformers. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pp. 2416–2427, 2023.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Xiusheng Huang, Jiaxiang Liu, Yequan Wang, Jun Zhao, and Kang Liu. Capability localization: Capabilities can be localized rather than individual knowledge. In International Conference on Learning Representations, volume 2025, pp. 40446–40462, 2025.

Zhuoran Jin, Pengfei Cao, Hongbang Yuan, Yubo Chen, Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu, and Jun Zhao. Cutting off the head ends the conflict: A mechanism for interpreting and mitigating knowledge conflicts in language models. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pp. 1193–1215, 2024.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. Advances in neural information processing systems, 35:17359–17372, 2022.

Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? Advances in neural information processing systems, 32, 2019.

Hieu Trung Nguyen, Bao Nguyen, and Viet Anh Nguyen. Structured pruning for diverse best-of-n reasoning optimization. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pp. 23911–23922, 2025.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022.

Abhishek Panigrahi, Nikunj Saunshi, Haoyu Zhao, and Sanjeev Arora. Task-specific skill localization in fine-tuned language models. In International Conference on Machine Learning, pp. 27011–27033. PMLR, 2023.

Scott Pesme and Nicolas Flammarion. Saddle-to-saddle dynamics in diagonal linear networks. Advances in Neural Information Processing Systems, 36:7475–7505, 2023.

A Yang Qwen, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengpeng Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, pp. 2412–15115, 2025.

Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446, 2021.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67, 2020.

Robert Tibshirani. Regression shrinkage and selection via the lasso. Journal ofthe Royal Statistical Society Series B: Statistical Methodology, 58(1):267–288, 1996.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In Proceedings of the 57th annual meeting of the association for computational linguistics, pp. 5797–5808, 2019.

Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. arXiv preprint arXiv:2211.00593, 2022.

Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.

Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. Retrieval head mechanistically explains long-context factuality. In International Conference on Learning Representations, volume 2025, pp. 62143–62156, 2025.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th annual meeting of the association for computational linguistics, pp. 4791–4800, 2019.

Wei Zhang, Chaoqun Wan, Yonggang Zhang, Yiu-ming Cheung, Xinmei Tian, Xu Shen, and Jieping Ye. Interpreting and improving large language models in arithmetic calculation. arXiv preprint arXiv:2409.01659, 2024.

## A ADDITIONAL INFORMATION FOR EXPERIMENT SETTING

Table 2: Model details. Here, H denotes query heads per layer, and $N = L H$ is the total number of ablatable heads. In our experiments, for convenience, SmolLM3 is evaluated in non-thinking mode.
<table><tr><td>MODEL</td><td>LAYERS L</td><td>HEADS H</td><td>TOTAL HEADS N</td></tr><tr><td>LLaMA-3.2-3B</td><td>28</td><td>24</td><td>672</td></tr><tr><td>Qwen2.5-3B</td><td>36</td><td>16</td><td>576</td></tr><tr><td>SmolLM3-3B</td><td>36</td><td>16</td><td>576</td></tr></table>

Table 3: Dataset sizes for discovery $D _ { \mathrm { d i s t } }$ and held-out evaluation $D _ { \mathrm { t e s t } }$
<table><tr><td>DATASET</td><td>DISCOVERY</td><td>HELD-OUT</td></tr><tr><td>Arithmetic</td><td>2,000</td><td>1,000</td></tr><tr><td>HellaSwag</td><td>2,000</td><td>10,042</td></tr><tr><td>BoolQ</td><td>2,000</td><td>3,270</td></tr><tr><td>MBPP</td><td>374</td><td>500</td></tr></table>

## B ADDITIONAL RESULTS AND OMITTED PROOFS FOR SECTION 4

In this section, we will provide the detailed proof for the results presented in Section 4.

## B.0.1 OMITTED PROOF FOR SECTION 4.2

First, we will provide the detailed proof for Proposition 4.1, which provides simplification for the feature response, predictor representation, population risks, and training optimum in our setting.

Proposition 4.1 (restated). Consider the three-head attention model described as in Section 4.1, with thefixed coordinate encoding, paired memory bank, and diagonal query, key, value and output projection matrices. For each head $\hat { h } \in \{ 1 , 2 , 3 \}$ andfeature $j \in \{ 0 , 1 , \overset { \cdot } { 2 } \}$ , define the effective headfeature coefficient $c _ { h j } = o _ { h j } v _ { h j }$ tanh $\left( q _ { h j } k _ { h j } \right)$ , and let $\begin{array} { r } { w _ { j } = \sum _ { h = 1 } ^ { 3 } c _ { h j } } \end{array}$ denote the total coefficient assigned to feature j after summing over three heads. Then we have

1. Head-feature response: For each input $\boldsymbol { x } = ( x _ { 0 } , x _ { 1 } , x _ { 2 } )$ , the contribution of head h associated with feature j is exactly $x _ { j } c _ { h j } e _ { j }$

2. Predictor representation: The model prediction can be written as $f _ { \theta } ( x ) = w ^ { \top } x ,$ , where $w =$ $( w _ { 0 } , w _ { 1 } , w _ { 2 } )$ and $w _ { i }$ is defined as above. In other words, the resulting predictor is linear in the input features. However, the effective coefficients w<sub>j</sub> are highly non-linear functions of the trainable weights.

3. Population risks: Under the two data distributions defined as in Section 4.1, the population risks reduce to

$$
R _ { A } ( w ) = w _ { 0 } ^ { 2 } + ( w _ { 1 } - b ) ^ { 2 } , \quad R _ { B } ( w ) = ( w _ { 0 } - a ) ^ { 2 } + ( w _ { 2 } - c ) ^ { 2 } .
$$

4. Joint training objective and optimum: As a result, the population objective $\begin{array} { r l } { { \mathcal { L } } ( \theta ) } & { { } = } \end{array}$ $\begin{array} { r } { \frac { 1 } { 4 } \left( R _ { A } ( w ( \theta ) ) + \dot { R _ { B } } ( w ( \theta ) ) \right) } \end{array}$ can be rewritten as

$$
\mathcal { L } ( \theta ) = \frac { 1 } { 2 } \left( w _ { 0 } - \frac { a } { 2 } \right) ^ { 2 } + \frac { 1 } { 4 } ( w _ { 1 } - b ) ^ { 2 } + \frac { 1 } { 4 } ( w _ { 2 } - c ) ^ { 2 } + \frac { a ^ { 2 } } { 8 } .
$$

This implies the optimal aggregate coefficient vector $\begin{array} { r } { w ^ { \star } = \beta = \left( \frac { a } { 2 } , b , c \right) } \end{array}$

Proof. We will present the proof in the following steps.

Step 1: Head-feature response: For a fixed head h and a feature $j ,$ we recall that the query associated with feature $j$ is $z _ { j } ( x ) ~ = ~ \sqrt { 3 } x _ { j } e _ { j }$ , while the two corresponding memory tokens are

$m _ { j , + } = e _ { j } , m _ { j , - } = - e _ { j }$ . From assumptions, the key and query matrices are diagonal, we have $Q _ { h } z _ { j } ( x ) = \sqrt { 3 } x _ { j } q _ { h j } e _ { j } , K _ { h } m _ { j , + } = k _ { h j } e _ { j }$ , and $K _ { h } m _ { j , - } = - k _ { h j } e _ { j }$ . Therefore, after the $( 1 / \sqrt { 3 } )$ scaling in the dot-product attention, the two logits are

$$
+ x _ { j } q _ { h j } k _ { h j } , \quad - x _ { j } q _ { h j } k _ { h j } .
$$

This means that the two softmax weights are

$$
A _ { h , j , + } ( x ) = \frac { e ^ { x _ { j } q _ { h j } k _ { h j } } } { e ^ { x _ { j } q _ { h j } k _ { h j } } + e ^ { - x _ { j } q _ { h j } k _ { h j } } } , \quad A _ { h , j , - } ( x ) = \frac { e ^ { - x _ { j } q _ { h j } k _ { h j } } } { e ^ { x _ { j } q _ { h j } k _ { h j } } + e ^ { - x _ { j } q _ { h j } k _ { h j } } } .
$$

The corresponding value vectors are $V _ { h } m _ { j , + } ~ = ~ v _ { h j } e _ { j } , ~ V _ { h } m _ { j , - } ~ = ~ - v _ { h j } e _ { j }$ , and the softmax weighted combination is

$$
v _ { h j } \frac { e ^ { x _ { j } q _ { h j } k _ { h j } } - e ^ { - x _ { j } q _ { h j } k _ { h j } } } { e ^ { x _ { j } q _ { h j } k _ { h j } } + e ^ { - x _ { j } q _ { h j } k _ { h j } } } = v _ { h j } \operatorname { t a n h } ( x _ { j } q _ { h j } k _ { h j } ) e _ { j } .
$$

After applying the output projection $O _ { h }$ , the contribution of the head $h$ for feature $j$ is $o _ { h j } v _ { h j } \operatorname { t a n h } ( x _ { j } q _ { h j } k _ { h j } ) e _ { j }$ . Note that $x _ { j } \in \{ - 1 , 0 , 1 \}$ , and tanh is an odd function, we have

$$
g _ { h , j } ( x ) = x _ { j } c _ { h j } e _ { j } , \quad c _ { h j } = o _ { h j } v _ { h j } \operatorname { t a n h } ( q _ { h j } k _ { h j } ) ,
$$

as expected.

Step 2: Predictor representation: We now sum over all heads to have $\begin{array} { r } { w _ { j } = \sum _ { h = 1 } ^ { 3 } c _ { h j } } \end{array}$ . After summing over all feature queries and all heads, and applying the scalar readout, the model prediction becomes $f _ { \theta } ( x ) = w ^ { \top } x$ as expected. Here, we can see that the model is linear in the input x. However, the coefficient $w _ { j }$ are not directly trained parameters, but complicated functions of parameters

$$
w _ { j } = \sum _ { h = 1 } ^ { 3 } c _ { h j } , c _ { h j } = o _ { h j } v _ { h j } \operatorname { t a n h } ( q _ { h j } k _ { h j } ) .
$$

This is an important observation for the later localization result.

Step 3: task-specific population risks and joint training objective: Using the definition of task A and $B ,$ the closed form of the predictor $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ , and the fact that the feature is Rademacher variables, we can directly calculate $R _ { A } ( \dot { w } ) = w _ { 0 } ^ { 2 } + ( \dot { w } _ { 1 } - b ) ^ { 2 }$ and $R _ { B } ( w ) = ( w _ { 0 } - a ) ^ { 2 } + ( w _ { 2 } - c ) ^ { 2 }$ . Therefore, the joint training objective is

$$
\mathcal { L } ( \theta ) = \frac { 1 } { 2 } \left( w _ { 0 } - \frac { a } { 2 } \right) ^ { 2 } + \frac { 1 } { 4 } ( w _ { 1 } - b ) ^ { 2 } + \frac { 1 } { 4 } ( w _ { 2 } - c ) ^ { 2 } + \frac { a ^ { 2 } } { 8 } ,
$$

and trivially the optimal aggregated coefficient $\begin{array} { r } { w ^ { \star } = \beta = \left( \frac { a } { 2 } , b , c \right) } \end{array}$

## B.1 ADDITIONAL RESULTS AND OMITTED PROOFS FOR SECTION 4.3

In this section, we will present the formal statements of supporting results and Theorem 4.2. The first result essentially says that, under small initialization and gradient flow trajectory (i.e., infinitesimal step-size) and for a given feature, there will be a corresponding leader head of which the weights will move away from initialization in a certain amount of time. Moreover, the earliest escaping time will be unique, i.e., there will not be two heads being the leaders for a single feature at the same time.

Lemma B.1 (Early competition among heads under small initialization). Fix afeature $j \in \{ 0 , 1 , 2 \}$ For each head $h \in \{ 1 , \dot { 2 } , 3 \}$ , the four trainable parameters associated with this feature are $\theta _ { h j } =$ $( q _ { h j } , k _ { h j } , v _ { h j } , o _ { h j } ) \in \mathbb { R } ^ { 4 }$ . Recall that they are initialized as

$$
\theta _ { h j } ^ { 0 } = \alpha G _ { h j } , \quad G _ { h j } \overset { \mathrm { i i d } } { \sim } N ( 0 , I _ { 4 } ) ,
$$

where $\alpha > 0$ is the small initialization scale. To describe their leading-order behavior near the (small) initialization, for a generic initial vector $g = ( g _ { q } , g _ { k } , g _ { v } , g _ { o } ) \in \bar { \mathbb R } ^ { 4 }$ , consider the auxiliary ODE

$$
\left\{ \begin{array} { l l } { \dot { u } _ { q } = u _ { k } u _ { v } u _ { o } , } \\ { \dot { q } _ { k } = u _ { q } u _ { v } u _ { o } , } \\ { \dot { u } _ { v } = u _ { q } u _ { k } u _ { o } , } \\ { \dot { u } _ { o } = u _ { q } u _ { k } u _ { v } , } \end{array} \right. \quad u ( 0 ) = g ,\tag{1}
$$

where $u ( t ) = ( u _ { q } ( t ) , u _ { k } ( t ) , u _ { v } ( t ) , u _ { o } ( t ) )$ describe the leading-order evolution of the normalized quartet $\theta _ { h j } / \alpha$ . Assume that the coordinate g are nonzero and that one coordinate has strictly smaller magnitude than the other three. Then:

1. Finite auxiliary escape time: The ODE in Eq. 1 has a unique maximal solution $u _ { g } : [ 0 , T ( g ) ) $ $\mathbb { R } ^ { 4 }$ , for some finite $T ( g )$ , and $\| u _ { g } ( t ) \| \to \infty a s t \to T ( g ) ^ { - }$ . We call $T ( g )$ the auxiliary escape time of the quartet.

2. Sign alignment and balancing: the product $u _ { q } ( t ) u _ { k } ( t ) u _ { v } ( t ) u _ { o } ( t )$ eventually becomes positive, and as $t \to T ( g ) ^ { - }$ , the weights are balanced

$$
\frac { u _ { r } ( t ) ^ { 2 } } { u _ { s } ( t ) ^ { 2 } }  1 , \quad \forall r , s \in \{ q , k , v , o \} .
$$

3. Unique leader for each feature: For three normalized Gaussian initialization $G _ { 1 j } , G _ { 2 j } , G _ { 3 j }$ iid ∼ ${ \cal N } ( 0 , I _ { 4 } ) $ , the escape times $T ( G _ { 1 j } ) , T ( G _ { 2 j } ) , T ( G _ { 3 j } )$ are almost surelyfinite and distinct. Therefore $\begin{array} { r } { \hat { h } _ { j } \triangleq \arg \operatorname* { m i n } _ { h \in \{ 1 , 2 , 3 \} } T ( G _ { h j } ) } \end{array}$ is almost surely uniquely defined. Moreover, $\begin{array} { r l r } { \mathrm { P r } ( \hat { h } _ { j } = 1 ) = } \end{array}$ $\operatorname* { P r } ( \hat { h } _ { j } = 2 ) = \operatorname* { P r } ( \hat { h } _ { j } = 3 ) = \textstyle { \frac { 1 } { 3 } }$ . The variable $\hat { h } _ { 0 } , \hat { h } _ { 1 } , \hat { h } _ { 2 }$ are independent acrossfeatures

Proof. We will organize the proof using the following steps.

Step 1: why the four-dimensional system is solvable. Fix one normalized initialization $g \ : = \ :$ $\left( g _ { q } , g _ { k } , g _ { v } , g _ { o } \right)$ , and let $u ( t ) = \smash { \bigl ( u _ { q } ( t ) , \dot { u } _ { k } ( t ) , u _ { v } ( t ) , u _ { o } ( t ) \bigr ) }$ be the solution of Eq. 1. For any coordinate $r \in \{ q , k , v , o \}$ , we have $\begin{array} { r } { \frac { d } { d t } u _ { r } ( t ) ^ { 2 } = 2 u _ { r } ( t ) \dot { u } _ { r } ( t ) } \end{array}$ . Note that $\begin{array} { r } { \frac { d } { d _ { t } } u _ { q } ^ { 2 } \ = \ 2 u _ { q } ( u _ { k } u _ { v } u _ { o } ) \ = } \end{array}$ $2 u _ { q } u _ { k } u _ { v } u _ { o } .$ , and the same identity holds for every coordinate:

$$
\frac { d } { d _ { t } } u _ { q } ^ { 2 } = \frac { d } { d _ { t } } u _ { k } ^ { 2 } = \frac { d } { d _ { t } } u _ { v } ^ { 2 } = \frac { d } { d _ { t } } u _ { o } ^ { 2 } = 2 u _ { q } u _ { k } u _ { v } u _ { o } .
$$

Therefore, the differences between squared coordinates are constant:

$$
u _ { r } ( t ) ^ { 2 } - u _ { s } ( t ) ^ { 2 } = g _ { r } ^ { 2 } - g _ { s } ^ { 2 } .
$$

Define $a _ { r } \triangleq g _ { r } ^ { 2 }$ , then there exists a single scalar function ξ(t) with $\xi ( 0 ) = 0$ such that

$$
u _ { r } ( t ) ^ { 2 } = a _ { r } + \xi ( t ) , \quad \forall r \in \{ q , k , v , o \} .\tag{2}
$$

Step 2: dynamic of the common scalar $\xi ( t )$ . Let $P ( t ) \ \triangleq \ u _ { q } ( t ) u _ { k } ( t ) u _ { v } ( t ) u _ { o } ( t )$ be the coordinate product. By Step 1 and $\operatorname { E q . 2 } ,$ we have $\dot { \xi } ( t ) ~ = ~ 2 P ( t )$ , and from Eq. 2, we have $\begin{array} { r } { P ( t ) ^ { 2 } = \dot { \prod } _ { r \in \{ q , k , v , o \} } \dot { ( a _ { r } + \xi ( t ) ) } } \end{array}$ . Therefore

$$
\dot { \xi } = \left\{ \begin{array} { l l } { 2 \sqrt { \prod _ { r } ( a _ { r } + \xi ) } , \quad P ( t ) > 0 , } \\ { - 2 \sqrt { \prod _ { r } ( a _ { r } + \xi ) } , \quad P ( t ) < 0 . } \end{array} \right.\tag{3}
$$

This means that the four-dimensional ODE has reduced to a one-dimensional separable equation as above.

Step 3: Positive initial products. Let $P _ { 0 } \triangleq g _ { q } g _ { k } g _ { v } g _ { 0 }$ . First, supposed that $P _ { 0 } > 0$ , then initially we have $\dot { \xi } > 0$ . Since all $u _ { r } ^ { 2 } = a _ { r } + \xi$ are increasing, no coordinate can cross zero, which means $P ( t )$ says positive. Therefore, $\xi ( t )$ increases monotonically from 0 toward $+ \infty$ , and separation of variables gives

$$
t = \int _ { 0 } ^ { \xi ( t ) } \frac { d s } { 2 \sqrt { \prod _ { r } ( a _ { r } + s ) } } .
$$

Define $\begin{array} { r } { I _ { + } ( g ) \triangleq \int _ { 0 } ^ { \infty } \frac { d s } { 2 \sqrt { \prod _ { s } ( a _ { r } + s ) } } } \end{array}$ , and for large s, note that $\textstyle { \sqrt { \prod _ { r } ( a _ { r } + s ) } } \approx s ^ { 2 }$ , so the integrand behaves as $\scriptstyle { \frac { 1 } { 2 s ^ { 2 } } }$ . Therefore $I _ { + } ( g ) < \infty$ , and $T ( g ) \ : = \ : I _ { + } ( g )$ when $P _ { 0 } > 0$ . Moreover, we have $\xi ( t ) \to \infty$ as $t \to T ( g ) ^ { - }$ , and therefore

$$
u _ { r } ( t ) ^ { 2 } = a _ { r } + \xi ( t )  \infty .
$$

This means that $\| u _ { g } ( t ) \| \to \infty$

Step 4: Negative initial products. Now assume that $P _ { 0 } < 0$ . Then initially, $\dot { \xi } < 0 ,$ so $\xi$ decreases. Let $m \triangleq \operatorname* { m i n } _ { r } a _ { r }$ . By assumption, this minimum is achieved by a unique coordinate, say $r _ { \star }$ . From $u _ { r _ { \star } } ( t ) ^ { 2 } = m + \xi ( t )$ , that coordinate reaches 0 when $\xi = - m$ . All other coordinate remain nonzero because their corresponding $ { a _ { r } } ^ { \mathrm { ~ \tiny ~ { ~ s ~ } ~ } }$ are strictly larger than m. At the point where $u _ { r _ { \star } } = 0$ , we have

$$
\dot { u } _ { r _ { \star } } = \prod _ { s \neq r _ { \star } } u _ { s } ( t ) \neq 0 ,
$$

which means the coordinate does not remain at zero but crossing through zero and changes sign. This change flips the sign of $P ( t ) = u _ { q } u _ { k } u _ { v } u _ { c }$ from negative to positive. Therefore, after the cross, $\dot { \xi } > 0$ , and the scalar trajectory follows $0 \to - m \to 0 \to + \infty$ . We define

$$
I _ { - } ( g ) \triangleq \int _ { - m } ^ { 0 } \frac { d s } { 2 \sqrt { \prod _ { r } ( a _ { r } + s ) } } > 0 .
$$

Near $s = - m ,$ , exactly one factor in the denominator vanishes linearly, so the integrand behave like $( s + m ) ^ { - 1 / 2 }$ , which is integrable, and therefore $0 < I _ { - } ( g ) < \infty$ . Now note that the trip from 0 to $- m$ takes time $I _ { - }$ , the trip fro $\mathbf { 1 } - m$ to 0 take time $I - ,$ , and the trip from 0 to ∞ take time $I _ { + }$ as shown above. Therefore, $\dot { T } ( g ) = I _ { + } ( g ) + 2 I _ { - } ( g )$ when $P _ { 0 } < 0$ . Hence, we conclude that in both case, $T ( g ) < \infty$

Step 5: Why the four factor magnitudes become balanced. From the above, we know that $u _ { r } ( \bar { t } ) ^ { 2 } = a _ { r } + \xi ( t )$ , and therefore

$$
\frac { u _ { r } ( t ) ^ { 2 } } { u _ { s } ( t ) ^ { 2 } } = \frac { a _ { r } + \xi ( t ) } { a _ { s } + \xi ( t ) } .
$$

As $t \to T ( g ) ^ { - }$ , we have $\xi ( t ) \to \infty$ , and therefore $\frac { u _ { r } ( t ) ^ { 2 } } { u _ { s } ( t ) ^ { 2 } }  1 .$

Step 6: Scaling the initialization. We have this useful observation: if the initial vector is multiplied by $r \mathrm { ~ > ~ } 0 , \operatorname { t }$ hen $a _ { i } \  \ r ^ { 2 } a _ { i }$ . Changing variables $\ s \ = \ r ^ { 2 } z$ in the integrals above gives $T ( r g ) =$ $r ^ { \dot { - } 2 } T ( g )$ This is also consistent with the actual initialization $\theta _ { h j } ^ { 0 } \ = \ \alpha G _ { h j }$ . If we applied the auxiliary ODE directly to the initialization α $\nu G _ { h j }$ , its escape time would be $T ( \alpha { \cal G } _ { h j } ) = \alpha ^ { - 2 } T ( { \cal G } _ { h j } )$ But the common factor $\alpha ^ { - 2 }$ is identical for all three heads. Therefore

$$
\arg \operatorname* { m i n } _ { h } T ( \alpha G _ { h j } ) = \arg \operatorname* { m i n } _ { h } T ( G _ { h j } ) .
$$

This is why we can skip the normalized factor α and compare the normalized escape times $T ( G _ { h j } )$ Step 7: Gaussian escape times have no atoms. Let $G \sim N ( 0 , I _ { 4 } )$ , and write $G = R \Omega$ , where $R = \| G \| = 0$ , and $\begin{array} { r } { \Omega = \frac { G } { \Vert G \Vert } } \end{array}$ . For an isotropic Gaussian vector, R and Ωare independent, and $R$ has a continuous density. From Step $^ { 6 , }$ we have $\begin{array} { r } { T ( G ) = T ( R \Omega ) = \frac { T ( \Omega ) } { R ^ { 2 } } } \end{array}$ . With probability one, all coordinates of Ω are nonzero and its smallest absolute coordinate is unique. By the preceding steps, $T ( \Omega ) \in ( 0 , \infty )$ for every such direction. Conditional on any such $\Omega , { \bar { T } } ( \Omega )$ is fixed, and the mapping $\dot { R } \mapsto \dot { T } ( \Omega ) \dot { / } R ^ { 2 }$ is strictly decreasing and continuous on $( 0 , \infty )$ . Therefore, for every fixed $t > 0$ , we have

$$
\operatorname* { P r } ( T ( G ) = t \mid \Omega ) = 0 .
$$

Averaging over $\Omega ,$ we have $\operatorname* { P r } ( T ( G ) = t ) = 0$

Step 8: Unique and uniformly early leader. For a fixed feature $j ,$ we have

$$
G _ { 1 j } , G _ { 2 j } , G _ { 3 j } \stackrel { \mathrm { i i d } } { \sim } N ( 0 , I _ { 4 } ) .
$$

Therefore, $T ( G _ { 1 j } ) , T ( G _ { 2 j } ) , T ( G _ { 3 j } )$ are i.i.d. atom-less random variables. Therefore, the ties occur with probability zero, and therefore

$$
\hat { h } _ { j } = \arg \operatorname* { m i n } _ { h } T ( G _ { h j } )
$$

is almost surely unique. Because the three head are exchangeable, we have

$$
\mathrm { P r } ( \hat { h } _ { j } = h ) = \frac { 1 } { 3 } , \quad h = 1 , 2 , 3 .
$$

Finally, the Gaussian initialization blocks for different feature are independent, therefore $\hat { h } _ { 0 } , \hat { h } _ { 1 } , \hat { h } _ { 2 }$ are independent, which concludes the proof.

Remark 1 (Interpretation of Lemma B.1). For each feature j, three heads begin from independent small random quartets. Their normalized leading-order dynamics have different escape times, so almost surely one head starts growing before the others. Lemma B.1 therefore identifies this early leader. However, it: (1) does NOT yet say that this head permanently wins the feature and (2) it establishes continuous time analysis. Both aspects will be discussed below.

Lemma B.1 establishes the leading head phenomenon for each features but under gradient flow assumption. In the following result, we connect these claims with the actually discrete GD iterates. Essentially, it shows that GD can track auxiliary ODE before escape times of the selected heads, provided the residual remains positive and unbounded.

Lemma B.2 (Discrete GD follows the auxiliary dynamics before escape). Fixed one feature and consider three heads $h \in \{ 1 , 2 , 3 \}$ . For each head $h ,$ let

$$
\theta _ { h } \triangleq ( q _ { h } , k _ { h } , v _ { h } , o _ { h } ) \in \mathbb { R } ^ { 4 } , c ( \theta _ { h } ) \triangleq o _ { h } v _ { h } \operatorname { t a n h } ( q _ { h } k _ { h } ) ,
$$

and define the total coefficient w $\triangleq \sum _ { h = 1 } ^ { 3 } c ( \theta _ { h } )$ . Let $\beta > 0$ be the target coefficient and $p > 0$ its loss weight. Consider the square loss $\begin{array} { r } { \mathcal { L } ( \theta ) = \frac { p } { 2 } ( w - \beta ) ^ { 2 } } \end{array}$ andfull-batch GD

$$
\theta _ { h } ^ { n + 1 } = \theta _ { h } ^ { n } - \eta \nabla _ { \theta _ { h } } \mathcal { L } ( \theta ^ { n } ) .
$$

Equivalently, writing $r _ { n } \triangleq \beta - w ^ { n }$ , and the update is

$$
\theta _ { h } ^ { n + 1 } = \theta _ { h } ^ { n } + n p r _ { n } \nabla c ( \theta _ { h } ^ { n } ) .
$$

Initialize $\theta _ { h } ^ { 0 } = \alpha g _ { h } , f o r g _ { h } \in \mathbb { R } ^ { 4 }$ and define the normalized parameters $u _ { h } ^ { n } \ \triangleq \ \frac { \theta _ { h } ^ { n } } { \alpha }$ . Also define $F _ { \alpha } ( u ) \triangleq \alpha ^ { - 3 } \nabla c ( \alpha u )$ , and the limiting four-factor vector field

$$
F ( u ) = \left[ \begin{array} { l } { u _ { k } u _ { v } u _ { o } } \\ { u _ { q } u _ { v } u _ { o } } \\ { u _ { q } u _ { k } u _ { o } } \\ { u _ { q } u _ { k } u _ { v } } \end{array} \right] .
$$

Assume that, over the GD iterates under consideration, $0 < r _ { n } \le \overline { { r } }$ , for some fixedfinite r. Define the residual-weighted auxiliary step and cumulative time by $\begin{array} { r } { \delta _ { n } \triangleq \alpha ^ { 2 } \eta p r _ { n } , \zeta _ { n } = \sum _ { k < n } \delta _ { k } } \end{array}$ . Let ${ \mathcal { T } } \subseteq \{ 1 , 2 , 3 \}$ be any nonempty set of heads andfix $S <$ min $\mathrel { \mathop { \gtrsim } } T ( g _ { h } )$ . Here, I specifies only the heads whose trajectories we approximate, and the residual $r ^ { n }$ and the GD updates still involve all three heads, then

1. The exact softmax field approaches the four-factor field: On every compact set $K \subset \mathbb { R } ^ { 4 }$ , we have $\| F _ { \alpha } - \mathbf { \bar { F } } \| _ { C ^ { 1 } ( K ) } \leq \hat { C } _ { K } ^ { \circ } \alpha ^ { 4 }$ for some constant $C _ { K }$ , and

$$
\| H \| _ { C ^ { 1 } ( K ) } \triangleq \operatorname* { s u p } _ { u \in K } \| H ( u ) \| + \operatorname* { s u p } _ { u \in K } \| D H ( u ) \| _ { \mathrm { o p } } .
$$

2. Normalized GD follows the auxiliary ODE before escape: there exists $C _ { S } < \infty$ such that, for all sufficiently small α,

$$
\operatorname* { m a x } _ { h \in \mathcal { T } } \| u _ { h } ^ { n } - u _ { h } ( \zeta _ { n } ) \| \leq C _ { S } \alpha ^ { 2 } ,
$$

for every iterate satisfying $\zeta _ { n } \leq S .$ . More precisely, the error is $\mathcal { O } ( \alpha ^ { 4 } + h _ { \alpha } )$ , where $h _ { \alpha } \triangleq \alpha ^ { 2 } \eta p \overline { { r } } ,$ and hence is $\mathcal { O } ( \alpha ^ { 2 } )$ for fixed $\eta , p , { \overline { { r } } } .$ . The same error bound also holds at the first crossing index $n _ { \star } = \operatorname* { m i n } \{ n : \zeta _ { n } \geq S \}$ , if it exists, for which $S \le \zeta _ { n \star } \le S + h _ { \alpha }$ . The step sizes $\delta _ { n }$ may depend on all current model parameters, that is, the approximation requires only their positivity and the bound $\delta _ { n } \leq h _ { \alpha }$ , not independencefrom the iterates.

Proof. We again divide the proof into the following steps.

Step 1: The exact softmax field approaches the four-factor field: From $c ( q , k , v , o ) ~ =$ ov tanh(qk), we have

$$
\begin{array} { r } { \nabla c ( q , k , v , o ) = \left( \begin{array} { c } { o v k \mathrm { s e c h } ^ { 2 } ( q k ) } \\ { o v q \mathrm { s e c h } ^ { 2 } ( q k ) } \\ { o \operatorname { t a n h } ( q k ) } \\ { v \operatorname { t a n h } ( q k ) } \end{array} \right) . } \end{array}
$$

Substituting $\theta \ = \ \alpha u$ , we have $( F _ { \alpha } ( u ) ) _ { q } ~ = ~ u _ { k } u _ { v } u _ { o } \mathrm { s e c h } ^ { 2 } ( \alpha ^ { 2 } u _ { q } u _ { k } )$ , and similarly $( F _ { \alpha } ( u ) ) _ { v } ~ =$ $u _ { o } \frac { \operatorname { t a n h } ( \alpha ^ { 2 } u _ { q } u _ { k } ) } { \alpha ^ { 2 } }$ . Using Taylor’s expansion, for small $z ,$ we have

$$
\mathrm { s e c h } ^ { 2 } ( z ) = 1 + \mathcal { O } ( z ^ { 2 } ) , \quad \operatorname { t a n h } z = z + \mathcal { O } ( z ^ { 3 } ) .
$$

This implies that $\| F _ { \alpha } - F \| _ { C ^ { 1 } ( K ) } \leq C _ { K } \alpha ^ { 4 }$

Step 2: Normalized GD follows the auxiliary ODE before escape: now divide the GD update

$$
\theta _ { h } ^ { n + 1 } = \theta _ { h } ^ { n } + n p r _ { n } \nabla c ( \theta _ { h } ^ { n } )
$$

by α, and since $\nabla c ( \alpha u ) = \alpha ^ { 3 } F _ { \alpha } ( u )$ by definition, we have

$$
u _ { h } ^ { n + 1 } = u _ { h } ^ { n } + \alpha ^ { 2 } \eta p r _ { n } F _ { \alpha } ( u _ { h } ^ { n } ) .
$$

Here, notice that $\alpha ^ { 2 } \eta p r _ { n } = \delta _ { n }$ . Therefore, normalized GD is a variable-step Euler discretization of the auxiliary ODE. Since $0 < r _ { n } \le \overline { { r } }$ , the steps $\delta _ { n }$ satisfies $0 < \delta _ { n } \leq h _ { \alpha } \triangleq \alpha ^ { 2 } \eta p \overline { { r } } \to 0$ for $\alpha  0$ Fix $S < \mathrm { m i n } _ { h \in \mathbb { Z } } T ( g _ { h } )$ . Since S lies strictly before every escape time, all exact ODE trajectories remain in a common compact set on a slightly larger interval $[ 0 , S + \rho ]$ . On this set $F$ is Lipschitz with some constant L. For one ODE steps, we have

$$
\begin{array} { r } { u _ { h } ( \zeta _ { n + 1 } ) = u _ { h } ( \zeta _ { n } ) + \delta _ { n } F ( u _ { h } ( \zeta _ { n } ) ) + \mathcal { O } ( \delta _ { n } ^ { 2 } ) . } \end{array}
$$

We then define

$$
e _ { n } \triangleq \operatorname* { m a x } _ { h \in \mathbb { Z } } \| u _ { h } ^ { n } - u _ { h } ( \zeta _ { n } ) \| .
$$

Choose $\rho > 0$ such that $S + \rho < \mathrm { { m i n } } _ { h \in \mathcal { T } } T ( g _ { h } )$ . The selected exact trajectories on $[ 0 , S + \rho ]$ lie in the interior of a compact cube $K ,$ , with a positive distance from its boundary. Take α sufficiently small that $h _ { \alpha } < \rho .$

As long as the numerical iterates lie in K, the preceding one-step estimate applies, including to a candidate step that would leave K. Since $e _ { 0 } = 0$ , applying discrete Gronwall inequality gives

$$
e _ { n } \leq C \exp \left( C \sum _ { k < n } \delta _ { k } \right) \left( \alpha ^ { 4 } \sum _ { k < n } \delta _ { k } + \sum _ { k < n } \delta _ { k } ^ { 2 } \right) .
$$

For iterates before the threshold and for the first crossing iterate, we have

$$
\sum _ { k < n } \delta _ { k } = \zeta _ { n } \leq S + h _ { \alpha } , \quad \sum _ { k < n } \delta _ { k } ^ { 2 } \leq h _ { \alpha } \sum _ { k < n } \delta _ { k } .
$$

Consequently, $e _ { n } \leq C _ { S } ( \alpha ^ { 4 } + h _ { \alpha } )$ . For sufficiently small α, this bound is smaller than the distance of the exact trajectories from the boundary $K .$ . Thus a first numerical exit from K is impossible, which implies the estimate throughout the claimed interval, including the first crossing step. If the clock never reaches S, the same argument applies to every finite prefix.

This argument uses only the realized bounds on the step sizes and their cumulative sum, and it does not require the mesh to be independent of the iterates. Finally, $h _ { \alpha } = \alpha ^ { 2 } \eta p \overline { { { r } } } = \mathcal { O } ( \alpha ^ { 2 } )$ for fixed $\eta , p ,$ and r. Therefore $e _ { n } \leq C _ { S } ^ { \prime } \alpha ^ { 2 }$ as desired. □

Remark 2. Lemma B.2 essentially says that as long as no auxiliary trajectory has reached its escape time, the normalized discrete GD trajectory stays close to the continuous four-factor trajectory from Lemma B.1. Therefore, the early head competition predicted by that auxiliary ODE genuinely describes the early stage ofthe actual GD dynamics.

The following lemma shows that, under suitable parameter and sufficiently small step-size, one GD steps preserves the positivity of the residual and does not increase it. In other words, the aggregated coefficient moves toward its target without overshooting. We will combine this result with the parameter bounds provided below to maintain these residual conditions throughout training.

Lemma B.3 (No residual overshoot). Fixed one feature and write $r = \beta - w , s = \eta p r ,$ , and $\theta _ { h } ^ { + } = \theta _ { h } + s \nabla c ( \theta _ { h } )$ , where $c ( q , k , v , o ) = o v \operatorname { t a n h } ( q k )$ , and $0 < p \le 1$ . Suppose that the weights of every head $\theta _ { h }$ satisfies $\theta _ { h } \in [ - B , B ] ^ { 4 }$ , and that $0 < r \leq D + 1$ , where $D = \operatorname* { m a x } ( 1 , t , b , c )$ $R = 1 2 ( D + 2 ) , B = 1 + \sqrt { R } ,$ and t, b, c are defined as in Section 4.1. $I f 0 < \eta \leq \eta _ { \star }$ , where

$$
\eta _ { \star } = \operatorname* { m i n } \left\{ \frac { 1 } { 2 ( D + 1 ) B ^ { 2 } } , \frac { 1 } { ( D + 1 ) L } , \frac { 1 } { 3 6 B ^ { 6 } } \right\} , \quad L = 4 ( 2 B ) ^ { 4 } + 3 ( 2 B ) ^ { 2 } + 1 ,
$$

then every update segment lies in $[ - 2 B , 2 B ] ^ { 4 }$ , andfor each head h

$$
\frac { s } { 2 } \| \nabla c ( \theta _ { h } ) \| _ { 2 } ^ { 2 } \leq c ( \theta _ { h } ^ { + } ) - c ( \theta _ { h } ) \leq \frac { 3 s } { 2 } \| \nabla c ( \theta _ { h } ) \| _ { 2 } ^ { 2 } .
$$

Consequently, $\begin{array} { r } { 0 \leq w ^ { + } - w \leq \frac { r } { 2 } , \frac { r } { 2 } \leq r ^ { + } \leq r , } \end{array}$ , where $\begin{array} { r } { w ^ { + } \triangleq \sum _ { h = 1 } ^ { 3 } c ( \theta _ { h } ^ { + } ) } \end{array}$ is the aggregate coefficient after one GD update, and $r ^ { + } \triangleq \beta - w ^ { + }$ . In other words, a positive residual remains positive after one GD step.

Proof. On a box $[ - b _ { 0 } , b _ { 0 } ] ^ { 4 }$ with $b _ { 0 } \geq 1$ , we have $\| \nabla c \| _ { 2 } ^ { 2 } \leq 4 b _ { 0 } ^ { 6 }$ , and $\| \nabla ^ { 2 } c \| _ { o p } \leq 4 b _ { 0 } ^ { 4 } + 3 b _ { 0 } ^ { 2 } + 1$ Since $s \leq \eta ( D + 1 )$ , the first step-size condition in the definition of η<sub>⋆</sub> implies s $\| \nabla c ( \bar { { \boldsymbol { \theta } } } _ { h } ) \| _ { \infty } \leq B$ so the whole update segment remains in $[ - 2 B , 2 B ] ^ { 4 }$ . On this box, we have $\| \nabla ^ { \dot { 2 } } c \| _ { o p } \leq \dot { L }$ . Using Taylor’s expansion, we have

$$
c ( \theta _ { h } ^ { + } ) - c ( \theta _ { h } ) = s \| \nabla c ( \theta _ { h } ) \| _ { 2 } ^ { 2 } + E _ { h } ,
$$

where the residual $\begin{array} { r } { | E _ { h } | \le \frac { s ^ { 2 } L } { 2 } \| \nabla c ( \theta _ { h } ) \| _ { 2 } ^ { 2 } } \end{array}$ . Since $s L \leq 1$ , we have

$$
\frac { s } { 2 } \| \nabla c ( \theta _ { h } ) \| _ { 2 } ^ { 2 } \leq c ( \theta _ { h } ^ { + } ) - c ( \theta _ { h } ) \leq \frac { 3 s } { 2 } \| \nabla c ( \theta _ { h } ) \| _ { 2 } ^ { 2 } .
$$

Summing over three heads, and note that $\| \nabla c ( \theta _ { h } ) \| _ { 2 } ^ { 2 } \leq 4 B ^ { 6 }$ , we have

$$
0 \leq w ^ { + } - w \leq 1 8 s B ^ { 6 } .
$$

Since $s = \eta p r \leq \eta r$ , and $\eta ~ \le ~ ( 3 6 B ^ { 6 } ) ^ { - 1 }$ , we have $w ^ { + } - w \leq \frac { r } { 2 }$ . Finally, $r ^ { + } = \beta - w ^ { + } =$ $r - ( w ^ { + } - w )$ , which implies $\begin{array} { r } { \frac { r } { 2 } \leq r ^ { + } \leq r } \end{array}$ as desired. □

We now establish an invariant region for the candidate winning head.

Lemma B.4 (Bounded invariant region for the winning head). Fix onefeature and let h<sup>⋆</sup> be the head with the smallest auxiliary escape time from Lemma B.1. Consider a discrete GD iterate obtained after applying Lemma B.2 at a time sufficiently close to that escape, so that the four parameters of $h ^ { \star }$ , that is $\theta = ( q , k , v , o )$ , have positive product and are approximately balanced. After afixed sign change that leave both c(θ) and the GD dynamics unchanged, we may assume $q , k , v , o > 0$ . Define

$$
Q = q ^ { 2 } + k ^ { 2 } , V = v ^ { 2 } + o ^ { 2 } , d = q ^ { 2 } - k ^ { 2 } , e = v ^ { 2 } - o ^ { 2 } .
$$

Suppose that at the current iterate

$$
| d | \leq \frac { Q } { 3 } , | e | \leq \frac { V } { 3 } , Q \leq 2 V , Q \leq R , V \leq R ,
$$

and whenever $Q \leq 1$ , we have $V \leq 3 Q .$ . Also, assume that Lemma B.3 applies to the next GD step, that the next total coefficient satisfies $\begin{array} { r } { w ^ { + } < \beta \le D , } \end{array}$ , and that the total next-step contribution of the two heads is at least −1. Then the updated parameters ofh<sup>⋆</sup> satisfy the same inequalities, that is

$$
{ q ^ { + 2 } , k ^ { + 2 } , v ^ { + 2 } , o ^ { + 2 } < R . }
$$

Therefore, allfour updated parameters remain in the box $[ - B , B ]$

Proof. Let $x = q k , y = v o$ . Since $q , k , v , o > 0$ , we have $x , y > 0$ . The exact GD update gives

$$
d ^ { + } = ( 1 - s ^ { 2 } y ^ { 2 } \mathrm { s e c h } ^ { 4 } x ) d , \quad e ^ { + } = ( 1 - s ^ { 2 } \operatorname { t a n h } ^ { 2 } x ) e .
$$

By the step-size conditions in Lemma B.3, both multipliers lie in $[ 0 , 1 ]$ , and therefore $| d ^ { + } | \leq | d | ,$ $| e ^ { + } | \leq | e |$ . At the same time, $Q$ and V are non-decreasing, so

$$
| d ^ { + } | \leq \frac { Q ^ { + } } { 3 } , | e ^ { + } | \leq \frac { V ^ { + } } { 3 } .
$$

Thus, the two parameter pairs remain balanced, which implies

$$
q k \geq { \frac { Q } { 3 } } , \quad v o \geq { \frac { V } { 3 } } .
$$

Moreover, the exact updates of $Q$ and $V$ satisfy

$$
\Delta Q = 4 s x y s \mathrm { e c h } ^ { 2 } x + s ^ { 2 } y ^ { 2 } \mathrm { s e c h } ^ { 4 } x Q , \quad \Delta V = 4 s y \mathrm { t a n h } x + s ^ { 2 } \mathrm { t a n h } ^ { 2 } x V .
$$

Using the fact that xsech<sup>2</sup>x ≤ tanh x, and together with the balanced bounds and the step-size restrictions, we have $\begin{array} { r } { \frac { \Delta Q } { \Delta V } \leq \frac { 7 } { 4 } < 2 } \end{array}$ . Since $Q \leq 2 V$ , we have

$$
Q ^ { + } = Q + \Delta Q \leq 2 V + 2 \Delta V = 2 V ^ { + } .
$$

The task left is to show that the boundedness. Let $c ^ { + }$ denote the contribution of $h ^ { \star }$ after the update. Since $w ^ { + } < \beta \leq D$ , and the other two heads contribute at least −1, we have $c ^ { + } \leq D + 1$ . By balance, we have

$$
q ^ { + } k ^ { + } \geq \frac { Q ^ { + } } { 3 } , \quad v ^ { + } o ^ { + } \geq \frac { V ^ { + } } { 3 } .
$$

Since $Q ^ { + } \leq 2 V ^ { + }$ , we have $V ^ { + } \geq \frac { Q ^ { + } } { 2 }$ . Therefore

$$
c ^ { + } = v ^ { + } o ^ { + } \operatorname { t a n h } ( q ^ { + } k ^ { + } ) \geq \frac { Q ^ { + } } { 6 } \operatorname { t a n h } \left( \frac { Q ^ { + } } { 3 } \right) .
$$

In the above, the RHS is increasing in $Q ^ { + }$ , and recall that $R = 1 2 ( D + 2 )$ by definition, it exceeds $D + 1$ at $Q ^ { + } = R$ . Therefore $Q ^ { + } < R$

Now consider $V ^ { + } . \mathrm { ~ I f ~ } Q ^ { + } \mathrm { ~ > ~ 1 ~ }$ , we have $\begin{array} { r } { q ^ { + } k ^ { + } \ge \frac { 1 } { 3 } , c ^ { + } \ge \frac { V ^ { + } } { 3 } } \end{array}$ tanh $\textstyle { \frac { 1 } { 3 } }$ . Since tanh $\textstyle { \frac { 1 } { 3 } } > 1 / 4$ , we have $V ^ { + } < 1 2 ( D + 1 ) < R$ . If instead $Q ^ { + } \leq 1$ , similarly, we can show that $\begin{array} { r } { \frac { \Delta V } { \Delta Q } < 3 . } \end{array}$ Since $V \leq 3 Q$ , we have

$$
V ^ { + } = V + \Delta V \leq 3 Q + 3 \Delta Q = 3 Q ^ { + } .
$$

Therefore

$$
V ^ { + } \leq 3 Q ^ { + } \leq 3 < R .
$$

Therefore, all defining inequalities are preserved after the update. Finally,

$$
q ^ { + 2 } , k ^ { + 2 } \leq Q ^ { + } < R , \quad v ^ { + 2 } , o ^ { + 2 } \leq V ^ { + } < R .
$$

This implies that every updated parameters have magnitude smaller than $\sqrt { R } < B$ , which concludes the proof. □

Finally, we show that the weights of other non-winning heads will stay small throughout the training.

Lemma B.5 (Remaining residual clock). Fix one feature, and let the designated winning head satisfy the invariant-region conditions of Lemma B.4 throughout the interval under consideration. Let $m _ { n } \triangleq \operatorname* { m i n } \{ q _ { n } , k _ { n } , v _ { n } , o _ { n } \}$ , and b <sup>≜</sup> min $\left\{ \mathrm { s e c h } ^ { 2 } ( R / 2 ) , { \frac { \mathrm { t a n h } ( R / 2 ) } { R / 2 } } \right\} > 0 .$ . Recall that the GD step is scaled by $s _ { n } = \eta p r _ { n } .$ . Then we have

$$
m _ { n + 1 } \geq m _ { n } ( 1 + b _ { \star } s _ { n } m _ { n } ^ { 2 } ) .
$$

Moreover, $\begin{array} { r } { m _ { n } ^ { - 2 } - m _ { n + 1 } ^ { - 2 } \geq \frac { 3 } { 4 } b _ { \star } s _ { n } } \end{array}$ . Therefore, for every $N > n _ { 0 }$ , we have

$$
\sum _ { n = n _ { 0 } } ^ { N - 1 } s _ { n } \leq \frac { 4 } { 3 b _ { \star } m _ { n _ { 0 } } ^ { 2 } } .
$$

In other words, the total remaining residual-weighted clock isfinite.

Proof. Denote $x _ { n } = q _ { k } k _ { n }$ . Because the winner remains in the invariant region as in Lemma B.4, we have $\begin{array} { r } { 0 \leq x _ { n } \leq \frac { Q _ { n } } { 2 } \leq \frac { R } { 2 } } \end{array}$ . Therefore, by the definition of $b _ { \star }$ , we have

$$
\operatorname { s e c h } ^ { 2 } ( x _ { n } ) \geq b _ { \star } , \quad \operatorname { t a n h } ( x _ { n } ) \geq b _ { \star } x _ { n } .
$$

Recall that

$$
\begin{array} { r } { \nabla c ( q , k , v , o ) = \left( \begin{array} { c } { o v k \mathrm { s e c h } ^ { 2 } ( q k ) } \\ { o v q \mathrm { s e c h } ^ { 2 } ( q k ) } \\ { o t a n \mathrm { h } ( q k ) } \\ { v \mathrm { t a n h } ( q k } \end{array} \right) . } \end{array}
$$

Since every factor is at least $m _ { n } .$ , each coordinate of the gradient is at least $b _ { \star } m _ { n } ^ { 3 }$ . Therefore, after one update, we have

$$
m _ { n + 1 } \geq m _ { n } + b _ { \star } s _ { n } m _ { n } ^ { 3 } = m _ { n } ( 1 + b _ { \star } s _ { n } m _ { n } ^ { 2 } ) .
$$

Now define $z _ { n } \triangleq b _ { \star } s _ { n } m _ { n } ^ { 2 }$ . The step-size bounds from Lemma B.3 imply $0 \leq z _ { n } \leq 1$ , which implies $m _ { n + 1 } ^ { - 2 } \leq m _ { n } ^ { - 2 } ( 1 + z _ { n } ) ^ { - 2 }$ , so

$$
m _ { n } ^ { - 2 } - m _ { n + 1 } ^ { - 2 } \geq m _ { n } ^ { - 2 } \left( 1 - \frac { 1 } { ( 1 + z _ { n } ) ^ { 2 } } \right) = b _ { \star } s _ { n } \frac { 2 + z _ { n } } { ( 1 + z _ { n } ) ^ { 2 } } .
$$

For $0 \leq z _ { n } \leq 1$ , we have $\begin{array} { r } { \frac { 2 + z _ { n } } { ( 1 + z _ { n } ) ^ { 2 } } \geq \frac { 3 } { 4 } } \end{array}$ , which implies $\begin{array} { r } { m _ { n } ^ { - 2 } - m _ { n + 1 } ^ { - 2 } \geq \frac { 3 } { 4 } b _ { \star } s _ { n } } \end{array}$ . Telescoping from $n = n _ { 0 }$ to $N - 1$ , we have

$$
\frac 3 4 b _ { \star } \sum _ { n = n _ { 0 } } ^ { N - 1 } s _ { n } \leq m _ { n _ { 0 } } ^ { - 2 } - m _ { N } ^ { - 2 } \leq m _ { n _ { 0 } } ^ { - 2 } \Rightarrow \sum _ { n = n _ { 0 } } ^ { N - 1 } s _ { n } \leq \frac { 4 } { 3 b _ { \star } m _ { n _ { 0 } } ^ { 2 } } ,
$$

as desired.

Remark 3. Lemma B.5 essentially shows that, once one head has established itself as the winner, there is not enough effective training time remaining for the losing heads to catch up. Combined with Lemmas B.1 and B.2, together with the stability result of Lemma B.4, this establishes that each feature becomes localized to a particular head. We formalize this localization property in the following result.

We are now ready to present the main theorem, which establishes that training induced feature localization.

Theorem 4.2 (restated). For heads $h \in \{ 1 , 2 , 3 \}$ and features $j \in \{ 0 , 1 , 2 \}$ , denote

$$
\theta _ { h j } = ( q _ { h j } , k _ { h j } , v _ { h j } , o _ { h j } ) , \quad c _ { h j } = o _ { h j } v _ { h j } \operatorname { t a n h } ( q _ { h j } k _ { h j } ) , \quad w _ { j } = \sum _ { h = 1 } ^ { 3 } c _ { h j } .
$$

Fix $a , b , c > 0 ,$ , set $\beta = \left( { \textstyle { \frac { a } { 2 } } } , b , c \right)$ and $p = \left( 1 , \frac { 1 } { 2 } , \frac { 1 } { 2 } \right)$ , and train $\begin{array} { r } { \mathcal { L } ( \theta ) = \frac { 1 } { 2 } \sum _ { j = 0 } ^ { 2 } p _ { j } ( w _ { j } - \beta _ { j } ) ^ { 2 } + \frac { a ^ { 2 } } { 8 } } \end{array}$ by full-batch GD from the initialization $\theta _ { h j } ^ { 0 } = \alpha G _ { h j }$ , where $G _ { h j } \stackrel { i . i . d . } { \sim } N ( 0 , I _ { 4 } )$ . Then there exists $\eta _ { \star } > 0$ , depending only on $a , b , c ,$ such that thefollowing holdsfor everyfixed step size $0 < \eta < \eta _ { \star }$ For almost every draw of G, all sufficiently small $\alpha > 0 _ { : }$ , we have:

1. Convergence: the parameter trajectory is bounded and converges to a finite limit with $w ^ { \infty } = \beta$ Each residual $r _ { j } ^ { n } = \beta _ { j } - w _ { j } ^ { n }$ is positive and non-increasing.

2. Localization: there are labels $h _ { j } = h _ { j } ( G )$ , independent and uniform on $\{ 1 , 2 , 3 \}$ , and a finite constant $K _ { G }$ , independent ofα, such that

$$
\operatorname* { s u p } _ { n \geq 0 } \| \theta _ { h j } ^ { n } \| _ { \infty } \leq K _ { G } \alpha , \quad \operatorname* { s u p } _ { n \geq 0 } | c _ { h j } ^ { n } | \leq K _ { G } \alpha ^ { 4 } { \mathrm { ~ f o r ~ } } h \neq h _ { j } .
$$

Consequently, $c _ { h _ { i } j } ^ { \infty } = \beta _ { j } + \mathcal { O } _ { G } ( \alpha ^ { 4 } )$

3. Finite checkpoint: define $\begin{array} { r } { N _ { \alpha } = \lceil \frac { \log ( 1 / \alpha ) } { \eta \alpha ^ { 2 } } \rceil } \end{array}$ , for every fixed $M > 0$ and sufficiently small $\alpha ,$ , we have

$$
\operatorname* { s u p } _ { n \geq N _ { \alpha } } \operatorname* { m a x } _ { j } r _ { j } ^ { n } \leq \alpha ^ { M } .
$$

Proof. Let $D = \operatorname* { m a x } \{ 1 , a / 2 , b , c \} , R = 1 2 ( D + 2 ) , B = 1 + \sqrt { R } ,$ and choose

$$
\eta _ { \star } = \operatorname * { m i n } \left\{ \frac { 1 } { 2 ( D + 1 ) B ^ { 2 } } , \frac { 1 } { ( D + 1 ) L } , \frac { 1 } { 3 6 B ^ { 6 } } \right\} , \quad L = 4 ( 2 B ) ^ { 4 } + 3 ( 2 B ) ^ { 2 } + 1 .
$$

Fix $0 < \eta \le \eta _ { \star }$ <sub>⋆</sub>. Because the objective separates across features, we first fix $j$ and suppress its index. Write

$$
\beta = \beta _ { j } , \quad p = p _ { j } , \quad c ( \theta ) = o v \operatorname { t a n h } ( q k ) ,
$$

$$
w ^ { n } = \sum _ { h = 1 } ^ { 3 } c ( \theta _ { h } ^ { n } ) , \quad r ^ { n } = \beta - w ^ { n } , \quad s _ { n } = \eta p r ^ { n } .
$$

The exact GD update is

$$
\theta _ { h } ^ { n + 1 } = \theta _ { h } ^ { n } + s _ { n } \nabla c ( \theta _ { h } ^ { n } ) .\tag{4}
$$

Fix a Gaussian draw for which the conclusions of Lemma B.1 hold and the three auxiliary escape times are finite and distinct, and note that this event has probability one. Let

$$
h _ { \star } = \arg \operatorname* { m i n } _ { h } T ( G _ { h } ) , \quad T _ { \star } = T ( G _ { h _ { \star } } ) , \quad T _ { ( 2 ) } = \operatorname* { m i n } _ { h \neq h _ { \star } } T ( G _ { h } ) .
$$

We will organize the proof in the following steps.

Step 1: Enter the invariant region. Choose $S _ { + }$ such that $T _ { \star } < S _ { + } < T _ { ( 2 ) }$ , and define

$$
b _ { \star } = \operatorname* { m i n } \left\{ \mathrm { s e c h } ^ { 2 } ( R / 2 ) , \frac { \operatorname { t a n h } ( R / 2 ) } { R / 2 } \right\} > 0 .
$$

Choose $A _ { 0 } < \infty$ sufficiently large such that

$$
\frac { 4 } { 3 b _ { \star } A _ { 0 } ^ { 2 } } < \frac { S _ { + } - T _ { \star } } { 4 } .\tag{5}
$$

By Lemma B.1, the winning auxiliary trajectory eventually has positive product, its factor magnitudes diverge, and their squared ratios converge to one. Therefore, we can choose $S _ { 0 } < T _ { \star }$ such that at $S _ { 0 } .$ , its smallest factor magnitude exceeds $2 A _ { 0 }$ and its largest-to-smallest squared-factor ratio is less than $\frac { 3 } { 2 }$

Introduce the normalized parameters and clock

$$
u _ { h } ^ { n } = \frac { \theta _ { h } ^ { n } } { \alpha } , \quad \zeta _ { n } = \alpha ^ { 2 } \sum _ { k < n } s _ { k } , \quad \delta _ { n } = \alpha ^ { 2 } s _ { n } .
$$

Then Eq. 4 becomes

$$
u _ { h } ^ { n + 1 } = u _ { h } ^ { n } + \delta _ { n } F _ { \alpha } ( u _ { h } ^ { n } ) , \mathrm { ~ w h e r e ~ } F _ { \alpha } ( u ) = \alpha ^ { - 3 } \nabla c ( \alpha u ) .\tag{6}
$$

Initially, $w ^ { 0 } = O _ { G } ( \alpha ^ { 4 } )$ Therefore, for sufficiently small $\alpha ,$ all parameters lie in $\lceil - B , B \rceil ^ { 4 }$ and $0 < r ^ { 0 } \le D + 1$ . Until the first crossing of $S _ { 0 }$ , a stopping argument using Lemmas B.2, B.3 gives

$$
0 < r ^ { n + 1 } \leq r ^ { n } \leq D + 1 , \quad 0 < \delta _ { n } \leq \alpha ^ { 2 } \eta p ( D + 1 ) ,\tag{7}
$$

and $\theta _ { h } ^ { n } = O _ { G } ( \alpha )$ for every head. Indeed, Lemma B.3 excludes residual failure, while Lemma B.2, including its candidate-step estimate, excludes a first exit from the parameter box.

The clock crosses $S _ { 0 }$ in finite time. Otherwise, Lemma B.2 would imply $w ^ { n } = O _ { G } ( \alpha ^ { 4 } )$ for every $n ,$ and therefore $r ^ { n } \geq \beta / 2$ for sufficiently small $\alpha$ . This would force $\delta _ { n } \geq \alpha ^ { 2 } \eta p \beta / 2 > 0 .$ , contradicting $\zeta _ { n } < S _ { 0 }$ for all $n .$

Let $n _ { 0 }$ be the first crossing index, Lemma B.2 gives

$$
\zeta _ { n _ { 0 } } = S _ { 0 } + \mathcal { O } ( \alpha ^ { 2 } ) , \quad \frac { \theta _ { h _ { \star } } ^ { n _ { 0 } } } { \alpha } \to u _ { h _ { \star } } ( S _ { 0 } ) .\tag{8}
$$

After a fixed coordinate-wise sign change, we may treat the winning factors as positive. This change preserves c and the GD dynamics because the quartet has positive product. For sufficiently small $\alpha ,$ we have

$$
m _ { n _ { 0 } } \triangleq \operatorname* { m i n } \{ q ^ { n _ { 0 } } , k ^ { n _ { 0 } } , v ^ { n _ { 0 } } , o ^ { n _ { 0 } } \} \geq \alpha A _ { 0 } ,\tag{9}
$$

and the squared-factor ratio is at most two.

Consequently, writing

$$
Q = q ^ { 2 } + k ^ { 2 } , \quad V = v ^ { 2 } + o ^ { 2 } , \quad d = q ^ { 2 } - k ^ { 2 } , \quad e = v ^ { 2 } - o ^ { 2 } ,
$$

we have

$$
| d | \leq Q / 3 , | e | \leq V / 3 , Q \leq 2 V , V \leq 2 Q .
$$

Since $Q , V = \mathcal { O } _ { G } ( \alpha ^ { 2 } )$ , the winner belongs to the invariant region of Lemma B.4.

Step 2: permanent exclusion of the losing heads. We prove by induction that, for every $n \geq n _ { 0 }$ the winner remains in the invariant region $\zeta _ { n } < S _ { + }$ , and the losing heads satisfy $\| \theta _ { h } ^ { n } \| _ { \infty } \leq K _ { + } \alpha$ for $h \neq h _ { \star }$ for a constant $K _ { + }$ independent of n and sufficiently small α. All parameters will consequently remain in the box required by Lemma B.3.

These properties hold at $n _ { 0 }$ . Suppose that they hold through iteration n. Lemma B.3 gives

$$
0 < r ^ { n + 1 } \leq r ^ { n } , \quad w ^ { n + 1 } < \beta .\tag{10}
$$

Applying Lemma B.5 to the controlled starting states $n _ { 0 }$ , , n and using Eq. 9, we have

$$
\xi _ { n + 1 } = \zeta _ { n _ { 0 } } + \alpha ^ { 2 } \sum _ { k = n _ { 0 } } ^ { n } s _ { k } \le \zeta _ { n _ { 0 } } + \frac { 4 \alpha ^ { 2 } } { 3 b _ { \star } m _ { n _ { 0 } } ^ { 2 } } \le S _ { 0 } + \mathcal O _ { G } ( \alpha ^ { 2 } ) + \frac { 4 } { 3 b _ { \star } A _ { 0 } ^ { 2 } } < S _ { + } .
$$

Crucially, this bound does NOT assume that the candidate terminal state already belongs to the invariant region.

The losing auxiliary trajectories are bounded through $S _ { + }$ since $S _ { + } < T _ { ( 2 ) }$ . Applying Lemma B.2 to these two heads alone, on the common residual-dependent mesh, gives

$$
\| \theta _ { h } ^ { n + 1 } \| _ { \infty } \le K _ { + } \alpha , \quad h \neq h _ { \star } .
$$

Using the fact that | tanh $x | \leq | x |$ , we have

$$
\begin{array} { r } { | c ( \theta _ { h } ^ { n + 1 } ) | \leq | o _ { h } ^ { n + 1 } v _ { h } ^ { n + 1 } q _ { h } ^ { n + 1 } k _ { h } ^ { n + 1 } | \leq K _ { + } ^ { 4 } \alpha ^ { 4 } . } \end{array}
$$

For sufficiently small α, the sum of the losing coefficients is therefore at least −1. Together with Eq. 10, Lemma B.4 implies that the winner remains in its invariant region and returns to the original parameter box. This finishes the induction.

Combining these estimates with the pre-entry bounds, and enlarging a constant if necessary, gives

$$
\operatorname* { s u p } _ { n \geq 0 } \| \theta _ { h } ^ { n } \| _ { \infty } \leq K _ { G } \alpha , \quad \operatorname* { s u p } _ { n \geq 0 } | c ( \theta _ { h } ^ { n } ) | \leq K _ { G } \alpha ^ { 4 } , \quad h \neq h _ { \star } .
$$

Lemma B.3 also establishes positivity and monotonicity of the residual through training.

Step 3: Parameter convergence and exact fitting. For each sufficiently small $\alpha ,$ Lemma B.5 gives

$$
\sum _ { n = n _ { 0 } } ^ { \infty } s _ { n } \leq \frac { 4 } { 3 b _ { \star } m _ { n _ { 0 } } ^ { 2 } } < \infty .
$$

Since $s _ { n } = \eta p r ^ { n }$ , it follows that

$$
\sum _ { n = n _ { 0 } } ^ { \infty } r ^ { n } < \infty , \quad r ^ { n } \to 0 .
$$

All parameter quartets remain in a compact box, so their gradients are uniformly bounded. Eq. 4, and the preceding summability bound imply

$$
\sum _ { n = n _ { 0 } } ^ { \infty } \| \theta _ { h } ^ { n + 1 } - \theta _ { h } ^ { n } \| _ { \infty } < \infty .
$$

Therefore, every parameter quartet converges to a finite limit. By continuity and the above, we have $w ^ { \infty } = \beta ,$ , and therefore

$$
c ( \theta _ { h _ { \star } } ^ { \infty } ) = \beta - \sum _ { h \neq h _ { \star } } c ( \theta _ { h } ^ { \infty } ) = \beta + \mathcal { O } _ { G } ( \alpha ^ { 4 } ) .\tag{11}
$$

Step 4: Finite-checkpoint bound. Let $\tau \triangleq \operatorname* { m i n } \{ n : w ^ { n } \geq \beta / 2 \}$ . This index is finite by convergence. For sufficiently small $\alpha ,$ it occurs after entry into the invariant region, since $w ^ { n _ { 0 } } = \mathsf { \bar { O } } _ { G } ( \alpha ^ { 4 } )$ From Step 2, we might further choose α sufficiently small that

$$
\sum _ { h \ne h _ { \star } } | c ( \theta _ { h } ^ { n } ) | \le \beta / 4 \quad \mathrm { ~ f o r ~ e v e r y ~ } n .
$$

Since $w ^ { n }$ is non-decreasing, we have

$$
c ( \theta _ { h _ { \star } } ^ { n } ) \geq \beta / 4 , \quad n \geq \tau .
$$

For the positive winning factors, the invariant-region bounds implies

$$
c = v o \operatorname { t a n h } ( q k ) \leq v o q k \leq \frac { V Q } { 4 } \leq \frac { R Q } { 4 } .
$$

Therefore, the above implies $Q \geq \beta / R$ . Moreover

$$
o ^ { 2 } \ge V / 3 \ge Q / 6 , \quad q k \ge Q / 3 ,
$$

Consequently

$$
\| \nabla c ( \theta _ { h _ { \star } } ^ { n } ) \| _ { 2 } ^ { 2 } \geq o ^ { 2 } \operatorname { t a n h } ^ { 2 } ( q k ) \geq \frac { \beta } { 6 R } \operatorname { t a n h } ^ { 2 } \left( \frac { \beta } { 3 R } \right) \triangleq m _ { \beta } > 0 , \quad n \geq \tau .\tag{12}
$$

Lemma B.3 gives

$$
w ^ { n + 1 } - w ^ { n } \geq \frac { s _ { n } } { 2 } \| \nabla c ( \theta _ { h _ { \star } } ^ { n } ) \| _ { 2 } ^ { 2 } \geq \frac { \eta p m _ { \beta } } { 2 } r ^ { n } ,
$$

and therefore

$$
r ^ { n + 1 } \leq \left( 1 - \frac { \eta p m _ { \beta } } { 2 } \right) r ^ { n } , \quad n \geq \tau .\tag{13}
$$

The contraction factor is in [0, 1): comparison with the no-overshoot bound $w ^ { n + 1 } - w ^ { n } \leq r ^ { n } / 2$ gives $\eta p m _ { \beta } / 2 \leq 1 / 2$

Before τ, we have $r ^ { n } > \beta / 2$ . The clock bound implies

$$
S _ { + } > \zeta _ { \tau } = \alpha ^ { 2 } \eta p \sum _ { n < \tau } r ^ { n } \ge \frac { 1 } { 2 } \alpha ^ { 2 } \eta p \beta \tau ,
$$

which implies $\begin{array} { r } { \tau \le \frac { 2 S _ { + } } { \eta p \beta \alpha ^ { 2 } } } \end{array}$ . It follows that $\begin{array} { r } { N _ { \alpha } = \lceil \frac { \log ( 1 / \alpha ) } { \eta \alpha ^ { 2 } } \rceil \geq 2 \tau } \end{array}$ for sufficiently small $\alpha .$ . Iterating Eq. 13, using $r ^ { \tau } \leq \beta / 2$ , and $1 - x \leq e ^ { - x }$ gives

$$
r ^ { N _ { \alpha } } \leq r ^ { \tau } \exp \left( - \frac { \eta p m _ { \beta } } { 2 } ( N _ { \alpha } - \tau ) \right) \leq \beta \exp \left( - \frac { p m _ { \beta } } { 4 \alpha ^ { 2 } } \log ( 1 / \alpha ) \right) .
$$

For every fixed $M \ > \ 0$ , the final expression is at most $\alpha ^ { M }$ for sufficiently small α. Residual monotonicity then gives

$$
\operatorname* { s u p } _ { n \geq N _ { \alpha } } r ^ { n } \leq \alpha ^ { M } .
$$

Finally, apply the argument to all three features, taking the minimum of their initialization thresholds and the maximum of their constants. Lemma B.1 identifies each winner as

$$
h _ { j } = \arg \operatorname* { m i n } _ { h } T ( G _ { h j } ) .
$$

The escape times are i.i.d. and atom-less across heads, so each winner is unique almost surely and uniformly distributed. The initialization blocks are independent across features, so the winner labels are independent. This concludes the proof. □

## B.2 ADDITIONAL RESULTS AND OMITTED PROOFS FOR SECTION 4.4

We now present the detailed proof for our main result, which establish a potential explanation for the emergence of CEL under training dynamics.

Theorem 4.3 (restated). Under the assumptions of Theorem $4 . 2 ,$ , let $t = a / 2 ,$ , and define the event

$$
E = \{ h _ { 0 } \notin \{ h _ { 1 } , h _ { 2 } \} \} ,
$$

where $h _ { j }$ is the head that wins the feature $j \in \{ 0 , 1 , 2 \}$ . Then $\textstyle \operatorname* { P r } ( E ) = { \frac { 4 } { 9 } }$ . Under such event, set $h _ { \star } = h _ { 0 } ,$ , and delete this head without changing any other parameters

$$
f _ { - h _ { \star } } ^ { n } = f ^ { n } - f _ { h _ { \star } } ^ { n } .
$$

For almost every initialization draw G in $E ,$ all sufficiently small $\alpha > 0 ,$ , and every $n \geq N _ { \alpha } ,$

$$
G _ { A } ^ { n } \triangleq R _ { A } ( f ^ { n } ) - R _ { A } ( f _ { - h _ { \star } } ^ { n } ) = t ^ { 2 } + \mathcal { O } _ { G } ( \alpha ^ { 4 } ) > 0 ,
$$

$$
D _ { B } ^ { n } \triangleq R _ { B } ( f _ { - h _ { \star } } ^ { n } ) - R _ { B } ( f ^ { n } ) = 3 t ^ { 2 } + \mathcal { O } _ { G } ( \alpha ^ { 4 } ) > 0 ,
$$

where the error bounds are uniform over $n \geq N _ { \alpha }$ . Consequently

$$
\operatorname* { l i m i n f } _ { \alpha \to 0 ^ { + } } \operatorname* { P r } _ { G } \left( \exists h : R _ { A } ( f ^ { N _ { \alpha } } ) - R _ { A } ( f _ { - h } ^ { N _ { \alpha } } ) \frac { t ^ { 2 } } { 2 } , R _ { B } ( f _ { - h } ^ { N _ { \alpha } } ) - R _ { B } ( f ^ { N _ { \alpha } } ) \geq \frac { 3 t ^ { 2 } } { 2 } \right) \geq \frac { 4 } { 9 } .
$$

Proof. By Theorem 4.2, the head labels $h _ { 0 } , h _ { 1 } , h _ { 2 }$ are independent and uniform on {1, 2, 3}. Conditional on $h _ { 0 } ,$ each $h _ { 1 } , h _ { 2 }$ avoid $h _ { 0 }$ with probability $\textstyle { \frac { 2 } { 3 } }$ . Therefore, $\textstyle \operatorname* { P r } ( E ) = \left( { \frac { 2 } { 3 } } \right) ^ { 2 } = { \frac { 4 } { 9 } }$

Fix a draw G for which Theorem 4.2 holds. At an iterate $n \geq N _ { \alpha } .$ , suppress the superscript $n ,$ and write

$$
e _ { j } = \beta _ { j } - w _ { j } , \quad z = \sum _ { h \neq h _ { \star } } c _ { h 0 } , \quad u _ { 1 } = c _ { h _ { \star } 1 } , \quad u _ { 2 } = c _ { h _ { \star } , 2 } .
$$

Theorem 4.2, with $M = 4$ , gives

$$
\operatorname* { m a x } _ { j } | e _ { j } | \leq \alpha ^ { 4 } .
$$

Moreover, every head other than $h _ { \star }$ loses feature 0, and $h _ { \star }$ loses features 1 and 2 on $E .$ . Hence, for a finite $K _ { G }$

$$
| z | \leq 2 K _ { G } \alpha ^ { 4 } , \quad | u _ { 1 } | , | u _ { 2 } | \leq K _ { G } \alpha ^ { 4 } .\tag{14}
$$

All these bounds hold uniformly for $n \geq N _ { \alpha }$ . The intact aggregate coefficients are

$$
w = ( t - e _ { 0 } , b - e _ { 1 } , c - e _ { 2 } ) .
$$

After deleting $h _ { \star }$ , they become

$$
w _ { - h } = ( z , b - e _ { 1 } - u _ { 1 } , c - e _ { 2 } - u _ { 2 } ) .
$$

Using $R _ { A } ( w ) = w _ { 0 } ^ { 2 } + ( w _ { 1 } - b ) ^ { 2 }$ , we obtain $G _ { A } = t ^ { 2 } + \mathcal { O } _ { G } ( \alpha ^ { 4 } )$ . Similarly, we can show that $D _ { B } = 3 t ^ { 2 } + \mathcal { O } _ { G } ( \overset { . } { \alpha ^ { 4 } } )$ . The uniform bounds in $\operatorname { E q }$ . 14 and on $e _ { j }$ make both remainder estimates uniform over all $n \geq N _ { \alpha }$ . Because $t > 0 ,$ , both quantities are positive for sufficiently small $\alpha ,$ and the second quantity measures the damage of head ablation on task $B .$

For the probability conclusion, let $A _ { \alpha }$ denote the event inside the display probability in the statement. The equations above imply that, for almost every $G \in E$ , the event $A _ { \alpha }$ holds for all sufficiently small α. Thus, along any sequence $\alpha _ { k }  0 ^ { + }$ , we have

$$
\operatorname* { l i m } _ { k \to \infty } \operatorname* { i n f } 1 _ { A _ { \alpha _ { k } } } ( G ) \geq 1 _ { E } ( G ) \quad { \mathrm { a l m o s t ~ s u r e l y } } .
$$

Fatou’s lemma gives

$$
\operatorname* { l i m i n f } _ { k \to \infty } \operatorname* { P r } ( A _ { \alpha _ { k } } ) \geq \mathbb { E } \left[ \operatorname* { l i m i n f } _ { k \to \infty } 1 _ { A _ { \alpha _ { k } } } \right] \geq \operatorname* { P r } ( E ) = { \frac { 4 } { 9 } } .
$$

Since the sequence was arbitrary, the lower-limit probability follows.