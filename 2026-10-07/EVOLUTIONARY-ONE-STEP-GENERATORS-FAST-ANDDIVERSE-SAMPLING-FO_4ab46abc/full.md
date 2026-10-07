# EVOLUTIONARY ONE-STEP GENERATORS: FAST ANDDIVERSE SAMPLING FOR DISCRETE DESIGN

Marcus Vukojevic<sup>1</sup> Erik Nielsen<sup>1</sup> Veronica Lachi<sup>2</sup> Andrea Passerini<sup>1</sup> Giovanni Iacca<sup>1</sup>

<sup>1</sup>University of Trento, Italy

<sup>2</sup>UiT The Arctic University of Norway, Tromsø, Norway

<sup>1</sup>{firstname.lastname}@unitn.it

<sup>2</sup>veronica.lachi@uit.no

## ABSTRACT

Several discrete design tasks, such as molecular discovery, require diverse collections of useful candidates at low computational cost. High validity alone does not guarantee a useful candidate library: repeatedly generating the same valid structures leaves few distinct alternatives. Training for both feasibility and diversity is challenging because many relevant criteria can only be evaluated after hard decoding. To address this challenge, we propose EGO (Evolutionary Generators with One-step inference), a framework for training compact generators directly on discrete outputs. The method combines distribution matching with structural constraints and optional diversity or history-dependent rewards, using antithetic low-rank evolution strategies without requiring criterion-specific differentiable surrogates. Once trained, the generator produces the entire graph in a single neuralnetwork evaluation. On molecular generation benchmarks, our compact generator achieves over 50× the valid-and-unique yield per estimated dense operation compared to recent one-step flow-map baselines while retaining high chemical validity. In scaffold completion, EGO achieves an observed 44.3× speedup over MoLeR in generation to SMILES and produces approximately 10× as many filter-passing proposals within matched time budgets for generation and screening. Beyond chemistry, EGO produces 1.54× as many distinct held-out elite architectures as relaxed gradient training on NAS-Bench-101. The low generation cost may enable real-time candidate generation across discrete design tasks, supporting interactive exploration of constrained design spaces and rapid construction of candidate sets for downstream evaluation.

## 1 INTRODUCTION

Many discrete generative applications ultimately require not an unlimited stream of plausible samples, but afinite collection ofuseful candidates (Renz et al., 2024). Molecular discovery constructs libraries of compounds for downstream screening, while neural architecture search considers a limited set of computational graphs for further evaluation (Brown et al., 2019). In such settings, distributional fidelity and validity remain important, but they are not sufficient: invalid outputs waste evaluation budget, repeated outputs add little to the candidate pool, and expensive sampling limits how many proposals can be considered. A useful generator must therefore produce many distinct qualifying candidates (Xie et al., 2023) under a finite proposal and compute budget.

This is challenging for discrete generative models because many relevant criteria are naturally defined only on the final discrete object. Connectivity, graph validity, exact identity, duplicate detection, and history-dependent rewards are straightforward to evaluate on generated structures, while optimizing them through categorical sampling typically requires task-specific relaxations or surrogate gradients. Continuous relaxations and straight-through estimators instead optimize differentiable surrogates (Bengio et al., 2013; Jang et al., 2017), which can be highly effective when the surrogate faithfully represents the desired hard-output criterion. Score-function estimators provide direct access to hard rewards, but introduce different optimization trade-offs.

One-step generators are particularly appealing in this setting because, once trained, they map latent noise and optional conditions directly to a complete proposal in a single learned-network evaluation (Cao and Kipf, 2018; Roos et al., 2026; Deng et al., 2026). The central training question is: How can this one-step proposal distribution be shaped directly by criteria evaluated on its final hard outputs?

We address this problem by proposing EGO, an objective-direct framework for training one-step discrete generators. Complete hard samples are scored using distribution-matching terms, structural constraints, and optional set- or history-dependent rewards, without requiring these criteria to be differentiable. We optimize the resulting objective with antithetic evolution strategies (ES) using shared sampling noise and low-rank parameter perturbations.

Building on established ES methods (Salimans et al., 2017; Sarkar et al., 2025), we contribute a training interface for reusable one-step discrete generators, a finite-budget formulation, and an empirical study of when the resulting training–inference trade-off is favorable. Population evaluation increases training cost, but after training each proposal requires only one evaluation of the central generator. The EGO workflow is shown in Figure 1.

Our experiments show that this trade-off can be favorable in structured candidate-generation tasks. On molecular benchmarks, EGO achieves high distinct valid yield at low estimated inference cost: relative to CFM-ECLD (Roos et al., 2026), its valid-and-unique yield per estimated dense operation is approximately 50× higher on QM9 (Ramakrishnan et al., 2014) and 200× higher on ZINC250K (Irwin et al., 2012), with a measurable distributional-fidelity trade-off. Under a matched 30-minute active-training budget on ZINC250K, EGO reaches a mean strict valid–unique–novel yield of 77.9%, compared with 5.8% for relaxed training, despite the gradient methods completing approximately six times as many optimizer updates. We further explore the potential of EGO as a reusable candidate generator through pilot experiments. On 100 held-out scaffold queries, EGO produces an average of 547.3 filter-passing proposals per query, compared with 54.5 for MoLeR (Maziarz et al., 2022), under matched generation and screening budgets averaging 6.08 seconds per query, giving a 10.05× gain in the number of filter-passing proposals. In a complementary evaluation at 100000 proposals, EGO returns 20245 distinct filter-passing completions versus 16684 for MoLeR, a 1.21× increase with an observed speedup of 44.3×. On NAS-Bench-101 (Ying et al., 2019), it produces 1.54× as many distinct held-out elite architectures as relaxed training at the same proposal budget, although relaxed training achieves higher mean accuracy and closer feature-distribution matching.

Together, these results suggest that objective-direct evolutionary training is most useful when hard feasibility and candidate diversity are poorly represented by a differentiable surrogate, and when low-cost repeated sampling is important. To summarize, our main contributions are:

1. We formulate discrete generation as a finite-budget candidate-generation problem, showing that distinct successful yield depends both on success probability and on how probability mass is distributed across successful identities.

2. We introduce an objective-direct training framework for one-step discrete generators that supports distributional, structural, and history-dependent criteria evaluated on complete hard samples.

3. We instantiate this interface with antithetic low-rank evolution strategies and evaluate it against diverse graph-generation baselines, demonstrating its effectiveness for distinct candidate generation with one-step inference while documenting the associated trade-offs.

The appendix further examines this trade-off through variations in optimization recipe and model scale, as well as controlled non-molecular experiments. It also discusses related work and provides full implementation and evaluation details for reproducibility.

## 2 FINITE-BUDGET DISCRETE GENERATION

We consider generation over a discrete space $x ,$ , with attributed graphs as our primary setting. When generation is conditional, let $c \in { \mathcal { C } }$ denote the condition. A stochastic generator with parameters θ induces the distribution

$$
q _ { \theta } ( x \mid c ) = \operatorname* { P r } _ { \theta } ( X = x \mid c ) ,\tag{1}
$$

![](images/a6fbe0d78a97bca9e651fea458ec9f711159c9a045d6b49ccbd1de75f0a54161.jpg)  
Figure 1: EGO. One evaluation of $f _ { \theta }$ turns $( \mathbf { z } , n )$ into a batch of complete graphs (1). The criterion is read off those hard graphs: distribution match, feasibility, and rarity (2). The update compares opposite probes $\theta \pm \varepsilon$ scored under identical noise (3). Adam carries the step back to the generator along the loop below. After training, each proposal costs one network evaluation (4).

for $x \in \mathcal { X }$ . For unconditional generation, we omit c. The following finite-budget analysis applies to independent complete samples regardless of the generator’s sampling procedure. Section 3 introduces EGO, which generates each proposal in one network evaluation.

Many applications evaluate a generator from a fixed number of samples rather than a single output. Let $\dot { X } _ { 1 } , \dots , X _ { N }$ be N independent samples for a given condition c. We denote by $a ( x , c ) \in \{ 0 , 1 \}$ whether x satisfies the evaluation criterion. To identify repeated outputs, let $\kappa _ { c } : \mathcal { X } \stackrel { } { \to } \mathcal { K } _ { c }$ map each output to the identity used for duplicate detection. The number of distinct successful outputs is $U _ { N } = | \hat { \{ \kappa _ { c } ( X _ { i } ) : a ( X _ { i } , \bar { c } ) = 1 , i \in [ \hat { N } ] \} } |$ |, with corresponding yield $Y _ { N } = U _ { N } / N$ . What counts as a successful output depends on the task. For example, it may be a valid, connected, training-novel molecule, an architecture in a target performance set, or a scaffold completion passing a set of chemical filters. Distinct yield complements, rather than replaces, measures of distributional fidelity, coverage, and average quality.

At finite N, not only the probability of success but also how this probability is distributed across successful identities affects the resulting candidate set. Let $\mathcal { A } _ { c } = \{ \kappa _ { c } ( x ) : x \in \mathcal { X } , a ( x , c ) = 1 \}$ and define $p _ { \theta } ( u \mid c ) = \mathrm { P r } _ { \theta } ( a ( X , c ) = 1 , \bar { \kappa _ { c } } ( X ) = u \mid c )$ for $u \in A _ { c }$

Proposition 1 (Finite-budget distinct yield). For N independent samples from $q _ { \theta } ( \cdot \mid c )$

$$
\mathbb { E } [ U _ { N } \mid c ] = \sum _ { u \in \mathcal { A } _ { c } } \left[ 1 - ( 1 - p _ { \theta } ( u \mid c ) ) ^ { N } \right] .\tag{2}
$$

Let $\begin{array} { r } { \alpha _ { \theta } ( c ) = \sum _ { u \in \mathcal { A } _ { c } } p _ { \theta } ( u \mid c ) } \end{array}$ denote the probability of producing a successful output, and let $\begin{array} { r } { \Gamma _ { \theta } ( c ) = \sum _ { u \in \mathcal { A } _ { c } } p _ { \theta } ( \bar { u } \mid c ) ^ { 2 } } \end{array}$ . Then

$$
\operatorname* { m a x } \biggl \{ 0 , N \alpha _ { \theta } ( c ) - \binom { N } { 2 } \Gamma _ { \theta } ( c ) \biggr \} \leq \mathbb { E } [ U _ { N } \mid c ] \leq N \alpha _ { \theta } ( c ) .
$$

The proposition separates two effects of the learned proposal distribution. The quantity $\alpha _ { \theta } ( c )$ measures how much probability mass is assigned to successful outputs, whereas $\Gamma _ { \theta } ( c )$ is the probability that two independent proposals produce the same successful identity. Thus, a high probability of success alone does not imply a large distinct candidate set at finite budget. The proof is given in Appendix A. This can also be seen directly from Eq. (2). Let $f _ { N } ( p ) \stackrel { - } { = } 1 - ( 1 \stackrel { - } { - } p ) ^ { N }$ , which is concave for $N \geq 2$ . Hence, $f _ { N } ( p _ { 1 } ) + f _ { N } ( \hat { p _ { 2 } } ) \leq 2 f _ { N } \hat { ( } ( p _ { 1 } + p _ { 2 } ) / 2 )$ . Replacing the probabilities of two successful identities by their average, while keeping their total probability fixed, therefore cannot decrease the expected number of distinct outputs.

Training on discrete samples. The criterion used for training need not coincide with the metric used for final evaluation. At iteration t, let ${ \cal S } _ { B } = \{ ( X _ { i } , c _ { i } ) \} _ { i = 1 } ^ { B }$ , with $X _ { i } \sim q _ { \theta } ( \cdot \mid c _ { i } )$ , be a batch of B generated discrete objects. Let R be an optional reference batch and let $H _ { t }$ denote an optional state accumulated before iteration t. We consider criteria of the form

$$
J _ { t } ( \theta \mid H _ { t } ) = \operatorname { \mathbb { E } } [ \mathcal { L } ( S _ { B } , R , H _ { t } ) \mid H _ { t } ] ,\tag{3}
$$

where the expectation is over the sampled conditions, generated objects, and reference data used at iteration $t . \mathrm { ~ A ~ }$ training criterion may contain some or all of the following components:

$$
\mathcal { L } ( S _ { B } , R , H _ { t } ) = D ( S _ { B } , R ) + \sum _ { j } \lambda _ { j } C _ { j } ( S _ { B } ) - \sum _ { r } \beta _ { r } \mathcal { R } _ { r } ( S _ { B } , H _ { t } ) .\tag{4}
$$

Here, D measures the discrepancy between generated and reference data, $C _ { j }$ are structural or feasibility penalties, and $\textstyle { \mathcal { R } } _ { r }$ are optional task-, set-, or history-dependent rewards. The state $H _ { t }$ is held fixed while the samples used for the current update are evaluated. All terms in Eq. (4) are evaluated on discrete generated objects and need not be differentiable.

Hard objectives and gradient approximations. Discrete sampling prevents ordinary backpropagation through the sampled choices, but still permits gradient-based optimization of the expected hard objective. A continuous relaxation replaces discrete variables with soft counterparts and optimizes a differentiable proxy. We refer to this relaxation-based approach as Soft. The straight-through estimator (STE) instead retains the hard forward computation while substituting derivatives during backpropagation (Bengio et al., 2013; Jang et al., 2017). Soft generally changes the optimized objective, whereas STE’s substituted derivatives do not generally correspond to the gradient of Eq. (3). Score-function estimators such as REINFORCE (Williams, 1992) provide another route: they combine hard-sample objective values with gradients of sampling log probabilities, without differentiating through the sampled objects. Our experiments (4) compare this approach with EGO and Appendix D.2.3 provides the full training and evaluation protocol. Evolution strategies construct updates from objective evaluations under parameter perturbations, without requiring backpropagation through the generator. We use this approach to incorporate structural checks and task-, set-, or history-dependent rewards through a common interface based on hard-objective evaluations. The additional forward evaluations during training are part of the computational trade-off assessed in our experiments. Section 3 describes the evolutionary, straight-through, and relaxed training procedures.

## 3 OBJECTIVE-DIRECT TRAINING OF ONE-STEP GENERATORS

In Section 2, we separated the criterion evaluated on discrete samples from the signal used to optimize the generator. We now connect the three components of our method: a generator that emits a complete discrete object in one pass, an objective evaluated on that hard object, and an evolutionary update that uses objective values without differentiating through sampling. The generator determines inference cost; the training interface determines which criteria shape its proposal distribution.

One-step discrete generation. Let $z \sim \mathcal { N } ( 0 , I )$ be a latent variable, c the condition, and $\xi$ the randomness used for discrete sampling. We parameterize the distribution in Eq. (1) as

$$
\hat { G } = d ( f _ { \theta } ( z , c ) , \xi ; c ) .\tag{5}
$$

where $f _ { \theta }$ predicts all output logits in parallel and d applies hard sampling together with task-specific structural masks. A proposal therefore requires one evaluation of $f _ { \boldsymbol { \theta } } ;$ discrete sampling, graph construction, and subsequent validity checks do not require additional generator evaluations.

Instantiations. For unconditional molecular generation, the condition $c = n$ specifies the number of heavy atoms. The generator concatenates z with an encoding of n and processes it with a multilayer perceptron: $h _ { 0 } = [ z ; \mathrm { o n e h o t } ( n ) ]$ ] and $h _ { \ell } = \mathrm { R e L U } ( W _ { \ell } h _ { \ell - 1 } + b _ { \ell } )$ . Two output heads predict atom-state logits $\varbar L ^ { A }$ and bond logits $L ^ { B }$ . Hard atom and bond states are sampled as

$$
\widehat { a } _ { i } = \arg \operatorname* { m a x } _ { k } ( L _ { i k } ^ { A } + \xi _ { i k } ^ { A } ) , \qquad \widehat { b } _ { i j } = \arg \operatorname* { m a x } _ { k } ( L _ { i j , k } ^ { B } + \xi _ { i j , k } ^ { B } ) , \quad i < j ,\tag{6}
$$

with independent standard Gumbel noise. Only the first n nodes are active, and bonds involving inactive nodes are masked out. This enforces the requested graph size and an undirected graph without self-loops, but does not enforce chemical validity or connectedness. During distributional generation, n is sampled from the training size distribution. For scaffold completion, we instead use a separate one-step decoder that preserves the supplied core atoms and bonds and predicts the added atoms, their parent assignments, and bond orders. Each added atom selects a parent from the scaffold or from an earlier added atom, which guarantees attachment to a connected nonempty core. Additional predicted edges can introduce cycles. The scaffold-only setting supplies only the scaffold, with the number of added atoms sampled from the training data. For NAS-Bench-101, an unconditional MLP predicts in parallel logits for the 21 possible forward edges on a seven-node graph and categorical operations for its five internal nodes. Restricting edges to the strict upper triangle ensures acyclicity. Input and output nodes are fixed, and nodes outside input-to-output paths are pruned. Full architectural details for all three generators are given in Appendix C.

Objectives on discrete samples. The objective in Eq. (4) must reward realistic samples without losing the hard constraints that determine whether they are usable. We therefore combine distribution matching with penalties evaluated on sampled discrete structures. Generated and reference objects are compared through task-specific feature maps $\phi _ { r }$ . For a generated batch $\{ \widehat { G } _ { i } \} _ { i = } ^ { B }$ and reference batch $\{ G _ { j } ^ { \mathrm { r e f } } \} _ { j = 1 } ^ { B _ { + } }$ , let $x _ { i } ^ { ( r ) } = \phi _ { r } ( \widehat { G } _ { i } ) / s _ { \prime }$ and $y _ { j } ^ { ( r ) } = \phi _ { r } ( G _ { j } ^ { \mathrm { r e f } } ) / s _ { r }$ . For each feature family, we use the empirical Energy discrepancy (Szekely and Rizzo, 2013)´

$$
\begin{array} { l } { { \displaystyle \mathcal E _ { r } = \frac { 2 } { B B _ { + } } \sum _ { i , j } \rho ( x _ { i } ^ { ( r ) } , y _ { j } ^ { ( r ) } ) - \frac { 1 } { B ( B - 1 ) } \sum _ { i \neq i ^ { \prime } } \rho ( x _ { i } ^ { ( r ) } , x _ { i ^ { \prime } } ^ { ( r ) } ) } } \\ { { \displaystyle \qquad - \frac { 1 } { B _ { + } ( B _ { + } - 1 ) } \sum _ { j \neq j ^ { \prime } } \rho ( y _ { j } ^ { ( r ) } , y _ { j ^ { \prime } } ^ { ( r ) } ) , } } \end{array}\tag{7}
$$

where $\rho ( a , b ) = \sqrt { \| a - b \| _ { 2 } ^ { 2 } + 1 0 ^ { - 1 2 } }$ . The first term brings generated features close to the reference data, while the second prevents concentration of the generated feature distribution. Feature maps describe task-relevant graph properties, such as node and edge attribute frequencies, degree statistics, local patterns, and spectral or connectivity descriptors. Their choice determines which aspects of the reference distribution are matched. We combine these discrepancies with graph-constraint penalties:

$$
\mathcal { L } _ { \mathrm { c o r e } } = \sum _ { r } \mathcal { E } _ { r } + \sum _ { j } \lambda _ { j } \frac { 1 } { B } \sum _ { i = 1 } ^ { B } C _ { j } ( \widehat { G } _ { i } ) .\tag{8}
$$

Each $C _ { j }$ measures violations of a task-specific structural or attribute constraint. The feature maps, penalties, and their weights are specified for each application in Appendix C. Because EGO scores hard samples directly, its distribution-matching objective is modular. For scaffold completion, our experiments favored Drift (Deng et al., 2026), a local attraction–repulsion objective, over the global Energy discrepancy in Eq. (7) (see Appendix E for details).

Some useful criteria concern a candidate set rather than an isolated sample. Equation (4) therefore also allows rewards that depend on previously generated samples. History-dependent rewards can encourage exploration of graph structures that have been sampled less frequently. Let $h ( G )$ be a task-specific graph signature and let $V _ { t } ( h ( G ) )$ count its visits before iteration t. We define the batch rarity reward as

$$
\mathcal { R } _ { t } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \frac { \mathbf { 1 } _ { \mathrm { f e a s i b l e } } ( \widehat { G } _ { i } ) } { \sqrt { 1 + V _ { t } ( h ( \widehat { G } _ { i } ) ) } } .\tag{9}
$$

The history is held fixed while the current population is evaluated and updated only afterwards. The feasibility indicator refers to the constraints used during training, which need not coincide with the final evaluation criterion. Likewise, the archive signature h need not coincide with the identity map $\kappa _ { c }$ used for evaluation. Additional downstream filters can therefore assess properties that were not explicitly optimized during training.

Evolutionary optimization. Evolution strategies optimize the generator from objective evaluations without differentiating through hard sampling, feature extraction, or structural checks (Salimans et al., 2017). At each update, we evaluate antithetic parameter perturbations using the same latent variables, conditions, reference data, and Gumbel noise. Sharing the sampling noise couples the categorical decisions and removes disagreements caused solely by resampling. Lemma 1, stated and proved in Appendix A.2, quantifies this stability by bounding the probability of different discrete outputs in terms of the changes in their logits.

For small logit changes, a shared-noise comparison changes a category only near an argmax boundary; independent noise introduces disagreement even between identical parameter vectors. The lemma therefore justifies coupling sampling noise within each antithetic pair. It does not imply that the rank-shaped ES direction is an unbiased gradient estimator. Appendix A gives the proof.

To reduce the memory required by the population, we use rank-one perturbations. For a weight matrix $W _ { \ell }$ , direction m is $\epsilon _ { m \ell } = u _ { m \ell } v _ { m \ell } ^ { \top }$ , with independent Gaussian vectors $u _ { m \ell }$ and $v _ { m \ell }$ . Each direction defines the antithetic pair

$$
\theta _ { m , \ell } ^ { \pm } = \theta _ { \ell } \pm \sigma _ { \ell } \epsilon _ { m \ell } , \qquad m = 1 , \ldots , M ,\tag{10}
$$

with analogous vector perturbations for biases. This follows scalable neural ES with antithetic sampling (Salimans et al., 2017) and low-rank population evaluation (Sarkar et al., 2025). Let $s _ { m } ^ { + }$

and $s _ { m } ^ { - }$ denote the shaped fitness values assigned to the two members of pair m. The search direction for parameter block ℓ is

$$
g _ { \ell } = \frac { 1 } { 2 M \sigma _ { \ell } } \sum _ { m = 1 } ^ { M } ( s _ { m } ^ { + } - s _ { m } ^ { - } ) \epsilon _ { m \ell } .\tag{11}
$$

We use Adam (Kingma and Ba, 2015) to update the central parameters from $g .$ Fitness shaping is experiment-dependent: the molecular experiments use centered ranks, with the rarity term ranked separately when present, whereas the controlled NAS uses centered negative loss without an explicit rarity reward. We use Eq. (11) as a search direction and do not interpret it as an unbiased gradient estimator. The population of networks is required only during training. After training, generation uses the central network in Eq. (1). Exact perturbation scales, population sizes, optimizer settings, feature normalization, and implementation details are reported in Appendix C.

## 4 EXPERIMENTS

We organize the evaluation into three questions. RQ1 asks what validity–efficiency trade-off one-step and multi-step molecular generators offer. RQ2 asks whether one-step generation supports useful downstream candidate libraries. RQ3 asks whether the approach extends beyond molecular graphs.

Baselines and controlled comparisons. We compare EGO with three gradient-based baselines sharing the generator architecture and training split within each controlled experiment. All methods are evaluated on hard discrete outputs. STE (Bengio et al., 2013) uses hard features in the forward pass and their soft counterparts’ derivatives: $\phi _ { \mathrm { S T E } } = \phi _ { \mathrm { s o f t } } + \mathrm { s t o p g r a d } ( \phi _ { \mathrm { h a r d } } - \phi _ { \mathrm { s o f t } } )$ . Soft (Jang et al., 2017) optimizes the relaxed objective. REINFORCE with leave-one-out (RLOO) (Williams, 1992) estimates gradients from hard rewards without differentiating through discrete outputs. The molecular size sweep includes $\mathrm { E G O _ { 0 } }$ , which omits the rarity reward in Eq. (9) from EGO to assess its contribution to candidate yield. A separate QM9 control compares EGO and RLOO using the same hard reward and fixed feature scales at matched hard-reward evaluation budgets, excluding rarity and rank shaping to isolate the update recipes. See Appendices C and D.2.3 for experiment-specific objectives, budgets, and checkpoint rules. Molecular comparisons also include MolGAN (Cao and Kipf, 2018), MoFlow (Zang and Wang, 2020), CFM-CSD/ECLD (Roos et al., 2026), Kernel-gradient drifting (Esteban-Casadevall et al., 2026), CatFlow (Eijkelboom et al., 2024), DeFoG (Qin et al., 2025), MELD (Seo et al., 2025), G2PT (Chen et al., 2025), and SMILES LSTM (Brown et al., 2019). The training-time comparison also includes Cometh (Siraudin et al., 2025). Additional baselines and protocols are given in Appendix D.1.

Evaluation protocol. Full validity requires chemical sanitization of the complete molecular graph. VU measures distinct valid molecules per attempt, while VUN additionally excludes training molecules; strict variants also require connectivity. These yields differ from uniqueness conditioned on validity. Estimated efficiency $\mathrm { \bar { \it E } _ { V U } }$ normalizes VU yield by reconstructed dense-projection work, including all neural evaluations; it excludes other computation and is not a runtime measurement. Scaffold uniqueness is measured per query. Matched-time budgets cover generation and screening, excluding final SMILES export for EGO (Appendix E). NAS measures distinct held-out elite yield alongside validity and average accuracy. Tables report mean ± sample SD across runs. Appendix C details metric definitions, seed counts, checkpoint selection, and computational accounting.

## 4.1 RQ1: EGO REACHES HIGH MOLECULAR YIELD AT LOW TRAINING AND ESTIMATED INFERENCE COST

Figure 2 places EGO in the high-yield, high-efficiency region of each molecular benchmark. With one network evaluation per proposal, it attains VU yields of 78.21%, 90.66%, and 77.61% on QM9, ZINC250K, and GuacaMol. Relative to CFM-ECLD, EGO achieves approximately 50× and 200× the valid-and-unique yield per estimated dense-projection operation on QM9 and ZINC250K, respectively, while VU is lower by 7.24 and 2.37 percentage points. These ratios compare reconstructed dense projection work under the source protocols rather than matched-hardware runtime. The efficiency gain trades off distributional fidelity. FCD, Frechet ChemNet Distance Preuer et al. (2018),´ is 3.53 versus 2.16 on QM9, 30.36 versus 20.0 on ZINC250K for EGO and CFM-ECLD, and on GuacaMol 34.62 ± 1.58. Thus, EGO favors a high yield of distinct valid candidates over closer matching of the reference distribution. For an extensive and detailed comparison, see Appendix D.

![](images/cc4e953d76d2d7e8a1bb1c64c1e076a570520c9b0c69ce8fda97c77115ba9384.jpg)  
Figure 2: Distinct valid yield versus estimated inference efficiency. Axes are valid-and-unique yield (VU) and VU per $1 0 ^ { 1 2 }$ estimated dense operations (E , log scale). The GuacaMol CFM-ECLD marker is a local 3/84-epoch feasibility checkpoint, not a converged baseline, and is excluded from ratio and ranking claims. Source protocols and sampling budgets differ, and $E _ { \mathrm { V U } }$ is not wall-clock throughput. Full accounting and methods references appear in Appendix D.

Figure 3a tests whether the yield advantage persists under controlled training budgets. In 30 minutes on ZINC250K, EGO reaches 77.9% Strict VUN, compared with 53.7% without rarity, 5.8% for Soft, and 29.5% for STE. This gives 13.47× Soft’s yield and a 9.29× advantage without rarity, although the gradient methods complete ≈ 6× as many updates (Appendix D.3.4). Across the six molecular size settings (two datasets, three size caps each), $\mathrm { E G O _ { 0 } }$ also outperforms STE, while rarity improves yield throughout (Appendix D.1.1). In a separate five-seed QM9 experiment matching architecture, reward, and fixed feature scales, EGO reaches 38.4 Strict VUN versus 12.0 for RLOO at 11.52 million hard-reward graph evaluations. Without rarity or rank shaping, this supports a yield advantage for EGO’s update recipe at this budget beyond differences in reward construction (Appendix D.2.4).

Figure 3b compares the computational resources required by EGO with those of baselines other than CFM. EGO records 0.30, 2.10, and 14.85 optimizer-update hours on QM9, ZINC250K, and GuacaMol, respectively. DeFoG reported times are 6.5, 14, and 141 hours, while Cometh reports 6 on QM9 and 96 hours on GuacaMol. CFM reports approximately 48 and 144 hours on QM9 and ZINC250K, while its configured GuacaMol schedule projects to 2.2–2.8k optimizer-update hours from our local three-epoch pilot (see Appendix D.4.3). Hardware, GPU count, and timing scope differ, so these values provide resource context rather than measured speedups. Even with this caveat, EGO lies at the low-time end on every dataset.

Overall, the controlled experiments support higher candidate yield under limited training time, while the external comparison shows higher yield per estimated inference work; both gains come with a measurable distributional-fidelity trade-off. Appendix D.1.1 reports the full six-setting molecular-size sweep, extended STE and RLOO comparisons, rarity ablations, and the complete training-time audit.

## 4.2 RQ2: EGO PRODUCES MORE FILTER-PASSING PROPOSALS AT MATCHED TIME

Scaffold completion tests whether low generation cost allows more candidates to be screened within a common time budget. We compare with MoLeR trained from scratch on the same ZINC250K scaffold split, using the same 100 held-out queries and screening rules (details in Appendix E). Both models run on the same GPU after warmup. For each query, the time MoLeR needs to generate and screen 256 proposals (a single batch) defines the deadline for both methods. EGO screens graph outputs directly and the final SMILES export is outside this interval. Screening requires a valid, connected completion containing the scaffold, QED (Quantitative Estimate of Drug-likeness) $\geq 0 . 7$ SA (Synthetic Accessibility) ≤ 4, and no additional PAINS (pan-assay interference compounds) alert types relative to the core. At a mean deadline of 6.08 seconds per query, EGO screens 1892.7 proposals on average, compared with 256 for MoLeR. Of these, 547.3 versus 54.5 pass the filters, a 10.05× increase at matched time (Table 1).

(a) Training-budget comparison  
![](images/fb318a0ff64cabdae57b4eff89d1c2b7246560818da962f0bf2d495c7cf16131.jpg)

(b) Training-resource context  
![](images/69875ea60f4ad20708f7767403b94a6691d80050a46626d17dd809bc6dc6ea76.jpg)  
Figure 3: Training efficiency and resource context. (a) Strict VUN under a shared active 30- minute ZINC250K wall-clock budget; RLOO versus EGO on QM9. (b) Training time (log scale). EGO reports optimizer-update time; DeFoG, Cometh, and CFM report elapsed training time on QM9/ZINC250K. <sup>‡</sup>Cometh GuacaMol used two A100s; <sup>†</sup>CFM GuacaMol: 84-epoch constantthroughput projection from a local three-epoch pilot. See Appendix D.4.

Table 1: Scaffold screening at matched time. Mean ± sample SD over three runs on 100 queries: three EGO training seeds and three sampling seeds from one MoLeR checkpoint. Fractions divide passing by screened counts within each run. Query wins compare mean passing counts across runs (13 ties). Both methods share each query’s deadline. Protocol: Appendix E.3.
<table><tr><td>Metric</td><td>EGO</td><td>MoLeR</td></tr><tr><td>Mean deadline/query (s)</td><td> $6 . 0 8 \pm 0 . 1 0$ </td><td> $6 . 0 8 \pm 0 . 1 0$ </td></tr><tr><td>Screened proposals/query Filter-passing proposals/query</td><td> $\mathbf { 1 8 9 2 . 7 \ : \pm { \ : 6 3 . 7 } }$   $\mathbf { 5 4 7 . 3 \ : \pm { \ : 1 5 . 9 } }$ </td><td> $2 5 6 . 0 \pm 0 . 0$   $5 4 . 5 \pm 0 . 4$ </td></tr><tr><td>Passing fraction (%) Scaffold coverage (/100)</td><td> ${ \bf 2 8 . 9 5 \pm 1 . 5 8 }$  79</td><td> $2 1 . 2 7 \pm 0 . 1 4$ </td></tr><tr><td>Query wins (/100)</td><td>73</td><td>85</td></tr><tr><td></td><td></td><td>14</td></tr></table>

![](images/2b43d9540b4b399a097a8ad539b0971dd2e9b8abc38c0227806a91a98a112f38.jpg)  
Figure 4: Highest-QED filterpassing completions for the prespecified scaffold 23 under a shared 5.13-second deadline. The supplied core is highlighted.

Averaging passing counts across the three replicates for each query, EGO returns more passing proposals on 73 of 100 scaffolds, with 13 ties and 14 MoLeR wins. MoLeR covers more queries and more often returns the higher-QED passing candidate (Appendix E.3). Figure 4 shows the selected completions for one query. In a complementary evaluation with 100,000 proposals, EGO produces 20245 distinct filter-passing completions versus 16684 for MoLeR, a 1.21× increase, while achieving a measured 44.3× speedup in generation including SMILES conversion (Appendix E.4).

## 4.3 RQ3: EGO PRODUCES MORE DISTINCT ELITE NEURAL ARCHITECTURES

NAS-Bench-101 (Ying et al., 2019) tests whether the candidate-library perspective extends to neural architectures. The task is to learn a one-step proposal distribution from an offline corpus of highperforming architectures and generate distinct candidates of similarly high quality beyond those used for training. EGO, STE, and Soft share the generator architecture, initialization within each seed, data split, sampling noise, and checkpoint selection rule. EGO evaluates Energy and a fixed structural-violation penalty on hard graphs. STE preserves the same numerical hard forward objective while using surrogate derivatives, and Soft optimizes their relaxed objective. This instantiation of EGO uses neither a rarity reward nor an explicit duplicate penalty, allowing us to assess candidate yield without these additional incentives. The Energy objective itself still encourages diversity among generated samples. All methods receive 15,000 training updates, matching update count rather than training work, with cumulative optimizer-update times of approximately 46 seconds for EGO and 34 seconds for STE and Soft (Appendix F.1).

Table 2: NAS-Bench-101 results with SD from 10,000 proposals per selected model. The elite cutoff is the 90th percentile of benchmark validation accuracy in the architecture training partition. Mean test accuracy is averaged over valid proposals, including repetitions. Feature MMD compares the feature distributions of generated architectures and the elite target; lower values indicate a closer match. Protocol details appear in Appendix F.1.
<table><tr><td>Metric</td><td>EGO (ES)</td><td>STE</td><td>Soft</td></tr><tr><td>Validity (%) ↑</td><td> $7 2 . 3 5 \pm 1 . 7 6$ </td><td> $7 7 . 4 5 \pm 6 . 2 7$ </td><td> ${ \bf 8 9 . 5 6 \pm 0 . 8 2 }$ </td></tr><tr><td>Valid-and-unique yield (%) ↑</td><td> ${ \bf 5 5 . 5 1 \pm 0 . 2 9 }$ </td><td> $3 3 . 2 5 \pm 6 . 0 2$ </td><td> $2 5 . 1 2 \pm 4 . 0 0$ </td></tr><tr><td>Held-out elite yield (%) ↑</td><td> ${ \bf 2 . 2 2 \pm 0 . 0 8 }$ </td><td> $1 . 3 8 \pm 0 . 2 9$ </td><td> $1 . 4 4 \pm 0 . 1 9$ </td></tr><tr><td>Mean test accuracy (%) ↑</td><td> $9 1 . 8 5 \pm 0 . 0 6$ </td><td> $9 2 . 3 8 \pm 0 . 2 9$ </td><td> ${ \bf 9 2 . 7 0 \pm 0 . 1 0 }$ </td></tr><tr><td>Feature MMD  $( \times 1 0 ^ { - 3 } ) .$  →</td><td> $2 7 . 7 0 \pm 2 . 7 0$ </td><td> $1 0 . 9 0 \pm 5 . 2 0$ </td><td> ${ \bf 3 . 1 0 \pm 0 . 7 0 }$ </td></tr></table>

Here, elite denotes an architecture whose benchmark validation accuracy is at or above the 90th percentile in the architecture training partition. The evaluation asks how many different elite architectures, excluded from both training and checkpoint selection, each model returns within 10000 proposals. The outcome of interest is the number of distinct qualifying alternatives available for further consideration. Table 2 shows that EGO produces the largest collection of distinct held-out elite architectures. From 10,000 proposals, it returns $2 2 1 . 7 \pm 7 . { \bar { 5 } }$ such candidates, compared with $1 3 8 . 0 \pm 2 8 . 9$ for STE and 143.7 ± 18.6 for Soft across the three seeds, providing 1.61× and 1.54× as many alternatives on average, respectively. The advantage also appears in the broader collection of valid architectures: over half of the proposal budget yields distinct valid candidates with EGO, compared with one third for STE and one quarter for Soft. Thus, the candidate-yield advantage observed in molecular generation also appears in this architecture-search setting.

This broader library comes with a trade-off. Soft achieves higher validity and mean test accuracy, and lower feature MMD (Maximum Mean Discrepancy), but repeats architectures more often. The additional distinct candidates from EGO offer more options for downstream comparison and screening beyond the accuracy cutoff. The advantage is task-dependent: on binarized MNIST, Soft achieves better distribution matching and broader predicted digit coverage than EGO (Appendix F.2).

## 5 CONCLUSION

We introduced EGO, a framework for training one-step discrete generators using objectives evaluated directly on hard outputs. Across molecular generation, scaffold completion, and NAS-Bench-101, the approach produces large sets of distinct qualifying candidates while retaining one-network-evaluation inference. These results support finite-budget candidate yield as a useful complement to validity, average quality, and distributional fidelity. The results also highlight the central trade-off of the approach: additional objective evaluations are spent during training to shape a cheap, reusable proposal distribution at deployment. This trade-off is most useful when hard feasibility, diversity, or downstream utility are not well captured by differentiable surrogates. Future work could improve training efficiency, scale the approach to larger generators and more complex discrete structures, and incorporate richer downstream feedback into the training objective.

Limitations. The benefits of EGO are task-dependent. Higher candidate yield can come at the expense of distributional fidelity, average proposal quality, or coverage across conditions, and relaxed training remains preferable in some settings. Population evaluation increases training cost, and scalability to substantially larger generators remains to be established. The learned distribution also depends on the chosen features, penalties, and rewards. Finally, the molecular screening results rely on heuristic filters and do not establish experimental usefulness.

## AI USE STATEMENT

Generative AI tools were used to assist with drafting portions of the manuscript and revising the text for clarity, organization, and readability. The authors take full responsibility for the final manuscript, including its originality, technical claims, reported results, and references.

## ETHICS STATEMENT

This work investigates discrete generative modeling using existing benchmark datasets and does not involve human participant studies or clinical experiments. The molecular experiments evaluate computationally generated candidates using structural criteria and heuristic property filters; these measures do not establish safety, biological efficacy, or experimental synthesizability. As with other molecular design methods, faster candidate generation could support beneficial discovery but could also be misused to propose harmful compounds. Any downstream use therefore requires application specific assessment, appropriate safety screening, and expert oversight before experimental testing. The learned distributions depend on the benchmark data and chosen objectives, and the reported improvements should not be interpreted as guarantees of usefulness outside the evaluated settings.

## REPRODUCIBILITY STATEMENT

Section 3 describes the proposed framework, and Appendix A provides the assumptions, derivations, and proofs of the theoretical results. Appendix C describes the generator architectures, objective functions, training procedures, and evaluation conventions. Appendices D, E, and F document task-specific data preparation, experimental protocols, checkpoint selection, sampling budgets, and additional results. Appendix D.5 explains the computational accounting and timing conventions, including the distinction between estimated dense operation counts and measured runtime. The exper imental appendices also state limitations of the available run records and qualify comparisons with published reference results. A documented code repository, including instructions for reproducing all experiments presented in this paper, will be provided as part of the supplementary materials. The repository will also be made publicly available when a public version of the paper is released.

## REFERENCES

Yoshua Bengio, Nicholas Leonard, and Aaron C. Courville. Estimating or propagating gradients´ through stochastic neurons for conditional computation. CoRR, abs/1308.3432, 2013.

G. Richard Bickerton, Gaia V. Paolini, Jer´ emy Besnard, Sorel Muresan, and Andrew L. Hopkins.´ Quantifying the chemical beauty of drugs. Nature Chemistry, 4(2):90–98, 2012. doi: 10.1038/ nchem.1243.

Nathan Brown, Marco Fiscato, Marwin H. S. Segler, and Alain C. Vaucher. Guacamol: Benchmarking models for de novo molecular design. J. Chem. Inf. Model., 59(3):1096–1108, 2019.

Nicola De Cao and Thomas Kipf. Molgan: An implicit generative model for small molecular graphs. CoRR, abs/1805.11973, 2018.

Xiaohui Chen, Yinkai Wang, Jiaxing He, Yuanqi Du, Soha Hassoun, Xiaolin Xu, and Liping Liu. Graph generative pre-trained transformer. In ICML, volume 267 of Proceedings of Machine Learning Research. PMLR / OpenReview.net, 2025.

Mingyang Deng, He Li, Tianhong Li, Yilun Du, and Kaiming He. Generative modeling via drifting. CoRR, abs/2602.04770, 2026.

Floor Eijkelboom, Grigory Bartosh, Christian Andersson Naesseth, Max Welling, and Jan-Willem van de Meent. Variational flow matching for graph generation. In NeurIPS, 2024.

Peter Ertl and Ansgar Schuffenhauer. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. J. Cheminformatics, 1:8, 2009.

Maria Esteban-Casadevall, Jorge Carrasco Pollo, Max Welling, Jan-Willem van de Meent, Erik J. Bekkers, and Floor Eijkelboom. Kernel-gradient drifting models. CoRR, abs/2605.10727, 2026.

Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Scholkopf, and Alexander J. Smola.¨ A kernel two-sample test. J. Mach. Learn. Res., 13:723–773, 2012.

John J. Irwin, Teague Sterling, Michael M. Mysinger, Erin S. Bolstad, and Ryan G. Coleman. ZINC: A free tool to discover chemistry for biology. J. Chem. Inf. Model., 52(7):1757–1768, 2012.

Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. In ICLR (Poster). OpenReview.net, 2017.

Jaehyeong Jo, Seul Lee, and Sung Ju Hwang. Score-based generative modeling of graphs via the system of stochastic differential equations. In ICML, volume 162 of Proceedings of Machine Learning Research, pages 10362–10383. PMLR, 2022.

Jaehyeong Jo, Dongki Kim, and Sung Ju Hwang. Graph generation with diffusion mixture. In ICML, volume 235 of Proceedings ofMachine Learning Research, pages 22371–22405. PMLR / OpenReview.net, 2024.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR (Poster), 2015.

Lingkai Kong, Jiaming Cui, Haotian Sun, Yuchen Zhuang, B. Aditya Prakash, and Chao Zhang. Autoregressive diffusion model for graph generation. In ICML, volume 202 of Proceedings of Machine Learning Research, pages 17391–17408. PMLR, 2023.

Yujia Li, Kevin Swersky, and Richard S. Zemel. Generative moment matching networks. In ICML, volume 37 of JMLR Workshop and Conference Proceedings, pages 1718–1727. JMLR.org, 2015.

Krzysztof Maziarz, Henry Richard Jackson-Flux, Pashmina Cameron, Finton Sirockin, Nadine Schneider, Nikolaus Stiefl, Marwin H. S. Segler, and Marc Brockschmidt. Learning to extend molecular scaffolds with structural motifs. In ICLR. OpenReview.net, 2022.

Mingue Park, Jisung Hwang, Seungwoo Yoo, Kyeongmin Yeo, and Minhyuk Sung. Pairflow: Closed-form source-target coupling for few-step generation in discrete flow models. CoRR, abs/2512.20063, 2025.

Kristina Preuer, Philipp Renz, Thomas Unterthiner, Sepp Hochreiter, and Gunter Klambauer. Fr¨ echet´ chemnet distance: A metric for generative models for molecules in drug discovery. J. Chem. Inf. Model., 58(9):1736–1741, 2018.

Yiming Qin, Manuel Madeira, Dorina Thanou, and Pascal Frossard. Defog: Discrete flow matching for graph generation. In ICML, volume 267 of Proceedings ofMachine Learning Research. PMLR / OpenReview.net, 2025.

Raghunathan Ramakrishnan, Pavlo O. Dral, Matthias Rupp, and O. Anatole von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. Scientific Data, 1:140022, 2014. doi: 10.1038/sdata.2014.22.

Philipp Renz, Sohvi Luukkonen, and Gunter Klambauer. Diverse hits in de novo molecule design:¨ Diversity-based comparison of goal-directed generators. J. Chem. Inf. Model., 64(15):5756–5761, 2024.

Daan Roos, Oscar Davis, Floor Eijkelboom, Michael M. Bronstein, Max Welling, .Ismail .Ilkan Ceylan, Luca Ambrogioni, and Jan-Willem van de Meent. Categorical flow maps. CoRR, abs/2602.12233, 2026.

Tim Salimans, Jonathan Ho, Xi Chen, and Ilya Sutskever. Evolution strategies as a scalable alternative to reinforcement learning. CoRR, abs/1703.03864, 2017.

Bidipta Sarkar, Mattie Fellows, Juan Agustin Duque, Alistair Letcher, Antonio Leon Villares, Anya´ Sims, Dylan Cope, Jarek Liesen, Lukas Seier, Theo Wolf, Uljad Berdica, Alexander David Goldie, Aaron C. Courville, Karin Sevegnani, Shimon Whiteson, and Jakob Nicolaus Foerster. Evolution strategies at the hyperscale. CoRR, abs/2511.16652, 2025.

Hyunjin Seo, Taewon Kim, Sihyun Yu, and Sungsoo Ahn. Learning flexible forward trajectories for masked molecular diffusion. CoRR, abs/2505.16790, 2025.

Antoine Siraudin, Fragkiskos D. Malliaros, and Christopher Morris. Cometh: A continuous-time discrete-state graph diffusion model. Trans. Mach. Learn. Res., 2025, 2025.

Gabor J. Sz ´ ekely and Maria L. Rizzo. Energy statistics: A class of statistics based on distances.´ Journal ofStatistical Planning and Inference, 143(8):1249–1272, 2013.

Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard.´ Digress: Discrete denoising diffusion for graph generation. In ICLR. OpenReview.net, 2023.

Zhe Wang, Jiaxin Shi, Nicolas Heess, Arthur Gretton, and Michalis K. Titsias. Learning-order autoregressive models with application to molecular graph generation. In ICML, volume 267 of Proceedings of Machine Learning Research. PMLR / OpenReview.net, 2025.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach. Learn., 8:229–256, 1992.

Yutong Xie, Ziqiao Xu, Jiaqi Ma, and Qiaozhu Mei. How much space has been explored? measuring the chemical space covered by databases and machine-generated molecules. In ICLR. OpenReview.net, 2023.

Chris Ying, Aaron Klein, Eric Christiansen, Esteban Real, Kevin Murphy, and Frank Hutter. Nasbench-101: Towards reproducible neural architecture search. In ICML, volume 97 of Proceedings of Machine Learning Research, pages 7105–7114. PMLR, 2019.

Chengxi Zang and Fei Wang. Moflow: An invertible flow model for generating molecular graphs. In KDD, pages 617–626. ACM, 2020.

Lingxiao Zhao, Xueying Ding, and Leman Akoglu. Pard: Permutation-invariant autoregressive diffusion for graph generation. In NeurIPS, 2024.

## APPENDIX CONTENTS

A Theoretical Results 13   
A.1 Finite-Budget Distinct Yield . . 13   
A.2 Stability under Shared Gumbel Noise . .14   
B Background and Related Work 16   
B.1 One-Step Generation and Discrete Training 16   
B.2 Structured Generation and Applications . . 17   
C Model, Training, and Evaluation Details 18   
C.1 Molecular Generation . . 19   
C.2 Scaffold Completion . . . 23   
C.3 Neural Architecture Generation . 26   
C.4 Binarized MNIST . 27   
D RQ1: Molecular Generation 28   
D.1 Shared Evaluation Protocol . . 29   
D.2 QM9 . . 30   
D.3 ZINC250K . 38   
D.4 GuacaMol . . 45   
D.5 Inference Cost and Training Time . . 50   
E RQ2: Scaffold Completion 53   
E.1 Training and Scaffold Only Inference . . 53   
E.4 Screening and Candidate Yield . . 56   
E.5 Generation to SMILES Time . . 57   
E.3 Generation and Screening at Matched Time . . 54   
F RQ3: Beyond Molecular Generation 58   
F.1 NAS-Bench-101 . . . 58   
F.2 Binarized MNIST Control . . 61

## A THEORETICAL RESULTS

## A.1 FINITE-BUDGET DISTINCT YIELD

Proof of Proposition 1 Fix a condition c and write

$$
p _ { u } = p _ { \theta } ( u \mid c ) , \qquad u \in \mathcal { A } _ { c } .
$$

For each successful identity u, define

$$
I _ { u } = { \bf 1 } \{ u \in \{ \kappa _ { c } ( X _ { i } ) : a ( X _ { i } , c ) = 1 , i \in [ N ] \} \} .
$$

Then

$$
U _ { N } = \sum _ { u \in \mathcal { A } _ { c } } I _ { u } .
$$

Since the summands are nonnegative, linearity of expectation (or Tonelli’s theorem if $\boldsymbol { A } _ { c }$ is countably infinite) gives

$$
\mathbb { E } [ U _ { N } \mid c ] = \sum _ { u \in \mathcal { A } _ { c } } \operatorname* { P r } ( I _ { u } = 1 \mid c ) .
$$

For a fixed $u ,$ each proposal produces the correct identity u with probability $p _ { u }$ . Because the N proposals are independent, the probability that u does not occur in any of them is $( \mathrm { \bar { 1 } } - p _ { u } ) ^ { N }$ . Therefore

$$
\operatorname* { P r } ( I _ { u } = 1 \mid c ) = 1 - ( 1 - p _ { u } ) ^ { N } ,
$$

and hence

$$
\mathbb { E } [ U _ { N } \mid c ] = \sum _ { u \in \mathcal { A } _ { c } } \left[ 1 - ( 1 - p _ { u } ) ^ { N } \right] .
$$

For the upper bound, the union bound gives

$$
\begin{array} { r } { 1 - ( 1 - p _ { u } ) ^ { N } \leq N p _ { u } . } \end{array}
$$

Summing over successful identities yields

$$
\mathbb { E } [ U _ { N } \mid c ] \leq N \sum _ { u \in \mathcal { A } _ { c } } p _ { u } = N \alpha _ { \theta } ( c ) .
$$

For the lower bound, let

$$
E _ { i , u } = \{ a ( X _ { i } , c ) = 1 , \kappa _ { c } ( X _ { i } ) = u \} .
$$

Then $\operatorname* { P r } ( E _ { i , u } \mid c ) = p _ { u } .$ , and independence gives $\mathrm { P r } ( E _ { i , u } \cap E _ { j , u } \mid c ) = p _ { u } ^ { 2 }$ for $i \neq j$ . The second-order Bonferroni inequality therefore gives

$$
1 - ( 1 - p _ { u } ) ^ { N } = \mathrm { P r } \left( \bigcup _ { i = 1 } ^ { N } E _ { i , u } \Bigg | c \right) \geq N p _ { u } - \binom { N } { 2 } p _ { u } ^ { 2 } .
$$

Summing over u gives

$$
\mathbb { E } [ U _ { N } \mid c ] \ge N \sum _ { u \in \mathcal { A } _ { c } } p _ { u } - \binom { N } { 2 } \sum _ { u \in \mathcal { A } _ { c } } p _ { u } ^ { 2 } = N \alpha _ { \theta } ( c ) - \binom { N } { 2 } \Gamma _ { \theta } ( c ) .
$$

Since $U _ { N } \geq 0$ almost surely, its expectation is also nonnegative. Combining the two lower bounds $\mathrm { g i v e s }$

$$
\begin{array} { r } { \mathbb { E } [ U _ { N } \mid c ] \ge \operatorname* { m a x } \left\{ 0 , { N \alpha } _ { \theta } ( c ) - \binom { N } { 2 } \Gamma _ { \theta } ( c ) \right\} . } \end{array}
$$

Together with the upper bound, this proves the result.

## A.2 STABILITY UNDER SHARED GUMBEL NOISE

Evolutionary updates compare objective values obtained from antithetic parameter perturbations. When categorical samples are drawn independently for the two evaluations, differences between their outputs can arise from both the parameter perturbation and the sampling randomness. Sharing the Gumbel noise instead couples the categorical decisions while preserving the marginal sampling distribution of each perturbed generator. The following result quantifies the resulting stability. Throughout this analysis, the latent input and conditioning information are held fixed.

Lemma 1 (Stability under shared Gumbel noise). Let $\ell ( { \boldsymbol { \theta } } ) \in \mathbb { R } ^ { K }$ denote the logits at one categorical site. For a fixed perturbation $\Delta , d e f i n e$

$$
\theta ^ { \pm } = \theta \pm \Delta , \qquad \delta ( \Delta ) = \operatorname* { m a x } _ { \eta \in \{ - 1 , + 1 \} } \left\| \ell ( \theta + \eta \Delta ) - \ell ( \theta ) \right\| _ { \infty } .
$$

$L e t \xi \in \mathbb { R } ^ { K }$ have independent standard Gumbel entries, shared between the two evaluations, and define

$$
X ^ { \pm } = \arg \operatorname* { m a x } _ { k \in \{ 1 , \ldots , K \} } \left\{ \ell _ { k } ( \theta ^ { \pm } ) + \xi _ { k } \right\} .
$$

Then

$$
\operatorname* { P r } ( X ^ { + } \neq X ^ { - } \mid \Delta ) \leq \operatorname* { m i n } \left\{ 1 , e ^ { 2 \delta ( \Delta ) } - 1 \right\} .
$$

Proof. Write

$$
\ell ^ { 0 } = \ell ( \theta ) , \qquad \ell ^ { \pm } = \ell ( \theta \pm \Delta ) , \qquad \delta = \delta ( \Delta ) .
$$

By definition,

$$
\| \ell ^ { \pm } - \ell ^ { 0 } \| _ { \infty } \leq \delta .
$$

Using the same Gumbel vector ξ, define the central noisy scores

$$
\begin{array} { r } { S _ { k } = \ell _ { k } ^ { 0 } + \xi _ { k } , } \end{array}
$$

and let

$$
X ^ { 0 } = \arg \operatorname* { m a x } _ { k } S _ { k } .
$$

Because the Gumbel distribution is continuous, ties occur with probability zero. Let

$$
M = S _ { X ^ { 0 } } - \operatorname* { m a x } _ { j \neq X ^ { 0 } } S _ { j }
$$

be the gap between the largest and second-largest central noisy scores.

Suppose that $M > 2 \delta$ . For every $j \neq X ^ { 0 }$ and $q \in \{ - 1 , + 1 \}$

$$
\begin{array} { c } { { ( \ell _ { X ^ { 0 } } ^ { q } + \xi _ { X ^ { 0 } } ) - ( \ell _ { j } ^ { q } + \xi _ { j } ) = ( S _ { X ^ { 0 } } - S _ { j } ) + ( \ell _ { X ^ { 0 } } ^ { q } - \ell _ { X ^ { 0 } } ^ { 0 } ) - ( \ell _ { j } ^ { q } - \ell _ { j } ^ { 0 } ) } } \\ { { \geq M - 2 \delta > 0 . } } \end{array}
$$

Hence both perturbed evaluations select the same category $X ^ { 0 }$ . Therefore

$$
\{ X ^ { + } \neq X ^ { - } \} \subseteq \{ M \leq 2 \delta \} ,
$$

and thus

$$
\operatorname* { P r } ( X ^ { + } \neq X ^ { - } \mid \Delta ) \leq \operatorname* { P r } ( M \leq 2 \delta \mid \Delta ) .
$$

It remains to bound the probability of a small top-two gap under Gumbel-max sampling. Let

$$
\lambda _ { k } = e ^ { \ell _ { k } ^ { 0 } } , \qquad \Lambda = \sum _ { k = 1 } ^ { K } \lambda _ { k } .
$$

Using the exponential-race representation of Gumbel-max, write

$$
\xi _ { k } = - \log E _ { k } , \qquad E _ { k } \overset { \mathrm { i i d } } { \sim } \mathrm { E x p } ( 1 ) ,
$$

and define

$$
T _ { k } = \frac { E _ { k } } { \lambda _ { k } } .
$$

Then the $T _ { k }$ are independent exponential random variables with rates $\lambda _ { k } .$ , and maximizing $S _ { k }$ is equivalent to minimizing $T _ { k }$ . If $T _ { ( 1 ) }$ and $T _ { ( 2 ) }$ denote the smallest and second-smallest race times, respectively, then

$$
M = \log \frac { T _ { ( 2 ) } } { T _ { ( 1 ) } } .
$$

For any $t \geq 0 ,$ , condition on category i winning the race at time s. The event $M > t$ requires every competing race time to exceed $e ^ { t _ { s } }$ . Hence

$$
\begin{array} { l } { \displaystyle \operatorname* { P r } ( M > t , X ^ { 0 } = i ) = \int _ { 0 } ^ { \infty } \lambda _ { i } e ^ { - \lambda _ { i } s } \prod _ { j \not = i } e ^ { - \lambda _ { j } e ^ { t } s } d s } \\ { \displaystyle = \frac { \lambda _ { i } } { \lambda _ { i } + e ^ { t } ( \Lambda - \lambda _ { i } ) } . } \end{array}
$$

Since

$$
\lambda _ { i } + e ^ { t } ( \Lambda - \lambda _ { i } ) \leq e ^ { t } \Lambda ,
$$

we obtain

$$
\operatorname* { P r } ( M > t , X ^ { 0 } = i ) \geq e ^ { - t } { \frac { \lambda _ { i } } { \Lambda } } .
$$

Summing over $i \ \mathrm { g i }$ ves

$$
\operatorname* { P r } ( M > t ) \geq e ^ { - t } ,
$$

and therefore

$$
\operatorname* { P r } ( M \leq t ) \leq 1 - e ^ { - t } \leq e ^ { t } - 1 .
$$

Setting $t = 2 \delta$ yields

$$
\operatorname* { P r } ( X ^ { + } \neq X ^ { - } \mid \Delta ) \leq e ^ { 2 \delta } - 1 .
$$

Combining this with the trivial upper bound of one proves the claim.

Corollary 1 (Multiple categorical sites). Suppose an object is represented by D categorical sites, and let $\delta _ { s } ( \Delta )$ denote the corresponding logit bound at site s. If the Gumbel noise at each site is shared between the two perturbed evaluations, then

$$
\operatorname* { P r } ( X ^ { + } \neq X ^ { - } \mid \Delta ) \leq \operatorname* { m i n } \left\{ 1 , \sum _ { s = 1 } ^ { D } \left( e ^ { 2 \delta _ { s } ( \Delta ) } - 1 \right) \right\} .
$$

The same bound applies to final objects obtained by applying a common deterministic decoder to the sampled categorical vectors.

Proof. If the complete sampled vectors differ, then at least one categorical site differs. By the union bound and Lemma 1,

$$
\begin{array} { r l } { \displaystyle \operatorname* { P r } ( X ^ { + } \neq X ^ { - } \mid \Delta ) \leq \sum _ { s = 1 } ^ { D } \operatorname* { P r } ( X _ { s } ^ { + } \neq X _ { s } ^ { - } \mid \Delta ) } & { } \\ { \displaystyle \leq \sum _ { s = 1 } ^ { D } \left( e ^ { 2 \delta _ { s } ( \Delta ) } - 1 \right) . } \end{array}
$$

Combining this with the trivial upper bound of one gives the stated result.

If a common deterministic decoder is applied to both sampled vectors, different decoded objects can occur only if the underlying categorical vectors differ. The same bound therefore applies after decoding. □

For comparison, if the two evaluations instead use independent Gumbel noise and $\Delta = 0$ , then they are independent draws from the same categorical distribution

$$
\pi = \operatorname { s o f t m a x } ( \ell ( \theta ) ) .
$$

Consequently,

$$
\operatorname* { P r } ( X ^ { + } \neq X ^ { - } \mid \Delta = 0 ) = 1 - \sum _ { k = 1 } ^ { K } \pi _ { k } ^ { 2 } .
$$

Thus, independent sampling retains a nonzero source of disagreement even when the two parameter vectors are identical.

By contrast, under shared noise, the bound in Lemma 1 vanishes with the size of the logit perturbation. For small δ,

$$
e ^ { 2 \delta } - 1 = 2 \delta + O ( \delta ^ { 2 } ) .
$$

Hence, sufficiently small parameter perturbations can change a sampled category only when the shared noisy logits lie near an argmax boundary.

This result characterizes the stability of the sampled decisions. It does not establish variance reduction for every possible hard objective, nor does it imply that the resulting rank-shaped ES direction is an unbiased gradient estimator. Its role is instead to motivate paired evaluations that remove disagreements caused solely by resampling, making the comparison more directly reflect the effect of the parameter perturbation.

## B BACKGROUND AND RELATED WORK

## B.1 ONE-STEP GENERATION AND DISCRETE TRAINING

One-step generation. Direct generators learn a mapping from latent noise to a complete output. Once trained, they can generate a new proposal with a single network evaluation. MolGAN applies this idea to molecular graphs by jointly predicting node and edge distributions with a feedforward network, followed by categorical sampling (Cao and Kipf, 2018). More recently, Categorical Flow Maps learn continuous maps toward categorical outputs through self-distillation, allowing generation in one or a few steps (Roos et al., 2026). Their CSD and ECLD variants use different consistency objectives within the same framework. We include both as one-step references. Our work studies how to train a direct generator when the objective is evaluated on complete discrete samples. Here, one-step generation refers to the network evaluation. Decoding, validity checks, and conversion into usable outputs also contribute to the total generation cost.

Distribution matching without explicit likelihoods. A generator can learn by comparing generated and reference samples without computing the probability of each output. Generative Moment Matching Networks train feedforward generators by minimizing maximum mean discrepancy (MMD), which compares distributions through kernel mean embeddings (Li et al., 2015; Gretton et al., 2012). Energy statistics provide a related approach based on distances within and between two sets of samples (Szekely and Rizzo, 2013). Drifting Models use attraction and repulsion to guide the´ generated distribution toward the data during training, while retaining one-step inference (Deng et al., 2026). These approaches motivate our batch objectives. We compare generated and reference structures through features that describe their properties, and study Energy and Drift as two distinct objectives. The choice of features matters because different graph distributions can share the same feature distribution. We therefore assess feature matching together with validity, diversity, and the number of useful candidates produced.

Learning through discrete decisions. Hard categorical sampling prevents ordinary backpropagation through the sampled decision. Gumbel-Softmax replaces this decision with a differentiable relaxation, while straight-through estimators use hard decisions in the forward pass and approximate their derivatives in the backward pass (Bengio et al., 2013; Jang et al., 2017). Both approaches provide practical training signals, but improving a relaxed objective or following an approximate gradient may not improve the final discrete outputs. Score-function estimators offer another option. They use gradients of log probabilities to estimate how parameters affect the expected score, without differentiating through the sampled output itself. Their variance can make training difficult (Bengio et al., 2013). Learned reward models provide a further route. MolGAN combines adversarial training with a reward network that approximates external molecular scores and supplies gradients for property optimization (Cao and Kipf, 2018). Our ES implementation uses scores computed on hard batches under parameter perturbations. This allows new criteria to enter training without requiring a differentiable approximation of each one.

Evolution strategies and low-rank perturbations. Evolution strategies update model parameters using evaluations of perturbed models. Salimans et al. show how this approach can scale to neural networks and use paired positive and negative perturbations to estimate an update (Salimans et al., 2017). Since the method only needs objective values, these evaluations can include discrete sampling, structural checks, and other non-differentiable operations. EGGROLL makes neural ES more efficient by representing matrix perturbations with low-rank factors, reducing the memory and computation needed to evaluate a population (Sarkar et al., 2025). Combining several low-rank perturbations can produce an update with higher rank. Our implementation builds on this approach using rank-one perturbations and scores computed on complete batches of discrete outputs. The population consists of perturbed generators during training. At inference, we sample from a single learned generator. The cost of evaluating the training population must therefore be assessed separately from the cost of generating new samples.

## B.2 STRUCTURED GENERATION AND APPLICATIONS

Molecular graph generation. Molecular generators use several approaches to construct graphs. MolGAN predicts a complete graph directly (Cao and Kipf, 2018), while G2PT represents graphs as token sequences and generates them through autoregressive next-token prediction (Chen et al., 2025). GDSS models the joint evolution of nodes and edges with stochastic differential equations (Jo et al., 2022). DiGress uses discrete noise and denoising steps to generate categorical graphs (Vignac et al., 2023). GruM and DeFoG further develop diffusion and flow-based methods for graph generation (Jo et al., 2024; Qin et al., 2025). These methods provide context for the balance between sample quality and generation cost. We examine this balance through distributional fidelity, validity, diversity, and computational cost. Comparisons with external models also depend on their data, decoding procedures, and evaluation protocols, which we distinguish from our controlled comparisons between training methods.

Scaffold completion and candidate libraries. Scaffold completion requires a generator to preserve a supplied molecular core while proposing extensions. MoLeR supports this task by adding atoms and motifs through sequential decisions (Maziarz et al., 2022). Our scaffold generator predicts the variables needed for a complete proposal in one network evaluation, and its decoder preserves the supplied core. We compare the resulting systems using the same public scaffold queries and proposal budget. This comparison concerns the candidate libraries returned by the complete systems, including their respective training and decoding choices. We measure distinct usable completions, distinct completions that pass the screening filters, scaffold coverage, and generation throughput. These quantities capture different aspects of performance. Producing many candidates for a few scaffolds can increase total yield while leaving other queries unsolved, and repeated valid outputs consume the budget without expanding the library.

Offline architecture generation. NAS-Bench-101 provides a finite space of neural architectures with recorded performance measurements, allowing reproducible evaluation without training every proposed architecture from scratch (Ying et al., 2019). We use an offline labeled corpus to learn a reusable distribution of architecture proposals. Evaluation measures how many distinct held-out elite architectures the generator produces within a fixed sampling budget. This setting differs from adaptive architecture search, where new performance evaluations guide later proposals. Access to the offline labels is part of the supervision budget and must be considered when comparing these settings. Our experiment asks whether training on discrete graph scores can produce a useful and diverse pool of architecture candidates.

Scope of our contribution. Direct generation, distribution matching, and evolution strategies are established ideas. Our contribution is to combine them in a practical training approach for discrete structures and evaluate when this combination is useful. Energy and Drift define distributional objectives, while ES, straight-through estimation, and continuous relaxation define different ways to train the generator. Keeping these choices separate helps identify what each comparison measures. Binarized MNIST provides a further setting in which to compare evolutionary and gradient-based training. Across the experiments, we assess sample quality, useful candidate yield, and training and inference costs to understand when evolutionary training offers a useful trade-off.

## C MODEL, TRAINING, AND EVALUATION DETAILS

This section covers the experiments discussed in the main paper, including the binarized-MNIST control cited in RQ3. For each task, we describe what the generator produces, how its fitness is built, and how we evaluate it. Across tasks, one-step generation means one evaluation of the neural generator per proposal. Sampling, graph assembly, chemical checks, and benchmark lookup can still add computation after that evaluation.

Shared evolutionary update. Let θ denote the generator parameters and let ℓ index a parameter block, such as a weight matrix or bias vector. Each update evaluates a population of $\bar { P } = 1 9 2$ candidates in $J = P / \mathrm { \bar { 2 } } = 9 6 $ pairs. Pair $j \in \{ 1 , \dotsc , J \}$ uses blocks $\begin{array} { r } { \theta _ { j \ell } ^ { \pm } = \dot { \theta _ { \ell } } \dot { \pm } \sigma _ { \ell } \epsilon _ { j \ell } . } \end{array}$ , where $\sigma _ { \ell } > 0$ is the perturbation scale. For a weight matrix of shape $d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } , \epsilon _ { j \ell } = a b ^ { \top }$ , with independent standard Gaussian vectors $a \in \mathbb { R } ^ { d _ { \mathrm { o u t } } }$ and $b \in \mathbb { R } ^ { d _ { \mathrm { i n } } } ; d _ { \mathrm { i n } }$ and $d _ { \mathrm { o u t } }$ are its input and output widths. Bias perturbations are standard Gaussian vectors of the same length as the bias. Perturbations are independent across pairs and parameter blocks. All candidates share latent noise, sampling noise, conditions, and reference examples. This makes their scores comparable without adding differences caused by resampling. For parameter block ℓ, the update direction is

$$
\widehat { g } _ { \ell } = \frac { 1 } { P \sigma _ { \ell } } \sum _ { j = 1 } ^ { J } ( u _ { j } ^ { + } - u _ { j } ^ { - } ) \epsilon _ { j \ell } ,\tag{12}
$$

where $u _ { j } ^ { \pm }$ are the scalar utilities of the two candidates and $\widehat { g } _ { \ell }$ is the ascent direction for block ℓ. Adam applies this direction as an ascent step. We use t to index optimizer updates and $q \in \{ 1 , \ldots , P \}$ to index individual candidates. For any array W, RMS(W) denotes the square root of the mean of its squared entries. The task descriptions specify whether utilities use ranks or raw losses. Equal update counts do not imply equal training work: an evolutionary update evaluates a population, whereas a gradient update optimizes one generated batch.

EGO training algorithm. Algorithm 1 summarizes the shared training procedure. The central network is the unperturbed generator with parameters θ. Each task supplies its own fitness components and utility rule, defined in Sections C.1–C.4.

Utilities are formed after the whole population is scored, so any ranks compare candidates under the same conditions. Fixed training-derived feature statistics remain unchanged throughout training. The QM9 matched-reward and ZINC250K active-time controls return the latest completed checkpoint within each budget; other experiments use the selection rules specified below. At deployment, each proposal uses one forward pass through the returned network.

```latex
Algorithm 1 EGO training
Require: Data splits, task settings, and training budget.
1: Initialize θ, Adam state, fixed feature statistics, and any rarity archive.
2: while the training budget remains do
3: Draw shared latent and sampling noise, references, and any conditions.
4: Generate an unperturbed batch if the task requires one.
5: Prepare dynamic scales and targets; age the archive when scheduled.
6: Hold all scoring quantities and archive counts fixed across candidates.
7: Sample J perturbation pairs $\begin{array} { r } { \theta _ { j \ell } ^ { \pm } = \theta _ { \ell } \pm \sigma _ { \ell } \epsilon _ { j \ell } . } \end{array}$
8: for each candidate $q = 1 , \ldots , \overset { } { P }$ do
9: Generate a batch of hard samples using the shared inputs.
10: Compute the task’s fitness components on the whole batch.
11: end for
12: Convert the population’s scores to utilities $\boldsymbol { u } _ { \boldsymbol { q } }$ using the task’s rule.
13: Compute $\widehat { g } _ { \ell }$ with Eq. (12) and apply Adam ascent.
14: If rarity is used, add eligible pre-update unperturbed samples to its archive.
15: Follow the task’s perturbation-scale and checkpoint schedules.
16: end while
17: return The central-network checkpoint prescribed by the experiment.
```

Table 3: Molecular generator configurations. Canvas size is the maximum number of atom positions in the output. Hidden layers are shown as depth × width. The ZINC250K active-time control uses the full 38-atom canvas with a smaller hidden width than the benchmark model.
<table><tr><td>Experiment</td><td>Canvas size</td><td>Hidden layers</td></tr><tr><td>QM9 benchmark and matched-reward control</td><td>9</td><td> $3 \times 7 6 8$ </td></tr><tr><td>ZINC250K, primary experiment</td><td>38</td><td>3 × 1024</td></tr><tr><td>Full GuacaMol</td><td>88</td><td>3 × 768</td></tr><tr><td>Controlled molecular size sweep</td><td>32</td><td> $3 \times 7 6 8$ </td></tr><tr><td>ZINC250K active-time control</td><td>38</td><td>3 × 768</td></tr></table>

## C.1 MOLECULAR GENERATION

Summary. We use QM9, ZINC250K, and GuacaMol to test whether a compact one-step generator can produce many distinct valid molecules at low inference cost. We measure validity, distinct yield, distributional fidelity, and estimated neural work. The size sweep and budget controls examine how these trade-offs change with molecular size and available training resources; the QM9 reward control compares update recipes under the same hard reward.

Takeaway. High validity alone can hide repeated outputs. We therefore assess distinct valid yield together with fidelity and cost, and check novelty separately against training molecules.

Generator and experimental settings. A ReLU MLP receives a 96-dimensional standard Gaussian latent vector and a one-hot encoding of the requested number of heavy atoms, sampled from the relevant training size distribution. Two output heads predict atom states and bonds. An atom state specifies element, charge, and hydrogen information; bonds have four states: absent, single, double, or triple. Aromatic molecules use a Kekule representation. Independent Gumbel-max draws sample´ the discrete states. We predict each unordered atom pair once, mirror its bond, and mask unused atoms. These operations enforce size and symmetry; they do not guarantee chemical validity. Output biases start from training atom and bond frequencies.

Table 3 distinguishes the full-dataset models from the controlled comparisons. Within each controlled experiment, the training methods use the same generator architecture.

Table 4: Molecular features and their role in the fitness. Counts and histograms are normalized by the active atom count. Local-neighborhood histograms use bond-aware Weisfeiler–Lehman (WL) label refinement.
<table><tr><td>Feature family</td><td>Information matched to training molecules</td></tr><tr><td>Atom states</td><td>Composition, including charge and hydrogen states.</td></tr><tr><td>Valence</td><td>Distribution of bond-order sums at atoms, in bins  $_ { 0 - 8 . }$ </td></tr><tr><td>Bond types</td><td>Relative counts of single, double, and triple bonds.</td></tr><tr><td>WL1</td><td>Local atom neighborhoods after one refinement round; 64 bins.</td></tr><tr><td>WL2</td><td>Larger neighborhoods after two refinement rounds; 64 bins.</td></tr><tr><td>Spectrum</td><td>Global connectivity, using 10 bins of normalized-Laplacian eigenval-</td></tr><tr><td>Global counts</td><td>ues. Bonds, connected components, and independent cycles per atom.</td></tr></table>

Molecular size and search difficulty. The canvas has $N$ available atom positions, of which $n \leq N$ are active in a generated molecule. For these active positions, the generator makes

$$
D ( n ) = n + { \binom { n } { 2 } } = { \frac { n ( n + 1 ) } { 2 } }\tag{13}
$$

categorical decisions: one atom state per position and one bond state per unordered pair. This gives 136, 300, and 528 decisions at 16, 24, and 32 atoms. With A available atom states and four bond states, the space before chemical constraints contains $A ^ { n } 4 ^ { \binom { n } { 2 } }$ labeled graph assignments. These are graph encodings, not distinct valid molecules.

Chemical validity restricts this space sharply. A fixed upper bound on atom valence permits only a number of present bonds proportional to n, although the number of possible pairs grows quadratically. Connectivity and local chemistry impose further constraints. Training must therefore coordinate many choices to place probability on this much smaller valid subset, while spreading that probability over different molecules.

The size sweep tests whether validity, Strict VUN, and feature matching are maintained as larger molecules enter the task. It uses ZINC250K and GuacaMol with cumulative caps of 16, 24, and 32 heavy atoms, keeping the canvas fixed at $N = 3 2$ . The vocabulary, chemistry tables, size-encoding width, and initialization also stay fixed. Equation (13) counts active decisions; masked positions do not contribute. Larger caps also change the molecular mixture and reference distribution, so changes in performance cannot be attributed to decision count alone.

Distribution matching. We compute features directly from every sampled graph, including invalid ones. Table 4 lists the seven feature families and what each contributes. Keeping invalid graphs in the batch allows them to affect training instead of disappearing from the objective.

For feature family $r ,$ let $\phi _ { r } ( x ) \in \mathbb { R } ^ { d _ { r } }$ denote the raw features of sample $x ,$ with $d _ { r }$ coordinates. Write $\phi _ { r }$ for the task-specific normalized feature map defined below. For generated batch $X = ( x _ { 1 } , \dots , x _ { B } )$ and reference batch $Y = ( y _ { 1 } , \dots , y _ { B _ { + } } )$ , set $f _ { i } ^ { ( r ) } = \widetilde { \phi } _ { r } ( x _ { i } )$ and $h _ { i } ^ { ( r ) } = \widetilde { \phi } _ { r } ( y _ { j } )$ . Here $B , B _ { + } \geq 2$ are batch sizes; batches retain repeated samples. We use the empirical Energy discrepancy

$$
\begin{array} { l } { \displaystyle \widehat { \mathcal { E } } _ { r } = \frac { 2 } { B B _ { + } } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { B _ { + } } d \big ( f _ { i } ^ { ( r ) } , h _ { j } ^ { ( r ) } \big ) } \\ { \displaystyle - \frac { 1 } { B ( B - 1 ) } \sum _ { i \neq i ^ { \prime } } d \big ( f _ { i } ^ { ( r ) } , f _ { i ^ { \prime } } ^ { ( r ) } \big ) - \frac { 1 } { B _ { + } ( B _ { + } - 1 ) } \sum _ { j \neq j ^ { \prime } } d \big ( h _ { j } ^ { ( r ) } , h _ { j ^ { \prime } } ^ { ( r ) } \big ) , } \end{array}\tag{14}
$$

where $d ( a , b ) = \sqrt { \| a - b \| _ { 2 } ^ { 2 } + 1 0 ^ { - 1 2 } }$ is a smoothed Euclidean distance between feature vectors. Within-batch sums run over all ordered distinct pairs: $i , i ^ { \prime } \in \{ 1 , \ldots , B \}$ and $j , j ^ { \prime } \in \{ 1 , \dots , B _ { + } \}$ The first term brings generated features toward the reference distribution. The second rewards spread among generated examples. The final term measures spread within the reference batch and is constant while candidates are compared. Thus, the objective already includes a diversity signal before adding rarity.

The molecular benchmark, size sweep, and active-time comparison use $\widetilde { \phi } _ { r } ( x ) = \phi _ { r } ( x ) / s _ { r }$

The scale uses an unperturbed batch $X ^ { 0 } = ( x _ { 1 } ^ { 0 } , \dots , x _ { B _ { 0 } } ^ { 0 } )$ , with $B _ { 0 } = 1 9 2$ , and $B _ { + } = 1 9 2$ references:

$$
s _ { r } = \operatorname* { m a x } \left\{ \frac { \displaystyle \sum _ { i = 1 } ^ { B _ { 0 } } \sum _ { j = 1 } ^ { B _ { + } } \sqrt { \operatorname* { m a x } \{ \| \phi _ { r } ( x _ { i } ^ { 0 } ) - \phi _ { r } ( y _ { j } ) \| _ { 2 } ^ { 2 } , 1 0 ^ { - 1 2 } \} } } { B _ { 0 } B _ { + } \sqrt { d _ { r } } } , 1 0 ^ { - 8 } \right\} .\tag{15}
$$

This positive scalar and the references remain fixed for all candidates. Scaling prevents feature units alone from setting their relative influence. These finite features summarize molecules; matching them does not guarantee that the full molecular distributions match.

Structural penalties and rarity. Distribution matching alone does not ensure valid chemistry. We therefore maximize the base fitness

$$
F _ { \mathrm { m o l } } = - \sum _ { r = 1 } ^ { 7 } \widehat { \mathcal { E } } _ { r } - 2 \overline { { C } } _ { \mathrm { v a l } } - 0 . 5 \overline { { C } } _ { \mathrm { c o n n } } - \overline { { C } } _ { \mathrm { l o c a l } } ,\tag{16}
$$

Here $F _ { \mathrm { m o l } }$ is a scalar batch fitness to maximize, and $\begin{array} { r } { \overline { { C } } = B ^ { - 1 } \sum _ { i = 1 } ^ { B } C ( x _ { i } ) } \end{array}$ averages a per-graph penalty. For a graph x with n active atoms, let $a _ { j }$ be atom $j ^ { \prime } { \bf s }$ state and $\nu _ { j }$ its sum of incident bond orders. To compute its valence error $\delta _ { j }$ , consider bins $v = 0 , \ldots , 1 2 \colon$ use cost $| \nu _ { j } - v |$ when v is allowed for $a _ { j }$ , and cost 13 otherwise, then take the minimum. If $K ( x )$ is the number of connected components and $L ( x )$ the size of the largest, the first two penalties are

$$
C _ { \mathrm { v a l } } ( x ) = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \delta _ { j } , \qquad C _ { \mathrm { c o n n } } ( x ) = { \bf 1 } \{ K ( x ) > 1 \} + 1 - \frac { L ( x ) } { n } .\tag{17}
$$

The indicator $\mathbf { 1 } \{ \cdot \}$ is one when its condition holds and zero otherwise. The local-chemistry penalty $C _ { \mathrm { l o c a l } } ( x )$ adds two quantities divided by $n \mathrm { : }$ each atom’s nearest-pattern $\ell _ { 1 }$ distance to a training pattern for the same atom state, summed over atoms, and the number of bonds with unsupported endpoint-state/bond-order combinations. The distance sums absolute differences in incident single/double/triple-bond counts. Pattern lookup clips counts to (6, 3, 2) and adds the clipped-off overflow; an atom state with no observed pattern receives lookup cost 13. Patterns and supported bond combinations are obtained from training data. The weights (2, 0.5, 1) are fixed experimental settings.

When enabled, rarity adds

$$
R _ { t } ( X ) = \frac { 1 } { | X | } \sum _ { x \in X } \frac { g ( x ) } { \sqrt { 1 + \widehat { v } _ { t } ( x ) } } ,\tag{18}
$$

Here X is the candidate’s generated batch, $| X | = B$ counts occurrences including repeats, and g(x) is a binary eligibility indicator: it is one when the graph is connected, every $\delta _ { j } = 0$ , and $C _ { \mathrm { l o c a l } } ( x ) \le 1 0 ^ { - 6 }$ , and zero otherwise. The approximate visit count $\widehat { v } _ { t } ( x )$ is the smaller of two hash-bucket counts recorded before update t. This reward favors eligible graphs that have been generated less often. It is neither an exact identity test nor a reward for novelty relative to the training set. The archive stays fixed during candidate scoring. After the update, it receives eligible occurrences from the pre-update unperturbed batch. Before every 1,000th update, its integer counts are halved, rounding down.

In these experiments, each candidate is scored on the same $B = 9 6$ inputs. For a population score vector $\boldsymbol { \zeta } = ( \zeta _ { 1 } , \ldots , \zeta _ { P } )$ , define its centered rank (CR) as $\mathrm { C R } ( \zeta ) _ { q } = \mathrm { \bar { r a n k } _ { a v g } } ( \zeta _ { q } ) / ( \bar { P } - 1 ) - 1 / 2$ where ranks are ascending, start at zero, and are averaged across ties. Let $F _ { q } = F _ { \mathrm { m o l } } ( X _ { q } , Y )$ and $R _ { q } = R _ { t } ( X _ { q } )$ be candidate $q ^ { * } { \bf s }$ base fitness and rarity score; $F$ and $R$ collect all $P$ scores. Its utility is $\dot { u } _ { q } = \mathrm { C R } ( \dot { F } ) _ { q } + \lambda _ { R } \mathrm { C R } ( \bar { R } ) _ { q }$ . We use $\lambda _ { R } = 0 . 2$ , or zero in the rarity ablation. Ranking the terms separately controls their influence without depending on their raw numerical ranges. Adam uses learning rate $1 0 ^ { - 2 }$ ; weight perturbations have scale max $\{ 0 . 1 \mathrm { R M S } ( W ) , 1 0 ^ { - 3 } \}$ and bias perturbations have scale 0.01.

Table 5: Budgets for the molecular comparisons. Each evaluated model produces 10,000 raw proposals for the primary yield metrics. Three-seed experiments use seeds 42, 43, and 44; the QM9 reward control uses seeds 42–46. Its row reports the largest matched reward budget.
<table><tr><td>Experiment</td><td>Seeds</td><td>Reported training budget</td></tr><tr><td>QM9 benchmark</td><td>3</td><td>120,000 updates</td></tr><tr><td>Full GuacaMol</td><td>3</td><td>240,000 updates</td></tr><tr><td>Molecular size sweep</td><td>5</td><td>30,000 updates</td></tr><tr><td>ZINC250K active-time control</td><td>3</td><td>30 active minutes</td></tr><tr><td>QM9 matched-reward control</td><td>5</td><td>11.52M graph evaluations</td></tr></table>

STE and Soft comparisons. The size-sweep and active-time comparisons share the generator and initialization within each seed. Soft trains on relaxed categorical outputs. STE keeps the numerical hard forward loss but obtains derivatives through differentiable feature and constraint approximations. Both use shared Gumbel noise and relaxation temperature 1. Their proxy features describe composition, valence, degree, element/bond combinations, and topology; they replace the hard hashes and spectral features that cannot be differentiated directly. Proxy coordinates use fixed training-derived scales, whereas EGO in these comparisons uses the dynamic family scales above. Neither baseline includes hashed rarity. Both use Adam with learning rate $3 \times 1 \dot { 0 } ^ { - 4 }$ and gradient-norm clipping at 10.

Matched-reward comparison on QM9. The separate QM9 control uses five paired seeds (42–46), the same generator, data split, and initialization, and the same hard reward $R _ { B } \stackrel { = } { = } F _ { \mathrm { m o l } }$ for EGO and REINFORCE-LOO (RLOO). Both methods use feature scales computed once from $8 , 1 9 2$ training pairs: the mean pairwise Euclidean distance in each family, divided by $\sqrt { d _ { r } }$ and floored at $1 0 ^ { - 8 }$ These replace the dynamic $s _ { r }$ in Eq. (15) and remain fixed. Neither method uses rarity or rank shaping. This keeps the reward construction identical while comparing the two update recipes.

RLOO uses 96 generated graphs and 192 training references per update. It assigns sample i the difference

$$
A _ { i } = R _ { B } ( X _ { 1 : B } ) - R _ { B - 1 } ( X _ { - i } ) , \qquad { \mathcal { L } } _ { \mathrm { R F } } = - \sum _ { i = 1 } ^ { B } { \mathrm { s t o p g r a d } } ( A _ { i } ) \log p \theta ( X _ { i } \mid z _ { i } , n _ { i } ) .\tag{19}
$$

Here $X _ { 1 : B }$ is the complete policy batch and $X _ { - i }$ is the same batch with graph $X _ { i }$ removed. $R _ { B - 1 }$ evaluates the reward using batch size $B - 1$ , and $A _ { i }$ is the resulting leave-one-out advantage. $p _ { \theta } ( X _ { i } \mid z _ { i } , n _ { i } )$ is the probability of the sampled atom and bond decisions given latent vector $z _ { i }$ and requested atom count $n _ { i }$ . Its log sums only active decisions. The operator stopgrad keeps its argument’s value but treats it as constant during differentiation; $\mathcal { L } _ { \mathrm { R F } }$ is minimized. The sum is intentional because $A _ { i }$ already includes the batch normalization. Removing a graph changes its distances to the other generated graphs, so the complete batch reward is recomputed with the correct smaller-batch normalization. References and feature scales remain fixed during this calculation. EGO uses the same batch sizes per candidate and raw paired reward differences across its 192 candidates. Both use Adam, with learning rates $1 0 ^ { - 2 }$ for $\mathrm { E G O }$ and $3 \times 1 0 ^ { - 4 }$ for RLOO.

At 11,520,000 hard-reward graph evaluations per method and seed, EGO completes 625 updates, and RLOO completes 120,000. We evaluate the current parameters at the declared budget boundaries, without validation-based checkpoint selection. This matches the number of generated graphs scored by the reward, rather than elapsed time or total computation. The update recipes still differ in optimizer settings and reference reuse. Appendix D.2.3 gives the protocol and results.

Budgets and checkpoint selection. Table 5 gives the budgets used for the molecular benchmarks and controlled comparisons. The primary ZINC250K benchmark retains its original checkpoint rule and its vocabulary and allowed-valence table collected before splitting. These differ from the controlled protocols; details are in Appendix D.

The QM9 and full-GuacaMol benchmark models select checkpoints using 2,000 fixed-noise validation proposals. The score is strict valid-and-unique yield divided by $1 + 8 M _ { w }$ , where $M _ { w } = ( 2 \dot { \operatorname* { m a x } } \{ M _ { \mathrm { a t o m } } , 0 \} + \operatorname* { m a x } \{ M _ { \mathrm { v a l } } , 0 \} + \operatorname* { m a x } \{ M _ { \mathrm { s p e c } } , 0 \} ) / 4$ . Here $M _ { \mathrm { a t o m } } , M _ { \mathrm { v a l } }$ , and $M _ { \mathrm { s p e c } }$ are estimates of squared maximum mean discrepancy $\mathrm { ( M M D ^ { 2 } ) }$ for atom-state, valence, and spectral features; $M _ { w }$ is their weighted, zero-clipped average. The yield in the selection score is a fraction between zero and one. This balances usable output yield with distribution matching. The size sweep instead prioritizes strict validity, then strict valid-and-unique yield, then lower $M _ { w }$ , using 10,000 fixed-noise proposals and a training reference panel. The ZINC250K active-time control uses a separate full-ZINC cache with a training-derived vocabulary and size prior, shared by all four methods. Methods run sequentially on the same allocated GPU, with their order rotated across seeds. Each reported point uses the latest completed update within its budget. Active time includes synchronized batch preparation, transfers, and optimizer/archive updates; it excludes setup, compilation, evaluation, and checkpoint I/O. These selection rules do not use training-set novelty or final test results.

Evaluation. Full validity requires RDKit sanitization of the complete generated graph; strict validity also requires connectedness. Full VU counts distinct full-valid molecules, while Strict VUN counts distinct, connected, valid molecules absent from the declared training reference. Both yield counts are divided by all attempted proposals, including invalid outputs and duplicates. Conditional uniqueness instead divides by valid proposals. Novelty uses full source-training identities for the benchmark models and QM9 reward control, and prepared training identities for the size-sweep and time controls. Identities use canonical non-isomeric SMILES. Disconnected molecules are not replaced by their largest component in these full-graph metrics.

Distributional evaluation uses feature $\mathbf { M M D ^ { 2 } }$ and Frechet ChemNet Distance (FCD), with lower´ values preferred. $\mathbf { M M D ^ { 2 } }$ excludes diagonal within-sample terms and can be negative at finite sample sizes. Its reference-derived bandwidth is fixed within each comparison. The size-sweep metric averages atom, valence, and spectral $\mathrm { { \bf M M D } ^ { 2 } }$ equally, whereas $M _ { w }$ above gives atom features twice the weight. FCD retains repeated valid molecules. The QM9 benchmark, QM9 reward control, and full GuacaMol use a separate 30,000-attempt bank to obtain up to 10,000 valid molecules for FCD; the historical ZINC evaluation uses valid outputs from its 10,000-attempt bank. These extra FCD samples do not change yield denominators. External datasets, splits, and FCD score conversions are documented in Appendix D.

External comparisons and computational cost. External molecular baselines retain their source training and evaluation protocols. We estimate dense inference work by counting two operations per dense multiply-accumulate across all neural evaluations needed for one proposal. With VU yield $Y _ { \mathrm { { V U } } }$ expressed as a fraction and estimated operation count per proposal $C _ { \mathrm { d e n s e } }$ , the reported efficiency is $E _ { \mathrm { V U } } ^ { ' } = 1 0 ^ { 1 2 } Y _ { \mathrm { V U } } / C _ { \mathrm { d e n s e } }$ . It measures distinct valid output per estimated neural work; it excludes feature extraction, graph processing, and chemistry, and is not throughput. The GuacaMol CFM pilot is a three-epoch feasibility checkpoint, excluded from ratio and ranking claims. Its projected full-training time is separate from measured training times. Appendix D.4 documents that pilot and the external training-time comparison.

## C.2 SCAFFOLD COMPLETION

Summary. We test whether one-step generation can turn a supplied molecular core into a useful library for downstream screening. At equal proposal budgets, we compare with ZINC-trained MoLeR on distinct filter-passing completions, coverage across cores, and generation-to-SMILES time. A separate matched-time experiment measures how many candidates can be generated and screened within a short deadline.

Takeaway. A generator can return more passing molecules while serving fewer scaffolds. Candidate count, query coverage, and generation time therefore measure different aspects of the resulting library.

Conditional generator and decoding. The model has a 38-atom canvas, three 768-unit global layers, and two relational blocks with 128-dimensional node states. It receives the core, the requested total size, and global and per-node Gaussian noise. Core features describe atom states, observed bonds, and available valence, giving the network information about where additions may fit. The decoder copies all core atoms and core-to-core pair states. Each added atom chooses an earlier atom as a parent, with a bond of at least order one. For a connected core, these parent edges connect every addition to the core; optional bonds allow cycles. All scores come from one network evaluation. The decoder does not repair invalid valences.

Conditional distribution fitness. The scaffold model uses Drift targets. Its nine raw feature families $\phi _ { r }$ , indexed by $r = 1 , \ldots , 9$ , include the seven in Table $^ { 4 , }$ plus an atom-state/valence histogram with 13A coordinates and a 256-bin hashed histogram of bonded atom-state pairs and bond orders. Here A is the scaffold atom-state vocabulary size; valences are clipped to bins 0–12. Both added histograms are normalized by active atom count. They help distinguish chemically different uses of the same elements. For completed graph x and core graph $c ,$ each joint family is $\psi _ { r } ( x , c ) = [ \phi _ { r } ( x ) , \lambda _ { s } \phi _ { r } ( c ) ] \in \mathbb { R } ^ { D _ { r } }$ , where brackets denote concatenation, $D _ { r } = 2 d _ { r } ,$ , and $\lambda _ { s }$ weights the core features relative to the completion features. Including the core makes the comparison sensitive to which input a completion belongs to.

Drift targets. At each update, the unperturbed and reference batches contain $B _ { 0 } = B _ { + } = 3 8 4$ completion/core pairs. We compute $s _ { r }$ using Eq. (15), replacing $\phi _ { r }$ by $\psi _ { r }$ and $d _ { r }$ by $D _ { r }$ . For one family, let $G$ and H be the normalized unperturbed and reference feature matrices: their rows are joint features divided by $s _ { r } .$ They have shapes $B _ { 0 } \times D _ { r }$ and $B _ { + } \times D _ { r }$ . The family index is omitted in the following construction.

For matrices $U , V$ with the same feature width, define the pairwise distance matrix by ${ \sf d } ( U , V ) _ { i j } = \sqrt { \operatorname* { m a x } \{ \| U _ { i } - V _ { j } \| _ { 2 } ^ { 2 } , 1 0 ^ { - 1 2 } \} }$ , where $U _ { i } , V _ { j }$ are rows. For each temperature $\tau \in { }$ $\sqrt { D _ { r } } \{ 0 . 0 2 , 0 . 0 5 , 0 . 2 \}$ , form

$$
\begin{array} { r l } & { L _ { \tau } = \left[ - \mathsf { d } ( G , H ) / \tau , \mathsf { \Pi } - ( \mathsf { d } ( G , G ) + 1 0 ^ { 9 } I _ { B _ { 0 } } ) / \tau \right] , } \\ & { Q _ { \tau } = \sqrt { \mathrm { s o f t m a x } _ { \mathrm { r o w } } ( L _ { \tau } ) \odot \mathrm { s o f t m a x } _ { \mathrm { c o l } } ( L _ { \tau } ) } = [ Q _ { \tau } ^ { + } , Q _ { \tau } ^ { - } ] . } \end{array}\tag{20}
$$

Here $I _ { B _ { 0 } }$ is the identity matrix, so the large diagonal term removes self-interactions. Row softmax normalizes across columns; column softmax normalizes across rows. The product ⊙ and square root act entrywise. The first $B _ { + }$ columns, $Q _ { \tau } ^ { + }$ , weight reference examples; the remaining $B _ { 0 }$ columns, $Q _ { \tau } ^ { - }$ , weight unperturbed examples.

Let $\begin{array} { r } { m _ { \tau , i } ^ { \pm } = \sum _ { j } Q _ { \tau , i j } ^ { \pm } } \end{array}$ denote their row sums. The attraction–repulsion field and target matrix are

$$
\begin{array} { l } { { \displaystyle V _ { \tau , i } = m _ { \tau , i } ^ { - } \sum _ { j = 1 } ^ { B _ { + } } Q _ { \tau , i j } ^ { + } H _ { j } - m _ { \tau , i } ^ { + } \sum _ { j = 1 } ^ { B _ { 0 } } Q _ { \tau , i j } ^ { - } G _ { j } } , } \\ { { \displaystyle T _ { r } = G + \sum _ { \tau } \frac { V _ { \tau } } { \mathrm { m a x } \{ \mathrm { R M S } ( V _ { \tau } ) , 1 0 ^ { - 8 } \} } . } } \end{array}\tag{21}
$$

$V _ { \tau }$ has $B _ { 0 }$ rows of width $D _ { r }$ ; RMS averages over all its entries. The three normalized fields are summed. Targets and scales stay fixed while candidates are scored. Candidate q receives

$$
O _ { q } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { r = 1 } ^ { 9 } \frac { 1 } { D _ { r } } \left\| \frac { \psi _ { r } ( x _ { q i } , c _ { i } ) } { s _ { r } } - T _ { r , i } \right\| _ { 2 } ^ { 2 } , \qquad B = 9 6 ,\tag{22}
$$

where $x _ { q i }$ is its completion of core $c _ { i } ,$ and $T _ { r , i }$ is row i of the target matrix. Only the first $B = 9 6$ shared conditions are used. $O _ { q }$ is a scalar score to maximize: it rewards movement toward the fixed targets, with a dimension correction for each family.

Chemical constraints and diversity. For each completed graph, use the per-atom valence errors $\delta _ { j }$ and the local penalty $C _ { \mathrm { l o c a l } }$ defined above. The scaffold risks are

$$
\begin{array} { l l } { \displaystyle C _ { \mathrm { v a l } } ^ { \mathrm { s c a f } } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \delta _ { j } + \alpha _ { \mathrm { m a x } } \operatorname* { m a x } _ { j } + \alpha _ { \mathrm { f a i l } } \mathbf { 1 } \{ \operatorname* { m a x } _ { j } \delta _ { j } > 1 0 ^ { - 6 } \} , } \\ { \displaystyle C _ { \mathrm { l o c a l } } ^ { \mathrm { s c a f } } = C _ { \mathrm { l o c a l } } + \alpha _ { \mathrm { l o c a l } } \mathbf { 1 } \{ C _ { \mathrm { l o c a l } } > 1 0 ^ { - 6 } \} . } \end{array}\tag{23}
$$

The nonnegative coefficients $\alpha _ { \mathrm { m a x } } , \alpha _ { \mathrm { f a i l } }$ , and $\alpha _ { \mathrm { l o c a l } }$ weight worst-atom error, any valence failure, and any local-chemistry failure, respectively. These terms stop one failing part of a molecule from being hidden by an average. With $n _ { \mathrm { a d d } }$ added atoms and $n _ { \mathrm { d e t } }$ of them having no path to the core, attachment risk is $C _ { \mathrm { a t t } } = { \bf 1 } \{ n _ { \mathrm { d e t } } > 0 \} + n _ { \mathrm { d e t } } / \operatorname* { m a x } \{ n _ { \mathrm { a d d } } , 1 \}$ . The candidate penalty is

$$
\Pi _ { q } = \overline { { C } } _ { \mathrm { v a l } , q } ^ { \mathrm { s c a f } } + 1 . 2 5 \overline { { C } } _ { \mathrm { a t t } , q } + 0 . 2 5 \overline { { C } } _ { \mathrm { l o c a l } , q } ^ { \mathrm { s c a f } } .\tag{24}
$$

Table 6: Scaffold coefficients
<table><tr><td>Coefficient</td><td>Value used in the experiments</td></tr><tr><td>Core feature weight  $\lambda _ { s }$ </td><td>1</td></tr><tr><td>Risk weights  $( \alpha _ { \mathrm { m a x } } , \alpha _ { \mathrm { f a i l } } , \alpha _ { \mathrm { l o c a l } } )$ </td><td>(1, 0.75,0.5)</td></tr><tr><td>Diversity thresholds  $( a , b )$ </td><td>(0.35, 0.75)</td></tr><tr><td>Rank weights  $( w _ { O } , w _ { \Pi } , w _ { v } , w _ { c } , w _ { U } )$ </td><td>(1, 0.5, 0.25, 0.75, 0.2)</td></tr></table>

Here each bar averages over candidate $q ^ { \prime } \mathrm { s } B = 9 6$ completions. Attachment risk is zero for connected supported cores under the forest decoder.

Conditional unique yield $U _ { q }$ sums the number of distinct eligible graph signatures in each repeated core/size group, then divides by the 64 conditional attempts, including ineligible ones. The same signature in two groups counts once per group. Rarity $R _ { q } \stackrel { - } { = } R _ { t } ( X _ { q } )$ uses all 96 candidate proposals and Eq. (18), with the scaffold gate $g _ { \mathrm { s c a f } } ( x ) = \mathbf { 1 } \{ \delta _ { j } = 0$ for all active $j \} { \bf 1 } \{ K  ( x ) = 1 \}$ in place of g. Thus eligibility requires allowed valences and connectedness. Conditional uniqueness discourages repeated answers to one query, while rarity discourages repeated graph signatures across updates. Let $\begin{array} { r } { \bar { g } _ { t } ^ { 0 } = B _ { 0 } ^ { - 1 } \sum _ { i = 1 } ^ { B _ { 0 } } g _ { \mathrm { s c a f } } ( x _ { i } ^ { 0 } ) } \end{array}$ be the eligible fraction in the 384 unperturbed completions. We use $\gamma _ { t } = \mathrm { c l i p } ( ( \bar { g } _ { t } ^ { 0 } - a ) / ( b - a ) , 0 , 1 )$ , where $0 \leq a < b \leq 1$ are the eligibility thresholds at which diversity pressure starts and reaches full weight. The clip operation restricts its argument to [0, 1].

We convert each score into a centered rank (CR) across the P = 192 candidates. For a score vector $\zeta = ( \zeta _ { 1 } , \ldots , \zeta _ { P } )$

$$
\mathrm { C R } ( \zeta ) _ { q } = \frac { \mathrm { r a n k } _ { \mathrm { a v g } } ( \zeta _ { q } ) } { P - 1 } - \frac { 1 } { 2 } ,\tag{25}
$$

where ranks run from zero for the lowest score to $P - 1$ for the highest, and tied scores receive their average rank. Thus, CR maps scores to [−0.5, 0.5] and removes differences in their numerical scales. We rank −Π so that lower penalties receive higher utility. The utility combines these separately ranked components:

$$
\begin{array} { r l } & { u _ { q } = w _ { O } \mathrm { C R } ( O ) _ { q } + w _ { \Pi } \mathrm { C R } ( - \Pi ) _ { q } + w _ { v } \mathrm { C R } ( v ) _ { q } + w _ { c } \mathrm { C R } ( c ) _ { q } } \\ & { \phantom { a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a } + \gamma _ { t } \left[ w _ { U } \mathrm { C R } ( U ) _ { q } + 0 . 2 \mathrm { C R } ( R ) _ { q } \right] , } \end{array}\tag{26}
$$

Here $v _ { q }$ and $c _ { q }$ are the fractions of candidate $q ^ { * } { \bf s }$ 96 completions with allowed valences and connected graphs, respectively. The vectors $O , \Pi , v , c , U , R$ collect the corresponding scores over all P candidates. Nonnegative weights $w _ { O } , w _ { \Pi } , w _ { v } , w _ { c } , w _ { U }$ multiply the ranks of Drift fit, negative penalty, valence success, connectivity success, and conditional unique yield, in that order. Separate ranks retain a role for each criterion despite their different numerical ranges. The logged value $O _ { q } - \Pi _ { q }$ is only a diagnostic. Table 6 defines the reference values of these coefficients.

Training and evaluation. The unperturbed batch has 384 examples. Each candidate uses the same first 96 conditions, including 64 examples in repeated core/size groups. Core identities are separated across training and held-out partitions. At inference, we sample the number of added atoms from training molecules with the same scaffold size, falling back to pooled training counts when that size is unseen. Only positive additions that fit the supported sizes and 38-atom limit are retained, and the distribution is renormalized. The public input is therefore only the scaffold, without the held-out target molecule or its total size.

Evaluation uses 100 held-out queries and 1,000 attempts per query. A usable completion must be chemically valid, connected, and contain the supplied core as a substructure. We deduplicate canonical non-isomeric SMILES within each query; failed attempts remain in the yield denominator. Screening then requires QED (quantitative estimate of drug-likeness) ≥ 0.7, SA (synthetic accessibility score) $\leq 4 .$ , and no new PAINS (pan-assay interference compounds) alert types relative to the core. These criteria measure the resulting candidate library and never enter the training fitness. Appendix E.1 gives the size prior, checkpoint selection, and available training budgets; Appendix E reports the screening protocol and results. The comparison uses three EGO training seeds (42–44) and three sampling seeds (41–43) of one ZINC-trained MoLeR checkpoint, selected from training seed 42. These represent different sources of variation. The library-generation timer includes conversion to SMILES and excludes subsequent screening.

Matched-time example. The example in Figure 4 shows the highest-QED filter-passing candidate from each method. Its deadline is the time MoLeR needs to generate and screen one batch of 256 completions, with both models already loaded and warmed up. EGO generates batches of 256 graphs and scores them directly; only candidates screened before the deadline count. The selected EGO molecule is converted to SMILES afterwards. This timer includes screening, so it measures a different operation from generation to SMILES.

## C.3 NEURAL ARCHITECTURE GENERATION

Summary. NAS-Bench-101 tests whether the approach is useful beyond chemistry. We train on high-performing architectures and compare EGO, STE, and Soft by the number of distinct held-out elite architectures obtained within 10,000 proposals. Validity, mean benchmark accuracy, and feature MMD measure the trade-offs between a larger candidate library, feasibility, and architecture quality.

Takeaway. Producing more distinct elite architectures can be useful even when mean accuracy is lower. We report both to distinguish the number of qualifying alternatives from their average quality.

Data and generator. Canonical benchmark architectures are split 80/10/10 with seed 2026. The elite threshold is the 90th percentile of validation accuracy among training architectures and is then fixed for the held-out partitions. A three-layer, 512-unit SiLU MLP maps 128-dimensional Gaussian noise to 21 edge logits and three operation logits for each of five internal vertices. Input and output vertices are fixed. Logits are clipped to [−10, 10]; edges use Bernoulli sampling, and operations use Gumbel-max. Output biases use training-elite frequencies.

Vertices outside input-to-output paths are pruned. A proposal is invalid if it has no such path or has more than nine retained edges. The benchmark’s canonical hash identifies architectures after pruning, avoiding duplicate counts caused by equivalent graph encodings.

Fitness and its purpose. Three feature families describe each pruned graph: its edges and operations (36 coordinates), in/out degrees (14), and size, reachability, and path counts (9). Together they capture local choices and how those choices connect input to output. For coordinate $k \in \{ 1 , \ldots , d _ { r } \}$ of raw family $\phi _ { r } .$ , the normalized feature is

$$
\widetilde { \phi } _ { r , k } ( \boldsymbol { x } ) = \frac { \phi _ { r , k } ( \boldsymbol { x } ) - \mu _ { r , k } ^ { \mathrm { t r a i n } } } { \operatorname* { m a x } \{ s _ { r , k } ^ { \mathrm { t r a i n } } , 0 . 0 5 \} } ,\tag{27}
$$

where $\mu _ { r , k } ^ { \mathrm { t r a i n } }$ and $s _ { r , k } ^ { \mathrm { t r a i n } }$ are the mean and population standard deviation of that coordinate over training elites. The statistics are fixed and shared by all methods and by generated and reference examples. Using the Energy discrepancy in Eq. (14), the fitness is

$$
F _ { \mathrm { N A S } } = - \frac { 1 } { 3 } \sum _ { r = 1 } ^ { 3 } \widehat { \mathcal { E } } _ { r } - \frac { 0 . 5 } { B } \sum _ { i = 1 } ^ { B } \left[ 1 - h _ { i } + \frac { \operatorname* { m a x } \{ 0 , e _ { i } - 9 \} } { 9 } \right] ,\tag{28}
$$

Here $B = 6 4 , B _ { + } = 1 9 2$ in each Energy term, $h _ { i } \in \{ 0 , 1 \}$ is one exactly when proposal i has an input-to-output path, and $e _ { i }$ counts its retained edges. The fitness is maximized. The Energy term matches elite graph structure; the penalty addresses the benchmark’s feasibility rules directly. Invalid proposals remain in the batch. We use neither a rarity archive nor an explicit duplicate penalty, so diversity pressure comes from Energy’s generated-to-generated term.

Training, selection, and evaluation. All methods use 64 generated examples and 192 training-elite references per objective evaluation. If $F _ { q }$ is candidate q’s NAS batch fitness, EGO uses

$$
u _ { q } = F _ { q } - { \frac { 1 } { P } } \sum _ { p = 1 } ^ { P } F _ { p } , \qquad P = 1 9 2 .\tag{29}
$$

Centering cancels in each paired difference, leaving raw fitness differences without ranking. Adam’s learning rate is $3 \times 1 0 ^ { - 3 }$ . For a weight block W, its perturbation scale is ω max{0.08 RMS(W), 0.001}; bias scales are $0 . 0 1 \omega _ { t }$ . Both output-layer scales receive an additional factor 0.5. The global multiplier starts at $\omega _ { 0 } = 1$ . At update 1 and every 100th update, we measure the fractions of edge and operation decisions changed by perturbations relative to the unperturbed network under shared noise. If the larger fraction exceeds 0.14, the multiplier is multiplied by 0.9, with a lower bound of 0.2; below 0.04, it is multiplied by 1.1, with an upper bound of 2. Otherwise, it is unchanged. STE and Soft use temperature 0.75 and an Adam learning rate of $3 \times 1 0 ^ { - 4 }$ . STE preserves the hard forward loss while differentiating through relaxed features and pruning; Soft evaluates the relaxed loss. The architecture, initialization, sampling streams, and hard evaluation sampler are shared within each seed.

Table 7: MNIST features used to build the fitness. All are computed directly from generated images, without digit labels.
<table><tr><td>Feature family</td><td>Dimension</td><td>Purpose</td></tr><tr><td>Pixels</td><td>784</td><td>Fine image detail.</td></tr><tr><td> $2 \times 2$  pooled means</td><td>196</td><td>Local stroke patterns.</td></tr><tr><td>4 × 4 pooled means</td><td>49</td><td>Coarse digit shape.</td></tr><tr><td>Row/column means</td><td>56</td><td>Horizontal and vertical ink placement.</td></tr><tr><td>Neighbor differences</td><td>56</td><td>Stroke transitions along rows and columns.</td></tr><tr><td>Global statistics</td><td>8</td><td>Ink amount, position, spread, and mirror asymme- try.</td></tr></table>

We train for 15,000 updates with seeds 42, 43, and 44. Every 500 updates, 2,000 validation proposals are scored by $S _ { \mathrm { N A S } } = 0 . 4 V + 0 . 2 U + 0 . 4 / ( 1 + \operatorname* { m a x } \{ \mathcal { M } _ { \mathrm { N A S } } , 0 \} )$ . The score $S _ { \mathrm { { N A S } } }$ is maximized. V is the valid fraction of all proposals; U is the number of distinct valid architectures divided by the valid count, or zero if none are valid. $\mathcal { M } _ { \mathrm { N A S } }$ is feature $\mathrm { { \bf M M D } ^ { 2 } }$ between up to 1,500 valid proposals and 1,500 validation elites, using raw concatenated features. If MMD cannot be computed, including with fewer than two valid proposals, its entire fidelity term contributes zero. This selects for feasibility, variety, and similarity to held-out elite structure.

Final evaluation uses 10,000 attempts. Held-out elite yield counts distinct generated test-partition elites divided by all attempts. Mean benchmark test accuracy instead includes repeated valid architectures. Evaluation $\mathrm { { \bf M M D } ^ { 2 } }$ uses the raw concatenated graph features and a bandwidth fixed from training elites; these differ from the standardized training features. Appendix F.1 gives the remaining benchmark diagnostics.

## C.4 BINARIZED MNIST

Summary. Binarized MNIST tests the training recipes in a setting without graph feasibility constraints: every sampled output is already a binary array. We compare EGO, STE, and Soft using image-distribution distances, classifier-predicted digit coverage, and exact uniqueness. The aim is to assess image quality and diversity when training relies only on distribution matching, without labels or classifier feedback.

Takeaway. Valid binary arrays need not be recognizable or varied digits. Exact pixel uniqueness must therefore be read alongside image quality and digit coverage.

Data and generator. Images are binarized at a pixel-intensity threshold of 128 on the 0–255 scale. We hold out 500 examples per digit from the official training set using seed 2026, leaving 55,000 training and 5,000 validation images. The official 10,000-image test set is used after checkpoint selection. A three-layer, 768-unit ReLU MLP maps 96-dimensional Gaussian noise to 784 logits. Independent Bernoulli draws produce all pixels of a $2 8 \times 2 8$ image in parallel. Output biases start from training pixel frequencies, and logits are not clipped.

Fitness and its purpose. Table 7 lists the six feature families. They describe image structure at several resolutions, so the objective can compare both individual pixels and larger shapes. We apply Eq. (27) to image features, now estimating $\bar { \mu } _ { r , k } ^ { \mathrm { t r a i n } }$ and $s _ { r , k } ^ { \mathrm { t r a i n } }$ over all 55,000 model-training images. These statistics remain fixed.

The fitness is

$$
F _ { \mathrm { M N I S T } } = - \frac { 1 } { 6 } \sum _ { r = 1 } ^ { 6 } \frac { \widehat { \mathcal { E } } _ { r } } { \sqrt { d _ { r } } } , \qquad ( d _ { 1 } , \ldots , d _ { 6 } ) = ( 7 8 4 , 1 9 6 , 4 9 , 5 6 , 5 6 , 8 ) .\tag{30}
$$

Here $d _ { r }$ is the number of coordinates in family $r ,$ and each Energy term uses $B = 9 6$ generated images and $B _ { + } = 1 9 2$ references. The fitness is maximized. The dimension factor limits the influence of larger feature families solely due to their coordinate count. There are no additional constraints, rarity rewards, or duplicate penalties. As in NAS, the Energy term itself rewards spread among generated examples.

Training and checkpoint selection. Each objective evaluation uses 96 generated images and 192 training references. EGO uses Eq. (29) with $F _ { q } = F _ { \mathrm { M N I S T } , q }$ and Adam learning rate $1 0 ^ { - 2 }$ . Its weight perturbation scale is max{0.1 RMS(W), 0.001} and its bias scale is 0.01. STE and Soft use temperature 0.75 and an Adam learning rate of $3 \times 1 0 ^ { - 4 }$ . They share the generator, initialization, noise streams, and fixed feature normalizers with EGO. STE preserves the hard forward loss; Soft trains on relaxed pixels. All methods sample binary images at evaluation.

We train for 30,000 updates with seeds 42, 43, and 44. Every 500 updates, 2,000 validation proposals are scored by $\begin{array} { r } { S _ { \mathrm { M N I S T } } = U _ { \mathrm { e x a c t } } e ^ { - \Delta / 2 } } \end{array}$ , which is maximized. Here $U _ { \mathrm { e x a c t } }$ is the number of distinct complete binary images divided by 2,000. For each raw feature coordinate $( r , k )$ , let $\mu _ { r , k } ^ { g } , v _ { r , k } ^ { g }$ be its mean and population variance over these generated images, and $\mu _ { r , k } ^ { v } , v _ { r , k } ^ { v }$ the corresponding statistics over all 5,000 validation images. Superscripts $g$ and v denote generated and validation data. We compute

$$
\Delta = \frac { 1 } { 6 } \sum _ { r = 1 } ^ { 6 } \frac { 1 } { d _ { r } } \sum _ { k = 1 } ^ { d _ { r } } \frac { ( \mu _ { r , k } ^ { g } - \mu _ { r , k } ^ { v } ) ^ { 2 } + \frac { 1 } { 4 } \left( \sqrt { v _ { r , k } ^ { g } + 1 0 ^ { - 6 } } - \sqrt { v _ { r , k } ^ { v } + 1 0 ^ { - 6 } } \right) ^ { 2 } } { v _ { r , k } ^ { v } + 1 0 ^ { - 3 } } .\tag{31}
$$

The $1 / 4$ coefficient reduces the weight of standard-deviation mismatch; the small constants stabilize square roots and denominators. This selector uses raw feature statistics and its own normalization, distinct from the training normalization. It combines variety with feature agreement and uses neither digit labels nor a classifier.

Evaluation. A frozen LeNet-style classifier, trained only on the 55,000 training images and their labels, supplies an independent evaluation of digit structure. We report Frechet distance in its 128-´ dimensional embedding, kernel inception distance (KID), here also computed on LeNet embeddings, and $\mathbf { M M D ^ { 2 } }$ with a radial basis function (RBF) kernel, together with predicted digit frequencies and confidence. Distances use 2,000 generated and 2,000 test images; confidence, class counts, and exact identities use all 10,000 generated images. A digit is covered if at least 50 images receive that argmax prediction, without confidence filtering. Confidence is reported separately using a maximum class probability threshold of 0.8. Exact novelty compares complete binary images with the 55,000 training images. Appendix F.2 gives the evaluator and distance settings.

Reporting across experiments. Unless stated otherwise, ± denotes the sample standard deviation across runs. Rates are computed separately for each run before averaging. Timing reports identify which stages are included, since neural sampling, graph processing, chemical conversion, and screening have different costs. Estimated dense-operation counts measure only the stated neural work; their ratios do not measure runtime speedups.

## D RQ1: MOLECULAR GENERATION

This section gives the evidence behind the molecular results in the main text. We first define the shared evaluation protocol, then present QM9, ZINC250K, and GuacaMol separately. Each dataset section includes the relevant external comparison and controlled experiments. The network size and training budget ablations appear in Sections D.2.2 and D.3.2. They vary the depth and width of EGO, while the molecular size sweep varies the data mixture with a fixed network architecture. We conclude with the inference cost calculations and training time audit used in the main figures. Throughout this section, EGO denotes our method, and $\mathrm { E G O _ { 0 } }$ denotes the version without the history-dependent rarity reward.

## D.1 SHARED EVALUATION PROTOCOL

Metrics. Full validity requires chemical sanitization of the complete generated molecule. Full validand-unique yield (VU) is the number of distinct valid molecular identities divided by all raw generation attempts. VUN additionally excludes molecules present in the training reference. Strict validity requires both chemical validity and connectivity, and Strict VUN applies both requirements before counting distinct molecules absent from training. These yields differ from uniqueness conditioned on validity. Invalid and repeated proposals remain in the denominator of every yield metric. EGO uses complete molecular identities, including disconnected molecules when reporting full validity and full VU.

External comparisons. Tables 8, 14, and 22 provide the complete external comparisons supporting the molecular results in the main text. They include the available sampling configurations and the source metrics needed to interpret these comparisons. Published values retain the original preprocessing, validity checks, identity rules, correction procedures, and sampling budgets. An approximate VU value reconstructed from published validity and conditional uniqueness is a sourcespecific estimate, not a measurement from a common proposal bank. We report FCD as a distance, with lower values preferred. For GuacaMol references that report the benchmark FCD score, we give its approximate distance equivalent and state the conversion. Unless stated otherwise, ± denotes the sample standard deviation across training seeds.

Inference efficiency. $E _ { \mathrm { V U } }$ measures VU per $1 0 ^ { 1 2 }$ estimated dense operations at the evaluated proposal budget. The estimate includes every network evaluation required for generation, but excludes other operations and molecular processing. It is distinct from measured VU per second. Section D.5 gives the formula, counting convention, architectural assumptions, and limits of this comparison.

## D.1.1 DESIGN OF THE CONTROLLED MOLECULAR SIZE SWEEP

Why this experiment. Larger molecular graphs require more categorical choices and can make valid, diverse generation harder. We therefore compare $\mathrm { E G O , E G O _ { 0 } , }$ , STE, and Soft across cumulative heavy-atom caps $k \in \{ 1 6 , 2 4 , 3 2 \}$ on ZINC250K and GuacaMol. The experiment asks whether the yield advantage persists across molecular size mixtures and whether it depends on the rarity reward. The full results are reported in Sections D.3.3 and D.4.2.

Shared setup. Each task contains molecules with $n \leq k ,$ , and generation is conditioned on a size sampled from the corresponding training mixture. All settings use a fixed 32-node canvas, a shared atom-state vocabulary derived from training, 96 latent dimensions, and three hidden layers of width 768. The conditioning width and output heads remain fixed across caps. Each method receives 30,000 updates and produces 10,000 final proposals, with five training seeds per setting. The checkpoint-selection bank and final evaluation bank are separate. Selection uses training-reference features; final feature evaluation uses test references. Checkpoints are selected first by strict validity, then by strict VU without a novelty requirement, and finally by a weighted feature MMD criterion. The last criterion weights the nonnegative atom, valence, and spectral MMD terms by $( 2 , 1 , 1 ) / 4$

For n active nodes, the generator samples n atom states and $\binom { n } { 2 }$ bond states. The maximum number of active categorical choices is therefore

$$
D ( k ) = k + { \binom { k } { 2 } } = { \frac { k ( k + 1 ) } { 2 } } , \qquad D ( 1 6 ) = 1 3 6 , \quad D ( 2 4 ) = 3 0 0 , \quad D ( 3 2 ) = 5 2 8 .\tag{32}
$$

These are upper bounds on active choices, not changes in network size. Increasing the cap also changes the data distribution, so the experiment does not isolate the effect of decision count alone.

Training comparisons. EGO combines the hard Energy objective and structural penalties with a hashed rarity reward. $\mathrm { E G O _ { 0 } }$ and STE evaluate the same numerical hard forward criterion. STE uses a soft feature Jacobian during backpropagation, while $\mathrm { E G O _ { 0 } }$ still uses population rank shaping. Soft optimizes the corresponding relaxed objective. Thus, the comparison with STE matches the forward criterion, and the comparison between the two EGO variants measures the effect of rarity. Update counts are equal, but training work is not: an EGO update evaluates 192 perturbed networks on 96 graphs each, whereas a gradient update uses 96 graphs. The separate experiment in Section D.3.4 controls active training time.

![](images/6c535e03ac81dd4fd282a4a5c3b47172f36b4ba68bd7812c3dc43071f4030301.jpg)  
Figure 5: Strict VUN across cumulative molecular size caps. Points and bars show mean ± sample standard deviation across five seeds, with 30,000 updates and 10,000 final proposals per run. EGO includes rarity; $\mathrm { E G O _ { 0 } }$ removes it. Increasing the cap changes the molecular size mixture while the output canvas remains fixed. Error bars are not truncated at zero.

Feature discrepancy and uncertainty. $\mathrm { M M D } _ { \mathrm { c e l l } }$ is the arithmetic mean of unbiased RBF $\mathrm { { \bf M M D } ^ { 2 } }$ estimates for atom, valence, and spectral features on raw generated graphs, using test references and up to 1,500 samples per side. It is compared only within a fixed dataset and cap, because reference distributions and kernel calibration can differ. Connectivity is measured on all proposals, including chemically invalid graphs. Paired differences use the five common seeds and unadjusted 95% Student-t intervals, ${ \overline { { \Delta } } } \pm t _ { 0 . 9 7 5 , 4 } s _ { \Delta } / { \sqrt { 5 } } .$ . The reported intervals use the rounded critical value $t _ { 0 . 9 7 5 , 4 } \approx 2 . 7 7 6$ . These intervals are distinct from the sample standard deviations in the result tables and are not adjusted for multiple comparisons.

Figure 5 summarizes all six size settings. Both EGO variants have higher mean Strict VUN than STE and Soft in every setting. The detailed tables below show the validity, connectivity, feature discrepancy, and paired uncertainty behind this pattern.

$$
\mathrm { ~ - \bullet - ~ E G O ~ } \mathrm { ~ -- ~ } \mathrm { ~ E G O ~ w / o ~ r a r i t y } \mathrm { ~ -- ~ } \mathrm { ~ s / E ~ } \mathrm { ~ -- ~ } \mathrm { ~ S o f t }
$$

Key takeaway. EGO maintains a yield advantage across the tested size mixtures, including settings with up to 528 active choices. Rarity further improves useful yield, with a small increase in feature MMD. This is evidence across equal update budgets, while the separate time control tests the effect of unequal work per update.

## D.2 QM9

QM9 contains small organic molecules with up to nine heavy atoms drawn from carbon, nitrogen, oxygen, and fluorine, with hydrogen completing their valences (Ramakrishnan et al., 2014). The original dataset contains 133,885 molecules. We use QM9 to study molecular quality and distinct yield on a small graph canvas. Our prepared split has 107,109 source training rows, corresponding to 107,061 unique training molecules, and 13,374 test rows. External studies can use different splits and evaluation rules.

## D.2.1 COMPARISON WITH PUBLISHED GENERATORS

Table 8 gives the complete QM9 comparison, including autoregressive models, iterative diffusion and flow models, SMILES generators, graph generators, and the REINFORCE training recipes. Validity and uniqueness describe different parts of molecular quality: uniqueness is conditional on validity, while FCD measures distributional distance. VU/s reports distinct valid outputs per generation second when compatible timing evidence is available. External rows retain the identity, sampling, postprocessing, and timing rules of their sources, as detailed below. Section D.5 separately reports estimated dense inference efficiency.

Table 8: QM9 generation: published baselines and training comparisons at 120,000 updates. EGO and REINFORCE use the same 3 × 768 generator and three training seeds, 42, 43, and 44; their dispersion is the sample SD. Uniqueness is conditional on validity, with the identity and postprocessing rules detailed in the text. VU/s counts valid, unique outputs per generation second. External rows retain the protocols, hardware, and uncertainty conventions of their sources. Gray rows identify one-step generators; dashes denote unavailable compatible evidence.
<table><tr><td>Method</td><td></td><td>Valid (%)</td><td>Unique (%)</td><td>FCD↓</td><td></td></tr><tr><td>Autoregressive generation</td><td>Steps</td><td>↑</td><td>↑</td><td></td><td>VU/s ↑</td></tr><tr><td>GraphARM (Kong et al.,</td><td> $n \leq 9$ </td><td>90.25</td><td>95.62</td><td>1.22</td><td></td></tr><tr><td>2023) G2PT-base (Chen et al., 2025) L tokens</td><td></td><td>99.0</td><td>96.8</td><td>0.06</td><td></td></tr><tr><td>Iterative diffusion / flow</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GDSSª (Jo et al., 2022)</td><td>1,000</td><td>95.72</td><td>98.46</td><td>2.900</td><td>≈86.4</td></tr><tr><td>DiGress (Vignac et al., 2023)</td><td>500</td><td>99.0</td><td>96.2</td><td></td><td></td></tr><tr><td>GruMª (Jo et al., 2024)</td><td>1,000</td><td>99.69</td><td>96.90</td><td>0.108</td><td></td></tr><tr><td>Cometh (Siraudin et al., 2025)</td><td>500</td><td>99.57</td><td>96.76</td><td>0.25</td><td></td></tr><tr><td>CatFlowb (Eijkelboom et al.,</td><td>100</td><td>99.81</td><td>99.95</td><td>0.441</td><td></td></tr><tr><td>2024) DeFoG (Qin et al., 2025)</td><td>50</td><td>98.9</td><td>96.2</td><td>0.26</td><td></td></tr><tr><td>DeFoG (Qin et al., 2025)</td><td>500</td><td>99.3</td><td>96.3</td><td>0.12</td><td></td></tr><tr><td>MELD (Seo et al., 2025)</td><td></td><td>100.00</td><td>96.49</td><td>0.09</td><td></td></tr><tr><td>Few step SMILES generationº</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UDLM + DCD</td><td> $1 + \mathrm { t a i l }$ </td><td>31.54</td><td> $\approx 9 9 . 3 5 $ </td><td></td><td></td></tr><tr><td>PairFlow + DCD</td><td> $1 + \mathrm { t a i l }$ </td><td>44.32</td><td> $\approx 9 9 . 5 6 $ </td><td></td><td></td></tr><tr><td> $\mathrm { U D L M + R e D i }$ </td><td> $1 + \mathrm { t a i l }$ </td><td>5.83</td><td> $\approx 1 0 0 . 0 0$ </td><td></td><td></td></tr><tr><td>PairFlow + ReDi</td><td> $1 + \mathrm { t a i l }$ </td><td>35.25</td><td> $\approx 9 9 . 5 8 $ </td><td></td><td></td></tr><tr><td>One-step SMILES generationd</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Euclidean Laplace drifting</td><td>1</td><td>22.0</td><td>40.0</td><td></td><td></td></tr><tr><td>Kernel-gradient drifting</td><td>1</td><td>38.9</td><td>44.1</td><td></td><td></td></tr><tr><td>One-step graph generation</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CFM-CSD (Roos et al., 2026)</td><td>1</td><td>90.0</td><td>91.8</td><td>2.14</td><td></td></tr><tr><td>CFM-ECLD (Roos et al., 2026)</td><td>1</td><td>95.8</td><td>89.2</td><td>2.16</td><td></td></tr><tr><td>MolGAN (Cao and Kipf,</td><td>1</td><td>98.1</td><td>10.4</td><td></td><td></td></tr><tr><td>2018) MoFlow (Zang and Wang,</td><td>1</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $9 9 . 2 0 \pm 0 . 1 2 $ </td><td></td><td></td></tr><tr><td>2020) RLOOe</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> ${ \mathrm { R L O O } } + { \mathrm { r a r i t y } } ^ { e }$ </td><td>1 1</td><td> $8 8 . 2 6 \pm 3 . 1 2$   $9 6 . 8 0 \pm 2 . 1 9$ </td><td> $5 0 . 6 8 \pm 9 . 2 1$   $5 1 . 3 2 \pm 3 . 1 4$ </td><td> $3 . 9 0 7 \pm 0 . 1 6 2$   $2 . 7 1 0 \pm 0 . 5 3 5$ </td><td>≈3,927 ≈ 4,425</td></tr><tr><td> $\mathbf { E G O } \left( 3 \times 7 6 8 \right)$ </td><td>1</td><td> $9 3 . 4 6 \pm 0 . 8 2$ </td><td> $8 3 . 6 9 \pm 1 . 4 9$ </td><td> $3 . 5 3 4 \pm 0 . 6 3 8$ </td><td> $\approx 6 { , } 7 4 6$ </td></tr></table>

Published graph protocols. In Table 8, n denotes heavy atoms and L the token sequence length; iterative steps need not equal network evaluations. For rows marked <sup>a</sup>, GDSS and GruM report validity before correction, whereas uniqueness and FCD use corrected outputs. GDSS VU/s uses a TITAN RTX and 20 CPU cores, while EGO uses an A100 and includes generation through SMILES/CSV, excluding later evaluation. For CatFlow, marked <sup>b</sup>, the step count follows Roos et al. (2026). DiGress, Cometh, and CFM use largest component identities. DeFoG reports standard validity and uniqueness among relaxed valid molecules using largest component identities. MELD reports 10,000 generated samples; its QM9 sampling step count and compatible VU/s are left unavailable here. MoFlow retains the validity correction used in its original Table 2, with mean and reported SD over five generation runs of 10,000 molecules (Zang and Wang, 2020). No throughput from a different MoFlow quality evaluation is attached to that row.

Published SMILES protocols. The four rows marked <sup>c</sup> come from Park et al. (2025) and use ten trials of 1,024 proposals. Validity is 100V/<sup>¯</sup> 1,024, and reconstructed uniqueness is $1 0 0 \hat { U } / \hat { V } ,$ , a ratio of mean counts. Here, U counts distinct valid SMILES strings without canonicalization. One reported sampling step is followed by greedy tail denoising, and the released sampler adds a final network evaluation. The rows marked <sup>d</sup> come from Esteban-Casadevall et al. (2026); kernel gradient drifting denotes the spherical Laplace gradient variant. Both drifting rows use canonical largest fragment uniqueness among valid samples, and their evaluation budgets differ from 10,000 proposals.

EGO and REINFORCE protocols. The rows marked <sup>e</sup> and EGO use full molecule identities. REINFORCE conditional uniqueness is aggregated over the same three training seeds. These training recipes match update counts, but differ in reward scalarization, hard reward evaluation counts, and training time. Section D.2.3 reports strict yields, checkpoint selection, and training costs.

EGO protocol and generation time. EGO uses three hidden layers of width 768, with 1,456,380 parameters and a canvas of nine heavy atoms. Molecular sizes follow the training distribution. Each seed completes 120,000 updates and uses a checkpoint selected by validation. Count metrics use 10,000 raw proposals, without rejection sampling or graph repair. FCD uses the first 10,000 full valid molecules from a separate bank of 30,000 proposals, retains duplicates, and compares them with 10,000 test reference molecules. These sample counts are met in all three seeds. The mean generation time through SMILES/CSV is $1 . 1 5 9 \pm \mathrm { { \bar { 0 } } . 0 0 8 }$ seconds on an A100 40GB, corresponding to approximately 6,746 valid unique outputs per generation second. This ratio uses the mean useful count and mean time, and excludes later deduplication, novelty checks, and distribution scoring.

Key takeaway. EGO combines 93.46±0.82% validity with 83.69±1.49% conditional uniqueness in a single network evaluation. Several published generators achieve higher validity or lower FCD, while both REINFORCE recipes produce more repetitions in this comparison. Table 26 shows the associated cost tradeoff: compared with CFM-ECLD, EGO has approximately 52.8× higher estimated VU efficiency, with VU lower by 7.24 percentage points and FCD higher by 1.37. Differences in source protocols and hardware prevent a direct runtime ranking from Table 8.

## D.2.2 NETWORK SIZE AND TRAINING BUDGET

Motivation. The molecular size sweep changes the distribution of graph sizes while keeping the network fixed. Here, we instead vary the depth and width of the EGO generator on QM9 to test whether more network capacity improves molecular quality and distinct yield. We also evaluate three training budgets to check whether a compact network can retain the quality of the main configuration and whether additional updates help each architecture.

Protocol. We evaluate five EGO architectures at 30,000, 60,000, and 120,000 updates, with training seeds 42, 43, and 44 for every architecture. The notation $L \times H$ gives the number of hidden layers and their width. The main configuration remains $3 \times 7 6 8 .$ as in Table $\begin{array} { r } { 8 ; } \end{array}$ the other networks vary in depth or width. The $2 \times 7 6 8$ network is a compact alternative, reported alongside every tested architecture. Equal update counts do not imply equal computation across network sizes. Differences between means over three seeds do not by themselves establish statistically significant superiority.

Each budget evaluates the checkpoint selected by validation within that budget, which can precede the final update. Count metrics use 10,000 raw proposals per seed. Full validity requires chemical sanitization of the complete output, and strict validity additionally requires connectedness. Strict VUN divides the number of distinct valid connected molecules absent from training by all proposals; it is not uniqueness conditioned on validity. Molecular sizes follow the distribution in the training data. A separate bank of 30,000 raw proposals provides up to 10,000 full valid molecules for FCD, retaining duplicates and using the test reference. FCD is a distance metric and does not share the denominator of the primary proposal bank. For $3 \times 7 6 8$ , every seed uses 10,000 generated molecules and 10,000 test reference molecules for FCD. The prepared split has 107,109 source training rows, including 107,061 unique molecules, and 13,374 test rows; external studies can use other splits.

Architecture comparison. Table 9 reports all five EGO architectures at 120,000 updates.

Table 9: Complete EGO network size sweep on QM9 at the 120,000 update budget. Metrics are mean ± sample standard deviation over seeds 42, 43, and 44. All columns except parameters and FCD are percentages. Parameter counts are rounded to millions; selected checkpoints can precede the budget milestone.
<table><tr><td>Architecture</td><td>Params. (M)</td><td>Full valid</td><td>Strict valid</td><td>Strict VUN</td><td>FCD↓</td></tr><tr><td> $2 \times 7 6 8$ </td><td>0.866</td><td> $9 4 . 0 5 \pm 1 . 3 6$ </td><td> $9 2 . 0 3 \pm 0 . 8 1$ </td><td> $6 8 . 7 0 \pm 0 . 9 8$ </td><td> $3 . 4 1 4 \pm 0 . 3 3 6$ </td></tr><tr><td> $3 \times 7 6 8$ </td><td>1.456</td><td> $9 3 . 4 6 \pm 0 . 8 2$ </td><td> $8 9 . 4 9 \pm 0 . 2 2$ </td><td> $6 8 . 3 1 \pm 1 . 0 3$ </td><td> $3 . 5 3 4 \pm 0 . 6 3 8$ </td></tr><tr><td> $4 \times 7 6 8$ </td><td>2.047</td><td> $9 3 . 5 0 \pm 0 . 3 1$ </td><td> $8 9 . 2 9 \pm 1 . 4 9$ </td><td> $6 7 . 1 0 \pm 1 . 9 4$ </td><td> $3 . 9 6 6 \pm 0 . 7 8 9$ </td></tr><tr><td> $3 \times 3 8 4$ </td><td>0.433</td><td> $9 4 . 3 0 \pm 1 . 1 3$ </td><td> $9 1 . 2 5 \pm 1 . 7 2$ </td><td> $6 7 . 0 6 \pm 2 . 1 6$ </td><td> $3 . 4 5 2 \pm 0 . 2 0 0$ </td></tr><tr><td> $3 \times 1 0 2 4$ </td><td>2.466</td><td> $9 3 . 0 6 \pm 0 . 7 2$ </td><td> $8 9 . 3 5 \pm 1 . 3 3 $ </td><td> $6 6 . 5 0 \pm 3 . 4 2$ </td><td> $3 . 6 7 9 \pm 0 . 2 9 3$ </td></tr></table>

The 2 × 768 EGO network has about 41% fewer parameters than the main $3 \times 7 6 8$ network. Its mean Strict VUN is higher by 0.39 percentage points, corresponding to about 39 molecules per 10,000 proposals. The ordering changes when connectedness is not required: full VUN is 71.99% for $3 \times 7 6 8$ and 70.57% for $2 \times 7 6 8 .$ . The main configuration has higher full VUN in every seed, with an aggregate of $7 1 . 9 9 \pm 0 . 8 5 \% ;$ only the mean was supplied for the compact configuration.  
The main EGO configuration has 1,456,380 parameters. Its connectedness is $9 5 . 9 3 \pm 0 . 6 5 \%$ conditional uniqueness among full valid outputs is $8 3 . 6 9 \pm 1 . 4 9 \%$ , and novelty among those unique outputs is $9 2 . 0 \bar { 7 } \pm 2 . 0 1 \%$ . Full VU is $7 8 . 2 \bar { 1 } \pm 0 . 8 4 \%$ , while strict unique yield is $7 \bar { 4 } . 5 3 \pm 1 . 1 \bar { 7 } \% ;$ novelty among strict unique outputs is $9 1 . 6 8 \pm 2 . 1 1 \%$ . These quantities use different denominators and should not be interchanged with Strict VUN. FCD against the training reference is $3 . 5 0 1 \pm 0 . 6 0 6$ separate from the test reference FCD in Table 9. The descriptor KL score is $0 . 9 1 2 \pm 0 . 0 2 2$ and uses the training reference.

Training budget and checkpoint selection. Table 10 shows all five trajectories across the three budgets. The milestones are repeated evaluations within each training run, rather than independent training repetitions. Strict VUN increases at both budget transitions in all 15 runs, across the five architectures and three seeds.

Table 10: EGO Strict VUN (%) on QM9 across training budgets, averaged over seeds 42, 43, and 44. Each result uses the checkpoint selected by validation within its budget. The dispersion at 120,000 updates is reported in Table 9.
<table><tr><td>Architecture</td><td>30,000 updates</td><td>60,000 updates</td><td>120,000 updates</td></tr><tr><td> $2 \times 7 6 8$ </td><td>63.04</td><td>66.23</td><td>68.70</td></tr><tr><td> $3 \times 7 6 8$ </td><td>63.19</td><td>66.19</td><td>68.31</td></tr><tr><td> $4 \times 7 6 8$ </td><td>62.54</td><td>64.33</td><td>67.10</td></tr><tr><td> $3 \times 3 8 4$ </td><td>61.02</td><td>63.40</td><td>67.06</td></tr><tr><td> $3 \times 1 0 2 4$ </td><td>62.16</td><td>64.66</td><td>66.50</td></tr></table>

At the 120,000 update budget, the selected updates for $2 \times 7 6 8$ are 118,000, 120,000, and 116,000 for seeds 42, 43, and 44, respectively. For $3 \times 7 6 8$ , they are 82,000, 90,000, and 100,000. These choices distinguish the completed training budget from the checkpoint used for evaluation. A larger budget does not improve every metric: $4 \times 7 6 8$ has worse FCD at 120,000 than at 60,000 updates despite its higher Strict VUN. The numerical FCD at 60,000 updates and the metric dispersions at the lower budgets were not supplied in this summary.

The earlier result of 64.39 ± 1.85% Strict VUN for $3 \times 7 6 8$ at 60,000 updates belongs to a separate record. It is not pooled with, or substituted for, the 66.19% mean at that budget in the present sweep.

Generation cost. Table 11 compares estimated dense work and measured generation time for the same five EGO architectures at 120,000 updates. The accounting follows Section D.5, and the timing scope follows Section D.2.1. This comparison tests whether the gain from reducing dense work also appears in the time required to obtain molecular outputs.

Table 11: EGO inference cost across the QM9 network size sweep at 120,000 updates. Dense MFLOP is the estimated work per proposal, and $E _ { \mathrm { V U } }$ is the number of distinct valid outputs per $1 0 ^ { 1 2 }$ estimated dense operations. Times are seconds per 10,000 proposals through SMILES/CSV on an A100 40GB, with mean ± sample SD over seeds 42, 43, and 44. VU/s is the mean distinct valid count divided by the mean time.
<table><tr><td colspan="4">Dense MFLOP</td></tr><tr><td>EGO architecture</td><td>↓</td><td> $E _ { \mathrm { V U } } \uparrow$ </td><td> ${ \mathrm { T i m e } } \left( { \mathrm { s } } \right) \downarrow$ </td><td>VU/s ↑</td></tr><tr><td> $3 \times 7 6 8 ( \mathrm { p r i m a r y } )$ </td><td>2.908</td><td>8268,980.3</td><td> $1 . 1 5 9 \pm 0 . 0 0 8 \approx 6 { , } 7 4 6$ </td><td></td></tr><tr><td> $2 \times 7 6 8$ </td><td></td><td>1.728445,409.0</td><td> $1 . 1 4 4 \pm 0 . 0 0 8 \approx 6 { , } 7 2 5$ </td><td></td></tr><tr><td> $3 \times 3 8 4$ </td><td></td><td>0.864 880,169.8</td><td> $1 . 1 6 4 \pm 0 . 0 1 0 \ \approx 6 { , } 5 3 6$ </td><td></td></tr><tr><td> $4 \times 7 6 8$ </td><td></td><td>4.087 186,692.3</td><td> $1 . 1 6 1 \pm 0 . 0 0 9 \approx 6 { , } 5 7 4$ </td><td></td></tr><tr><td> $3 \times 1 0 2 4$ </td><td></td><td>4.925 157,535.8</td><td> $1 . 1 6 7 \pm 0 . 0 1 3 \approx 6 { , } 6 4 7$ </td><td></td></tr></table>

Dense work includes all hidden linear maps and both output heads, with two operations per multiplyand-accumulate. It excludes biases, activations, feature construction, sampling, and chemistry processing. The reported $E _ { \mathrm { V U } }$ is the mean across seeds and uses full VU, without a connectedness or novelty requirement. VU/s is a ratio of means, not the mean of paired rates; no rate standard deviation is inferred from separate aggregate measurements.

The main 3 × 768 EGO generator takes $1 . 1 5 9 \pm 0 . 0 0 8$ seconds through SMILES/CSV, including $0 . 0 3 6 4 8 \pm 0 . 0 0 0 4 2$ seconds for raw graph generation. Its mean full valid unique count is 7,821, and its mean Strict VUN count is 6,831. Dividing these counts by the unrounded mean generation time gives approximately 6,746 VU/s and 5,892 Strict VUN outputs per generation second. The timer includes graph generation, host transfer, chemistry conversion, and SMILES/CSV writing, but excludes loading, compilation, warmup, and later screening. Useful counts come from the later evaluation, so these rates do not measure the complete screening process. The total time spent in optimizer updates over the 120,000 update budget is $1 , 0 7 9 . 5 6 \pm 4 . { \overset { \cdot } { 4 } } { \overset { \cdot } { 3 } }$ seconds, excluding preparation, compilation, validation, and checkpointing. The separately reported evaluation timer is $1 1 4 . 1 4 \pm 0 . 7 2$ seconds and is not used as generation time.

Mean generation times range from 1.144 to 1.167 seconds per 10,000 proposals across the tested networks. The 3 × 384 EGO network has higher VU per estimated dense work than the primary configuration but produces approximately 6,536 VU/s, compared with $6 { , } 7 4 6$ for $3 \times 7 6 8$ . Fewer dense operations therefore do not imply higher measured throughput through SMILES/CSV in these runs.

Key takeaway. A larger EGO network does not consistently improve molecular quality or Strict VUN on QM9. The $2 \times 7 6 8$ network uses about 41% fewer parameters than the main configuration and reaches a similar mean Strict VUN, while the main configuration retains higher full VUN. Longer training improves Strict VUN at both budget transitions in every run, although FCD does not always improve. Compact networks reduce estimated dense work, while the measured times through molecular conversion remain similar across architectures.

## D.2.3 REINFORCE COMPARISON AT 120,000 UPDATES

Motivation. REINFORCE can optimize rewards on hard molecular outputs without differentiating through sampling. We first compare EGO with REINFORCE under the same architecture and update count to test whether this direct reward access also preserves a large pool of distinct novel molecules. This experiment compares the full training recipes; the following experiment then matches the reward and the number of generated graphs scored.

Why leave-one-out. We use REINFORCE with a leave-one-out baseline (RLOO) to control the variance of the gradient estimate. Here, vanilla REINFORCE means REINFORCE without a baseline.

Our reward is defined over a batch and includes pairwise Energy terms. Without a baseline, each sampled molecule is weighted by the full batch reward, including variation caused by the other molecules. For each molecule, RLOO subtracts the reward recomputed after removing that molecule and renormalizing the remaining terms. The name refers to this removal of one molecule at a time. This measures a contribution relative to the remaining batch. The baseline reuses the sampled graphs and distance matrices, so it requires no additional generated molecules or learned critic. Throughout this appendix, “REINFORCE” refers to our implementation with a leave-one-out baseline (RLOO), unless explicitly described as vanilla REINFORCE. We use this baseline with the aim of reducing gradient variance in the comparison with EGO. The estimator preserves the expected gradient under the conditions stated in Section D.2.4. We do not measure its variance or performance relative to vanilla REINFORCE in these experiments.

Protocol and training objective. We compare EGO with RLOO, with and without rarity, using the same one-step QM9 generator: three hidden layers of width 768, 1,456,380 parameters, a canvas of nine heavy atoms, and a training-derived size prior. Each configuration uses seeds 42, 43, and 44 and completes 120,000 updates. REINFORCE uses 96 policy samples and 192 reference samples per update. Final count metrics use 10,000 raw proposals per seed, with no rejection sampling or graph repair; FCD uses a separate sampling bank. REINFORCE scores the hard Energy criterion and structural penalties directly, with feature scales fixed from training data. The rarity variant adds the hashed archive reward with weight 0.20, applied only to valid graphs.

Reward evaluation. The leave-one-out estimator is defined in Section D.2.4. For this experiment, feature scales and the rarity archive remain fixed while an update is scored, and archive entries are committed afterward. Log probabilities include only active node and edge decisions. The runs use no advantage whitening, entropy bonus, or gradient clipping.

Validity, distinct yield, and fidelity. Table 12 shows a tradeoff: RLOO with rarity achieves higher full and strict validity and lower FCD, whereas EGO produces more distinct valid molecules and more distinct novel connected molecules. EGO reaches 68.31% Strict VUN versus 42.07% for REINFORCE with rarity, a difference of 26.24 percentage points and a ratio of approximately 1.62×. This corresponds to about 6,831 versus 4,207 distinct novel connected molecules per 10,000 attempts. Conditional uniqueness among full valid outputs is 83.69% for EGO and 51.32% for REINFORCE with rarity; these percentages have a different denominator from strict unique yield. FCD evaluates the feature distribution of valid outputs while retaining duplicates, so lower FCD can coexist with lower distinct yield.

Table 12: QM9 training comparison at 120,000 updates: mean ± sample SD across seeds 42, 43, and 44. All columns except raw FCD are percentages. Strict validity requires chemical validity and connectedness. Strict unique yield divides distinct valid connected molecules by all attempts; Strict VUN additionally requires absence from training. Bold values identify the best mean within this comparison.
<table><tr><td></td><td></td><td></td><td>Strict unique</td><td></td><td></td></tr><tr><td>Method</td><td>Full valid ↑</td><td>Strict valid ↑</td><td>yield ↑</td><td>Strict VUN ↑</td><td>FCD↓</td></tr><tr><td>EGO</td><td> $9 3 . 4 6 \pm 0 . 8 2$ </td><td> $8 9 . 4 9 \pm 0 . 2 2 $ </td><td> ${ \bf 7 4 . 5 3 \pm 1 . 1 7 }$ </td><td> ${ \bf 6 8 . 3 1 \pm 1 . 0 3 }$ </td><td> $3 . 5 3 4 \pm 0 . 6 3 8$ </td></tr><tr><td>RLOO</td><td> $8 8 . 2 6 \pm 3 . 1 2$ </td><td> $8 5 . 8 9 \pm 2 . 4 3$ </td><td> $4 2 . 5 8 \pm 7 . 0 9$ </td><td> $3 9 . 3 9 \pm 7 . 6 1$ </td><td> $3 . 9 0 7 \pm 0 . 1 6 2$ </td></tr><tr><td> $\mathrm { R L O O } + \mathrm { r a r i t y }$ </td><td> ${ \bf 9 6 . 8 0 \pm 2 . 1 9 }$ </td><td> ${ \bf 9 5 . 7 7 \pm 2 . 1 0 }$ </td><td> $4 8 . 8 6 \pm 4 . 1 5$ </td><td> $4 2 . 0 7 \pm 2 . 3 8$ </td><td> $\mathbf { 2 . 7 1 0 \pm 0 . 5 3 5 }$ </td></tr></table>

Effect of rarity. Adding rarity to REINFORCE increases strict validity by 9.88 percentage points and strict unique yield by 6.28 points, but Strict VUN increases by only 2.68 points. The gain in distinct valid connected outputs therefore does not translate into an equally large gain in distinct novel outputs. Rarity improves validity and distinct yield without closing the Strict VUN gap to EGO under this training recipe; this result does not imply that history-dependent rewards are ineffective for all score function methods.

Budget milestones and checkpoint selection. Each budget evaluates the best checkpoint found so far under the validation criterion. For REINFORCE, the selected parameters are already available at the 30,000 update milestone and remain unchanged at 60,000 and 120,000 updates. Without rarity, the selected update is 2,000 in all three seeds. With rarity, the selected updates are 28,000, 30,000, and 10,000 for seeds 42, 43, and 44. No subsequent checkpoint improves that criterion, and the fixed generation banks reproduce the same metrics for the same parameters. All RLOO runs nevertheless complete 120,000 updates and 11,520,000 policy graph draws. For EGO, the selected updates are 82,000, 90,000, and 100,000 for seeds 42, 43, and 44. The selected checkpoints should therefore be distinguished from the training budget consumed.

Training cost and scope of the comparison. REINFORCE update time is approximately 361 to 366 seconds across the two configurations, compared with approximately 1,080 seconds for EGO. These timers exclude compilation/warmup, validation, checkpointing, and subsequent generation and evaluation. The comparison matches architecture and update count, but not training time or objective evaluation workload. EGO uses dynamic Energy scales and centered rank fitness shaping, whereas REINFORCE uses fixed scales and a raw scalar reward. Consequently, this experiment compares complete training recipes and does not isolate the estimator alone. It demonstrates that direct access to hard molecular criteria can yield high validity and favorable FCD without preserving the distinct novel yield achieved by EGO in this comparison. All variants retain one-step inference; no inference speedup of EGO over REINFORCE is claimed.

Distribution metrics. Validity, yield, and raw FCD evaluations are complete. Only the optional descriptor KL metric is incomplete and is omitted from this comparison.

Key takeaway. At 120,000 updates, EGO produces about 1.62× as many distinct novel connected molecules as RLOO with rarity, while RLOO with rarity has higher validity and lower FCD. The difference concerns the complete training recipes, including their reward scaling, fitness shaping, and training work. The next comparison checks whether the advantage in distinct yield remains under a shared reward and matched reward evaluation budgets.

## D.2.4 REINFORCE COMPARISON AT MATCHED REWARD BUDGETS

Motivation. The comparison at 120,000 updates gives each method a different number of reward evaluations. We therefore run a separate experiment to ask whether EGO still produces more distinct novel molecules when both methods score the same number of generated graphs with the same reward. We also measure validity and FCD to show the quality tradeoffs that can accompany a gain in distinct yield.

Protocol and shared training objective. We compare EGO with RLOO in a dedicated QM9 experiment with five seeds, separate from the runs used for the main molecular generation benchmarks. Both methods use the same one-step generator: three hidden layers of width 768, 1,456,380 parameters, a canvas of nine heavy atoms, and a training-derived size prior. The prepared QM9 split and initialization are paired by seeds 42 to 46. Both optimize the same hard Energy reward minus the same structural penalties, with feature scales computed once from 8,192 deterministic training pairs and then held fixed. Rarity and rank shaping are disabled in both methods. The recorded definitions of the reward and feature scales agree across all ten runs.

Leave-one-out estimator. For a batch reward $R _ { B }$ evaluated on samples $X _ { 1 : B }$ , RLOO uses

$$
A _ { i } = R _ { B } ( X _ { 1 : B } ) - R _ { B - 1 } ( X _ { - i } ) , \qquad { \mathcal { L } } _ { \mathrm { L O O } } = - \sum _ { i = 1 } ^ { B } { \mathrm { s t o p g r a d } } ( A _ { i } ) \log p _ { \theta } ( X _ { i } \mid z _ { i } , n _ { i } ) .\tag{33}
$$

The delete-one reward removes both the row and column of sample i from the Energy distances between generated samples and renormalizes the U-statistic. Feature scales and positive references remain fixed while scoring the update. Log probabilities include only active node and edge decisions. RLOO uses 96 policy graphs and 192 positive training references per update.

Conditional on the current parameters and training history, latent inputs, requested sizes, fixed feature scales, and reference panel, the generated graphs are sampled independently. The baseline $b _ { i } = R _ { B - 1 } ( X _ { - i } )$ therefore does not depend on the sampled graph $X _ { i }$ . Its contribution to the expected gradient is zero:

$$
\mathbb { E } [ b _ { i } \nabla _ { \theta } \log p _ { \theta } ( X _ { i } \mid z _ { i } , n _ { i } ) ] = 0 .\tag{34}
$$

Subtracting this baseline leaves the expected gradient of the batch reward unchanged, following the baseline principle of REINFORCE (Williams, 1992). Keeping feature scales and, when used, the rarity archive fixed during the update is necessary for this independence. LOO is thus a variance control within REINFORCE, with the same hard reward and no learned value model.

Evolutionary updates and optimization settings. EGO uses 96 antithetic rank-one perturbation pairs (192 candidate networks), with 96 generated graphs per candidate. Candidates share the positive references and generation random numbers within an update. Updates use the numerical differences between paired rewards, without a rank transformation. Matrix perturbation scales are $\operatorname* { m a x } ( 0 . 1 \operatorname { R M S } ( \theta _ { \ell } ) , \bar { 1 0 ^ { - 3 } } )$ , recomputed at each update; vector parameters use scale 0.01. Candidates are evaluated in chunks of 16. Both methods use Adam with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 .$ , and $\epsilon = 1 0 ^ { - 8 }$ with learning rates $1 0 ^ { - 2 }$ for EGO and $3 \times 1 0 ^ { - 4 }$ for RLOO. Neither method uses advantage whitening, an entropy bonus, or gradient clipping. The comparison therefore concerns the estimator and update recipes, including method-specific optimization settings.

Matched reward budgets and checkpoint selection. An RLOO update scores 96 generated graphs, whereas an EGO update scores $1 9 2 \times \mathrm { { \bar { 9 6 } } = 1 8 , 4 3 2 }$ . At the largest budget, 11,520,000 evaluations per method and seed, EGO completes 625 updates, and RLOO completes 120,000. Every reported result evaluates the current parameters at the declared boundary, without substituting a validation-selected checkpoint.

Matching these evaluations controls the number of generated graphs scored by the shared reward, not total FLOPs or elapsed training time. Reference reuse and gradient computation differ. For example, EGO presents 120,000 positive reference rows at the largest budget, versus 23,040,000 for RLOO, because its candidates reuse each reference panel. Thus the reference sampling rule is shared, but the total number of reference presentations is not matched.

Evaluation and uncertainty. Each checkpoint produces 10,000 raw proposals per seed, without rejection sampling or graph repair. Strict validity requires chemical validity and connectedness. Strict unique yield divides the number of distinct valid connected molecules by all attempts; Strict VUN additionally requires absence from training. FCD uses the first 10,000 chemically valid outputs from a separate bank of 30,000 raw proposals, retaining duplicates, and a held-out reference sample. All endpoint FCD evaluations meet this required sample count. Descriptor KL is omitted because its required number of unique samples is not available for all five paired runs.

We report means and sample standard deviations across seeds 42 to 46. Paired contrasts are EGO minus RLOO. Their two-sided 95% Student-t intervals use the five paired differences and four degrees of freedom. These are descriptive, pointwise intervals without correction across budgets or metrics.

Table 13: QM9 comparison at 11,520,000 hard reward graph evaluations per method and seed: mean ± sample SD over seeds 42 to 46. The current checkpoints are at 625 updates for EGO and 120,000 for RLOO. All columns except held-out raw FCD are percentages; yield metrics use all 10,000 proposals as the denominator. Both methods use the same reward and fixed scales, without rarity or rank shaping.
<table><tr><td>Method</td><td>Full valid ↑</td><td>Strict valid ↑</td><td>Strict unique yield ↑</td><td>Strict VUN ↑</td><td>FCD↓</td></tr><tr><td>EGO</td><td> $7 6 . 7 1 \pm 3 . 6 7$ </td><td> $7 0 . 7 0 \pm 2 . 5 9$ </td><td> $4 0 . 7 1 \pm 2 . 5 3$ </td><td> $3 8 . 3 9 \pm 2 . 8 9$ </td><td> $5 . 2 5 0 \pm 0 . 6 6 0$ </td></tr><tr><td>RLOO</td><td> $9 8 . 6 4 \pm 0 . 8 6$ </td><td> $9 7 . 7 7 \pm 1 . 0 4$ </td><td> $1 4 . 0 0 \pm 1 . 0 5$ </td><td> $1 2 . 0 5 \pm 0 . 5 3$ </td><td> $4 . 1 6 1 \pm 0 . 5 0 8$ </td></tr></table>

Validity, distinct yield, and fidelity. Table 13 reports the final comparison. At the largest matched reward budget, EGO reaches 38.39±2.89% Strict VUN versus 12.05±0.53% for RLOO. The paired difference is +26.34 percentage points, with a 95% interval of [22.81, 29.86], and is positive in all five seeds. This corresponds to approximately 3,839 versus 1,205 distinct novel connected molecules per 10,000 attempts. The strict unique yield is also higher, 40.71 ± 2.53% versus 14.00 ± 1.05%.

RLOO instead attains higher strict $\mathrm { v a l i d i t y , 9 7 . 7 7 \pm 1 . 0 4 \% }$ versus $7 0 . 7 0 \pm 2 . 5 9 \%$ , and a lower mean FCD, $4 . 1 6 1 \pm 0 . 5 0 8$ versus $5 . 2 5 0 \pm 0 . { \dot { 6 } } 6 0$ . The paired FCD difference is +1.089, with interval [−0.015, 2.193], which includes zero. Thus, the lower mean FCD is less conclusive than the paired differences in Strict VUN and strict validity. Because FCD retains repeated valid outputs, lower FCD can coexist with a lower yield of distinct molecules.

Key takeaway. The control supports a yield advantage for the update recipe of EGO at the matched reward budgets even with identical reward construction and counts of generated graphs scored. Method-specific learning rates, perturbations, and reference reuse remain part of the compared recipes, so the result does not isolate the estimator independently of these choices. It also does not establish uniform superiority across budgets, a wall clock speedup, or an inference cost advantage over RLOO: both methods retain the same one-step generator.

## D.3 ZINC250K

ZINC250K is a subset of the ZINC molecular collection (Irwin et al., 2012). It extends the evaluation from the small molecules in QM9 to graphs with up to 38 heavy atoms. We use it to test whether EGO can maintain a high yield of valid and distinct molecules on larger graphs. The external comparison measures generation quality and estimated inference cost. A separate experiment compares training methods under equal active training time.

## D.3.1 COMPARISON WITH PUBLISHED GENERATORS

Table 14 provides the full ZINC250K comparison with published generators. It includes autoregressive generation, iterative diffusion and flow, and one-step generators. The table reports chemical validity, uniqueness among valid molecules, FCD, and measured valid and unique outputs per second where available. External rows retain the evaluation protocols of their sources. Estimated inference cost is reported separately in Table 26; it is not a runtime measurement.

Table 14: Full ZINC250K comparison with published generators. Validity and conditional uniqueness are percentages, FCD is a distance, and VU/s is a measured output rate. EGO uses 120,000 training updates and reports mean ± sample SD where available. Its FCD is a single recovered evaluation. Gray rows identify one-step generators; dashes denote unavailable measurements.
<table><tr><td>Method</td><td>Steps</td><td></td><td>Valid (%) ↑ Unique (%) ↑ FCD↓</td><td></td><td>VU/s ↑</td></tr><tr><td colspan="6">Autoregressive generation and autoregressive diffusion</td></tr><tr><td>GraphARM (Kong et al., 2023)</td><td> $n \leq 3 8$ </td><td>88.23</td><td>99.46</td><td>16.26</td><td></td></tr><tr><td>LO-ARM-st-sep (Wang et al., 2025)</td><td>D(n)</td><td>96.26</td><td>100.00</td><td>3.229</td><td></td></tr><tr><td>PARD (Zhao et al., 2024)</td><td>blockwise</td><td>95.23</td><td>99.99</td><td>1.98</td><td></td></tr><tr><td colspan="6">Iterative diffusion and flow</td></tr><tr><td>GDSSª (Jo et al., 2022)</td><td>1,000</td><td>97.01</td><td>99.64</td><td>14.656</td><td>≈4.93</td></tr><tr><td>DiGressb (Jo et al., 2024)</td><td>1,000</td><td>94.99</td><td>99.97</td><td>3.482</td><td></td></tr><tr><td>GruM (Jo et al., 2024)</td><td>1,000</td><td>98.65</td><td>99.97</td><td>2.257</td><td></td></tr><tr><td>CatFlow (Eijkelboom et al., 2024)</td><td>100</td><td>99.21</td><td></td><td>13.211</td><td></td></tr><tr><td>MELD (Seo et al., 2025)</td><td>500</td><td>100.00</td><td>99.99</td><td>1.51</td><td></td></tr><tr><td>DeFoG (Qin et al., 2025)</td><td>50</td><td>96.65</td><td>99.99</td><td>2.123</td><td></td></tr><tr><td>DeFoG (Qin et al., 2025)</td><td>500</td><td>99.22</td><td>99.99</td><td>1.425</td><td></td></tr><tr><td colspan="6">One-step generation</td></tr><tr><td>MoFlow (Zang and Wang, 2020)</td><td>1</td><td>100.00</td><td></td><td></td><td></td></tr><tr><td>CFM-CSDc (Roos et al., 2026)</td><td>1</td><td>73.5</td><td>99.9</td><td>19.3</td><td></td></tr><tr><td>CFM-ECLDc (Roos et al., 2026)</td><td>1</td><td>93.5</td><td>99.5</td><td>20.0</td><td></td></tr><tr><td>EGO (ours)</td><td>1</td><td>94.02 ± 3.94</td><td>96.42 ± 0.21</td><td>30.36d</td><td>2,525±160</td></tr></table>

Source protocols and timing. For LO-ARM, $D ( n ) = n ( n + 1 ) / 2$ counts node and edge decisions, and the reported row uses 16,384 samples. PARD generates blocks autoregressively and uses diffusion within each block. The Steps column therefore does not provide a uniform count of neural evaluations. <sup>a</sup>GDSS reports validity before correction, while its uniqueness, FCD, and VU/s use corrected outputs on a TITAN RTX with 20 CPU cores. <sup>b</sup>The DiGress row is the reproduction with 1,000 steps reported by GruM. <sup>c</sup>CFM computes uniqueness from identities of the largest component. MoFlow retains the molecular correction used in its source evaluation. <sup>d</sup>The EGO FCD is 30.3573 before rounding, from seed 42 at a budget of 120,000 updates.

EGO uniqueness is computed within each run among full valid molecules before aggregation. Values recovered from rounded logs are approximate. The EGO rate of $2 { , } 5 2 5 \pm 1 6 0$ VU/s is measured through molecular conversion and SMILES/CSV export, then aggregated across runs. It excludes later FCD and other distributional evaluations. Hardware and timing boundaries differ across sources, so these rates do not establish a speedup under matched conditions.

EGO configuration and evaluation. EGO uses three hidden layers of width 1,024, a canvas of 38 nodes, and a budget of 120,000 updates. Molecular sizes are sampled from the training distribution. This architecture was selected after inspecting the capacity sweep in Section D.3.2 because it had the highest observed mean full validity and full VU at this budget. The choice was therefore not fixed before that sweep. Training uses seven hard feature families, a population of 192 perturbed networks, fitness batches of 96 graphs, and a rarity weight of 0.2.

The historical preparation collects the atom state vocabulary and allowed valences before the split and applies no additional deduplication. A fixed permutation with seed 0 reserves up to 10,000 accepted rows for testing. The positive feature panel and validation references use separate portions of the remaining training pool. Checkpoint selection uses a fixed validation noise bank and first prefers full validity of at least 85% and connectivity of at least 97%. It then uses uniqueness of the largest component adjusted by feature MMD; before feasibility, it primarily rewards full validity. This differs from the protocol of the controlled size sweep in Section D.1.1.

Each final evaluation uses 10,000 raw proposals. Full validity tests the complete molecule, and full VU counts distinct full valid molecular identities per attempt. FCD retains all full valid outputs from this bank, including duplicates, and compares them with the valid held-out reference. It does not use the separate replenishment bank used for QM9. The FCD evaluation for this configuration is from seed 42.

Key takeaway. EGO reaches 94. $. 0 2 \pm 3 . 9 4 \%$ full validity and $9 6 . 4 2 \pm 0 . 2 1 \%$ uniqueness among valid molecules with one generator evaluation. Its full VU yield is $9 0 . 6 6 \pm 4 . 0 0 \%$ , with a measured rate of $2 { , } 5 2 5 \pm 1 6 0$ valid and unique molecules per second through export. Table 26 reports an estimated efficiency of about 208× the value for CFM-ECLD, while the mean VU yield of EGO is lower by 2.37 percentage points. The available FCD value of 30.36 is higher than 20.0 for CFM-ECLD and the values of the iterative methods in Table 14. EGO therefore combines high useful yield with low estimated inference cost, while distributional fidelity remains a limitation under the reported protocols.

## D.3.2 NETWORK SIZE AND TRAINING BUDGET

Motivation. We vary the network depth and width to test whether more capacity improves the validity, useful yield, and distributional fidelity of EGO. We also compare three training budgets to examine the effect of longer optimization. This study changes the generator network while keeping the molecular dataset fixed. The controlled molecular size sweep in Section D.1.1 asks a different question by changing the molecular mixture while keeping the network fixed.

Architectures and training. We compare five EGO architectures at 30,000, 60,000, and 120,000 updates. Architecture names give the number and width of the hidden layers. The campaign schedules seeds 42, 43, and 44 for each architecture, with intermediate milestones along continuous training trajectories. The original reference architecture is $3 \times 7 6 8$ . We use $3 \times 1 0 2 4$ in the main comparison because it has the highest observed mean full validity and full VU at 120,000 updates. This choice was made after inspecting the sweep summary. Equal update budgets do not imply equal computation across network sizes.

Training follows the historical ZINC preparation and EGO protocol on a canvas of 38 nodes. Molecular sizes are sampled from the training distribution. The configuration retains seven hard feature families, a population of 192 perturbed networks, fitness batches of 96 graphs, and a rarity weight of 0.2. The atom state vocabulary and allowed valences are collected before the data split, and the split has no additional deduplication. A fixed permutation with seed 0 reserves up to 10,000 accepted rows for testing. The positive feature panel and validation references use separate portions of the remaining training pool.

Checkpoint selection. A fixed validation noise bank is used to select checkpoints. The rule first prefers full validity of at least 85% and connectivity of at least 97%, then uses uniqueness of the largest component adjusted by feature MMD. Before these thresholds are met, it primarily rewards full validity. Each budget evaluates the best available checkpoint under this rule, which can differ from the last iterate. This selection rule differs from the rule used in the controlled molecular size sweep.

Metrics and available evidence. Each final evaluation uses 10,000 raw proposals. Full validity tests the complete molecule, and strict validity additionally requires connectivity. Connectivity is measured over all proposals, including invalid ones. Full VU counts distinct full valid molecular identities per attempt. Strict VUN counts distinct valid, connected molecules absent from the training reference per attempt.

Conditional uniqueness is computed within each seed as 100 times the full VU percentage divided by the full validity percentage. The mean and sample SD summarize these ratios rather than the ratio of the aggregate means. Reconstruction from rounded log percentages introduces a small approximation. FCD uses all full valid molecules from the original 10,000 proposals, including duplicates, against the valid held-out reference. The generated sample count therefore varies with validity, without the separate replenishment bank used for QM9.

The mean and sample SD use the available runs for each metric. The starred FCD of 3 × 1024 is a single available evaluation from seed 42. The starred Strict VUN and FCD values of 4 × 768 come from seed 44. These values have no sample SD, and missing measurements remain unavailable. Table 15 reports all architecture results at 120,000 updates.

Table 15: EGO architecture comparison on ZINC250K at 120,000 updates. Values are the mean ± sample SD over the available runs per metric. All columns except FCD are percentages. Unique given valid is computed within each run among full valid molecules before aggregation; reconstruction from rounded logs is approximate. Stars mark single available evaluations, with no sample SD; dashes denote unavailable metrics. The highlighted architecture is the profile used in the main comparison.
<table><tr><td>Architecture</td><td>Full valid</td><td>Strict valid</td><td>Connected</td><td>Unique given valid</td><td>1Full VU</td><td>Strict VUN</td><td>FCD↓</td></tr><tr><td rowspan="2">3 × 384</td><td>84.42</td><td>79.53</td><td>95.08</td><td>93.16</td><td>78.62</td><td>74.79</td><td>29.58</td></tr><tr><td>±4.62</td><td>±1.86</td><td>±2.76</td><td>±1.03</td><td>±3.88</td><td>±2.01</td><td>±1.49</td></tr><tr><td rowspan="2">2 × 768</td><td>90.72</td><td>86.71</td><td>95.93</td><td>97.06</td><td>88.08</td><td></td><td></td></tr><tr><td>±5.31</td><td>±7.76</td><td>±2.50</td><td>±0.68</td><td>±5.66</td><td></td><td></td></tr><tr><td rowspan="2">3 × 768</td><td>88.35</td><td>85.59</td><td>97.20</td><td>96.69</td><td>85.43</td><td></td><td></td></tr><tr><td>±2.75</td><td>±4.20</td><td>±2.88</td><td>±0.24</td><td>±2.86</td><td></td><td></td></tr><tr><td rowspan="2">4 × 768</td><td>91.23</td><td>86.39</td><td>95.16</td><td>94.89</td><td>86.56</td><td></td><td></td></tr><tr><td>±0.29</td><td>±1.83</td><td>±1.59</td><td>±1.00</td><td>±0.73</td><td>81.70*</td><td>28.89*</td></tr><tr><td>3 × 1024</td><td>94.02 ±3.94</td><td>90.84 ±6.97</td><td>96.82 ±3.03</td><td>96.42 ±0.21</td><td>90.66 ±4.00</td><td>88.22 ±6.86</td><td>30.36*</td></tr></table>

Effect of network capacity. At 120,000 updates, 3 × 1024 has the highest reported mean full validity, strict validity, and full VU in Table 15. It also has the highest reported Strict VUN among configurations for which this metric is available. The 2 × 768 model has the highest mean conditional uniqueness, $9 7 . 0 6 \pm 0 . 6 8 \% ,$ compared with 96.42 ± 0.21% for $3 \times 1 0 2 4$ . The choice of the wider model therefore reflects its higher useful yield rather than conditional uniqueness alone.

Relative to $3 \times 7 6 8$ , the wider network improves mean full validity, strict validity, and full VU by 5.67, 5.25, and 5.23 percentage points, respectively. However, its SD of 6.97 points for strict validity and 6.86 points for Strict VUN shows substantial variation across the available runs. Missing Strict VUN values for two architectures and incomplete FCD coverage further limit the comparison. The aggregates do not establish a statistically resolved advantage of the wider network or a benefit for distributional fidelity.

Effect of training budget. Table 16 reports full and strict validity at each training budget for all five architectures. It uses checkpoints selected on validation data within each budget, without pooling the milestones as independent runs.

Table 16: EGO validity on ZINC250K across training budgets. Each cell reports mean $\pm$ sample SD over the available runs. Each milestone evaluates a checkpoint selected on validation data within the corresponding budget. Improvements describe aggregate means, rather than verified improvements in every individual trajectory.
<table><tr><td>Architecture 30,000 updates 60,000 updates 120,000 updates</td><td></td><td></td><td></td></tr><tr><td colspan="4"> $F u l l \nu a l i d i t y ( \% )$ </td></tr><tr><td> $3 \times 3 8 4$ </td><td> $8 2 . 5 3 \pm 4 . 5 3$ </td><td> $8 3 . 6 2 \pm 4 . 2 7$ </td><td> $8 4 . 4 2 \pm 4 . 6 2$ </td></tr><tr><td> $2 \times 7 6 8$ </td><td> $8 5 . 6 2 \pm 6 . 9 8$ </td><td> $8 8 . 7 3 \pm 5 . 7 8$ </td><td> $9 0 . 7 2 \pm 5 . 3 1$ </td></tr><tr><td> $3 \times 7 6 8$ </td><td> $8 5 . 2 0 \pm 3 . 0 0$ </td><td> $8 6 . 2 3 \pm 2 . 3 6$ </td><td> $8 8 . 3 5 \pm 2 . 7 5$ </td></tr><tr><td> $4 \times 7 6 8$ </td><td> $8 7 . 8 2 \pm 2 . 0 5$ </td><td> $9 0 . 6 8 \pm 0 . 1 3$ </td><td> $9 1 . 2 3 \pm 0 . 2 9$ </td></tr><tr><td> $3 \times 1 0 2 4$ </td><td> $9 1 . 1 3 \pm 1 . 1 7$ </td><td> $9 2 . 5 4 \pm 2 . 2 9$ </td><td> $9 4 . 0 2 \pm 3 . 9 4$ </td></tr><tr><td colspan="4">Strict validity (%)</td></tr><tr><td> $3 \times 3 8 4$ </td><td> $7 7 . 2 0 \pm 1 . 8 7$ </td><td> $7 8 . 5 2 \pm 1 . 0 5$ </td><td> $7 9 . 5 3 \pm 1 . 8 6$ </td></tr><tr><td> $2 \times 7 6 8$ </td><td> $8 1 . 4 1 \pm 9 . 2 0$ </td><td> $8 4 . 6 6 \pm 8 . 1 7$ </td><td> $8 6 . 7 1 \pm 7 . 7 6$ </td></tr><tr><td> $3 \times 7 6 8$ </td><td> $8 2 . 1 1 \pm 4 . 5 8$ </td><td> $8 3 . 2 8 \pm 3 . 5 4$ </td><td> $8 5 . 5 9 \pm 4 . 2 0$ </td></tr><tr><td> $4 \times 7 6 8$ </td><td> $8 2 . 6 3 \pm 0 . 6 5$ </td><td> $8 5 . 7 1 \pm 1 . 7 2$ </td><td> $8 6 . 3 9 \pm 1 . 8 3$ </td></tr><tr><td> $3 \times 1 0 2 4$ </td><td> $8 7 . 9 9 \pm 4 . 1 1$ </td><td> $8 9 . 3 2 \pm 5 . 3 7$ </td><td> $9 0 . 8 4 \pm 6 . 9 7$ </td></tr></table>

Mean full and strict validity increase at both budget transitions for every architecture in Table 16. From 30,000 to 120,000 updates, full validity improves by 1.89 to 5.10 percentage points and strict validity by 2.33 to 5.30 points. For $3 \times 1 0 2 4 .$ , these gains are 2.89 and 2.85 points, respectively. The aggregate trend supports a benefit from longer optimization for validity. The available summaries do not establish improvement in every seed or a corresponding budget trend for Strict VUN or FCD.

Graph generation throughput. Table 17 reports the number of full valid and unique molecules per second of graph generation at 120,000 updates. For each available run $s ,$ the rate is $q _ { s } = r _ { s } y _ { s } / 1 0 0$ where $r _ { s }$ is the reported sampling rate and $y _ { s }$ is the full VU percentage. The table summarizes these rates across runs. We use the reported sampling rate instead of inverting the recorded generation time, which is rounded to 0.01 seconds in the CSV.

Molecular evaluation determines the valid and unique count after graph generation. The timing denominator excludes SMILES conversion, RDKit validity checks, canonicalization, and deduplication. It measures graph generation cost, while the throughput in Table 14 includes molecular conversion and export. The available summaries provide rates for every architecture except $3 \times 3 8 4$ . The $4 \times 7 6 8$ rate is based on two available runs.

Table 17: Full VU per second of graph generation for the EGO architectures on ZINC250K at 120,000 updates: mean ± sample SD of rates computed within each run. Molecular conversion and evaluation are excluded from the timing denominator. <sup>a</sup>The $4 \times 7 6 8$ aggregate uses two available runs. Dashes denote unavailable rates; the highlighted profile is the configuration used in the main table.
<table><tr><td></td><td>Architecture VU/s (graph generation only) ↑</td></tr><tr><td> $3 \times 3 8 4$ </td><td></td></tr><tr><td> $2 \times 7 6 8$ </td><td> $7 2 2 , 7 3 3 \pm 2 5 7 , 1 8 3$ </td></tr><tr><td> $3 \times 7 6 8$ </td><td> $8 4 1 , 6 1 9 \pm 2 8 , 3 2 3$ </td></tr><tr><td> $4 \times 7 6 8$ </td><td> $( 8 4 1 , 3 6 5 \pm 2 5 , 4 1 3 ) ^ { a }$ </td></tr><tr><td> $\mathbf { 3 \times 1 0 2 4 }$ </td><td> $8 6 2 , 1 4 2 \pm 4 6 , 1 0 9$ </td></tr></table>

Timing through molecular export. For $3 \times 1 0 2 4$ at 120,000 updates, Table 18 separates graph generation from generation through SMILES/CSV export. Both rates use the full VU yield of $\bar { 9 } 0 . 6 6 \pm 4 . 0 0 \%$ , with the rate computed within each run before aggregation. The broader timing boundary includes RDKit conversion and CSV export and excludes subsequent FCD and other distributional evaluations.

The approximate times for 10,000 raw proposals are 0.011 seconds for graph generation and 3.59 seconds through export. Graph generation accounts for about 0.3% of the reported time through export, so molecular conversion and export dominate this measurement. These rounded time summaries have no supplied SD. The throughput means are not reconstructed by dividing aggregate counts by the rounded times.

Table 18: Measured timing boundaries for $\mathrm { E G O ~ 3 ~ } \times \mathrm { 1 0 2 4 }$ on ZINC250K at 120,000 updates. Time refers to 10,000 raw proposals and is reported approximately. VU/s uses fully valid distinct outputs, and reports mean ± sample SD of rates computed within each run. The broader boundary includes molecular conversion and CSV export; subsequent distributional evaluation is excluded.
<table><tr><td>Timing boundary</td><td>Time (s)</td><td>Full VU/s ↑</td></tr><tr><td>Graph generation only</td><td> $\approx 0 . 0 1 1$ </td><td> $8 6 2 , 1 4 2 \pm 4 6 , 1 0 9$ </td></tr><tr><td>Generation through SMILES/CSV</td><td> $\approx 3 . 5 9$ </td><td> $2 { , } 5 2 5 \pm 1 6 0$ </td></tr></table>

Key takeaway. The wider $3 \times 1 0 2 4$ network has the highest reported mean full VU in this sweep, and more training updates improve mean validity across all five architectures. The available FCD results provide no evidence that greater width improves distributional fidelity. The timing measurements also show that molecular conversion and export dominate the cost of obtaining usable EGO outputs. The external comparison therefore uses the measured export rate in Table 18, while Table 17 describes graph generation alone. Differences in hardware and timing boundaries prevent these rates from establishing a speedup against external methods. The historical ZINC validity result of 93.20% comes from a separate run and is not pooled into this campaign.

## D.3.3 CONTROLLED MOLECULAR SIZE SWEEP

Why this experiment. We use the shared size sweep in Section D.1.1 to test whether EGO retains useful yield as the ZINC250K task includes larger molecules. The fixed architecture separates changes in the molecular mixture from changes in network capacity. The comparison also tests whether EGO needs the rarity reward to improve on the gradient baselines.

Results. Table 19 reports all twelve configurations. Mean Strict VUN for EGO decreases from 90.84% at k = 16 to 78.40% at k = 32, but remains above both gradient baselines at every cap. At $k = 3 2 .$ , Soft reaches higher strict validity than EGO, 87.35% versus 82.48%, but much lower Strict VUN, 13.25% versus 78.40%. Thus, high validity alone does not guarantee a large set of distinct novel molecules.

Paired comparisons. Table 20 gives the paired differences and their uncertainty. Without rarity, EGO exceeds STE in mean Strict VUN by 69.88, 54.41, and 32.22 percentage points at caps 16, 24, and 32, respectively. All three intervals lie above zero. Adding rarity raises Strict VUN by 13.45, 26.15, and 24.44 points, again with positive intervals. $\mathrm { E G O _ { 0 } }$ has the lowest mean feature MMD at each cap, while rarity raises MMD as it improves useful yield. MMD comparisons apply within each cap, not across different molecular mixtures.

Table 19: Complete cumulative ZINC250K sweep. Values are mean ± sample standard deviation across five seeds, with 30,000 updates per run. Full, Strict, Strict VUN, and Conn. are percentages; all use the complete attempt budget as the denominator. Lower $\mathrm { M M D } _ { \mathrm { c e l l } }$ is better, with comparisons restricted to a fixed cap.
<table><tr><td>k Method</td><td>Full ↑</td><td>Strict ↑</td><td>Strict VUN ↑</td><td>Conn. ↑</td><td> $\mathrm { M M D } _ { \mathrm { c e l l } } \downarrow$ </td></tr><tr><td>16 EGO</td><td> $9 7 . 3 3 \pm 1 . 1 1$ </td><td> $9 4 . 9 1 \pm 1 . 5 7$ </td><td> $9 0 . 8 4 \pm 0 . 7 7$ </td><td> $9 7 . 5 5 \pm 0 . 9 4$ </td><td> $0 . 0 0 8 4 \pm 0 . 0 0 0 7$ </td></tr><tr><td>16  $\mathrm { E G O _ { 0 } }$ </td><td> $8 6 . 6 0 \pm 2 . 8 1$ </td><td> $8 3 . 8 2 \pm 4 . 1 8$ </td><td> $7 7 . 3 9 \pm 3 . 7 5$ </td><td> $9 7 . 0 7 \pm 1 . 8 5$ </td><td> $0 . 0 0 5 2 \pm 0 . 0 0 0 9$ </td></tr><tr><td>STE 16 (feature-ST)</td><td> $9 9 . 2 1 \pm 0 . 9 1 $ </td><td> $9 6 . 5 2 \pm 1 . 7 5$ </td><td> $7 . 5 1 \pm 4 . 8 3$ </td><td> $9 7 . 3 1 \pm 2 . 2 9$ </td><td> $0 . 0 5 3 8 \pm 0 . 0 3 5 1$ </td></tr><tr><td>16 Soft</td><td> $9 5 . 0 1 \pm 1 . 7 4$ </td><td> $8 9 . 3 3 \pm 7 . 1 9$ </td><td> $1 0 . 8 1 \pm 4 . 2 1$ </td><td> $9 4 . 2 9 \pm 6 . 8 2$ </td><td> $0 . 0 4 3 0 \pm 0 . 0 3 1 3$ </td></tr><tr><td>24 EGO</td><td> $9 5 . 4 8 \pm 2 . 9 8$ </td><td> $9 1 . 6 9 \pm 3 . 1 9$ </td><td> $8 9 . 4 5 \pm 2 . 6 7$ </td><td> $9 6 . 1 8 \pm 2 . 1 2$ </td><td> $0 . 0 0 9 9 \pm 0 . 0 0 1 5$ </td></tr><tr><td>24  $\mathrm { E G O _ { 0 } }$  STE</td><td> $7 5 . 5 8 \pm 1 1 . 0 0$ </td><td> $6 9 . 3 8 \pm 1 4 . 0 1$ </td><td> $6 3 . 3 0 \pm 1 0 . 9 6$ </td><td> $9 3 . 1 7 \pm 4 . 7 9$ </td><td> $0 . 0 0 6 5 \pm 0 . 0 0 1 2$ </td></tr><tr><td>24 (feature-ST)</td><td> $9 7 . 6 3 \pm 1 . 9 9$ </td><td> $8 1 . 4 0 \pm 3 2 . 0 5$ </td><td> $8 . 8 8 \pm 6 . 8 2$ </td><td> $8 3 . 1 2 \pm 3 1 . 5 9$ </td><td> $0 . 0 5 9 3 \pm 0 . 0 1 8 9$ </td></tr><tr><td>24 Soft</td><td> $9 0 . 4 4 \pm 6 . 2 8$ </td><td> $7 5 . 2 0 \pm 1 3 . 8 9$ </td><td> $1 4 . 1 3 \pm 1 0 . 3 8$ </td><td> $8 4 . 4 1 \pm 1 1 . 2 5$ </td><td> $0 . 0 6 1 4 \pm 0 . 0 4 0 2$ </td></tr><tr><td>32 EGO 32  $\mathrm { E G O _ { 0 } }$ </td><td> $8 6 . 3 1 \pm 8 . 5 8$ </td><td> $8 2 . 4 8 \pm 9 . 3 3$ </td><td> $7 8 . 4 0 \pm 8 . 6 8$   $5 3 . 9 6 \pm 7 . 8 2$ </td><td> $9 6 . 0 9 \pm 1 . 5 3$ </td><td> $0 . 0 1 2 5 \pm 0 . 0 0 3 2$ </td></tr><tr><td>STE</td><td> $6 5 . 9 8 \pm 7 . 8 5$ </td><td> $6 0 . 6 4 \pm 9 . 1 8$ </td><td></td><td> $9 3 . 9 2 \pm 5 . 5 2$ </td><td> $0 . 0 0 8 4 \pm 0 . 0 0 1 7$ </td></tr><tr><td>32 (feature-ST)</td><td> $8 0 . 4 1 \pm 2 7 . 5 9$ </td><td> $7 6 . 1 0 \pm 2 6 . 0 7$ </td><td> $2 1 . 7 4 \pm 9 . 6 1$ </td><td> $9 4 . 3 0 \pm 3 . 2 1$ </td><td> $0 . 0 5 5 5 \pm 0 . 0 1 2 2$ </td></tr><tr><td>32 Soft</td><td> $9 2 . 4 7 \pm 5 . 2 5$ </td><td> $8 7 . 3 5 \pm 7 . 8 8$ </td><td> $1 3 . 2 5 \pm 8 . 6 1$ </td><td> $9 4 . 4 1 \pm 4 . 9 1$ </td><td> $0 . 0 6 5 3 \pm 0 . 0 2 2 8$ </td></tr></table>

Table 20: ZINC250K paired contrasts: mean difference with its unadjusted paired 95% confidence interval in brackets, over five seeds. Each difference is the first method minus the second. Validity and VUN use the strict definition, with differences reported in percentage points; MMD differences are multiplied by $1 0 ^ { 3 } .$ . Positive validity or yield differences and negative MMD differences favor the first method.
<table><tr><td>k Contrast</td><td>∆Validity</td><td>∆VUN</td><td> $1 0 ^ { 3 } \Delta \mathrm { M M D } _ { \mathrm { c e l l } }$ </td></tr><tr><td>16  $\mathrm { E G O } - \mathrm { E G O } _ { 0 }$ </td><td>11.08 [6.72, 15.44]</td><td>13.45 [8.62, 18.28]</td><td>3.23 [2.35, 4.12]</td></tr><tr><td>16  $\mathrm { E G O _ { 0 } } - \mathrm { S o f t }$ </td><td> $- 5 . 5 1 \ [ - 1 3 . 4 5 , \ 2 . 4 4 ]$ </td><td>66.58 [59.67, 73.48]</td><td>-37.80 [-76.99, 1.39]</td></tr><tr><td>16  $\mathrm { E G O _ { 0 } } - \mathrm { S T E }$ </td><td>-12.70 [−19.28, −6.12]</td><td>69.88 [61.27, 78.49]</td><td>-48.64 [-91.73, -5.55]</td></tr><tr><td>16  $\mathrm { S T E - S o f t }$ </td><td>7.19 [-1.20, 15.58]</td><td>-3.30 [−13.92, 7.32]</td><td>10.84 [-66.83, 88.50]</td></tr><tr><td>24  $\mathrm { E G O } - \mathrm { E G O } _ { 0 }$ </td><td>22.31 [6.40, 38.21]</td><td>26.15 [13.23, 39.08]</td><td>3.40 [0.85, 5.95]</td></tr><tr><td>24  $\mathrm { E G O _ { 0 } } - \mathrm { S o f t }$ </td><td>-5.82 [-37.61, 25.97]</td><td>49.17 [40.16, 58.17]</td><td>-54.89 [−104.17, -5.60]</td></tr><tr><td>24  $\mathrm { E G O _ { 0 } } - \mathrm { S T E }$ </td><td>-12.02 [−62.42, 38.38]</td><td>54.41 [36.32, 72.50]</td><td>-52.85 [−76.76, -28.94]</td></tr><tr><td>24  $\mathrm { S T E - S o f t }$ </td><td>6.20 [-27.44, 39.84]</td><td>-5.24 [-19.20, 8.71]</td><td>-2.04 [-54.14, 50.07]</td></tr><tr><td>32  $\mathrm { E G O } - \mathrm { E G O } _ { 0 }$ </td><td>21.84 [3.92, 39.77]</td><td>24.44 [10.18, 38.70]</td><td>4.09 [0.54, 7.64]</td></tr><tr><td>32  $\mathrm { E G O _ { 0 } } - \mathrm { S o f t }$ </td><td> $- 2 6 . 7 2 \ [ - 3 6 . 7 0 , - 1 6 . 7 3 ]$ </td><td>40.71 [21.60, 59.82]</td><td> -56.90 [-86.71, -27.08]</td></tr><tr><td>32  $\mathrm { E G O _ { 0 } } - \mathrm { S T E }$ </td><td>-15.47 [-49.48, 18.55]</td><td>32.22 [15.17, 49.27] -47.15 [−64.08, -30.23]</td><td></td></tr><tr><td>32  $\mathrm { S T E - S o f t }$ </td><td> $- 1 1 . 2 5 \ : [ - 4 9 . 8 7 , 2 7 . 3 7 ]$ </td><td> $8 . 4 9 \ [ - 1 0 . 4 9 , 2 7 . 4 7 ]$ </td><td>-9.75 [-28.28, 8.79]</td></tr></table>

Key takeaway. On the tested ZINC250K mixtures, EGO produces more distinct novel connected molecules than STE or Soft, even without rarity. Rarity further improves this yield but weakens the match to the selected feature distributions. Equal update counts leave training work unmatched, which motivates the active time comparison below.

## D.3.4 MATCHED ACTIVE TRAINING BUDGETS

Motivation. Equal update budgets can hide differences in the cost of a training update. We therefore compare EGO, $\begin{array} { r l } { \operatorname { E G O } _ { 0 } . } \end{array}$ , STE, and Soft under the same active training time on the same GPU. This experiment asks whether the higher useful yield of EGO remains when the gradient methods can complete more updates. The comparison of $\mathrm { E G O _ { 0 } }$ with STE tests the update recipes under the same hard forward criterion. Comparing the two EGO variants tests the additional rarity term.

Shared protocol. This experiment uses the full accepted ZINC250K data on a canvas of 38 nodes, including molecules with more than 32 heavy atoms. All four methods use the same one-step generator with three hidden layers of width 768 and latent dimension 96. They share the train/test split, molecular size prior, evaluation inputs, and initialization within each of the three seeds, 42, 43, and 44. The methods run sequentially on one allocated GPU, with their order rotated across seeds. The vocabulary is derived only from ZINC training molecules. This differs from the joint ZINC/GuacaMol vocabulary used for the controlled size sweep, so the results are compared within this experiment.

Clock and checkpoint policy. We evaluate 10,000 proposals at 0, 1, 2, 5, 10, 20, and 30 minutes. Each evaluation uses the last completed update within its budget, without retrospective checkpoint selection or a cap on update count. Active time includes synchronized updates, batch preparation, device transfers, and archive updates. It excludes initial setup, compilation and warmup, evaluation, and checkpoint I/O. Strict VUN counts distinct valid, connected molecules absent from training, divided by all attempts, without repair by extracting the largest component. Table 21 reports the final measurements that support Figure 3.

Table 21: Full ZINC250K after 30 minutes of active training. Values are mean ± sample SD across three seeds, with 10,000 proposals per seed. Quality metrics are percentages; updates are in thousands.
<table><tr><td>Method</td><td>Full valid</td><td>Strict valid</td><td>Strict VUN</td><td> $\mathrm { U p d a t e s } \left( 1 0 ^ { 3 } \right)$ </td></tr><tr><td>EGO</td><td> $8 6 . 9 4 \pm 5 . 2 6$ </td><td> $7 9 . 6 9 \pm 8 . 4 7$ </td><td> $7 7 . 9 1 \pm 7 . 9 2$ </td><td> $2 5 . 8 4 \pm 0 . 2 9$ </td></tr><tr><td> $\mathbf { E G O } _ { 0 }$ </td><td> $7 1 . 5 3 \pm 4 . 9 5$ </td><td> $6 1 . 7 1 \pm 3 . 5 8$ </td><td> $5 3 . 7 3 \pm 3 . 7 0$ </td><td> $2 5 . 9 5 \pm 0 . 2 6$ </td></tr><tr><td>STE</td><td> $7 9 . 1 2 \pm 4 . 1 5$ </td><td> $5 7 . 2 6 \pm 1 1 . 2 7$ </td><td> $2 9 . 5 0 \pm 1 . 7 6$ </td><td> $1 4 5 . 7 3 \pm 2 . 3 9$ </td></tr><tr><td>Soft</td><td> $9 2 . 8 2 \pm 4 . 7 0$ </td><td> $6 6 . 8 1 \pm 2 5 . 4 6$ </td><td> $5 . 7 8 \pm 9 . 8 6$ </td><td> $1 5 6 . 5 8 \pm 3 . 3 0$ </td></tr></table>

Results. At 30 minutes, EGO reaches $7 7 . 9 1 { \pm } 7 . 9 2 \%$ Strict VUN, and $\mathrm { E G O _ { 0 } }$ reaches 53.73±3.70%. The corresponding values are $2 9 . 5 0 \pm 1 . 7 6 \%$ for STE and $5 . 7 8 \pm 9 . 8 6 \%$ for Soft in Table 21. The ratios of mean yield relative to Soft are 13.47× for EGO and 9.29× for EGO . The rarity term adds 24.19 percentage points to the mean yield. Both EGO variants have higher mean Strict VUN than both gradient baselines at all six positive budgets. The aggregate data support this statement about means, without establishing the same ordering in every individual seed.

STE and Soft complete 5.64× and 6.06× as many updates as EGO, respectively. Soft has the highest mean full validity, 92.82%, but a low Strict VUN of 5.78%. This gap shows why full validity alone is insufficient to describe useful output yield. The supplied aggregate metrics do not separate the losses due to disconnection, duplication, and recovery of training molecules.

Key takeaway. EGO maintains a higher useful molecular yield under equal active training time, even though STE and Soft complete about six times as many updates. The advantage remains when rarity is removed, and the rarity term increases it further. This result concerns active training time on full ZINC250K. It does not measure total job time, inference speed, or distributional fidelity.

## D.4 GUACAMOL

GuacaMol (Brown et al., 2019) uses a curated subset of ChEMBL to evaluate molecular generation. It includes a wider range of molecular sizes and structures than QM9. Our full GuacaMol experiment supports up to 88 heavy atoms and uses the supplied training, validation, and test partitions. This setting tests whether EGO can still produce a large collection of distinct valid molecules as graph size increases. Experiments on subsets defined by a smaller size limit are discussed separately below.

## D.4.1 COMPARISON WITH EXTERNAL GENERATORS

Table 22 reports the complete comparison with published GuacaMol generators and the local CFM-ECLD pilot described in Section D.4.3. The EGO experiment restricted to $n \leq 3 2$ is shown in a separate block because it uses a different data distribution and a smaller graph canvas. We report FCD as a distance throughout this table, so lower values indicate a closer distributional match. For published rows, we convert the rounded benchmark score s<sub>FCD</sub> using $D _ { \mathrm { F C D } } = - 5$ log s<sub>FCD</sub>, following the definition in Brown et al. (2019). These converted distances are approximate. For EGO, we aggregate the recorded distance in each seed directly.

Separate experiment on the n ≤ 32 subset

Table 22: GuacaMol generation, including the local CFM-ECLD pilot after three epochs and 14,919 updates, using seed 42. The pilot is an early checkpoint, not a converged baseline, and is excluded from efficiency ratio and ranking claims. EGO reports mean ± sample SD over three training seeds. VU is the percentage of raw attempts that yield distinct valid molecules, except for the proxies marked <sup>a</sup>. FCD is a distance, with lower values preferred. Gray rows identify generators that use one network evaluation. Published results retain their source protocols; EGO and the local CFM-ECLD pilot use our offline evaluation protocol. The final block reports the separate EGO experiment on n ≤ 32.
<table><tr><td>Method</td><td>Steps</td><td>Valid (%) ↑</td><td>VU (%) ↑</td><td>FCD↓</td><td>VU/s ↑</td></tr><tr><td>Autoregressive generation SMILES LSTM (Brown et al., L tokens</td><td></td><td>95.9</td><td>95.9ª</td><td>≈0.46</td><td></td></tr><tr><td>2019)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>G2PT-base (Chen et al., 2025)</td><td>L tokens</td><td>94.6</td><td>94.6ª</td><td>≈0.34</td><td></td></tr><tr><td>G2PT-large (Chen et al., 2025)</td><td>L tokens</td><td>95.3</td><td>95.3a</td><td>≈0.38</td><td></td></tr><tr><td>Iterative diffusion and flow</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DiGress (Vignac et al., 2023)</td><td>500</td><td>85.2</td><td>85.2ª</td><td>≈1.93</td><td></td></tr><tr><td>Cometh (Siraudin et al., 2025)</td><td>500</td><td>94.4</td><td>94.4</td><td>≈1.97</td><td></td></tr><tr><td>Cometh-PC (Siraudin et al., 2025)</td><td>500 + PC</td><td>98.9</td><td>98.9</td><td>≈1.59</td><td></td></tr><tr><td>DeFoG (Qin et al., 2025)</td><td>50</td><td>91.7</td><td>91.7</td><td>≈ 2.73</td><td></td></tr><tr><td>DeFoG (Qin et al., 2025)</td><td>500</td><td>99.0</td><td>99.0</td><td>≈1.52</td><td></td></tr><tr><td>MELD (Seo et al., 2025)</td><td></td><td>100.0</td><td>100.0ª</td><td>≈1.87</td><td></td></tr><tr><td>One-step generation</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CFM-ECLD†</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(3 epochs)</td><td>1</td><td>19.67</td><td></td><td></td><td></td></tr><tr><td>EGO (full)</td><td>1</td><td>80.80 ± 10.36</td><td>19.67  $7 7 . 6 1 \pm 9 . 3 5$ </td><td>33.36 34.62 ±1.58</td><td>28.27  $6 8 6 \pm 7 6$ </td></tr></table>

$$
8 7 . 7 1 \pm 3 . 5 4
$$

$$
3 3 . 2 9 \pm 3 . 0 6
$$

$$
2 , 4 4 2 \pm 6 3
$$

L denotes token sequence length; PC denotes predictor and corrector sampling, which adds computation. Iterative steps need not equal neural network evaluations. <sup>a</sup>VU is approximated from reported validity and conditional uniqueness, using rounded source values. The two measurements may use separate sampling banks, so these proxies are not measured distinct yields from a shared proposal bank. <sup>†</sup>The CFM-ECLD pilot uses 10,000 raw proposals for validity and VU. Its FCD uses 10,000 valid molecules from a separate bank of 60,000 attempts and the test reference. No across-seed SD is available for this single run. Full pilot results appear in Table 25. EGO uses three hidden layers of width 768, 240,000 training updates on full GuacaMol and 120,000 on $n \leq 3 2$ , with validation selection. Each seed generates 10,000 raw proposals for validity and VU. Its FCD uses 10,000 valid samples from a separate bank of 30,000 raw proposals and 10,000 test reference molecules; the $n \leq 3 2$ reference is filtered to that subset. Full molecular identities allow disconnected valid outputs; no novelty requirement is applied to VU. EGO VU/s is the mean and sample SD of per-seed rates on an A100 40GB, including generation, molecular conversion, and SMILES/CSV export, but excluding setup, warmup, and subsequent distribution evaluation. The CFM pilot rate includes conversion and CSV export on one A100; its sampler does not perform a separate warmup. These measurements do not establish a matched-hardware speedup. Estimated dense costs are reported separately in Table 26. A dash denotes an unavailable compatible measurement.

Key takeaway. EGO produces $7 7 . 6 1 \pm 9 . 3 5 \%$ distinct valid molecules from its proposal budget with one network evaluation per attempt. Its measured full-data rate is $6 8 6 \pm 7 6$ distinct valid molecules per second through CSV export. The smaller $n \leq 3 2$ experiment achieves higher yield and throughput, but changes both the data distribution and the training budget. The cost estimates in Table 26 provide additional computational context. The FCD of EGO remains much larger than the published reference values. These results show that producing many valid candidates and matching the data distribution are separate objectives. Differences in evaluation protocol prevent a controlled ranking of distributional quality across all rows.

## D.4.2 CONTROLLED MOLECULAR SIZE SWEEP

Why this experiment. We repeat the size sweep on GuacaMol to test whether the result extends beyond the ZINC250K mixtures. These cumulative subsets use the shared protocol in Section D.1.1 and remain separate from the full GuacaMol comparison. The experiment asks whether useful yield persists as the task includes larger molecules and whether rarity gives a consistent benefit.

Results. Table 23 reports all twelve configurations. EGO reaches mean Strict VUN of 82.12%, 81.53%, and 73.69% at caps 16, 24, and 32. Both EGO variants remain above STE and Soft in mean Strict VUN at every cap. $\mathbf { A } \mathbf { t } \ k = 3 2 .$ , STE has higher strict validity than EGO, 83.64% versus 78.41%, but much lower Strict VUN, 15.02% versus 73.69%. The difference between these metrics combines repetitions and training-set overlap; it does not separate their individual contributions.

Table 23: Complete cumulative GuacaMol sweep. Values are mean ± sample standard deviation across five seeds, with 30,000 updates per run. Full, Strict, Strict VUN, and Conn. are percentages; all use the complete attempt budget as the denominator. Lower $\mathrm { M M D } _ { \mathrm { c e l l } }$ is better, with comparisons restricted to a fixed cap.
<table><tr><td>k Method</td><td></td><td>Full ↑</td><td>Strict↑</td><td>Strict VUN ↑</td><td>Conn. ↑</td><td> $\mathrm { M M D } _ { \mathrm { c e l l } } \downarrow$ </td></tr><tr><td></td><td>16 EGO</td><td> $9 4 . 5 7 \pm 2 . 6 6$ </td><td> $9 0 . 2 7 \pm 2 . 3 4$ </td><td> $8 2 . 1 2 \pm 2 . 1 9$ </td><td> $9 5 . 6 8 \pm 1 . 4 3$ </td><td> $0 . 0 0 9 1 \pm 0 . 0 0 2 4$ </td></tr><tr><td>16</td><td> $\mathrm { E G O _ { 0 } }$  STE</td><td> $8 6 . 7 1 \pm 5 . 4 9$ </td><td> $8 2 . 6 0 \pm 6 . 8 2$ </td><td> $6 6 . 0 1 \pm 4 . 0 8$ </td><td> $9 5 . 7 9 \pm 1 . 5 9$ </td><td> $0 . 0 0 6 4 \pm 0 . 0 0 1 6$ </td></tr><tr><td></td><td>16 (feature-ST)</td><td> $9 9 . 7 2 \pm 0 . 3 8$ </td><td> $9 7 . 8 8 \pm 1 . 9 5$ </td><td> $6 . 3 9 \pm 1 . 3 8$ </td><td> $9 8 . 1 6 \pm 1 . 6 3 $ </td><td> $0 . 0 3 6 4 \pm 0 . 0 1 8 9$ </td></tr><tr><td>16 Soft</td><td></td><td> $9 7 . 7 5 \pm 2 . 0 9$ </td><td> $9 2 . 5 2 \pm 4 . 4 1$ </td><td> $7 . 7 4 \pm 7 . 5 3$ </td><td> $9 4 . 7 7 \pm 2 . 6 2$ </td><td> $0 . 0 3 8 7 \pm 0 . 0 1 6 2$ </td></tr><tr><td>24 EGO</td><td></td><td> $9 2 . 8 0 \pm 5 . 4 2$ </td><td> $8 6 . 2 7 \pm 6 . 4 7$ </td><td> $8 1 . 5 3 \pm 6 . 8 1$ </td><td> $9 3 . 4 4 \pm 1 . 5 4$ </td><td> $0 . 0 1 0 9 \pm 0 . 0 0 3 7$ </td></tr><tr><td>24</td><td> $\mathrm { E G O _ { 0 } }$ </td><td> $7 4 . 4 2 \pm 7 . 9 1$ </td><td> $6 8 . 6 5 \pm 7 . 9 3$ </td><td> $5 8 . 6 8 \pm 8 . 2 2$ </td><td> $9 3 . 9 9 \pm 3 . 1 7$ </td><td> $0 . 0 0 8 8 \pm 0 . 0 0 2 4$ </td></tr><tr><td>24</td><td>STE (feature-ST)</td><td> $6 6 . 5 2 \pm 3 2 . 1 8$ </td><td> $6 2 . 3 7 \pm 3 1 . 2 1$ </td><td> $8 . 8 6 \pm 6 . 3 5$ </td><td> $9 1 . 0 6 \pm 6 . 5 0$ </td><td> $0 . 0 6 2 0 \pm 0 . 0 2 8 1$ </td></tr><tr><td>24</td><td>Soft</td><td> $9 8 . 9 7 \pm 1 . 6 5$ </td><td> $9 0 . 7 3 \pm 5 . 5 7$ </td><td> $7 . 9 5 \pm 9 . 5 8$ </td><td> $9 1 . 7 5 \pm 4 . 0 5$ </td><td> $0 . 0 9 8 4 \pm 0 . 0 4 9 1$ </td></tr><tr><td></td><td>32 EGO</td><td> $8 4 . 2 8 \pm 6 . 0 5$ </td><td> $7 8 . 4 1 \pm 5 . 7 9$ </td><td> $7 3 . 6 9 \pm 5 . 1 0$ </td><td> $9 4 . 0 7 \pm 1 . 5 7$ </td><td> $0 . 0 1 0 6 \pm 0 . 0 0 1 8$ </td></tr><tr><td>32</td><td> $\mathrm { E G O _ { 0 } }$  STE</td><td> $6 4 . 9 8 \pm 4 . 9 7$ </td><td> $5 8 . 2 2 \pm 7 . 8 1$ </td><td> $4 8 . 8 7 \pm 9 . 5 1$ </td><td> $9 2 . 9 0 \pm 4 . 8 1$ </td><td> $0 . 0 1 0 0 \pm 0 . 0 0 2 0$ </td></tr><tr><td>32</td><td>(feature-ST)</td><td> $9 3 . 2 3 \pm 3 . 1 2$ </td><td> $8 3 . 6 4 \pm 5 . 5 3$ </td><td> $1 5 . 0 2 \pm 7 . 8 5$ </td><td> $9 0 . 2 1 \pm 7 . 6 9$ </td><td> $0 . 0 7 3 6 \pm 0 . 0 0 3 7$ </td></tr><tr><td>32 Soft</td><td></td><td> $8 6 . 9 5 \pm 5 . 8 3$ </td><td> $4 9 . 0 4 \pm 3 4 . 2 8$ </td><td> $1 7 . 2 5 \pm 2 3 . 2 8$ </td><td> $5 9 . 2 6 \pm 3 2 . 5 9$ </td><td> $0 . 0 7 5 2 \pm 0 . 0 2 5 5$ </td></tr></table>

Paired comparisons. Table 24 reports the complete paired contrasts. $\mathrm { E G O _ { 0 } }$ exceeds STE in Strict VUN by 59.62, 49.82, and 33.86 percentage points at caps 16, 24, and 32, with all three intervals above zero. The rarity gains are 16.12, 22.84, and 24.81 points, also with positive intervals. The comparison of $\mathrm { E G O _ { 0 } }$ with Soft at $k = 3 2$ has a positive mean difference but an interval that includes zero. We therefore distinguish a consistent ordering of means from evidence for each individual contrast.

Table 24: GuacaMol paired contrasts: mean difference with its unadjusted paired 95% confidence interval in brackets, over five seeds. Each difference is the first method minus the second. Validity and VUN use the strict definition, with differences reported in percentage points; MMD differences are multiplied by $1 0 ^ { 3 }$ . Positive validity or yield differences and negative MMD differences favor the first method.
<table><tr><td>k Contrast</td><td>∆Validity</td><td>∆VUN</td><td> $1 0 ^ { 3 } \Delta \mathrm { M M D } _ { \mathrm { c e l l } }$ </td></tr><tr><td></td><td>7.68</td><td>16.12</td><td>2.63</td></tr><tr><td>16  $\mathrm { E G O } - \mathrm { E G O } _ { 0 }$ </td><td>[−0.68, 16.03] -9.92</td><td>[10.91, 21.32] 58.27</td><td>[0.49, 4.76] -32.20</td></tr><tr><td>16  $\mathrm { E G O _ { 0 } } - \mathrm { S o f t }$ </td><td>[−15.07, -4.78] -15.28</td><td>[50.18, 66.35] 59.62</td><td>[−52.40, -12.01] -29.99</td></tr><tr><td>16  $\mathrm { E G O _ { 0 } } - \mathrm { S T E }$ </td><td>[-22.69, -7.88] 5.36</td><td>[55.76, 63.48] -1.35</td><td>[−52.55, -7.43] -2.21</td></tr><tr><td>16  $\mathrm { S T E - S o f t }$ </td><td>[0.12, 10.61]</td><td>[-9.21, 6.50]</td><td>[-25.74, 21.31]</td></tr><tr><td>24  $\mathrm { E G O } - \mathrm { E G O } _ { 0 }$ </td><td>17.62 [5.49, 29.75] -22.08</td><td>22.84 [12.98, 32.70] 50.73</td><td>2.03 [−2.51, 6.58] -89.57</td></tr><tr><td>24  $\mathrm { E G O _ { 0 } } - \mathrm { S o f t }$ </td><td>[-35.91, -8.26] 6.28</td><td>[33.29, 68.17] 49.82</td><td>[-152.35, -26.79] -53.20</td></tr><tr><td>24  $\mathrm { E G O _ { 0 } } - \mathrm { S T E }$ </td><td>[-39.97, 52.53] -28.37</td><td>[33.59, 66.06] 0.91</td><td>[-86.49, -19.91] -36.37</td></tr><tr><td>24  $\mathrm { S T E - S o f t }$ </td><td>[−70.36, 13.63]</td><td>[-11.69, 13.50]</td><td>[-102.75, 30.01]</td></tr><tr><td>32  $\mathrm { E G O } - \mathrm { E G O } _ { 0 }$ </td><td>20.19 [13.34, 27.04]</td><td>24.81 [14.86, 34.76] 31.62</td><td>0.63 [−3.49, 4.75] -65.22</td></tr><tr><td>32  $\mathrm { E G O _ { 0 } } - \mathrm { S o f t }$ </td><td>9.17 [-33.97, 52.31] -25.42</td><td>[-2.70, 65.94]</td><td>[-94.99, -35.45]</td></tr><tr><td>32  $\mathrm { E G O _ { 0 } } - \mathrm { S T E }$ </td><td>[-37.09, -13.75]</td><td>33.86 [28.24, 39.47]</td><td>-63.64 [-69.10, -58.19]</td></tr><tr><td>32  $\mathrm { S T E - S o f t }$ </td><td>34.59 [-8.87, 78.05]</td><td>-2.24 [-36.98, 32.51]</td><td>-1.58 [-36.53, 33.37]</td></tr></table>

Key takeaway. The GuacaMol sweep supports the same pattern as ZINC250K: EGO has higher distinct novel yield across the tested mixtures, and rarity provides an additional gain. The gain does not imply better performance on every metric, since $\mathrm { E G O _ { 0 } }$ has the lowest mean feature MMD at each cap and some validity comparisons favor the gradient methods.

## D.4.3 A CFM PILOT ON FULL GUACAMOL

Motivation. We selected CFM-ECLD (Roos et al., 2026) for its strong one-step graph generation results on both QM9 and ZINC250K. Kernel-gradient drifting (Esteban-Casadevall et al., 2026) reports molecular experiments only on QM9, using SMILES. We therefore adapted CFM-ECLD to full GuacaMol, absent from its published evaluation, and ran a limited pilot to measure training cost and sample quality early in training on larger graphs.

Setup. The adaptation uses the public CFM model and loss implementation, with 12 transformer layers, node, edge, and global hidden dimensions of 256, 64, and 64, respectively, and eight attention heads. The global feedforward width is explicitly set to 128. This explicitly forwards a configuration argument that the public constructor call omitted, leaving its default at 2048. The adaptation therefore differs from that code path and is not a reproduction of an authenticated published checkpoint. The data use 58 atom states and a fixed canvas of 88 nodes. Fixed padding preserves the global feature statistics of the public model. Training uses AdamW with learning rate $\mathrm { \dot { 1 } 0 ^ { - 4 } }$ , microbatches of 8, an effective batch size of 256, and EMA decay 0.999.

We train one seed, 42, for three epochs, corresponding to 14,919 optimizer updates. The cosine learning rate schedule remains configured for 84 epochs. Evaluation uses the EMA parameters at the final pilot checkpoint, rather than a checkpoint selected on validation. Sampling uses one network evaluation per attempt and batches of 16. We apply the same offline molecular evaluation protocol as for EGO, with no repair, replacement of invalid samples, or selection of the largest fragment.

Results. Table 25 reports the completed evaluation. The primary bank contains 10,000 raw attempts. A separate fixed bank of 60,000 attempts produces 11,694 valid molecules, which supplies the 10,000 generated samples required for FCD and KL evaluation. FCD uses the test reference, and KL uses the training reference. The table uses the completed evaluation with the larger bank, with no additiona training.

Training cost. The logs record approximately 19 to 24 seconds per optimizer update on one A100-SXM-64GB. With 4,973 updates per epoch, extrapolating to the configured 84 epochs gives

$$
T _ { 8 4 } \approx { \frac { 8 4 \times 4 . 9 7 3 \times [ 1 9 , 2 4 ] } { 3 , 6 0 0 } } = [ 2 , 2 0 5 , 2 , 7 8 5 ] { \mathrm { ~ h o u r s } } .
$$

This is the source of the rounded 2,200 to 2,800-hour projection in Table 27. It assumes unchanged update throughput and excludes setup, validation, checkpointing, final evaluation, restarts, and queue time. The range describes observed throughput scenarios, not a confidence interval. The exact cumulative duration of the pilot across restarts was not recovered. The projection does not establish that 84 epochs would be needed or sufficient for convergence.

Table 25: CFM-ECLD pilot on full GuacaMol, seed 42, after three epochs and 14,919 updates. This is a single pilot checkpoint without an estimate of variation across training seeds. Full validity allows disconnected molecules; strict validity also requires connectivity.
<table><tr><td>Metric</td><td>CFM-ECLD pilot</td></tr><tr><td>Full validity (%)</td><td>19.67</td></tr><tr><td>Strict validity (%)</td><td>4.64</td></tr><tr><td>Uniqueness among full valid molecules (%)</td><td>100.00</td></tr><tr><td>Full VU (%)</td><td>19.67</td></tr><tr><td>Strict VU (%)</td><td>4.64</td></tr><tr><td>Full VUN (%)</td><td>19.66</td></tr><tr><td>Strict VUN (%)</td><td>4.63</td></tr><tr><td>FCD to test reference ↓</td><td>33.3561</td></tr><tr><td>GuacaMol KL score ↑</td><td>0.6814</td></tr><tr><td>Graph generation, 10,000 attempts (s)</td><td>64.67</td></tr><tr><td>Generation through SMILES/CSV, 10,000 attempts (s)</td><td>69.59</td></tr><tr><td>Full VU/s through SMILES/CSV</td><td>28.27</td></tr><tr><td>Estimated dense MFLOP per proposal</td><td>13,312.918</td></tr><tr><td>Evu</td><td>14.8</td></tr></table>

Validity, VU, and VUN use the primary bank. VUN additionally requires absence from the training reference. FCD and KL use the separate distribution bank. Timings come from the first evaluation of the same checkpoint and primary bank on one A100, include synchronized generation and transfer to CPU, and separate chemistry conversion and CSV export. They exclude subsequent FCD and KL computation; the sampler does not run a separate warmup. Dense cost is reconstructed from the local 58-state architecture at 88 nodes and one network evaluation, using the partial dense operation convention of this appendix. It is not a measured hardware cost.

Key takeaway. The pilot makes the missing GuacaMol comparison concrete, but it remains an early checkpoint. All 1,967 valid molecules in the primary bank are distinct, yet only 464 are connected. Thus, approximately 76.4% of valid outputs contain multiple fragments, and strict VUN is only 4.63%. The run exposes a substantial training cost and a connectivity limitation at this budget. It does not establish the quality of a fully trained CFM model and is excluded from efficiency ratio and ranking claims against EGO.

## D.5 INFERENCE COST AND TRAINING TIME

## D.5.1 ESTIMATED DENSE INFERENCE WORK

Why this analysis. The main comparison asks how many distinct valid candidates a generator produces relative to its inference work. A single sampling step does not imply the same cost across architectures. We therefore count the dense projections executed during generation and report measured training time separately. The resulting estimates support Figure 2 in the main text.

Definition. Let N be the number of raw proposals, $U _ { N }$ the number of distinct valid outputs, and $W _ { N }$ the estimated dense work for that bank. With $Y _ { \mathrm { V U } } = U _ { N } / N$ and average work $C _ { \mathrm { d e n s e } } \overset { \cdot } { = } W _ { N } / N$ per proposal,

$$
E _ { \mathrm { V U } } = 1 0 ^ { 1 2 } { \frac { U _ { N } } { W _ { N } } } = 1 0 ^ { 1 2 } { \frac { Y _ { \mathrm { V U } } } { C _ { \mathrm { d e n s e } } } } .\tag{35}
$$

Yield is a fraction in this equation. Invalid and duplicate proposals consume work and remain in the denominator. Because distinct yield depends on the number of attempts, this normalization does not predict how many distinct molecules a new run would find under a common compute budget.

Counting convention. We count two operations per multiply-accumulate in each executed learned weight projection. This includes linear layers, recurrent projections, and convolutional projections where applicable, summed over all network evaluations. For a linear map from width a to width b applied at m positions, the count is 2mab. Convolutions use the corresponding kernel dimensions and output positions. Bias additions, activations, normalization, attention products and reductions, graph aggregation, feature construction, sampling, molecular conversion, deduplication, and distribution evaluation are excluded. For MoFlow, the count includes inverse $1 \times 1$ projections, but excludes weight-inverse setup and elementwise flow operations. These are partial inference operation counts, not complete FLOPs or runtime measurements.

EGO architectures. For an EGO multilayer perceptron with input width d, L hidden layers of width H, and output width $^ { O , }$

$$
C _ { \mathrm { d e n s e } } = 2 \left[ d H + ( L - 1 ) H ^ { 2 } + H o \right] .\tag{36}
$$

The output width includes both atom and bond heads. For QM9, d = 105, L = 3, H = 768, and $o = 2 5 2$ , which gives 2,907,648 dense operations per proposal. Using the mean VU of 0.7821 gives $E _ { \mathrm { V U } } = 2 6 8 , 9 8 0$ . The corresponding ZINC250K and full GuacaMol counts are 12,630,016 and 33,991,680 operations. The corresponding $( d , L , H , o )$ values are (129, 3, 1024, 3990) and (178, 3, 768, 20416). Their output canvases have 38 and 88 nodes, respectively; the size encoding widths are 33 and 82. Table 26 preserves the available efficiency estimates from the main molecular comparison and gives the counts supporting the EGO versus CFM ratios.

Table 26: Dense inference work and efficiency for the molecular comparison. VU is a percentage of raw proposals, and MFLOP denotes $1 0 ^ { 6 }$ counted operations per proposal, including all network evaluations. EGO counts use local architectures; published CFM counts use reconstructed architectures. The GuacaMol CFM count uses the local three-epoch pilot. Estimates marked <sup>u</sup> retain the values used in the main figure, but lack a complete recovered layer count. Dashes indicate unavailable counts. The pilot is excluded from ranking and ratio claims.
<table><tr><td rowspan=1 colspan=3>Dataset      Method                        VU (%)  Dense MFLOP      $E _ { \mathrm { V U } }$ </td></tr><tr><td rowspan=1 colspan=3>QM9        GraphARM                   ≈ 86.30</td></tr><tr><td rowspan=6 colspan=3>DeFoG (500)                 ≈ 95.72      86,234.240MELD                        ≈ 96.49MolGANMoFlow                          99.20           47.024CFM-CSD                    ≈82.62          167.689</td></tr><tr><td rowspan=1 colspan=1>11.1</td></tr><tr><td rowspan=1 colspan=1>1.72u</td></tr><tr><td rowspan=1 colspan=1>142,945</td></tr><tr><td rowspan=1 colspan=1>21,096</td></tr><tr><td rowspan=1 colspan=1>4,927</td></tr><tr><td rowspan=2 colspan=3>CFM-ECLD                  ≈85.45          167.689EGO                     $7 8 . 2 1 \pm 0 . 8 4$             2.908</td></tr><tr><td rowspan=1 colspan=1>268,980</td></tr><tr><td rowspan=8 colspan=3>ZINC250K  GraphARM                   ≈87.75CatFlow (100)                ≈ 99.21DeFoG (500)                 ≈ 99.21   1,359,621.888MELD                        ≈ 99.99MoFlow                          99.99CFM-CSD                    ≈73.43       2,694.691CFM-ECLD                  ≈ 93.03       2,694.691EGO                    90.66 ± 4.00           12.630</td></tr><tr><td rowspan=1 colspan=1>14.3u</td></tr><tr><td rowspan=1 colspan=1>0.730</td></tr><tr><td rowspan=1 colspan=1>0.265u</td></tr><tr><td rowspan=1 colspan=1>666u</td></tr><tr><td rowspan=1 colspan=1>272</td></tr><tr><td rowspan=1 colspan=1>345</td></tr><tr><td rowspan=1 colspan=1>71,781</td></tr><tr><td rowspan=1 colspan=3>GuacaMol   SMILES LSTM                ≈ 95.9</td></tr><tr><td rowspan=3 colspan=3>G2PT-base                     ≈ 94.6DeFoG (500)MELD                        ≈ 100.0</td></tr><tr><td rowspan=1 colspan=1>0.148</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>CFM-ECLD, 3 epochs          19.67      13,312.918</td></tr><tr><td rowspan=1 colspan=3>EGO                     $7 7 . 6 1 \pm 9 . 3 5$            33.992   22,831</td></tr></table>

External architectures and sampling. Graph Transformer estimates use reference canvases of 9, 38, and 88 nodes for QM9, ZINC250K, and GuacaMol. These describe a maximum-canvas workload, not an observed average over the sampled molecular sizes or padded batches. Autoregressive costs must include the full generated sequence, and iterative costs must include all network calls, including any final denoiser. We do not replace these counts with parameter count multiplied by nominal sampling steps. The DeFoG reconstruction gives 86,234.240, 1,359,621.888, and 6,685,054.720 MFLOP for 500 evaluations on the three datasets. The audited QM9 counts for MolGAN and MoFlow are 0.714 and 47.024 MFLOP, respectively. External configurations are associated with published quality, but the precise checkpoints that produced those values were not authenticated. Estimates marked <sup>u</sup> in Table 26 retain the values used in the main figure; a complete layer-level reconstruction was not recovered for those entries. They do not support the EGO versus CFM ratios. The approximate QM9 VU for DeFoG uses the published relaxed validity of 99.4% and uniqueness of 96.3%; Table 8 retains the standard validity of 99.3%. We therefore do not multiply columns based on different validity rules.

Sensitivity of the CFM reconstruction. The QM9 reconstruction uses a global feed-forward width of 128 based on the paper description. Using the constructor default of 2048 changes the parameter count from 5.925 to 8.154 million and dense work from 167.689 to 172.112 MFLOP. Holding published quality fixed, the corresponding CFM-ECLD efficiencies are approximately 5,096 and 4,965. The EGO efficiency ratio is then 52.8 or 54.2, respectively. Thus, the reported advantage is stable across these two assumptions, although neither scenario authenticates the original checkpoint. The ZINC250K CFM reconstruction gives the approximately 208× ratio reported in the main text. The local GuacaMol configuration and its limited training budget are described in Section D.4.3.

Key takeaway. The compact EGO architecture requires much less dense projection work per proposal than the reconstructed CFM architectures. This yields the reported efficiency advantage despite lower VU and worse FCD on QM9 and ZINC250K. The ratios describe the stated partial operation count and source protocols, not measured generation speedups.

## D.5.2 TRAINING TIME AUDIT

Why this comparison. Low inference cost can require substantial training work. Table 27 therefore reports the training durations used in the resource panel of the main text. It provides context across EGO, DeFoG, Cometh, and CFM while preserving differences in timing scope and hardware.

Table 27: Training time in hours for the methods discussed in the main text. EGO measures synchronized optimizer updates; external values use the reported elapsed training durations. The GuacaMol CFM entry is a constant-throughput projection to 84 epochs from the local three-epoch pilot. It is not a measured full training duration. A dash denotes unavailable evidence.
<table><tr><td>Method</td><td>QM9</td><td>ZINC250K</td><td>GuacaMol</td></tr><tr><td>EGO</td><td>0.2999 ± 0.0012</td><td>2.0983</td><td>14.8468 ± 0.0697</td></tr><tr><td>DeFoG (Qin et al., 2025)</td><td>6.5</td><td>14</td><td>141</td></tr><tr><td>Cometh (Siraudin et al., 2025)</td><td>6</td><td></td><td>≈ 96</td></tr><tr><td>CFM (Roos et al., 2026)</td><td>≈48</td><td>≈144</td><td>≈ 2,200 to 2,800</td></tr></table>

EGO measurements. QM9 uses the 3 × 768 generator trained for 120,000 updates, and full GuacaMol uses the same depth and width for 240,000 updates. Both report the mean and sample standard deviation across seeds 42, 43, and 44. The ZINC250K duration is the available seed-42 measurement for the 3×1024 model at 120,000 updates, so no across-seed deviation is reported. These timers cover the completed training budget even when validation selects an earlier checkpoint. They exclude setup, compilation, warmup, validation, checkpointing, and final evaluation. The historical ZINC timer also excludes input preparation and perturbation-scale calculation. The recovered timing records do not certify the GPU model for every EGO campaign, so we do not assign a common GPU specification to these durations.

Published measurements. DeFoG reports 6.5 hours for QM9 without explicit hydrogen nodes, 14 hours for ZINC250K, and 141 hours for GuacaMol on one A100-SXM4 80GB. Cometh reports 6 hours on QM9 using a V100 or A10, without resolving which device produced that duration, and about four days on GuacaMol using two A100 GPUs. The latter corresponds to about 192 allocated GPU-hours. CFM reports about two days per QM9 model and six days per ZINC250K model with RTX A6000 hardware, with the GPU count unspecified. Its reported schedules use 5,000 and 500 epochs, respectively. These measurements follow the training-time sections of the original papers. The local GuacaMol projection follows the calculation in Section D.4.3; it is not a published CFM result.

Key takeaway. EGO has the lowest recorded training durations among these entries, but the table does not establish a speedup under matched hardware, timing scope, or final quality. The controlled ZINC250K experiment in Section D.3.4 supplies the direct comparison under a common active training budget.

## E RQ2: SCAFFOLD COMPLETION

Scaffold completion tests whether a compact generator can rapidly supply molecular candidates that preserve a given core and satisfy screening criteria. We train MoLeR (Maziarz et al., 2022) from scratch on the same ZINC scaffold split as EGO. Both methods receive the same held-out scaffold queries and are evaluated with the same screening rules. Architectures, training objectives, and training budgets differ.

We evaluate two resource constraints. The primary experiment matches the time available for generation and screening on each query. A complementary experiment matches the number of proposals and measures the resulting distinct candidate libraries. Generation-to-SMILES timing is recorded separately for that fixed proposal bank.

Summary. At matched time, EGO screens 7.39× as many proposals on average and returns 10.05× as many filter-passing proposals. MoLeR covers more queries and more often returns the higher-QED passing candidate. We report outcomes over all 100 queries in each replicate, including failures of either or both methods. At a fixed budget of 100,000 proposals, EGO returns 20,245 ± 755 distinct filter-passing completions versus 16,684 ± 79 for MoLeR, with an observed 44.3× reduction in generation-to-SMILES time under that separate protocol. Tables 28 and 29 report the two evaluations.

## E.1 SHARED TRAINING AND SCAFFOLD-ONLY INFERENCE

The benchmark contains 100 scaffold queries from the held-out ZINC partition. Molecules are grouped by canonical, nonisomeric Bemis-Murcko scaffold identity before splitting. The frozen split contains 68,296 training, 4,017 validation, and 8,035 test scaffold identities. The query scaffold identities are therefore absent from the training partition of both methods. The decoders receive only the scaffold, without the corresponding held-out molecule or its target size.

EGO uses the scaffold-conditioned rooted forest decoder described in Appendix C. Training combines the Drift distribution matching objective, structural constraints, and a rarity reward, as specified in Appendices C and C. During training, the condition includes the number of atoms added to the scaffold in each example. At inference, this number is sampled from a prior estimated only from the training partition. The selected checkpoints are reused without further training.

The three EGO checkpoints use training seeds 42, 43, and 44. Seeds 43 and 44 complete 30,000 updates and use checkpoints selected at updates 21,000 and 20,000, respectively. Seed 42 uses the archived checkpoint from the original experiment. The original validation procedure supplies a target size and first prioritizes feasibility. It then combines completion success, distinct completion yield, and scaffold coverage, with penalties for feature discrepancy and constraint violations. The selected checkpoints for seeds 43 and 44 do not meet the original validation feasibility thresholds. They are retained under the same selection rule and are not reselected using the screening results reported here.

The size prior records the number of heavy atoms added to each training scaffold. It is conditioned on scaffold size when that size occurs in training, with the pooled distribution of training additions as a fallback. Only positive additions that yield a supported total size of at most 38 heavy atoms are retained, and the distribution is renormalized. The procedure uses neither the held-out target size nor clipping based on that target. Core atom states and their ordering are reconstructed from the public scaffold SMILES. Randomness is assigned deterministically from the generation seed, query identity, and attempt index. Unsupported cores and failed conversions remain counted attempts. Failed proposals are not replaced, and conversion uses the complete graph without chemical repair or rescue of the largest component.

MoLeR is trained from scratch, without external pretrained weights. The training runner exports the original ZINC training and validation partitions using the frozen scaffold assignments. It does not create a new random split. The benchmark test partition is excluded from preprocessing and checkpoint selection. The test input required by the MoLeR pipeline is populated with a copy of the validation partition, rather than the benchmark test molecules. Training uses the MoLeR VAE procedure with validation-based checkpoint selection.

We evaluate one fixed MoLeR checkpoint with sampling seeds 41, 42, and 43 and beam size 1. The decoder receives the scaffold and a latent sampled from the prior, without a requested final size. The initial diagnostic run with sampling seed 123 is excluded from the reported means. Thus, the EGO results describe three trained checkpoints, while the MoLeR results describe sampling variability for one trained checkpoint. The shared scaffold queries provide the pairing for comparisons within each query. The repeats do not constitute paired training experiments.

## E.2 SHARED USABILITY AND SCREENING CRITERIA

A usable completion must pass chemical sanitization of the complete molecule, be connected, and contain the supplied scaffold as a substructure. An induced exact scaffold match is recorded separately and is not required by the primary criterion. The completion does not need to match the heavy atom count of the held-out reference molecule. For distinct-library metrics, usable outputs are deduplicated by canonical, nonisomeric SMILES within each query. The same molecular identity can contribute to different queries, but repeated identities within a query count only once.

Screening requires $\mathrm { Q E D } \geq 0 . 7 , \mathrm { S A } \leq 4$ , and no additional PAINS alert types relative to the scaffold. QED measures drug likeness (Bickerton et al., 2012), and SA estimates synthetic accessibility (Ertl and Schuffenhauer, 2009). We use the RDKit implementation of QED, the RDKit Contrib implementation of SA, and the RDKit PAINS filter catalog. If A(x) and A(c) are the sets of alert types for completion x and scaffold $c ,$ the PAINS rule is

$$
A ( x ) \setminus A ( c ) = \emptyset .
$$

Alert types already present in the core are permitted. The rule therefore does not require every completion to be free of PAINS alerts. These screening criteria are applied during evaluation and are not included in the training objective of EGO or used to select the reported checkpoints.

Novelty relative to training molecules is not required by the screening criterion.

## E.3 GENERATION AND SCREENING AT MATCHED TIME

This primary application experiment tests whether faster generation allows more candidates to be evaluated within a short time budget; its aggregate results appear in the main text. The experiment uses the same 100 scaffold queries and the same selected checkpoints, with no further training. The three replicates combine EGO training seeds 42, 43, and 44 with MoLeR sampling seeds 41, 42, and 43, respectively. Both methods run on the same NVIDIA A100-SXM4-40GB GPU, with models loaded and warmed up before timing.

For each scaffold and replicate, MoLeR generates one batch of 256 proposals with beam size 1 and six decoding workers. The measured time T includes query preparation, latent sampling, decoding to SMILES, communication with the scoring process, screening, and selection of the eligible candidate with the highest QED. This measured interval defines the time budget for EGO on the same query. The budget therefore varies across queries and replicates.

EGO generates batches of 256 graphs and converts them directly into RDKit molecules for screening. It does not serialize every candidate to SMILES inside the timed loop. Only candidates fully scored and selected by time T are eligible. Both methods apply the same usability checks and QED, SA, and scaffold relative PAINS filters. A query with no eligible candidate is recorded as a failure. The first eligible candidate attaining the maximum QED is retained in a tie, and comparisons between methods use a tolerance of $1 0 ^ { - 1 2 }$ . Model loading, warmup, result I/O, and export of the selected EGO molecule to SMILES are outside the timed interval. The largest recorded overrun of a final EGO operation is 0.005853 seconds, and results completed after $\bar { \mathbf { \Gamma } } _ { T }$ are excluded. This protocol measures candidate selection within a common time budget. Its output representation and timing boundaries differ from the generation-to-SMILES measurement in Section E.5.

Table 28: Generation and screening at matched time. Each column reports one replicate on the same 100 scaffold queries; headers identify the EGO training seed and MoLeR sampling seed. Screened counts are means per query; passing counts are totals over the panel. Passing fractions use all screened proposals. The outcome block partitions all 100 queries. Higher-QED counts require passing candidates from both methods; exclusive successes and failures of both methods remain explicit.
<table><tr><td>Metric</td><td>42/ 41</td><td>43 / 42</td><td>44 / 43</td></tr><tr><td>Mean deadline per query (s)</td><td>6.176</td><td>6.105</td><td>5.974</td></tr><tr><td>Mean proposals screened, EGO</td><td>1,822.94</td><td>1,947.70</td><td>1,907.35</td></tr><tr><td>Mean proposals screened, MoLeR</td><td>256.0</td><td>256.0</td><td>256.0</td></tr><tr><td>Filter-passing proposals, EGO</td><td>55,152</td><td>52,975</td><td>56,064</td></tr><tr><td>Filter-passing proposals, MoLeR</td><td>5,434</td><td>5,485</td><td>5,417</td></tr><tr><td>Passing fraction (%), EGO</td><td>30.25</td><td>27.20</td><td>29.39</td></tr><tr><td>Passing fraction (%), MoLeR</td><td>21.23</td><td>21.43</td><td>21.16</td></tr><tr><td>Queries with a hit, EGO</td><td>79</td><td>79</td><td>79</td></tr><tr><td>Queries with a hit, MoLeR</td><td>85</td><td>85</td><td>85</td></tr><tr><td colspan="4">Query outcomes (counts per 100 queries)</td></tr><tr><td>Both pass: higher QED, EGO</td><td>23</td><td>34</td><td>33</td></tr><tr><td>Both pass: higher QED, MoLeR</td><td>55</td><td>44</td><td>46</td></tr><tr><td>Equal QED with hits from both</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Hit from EGO only</td><td>1</td><td>1</td><td>0</td></tr><tr><td>Hit from MoLeR only</td><td>7</td><td>7</td><td>6</td></tr><tr><td>No hit from either method</td><td>14</td><td>14</td><td>15</td></tr></table>

Table 28 reports all three replicates over the same 100-query panel. The mean of the three replicatelevel mean deadlines is $6 . 0 8 5 \pm 0 . 1 0 3$ seconds per query. The corresponding numbers of proposals fully screened per query are $1 , 8 9 2 . 7 \pm 6 3 . 7$ for EGO and $2 5 6 . 0 \pm 0 . 0 \dot { }$ for MoLeR. The ratio of these means is 7.39.

The saved per-query records also contain the number of distinct proposals that pass all screening criteria before the deadline. These counts are $5 4 7 . 3 \pm 1 5 . 9$ per query for EGO and $5 4 . 5 \pm 0 . 4$ for MoLeR. Their ratio of means is 10.05.

Let $P _ { s r } ^ { ( m ) }$ and ${ \cal B } _ { s r } ^ { ( m ) }$ denote passing and screened proposal counts for method $m _ { : }$ , scaffold $s ,$ and replicate r. Each replicate contains $S = 1 0 0$ queries. Its mean passing count per query and passing fraction are

$$
\overline { { { P } } } _ { r } ^ { ( m ) } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } P _ { s r } ^ { ( m ) } , \qquad \pi _ { r } ^ { ( m ) } = \frac { \sum _ { s = 1 } ^ { S } P _ { s r } ^ { ( m ) } } { \sum _ { s = 1 } ^ { S } B _ { s r } ^ { ( m ) } } .
$$

We report means and sample SDs of these replicate summaries. Passing fractions are $2 8 . 9 5 \pm 1 . 5 8 \%$ for EGO and 21. $. 2 7 \pm 0 .$ .14% for MoLeR. These are averages of the three replicate fractions, rather than an average of per-query percentages.

All aggregate uncertainties in the main matched-time table are sample standard deviations of the three replicate summaries, not standard deviations across individual queries or standard errors. The means and SDs are computed from the full per-query records before rounding. The same scaffolds are reused across replicates, and the two methods’ seeds represent different sources of variation.

To account for every query, we distinguish outcomes where both methods return passing candidates, and one has a higher maximum QED, where only one method returns a passing candidate, and where neither succeeds. Equal-QED outcomes are recorded separately; none occur here. These categories partition the 100 queries in each replicate. Coverage is a separate summary that counts all queries where the specified method returns a passing candidate.

For the comparison of passing counts across scaffolds, we first average the three replicate counts separately for each query and method. EGO has a higher mean passing count on 73 queries,

MoLeR on 14, and 13 queries tie. These are counts over the same 100-query panel. Within individual replicates, the numbers of EGO wins, MoLeR wins, and ties are respectively (71, 14, 15), (72, 14, 14), and (71, 14, 15).

Averaging the outcome counts across replicates, both methods pass, and EGO has a higher maximum QED on $3 0 . 0 \pm 6 . 1$ queries per 100; MoLeR has higher maximum QED on $4 8 . 3 \pm 5 . 9$ . Only EGO passes on $0 . 7 \pm 0 . 6$ queries, only MoLeR passes on $6 . 7 \pm 0 . 6 $ , and neither passes on $1 4 . 3 \pm 0 . 6$ These five observed outcomes account for the entire query panel, without assigning an artificial QED to failures or combining feasibility and quality into a new score.

MoLeR covers 85 queries in every replicate, compared with 79 for EGO. Its higher coverage and more frequent higher-QED outcomes coexist with the greater proposal throughput of EGO. These are descriptive comparisons on repeated evaluations of the same 100 scaffolds, not significance tests on 300 independent queries.

Figure 6 shows the selected candidates for scaffold 23 in all three replicates. The scaffold was specified before examining the timed results. The third replicate corresponds to the 5.13-second example in the main text. Within this interval, EGO generates 3,072 graphs and completes evaluation of 2,830 proposals, of which 469 pass screening. MoLeR generates and evaluates 256 proposals, of which 102 pass screening. The highest QED is 0.919 for EGO and 0.946 for MoLeR, with corresponding SA scores of 3.77 and 3.32. Generated graphs that remain unevaluated at the deadline do not enter the selection.

The matched-time experiment establishes higher screening throughput and more filter-passing proposals for EGO under the stated protocol. The next experiment assesses distinct candidate libraries at a fixed proposal budget. The screening filters are heuristic and do not establish experimental usefulness.

## E.4 DISTINCT CANDIDATE LIBRARIES AT A FIXED PROPOSAL BUDGET

This complementary evaluation uses the same selected checkpoints and 100 held-out scaffold queries. Each method makes 1,000 proposals per query, giving 100,000 proposals per run. There is no shared time deadline in this experiment. Screening uses the criteria in Section E.2, and distinct counts remove repeated identities within each query.

Let $U _ { s r }$ be the number of distinct usable completions for scaffold s in run r, and let $F _ { s r }$ be the number that also pass screening. The distinct usable yield is $1 0 0 \sum _ { s } U _ { s r } / 1 0 0 , 0 0 0 .$ . The distinct filtered count is $\sum _ { s } F _ { s r }$ , and the conditional fraction that passes screening is $\begin{array} { r } { 1 0 0 \sum _ { s } F _ { s r } / \sum _ { s } U _ { s r } , } \end{array}$ Coverage counts queries with $F _ { s r } > 0$ . The usable completion rate includes repeated proposals, while distinct usable yield removes them. Invalid and failed attempts remain in all proposal-based denominators.

Table 29 reports the aggregate results. MoLeR produces more distinct usable completions, with a yield of 85.80% compared with 66.78% for EGO. After screening, the ordering reverses. The fraction of distinct usable completions that passes the filters is approximately 30.3% for EGO and 19.4% for MoLeR. Consequently, EGO returns 20,245 distinct filtered completions on average, compared with 16,684 for MoLeR. The ratio of these mean counts is 1.21.

Table 30 gives the exact filtered counts for the individual runs. Means are unweighted across runs, and sample SD uses denominator $n - 1$ . Conditional fractions are calculated within each run before averaging. Outputs from different runs are not pooled into a larger library.

For the query comparison, we first average the distinct filtered counts across the three runs separately for each scaffold and method. EGO has a higher mean count on 51 scaffolds, with 13 ties and 36 MoLeR wins. Its larger total library therefore coexists with lower coverage, at $7 7 . 7 \pm 1 . 5$ queries compared with $8 6 . 3 \overset { \cdot } { \pm } 0 . 6$ for MoLeR. These counts describe the observed query comparisons and are not significance tests. The reported SD values also describe different sources of variation for the two methods. They do not estimate uncertainty over independently trained MoLeR models.

Figure 7 shows examples of distinct completions around the same supplied core. Figure 8 illustrates completions with larger additions to the core. Both figures retain qualitative examples from the earlier comparison with externally pretrained MoLeR. The quantitative results in Table 29 use the new MoLeR checkpoint trained from scratch on ZINC.

![](images/95b773e20c0ed6c07c160c8c573eafb389532734e8f06777426227eb1137deb3.jpg)  
Figure 6: Highest QED completions that pass screening for the prespecified scaffold 23 at matched time. Each row shows one replicate, using EGO training seeds 42, 43, and 44 and sampling seeds 41, 42, and 43 from the same MoLeR checkpoint trained on ZINC. The common budgets are 5.64, 5.55, and 5.13 seconds. The supplied scaffold is highlighted. Table 28 reports results over all 100 queries.

## E.5 GENERATION TO SMILES TIME

The timing row in Table 29 measures generation of the full proposal bank and conversion to SMILES. For EGO, the interval includes size sampling, synchronized graph generation, and conversion of the complete graphs. It excludes setup, compilation, warmup, file I/O, and subsequent screening. For MoLeR, the interval includes latent sampling and decoding to SMILES, including the first decoding call. It excludes model loading, file I/O, and subsequent screening.

The recorded EGO times are 32.140, 46.343, and 46.973 seconds for training seeds 42, 43, and 44. Their mean is 41.82 ± 8.39 seconds, compared with 1,851.82 ± 8.05 seconds for the three MoLeR sampling runs. The ratio of mean times is

$$
{ \frac { 1 , 8 5 1 . 8 2 } { 4 1 . 8 1 8 7 } } \approx 4 4 . 3 .
$$

This ratio describes the recorded generation runs under the stated timing boundaries. It is not the mean of ratios between seeds and does not measure the complete screening pipeline. The experiment in Section E.3 separately matches the available time for generation and screening on the same GPU.

Table 29: Scaffold completion at the same proposal budget. Each run uses 100 queries and 1,000 proposals per query. Values are the mean ± sample SD over three EGO training seeds and three sampling seeds from one MoLeR checkpoint trained from scratch on ZINC. Distinct-library counts are deduplicated within each query. The filter passing fraction is conditional on distinct usable completions. Generation time excludes screening.
<table><tr><td>Metric</td><td>EGO</td><td>MoLeR</td></tr><tr><td>Usable completions (%)</td><td> $7 7 . 6 7 \pm 2 . 5 7$ </td><td> $9 7 . 5 8 \pm 0 . 0 9$ </td></tr><tr><td>Distinct usable yield (%)</td><td> $6 6 . 7 8 \pm 1 . 1 5$ </td><td> $8 5 . 8 0 \pm 0 . 1 2$ </td></tr><tr><td>Filter passing fraction (%)</td><td> $3 0 . 3 \pm 1 . 1$ </td><td> $1 9 . 4 \pm 0 . 1$ </td></tr><tr><td>Distinct filtered completions</td><td> $2 0 , 2 4 5 \pm 7 5 5$ </td><td> $1 6 { , } 6 8 4 \pm 7 9$ </td></tr><tr><td>Scaffold coverage (/100)</td><td> $7 7 . 7 \pm 1 . 5$ </td><td> $8 6 . 3 \pm 0 . 6$ </td></tr><tr><td>Generation to SMILES (s)</td><td> $4 1 . 8 2 \pm 8 . 3 9$ </td><td> $1 , 8 5 1 . 8 2 \pm 8 . 0 5$ </td></tr></table>

Table 30: Distinct completions that pass screening in each run. Each count is the sum of distinct counts within 100 queries at 1,000 proposals per query. MoLeR sampling seeds share one checkpoint trained with seed 42.
<table><tr><td>Method</td><td>Seed type</td><td>Seed</td><td>Distinct filtered count</td></tr><tr><td>EGO</td><td>Training</td><td>42</td><td>19,527</td></tr><tr><td>EGO</td><td>Training</td><td>43</td><td>20,175</td></tr><tr><td>EGO</td><td>Training</td><td>44</td><td>21,033</td></tr><tr><td>MoLeR</td><td>Sampling</td><td>41</td><td>16,763</td></tr><tr><td>MoLeR</td><td>Sampling</td><td>42</td><td>16,683</td></tr><tr><td>MoLeR</td><td>Sampling</td><td>43</td><td>16,606</td></tr></table>

## F RQ3: BEYOND MOLECULAR GENERATION

We test the same one-step generator framework on neural architectures and binary images. NAS measures the yield of distinct high-performing candidates; MNIST tests distribution matching when graph feasibility is no longer a constraint. These experiments help identify where the training approach is useful and where gradient-based alternatives perform better.

## F.1 NAS-BENCH-101

Summary. NAS-Bench-101 (Ying et al., 2019) tests whether the candidate-library result extends beyond chemistry. We train on high-performing architectures and measure how many distinct held-out elite architectures each method returns within 10,000 proposals. Validity, mean accuracy, and feature distance reveal the trade-offs involved in producing a larger library.

Takeaway. EGO returns more distinct held-out elites than STE or Soft (Table 2). Soft achieves higher validity and mean accuracy, and closer feature-distribution matching. The result supports EGO when the aim is to obtain more qualifying alternatives for subsequent comparison.

Shared protocol and fitness. We compare EGO, the straight-through estimator (STE), and Soft using seeds 42–44 and 15,000 updates per run. The generator, initialization, sampling streams, and checkpoint-selection rule are shared within each seed. Each generator maps 128-dimensional noise through three 512-unit hidden layers to 21 binary edge decisions and five operation choices on a seven-vertex graph. Pruning removes vertices outside input-to-output paths.

The fitness in Eq. (28) combines negative Energy with a structural penalty. Energy compares edge and operation choices, degrees, and path structure with those of training elites. The penalty, with fixed weight 0.5, discourages missing input-to-output paths and more than nine edges after pruning. Its purpose is to enforce the benchmark’s feasibility rules while learning the structure of high-performing designs. There is no rarity reward or explicit duplicate penalty; Energy itself rewards spread among generated samples.

Supplied core highlighted. All shown completions pass QED >= 0.7, SA <= 4 and no-new-PAINS.  
![](images/61c6aa963d4e2f0396a0293d5e8b6b7e364c0c3a1c251180ee6b925e74b8536f.jpg)  
Figure 7: Examples of scaffold completions from EGO with training seed 43 and externally pretrained MoLeR with sampling seed 42. The supplied core is highlighted. All displayed molecules pass the screening criteria. Three scaffolds were selected by a fixed hash ordering among queries with at least two distinct passing completions from both methods. The first two distinct passing completions in proposal order are shown for each method and scaffold.

EGO evaluates this fitness on hard graphs. STE uses the same hard forward values with derivatives from relaxed features; Soft evaluates the relaxed objective. Each objective evaluation uses 64 generated graphs and 192 training-elite references; EGO evaluates a population of 192 candidates. Appendix C.3 defines the features, normalization, fitness terms, optimizer settings, and common selector.

Data split and checkpoint selection. We use the benchmark’s 108-epoch records and split architectures 80/10/10 with seed 2026. An architecture is elite when its benchmark validation accuracy is at or above the 90th percentile in the architecture training partition. This threshold is fixed for all three partitions. Training elites supply the target distribution; validation-partition elites and a fixed noise bank support checkpoint selection. Initialization is not eligible for selection.

The architecture partition and benchmark accuracy split serve different purposes. A held-out test elite is an architecture in the test partition that qualifies using its benchmark validation accuracy. Its benchmark test accuracy is a separate final metric. Test-partition architectures are excluded from the training and selection reference sets, and their benchmark labels are used only for final evaluation. Benchmark test accuracy is never used for training or checkpoint selection. Feature normalization and kernel bandwidth are fitted only on training elites.

Final metrics and interpretation. Each selected model produces 10,000 proposals. Held-out elite yield divides the number of distinct generated test elites by all 10,000 attempts, including invalid graphs and duplicates. Valid-and-unique yield in Table 2 likewise divides distinct valid architectures by all attempts. Architectures are identified by the benchmark’s canonical hash after pruning, so

![](images/67f65ef1c7c68c4db4057891b919b6ad45bfa0baaebd9f56df280255cddebaff.jpg)  
Supplied core highlighted. Examples selected for larger extensions; all pass the screening filters. Added atoms exclude hydrogen. QED >= 0.7; SA <= 4; no new PAINS alert types.

Figure 8: Selected examples of larger scaffold extensions from EGO with training seed 44 and externally pretrained MoLeR with sampling seed 42. The supplied core is highlighted, and all displayed molecules pass screening. Scaffolds were ranked by the second-largest addition of heavy atoms from EGO among queries with at least two distinct passing completions from both methods. The three selected scaffolds show the two largest passing extensions from each method. These examples illustrate larger extensions rather than typical outputs.

equivalent graph encodings do not count as different designs. All valid final proposals are found in the benchmark lookup. Mean test accuracy averages over these valid proposals, retaining duplicates.

Feature distance is the squared maximum mean discrepancy (MMD<sup>2</sup>) between raw concatenated graph features from up to 1,500 valid proposals and 1,500 test-partition elites. Its radial basis function kernel has a bandwidth fixed from training elites. It measures agreement with elite structure, separately from accuracy and distinct candidate yield.

The held-out elite yields in Table 2 correspond to 221.7 ± 7.5 distinct test elites for EGO, compared with $1 3 8 . 0 \pm 2 8 . 9$ for STE and 143.7 ± 18.6 for Soft. Among valid proposals, the fractions qualifying as elite across all architecture partitions are 42.81 ± 2.97%, 60.27 ± 2.93%, and 64.35 ± 1.96%, respectively. These fractions include repeated architectures. Together, the metrics show why a higher concentration of elites need not produce a larger collection of distinct held-out elites. All ± values are sample standard deviations across the three runs.

Training work and timing. The runs match update counts, not objective evaluations or training time. Over 15,000 updates, EGO evaluates 184.32 million candidate graphs plus 0.96 million unperturbed diagnostic graphs. STE and Soft each evaluate 0.96 million candidate graphs. Each method draws 2.88 million reference examples. Table 31 separates optimizer-update time, compilation, and checkpoint validation; initialization and final evaluation are excluded

Table 31: NAS training time in seconds, mean ± sample standard deviation across seeds 42–44. Columns measure separate stages of the 15,000-update runs; none alone is total runtime.
<table><tr><td>Method</td><td>Update time</td><td>Compilation</td><td>Validation</td></tr><tr><td>EGO</td><td> $4 6 . 1 2 \pm 0 . 5 9$ </td><td> $4 8 . 4 1 \pm 0 . 7 5$ </td><td> $3 9 . 5 8 \pm 1 . 9 1 $ </td></tr><tr><td>STE</td><td> $3 4 . 0 6 \pm 0 . 3 8$ </td><td> $5 4 . 1 8 \pm 0 . 8 8$ </td><td> $3 0 . 8 4 \pm 1 . 0 1$ </td></tr><tr><td>Soft</td><td> $3 4 . 3 7 \pm 0 . 6 5$ </td><td> $5 4 . 5 1 \pm 1 . 1 8$ </td><td> $1 6 . 3 6 \pm 1 . 2 8$ </td></tr></table>

## F.2 BINARIZED MNIST CONTROL

Summary. Binarized MNIST tests whether the NAS result extends to image generation when every sampled output is already a valid binary array. We compare image-distribution distances and predicted digit coverage to assess what the methods learn beyond producing distinct outputs. The generator learns from image features without digit labels or classifier feedback.

Takeaway. Soft achieves better distribution matching and broader predicted digit coverage than EGO in every paired seed (Tables 32 and 33). EGO produces almost no exact duplicates, but small pixel differences do not establish semantic diversity. Its NAS candidate-yield advantage therefore does not imply better generation on every discrete task.

Shared protocol and fitness. Images are $2 8 \times 2 8$ and binarized at intensity 128 on the 0–255 scale. With split seed 2026, we reserve 500 images per digit for validation, leaving 55,000 training and 5,000 validation images. The official test set is used only after checkpoint selection.

All methods use the same three-layer, 768-unit generator, with 96-dimensional noise and 784 pixel logits. One forward pass and parallel Bernoulli sampling produce each binary image. Within each seed, the initial parameters, latent noise, pixel-sampling noise, and reference samples are shared. Each run completes 30,000 updates with 96 generated images and 192 references per objective evaluation; EGO uses 192 candidates.

The fitness is the negative average Energy over six feature families: pixels, two pooling scales, row and column projections, neighboring-pixel differences, and global image statistics. These features compare fine detail and larger shapes without requiring a trained classifier. Normalization is fixed from training images, and each family’s Energy is divided by the square root of its dimension so that a larger feature vector does not dominate solely through its size. There are no extra penalties or rewards. Table 7 lists the features, and Eq. (30) defines the complete fitness.

EGO uses hard images and mean-centered fitness without rank shaping. STE preserves the hard forward Energy while using relaxed derivatives; Soft evaluates relaxed images. Both gradient methods use temperature 0.75, and all final evaluations use binary samples. Appendix C.4 gives the remaining training settings.

Checkpoint selection. Every 500 updates, the same selector scores 2,000 proposals from a fixed noise bank. It combines exact image uniqueness with agreement between generated and validation feature means and variances, as defined in Eq. (31) and the accompanying selection rule. This rewards variety and distribution matching without classifier feedback. Initialization is excluded. Digit label are used only to stratify the split and train the separate evaluator.

Independent evaluation. One frozen LeNet-style classifier is shared across all nine generator runs. It is trained on the 55,000 model-training images and labels using seed 2026, 2,500 updates, batch size 256, and learning rate $1 0 ^ { - 3 }$ . Its accuracy on real test images is 98.98%; this is not an accuracy measurement for generated images.

Distribution distances use 2,000 generated and 2,000 real test images. LeNet Frechet distance´ compares means and covariances of raw 128-dimensional classifier embeddings. Kernel inception distance (KID) is computed here on LeNet embeddings, using the polynomial kernel $k ( a , b ) =$ $( a ^ { \top } b / 1 2 8 + 1 ) ^ { 3 }$ , where $a , b \in \mathbb { R } ^ { 1 2 8 }$ are standardized embeddings. Thus, these scores do not use Inception features. Table 32 reports Frechet distance alongside predicted digit coverage.´

Table 32: Controlled Energy comparison on binarized MNIST, mean ± sample standard deviation over seeds 42–44. Distances use 2,000 generated and 2,000 test images. Coverage uses all 10,000 generated images, with at least 50 predictions required to count a digit class.
<table><tr><td>Method</td><td></td><td>LeNet Fréchet ↓ Predicted classes / 10 ↑</td></tr><tr><td>EGO</td><td> $5 9 9 . 8 \pm 4 7 . 1$ </td><td> $6 . 3 3 \pm 0 . 5 8$ </td></tr><tr><td>STE</td><td> $5 2 2 . 4 \pm 9 5 . 6$ </td><td> $7 . 6 7 \pm 0 . 5 8$ </td></tr><tr><td>Soft</td><td> ${ \bf 4 2 0 . 6 \pm 8 9 . 1 }$ </td><td> ${ \bf 8 . 3 3 \pm 0 . 5 8 }$ </td></tr></table>

We also compute $\mathrm { { \bf M M D } ^ { 2 } }$ using a radial basis function (RBF) kernel, both on standardized LeNet embeddings and on the 49 raw means from nonoverlapping $4 \times 4$ image blocks. The latter provides an image distance independent of the classifier. For feature vectors a and b from either representation, the kernel is $k ( a , b ) = \mathrm { { \dot { \ e x p } } } [ - \| a - b \| _ { 2 } ^ { 2 } / ( 2 \beta ) ]$

For each representation, β is the median positive squared pairwise distance on a fixed subset of 2,000 training images (seed 2028), floored at $\bar { 1 } 0 ^ { - 8 }$ . Embedding means and standard deviations are fitted on the same subset, with a standard-deviation floor of 0.05. KID and $\mathrm { R B F - M M D ^ { 2 } }$ use unbiased estimators. All metric settings are fixed before test evaluation.

Distribution matching and digit coverage. Table 33 reports complementary distances and classbalance diagnostics. Class metrics use the most likely LeNet label for each of 10,000 generated images, without confidence filtering. A digit is covered when at least 50 predictions belong to it. Total variation (TV) is half the sum of absolute differences between predicted class frequencies and real test-label frequencies. Lower TV indicates closer agreement. Class entropy is Shannon entropy divided by log(10), so one denotes uniform predicted frequencies across all ten digits.

Table 33: Additional MNIST diagnostics, mean ± sample standard deviation across seeds 42–44. LeNet KID uses classifier embeddings; pooled-image $\mathrm { { \bf M M D ^ { 2 } } }$ uses raw block means. Class metrics use predicted labels on 10,000 images.
<table><tr><td>Metric</td><td>EGO</td><td>STE</td><td>Soft</td></tr><tr><td>LeNet KID ↓</td><td> $0 . 4 5 1 7 \pm 0 . 0 4 9 0$ </td><td> $0 . 3 8 6 8 \pm 0 . 0 8 5 3$ </td><td> $0 . 3 0 6 9 \pm 0 . 0 7 1 8$ </td></tr><tr><td>Pooled-image  $\mathrm { { \bf M M D } ^ { 2 } \downarrow }$ </td><td> $0 . 0 4 1 4 \pm 0 . 0 0 0 8$ </td><td> $0 . 0 2 3 5 \pm 0 . 0 0 9 4$ </td><td> $0 . 0 1 4 5 \pm 0 . 0 0 6 2$ </td></tr><tr><td>Class-balance  $\mathrm { T V } \downarrow$ </td><td> $0 . 5 4 5 \pm 0 . 0 2 0$ </td><td> $0 . 4 6 3 \pm 0 . 0 4 5$ </td><td> $0 . 4 0 1 \pm 0 . 0 9 9$ </td></tr><tr><td>Normalized class entropy ↑</td><td> $0 . 6 2 8 \pm 0 . 0 3 5$ </td><td> $0 . 7 3 8 \pm 0 . 0 4 4$ </td><td> $0 . 7 5 4 \pm 0 . 1 0 8$ </td></tr></table>

Soft improves on EGO in Frechet distance, KID, both RBF-MMD ´ <sup>2</sup> measures, coverage, entropy, and class-balance TV in each paired seed. STE also improves mean Frechet distance and coverage over´ EGO (Table 32). Coverage remains incomplete for all methods (Table 34).

EGO’s exact image uniqueness is $9 9 . 9 8 \pm 0 . 0 3 \%$ , despite its lower predicted digit coverage. No generated image exactly matches a training image in any run, but this pixel-level test cannot rule out copies with small changes. Figure 9 presents sample grids for the three methods; the quantitative comparison uses the full evaluation sets described above.

Selected checkpoints and cost. Table 34 records the selected update for each seed. Some STE and Soft checkpoints are selected early, but every run still completes 30,000 updates. The selected iteration therefore differs from the completed training budget.

EGO spends $1 1 8 . 6 0 \pm 3 . 5 0$ seconds on optimizer updates, compared with $6 7 . 2 7 \pm 0 . 2 3$ for STE and $6 7 . 7 0 \pm 0 . 1 5$ for Soft. These times exclude compilation, checkpoint validation, evaluator training, and final evaluation. EGO evaluates 552.96 million candidate images plus 2.88 million unperturbed diagnostic images; each gradient method evaluates 2.88 million candidate images. All methods draw 5.76 million reference images.

Generating 10,000 binary images takes about 29–30 milliseconds for each method. This includes sampling and transfer to host arrays, but excludes warmup, classifier evaluation, metrics, and file

Table 34: MNIST checkpoint selection and predicted digit coverage. Entries follow seed order $4 2 .$ 43, 44. Every run completes 30,000 updates; coverage is measured on 10,000 final images.
<table><tr><td>Method</td><td></td><td>Selected update Predicted classes / 10</td></tr><tr><td>EGO</td><td> $2 9 , 0 0 0 / 3 0 , 0 0 0 / 3 0 , 0 0 0$ </td><td> $6 / 6 / 7$ </td></tr><tr><td>STE</td><td> $5 0 0 / 3 0 { , } 0 0 0 / 5 0 0$ </td><td> $8 / 8 / 7$ </td></tr><tr><td>Soft</td><td> $1 , 5 0 0 / 2 8 , 0 0 0 / 1 , 0 0 0$ </td><td> $8 / 9 / 8$ </td></tr></table>

writing. All three methods retain the same one-step inference procedure; their differences concern training cost and the distribution learned.

![](images/a579cd32aea206ed9e5a9958198a77801f0a6f7f052eaaa363ba681fdca4fad1.jpg)  
Soft  
Figure 9: Binarized MNIST samples generated by EGO, STE, and Soft, from top to bottom.