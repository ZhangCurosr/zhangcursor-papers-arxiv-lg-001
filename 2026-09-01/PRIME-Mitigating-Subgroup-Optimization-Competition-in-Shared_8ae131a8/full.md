# PRIME: Mitigating Subgroup Optimization Competition in Shared CTR Top Networks with Plug-in Residual Input-Conditioned Mixture of Experts

Heng Yao<sup>†</sup> Ant Group

Siyun Hou<sup>†</sup> Henan Polytechnic University

Tianying Liu Independent Researcher

Yulou Shu Alibaba Inc.

## Abstract

Click-through rate (CTR) models vary in feature-interaction design, yet their top networks usually remain a single multilayer perceptron shared by all examples. Heterogeneous user, item, and context subgroups therefore update the same parameters; weakly aligned learning signals make the aggregate gradient a compromise among competing directions. We study the competition on Avazu with 4 models and 4 semantic fields. Across all architectures, semantic subgroups show lower Top-NN gradient cosine similarity than random groups matched by sample size and label ratio, with reductions of 0.23–0.37. This gap shows that semantic heterogeneity imposes less compatible optimization demands on a shared Top-NN.

This competition motivates input-conditioned experts, but directly replacing an established Dense mapping changes its initial function, sharing patern, and capacity, obscuring the source of gains. We introduce PRIME (Plug-in Residual Input-conditioned Mixture of Experts), a Dense-anchored mixture of low-rank residual experts. PRIME anchors the original prediction and uses zeroresidual initialization to match the Dense baseline exactly at training onset. Input-dependent routing weights low-rank experts for example-specific logit corrections; multi-bag aggregation and EMA load biases stabilize conditional estimation.

We evaluate PRIME on held-out Avazu and Criteo test sets across 13 CTR architectures and five paired seeds. PRIME improves mean AUC for 11 of 13 architectures on each dataset. Median paired AUC gains are +0.0022 and +0.0066, with LogLoss reductions of 0.0011 and 0.0081, respectively. On FiBiNET and DCNv2, PRIME outperforms APG in all ten seed-level AUC comparisons while using fewer parameters and lower inference latency on both backbones. These results show that function-preserving conditional residuals add input-dependent capacity while preserving the Dense path and its optimization stability. Code is available at https://github.c om/YH-learning/PRIME.

## CCS Concepts

• Information systems → Recommender systems; • Comput ing methodologies → Machine learning.

Yong He, Chuan Yuan, Kaibin   
Qiu, Guowei Chen, Jiayu Zhao, Chao Yu, and Ke Ding Ant Group

## Keywords

click-through rate prediction, recommendation ranking, mixture of experts, shared top network, low-rank residual, conditional routing

## 1 Introduction

## 1.1 CTR Prediction and the Fully Shared Top-NN

Click-through rate (CTR) prediction is a basic component of recommendation ranking and online advertising. Given user, item, and context features, a CTR model estimates the probability that an impression will receive a click. This probability afects candidate ranking, trafic allocation, ad auctions, and downstream value estimation. A useful CTR model must therefore balance ranking discrimination, probability quality, training stability, and inference cost [4, 23].<sup>1</sup>

A typical CTR model comprises an embedding layer, a featureinteraction module, and a top prediction network. Sparse categorical features are first mapped to dense vectors. An interaction module then combines information across fields, and a shared decision path produces the click probability. We use Top-NN to denote the shared MLP decision path found in most deep CTR architectures. Architectural innovation over the past decade has concentrated largely on the interaction module. The DCN family constructs explicit feature crosses [18, 19], AutoInt learns field relations through self-atention [15], FiBiNET introduces bilinear interactions [8], and MaskNet modulates features with instance-specific masks [20]. Despite their diferent inductive biases, these models typically retain a Dense Top-NN whose parameters are updated by every training example.

This design makes a consequential but rarely tested assumption: examples may require diferent interaction mechanisms, but a single post-interaction mapping is suficient for all of them. A shared MLP is parameter-eficient and expressive, which has made it a reliable default. CTR data, however, are heterogeneous along user, advertisement, site, device, and temporal dimensions. Whether one fully shared top mapping remains the right parameterization under such heterogeneity has received litle direct examination.

Semantic subgroups reduce shared Top-NN gradient alignment

![](images/be59da180d27fab861cc3973e89788eeb1e781367eef17a7b54209ca1d52d1b9.jpg)  
Figure 1: Semantic subgroups exhibit lower gradient align ment in the shared Top-NN. Each bar averages three trained checkpoints and four semantic fields. Blue bars denote matched random groups, and orange bars denote seman tic subgroups. The short horizontal markers and arrows show the diference between them. Semantic subgroups have lower gradient similarity in all four architectures.

Semantic subgroups may rely on diferent predictive cues. The same advertisement can elicit diferent responses across mobile and desktop trafic, while in-app and web trafic may depend on diferent feature combinations. A shared Top-NN represents these paterns with the same parameters. Sharing pools observations but provides no mechanism for selecting input-specific parameter di rections. The resulting constraint concerns optimization, not capacity alone. If subgroup gradients align, an aggregate update can im prove several groups; as alignment decreases, the optimizer must compromise among subgroup objectives. We call this subgroup gradient competition. Widening the MLP increases capacity but leaves every added unit active for every example, so it does not alter the underlying sharing constraint.

Ordinary scaling asks how many parameters are available; we ask whether diferent examples should use diferent parameters. If observable subgroup competition can be reduced by input-conditioned parameters, MoE becomes a way to reorganize sharing rather than merely enlarge the model. Section 3.1 formalizes this distinction.

## 1.2 Diagnosing Subgroup Gradient Competition

We diagnose this constraint on trained Dense models from Avazu. Model parameters are frozen, and semantic subgroups are constructed from site\_category, app\_category, device\_type, and hour. For each subgroup, we compute the loss gradient with respect to the shared Top-NN parameters. Gradient cosine similarity measures agreement between the corresponding update directions. Each semantic grouping is compared against random groups with the same sample size and positive-label ratio, controlling for two basic sources of gradient variation.

As Figure 1 shows, semantic partitioning consistently reduces gradient alignment relative to the matched random control: the gaps are 0.26 for DNN, 0.34 for AutoInt, 0.23 for FiBiNET, and 0.37 for DCNv2. Because each control preserves subgroup size and label prevalence, these substantial gaps cannot be explained by either factor. They instead reveal a systematic optimization efect: real semantic groups impose less compatible update directions on the shared Top-NN than composition-matched random groups. We refer to this excess loss of alignment as the subgroup competition gap. Section 4.2 formalizes the diagnostic and tests whether PRIME narrows this gap by providing input-conditioned residual directions.

## 1.3 From Diagnosis to Function-Preserving Conditional Adaptation

Mixture-of-Experts models change the granularity of parameter sharing by routing examples to local experts [9, 14, 16]. Modern routing improves expert organization and stability [34–36, 42, 43], but transferring this conditional parameterization to CTR is not straightforward.

CTR prediction difers from language modeling in several respects. Each example usually supplies only one binary label; predicted probabilities afect both ranking and calibration; and the backbone already contains an explicit, architecture-specific interaction module. Adding MoE to a CTR Top-NN must therefore address three linked questions. The method should protect a strong Dense function, separate conditional specialization from ordinary capacity growth, and avoid duplicating interaction paterns already modeled by the backbone.

These requirements determine PRIME’s design. New parameters start from the original Dense function, isolating the value of conditional adaptation from changes to the baseline mapping. Expert weighting depends explicitly on the input and delivers gains beyond parameter- and compute-matched controls. Each expert produces a compact low-rank logit correction instead of replicating the interaction module. Multi-bag aggregation combines several conditional estimates, while auxiliary-loss-free load adjustment maintains efective expert utilization without changing the CTR objective. Together, these mechanisms provide a stable operating layer around PRIME’s function-preserving conditional parameterization.

Our central claim is that the limitation of a shared Top-NN lies in the granularity of parameter sharing, not in an inherent lack of MLP expressiveness. An input-conditioned low-rank residual can reduce the optimization compromise among heterogeneous examples while retaining the common Dense mapping.

This paper asks:

How can a shared CTR Top-NN acquire input-conditioned correction directions without changing the backbone   
interaction module, disturbing the initial Dense function, or incurring uncontrolled computation?

The question leads to three primary requirements: function preservation, input-conditioned specialization, and budget-controlled attachment across backbones. Multi-bag aggregation and load adjustment are supporting implementation choices. Section 3 derives PRIME from this hierarchy after formalizing subgroup optimization under a shared Top-NN.

The resulting contributions are threefold:

(1) To the best of our knowledge, we present the first controlled empirical study of semantic subgroup competition in the shared Top-NN of single-task CTR models. The diagnostic compares real semantic partitions with random partitions matched for size and label ratio, and verifies the competition gap across CTR architectures and seeds.

(2) We propose PRIME, a function-preserving conditional residual expert model. PRIME retains the shared Dense mapping as an anchor and uses input-dependent routing to combine low-rank logit corrections, separating conditional capacity from the backbone’s feature-interaction mechanism.

(3) Across 13 CTR architectures, two datasets, and five paired seeds, matched-budget controls, ablations, and gradient di agnostics separate input-conditioned organization from ordinary capacity expansion.

## 2 Related Work

## 2.1 CTR Feature Interaction and Shared Prediction Layers

CTR modeling has long focused on interactions among sparse features. Factorization Machines represent second-order interactions with low-rank inner products [13]. Wide&Deep and DeepFM combine low-order and deep nonlinear paths [1, 7], while NFM, PNN, AFM, and FwFM refine interaction pooling, product layers, atention, and field weighting, respectively [6, 11, 12, 22]. DCN/DCNv2, xDeepFM, AutoInt, FiBiNET, and MaskNet build richer representations through explicit crosses, compressed interactions, self-atention bilinear interactions, and instance-guided masks [8, 15, 18–20, 26]. DIN activates historical interests relative to a target item [27], and FinalMLP constructs complementary MLP streams through streamspecific feature selection [28]. These methods alter the input representation or interaction path, but their final decisions are usually produced by MLP parameters shared across all examples. PRIME leaves the interaction module unchanged and instead modifies how Top-NN capacity is shared.

Viewed from the decision layer, these architectures difer in how they transform the shared embedding �, yet converge on a globally shared prediction parameterization. PRIME targets this common endpoint rather than adding another interaction operator, allowing the same conditional adapter to be evaluated across otherwise heterogeneous backbones.

## 2.2 Conditional Parameterization and Expert Models for Recommendation

Conditional parameterization changes network weights as a function of the input or task. HyperNetworks generate the parameters of another network [29]. APG dynamically produces low-rank prediction parameters for individual CTR examples [30] and is one of the closest published methods to PRIME. STAR combines shared central parameters with domain-specific parameters [31], while PEPNet modulates embeddings and DNN units using user, domain, and task priors [32]. More recent work has combined low-rank adaptation with expert selection. iLoRA uses sequence representations to gate LoRA experts for instance-wise adaptation in sequential recommendation [37]. MLoRA assigns a low-rank adapter to each CTR domain and reports both ofline and online gains [38]. HiLoMoE uses hierarchical rank-one experts to scale CTR models in depth and width [39], whereas MoE-MLoRA first trains domain experts and then learns a gate for multi-domain CTR [40].

MLoRA+ fuses domain-specific LoRA adapters with a Transformer to model cross-domain dependencies [44]. These studies support low-rank experts as a compact representation of conditional variation, but they rely on pretrained language models, explicit domain boundaries, or multi-stage optimization. PRIME targets conventional single-task CTR without domain labels or staged training.

Expert architectures in recommendation have also been developed mainly for multi-task or multi-scenario sharing. MMoE combines shared experts through task-specific gates [10], and PLE separates shared from task-specific experts to reduce negative transfer [17]. PRIME operates in one label space with one CTR objective. Latent semantic groups are used only for post-training diagnosis and never supervise the router. The problem is not expert allocation across observed tasks, but update competition among heterogeneous examples within a single task.

## 2.3 MoE Stability and Controlled Residual Adaptation

Classical MoE learns local specialization through gating [9], and sparse gating and Switch Transformers use conditional activation to scale model capacity [5, 14]. Routing instability, expert redundancy, and load imbalance have consequently become central concerns. ST-MoE studies stable training for sparse expert models [16], Expert Choice Routing assigns a fixed capacity to each expert [33], Soft MoE avoids token dropping and discrete assignment through continuous soft routing [34], DeepSeekMoE increases specialization with fine-grained routed and shared experts [35], and ReMoE replaces discontinuous Top-� selection with ReLU routing [36]. More recent work addresses whether routes reflect expert capabilities: ERC loss explicitly couples routers and experts [42], while RoMA aligns routing-weight and task-semantic manifolds to improve generalization [43]. Auxiliary-loss-free balancing instead adjusts dynamic expert biases without adding balancing gradients to the task objective [21]. These methods primarily address sparse computation at language-model scale. PRIME uses a small low-rank softrouted module and concentrates on function preservation, controlled conditional estimation, and adaptation across CTR architectures.

PRIME also draws on low-rank adaptation, residual learning, and ensemble estimation. LoRA confines parameter updates to a low-dimensional subspace [2], and iLoRA, MLoRA, and HiLoMoE extend low-rank parameterization to instance-, domain-, and hierarchy conditioned recommendation [37–39]. Residual learning preserves a baseline mapping [3], while bagging reduces variance by averaging diverse predictors [24]. PRIME combines these ideas in a single end-to-end optimization: low rank restricts expert freedom, zeroresidual initialization preserves the initial Dense function, and averaging smooths multiple conditional estimates. Unlike a fixed global low-rank update, a staged expert system, or independently trained full predictors, PRIME routes residual directions by input while sharing the same CTR backbone.

Existing work studies task-level negative transfer and domainconditioned parameters. PRIME instead identifies semantic subgroup competition within the shared Top-NN of a single CTR task. Controls matched for group size and click rate separate this gap from composition efects, motivating a conditional residual design that preserves the Dense function.

![](images/825d9930ac83904b6cd9d863ed25500ea35423227a67e7c499f0f7b00ee627b5.jpg)

Figure 2: PRIME architecture. The original model produces �<sub>�</sub>; � = 4 bags route � to rank-� logit-residual experts and average their predictions into $\mathit { p _ { s } }$ . The final prediction mixes $\hbar d$ and $\mathit { p _ { s } }$ with $\alpha = 0 . 5$ . EMA load biases adjust utilization without auxiliary gradients, and zero-residual initialization gives $ { \boldsymbol { p } } =  { \boldsymbol { p } } _ { d }$ at the start of training.

## 3 PRIME

Figure 2 summarizes PRIME’s two-path architecture. The original CTR backbone produces the anchored probability $\mathit { p } _ { d } ,$ , while � routed low-rank expert bags estimate the conditional probability ${ \mathit { p } } _ { s } ;$ their convex combination yields the final prediction $\mathcal { P } \cdot$

## 3.1 Problem Formulation and Shared Optimization

Let � denote an input example. The embedding layer $E _ { \omega }$ produces the flatened representation �, and the complete original CTR back bone $B _ { \phi , \theta }$ produces the baseline logit $\ell _ { d }$ and probability $\mathit { p } _ { d } \mathbf { : }$

$$
z = \mathrm { v e c } ( E _ { \omega } ( x ) ) , \qquad \ell _ { d } = B _ { \phi , \theta } ( z ) , \qquad \hbar { } d = \sigma ( \ell _ { d } ) .\tag{1}
$$

Here $\phi$ denotes the interaction parameters and � the shared Top-NN parameters. When a backbone contains parallel shallow or interaction logits, $B _ { \phi , \theta }$ includes those paths as well, so $\hbar d$ is always the complete original prediction. PRIME leaves every component of $B _ { \phi , \theta }$ unchanged and ataches its residual branch to � and ${ \mathit { p } } _ { d } .$

Suppose the training distribution contains � latent subgroups. Subgroup � has probability mass $\pi _ { k }$ and loss ${ \mathcal { L } } _ { k } ( \theta )$ . The objective and gradient of the shared Top-NN are

$$
\mathcal { L } ( \boldsymbol { \theta } ) = \sum _ { k = 1 } ^ { K } \pi _ { k } \mathcal { L } _ { k } ( \boldsymbol { \theta } ) , \qquad \boldsymbol { g } = \nabla _ { \boldsymbol { \theta } } \mathcal { L } ( \boldsymbol { \theta } ) = \sum _ { k = 1 } ^ { K } \pi _ { k } \boldsymbol { g } _ { k } ,\tag{2}
$$

where $g _ { k } = \nabla _ { \theta } \mathcal { L } _ { k } ( \theta )$ . For an update $\theta ^ { \prime } = \theta - \eta g$ , the first-order change in subgroup �’s loss is

$$
\begin{array} { l } { \mathcal { L } _ { k } ( { \boldsymbol { \theta } } ^ { \prime } ) - \mathcal { L } _ { k } ( { \boldsymbol { \theta } } ) \approx - \eta g _ { k } ^ { \top } g } \\ { \qquad = - \eta \left[ \pi _ { k } \| g _ { k } \| _ { 2 } ^ { 2 } + \displaystyle \sum _ { j \neq k } \pi _ { j } \| g _ { k } \| _ { 2 } \| g _ { j } \| _ { 2 } \cos ( g _ { k } , g _ { j } ) \right] . } \end{array}\tag{3}
$$

Equation 3 shows that the efective descent a subgroup receives from a shared update depends on its gradient alignment with the other subgroups. Lower cosine similarity weakens the aggregate update for that subgroup. If the similarity becomes negative, an update that benefits one subgroup can increase another subgroup’s loss. A fully shared Top-NN must therefore compromise among subgroup objectives.

Let $\theta _ { k } ^ { \star }$ be the optimum obtained by training only on subgroup $k ,$ and let $\theta ^ { \star }$ be the optimum under fully shared parameters. The compromise due to sharing can be writen as

$$
\mathcal G _ { \mathrm { s h a r e } } = \sum _ { k = 1 } ^ { K } \pi _ { k } \left[ \mathcal L _ { k } ( \theta ^ { \star } ) - \mathcal L _ { k } ( \theta _ { k } ^ { \star } ) \right] \ge 0 .\tag{4}
$$

Equation 4 motivates a compact intervention: input-dependent low-dimensional corrections reduce the sharing compromise while retaining the statistical eficiency of a common baseline. PRIME therefore preserves the complete baseline function and introduces a conditional adapter $A _ { \psi } \mathrm { : }$

$$
\begin{array} { r } { p = A _ { \psi } ( z , p _ { d } ) . } \end{array}\tag{5}
$$

The adapter must satisfy three primary conditions.

Function preservation. At initialization $\psi _ { 0 }$

$$
A _ { \psi _ { 0 } } ( z , p _ { d } ) = p _ { d } , \qquad \forall z .\tag{6}
$$

Input-conditioned parameter organization. For two inputs $x _ { i }$ and $x _ { j } ,$ the routing weights may satisfy $r ( x _ { i } ) \neq r ( x _ { j } )$ , preventing the added capacity from collapsing into a fixed ensemble. Conditional organization should retain a measurable advantage when parameter count or computation is matched.

Budget control and backbone decoupling. The added parameters and operations should be governed by a low rank $q \ll d .$ Experts output generic logit residuals rather than replicating a complete Top-NN or the backbone’s interaction module.

PRIME adds two supporting implementation choices: multiple expert bags aggregate alternative conditional estimates, and a gradientfree load bias limits extreme routing imbalance. Both are trained within the same end-to-end binary cross-entropy objective. Ablations distinguish these refinements from the three conditions that define PRIME’s core parameterization.

## 3.2 Dense-Anchored Conditional Residual Parameterization

For latent subgroup �, write its ideal logit as a shared baseline plus a subgroup-specific correction:

$$
\ell _ { k } ^ { \star } ( x ) = \ell _ { d } ( x ) + \Delta _ { k } ( x ) .\tag{7}
$$

This decomposition assigns common predictive structure to $\ell _ { d }$ and focuses each expert on a residual $\Delta _ { k } .$ . Following the baselinepreserving principle ofresidual learning [3], the new branch learns conditional deviations around an established prediction function. Since subgroup labels are unavailable at inference time, PRIME uses input-dependent routing to weight candidate residuals:

$$
\Delta ( x ) \approx \sum _ { e } r _ { e } ( x ) \delta _ { e } ( x ) .\tag{8}
$$

The Dense path is a functional anchor shared by all experts, while the routed branch learns conditional corrections around the

original model function. A fixed mixture weight bounds the expert contribution and guarantees a persistent Dense component throughout training.

## 3.3 Low-Rank Residual Experts

PRIME obtains a flatened embedding representation $z \in \mathbb { R } ^ { d }$ from the original model and applies Layer Normalization without afine parameters:

$$
\tilde { z } = \mathrm { L N } ( z ) .\tag{9}
$$

Expert � in bag � is defined as

$$
\delta _ { g , e } ( x ) = u _ { g , e } ^ { \top } \mathrm { S i L U } \big ( V _ { g , e } \tilde { z } \big ) + b _ { g , e } ,\tag{10}
$$

where $V _ { g , e } \in \mathbb { R } ^ { q \times d } , u _ { g , e } \in \mathbb { R } ^ { q }$ , and $q \ll d .$ Matrix $V _ { g , e }$ projects the input into a �-dimensional subspace, after which $u _ { \mathrm { g } , e }$ maps the nonlinear response to a scalar logit correction. As in low-rank adaptation [2], this restriction limits each expert to $O ( q d )$ parameters, far fewer than a replicated Top-NN. Distinct matrices $V _ { g , e }$ represent diferent low-dimensional correction directions, which the router combines according to the input.

To satisfy the function-preservation condition in Equation $^ { 6 , }$ PRIME initializes $u _ { g , e }$ and $b _ { g , e }$ to zero:

$$
\delta _ { g , e } ( x ) = 0 , \qquad \forall g , e , x .\tag{11}
$$

The projection matrices and routers may start from diferent directions without changing the initial prediction. Optimization first develops the expert output projections and only then establishes router-expert specialization, preventing untrained experts from dominating early predictions.

## 3.4 Input-Conditioned Routing, Bag Aggregation, and Functional Equivalence

Each expert bag has an independent router. The routing probability within bag � is

$$
r _ { g , e } ( x ) = \frac { \exp \left( ( a _ { g , e } ^ { \top } \tilde { z } + c _ { g , e } ) / T \right) } { \sum _ { j = 1 } ^ { M } \exp \left( ( a _ { g , j } ^ { \top } \tilde { z } + c _ { g , j } ) / T \right) } ,\tag{12}
$$

where � is the number of experts per bag, � is the routing temperature, and $c _ { \mathrm { g } , e }$ is a load-correction bias. The main experiments use $T = 1$ . Each expert corrects the baseline logit:

$$
\begin{array} { r } { \dot { p } _ { \mathrm { g } , e } ( \boldsymbol { x } ) = \sigma \big ( \mathrm { l o g i t } ( p _ { d } ( \boldsymbol { x } ) ) + \delta _ { \mathrm { g } , e } ( \boldsymbol { x } ) \big ) . } \end{array}\tag{13}
$$

The bag prediction is the conditional expectation of the expert probabilities:

$$
p _ { g } ( x ) = \sum _ { e = 1 } ^ { M } r _ { g , e } ( x ) p _ { g , e } ( x ) .\tag{14}
$$

Equations 12 and 14 make the expert mixture input-dependent. Replacing $r _ { g , e } ( x )$ with a constant turns the branch into an unconditional ensemble. Permuting $r ( x )$ across examples preserves aggregate load while breaking the association between an input and its expert weights. Section 4.3 uses both interventions to test the contribution of conditional routing.

Bag aggregation. PRIME partitions $E = G M$ experts into � bags to obtain multiple conditional estimates under a fixed total expert count. Each bag independently evaluates Equations 12–14, and the bag predictions are averaged:

$$
\begin{array} { r } { p _ { s } ( x ) = \displaystyle \frac { 1 } { G } \sum _ { g = 1 } ^ { G } p _ { g } ( x ) . } \end{array}\tag{15}
$$

If bag predictions have variance $\sigma _ { p } ^ { 2 }$ and an approximate pairwise correlation $\rho _ { p } ,$ the variance of their mean is

$$
\operatorname { V a r } [ { p _ { s } } ] \approx \frac { \sigma _ { \mathscr { P } } ^ { 2 } } { G } \left[ 1 + ( G - 1 ) \rho _ { \mathscr { P } } \right] .\tag{16}
$$

Equation 16 motivates prediction averaging as an output-smoothing mechanism under correlated conditional estimates. The bags share a backbone and are optimized jointly, so the expression motivates the design without assuming independent bagging estimators. The single-bag comparison in Section 4.3 shows that multi-bag aggregation contributes complementary output smoothing, with gains concentrated on backbones that benefit most from routing diversity.

Dense probability anchor. The final prediction is a convex combination of the Dense and expert probabilities:

$$
p ( x ) = \alpha p _ { d } ( x ) + ( 1 - \alpha ) p _ { s } ( x ) , \qquad 0 \leq \alpha \leq 1 .\tag{17}
$$

We use $\alpha = 0 . 5$ . Equation 17 bounds expert corrections in probability space and retains an explicit Dense contribution throughout training. The original path remains in the computation graph even if routing becomes highly concentrated.

From Equation 11,

$$
\begin{array} { r } { \boldsymbol { p } _ { g , e } ( \boldsymbol { x } ) = \sigma ( \mathrm { l o g i t } ( \boldsymbol { p } _ { d } ( \boldsymbol { x } ) ) ) = \boldsymbol { p } _ { d } ( \boldsymbol { x } ) . } \end{array}
$$

Substitution into Equations 14–17 gives

$$
\begin{array} { r } { p _ { g } ( x ) = p _ { s } ( x ) = p ( x ) = p _ { d } ( x ) . } \end{array}\tag{18}
$$

PRIME is therefore exactly function-equivalent to the Dense model at initialization. This property follows directly from zeroresidual initialization and convex mixing, without relying on hyperparameter tuning.

## 3.5 Auxiliary-Loss-Free Load Adjustment

Sparse MoE models often add a load-balancing term to the task loss to prevent a small number of experts from absorbing most inputs [14]:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { C T R } } + \lambda _ { \mathrm { b a l } } \mathcal { L } _ { \mathrm { b a l } } .\tag{19}
$$

The coeficient $\lambda _ { \mathrm { b a l } }$ creates an additional trade-of between expert utilization and the CTR objective. Following auxiliary-lossfree load balancing [21], PRIME does not backpropagate ${ \mathcal { L } } _ { \mathrm { b a l } }$ . It instead tracks average routing load within each bag. For batch �,

$$
\hat { q } _ { g , e } ^ { ( t ) } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } r _ { g , e } ( x _ { i } ) ,\tag{20}
$$

and the exponential moving average is updated as

$$
\begin{array} { r } { \bar { q } _ { g , e } ^ { ( t ) } = \mu \bar { q } _ { g , e } ^ { ( t - 1 ) } + ( 1 - \mu ) \hat { q } _ { g , e } ^ { ( t ) } . } \end{array}\tag{21}
$$

The load bias is corrected without gradients toward the target load $1 / M \mathrm { : }$

$$
\begin{array} { l } { \displaystyle \tilde { c } _ { g , e } ^ { ( t + 1 ) } = c _ { g , e } ^ { ( t ) } + \eta _ { b } \left( \frac { 1 } { M } - \bar { q } _ { g , e } ^ { ( t ) } \right) , } \\ { \displaystyle c _ { g , e } ^ { ( t + 1 ) } = \mathrm { c l i p } \left( \tilde { c } _ { g , e } ^ { ( t + 1 ) } - \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \tilde { c } _ { g , j } ^ { ( t + 1 ) } , - c _ { \operatorname* { m a x } } , c _ { \operatorname* { m a x } } \right) . } \end{array}\tag{22}
$$

Equation 22 updates each bias from its load error and removes the shared ofset by centering within a bag. We set $\mu ~ = ~ 0 . 9 9 _ { ; }$ $\eta _ { b } = 1 0 ^ { - 3 }$ , and $c _ { \operatorname* { m a x } } = 2$ . The update changes routing biases for subsequent batches but sends no additional gradient to model parameters. PRIME is consequently trained end to end with a single CTR objective:

$$
\mathcal { L } _ { \mathrm { P R I M E } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ y _ { i } \log p _ { i } + \left( 1 - y _ { i } \right) \log ( 1 - p _ { i } ) \right] + \lambda _ { 2 } \| \Theta \| _ { 2 } ^ { 2 } .\tag{23}
$$

## 3.6 Parameter Count, Computation, and Plug-in Implementation

Let � be the input dimension, � the total number of experts, and � the expert rank. Ignoring scalar activation costs, the routers require �� multiply-accumulate operations (MACs), and the low-rank experts require $E q ( d + 1 )$ . The additional computation is

$$
C _ { \mathrm { { e x t r a } } } = E d + E q ( d + 1 ) .\tag{24}
$$

The number of added trainable parameters is

$$
P _ { \mathrm { { e x t r a } } } = E d + E q d + E q + E = E ( q + 1 ) ( d + 1 ) .\tag{25}
$$

Both quantities are controlled by $q ,$ avoiding replication of � complete Top-NNs.

At runtime, PRIME executes the original model to obtain �<sub>�</sub> and captures the flatened embedding representation � through a hook, as shown in Figure 2(a). It then evaluates Equations 9–17. Feature processing, the interaction module, the Dense Top-NN, the training entry point, and the record format remain unchanged. Enabling the plug-in reconstructs the optimizer so that the new parameters participate in the same end-to-end training process.

These design choices yield three primary predictions: matched non-conditional capacity should not reproduce PRIME’s full gain, breaking input-route correspondence should reduce performance, and preserving the Dense anchor should improve robustness across backbones. The experiments test these predictions before the configuration is frozen for cross-architecture evaluation.

## 4 Experiments

## 4.1 Research Questions and Experimental Protocol

The experiments address five research questions in the order of the evidence presented.

• RQ1: Problem diagnosis (Section 4.2). Do semantic subgroups exhibit lower gradient alignment in a shared Top-NN than matched random groups?

• RQ2: Mechanism association (Section 4.2). Does the semantic subgroup competition gap decrease after training with PRIME?

• RQ3: Capacity and conditioning (Section 4.3). Can nonconditional Dense parameters or matched computation explain PRIME’s gains, and is input-dependent routing necessary?

• RQ4: Design hierarchy and eficiency (Sections 4.3– 4.4). Which PRIME components are necessary, and how does the resulting model compare with related conditional parameterization in accuracy and end-to-end cost?

• RQ5: Cross-architecture and cross-dataset efectiveness (Section 4.5). After freezing the configuration, how broadly does PRIME improve held-out performance across CTR interaction architectures on Avazu and Criteo?

We use the public FuxiCTR/BARS-CTR processing pipeline [4, 23]. Avazu contains 22 fields and 28.30M, 4.04M, and 8.09M examples in its training, validation, and test splits, respectively. Criteo contains 39 fields, with corresponding split sizes of 36.67M, 4.58M, and 4.58M. The evaluation covers 13 CTR architectures: AFM [22], AutoInt [15], DCN [18], DNN, DeepFM [7], FiBiNET [8], FwFM [12], GDCN [25], MaskNet [20], NFM [6], PNN [11], Wide&Deep [1], and DCNv2 [19]. For backbones without a standalone MLP head, Dense denotes the complete original prediction path; PRIME still anchors its unmodified probability ${ \mathit { p } } _ { d } .$ . Every Dense/PRIME pair shares its split, preprocessing, backbone configuration, and random seed. Model selection uses validation AUC, after which the untouched test set is evaluated over five paired seeds.<sup>2</sup>

All main experiments use Adam with learning rate $1 0 ^ { - 3 }$ , batch size 4,096, and the public backbone hyperparameters. Training runs for at most 100 epochs, stops after two validation checks without improvement, and restores the best validation checkpoint. The em bedding dimension is 10. PRIME is frozen at � = 4 bags, $M = 8$ experts per bag, rank $q = 1 6 ,$ , Dense weight $\alpha = 0 . 5$ , and routing temperature $T = 1$ . Completed runs are retained; a job is restarted only if it exits abnormally or produces no valid record. OpenAI Codex was used to assist with experimental script development and code debugging; all generated code changes were reviewed and validated by the authors.

We define the paired metrics as

$$
\Delta \mathrm { A U C } = \mathrm { A U C } _ { \mathrm { P R I M E } } - \mathrm { A U C } _ { \mathrm { D e n s e } } ,\tag{26}
$$

$$
\Delta \mathrm { L o g L o s s } = \mathrm { L o g L o s s } _ { \mathrm { D e n s e } } - \mathrm { L o g L o s s } _ { \mathrm { p R I M E } } .\tag{27}
$$

Positive values indicate an improvement by PRIME for both definitions. We report the original metrics and five-seed means at the architecture level.

![](images/3794d39528e5d482d344c884faa3d8ae6807ad4e5f121ab0a625f81f7829c47b.jpg)  
Figure 3: PRIME reduces the competition gap between semantic and matched random groups. Bars report architecture-level means over three paired checkpoints and four semantic fields. Percentages denote the relative reduction from Dense to PRIME; lower values indicate more consistent subgroup updates.

## 4.2 Diagnosing Shared Top-NN Competition and Its Change under PRIME

To answer RQ1, we evaluate DNN, AutoInt, FiBiNET, and DCNv2 with three trained checkpoints per architecture and four semantic fields. This gives 48 comparisons across architectures, checkpoints, and fields. Semantic subgroups are defined by site\_category, app\_ category, device\_type, and hour. Each group is sampled into eight disjoint blocks containing 256 positive and 256 negative examples per block, for 4,096 examples in total. Matched random groups use exactly the same sample size and label ratio, with 2,000 random permutations. Models are placed in evaluation mode, and unregularized binary cross-entropy gradients are computed on the validation set. The gradient parameter set is restricted to all trainable weights in the shared Top-NN. Gradients are flatened before pairwise cosine similarities are calculated. Every architecture in Figure 1 satisfies $A _ { f } ^ { \mathrm { s e m } } < A _ { f } ^ { \mathrm { r a n d } }$ , establishing semantic grouping as a consistent source of lower agreement among shared updates after controlling for group size and label ratio.

For semantic field $f ,$ let $A _ { f } ^ { \mathrm { s e m } }$ be the mean pairwise gradient similarity among semantic subgroups and $A _ { f } ^ { \mathrm { r a n d } }$ that of matched random groups. Define the competition gap as

$$
\Gamma _ { f } = A _ { f } ^ { \mathrm { r a n d } } - A _ { f } ^ { \mathrm { s e m } } .\tag{28}
$$

A larger $\Gamma _ { f }$ indicates greater gradient heterogeneity associated with semantic grouping. To address RQ2, we compare trained Dense and PRIME models using the same architectures, seeds, fields, and examples.

As shown in Figure 3, the mean competition gap across the four architectures decreases from $\bar { \Gamma } _ { \mathrm { D e n s e } } = 0 . 3 0 1 6$ to $\bar { \Gamma } _ { \mathrm { { P R I M E } } } = 0 . 1 9 8 1$ corresponding to a relative reduction of $R _ { \Gamma } = 1 - 0 . 1 9 8 1 / 0 . 3 0 1 6 =$ 34.3%.

Gradient alignment increases more for semantic subgroups than for matched random groups. The reduction in Γ therefore captures selective convergence of the update directions that exhibit the strongest competition, matching the optimization behavior in Equation 3.

The matched non-conditional control next separates input conditioning from residual capacity.

We next construct a parameter-matched, non-conditional Dense residual to test whether any residual expansion or altered training trajectory would reduce Γ. The control uses the same Dense probability anchor and zero-residual initialization as PRIME, but all examples share the same added parameters and there is no inputdependent routing. Averaged over four architectures, three training seeds, and four semantic fields, the mean competition gaps are $\bar { \Gamma } _ { \mathrm { D e n s e } } = 0 . 3 0 1 6 , \bar { \Gamma } _ { \mathrm { N o n C o n d } } = 0 . 2 2 4 6 ,$ and $\bar { \Gamma } _ { \mathrm { { P R I M E } } } = 0 . 1 9 8 1$

The non-conditional residual reduces the gap by 0.0770 relative to Dense, showing that function-preserving residual capacity alleviates part of the shared compromise. PRIME provides a further reduction of0.0265. Under the same budget and initialization, this additional reduction establishes a contribution from input-conditioned parameter organization beyond residual capacity and links PRIME’s predictive gains to stronger optimization agreement.

## 4.3 Capacity-Matched Controls and the Contribution of Input Conditioning

This section answers RQ3 and evaluates the component-level design hierarchy in RQ4. We separate conditional organization from added capacity using two non-conditional Dense residual adapters. One approximately matches PRIME’s added parameter count; the other approximately matches its added activation MACs. Both controls use the same zero-residual initialization and probability anchor as PRIME but contain no expert router. For budget-matched control $b ,$ define

$$
\mathcal { A } _ { b } = \mathrm { A U C } _ { \mathrm { P R I M E } } - \mathrm { A U C } _ { b } .\tag{29}
$$

The experiment uses the held-out Avazu test set, DNN, AutoInt, FiBiNET, DCNv2, and five paired seeds, yielding 20 paired runs for each comparison. These comparisons serve as diagnostic controls that isolate the efects of capacity, routing, anchoring, and aggregation. All variant configurations were fixed using validation data before held-out test evaluation, and no test result was used for subsequent tuning. Relative to the original Dense models, the parameter-matched and MAC-matched residuals improve mean AUC by 0.0019 and 0.0020, respectively. Non-conditional resid ual capacity is therefore useful. Both controls nevertheless remain below PRIME by 0.0012 and 0.0011 AUC. The resulting $\mathcal { A } _ { b }$ measures the value of input-dependent expert organization after parameter or activation cost is matched.

Figure 4 reveals a clear design hierarchy. Uniform routing retains all expert parameters but removes input-dependent allocation. Permuted routing preserves aggregate load but breaks the input-expert correspondence. These interventions reduce mean AUC by 0.0018 and 0.0017, respectively, providing direct evidence for conditional specialization. Retaining the Dense anchor contributes 0.0005 mean AUC, confirming the function-preservation role of Equations 6 and 17. Together with the parameter- and MAC-matched controls, these results establish function-preserving, input-conditioned parameter organization as the source of PRIME’s advantage over ordinary capacity growth and unconditional ensembling. Multibag aggregation contributes a further 0.0005 mean AUC through multiple conditional estimates. Auxiliary-loss-free balancing matches the accuracy of loss-based balancing while preserving the singletask objective in Equation 23 and eliminating an additional loss coeficient.

![](images/0d9aa7604e5890a2a20e69e5f4543114037955402e461737d28ca6679f29122d.jpg)  
Figure 4: AUC diferences between PRIME and capacity or mechanism controls. Bars report means over four backbones and five paired seeds. Every horizontal bar is oriented as PRIME, or the retained mechanism, minus the corresponding control or ablation. Positive values favor the PRIME design.

## 4.4 Comparison with Conditional Parameterization and End-to-End Cost

This section completes RQ4 by comparing PRIME with the oficial FuxiCTR implementation of APG [30], a related input-conditioned parameterization method, and by measuring end-to-end cost. The comparison covers FiBiNET and DCNv2. For FiBiNET, APG is attached before bilinear interaction and retains the original Dense Top-NN. The methods use identical data splits, embedding dimensions, backbone hidden layers, optimizers, batch sizes, and earlystopping protocols. APG-specific hyperparameters are selected on the validation set without access to test data.

All methods are profiled on the same accelerator with batches of 4,096 examples, after 20 warm-up steps, using 100 timed inference steps and three independent repetitions.

The APG configurations train reliably on Avazu. Table 1 reports held-out test results over five seeds for the shared backbones. PRIME exceeds APG by 0.0030 AUC on FiBiNET and 0.0032 AUC on DCNv2, and all ten seed-level comparisons favor PRIME. On FiBiNET, APG improves over Dense, while PRIME provides a further gain with fewer parameters and lower latency; on DCNv2, PRIME exceeds both Dense and APG. Across both shared backbones, Dense-anchored conditional residuals are more consistent than direct generation of dynamic prediction parameters.

Table 1 shows that conditional capacity need not require a large deployment footprint. On FiBiNET, PRIME adds 0.6% parameters and 8.0% latency over Dense, while using 0.7% fewer parameters and 1.8% lower latency than APG. On DCNv2, the corresponding overheads over Dense are 0.8% and 29.0%, whereas the reductions relative to APG are 5.2% and 15.9%. PRIME therefore retains near-Dense parameter scale while improving both accuracy and eficiency over APG.

Table 1: Test accuracy and inference cost on Avazu. AUC and LogLoss are five-seed means; latency is the median per 4,096-example batch over three profiling repetitions. Bold accuracy values are best per backbone, while bold cost values mark the more eficient input-conditioned method.
<table><tr><td>Backbone</td><td>Method</td><td>Test AUC</td><td>Test LogLoss</td><td>Params (M)</td><td>Latency (ms)</td></tr><tr><td rowspan="3">FiBiNET</td><td>Dense</td><td>0.7600</td><td>0.3695</td><td>19.1582</td><td>2.1993</td></tr><tr><td>APG</td><td>0.7620</td><td>0.3684</td><td>19.4240</td><td>2.4191</td></tr><tr><td>PRIME</td><td>0.7650</td><td>0.3663</td><td>19.2784</td><td>2.3752</td></tr><tr><td rowspan="3">DCNv2</td><td>Dense</td><td>0.7626</td><td>0.3680</td><td>16.0000</td><td>1.1260</td></tr><tr><td>APG</td><td>0.7610</td><td>0.3689</td><td>17.0100</td><td>1.7270</td></tr><tr><td>PRIME</td><td>0.7642</td><td>0.3671</td><td>16.1200</td><td>1.4521</td></tr></table>

## 4.5 Final Held-Out Results on Avazu and Criteo

To answer RQ5, both the PRIME configuration and all backbone setings are frozen after model selection on the validation set. Table 2 reports the held-out test performance over five seeds for each architecture. Dense denotes the complete original backbone, and PRIME denotes the same backbone trained end to end with the plugin enabled.

On Avazu, PRIME raises macro-average AUC from 0.7588 to 0.7611 and lowers LogLoss from 0.3701 to 0.3689, with a median paired ΔAUC of 0.0022. When the five-seed standard deviation is averaged across architectures, PRIME reduces AUC dispersion from 0.0018 to 0.0011 and LogLoss dispersion from 0.0010 to 0.0007. These simultaneous improvements in predictive quality and average cross-seed dispersion hold across a diverse suite of plain DNNs, explicit crosses, atention, bilinear interactions, factorization-based models, and mask-based networks.

On Criteo, every Dense/PRIME pair uses identical configurations for the shared backbone and training protocol; enabling PRIME is the only structural change. Under this frozen common protocol, Dense AutoInt and DeepFM repeatedly sufer severe optimization degradation, whereas their PRIME counterparts remain stable. We retain these runs and include them in the macro-average to preserve the uniform evaluation standard. Consequently, the aggregate gains of +0.0252 AUC and +0.4479 LogLoss summarize the complete protocol but combine conventional predictive improvements with recovery from Dense optimization failure. Crossarchitecture efectiveness is established separately by the positive mean ΔAUC on 11 of13 architectures and the +0.0066 median over all paired runs, both of which are substantially less sensitive to the two degraded Dense baselines.

Table 2: Architecture-level test results on Avazu and Criteo. Dense and PRIME entries are mean ± sample standard deviation over five paired seeds. Δ is PRIME−Dense for AUC and Dense−PRIME for LogLoss; positive values favor PRIME and are bold. The Mean row averages architecture-level means and within-architecture standard deviations over all 13 architectures; the paired-median row reports the median Δ over all 65 architecture–seed pairs.
<table><tr><td rowspan="3">Backbone</td><td colspan="6">Avazu</td><td colspan="6">Criteo</td></tr><tr><td colspan="3">AUC</td><td colspan="3">LogLoss</td><td colspan="3">AUC</td><td colspan="3">LogLoss</td></tr><tr><td>Dense</td><td>PRIME</td><td>Δ</td><td>Dense</td><td>PRIME</td><td>Δ</td><td>Dense</td><td>PRIME</td><td>Δ</td><td>Dense</td><td>PRIME</td><td>Δ</td></tr><tr><td>AFM</td><td>0.7402 ± 0.0010</td><td>0.7455 ± 0.0004</td><td>+0.0053</td><td>0.3812 ± 0.0013</td><td>0.3785 ± 0.0008</td><td>+0.0027</td><td>0.7711 ± 0.0135</td><td>0.7859 ± 0.0025</td><td>+0.0148</td><td>0.7623 ± 0.5594</td><td>0.4688 ± 0.0045</td><td>+0.2935</td></tr><tr><td>AutoInt</td><td>0.7599 ± 0.0048</td><td>0.7646 ± 0.0008</td><td>+0.0047</td><td>0.3695 ± 0.0023</td><td>0.3668 ± 0.0004</td><td>+0.0027</td><td>0.6740 ± 0.0306</td><td>0.7944 ± 0.0018</td><td>+0.1204</td><td>2.2000 ± 1.4766</td><td>0.4557 ± 0.0015</td><td>+1.7442</td></tr><tr><td>DCN</td><td>0.7641 ± 0.0003</td><td>0.7645 ± 0.0007</td><td>+0.0004</td><td>0.3671 ± 0.0003</td><td>0.3670 ± 0.0003</td><td>+0.0001</td><td>0.7799 ± 0.0091</td><td>0.8033 ± 0.0010</td><td>+0.0234</td><td>0.4682 ± 0.0072</td><td>0.4478 ± 0.0009</td><td>+0.0203</td></tr><tr><td>DNN</td><td>0.7630 ± 0.0001</td><td>0.7641 ± 0.0013</td><td>+0.0011</td><td>0.3675 ± 0.0001</td><td>0.3668 ± 0.0007</td><td>+0.0007</td><td>0.7985 ± 0.0042</td><td>0.8019 ± 0.0018</td><td>+0.0034</td><td>0.4579 ± 0.0054</td><td>0.4504 ± 0.0016</td><td>+0.0075</td></tr><tr><td>DeepFM</td><td>0.7636 ± 0.0015</td><td>0.7630 ± 0.0022</td><td>-0.0006</td><td>0.3672 ± 0.0008</td><td>0.3678 ± 0.0011</td><td>-0.0005</td><td>0.7048 ± 0.0411</td><td>0.7927 ± 0.0015</td><td>+0.0879</td><td>4.0806 ± 4.2571</td><td>0.4582 ± 0.0017</td><td>+3.6224</td></tr><tr><td>FiBiNET</td><td>0.7600 ± 0.0037</td><td>0.7650 ± 0.0007</td><td>+0.0050</td><td>0.3695 ± 0.0017</td><td>0.3663 ± 0.0004</td><td>+0.0031</td><td>0.7969 ± 0.0010</td><td>0.8038 ± 0.0007</td><td>+0.0069</td><td>0.4535 ± 0.0009</td><td>0.4473 ± 0.0006</td><td>+0.0062</td></tr><tr><td>FwFM</td><td>0.7488 ± 0.0012</td><td>0.7459 ± 0.0015</td><td>-0.0030</td><td>0.3772 ± 0.0009</td><td>0.3782 ± 0.0008</td><td>-0.0010</td><td>0.7657 ± 0.0027</td><td>0.7820 ± 0.0033</td><td>+0.0163</td><td>0.5288 ± 0.0204</td><td>0.4707 ± 0.0035</td><td>+0.0580</td></tr><tr><td>GDCN</td><td>0.7620 ± 0.0014</td><td>0.7629 ± 0.0014</td><td>+0.0009</td><td>0.3685 ± 0.0007</td><td>0.3679 ± 0.0008</td><td>+0.0006</td><td>0.8066 ± 0.0020</td><td>0.8041 ± 0.0061</td><td>-0.0025</td><td>0.4447 ± 0.0020</td><td>0.4469 ± 0.0054</td><td>-0.0022</td></tr><tr><td>MaskNet</td><td>0.7601 ± 0.0002</td><td>0.7619 ± 0.0022</td><td>+0.0018</td><td>0.3693 ± 0.0004</td><td>0.3685 ± 0.0010</td><td>+0.0008</td><td>0.8100 ± 0.0004</td><td>0.8099 ± 0.0013</td><td>-0.0001</td><td>0.4416 ± 0.0004</td><td>0.4417 ± 0.0012</td><td>-0.0001</td></tr><tr><td>NFM</td><td>0.7568 ± 0.0009</td><td>0.7644 ± 0.0004</td><td>+0.0076</td><td>0.3702 ± 0.0004</td><td>0.3669 ± 0.0004</td><td>+0.0033</td><td>0.7768 ± 0.0373</td><td>0.7943 ± 0.0020</td><td>+0.0175</td><td>0.4993 ± 0.0948</td><td>0.4605 ± 0.0070</td><td>+0.0389</td></tr><tr><td>PNN</td><td>0.7600 ± 0.0042</td><td>0.7634 ± 0.0013</td><td>+0.0034</td><td>0.3692 ± 0.0022</td><td>0.3673 ± 0.0007</td><td>+0.0018</td><td>0.8037 ± 0.0011</td><td>0.8057 ± 0.0003</td><td>+0.0020</td><td>0.4473 ± 0.0010</td><td>0.4454 ± 0.0003</td><td>+0.0019</td></tr><tr><td>Wide&amp;Deep</td><td>0.7632 ± 0.0024</td><td>0.7643 ± 0.0002</td><td>+0.0011 +0.0016</td><td>0.3675 ± 0.0009</td><td>0.3670 ± 0.0001</td><td>+0.0005</td><td>0.7636 ± 0.0266</td><td>0.7960 ± 0.0004</td><td>+0.0324</td><td>0.4813 ± 0.0206</td><td>0.4545 ± 0.0004</td><td>+0.0268</td></tr><tr><td>DCNv2</td><td>0.7626 ± 0.0015</td><td>0.7642 ± 0.0013</td><td></td><td>0.3680 ± 0.0010</td><td>0.3671 ± 0.0008</td><td>+0.0009</td><td>0.7977 ± 0.0055</td><td>0.8031 ± 0.0027</td><td>+0.0054</td><td>0.4530 ± 0.0050</td><td>0.4482 ± 0.0026</td><td>+0.0048</td></tr><tr><td>Mean</td><td>0.7588 ± 0.0018</td><td>0.7611 ± 0.0011</td><td>+0.0023</td><td>0.3701 ± 0.0010</td><td>0.3689 ± 0.0007</td><td>+0.0012</td><td>0.7730 ± 0.0135</td><td>0.7982 ± 0.0020</td><td>+0.0252</td><td>0.9014 ± 0.4962</td><td>0.4535 ± 0.0024</td><td>+0.4479</td></tr><tr><td colspan="2">Paired median ∆</td><td></td><td>+0.0022</td><td></td><td></td><td>+0.0011</td><td></td><td></td><td>+0.0066</td><td></td><td></td><td>+0.0081</td></tr></table>

## 5 Discussion

## 5.1 How Conditional Residual Experts Reduce Shared Top-NN Competition

PRIME changes where heterogeneous examples are allowed to disagree. In a Dense baseline, every example updates the same latestage coordinates, forcing subgroup-specific signals into a single shared direction. PRIME instead decomposes the decision into a shared Dense prediction and an input-conditioned residual. Common regularities remain in the anchor, while input-specific deviations are absorbed by selected low-rank experts. It therefore reorganizes parameter sharing without subgroup labels or replicated prediction networks.

The 34.3% reduction in the semantic subgroup competition gap provides direct optimization evidence for this mechanism. Parameterand MAC-matched Dense controls remain below PRIME, a nonconditional residual recovers only part of the gain, and uniform or permuted routing removes the remaining advantage. The improvement consequently requires both residual capacity and inputconditioned organization. Routed experts act as local correction subspaces around a common function, making updates at the shared path more compatible.

To test whether function preservation maters, we compared PRIME with two development-stage alternatives that modify the Dense path: direct hidden-layer subspace replacement and multilayer bagged replacement. Across four Avazu architectures and five seeds, a win required simultaneous validation AUC and LogLoss improvements over Dense. PRIME won 15 of 20 pairs, versus 4 of 20 and 8 of 20 for the two alternatives. Zero-residual initialization and bounded mixing preserve a usable predictor while specialization develops; direct replacement instead changes the baseline immediately and propagates early expert errors.

## 5.2 When PRIME Works: Aligning Experts with the Shared Decision Path

Cross-architecture results identify one atachment principle: the router must observe the representation consumed by the shared decision mapping, and the residual must correct that same stage. This alignment, rather than the interaction operator itself, separates consistently positive backbones from the boundary cases.

FwFM and GDCN expose the boundary because their improvement directions reverse across datasets. FwFM has no shared MLP Top-NN; it predicts from linear and field-weighted pairwise terms [12]. PRIME raises its mean AUC by 0.0163 on Criteo but reduces it by 0.0030 on Avazu. Here the plug-in acts as an auxiliary conditional predictor: extra nonlinear capacity can benefit one distribution, but it does not systematically reorganize a contested shared mapping.

The subgroup diagnostic explains why atachment location matters. The competition gap Γ is defined from gradients with respect to a particular shared mapping; it can be reduced systematically only when routing exposes the subgroup distinctions driving those gradients. GDCN improves by 0.0009 AUC on Avazu but declines by 0.0025 on Criteo. Its decision uses a gated crossed representation [25], whereas PRIME currently routes on pre-cross embeddings. The controls identify pre-cross router conditioning as a major contributor to GDCN’s degradation. An additional five-seed Criteo experiment in Appendix D supports this interpretation: moving the router to an intermediate cross layer raises mean AUC from 0.8041 to 0.8063. When subgroup distinctions emerge after crossing, experts are assigned before the competing directions become visible. They may still learn useful correlations, but their specialization is no longer coupled to GDCN’s final decision path.

Together, these cases motivate architecture-aware atachment: routing should use decision-relevant representations, and the residual should target an explicit shared prediction path.

## 6 Conclusion

This paper identifies a parameter-sharing constraint in fully shared CTR top networks. On Avazu, semantic subgroups consistently produce lower Top-NN gradient alignment than size- and labelmatched random groups. Heterogeneous examples therefore impose less compatible update directions on a single shared mapping, motivating conditional parameter organization within the decision layer.

PRIME addresses this constraint through function-preserving conditional adaptation. The Dense prediction remains anchored, and zero-residual initialization preserves the initial function. Input dependent routing combines bounded low-rank logit corrections without replicating the backbone interaction module. Multi-bag aggregation supplies complementary conditional estimates, while EMA load adjustment limits expert imbalance. This design introduces specialization while retaining the optimization stability and statistical strength of the shared path.

Across held-out Avazu and Criteo tests, PRIME improves mean AUC for 11 of 13 architectures on each dataset. On the FiBiNET and DCNv2 backbones shared with APG, PRIME wins all ten Avazu seed-level comparisons. It also remains above parameter- and MACmatched Dense controls, while routing controls show that the gain arises from input-conditioned expert organization rather than capacity alone. After PRIME training, the semantic subgroup competition gap decreases by 34.3%, connecting stronger predictions with more compatible shared updates. Overall, PRIME provides a compact, function-preserving, and budget-controlled route to conditional adaptation of shared CTR Top-NNs across heterogeneous backbones.

## Ethical Considerations

This study uses public, de-identified CTR benchmark data and involves no human recruitment or personally identifiable information.

## References

[1] H.-T. Cheng, L. Koc, J. Harmsen, T. Shaked, T. Chandra, H. Aradhye, G. Anderson, G. Corrado, W. Chai, M. Ispir, R. Anil, Z. Haque, L. Hong, V. Jain, X. Liu, and H. Shah. 2016. Wide & Deep Learning for Recommender Systems. In Pro ceedings of the 1st Workshop on Deep Learning for Recommender Systems (DLRS ’16). 7–10. https://doi.org/10.1145/2988450.2988454

[2] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In Proceedings of the 10th International Conference on Learning Representations (ICLR ’22). https: //openreview.net/forum?id=nZeVKeeFYf9

[3] K. He, X. Zhang, S. Ren, and J. Sun. 2016. Deep Residual Learning for Image Recognition. In Proceedings ofthe IEEE Conference on Computer Vision and Pat tern Recognition (CVPR ’16). 770–778. https://doi.org/10.1109/CVPR.2016.90

[4] J. Zhu, J. Liu, S. Yang, Q. Zhang, and X. He. 2021. Open Benchmarking for Click-Through Rate Prediction. In Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM ’21). 2759–2769. https://doi.org/10.1145/3459637.3482486

[5] W. Fedus, B. Zoph, and N. Shazeer. 2022. Switch Transformers: Scaling to Tril lion Parameter Models with Simple and Eficient Sparsity. Journal ofMachine Learning Research 23, 120 (2022), 1–39. https://jmlr.org/papers/v23/21-0998.htm

[6] X. He and T.-S. Chua. 2017. Neural Factorization Machines for Sparse Predictive Analytics. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’17). 355–364. https://doi.org/10.1145/3077136.3080777

[7] H. Guo, R. Tang, Y. Ye, Z. Li, and X. He. 2017. DeepFM: A Factorization-Machine Based Neural Network for CTR Prediction. In Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI ’17). 1725–1731. https: //doi.org/10.24963/ijcai.2017/239

[8] T. Huang, Z. Zhang, and J. Zhang. 2019. FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for Click-Through Rate Prediction. In Proceedings ofthe 13th ACM Conference on Recommender Systems (RecSys ’19). 169–177. https://doi.org/10.1145/3298689.3347043

[9] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton. 1991. Adaptive Mixtures of Local Experts. Neural Computation 3, 1 (1991), 79–87. https://doi.org/10.116 2/neco.1991.3.1.79

[10] J. Ma, Z. Zhao, X. Yi, J. Chen, L. Hong, and E. H. Chi. 2018. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts. In Proceedings ofthe 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’18). 1930–1939. https://doi.org/10.1145/3219819.3220007

[11] Y. Qu, H. Cai, K. Ren, W. Zhang, Y. Yu, Y. Wen, and J. Wang. 2016. Product-based Neural Networks for User Response Prediction. In Proceedings ofthe 16th IEEE International Conference on Data Mining (ICDM ’16). 1149–1154. https://doi.org/ 10.1109/ICDM.2016.0151

[12] J. Pan, J. Xu, A. L. Ruiz, W. Zhao, S. Pan, Y. Sun, and Q. Lu. 2018. Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising. In Proceedings ofthe 2018 World Wide Web Conference (WWW ’18). 1349– 1357. https://doi.org/10.1145/3178876.3186040

[13] S. Rendle. 2010. Factorization Machines. In Proceedings of the 10th IEEE International Conference on Data Mining (ICDM ’10). 995–1000. https://doi.org/10.1109/ ICDM.2010.127

[14] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. V. Le, G. E. Hinton, and J. Dean. 2017. Outrageously Large Neural Networks: The Sparsely-Gated Mixtureof-Experts Layer. In Proceedings of the 5th International Conference on Learning Representations (ICLR ’17). https://openreview.net/forum?id=B1ckMDqlg

[15] W. Song, C. Shi, Z. Xiao, Z. Duan, Y. Xu, M. Zhang, andJ. Tang. 2019. AutoInt: Automatic Feature Interaction Learning via Self-Atentive Neural Networks. In Proceedings ofthe 28th ACM International Conference on Information and Knowledge Management (CIKM ’19). 1161–1170. https://doi.org/10.1145/3357384.3357925

[16] B. Zoph, I. Bello, S. Kumar, N. Du, Y. Huang, J. Dean, N. Shazeer, and W. Fedus. 2022. ST-MoE: Designing Stable and Transferable Sparse Expert Models. arXiv:2202.08906. https://arxiv.org/abs/2202.0890

[17] H. Tang, J. Liu, M. Zhao, and X. Gong. 2020. Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations. In Proceedings of the 14th ACM Conference on Recommender Systems (RecSys ’20). 269–278. https://doi.org/10.1145/3383313.3412236

[18] R. Wang, B. Fu, G. Fu, and M. Wang. 2017. Deep & Cross Network for Ad Click Predictions. In Proceedings ofthe ADKDD ’17 Workshop. Article 12, 1–7. https: //doi.org/10.1145/3124749.3124754

[19] R. Wang, R. Shivanna, D. Z. Cheng, S. Jain, D. Lin, L. Hong, and E. H. Chi. 2021. DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems. In Proceedings of the Web Conference 2021 (WWW ’21). 1785–1797. https://doi.org/10.1145/3442381.3450078

[20] Z. Wang, Q. She, and J. Zhang. 2021. MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask. In Proceedings of DLP-KDD ’21. arXiv:2102.07619. https://arxiv.org/abs/2102.07619

[21] L. Wang, H. Gao, C. Zhao, X. Sun, and D. Dai. 2024. Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts. arXiv:2408.15664. https://arxiv.org/ abs/2408.15664

[22] J. Xiao, H. Ye, X. He, H. Zhang, F. Wu, and T.-S. Chua. 2017. Atentional Factorization Machines: Learning the Weight of Feature Interactions via Atention Networks. In Proceedings ofthe 26th International Joint Conference on Artificial Intelligence (IJCAI ’17). 3119–3125. https://doi.org/10.24963/ijcai.2017/435

[23] J. Zhu, K. Mao, Q. Dai, L. Su, R. Ma, J. Liu, G. Cai, Z. Dou, X. Xiao, and R. Zhang. 2022. BARS: Towards Open Benchmarking for Recommender Systems. In Proceedings ofthe 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’22). 2912–2923. https://doi.org/10.1145/ 3477495.3531723

[24] L. Breiman. 1996. Bagging Predictors. Machine Learning 24, 2 (1996), 123–140. https://doi.org/10.1007/BF00058655

[25] F. Wang, H. Gu, D. Li, T. Lu, P. Zhang, and N. Gu. 2023. Towards Deeper, Lighter and Interpretable Cross Network for CTR Prediction. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM ’23). 2523–2533. https://doi.org/10.1145/3583780.3615089

[26] J. Lian, X. Zhou, F. Zhang, Z. Chen, X. Xie, and G. Sun. 2018. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. In Proceedings ofthe 24th ACM SIGKDD International Conference on Knowledge Dis covery & Data Mining (KDD ’18). 1754–1763. https://doi.org/10.1145/3219819.32 20023

[27] G. Zhou, X. Zhu, C. Song, Y. Fan, H. Zhu, X. Ma, Y. Yan, J. Jin, H. Li, and K. Gai. 2018. Deep Interest Network for Click-Through Rate Prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’18). 1059–1068. https://doi.org/10.1145/3219819.3219823

[28] K. Mao, J. Zhu, L. Su, G. Cai, Y. Li, and Z. Dong. 2023. FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction. In Proceedings of the 37th AAAI Conference on Artificial Intelligence (AAAI ’23), Vol. 37, No. 4. 4552–4560. https: //doi.org/10.1609/aaai.v37i4.25577

[29] D. Ha, A. M. Dai, and Q. V. Le. 2017. HyperNetworks. In Proceedings of the 5th International Conference on Learning Representations (ICLR ’17). https://openre view.net/forum?id=rkpACe1lx

[30] B. Yan, P. Wang, K. Zhang, F. Li, H. Deng, J. Xu, and B. Zheng. 2022. APG: Adap tive Parameter Generation Network for Click-Through Rate Prediction. In Advances in Neural Information Processing Systems 35 (NeurIPS ’22). 24740–24752. https://proceedings.neurips.cc/paper\_files/paper/2022/hash/9cd0c57170f4852

0749d5ae62838241f-Abstract-Conference.html

[31] X.-R. Sheng, L. Zhao, G. Zhou, X. Ding, B. Dai, Q. Luo, S. Yang, J. Lv, C. Zhang, H. Deng, and X. Zhu. 2021. One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction. In Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM ’21). 4104–4113. https://doi.org/10.1145/3459637.3481941

[32] J. Chang, C. Zhang, Y. Hui, D. Leng, Y. Niu, Y. Song, and K. Gai. 2023. PEP-Net: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23). 3795–3804. https: //doi.org/10.1145/3580305.3599884

[33] Y. Zhou, T. Lei, H. Liu, N. Du, Y. Huang, V. Y. Zhao, A. M. Dai, Z. Chen, Q. V. Le, and J. Laudon. 2022. Mixture-of-Experts with Expert Choice Routing. In Advances in Neural Information Processing Systems 35 (NeurIPS ’22). 7103–7114. https://proceedings.neurips.cc/paper\_files/paper/2022/hash/2f00ecd787b432c 1d36f3de9800728eb-Abstract-Conference.htm

[34] J. Puigcerver, C. Riquelme Ruiz, B. Mustafa, and N. Houlsby. 2024. From Sparse to Soft Mixtures of Experts. In Proceedings ofthe 12th International Conference on Learning Representations (ICLR ’24). https://openreview.net/forum?id=jxps Aj7ltE

[35] D. Dai, C. Deng, C. Zhao, R. X. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu, Z. Xie, Y. K. Li, P. Huang, F. Luo, C. Ruan, Z. Sui, and W. Liang. 2024. DeepSeek MoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (ACL ’24). 1280–1297. https://doi.org/10.18653/v1/2024.acllong.70

[36] Z. Wang, J. Zhu, and J. Chen. 2025. ReMoE: Fully Diferentiable Mixture-of-Experts with ReLU Routing. In Proceedings ofthe 13th International Conference on Learning Representations (ICLR ’25). https://proceedings.iclr.cc/paper\_ files/paper/2025/hash/94dc604e115237a7f4a758b3146cd976- Abstract-Conference.html

[37] X. Kong, J. Wu, A. Zhang, L. Sheng, H. Lin, X. Wang, and X. He. 2024. Customizing Language Models with Instance-wise LoRA for Sequential Recommendation. In Advances in Neural Information Processing Systems 37 (NeurIPS ’24). https://doi.org/10.52202/079017-3593

[38] Z. Yang, H. Gao, D. Gao, L. Yang, L. Yang, X. Cai, W. Ning, and G. Zhang. 2024. MLoRA: Multi-Domain Low-Rank Adaptive Network for CTR Prediction. In Proceedings of the 18th ACM Conference on Recommender Systems (RecSys ’24). 287– 297. https://doi.org/10.1145/3640457.3688134

[39] Z. Zeng, M. Hang, X. Liu, X. Liu, X. Lin, R. Qiu, T. Wei, Z. Liu, S. Yuan, C. Yang, Y. Liu, H. Yin, J. Yang, and H. Tong. 2025. Hierarchical LoRA MoE for Eficient CTR Model Scaling. arXiv:2510.10432. https://arxiv.org/abs/2510.10432

[40] K. Yagel, E. German, and A. Ben Siman Tov. 2025. MoE-MLoRA for Multi-Domain CTR Prediction: Eficient Adaptation with Expert Specialization. arXiv:2506.07563. https://arxiv.org/abs/2506.07563

[41] H. Huang, N. Ardalani, A. Sun, L. Ke, H.-H. S. Lee, S. Bhosale, C.-J. Wu, and B. Lee. 2024. Toward Eficient Inference for Mixture of Experts. In Advances in Neural Information Processing Systems 37(NeurIPS ’24). https://doi.org/10.52202/079017- 2670

[42] A. Lv, J. Ma, Y. Ma, and S. Qiao. 2026. Coupling Experts and Routers in Mixtureof-Experts via an Auxiliary Loss. In Proceedings ofthe 14th International Conference on Learning Representations (ICLR ’26). Oral presentation. https://openrevi ew.net/forum?id=MpeyjgWbKt

[43] Z. Li, Z. Li, and T. Zhou. 2026. Routing Manifold Alignment Improves Generalization of Mixture-of-Experts LLMs. In Proceedings of the 14th International Conference on Learning Representations (ICLR ’26). https://openreview.net/for um?id=3lskwxB653

[44] D. Gao, S. Chen, Z. Yang, L. Yang, H. Gao, M. Wu, S. Yu, Q. Xuan, W. Zhang, L. Yang, and X. Cai. 2026. MLoRA+: Transformer-Fusion Mixture-of-LoRA Net work for Multi-Domain Click-Through Rate Prediction. Expert Systems with Ap plications 297: 129486. https://doi.org/10.1016/j.eswa.2025.129486

## Supplementary Material

## A Reproducibility and Design Map

All main experiments use Adam with learning rate $1 0 ^ { - 3 }$ , batch size 4,096, and validation AUC for checkpoint selection. Training proceeds for at most 100 epochs, stops after two consecutive validation checks without improvement, and restores the best checkpoint. The embedding dimension is 10. A typical Top-NN has three hidden layers of width 400; dropout, batch normalization, and regularization follow the public configuration of each backbone and remain fixed within every Dense/PRIME pair.

The frozen PRIME configuration uses $G = 4$ bags, $M = 8$ experts per bag, rank $q = 1 6 ,$ Dense weight $\alpha = 0 . 5$ , routing temperature $T = 1$ , moving-load coeficient $\mu = 0 . 9 9$ , bias step size $\eta _ { b } = 1 0 ^ { - 3 }$ , and correction bound $c _ { \operatorname* { m a x } } = 2$ . The software stack consists of Python 3.10.16, PyTorch 2.6.0, CUDA 12.6, cuDNN 8.9.5, and FuxiCTR 2.3.10. Training uses 16 PPU-ZW810E accelerators. Checkpoints, resolved configurations, validation records, and test records are stored separately, while hashes of the feature map, configuration, and code revision permit run-level verification. No job is repeated because of an unfavorable metric.

The design hypotheses and their corresponding tests are summarized below.

(1) Conditional correction. Weak alignment across semantic subgroups motivates input-conditioned routing. Uniform and permuted-routing controls test whether preserving the input–expert correspondence maters.

(2) Function preservation. Perturbing a validated prediction path motivates the Dense anchor and zero-residual initialization. The no-anchor ablation tests this requirement.

(3) Budget control. Replicating complete Top-NNs is unnecessary for residual correction. Parameter- and MAC-matched non-conditional adapters test whether low-rank conditional organization contributes beyond capacity alone.

(4) Conditional aggregation. Multiple expert bags provide several conditional estimates. The single-bag control tests whether their average supplies an additional empirical benefit.

(5) Objective preservation. EMA load bias separates load adjustment from CTR gradients. The auxiliary-loss control tests whether comparable accuracy can be obtained with out introducing another loss coeficient.

The first three items define PRIME’s core parameterization and eficiency constraints; the final two provide complementary stabilization and aggregation.

## B Complete APG Comparison

Table S1 reports the complete Avazu comparison with APG on FiBiNET and DCNv2. For FiBiNET, APG applies an input-conditioned low-rank transform to field embeddings before bilinear interaction while retaining the original bilinear module and Dense Top-NN; DCNv2 uses self-wise APG parameter generation in its prediction path. All configurations use the same data split, embedding dimension, optimizer, batch size, early-stopping protocol, and five paired random seeds.

PRIME achieves higher AUC in all ten paired seed-level comparisons; the smallest margin is +0.0002 on FiBiNET at seed 190034. On FiBiNET, APG improves AUC over Dense by 0.0020, while PRIME adds a further 0.0030 and reduces LogLoss by another 0.0021 relative to APG. On DCNv2, PRIME exceeds both Dense and APG in predictive accuracy. Across both backbones, PRIME also uses fewer parameters and has lower inference latency than APG.

## C Routing Diagnostics and Dense-Weight Sensitivity

Across 16 Avazu runs with complete diagnostic records, normalized routing entropy over eight experts per bag is $0 . 4 3 2 \pm 0 . 0 8 5 ,$ corresponding to 2.49±0.47 efective experts. The maximum mean load ofone expert is 0.425±0.116, above the uniform value of0.125. Multiple experts remain active while preserving pronounced specialization. The observed load profile is consistent with the EMA load-bias update preventing single-expert collapse without erasing the expert preferences required for conditional routing.

With development seed 190034 and four architectures, $\alpha = 0 . 2 5 ,$ 0.5, and 0.75 produce mean validation AUCs of 0.7493, 0.7480, and 0.7474, with LogLoss values of 0.3947, 0.3955, and 0.3959. We then compare 0.25 and the frozen default 0.5 across four architectures and five seeds. Their mean AUCs are 0.7483 and 0.7480, with closely matched LogLoss. DNN and FiBiNET favor 0.5, whereas AutoInt and DCNv2 favor 0.25. These complementary directions support retaining the prespecified $\alpha \ = \ 0 . 5$ as a balanced seting across backbones.

## D GDCN Attachment Controls on Criteo

To examine why the original PRIME configuration trails GDCN on Criteo, we keep the GDCN backbone and PRIME capacity fixed while changing the representations supplied to the router and experts. Tables S2 and S3 report the seed-level AUCs and complete five-seed atachment controls, respectively. Pre denotes flatened embeddings before the gated cross network, Cross-1 and Cross-2 denote the outputs of its first and second cross layers, and Post denotes the final crossed representation. Unless stated otherwise, the controls use input LayerNorm, an atached router gradient, and the bounded probability-mixture output. All results are held-out test metrics over the same five paired seeds.

Table S1: Complete held-out Avazu comparison with APG. AUC and LogLoss are five-seed means; latency is the median per 4,096-example batch over three profiling repetitions. Seed columns report held-out AUC. Δ is PRIME−APG; positive AUC and negative LogLoss or cost diferences favor PRIME. Bold values mark improvements over APG.
<table><tr><td>Backbone</td><td></td><td colspan="2">Five-seed mean</td><td colspan="2">Deployment</td><td colspan="5">Held-out AUC by seed</td></tr><tr><td></td><td>Method</td><td>AUC</td><td>LogLoss</td><td>Params (M)</td><td>Latency (ms)</td><td>2021</td><td>190034</td><td>27011</td><td>948432</td><td>992817</td></tr><tr><td>FiBiNET</td><td>Dense</td><td>0.7600</td><td>0.3695</td><td>19.1582</td><td>2.1993</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>APG</td><td>0.7620</td><td>0.3684</td><td>19.4240</td><td>2.4191</td><td>0.7619</td><td>0.7641</td><td>0.7616</td><td>0.7619</td><td>0.7606</td></tr><tr><td></td><td>PRIME</td><td>0.7650</td><td>0.3663</td><td>19.2784</td><td>2.3752</td><td>0.7648</td><td>0.7644</td><td>0.7657</td><td>0.7643</td><td>0.7657</td></tr><tr><td></td><td>∆</td><td>+0.0030</td><td>-0.0021</td><td>-0.1456</td><td>-0.0439</td><td>+0.0030</td><td>+0.0002</td><td>+0.0041</td><td>+0.0024</td><td>+0.0051</td></tr><tr><td>DCNv2</td><td>Dense</td><td>0.7626</td><td>0.3680</td><td>16.0000</td><td>1.1260</td><td></td><td>一</td><td></td><td>一</td><td></td></tr><tr><td></td><td>APG</td><td>0.7610</td><td>0.3689</td><td>17.0100</td><td>1.7270</td><td>0.7609</td><td>0.7601</td><td>0.7611</td><td>0.7621</td><td>0.7609</td></tr><tr><td></td><td>PRIME</td><td>0.7642</td><td>0.3671</td><td>16.1200</td><td>1.4521</td><td>0.7622</td><td>0.7642</td><td>0.7659</td><td>0.7641</td><td>0.7645</td></tr><tr><td></td><td>Δ</td><td>+0.0032</td><td>-0.0018</td><td>-0.8900</td><td>-0.2749</td><td>+0.0013</td><td>+0.0041</td><td>+0.0048</td><td>+0.0020</td><td>+0.0036</td></tr></table>

Table S2: Seed-level AUC for GDCN/Criteo router controls, with means computed from unrounded run-level metrics. Experts remain on pre-cross embeddings in the cross-layer variants.
<table><tr><td>Seed</td><td>Dense</td><td>Original</td><td>Cross-1</td><td>Cross-2</td><td>Cross-2, no LN</td></tr><tr><td>2021</td><td>0.8081</td><td>0.8088</td><td>0.8056</td><td>0.8046</td><td>0.8046</td></tr><tr><td>190034</td><td>0.8031</td><td>0.8010</td><td>0.8070</td><td>0.8081</td><td>0.8084</td></tr><tr><td>27011</td><td>0.8078</td><td>0.8071</td><td>0.8054</td><td>0.8071</td><td>0.8073</td></tr><tr><td>948432</td><td>0.8070</td><td>0.8090</td><td>0.8063</td><td>0.8082</td><td>0.8061</td></tr><tr><td>992817</td><td>0.8073</td><td>0.7949</td><td>0.8073</td><td>0.8033</td><td>0.8066</td></tr><tr><td>Mean</td><td>0.8066</td><td>0.8041</td><td>0.8063</td><td>0.8063</td><td>0.8066</td></tr></table>

Table S3: Complete five-seed GDCN/Criteo attachment controls. Entropy is the unnormalized mean router entropy in nats over eight experts per bag; max load is the largest mean expert probability. The logit row replaces bounded probability mixing with a direct logit residual.
<table><tr><td>Configuration</td><td>Router</td><td>Expert</td><td>AUC</td><td>LogLoss</td><td>Entropy</td><td>Max load</td></tr><tr><td>Dense</td><td>一</td><td>一</td><td>0.8066</td><td>0.4447</td><td>一</td><td>一</td></tr><tr><td>Original PRIME</td><td>Pre</td><td>Pre</td><td>0.8041</td><td>0.4469</td><td>一</td><td></td></tr><tr><td>Post/Post</td><td>Post</td><td>Post</td><td>0.8059</td><td>0.4452</td><td>0.9674</td><td>0.5328</td></tr><tr><td>Post/Pre</td><td>Post</td><td>Pre</td><td>0.8061</td><td>0.4452</td><td>0.9843</td><td>0.5418</td></tr><tr><td>Pre/Post</td><td>Pre</td><td>Post</td><td>0.8035</td><td>0.4473</td><td>0.4086</td><td>0.9021</td></tr><tr><td>Post/Pre, detached</td><td>Post</td><td>Pre</td><td>0.8051</td><td>0.4460</td><td>0.8986</td><td>0.6563</td></tr><tr><td>Post/Pre, logit</td><td>Post</td><td>Pre</td><td>0.7975</td><td>0.4529</td><td>0.9375</td><td>0.4424</td></tr><tr><td>Cross-1/Pre</td><td>Cross-1</td><td>Pre</td><td>0.8063</td><td>0.4450</td><td>0.9273</td><td>0.5928</td></tr><tr><td>Cross-2/Pre</td><td>Cross-2</td><td>Pre</td><td>0.8063</td><td>0.4451</td><td>0.9817</td><td>0.5672</td></tr><tr><td>Cross-2/Pre, no LN</td><td>Cross-2</td><td>Pre</td><td>0.8066</td><td>0.4448</td><td>1.1695</td><td>0.5302</td></tr></table>

Moving the router from pre-cross embeddings to an intermediate cross layer raises mean AUC from 0.8041 to 0.8063. The more diagnostic reversal holds the expert source fixed: Post/Pre reaches 0.8061 AUC, whereas Pre/Post falls to 0.8035. The later also has substantially lower router entropy (0.4086 versus 0.9843) and a higher maximum expert load (0.9021 versus 0.5418). These controls identify pre-cross router conditioning, rather than insuficient expert depth, as a major contributor to the original degradation.

The remaining variants delimit the mechanism. Post/Pre is slightly stronger than Post/Post, so exact coincidence between router and expert inputs is unnecessary. Detaching the router gradient lowers mean AUC, while direct logit residuals produce the weakest result. GDCN therefore benefits from routing on decision-relevant crossed representations while retaining PRIME’s Dense anchor and bounded probability mixture.

## E End-to-End Profiling Details

Each method is profiled on the same accelerator with batch size 4,096, warmed up for 20 steps, timed for 100 inference steps and 50 training steps, and repeated independently three times. A training step includes forward propagation, backpropagation, and an optimizer update. Inference uses model.eval() without gradient tracking. PRIME’s deployment path omits balancing losses, orthogonality Gram matrices, and host-side routing diagnostics used only during training or analysis. Table S4 reports median absolute latency and the median paired ratio to the corresponding Dense backbone within each repetition.

Across the two backbones, PRIME adds only 0.6%–0.8% parameters over Dense. Its training-step overhead ranges from 8.4% on FiBiNET to 16.9% on DCNv2. PRIME is faster and substantially more memory-eficient than APG on DCNv2. On FiBiNET, prebilinear APG is 2.6% faster per training step, whereas PRIME lowers peak training memory by 36.0% and remains faster at inference. Its soft router evaluates all low-rank experts in every bag, so the measured latency reflects full conditional estimation [41]. Sparse scheduling and operator fusion ofer direct paths to further acceleration.

## F Development-Stage Structural Controls

During development, we evaluated two expert designs that modify the Dense path. The first replaces hidden-layer linear transformations with subspace experts. The second inserts bagged experts at multiple hidden stages. These alternatives provide structural controls for the function-preservation requirement. Because they alter the baseline mapping, early expert errors propagate through subsequent nonlinear layers and their optimization trajectories begin from a diferent function than the original model.

Across four Avazu architectures and five seeds, PRIME improves both validation AUC and LogLoss in 15 of 20 pairs, compared with for the best multi-layer bagged replacement. A win requires simultaneous improvement in both metrics. This decisive margin isolates function preservation as a central requirement and motivates PRIME’s Dense-anchored design.

Table S4: End-to-end profiling on Avazu. Absolute latencies are medians of three independent repetitions, and ratios use the matching Dense backbone. Each inference batch and training step processes 4,096 examples. Bold values mark the lower cost between APG and PRIME for a given backbone.
<table><tr><td></td><td>Method</td><td></td><td colspan="2">Inference</td><td colspan="2">Training</td><td></td></tr><tr><td>Backbone</td><td></td><td>Parameter ratio</td><td>Latency (ms)</td><td>Ratio</td><td>Latency (ms)</td><td>Ratio</td><td>Peak-memory ratio</td></tr><tr><td>FiBiNET</td><td>Dense</td><td>1.0000</td><td>2.1993</td><td>1.0000</td><td>14.9712</td><td>1.0000</td><td>1.0000</td></tr><tr><td></td><td>APG</td><td>1.0139</td><td>2.4191</td><td>1.1001</td><td>15.7662</td><td>1.0535</td><td>1.5712</td></tr><tr><td></td><td>PRIME</td><td>1.0063</td><td>2.3752</td><td>1.0849</td><td>16.1865</td><td>1.0840</td><td>1.0062</td></tr><tr><td>DCNv2</td><td>Dense</td><td>1.0000</td><td>1.1260</td><td>1.0000</td><td>8.8583</td><td>1.0000</td><td>1.0000</td></tr><tr><td></td><td>APG</td><td>1.0632</td><td>1.7270</td><td>1.5337</td><td>10.9516</td><td>1.2402</td><td>1.8738</td></tr><tr><td></td><td>PRIME</td><td>1.0075</td><td>1.4521</td><td>1.2896</td><td>10.3558</td><td>1.1690</td><td>1.0231</td></tr></table>

4 of 20 for direct hidden-layer subspace replacement and 8 of 20