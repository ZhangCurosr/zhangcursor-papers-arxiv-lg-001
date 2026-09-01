# FINE-TUNING LOW-BIT MODELS WITH GRADIENT IN QUANTIZED CODE SPACE

Shiguang Wu<sup>1</sup>, Zhouchen Lin<sup>2</sup>, Quanming Yao<sup>1</sup>

<sup>1</sup>Department of Electronic Engineering, Tsinghua University

<sup>2</sup>School of Intelligence Science and Technology, Peking University

wsg23@mails.tsinghua.edu.cn, qyaoaa@tsinghua.edu.cn

## ABSTRACT

Fine-tuning Low-bit models aims to adapt a quantized model while keeping the final deployed checkpoint in the same low-bit form. This setting is practically important as it reduces memory and inference cost for storage and deployment. Under this constraint, adaptation becomes an optimization problem over quantization codes and scales. Existing continuous low-bit training is efficient, but it can be distorted by straight through estimation error or by post-quantize gap; discrete search is deployment-faithful, but it is often too inefficient under a finite training budget. We propose code surrogate gradient as the first order signal in deployable code space to acceleate optimization, and performing guided search to preserve deployment faithfulness. Experiments across arithmetic reasoning, instruction following, and structured language understanding show that GradCodeS consistently improves fine-tuning low-bit models across different quantization datatypes. Code is provided at https://github.com/ovo67/GradCodes.

## 1 INTRODUCTION

Large language models (LLMs) are increasingly deployed in low-precision formats such as INT4, NF4, and MXFP4 to reduce storage, memory traffic, and inference cost (Xiao et al., 2023; Lin et al., 2024). These savings matter not only for a shared backbone, but also when many task- or user-specific checkpoints must be stored under a fixed memory budget. We study a strict fully low-bit setting in which the targeted weights of every adapted checkpoint remain in the target quantized representation, with only datatype-permitted scale metadata and no high-precision residual adapter at inference time (Jeon et al., 2025; Malinovskii et al., 2024). Under this constraint, fine-tuning is no longer optimization in a nearly continuous floating-point space; it is optimization over quantization codes and scales.

One practical strategy retains continuous optimization. Quantized-backbone adapter methods train high-precision LoRA-style parameters and either keep them at inference time or merge and requantize them afterward (Dettmers et al., 2023; Li et al., 2024; Loeschcke et al., 2024; Xu et al., 2024). This is effective when mixed-precision deployment or a separate conversion stage is acceptable. Under the stricter setting above, however, retaining the adapter violates the deployment constraint, while merging and requantizing can make the evaluated low-bit state differ from the state optimized during training. Quantization-aware and compressed-representation methods move optimization closer to the final checkpoint (Jeon et al., 2025; Malinovskii et al., 2024), but still leave a fundamental question: how should first-order information be expressed when the actual decision variable is a quantization-code index?

Direct discrete and zeroth-order approaches instead evaluate candidates in or near the quantized model family (Zhou et al., 2025; Shang et al., 2026; Xu et al., 2026), but random or population proposals can require many forward evaluations in high-dimensional code spaces. The central challenge is to combine deployment-faithful selection with first-order directional guidance.

To be deployment-faithful, we optimize over quantized code coordinates rather than weight. To be efficient, we try to find more useful information beyond zero-order. Based on this principle, we propose GradCodeS, a gradient-guided search method. We first identify that mapping ordinary gradient in weight space to code space is generally not the steepest descent direction in code space, as illustrated in Figure 1(a), thus introducing a code-coordinate surrogate gradient by codebook continuation, serving as first-order information in code space with theoretical support. GradCodeS uses code-coordinate surrogate gradient descent to provide reference and then samples multiple nearby code candidates biased toward that reference and selects an update only according to its realized deployed loss. Thus every selected iterate is a valid low-bit checkpoint. This rule is parameterization-agnostic and datatype-agnostic: it can be instantiated on full weights, integrated with structured parameterizations such as LoRA, combined with joint scale optimization, and paired with different quantization datatypes.

![](images/a4fcc4f1f00e85fca26be89635d4df925a76a8bb3ef21c6d353f3f6c8534ffc2.jpg)  
(a) Geometry mismatch between weight and code spaces.

![](images/e17a70b5d4fad11e648fdd99f99374c93f6eb52fef9d59367b13078ac269432e.jpg)  
(b) Accuracy under different deployment regimes.  
Figure 1: Motivation for deployment-faithful code-space optimization. (a) Group scales and nonuniform codebook gaps distort local geometry, so a weight gradient does not identify the best discrete code move. (b) In the Llama-3.2-1B GSM8K example, continuous adaptation either retains an FP16 adapter or loses 13.4 points after QLoRA requantization; GradCodeS improves the fully 4-bit result while retaining a deployable checkpoint.

Our contributions can be summarized as follows:

• We derive a code-coordinate surrogate gradient and establish its local steepest-descent interpretation under codebook continuation.

• We develop a guided candidate-sampling algorithm that combines first-order guidance with lossbased selection over deployable low-bit candidates.

• We evaluate GradCodeS across three task families, multiple quantization datatypes, and both full and structured code parameterizations, observing consistent gains in the fully low-bit setting.

## 2 RELATED WORK

Continuous low-bit adaptation. Low-bit adaptation commonly retains continuous optimization. Quantized-backbone parameter-efficient methods train high-precision adapters, improve their quantized initialization, or design updates that can later be merged (Dettmers et al., 2023; Li et al., 2024; Xu et al., 2024; Loeschcke et al., 2024). They are effective when mixed-precision inference or a separate conversion stage is acceptable, but under strict fully low-bit deployment the optimized and deployed states may differ because an adapter remains at inference time or a merged update must be requantized. Quantization-aware and compressed-representation methods, including L4Q and PV-Tuning, optimize closer to the final quantized checkpoint (Jeon et al., 2025; Malinovskii et al., 2024). Their focus is efficient optimization of differentiable or structured compressed representations. Our complementary question is how a weight gradient should be expressed in ordered code coordinates without treating the resulting continuous reference as an accepted update.

Direct low-bit optimization. This approach avoids a separate deployment conversion. QuZO and QZO use zeroth-order estimates for low-precision or quantization-aware fine-tuning, while QES applies evolution strategies to quantized parameters (Zhou et al., 2025; Shang et al., 2026; Xu et al., 2026). These methods reduce or avoid backpropagation and evaluate search information through forward perturbations or population fitness. Although candidate quality can therefore be measured in or near the quantized model family, proposal efficiency remains difficult in a high-dimensional space under a finite evaluation budget. GradCodeS instead spends one backward pass to obtain an analytical local signal, uses it only to guide where discrete search looks, and still selects updates by the realized loss of deployable low-bit candidates.

Gradient-informed discrete proposals. A derivative can shape a discrete proposal without being followed deterministically. Representative samplers construct local, Langevin-style, Newton-style, or importance proposals from gradients with respect to discrete inputs (Grathwohl et al., 2021; Zhang et al., 2022; Sun et al., 2023a; Xiang et al., 2023; Liu et al., 2023); related optimization methods bias random search with surrogate gradients or adapt discrete sampling to combinatorial objectives (Maheswaranathan et al., 2019; Sun et al., 2023b; Li & Zhang, 2025). These works motivate separating proposal construction from state selection, but they do not resolve two features of low-bit fine-tuning. First, the available derivative is with respect to recovered weights, whereas the decision variable is a code index whose geometry is distorted by group scales and non-uniform codebook gaps. Second, our goal is loss minimization over a finite candidate set rather than preservation of a stationary distribution. GradCodeS addresses these differences through a quantization-specific code-coordinate signal, proposal-only gradient guidance, and loss-based selection of deployable states.

## 3 PROBLEM SETUP

Fix a target quantization datatype $q ,$ specified by its bitwidth $b _ { q } ,$ numerical codebook $\mathcal { C } _ { q } ,$ grouping structure $\mathcal { G } _ { q }$ , and admissible scale space $\textstyle { \mathcal { S } } _ { q }$ . Its ordered codebook is

$$
{ \mathcal C } _ { q } = \{ c _ { 1 } ^ { ( q ) } , \dots , c _ { K _ { q } } ^ { ( q ) } \} , \qquad c _ { 1 } ^ { ( q ) } \le \cdots \le c _ { K _ { q } } ^ { ( q ) } , \qquad K _ { q } \le 2 ^ { b _ { q } } .
$$

The indices follow numerical order and need not match hardware bit-pattern order. For a weight matrix $W \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ , let $[ K _ { q } ] : = \{ 1 , \dots , K _ { q } \}$ and $\mathcal { Z } _ { q } : = [ K _ { q } ] ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ . A code matrix $Z \in { \mathcal { Z } } _ { q }$ selects values entrywise as $\mathcal { C } _ { q } ( Z ) = [ c _ { Z _ { i j } } ^ { ( q ) } ] _ { i j }$ , where $\left[ \psi ( i , j ) \right] _ { i j }$ denotes the matrix whose $( i , j )$ -th element is $\psi ( i , j )$ . Each entry belongs to a group $g _ { q } ( i , j ) \in \check { \mathcal { G } } _ { q }$ with scale metadata $s = \{ s _ { g } \} _ { g \in { \mathcal G } _ { q } } \in S _ { q }$ Defining $S _ { q } ( s ) = [ s _ { g _ { q } ( i , j ) } ] _ { i j }$ , the deployed weight is

$$
\widehat { W } _ { q } ( Z , s ) = S _ { q } ( s ) \odot { \mathcal C } _ { q } ( Z ) ,\tag{1}
$$

where $\odot$ is element-wise multiplication. This representation covers the evaluated datatypes NF4, INT4, and MXFP4; their datatype-specific components are given in Appendix B.

For a model, let $\mathcal { L }$ be the set of targeted quantized linear layers. We use $Z$ and s for their layer-wise collections and let $\mathcal { Q } _ { q }$ denote the product of the corresponding layer-wise feasible code and scale sets. We use $\widehat { { \bf W } } _ { q } ( Z , s )$ for the recovered weights. Parameters outside $\mathcal { L }$ are left unquantized.

Given an initial feasible state $( Z _ { 0 } , s _ { 0 } )$ , a target distribution $\mathcal { D } _ { : }$ , a model $f ,$ , and a loss function $\ell ,$ we formulate fully low-bit fine-tuning as

$$
\begin{array} { r l } & { \operatorname* { m i n } _ { ( Z , s ) \in { \mathscr { Q } _ { q } } } L ( Z , s ) : = { \mathbb E } _ { ( x , y ) \sim \mathcal { D } } \left[ \ell \left( f \left( x ; \widehat { { \mathbf W } } _ { q } ( Z , s ) \right) , y \right) \right] . } \end{array}\tag{2}
$$

Unlike approaches that optimize a continuous auxiliary state and quantize only after training, $( 2 )$ is defined directly over deployable states. Here $Z$ is discrete, while s is continuous in the default setting but constrained to its admissible space. Since q is fixed, we henceforth omit datatype annotations from $K _ { q } , c _ { k } ^ { ( q ) } , \mathcal { C } _ { q } , \mathcal { Z } _ { q } , \mathcal { S } _ { q } ,$ , and $S _ { q }$ . Thus, for a single matrix, ${ \widehat W } ( Z , s ) = S ( s ) \odot { \mathcal { C } } ( Z )$ . Bold $\widehat { \bf W }$ is reserved for the full collection of targeted weights, while L denotes the corresponding objective.

## 4 METHOD

To optimize (2), GradCodeS separates gradient guidance from discrete state selection. Section 4.1 constructs a geometry-aware code surrogate gradient, and Section 4.2 establishes its local and discrete first-order properties. Section 4.3 then incorporates this signal into a guide–sample–evaluate–select procedure that accepts updates according to the realized loss of deployable low-bit states.

## 4.1 CONSTRUCTING THE CODE SURROGATE GRADIENT

The ordinary weight gradient $\nabla _ { \widehat { W } } L$ is not directly a signal for the discrete code index because group scales and local codebook gaps rescale a unit code move. Thus in general, the direction in code space corresponding to the steepest descent direction in weight space is not the steepest descent direction in code space, as illustrated in Figure 2.

For a matrix A, write $A _ { + } = [ \operatorname* { m a x } \{ A _ { i j } , 0 \} ] _ { i j }$ and $A _ { - } = [ \operatorname* { m a x } \{ - A _ { i j } , 0 \} ] _ { i j }$ . Let $Z ^ { \pm } = \mathrm { c l i p } ( Z \pm$ $\mathbb { 1 } , 1 , K )$ and define the forward and backward gaps

$$
G ^ { + } ( Z ) = \left[ c _ { Z _ { i j } ^ { + } } - c _ { Z _ { i j } } \right] _ { i j } , ~ G ^ { - } ( Z ) ~ = \left[ c _ { Z _ { i j } } - c _ { Z _ { i j } ^ { - } } \right] _ { i j } .
$$

The quantities above characterize the local geometry of the two neighboring code moves. To use a weight gradient for discrete search, we need a code-coordinate signal that reflects this geometry; Definition 1 constructs such a signal.

Definition 1 (Local effective step and code surrogate gradient). Given a quantized state $( Z , s )$ with fixed codebook C, let $\sigma = \mathrm { s i g n } ( \nabla _ { \widehat { W } } L ( Z , s ) )$ ) and define the descent-aligned effective step by $D \big ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \big ) = S ( s ) \odot \big ( G ^ { + } ( \ddot { Z } ) \odot \sigma _ { - } + G ^ { - } ( Z ) \odot \sigma _ { + } \big )$ . The directional code surrogate gradient is then

$$
\nabla _ { Z } L ( Z , s ) = \nabla _ { \widehat { W } } L ( Z , s ) \odot D \big ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \big ) .\tag{3}
$$

Figure 2 gives a simplified illustration of Definition 1. It uses an identity codebook, whose adjacent gaps are all one, so the effective step reduces to $D = S ( s )$ . The example therefore shows even without nonuniform codebook gaps, accounting for the deployed-weight change of each code coordinate yields a substantially better descent direction. Appendix C derives this construction from the local piecewise-linear continuation.

$$
\mathcal { C } ( Z ) = Z , S ( s ) = ( 0 . 1 , 1 0 ) , L ( \hat { W } ) = \hat { W } _ { 1 } + 0 . 6 \hat { W } _ { 2 } . \nabla _ { \widehat { W } } L = ( 1 , 0 . 6 )
$$

$$
\nabla _ { \widehat { W } } L ) \mathrm { i s } \ ( - 1 0 , - 0 . 0 6 ) ;
$$

$$
\mathbf { \boldsymbol { D } } = \mathbf { \boldsymbol { S } } ( \mathbf { \boldsymbol { s } } )
$$

$$
\nabla _ { Z } L \overset { \because } { = } ( 0 . 1 , 6 )
$$

$$
\Delta Z = \epsilon d \stackrel { \ r { \prime } \bar { \operatorname { w i t h } } } { \mathrm { w i t h } } \lVert d \rVert _ { 2 } ^ { - } = 1 \mathrm { { : } }
$$

<table><tr><td>Direction</td><td>Unit direction d</td><td>Loss change  $\Delta L$ </td></tr><tr><td>Mapped weight gradient</td><td> $( - 1 0 , - 0 . 0 6 ) / \sqrt { 1 0 0 . 0 0 3 6 }$ </td><td>-0.136€</td></tr><tr><td>Definition 1 surrogate</td><td> $( - 0 . 1 , - 6 ) / \sqrt { 3 6 . 0 1 }$ </td><td>-6.001€</td></tr></table>

Result. The effective-step correction yields about 44× more loss decrease for the same code-space step.

Figure 2: A two-coordinate illustration of the effective-step correction in Definition 1.

## 4.2 FIRST-ORDER PROPERTIES

The results below summarize the overall argument. Lemma 1 gives the surrogate a local continuous interpretation. Proposition 1 establishes the central discrete connection by showing exact firstorder consistency for feasible one-level code moves. Corollary 1 then converts this identity into a realized-loss bound used by the candidate-search analysis in Proposition 2.

Lemma 1 (Local steepest-descent interpretation). Let $D _ { Z , s } : = D \big ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \big )$ . Define the local directional surrogate function of $L ( Z , s )$ by extending it to real-valued interpolation $\Delta Z \in$ $\mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } } \ b y$

$$
\bar { L } _ { Z , s } ( \Delta Z ) = L \Bigl ( \widehat { W } ( Z , s ) + D _ { Z , s } \odot \Delta Z \Bigr ) .
$$

Then $\nabla _ { \Delta Z } \bar { L } _ { Z , s } ( \Delta Z ) \big | _ { \Delta Z = 0 } = \nabla _ { Z } L ( Z , s )$ . Consequently, whenever $\nabla _ { Z } L ( Z , s ) \neq 0 ;$ , the direction $- \nabla _ { Z } L ( Z , s )$ is the steepest descent direction of $\bar { L } _ { Z , s } a t \Delta Z = 0$ under the Frobenius norm.

Lemma $1 \ \mathrm { g i v e s } - \nabla _ { Z } L$ a precise but deliberately local interpretation: it is the steepest direction of the active continuous surrogate, not an automatically valid code update. We next connect its score to an actual one-level code move.

Proposition 1 (One-step consistency). Let $\Delta Z \in \{ - 1 , 0 , 1 \} ^ { d _ { \mathrm { o u t } } }$ <sup>t×din</sup> be a feasible one-step code move such that $Z + \Delta Z \in { \mathcal { Z } } , \Delta Z \odot \nabla _ { \widehat { W } } L ( Z , s ) \leq 0$ entrywise. Then the surrogate linear term matches the exactfirst-order deployed-weight change:

$$
\langle \nabla _ { Z } L ( Z , s ) , \Delta Z \rangle _ { F } = \Bigl \langle \nabla _ { \widehat { W } } L ( Z , s ) , \widehat { W } ( Z + \Delta Z , s ) - \widehat { W } ( Z , s ) \Bigr \rangle _ { F } .\tag{4}
$$

Algorithm 1 GradCodeS: Guide–Sample–Evaluate–Select in Quantized Code Space   
1: Input: state $( Z , s ) ;$ codebook C; data $\mathcal { D } ;$ feasible sets $( { \mathcal { Z } } , S ) ;$ step sizes $( \eta _ { Z } , \eta _ { s } )$ ; proposal p;   
candidates $M ;$ iterations $T .$   
2: for $t = 1 , \dots , T$ do   
3: Sample mini-batch $\boldsymbol { B } _ { t } \sim \mathcal { D }$   
4: Update scales by $s \gets \Pi _ { S } ( s - \eta _ { s } \nabla _ { s } \widehat { L } _ { B _ { t } } ( Z , s ) )$   
5: Guide: compute $\nabla _ { Z } \widehat { L } _ { B _ { t } } ( Z , s )$ using (3), then $Z ^ { \mathrm { r e f } }$ using (5)   
6: Sample: sample M code candidates $\{ Z ^ { m } \} _ { m = 1 } ^ { M }$ from $p ( \cdot \mid Z ^ { \mathrm { r e f } } )$   
7: Evaluate: compute $\widehat { L } _ { B _ { t } } ( Z ^ { m } , s )$ for each candidate   
8: Select: Z ← arg min $\mathsf { L } _ { \mathsf { \Lambda } ^ { \prime } \in \{ Z , Z ^ { 1 } , \ldots , Z ^ { M } \} } \widehat { L } _ { { \mathsf { B } } _ { t } } ( \widetilde { Z } , s )$   
9: end for   
10: Return: ${ \widehat W } ( Z , s ) = S ( s ) \odot { \mathcal { C } } ( Z )$

Proposition 1 provides the central discrete connection: by selecting the adjacent gap in the descent direction, the surrogate score exactly matches the first-order deployed-weight change of every feasible gradient-aligned one-level move. Because the network loss is nonlinear, the next result controls the discrepancy from the realized loss.

Corollary 1 (One-step smooth-loss upper bound). Under the assumptions ofProposition 1, $i f L$ is $\beta .$ -smooth as a function of the deployed weight in a neighborhood of $\widehat { W } ( Z , s )$ , then

$$
L ( Z + \Delta Z , s ) \leq L ( Z , s ) + \langle \nabla _ { Z } L ( Z , s ) , \Delta Z \rangle _ { F } + \frac { \beta } { 2 } \left\| \widehat { W } ( Z + \Delta Z , s ) - \widehat { W } ( Z , s ) \right\| _ { F } ^ { 2 } .
$$

Corollary 1 bounds the realized loss change by the surrogate prediction plus a curvature penalty; a move is guaranteed to decrease the loss when its negative surrogate term dominates this penalty. Thus the signal is first-order faithful for feasible local moves, while candidate evaluation accounts for nonlinear and higher-order effects. This bound underlies Proposition $2 ;$ proofs of the present results are provided in Appendix D.

## 4.3 GRADIENT-GUIDED DISCRETE OPTIMIZATION

The results in Section 4.2 show how to rank feasible local code moves. We now turn this signal into a complete procedure that alternates differentiable scale updates with deployment-faithful code search. The surrogate gradient guides where to search, while the deployed loss determines what to accept.

Performing gradient descent with the code surrogate gradient (3) gives a real-valued reference point in code space:

$$
Z ^ { \mathrm { { r e f } } } = Z - \eta _ { Z } \nabla _ { Z } L ( Z , s ) ,\tag{5}
$$

where $\eta _ { Z } > 0$ is the step size. Because $Z ^ { \mathrm { r e f } }$ is generally non-integer, it only centers a proposal over valid codes. The guide–sample–evaluate–select pipeline draws $M \ll | \mathcal { Z } |$ candidates with greater mass near $Z ^ { \mathrm { r e f } }$ and retains the lowest-loss state. Appendix E gives an efficiently sampled coordinate-factorized proposal.

In practice, $\widehat { L } _ { B } ( Z , s )$ denotes the average loss over mini-batch $B ,$ an unbiased estimator of $L ( Z , s )$ Each iteration first updates the differentiable scales by projected gradient descent with Z fixed and then searches the codes at the updated scales. Algorithm 1 summarizes the procedure; Π<sub>S</sub> projects onto the admissible scale space, and selection ties retain the current Z.

The next result isolates the role of gradient guidance in the deterministic full-batch code-search substep; it is conditional on the proposal mass placed near the reference step.

Proposition 2 (Guided best-of-M descent). Fix the scales $s ,$ let $q _ { t }$ be the proposal mass ofgradientaligned one-stepfeasible updates ∆ satisfying $\| \boldsymbol { \Delta } + \eta _ { Z } \nabla _ { Z } L ( Z _ { t } , s ) \| _ { F } \leq c \eta _ { Z } \| \nabla _ { Z } L ( Z _ { t } , s ) \| _ { F } ,$ and suppose that Corollary 1 applies with $\| \widehat { W } ( Z _ { t } + \Delta , s ) - \widehat { W } ( Z _ { t } , s ) \| _ { F } \leq \kappa \| \Delta \| _ { F } . \ H f c \in [ 0 , 1 )$ and $\begin{array} { r } { \alpha : = 1 - c - \frac { \beta \kappa ^ { 2 } \eta _ { Z } } { 2 } ( 1 + c ) ^ { 2 } > 0 , } \end{array}$ , then M independent candidatesfollowed by the selection rule in Algorithm 1 satisfy $\begin{array} { r } { \mathbb { E } [ L ( Z _ { t + 1 } , s ) \mid Z _ { t } ] \leq L ( Z _ { t } , s ) - \alpha \eta _ { Z } \left[ 1 - ( 1 - q _ { t } ) ^ { M } \right] \| \nabla _ { Z } L ( Z _ { t } , s ) \| _ { F } ^ { 2 } } \end{array}$

The factor $1 - ( 1 - q _ { t } ) ^ { M }$ is the probability that at least one candidate lies in the specified nearreference set, showing how proposal concentration and candidate count control expected progress. The proof is provided in Appendix F.

Operational distinction. Continuous or STE-style approaches use a gradient to produce an update that is retained, projected, or requantized, whereas zeroth-order and evolutionary approaches preserve discrete evaluation but rely on sampled losses to determine where to search. GradCodeS separates these roles: a quantization-aware backward signal shapes a proposal over valid codes, while realized discrete losses select the accepted deployable state. Section 5 evaluates the two practical consequences of this separation: deployment-faithful performance and more effective candidate search.

## 5 EXPERIMENTS

We evaluate GradCodeS to address three questions: whether GradCodeS improves performance while remaining in the fully 4-bit deployment space; whether the results are robust across data regimes, code parameterizations, and quantization datatypes; and whether gradient guidance improves candidate quality and end-to-end search efficiency.

## 5.1 EXPERIMENTAL SETTINGS

We evaluate Qwen3-0.6B, Llama-3.2-1B and 3B-Instruct on GSM8K, AlpacaEval, and MASSIVE en-US, covering arithmetic reasoning, instruction following, and structured semantic parsing. We quantize all transformer-block linear layers while leaving the token embedding and LM head unquan tized. Under matched data and evaluation protocols, we compare unquantized references (Base-16 and 16-bit SFT), mixed-precision adapter methods (QLoRA-16A and QA-LoRA), and fully 4-bit baselines (Base-4, QLoRA-4Merge, LoQT, L4Q, PV-Tuning, QuZO, QZO, and QES). We evaluate both full-matrix and low-rank code parameterizations of GradCodeS; complete dataset, metric, quantization, and baseline details are provided in Appendix A. All results are obtained with three independet runs with different random seeds, reported in mean<sub>std</sub>.

## 5.2 MAIN RESULTS

Tables 1 and 2 report results across arithmetic reasoning, instruction following, and structured semantic parsing. Across both backbones and all three tasks, GradCodeS is the strongest fully 4-bit method. Direct 4-bit quantization causes a substantial drop, and the competing fully 4-bit methods recover this loss to varying degrees; GradCodeS provides the most consistent recovery across the reported metrics. We skip relatively weak baselines on 3B model due to computation budget.

The comparison between QLoRA-16A and QLoRA-4Merge isolates the deployment-state mismatch. QLoRA-16A retains a high-precision adapter at inference time, whereas QLoRA-4Merge collapses that adapter into the backbone and requantizes the resulting weights. The degradation after this conversion, also highlighted in Figure 1(b), supports deployment-faithful optimizing rather than relying on an optimize-then-quantize pipeline. Within the fully 4-bit regime, both GradCodeS parameterizations are effective: the LoRA variant is generally strongest, while the full-matrix variant remains competitive. We examine their dependence on training-set size in Section 5.3.

Compared with 16-bit SFT, the best GradCodeS variant obtains higher GSM8K accuracy and stronger MASSIVE results, while the 16-bit reference remains strongest on Qwen AlpacaEval and is marginally higher on Llama AlpacaEval. We interpret the gains with low-bit codes as regularization in Section 5.3, not as a general claim that lower precision is universally superior.

Deployment cost view. Figure 1(b) complements the task tables by plotting Llama-3.2-1B-Instruct GSM8K accuracy across deployment regimes. The comparison is computed on the target quantized modules and excludes modules intentionally left unquantized. GradCodeS attains the best accuracy on the fully 4-bit deployment line, whereas mixed-precision methods retain additional high-precision adapter parameters.

Structured prediction. Table 2 evaluates a more application-like structured prediction setting. GradCodeS gives the best fully 4-bit results on both backbones and all three metrics. Exact Match

Table 1: Results on GSM8K and AlpacaEval. Values are means over three independent runs, with standard deviations shown as subscripts. The second column reports the final deployment regime: 16 denotes unquantized 16-bit deployment, Mixed denotes a 4-bit backbone with high-precision adapter parameters, and 4 denotes fully 4-bit deployment for the quantized modules.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Deployment bits</td><td colspan="2">Qwen3-0.6B</td><td colspan="2">Llama-3.2-1B-Inst.</td><td colspan="2">Llama-3.2-3B-Inst.</td></tr><tr><td>GSM8K Acc.</td><td>Alpaca WR</td><td>GSM8K Acc.</td><td>Alpaca WR</td><td>GSM8K Acc.</td><td>Alpaca WR</td></tr><tr><td>Base-16</td><td>16</td><td> $3 7 . 4 5 _ { 0 . 0 0 }$ </td><td>base</td><td> $3 6 . 3 2 _ { 0 . 0 0 }$ </td><td>base</td><td> $5 6 . 9 4 _ { 0 . 0 0 }$ </td><td>base</td></tr><tr><td>16-bit SFT</td><td>16</td><td> $3 9 . 6 5 _ { 0 . 5 4 }$ </td><td> $\mathbf { 8 8 . 2 6 } _ { 2 . 1 6 }$ </td><td> $3 8 . 9 3 _ { 0 . 6 3 }$ </td><td> ${ \bf 5 6 . 6 4 } _ { 2 . 3 6 }$ </td><td> $\mathbf { 6 7 . 1 0 } _ { 0 . 2 8 }$ </td><td> ${ \bf 7 8 . 5 7 _ { 2 . 8 6 } }$ </td></tr><tr><td>QLoRA-16A QA-LoRA</td><td>Mixed (4.08)</td><td> $3 9 . 5 7 _ { 0 . 4 7 }$ </td><td> $5 4 . 9 0 _ { 2 . 8 7 }$ </td><td> $3 8 . 2 1 _ { 0 . 5 1 }$ </td><td> $5 3 . 8 9 _ { 2 . 7 4 }$ </td><td> $6 5 . 0 5 _ { 0 . 3 9 }$ </td><td> $6 8 . 2 6 _ { 3 . 1 2 }$ </td></tr><tr><td></td><td>Mixed (4.05)</td><td> $3 6 . 1 3 _ { 0 . 6 2 }$ </td><td> $5 3 . 3 4 _ { 3 . 1 2 }$ </td><td> $3 4 . 1 5 _ { 0 . 7 4 }$ </td><td> $4 9 . 2 3 _ { 3 . 1 8 }$ </td><td></td><td></td></tr><tr><td>Base-4</td><td>4</td><td> $2 7 . 1 4 _ { 0 . 0 0 }$ </td><td> $2 6 . 5 8 _ { 1 . 9 4 }$ </td><td> $2 5 . 8 5 _ { 0 . 0 0 }$ </td><td> $4 6 . 1 6 _ { 2 . 4 2 }$ </td><td> $4 9 . 1 3 _ { 0 . 0 0 }$ </td><td> $3 5 . 9 4 _ { 2 . 6 8 }$ </td></tr><tr><td>QLoRA-4Merge</td><td>4</td><td> $2 6 . 4 6 _ { 0 . 7 1 }$ </td><td> $3 8 . 3 5 _ { 2 . 7 6 }$ </td><td> $2 4 . 7 9 \mathrm { _ { 0 . 6 6 } }$ </td><td> $3 3 . 6 8 _ { 3 . 5 5 }$ </td><td> $4 9 . 1 3 _ { 0 . 5 2 }$ </td><td> $3 6 . 2 0 _ { 3 . 0 7 }$ </td></tr><tr><td>LoQT</td><td>4</td><td> $3 0 . 9 4 _ { 0 . 5 8 }$ </td><td> $4 4 . 6 7 _ { 2 . 6 3 }$ </td><td> $2 9 . 0 4 _ { 0 . 6 1 }$ </td><td> $4 8 . 8 5 _ { 2 . 8 1 }$ </td><td></td><td></td></tr><tr><td>L4Q</td><td>4</td><td> $2 9 . 8 9 _ { 0 . 6 7 }$ </td><td> $4 0 . 2 8 _ { 2 . 9 1 }$ </td><td> $3 1 . 0 6 _ { 0 . 5 7 }$ </td><td> $5 1 . 2 1 _ { 2 . 6 6 }$ </td><td></td><td></td></tr><tr><td>PV-Tuning</td><td>4</td><td> $3 8 . 7 0 _ { 0 . 4 9 }$ </td><td> $7 0 . 5 1 _ { 3 . 2 1 }$ </td><td> $3 6 . 9 2 _ { 0 . 4 5 }$ </td><td> $5 2 . 6 3 _ { 1 . 8 4 }$ </td><td> $6 3 . 0 8 _ { 0 . 4 3 }$ </td><td> $6 2 . 3 6 _ { 2 . 7 9 }$ </td></tr><tr><td>QuZO</td><td>4</td><td> $2 9 . 4 5 _ { 0 . 8 4 }$ </td><td> $3 4 . 3 5 _ { 3 . 4 7 }$ </td><td> $2 6 . 5 4 _ { 0 . 7 8 }$ </td><td> $4 1 . 5 7 _ { 3 . 2 6 }$ </td><td></td><td></td></tr><tr><td>QZO QES</td><td>4</td><td> $2 9 . 7 6 _ { 0 . 7 6 }$ </td><td> $3 8 . 9 9 _ { 3 . 1 4 }$ </td><td> $2 7 . 6 0 _ { 0 . 8 1 }$ </td><td> $3 9 . 3 9 _ { 3 . 4 1 }$ </td><td></td><td></td></tr><tr><td></td><td>4</td><td> $2 8 . 8 0 _ { 0 . 9 2 }$ </td><td> $4 1 . 1 3 _ { 3 . 3 6 }$ </td><td> $2 6 . 3 9 _ { 0 . 8 5 }$ </td><td> $3 8 . 1 2 _ { 3 . 6 7 }$ </td><td></td><td></td></tr><tr><td>GradCodeS (LoRA)</td><td>4</td><td> $\mathbf { 4 5 . 7 2 } _ { 0 . 4 2 }$ </td><td> $8 2 . 2 4 _ { 2 . 5 7 }$ </td><td> $\underline { { 4 0 . 5 2 } } _ { 0 . 4 7 }$ </td><td> $5 4 . 6 6 _ { 1 . 8 3 }$ </td><td> $\underline { { 6 7 . 0 8 } } _ { 0 . 3 1 }$ </td><td> $6 9 . 5 2 _ { 3 . 2 6 }$ </td></tr><tr><td>GradCodeS (Full)</td><td>4</td><td> $\underline { { 4 3 . 6 8 } } _ { 0 . 3 8 }$ </td><td> $\underline { { 8 3 . 0 9 } } _ { 2 . 4 8 }$ </td><td> $\mathbf { 4 1 . 6 3 } _ { 0 . 5 8 }$ </td><td> $\underline { { 5 6 . 3 4 } } _ { 2 . 3 1 }$ </td><td> $6 6 . 3 9 _ { 0 . 3 5 }$ </td><td> $\underline { { 7 2 . 0 4 } } _ { 3 . 1 4 }$ </td></tr></table>

requires the entire semantic frame to be correct, while Slot F1 measures the quality of the predicted slot structure. The gains on both metrics indicate that deployment-faithful code updates preserve structured output behavior after quantization.

Table 2: MASSIVE en-US test results. Values are means over three independent runs, with standard deviations shown as subscripts. Models are fine-tuned on the English-US training split and evaluated on the English-US test split. EM denotes exact semantic-frame match.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Deployment bits</td><td colspan="3">Qwen3-0.6B</td><td colspan="3">Llama-3.2-1B-Inst</td><td colspan="3">Llama-3.2-3B-Inst.</td></tr><tr><td>EM</td><td>Intent</td><td>Slot F1</td><td>EM</td><td>Intent</td><td>Slot F1</td><td>EM</td><td>Intent</td><td>Slot F1</td></tr><tr><td>Base-16</td><td>16</td><td> $0 . 0 0 _ { 0 . 0 0 }$ </td><td> $0 . 0 0 _ { 0 . 0 0 }$ </td><td> $3 . 3 8 _ { 0 . 0 0 }$ </td><td> $0 . 0 0 _ { 0 . 0 0 }$ </td><td> $0 . 1 3 _ { 0 . 0 0 }$ </td><td> $1 . 3 9 _ { 0 . 0 0 }$ </td><td>0.070.00</td><td> $3 . 0 3 _ { 0 . 0 0 }$ </td><td> $3 . 6 3 _ { 0 . 0 0 }$ </td></tr><tr><td>16-bit SFT</td><td>16</td><td> $5 7 . 9 2 _ { 0 . 3 6 }$ </td><td> $8 3 . 9 5 _ { 0 . 2 2 }$ </td><td> $6 5 . 3 1 _ { 0 . 4 4 }$ </td><td> $6 3 . 0 6 _ { 0 . 4 1 }$ </td><td> $8 5 . 5 7 _ { 0 . 2 9 }$ </td><td> $7 3 . 3 9 _ { 0 . 3 8 }$ </td><td> $6 6 . 1 6 _ { 0 . 3 3 }$ </td><td> $\underline { { 8 8 . 9 7 } } _ { 0 . 2 4 }$ </td><td> $7 5 . 2 1 _ { 0 . 3 5 }$ </td></tr><tr><td>QLoRA-16A</td><td>Mixed (4.08)</td><td> $5 7 . 1 6 _ { 0 . 4 2 }$ </td><td> $8 3 . 4 9 _ { 0 . 3 1 }$ </td><td> $6 6 . 7 4 _ { 0 . 4 8 }$ </td><td> $6 3 . 3 9 _ { 0 . 3 7 }$ </td><td> $8 5 . 8 1 _ { 0 . 2 6 }$ </td><td> $7 4 . 1 3 _ { 0 . 4 3 }$ </td><td> $6 5 . 2 3 _ { 0 . 3 9 }$ </td><td> $8 8 . 0 6 _ { 0 . 2 8 }$ </td><td> $7 5 . 1 4 _ { 0 . 4 1 }$ </td></tr><tr><td>QA-LoRA</td><td>Mixed (4.05)</td><td> $5 0 . 1 7 _ { 0 . 5 8 }$ </td><td> $8 0 . 7 7 _ { 0 . 4 7 }$ </td><td> $6 1 . 1 9 _ { 0 . 6 2 }$ </td><td> $5 4 . 1 0 _ { 0 . 5 5 }$ </td><td> $8 4 . 3 0 _ { 0 . 3 5 }$ </td><td> $6 5 . 4 1 _ { 0 . 6 6 }$ </td><td></td><td></td><td></td></tr><tr><td>Base-4</td><td>4</td><td> $0 . 0 0 _ { 0 . 0 0 }$ </td><td>0.610.00</td><td> $3 . 9 7 _ { 0 . 0 0 }$ </td><td> $0 . 0 0 _ { 0 . 0 0 }$ </td><td>0.030.00</td><td> $0 . 6 2 _ { 0 . 0 0 }$ </td><td> $0 . 0 0 _ { 0 . 0 0 }$ </td><td> $1 . 9 5 _ { 0 . 0 0 }$ </td><td> $3 . 3 2 _ { 0 . 0 0 }$ </td></tr><tr><td>QLoRA-4Merge</td><td>4</td><td> $0 . 0 0 _ { 0 . 0 0 }$ </td><td> $0 . 6 7 _ { 0 . 1 1 }$ </td><td> $3 . 6 1 _ { 0 . 1 8 }$ </td><td> $0 . 0 0 _ { 0 . 0 0 }$ </td><td> $0 . 0 3 _ { 0 . 0 2 }$ </td><td> $0 . 4 8 _ { 0 . 1 4 }$ </td><td> $0 . 0 0 _ { 0 . 0 0 }$ </td><td>1.950.13</td><td> $3 . 2 9 _ { 0 . 2 1 }$ </td></tr><tr><td>LoQT</td><td>4</td><td> $5 1 . 4 2 _ { 0 . 6 3 }$ </td><td> $7 2 . 0 9 _ { 0 . 5 1 }$ </td><td> $4 5 . 8 3 _ { 0 . 7 8 }$ </td><td> $5 2 . 1 4 _ { 0 . 5 9 }$ </td><td> $7 5 . 2 7 _ { 0 . 4 6 }$ </td><td> $4 9 . 9 0 _ { 0 . 8 2 }$ </td><td></td><td></td><td></td></tr><tr><td>L4Q</td><td>4</td><td> $4 8 . 2 7 _ { 0 . 7 1 }$ </td><td> $7 2 . 0 3 _ { 0 . 5 6 }$ </td><td> $4 6 . 0 2 _ { 0 . 7 4 }$ </td><td> $5 3 . 4 3 _ { 0 . 6 4 }$ </td><td> $7 5 . 8 8 _ { 0 . 4 9 }$ </td><td> $5 8 . 6 2 _ { 0 . 6 9 }$ </td><td></td><td></td><td></td></tr><tr><td>PV-Tuning</td><td>4</td><td> $5 8 . 7 1 _ { 0 . 4 5 }$ </td><td> $8 2 . 9 2 _ { 0 . 3 7 }$ </td><td> $6 6 . 2 9 _ { 0 . 5 3 }$ </td><td> $6 4 . 5 0 _ { 0 . 3 8 }$ </td><td> $8 6 . 3 1 _ { 0 . 3 1 }$ </td><td> $7 4 . 8 8 _ { 0 . 4 6 }$ </td><td> $6 5 . 8 9 _ { 0 . 3 5 }$ </td><td>86.950.27</td><td>76.200.42</td></tr><tr><td>QuZO</td><td>4</td><td> $3 1 . 2 6 _ { 0 . 8 6 }$ </td><td> $5 7 . 3 0 _ { 0 . 7 2 }$ </td><td> $3 8 . 5 5 _ { 0 . 9 1 }$ </td><td> $4 6 . 3 5 _ { 0 . 7 7 }$ </td><td> $6 4 . 1 7 _ { 0 . 6 8 }$ </td><td> $4 6 . 8 0 _ { 0 . 8 8 }$ </td><td></td><td></td><td></td></tr><tr><td>QZO</td><td>4</td><td> $3 0 . 0 6 _ { 0 . 7 9 }$ </td><td> $5 9 . 7 9 _ { 0 . 6 4 }$ </td><td> $4 2 . 5 0 _ { 0 . 8 3 }$ </td><td> $4 5 . 1 8 _ { 0 . 8 1 }$ </td><td> $6 2 . 5 0 _ { 0 . 7 4 }$ </td><td> $4 1 . 3 3 _ { 0 . 9 5 }$ </td><td></td><td></td><td></td></tr><tr><td>QES</td><td>4</td><td> $2 9 . 2 5 _ { 0 . 9 3 }$ </td><td> $5 2 . 5 3 _ { 0 . 8 1 }$ </td><td> $3 4 . 4 1 _ { 1 . 0 2 }$ </td><td> $4 3 . 5 6 _ { 0 . 8 5 }$ </td><td> $5 8 . 3 5 _ { 0 . 7 9 }$ </td><td> $3 9 . 8 1 _ { 0 . 9 7 }$ </td><td></td><td></td><td></td></tr><tr><td>GradCodeS (LoRA)</td><td>4</td><td> $\underline { { 6 5 . 8 8 } } _ { 0 . 3 4 }$ </td><td> $\underline { { 8 5 . 5 3 } } _ { 0 . 2 8 }$ </td><td> $\mathbf { 7 6 . 6 0 } _ { 0 . 3 9 }$ </td><td> $\underline { { 6 7 . 3 7 } } _ { 0 . 3 1 }$ </td><td> $8 6 . 8 9 _ { 0 . 3 3 }$ </td><td> $\underline { { 7 6 . 2 5 } } _ { 0 . 3 7 }$ </td><td> $\mathbf { 7 0 . 4 8 } _ { 0 . 2 7 }$ </td><td> $\mathbf { 9 0 . 9 7 } _ { 0 . 1 9 }$ </td><td> $\mathbf { 7 9 . 8 5 } _ { 0 . 3 1 }$ </td></tr><tr><td>GradCodeS (Full)</td><td>4</td><td> ${ \bf 6 6 . 6 1 } _ { 0 . 3 2 }$ </td><td>86.850.24</td><td>75.880.41</td><td>69.170.28</td><td>88.130.21</td><td> $7 8 . 7 4 _ { 0 . 3 4 }$ </td><td> $\underline { { 6 7 . 2 6 } } _ { 0 . 2 9 }$ </td><td> $8 7 . 5 3 _ { 0 . 2 5 }$ </td><td> $\underline { { 7 8 . 2 4 } } _ { 0 . 3 6 }$ </td></tr></table>

## 5.3 ROBUSTNESS ACROSS DATA, PARAMETERIZATIONS, AND DATATYPES

Data size and code parameterization. The comparison with 16-bit SFT suggests that the restricted code-space parameterization may provide useful regularization when fine-tuning data are limited. We treat this as an empirical hypothesis rather than a consequence of the local optimization theory. Table 3 examines the hypothesis by comparing the original 7.0k-example GSM8K training set with a 167k-example training set formed by adding 160k MetaMathQA examples (Yu et al., 2024).

With 7.0k examples, GradCodeS (LoRA) performs best; with 167k examples, 16-bit SFT becomes strongest. This result is consistent with, but does not prove, a regularization benefit from the low-rank code parameterization in the smaller-data regime.

Table 3: GSM8K test accuracy (%) across training-set sizes and parameterizations on Llama-3.2-1B-Instruct. Values are means over three independent runs, with standard deviations shown as subscripts.
<table><tr><td>Method</td><td>Parameterization</td><td>7.0k examples</td><td>167k examples</td></tr><tr><td>16-bit SFT</td><td>LoRA (rank 8)</td><td> ${ \underline { { 3 8 . 9 } } } _ { 0 . 6 }$ </td><td> $\mathbf { 5 5 . 9 } _ { 0 . 7 }$ </td></tr><tr><td rowspan="2">GradCodeS</td><td>LoRA (rank 8)</td><td> $4 0 . 5 _ { 0 . 6 }$ </td><td> $5 4 . 3 _ { 0 . 6 }$ </td></tr><tr><td>Full matrix</td><td> $\mathbf { 4 1 . 6 } _ { 0 . 5 }$ </td><td> $\underline { { 5 4 . 5 } } _ { 0 . 5 }$ </td></tr></table>

Quantization datatype. Based on the unified datatype-specific recovery map in (1), GradCodeS does not rely on a single quantization datatype. Different datatypes change the codebook C and scale grouping, while the search rule requires only ordered code levels and the local effective-step map in Definition 1. Table 4 shows that GradCodeS can be applied to NF4, INT4, and MXFP4. NF4 and MXFP4 are stronger than INT4 in this setting, and all three variants remain competitive with the adaptive PV-Tuning baseline. These results establish compatibility across the tested datatype-specific geometries without attributing the performance differences to any single codebook property.

Table 4: Results of GradCodeS across quantization datatypes on Llama-3.2-1B-Instruct. Values are means over three independent runs, with standard deviations shown as subscripts.
<table><tr><td>Method</td><td>Datatype</td><td>GSM8K ACC.</td><td>AlpacaEval WR.</td><td>MASSIVE EM.</td></tr><tr><td rowspan="3">GradCodeS</td><td>NF4</td><td> $\mathbf { 4 1 . 6 } _ { 0 . 5 }$ </td><td> $5 6 . 3 _ { 1 . 8 }$ </td><td> $\mathbf { 6 9 . 2 } _ { 0 . 3 }$ </td></tr><tr><td>INT4</td><td> $3 8 . 9 _ { 0 . 6 }$ </td><td> $5 2 . 0 _ { 2 . 4 }$ </td><td> $6 5 . 1 _ { 0 . 5 }$ </td></tr><tr><td>MXFP4</td><td> $4 1 . 1 _ { 0 . 4 }$ </td><td> $\mathbf { 5 7 . 1 } _ { 2 . 2 }$ </td><td> $6 8 . 8 _ { 0 . 4 }$ </td></tr><tr><td>PV-Tuning</td><td>Adaptive</td><td> $3 6 . 9 _ { 0 . 5 }$ </td><td> $5 2 . 6 _ { 1 . 8 }$ </td><td> $6 4 . 5 _ { 0 . 4 }$ </td></tr></table>

## 5.4 SEARCH EFFICIENCY AND COMPONENT ANALYSIS

End-to-end efficiency. Figure 3(a) compares wall-clock time against the best test accuracy reached so far. GradCodeS is not the cheapest possible step: it uses one backward pass to build the guided reference and then evaluates multiple discrete candidates. The benefit is that these evaluations are targeted. The baselines improve quickly but plateau below the final GradCodeS accuracy, while GradCodeS reaches a stronger result within the displayed time range. This supports the intended trade-off of spending one gradient computation to reduce wasted discrete search.

![](images/a6b98925aca78a62d03e887a602cdb9c8e09fb2d30fe4861127dbc6e8db5582a.jpg)  
(a) Wall-clock time versus the best test accuracy reached so far, evaluated once per epoch.

![](images/b1147a221e55b7e5f80b91dbbae71d3638f6bf723cd36dd742a05d455e7d1de4.jpg)  
(b) Candidate loss versus distance from the guided reference in one sampling step.  
Figure 3: Training efficiency and candidate-set quality for Llama-3.2-1B-Instruct on GSM8K.

Candidate-set quality. Figure 3(b) gives a candidate-level view of one illustrative sampling step. Candidate loss tends to be lower near the guided reference, but the relationship is imperfect. This is why GradCodeS does not simply project to the nearest code point: it keeps several nearby candidates reachable and lets their realized discrete losses determine the accepted update.

Table 5 reports the empirical selection probability and selected mean loss under the same candidate budget. The guided sampler is selected substantially more often than Gaussian or one-hop sampling, indicating that gradient guidance improves candidate-set quality rather than merely increasing the number of forward evaluations. And the proposed code surrogate gradient $( \nabla _ { Z } L )$ -guided sample better candidates than mapped ordinary weight gra-

Table 5: Candidate-selection statistics for different sampling distributions under the same candidate budget. Values are means over three independent runs, with standard deviations shown as subscripts.
<table><tr><td>Sampling</td><td>Sel. Prob.</td><td>Sel. Mean Loss</td></tr><tr><td>Gaussian</td><td>5.83%</td><td>0.2955</td></tr><tr><td>1-Hop Neighbor</td><td>38.16%</td><td>0.2963</td></tr><tr><td> $\nabla _ { \hat { W } } \bar { L }$  -Guided</td><td>48.62%</td><td>0.2951</td></tr><tr><td> $\nabla _ { Z } ^ { \cdot \cdot } L$  -Guided (Ours)</td><td>67.97%</td><td>0.2949</td></tr></table>

dient $( \nabla _ { \hat { W } } L ) { \mathrm { - g u i d e d } }$ , indicating it is a more effective first order signal. Together, Figure 3(b) and Table 5 are consistent with the proposal-guidance role analyzed in Proposition 2, without directly testing its candidate-count dependence.

Component ablations. We ablate three components of the default GradCodeS configuration: gradient guidance, code-index updates, and scale updates. The w/o guide variant samples uniformly from a neighborhood of the current code without conditioning on (5). The $\nabla _ { \boldsymbol { \hat { W } } } L$ -guide replace code surrogate gradient with mapped ordinary weight gradient. The w/o scale update variant freezes group-wise scales and optimizes only codes, while the w/o code update variant freezes the indices and optimizes only group-wise scales.

![](images/4f6ccc697edda13aedff33e4628578e64b8723bafe2d395cacd9b069fa244524.jpg)  
Figure 4: Component ablations for GradCodeS (LoRA) on Llama-3.2-1B-Instruct.

Figure 4 shows that all three components contribute. Both removing gradient guidance and replacing $\nabla _ { Z } L$ signal with $\nabla _ { \hat { W } } L$ consistently reduce performance, supporting the use of the code surrogate gradient to shape the proposal. Freezing either codes or scales causes larger task-dependent drops, indicating that both variables are important to the complete optimization procedure.

## 6 CONCLUSION

We presented GradCodeS, a gradient-guided code-search method for fine-tuning low-bit model. The key idea is to analyze quantization geometry to find the steepest descent direction in code space, and use such first-order information to shape a candidate-sampling distribution, while letting the realized discrete loss decide which state to select to be deployment-faithful. Across reasoning, instructionfollowing, and structured language-understanding tasks, GradCodeS consistently improves fully 4-bit adaptation and remains compatible with different codebooks, low-rank code parameterizations, and alternating scale updates. The local analysis further clarifies why guided best-of-M code search can dominate both blind discrete search and single-point projected updates. An important next step is to scale the method to larger backbones and richer structured candidate-sampling families while preserving the same deployment-faithful update principle.

## REFERENCES

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient finetuning of quantized LLMs. In Advances in Neural Information Processing Systems, volume 36, pp. 10088–10115, 2023.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate post-training quantization for generative pre-trained transformers. In International Conference on Learning Representations, 2023.

Will Grathwohl, Kevin Swersky, Milad Hashemi, David Duvenaud, and Chris J. Maddison. Oops i took a gradient: Scalable sampling for discrete distributions. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139, pp. 3831–3841, 2021.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Hyesung Jeon, Yulhwa Kim, and Jae-Joon Kim. L4Q: Parameter efficient quantization-aware finetuning on large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2002–2024, 2025.

Muheng Li and Ruqi Zhang. Reheated gradient-based discrete sampling for combinatorial optimization. Transactions on Machine Learning Research, 2025.

Yixiao Li, Yifan Yu, Chen Liang, Nikos Karampatziakis, Pengcheng He, Weizhu Chen, and Tuo Zhao. LoftQ: LoRA-fine-tuning-aware quantization for large language models. In International Conference on Learning Representations, 2024.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-aware weight quantization for LLM compression and acceleration. In Proceedings of Machine Learning and Systems, volume 6, 2024.

Meng Liu, Haoran Liu, and Shuiwang Ji. Gradient-guided importance sampling for learning binary energy-based models. In International Conference on Learning Representations, 2023.

Sebastian Loeschcke, Mads Toftrup, Michael J. Kastoryano, Serge Belongie, and Vésteinn Snæbjarnarson. LoQT: Low-rank adapters for quantized pretraining. In Advances in Neural Information Processing Systems, volume 37, pp. 115282–115308, 2024.

Niru Maheswaranathan, Luke Metz, George Tucker, Dami Choi, and Jascha Sohl-Dickstein. Guided evolutionary strategies: Augmenting random search with surrogate gradients. In Proceedings of the 36th International Conference on Machine Learning, volume 97, pp. 4264–4273, 2019.

Vladimir Malinovskii, Denis Mazur, Ivan Ilin, Denis Kuznedelev, Konstantin Burlachenko, Kai Yi, Dan Alistarh, and Peter Richtárik. PV-Tuning: Beyond straight-through estimation for extreme LLM compression. In Advances in Neural Information Processing Systems, volume 37, pp. 5074–5121, 2024.

Sifeng Shang, Jiayi Zhou, Chenyu Lin, Minxian Li, and Kaiyang Zhou. Fine-tuning quantized neural networks with zeroth-order optimization. In International Conference on Learning Representations, 2026.

Haoran Sun, Hanjun Dai, Bo Dai, Haomin Zhou, and Dale Schuurmans. Discrete langevin samplers via wasserstein gradient flow. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, volume 206, pp. 6290–6313, 2023a.

Haoran Sun, Katayoon Goshvadi, Azade Nova, Dale Schuurmans, and Hanjun Dai. Revisiting sampling for combinatorial optimization. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202, pp. 32859–32874, 2023b.

Yue Xiang, Dongyao Zhu, Bowen Lei, Dongkuan Xu, and Ruqi Zhang. Efficient informed proposals for discrete distributions via Newton’s series approximation. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, volume 206, pp. 7288–7310, 2023.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and efficient post-training quantization for large language models. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202, 2023.

Yinggan Xu, Kajetan Schweighofer, Risto Miikkulainen, and Xin Qiu. Quantized evolution strategies: High-precision fine-tuning of quantized LLMs at low-precision cost. arXiv preprint arXiv:2602.03120, 2026.

Yuhui Xu, Lingxi Xie, Xiaotao Gu, Xin Chen, Heng Chang, Hengheng Zhang, Zhengsu Chen, Xiaopeng Zhang, and Qi Tian. QA-LoRA: Quantization-aware low-rank adaptation of large language models. In International Conference on Learning Representations, 2024.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. MetaMath: Bootstrap your own mathematical questions for large language models. In International Conference on Learning Representations, 2024.

Ruqi Zhang, Xingchao Liu, and Qiang Liu. A Langevin-like sampler for discrete distributions. In Proceedings of the 39th International Conference on Machine Learning, volume 162, pp. 26375–26396, 2022.

Jiajun Zhou, Yifan Yang, Kai Zhen, Ziyue Liu, Yequan Zhao, Ershad Banijamali, Athanasios Mouchtaris, Ngai Wong, and Zheng Zhang. QuZO: Quantized zeroth-order fine-tuning for large language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 5341–5359, 2025.

## A EXPERIMENTAL DETAILS

## A.1 MODELS AND EVALUATION TASKS

Models. We evaluate three instruction-tuned decoder-only language models, Qwen3-0.6B and Llama-3.2-1B and 3B-Instruct. Compact backbones are a useful stress test for fully low-bit adaptation: they are realistic for mobile, edge, and other memory-constrained deployment scenarios, and they make it easier to attribute performance differences to the optimization method rather than to very large overparameterized capacity. For each backbone, we compare the original 16-bit checkpoint, the directly quantized 4-bit checkpoint, mixed-precision adapter baselines, and fully 4-bit fine-tuned variants under the same data and evaluation protocol.

Datasets and metrics. We evaluate on three datasets covering arithmetic reasoning, general instruction following, and mobile-style language understanding. For GSM8K, models are fine-tuned on the official training split and evaluated on the official test split; we report answer accuracy after extracting the final numerical answer. For Alpaca, models are fine-tuned on Alpaca Cleaned and evaluated with AlpacaEval; we report win rate under the same evaluator for all methods. For MASSIVE en-US, models are fine-tuned on the English-US training split and evaluated on the English-US test split. We formulate MASSIVE as structured semantic parsing and report Exact Match, Intent Accuracy, and Slot F1.

## A.2 QUANTIZATION PROTOCOL

Following the block-wise setup in Section 3, we quantize all linear parameters in the transformer blocks, including the attention and MLP projection matrices in every layer, while leaving the token embedding and LM head unquantized, following common LLM quantization practice (Dettmers et al., 2023; Frantar et al., 2023; Lin et al., 2024). We use the same target modules, quantization format, training data, and evaluation scripts whenever applicable. The reported 4-bit scores reflect the actually deployable checkpoint rather than the relaxed training state.

## A.3 BASELINES

We compare with three groups of baselines. A method is considered fully low-bit deployable only if inference does not require high-precision adapters or residual branches for the quantized modules. A method is considered mixed if it uses a quantized backbone but keeps high-precision inference-time adapter parameters. We also include unquantized references to separate adaptation quality from deployment cost.

Unquantized references.

• Base-16 evaluates the original 16-bit checkpoint without task fine-tuning.

• 16-bit SFT fine-tunes 16-bit LoRA adapters on the 16-bit backbone and serves as a high-precision parameter-efficient adaptation reference (Hu et al., 2022).

Mixed-precision baselines.

• QLoRA-16A freezes a 4-bit backbone and fine-tunes a 16-bit LoRA adapter, giving a strong low-memory training baseline whose deployed model still contains high-precision adapter weights (Dettmers et al., 2023).

• QA-LoRA uses quantization-aware group-wise low-rank adapters to improve compatibility between LoRA adaptation and quantized deployment, but optimizes continuous adapter variables (Xu et al., 2024).

Fully 4-bit baselines.

• Base-4 directly evaluates the quantized 4-bit checkpoint before adaptation.

• QLoRA-4Merge follows QLoRA training, then merges the learned adapter into the backbone and requantizes the merged weights to 4-bit for inference (Dettmers et al., 2023).

• LoQT trains low-rank adapters for quantized weights and periodically merges them into quantized full-rank weights; we evaluate its final merged-and-requantized checkpoint (Loeschcke et al., 2024).

• L4Q is a representative projection/STE-style low-bit fine-tuning baseline (Jeon et al., 2025).

• PV-Tuning optimizes a compressed representation and alternates quantized variables with auxiliary continuous quantities (Malinovskii et al., 2024).

• QuZO, QZO, and QES represent deployment-faithful discrete or zeroth-order low-bit optimization baselines (Zhou et al., 2025; Shang et al., 2026; Xu et al., 2026).

Our method, GradCodeS, searches over low-bit code states directly. Every selected update is a deployable low-bit state, so the final model does not require a 16-bit adapter or high-precision residual branch at inference time. We evaluate both the full-matrix code update, GradCodeS (Full), and the low-rank code parameterization from Appendix G, GradCodeS (LoRA).

## B DATATYPE-SPECIFIC INSTANTIATIONS OF THE UNIFIED QUANTIZATION MODEL

The common update rule depends on each datatype only through its ordered codebook, scale grouping, and recovery map. Table 6 summarizes these components for the evaluated datatypes.

<table><tr><td>Datatype</td><td> $b _ { q }$ </td><td> $K _ { q }$ </td><td>Codebook</td><td>Scale/group structure</td></tr><tr><td>NF4</td><td>4</td><td>16</td><td>Non-uniform</td><td>Group-wise scaling</td></tr><tr><td>INT4</td><td>4</td><td>16</td><td>Uniform</td><td>Group-wise scaling</td></tr><tr><td>MXFP4</td><td>4</td><td>16</td><td>E2M1-style</td><td>Block/microscaling</td></tr></table>

Table 6: Datatype-specific components used by the unified quantization representation.

For datatype q, let $\mathcal { C } _ { q } = \{ c _ { 1 } ^ { ( q ) } , \ldots , c _ { K _ { q } } ^ { ( q ) } \}$ , let $g _ { q } ( i , j )$ denote the scale group of entry $( i , j )$ , and let $S _ { q } ( s ) = [ s _ { g _ { q } ( i , j ) } ] _ { i j }$ . The deployed weight is

$$
\widehat { W } _ { q } ( Z , s ) = S _ { q } ( s ) \odot { \mathcal C } _ { q } ( Z ) ,\tag{6}
$$

with datatype-specific code and scale constraints. A feasible code move changes an entry by

$$
\Delta \widehat { W } _ { q , i j } = s _ { g _ { q } ( i , j ) } \Big ( c _ { Z _ { i j } + \Delta Z _ { i j } } ^ { ( q ) } - c _ { Z _ { i j } } ^ { ( q ) } \Big ) .
$$

Thus the effective step automatically incorporates the datatype-specific codebook gap and scale. Here datatype-agnostic refers to the common optimization rule, not to identical feasible spaces; every candidate is still evaluated through the appropriate recovery map in (6).

## C DERIVATION OF THE CODE SURROGATE GRADIENT

The chain rule would suggest $\nabla _ { Z } L = \nabla _ { \widehat { W } } L \odot \nabla _ { Z } \widehat { W }$ , but the codebook lookup has no standard derivative. For an actual code-coordinate move $\Delta Z \in \mathbb { Z } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ , the deployed-weight change is exactly

$$
\widehat { W } ( \mathrm { c l i p } ( Z + \Delta Z , 1 , K ) , s ) - \widehat { W } ( Z , s ) = S ( s ) \odot \Big ( \mathcal { C } ( \mathrm { c l i p } ( Z + \Delta Z , 1 , K ) ) - \mathcal { C } ( Z ) \Big ) .
$$

We therefore extend $\mathcal { C } ( \cdot )$ piecewise linearly around the current code, using the adjacent gaps $G ^ { + } ( Z )$ and $G ^ { - } ( Z )$ from Section 4.1 as directional slopes. For $\| \Delta Z \| _ { \infty } \le 1$ , this continuation satisfies

$$
\widehat { W } ( \mathrm { c l i p } ( Z + \Delta Z , 1 , K ) , s ) - \widehat { W } ( Z , s ) = S ( s ) \odot \Big ( G ^ { + } ( Z ) \odot ( \Delta Z ) _ { + } - G ^ { - } ( Z ) \odot ( \Delta Z ) _ { - } \Big ) .
$$

For a descent-aligned move, a negative weight-gradient entry calls for a forward code move, whereas a positive entry calls for a backward code move. Writing $\sigma = \mathrm { s i g n } ( \nabla _ { \widehat { W } } L )$ , the corresponding nonnegative directional slope is

$$
S ( s ) \odot \Bigl ( G ^ { + } ( Z ) \odot \sigma _ { - } + G ^ { - } ( Z ) \odot \sigma _ { + } \Bigr ) ,
$$

which is precisely the effective step $D \big ( Z , s , \nabla _ { \widehat { W } } L \big )$ in Definition 1. Applying the chain rule to this local continuation yields

$$
\nabla _ { Z } L ( Z , s ) = \nabla _ { \widehat { W } } L ( Z , s ) \odot D \big ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \big ) ,
$$

as stated in (3). For a uniform codebook and interior coordinates, the adjacent gaps coincide, so D reduces to the quantization gap multiplied by $S ( s )$

## D PROOFS FOR SECTION 4.2

Restatement of Lemma 1. Define the local directional surrogate function of $L ( Z , s )$ by extending it to real-valued interpolation $\Delta Z \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ by

$$
\bar { L } _ { Z , s } ( \Delta Z ) = L \Bigl ( \widehat { W } ( Z , s ) + D \bigl ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \bigr ) \odot \Delta Z \Bigr ) .
$$

Then

$$
\nabla _ { \Delta Z } \bar { L } _ { Z , s } ( \Delta Z ) \big | _ { \Delta Z = 0 } = \nabla _ { Z } L ( Z , s ) .
$$

Consequently, whenever $\nabla _ { Z } L ( Z , s ) \neq 0$ , the direction $- \nabla _ { Z } L ( Z , s )$ is the steepest descent direction of $\bar { L } _ { Z , \varepsilon }$ <sub>s</sub> at $\Delta Z = 0$ under the Frobenius norm.

Proof of Lemma 1. By definition,

$$
\bar { L } _ { Z , s } ( \Delta Z ) = L \Bigl ( \widehat { W } ( Z , s ) + D \bigl ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \bigr ) \odot \Delta Z \Bigr ) .
$$

Since ${ \cal D } \big ( Z , s , \nabla _ { \widehat { \cal W } } L ( Z , s ) \big )$ is fixed at the current point, the map

$$
\Delta Z \mapsto \widehat { W } ( Z , s ) + D \big ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \big ) \odot \Delta Z
$$

is affine, and its differential in direction H is

$$
D \big ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \big ) \odot H .
$$

Therefore, by the chain rule,

$$
\nabla _ { \Delta Z } \bar { L } _ { Z , s } ( \Delta Z ) = \nabla _ { \widehat { W } } L \Bigl ( \widehat { W } ( Z , s ) + D \bigl ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \bigr ) \odot \Delta Z \Bigr ) \odot D \bigl ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \bigr ) .
$$

Evaluating at $\Delta Z = 0$ gives

$$
\begin{array} { r l } & { \nabla _ { \Delta Z } \widehat { L } _ { Z , s } ( \Delta Z ) \big | _ { \Delta Z = 0 } } \\ & { = \nabla _ { \widehat { W } } L \big ( \widehat { W } ( Z , s ) \big ) \odot D \big ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \big ) } \\ & { = \nabla _ { \widehat { W } } L ( Z , s ) \odot D \big ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \big ) = \nabla _ { Z } L ( Z , s ) , } \end{array}
$$

which proves the exact-gradient claim.

For the steepest-descent statement, consider the first-order model

$$
m ( \Delta Z ) = \bar { L } _ { Z , s } ( 0 ) + \langle \nabla _ { Z } L ( Z , s ) , \Delta Z \rangle _ { F } .
$$

For any $\Delta Z$ with $\| \Delta Z \| _ { F } \leq r ,$ , Cauchy–Schwarz gives

$$
\begin{array} { r } { \langle \nabla _ { Z } L ( Z , s ) , \Delta Z \rangle _ { F } \geq - \| \nabla _ { Z } L ( Z , s ) \| _ { F } \left\| \Delta Z \right\| _ { F } \geq - r \| \nabla _ { Z } L ( Z , s ) \| _ { F } . } \end{array}
$$

Equality is attained at

$$
\Delta Z = - r \frac { \nabla _ { Z } L ( Z , s ) } { \| \nabla _ { Z } L ( Z , s ) \| _ { F } } ,
$$

whenever $\nabla _ { Z } L ( Z , s ) \neq 0$ . Hence $- \nabla _ { Z } L ( Z , s )$ is the steepest descent direction of $\bar { L } _ { Z , s }$ at $\Delta Z = 0$ under the Frobenius norm. □

Restatement of Proposition 1. Let $\Delta Z \in \{ - 1 , 0 , 1 \}$ }<sup>d</sup>out<sup>×d</sup>in be a feasible one-step code move such that

$$
Z + \Delta Z \in { \mathcal { Z } } , \qquad \Delta Z \odot \nabla _ { \widehat { W } } L ( Z , s ) \leq 0 \quad \mathrm { e n t r y w i s e } .
$$

Then the surrogate linear term matches the exact first-order deployed-weight change:

$$
\langle \nabla _ { Z } L ( Z , s ) , \Delta Z \rangle _ { F } = \Bigl \langle \nabla _ { \widehat { W } } L ( Z , s ) , \widehat { W } ( Z + \Delta Z , s ) - \widehat { W } ( Z , s ) \Bigr \rangle _ { F } .
$$

Proof of Proposition 1. We first prove the exact first-order identity

$$
\langle \nabla _ { Z } L ( Z , s ) , \Delta Z \rangle _ { F } = \Bigl \langle \nabla _ { \widehat { W } } L ( Z , s ) , \widehat { W } ( Z + \Delta Z , s ) - \widehat { W } ( Z , s ) \Bigr \rangle _ { F } .
$$

Fix one coordinate $( i , j )$ . If $\Delta Z _ { i j } = 0$ , both sides contribute zero. If $\Delta Z _ { i j } = 1$ , feasibility gives $Z _ { i j } ~ < ~ K$ , and gradient alignment gives $( \nabla _ { \widehat { W } } L ( Z , s ) ) _ { i j } \ \leq \ 0$ . When $( \nabla _ { \widehat { W } } L ( Z , s ) ) _ { i j } < 0$ , the definition of ${ \cal D } \big ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \big )$ selects the forward gap, so

$$
\begin{array} { r l } & { \widehat { W } _ { i j } ( Z + \Delta Z , s ) - \widehat { W } _ { i j } ( Z , s ) = S _ { i j } ( s ) \big ( c _ { Z _ { i j } + 1 } - c _ { Z _ { i j } } \big ) } \\ & { \qquad = d _ { i j } ^ { + } ( Z , s ) } \\ & { \qquad = \big [ D \big ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \big ) \big ] _ { i j } \Delta Z _ { i j } . } \end{array}
$$

If instead $( \nabla _ { \widehat { W } } L ( Z , s ) ) _ { i j } = 0$ , then both first-order contributions are zero. If $\Delta Z _ { i j } = - 1$ , feasibility gives $Z _ { i j } > \ddot { 1 }$ , and gradient alignment gives $( \nabla _ { \widehat { W } } L ( Z , s ) ) _ { i j } \geq 0$ . When $( \nabla _ { \widehat { W } } \bar { L } ( Z , s ) ) _ { i j } > 0$ , the definition of ${ \cal D } \big ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \big )$ selects the backward gap, so

$$
\begin{array} { r l } & { \widehat { W } _ { i j } ( Z + \Delta Z , s ) - \widehat { W } _ { i j } ( Z , s ) = S _ { i j } ( s ) \big ( c _ { Z _ { i j } - 1 } - c _ { Z _ { i j } } \big ) } \\ & { \phantom { \qquad = } - d _ { i j } ^ { - } ( Z , s ) } \\ & { \phantom { \qquad = } \left[ D \big ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \big ) \right] _ { i j } \Delta Z _ { i j } . } \end{array}
$$

Again, if $( \nabla _ { \widehat { W } } L ( Z , s ) ) _ { i j } = 0$ , both first-order contributions are zero. Therefore, for every coordinate,

$$
\begin{array} { r } { ( \nabla _ { \widehat { W } } L ( Z , s ) ) _ { i j } \left( \widehat { W } _ { i j } ( Z + \Delta Z , s ) - \widehat { W } _ { i j } ( Z , s ) \right) = ( \nabla _ { \widehat { W } } L ( Z , s ) ) _ { i j } \left[ D \big ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \big ) \right] _ { i j } \Delta Z _ { i j } . } \end{array}
$$

Summing over coordinates yields

$$
\begin{array} { r l r } & { } & { \Big \langle \nabla _ { \widehat { W } } L ( Z , s ) , \widehat { W } ( Z + \Delta Z , s ) - \widehat { W } ( Z , s ) \Big \rangle _ { F } = \big \langle \nabla _ { \widehat { W } } L ( Z , s ) \odot D \big ( Z , s , \nabla _ { \widehat { W } } L ( Z , s ) \big ) , \Delta Z \big \rangle _ { F } } \\ & { } & { = \langle \nabla _ { Z } L ( Z , s ) , \Delta Z \rangle _ { F } , } \end{array}
$$

which proves (4).

Restatement of Corollary 1. Under the assumptions of Proposition 1, if L is $\beta \mathrm { . }$ -smooth as a function of the deployed weight in a neighborhood of $\widehat { W } ( Z , s )$ , then

$$
L ( Z + \Delta Z , s ) \leq L ( Z , s ) + \langle \nabla _ { Z } L ( Z , s ) , \Delta Z \rangle _ { F } + \frac { \beta } { 2 } \left\| \widehat { W } ( Z + \Delta Z , s ) - \widehat { W } ( Z , s ) \right\| _ { F } ^ { 2 } .
$$

ProofofCorollary 1. If L is β-smooth as a function of the deployed weight in a neighborhood of $\widehat { W } ( Z , s )$ , the descent lemma gives

$$
\begin{array} { r } { L \bigl ( \widehat { W } ( Z + \Delta Z , s ) \bigr ) \leq L \bigl ( \widehat { W } ( Z , s ) \bigr ) + \Bigl \langle \nabla _ { \widehat { W } } L \bigl ( \widehat { W } ( Z , s ) \bigr ) , \widehat { W } ( Z + \Delta Z , s ) - \widehat { W } ( Z , s ) \Bigr \rangle _ { F } } \\ { + \displaystyle \frac { \beta } { 2 } \left\| \widehat { W } ( Z + \Delta Z , s ) - \widehat { W } ( Z , s ) \right\| _ { F } ^ { 2 } . } \end{array}
$$

By Proposition 1,

$$
\begin{array} { r } { \Big \langle \nabla _ { \widehat { W } } L \big ( \widehat { W } ( Z , s ) \big ) , \widehat { W } ( Z + \Delta Z , s ) - \widehat { W } ( Z , s ) \Big \rangle _ { F } = \langle \nabla _ { Z } L ( Z , s ) , \Delta Z \rangle _ { F } . } \end{array}
$$

Substituting this identity together with

$$
L ( Z , s ) = L { \bigl ( } { \widehat { W } } ( Z , s ) { \bigr ) } , \qquad L ( Z + \Delta Z , s ) = L { \bigl ( } { \widehat { W } } ( Z + \Delta Z , s ) { \bigr ) }
$$

yields

$$
L ( Z + \Delta Z , s ) \leq L ( Z , s ) + \langle \nabla _ { Z } L ( Z , s ) , \Delta Z \rangle _ { F } + \frac { \beta } { 2 } \left\| \widehat { W } ( Z + \Delta Z , s ) - \widehat { W } ( Z , s ) \right\| _ { F } ^ { 2 } ,
$$

as claimed.

## E A CONCRETE FACTORIZED CANDIDATE-SAMPLING DISTRIBUTION

To instantiate the local candidate-sampling rule in Section 4.3, define the neighborhood around the guided reference $Z ^ { \mathrm { r e f } }$ by

$$
\begin{array} { r } { \mathcal { N } ^ { ( \rho ) } ( Z ^ { \mathrm { { r e f } } } ) = \left\{ Z ^ { \prime } \in \mathcal { Z } \vert \Vert Z ^ { \prime } - Z ^ { \mathrm { { r e f } } } \Vert _ { \infty } \leq \rho \right\} , } \end{array}
$$

where $\rho \in \mathbb { Z } ^ { + }$ is the code search radius. For efficient sampling in high-dimensional spaces, we sample each matrix entry $( i , j ) \in [ 1 , d _ { \mathrm { o u t } } ] \times [ 1 , d _ { \mathrm { i n } } ]$ independently from a discrete scalar distribution $p _ { i j } ( \bar { u } \mid Z ^ { \mathrm { r e f } } )$ , which jointly induces

$$
p ( Z ^ { m } \mid Z ^ { \mathrm { r e f } } ) = \prod _ { i = 1 } ^ { d _ { \mathrm { o u t } } } \prod _ { j = 1 } ^ { d _ { \mathrm { i n } } } p _ { i j } ( Z _ { i j } ^ { m } \mid Z ^ { \mathrm { r e f } } ) .
$$

We choose

$$
p _ { i j } ( u \mid Z ^ { \mathrm { r e f } } ) = \frac { \phi ( | u - Z _ { i j } ^ { \mathrm { r e f } } | ) } { \sum _ { v \in { \mathcal { N } _ { i j } ^ { ( p ) } } ( Z ^ { \mathrm { r e f } } ) } \phi ( | v - Z _ { i j } ^ { \mathrm { r e f } } | ) } , \qquad { \mathcal { N } _ { i j } ^ { ( \rho ) } } ( Z ^ { \mathrm { r e f } } ) = \left\{ u \in [ K ] \mid | u - Z _ { i j } ^ { \mathrm { r e f } } | \leq \rho \right\}
$$

where $\phi ( x )$ is a monotonically decreasing function, such as $( x + \varepsilon ) ^ { - 1 }$ with $\varepsilon > 0$ or $\exp ( - x ^ { 2 } )$ , so that code states closer to $Z ^ { \mathrm { r e f } }$ receive larger coordinatewise probability mass.

## F EXPECTED DESCENT OF THE CODE-SEARCH SUBSTEP

For the proof of Proposition 2, we analyze the deterministic full-batch code-search substep of Algorithm 1 with a fixed scale s. At iteration t, define the code-space surrogate gradient and continuous reference step by

$$
g _ { t } = \nabla _ { Z } L ( Z _ { t } , s ) , \qquad \bar { \Delta } _ { t } = - \eta _ { Z } g _ { t } , \qquad Z _ { t } ^ { \mathrm { r e f } } = Z _ { t } + \bar { \Delta } _ { t } .
$$

For a sampled candidate state $Z _ { t } ^ { m }$ , let

$$
\Delta _ { t } ^ { m } = Z _ { t } ^ { m } - Z _ { t }
$$

denote its code update relative to the current state. The proof uses the following assumptions.

Assumption A1 (smoothness on admissible sampled moves). There exist constants $\beta > 0$ and $\kappa > 0$ such that, for every iteration t and every candidate update $\Delta$ considered below, Corollary 1 applies and

$$
\begin{array} { r } { \left\| \widehat { W } ( Z _ { t } + \Delta , s ) - \widehat { W } ( Z _ { t } , s ) \right\| _ { F } \leq \kappa \| \Delta \| _ { F } . } \end{array}\tag{7}
$$

Assumption A2 (near-reference proposal mass). Fix some $c \in [ 0 , 1 )$ . For each iteration $t ,$ define the good-update set

$$
\mathcal G _ { t } = \Big \{ \Delta \ \Big | \ \Delta \ \mathrm { i s ~ g r a d i e n t - a l i g n e d ~ a n d ~ o n e - s t e p ~ f e a s i b l e , ~ a n d } \ \| \Delta + \eta _ { Z } g _ { t } \| _ { F } \leq c \eta _ { Z } \| g _ { t } \| _ { F } \Big \} .
$$

Let $q _ { t }$ denote the total proposal mass of candidate states whose update belongs to $\mathcal { G } _ { t } \mathrm { : }$ :

$$
q _ { t } = \sum _ { Z ^ { \prime } : Z ^ { \prime } - Z _ { t } \in \mathcal { G } _ { t } } p ( Z ^ { \prime } \mid Z _ { t } ^ { \mathrm { r e f } } ) .
$$

Proof of Proposition 2. Fix an iteration t. If $\Delta \in { \mathcal { G } } _ { t }$ , then by definition of $\mathcal { G } _ { t }$

$$
\begin{array} { r } { \langle g _ { t } , \Delta \rangle = \langle g _ { t } , - \eta _ { Z } g _ { t } \rangle + \langle g _ { t } , \Delta + \eta _ { Z } g _ { t } \rangle \leq - \eta _ { Z } \| g _ { t } \| _ { F } ^ { 2 } + \| g _ { t } \| _ { F } \| \Delta + \eta _ { Z } g _ { t } \| _ { F } \leq - ( 1 - c ) \eta _ { Z } \| g _ { t } \| _ { F } ^ { 2 } , } \end{array}
$$

and also

$$
\begin{array} { r } { \| \Delta \| _ { F } \leq \| \eta _ { Z } g _ { t } \| _ { F } + \| \Delta + \eta _ { Z } g _ { t } \| _ { F } \leq ( 1 + c ) \eta _ { Z } \| g _ { t } \| _ { F } . } \end{array}
$$

Combining this with (7) gives

$$
\begin{array} { r } { \left\| \widehat { W } ( Z _ { t } + \Delta , s ) - \widehat { W } ( Z _ { t } , s ) \right\| _ { F } ^ { 2 } \leq \kappa ^ { 2 } ( 1 + c ) ^ { 2 } \eta _ { Z } ^ { 2 } \| g _ { t } \| _ { F } ^ { 2 } . } \end{array}
$$

Therefore, by Corollary 1,

$$
L ( Z _ { t } + \Delta , s ) \le L ( Z _ { t } , s ) - \left( 1 - c - \frac { \beta \kappa ^ { 2 } \eta _ { Z } } { 2 } ( 1 + c ) ^ { 2 } \right) \eta _ { Z } \| g _ { t } \| _ { F } ^ { 2 } = L ( Z _ { t } , s ) - \alpha \eta _ { Z } \| g _ { t } \| _ { F } ^ { 2 } .
$$

Now let

Et = nat least one of the M sampled candidates has update in $\mathcal { G } _ { t } \bigg \}$

By independence of the M samples,

$$
\operatorname* { P r } ( E _ { t } \mid Z _ { t } ) = 1 - ( 1 - q _ { t } ) ^ { M } .
$$

On the event $E _ { t }$ , the sampled set contains a candidate whose loss is at most $L ( Z _ { t } , s ) - \alpha \eta _ { Z } \lVert g _ { t } \rVert _ { F } ^ { 2 }$ On $E _ { t }$ , the best-of-M rule therefore returns a candidate whose loss is no larger than this value. On the complementary event $E _ { t } ^ { c }$ , the strict accept-if-improving rule gives $L ( \bar { Z _ { t + 1 } } , s ) \leq L ( Z _ { t } , s )$ Combining the two cases yields the pointwise bound

$$
L ( Z _ { t + 1 } , s ) \leq L ( Z _ { t } , s ) - \alpha \eta _ { Z } \| g _ { t } \| _ { F } ^ { 2 } \mathbf { 1 } _ { E _ { t } } .
$$

Taking conditional expectation gives

$$
\begin{array} { r } { \mathbb { E } [ L ( Z _ { t + 1 } , s ) \mid Z _ { t } ] \leq L ( Z _ { t } , s ) - \alpha \eta _ { Z } \big [ 1 - ( 1 - q _ { t } ) ^ { M } \big ] \| g _ { t } \| _ { F } ^ { 2 } , } \end{array}
$$

which proves the claimed one-step descent inequality.

## G LOW-RANK CODE UPDATE

The core search rule described above only requires a code matrix $Z \in { \mathcal { Z } }$ to induce weight $\widehat { W } ( Z )$ and a differentiable loss to compute (3). As a result, the same rule can also be used with other parameterizations. Low-rank parameterizations are widely used in LLM fine-tuning to reduce the number of optimization variables and improve parameter efficiency. We apply this principle in code space by parameterizing a local code update as a low-rank decomposition with two integer-valued factors:

$$
\Delta Z ( A , B ) = A B ^ { \top } , \qquad A \in \mathbb { Z } ^ { d _ { \mathrm { o u t } } \times R } , \quad B \in \mathbb { Z } ^ { d _ { \mathrm { i n } } \times R } .
$$

This reduces the number of searched coordinates from $d _ { \mathrm { o u t } } d _ { \mathrm { i n } }$ to $R ( d _ { \mathrm { o u t } } + d _ { \mathrm { i n } } )$ . Unlike ordinary $\mathrm { L o R A } , A$ and B are not inference-time residual adapters; after training, only the deployed code state clip $\displaystyle Z + \Delta Z ( A , B ) , 1 , K )$ needs to be stored. The factor-wise guidance follows from the same surrogate chain rule. Given $\nabla _ { Z } L$ by (3) at the induced code state, we use

$$
\nabla _ { \boldsymbol { A } } L = \nabla _ { Z } L \boldsymbol { B } , \qquad \nabla _ { \boldsymbol { B } } L = \left( \nabla _ { Z } L \right) ^ { \top } \boldsymbol { A } .
$$

We form guided references for A and B, sample integer-valued factor candidates, and select by the deployed loss after merging the induced code update.