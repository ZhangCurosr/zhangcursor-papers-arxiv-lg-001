# Reservoir of Importance: Learning Semi-Structured Sparsity with Differentiable Subset Sampling

Ha Dinh<sup>1</sup> \* Xuan Duy Ta<sup>1</sup> \* Khoat Than<sup>2</sup> Khac-Hoai Nam Bui<sup>1</sup> <sup>†</sup> <sup>1</sup>Viettel AI, Viettel Group, Vietnam

<sup>2</sup>Hanoi University of Science and Technology, Hanoi, Vietnam {hadv17,duytx}@viettel.com.vn, khoattq@soict.hust.edu.vn, nambkh@viettel.com.vn

## Abstract

Semi-structured N:M sparsity has emerged as a practical direction for accelerating large language models (LLMs). However, existing learnable-mask approaches incur substantial parameter and memory overhead, limiting their scalability to large models and aggressive sparsity regimes. In this work, we revisit semistructured pruning from a perspective that reconciles efficiency with scalability. We propose Reservoir of Importance (RoI) <sup>1</sup>, a lightweight semi-structured pruning framework that learns sparsity masks through differentiable subset sampling. Unlike prior methods that model full categorical distributions over all feasible N:M patterns, RoI introduces a compact-logit parameterization for sparsity mask learning and performs sampling without replacement to select masks, thereby reducing trainable parameters from combinatorial complexity to O(M). As a result, RoI requires 1.5–8.75× fewer learnable parameters and significantly lower memory cost, while remaining fully aligned with hardware-friendly sparsity patterns. Extensive evaluations across multiple scales of the Qwen2.5 LLM family (0.5-7B parameters) demonstrate that RoI achieves competitive performance with strong memory efficiency, stability, and scalability to more aggressive N:M sparsity patterns, offering a practical path toward efficient LLM deployment.

## 1 Introduction

Pruning techniques play a central role in developing sparse LLMs that reduce memory footprint and improve inference efficiency. Post-training pruning methods generally fall into three categories: (i) unstructured pruning, which removes individual weight parameters without considering architectural constraints (Sun et al., 2024); (ii) structured pruning, which removes entire components such as neurons, attention heads, or layers (Xia et al., 2024; Le et al., 2025); and (iii) semi-structured pruning, which balances flexibility and regularity by enforcing hardware-friendly sparsity patterns such as N:M sparsity (Fang et al., 2024; Huang et al., 2025).

![](images/ae8027c51e4697c781252288eeb1c629e7b81b54010e6a5a5c6cea99845a2b15.jpg)  
a) Existing mask learning approach

![](images/51a7289c36c319b96f75dc1fe6094ff0ec9186f310dd9dd5ae96c294e77c5bd2.jpg)  
b) The proposed method  
Figure 1: Learnable semi-structured N:M sparsity methods: a) modeling the mask selection process using a categorical distribution over feasible masks, and b) our proposed method by learning to sample subsets without replacement of model parameters. The proposed method is more memory efficient than previous works for most practical N:M sparsity patterns. The memory advantage becomes more pronounced as M increases or when N is around M/2.

In this work, we focus on semi-structured pruning because it removes redundant weights while maintaining regular sparsity patterns that are compatible with modern accelerators. In particular, enforcing N:M sparsity patterns (Hubara et al., 2021) provides principled trade-offs between sparsity flexibility and hardware efficiency, enabling practical inference acceleration.

Recent progress in semi-structured pruning has increasingly focused on learnable-mask approaches that directly optimize N:M sparsity patterns through gradient-based training. Representative methods (e.g., MaskLLM (Fang et al., 2024) and HyperPrune (Sun and Sakuma, 2026)) formulate each pruning group as a categorical distribution over all feasible N:M configurations, which demonstrate strong pruning performance and generalization. However, the major challenge of this approach is the combinatorial parameterization required to model a multinomial distribution of size $\bar { ( } _ { N } ^ { M } )$ for each weight group. Specifically, as M increases, this design imposes substantial memory and parameter overhead, as demonstrated in Figure 1 (a). Motivated by this limitation, we propose a lightweight alternative termed Reservoir of Importance (RoI), which replaces full categorical modeling with differentiable subset sampling. Specifically, instead of parameterizing the probability of every possible N:M pattern, RoI samples N entries without replacement from a compact logit vector of size M using Weighted Reservoir Sampling and Gumbel-Top-K relaxation. This reduces the learnable mask parameters from $\mathcal { O } ( ( { } _ { N } ^ { M } ) )$ to a simple $\mathcal O ( M )$ per group while retaining differentiability and hardware alignment, as shown in Figure 1 (b). By avoiding combinatorial distributions and relying on efficient subset-sampling dynamics, the proposed method enables scalable, stable, and memory-efficient mask learning for semistructured pruning of modern LLMs.

## 2 Related Work

Semi-structured pruning has emerged as an effective compromise between structured pruning (An et al., 2024; Liu et al., 2025b) and unstructured pruning (Dong et al., 2024; Sun et al., 2024). By enforcing regular sparsity patterns (i.e., N:M sparsity), semi-structured pruning enables efficient hardware acceleration while preserving model accuracy (Hubara et al., 2021).

## 2.1 Saliency-Based Methods

Saliency-based approaches, including SparseGPT (Frantar and Alistarh, 2023) and Wanda (Sun et al.,

2024), rely on a small calibration dataset, typically a subset of the pre-training data, to approximate the knowledge encoded in a pre-trained language model. These methods estimate the importance of individual weights or weight groups using criteria such as weight magnitude, gradient information, or second-order statistics (e.g., Hessian approximations inspired by Optimal Brain Damage (LeCun et al., 1989) and Optimal Brain Surgeon (Hassibi and Stork, 1992)), and prune parameters accordingly. Although computationally efficient, the effectiveness of these importance estimates strongly depends on the underlying saliency criterion and the representativeness of the calibration data. In practice, the limited calibration set may fail to adequately capture the rich and diverse knowledge embedded in large language models, potentially leading to suboptimal pruning decisions.

## 2.2 Learning-Based Methods

Learning-based semi-structured pruning has recently gained attention due to its ability to optimize pruning decisions directly through gradientbased training, rather than relying on handcrafted importance metrics. Under the N:M sparsity constraint, the objective is to retain exactly N nonzero weights within each group of M parameters, thereby producing hardware-friendly sparsity patterns while minimizing performance degradation (Zhou et al., 2021). A prominent line of work formulates mask learning as an optimization problem over categorical distributions that enumerate all feasible N:M configurations. For example, MaskLLM (Fang et al., 2024) models each group’s pruning mask as a multinomial distribution and employs Gumbel-Softmax sampling to enable differentiable training. This approach achieves strong pruning performance and good generalization across tasks; however, it incurs substantial computational and memory overhead due to the $\mathcal { O } \big ( \big ( _ { N } ^ { \hat { M } } \big ) \big )$ ) parameterization per group. On the other hand, ProxSparse (Liu et al., 2025a) reduces training complexity via regularized optimization and smaller learning samples. Despite these improvements, challenges related to large-scale learning efficiency and performance generalization remain open.

In contrast to prior work, our study follows a large-scale learning-based paradigm while substantially reducing the parameterization cost of mask learning. Specifically, we improve the categorical formulation with an O(M) parameterization per group by leveraging weighted reservoir sampling. This design achieves significant memory savings while retaining high pruning accuracy, making it well-suited for large-scale semi-structured pruning, as detailed in the following section.

## 3 Preliminaries

## 3.1 Weighted Reservoir Sampling

Weighted Reservoir Sampling (WRS) (Efraimidis and Spirakis, 2006) is an extension of the Reservoir Sampling class of algorithms (Vitter, 1985), which aims to sample K elements from a population of size L. In WRS, each item is associated with a positive weight, and items with larger relative weights are preferentially sampled. Formally, consider a finite population $\mathcal { X } = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { L } \}$ equipped with weights $\pmb { w } = [ w _ { 1 } , w _ { 2 } , \dots , w _ { L } ]$ , WRS produces an ordered subset $\mathcal { Y } ~ = ~ \{ y _ { 1 } , y _ { 2 } , . ~ . ~ . ~ , y _ { K } \}$ given by:

$$
\begin{array} { r } { \underset { \mathbb { W } \times \mathbb { S } } { \operatorname* { P r } } ( \mathcal { V } | \pmb { w } ) = \frac { w _ { y _ { 1 } } } { W } \times \frac { w _ { y _ { 2 } } } { W - w _ { y _ { 1 } } } \times . . . } \\ { \times \frac { w _ { y _ { K } } } { W - \sum _ { j = 1 } ^ { K - 1 } w _ { y _ { j } } } , } \end{array}\tag{1}
$$

where $\begin{array} { r } { W = \sum _ { i = 1 } ^ { L } w _ { i } } \end{array}$ denotes the total weight and $w _ { y _ { i } }$ is the weight of element $y _ { i }$ . This distribution corresponds to a without-replacement sampling process, where the probability of selecting a subset is proportional to its item weights.

## 3.2 Gumbel-Top-K Trick

Gumbel-Max (Gumbel, 1954) can be viewed as a monotonic transformation of the WRS technique, providing a reparameterization trick to sample from a categorical distribution by perturbing the logprobabilities with Gumbel noise. Consider a categorical distribution over $\{ x _ { 1 } , \ldots , x _ { L } \}$ , parameterized by logits $\phi = [ \phi _ { 1 } , \ldots , \phi _ { L } ]$ , with probabilities $\pi _ { i } = \exp ( \phi _ { i } ) / { \sum _ { j = 1 } ^ { L } \exp ( \phi _ { j } ) }$ . The Gumbel-Max trick samples from this distribution by first associating each item with a random key obtained via Gumbel perturbation:

$$
\kappa _ { i } = \phi _ { i } + g _ { i } , \quad g _ { i } \stackrel { \mathrm { i . i . d } } { \sim } \mathrm { G u m b e l } ( 0 , 1 ) ,\tag{2}
$$

where each $g _ { i }$ is independently drawn from the Gumbel(0, 1) distribution. The sampler outputs the item x<sub>j</sub> having the largest key $\kappa _ { j }$ . The index $j$ is the output of taking argmax over every perturbed key values $( j = \mathrm { a r g m a x } _ { i } \kappa _ { i } )$

Gumbel-Top-K (Xie and Ermon, 2019) is a generalization of the Gumbel-Max trick, selecting the top-K items with the highest keys, rather than just the maximum. This corresponds to drawing K items without replacement from a categorical distribution over L candidates. By replacing the argtop operator with a sequence of softmax relaxations (Plötz and Roth, 2018), the sampling process becomes differentiable, thereby enabling endto-end learning via backpropagation.

To sample a size-K subset using the Gumbel-Top-K trick, logits are first independently perturbed with Gumbel noise to form random keys $\kappa _ { i } ,$ , analogously to the Gumbel-Max trick. Subsequently, a sequence of softmax operations is applied to generate differentiable approximations of the selected one-hot indicators. Let $\alpha ^ { ( k ) } =$ $[ \alpha _ { 1 } , \dots , \alpha _ { L } ]$ denote adjusted keys at sampling step k, which are defined recursively as follows:

$$
\begin{array} { l } { { \pmb { \alpha } } ^ { ( 1 ) } = [ \kappa _ { 1 } , \ldots , \kappa _ { L } ] , } \\ { { \pmb { \alpha } } ^ { ( k ) } = { \pmb { \alpha } } ^ { ( k - 1 ) } + \log ( 1 - { \pmb { \mu } } ^ { ( k - 1 ) } ) , } \end{array}\tag{3}
$$

where $\pmb { \mu } ^ { ( k - 1 ) } \ = \ [ \mu _ { 1 } ^ { ( k - 1 ) } , \dots , \mu _ { L } ^ { ( k - 1 ) } ]$ is the relaxed one-hot indicator of the item selected at step $k - 1$ . This representation is achieved by applying a softmax with temperature τ to the adjusted keys:

$$
\mu _ { i } ^ { ( k - 1 ) } = \frac { \exp ( \alpha _ { i } ^ { ( k - 1 ) } / \tau ) } { \sum _ { j = 1 } ^ { L } \exp ( \alpha _ { j } ^ { ( k - 1 ) } / \tau ) } .\tag{4}
$$

After K iterations, the procedure produces an ordered collection of relaxed one-hot vectors ${ \boldsymbol { s } } =$ $\{ \pmb { \mu } ^ { ( 1 ) } , \ldots , \pmb { \mu } ^ { ( K ) } \}$ . Summing these vectors yields a soft K-hot representation, and the mapping from logits $\phi _ { i }$ to this representation is differentiable, enabling gradient-based training.

## 4 Methodology

## 4.1 Problem Statement

The task of determining the optimal N:M sparsity pattern can be formulated as selecting, for each contiguous group of M parameters, a length-M binary mask containing exactly N non-zero entries that minimizes the calibration loss. Let G be the number of such groups, $\mathbf { W } = \{ \mathbf { w } _ { 1 } , \dots , \mathbf { w } _ { G } \}$ the corresponding parameter groups, and ${ \textbf { M } } =$ $\{ \mathbf { m } _ { 1 } , \hdots , \mathbf { m } _ { G } \}$ their associated masks. The resulting optimization problem is:

$$
\mathbf { M } ^ { * } = \underset { \mathbf { M } } { \operatorname { a r g m i n } } \ \mathcal { L } _ { \mathrm { C E } } ( \mathcal { D } ; \mathbf { W } \odot \mathbf { M } ) .\tag{5}
$$

Here, $\mathcal { L } _ { \mathrm { C E } }$ denotes the cross-entropy loss for language modeling, D is the calibration dataset, and ⊙ represents the element-wise multiplication between each weight group and its corresponding mask. This optimization problem is NP-hard due to the combinatorial search space, containing ${ \binom { M } { N } } ^ { G }$ feasible configurations. For LLMs, the number of weight groups G is extremely large, rendering exhaustive search intractable. Therefore, in the following section, we reformulate the objective as a stochastic variational optimization problem to obtain a tractable and efficient approximation.

![](images/2a26b0c6262046250aeadee9b3112fbfbf9a6195e153c4f0535d4920507160ca.jpg)  
Figure 2: Overview of the RoI framework for semi-structure sparsity learning via differentiable subset sampling, illustrating the training and inference phases.

## 4.2 Proposed Method

An overview of the proposed RoI method is presented in Figure 2. Stochastic variational optimization (Bird et al., 2018) relies on the observation that for any arbitrary distribution $q ( x )$ , the expected value of a function $f ( x )$ provides an upper bound for its minimum:

$$
\operatorname* { m i n } _ { x } f ( x ) \leq \mathbb { E } _ { q ( x ) } [ f ( x ) ] .\tag{6}
$$

By modeling pruning masks as random variables, the optimization problem in Equation 5 can be reframed as minimizing a variational upper bound of the objective with respect to the variational distribution parameters. Specifically, we solve:

$$
\begin{array} { r }  \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf { \Phi } \mathbf  \Phi  \end{array}\tag{7}
$$

where $\begin{array} { c c l } { \Phi } & { = } & { \left\{ \phi _ { 1 } , . . . , \phi _ { G } \right\} } \end{array}$ parameterizes independent variational factors $\mathrm { P r } ( \mathbf { m } _ { 1 } | \phi _ { 1 } )$ $\mathrm { P r } ( \mathbf { m } _ { G } | \phi _ { G } )$ , with joint distribution $\begin{array} { r l } { \mathrm { P r } ( \mathbf { M } | \boldsymbol { \Phi } ) = } \end{array}$ $\begin{array} { r l } { \prod _ { i = 1 } ^ { G } \operatorname* { P r } ( \mathbf { m } _ { i } | \boldsymbol { \phi } _ { i } ) } \end{array}$ . Under this stochastic variational formulation problem, mask sampling can be reparameterized and continuously relaxed into a differentiable mapping with respect to Φ, making it possible to learn through gradient-based methods.

## 4.2.1 Variational Distribution Selection

Since pruning masks are N-hot vectors of length M, each mask can take one of $\textstyle { \binom { M } { N } }$ possible values. Modeling an unconstrained distribution over all possible values therefore requires ${ \binom { M } { N } } \ - \ 1$ free parameters, which grows combinatorially with M. To efficiently learn masks with a manageable number of parameters, we instead model mask distributions $\mathrm { P r } ( \mathbf { m } _ { i } | \phi _ { i } )$ using the WRS distribution (Equation 1) over ordered subsets. Let $S _ { i } = \{ \mu _ { i } ^ { ( 1 ) } , \ldots , \mu _ { i } ^ { ( N ) } \}$ be a set of N one-hot vectors corresponding to the selected weights within the i-th group, sampled from the WRS distribution using the Gumbel-Top-K trick. The probability of a mask $\mathbf { m } _ { i }$ is then:

$$
\mathrm { P r } ( \mathbf { m } _ { i } | \phi _ { i } ) = \sum _ { \cal S _ { \mathbf { m } _ { i } } } \mathrm { P r } _ { } ( \cal S _ { \mathbf { m } _ { i } } | \exp ( \phi _ { i } ) ) ,\tag{8}
$$

where $\boldsymbol { S } _ { \mathbf { m } _ { i } }$ denotes the collection of ordered subsets whose elements sum to m , i.e., $\begin{array} { r l } { \mathbf { m } _ { i } } & { { } = } \end{array}$ $\textstyle \sum _ { j = 1 } ^ { N } \mu _ { i } ^ { ( j ) }$ . Exactly computing this probability is expensive and unnecessary, as the construction of $\mathbf { m } _ { i }$ discards the ordering of items in the sampled subset. Instead, the expected loss can be computed via Monte Carlo sampling. Under this formulation, the original problem reduces to learning importance scores $\exp ( \phi _ { i j } )$ to select salient weights so as to minimize the objective. Denoting the WRS distribution of an arbitrary subset $S _ { i }$ by $\mathrm { P r } _ { \mathrm { W R S } } ( S _ { i } | \phi _ { i } )$ for brevity, the optimization problem in Equation 7 can be reformulated as follows:

$$
\Phi ^ { * } = \underset { \Phi } { \operatorname { a r g m i n } } \ \mathbb { E } _ { \operatorname* { P r w e s } } ( \mathcal { S } | \Phi ) \big [ \mathcal { L } _ { \mathrm { C E } } \big ( \mathcal { D } ; \mathbf { W } \odot \mathbf { M } \big ) \big ] ,\tag{9}
$$

where $\begin{array} { c c l } { \mathcal { S } } & { = } & { \{ { S } _ { 1 } , . . . , { S } _ { G } \} } \end{array}$ is a collection of G subsets sampled from the joint distribution $\begin{array} { r } { \operatorname* { P r } _ { \mathrm { W R S } } ( \cal { S } | \Phi ) \ = \ \prod _ { i = 1 } ^ { G } \operatorname* { P r } _ { \mathrm { W R S } } ( S _ { i } | \phi _ { i } ) } \end{array}$ Each $N -$ hot pruning mask m<sub>i</sub> in the mask collection M is constructed as $\begin{array} { r } { \mathbf { m } _ { i } = \sum _ { \pmb { \mu } _ { i } ^ { ( j ) } \in S _ { i } } \pmb { \mu } _ { i } ^ { ( j ) } } \end{array}$ . Parameterizing masks as sums of subsets sampled from WRS-restricted distributions yields the same expected loss as sampling masks from the target distributions, as formalized in Theorem 1. Our approach reduces parameter complexity by viewing N-hot sampling as a sequential sampling without replacement process. Rather than maintaining a full categorical distribution over all $\textstyle { \binom { M } { N } }$ possible configurations, we instead model only a single categorical distribution over the M model parameters, requiring exactly M parameters independent of $N$ . The proposed method achieves a reduction in parameter complexity from $\mathcal { O } ( \binom { M } { N } )$ to $\mathcal O ( M )$ , yielding an exponential improvement in memory efficiency.

## 4.2.2 Mask Selection Relaxation

To enable differentiation of the objective with respect to the variational distributions’ parameters, we relax the discrete sampling process using the Gumbel-Top-K trick. Given logits $\phi _ { i } = $ $[ \phi _ { i 1 } , . . . , \phi _ { i M } ]$ forming a categorical distribution over the M consecutive model weights within the i-th group, the probability of selecting the j-th weight is achieved through the softmax function, $\pi _ { i j } = \exp ( \phi _ { i j } ) / \sum _ { k = 1 } ^ { M } \exp ( \phi _ { i k } )$ . To sample a subset $s _ { i }$ without replacement from this distribution, we first perturb the logits with independent Gumbel noise to obtain random keys:

$$
\kappa _ { i j } = \phi _ { i j } + g _ { i j } , g _ { i j } \stackrel { \mathrm { i . i . d } } { \sim } \mathrm { G u m b e l } ( 0 , 1 ) .\tag{10}
$$

We denote the adjusted keys for the i-th group at sampling step k by $\pmb { \alpha } _ { i } ^ { ( k ) } = [ \alpha _ { i 1 } ^ { ( k ) } , \dots , \alpha _ { i M } ^ { ( \overline { { k } } ) } ]$ . The update rule is defined recursively. At each subsequent step k, the adjusted keys are updated by accumulating a correction term based on the previously selected probabilities as follows:

$$
\begin{array} { l } { \pmb { \alpha } _ { i } ^ { ( 1 ) } = [ \kappa _ { i 1 } , \ldots , \kappa _ { i M } ] , } \\ { \pmb { \alpha } _ { i } ^ { ( k ) } = \pmb { \alpha } _ { i } ^ { ( k - 1 ) } + \log ( 1 - \pmb { \mu } _ { i } ^ { ( k - 1 ) } ) . } \end{array}\tag{11}
$$

Finally, a differentiable relaxed one-hot vector $\pmb { \mu } _ { i } ^ { ( k ) } = [ \mu _ { i 1 } ^ { ( k ) } , \dots , \mu _ { i M } ^ { ( k ) } ]$ , representing the selected item at the k-th sampling step, is achieved by applying a softmax operation to the adjusted keys with temperature $\tau > 0$

$$
\mu _ { i j } ^ { ( k ) } = \frac { \exp ( \alpha _ { i j } ^ { ( k ) } / \tau ) } { \sum _ { k = 1 } ^ { M } \exp ( \alpha _ { i k } ^ { ( k ) } / \tau ) } .\tag{12}
$$

After N sampling steps, a set of soft one-hot vectors representing selected weights is obtained. By summing up these vectors, a continuous relaxation of the N-hot pruning mask can be constructed, enabling gradient-based optimization of the objective.

## 4.2.3 Temperature Annealing

Having previously defined the temperature hyperparameter τ to control the sharpness of one-hot approximations, we now introduce a separate hyperparameter λ to explicitly govern the level of stochasticity in the sampling process. This is realized by applying the Gumbel-Top-K trick to the scaled logits, ${ \overline { { \phi } } } _ { i } = \phi _ { i } / \lambda .$ , such that larger values of λ induce higher sampling randomness. In our experiments, we implement annealing schedules for both τ and λ to progressively guide the mask learning process, starting with high randomness to encourage broad exploration and gradually converging to a small set of high-confidence solutions toward the end of training. Specifically, we adopt exponential annealing schedules: at the t-th training step, the temperatures are:

$$
\tau _ { t } = \operatorname* { m a x } \left\{ \tau _ { \mathrm { e n d } } , \tau _ { \mathrm { i n i t } } \times \left( \frac { \tau _ { \mathrm { e n d } } } { \tau _ { \mathrm { i n i t } } } \right) ^ { \frac { t } { T _ { \mathrm { a n n e a l } } } } \right\} ,\tag{13}
$$

$$
\lambda _ { t } = \operatorname* { m a x } \left\{ \lambda _ { \mathrm { e n d } } , \lambda _ { \mathrm { i n i t } } \times \left( \frac { \lambda _ { \mathrm { e n d } } } { \lambda _ { \mathrm { i n i t } } } \right) ^ { \frac { t } { T _ { \mathrm { a n n e a l } } } } \right\} ,
$$

with $T _ { \mathrm { a n n e a l } }$ being the number of annealing steps.

## 5 Experiment

## 5.1 Experimental Setting

LLM backbones: The proposed method is evaluated on the Qwen2.5 model family (Yang et al., 2024) at multiple scales, from 0.5B to 7B, to assess its stability and scalability under semi-structured pruning. All main experiments adopt a 2:4 sparsity pattern, compatible with NVIDIA hardware acceleration. Training is performed for 2,000 steps with a batch size of 256 and a sequence length of 4096, processing approximately 2B tokens in total. The training data is drawn from the Nemotron-CCv2 corpus (NVIDIA et al., 2025), a clean English dataset suitable for pre-training modern LLMs. We detail hyperparameters in Appendix A.4.

Baselines: To assess the effectiveness of the proposed method, we select five representative semistructured pruning baselines spanning both classical and modern approaches: Magnitude (Han et al.,

<table><tr><td>Method</td><td>W/U</td><td>ARC-C</td><td>ARC-E</td><td>BoolQ</td><td>HellaS.</td><td>PIQA</td><td>RACE</td><td>SciQ</td><td>Average ↑</td></tr><tr><td>Base Model: Qwen2.5-0.5B</td><td>=</td><td>29.01</td><td>64.48</td><td>61.80</td><td>40.54</td><td>70.51</td><td>34.74</td><td>92.90</td><td>56.28</td></tr><tr><td>Magnitude</td><td>x</td><td>19.11</td><td>31.19</td><td>38.38</td><td>26.65</td><td>54.13</td><td>23.06</td><td>43.40</td><td>33.70</td></tr><tr><td>Wanda</td><td>x</td><td>19.80</td><td>42.85</td><td>56.82</td><td>28.93</td><td>59.03</td><td>26.70</td><td>83.20</td><td>45.33</td></tr><tr><td>SparseGPT</td><td>√</td><td>19.97</td><td>47.22</td><td>59.85</td><td>30.80</td><td>60.12</td><td>27.85</td><td>84.30</td><td>47.16</td></tr><tr><td>ProxSparse</td><td>x</td><td>21.42</td><td>46.63</td><td>61.63</td><td>32.00</td><td>61.63</td><td>29.09</td><td>79.80</td><td>47.46</td></tr><tr><td>MaskLLM</td><td>x</td><td>22.16</td><td>54.02</td><td>62.10</td><td>32.28</td><td>63.41</td><td>30.19</td><td>85.90</td><td>50.01</td></tr><tr><td>RoI (Ours)</td><td>x</td><td>23.63</td><td>54.97</td><td>60.95</td><td>32.34</td><td>63.82</td><td>31.00</td><td>87.20</td><td>50.56</td></tr><tr><td>Base Model: Qwen2.5-1.5B</td><td></td><td>41.55</td><td>75.21</td><td>72.48</td><td>50.18</td><td>75.52</td><td>36.65</td><td>94.20</td><td>63.68</td></tr><tr><td>Magnitude</td><td>x</td><td>20.48</td><td>32.74</td><td>39.42</td><td>27.20</td><td>57.07</td><td>24.59</td><td>60.10</td><td>37.37</td></tr><tr><td>Wanda</td><td>x</td><td>24.06</td><td>51.26</td><td>63.64</td><td>32.36</td><td>62.46</td><td>29.28</td><td>88.30</td><td>50.19</td></tr><tr><td>SparseGPT</td><td>√</td><td>28.33</td><td>56.69</td><td>62.66</td><td>34.54</td><td>65.07</td><td>33.49</td><td>89.00</td><td>52.83</td></tr><tr><td>ProxSparse</td><td>x</td><td>29.25</td><td>59.55</td><td>42.87</td><td>39.61</td><td>67.25</td><td>32.73</td><td>89.80</td><td>51.58</td></tr><tr><td>MaskLLM</td><td>x</td><td>29.45</td><td>63.51</td><td>61.33</td><td>39.10</td><td>69.07</td><td>32.51</td><td>89.90</td><td>54.98</td></tr><tr><td>RoI (Ours)</td><td>x</td><td>29.37</td><td>63.01</td><td>61.93</td><td>39.29</td><td>69.31</td><td>32.44</td><td>89.90</td><td>55.04</td></tr><tr><td>Base Model: Qwen2.5-3B</td><td>=</td><td>44.62</td><td>77.57</td><td>77.22</td><td>55.04</td><td>78.29</td><td>38.47</td><td>95.90</td><td>66.73</td></tr><tr><td>Magnitude</td><td>x</td><td>23.81</td><td>24.66</td><td>37.83</td><td>25.98</td><td>51.96</td><td>20.77</td><td>18.50</td><td>29.07</td></tr><tr><td>Wanda</td><td>x</td><td>27.13</td><td>60.02</td><td>62.26</td><td>35.88</td><td>67.14</td><td>34.35</td><td>91.00</td><td>53.97</td></tr><tr><td>SparseGPT</td><td>√</td><td>31.40</td><td>66.12</td><td>62.66</td><td>41.05</td><td>69.75</td><td>35.12</td><td>92.50</td><td>56.94</td></tr><tr><td>ProxSparse</td><td>x</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MaskLLM</td><td>x</td><td>32.95</td><td>66.42</td><td>65.11</td><td>43.18</td><td>70.44</td><td>35.35</td><td>93.00</td><td>58.06</td></tr><tr><td>RoI (Ours)</td><td>X</td><td>32.08</td><td>66.71</td><td>65.26</td><td>43.30</td><td>70.57</td><td>35.22</td><td>93.00</td><td>58.02</td></tr><tr><td>Base Model: Qwen2.5-7B</td><td>=</td><td>48.21</td><td>80.43</td><td>84.65</td><td>59.98</td><td>78.78</td><td>41.91</td><td>96.60</td><td>70.08</td></tr><tr><td>Magnitude</td><td>x</td><td>24.40</td><td>38.34</td><td>64.16</td><td>30.05</td><td>55.60</td><td>23.25</td><td>74.30</td><td>44.30</td></tr><tr><td>Wanda</td><td>x</td><td>38.99</td><td>72.18</td><td>73.21</td><td>44.53</td><td>71.87</td><td>35.79</td><td>94.50</td><td>61.58</td></tr><tr><td>SparseGPT</td><td>√</td><td>39.65</td><td>73.54</td><td>75.83</td><td>45.89</td><td>73.38</td><td>36.67</td><td>94.80</td><td>62.82</td></tr><tr><td>ProxSparse</td><td>x</td><td>38.69</td><td>74.46</td><td>74.77</td><td>48.63</td><td>73.22</td><td>36.08</td><td>95.60</td><td>63.06</td></tr><tr><td>MaskLLM</td><td>x</td><td>39.27</td><td>73.68</td><td>81.55</td><td>49.12</td><td>74.88</td><td>37.46</td><td>95.10</td><td>64.44</td></tr><tr><td>RoI (Ours)</td><td>x</td><td>38.14</td><td>74.20</td><td>82.14</td><td>49.04</td><td>75.35</td><td>37.80</td><td>95.10</td><td>64.54</td></tr></table>

Table 1: Comparative evaluation of zero-shot accuracy across multiple benchmark datasets for various pruning methods on the Qwen2.5 model family at varying scales with 2:4 sparsity pattern. Bold and underlined values indicate the highest and second-highest performance, respectively. The ‘W/U’ column denotes whether weight updates are applied during pruning. ProxSparse fails to converge for the 3B model.

2015), Wanda (Sun et al., 2024), SparseGPT (Frantar and Alistarh, 2023), ProxSparse (Liu et al., 2025a), and MaskLLM (Fang et al., 2024). For ProxSparse, we use the same amount of training data as RoI and MaskLLM to ensure fair comparisons, even though ProxSparse typically requires less data. Performance across different data scales is reported in Section 5.3.2, and while detailed methodological descriptions of baselines are provided in Appendix A.2.

## 5.2 Main Results

Following prior work, we begin by evaluating zeroshot accuracies on a broad suite of standard benchmark datasets (Table 1). Across all model scales and evaluation tasks, sparsity-mask learning approaches (ProxSparse, MaskLLM, and RoI) consistently outperform more saliency-based pruning baselines, with only minor exceptions. Among these methods, approaches that explicitly learn the pruning mask (MaskLLM and RoI) via variational optimization substantially exceed deterministic regularization-based learning (ProxSparse), improving accuracy by $2 . 5 6 ^ { 2 }$ percentage points on average across model scales and benchmarks. Despite RoI and MaskLLM both achieving comparable results (e.g. 64.54% vs. 64.44% average accuracy for the 7B model), RoI incurs substantially lower pruning overhead. Specifically, for 2:4 sparsity, RoI reduces the number of trainable parameters by approximately 33%, using only 1× the original model’s parameter count, compared to 1.5× for MaskLLM (illustrated in Table 3).

Additionally, Table 2 reports perplexity (PPL) values on the WikiText-2 dataset, further highlighting the advantages of our proposed method: (i) Effectiveness of Differentiable Subset Sampling: RoI consistently outperforms all competing baselines at every model scale, indicating that its differentiable subset sampling mechanism reliably discovers high-quality sparsity patterns that better preserve model performance after pruning. (ii) Scalability across Model Sizes: RoI inherits the strong generalization properties of MaskLLM and delivers consistent improvements as the model scale increases, enabled by a closely related for mulation and optimization strategy. RoI maintains its advantage over simpler methods from small to large models: achieving 14.89 perplexity at 7B parameters, compared to 20.87 for Wanda and 16.44 for SparseGPT. This stable scaling behavior contrasts sharply with Magnitude pruning - which attains its best perplexity at 1.5B parameters rather than at 7B, further emphasizing RoI’s ability to maintain reliable performance as model capacity grows. In contrast, ProxSparse exhibits unstable optimization behavior at larger scales. In particular, when applied to the Qwen2.5-3B model, ProxSparse fails to reliably converge, preventing it from achieving competitive performance. (iii) Robustness of Learnable Sparsity-Mask Approaches: Magnitude pruning performs poorly (consistently yielding perplexity values exceeding 5,000), and other naive baselines fall far behind methods that explicitly learn the pruning mask, reaffirming that simple pruning strategies substantially impair language modeling performance. The strong results of RoI and MaskLLM underscore the importance of principled, learnable pruning mechanisms.

<table><tr><td rowspan="2">Method</td><td colspan="4">Model Sizes</td></tr><tr><td>0.5B</td><td>1.5B</td><td>3B</td><td>7B</td></tr><tr><td>w/o Pruning</td><td>18.45</td><td>12.59</td><td>12.32</td><td>9.03</td></tr><tr><td>Magnitude</td><td>3.0E+4</td><td>5.2E+3</td><td>1.6E+6</td><td>6.5E+3</td></tr><tr><td>Wanda</td><td>159.05</td><td>67.97</td><td>37.31</td><td>20.87</td></tr><tr><td>SparseGPT</td><td>73.78</td><td>38.95</td><td>22.82</td><td>16.44</td></tr><tr><td>ProxSparse</td><td>146.42</td><td>28.68</td><td></td><td>14.91</td></tr><tr><td>MaskLLM</td><td>35.16</td><td>21.90</td><td>19.85</td><td>14.91</td></tr><tr><td>RoI (Ours)</td><td>32.63</td><td>21.27</td><td>19.37</td><td>14.89</td></tr></table>

Table 2: Perplexity (PPL) scores on WikiText-2 benchmark dataset and training flops (FLOPs) across backbone models with 2:4 sparsity pattern setting. Bold and underlined values indicate the best and second-best performance, respectively.

RoI and MaskLLM adopt similar problem formulations and largely comparable optimization procedures. Although RoI overall outperforms MaskLLM in both zero-shot accuracy and perplexity, the raw performance gap remains relatively modest. The primary advantage of RoI instead lies in its significantly lower training cost, as illustrated in Table 3, which compares the number of trainable parameters required by each method. In particular, the training cost of MaskLLM grows rapidly with model size. Even under 2:4 sparsity, MaskLLM requires 1.5× more trainable parameters than RoI, a difference that may appear moderate in ratio but translates into a large absolute increase at scale. For the Qwen2.5-7B model, achieving 2:4 sparsity with MaskLLM requires ≈9.75B additional trainable parameters.

<table><tr><td rowspan="2">Method</td><td colspan="4">Model Sizes</td></tr><tr><td>0.5B</td><td>1.5B</td><td>3B</td><td>7B</td></tr><tr><td colspan="5">2:4 Sparsity</td></tr><tr><td>ProxSparse</td><td>0.4B (1×)</td><td>1.3B (1×)</td><td>2.7B (1×)</td><td>6.5B (1×)</td></tr><tr><td>MaskLLM</td><td>0.6B (1.5×)</td><td>2.0B (1.5×)</td><td>4.0B (1.5×)</td><td>9.8B (1.5×)</td></tr><tr><td>RoI (Ours)</td><td>0.4B (1×)</td><td>1.3B (1×)</td><td>2.7B (1×)</td><td>6.5B (1×)</td></tr><tr><td colspan="5">2:8 Sparsity</td></tr><tr><td>MaskLLM</td><td>1.4B (3.5×)</td><td>4.6B (3.5×)</td><td>9.5B (3.5×)</td><td>22.8B (3.5×)</td></tr><tr><td>RoI (Ours)</td><td>0.4B (1×)</td><td>1.3B (1×)</td><td>2.7B (1×)</td><td>6.5B (1×)</td></tr></table>

Table 3: Number of trainable parameters (in billions) under different sparsity patterns and model sizes. Values in parentheses indicate the ratio relative to the original dense model parameters (excluding embeddings). For mask-learning methods, only mask parameters are trainable while the underlying model weights remain frozen.

## 5.3 Detailed Analysis

## 5.3.1 2:8 Sparsity as a Failure Regime

To further evaluate the generalization of RoI, we conduct a stress test under the 2:8 configuration. Because 2:8 sparsity is not broadly supported by mainstream accelerators, this experiment is intended for scalability evaluation rather than deployment. This aggressive sparsity exposes the challenges faced by semi-structured pruning methods, particularly those that do not learn pruning masks.

As shown in Table 4, only methods that learn pruning masks achieve reasonable perplexity values, while saliency-based baselines collapse, producing perplexities exceeding 100. This behavior is congruent with the 2:4 setting and highlight the critical necessity of learnable mask parameterization. Accuracy results under 2:8 sparsity also exhibit the same trend and are deferred to Appendix A.12.

In conjunction with Table 3, we again observe the rapid growth in training cost of MaskLLM as model size increases. Moreover, the parameterrequirement gap between RoI and MaskLLM widens substantially under more aggressive sparsity. For the 7B model at 2:8 sparsity, MaskLLM requires more than twice the auxiliary parameters compared to the 2:4 setting. This corresponds to ≈22.75B additional parameters relative to RoI, resulting in a 3.5× increase in parameters participating in memory allocation, backpropagation, optimizer and scheduler states, while only achieving perplexity levels comparable to RoI.

<table><tr><td rowspan="2">Method</td><td colspan="4">Model Sizes</td></tr><tr><td>0.5B</td><td>1.5B</td><td>3B</td><td>7B</td></tr><tr><td>w/o Pruning</td><td>18.45</td><td>12.59</td><td>12.32</td><td>9.03</td></tr><tr><td>Magnitude</td><td>1.5E+7</td><td>1.0E+7</td><td>6.6E+7</td><td>infinity³</td></tr><tr><td>Wanda</td><td>8.1E+5</td><td>1.0E+6</td><td>3.2E+6</td><td>4.7E+4</td></tr><tr><td>SparseGPT</td><td>1.1E+4</td><td>4.0E+4</td><td>257.22</td><td>158.70</td></tr><tr><td>MaskLLM</td><td>63.11</td><td>38.14</td><td>38.89</td><td>34.72</td></tr><tr><td>RoI (Ours)</td><td>61.89</td><td>36.87</td><td>37.72</td><td>32.33</td></tr></table>

Table 4: WikiText-2 Perplexity (PPL) scores across backbone models with 2:8 sparsity pattern setting. Bold and underlined values indicate the best and second-best performance, respectively. ProxSparse is excluded due to its inability to learn 2:8 sparsity patterns.

These results demonstrate that differentiable subset sampling is sufficient to sustain language modeling capability under extreme sparsity, while requiring only 28.57% of the trainable parameters of the categorical parameterization. This establishes RoI as the minimal learnable-mask formulation capable of minimizing accuracy loss.

## 5.3.2 Perplexity over Training Tokens

Among mask-learning pruning approaches, deterministic regularization-based learning fundamentally limits how effectively a model can explore in the parameter space due to the bias-variance tradeoff. While regularization can accelerate early-stage optimization, it introduces a persistent bias that constrains convergence as training data grows. In contrast, directly learning sparsity masks in a variational manner avoids this limitation and is therefore intrinsically better suited to large-scale learning regimes. This difference is directly reflected in Figure 3. During the early stage of training (until approximately 0.5B tokens), the regularization-based pruning baseline ProxSparse attains lower perplexity. However, this advantage is temporary: as additional training data is incorporated, the performance of ProxSparse rapidly plateaus. Meanwhile, RoI and MaskLLM continue to improve steadily and monotonically, consistently translating additional data into lower perplexity. Although RoI and MaskLLM optimize the same variational objective, RoI exhibits stronger scalability in practice; see Appendix A.7 for a detailed analysis. These results provide clear empirical evidence that learning masks directly is critical for unlocking improvements under large-scale learning settings and achieving superior asymptotic performance.

![](images/111c54b016b9da14ffb3c3bb4f14444e5480af46c1a464e03b818b00922ce746.jpg)  
Figure 3: RoI demonstrates better scalability than MaskLLM and ProxSparse, with WikiText-2 perplexity (on the Qwen2.5-1.5B model) improving more consistently as the number of training tokens increases.

## 6 Conclusion

This work introduced RoI, a novel semi-structured pruning technique that employs differentiable subset sampling to efficiently derive N:M sparsity masks. By substantially reducing the number of trainable parameters and associated memory overhead, RoI enables effective compression of LLMs while maintaining strong downstream performance. Across Qwen2.5 models ranging 0.5 to 7B parameters, RoI consistently achieves lower perplexity on WikiText-2 compared to existing pruning approaches, and maintains competitive zero-shot accuracy on a diverse set of benchmarks. Moreover, RoI demonstrates enhanced data efficiency and favorable scalability as the amount of calibration data increases. Overall, these results position RoI as a practical and scalable solution for semi-structured pruning of LLMs, offering a strong balancing between compression efficiency and performance preservation for resource-constrained deployment environments.

## Limitations

Despite the promising performance and efficiency demonstrated by RoI, several limitations remain: (i) First, the deployment of semi-structured sparsity is inherently hardware-dependent. At present, substantial throughput gains are realized only on select platforms (e.g., AMD ROCm and certain NVIDIA Ampere and Hopper GPUs) where 2:4 structured sparsity is natively supported and accelerated at the kernel level. Although RoI can, in principle, be extended to arbitrary N : M sparsity patterns, its practical utility is constrained by the absence of hardware kernels and vendor-optimized libraries for ratios other than 2:4. On accelerators or CPUs lacking such specialized support, pruning yields only marginal reductions in memory footprint and fails to deliver meaningful inference speed up. This hardware dependency poses a significant challenge for widespread adoption in heterogeneous production environments, where deployment targets may vary; (ii) Second, the current evaluation focuses exclusively on English-centric, dense models and a limited set of standard NLP benchmarks. Future research should investigate the applicability of RoI to multilingual LLMs, different architectures like Mixture-of-Experts, larger-scale models, and domain-specific tasks (e.g., code generation, reasoning-intensive applications) to assess its generalization and scalability comprehensively.

## References

Yongqi An, Xu Zhao, Tao Yu, Ming Tang, and Jinqiao Wang. 2024. Fluctuation-based adaptive structured pruning for large language models. In Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications ofArtificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2024, February 20-27, 2024, Vancouver, Canada, pages 10865–10873. AAAI Press.

Thomas Bird, Julius Kunze, and David Barber. 2018. Stochastic variational optimization. CoRR, abs/1809.04855.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. PIQA: reasoning about physical commonsense in natural language. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications ofArtificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 7432– 7439. AAAI Press.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. Boolq: Exploring the surprising difficulty of natural yes/no questions. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 2924–2936. Association for Computational Linguistics.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the AI2 reasoning challenge. CoRR, abs/1803.05457.

Peijie Dong, Lujun Li, Zhenheng Tang, Xiang Liu, Xinglin Pan, Qiang Wang, and Xiaowen Chu. 2024. Pruner-zero: Evolving symbolic pruning metric from scratch for large language models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

Pavlos S. Efraimidis and Paul G. Spirakis. 2006. Weighted random sampling with a reservoir. Inf. Process. Lett., 97(5):181–185.

Gongfan Fang, Hongxu Yin, Saurav Muralidharan, Greg Heinrich, Jeff Pool, Jan Kautz, Pavlo Molchanov, and Xinchao Wang. 2024. Maskllm: Learnable semistructured sparsity for large language models. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Elias Frantar and Dan Alistarh. 2023. Sparsegpt: Massive language models can be accurately pruned in one-shot. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings ofMachine Learning Research, pages 10323–10337. PMLR.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, and 5 others. 2024. The language model evaluation harness.

Emil Julius Gumbel. 1954. Statistical theory ofextreme values and some practical applications: a series of lectures, volume 33. US Government Printing Office.

Song Han, Jeff Pool, John Tran, and William J. Dally. 2015. Learning both weights and connections for efficient neural network. In Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada, pages 1135–1143.

Babak Hassibi and David G. Stork. 1992. Second order derivatives for network pruning: Optimal brain surgeon. In Advances in Neural Information Processing Systems 5, [NIPS Conference, Denver, Colorado, USA, November 30 - December 3, 1992], pages 164– 171. Morgan Kaufmann.

Weiyu Huang, Yuezhou Hu, Guohao Jian, Jun Zhu, and Jianfei Chen. 2025. Pruning large language models with semi-structural adaptive sparse training. In AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA, pages 24167– 24175. AAAI Press.

Itay Hubara, Brian Chmiel, Moshe Island, Ron Banner, Joseph Naor, and Daniel Soudry. 2021. Accelerated sparse neural training: A provable and efficient method to find N: M transposable masks. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 21099–21111.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard H. Hovy. 2017. RACE: large-scale reading comprehension dataset from examinations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, EMNLP 2017, Copenhagen, Denmark, September 9-11, 2017, pages 785–794. Association for Computational Linguistics.

Qi Le, Enmao Diao, Ziyan Wang, Xinran Wang, Jie Ding, Li Yang, and Ali Anwar. 2025. Probe pruning: Accelerating llms through dynamic pruning via model-probing. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Yann LeCun, John S. Denker, and Sara A. Solla. 1989. Optimal brain damage. In Advances in Neural Information Processing Systems 2, [NIPS Conference, Denver, Colorado, USA, November 27-30, 1989], pages 598–605. Morgan Kaufmann.

Hongyi Liu, Rajarshi Saha, Zhen Jia, Youngsuk Park, Jiaji Huang, Shoham Sabach, Yu-Xiang Wang, and George Karypis. 2025a. Proxsparse: Regularized learning of semi-structured sparsity masks for pretrained LLMS. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025. OpenReview.net.

Yijiang Liu, Huanrui Yang, Youxin Chen, Rongyu Zhang, Miao Wang, Yuan Du, and Li Du. 2025b. PAT: pruning-aware tuning for large language models. In AAAI-25, Sponsored by the Association for the Advancement ofArtificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA, pages 24686–24695. AAAI Press.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2017. Pointer sentinel mixture models. In 5th International Conference on Learning

Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. Open-Review.net.

NVIDIA, :, Aarti Basant, Abhijit Khairnar, Abhijit Paithankar, Abhinav Khattar, Adithya Renduchintala, Aditya Malte, Akhiad Bercovich, Akshay Hazare, Alejandra Rico, Aleksander Ficek, Alex Kondratenko, Alex Shaposhnikov, Alexander Bukharin, Ali Taghibakhshi, Amelia Barton, Ameya Sunil Mahabaleshwarkar, Amy Shen, and 198 others. 2025. Nvidia nemotron nano 2: An accurate and efficient hybrid mamba-transformer reasoning model. Preprint, arXiv:2508.14444.

Tobias Plötz and Stefan Roth. 2018. Neural nearest neighbors networks. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada, pages 1095–1106.

Lu Sun and Jun Sakuma. 2026. Learning semistructured sparsity for llms via shared and contextaware hypernetwork. In The Fourteenth International Conference on Learning Representations.

Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. 2024. A simple and effective pruning approach for large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Gemma Team. 2024. Gemma 2: Improving open language models at a practical size. CoRR, abs/2408.00118.

Jeffrey Scott Vitter. 1985. Random sampling with a reservoir. ACM Trans. Math. Softw., 11(1):37–57.

Johannes Welbl, Nelson F. Liu, and Matt Gardner. 2017. Crowdsourcing multiple choice science questions. In Proceedings of the 3rd Workshop on Noisy Usergenerated Text, NUT@EMNLP 2017, Copenhagen, Denmark, September 7, 2017, pages 94–106. Association for Computational Linguistics.

Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi Chen. 2024. Sheared llama: Accelerating language model pre-training via structured pruning. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Sang Michael Xie and Stefano Ermon. 2019. Reparameterizable subset sampling via continuous relaxations. In Proceedings ofthe Twenty-Eighth International Joint Conference on Artificial Intelligence, IJ-CAI 2019, Macao, China, August 10-16, 2019, pages 3919–3925. ijcai.org.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and

22 others. 2024. Qwen2.5 technical report. CoRR, abs/2412.15115.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? In Proceedings ofthe 57th Conference ofthe Associationfor Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 4791–4800. Association for Computational Linguistics.

Aojun Zhou, Yukun Ma, Junnan Zhu, Jianbo Liu, Zhijie Zhang, Kun Yuan, Wenxiu Sun, and Hongsheng Li. 2021. Learning N: M fine-grained structured sparse neural networks from scratch. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

## A Appendix

## A.1 WRS Yields Equivalent Variational Objective

Theorem 1. Let $\mathrm { P r } ( \mathbf { m } _ { i } | \phi _ { i } )$ be the target distribution ofeach mask $\mathbf { m } _ { i }$ as defined in Equation 8. The expected loss obtained by sampling each mask from its target distribution is equal to the expected loss obtained when each mask is parameterized as a sum of elements of an ordered subset $s _ { i }$ sampled from the corresponding restricted distribution Pr<sub>WRS</sub> $( S _ { i } | \phi _ { i } )$

We show that:

$$
\mathbb { E } _ { \mathrm { P r } ( \mathbf { m } | \phi ) } [ f ( \mathbf { m } ) ] = \mathbb { E } _ { \mathrm { P r } _ { \mathrm { W R S } } ( S | \phi ) } \left[ f \left( \sum _ { \mu \in \mathcal { S } } \mu \right) \right] ,\tag{14}
$$

where $f ( \mathbf { m } )$ is an objective function of m, and $\begin{array} { r } { \mathrm { P r } ( \mathbf { m } | \phi ) = \sum _ { \mathcal { S } _ { \mathbf { m } } } \mathrm { P r } _ { \mathbb { W } \mathrm { R } \mathrm { S } } ( \mathcal { S } _ { \mathbf { m } } | \phi ) } \end{array}$ with $\scriptstyle { S _ { \mathbf { m } } }$ ranging over all sets whose elements sum to m.

Proof. Given any set of binary masks M satisfying N:M sparsity, by definition of $\mathrm { P r } ( \mathbf { m } \mid \phi )$

$$
\begin{array} { l } { { \displaystyle \mathbb { E } _ { \mathbf { P r } ( \mathbf { m } | \phi ) } [ f ( \mathbf { m } ) ] = \sum _ { \mathbf { m } \in \mathcal { M } } \mathrm { P r } ( \mathbf { m } \mid \phi ) f ( \mathbf { m } ) } \ ~ } \\ { { \displaystyle = \sum _ { \mathbf { m } \in \mathcal { M } } \left( \sum _ { \mathcal { S } _ { \mathbf { m } } } \mathrm { P r } _ { \mathrm { N S } } ( \mathcal { S } _ { \mathbf { m } } \mid \phi ) \right) f \left( \sum _ { \mu \in \mathcal { S } _ { \mathbf { m } } } \mu \right) . } } \end{array}
$$

Reordering the sums yields:

$$
\begin{array} { l } { { \displaystyle \sum _ { \mathbf { m } \in \mathcal { M } } \sum _ { S _ { \mathbf { m } } } \mathbf { P } _ { \mathrm { T R S } } ( S _ { \mathbf { m } } \mid \phi ) f \left( \sum _ { \mu \in \mathcal { S } _ { \mathbf { m } } } \mu \right) } } \\ { { \displaystyle = \sum _ { S } \mathbf { P } _ { \mathrm { R R S } } ( S \mid \phi ) f \left( \sum _ { \mu \in S } \mu \right) , } } \end{array}
$$

which equals:

$$
\mathbb { E } _ { \mathrm { P r } _ { \mathrm { W R S } } ( S | \phi ) } \Bigg [ f \Bigg ( \sum _ { \mu \in \mathcal { S } } \mu \Bigg ) \Bigg ] .
$$

## A.2 Baseline Methods

i) Magnitude is a simple, data-free method that removes parameters with the smallest absolute values. While this method is easy to implement, it often yields subpar results due to the limitations of parameter sensitivity and model dynamics; ii) Wanda (Sun et al., 2024) combines parameter magnitudes with activation statistics at each layer, achieving better performance than pure magnitude pruning, especially under higher sparsity constraints, while maintaining favorable computational efficiency; iii) SparseGPT (Frantar and Alistarh, 2023) incorporates activation outputs and Hessian information to estimate parameter importance, followed by parameter updates to further reduce output error. This method can yield higher accuracy but is more computationally demanding; iv) ProxSparse (Liu et al., 2025a) leverages regularization to quickly find sparsity masks under very constrained data and computational settings. Owing to its deterministic optimization procedure, ProxSparse typically converges rapidly in the early stages of training. However, this early convergence can limit its ability to further explore the solution space, thereby restricting its capacity to fully exploit additional training data or extended optimization; and v) MaskLLM (Fang et al., 2024) is quite similar to the proposed method in this study, by learning pruning masks with minimizing calibration loss under an N:M sparsity constraint, modeled via a multinomial distribution. It delivers strong performance across benchmarks but suffers from high computational cost.

## A.3 Evaluation Metrics and Benchmark Datasets

Following previous works in this research field, three automated metrics are considered for the evaluation, including both quantitative and qualitative metrics to capture the full impact of pruning: i) Task Accuracy (ACC): on common NLP tasks such as question answering in reading comprehension, mathematics, and science. These tasks are typically assessed in zero-shot or few-shot settings using benchmark datasets; Perplexity (PPL): is a standard metric for assessing language model quality. It measures how well the model predicts the next word in a sequence, with lower values indicating better predictive performance.

<table><tr><td>Dataset</td><td>Questions</td><td>Task Type</td></tr><tr><td>ARC-Easy</td><td>2,376</td><td>Multiple-choice science</td></tr><tr><td>ARC-Challenge</td><td>1,172</td><td>Multiple-choice science</td></tr><tr><td>BoolQ</td><td>3,270</td><td>Yes/no questions</td></tr><tr><td>HellaSwag</td><td>10,042</td><td>Sentence completion</td></tr><tr><td>PIQA</td><td>1,838</td><td>Physical interaction QA</td></tr><tr><td>RACE</td><td>1,045</td><td>Multiple-choice comprehension</td></tr><tr><td>SciQ</td><td>1,000</td><td>Multiple-choice science</td></tr></table>

Table 5: Dataset statistics for zero-shot evaluation. We report the number of questions and the corresponding task type for each benchmark.

The benchmark datasets used to assess the effectiveness of pruning methods include WikiText-2 (Merity et al., 2017) for perplexity evaluation and a range of NLP benchmark datasets for zeroshot evaluation, which cover diverse task types and reasoning requirements, including ARC (Clark et al., 2018), BoolQ (Clark et al., 2019), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), SciQ (Welbl et al., 2017), and RACE (Lai et al., 2017). All evaluations are conducted using the LM-Evaluation-Harness toolkit (Gao et al., 2024). Table 5 provides a comprehensive summary of the datasets used for zero-shot evaluation across multiple tasks. These datasets span a range of domains, including commonsense reasoning, science question answering, and reading comprehension, thereby ensuring a rigorous and diverse assessment of pruning performance.

## A.4 Hyperparameter Setting

The hyperparameters used for training RoI are listed in Table 6. These settings were carefully chosen to balance convergence stability and computational efficiency across all evaluated models.

Specifically, model weights remain frozen during training. The variational distribution is initialized from a standard normal $( \mu = 0 . 0 , \sigma = 0 . 0 1 )$ and a simulated annealing process gradually reduces randomness. For 1500 first training steps, temperatures τ and λ exponentially decay from 1.0 to 0.05 and from 1.0 to 0.002, respectively. Optimization uses AdamW with a learning rate decaying from $1 \times 1 0 ^ { - 3 } \mathrm { t o } 1 \times 1 0 ^ { - 4 }$ , weight decay value of 0.05, and $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$

<table><tr><td>Parameter</td><td>Values</td></tr><tr><td>Initialization distribution Gumbel-Softmax temperature</td><td> $\mathcal { N } ( 0 , 0 . 0 1 )$   $\tau = 1 . 0  0 . 0 5$ </td></tr><tr><td>Sampling temperature Weight decay</td><td> $\lambda = 1 . 0 \to 0 . 0 0 2$  0.05</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 3 }  1 0 ^ { - 4 }$ </td></tr><tr><td>AdamW parameters Batch size</td><td> $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ </td></tr><tr><td>Sequence length</td><td>256</td></tr><tr><td></td><td>4096</td></tr><tr><td></td><td></td></tr><tr><td>Training steps Annealing steps</td><td>2000  $T _ { \mathrm { a n n e a l } } = 1 5 0 0$ </td></tr></table>

Table 6: Hyperparameter configuration used in training.

## A.5 Annealing Schedule Ablation

We further study the effect of the annealing schedule for the sampling temperature λ and the Gumbel-Softmax temperature τ. Table 7 compares exponential and linear schedules with different final values $( \lambda _ { \mathrm { m i n } } , \tau _ { \mathrm { m i n } } )$ on the Qwen2.5-1.5B model. Overall, exponential annealing consistently outperforms linear annealing in both perplexity and average zero-shot accuracy. This suggests that reducing the temperatures more aggressively in the early stage, followed by smaller refinements near the end of training, provides a better balance between exploration and convergence. Among the exponential schedules, (0.001, 0.05) yields the best average zero-shot accuracy, while (0.002, 0.05) gives the lowest perplexity. We therefore use the exponential schedule with $( \lambda _ { \mathrm { m i n } } , \tau _ { \mathrm { m i n } } ) = ( 0 . 0 0 2 , 0 . 0 5 )$ as the default setting, as it provides the strongest language modeling performance while maintaining competitive downstream accuracy.

## A.6 Training Time Comparison

To complement the parameter-efficiency analysis in Table 3, we report the end-to-end training time of learnable semi-structured pruning methods across model scales in Table 8.

<table><tr><td rowspan="2">Method</td><td colspan="4">Model Sizes</td></tr><tr><td>0.5B</td><td>1.5B</td><td>3B</td><td>7B</td></tr><tr><td>ProxSparse</td><td>12.05</td><td>16.62</td><td>32.66</td><td>52.41</td></tr><tr><td>MaskLLM</td><td>9.00</td><td>16.93</td><td>30.95</td><td>55.39</td></tr><tr><td>RoI (Ours)</td><td>7.12</td><td>13.17</td><td>24.36</td><td>46.10</td></tr></table>

Table 8: Reported training time (in GPU hours) across various sizes of the Qwen 2.5 model family for learnable semi-structured pruning methods. Lower values indicate faster training.

<table><tr><td>Schedule</td><td> $( \lambda _ { \mathrm { { m i n } } } , \tau _ { \mathrm { { m i n } } } )$ </td><td>PPL↓</td><td>ARC-C</td><td>ARC-E</td><td>BoolQ</td><td>HellaS.</td><td>PIQA</td><td>RACE</td><td>SciQ</td><td>Avg. ↑</td></tr><tr><td rowspan="3">Exponential</td><td>(0.001, 0.05)</td><td>21.57</td><td>28.50</td><td>63.47</td><td>62.75</td><td>40.06</td><td>70.67</td><td>34.26</td><td>89.00</td><td>55.53</td></tr><tr><td>(0.002, 0.05)</td><td>21.27</td><td>29.37</td><td>63.01</td><td>61.93</td><td>39.29</td><td>69.31</td><td>32.44</td><td>89.90</td><td>55.04</td></tr><tr><td>(0.003, 0.05)</td><td>22.29</td><td>29.18</td><td>61.99</td><td>63.85</td><td>39.24</td><td>68.55</td><td>33.68</td><td>88.30</td><td>54.97</td></tr><tr><td rowspan="3">Linear</td><td>(0.001, 0.05)</td><td>26.80</td><td>28.18</td><td>60.85</td><td>61.70</td><td>38.42</td><td>68.21</td><td>31.83</td><td>87.70</td><td>53.84</td></tr><tr><td>(0.002, 0.05)</td><td>27.04</td><td>27.59</td><td>60.31</td><td>62.11</td><td>37.64</td><td>67.53</td><td>32.29</td><td>87.10</td><td>53.51</td></tr><tr><td>(0.003, 0.05)</td><td>27.51</td><td>28.51</td><td>61.99</td><td>61.25</td><td>37.07</td><td>67.12</td><td>32.29</td><td>87.10</td><td>53.62</td></tr></table>

Table 7: Ablation results for different annealing schedules and final temperature values on the Qwen2.5-1.5B model. PPL denotes WikiText-2 perplexity, and Avg. denotes the average zero-shot accuracy over the downstream tasks.

All methods are trained under the same 2:4 sparsity setting on NVIDIA B300 GPUs and use the same training budget described in Appendix A.4. The results show that RoI consistently requires less training time than both ProxSparse and MaskLLM across all evaluated model sizes. This advantage becomes more pronounced as the model scale increases: on the 7B model, RoI reduces training time from 52.41 hours to 46.10 hours compared with ProxSparse, and from 55.39 hours to 46.10 hours compared with MaskLLM. These results support the claim that the compact subset-sampling parameterization improves practical training efficiency in addition to reducing trainable parameters.

## A.7 Optimization Efficiency

Although RoI and MaskLLM optimize equivalent variational objectives over N:M masks, they use different parameterizations of the same mask distribution. MaskLLM assigns a separate logit to each feasible mask configuration, requiring $\mathbf { \bar { \boldsymbol { C } } } = \mathbf { \boldsymbol { ( \lambda _ { N } ^ { M } ) } }$ logits per pruning group. In contrast, RoI assigns one logit to each item in the group and induces a distribution over masks through weighted sampling without replacement, requiring only M logits. This distinction does not change the target objective, but it changes the geometry of the optimization problem and the efficiency of gradient propagation.

Let M denote the set of all feasible N-hot masks in a group. A full categorical mask learner parameterizes

$$
\operatorname* { P r } _ { \mathbf { c } \mathbf { a } } ( \mathbf { m } | \alpha ) = \frac { \exp ( \alpha _ { \mathbf { m } } ) } { \sum _ { \mathbf { m ^ { \prime } } \in \mathcal { M } } \exp ( \alpha _ { \mathbf { m ^ { \prime } } } ) } , \mathbf { m } \in \mathcal { M } ,\tag{15}
$$

where each configuration has its own independent parameter $\alpha _ { \mathbf { m } }$ . Therefore, a gradient update observed for one sampled mask primarily updates that individual configuration. Information about the usefulness of a particular retained weight is not directly shared with other masks that contain the same weight. Since each weight participates in many feasible masks, this representation can require repeated observations of many related configurations before the optimizer consistently increases the probability of important weights.

RoI instead parameterizes the same decision through item-level logits $\phi = [ \phi _ { 1 } , \ldots , \phi _ { M } ]$ . The probability of an unordered mask m is induced by summing over WRS orderings as in Equation 8. Consequently, the gradient with respect to an item logit aggregates contributions from all sampled masks containing that item:

$$
\begin{array} { r l } & { \displaystyle \frac { \partial } { \partial \phi _ { j } } \mathbb { E } _ { \mathrm { P r } _ { \mathrm { R o I } } ( \mathbf { m } | \phi ) } [ \mathcal { L } ( \mathbf { m } ) ] } \\ & { \displaystyle = \sum _ { \mathbf { m } \in \mathcal { M } } \mathcal { L } ( \mathbf { m } ) \frac { \partial \mathrm { P r } _ { \mathrm { R o I } } ( \mathbf { m } | \phi ) } { \partial \phi _ { j } } . } \end{array}\tag{16}
$$

Because $\phi _ { j }$ influences every mask in which item j can be selected, each update transfers evidence across a combinatorial family of masks rather than only a single categorical state. This parameter sharing provides a useful inductive bias for pruning: if a weight is consistently important, RoI can increase its selection probability across many masks simultaneously.

This explains the empirical convergence gap. The categorical MaskLLM parameterization is more expressive, but it is also higher-dimensional and less statistically efficient; optimization must distribute probability mass over $\textstyle { \binom { M } { N } }$ configurations. RoI restricts the search to a structured WRS-induced family with only M degrees of freedom, which reduces memory traffic, optimizerstate overhead, and Monte Carlo estimation noise. As a result, even though both methods correspond to equivalent variational objectives at the maskdistribution level, RoI follows a better-conditioned and more sample-efficient optimization path.

## A.8 Gumbel-Top-K Algorithm

The algorithm for Gumbel-Top-K is illustrated in Algorithm 1. Specifically, we provide a clear description of the Gumbel-Top-K sampling procedure employed to enable differentiable mask learning. This formulation allows for efficient sampling of K items without replacement while preserving differentiability for gradient-based optimization.

Algorithm 1 Gumbel-Top-K Sampling Algorithm   
(Differentiable)   
Input: Set of candidates $\mathcal { X } = \{ x _ { 1 } , \ldots , x _ { L } \}$ with   
corresponding logits $\phi = [ \phi _ { 1 } , \ldots , \phi _ { L } ]$ , number of   
samples K, temperature $\tau > 0$   
Output: Soft K-hot selection vector ${ \textbf { S } } \in$   
$\mathbb { R } ^ { N }$   
1: for $i \gets 1$ to N do   
2: u<sub>i</sub> ∼ Uniform(0, 1)   
3: $g _ { i }  - \log ( - \log ( u _ { i } ) )$ {Gumbel noise}   
4: $\kappa _ { i } \gets \phi _ { i } + g _ { i }$ {Compute perturbed key}   
5: end for   
6: $\pmb { \alpha } ^ { ( 1 ) }  [ \kappa _ { 1 } , . . . , \kappa _ { N } ]$   
7: for $k \gets 1$ to K do   
8: $\mu ^ { ( k ) } \gets$ softmax $( \alpha ^ { ( k ) } / \tau )$   
9: $\begin{array} { r } { \alpha ^ { ( k + 1 ) }  \alpha ^ { ( k ) } . } \end{array}$ + log $\left( 1 - \mu ^ { ( k ) } \right)$   
10: end for   
11: S $\textstyle  \sum _ { k = 1 } ^ { K } \mu ^ { ( k ) }$ {Soft K-hot vector}   
12: return S

## A.9 Comparison with Straight-Through Gumbel-Top-K

We further examined whether adopting a straightthrough (ST) Gumbel-Top-K estimator benefits pruning performance. In this variant, the forward pass generates discrete masks by directly applying $\mathrm { a r g t o p } _ { K }$ over Gumbel-perturbed logits, while the backward pass propagates gradients through the continuous Gumbel-Softmax relaxation. This strategy enforces discretization earlier in training, which should be able to improve mask interpretability. However, our empirical results in Table 9 show that ST Gumbel-Top-K leads to slightly inferior performance compared to the pure soft relaxation.

For instance, on Qwen2.5-1.5B, ST achieves 30.58 perplexity and 53.10% average accuracy, while the soft approach (RoI) reaches 21.27 perplexity and 55.04% accuracy. Similarly, on Qwen2.5-3B, the gap widens (23.57 vs. 19.37 perplexity). These observations suggest that the bias introduced by the ST estimator hampers generalization, outweighing the potential benefits of earlier discretization. Overall, the soft Gumbel-Top-K relaxation used in RoI provides a more effective balance between trainability and performance.

<table><tr><td></td><td colspan="2">Qwen2.5-1.5B</td><td colspan="2">Qwen2.5-3B</td></tr><tr><td>Metric</td><td>w STE</td><td>w/o STE (Ours)</td><td>w STE</td><td>w/o STE (Ours)</td></tr><tr><td>PPL (↓)</td><td>30.58</td><td>21.27</td><td>23.75</td><td>19.37</td></tr><tr><td>Avg. Acc (↑)</td><td>53.10</td><td>55.04</td><td>55.26</td><td>58.02</td></tr></table>

Table 9: Comparison of pruning results with 2:4 sparsity, both with and without ST Gumbel-Top-K estimator (denoted as "w STE" and "w/o STE").

## A.10 Throughput

Table 10 reports the decoding throughput of dense base models and their corresponding 2:4 sparse variants under different input/output token configurations. All models are deployed using vLLM with identical serving settings on a single NVIDIA B300 GPU to ensure a fair comparison.

<table><tr><td>I/O tokens</td><td>Models</td><td>Throughput</td><td>Acceleration</td></tr><tr><td rowspan="3">1024/4096</td><td>Qwen2.5-3B</td><td>190.60</td><td>1.0x</td></tr><tr><td>Qwen2.5-3B 2:4</td><td>248.50</td><td>1.3x</td></tr><tr><td>Qwen2.5-7B</td><td>165.00</td><td>1.0x</td></tr><tr><td rowspan="4">2048/4096</td><td>Qwen2.5-7B 2:4</td><td>210.10</td><td>1.27x</td></tr><tr><td>Qwen2.5-3B Qwen2.5-3B 2:4</td><td>185.20</td><td>1.0x</td></tr><tr><td></td><td>241.40</td><td>1.3x</td></tr><tr><td>Qwen2.5-7B</td><td>162.00</td><td>1.0x</td></tr><tr><td></td><td>Qwen2.5-7B 2:4</td><td>204.40</td><td>1.26x</td></tr></table>

Table 10: Throughput comparison between dense base models and their 2:4 sparse variants. All models are deployed using vLLM on a single NVIDIA B300 GPU. Throughput is reported under different input/output token configurations. The 2:4 sparse models consistently achieve higher throughput than their dense counterparts, demonstrating effective exploitation of semi-structured sparsity at inference time.

Across both Qwen2.5-3B and Qwen2.5-7B, the 2:4 sparse variants consistently achieve higher throughput than their dense counterparts. For the 1024/4096 I/O setting, the 2:4 models yield up to 1.3× speedup for Qwen2.5-3B and 1.27× speedup for Qwen2.5-7B. Similar improvements are observed under the 2048/4096 configuration, indicating that the throughput gains are robust across different sequence lengths. These results demonstrate that semi-structured 2:4 sparsity can be effectively exploited at inference time, translating sparsity into tangible system-level acceleration without modifying the serving stack. Notably, the relative speedups remain stable across model scales, suggesting that the benefits of 2:4 sparsity generalize well from smaller to larger models.

![](images/56756c0e665850a7654d06c7f33f08a0c91a7df279fe4dea4d9b68db645aa34f.jpg)  
Figure 4: Mask difference comparison between RoI and previous works on the Qwen2.5-7B model. Besides the name of each baseline, place an overlapping percentage indicating the similarity of the produced masks between that baseline and RoI. Bold pixels indicate masks corresponding to retained weights.

## A.11 Mask Difference Analysis

To investigate how different pruning strategies select weights, we measure the overlap between masks produced by various methods on the same model. Figure 4 presents a qualitative comparison of the sparsity masks produced by RoI and several representative pruning baselines, including Magnitude, Wanda, SparseGPT, ProxSparse, and MaskLLM. For each method, we visualize the top-left 8×8 sub-matrix of the learned masks from three representative layers: self\_attn.q\_proj, self\_attn.k\_proj, and mlp.up\_proj. Light pixels denote zero entries corresponding to pruned weights. In addition, we report the overlapping percentage between the masks generated by each baseline and RoI, indicating the degree of structural similarity. Across all layers, RoI exhibits a clear alignment with MaskLLM, as reflected by the higher overlap ratios. This suggests that RoI is able to recover pruning patterns that are consistent with more expensive or data-intensive approaches, despite relying on a more efficient differentiable subset selection mechanism. In contrast, Magnitude pruning shows substantially lower overlap, indicating that purely weight-based criteria lead to structurally different sparsity patterns that deviate from those discovered by data-aware methods. Notably, ProxSparse achieves an overlap comparable to SparseGPT in certain layers, but still demonstrates noticeable structural discrepancies, particularly in attention projections. This observation is consistent with ProxSparse’s deterministic and early-converging optimization behavior, which can restrict exploration of alternative sparsity configurations. Overall, this analysis highlights that RoI not only improves quantitative performance but also produces structured masks that closely resemble those of strong, data-aware pruning baselines, reinforcing its effectiveness in learning meaningful semi-structured sparsity patterns.

## A.12 Accuracy Results for 2:8 Sparsity

Table 12 re-illustrates the severity of the 2:8 sparsity constraint. On ARC-Challenge, pruned models of all sizes and methods fail to exceed the 25% random-guess baseline. Across the remaining benchmarks (Table 11), heuristic pruning methods generally suffer substantial performance degradation, although SparseGPT retains non-trivial accuracy on several tasks. In contrast, approaches that explicitly learn the sparsity mask, namely RoI and MaskLLM, consistently achieve much higher average accuracy across model sizes.

Among saliency-based strategies, SparseGPT is the only method that preserves non-negligible performance, due to permitting parameter updates, and only on SciQ. These accuracy trends indicate that explicit mask learning is the most effective strategy. This observation is consistent with the 2:4 accuracy results in Table 1, as well as the 2:4 and 2:8 perplexity results reported in Tables 2 and 4.

<table><tr><td>Method</td><td>W/U</td><td>ARC-E</td><td>BoolQ</td><td>HellaS.</td><td>PIQA</td><td>RACE</td><td>SciQ</td><td>Average ↑</td></tr><tr><td>Base Model: Qwen2.5-0.5B</td><td>-</td><td>64.48</td><td>61.80</td><td>40.54</td><td>70.51</td><td>34.74</td><td>92.90</td><td>60.83</td></tr><tr><td>Magnitude</td><td>x</td><td>24.96</td><td>38.72</td><td>25.76</td><td>51.90</td><td>20.96</td><td>19.30</td><td>30.27</td></tr><tr><td>Wanda</td><td>x</td><td>26.22</td><td>37.83</td><td>26.43</td><td>54.84</td><td>23.54</td><td>29.40</td><td>33.04</td></tr><tr><td>SparseGPT</td><td>√</td><td>28.37</td><td>38.56</td><td>26.65</td><td>53.05</td><td>24.11</td><td>41.70</td><td>35.41</td></tr><tr><td>MaskLLM</td><td>x</td><td>42.75</td><td>61.65</td><td>26.92</td><td>59.42</td><td>25.86</td><td>73.80</td><td>48.40</td></tr><tr><td>RoI (Ours)</td><td>x</td><td>43.98</td><td>61.65</td><td>27.11</td><td>59.36</td><td>25.93</td><td>73.10</td><td>48.52</td></tr><tr><td>Base Model: Qwen2.5-1.5B</td><td>–</td><td>75.21</td><td>72.48</td><td>50.18</td><td>75.52</td><td>36.65</td><td>94.20</td><td>67.37</td></tr><tr><td>Magnitude</td><td>X</td><td>26.09</td><td>41.41</td><td>25.77</td><td>51.69</td><td>21.24</td><td>20.30</td><td>31.08</td></tr><tr><td>Wanda</td><td>x</td><td>25.76</td><td>47.98</td><td>25.65</td><td>51.63</td><td>23.44</td><td>22.00</td><td>32.74</td></tr><tr><td>SparseGPT</td><td>√</td><td>28.54</td><td>54.31</td><td>26.52</td><td>53.16</td><td>24.69</td><td>45.20</td><td>38.74</td></tr><tr><td>MaskLLM</td><td>x</td><td>50.12</td><td>58.87</td><td>30.20</td><td>63.32</td><td>27.80</td><td>77.40</td><td>51.29</td></tr><tr><td>RoI (Ours)</td><td>x</td><td>50.25</td><td>58.99</td><td>29.97</td><td>63.49</td><td>28.08</td><td>77.20</td><td>51.33</td></tr><tr><td>Base Model: Qwen2.5-3B</td><td>-</td><td>77.57</td><td>77.22</td><td>55.04</td><td>78.29</td><td>38.47</td><td>95.90</td><td>70.42</td></tr><tr><td>Magnitude</td><td>x</td><td>24.24</td><td>37.83</td><td>25.59</td><td>52.23</td><td>20.77</td><td>19.70</td><td>30.06</td></tr><tr><td>Wanda</td><td>x</td><td>26.89</td><td>38.41</td><td>26.42</td><td>53.16</td><td>23.54</td><td>25.50</td><td>32.32</td></tr><tr><td>SparseGPT</td><td>√</td><td>35.14</td><td>58.99</td><td>27.46</td><td>55.11</td><td>24.02</td><td>68.90</td><td>44.94</td></tr><tr><td>MaskLLM</td><td>x</td><td>50.57</td><td>61.91</td><td>30.70</td><td>64.08</td><td>28.22</td><td>82.00</td><td>52.91</td></tr><tr><td>RoI (Ours)</td><td>X</td><td>51.01</td><td>62.32</td><td>30.06</td><td>64.69</td><td>28.80</td><td>81.60</td><td>53.08</td></tr><tr><td>Base Model: Qwen2.5-7B</td><td>-</td><td>80.43</td><td>84.65</td><td>59.98</td><td>78.78</td><td>41.91</td><td>96.60</td><td>73.73</td></tr><tr><td>Magnitude</td><td>x</td><td>26.26</td><td>46.64</td><td>25.42</td><td>52.45</td><td>22.01</td><td>22.20</td><td>32.50</td></tr><tr><td>Wanda</td><td>x</td><td>27.48</td><td>37.83</td><td>26.35</td><td>53.59</td><td>22.68</td><td>32.60</td><td>33.42</td></tr><tr><td>SparseGPT</td><td>√</td><td>32.91</td><td>59.33</td><td>27.87</td><td>55.28</td><td>27.37</td><td>75.20</td><td>46.33</td></tr><tr><td>MaskLLM</td><td>x</td><td>51.20</td><td>60.98</td><td>29.32</td><td>62.90</td><td>28.98</td><td>83.60</td><td>52.83</td></tr><tr><td>RoI (Ours)</td><td>x</td><td>50.67</td><td>62.35</td><td>29.70</td><td>63.44</td><td>28.85</td><td>83.40</td><td>53.07</td></tr></table>

Table 11: Comparative evaluation of zero-shot accuracy across multiple benchmark datasets for various pruning methods on the Qwen2.5 model family of different sizes with 2:8 sparsity pattern. Bold and underlined values indicate the highest and second-highest performance, respectively. ProxSparse is excluded due to its inability to learn 2:8 sparsity patterns.

<table><tr><td rowspan="2">Method</td><td colspan="4">Model Sizes</td></tr><tr><td>0.5B</td><td>1.5B</td><td>3B</td><td>7B</td></tr><tr><td>w/o Pruning</td><td>29.01</td><td>41.55</td><td>44.62</td><td>48.21</td></tr><tr><td>Magnitude</td><td>20.99</td><td>21.93</td><td>23.21</td><td>21.50</td></tr><tr><td>Wanda</td><td>19.80</td><td>19.62</td><td>19.37</td><td>19.88</td></tr><tr><td>SparseGPT</td><td>20.48</td><td>19.80</td><td>17.92</td><td>18.69</td></tr><tr><td>MaskLLM</td><td>18.88</td><td>21.51</td><td>20.81</td><td>21.41</td></tr><tr><td>RoI (Ours)</td><td>18.26</td><td>21.33</td><td>20.48</td><td>21.33</td></tr></table>

Table 12: ARC-Challenge zero-shot accuracy scores across backbone models with 2:8 sparsity pattern setting. ProxSparse is excluded due to its inability to learn 2:8 sparsity patterns.

Collectively, these results further emphasize the advantages of RoI over MaskLLM, showing that RoI achieves competitive or superior performance at a fraction of the computational cost (Table 3), and remains robust across both tasks and sparsity regimes.

## A.13 Additional Results

To evaluate whether the effectiveness of RoI generalizes beyond the Qwen2.5 model family, we further conduct experiments on Gemma2-9B (Team, 2024) under the 2:4 sparsity pattern. We compare RoI with both saliency-based methods and MaskLLM using WikiText-2 perplexity and zeroshot accuracy on seven downstream benchmarks. The results are reported in Table 13.

RoI achieves the lowest perplexity of 16.30, improving upon MaskLLM by 0.95 points and substantially outperforming the saliency-based baselines. It also obtains the highest average zeroshot accuracy of 64.08, compared with 63.73 for MaskLLM and 60.62 for the strongest saliencybased baseline, SparseGPT. Across individual tasks, RoI achieves the best or tied-best performance on six of the seven benchmarks, while remaining competitive on ARC-E. These results demonstrate that RoI transfers effectively to a different model family and consistently preserves language-modeling and downstream-task performance under semi-structured sparsity.

<table><tr><td>Method</td><td>PPL↓</td><td>ARC-C</td><td>ARC-E</td><td>BoolQ</td><td>HellaS.</td><td>PIQA</td><td>RACE</td><td>SciQ</td><td>Average ↑</td></tr><tr><td>Base Model: Gemma2-9B</td><td>10.54</td><td>61.52</td><td>87.16</td><td>84.16</td><td>61.20</td><td>81.18</td><td>42.49</td><td>97.20</td><td>73.56</td></tr><tr><td>Magnitude</td><td>537.39</td><td>34.39</td><td>68.31</td><td>69.79</td><td>42.46</td><td>70.78</td><td>33.21</td><td>88.10</td><td>58.15</td></tr><tr><td>Wanda</td><td>94.49</td><td>37.88</td><td>71.09</td><td>68.62</td><td>42.22</td><td>71.82</td><td>34.55</td><td>94.00</td><td>60.03</td></tr><tr><td>SparseGPT</td><td>68.54</td><td>36.35</td><td>69.49</td><td>72.54</td><td>44.25</td><td>71.82</td><td>36.17</td><td>93.70</td><td>60.62</td></tr><tr><td>MaskLLM</td><td>17.25</td><td>40.36</td><td>74.10</td><td>73.08</td><td>50.37</td><td>75.41</td><td>37.53</td><td>95.30</td><td>63.73</td></tr><tr><td>RoI</td><td>16.30</td><td>40.87</td><td>73.82</td><td>73.55</td><td>50.98</td><td>75.79</td><td>38.28</td><td>95.30</td><td>64.08</td></tr></table>

Table 13: Perplexity and zero-shot accuracy of different pruning methods on the Gemma-2-9B model under 2:4 sparsity. Bold and underlined values indicate the best and second-best results among the pruning methods, respectively.