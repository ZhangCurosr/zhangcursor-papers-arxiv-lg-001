# NORMALIZED LOW-RANK ADAPTATION

Jiale Kang<sup>1,3</sup> Ziyin Yue Zheng Zhan<sup>2</sup> Yangyi Huang<sup>3</sup> Weiyang Liu<sup>3,4</sup>

<sup>1</sup>Yuanshi Intelligence <sup>2</sup>Microsoft Research <sup>3</sup>The Chinese University of Hong Kong <sup>4</sup>Shenzhen Loop Area Institute

spherelab.ai/NoRA

## ABSTRACT

While low-rank adaptation (LoRA) is widely used for parameter-efficient model adaptation, how to regularize its training dynamics for stable and effective optimization remains underexplored. Because LoRA initializes the up-projection to zero, its early optimization dynamics are largely governed by the down-projection. Building on this observation, we introduce Normalized Low-Rank Adaptation (NoRA), a simple yet effective method that normalizes the down-projection matrices during training. We further show that the same normalization can be applied only at initialization, improving standard LoRA without requiring repeated normalization throughout training. Across pretraining, supervised finetuning, and reinforcement learning, NoRA consistently accelerates convergence, improves performance and training stability, and mitigates catastrophic forgetting. These benefits require neither additional trainable parameters nor inference-time computation, making NoRA a simple and broadly applicable enhancement to LoRA.

## 1 INTRODUCTION

As large language models (LLMs) continue to scale, their training costs grow rapidly, increasing the need for parameter-efficient learning. Among parameter-efficient finetuning (PEFT) approaches, Low-rank adaptation (LoRA) (Hu et al., 2021) has become particularly popular due to its simplicity, efficiency, and strong empirical performance. By representing weight updates through low-rank factorization, LoRA greatly reduces the number of trainable parameters and has been successfully applied to supervised finetuning, pretraining, and reinforcement learning (Wang et al., 2026; Shen et al., 2026). Despite its effectiveness, LoRA’s optimization dynamics remain poorly understood. Standard LoRA parameterizes ∆W = αBA with randomly initialized A and zero-initialized B. Although this preserves the pretrained initialization, early optimization depends entirely on the random features induced by A. This observation motivates the question below:

Does the regularization ofthe down-projection matrix A provide a useful design dimension for improving LoRA optimization?

Existing work suggests that initialization can substantially affect low-rank optimization. Methods such as PiSSA (Meng et al., 2024) and MiLoRA (Wang et al., 2025) construct the initial low-rank subspace from the spectral structure of pretrained weights and often converge faster than random initialization. However, they require singular-value decomposition and modify the initial decomposition of the pretrained weights, making them less convenient for large-scale training and potentially fragile in reinforcement-learning settings (Yin et al., 2025). Other approaches improve LoRA through modified scaling (Kalajdzievski, 2023), constrained parameterizations (Liu et al., 2024b), or fixed projections (Kopiczko et al., 2024). However, how the regularization of the down-projection matrix affects or improves LoRA’s optimization remains poorly understood.

Our investigation begins with the observation that the normalized latent bottleneck used in Multihead Latent Attention (MLA) substantially stabilizes training and also improves convergence. Similar observations have also been made in multiple training settings (Liu et al., 2017; 2018; Loshchilov et al., 2025). These phenomena suggest that the numerical structure of the normalized latent representation presented to the up-projection can play an important role in stable optimization. Directly introducing latent normalization into LoRA changes the standard linear update from BAx to B · Norm(Ax). Because the normalization depends on the input, the resulting transformation is nonlinear with respect to x and can no longer be exactly merged into the pretrained weight matrix like LoRA. Inspired by MLA, we therefore study whether the useful numerical effect of latent normalization can instead be encoded as a regularization of the linear down-projection such that we can have both LoRA’s exact mergeability and MLA’s favorable optimization benefits.

To this end, we adapt MLA’s normalization principle from the latent feature to the low-rank projection itself. Each column of $\pmb { A } \in \mathbb { R } ^ { r \times k }$ represents how an input coordinate is projected into the r-dimensional latent space. Guided by the normalization dimension in MLA, we propose to normalize these projection vectors along the rank dimension, yielding the linear form $\bar { B } \cdot \bar { \bf N } \mathrm { o r m } ( A ) x$ We refer to this formulation as Normalized LoRA (NoRA), which constrains the input-to-latent projection magnitudes throughout training while preserving exact weight mergeability. This normalization scheme also ensures that the gradient norm of NoRA can be better aligned with full finetuning, bridging the learnability gap between LoRA and full finetuning.

NoRA requires the down-projection matrix A to be normalized along the output rank dimension. This principle motivates two ways of applying normalization: (1) as a training constraint $( i . e . ,$ NoRA) and (2) as an initialization strategy (i.e., NoRA-init). In the former, NoRA continuously normalizes the down-projection matrix A along the low-rank dimension r throughout training. In the latter, NoRA-init normalizes A only once at initialization and subsequently optimizes it without constraints during training. For NoRA-init, we propose two initialization variants: (1) random normalization, where A is randomly initialized and then normalized; and (2) Block Identity Matrix Initialization (BIMI), where A is constructed as a block-identity matrix that inherently satisfies the rank-dimensional normalization requirement. NoRA improves training by removing undesirable scale imbalance across the low-rank dimensions, providing a better-conditioned parameterization than standard LoRA.

We evaluate both NoRA and NoRA-init across pretraining, supervised finetuning, and reinforcement learning with verifiable rewards. Across a diverse range of base models, tasks, and training configurations, both methods consistently accelerate convergence, improve training stability, and enhance downstream performance. They also mitigate catastrophic forgetting during supervised adaptation and remain robust in reinforcement-learning settings where other spectral initialization methods can be fragile. In general, we can conclude that these results establish rank-dimension normalization of the down-projection as an important design principle for improving LoRA optimization. Our contributions are summarized as follows:

• We identify the down-projection matrix A as a previously underexplored yet important design dimension in LoRA, showing that its magnitude can play a critical role in shaping optimization dynamics and ultimately improving training efficiency and downstream performance.

• We propose NoRA, which normalizes the down-projection matrix A along the rank dimension in the LoRA parameterization. We further introduce NoRA-init, an initialization-only variant that applies this normalization at initialization and requires no re-normalization during training, enabling seamless performance improvements without modifying the optimization procedure.

• Extensive experiments across pretraining, SFT, and RLVR demonstrate NoRA’s improved convergence, stability, downstream performance, and resistance to catastrophic forgetting.

## 2 PRELIMINARIES

LoRA parameterizes the weight update of a pretrained matrix as

$$
\Delta W = W - W _ { 0 } = \alpha B A ,\tag{1}
$$

where $\pmb { A } \in \mathbb { R } ^ { r \times k } , \pmb { B } \in \mathbb { R } ^ { d \times r }$ , and $r \ll \operatorname* { m i n } ( d , k )$ . Standard LoRA initializes $\pmb { { \cal B } } ^ { ( 0 ) } = \mathbf { 0 }$ to preserve the pretrained model. Let $G = \partial \mathcal { L } / \partial W$ denote the gradient of the corresponding full weight matrix. At initialization, we have the following gradients with respect to B and A:

$$
\left. \frac { \partial \mathcal { L } } { \partial \boldsymbol { B } } \right| _ { t = 0 } = \alpha G ( \boldsymbol { A } ^ { ( 0 ) } ) ^ { \top } , \qquad \left. \frac { \partial \mathcal { L } } { \partial \boldsymbol { A } } \right| _ { t = 0 } = \mathbf { 0 } .\tag{2}
$$

Therefore, the earliest optimization of LoRA is entirely determined by the latent representation induced by the initial projection $A ^ { ( 0 ) }$

<table><tr><td>Method</td><td>Forward Computation</td><td>Initialization / Prior</td></tr><tr><td colspan="3">Full-Parameter and Standard Low-Rank Adaptation</td></tr><tr><td>Full Finetuning</td><td> $y = W _ { 0 } x$ </td><td>N/A</td></tr><tr><td>LoRA (Hu et al., 2021)</td><td> $y = W _ { 0 } x + B A x$ </td><td> $\pmb { A } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) , \pmb { B } = 0$ </td></tr><tr><td>Optimization-Strategy Variants</td><td></td><td></td></tr><tr><td>LoRA+ (Hayou et al., 2024)</td><td> $y = W _ { 0 } x + B A x$ </td><td> $\eta _ { B } = \lambda \eta _ { A }$ </td></tr><tr><td colspan="3">Structured or Constrained Low-Rank Parameterizations</td></tr><tr><td>DoRA (Liu et al., 2024b)</td><td> $y = m \left( { \frac { W _ { 0 } x + B A x } { \| W _ { 0 } + B A \| c } } \right)$ </td><td>Kaiming init, B = 0</td></tr><tr><td>MiSS (Kang &amp; Yin, 2026)</td><td> $y = W _ { 0 } x + \mathrm { e x p a n d } ( D ) x$ </td><td>Fixed structured projection</td></tr><tr><td>AdaLoRA (Zhang et al., 2023b)</td><td> $y = W _ { 0 } x + P \Lambda Q x$ </td><td> $\Lambda = 0 , P , Q \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ </td></tr><tr><td>LoRA-FA (Zhang et al., 2023a)</td><td> $y = W _ { 0 } x + B A x$ </td><td>Frozen random A, B = 0</td></tr><tr><td>VeRA (Ye et al., 2025)</td><td> $y = W _ { 0 } x + \Lambda _ { b } B \Lambda _ { d } A x$ </td><td>Frozen random A, B</td></tr><tr><td colspan="3">Initialization-Based Low-Rank Adaptation</td></tr><tr><td>PiSSA (Meng et al., 2024)</td><td> $y = ( W _ { 0 } - B A ) x + B A x$ </td><td>Top-r SVD initialization</td></tr><tr><td>MiLoRA (Wang et al., 2025)</td><td> $y = ( W _ { 0 } - B A ) x + B A x$ </td><td>Minor-component SVD init</td></tr><tr><td>NoRA-init</td><td> $y = W _ { 0 } x + B A x$ </td><td> $\pmb { A } = \mathrm { N o r m } ( \pmb { A } ) , \pmb { B } = 0$ </td></tr><tr><td>NoRA</td><td> $y = W _ { 0 } x + B \operatorname { N o r m } ( A ) x$ </td><td> $\scriptstyle B = 0$ </td></tr></table>

Table 1: Overview of representative PEFT methods categorized by their primary mechanism, including forward parameterization and initialization strategy.

Motivated by this observation, we consider a unified latent-feature formulation:

$$
\Delta { \boldsymbol { y } } = \alpha B \phi ( { \boldsymbol { x } } ) ,\tag{3}
$$

where $\ b { \phi } ( \ b { x } ) \in \mathbb { R } ^ { r }$ denotes the latent feature presented to the up-projection. Existing low-rank methods can be interpreted as constructing this latent feature in different ways:

$$
\phi ( \pmb { x } ) = \left\{ \begin{array} { l l } { A \pmb { x } , } & { \mathrm { r a n d o m ~ p r o j e c t i o n ~ ( L o R A ) } , } \\ { \pmb { A } _ { \mathrm { i n i t } } \pmb { x } , } & { \mathrm { s t r u c t u r e d ~ p r o j e c t i o n ~ ( e . g . , ~ P i S S A , ~ M i L o R A ) } , } \\ { \mathrm { N o r m } ( A \pmb { x } ) , } & { \mathrm { n o r m a l i z e d ~ p r o j e c t i o n ~ ( e . g . , ~ M L A ) } . } \end{array} \right.\tag{4}
$$

Although these approaches adopt different implementations, they can all be viewed as designing the latent feature presented to the up-projection. This latent-projection perspective motivates us to revisit how to regularize the latent projection without losing the exact mergeability of LoRA.

## 3 NORA: NORMALIZED LOW-RANK ADAPTATION

## 3.1 FROM NORMALIZED LATENT FEATURES TO NORMALIZED LOW-RANK PROJECTION

The normalized latent projection in MLA suggests that the representation presented to the upprojection plays an important role in training dynamics. Specifically, MLA constructs the lowdimensional latent feature as

$$
\phi ( \pmb { x } ) = \mathrm { N o r m } ( A \pmb { x } ) ,\tag{5}
$$

where the normalization operator explicitly regulates the projected feature before it is mapped back to the output space. Directly introducing this operation into LoRA changes the standard linear update from BAx to BNorm(Ax). Since the normalization depends on the input, this transformation becomes nonlinear with respect to x and can no longer be exactly merged into the pretrained weight matrix. We therefore seek to transfer the scale-control effect from the latent feature to the downprojection matrix itself. Modern Transformer architectures are typically pre-normalized, such that the magnitude of the input x to each linear layer is already well controlled by LayerNorm. In this setting, variations in the column norms of A introduce additional scale imbalance in how different input coordinates are projected into the low-rank latent space. We therefore normalize each column of A along the LoRA rank dimension. For $\pmb { A } \in \mathbb { R } ^ { r \times k }$ , we write

$$
\mathbf { \pmb { A } } = [ \pmb { a } _ { 1 } , \pmb { a } _ { 2 } , \dots , \pmb { a } _ { k } ] , \qquad \pmb { a } _ { j } \in \mathbb { R } ^ { r } ,\tag{6}
$$

where $\mathbf { \delta } _ { \mathbf { a } _ { j } }$ denotes the latent projection vector associated with the j-th input coordinate. We define

$$
\operatorname { N o r m } ( A ) = \left[ { \frac { \mathbf { \mathfrak { a } } _ { 1 } } { \operatorname* { m a x } ( \lVert { \mathbf { \mathfrak { a } } } _ { 1 } \rVert _ { 2 } , \epsilon ) } } , \dots , { \frac { \mathbf { \mathfrak { a } } _ { k } } { \operatorname* { m a x } ( \lVert { \mathbf { \mathfrak { a } } } _ { k } \rVert _ { 2 } , \epsilon ) } } \right] ,\tag{7}
$$

![](images/b7cd90ad5f5fbc6b0d65cc8e62c02d6bf1363e76d8f4500440439609e014d7b0.jpg)

![](images/4da27216231d8e68d460b5afabffa159a6e2065f205362133c40722c90f4af29.jpg)  
Figure 1: Illustration and optimization behavior NoRA. Left: Each input-to-latent projection vector is normalized along the LoRA rank dimension. Right: Gradient-norm dynamics of different initialization methods and LoRA ranks on LLaMA-3.2-3B trained on the Math dataset.

where ϵ is a small constant for numerical stability. Thus, we have that

$$
\left\| \left[ \operatorname { N o r m } ( \pmb { A } ) \right] _ { : , j } \right\| _ { 2 } = 1 , \qquad j = 1 , \ldots , k .\tag{8}
$$

Based on this rank-dimension normalization, we introduce normalized low-rank adaptation, which parameterizes the low-rank update as the following form:

$$
\Delta \pmb { y } = \alpha B \mathrm { N o r m } ( \pmb { A } ) \pmb { x } .\tag{9}
$$

Unlike latent-feature normalization in MLA, Norm(·) depends only on the LoRA parameters rather than the input. NoRA therefore imposes a normalization constraint on the down-projection throughout training while remaining linear with respect to x. After training, the normalized down-projection can be absorbed into the low-rank update, preserving exact weight mergeability. In this way, every input coordinate is continuously projected through a unit-norm latent direction, removing arbitrary column-wise scale variation in A. The resulting transition can be summarized as

$$
\begin{array} { r l r } { \underbrace { B \mathrm { N o r m } ( A x ) } _ { \mathrm { \bf { M L A } : i n p u t - d e p e n d e n t l a t e n t n o r m a l i z a t i o n } } } & { \to } & { \underbrace { B \mathrm { N o r m } ( A ) x } _ { \mathrm { \bf { N o R A } : n o r m a l i z e d ~ l o w - r a n k ~ p r o j e c t i o n } } . } \end{array}\tag{10}
$$

Our initialization and gradient-norm analyses further suggest that much of the benefit of NoRA is already established at the beginning of optimization. This motivates a simpler initialization-only variant, which we denote as NoRA-init. Specifically, we initialize

$$
A ^ { ( 0 ) } = \mathrm { N o r m } ( A _ { \mathrm { i n i t } } ) , \qquad B ^ { ( 0 ) } = { \bf 0 } ,\tag{11}
$$

and subsequently optimize A and $\textbf {  { B } }$ using the standard LoRA parameterization without performing normalization throughout training. $A _ { \mathrm { i n i t } }$ denotes the random initialization in LoRA. Despite removing the persistent constraint during training, NoRA-Init retains most of the optimization benefit of NoRA in our experiments. This observation suggests that controlling the scale of the input-to-latent projection at initialization is particularly important for the early optimization of low-rank adaptation.

## 3.2 CONNECTION TO MISS AND BLOCK IDENTITY MATRIX INITIALIZATION

Having established the effectiveness of rank-dimension normalization, we further investigate alternative approaches for achieving this form of normalization. MiSS (Kang & Yin, 2026) is a recent PEFT method that improves parameter efficiency through a matrix-shard sharing strategy. Consider an input $\mathbf { \boldsymbol { x } } \in { \mathbb { R } ^ { k } }$ with $k = b r$ , partitioned into b blocks $\bar { \mathbf { x } } ^ { ( i ) } \in \mathbb { R } ^ { r }$ . The MiSS update is written as

$$
\Delta { y } _ { \mathrm { M i S S } } = \Delta W x = \alpha \cdot \mathrm { e x p a n d } ( B ) x = \alpha B \sum _ { i = 1 } ^ { b } x ^ { ( i ) } = \alpha B \underbrace { \left[ { I } _ { r } , \dots , { I } _ { r } \right] } _ { A _ { \mathrm { f i x } } } \mathbf { x } .\tag{12}
$$

Thus, MiSS is equivalent to a LoRA-style update with a fixed down-projection $\pmb { A } _ { \mathrm { f i x } }$ composed of repeated identity matrices. Importantly, each column of this projection has unit norm:

$$
\begin{array} { r } { \| ( A _ { \mathrm { f i x } } ) _ { : , j } \| _ { 2 } = 1 , \qquad \forall j . } \end{array}\tag{13}
$$

MiSS therefore can be interpreted as a special case of NoRA, as its down-projection matrix is normalized by a block-identity construction $( i . e . , A$ is fixed to the constant $\pmb { A } _ { \mathrm { f i x } } )$ . As shown in Figure 1, NoRA produces substantially stronger early gradients than standard LoRA, with gradient norms approaching those of full finetuning. Interestingly, MiSS also exhibits a highly similar pattern. These results well justifies the empirical effectiveness of NoRA and its special case MiSS.

Motivated by the connection to MiSS, we introduce Block Identity Matrix Initialization (BIMI) as a simple structured realization of NoRA-init. Rather than fixing the block-identity projection throughout training, BIMI uses it only to initialize the LoRA down-projection matrix and allows it to remain fully trainable afterward. Specifically, for $k = b r + q ,$ , where $0 \leq q < r$ , we initialize

$$
\begin{array} { r } { \pmb { A } _ { \mathrm { b i m i } } = \left[ \underbrace { \pmb { I } _ { r } , \pmb { I } _ { r } , \dots , \pmb { I } _ { r } } _ { b \mathrm { b l o c k s } } , \pmb { E } _ { q } \right] \in \mathbb { R } ^ { r \times k } , } \end{array}\tag{14}
$$

where $\pmb { { E } } _ { a } \in \mathbb { R } ^ { r \times q }$ contains the first q columns of ${ { I } _ { r } }$ and is omitted when $q = 0$ . By construction, we have $\lVert ( \dot { A _ { \mathrm { b i m i } } } ) _ { : , j } \rVert _ { 2 } = 1 , \forall j ,$ , making BIMI directly a special case of NoRA-init. The resulting update retains the standard LoRA form, $\bar { \Delta { \boldsymbol { y } } } = \alpha B A { \boldsymbol { x } } ,$ , with both A and B optimized during training.

## 3.3 WHY DOES NORA WORK? A PRECONDITIONING PERSPECTIVE

The early optimization of LoRA is dictated by the initialization of the down-projection matrix A. This is because LoRA can be viewed as performing full-finetuning gradient descent under a hidden preconditioner that is determined by A and acts on the input coordinates. Let $G = \partial \mathcal { L } / \partial W \in \mathbb { R } ^ { d \times k }$ be the full-finetuning gradient. A gradient step of size $\eta$ on B at initialization is $\Delta B \stackrel { \cdot } { = } - \eta \alpha G A ^ { \top }$ while A receives no gradient and stays fixed. The induced change of the merged weight is therefore

$$
\Delta W = \alpha \Delta B A = - \eta G P , \qquad P = \alpha ^ { 2 } A ^ { \top } A \in \mathbb { R } ^ { k \times k } .\tag{15}
$$

Full finetuning takes the step $\Delta W = - \eta G , i . e .$ ., equation 15 with $P = I .$ Hence, at initialization, and approximately for as long as B remains small (the neglected term $\alpha B \Delta A = - \eta \alpha ^ { 2 } B B ^ { \top } G$ is second order in $B )$ , LoRA is full finetuning with the gradient right-multiplied by a positive semidefinite matrix of rank r that acts on the input coordinates. Right-multiplication by a $k \times k$ matrix is exactly where curvature-based methods (Martens & Grosse, 2015) place an input-side preconditioner. Moreover, we observe that column norms are per-coordinate learning rates:

$$
\Delta { W } _ { : , j } = - \eta \Big [ \underbrace { \alpha ^ { 2 } \lVert { \pmb a } _ { j } \rVert _ { 2 } ^ { 2 } { \pmb G } _ { : , j } } _ { \mathrm { c o o r d i n a t e ~ } j \mathrm { ~ s ~ o w n ~ u p d a t e } } + \underbrace { \sum _ { i \neq j } \alpha ^ { 2 } ( \pmb a _ { i } ^ { \top } { \pmb a } _ { j } ) { \pmb G } _ { : , i } } _ { \mathrm { c r o s s t a l k ~ f r o m ~ t h e ~ r a n k ~ b o t l e n e c k } } \Big ] .\tag{16}
$$

The first term is the full-finetuning update of the weight column with respect to input coordinate $j ,$ scaled by the squared length of its latent $\mathbf { \delta } _ { \mathbf { a } _ { j } }$ . The second term is crosstalk between input coordinates induced by the rank bottleneck, and its coefficients are inner products of latent directions.

Random initialization of A makes learning rates unbalanced at each coordinate. For example, when $A ^ { ( 0 ) }$ has zero-mean i.i.d. entries of variance $\sigma ^ { 2 } \propto 1 / k , \mathbb { E } \| \pmb { a } _ { j } \| _ { 2 } ^ { 2 } = r \sigma ^ { 2 } \propto r / k \ll 1$ . Therefore, at any moderate $\alpha ,$ , every learning rate is far too small, so gradient flows through the adapter at a small fraction of the full-finetuning rate. This is empirically verified by the small gradient norm of LoRA in Figure 1. The learning rates are also unbalanced. $\| \boldsymbol { a } _ { j } \| _ { 2 } ^ { 2 }$ fluctuates around its mean with relative spread of order $1 / { \sqrt { r } } _ { \mathrm { { } } }$ , so at small rank each coordinate gets its learning rate randomly, unrelated to data or curvature. A preconditioner should remove distortion, not add its own.

Normalization is exactly the fix. Setting every $\| \pmb { a } _ { j } \| _ { 2 } = 1$ at unit scaling (rank-dimension normalization in NoRA), $\alpha = 1$ (equivalently $\alpha = r$ under the $\alpha / r$ implementation convention), fixes the problems. Specifically, this leads to $\operatorname { D i a g } ( P ) = I$ deterministically, $\mathbb { E } [ P ] = I$ over the random directions, and the adapter’s gradient norm can match full finetuning’s gradient norm in expectation, independently of $r .$ Therefore, we can get the flat NoRA curve in Figure 1. In contrast, row normalization $( \mathrm { N o r m } _ { k } )$ leaves $\operatorname { D i a g } ( P )$ random of order $r / k$ , and brings no empirical gain (see Table 3). Moreover, BIMI uses unit basis vectors as columns (the same diagonal through a completely different crosstalk pattern) and performs similarly to rank-dimension normalization of random initializations (see Table 3), which points to the diagonal of $_ { P }$ as the decisive quantity.

Initialization does most of the work; the constraint adds stability. Since $A \mathrm { { } ^ { \circ } s }$ gradient, $\alpha B ^ { \top } G$ vanishes with $B , A ^ { ( 0 ) }$ alone fixes the learning rates and the input subspace during the decisive early phase, correcting the rates once at initialization thus captures most of the benefit. This is exactly what NoRA-init does. Performing Norm(A) in the forward pass adds two things: (1) the loss becomes invariant to each column’s scale, so gradients are tangential and, under gradient descent, norms can only grow while the effective step anneals automatically, as in weight-normalized training (Liu et al.,

<table><tr><td>Stage</td><td>Repository</td><td>Model</td><td>Train Data</td><td>Benchmark</td></tr><tr><td>Pretrain</td><td>FLA (Yang &amp; Zhang, 2024)</td><td>MLA (Liu et al., 2024a) MHA (Vaswani et al., 2017)</td><td>SlimPajama (Sobol- eva et al., 2023)</td><td>LAMBADA (Paperno et al., 2016) WikiText (Merity et al., 2016) Arc-Easy/Arc-Challenge (Clark et al., 2018) HellaSwag (Zellers et al., 2019) PIQA (Bisk et al., 2020) OpenBookQA (Mihaylov et al., 2018)</td></tr><tr><td>SFT</td><td>et al., 2026)</td><td>PEFT-Arena (Huang</td><td>Llama-3.2- 3B (Grattafiori et al., 2024)</td><td>MetaMath (Yu et al., 2024) CodeFeedback (Zheng et al., 2024)</td><td>Math: GSM8k (Cobbe et al., 2021) Math500 (Yu et al., 2024) Code: HumanEval (Chen et al., 2021) Mbpp (Austin et al., 2021)</td></tr><tr><td>RL</td><td>2025)</td><td>PeRL (Yin et al.,</td><td>DeepSeek-R1-Distill- Qwen1.5B (Liu et al., 2024a)</td><td>open-r1/DAPO-Math- 17k-Processed (Yu et al., 2026)</td><td>AIME24 (Zhang &amp; Math-AI, 2025) (Avg@32) AIME25 (Zhang &amp; Math-AI, 2025) (Avg@32) MATH500 (Lightman et al., 2024) (Avg@4) Minerva (Lewkowycz et al., 2022) (Avg@4) AMC (Li et al., 2024) (Avg@32) HMMT (Balunović et al., 2025) (Avg@32)</td></tr></table>

Table 2: Overview of all the training settings in our experiments.

2017; Salimans & Kingma, 2016; Liu et al., 2018); and (2) Diag(P) = I is enforced for the whole run on a compact set of directions, so the latent scale can neither collapse nor explode and the update stays linear in x and can be merged exactly. To summarize, NoRA sets the diagonal gains of LoRA’s hidden preconditioner to one, and keeps them throughout training.

## 4 EXPERIMENTS AND RESULTS

To thoroughly evaluate our method, we conduct experiments across pretraining, supervised finetuning (SFT), and reinforcement learning. We further perform ablation studies to investigate the effect of the normalization dimension and initialization distribution. Evaluation settings are in Table 2.

Ablation Study. We investigate the effect of the normalization dimension of the down-projection matrix A by comparing column-wise and row-wise normalization, and further evaluate the effectiveness of NoRA across different random or deterministic initialization schemes.

Pretraining. We begin with controlled studies on the MLA architecture under the L24-D1024 setting to examine the effect of intermediate normalization within low-rank latent representations. Building on these observations, we further evaluate NoRA on standard multi-head attention (MHA), where all attention projection layers are adapted with NoRA.

Supervised finetuning. We evaluate NoRA on Llama 3.2-3B across a diverse set of downstream tasks, including mathematical reasoning and code generation, as well as its robustness against catastrophic forgetting (Huang et al., 2026). More detailed settings are given in Appendix B.

Reinforcement learning with verifiable rewards. To assess its effectiveness in post-training optimization, we additionally evaluate NoRA under reinforcement learning using the ReRL framework.

## 4.1 EFFECT OF THE NORMALIZATION DIMENSION

We first study whether the effect of the normalization dimension under different initialization schemes. Given $\textbf { \textit { A } } \in \mathbb { R } ^ { r \times k }$ , Norm normalizes each row, while Norm normalizes each column $A _ { : , j } ~ \in ~ \mathbb { R } ^ { r }$ As shown in Table 3, Norm provides little improvement, whereas Norm consistently improves Random Uniform, Gaussian, and Kaiming Uniform initializations. After applying NoRA, all three initialization schemes achieve similar performance, indicating that its benefit is largely independent of the initialization distribution. BIMI achieves comparable performance and satisfies NoRA by construction, serving as a structured and deterministic instance of NoRA.

<table><tr><td>Initialization</td><td>Method</td><td>GSM8K</td><td>Math</td><td>Avg.</td></tr><tr><td rowspan="3">u(−1/√k,1/√k)</td><td>None</td><td>47.68</td><td>10.86</td><td>29.27</td></tr><tr><td>Normk</td><td>48.67</td><td>10.82</td><td>29.75</td></tr><tr><td>Normr</td><td>60.12</td><td>14.20</td><td>37.16</td></tr><tr><td rowspan="3"> $\mathcal { N } ( 0 , 1 / r ^ { 2 } )$ </td><td>None</td><td>50.41</td><td>11.68</td><td>31.05</td></tr><tr><td>Normk</td><td>49.73</td><td>11.34</td><td>30.54</td></tr><tr><td>Normr</td><td>59.96</td><td>13.93</td><td>36.95</td></tr><tr><td rowspan="3">U(−1,1)</td><td>None</td><td>48.19</td><td>11.02</td><td>29.61</td></tr><tr><td>Normk</td><td>48.29</td><td>10.86</td><td>29.58</td></tr><tr><td>Norm</td><td>58.98</td><td>14.34</td><td>36.66</td></tr><tr><td> $[ \pmb { I } _ { r } , \dots , \pmb { I } _ { r } ]$ </td><td>BIMI</td><td>59.89</td><td>14.24</td><td>37.07</td></tr></table>

Table 3: Effect of different normalization dimensions under supervised finetuning. The average is computed over accuracies on GSM8K and Math.

<table><tr><td>Model &amp; Scale</td><td>Parameters</td><td>Lamb.† ppl↓</td><td>Wiki. ppl↓</td><td>Lamb. acc↑</td><td>ARCe acc↑</td><td>ARCc accn↑</td><td>Hella. accn↑</td><td>PIQA</td><td>Wino.</td><td>OBQA acc↑</td><td>Avg.</td></tr><tr><td colspan="8">MLA with 10B training tokens and 0.5M batchsize tokens</td><td>acc↑</td><td>acc↑</td><td></td><td></td><td>acc↑</td></tr><tr><td>BNorm(Ax)</td><td>342.4M</td><td>48.73</td><td>32.24</td><td>30.62</td><td>56.99</td><td>27.56</td><td>36.26</td><td>63.49</td><td>51.30</td><td>22.00</td><td>41.17</td></tr><tr><td>BAx</td><td>342.4M</td><td>61.83</td><td>32.89</td><td>28.70</td><td>54.63</td><td>26.02</td><td>35.88</td><td>63.66</td><td>51.62</td><td>22.00</td><td>40.36</td></tr><tr><td> $B A _ { \mathrm { N o R A - i n i t } } { \pmb x }$ </td><td>342.4M</td><td>50.77</td><td>32.42</td><td>29.87</td><td>54.04</td><td>26.37</td><td>36.11</td><td>63.76</td><td>52.72</td><td>21.60</td><td>40.64</td></tr><tr><td colspan="10">MHA with 10B training tokens and 0.5M batchsize tokens</td><td></td></tr><tr><td>Wx</td><td>348.7M</td><td>47.95</td><td>31.31</td><td>30.55</td><td>54.59</td><td>26.45</td><td>36.95</td><td>65.07</td><td>51.22</td><td>22.60</td><td>41.06</td></tr><tr><td>BAx</td><td>289.3M</td><td></td><td></td><td>0.00</td><td>25.17</td><td>28.07</td><td>25.97</td><td>49.78</td><td>49.49</td><td>16.20</td><td>27.81</td></tr><tr><td> $B A _ { \mathrm { N o R A - i n i t } } { \pmb x }$ </td><td>289.3M</td><td>63.45</td><td>34.39</td><td>28.29</td><td>54.38</td><td>26.02</td><td>35.10</td><td>63.49</td><td>50.91</td><td>19.60</td><td>39.69</td></tr></table>

Table 4: Zero-shot performance of 340M models trained on SlimPajama (Soboleva et al., 2023). Commonsense reasoning tasks are evaluated with lm-evaluation-harness (Gao et al., 2024); the recall-intensive task follows prefix-linear-attention (Arora et al., 2024) with 2K input tokens. “-” indicates a collapse in perplexity.

<table><tr><td rowspan="2">Method</td><td colspan="6">Supervised Finetuning</td><td colspan="4">Forgetting</td></tr><tr><td>Trainable</td><td>GSM8K</td><td>Math</td><td>HumanEval</td><td>MBPP</td><td> $\mathbf { A v } \mathbf { g }$ </td><td>MMLU</td><td>AGIEval</td><td>ARC-C</td><td>Avg ∆</td></tr><tr><td>Base Model</td><td>-</td><td></td><td>一</td><td></td><td>一</td><td></td><td> $5 5 . 1 0 _ { + 0 . 0 0 }$ </td><td> $2 3 . 9 9 _ { + 0 . 0 0 }$ </td><td> $4 2 . 9 2 _ { + 0 . 0 0 }$ </td><td>0.00</td></tr><tr><td>Full</td><td>3B</td><td>65.12</td><td>17.96</td><td>36.15</td><td>49.05</td><td>42.07</td><td> $5 4 . 1 7 _ { - 0 . 9 3 }$ </td><td> $2 6 . 2 5 _ { + 2 . 2 6 }$ </td><td> $4 1 . 6 4 _ { - 1 . 2 8 }$ </td><td>+0.02</td></tr><tr><td>PiSSA</td><td>48.6M</td><td>54.66</td><td>12.08</td><td>39.00</td><td>55.30</td><td>40.26</td><td> $5 4 . 4 3 _ { - 0 . 6 7 }$ </td><td> $2 3 . 9 9 _ { + 0 . 0 0 }$ </td><td> $4 2 . 7 5 _ { - 0 . 1 7 }$ </td><td>-0.28</td></tr><tr><td>OFT</td><td>53.7M</td><td>56.10</td><td>13.02</td><td>38.40</td><td>54.20</td><td>40.43</td><td>54.72-0.38</td><td> $2 5 . 2 3 _ { + 1 . 2 5 }$ </td><td>43.60+0.68</td><td>+0.52</td></tr><tr><td>RSLoRA</td><td>48.6M</td><td>57.05</td><td>12.32</td><td>39.65</td><td>56.10</td><td>41.28</td><td> $5 4 . 0 7 _ { - 1 . 0 3 }$ </td><td>24.58+0.60</td><td>41.81-1.11</td><td>-0.51</td></tr><tr><td>MiSS</td><td>49.5M</td><td>60.80</td><td>14.60</td><td>39.60</td><td>55.80</td><td>42.70</td><td> $5 3 . 4 0 _ { - 1 . 5 0 }$ </td><td>24.33+0.34</td><td>41.98-0.94</td><td>-0.70</td></tr><tr><td colspan="9">LoRA-style Parameterization</td><td></td></tr><tr><td>LoRA</td><td>48.6M</td><td>50.94</td><td>10.38</td><td>37.20</td><td>53.20</td><td>37.93</td><td> ${ \bar { \bf s } } 4 . 2 7 _ { - 0 . 8 3 }$ </td><td>24.40+0.42</td><td>41.64-1.28</td><td>-0.56</td></tr><tr><td>NoRA-init</td><td>48.6M</td><td>59.89</td><td>14.24</td><td>39.60</td><td>55.80</td><td>42.38</td><td> $5 4 . 0 9 _ { - 1 . 0 1 }$ </td><td> $2 5 . 1 0 _ { + 1 . 1 1 }$ </td><td>42.24-0.68</td><td>-0.19</td></tr><tr><td>NoRA</td><td>48.6M</td><td>61.63</td><td>14.46</td><td>42.10</td><td>55.30</td><td>43.37</td><td> $5 4 . 0 0 _ { - 1 . 1 0 }$ </td><td> $2 5 . 3 2 _ { + 1 . 3 3 }$ </td><td> $\mathbf { 4 2 . 7 5 _ { - 0 . 1 7 } }$ </td><td>+0.02</td></tr><tr><td colspan="9">DoRA-style Parameterization</td><td></td></tr><tr><td>DoRA</td><td>49.4M</td><td>51.63</td><td>11.36</td><td>37.80</td><td>52.40</td><td>38.30</td><td> ${ \pmb 5 4 . 2 0 } _ { - 0 . 9 0 }$ </td><td> $2 4 . 0 9 _ { + 0 . 1 0 }$ </td><td> $4 1 . 8 1 _ { - 1 . 1 1 }$ </td><td>-0.64</td></tr><tr><td>NoRA-init</td><td>49.4M</td><td>59.59</td><td>14.10</td><td>38.70</td><td>53.20</td><td>41.40</td><td> $5 3 . 6 5 _ { - 1 . 4 5 }$ </td><td> $2 4 . 8 4 _ { + 0 . 8 6 }$ </td><td> $4 2 . 3 2 _ { - 0 . 6 0 }$ </td><td>-0.40</td></tr><tr><td>NoRA</td><td>49.4M</td><td>61.33</td><td>14.44</td><td>42.10</td><td>55.60</td><td>43.34</td><td> $5 4 . 0 0 _ { - 1 . 1 0 }$ </td><td> $2 4 . 5 9 _ { + 0 . 5 8 }$ </td><td> $4 2 . 1 5 _ { - 0 . 7 7 }$ </td><td>-0.43</td></tr></table>

Table 5: Comparison of PEFT methods under supervised finetuning and retained benchmark settings.

## 4.2 LLM PRETRAINING

We first conduct LLM pretraining experiments using the standard MLA architecture. Normalizing the latent representation Ax consistently improves model performance and accelerates convergence during pretraining. Motivated by these results, we extend the same normalization principle to the linear low-rank formulation by applying normalization to the down-projection A, yielding NoRA. To evaluate the effectiveness of NoRA for LLM pretraining, we adopt NoRA-init and compare it with the standard low-rank parameterization. As shown in Table 4, NoRA-init achieves faster convergence and better downstream performance. Although its performance remains slightly below that of directly normalizing Ax, NoRA-init preserves the linear structure BAx, thereby retaining the parameter mergeability and deployment efficiency of conventional low-rank parameterizations.

We further evaluate NoRA-init under the standard MHA architecture for LLM pretraining. Specifically, we parameterize all linear projections in the attention module, including ${ q _ { \mathrm { p r o j } } } , k _ { \mathrm { p r o j } } , \boldsymbol { v } _ { \mathrm { p r o j } }$ , and $\pmb { o } _ { \mathrm { p r o j } }$ , using low-rank factors with rank r = 64. As shown in Table 4, the standard low-rank parameterization exhibits a severe performance collapse on LAMBADA, along with substantial degradation on several other benchmarks. An examination of the optimization dynamics reveals that its gradient norms diminish to abnormally low levels during training, suggesting insufficient optimization. In contrast, NoRA-init achieves significantly better performance than standard low-rank parameterization and preserves well-scaled gradient magnitudes throughout training, thereby avoiding such optimization collapse and yielding more stable loss trajectories and downstream performance.

This observation is related to the capacity-limited behavior of low-rank adaptation discussed in Schulman & Lab (2025), where insufficient low-rank capacity can largely constrain the flexibility of optimization trajectory. Our results further show that capacity alone does not fully determine optimization behavior. Under the same rank constraint, modifying the initialization of the low-rank projection substantially changes the training dynamics. This suggests that controlling the scale and structure of the input-to-latent projection provides a more favorable optimization geometry under limited-rank parameterizations, thereby alleviating performance degradation in pretraining.

![](images/674273da6f16e8208ee306c1cf7ae67fb68807f015c1d4991a840419a5adcb21.jpg)

![](images/1c287e9bf6fddf1f7694fe39355e814edd7b8a108aab9cabe65657a710ae508f.jpg)  
Figure 2: Training dynamics comparison of full finetuning, LoRA, PiSSA, and NoRA on the SFT task.

## 4.3 SUPERVISED FINETUNING

Table 5 compares NoRA with representative PEFT methods under supervised finetuning. NoRA achieves the best overall SFT performance among the compared methods, improving the average score from 37.93 with standard LoRA to 43.37, a gain of 5.44 points. In particular, NoRA substantially improves performance on GSM8K and HumanEval, reaching 61.63 and 42.10, respectively, while also outperforming PiSSA (Meng et al., 2024), OFT (Qiu et al., 2023; 2025; Liu et al., 2024c), RSLoRA (Kalajdzievski, 2023), and MiSS (Kang & Yin, 2026) in average performance. These results demonstrate the effectiveness of rank-dimension normalization for improving LoRA.

We further compare NoRA with its initialization-only variant, NoRA-init. NoRA-init applies normalization constraint only to the initial down-projection and then follows standard LoRA training, whereas NoRA continuously enforces the normalization constraint throughout training. NoRA-ini already improves the SFT average from 37.93 to 42.38, capturing a large portion of the gain achieved by NoRA. With the normalization constraint maintained throughout training, NoRA further im proves the average score to 43.37. The comparison indicates that controlling the input-to-latent projection magnitude at initialization accounts for a large portion of the gain, while persistent normalization provides additional improvements. Comparison of training dynamics is given in Figure 2. NoRA shows stronger learnability than LoRA without introducing extra parameters.

NoRA also outperforms MiSS, which achieves an average score of 42.70. Recall that MiSS can be reformulated as a fixed block-identity down-projection satisfying the normalization constraint. The strong performance of both MiSS and NoRA-init therefore provides further evidence that normalized low-rank projections play an important role in their optimization. Notably, NoRA-init achieves comparable performance to MiSS while keeping the down-projection fully trainable, avoiding the constraint of fixing the projection throughout training.

Finally, we examine whether the NoRA principle generalizes beyond the standard LoRA parameterization. We apply Norm -based normalization to DoRA and obtain consistent improvements: NoRAinit-DoRA improves the SFT average from 38.30 for DoRA to 41.40. This result demonstrates that the benefit of rank-dimension normalization is not specific to LoRA, but can also be transferred to other low-rank adaptation variants. Together with the improvements observed for LoRA, this suggests that NoRA captures a general optimization principle for low-rank adaptation. NoRA also exhibits favorable knowledge retention, achieving an average change of +0.02 on MMLU, AGIEval, and ARC-C, compared with −0.56 for LoRA and -0.70 for MiSS. Thus, NoRA improves adaptation performance while maintaining the pretrained model’s retained knowledge.

## 4.4 RLVR ON MATHEMATICAL REASONING

We further evaluate NoRA under reinforcement learning with verifiable rewards (RLVR) on challenging mathematical reasoning benchmarks. As shown in Table 6, standard LoRA improves the overall average score from 41.0 for the base model to 42.8, while NoRA further increases it to 44.4, outperforming the base model and standard LoRA by 3.4 and 1.6 points, respectively. NoRA yields broad improvements across the evaluated benchmarks, with particularly clear gains on AMC,

<table><tr><td rowspan="2">Method</td><td colspan="2">AIME24@32</td><td colspan="2">AIME25@32</td><td colspan="2">AMC@32</td><td colspan="2">HMMT@32</td><td colspan="2">MATH500@4</td><td colspan="2">Minerva@4</td><td rowspan="2">Avg.</td></tr><tr><td>Avg.</td><td>Pass</td><td>Avg.</td><td>Pass</td><td>Avg.</td><td>Pass</td><td>Avg.</td><td>Pass</td><td>Avg.</td><td>Pass</td><td>Avg.</td><td>Pass</td></tr><tr><td>Base</td><td>24.3</td><td>70.0</td><td>20.4</td><td>53.3</td><td>64.8</td><td>95.0</td><td>9.9</td><td>36.7</td><td>80.9</td><td>92.8</td><td>27.8</td><td>43.0</td><td>41.0</td></tr><tr><td>MiLoRA</td><td>4.2</td><td>6.7</td><td>0.0</td><td>0.0</td><td>19.6</td><td>47.5</td><td>0.0</td><td>0.0</td><td>44.5</td><td>63.4</td><td>11.7</td><td>19.9</td><td>18.0</td></tr><tr><td>PiSSA</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.6</td><td>1.0</td><td>0.1</td><td>0.4</td><td>0.2</td></tr><tr><td>LoRA</td><td>28.0</td><td>60.0</td><td>23.1</td><td>53.3</td><td>68.0</td><td>97.5</td><td>10.3</td><td>35.0</td><td>81.3</td><td>92.2</td><td>30.1</td><td>46.3</td><td>42.8</td></tr><tr><td>NoRA</td><td>26.7</td><td>66.7</td><td>23.5</td><td>46.7</td><td>72.7</td><td>97.5</td><td>10.3</td><td>35.0</td><td>84.5</td><td>92.8</td><td>32.0</td><td>47.8</td><td>44.4</td></tr></table>

Table 6: Comparison of accuracy and pass rate on the RLVR task. All numbers are reported in percentages.

MATH500, and Minerva. These results suggest that rank-dimension normalization remains effective under RLVR and facilitates more effective optimization of low-rank adapters.

A notable difference emerges for initialization methods based on pretrained-weight spectral decomposition. Although PiSSA (Meng et al., 2024) and MiLoRA (Wang et al., 2025) are effective under supervised finetuning, their performance degrades substantially in the RLVR setting. This behavior is consistent with prior observations that spectral initialization methods can suffer from optimization instability during reinforcement learning (Yin et al., 2025). In contrast, NoRA remains stable and consistently improves the overall performance of standard LoRA without relying on singular-value decomposition of the pretrained weights. These results highlight the robustness of NoRA across different optimization regimes. Unlike PiSSA and MiLoRA, NoRA does not depend on the spectral structure of pretrained weights, while preserving the lightweight low-rank parameterization and deployment efficiency of standard LoRA. Together with the supervised finetuning results, the RLVR experiments suggest that controlling the scale of the input-to-latent projection provides a simple and broadly effective mechanism for improving low-rank optimization.

## 5 RELATED WORK AND CONCLUDING REMARKS

Parameter-efficient finetuning. As LLMs continue to grow in scale and capability, there has been increasing interest in adapting them to downstream tasks in a parameter-efficient manner (Houlsby et al., 2019; Aghajanyan et al., 2020; Hu et al., 2021; Edalati et al., 2022; Wang et al., 2022; Gheini et al., 2021; Zaken et al., 2022; Guo et al., 2020; Sung et al., 2021; Ansell et al., 2022; Lester et al., 2021; Li & Liang, 2021; Vu et al., 2022; He et al., 2021; Mao et al., 2021; Karimi Mahabadi et al., 2021; Liu et al., 2022; Sung et al., 2022; Chen et al., 2023; Jia et al., 2022; Chen et al., 2022; Zhang et al., 2022; Jie & Deng, 2023; Lian et al., 2022; Qiu et al., 2023; 2025; Liu et al., 2024c; Wu et al., 2024; Zi et al., 2023). Among them, LoRA (Hu et al., 2021) is widely adopted due to its simplicity, efficiency, and mergeability at inference time. Subsequent variants improve LoRA from different perspectives, such as adaptive rank allocation (Zhang et al., 2023b), low-bit quantization (Dettmers et al., 2023) and weight-decomposed low-rank updates (DoRA) (Liu et al., 2024b). While these methods improve the capacity, efficiency, or stability of low-rank adaptation, our work focuses on a complementary question: how the low-rank subspace should be initialized for better optimization.

LoRA initialization and optimization. Recent studies show that LoRA is sensitive to initialization, scaling, and early-stage optimization dynamics. PiSSA initializes LoRA from principal singular components of pretrained weights (Meng et al., 2024), OLoRA (Buy¨ ukaky¨ uz¨ , 2024) adopts orthogonal initialization, LoRA-GA (Wang et al., 2024) uses gradient information to guide initialization, EVA leverages activation statistics, and rsLoRA improves stability through rank-dependent scaling (Kalajdzievski, 2023). These methods demonstrate the importance of initialization, but often rely on SVD, gradient estimation, activation statistics, or scaling heuristics. In contrast, NoRA requires no additional data, SVD, or test-time overhead while preserving the mergeability of LoRA.

Concluding remarks. This work shows that LoRA’s effectiveness depends not only on its rank, but also critically on the geometry and scale of its down-projection. Inspired by the normalized latent representations in MLA, we ask whether the benefits of latent normalization can be transferred to LoRA without sacrificing its linear structure and exact mergeability. This leads to NoRA, which normalizes the down-projection along the rank dimension and thereby controls the scale of the inputto-latent mapping. From a preconditioning perspective, LoRA can be viewed as full finetuning under an implicit low-rank, input-side preconditioner determined by the down-projection; NoRA’s rankdimension normalization corrects undesirable scale imbalance in this preconditioner and yields a better-conditioned optimization geometry. NoRA can consistently improve convergence, training stability and downstream performance across pretraining, supervised finetuning, and RLVR.

## REFERENCES

Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. arXiv preprint arXiv:2012.13255, 2020. 9

Alan Ansell, Edoardo Ponti, Anna Korhonen, and Ivan Vulic. Composable sparse fine-tuning for ´ cross-lingual transfer. In ACL, 2022. 9

Simran Arora, Aman Timalsina, Aaryan Singhal, Benjamin Spector, Sabri Eyuboglu, Xinyi Zhao, Ashish Rao, Atri Rudra, and Christopher Re. Just read twice: closing the recall gap for recurrent ´ language models. arXiv preprint arXiv:2407.05483, 2024. 7

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021. 6

Mislav Balunovic, Jasper Dekoninck, Ivo Petrov, Nikola Jovanovi´ c, and Martin Vechev. Matharena:´ Evaluating llms on uncontaminated math competitions. arXiv preprint arXiv:2505.23281, 2025. 6

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In AAAI, 2020. 6

Kerim Buy¨ ukaky¨ uz. Olora: Orthonormal low-rank adaptation of large language models.¨ arXiv preprint arXiv:2406.01775, 2024. 9

Jiaao Chen, Aston Zhang, Xingjian Shi, Mu Li, Alex Smola, and Diyi Yang. Parameter-efficient fine-tuning design spaces. In ICLR, 2023. 9

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. 6

Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, and Ping Luo. Adaptformer: Adapting vision transformers for scalable visual recognition. In NeurIPS, 2022. 9

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018. 6

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 6

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. In NeurIPS, 2023. 9

Ali Edalati, Marzieh Tahaei, Ivan Kobyzev, Vahid Partovi Nia, James J Clark, and Mehdi Rezagholizadeh. Krona: Parameter efficient tuning with kronecker adapter. arXiv preprint arXiv:2212.10650, 2022. 9

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/12608602. 7

Mozhdeh Gheini, Xiang Ren, and Jonathan May. Cross-attention is all you need: Adapting pretrained transformers for machine translation. In EMNLP, 2021. 9

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 6

Demi Guo, Alexander M Rush, and Yoon Kim. Parameter-efficient transfer learning with diff pruning. arXiv preprint arXiv:2012.07463, 2020. 9

Soufiane Hayou, Nikhil Ghosh, and Bin Yu. Lora+: Efficient low rank adaptation of large models. arXiv preprint arXiv:2402.12354, 2024. 3

Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham Neubig. Towards a unified view of parameter-efficient transfer learning. arXiv preprint arXiv:2110.04366, 2021. 9

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In ICML, 2019. 9

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 1, 3, 9

Yangyi Huang, Ruotian Peng, Zeju Qiu, Jiale Kang, Yandong Wen, Bernhard Scholkopf, and¨ Weiyang Liu. Peft-arena: Understanding parameter-efficient finetuning from a stability-plasticity perspective. arXiv preprint arXiv:2605.28819, 2026. 6

Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In ECCV, 2022. 9

Shibo Jie and Zhi-Hong Deng. Fact: Factor-tuning for lightweight adaptation on vision transformer. In AAAI, 2023. 9

Damjan Kalajdzievski. A rank stabilization scaling factor for fine-tuning with lora. arXiv preprint arXiv:2312.03732, 2023. 1, 8, 9

Jiale Kang and Qingyu Yin. Miss: Revisiting the trade-off in lora with an efficient shard-sharing structure. In ICLR, 2026. 3, 4, 8

Rabeeh Karimi Mahabadi, James Henderson, and Sebastian Ruder. Compacter: Efficient low-rank hypercomplex adapter layers. In NeurIPS, 2021. 9

Dawid Kopiczko, Tijmen Blankevoort, and Yuki Asano. Vera: Vector-based random matrix adaptation. In ICLR, 2024. 1

Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In EMNLP, 2021. 9

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. In NeurIPS, 2022. 6

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9):9, 2024. 6

Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In ACL, 2021. 9

Dongze Lian, Daquan Zhou, Jiashi Feng, and Xinchao Wang. Scaling & shifting your features: A new baseline for efficient model tuning. In NeurIPS, 2022. 9

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In ICLR, 2024. 6

Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture of-experts language model. arXiv preprint arXiv:2405.04434, 2024a. 6

Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. In NeurIPS, 2022. 9

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. Dora: Weight-decomposed low-rank adaptation. arXiv preprint arXiv:2402.09353, 2024b. 1, 3, 9

Weiyang Liu, Yan-Ming Zhang, Xingguo Li, Zhiding Yu, Bo Dai, Tuo Zhao, and Le Song. Deep hyperspherical learning. In NeurIPS, 2017. 1, 5

Weiyang Liu, Zhen Liu, Zhiding Yu, Bo Dai, Rongmei Lin, Yisen Wang, James M Rehg, and Le Song. Decoupled networks. In CVPR, 2018. 1, 6

Weiyang Liu, Zeju Qiu, Yao Feng, Yuliang Xiu, Yuxuan Xue, Longhui Yu, Haiwen Feng, Zhen Liu, Juyeon Heo, Songyou Peng, Yandong Wen, Michael J. Black, Adrian Weller, and Bernhard Scholkopf. Parameter-efficient orthogonal finetuning via butterfly factorization. In¨ ICLR, 2024c. 8, 9

Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, and Boris Ginsburg. ngpt: Normalized transformer with representation learning on the hypersphere. In ICLR, 2025. 1

Yuning Mao, Lambert Mathias, Rui Hou, Amjad Almahairi, Hao Ma, Jiawei Han, Wen-tau Yih, and Madian Khabsa. Unipelt: A unified framework for parameter-efficient language model tuning. arXiv preprint arXiv:2110.07577, 2021. 9

James Martens and Roger Grosse. Optimizing neural networks with kronecker-factored approximate curvature. In ICML, 2015. 5

Fanxu Meng, Zhaohui Wang, and Muhan Zhang. Pissa: Principal singular values and singular vectors adaptation of large language models. In NeurIPS, 2024. 1, 3, 8, 9

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016. 6

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018. 6

Denis Paperno, German Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi,´ Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The lambada dataset:´ Word prediction requiring a broad discourse context. In ACL, 2016. 6

Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, and Bernhard Scholkopf. Controlling text-to-image diffusion by orthogonal finetuning.¨ In NeurIPS, 2023. 8, 9

Zeju Qiu, Weiyang Liu, Adrian Weller, and Bernhard Scholkopf. Orthogonal finetuning made scal- ¨ able. In EMNLP, 2025. 8, 9

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications ofthe ACM, 64(9):99–106, 2021. 6

Tim Salimans and Durk P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In NeurIPS, 2016. 6

John Schulman and Thinking Machines Lab. Lora without regret. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20250929. https://thinkingmachines.ai/blog/lora/. 7

Zhennan Shen, Yanshu Li, Qingyu Yin, Chak Tou Leong, Zhilin Wang, Yanxu Chen, Rongduo Han, Sunbowen Lee, and Yi R Fung. On the geometry of on-policy distillation. arXiv preprint arXiv:2606.07082, 2026. 1

Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama, 2023. 6, 7

Yi-Lin Sung, Varun Nair, and Colin A Raffel. Training neural networks with fixed sparse masks. NeurIPS, 2021. 9

Yi-Lin Sung, Jaemin Cho, and Mohit Bansal. Lst: Ladder side-tuning for parameter and memory efficient transfer learning. In NeurIPS, 2022. 9

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural informa tion processing systems, 30, 2017. 6

Tu Vu, Brian Lester, Noah Constant, Rami Al-Rfou, and Daniel Cer. Spot: Better frozen model adaptation through soft prompt transfer. In ACL, 2022. 9

Hanqing Wang, Yixia Li, Shuo Wang, Guanhua Chen, and Yun Chen. Milora: Harnessing minor singular components for parameter-efficient llm finetuning. In NAACL-HLT, 2025. 1, 3, 9

Shangshang Wang, Julian Asilis, Omer Faruk Akg<sup>¨</sup> ul, Enes Bilgin, Ollie Liu, and Willie Neiswanger.¨ Tina: Tiny reasoning models via lora. In ICLR, 2026. 1

Shaowen Wang, Linxi Yu, and Jian Li. Lora-ga: Low-rank adaptation with gradient approximation. In NeurIPS, 2024. 9

Yaqing Wang, Subhabrata Mukherjee, Xiaodong Liu, Jing Gao, Ahmed Hassan Awadallah, and Jianfeng Gao. Adamix: Mixture-of-adapter for parameter-efficient tuning of large language models. In EMNLP, 2022. 9

Taiqiang Wu, Jiahao Wang, Zhe Zhao, and Ngai Wong. Mixture-of-subspaces in low-rank adaptation. arXiv preprint arXiv:2406.11909, 2024. 9

Songlin Yang and Yu Zhang. Fla: A triton-based library for hardware-efficient implementations of linear attention mechanism, January 2024. URL https://github.com/fla-org/ flash-linear-attention. 6

Muchao Ye, Weiyang Liu, and Pan He. Vera: Explainable video anomaly detection via verbalized learning of vision-language models. In CVPR, 2025. 3

Qingyu Yin, Yulun Wu, Zhennan Shen, Sunbowen Li, Zhilin Wang, Yanshu Li, Chak Tou Leong, Jiale Kang, and Jinjin Gu. Evaluating parameter efficient methods for rlvr. arXiv preprint arXiv:2512.23165, 2025. 1, 6, 9

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. In ICLR, 2024. 6

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. In NeurIPS, 2026. 6

Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel. BitFit: Simple Parameter-efficient Finetuning for Transformer-based Masked Language-models. In ACL, 2022. 9

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In ACL, 2019. 6

Longteng Zhang, Lin Zhang, Shaohuai Shi, Xiaowen Chu, and Bo Li. Lora-fa: Efficient and effective low rank representation fine-tuning. arXiv preprint arXiv:2308.03303, 2023a. 3

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adalora: Adaptive budget allocation for parameterefficient fine-tuning. arXiv preprint arXiv:2303.10512, 2023b. 3, 9

Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2025, 2025. 6

Yuanhan Zhang, Kaiyang Zhou, and Ziwei Liu. Neural prompt search. arXiv preprint arXiv:2206.04673, 2022. 9

Tianyu Zheng, Ge Zhang, Tianhao Shen, Xueling Liu, Bill Yuchen Lin, Jie Fu, Wenhu Chen, and Xiang Yue. Opencodeinterpreter: Integrating code generation with execution and refinement. In Findings ofACL, 2024. 6

Bojia Zi, Xianbiao Qi, Lingzhi Wang, Jianan Wang, Kam-Fai Wong, and Lei Zhang. Deltalora: Fine-tuning high-rank parameters with the delta of low-rank matrices. arXiv preprint arXiv:2309.02411, 2023. 9

## A PRETRAINING SETTINGS

We conduct the pre-training experiments on FineWeb-10BT. Unless otherwise specified, all models are trained for 20,480 optimization steps with a sequence length of 2,048 and a global batch size of 256. We use AdamW with a peak learning rate of $3 \times \bar { 1 0 } ^ { - 4 }$ and $\epsilon = 1 0 ^ { - \bar { 1 5 } }$ The learning rate is warmed up for the first 1,024 steps and then decayed using a cosine schedule to 10% of the peak learning rate. The gradient norm is clipped at 1.0. All experiments use the same training configuration and random seed for controlled comparison.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Dataset</td><td>FineWeb-10BT</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Adam €</td><td> $1 0 ^ { - 1 5 }$ </td></tr><tr><td>LR scheduler</td><td>Cosine decay</td></tr><tr><td>Warmup steps</td><td>1,024</td></tr><tr><td>Minimum LR ratio</td><td>0.1</td></tr><tr><td>Global batch size</td><td>256</td></tr><tr><td>Sequence length</td><td>2,048</td></tr><tr><td>Training steps</td><td>20,480</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Random seed</td><td>42</td></tr></table>

Table 7: Hyperparameter settings for the pretraining experiments.

## B SUPERVISED FINETUNING SETTINGS

For the SFT experiments, we tune the hyperparameters of each method individually to ensure competitive performance. For NoRA, we recommend using $\alpha = r$ as the default setting, under which the early gradient norm is close to that of full finetuning. Although a larger scaling factor, such as $\alpha = 2 r$ , can yield better loss fitting in some cases, we observe that it may also lead to increased forgetting of the pretrained knowledge. Considering this trade-off between adaptation and retention, we use and recommend α : $: r = 1 : 1$ as the default configuration.

<table><tr><td>Hyperparameters</td><td>LoRA</td><td>DoRA</td><td>PiSSA</td><td>rsLoRA</td><td>OFT</td><td>NoRA</td></tr><tr><td>rank</td><td>32</td><td>32</td><td>32</td><td>32</td><td>32</td><td>32</td></tr><tr><td>α</td><td>64</td><td>64</td><td>32</td><td>64</td><td>-</td><td>32</td></tr><tr><td>dropout</td><td></td><td></td><td>0.0</td><td></td><td></td><td></td></tr><tr><td>optimizer</td><td></td><td></td><td>AdamW</td><td></td><td></td><td></td></tr><tr><td>lr</td><td></td><td></td><td>2e-5</td><td></td><td></td><td></td></tr><tr><td>lr scheduler</td><td></td><td></td><td>Cosine decay</td><td></td><td></td><td></td></tr><tr><td>batch size</td><td></td><td></td><td>128</td><td></td><td></td><td></td></tr><tr><td>warmup ratio</td><td></td><td></td><td>0.3</td><td></td><td></td><td></td></tr><tr><td>epochs</td><td></td><td></td><td>1</td><td></td><td></td><td></td></tr><tr><td>target_modules</td><td></td><td></td><td>Q,K,V,O,Up,Down,Gate</td><td></td><td></td><td></td></tr></table>

Table 8: Hyperparameter settings for the Llama-3.2-3B supervised finetuning experiments.

## C REINFORCEMENT LEARNING SETTINGS

We conduct reinforcement learning experiments on DeepSeek-R1-Distill-Qwen-1.5B using the DAPO objective and the DAPO-Math-17K dataset. All experiments are performed with 8 GPUs using bfloat16 precision. We train for 1,024 optimization steps with a global batch size of 128 and a learning rate of $1 \times 1 0 ^ { - 5 }$ under a cosine learning-rate schedule without warmup. For each prompt, we sample 8 responses with a maximum completion length of 16,384 tokens. The maximum prompt length is set to 512 tokens.

For parameter-efficient training, we use a rank of 32 and set $\alpha = 6 4$ . The low-rank modules are applied to all attention projections (q, k, v, and o) and MLP projections (up, down, and gate). We use the same training configuration across different methods to ensure a controlled comparison.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Model</td><td>DeepSeek-R1-Distill-Qwen-1.5B</td></tr><tr><td>Dataset</td><td>DAPO-Math-17K</td></tr><tr><td>RL objective</td><td>DAPO</td></tr><tr><td>Precision</td><td>bfloat16</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>LR scheduler</td><td>Cosine decay</td></tr><tr><td>Warmup ratio</td><td>0</td></tr><tr><td>Global batch size</td><td>128</td></tr><tr><td>Training steps</td><td>1,024</td></tr><tr><td>LoRA rank r</td><td>32</td></tr><tr><td>LoRA α</td><td>64</td></tr><tr><td>LoRA dropout</td><td>0.05</td></tr><tr><td>Responses per prompt</td><td></td></tr><tr><td></td><td>8</td></tr><tr><td>Maximum prompt length</td><td>512</td></tr><tr><td>Maximum completion length</td><td>16,384</td></tr><tr><td>€high</td><td>0.28</td></tr><tr><td>β</td><td>0.0</td></tr><tr><td>Random seed</td><td>42</td></tr></table>

Table 9: Hyperparameter settings for the reinforcement learning experiments.