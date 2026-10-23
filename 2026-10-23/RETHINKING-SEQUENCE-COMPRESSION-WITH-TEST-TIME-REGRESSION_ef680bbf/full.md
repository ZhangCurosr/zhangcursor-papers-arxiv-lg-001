![](images/2fc75ac7700fe1ee5b14dce434c05acb8686f5771d86b3912a63268fc7baf4ac.jpg)

# RETHINKING SEQUENCE COMPRESSION WITH TEST-TIME REGRESSION

Sihan Yang<sup>1</sup> Jie Huang<sup>2</sup> Hanxuan Li<sup>1</sup> Yifan Guo<sup>1</sup> Weigao Sun<sup>3</sup> Yiran Zhong<sup>3</sup> Weiyang Liu<sup>1,\*</sup> <sup>1</sup>The Chinese University of Hong Kong <sup>2</sup>Independent Researcher <sup>3</sup>Token Foundry, Alibaba Group <sup>\*</sup>Corresponding author spherelab.ai/ttr-compressor

![](images/fb3eea75ad4c0029854a8145306c6c1feb4b14630927a3fde388481607eb97b8.jpg)

![](images/54e6da873e79f103de8d4cde0bdb1c0c353045bb3e9ad42a71f03060815b60c4.jpg)

![](images/1dc9ae5d8cf38926ca692d500f0799a03ebd383fae5695e941ad0ea0739cc78a.jpg)

![](images/f7f56dbf699bb510e6b78cfd0518a1f9192caa5ad0ff0b3b77b9ba4bc50090db.jpg)  
(b) Applications and empirical comparison  
Figure 1: Overview of TTR-Compressor. (a) Unlike pooling and projection, which process tokens independently, TTR-Compressor fits all tokens jointly and thereby captures the interplay among them. (b) TTR-Compressor outperforms existing methods on both block-wise sparse attention and visual token merging.

## ABSTRACT

Sequence compression is a fundamental operation and a critical information bottleneck in efficient language and multimodal models, and yet prevailing approaches based on pooling or learned projections remain underperforming. We revisit sequence compression through the lens of test-time regression: input tokens define a regression problem whose fitted coefficients form the compressed representation. Importantly, the regression problem is learned end-to-end via backpropagation, while its coefficients are computed efficiently in closed form during the forward pass. This leads to TTR-Compressor, a general alternative to conventional pooling and projection-based compression. We evaluate TTR-Compressor as the compressor for block-wise sparse attention and the visual token merger for multimodal large language models. On block-wise sparse attention, TTR-Compressor improves RULER performance over attention-based compressors by a large margin. For visual token merging, it consistently outperforms the standard MLP merger and matches its performance with 55.6% fewer visual tokens. These results establish test-time regression as an effective primitive for sequence compression in efficient language and multimodal models.

## 1 INTRODUCTION

Sequence compression is a ubiquitous operation in modern efficient large model architectures (Qwen Team, 2026b; DeepSeek-AI, 2026), and yet remains surprisingly underexplored. For example, block-wise sparse attention compresses each token block into a summary vector used to score and select relevant blocks (Jiang et al., 2024; Lu et al., 2025; Yuan et al., 2025), while multimodal large language models (MLLMs) merge neighboring visual features into compact tokens before feeding them into the language backbone (Wang et al., 2025; Bai et al., 2025; Team et al., 2026a). In both settings, the compressor acts as a critical information bottleneck. Once information is discarded, it cannot be recovered downstream. One would expect compression to be a carefully designed architectural component. However, most models rely on simple mechanisms, such as parameter-free pooling, learned projections over concatenated tokens, or at best, attention-weighted pooling.

We argue that these default compressors share a fundamental structural limitation, i.e., they aggregate tokens using weights that do not account for inter-token relationships. Mean pooling uses uniform weights, linear projection uses fixed position-dependent weights, and attention-weighted pooling scores each token independently. As a result, redundant tokens can reinforce one another, while distinctive information is diluted. From a statistical perspective, these methods primarily exploit first-order token statistics while ignoring second-order inter-token correlations, the very structure that reveals which information is redundant and which is distinctive.

We introduce a fundamentally different compression strategy. Rather than asking which weighted average best summarizes a set of tokens, we let the tokens define a regression problem and take its fitted coefficients as the summary. Specifically, our proposed TTR-Compressor, constructs a selfsupervised ridge-regression problem directly from the input tokens, requiring no external labels, and uses its closed-form solution as the compressed representation. This formulation creates a natural division: the outer training loop meta-learns which regression problem produces task-usefu representations, while the inner closed-form solution determines how that problem should be fitted to each input. The key distinction from conventional pooling and projection arises from the Gram matrix in the regression solution. By explicitly capturing inter-token correlations, TTR-Compressor fits correlated tokens jointly, so redundant evidence has diminishing rather than purely additive influence. This input-adaptive behavior cannot be reproduced by a fixed projection or additive token combination. Moreover, learning the regularizer removes the need for ridge hyperparameter tuning in large-scale pretraining. Because the compression window is typically smaller than the channel dimension (i.e., T < d), we solve the dual problem with an efficient $T \times T$ Cholesky factorization. Thus, TTR-Compressor incurs minimal overhead and introduces only 2d learnable parameters per compressor, making it an effective drop-in replacement for sequence compression.

We evaluate TTR-Compressor on two applications that impose fundamentally different requirements on compression. In block-wise sparse attention, the compressed summary must act as a faithful retrieval key. In this setting, TTR-Compressor consistently outperforms traditional baselines on RULER (Hsieh et al., 2024) retrieval tasks, surpassing sophisticated attention-based compressors (Hu et al., 2026) by a considerable margin. In visual token merging, by contrast, the compressed summary must preserve a faithful representation. Here, TTR-Compressor consistently outperforms the widely used MLP merger across diverse compression ratios, and achieves comparable performance with 55.6% fewer visual tokens. Our contributions are summarized below:

• A novel compression framework. We introduce a test-time regression framework for sequence compression, reframing token summarization as an instance-adaptive optimization problem.

• A practical operator. We derive an efficient dual closed-form solution and eliminate hyperparameter tuning, making TTR-Compressor fully practical and scalable for large-scale pretraining.

• Empirical effectiveness. As a drop-in component in large models, TTR-Compressor consistently outperforms conventional baselines in both block-wise sparse attention and visual token merging.

## 2 RELATED WORK

Sequence compression. Compressing a sequence of tokens into a single vector is a fundamental operation across modern architectures, with applications ranging from multimodal token reduction (Bolya et al., 2022; Li et al., 2023; Yao et al., 2024; Cha et al., 2024; Chu et al., 2024) to block-wise sparse attention (Jiang et al., 2024; Lu et al., 2025; Gao et al., 2024; Lai et al., 2025). Existing compression operators largely fall into three paradigms: (1) parameter-free pooling, such as mean pooling; (2) concatenation followed by learned projection; and (3) attention-based compression, which summarizes tokens via attention-weighted averaging (Mohtashami & Jaggi, 2023; Yang et al., 2025b). A fundamental limitation of these families is that they compute token combinations without explicitly accounting for the inter-token correlations. In contrast, TTR-Compressor derives its representation as the solution to a linear system whose Gram matrix directly encodes these correlations, providing an expressive, drop-in replacement for the operators above.

Test-time regression and test-time training. The idea of adapting part of a network on each input at inference time dates back to fast-weight programmers (Schmidhuber, 1992; Schlag et al., 2021) and was revived as test-time training (TTT), first for robustness under distribution shift (Sun et al., 2020) and then as sequence-modeling layers whose hidden states are small models updated by gradient descent on a self-supervised loss (Sun et al., 2024; Behrouz et al., 2026b;a; Han et al., 2026). In-Place TTT (Feng et al., 2026) further extends this paradigm by directly adapting the existing MLP projection weights of pretrained Transformers. A parallel line of work reinterprets linear attention and its descendants (e.g., DeltaNet (Yang et al., 2024), Gated DeltaNet (Yang et al., 2025c), Longhorn (Liu et al., 2025), Titans (Behrouz et al., 2026b)) as online regression: each layer maintains a key-to-value regressor that is refined with one gradient or delta-rule step per token and read out by the query. MesaNet (von Oswald et al., 2026) pushes the inner optimization to convergence, solving the problem at every time step with conjugate gradients, while LaCT (Zhang et al., 2026) updates fast weights over large, unordered chunks to improve hardware utilization.

TTR-Compressor differs from these methods in two fundamental ways. First, because compression operates independently on small, disjoint blocks without cross-block dependencies, the regression admits an efficient, exact closed-form solution via a single $T \times T$ dual solve. This avoids both iterative gradient-based inner optimization (as in TTT, $\mathrm { V i T ^ { 3 } }$ , In-Place TTT, and MesaNet) and singlestep approximations (such as delta-rule updates). Second, rather than using test-time regression to produce input-dependent model weights (Bertinetto et al., 2018; Sun et al., 2020) or a fast-weight memory that is subsequently queried (Schlag et al., 2021; Yang et al., 2025c; Behrouz et al., 2026b), we use regression to directly yield a representation of a token block. Specifically, the fitted coefficients themselves serve as the compressed representation (e.g., a visual token or a retrieval key).

## 3 JOINT SEQUENCE COMPRESSION VIA TEST-TIME REGRESSION

We introduce TTR-Compressor, a sequence compression operator based on test-time regression, as shown in Figure 2(a). Given a sequence of token representations, TTR-Compressor constructs a regression problem from the tokens and uses its fitted coefficients as the compressed representation. We first present the regression formulation of compression and specify its construction, then derive an efficient solution and describe how TTR-Compressor is learned end-to-end within a model.

## 3.1 COMPRESSION AS TEST-TIME REGRESSION

We formulate sequence compression as ridge regression (Hoerl & Kennard, 1970). The tokens in a sequence provide regression observations, and the fitted regression coefficients serve as a compact representation of that sequence. The regression objective encourages this representation to capture the relationships among tokens, while regularization controls the magnitude of the fitted coefficients.

Given a sequence $\pmb { X } = [ \pmb { x } _ { 1 } , \dots , \pmb { x } _ { T } ] ^ { \top } \in \mathbb { R } ^ { T \times d }$ , our goal is to compress its $T$ token representations into a single vector $\pmb { h } \in \mathbb { R } ^ { d }$ . We associate each token with a regression key $\boldsymbol { k } _ { t } \in \mathbb { R } ^ { d }$ and a scalar target $v _ { t } \in \mathbb { R }$ , and define the compressed representation as:

$$
h ^ { \star } = \arg \operatorname* { m i n } _ { h \in \mathbb { R } ^ { d } } \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } \omega _ { t } \big ( \boldsymbol { k } _ { t } ^ { \top } \boldsymbol { h } - \boldsymbol { v } _ { t } \big ) ^ { 2 } + \frac { \lambda } { 2 } \| \boldsymbol { h } \| _ { 2 } ^ { 2 } ,\tag{1}
$$

where $\omega _ { t } \geq 0$ determines the contribution of each token and $\lambda > 0$ controls regularization. $h ^ { \star }$ summarizes the sequence by fitting the key-target relationships across its tokens. The regression target $v _ { t }$ is meta-learned through the task loss, while the regression itself is constructed and solved for each input sequence during both training and inference. Overall, we are solving a bi-level optimization, where the outer objective is the task loss that finds the task-useful target $v _ { t } ,$ and the inner objective is the test-time regression Eq. 1 that solves the compressed representation $h ^ { \star }$ . Here, test-time refers to fitting these input-specific coefficients; the learned model parameters remain fixed at inference.

## 3.2 CONSTRUCTING THE REGRESSION PROBLEM

Keys and Targets. We construct regression keys by normalizing the input features and generate scalar targets using a learned projection: $\begin{array} { r } { \pmb { k } _ { t } = \frac { \mathbf { \tilde { x } } _ { t } } { \lVert \mathbf { \tilde { x } } _ { t } \rVert _ { 2 } } , v _ { t } = \pmb { a } ^ { \top } \pmb { x } _ { t } } \end{array}$ , where $\pmb { a } \in \mathbb { R } ^ { d }$ is shared across tokens and sequences within a compressor. Key normalization ensures that $\kappa ^ { + } \kappa$ measures directional (semantic) similarity alone (Liu et al., 2017). Crucially, this normalization is asymmetric, as targets retain original token magnitudes $( v _ { t } = \pmb { a } ^ { \top } \pmb { x } _ { t } )$ . If targets were also normalized $( v _ { t } = \pmb { a } ^ { \top } k _ { t } )$ the fitting term would reduce to $\overset { \vartriangle } { \frac { 1 } { 2 } } \| \boldsymbol { K } ( \boldsymbol { h } - \boldsymbol { a } ) \| _ { 2 } ^ { 2 }$ , trivially collapsing $h ^ { \star }$ toward the sequence-invariant static vector $\mathbf { } \mathbf { } a .$ By preserving token norms in the targets, no constant vector can fit varying magnitudes, forcing $h ^ { \star }$ to dynamically integrate both token semantics and their relative importance. The targets are derived entirely from the input, so fitting $h ^ { \star }$ requires no additional labels. Optionally, targets can be further enriched by first applying a local convolution (1D for language and 2D for vision) to aggregate neighboring context before the projection, i.e., $v _ { t } = \pmb { a } ^ { \top } \operatorname { C o n v } ( \pmb { X } )$ , where Conv(·) denotes the convolution operator and X represents the local context window of n neighboring tokens around position $t .$ Alternative parameterizations of $v _ { t }$ are explored in the Appendix E.2.

![](images/679a144367a42ec9defb84fb0976572e72cbffbe71e4cf90f86f504b452b42b0.jpg)  
Figure 2: Overview of TTR-Compressor. (a) Bilevel learning: The inner solve formulates sequence compression as test-time ridge regression with an exact dual Cholesky solve to obtain the representation $\boldsymbol { h } ^ { * }$ , while the regression parameters are learned end-to-end via task gradients. (b) Input-dependent second-order adaptation: By jointly coupling tokens through the Gram matrix, TTR naturally diminishes the relative influence of redundant tokens compared to additive pooling. For intuitive illustration, we set $D = I$ and uniform targets ${ \pmb v } = { \bf 1 } ;$ as typically learned in practice, a smaller regularizer D further amplifies this deduplication effect, yielding an even higher relative share for distinct tokens.

Uniform Token Contributions. We use uniform weights $\omega _ { t } = 1$ , assigning the same coefficient to every token’s regression error. Let $K = [ k _ { 1 } , \ldots , k _ { T } ^ { - } ] ^ { \top } \in \mathbb { R } ^ { T \times d }$ and $\mathbf { \bar { \pmb { v } } } = [ v _ { 1 } , \dots , v _ { T } ] ^ { \top } \in \mathbb { R } ^ { T }$ Setting the gradient of Eq. 1 to zero gives us a closed-form solution:

$$
( \boldsymbol { K } ^ { \top } \boldsymbol { K } + \lambda \boldsymbol { I } _ { d } ) \boldsymbol { h } ^ { \star } = \boldsymbol { K } ^ { \top } \boldsymbol { v } , \qquad \boldsymbol { h } ^ { \star } = ( \boldsymbol { K } ^ { \top } \boldsymbol { K } + \lambda \boldsymbol { I } _ { d } ) ^ { - 1 } \boldsymbol { K } ^ { \top } \boldsymbol { v } .\tag{2}
$$

Since $\lambda > 0$ , the objective is strictly convex and has a unique solution. The solution combines the key-target statistic $\check { \pmb { K } } ^ { \top } \cdot$ v with the sequence-dependent feature geometry $\kappa ^ { \top } \kappa$

Learnable Diagonal Regularization. In conventional ridge regression, λ is a tunable hyperparameter. Such tuning is expensive during large-model pretraining and complicates scaling to different model sizes. To avoid this, we replace $\lambda I _ { d }$ with a learnable positive diagonal matrix (meta-learned via the task loss), yielding a generalized ridge regression objective (Tikhonov & Arsenin, 1977):

$$
D = \operatorname { d i a g } ( \delta _ { 1 } , \ldots , \delta _ { d } ) , \qquad \delta _ { j } = \mathrm { s o f t p l u s } ( \rho _ { j } ) ,\tag{3}
$$

where $\pmb { \rho } \in \mathbb { R } ^ { d }$ is trainable. The resulting objective and its solution are:

$$
h ^ { \star } = \arg \operatorname* { m i n } _ { h } \frac { 1 } { 2 } \| K h - { \pmb v } \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 } h ^ { \top } D h ,\tag{4}
$$

$$
{ \bf \Lambda } = ( { \pmb K } ^ { \top } { \pmb K } + { \pmb D } ) ^ { - 1 } { \pmb K } ^ { \top } { \pmb v } ,\tag{5}
$$

which solves a regularized linear system that takes the pairwise interaction of tokens into account.

## 3.3 EFFICIENT DUAL-FORM SOLUTION

Directly evaluating Eq. 2 requires solving a $d \times d$ linear system. When the number of tokens being compressed is typically smaller than the feature dimension, we instead use the equivalent dual

form (Saunders et al., 1998) in order to improve the computational efficiency:

$$
\pmb { h } ^ { \star } = \pmb { K } ^ { \top } ( \pmb { K } \pmb { K } ^ { \top } + \lambda \pmb { I } _ { T } ) ^ { - 1 } \pmb { v } .\tag{6}
$$

This replaces the feature-space system with a $T \times T$ token-space system. Its entries depend on pairwise similarities between regression keys, allowing the fitted representation to account for correlations among tokens. We evaluate Eq. 6 by solving $( K K ^ { \top } + \lambda I _ { T } ) \pmb { \alpha } = \pmb { v }$ and computing $h ^ { \star } = K ^ { \top } \alpha$ , without explicitly forming an inverse. The resulting computational cost is $\mathcal { O } ( T ^ { \tilde { 2 } } d + T ^ { 3 } )$ , compared with $\mathcal { \ O } ( T d ^ { 2 } + d ^ { 3 } )$ for the direct primal formulation. The dual form is therefore particularly suitable for compressing short sequences of high-dimensional features.

Obtaining an Efficient Dual Solution. With the learnable diagonal regularization introduced in Section $3 . 2 , \lambda I _ { d }$ is replaced by D. To obtain an efficient dual solution, we rescale the keys as $\widetilde { K } = K D ^ { - 1 / 2 }$ , yielding the final form of TTR-Compressor:

$$
\boxed { \mathrm { T T R } ( \pmb { X } ) } = \pmb { D } ^ { - 1 / 2 } \widetilde { \pmb { K } } ^ { \top } \big ( \pmb { I } _ { T } + \widetilde { \pmb { K } } \widetilde { \pmb { K } } ^ { \top } \big ) ^ { - 1 } \pmb { v } \ \vert\tag{7}
$$

Because D is diagonal, rescaling the keys costs only $\mathcal { O } ( T d )$ , preserving the dual formulation’s overall complexity. In practice, we compute a Cholesky factorization $L L ^ { \top } = I _ { T } + \widetilde { K } \widetilde { K } ^ { \top }$ , solve $\mathbf { \nabla } _ { L L } \top _ { \alpha } = v$ through triangular solutions, and return $D ^ { - 1 / 2 } \widetilde { K } ^ { \intercal } \alpha$

## 3.4 FULLY END-TO-END TRAINING

The target projection a and diagonal parameters $\rho$ are optimized end-to-end through the differentiable closed-form solution using the original model’s training objective. Each compressor maintains its own parameters, which are shared across all input sequences. At inference time, these learned parameters remain fixed, while $h ^ { \star }$ is recomputed independently for each input sequence. TTR-Compressor therefore combines a learned regression objective with an exact, input-adaptive solution, without requiring iterative gradient-based optimization at test time.

## 4 UNDERSTANDING TTR-COMPRESSOR

The learning process of TTR-Compressor can be decomposed into two distinct mechanisms: outerloop training learns which regression problem is useful for compression, while the inner solution adapts the compressed representation to the current input’s statistics. We examine them separately.

## 4.1 TASK-SUPERVISED LEARNING OF THE COMPRESSION OBJECTIVE

The regression objective is not prescribed independently of the task. Its targets $v _ { t } = \mathbf { \nabla } \mathbf { a } ^ { \top } \mathbf { x } _ { t }$ and regularization matrix D are learned through the downstream use of the compressed representation. Writing $h _ { a , \rho } ^ { \star }$ for the regression solution, the training objective takes the form:

$$
\operatorname* { m i n } _ { \theta , a , \rho } \mathbb { E } _ { ( x , y ) } \left[ \mathcal { L } _ { \mathrm { t a s k } } \left( F _ { \theta } \left( h _ { a , \rho } ^ { \star } ( X _ { \theta } ( x ) ) \right) , y \right) \right] ,\tag{8}
$$

where $X _ { \theta }$ produces the input representations and $F _ { \theta }$ denotes the subsequent computation. Thus, the outer objective evaluates the usefulness of $h ^ { \star }$ for the final task, rather than leveraging its learnability to merely minimize the inner regression error.

Learning the Targets and Regularization. This coupling can be seen directly from the gradients. Let $\pmb { A } = \mathbf { \bar { K } } ^ { \top } \pmb { K } + \mathbf { \bar { D } } , \pmb { g } = \nabla _ { \pmb { h } ^ { \star } } \mathcal { L } _ { \mathrm { t a s k } }$ , and $\pmb { s } = \pmb { A } ^ { - 1 } \pmb { g }$ . Differentiating the regression solution gives:

$$
\nabla _ { a } \mathcal { L } _ { \mathrm { t a s k } } = \pmb { X } ^ { \top } \pmb { K } \pmb { s } , \qquad \nabla _ { \rho } \mathcal { L } _ { \mathrm { t a s k } } = - \mathrm { s i g m o i d } ( \pmb { \rho } ) \odot \pmb { h } ^ { \star } \odot \pmb { s } ,\tag{9}
$$

where ⊙ denotes element-wise multiplication. The downstream gradient therefore shapes both the scalar targets derived from tokens via a and the coordinate-wise penalties imposed through $\rho .$ Consequently, the scalar targets require no predefined semantics; instead, they serve as learned measurements whose fitted coefficients yield informative representations for downstream layers. In this sense, the regression provides a structural inductive bias for compression, while outer-loop optimization steers this structural bias towards better handling the downstream task.

## 4.2 INPUT-DEPENDENT SECOND-ORDER ADAPTATION

Beyond learning the regression objective, the inner solve adapts the compressed representation to the correlation structure of each input. For isotropic regularization, the dual-form solution is

$$
\pmb { h } ^ { \star } = \pmb { K } ^ { \top } ( \pmb { K } \pmb { K } ^ { \top } + \lambda \pmb { I } ) ^ { - 1 } \pmb { v } .\tag{10}
$$

Let $\pmb { G } = \pmb { K } \pmb { K } ^ { \top } \in \mathbb { R } ^ { T \times T }$ . Since the keys are normalized, each entry $G _ { i j } = { \pmb k } _ { i } ^ { \top } { \pmb k } _ { j }$ directly measures the similarity between tokens i and j. Thus, G captures the input-dependent second-order correlation structure of the token group.

The compressed representation can equivalently be written as

$$
\pmb { h } ^ { \star } = \pmb { K } ^ { \top } \pmb { \alpha } , \qquad \pmb { \alpha } = ( \pmb { G } + \lambda \pmb { I } ) ^ { - 1 } \pmb { v } .\tag{11}
$$

Unlike pooling and projection, where token contributions are aggregated independently, the effective weights α are determined jointly by the target responses and all pairwise token correlations. Highly similar tokens are explicitly coupled through the Gram matrix, so redundant evidence exhibits di minishing rather than additive influence. Conversely, information not already explained by strongly correlated tokens can retain a larger relative influence when supported by the regression targets, as shown in Figure 2(b). TTR therefore adapts its compression rule to the second-order structure of every input rather than applying a fixed or independently weighted combination of tokens.

Learning the inter-token geometry. With the learnable diagonal regularizer D, this interpretation extends to the rescaled keys $\widetilde { K } = K D ^ { - 1 / 2 }$ . The corresponding token Gram matrix becomes

$$
\widetilde { G } = \widetilde { K } \widetilde { K } ^ { \intercal } = K D ^ { - 1 } K ^ { \intercal } .\tag{12}
$$

Training therefore learns which feature dimensions should matter more when measuring token similarity. The outer loop learns a task-adapted correlation geometry, while the inner solve adapts the compressed representation to the correlation structure of each individual input.

## 5 APPLICATIONS AND EXPERIMENTAL RESULTS

We evaluate TTR-Compressor in two settings: block-wise sparse attention for efficient language modeling, where compressed representations guide context selection, and visual token merging for efficient multimodal understanding, where they serve as the visual input to the language backbone.

## 5.1 BLOCK-WISE SPARSE ATTENTION

Architecture and Baselines. Block-wise sparse attention partitions the context into blocks, scores their compressed representations against each query, and attends to tokens in selected blocks alongside a local window. The compressor therefore determines which distant information remains accessible: an uninformative summary can prevent a relevant block from being selected, regardless of the quality of its individual token representations.

We adopt HiLS-Attention (Hu et al., 2026) as a base framework and vary only the compressor. HiLS-Attention has three effective architectural components: hierarchical softmax attention, which combines within-block attention with learned block-level weights; low-rank query calibration, which adapts queries for scoring compressed representations; and HoPE, which applies RoPE to only a subset of feature dimensions to support length extrapolation. We compare four compressors: mean pooling, concat & projection, the attention-weighted pooling used in the original HiLS-Attention, and TTR-Compressor. Full attention and sliding-window attention serve as additional references.

Training and Evaluation. We train 345M-parameter language models on the FineWeb dataset (Penedo et al., 2024) with an 8K context length. Training runs for 30,000 optimization steps with a global batch size of 128 sequences, corresponding to approximately 30B tokens, and uses AdamW with a learning rate of $3 \stackrel { \textstyle - } { \times } 1 0 ^ { - 4 }$ . All sparse attention variants use 64-token blocks, select up to 32 distant blocks per query, and retain a 512-token local window.

Evaluation covers two complementary aspects. First, since queries use the compressed representations to retrieve relevant blocks, retrieval accuracy directly reflects compression quality. Following (Hatamizadeh et al., 2026), we evaluate on the single-needle (S-NIAH) and multi-key (MK NIAH) retrieval tasks from RULER (Hsieh et al., 2024), which require locating relevant information among distractors, as the most direct and demanding test of the compressor. We evaluate at 4K–32K context lengths, including extrapolation beyond the 8K training length. Additionally, LAMBADA accuracy and the zero-shot benchmarks in Table 2 assess contextual prediction, commonsense reasoning, and question answering.

<table><tr><td rowspan="2">Model</td><td colspan="4">S-NIAH-1</td><td colspan="4">S-NIAH-2</td><td colspan="4">S-NIAH-3</td><td colspan="4">MK-NIAH-1</td><td rowspan="2">Avg.</td></tr><tr><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>4K</td><td>8K</td><td>16K</td><td>32K</td></tr><tr><td>Full Attention</td><td>0.834</td><td>0.428</td><td>0</td><td>0</td><td>0.850</td><td>0.412</td><td>0</td><td>0</td><td>0.572</td><td>0.470</td><td>0</td><td>0</td><td>0.374</td><td>0.272</td><td>0</td><td>0</td><td>0.2632</td></tr><tr><td>Sliding Window Attention</td><td>0.116</td><td>0.050</td><td>0.044</td><td>0.016</td><td>0.148</td><td>0.048</td><td>0.034</td><td>0.030</td><td>0.112</td><td>0.094</td><td>0.030</td><td>0.034</td><td>0.126</td><td>0.080</td><td>0.026</td><td>0.028</td><td>0.0635</td></tr><tr><td>Mean Pooling</td><td>0.938</td><td>0.536</td><td>0.470</td><td>0.270</td><td>0.840</td><td>0.258</td><td>0.042</td><td>0.050</td><td>0.292</td><td>0.282</td><td>0.040</td><td>0.042</td><td>0.366</td><td>0.230</td><td>0.042</td><td>0.056</td><td>0.2971</td></tr><tr><td>Concat &amp; Projection</td><td>0.892</td><td>0.840</td><td>0.792</td><td>0.582</td><td>0.764</td><td>0.232</td><td>0.124</td><td>0.068</td><td>0.512</td><td>0.324</td><td>0.174</td><td>0.076</td><td>0.476</td><td>0.278</td><td>0.084</td><td>0.052</td><td>0.3919</td></tr><tr><td>Attention-Weighted Pooling</td><td>0.944</td><td>0.914</td><td>0.876</td><td>0.740</td><td>0.926</td><td>0.274</td><td>0.136</td><td>0.060</td><td>0.340</td><td>0.082</td><td>0.026</td><td>0.026</td><td>0.474</td><td>0.258</td><td>0.124</td><td>0.072</td><td>0.3920</td></tr><tr><td>TTR-Compressor</td><td>0.996</td><td>0.996</td><td>0.992</td><td>0.650</td><td>0.982</td><td>0.566</td><td>0.288</td><td>0.126</td><td>0.770</td><td>0.376</td><td>0.188</td><td>0.066</td><td>0.528</td><td>0.302</td><td>0.212</td><td>0.110</td><td>0.5092</td></tr><tr><td>TTR-Compressor (w/ Conv)</td><td>0.998</td><td>1.000</td><td>0.994</td><td>0.988</td><td>0.960</td><td>0.454</td><td>0.128</td><td>0.102</td><td>0.834</td><td>0.482</td><td>0.130</td><td>0.096</td><td>0.542</td><td>0.298</td><td>0.116</td><td>0.108</td><td>0.5144</td></tr></table>

Table 1: Accuracy on Single Needle-In-A-Haystack (S-NIAH) and Multi-Key Needle-In-A-Haystack (MK-NIAH) tasks from RULER across varied context lengths. Best values for each context length are bolded, and second-best values are underlined.

<table><tr><td>Model</td><td>ARC-c acc ↑</td><td>ARC-e acc ↑</td><td>BoolQ acc ↑</td><td>COPA acc ↑</td><td>Hella. acc ↑</td><td>LMB. acc ↑</td><td>OBQA acc ↑</td><td>PIQA acc ↑</td><td>SciQ acc ↑</td><td>Wino. acc ↑</td><td>Avg. acc ↑</td></tr><tr><td>Full Attention</td><td>24.74</td><td>42.09</td><td>60.15</td><td>68.00</td><td>39.85</td><td>35.71</td><td>29.60</td><td>67.74</td><td>62.70</td><td>51.07</td><td>48.16</td></tr><tr><td>Sliding Window Attention</td><td>24.91</td><td>41.96</td><td>55.02</td><td>69.00</td><td>39.53</td><td>35.47</td><td>29.00</td><td>67.14</td><td>63.00</td><td>50.75</td><td>47.58</td></tr><tr><td>Mean Pooling</td><td>24.66</td><td>42.38</td><td>55.99</td><td>66.00</td><td>39.62</td><td>35.88</td><td>28.40</td><td>69.31</td><td>65.50</td><td>53.04</td><td>48.08</td></tr><tr><td>Concat &amp; Projection</td><td>24.83</td><td>42.30</td><td>58.65</td><td>68.00</td><td>39.58</td><td>36.62</td><td>30.00</td><td>67.74</td><td>62.20</td><td>51.78</td><td>48.17</td></tr><tr><td>Attention-Weighted Pooling</td><td>25.00</td><td>41.75</td><td>61.22</td><td>65.00</td><td>39.61</td><td>35.61</td><td>29.40</td><td>68.93</td><td>63.80</td><td>50.51</td><td>48.08</td></tr><tr><td>TTR-Compressor</td><td>23.29</td><td>41.58</td><td>57.95</td><td>71.00</td><td>39.41</td><td>36.27</td><td>30.20</td><td>69.31</td><td>63.00</td><td>49.57</td><td>48.16</td></tr><tr><td>TTR-Compressor (w/ Conv)</td><td>24.15</td><td>41.33</td><td>59.94</td><td>70.00</td><td>39.63</td><td>35.36</td><td>30.20</td><td>67.85</td><td>62.80</td><td>49.96</td><td>48.12</td></tr></table>

Table 2: Performance comparison on language modeling and zero-shot common-sense reasoning. The evaluation context lengths are too short to activate sparse attention, so the compressor is not involved during inference; these results therefore only reflect the common-sense reasoning ability acquired during language model pretraining. Note that Concat & Projection has significantly more parameters than the others. Best values are bolded, and second-best values are underlined.

Results. As shown in Table 1, the choice of compressor largely determines retrieval quality. TTR-Compressor outperforms all baselines by a large margin, and its advantage persists when extrapolating to 16K and 32K. We attribute this to its second-order modeling: pooling and projection add up tokens independently, so the redundant context that fills most of a block drowns out the few informative tokens. The Gram matrix in TTR-Compressor fits correlated tokens jointly and reduces the influence of redundant evidence, so the resulting key keeps the distinctive content that queries need to find. The convolutional variant brings a further gain, reaching 12.24 points over the strongest baseline, as convolution makes the regression targets more informative and thus improves compression quality. On the benchmarks in Table 2, the evaluation contexts are too short to activate sparse attention, so these results only reflect the common-sense reasoning ability acquired during pretraining. Concat & Projection has substantially more parameters than the other compressors, which may benefit pretraining. TTR-Compressor performs well and on par with full attention.

## 5.2 VISUAL TOKEN MERGER

Architecture and Baselines. High-resolution images produce many visual tokens, making their direct processing by a language backbone expensive. Modern multimodal models therefore commonly use a spatial merger to reduce the visual sequence length. A widely adopted design concatenates each 2 × 2 group of neighboring visual features and maps it to one token through a two-layer MLP (Wang et al., 2025; Bai et al., 2025; Qwen Team, 2026a; Team et al., 2026b). Longer multimodal contexts and agentic workflows with repeated visual observations further motivate stronger compression, such as 3 × 3 merging (Team et al., 2026a; DeepSeek-AI, 2026).

We compare TTR-Compressor with the standard MLP merger. TTR-Compressor fits one regression per spatial group to compress its four or nine visual features into a single vector, which a two-layer MLP then projects to the language model’s hidden dimension. Its convolutional variant applies a 3 × 3 depthwise convolution to the feature map before constructing the regression targets.

Training and Evaluation. We use the ViT of Qwen3-VL (Bai et al., 2025) as the vision encoder and Qwen3-4B-Instruct (Yang et al., 2025a) as the language model. Training consists of two stages. The first stage trains only the merger modules on LLaVA-558K (Liu et al., 2024a) for one epoch with a learning rate of $1 0 ^ { - 3 }$ . The second stage jointly trains the merger modules and language backbone on 1.3M instruction samples mixing image (Liu et al., 2024b), multi-image (Li et al., 2024), and video (Zhang et al., 2024) data for one epoch with a learning rate of $2 \times 1 0 ^ { - 5 }$

<table><tr><td rowspan="2">Method</td><td colspan="3">General VQA</td><td colspan="2">Knowledge Reasoning</td><td colspan="3">Document Understanding</td><td colspan="2">Spatial Under.</td><td rowspan="2">Avg.</td></tr><tr><td>MMBench MMStar</td><td></td><td>RWQA</td><td>MMMU</td><td>AI2D</td><td>OCRBench InfoVQA</td><td></td><td>ChartQA</td><td>MMSI</td><td>ERQA</td></tr><tr><td>2 × 2 Token Compression</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MLP</td><td>76.63</td><td>49.93</td><td>64.44</td><td>46.22</td><td>75.49</td><td>62.80</td><td>41.72</td><td>70.64</td><td>26.20</td><td>35.25</td><td>54.93</td></tr><tr><td>TTR-Compressor</td><td>76.89</td><td>53.94</td><td>63.40</td><td>45.44</td><td>76.94</td><td>62.90</td><td>43.74</td><td>73.04</td><td>26.90</td><td>36.00</td><td>55.92</td></tr><tr><td>TTR-Compressor (w/ Conv)</td><td>75.86</td><td>52.26</td><td>66.93</td><td>45.67</td><td>76.65</td><td>63.80</td><td>44.33</td><td>72.52</td><td>26.30</td><td>36.00</td><td>56.03</td></tr><tr><td>3 × 3 Token Compression</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MLP</td><td>74.74</td><td>50.89</td><td>64.31</td><td>44.44</td><td>74.68</td><td>55.90</td><td>35.16</td><td>63.60</td><td>26.00</td><td>33.00</td><td>52.27</td></tr><tr><td>TTR-Compressor</td><td>76.46</td><td>52.40</td><td>65.62</td><td>45.89</td><td>75.23</td><td>58.70</td><td>38.27</td><td>67.16</td><td>26.80</td><td>36.50</td><td>54.30</td></tr><tr><td>TTR-Compressor (w/ Conv)</td><td>76.63</td><td>51.16</td><td>65.88</td><td>47.44</td><td>75.16</td><td>60.00</td><td>39.70</td><td>67.16</td><td>28.90</td><td>36.50</td><td>54.85</td></tr></table>

Table 3: Evaluation across multimodal benchmarks under different token compression ratios. Results are grouped by $2 \times 2$ and $3 \times 3$ token compression. Best values within each compression group are bolded, and second-best values are underlined

We evaluate on 4 types of benchmarks: general visual question answering (MMBench (Liu et al., 2024c), MMStar (Chen et al., 2024), RealWorldQA (xAI, 2024)), knowledge-intensive multimodal reasoning (MMMU (Yue et al., 2024), AI2D (Kembhavi et al., 2016)), text, infographic, and chart understanding (OCRBench (Liu et al., 2024d), InfoVQA (Tito et al., 2021), ChartQA (Masry et al., 2022)), and spatial reasoning (MMSI (Yang et al., 2026), ERQA (Team et al., 2025)).

Results. As shown in Table 3, TTR-Compressor outperforms the MLP merger under both compression ratios. Under $2 \times 2$ merging, it raises the average from 54.93 to 55.92, and its convolutional variant further improves it to 56.03. The advantage grows under stronger $3 \times 3$ merging. The gains are most pronounced on text-rich benchmarks, where fine-grained details are easily lost during merging: under $3 \times 3$ merging, the convolutional variant improves over the MLP merger by 4.10 on OCRBench, 4.54 on InfoVQA, and 3.56 on ChartQA. On average, it outperforms the MLP merger by 2.58 points. Notably, TTR-Compressor (w/ Conv) with $3 \times 3$ merging achieves an average of 54.85, close to the 54.93 of the MLP merger with $2 \times 2$ merging, while using fewer than half as many visual tokens. We attribute the strong performance of TTR-Compressor to its secondorder modeling. The MLP merger concatenates the features in each group and maps them with fixed weights, so the mixing does not depend on how the features relate to each other. When most features in a group are redundant, such as background patches, they dominate the merged token and wash out fine-grained details. In contrast, TTR-Compressor fits correlated features jointly and reduces the influence of redundant ones, so the merged token keeps the distinctive content within the group.

## 5.3 EFFICIENCY ANALYSIS

Theoretical Complexity and Parameter Efficiency. For a sequence $\boldsymbol { X } ^ { \bullet } \in \mathbb { R } ^ { \tilde { T } \times d }$ compressed into $h \in \mathbb { R } ^ { \bar { d } }$ Table 4 summarizes the asymptotic complexity and parameter count across different paradigms. Under the targeted regime where $T \ll \bar { d } ( { \bf e . g . } , { \bf \bar { \Delta } } T \in \{ 4 , 9 , 6 4 \} $ vs. $d \geq 1 0 2 4 )$ , TTR-Compressor fundamentally avoids the $O ( d ^ { 2 } )$ overhead and delivers strong, data-dependent expressiveness with only 2d parameters.

<table><tr><td>Paradigm</td><td>Complexity</td><td>Params</td></tr><tr><td>Mean Pooling</td><td>O(Td)</td><td>0</td></tr><tr><td>Linear Projection</td><td>O(Td2)</td><td> $T d ^ { 2 }$ </td></tr><tr><td>Attention (Reusing)</td><td>O(Td)</td><td> $^ d$ </td></tr><tr><td>Attention (Standalone)</td><td> $\mathcal { O } ( T d ^ { 2 } )$ </td><td> $\sim 4 d ^ { 2 }$ </td></tr><tr><td>TTR-Compressor</td><td> $\mathcal { O } ( T ^ { 2 } d )$ </td><td>2d</td></tr></table>

Table 4: Complexity comparison of different token compression methods $( T \ll d )$

Empirical Training Efficiency. Following the setups in Sections 5.1 and 5.2, Figure 3 benchmarks the actual wallclock training efficiency. For block-wise sparse attention (8 × H100 GPUs), while mean pooling is the most efficient, its representational capacity is inherently limited; other compression variants exhibit comparable training times. For visual token merging $( 2 \times 8 \times \mathrm { { H 1 0 0 } G P U s ) }$ , TTR-

![](images/865a685dfe3e21449ea2350bdc471fad6b7a4a58ca5fba9892e4cfce7230313c.jpg)

![](images/09fe9dd6db1b5972ee84432c476ce914c68ee43656b26d894b2bf356f152cc54.jpg)  
Figure 3: Wall-clock training time comparisons.

Compressor achieves efficiency on par with the widely used MLP. Notably, while marginally slower under $2 \times 2$ merging, TTR-Compressor outpaces MLP under $3 \times 3$ merging, highlighting its superior efficiency and scalability at higher compression ratios.

<table><tr><td rowspan="2">Model</td><td colspan="4">S-NIAH-1</td><td colspan="4">S-NIAH-2</td><td colspan="4">S-NIAH-3</td><td colspan="4">MK-NIAH-1</td><td rowspan="2">Avg.</td></tr><tr><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>4K</td><td>8K</td><td>16K</td><td>32K 4K</td><td>8K</td><td>16K</td><td>32K</td></tr><tr><td>TTR-Compressor</td><td>0.996</td><td>0.996</td><td>0.992</td><td>0.650</td><td>0.982</td><td>0.566</td><td>0.288</td><td>0.126</td><td>0.770 0.376</td><td>0.188</td><td>0.066</td><td>0.528</td><td>0.302</td><td>0.212</td><td>0.110</td><td>0.5092</td></tr><tr><td>w/o normalize k</td><td>0.010</td><td>0.000</td><td>0.002</td><td>0.000</td><td>0.938</td><td>0.254 0.136</td><td>0.104</td><td>0.764</td><td>0.592</td><td>0.186</td><td>0.092</td><td>0.500</td><td>0.258</td><td>0.098</td><td>0.116</td><td>0.2531</td></tr></table>

Table 5: Ablation study on key normalization in TTR-Compressor. We compare the full model with a variant without normalizing k (w/o normalize k). Best values for each context length are bolded.

<table><tr><td rowspan="2">Model</td><td colspan="4">S-NIAH-1</td><td colspan="4">S-NIAH-2</td><td colspan="4">S-NIAH-3</td><td colspan="4">MK-NIAH-1</td><td rowspan="2"> $\mathbf { A v } \mathbf { g } .$ </td></tr><tr><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>4K</td><td>8K</td><td>16K</td><td>32K</td></tr><tr><td> $\lambda = 0 . 0 0 0 1$ </td><td>0.986</td><td>0.896</td><td>0.826</td><td>0.628</td><td>0.888</td><td>0.348</td><td>0.134</td><td>0.054</td><td>0.654</td><td>0.528</td><td>0.086</td><td>0.108</td><td>0.384</td><td>0.242</td><td>0.118</td><td>0.102</td><td>0.4364</td></tr><tr><td> $\lambda = 0 . 0 0 1$ </td><td>0.998</td><td>0.996</td><td>0.990</td><td>0.948</td><td>0.998</td><td>0.448</td><td>0.186</td><td>0.112</td><td>0.686</td><td>0.420</td><td>0.186</td><td>0.072</td><td>0.466</td><td>0.274</td><td>0.174</td><td>0.082</td><td>0.5023</td></tr><tr><td> $\lambda = 0 . 0 1$ </td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.936</td><td>0.486</td><td>0.148</td><td>0.076</td><td>0.692</td><td>0.374</td><td>0.042</td><td>0.022</td><td>0.484</td><td>0.288</td><td>0.146</td><td>0.068</td><td>0.4851</td></tr><tr><td> $\lambda = 0 . 1$ </td><td>0.990</td><td>0.986</td><td>0.982</td><td>0.978</td><td>0.932</td><td>0.312</td><td>0.054</td><td>0.068</td><td>0.728</td><td>0.424</td><td>0.088</td><td>0.078</td><td>0.470</td><td>0.252</td><td>0.112</td><td>0.074</td><td>0.4705</td></tr><tr><td>λ = 1</td><td>0.872</td><td>0.552</td><td>0.826</td><td>0.794</td><td>0.896</td><td>0.204</td><td>0.062</td><td>0.072</td><td>0.772</td><td>0.496</td><td>0.184</td><td>0.088</td><td>0.526</td><td>0.278</td><td>0.090</td><td>0.068</td><td>0.4238</td></tr><tr><td>Learnable Scalar</td><td>0.988</td><td>0.986</td><td>0.970</td><td>0.806</td><td>0.938</td><td>0.368</td><td>0.104</td><td>0.104</td><td>0.788</td><td>0.536</td><td>0.212</td><td>0.112</td><td>0.554</td><td>0.274</td><td>0.132</td><td>0.082</td><td>0.4971</td></tr><tr><td>Learnable Diagonal</td><td>0.996</td><td>0.996</td><td>0.992</td><td>0.650</td><td>0.982</td><td>0.566</td><td>0.288</td><td>0.126</td><td>0.770</td><td>0.376</td><td>0.188</td><td>0.066</td><td>0.528</td><td>0.302</td><td>0.212</td><td>0.110</td><td>0.5092</td></tr></table>

Table 6: Ablation study on the regularization parameter λ in TTR-Compressor. Best values for each context length are bolded, and second-best values are underlined

## 5.4 ABLATION STUDY AND ANALYSIS

Criticality of Key Normalization. As shown in Table 5, we find that omitting key normalization $( \mathbf { k } _ { t } = \mathbf { x } _ { t } / \Vert \mathbf { x } _ { t } \Vert _ { 2 } )$ precipitates a collapse across retrieval tasks. We attribute this to two reasons. First, from an optimization perspective, since scalar targets are linearly generated via $v _ { t } = \mathbf { a } ^ { \top } \mathbf { x } _ { t } ,$ using raw keys $( \mathbf { k } _ { t } = \mathbf { x } _ { t } )$ makes the static weight a a trivial exact fit $( \mathbf { k } _ { t } ^ { \top } \mathbf { a } \equiv v _ { t } )$ . Consequently, the regularized solution $\mathbf { h } ^ { \star }$ is biased toward the fixed vector a, which weakens the instance adaptivity of test-time regression and pushes it toward an input-invariant projection. Second, from a geometric and numerical viewpoint, outlier tokens with large norms dominate the Gram matrix $\mathbf { K } \mathbf { K } ^ { \top }$ , causing correlation modeling to reflect token norms rather than semantic similarity. Normalization resolves both issues at once: it breaks the trivial alignment with static parameters and prevents large-norm tokens from dominating the Gram matrix.

From Static Heuristics to Anisotropic Regularization. Table 6 compares fixed regularization scalars against learnable formulations. All fixed settings outperform the baseline. Moderate val ues around $\lambda = 1 0 ^ { - 2 }$ perform well, while performance decreases when λ is too small (0.4364 at $\lambda = 1 0 ^ { - 4 } )$ or too large (0.4238 at $\lambda = 1 )$ . Notably, a Learnable Scalar already achieves a competitive average accuracy of 0.4971, demonstrating that the outer-loop task objective can reliably discover effective regularization strengths end-to-end. Extending this to a coordinate-wise Learnable Diagonal $( \mathbf { D } = \mathrm { d i a g } ( \mathrm { s o f t p l u s } ( \pmb { \rho } ) ) )$ achieves the overall best performance. This progression shows the benefit of learning a separate regularization strength for each channel: the model penalizes less useful feature directions more strongly while preserving task-critical ones.

## 6 CONCLUDING REMARKS

In this work, we revisit sequence compression, a pervasive yet often overlooked bottleneck in modern architectures, through the lens of test-time regression. Instead of relying on fixed pooling or learned projections, we formulate token compression as an instance-adaptive ridge regression problem, allowing the compressed representation to be solved directly from each input sequence. By exploiting inter-token correlations through second-order interactions, TTR-Compressor offers a more expressive way to preserve sequence information while remaining lightweight and parameterefficient. Its dual-form solution further makes this formulation scalable in practice. Empirically, TTR-Compressor performs strongly across both block-wise sparse attention and multimodal token reduction, improving long-context retrieval while matching standard visual mergers with significantly fewer tokens. More broadly, our results suggest that sequence compression need not be viewed as static aggregation, but as an input-dependent inference problem that adapts to the structure of each sequence. We expect this perspective establishes test-time regression as a simple, expressive, and scalable primitive for efficient foundation models.

## REFERENCES

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 2, 7

Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni. Nested learning: The illusion of deep learning architectures. NeurIPS, 2026a. 3

Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. NeurIPS, 2026b. 3

Luca Bertinetto, Joao F Henriques, Philip HS Torr, and Andrea Vedaldi. Meta-learning with differentiable closed-form solvers. arXiv preprint arXiv:1805.08136, 2018. 3

Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022. 2

Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Locality-enhanced projector for multimodal llm. In CVPR, 2024. 2

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? NeurIPS, 2024. 8

Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024. 2

DeepSeek-AI. Deepseek-v4.1-flash: Pushing the limits of kv cache compression, 2026. 1, 7

Guhao Feng, Shengjie Luo, Kai Hua, Ge Zhang, Wenhao Huang, Di He, and Tianle Cai. In-place test-time training. In ICLR, 2026. 3

Yizhao Gao, Zhichen Zeng, Dayou Du, Shijie Cao, Peiyuan Zhou, Jiaxing Qi, Junjie Lai, Hayden Kwok-Hay So, Ting Cao, Fan Yang, et al. Seerattention: Learning intrinsic sparse attention in your llms. arXiv preprint arXiv:2410.13276, 2024. 2

Dongchen Han, Yining Li, Tianyu Li, Zixuan Cao, Ziming Wang, Jun Song, Yu Cheng, Bo Zheng, and Gao Huang. Vit3: Unlocking test-time training in vision. In CVPR, 2026. 3

Ali Hatamizadeh, Yejin Choi, and Jan Kautz. Gated deltanet-2: Decoupling erase and write in linear attention. arXiv preprint arXiv:2605.22791, 2026. 6

Arthur E Hoerl and Robert W Kennard. Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1):55–67, 1970. 3

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024. 2, 6

Xiang Hu, Xinyu Wei, Hao Gu, Minshen Zhang, Tian Liang, Huayang Li, Lei Zhu, Yan Wang, Sirui Han, Yushi Bai, et al. Hierarchical sparse attention done right: Toward infinite context modeling. arXiv preprint arXiv:2607.02980, 2026. 2, 6, 18

Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H Abdi, Dongsheng Li, Chin-Yew Lin, et al. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. NeurIPS, 2024. 2

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, 2016. 8

Xunhao Lai, Jianqiao Lu, Yao Luo, Yiyuan Ma, and Xun Zhou. Flexprefill: A context-aware sparse attention mechanism for efficient long-sequence inference. arXiv preprint arXiv:2502.20766, 2025. 2

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next: Tackling multi-image, video, and 3d in large multimodal models, June 2024. URL https://llava-vl.github.io/blog/ 2024-06-16-llava-next-interleave/. 8

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023. 2

Bo Liu, Rui Wang, Lemeng Wu, Yihao Feng, Peter Stone, and Qiang Liu. Longhorn: State space models are amortized online learners. In ICLR, 2025. 3

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26286–26296. IEEE, 2024a. 8

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024b. URL https:// llava-vl.github.io/blog/2024-01-30-llava-next/. 8

Weiyang Liu, Yan-Ming Zhang, Xingguo Li, Zhiding Yu, Bo Dai, Tuo Zhao, and Le Song. Deep hyperspherical learning. In NeurIPS, 2017. 4

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In ECCV, 2024c. 8

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 2024d. 8

Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang, Chao Hong, Shaowei Liu, Weiran He, Enming Yuan, Yuzhi Wang, Zhiqi Huang, Huan Yuan, Suting Xu, Xinran Xu, Guokun Lai, Yanru Chen, Huabin Zheng, Junjie Yan, Jianlin Su, Yuxin Wu, Yutao Zhang, Zhilin Yang, Xinyu Zhou, Mingxing Zhang, and Jiezhong Qiu. Moba: Mixture of block attention for long-context llms. arXiv preprint arXiv:2502.13189, 2025. 2

Ahmed Masry, Jia Qing Tan, Shafiq Joty, Enamul Hoque, et al. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings ofACL, 2022. 8

Amirkeivan Mohtashami and Martin Jaggi. Landmark attention: Random-access infinite context length for transformers. arXiv preprint arXiv:2305.16300, 2023. 2

Guilherme Penedo, Hynek Kydlícek, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandroˇ Von Werra, Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale. NeurIPS, 2024. 6

Qwen Team. Qwen3.5: Towards native multimodal agents, 2026a. URL https://qwen.ai/ blog?id=qwen3.5. 7

Qwen Team. Qwen3.8-Flash-Next: A new architecture, towards ultimate cost-efficiency, 2026b. 1

Craig Saunders, Alexander Gammerman, and Volodya Vovk. Ridge regression learning algorithm in dual variables. In ICML, 1998. 5

Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers. In ICML, 2021. 3

Jürgen Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic recurrent networks. Neural Computation, 4(1):131–139, 1992. 3

Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt. Test-time training with self-supervision for generalization under distribution shifts. In ICML, 2020. 3

Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, et al. Learning to (learn at test time): Rnns with expressive hidden states. arXiv preprint arXiv:2407.04620, 2024. 3

Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025. 8

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle Casbon, et al. Gemma 4 technical report.˘ arXiv preprint arXiv:2607.02770, 2026a. 2, 7

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi k3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026b. 7

Andrey N. Tikhonov and Vasiliy Y. Arsenin. Solutions of ill-posed problems. V. H. Winston & Sons, Washington, D.C.: John Wiley & Sons, New York, 1977. Translated from the Russian, Preface by translation editor Fritz John, Scripta Series in Mathematics. 4

Rubèn Tito, Minesh Mathew, CV Jawahar, Ernest Valveny, and Dimosthenis Karatzas. Icdar 2021 competition on document visual question answering. In ICDAR, 2021. 8

Johannes von Oswald, Nino Scherrer, Seijin Kobayashi, Luca Versari, Songlin Yang, Maximil ian Schlegel, Kaitlin Maile, Yanick Schimpf, Oliver Sieberling, Alexander Meulemans, et al. Mesanet: Sequence modeling by locally optimal test-time training. In ICLR, 2026. 3

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 2, 7

xAI. Grok-1.5 vision preview. https://x.ai/news/grok-1.5v, April 2024. 8

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a. 7

Sihan Yang, Runsen Xu, Chenhang Cui, Tai Wang, Dahua Lin, and Jiangmiao Pang. Vflowopt: A token pruning framework for lmms with visual information flow-guided optimization. In ICCV, 2025b. 2

Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multi-image spatial intelligence. In ICLR, 2026. 8

Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. NeurIPS, 2024. 3

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In ICLR, 2025c. 3

Linli Yao, Lei Li, Shuhuai Ren, Lean Wang, Yuanxin Liu, Xu Sun, and Lu Hou. Deco: Decoupling token compression from semantic abstraction in multimodal large language models. arXiv preprint arXiv:2405.20985, 2024. 2

Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Yuxing Wei, Lean Wang, Zhiping Xiao, et al. Native sparse attention: Hardware-aligned and natively trainable sparse attention. In ACL, 2025. 2

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024. 8

Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William Freeman, and Hao Tan. Test-time training done right. In ICLR, 2026. 3

Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 8

## Appendix

A Derivation of the Dual-form Solutions 15   
B Redundancy-Aware Weighting of TTR-Compressor 16   
C Implementation Details 18   
D Experimental Details 18   
E Additional Experimental Results 18   
E.1 Scaling Up Model Size and Training Budget 18   
E.2 Ablation on the Value Parameterization 18   
F Inference Latency of Block-wise Sparse Attention 19

## A DERIVATION OF THE DUAL-FORM SOLUTIONS

We first prove the equivalence between the primal and dual forms of ridge regression (Eq. 6). We then show that, with learnable diagonal regularization, rescaling the keys yields the dual solution in Eq. 7.

Equivalence of the Primal and Dual Forms.

Lemma 1 (Push-through identity). For any $K \in \mathbb { R } ^ { T \times d }$ and $\lambda > 0 ;$

$$
( { \pmb { K } } ^ { \top } { \pmb { K } } + \lambda { \pmb { I } } _ { d } ) ^ { - 1 } { \pmb { K } } ^ { \top } = { \pmb { K } } ^ { \top } ( { \pmb { K } } { \pmb { K } } ^ { \top } + \lambda { \pmb { I } } _ { T } ) ^ { - 1 } .\tag{13}
$$

Proof. Both $\kappa ^ { \top } \kappa$ and K $\pmb { K } ^ { \top }$ are positive semidefinite. Hence, for $\lambda > 0$ , the matrices $\pmb { K } ^ { \top } \pmb { K } +$ $\lambda I _ { d }$ and $K K ^ { \top } + \lambda I _ { T }$ are symmetric positive definite and therefore invertible. Expanding both sides directly gives

$$
( \pmb { K } ^ { \top } \pmb { K } + \lambda \pmb { I } _ { d } ) \pmb { K } ^ { \top } = \pmb { K } ^ { \top } \pmb { K } \pmb { K } ^ { \top } + \lambda \pmb { K } ^ { \top } = \pmb { K } ^ { \top } ( \pmb { K } \pmb { K } ^ { \top } + \lambda \pmb { I } _ { T } ) .\tag{14}
$$

Left-multiplying by $( K ^ { \top } K + \lambda I _ { d } ) ^ { - 1 }$ and right-multiplying by $( K K ^ { \top } + \lambda I _ { T } ) ^ { - 1 }$ yields Eq. 13.

Right-multiplying Eq. 13 by v turns the primal solution in Eq. 2 into the dual form in Eq. 6:

$$
\pmb { h } ^ { \star } = ( \pmb { K } ^ { \top } \pmb { K } + \lambda \pmb { I } _ { d } ) ^ { - 1 } \pmb { K } ^ { \top } \pmb { v } = \pmb { K } ^ { \top } ( \pmb { K } \pmb { K } ^ { \top } + \lambda \pmb { I } _ { T } ) ^ { - 1 } \pmb { v } .\tag{15}
$$

Therefore, $h ^ { \star }$ lies in the row space of $\kappa$ . Its dual coefficients $\pmb { \alpha } = ( \pmb { K } \pmb { K } ^ { \top } + \lambda \pmb { I } _ { T } ) ^ { - 1 } \pmb { v } \in \mathbb { R } ^ { T }$ are obtained by solving a single $T \times T$ system.

Dual Solution under Diagonal Regularization.

Proposition 1. Let $\pmb { D } = \mathrm { d i a g } ( \delta _ { 1 } , \dots , \delta _ { d } )$ with $\delta _ { j } > 0 ,$ , and let $\widetilde { K } = K D ^ { - 1 / 2 }$ . Then the solution of $E q . 4$ satisfies

$$
\begin{array} { r } { ( K ^ { \top } K + D ) ^ { - 1 } K ^ { \top } { \boldsymbol v } = D ^ { - 1 / 2 } \widetilde { \boldsymbol K } ^ { \top } \big ( \boldsymbol I _ { T } + \widetilde { K } \widetilde { \boldsymbol K } ^ { \top } \big ) ^ { - 1 } \boldsymbol v . } \end{array}\tag{16}
$$

Proof. Since D is diagonal with positive entries, $D ^ { \pm 1 / 2 }$ are well defined and $D ^ { 1 / 2 } D ^ { - 1 / 2 } = I _ { d }$ Using $K = \widetilde { K } D ^ { 1 / 2 }$ , we factorize

$$
\begin{array} { r } { K ^ { \top } K + D = D ^ { 1 / 2 } \big ( \widetilde { K } ^ { \top } \widetilde { K } + I _ { d } \big ) D ^ { 1 / 2 } . } \end{array}\tag{17}
$$

Inverting this product and using $K ^ { \top } = D ^ { 1 / 2 } \widetilde { K } ^ { \top }$ gives

$$
\begin{array} { r } { ( K ^ { \top } K + D ) ^ { - 1 } K ^ { \top } v = D ^ { - 1 / 2 } \big ( \widetilde { K } ^ { \top } \widetilde { K } + I _ { d } \big ) ^ { - 1 } D ^ { - 1 / 2 } D ^ { 1 / 2 } \widetilde { K } ^ { \top } v = D ^ { - 1 / 2 } \big ( \widetilde { K } ^ { \top } \widetilde { K } + I _ { d } \big ) ^ { - 1 } \widetilde { K } ^ { \top } v . } \end{array}\tag{18}
$$

Applying Lemma 1 with $K \gets \widetilde K$ and $\lambda = 1$ gives Eq. 16.

Equivalently, the change of variables $\pmb { g } = \pmb { D } ^ { 1 / 2 } \pmb { h }$ turns Eq. 4 into a standard ridge problem, $\frac { 1 } { 2 } \lVert \widetilde { K } g -$ $\pmb { v } \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 } \| \pmb { g } \| _ { 2 } ^ { 2 }$ . Its dual solution is then mapped back through ${ \pmb h } ^ { \star } = { \pmb D } ^ { - 1 / 2 } { \pmb g } ^ { \star }$

Two further properties follow:

• The matrix $I _ { T } + \widetilde { K } \widetilde { K } ^ { \intercal }$ is symmetric positive definite, so its Cholesky factorization exists and is unique.

• Because D is diagonal, forming $\widetilde { \kappa }$ and applying $D ^ { - 1 / 2 }$ each cost only $\mathcal { O } ( T d )$ and $\mathcal O ( d )$ The overall complexity therefore remains $\mathcal { O } ( T ^ { 2 } \mathsf { \breve { d } } + T ^ { 3 } )$

Setting $\pmb { D } = \lambda \pmb { I } _ { d }$ recovers Eq. 6.

## B REDUNDANCY-AWARE WEIGHTING OF TTR-COMPRESSOR

We formalize the claim in Section 4.2 that redundant tokens have diminishing influence, while information not explained by correlated tokens keeps a larger relative influence. We take isotropic regularization $\lambda >$ 0 and treat the targets v as arbitrary, so the results hold for any learned projection a. The extension to the learnable D is given at the end. Throughout, let $\pmb { A } = \pmb { \dot { K } } ^ { \top } \pmb { K } + \lambda \bar { \pmb { I } } _ { d }$ , and let ${ \pmb { \alpha } } = ( { \pmb G } + \lambda { \pmb I } _ { T } ) ^ { - 1 } { \pmb v }$ , so that $h ^ { \star } = K ^ { \top } \alpha$

## Effective Weights as Unexplained Residuals.

Proposition 2 (Residual form). For any keys K and targets v, the effective weights satisfy

$$
\pmb { \alpha } = \frac { 1 } { \lambda } \big ( \pmb { v } - \pmb { K } \pmb { h } ^ { \star } \big ) .\tag{19}
$$

Proof. From $( G + \lambda I _ { T } ) \pmb { \alpha } = \pmb { v }$ we have $\lambda \pmb { \alpha } = \pmb { v } - \pmb { K } \pmb { K } ^ { \top } \pmb { \alpha } = \pmb { v } - \pmb { K } \pmb { h } ^ { \star }$

Thus, a token is weighted by the part of its target that the jointly fitted representation leaves unexplained. By contrast, additive pooling assigns weights v regardless of how the other tokens are arranged. To separate the role of the other tokens, let $\bar { \ b { K } } _ { - t }$ and ${ \pmb v } _ { - t }$ denote the keys and targets with token t removed. Define

$$
A _ { - t } = K _ { - t } ^ { \top } K _ { - t } + \lambda I _ { d } , \qquad h _ { - t } = A _ { - t } ^ { - 1 } K _ { - t } ^ { \top } v _ { - t } ,\tag{20}
$$

where $\mathbf { \delta } _ { h _ { - t } }$ is the solution fitted without token t.

Proposition 3 (Leave-one-out form). Let $r _ { t } = v _ { t } - k _ { t } ^ { \top } h _ { - t }$ be the residual of token t under the other tokens, and let $q _ { t } = k _ { t } ^ { \top } A _ { - t } ^ { - 1 } k _ { t } \geq 0$ . Then

$$
\alpha _ { t } = \frac { r _ { t } } { \lambda ( 1 + q _ { t } ) } , \qquad h ^ { \star } = h _ { - t } + \frac { r _ { t } } { 1 + q _ { t } } A _ { - t } ^ { - 1 } k _ { t } .\tag{21}
$$

Proof. Write $\pmb { A } = \pmb { A } _ { - t } + \pmb { k } _ { t } \pmb { k } _ { t } ^ { \top }$ and $K ^ { \top } \pmb { v } = \pmb { K } _ { - t } ^ { \top } \pmb { v } _ { - t } + v _ { t } \pmb { k } _ { t }$ . Substituting into $A h ^ { \star } = K ^ { \top } \cdot$ v gives

$$
\pmb { h } ^ { \star } = \pmb { h } _ { - t } + \left( v _ { t } - \pmb { k } _ { t } ^ { \top } \pmb { h } ^ { \star } \right) \pmb { A } _ { - t } ^ { - 1 } \pmb { k } _ { t } .\tag{22}
$$

Taking the inner product with $k _ { t }$ gives $\pmb { k } _ { t } ^ { \top } \pmb { h } ^ { \star } = \pmb { k } _ { t } ^ { \top } \pmb { h } _ { - t } + \left( v _ { t } - \pmb { k } _ { t } ^ { \top } \pmb { h } ^ { \star } \right) q _ { t }$ , and hence $v _ { t } - k _ { t } ^ { \top } h ^ { \star } =$ $r _ { t } / ( 1 + q _ { t } )$ . Substituting this into the display above and into Eq. 19 yields Eq. 21. □

Proposition 3 shows that token t changes the representation only through $r _ { t }$ , the part of its target that the remaining tokens cannot predict. In particular, if the other tokens fully explain token $t \left( r _ { t } = 0 \right)$ then $\alpha _ { t } = 0$ and $h ^ { \star } = h _ { - t }$ , so the token has no influence. Additive pooling instead always adds $v _ { t } k _ { t }$

Saturation under Repeated Evidence. We now quantify this effect for groups of correlated tokens, generalizing the example in Figure 2(b).

Proposition 4 (Group saturation). Suppose the tokens form C groups. Group c contains $m _ { c }$ tokens that share the key ${ \mathbf { } } u _ { c } ,$ with $\| \pmb { u } _ { c } \| _ { 2 } = \bar { 1 }$ and $\pmb { u } _ { c } ^ { \top } \pmb { u } _ { c ^ { \prime } } = \bar { 0 } f o r c \neq c ^ { \prime } .$ . Let $S _ { c }$ be the sum of the targets in group c. Then

$$
\begin{array} { r } { \pmb { h } ^ { \star } = \displaystyle \sum _ { c = 1 } ^ { C } \frac { S _ { c } } { \lambda + m _ { c } } \pmb { u } _ { c } , \qquad w h e r e a s \qquad \pmb { K } ^ { \top } \pmb { v } = \displaystyle \sum _ { c = 1 } ^ { C } S _ { c } \pmb { u } _ { c } . } \end{array}\tag{23}
$$

Proof. Under this assumption G is block diagonal, and the block for group c is $J _ { m _ { c } } = \mathbf { 1 1 } ^ { \top }$ . Since $J _ { m } ^ { 2 } = m J _ { m }$ , one can verify that $( \pmb { J _ { m } } + \lambda \pmb { I _ { m } } ) ^ { - 1 } = \lambda ^ { - 1 } \big ( \pmb { I _ { m } } - \pmb { J _ { m } } / ( \lambda + m ) \big )$ . Hence the weights $\pmb { \alpha } _ { c }$ of group c sum to

$$
\mathbf { 1 } ^ { \top } \alpha _ { c } = { \frac { 1 } { \lambda } } { \Big ( } S _ { c } - { \frac { m _ { c } S _ { c } } { \lambda + m _ { c } } } { \Big ) } = { \frac { S _ { c } } { \lambda + m _ { c } } } .\tag{24}
$$

Therefore $\begin{array} { r } { \pmb { h } ^ { \star } = \sum _ { c } \pmb { u } _ { c } \pmb { 1 } ^ { \top } \pmb { \alpha } _ { c } , } \end{array}$ which gives Eq. 23.

Corollary 1 (Diminishing influence of redundancy). Assume every token has the same target $v > 0 ,$ so that $S _ { c } = m _ { c } v$ . Compare a group of m repeated tokens with a single distinct token. Their relative influence is

$$
\frac { m / ( \lambda + m ) } { 1 / ( \lambda + 1 ) } = \frac { m ( \lambda + 1 ) } { \lambda + m } < m \mathrm { f o r } m > 1 .\tag{25}
$$

This ratio is bounded by $\lambda + 1$ for every m, whereas the corresponding ratio for additive pooling is m. Moreover, the n-th repeated token increases the group’s coefficient by

$$
{ \frac { n v } { \lambda + n } } - { \frac { ( n - 1 ) v } { \lambda + n - 1 } } = { \frac { \lambda v } { ( \lambda + n ) ( \lambda + n - 1 ) } } = \mathcal { O } ( n ^ { - 2 } ) ,\tag{26}
$$

which decays with n, so repeated evidence has diminishing returns.

The ridge strength controls the transition between two regimes:

$\operatorname { A s } \lambda \to \infty , \lambda \pmb { h } ^ { \star }  \pmb { K } ^ { \top } \imath$ v, which is additive pooling.

$\mathrm { { A s } } \lambda  0$ , each group contributes $S _ { c } / m _ { c } ,$ , its mean target, regardless of how many tokens it contains.

Setting λ = 1, m = 3 recovers Figure 2(b): the ratio is 3:2 and $\begin{array} { r } { \pmb { \alpha } = ( \frac { 1 } { 4 } , \frac { 1 } { 4 } , \frac { 1 } { 4 } , \frac { 1 } { 2 } ) ^ { \top } } \end{array}$

Extension to Learnable Diagonal Regularization. All results carry over to learnable diagonal regularization. Using the change of variables in Proposition 1, set $\pmb { \alpha } = ( \pmb { I } _ { T } + \widetilde { \pmb { K } } \widetilde { \pmb { K } } ^ { \top } ) ^ { - 1 } \pmb { v }$ , so that $h ^ { \star } = D ^ { - 1 } K ^ { \top } \alpha$ . Since $K h ^ { \star } = \widetilde { K } \widetilde { K } ^ { \top } \alpha$ , the argument of Proposition 2 gives $\alpha = v - K h ^ { \star }$ Proposition 3 then holds with $\lambda = 1$ and $\pmb { A } _ { - t } = \mathbf { \bar { K } } _ { - t } ^ { \top } \pmb { K } _ { - t } + \bar { \pmb { D } }$ . Proposition 4 holds when the groups are orthogonal under the learned metric $( { \pmb u } _ { c } ^ { \top } D ^ { - 1 } { \pmb u } _ { c ^ { \prime } } = 0$ for $c \neq c ^ { \prime } )$ , with λ replaced by the group-specific $\lambda _ { c } = ( { \pmb u } _ { c } ^ { \top } D ^ { - 1 } { \pmb u } _ { c } ) ^ { - 1 }$ and ${ \bf u } _ { c }$ replaced by $\lambda _ { c } D ^ { - 1 } { \pmb u } _ { c }$ The learned D therefore decides which feature directions count as redundant, while the diminishing-influence mechanism itself is unchanged.

## C IMPLEMENTATION DETAILS

Numerical Precision. Convolution and target projection use the input precision (BF16), while regression keys and targets are cast to FP32 before the dual solve. Diagonal regularization is evaluated in FP32. We solve the positive-definite dual system using Cholesky factorization without explicitly forming an inverse, and cast the compressed representation back to the input dtype.

Local Target Construction. Only the regression targets incorporate convolutional context; keys are computed from the original inputs. For language, we apply a non-causal depthwise 1D convolution with kernel size 5, independently within each compression block, using two-sided zero padding of width 2. The convolution has neither bias nor activation and is initialized to the identity. For vision, we use a 3×3 depthwise convolution. Each channel has its own spatial kernel, with no cross-channel mixing before the scalar target projection.

The base compressor has 2d effective trainable parameters: d for the target projection and d for diagonal regularization. A bias-free depthwise convolution with s kernel elements adds sd parameters and $\mathcal { O } ( s \bar { T } d )$ computation. Thus, the language and vision variants require 7d and 11d parameters, respectively. Their total computational complexity remains $\mathcal { O } ( T ^ { 2 } d )$ for fixed kernel sizes and $T \ll d .$

## D EXPERIMENTAL DETAILS

Block-wise Sparse Attention. HiLS-Attention (Hu et al., 2026) argues that small models provide a more diagnostic setting for RULER’s predominantly “find-and-copy” tasks, as larger models may compensate for retrieval inaccuracies through greater information bandwidth. We therefore follow their setup and use 345M-parameter models to evaluate compression quality. All models use 16 Transformer layers, a hidden dimension of 1,024, an FFN dimension of 4,096, and grouped-query attention with 16 query heads and 2 key–value heads. We use rank-64 query calibration. Training uses AdamW weight decay of 0.01 and gradient clipping at a norm of 1.0; normalization parameters, biases, and embeddings are excluded from weight decay. The learning rate warms up over the first 0.7% of training and remains constant thereafter.

Visual Token Merger. We use BF16 training with DeepSpeed ZeRO-1, gradient checkpointing, and a maximum sequence length of 16,384 tokens. The two stages use per-device batch sizes of 4 and 1, with gradient accumulation over 4 and 8 steps, respectively. Both stages apply a 3% learning-rate warmup followed by cosine decay, with minimum learning rates of $1 0 ^ { - 5 }$ and 0 and weight decay of 0.01 and 0.02, respectively. For TTR-Compressor, the regression systems are solved in FP32.

## E ADDITIONAL EXPERIMENTAL RESULTS

## E.1 SCALING UP MODEL SIZE AND TRAINING BUDGET

To further verify that the advantage of TTR-Compressor is not merely an artifact of the smaller-scale setting used in the main paper, we scale up both the model size and the training budget. Specifically, we train a 1B-parameter model on 100B tokens and evaluate it on the same RULER sub tasks across context lengths ranging from 4K to 32K. For comparison, we also scale up Attention-Weighted Pooling—the strongest baseline among all compression methods evaluated in the main paper’s 345M-parameter experiments—to the same 1B-parameter, 100B-token setting.

As shown in Table 7, TTR-Compressor consistently outperforms the Attention-Weighted Pooling baseline across almost all context lengths and subtasks, achieving an average accuracy of 0.7996 compared to 0.735. Notably, the performance gap widens as the context length increases, indicating that TTR-Compressor scales more gracefully to longer contexts even when both model size and training budget are substantially increased relative to the main paper’s 345M-parameter setting.

## E.2 ABLATION ON THE VALUE PARAMETERIZATION

Under the $3 \times 3$ compression setup, we compare five parameterizations of v : (i) constant $( v _ { t } = 1 )$ (ii) linear $( v _ { t } = \pmb { a } ^ { \top } \pmb { x } _ { t } ) ;$ (iii) $M L P \left( v _ { t } = \mathrm { M L P } ( \pmb { x } _ { t } ) \right)$ (iv) bilinear $\begin{array} { r } { ( v _ { t } = \mathrm { M L P } ( \pmb { x } _ { t } ) ^ { \top } \pmb { x } _ { t } ) ; } \end{array}$ and (v) their convolutional variants using local $3 \times 3$ features $\operatorname { C o n v } ( X )$ <sub>t</sub>.

<table><tr><td rowspan="2">Model</td><td colspan="4">S-NIAH-1</td><td colspan="4">S-NIAH-2</td><td colspan="3">S-NIAH-3</td><td colspan="4">MK-NIAH-1</td><td rowspan="2">Avg.</td><td rowspan="2"></td></tr><tr><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>4K</td><td>8K</td><td>16K</td><td>32K</td><td>4K</td><td>8K</td><td>16K</td><td>32K 4K</td><td>8K</td><td>16K 32K</td></tr><tr><td>Attention-Weighted Pooling</td><td>0.978</td><td>0.972</td><td>0.984</td><td>0.990</td><td>1.000</td><td>0.982</td><td>0.860</td><td>0.732</td><td>0.898</td><td>0.856</td><td>0.456</td><td>0.550</td><td>0.494 0.358</td><td>0.336</td><td>0.314</td><td>0.735</td></tr><tr><td>TTR-Compressor</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.956</td><td>0.862</td><td>0.750</td><td>0.980</td><td>0.918 0.740</td><td>0.662</td><td>0.626</td><td>0.442</td><td>0.446</td><td>0.412</td><td>0.7996</td></tr></table>

Table 7: Accuracy on Single Needle-In-A-Haystack (S-NIAH) and Multi-Key Needle-In-A-Haystack (MK-NIAH) tasks from RULER across varied context lengths. Results are based on a 1B-parameter model trained on 100B tokens. Best values for each context length are bolded.

<table><tr><td rowspan="2">Method</td><td rowspan="2">vt</td><td colspan="3">General VQA</td><td colspan="2">Knowledge Reasoning</td><td colspan="3">Document Understanding</td><td colspan="2">Spatial Under. MMSI ERQA</td><td rowspan="2">Avg.</td></tr><tr><td>MMBench MMStar</td><td></td><td>RWQA</td><td>MMMU</td><td>AI2D</td><td></td><td>OCRBench InfoVQA</td><td>ChartQA</td><td></td><td></td></tr><tr><td>Constant value</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Constant</td><td>1</td><td>75.26</td><td>51.63</td><td>64.31</td><td>45.67</td><td>76.07</td><td>55.90</td><td>37.84</td><td>67.20</td><td>26.80</td><td>35.00</td><td>53.56</td></tr><tr><td>Linear value</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TTR-Compressor</td><td> $\mathbf { \pmb { a } } ^ { \top } \mathbf { \pmb { x } } _ { t }$ </td><td>76.46</td><td>52.40</td><td>65.62</td><td>45.89</td><td>75.23</td><td>58.70</td><td>38.27</td><td>67.16</td><td>26.80</td><td>36.50</td><td>54.30</td></tr><tr><td>TTR-Compressor (w/ Conv)</td><td> $\underline { { \pmb { a } ^ { \top } \operatorname { C o n v } ( \pmb { X } ) _ { t } } }$ </td><td>76.63</td><td>51.16</td><td>65.88</td><td>47.44</td><td>75.16</td><td>60.00</td><td>39.70</td><td>67.16</td><td>28.90</td><td>36.50</td><td>54.85</td></tr><tr><td>MLP value</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MLP</td><td> $\mathrm { M L P } ( \pmb { x } _ { t } )$ </td><td>76.29</td><td>52.06</td><td>63.92</td><td>44.78</td><td>76.42</td><td>58.30</td><td>38.66</td><td>67.00</td><td>28.10</td><td>37.50</td><td>54.30</td></tr><tr><td>MLP (w/ Conv)</td><td> $\underline { { \mathrm { M L P } ( \mathrm { C o n v } ( \pmb X ) _ { t } ) } }$ </td><td>76.46</td><td>50.87</td><td>64.18</td><td>44.56</td><td>75.32</td><td>58.10</td><td>38.09</td><td>66.80</td><td>26.50</td><td>38.25</td><td>53.91</td></tr><tr><td>MLP-dot value</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MLP-dot</td><td> $\mathrm { M L P } ( \mathbf { x } _ { t } ) ^ { \top } \mathbf { x } _ { t }$ </td><td>76.20 76.29</td><td>51.84</td><td>65.23</td><td>47.33</td><td>75.55</td><td>59.60</td><td>38.47</td><td>68.88</td><td>27.40</td><td>35.00</td><td>54.55</td></tr><tr><td>MLP-dot (w/ Conv)</td><td> $\mathrm { M L P } ( \mathrm { C o n v } ( \pmb { X } ) _ { t } ) ^ { \top } \pmb { x } _ { t }$ </td><td></td><td>51.08</td><td>64.05</td><td>46.56</td><td>75.26</td><td>58.10</td><td>38.81</td><td>66.36</td><td>29.90</td><td>37.25</td><td>54.37</td></tr></table>

Table 8: Ablation on the parameterization of the value target ${ \mathbf { } } v _ { t }$ in the TTR objective $\begin{array} { r } { \pmb { h } ^ { \star } = \arg \operatorname* { m i n } _ { \pmb { h } } \frac { 1 } { 2 } \| \pmb { K } \pmb { h } - \pmb { \tilde { h } } \| _ { 2 } } \end{array}$ $\pmb { v } \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 } \pmb { h } ^ { \top } \pmb { D } \pmb { h }$ . All variants use $3 \times 3$ token compression. Gray rows denote the TTR-Compressor configu rations used in the main paper. Best values are bolded, and second-best values are underlined.

As shown in Table 8, we observe: (1) Content dependence is critical: The constant target yields the lowest average (53.56), while the linear target gains +0.74, showing that regression targets must encode token content to form effective summaries. (2) Deeper heads offer marginal gains: Plain MLP matches the linear baseline (54.30), and adding convolution degrades it (53.91). Although the bilinear head slightly improves performance (54.55), its convolutional counterpart sees no further gain (54.37). (3) Spatial context outweighs head capacity: Combining local convolution with a linear readout, $v _ { t } = \pmb { a } ^ { \top } \mathrm { C o n v } ( \pmb { X } ) _ { t } .$ , achieves the best overall score (54.85). These results indicate that value capacity is not the bottleneck; local neighborhood aggregation provides a more effective inductive bias than non-linear projection.

## F INFERENCE LATENCY OF BLOCK-WISE SPARSE ATTENTION

As shown in Figure 4, all compression methods exhibit similar prefill and decode latency and substantially outperform full attention for context lengths beyond 16K tokens.

![](images/8c4a3e2ebdca570fe0085374be35dbc77a3926376adebef9edc621f586dec69b.jpg)  
Figure 4: Prefill and decode latency across context lengths.