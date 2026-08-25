# Spectrum-Aware Bounds on Invertibility for Privacy-Enhancing Instance Encoding

Seokjin Hwang<sup>∗</sup> The Pennsylvania State University

Yuting (Ray) Li<sup>∗</sup>

The Pennsylvania State University

Kiwan Maeng The Pennsylvania State University

## Abstract

Instance encoding is a popular empirical technique for privacy enhancement when sharing data to an untrusted server. It transforms sensitive data through an encoding process before sharing, with the hope that the encoding process retains utility but makes it hard to reconstruct the original data. However, most work offers no theoretical guarantee that the encoding process is actually irreversible. A recent work derived a mean-squared error (MSE) bound limiting any adversary’s reconstruction accuracy, offering one of the first theoretical results in this domain. This bound, however, has three critical limitations: it is often too loose, only works with randomized encoders (excluding many deterministic encoders practitioners use), and only bounds MSE. We introduce a family of new bounds that (1) are tighter, (2) applicable even to fully deterministic encoders, and (3) can extend beyond MSE to other norm-based similarity metrics, by properly accounting for the encoder’s spectral structure. We evaluate our bounds across a range of encoders, datasets, and attacks, showing they hold consistently and improve upon the existing bound.

## 1 Introduction

Machine learning (ML) applications frequently operate on sensitive data during both training and inference. A medical model that predicts a diagnosis from a patient’s X-ray image [1], for instance, takes protected health information as its input. When training or inference runs on a remote server that users neither trust nor control, users must send their sensitive data to the server—exposing it to privacy risks.

Instance encoding [3] is a popular heuristic approach to improving privacy in such settings. The idea is to process the sensitive data with a privacy-enhancing encoder that outputs an embedding, and to release the embedding instead of the raw data (Figure 1). The hope is that a suitably designed encoder makes it hard to invert the embedding and recover the original data, while still preserving enough information for downstream tasks, such as training or inference. The same idea appears under a variety of names: data obfuscation [61, 63], learnable encryption [22,58,59,61,62], split learning [42, 52], split inference [10,26,33], and vertical federated learning (vFL) [32, 49, 65] all build on the same idea.

Unfortunately, most existing work in this area is heuristic and lacks a rigorous theoretical foundation: it is unclear what privacy properties, if any, these encoders actually provide. Relying on empirical evaluation alone is unsound, since a scheme that can resist all known attacks and considered safe may later be successfully attacked. In fact, the community has already seen several such failures [3, 4]. It is therefore important to understand, in a principled way, which privacy properties these encoders do and do not provide, and how the invertibility of an encoder can be quantified.

One notable step in this direction is the work of Maeng et al. [35], who derived a mean-squared error (MSE) bound limiting how precisely an (arbitrarily strong) adversary can reconstruct the original data from its embedding (Section 3.1). While this work made important progress towards a theoretical foundation, its bound has several limitations. First, although tight in some cases, the bound becomes very loose in other cases. Second, it applies only to randomized encoders. This renders it inapplicable to many DNN-based encoders proposed by practitioners, which has no randomness [8, 31–33, 45, 53, 54, 63]. Finally, the bound only constrains the MSE between the original and reconstructed data. MSE, while frequently used as a proxy to data similarity [17, 35], is not always a faithful measure of the reconstruction quality: a reconstruction can have high MSE while still leaking sensitive semantic information, or low MSE without meaningful leakage.

In this paper, we provide a set of improved bounds that address these limitations. Our new bounds better account for the spectral geometry of the encoder, yielding tighter bounds in cases where the prior work [35] fell short. Our bounds also stay non-trivial even when the encoder induces no randomness, by correctly capturing how the information loss through the encoder’s spectral geometry makes reconstruction harder. Finally, our bounds generalize beyond MSE to other norm-based similarity metrics, such as LPIPS [66] or CLIP score [43]. Our bounds are not only tighter but also match our intuition. The bounds go up (improved privacy) when the encoder discards informative directions, especially those the prior constrains weakly or that the similarity metric (e.g., LPIPS) is most sensitive to. Below, we summarize our contributions:

![](images/f463c7a939380d831c0a7f864403ccec4f858e0a9f463797fc44d3624aa89815.jpg)  
Figure 1: The idea of instance encoding and its invertibility.

• We improve the bound from Maeng et al. [35] (Corollary 1), yielding a much tighter bound for cases where the encoder’s randomness is small and privacy mainly stems from the encoder’s geometry.

• We extend the bound to hold even when the encoder adds no randomness at all (Corollary 2). This adds practicality to the bound, as many encoders used by practitioners do not incorporate any randomness [8, 31–33, 45, 53, 54, 63].

• We extend the bounds to other similarity metrics (Theorem 2 and Corollary 3), such as LPIPS. This allows us to bound a more semantically meaningful metric than MSE.

• We develop an improved method (compared to that of Maeng et al. [35]) for estimating the dataset prior, which is required for the bound, using well-trained generative models (e.g., DDPM [21]). This is not our core contribution but has practical value to the community.

• We evaluate our new set of bounds on two image datasets (MNIST [7] and CIFAR-10 [29]), two similarity metrics (MSE and LPIPS [66]), and various encoders, showing that the bounds are effective and tight in many cases. We also show that the bound can serve as a reasonable proxy for measuring the invertibility of privacy-enhancing encoders.

## 2 Background: Instance Encoding

Instance encoding refers to a family of techniques that transform a secret input $x \in \mathbb { R } ^ { d }$ into an embedding $e \in \mathbb { R } ^ { m }$ using a privacy-enhancing encoder Enc : $\mathbb { R } ^ { d } \to \mathbb { R } ^ { m }$ , and share e with untrusted parties instead of x (Figure 1). The goal is to design an encoder such that training and inference can still be performed using the embedding e, while the original data x cannot be reconstructed from e. While this idea has been popular in the community [11, 19, 22, 32, 34, 38, 39, 49, 52, 54, 56, 58, 62, 63], most prior work relies on empirical arguments rather than theoretical ones to claim that their encoders are non-invertible, e.g., by testing against a few known attacks.

Types of encoders. Numerous different proposals exist for how to design such a privacy-enhancing encoder. The vast majority of proposals in this area use a deep neural network (DNN) as the encoder [32, 33, 42, 49, 52–54, 59, 62], with variations. Most use a DNN trained with various techniques to (hopefully) resist input reconstruction attacks [53, 54], while others use DNNs with random weights [59,62,63]. Some add noise to the encoder to introduce randomness [11, 20, 27, 35, 38–40, 50, 56], while others use a deterministic DNN [32, 33, 45, 53, 63]. This paper mostly focuses on such DNN-based encoders, although our bounds are applicable to a broader class of encoders (Section 3).

Theoretical arguments for instance encoding. The field was largely devoid of theoretical analysis [35], but a few recent works have made progress. Carlini et al. [3] proved a negative result: instance encoding cannot simultaneously achieve strong data indistinguishability and utility [3]. Thus, one must aim for a weaker guarantee, e.g., non-invertibility, which prohibits near-exact reconstruction of the original data. This is also why simply applying the popular differential privacy (DP) framework to instance encoding [11, 56] and claiming safety is insufficient: DP guarantees data indistinguishability, but any meaningfully strong DP guarantee is incompatible with utility in instance encoding [3].

Maeng et al. [35] were among the first to derive a meansquared error (MSE) bound that limits the adversary’s reconstruction accuracy. Our work mainly compares against this prior work, so we discuss it in more detail in Section 3.1. There is other relevant work that is largely orthogonal to ours. One line of work [13, 14, 24, 37, 46] instead adopts metric DP, guaranteeing that changes within some $\ell _ { p }$ ball of the input cannot be distinguished by looking at the embedding. These approaches have two limitations: (1) in many cases, changes in the $\ell _ { p }$ do not have an interpretable meaning, especially when defined in some latent space [37, 44, 46], and their privacy becomes highly empirical; and (2) they cannot easily be applied to DNN-based encoders, since it requires computing the sensitivity (Lipschitz constant) of the DNN encoder, which is known to be NP-hard [57]. Another line of work [60, 61] proposed a framework called PAC privacy and showed that a bound more general than that of Maeng et al. [35] (and ours) is possible; however, their possibility results are purely theoretical, and their bound has never been evaluated against realistic attacks. We do not directly compare with these alternate directions.

## 3 Bounding the Encoder’s Invertibility Through its Spectral Geometry

Threat model. We assume the problem setup shown in Figure 1. Our goal is to estimate how accurately an attacker can reconstruct the input $x \sim \pi$ by observing $e ,$ through an attack ${ \hat { x } } = \operatorname { A t t } ( e )$ . We assume a strong attacker who knows the encoder Enc and the data prior π, and who can leverage this knowledge arbitrarily to mount the best possible attack. This threat model matches that of Maeng et al. [35] and is much stronger than the adversaries commonly assumed in other work, which assume that the adversary does not know π [17] and/or Enc [33, 58, 59, 62, 63].

## 3.1 Bound from Prior Work (Maeng et al. [35])

Maeng et al. [35] proposed one of the first reconstruction bounds in this setup. Their bound requires the encoder to be differentiable and randomized. To accommodate DNNbased encoders popular among practitioners [8, 31–33, 45, 53, 54, 63]—which are generally neither differentiable nor randomized—they made two changes to commonly used DNNs: (1) non-differentiable operators $( e . g .$ , ReLU or Max-Pool) were replaced with differentiable approximations $( e . g .$ GELU or AvgPool), and (2) Gaussian noise was added to the encoder output. That ${ \mathrm { i s } } ,$ if $\operatorname { E n c } _ { D } ( x )$ is the deterministic encoder DNN (with non-differentiable components replaced), the encoder they studied can be expressed as $\operatorname { E n c } ( x ) =$ Enc $: \boldsymbol { D } ( \boldsymbol { x } ) + \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { m } )$

For this type of encoder, Maeng et al. [35] showed that the mean-squared error (MSE) between the original input x and the reconstruction xˆ obtained by any adversary is lowerbounded in terms of the encoder’s Fisher information matrix (FIM), defined as:

$$
I _ { e } ( x ) = \frac { 1 } { { \sigma } ^ { 2 } } J _ { \mathrm { E n c _ { D } } } ( x ) ^ { \top } J _ { \mathrm { E n c _ { D } } } ( x ) ,\tag{1}
$$

where $J _ { \mathrm { E n c _ { D } } } ( x )$ is the Jacobian of $\mathrm { E n c } _ { \mathrm { D } }$ with respect to x. Their bound, and all of ours, rests on the following regularity conditions.

Assumption 1 The prior π admits a density; thejoint density $p ( e , x ) = \pi ( x ) p ( e \mid x )$ is absolutely continuous in x for a.e. $e ;$ the score $s _ { \pi } ( x ) = \nabla _ { \boldsymbol { x } }$ <sub>x</sub> logπ(x) and $S = \mathbb { E } [ s _ { \pi } ( x ) s _ { \pi } ( x ) ^ { \top } ]$ exist and are finite; and $I _ { e } ( x )$ exists with diag $( I _ { e } ( x ) ) ^ { 1 / 2 }$ locally integrable in x.

While some data may naturally satisfy these assumptions, the distribution of commonly used datasets, $e . g .$ ., natural images, are not absolutely continuous. Maeng et al. [35] therefore worked with smoothed images [5] instead, where a small amount of Gaussian noise was added to each image. Our evaluation also uses such smoothed images. Under Assumption 1, Maeng et al. [35] showed the following.

Theorem 1 (Bound from [35]) Under Assumption 1, suppose further that $p ( e , x )  0$ as $\| x \| $ ∞ for a.e. e, and that E $\lceil \mathrm { T r } ( I _ { e } ( x ) ) \rceil + \mathrm { T r } ( S )$ is finite and nonzero. For a $d -$ dimensional input $x ,$ the following holds for any $\hat { x } = \operatorname { A t t } ( e )$

$$
\mathbb { E } \left[ \Vert \hat { x } - x \Vert _ { 2 } ^ { 2 } / d \right] \geq \frac { d } { \mathbb { E } \left[ \mathrm { T r } ( I _ { e } ( x ) ) \right] + \mathrm { T r } ( S ) } .\tag{2}
$$

Limitations of Maeng et al. [35] The bound from Theorem 1 has critical issues that limit its applicability. First, the bound becomes smaller as less noise σ is added, and becomes trivial $( i . e . , \mathrm { z e r o } )$ when $\sigma = 0$ . This may be tight for encoders that are genuinely easy to invert without much noise, but it is very loose for others; practitioners have shown that many encoders, especially DNN-based ones, are hard to invert even when little or no randomness is added [8,31–33,45,53,54,63]. For these encoders, the bound incorrectly suggests that the encoder is easy to invert (because little or no noise is added), when in fact it is not.

Second, the bound only constrains the MSE of the reconstruction. While MSE between the original and reconstructed inputs has been used as a proxy for reconstruction quality in much prior work [17,18,35], it is known that MSE is not an accurate measure of semantic similarity. Other similarity metrics are considered better proxies for semantic similarity—such as LPIPS [66] or CLIP score [43] for images, and sentence embedding similarity [30] for natural language. We introduce a set of alternative bounds that address these issues.

## 3.2 Bounds with the Spectral Geometry

First, we introduce a new bound that is tighter than the bound of Maeng et al. [35] (Theorem 1) when the noise added to the encoder is small. Unlike Theorem 1, which relies on a scalar summary of the encoder’s spectral geometry (through the term $\mathrm { T r } ( I _ { e } ( x ) ) )$ ), our bound makes better use of the full spectral geometry information. We derive the bound by extending the van Trees inequality, and hence present it as a Corollary. Maeng et al. [35] also derived their bound from the van Trees inequality [16], but we use an alternative matrix form [2]. For completeness, the form we use is restated in Appendix $\mathrm { A . } 2 .$ and the proof is given in Appendix $\mathrm { A . 4 } .$

Corollary 1 (Improved MSE bound) Under Assumption 1, suppose further that $x p ( e , x ) \to 0$ as $\| x \|  { \mathsf { c } }$ ∞for a.e. e, and that $\mathbb { E } \big [ I _ { e } ( x ) \big ] + S$ is non-singular. Then the following holds for any attack ${ \hat { x } } = \operatorname { A t t } ( e )$

$$
\mathbb { E } \big [ \| \hat { x } - x \| _ { 2 } ^ { 2 } / d \big ] \geq \frac { 1 } { d } \operatorname { T r } \Big ( \big ( \mathbb { E } \big [ I _ { e } ( x ) \big ] + S \big ) ^ { - 1 } \Big ) .\tag{3}
$$

Again, using smoothed images (or any other dataset that naturally meets the assumptions) allow us to use the bound.

The bound from Corollary 1 looks similar to that from Theorem 1 but has one crucial difference that makes it tighter for smaller σ: Theorem 1 takes the inverse ofthe trace, while our bound (Corollary 1) takes the trace of the inverse. To highlight this difference, simply let $S = 0$ , and let $\lambda _ { 1 } , \lambda _ { 2 } , \ldots , \lambda _ { d }$ be the eigenvalues of E $\big [ I _ { e } ( x ) \big ]$ . Since the trace of a square matrix equals the sum of its eigenvalues, the right-hand side of Theorem 1 becomes $\frac { d } { \lambda _ { 1 } + \lambda _ { 2 } + \cdots + \lambda _ { d } }$ , whereas the right-hand side of Corollary 1 becomes $\begin{array} { r } { \frac { 1 } { d } \left( \frac { 1 } { \lambda _ { 1 } } + \frac { 1 } { \lambda _ { 2 } } + \cdots + \frac { 1 } { \lambda _ { d } } \right) } \end{array}$ . The two become identical when all eigenvalues are equal $( i . e .$ , when the encoder is isotropic), but otherwise, the latter is always at least as large.

Colloquially, an encoder can be seen as projecting the input into an embedding space, where the eigenvalues of its (expected) FIM describe how strongly information along each direction survives. An isotropic encoder preserves information equally along every direction. An anisotropic encoder, on the other hand, amplifies certain directions while attenuating or discarding others—the latter intuitively enhances privacy better. Our bound is tighter for anisotropic encoders because it explicitly accounts for this directional structure $( e . g .$ , the encoder’s spectral geometry), correctly predicting that some encoders are harder to invert—even without much noise—if they discard enough directional information. Later, in Section 6, we show that our new bound (Corollary 1) is indeed much tighter than that from prior work (Theorem 1).

## 3.3 Bound in the Case of Zero Noise Addition

Our new bound (Corollary 1) still collapses to zero when there is no noise at all $( \sigma = 0 )$ . This makes the bound still inapplicable to many privacy-enhancing encoders that do not add any noise. We therefore extend Corollary 1 to cover such noise-free encoders. For notational simplicity, we define $\begin{array} { r } { G ( x ) = J _ { \mathrm { E n c } _ { \mathrm { D } } } ( x ) ^ { \top } J _ { \mathrm { E n c } _ { \mathrm { D } } } ( x ) \ ( i . e . , \ I _ { e } ( x ) = \frac { 1 } { \mathfrak { c } ^ { 2 } } G ( x ) ) } \end{array}$ ). The proof is given in Appendix A.4.

Corollary 2 (Noise-free MSE bound) Let $r = \operatorname { r a n k } ( \mathbb { E } \lceil G \rceil )$ , and let $U \in \mathbb { R } ^ { d \times ( d - r ) }$ be an orthonormal basis for null(E-G). Then,

$$
\mathbb { E } [ | | \hat { x } - x | | _ { 2 } ^ { 2 } / d ] \geq \operatorname* { l i m } _ { \sigma  0 } \frac { 1 } { d } \operatorname { T r } ( ( \mathbb { E } [ I _ { e } ( x ) ] + S ) ^ { - 1 } )\tag{4}
$$

$$
= \operatorname* { l i m } _ { \sigma  0 } \frac { 1 } { d } \operatorname { T r } ( \big ( \frac { 1 } { \sigma ^ { 2 } } \mathbb { E } \big [ G \big ] + S \big ) ^ { - 1 } )\tag{5}
$$

$$
= { \frac { 1 } { d } } \operatorname { T r } \left( ( U ^ { \top } S U ) ^ { - 1 } \right) .\tag{6}
$$

When E-G is full rank $( r = d ) ,$ , the bound collapses to zero.

Corollary 2 gives a non-trivial bound even when no noise is added to the encoder. Even when the encoder is deterministic, the rank of the expected FIM over the prior limits the adversary’s reconstruction accuracy. Specifically, the attack becomes harder as $\mathbb { E } \big [ G \big ]$ becomes lower-rank $( r < d )$ . This matches our intuition: if the encoder captures information only along a consistent, low-dimensional subspace and discards the rest, the privacy of the input will enhance.

The bound also shows that reconstruction is easier $( i . e .$ the bound is smaller) when the null space contains directions along which S has large eigenvalues. This again matches our intuition: even if the encoder discards directions where S has large eigenvalues $( i . e .$ , directions that the prior already constrains tightly), such directions are easier to guess from the prior. To summarize, our new bounds show that reconstruction is harder when: (1) the encoder discards more informative directions, and (2) the discarded directions are ones that are hard to guess from the prior (small eigenvalues in S).

## 4 Extending to Metrics Beyond MSE

Next, we discuss how the bounds from Section 3 can be extended to other popular metrics that are considered to better capture the semantic similarity between data points. Specifically, we consider norm-based metrics, where the input is projected into some high-dimensional space via a function $\Psi ( x )$ ), and similarity is measured by the (squared) $\ell _ { 2 } { \mathrm { - n o r m } }$ in that space, i.e., $\| \Psi ( x _ { 1 } ) - \Psi ( x _ { 2 } ) \| ^ { 2 }$ . This general form covers a wide range of popular metrics: when comparing two images, popular metrics such as LPIPS [66] and CLIP [43]; and in the data retrieval literature, vector databases [30] often measure relatedness between data points using the $\ell _ { 2 }$ distance between their representations in a high-dimensional embedding space. This section applies broadly to this large family of $\ell _ { 2 }$ -based similarity metrics.

## 4.1 Bounding the ℓ<sub>2</sub>-Norm for Function Ψ

First, we derive a bound for a general function Ψ, when the encoder has randomness. Our bound is an extension of the generalized form of the multivariate van Trees inequality from Gill & Levit [16], restated in Appendix A.3, and proof is in Appendix A.4.

Theorem 2 (Ψ-bound) Let $\Psi : \mathbb { R } ^ { d }  \mathbb { R } ^ { k }$ be a twicedifferentiable function, with $J _ { \Psi } = \partial \Psi / \partial x$ . Let $P _ { 1 }$ and $P _ { 2 }$ be two orthogonal projectors that are orthogonal to each other and $P _ { 1 } + P _ { 2 } = I _ { d }$ (it can be generalized to arbitrary number of projectors, but we keep it two for simplicity). Define for $g \in \{ 1 , 2 \}$

$$
n _ { g } = \mathbb { E } \big [ \big \| J _ { \Psi } P _ { g } \big \| _ { F } ^ { 2 } \big ] , \qquad t _ { g } = \Delta _ { g } \Psi + J _ { \Psi } P _ { g } s _ { \pi } ,
$$

where $\Psi _ { i } : \mathbb { R } ^ { d } $ R is the i-th component $o f \Psi ,$ and the i-th row of $\Delta _ { g } \Psi i s \left( \Delta _ { g } \Psi \right) _ { i } = \mathrm { T r } \left( P _ { g } \nabla ^ { 2 } \Psi _ { i } \right)$ . Next, we define a $2 \times 2$ matrix $D ,$ with thefollowing $( i , j )$ -th elements:

$$
D _ { i j } = \frac { 1 } { \sigma ^ { 2 } } \mathbb { E } \left[ \langle J _ { \mathrm { E n c } _ { \mathrm { D } } } P _ { i } J _ { \Psi } ^ { \top } , J _ { \mathrm { E n c } _ { \mathrm { D } } } P _ { j } J _ { \Psi } ^ { \top } \rangle _ { F } \right] + \mathbb { E } \left[ \langle t _ { i } , t _ { j } \rangle \right] ,
$$

where $\langle \cdot , \cdot \rangle _ { F }$ is a Frobenius inner product. Suppose Assumption 1 holds, that $J _ { \Psi } ( x ) P _ { g } p ( e , x ) \to 0 a s \| x \| \to \infty f o r a . e . \ \epsilon$

and each $^ { g , }$ , that the moments $n _ { g }$ and $D _ { i j }$ arefinite, and that $D \succ 0 .$ . Thenfor any attacker $\hat { x } = \operatorname { A t t } ( e )$

$$
\begin{array} { r } { \mathbb { E } \| \Psi ( \hat { x } ) - \Psi ( x ) \| ^ { 2 } \geq n ^ { \top } D ^ { - 1 } n , \qquad n = [ n _ { 1 } \ n _ { 2 } ] ^ { \top } . } \end{array}\tag{7}
$$

The LPIPS bound may be hard to interpret. To make the interpretation easier, consider the case where $P _ { 1 }$ and $P _ { 2 }$ are chosen to be the projectors onto the range space and null space of E-G, respectively (call them $P _ { \mathrm r }$ and $P _ { \mathrm { n } } .$ , with their respective t and n vectors denoted $t _ { \mathrm { r } } , t _ { \mathrm { n } } , n _ { \mathrm { r } } ,$ and $n _ { \mathrm { { n } } } ;$ note that $\mathbb { E } [ G ]$ is symmetric, and nul $\begin{array} { r } { \iota ( \mathbb { E } \big [ G \big ] ^ { \top } ) = \mathrm { n u l l } ( \mathbb { E } \big [ G \big ] ) ) } \end{array}$ . Then $J _ { \mathrm { E n c } _ { \mathrm { D } } } P _ { \mathrm { r } } = J _ { \mathrm { E n c } _ { \mathrm { D } } }$ and $J _ { \mathrm { E n c } _ { \mathrm { D } } } P _ { \mathrm { n } } = 0$ for π-almost every x, and:

$$
\begin{array} { r } { D = \left( \begin{array} { c c } { \frac { 1 } { \mathfrak { C } ^ { 2 } } \mathbb { E } \left[ \| J _ { \mathrm { E n c } _ { \mathrm { D } } } J _ { \Psi } ^ { \top } \| _ { F } ^ { 2 } \right] + \mathbb { E } \left[ \left. t _ { \mathrm { r } } , t _ { \mathrm { r } } \right. \right] } & { \mathbb { E } \left[ \left. t _ { \mathrm { r } } , t _ { \mathrm { n } } \right. \right] } \\ { \mathbb { E } \left[ \left. t _ { \mathrm { n } } , t _ { \mathrm { r } } \right. \right] } & { \mathbb { E } \left[ \left. t _ { \mathrm { n } } , t _ { \mathrm { n } } \right. \right] } \end{array} \right) . } \end{array}
$$

When the encoder discards exactly the directions that $\Psi$ cares about $( i . e . , J _ { \Psi }$ and $P _ { \mathrm { n } }$ are well aligned), $n _ { \mathrm { n } }$ is large while $n _ { \mathrm { r } }$ is small, and the dominating term in $n ^ { \top } D ^ { - 1 } n$ becomes $n _ { \mathrm { n } } ^ { 2 } / \mathbb { E } \left[ \left. t _ { \mathrm { n } } , t _ { \mathrm { n } } \right. \right]$ , which is finite and does not scale with $1 / \sigma ^ { 2 }$ In other words, the bound stays large no matter how noisy the encoder is—because the encoder’s spectral geometry discards the very information that Ψ needs anyway.

If the encoder instead preserves the directions that Ψ captures $( i . e . , J _ { \Psi }$ and $P _ { \mathrm { r } }$ are well aligned), $n _ { \mathrm { r } }$ is large while $n _ { \mathrm { n } }$ is small. Now the term with $1 / \sigma ^ { 2 }$ dominates. This makes the bound driven primarily by the noise in the encoder, and the bound collapses as the noise approaches zero. Again, this is intuitive—since the encoder preserves the information relevant to $\Psi ,$ a large amount of noise must be added to prevent precise reconstruction under the Ψ-metric.

## 4.2 Extending with Zero Noise

Next, we extend the Ψ-bound to the case where no noise is added to the encoder output (σ = 0), analogous to Corollary 2. The bound is nontrivial only when $\mathbb { E } \big [ G \big ]$ is not full rank.

Corollary 3 (Zero-noise Ψ-bound) From Theorem 2, let $P _ { \mathrm { r } }$ be the projector onto the range space range $\left( \mathbb { E } \left\lceil G \right\rceil \right)$ , and let $P _ { \mathrm { n } }$ be the projector onto the null space null(E-G). Let $t _ { \mathrm { r } } , t _ { \mathrm { n } } , n _ { \mathrm { r } }$ and $n _ { \mathrm { n } }$ be the respective t and n vectorsfor $P _ { \mathrm r }$ and $P _ { \mathrm { n \cdot } }$ . Then, as $\sigma  0$

$$
\begin{array} { r } { n ^ { \top } D ^ { - 1 } n \longrightarrow \frac { n _ { \mathrm { n } } ^ { 2 } } { \mathbb { E } \left[ \left. t _ { \mathrm { n } } \right. ^ { 2 } \right] } = \frac { \left( \mathbb { E } \left[ \left. J _ { \Psi } P _ { \mathrm { n } } \right. _ { F } ^ { 2 } \right] \right) ^ { 2 } } { \mathbb { E } \left[ \left. \Delta _ { \mathrm { n } } \Psi + J _ { \Psi } P _ { \mathrm { n } } s _ { \pi } \right. ^ { 2 } \right] } . } \end{array}
$$

Again, the numerator $n _ { \mathrm { n } } = \mathbb { E } \left[ \Vert J _ { \Psi } P _ { \mathrm { n } } \Vert _ { F } ^ { 2 } \right]$ measures how sen sitive Ψ is to the directions the encoder discards. A large numerator means the encoder is discarding the directions that Ψ cares, and hence privacy improves (bound goes up). If the encoder instead preserves the directions that Ψ cares, the numerator shrinks and the bound goes down. When $\mathbb { E } \big [ G \big ]$ is full rank $( r = d )$ , the bound becomes zero. The lesson from this bound is similar to that of Theorem 2: a privacy-enhancing encoder with respect to Ψ should (1) have low rank $( r \ll d )$ $i . e .$ , discard as many informative directions as possible, and, especially, (2) discard the directions that Ψ cares about.

## 5 Practical Considerations

In this section, we discuss several practical considerations for calculating and applying the bound in real-world use cases.

## 5.1 Choosing $P _ { 1 }$ and $P _ { 2 }$

In Theorem 2, any choice of orthogonal projectors $P _ { 1 }$ and $P _ { 2 }$ (or even more than two projectors) gives a valid bound, though different choices yield different tightness. Throughout the rest of the paper, we choose $P _ { 1 } = P _ { \mathrm { r } }$ and $P _ { 2 } = P _ { \mathrm { n } } .$ —the projectors onto the range space and null space of $\mathbb { E } [ G ]$ , respectively. This choice is easy to interpret and connects well to Corollary 3, but it may not yield an optimal bound. As our evaluation will show, our LPIPS bound is not always tight, and a tighter bound may be achievable by choosing a different set of projectors (likely more than two). We leave such improvements to future work.

## 5.2 Estimating S

To use the new bounds, one must calculate ${ \boldsymbol { S } } =$ $\mathbb { E } _ { \pi } [ s _ { \pi } ( x ) s _ { \pi } ( x ) ^ { \top } ]$ , where $s _ { \pi } ( x ) = \nabla _ { x } \log \pi ( x )$ is the score of each input x. For image datasets, Maeng et al. [35] used a technique called sliced score matching [47] to train a model called NICE [9], which predicts $s _ { \pi } ( x )$ from x. Instead, we show that a well-trained, off-the-shelf diffusion model (in particular, DDPM [21]) can be used to predict $s _ { \pi } ( x )$ . We show in Section 6.4.1 that our method provides a better estimate than a trained NICE [9] model; this is an additional contribution of our work.

DDPM starts from a maximally noisy image $x _ { T }$ and denoises it back to $x _ { T - 1 } , x _ { T - 2 } , . . . , x _ { 0 }$ , where $x _ { t }$ is the noisy image at step t and x is the original, noise-free image. A key component of DDPM is training a model $ { \varepsilon } _ { \boldsymbol { \Theta } } ( x _ { t } , t )$ , which predicts the noise present in the image at step t. Song et al. [48] showed that there is a relationship between $ { \varepsilon } _ { \boldsymbol { \Theta } } ( x _ { t } , t )$ and $s _ { \pi } ( x )$ specifically, $s _ { \pi _ { t } } ( x _ { t } ) = - \mathfrak { E } _ { \boldsymbol { \theta } } ( x _ { t } , t ) / \sqrt { 1 - \bar { \mathfrak { a } } _ { t } }$ , where $\bar { \bf { q } } _ { t }$ is a coefficient determined by the noise schedule (provided with the pretrained model), and $\pi _ { t }$ is the distribution of $x _ { t }$

Recall that we operate on a smoothed (i.e., slightly noisy) image $x = x _ { 0 } + \mathcal { N } ( 0 , \sigma _ { s } ^ { 2 } I _ { d } )$ , for some small $\pmb { \sigma } _ { s }$ . In DDPM, $x _ { t } = \sqrt { \bar { \alpha } _ { t } } \big ( x _ { 0 } + \sqrt { ( 1 - \bar { \alpha } _ { t } ) / \bar { \alpha } _ { t } } z \big )$ , where $z \sim \mathcal { N } ( 0 , I _ { d } )$ . If we choose t such that $\sqrt { ( 1 - \bar { \alpha } _ { t } ) / \bar { \alpha } _ { t } } = \sigma _ { s }$ , then $x _ { t } = \sqrt { \bar { \alpha } _ { t } } x ,$ , and

$$
s _ { \pi } ( x ) = \sqrt { \bar { \alpha } _ { t } } s _ { \pi _ { t } } ( x _ { t } ) = - \sqrt { \bar { \alpha } _ { t } / ( 1 - \bar { \alpha } _ { t } ) } \varepsilon _ { \Theta } \left( \sqrt { \bar { \alpha } _ { t } } x , t \right) .
$$

This allows us to calculate the score function $s _ { \pi } ( x )$ using a trained DDPM.

Throughout our evaluation, we download publicly available, well-trained DDPMs (for MNIST [12] and CIFAR-10 [55]) and use them to estimate $s _ { \pi } ( x )$ . Then, $S = \mathbb { E } [ s _ { \pi } ( x ) s _ { \pi } ( x ) ^ { \top } ]$ is calculated by averaging over all images in the training set. Note that DDPM assumes pixel values in $[ - 1 , 1 ] ,$ , so a scaling factor of 2 must be applied when using pixel values in [0,1].

## 5.3 Estimating E-G

Calculating the bounds requires (1) estimating $\mathbb { E } [ G ]$ , and (2) separating its range space and null space. Correctly estimating $\mathbb { E } [ G ]$ is required for all four bounds. Correctly separating its null space and range space is needed for the noiseless bounds (Corollary 2 and Corollary 3).

Estimating $\mathbb { E } [ G ]$ . Estimating $\mathbb { E } [ G ]$ over π requires calculating $J _ { \mathrm { E n c _ { D } } } ( x )$ and averaging over $x \sim \pi .$ For efficiency, we sample k images from the training data and average over the sampled images. Ideally, using the entire training set gives the most accurate estimate of the bound, but we empirically find that a smaller k suffices in most cases. The effect of the choice of k is further evaluated in Section 6.4.2.

Splitting range and null space. The bounds from Corollary 2 and Corollary 3 (the bounds for $\sigma = 0 )$ require splitting the range space and null space of E-G. While one can obtain the exact null space from $\mathbb { E } [ G ]$ once it is estimated, using this exact null space often gives a correct but conservative bound, because any subspace with a negligibly small but still nonzero eigenvalue is treated as part of the range space. Alternatively, we explore setting a threshold τ and treating subspaces whose eigenvalues are smaller than $\tau \cdot \lambda _ { \mathrm { m a x } }$ as null, where $\lambda _ { \operatorname* { m a x } }$ is the largest eigenvalue of $\mathbb { E } \big [ G \big ]$ . We empirically find that this gives a tighter bound. The effect of different choices of τ is evaluated in Section 6.4.3.

Calculating $J _ { \mathrm { E n c } _ { \mathrm { D } } } ( x ) . J _ { \mathrm { E n c } _ { \mathrm { D } } } ( x )$ is an $m \times d$ matrix. Since privacy-enhancing encoders usually have a small output dimension m, the entire $J _ { \mathrm { E n c _ { D } } } ( x )$ can typically be materialized at once in modern GPU memory. When this is not possible, we instead calculate it column-by-column using a Jacobianvector product (jvp).

## 5.4 Estimating Other Terms in Theorem 2

Estimating $n _ { g } = \mathbb { E } \left[ \Vert J _ { \Psi } P _ { g } \Vert _ { F } ^ { 2 } \right] , \mathbb { E } \left[ \langle t _ { i } , t _ { j } \rangle \right] , \mathbb { E } \left[ \Vert J _ { \mathrm { E n c _ { D } } } J _ { \Psi } ^ { \top } \Vert _ { F } ^ { 2 } \right]$ and Tr $\left( P _ { g } \nabla ^ { 2 } \Psi _ { i } \right)$ —the remaining terms of Theorem $2 { \mathrm { - } } { \mathrm { i s } }$ even more expensive, because $\Psi \boldsymbol { \mathrm { s } }$ output dimension is large for LPIPS. Thus, we estimate these quantities through Rademacher probes [23]. Throughout we use 16 probes per input, averaged over 8 inputs. We confirmed that increasing the number of probes beyond this provides diminishing return. Admittedly, averaging over more inputs may improve the bound further, but our LPIPS bounds are always valid with this approximated estimation.

## 5.5 Bound as a Proxy for Invertibility

As we will show in our evaluation, when the MSE or LPIPS bound is small, it is a strong indicator that the encoder’s privacy-enhancing property is weak, $i . e .$ , that the original input can be reconstructed easily. However, the absolute value of the bound becomes less informative as it grows and starts to saturate. For example, for the MNIST dataset, an attacker can still achieve a reasonably low MSE or LPIPS by drawing white, squiggly, “number-like” lines on a black background, even when the attack fails to reconstruct any meaningful information. This is a fundamental issue with norm-based similarity metrics—the meaning of their absolute value heavily depends on context. This makes using the absolute number of the bound ill-suited as a measure for privacy.

However, we empirically observe that the ratio between the bound and its ceiling—the bound that would be achieved if the encoder conveyed no information at all—is still a reasonable indicator of the encoder’s privacy-enhancing properties. In particular, we find that the ratio computed from the MSE bound works better, because this bound is tighter and less flat in the hard-to-reconstruct regions (as we later show in Section 6). From the observation, we define the following metric and call it the ratio to ceiling, or RTC:

$$
{ \mathrm { R T C } } : = \log { \frac { ( { \mathrm { c e i l i n g ~ o f ~ t h e ~ M S E ~ b o u n d } } ) } { ( { \mathrm { a c t u a l ~ M S E ~ b o u n d } } ) } } ,
$$

where log is the natural logarithm (base e). The idea behind RTC follows popular privacy definitions that measure privacy as the (log of the) ratio between the adversary’s belief before and after a privacy-leaking observation [15, 28, 64]. Essentially, RTC measures how much the bound has shifted—i.e., how much easier reconstruction became—after the adversary observes the encoder’s output, compared to guessing based on the prior π alone. A lower RTC indicates better privacy, and $\mathbf { R } \mathbf { T } \mathbf { C } = 0$ implies that the bound collapses to the prior-only bound (the ceiling). We show in Section 6.5 that RTC serves as a reasonable empirical metric for privacy.

To summarize, our bounds serve two practical purposes: (1) when the absolute value of the bounded metric has direct interpretation (e.g., bounding the MSE of weight or height of a patient), the bound can directly limit them from being reconstructed; and (2) even when the bounded metric is only loosely indicative of the reconstruction quality $( e . g .$ , as in MSE for natural images), the ratio between the bound and the ceiling of the bound (i.e., RTC) serves as a theoreticallyprincipled indicator for reconstruction hardness.

## 6 Evaluation

Our evaluation aims to answer the following questions regarding our new bounds (Corollaries 1, 2, and 3, and Theorem 2):

• Do they work across attacks, encoders, and datasets?

• Are they tighter than the bound of Maeng et al. [35]?

![](images/b9884302ccce55c5a285522e7c49ad08e462085159865c17a0c08b966135c230.jpg)  
Embedding dim (m)  
(a) Gaussian, σ = 10<sup>−3</sup>

![](images/31767e7dc6e6b5f53dd9c9e8fc137b0614409aed4b0f50abcca0ed007be7321d.jpg)  
Embedding dim (m)

![](images/bfd0daaac0ac67dc18937fcf28ffe5cfa6cef8c5646f9b61787e8f8d3cbc10fd.jpg)  
Embedding dim (m)  
(d) Gaussian, σ = 0  
(b) MNIST, σ = 10<sup>−3</sup>

![](images/ce866c2bba750602b596a6f27f706bdc9d91a4d0c7fec333aad83d7a823b5b5b.jpg)  
Embedding dim (m)

![](images/b4b1dd5daa0243df0cf46c250fab96f3a95e54834718a2996db061269899a9f5.jpg)  
Embedding dim (m)  
(e) MNIST, σ = 0

attack DNN attack  
(c) CIFAR-10, σ = 10<sup>−3</sup>  
![](images/16a0f6aed05ef9f217377560f53716fcc7edc0d57e5d0e6a3b3c5a42fe858b4a.jpg)  
Embedding dim (m)  
(f) CIFAR-10, σ = 0  
Figure 2: Our MSE bounds (solid blue line) closely lower-bound the attackers’ achievable MSE for all the attackers. Bound from Maeng et al. [35] (dotted line) becomes very loose when $r < d$ and is always zero when $\sigma = 0$ (no encoder randomness).

• How do the bounds change with the heuristics from Section 5?

• Is RTC (Section 5.5) indicative of reconstruction hardness?

## 6.1 Evaluation Setup and Methodologies

Datasets. We evaluate our bounds on three datasets: (1) isotropic Gaussian noise (matching the dimensionality of MNIST for MSE and CIFAR-10 for LPIPS bound), sampled from $\mathcal { N } ( 0 . 5 , 0 . 2 ^ { 2 } I _ { d } )$ ; (2) MNIST [7]; and (3) CIFAR-10 [29]. For MNIST and CIFAR-10, we add smoothing noise of $\mathcal { N } ( 0 , 0 . 1 ^ { 2 } I _ { d } )$ , following Maeng et al. [35]. For the Gaussian dataset, S is calculated analytically (inverse of the covariance matrix). Others use DDPM to calculate S (Section 6.4.1).

Encoders. We test encoders with various architectures and weights. The first are linear encoders $( \operatorname { E n c } ( x ) = W x +$ $\mathcal { N } ( 0 , \sigma ^ { 2 } I _ { d } ) , W \in \mathbb { R } ^ { m \times d } )$ with varying output dimension m and $\mathfrak { O } \in \{ 0 , 1 0 ^ { - 3 } \}$ (small noise or no noise). The second are DNN encoders, consisting of the first few layers of popular CNNs, followed by an additional compression block to control the output dimension m. The compression block consists of a compressing convolutional layer, followed by GELU and an optional AvgPool layer. For MNIST, we use a simple 3-block CNN with three split points (early: after block 1; middle: after block 2; late: after block 3). For CIFAR-10, we use ResNet-18 [41] with three split points (early: after the first convolutional layer; middle: after ResNet block 4; late: after ResNet block 6). To make the encoder differentiable, we replace MaxPool with AvgPool and ReLU with GELU. The encoder is followed by a downstream model that consumes the embedding and performs the downstream task; this model consists of a decompression block followed by the remainder of the split DNN. Details are in Appendix A.1.1.

We study three different weights: (1) random weights, frozen after initialization; (2) weights trained jointly with the downstream model on the target task; and, for linear encoders only, (3) a weight matrix W whose eigenvalues are analytically controlled to be $\begin{array} { r } { \lambda _ { i } = d \cdot i ^ { - \alpha } / \sum _ { j } j ^ { - \alpha } , } \end{array}$ , with a varying α to control the skewness. This is achieved via $W = U \mathrm { d i a g } ( \sqrt { \lambda _ { i } } ) V ^ { \top }$ with random unitary matrices U and V.

Attacks. For reconstructing the Gaussian inputs x ∼ $\mathcal { N } ( \mu , \boldsymbol { \eta } ^ { 2 } I _ { d } )$ from the linear encoder $e = W x + \mathscr { N } ( 0 , \sigma ^ { 2 } I _ { m } )$ we can analytically arrive to a minimum mean-squared error (MMSE) estimator, which is the posterior mean. For $\sigma > 0$

$$
\hat { x } = \mathbb { E } \big [ x | e \big ] = \mu \mathbf { 1 } + \big ( \frac { W ^ { \top } W } { \sigma ^ { 2 } } + \frac { I _ { d } } { \eta ^ { 2 } } \big ) ^ { - 1 } \frac { W ^ { \top } ( e - W \mu \mathbf { 1 } ) } { \sigma ^ { 2 } } .\tag{8}
$$

For σ = 0,

$$
\hat { x } = \mathbb { E } \big [ x | e \big ] = \mu \mathbf { 1 } + W ^ { + } ( e - W \mu \mathbf { 1 } ) ,\tag{9}
$$

where $W ^ { + }$ is a Moore–Penrose pseudoinverse of W (proofs in Appendix A.4). We call this MMSE attack. MMSE attacks are only defined for Gaussian inputs and linear encoders.

For natural image datasets, we test two attacks. The first is an optimization-based attack (Opt. attack), which minimizes a MAP-style objective with a tunable prior weight β:

$$
{ \hat { x } } ( e ) = \underset { x ^ { * } } { \operatorname { a r g m i n } } \ \| e - \mathrm { E n c } _ { D } ( x ^ { * } ) \| _ { 2 } ^ { 2 } - \beta \log \pi ( x ^ { * } ) .
$$

Embedding dim (m)  
![](images/b89832a21f1a9407458986ddee442ee29be850cf5071c0a0a85dd7e910e56f41.jpg)

(a) Gaussian, σ = 10<sup>−3</sup>  
![](images/c1368c0bb5943e804ab23f4e95ff2c53eda134cf6d2a415932a1b510b2233141.jpg)

![](images/95d9d1e17739d085bb5a28f28e4ed18cf15f47de866e272ed807dc03ed3b5bc6.jpg)  
(d) Gaussian, σ = 0  
(b) MNIST, σ = 10<sup>−3</sup>

![](images/ee4cf93e3d2a8da78e602823eaa12f8300eb00be969f1c1cdf60df7bf228e9ec.jpg)  
(c) CIFAR-10, σ = 10<sup>−3</sup>

![](images/525fdbb6e75da353f3b2bb5eabf5d0db7c2aaa756052175f021db1b314a4ae03.jpg)  
Embedding dim (m)  
(e) MNIST, σ = 0

![](images/e33612e6b582771e40223d72423bd8d317cd7ae95a93cd74ead58ff81c7c9615.jpg)  
Embedding dim (m)  
(f) CIFAR-10, σ = 0  
Figure 3: Our LPIPS bounds (solid blue line) are always met, and tight near $m \sim d .$ For $m \ll d ,$ the bound is unfortunately looser and near-flat, being less informative than the MSE bounds. Still, this is the first working LPIPS bound to our knowledge.

For $\sigma > 0 ,$ setting $\beta = 2 \sigma ^ { 2 }$ recovers the exact maximum a posteriori (MAP) estimate. For $\sigma = 0 , \beta$ is instead treated as a tunable hyperparameter. For the data prior $\pi ( x )$ , we use the learned score from the pre-trained DDPM (Section 5.2). This generalizes existing optimization-based attacks that use heuristically designed proxies for the data prior [36, 51]. The second is a DNN-based attack (DNN attack), which uses a trained DNN as the attacker, following the recipe of [32]. Details of the attacks are in Appendix A.1.2.

Additional design parameters. We calculate S and $\mathbb { E } \big [ G \big ]$ exactly whenever possible. When we need to approximate, we use the following parameters unless stated otherwise: when calculating $\mathbb { E } [ G ]$ with subsampled images, we use $k = 2 5 6$ images; when approximating small-signal subspaces as the null space, we treat directions whose eigenvalues are smaller than 0.0001 times the largest eigenvalue $( \tau = 1 0 ^ { - 4 } )$ as null. The effect of these choices is evaluated in Section 6.4. Finally, for LPIPS, we use the pretrained VGG-16 backbone with ReLU/MaxPool replaced by GELU/AvgPool (for twicedifferentiability). We confirme that this does not significantly affect the calculated LPIPS score. For MNIST, we replicate the channel dimension, since LPIPS assumes three channels.

## 6.2 Bounds with Linear Encoders

We first evaluate our new bounds on linear encoders. For linear encoders, the FIM is not data-dependent, so $\mathbb { E } \big [ G \big ]$ can be calculated exactly, and $r = \operatorname { r a n k } ( \mathbb { E } \left[ { \tilde { G } } \right] )$ is simply $r = \operatorname* { m i n } ( d , m )$ provided the weights do not introduce additional null directions. These two properties eliminate the need for many of the heuristics in Section 5.3, isolating the effectiveness of the bound from potential errors introduced by the heuristics.

## 6.2.1 Controlling the Rank

First, we evaluate the bound with a random W, which makes W isotropic, while controlling the rank.

MSE bound. Figure 2 plots our MSE bounds (blue solid line) from Corollary 1 (when noise with $\sigma = 1 0 ^ { - 3 }$ is added; Figure 2(a–c)) and Corollary 2 (when no noise is added; Figure 2(d–f)), along with the MSE actually achieved by the attacks. For setups with noise $\sigma = 1 0 ^ { - 3 }$ (top row; a–c), we also plot the bound from Maeng et al. [35] (Theorem 1; dotted line). For setups with no noise (bottom row; d–f), the bound from Maeng et al. [35] becomes zero.

The Gaussian MMSE attack exactly lands on top of our bounds. For natural images, Opt. attack performs better when $m > d$ , while DNN attack performs better when $m < d$ (we only show the Opt. attack in the $m > d ~ \mathrm { r e g i o n } )$ . Still, our bound is valid for both attacks throughout, and is reasonably tight in most cases. On the other hand, the bound from Maeng et al. [35] (dotted line) is tight only when $d \ll m$ and $\sigma >$ 0. Again, this is because our bounds better account for the information lost through the encoder’s spectral geometry.

We can also see that our bounds for noisy embeddings (Figure 2(a–c)) increase sharply around $m = r = d .$ This is expected, since information starts to be lost through the encoder’s spectral geometry once we cross the $m = r = d$ line.

(b) MNIST  
(a) Gaussian  
![](images/6e33aaefe0f99f4c52bfeb9f4315a57ebfb1526e91457daeb407a8d8bf3db622.jpg)

(c) CIFAR-10  
![](images/dde49ee3809c4207ea8689e4aef3382ebc30e83274c837b7ad2bd4bf31589d5a.jpg)  
(d) Gaussian (e) MNIST (f) CIFAR-10  
Figure 4: MSE and LPIPS bound with different skewness α.

Similarly, Figure 2(d–f) shows that the bound becomes nontrivial as this line is crossed, which is exactly when the encoder starts to lose information even without any added noise.

LPIPS bound. Figure 3 shows the same plot using LPIPS as the similarity metric instead of MSE, based on Theorem 2 and Corollary 3. The bounds are again always valid, but looser and nearly flat in the $m < d$ region, making them less informative than the MSE bound. One potential reason for this, as explained in Section 5.1, is that we simply choose the two projectors in Theorem 2 to be the projectors onto the range and null spaces, whereas a different (and larger) set of projectors could provide a tighter bound. Again, we leave further improvement of the bound to future work.

## 6.2.2 Controlling the Skewness of E-G

Next, we keep $r = \operatorname { r a n k } ( \mathbb { E } [ G ] )$ constant at $m = r = d$ and vary the distribution of the eigenvalues using a controlled parameter $\alpha ,$ as discussed in Section 6.1. We only show the case with small noise $( \sigma = 1 0 ^ { - 3 } )$ ), since E-G is full rank, and Corollary 2 would give a trivial bound. Again, the trend is similar: Opt. attack performs better for easier cases (lower α), while DNN attack performs better for harder cases (higher α). Still, our bounds hold for both attacks.

Figure 4(a–c) shows that our bounds hold for all the attacks. The bound remains tight for Gaussian data, but becomes looser on natural images with large α (but still tighter than the bound from Maeng et al. [35], which is the flat dotted line). When the encoder is isotropic $( { \bf { \alpha } } { \bf { \alpha } } { \bf { \alpha } } = 0 )$ , our bound coincides with that of Maeng et al. [35]. However, as the spectrum becomes more skewed $( { \bf { \alpha } } { \bf { \alpha } } ) > 0 )$ , our bound starts to increase, correctly capturing the fact that information loss along certain directions makes reconstruction harder. The bound from Maeng et al. [35], on the other hand, stays flat. Figure 4(d–f) again shows a similar trend: our bounds always hold against both attackers, but they are much looser than the MSE bound.

![](images/d1f5877c05fd3632be4a4c8a2fc4a50fe23fbf4a84eca0407df2c954c9f6f61a.jpg)

![](images/3727088ada99f253e0b0f0dc1f9719c1989c972749878aead17638f8d56b7ec6.jpg)  
(a) MNIST, $\sigma = 1 0 ^ { - 3 }$ (b) ${ \mathsf { C l F A R - 1 0 } } .$ $\sigma = 1 0 ^ { - 3 }$

![](images/ba008e0f4a8e620cc0921b9a312952375cb6e629b58b93b232b8b1ae13c64ea1.jpg)

![](images/832a2527251ffc8f0b423c26b4c2fecd2ad2dd9628e3ccec59e2718081684cd3.jpg)  
(c) MNIST, σ = 0  
(d) CIFAR-10, σ = 0  
Figure 5: MSE bound vs. achieved MSE by the attacker.

## 6.3 Bounds with DNN-based Encoders

Next, we show results for general DNN-based encoders from Section 6.1. With DNN-based encoders, we do not have a straightforward way to control the rank or the skewness of E-G. We therefore simply plot the achieved attack MSE and LPIPS against the bounds for various encoder architectures (linear and DNN with early/middle/late splits) and weights (random, skew-controlled via α (only possible for linear encoders), and fully trained). Here, we only plot the MSE and LPIPS from the Opt. attack, except for the linear encoders, whose DNN attackers were already trained from the previous section. This is because training DNN attackers for each encoder and takes too much time.

MSE bound. Figures 5(a–b) plot the bound from Corollary 1 against the attack MSE. Across various encoder architectures and their weights, the bound always holds—no marker falls into the red-shaded region. This shows that our bound (Corollary 1) holds reliably across various encoder architectures and training methods for their weights.

In general, bounds for linear or split-early encoders, especially with random or skewed weights, appear tighter. Meanwhile, bounds for split-middle and split-late encoders, especially with trained weights, appear looser, though still valid. This matches empirical observations from prior work showing that reconstruction becomes harder as data passes through a deeper network [33, 35]. It may be that our bound still does not precisely capture this additional hardness due to encoder depth. Alternatively, it may simply be that our attack methods are suboptimal for deeper encoders.

![](images/19932bb44cb2d0ad6bfa0cc9911794268f2dc0c940660d7938d0ab6688a342ce.jpg)

![](images/4fd884a8bf26e0b9f9f64905ebf5ca267f9c79ced5566659862d46e11a3b7ecc.jpg)  
(a) MNIST, $\sigma = 1 0 ^ { - 3 }$ (b) CIFAR-10, $\sigma = 1 0 ^ { - 3 }$

![](images/9488d77beace7ca7e43b9da36b97f36b4d0168963ce5c40f52e721cfb853ed96.jpg)  
(c) MNIST, σ = 0

![](images/b1a63c6c5b4b83d434f7f43c93513bd72fe7bd0d90ea5a6586674e6e46f153fa.jpg)  
(d) CIFAR-10, σ = 0  
Figure 6: LPIPS bound vs. achieved LPIPS by the attacker.

Figures 5(c–d) show the bound from Corollary 2, with no randomness in the encoder. Again, the bound always holds, albeit often loose. Here, weights with skewness controlled by α (green markers) are not shown, because this bound is controlled only by the null space of E-G, and the skewnesscontrolled linear encoders are still full rank. In this sense, the zero-noise bound (Corollary 2) is inherently looser than the bound for noisy encoders (Corollary 1)—Corollary 2, unlike Corollary 1, cannot capture the information loss due to the skewed spectral geometry, unless it introduces a null space.

LPIPS bound. Figures 6(a–b) show the LPIPS bound from our Theorem 2 versus the actual LPIPS achieved by running the attack. Figures 6(c–d) show the LPIPS bound when no noise is added to the encoder output (Corollary 3). The bounds in Figure 6 always hold, but are generally looser than the MSE bound (Figure 5); this may again be improved by selecting a better set of projectors in Theorem 2 or with a better attack.

![](images/5ad1eecccfdc00f8021b8d3d3d4166ab519456bfe9b6fea53821f7e48949c193.jpg)  
Figure 7: S quality check: all the eigenvalues $\mu _ { i }$ of $\Sigma ^ { 1 / 2 } S \Sigma ^ { 1 / 2 }$ must be equal to or larger than 1.

## 6.4 Sensitivity Study

In this section, we evaluate the heuristics from Section 5.

## 6.4.1 Using Different Score Models

We introduced a new practical method for calculating $S =$ $\mathbb { E } _ { \pi } [ s _ { \pi } ( x ) s _ { \pi } ( x ) ^ { \top } ]$ using a well-trained DDPM (Section 5.2). Previously, Maeng et al. [35] used a technique called sliced score matching [47] to train a NICE model [9] to calculate the same matrix. Here, we compare the two approaches, showing that our new method is more reliable. However, we do not claim that our method is fundamentally superior; rather, we believe the better estimation results are because open-source DDPMs have been well-trained over time by the community, and have therefore internalized better score models than one could obtain by training a score model from scratch.

We measure the quality of S as follows. Since S is the Fisher information of $x \sim \pi .$ , the matrix form of the Cramér-Rao inequality [6] gives $S \succeq \Sigma ^ { - 1 }$ , where Σ is the covariance of x ∼ π and $\succeq$ indicates the Loewner order. Multiplying both sides by $\Sigma ^ { 1 / 2 }$ gives $\Sigma ^ { 1 / 2 } S \Sigma ^ { 1 / 2 } \succeq I _ { d }$ . Since $\Sigma ^ { 1 / 2 } S \bar { \Sigma } ^ { 1 \bar { / } 2 }$ is symmetric positive semi-definite, $\Sigma ^ { 1 / 2 } S \Sigma ^ { 1 / 2 } \succeq I _ { d }$ holds if and only if all of its eigenvalues $\mu _ { i }$ satisfy $\mu _ { i } \geq 1$ for all i. This is exactly what we check.

We calculate Σ from the training images and run this test on S obtained from both our method and the method of prior work [35]. Figure 7 shows the results. While the eigenvalues (µ ) from our approach (blue line) also sometimes fall below 1, the violation is much rarer and milder than that of prior approach (red line). This shows that our DDPM-based method more closely estimates the true S.

## 6.4.2 Effect of Sample Count k in Estimating E-G

The most accurate way to estimate $\mathbb { E } [ G ]$ is to calculate $G =$ $J _ { \mathrm { E n c } _ { \mathrm { D } } } ( x ) ^ { \top } J _ { \mathrm { E n c } _ { \mathrm { D } } } ( x )$ for each x and average across all training samples, but this requires heavy compute, especially for deep encoders. Figure 8 shows, for the CIFAR-10 dataset, how the four bounds (Theorem 2 and Corollaries 1, 2, and 3) change if we use only k samples randomly selected from the training set instead. The figure shows that the bound (i.e., the distribution of the markers) does not change significantly beyond $k = 2 5 6 ;$ thus, we use $k = 2 5 6$ throughout our evaluation.

![](images/d629fdbb5405a2e737421f7b509d23e273e76b6e7add0666dfed9a7aa96cffda.jpg)  
Figure 8: Bound vs. attack MSE/LPIPS with while varying the number of data k used to estimate E-G. Result from the CIFAR-10 dataset; MNIST dataset shows similar trends and is omitted for space.

## 6.4.3 Effect of Varying the Null Space Threshold τ

The bounds from Theorem 2 and Corollaries 2 and 3 require splitting the domain into the range space and null space of E-G. For Theorem 2, this split does not need to be accurate— any orthogonal split leads to a valid bound (although its tight ness will vary), and using the range and null spaces was simply our choice for simplicity (Section 4.1). However, the bounds for encoders with no randomness (Corollaries 2 and 3) require the split to be accurate, since these bounds rely critically on the null space. If part of the range space is incorrectly classified as null, the bound can be overestimated (the reverse is fine, albeit will lead to a looser bound).

Thus, the safest approach is to consider only directions whose eigenvalues are exactly 0 as the null space. However, this may make the bound loose, because our estimate of E-G cannot be exact for real-world data—since it relies on a sampled training dataset—and some null directions’ eigenvalues may be estimated as small but nonzero due to estimation noise. Hence, we treat eigenvalues smaller than $\tau \cdot \lambda _ { \mathrm { m a x } }$ as zero, where $\lambda _ { \operatorname* { m a x } }$ is the largest eigenvalue (Section 6.4.3).

Figure 9 plots the LPIPS bounds for various datasets, with and without noise, to evaluate the effect of choosing different τ. Again, the bounds with noise (Theorem 2; the first two columns) are always correct, and τ only controls the tightness. For these bounds in the first two columns, $\tau = 1 0 ^ { - 4 }$ and $\tau =$ $1 0 ^ { - 2 }$ appear to be the tightest, but the differences across all four rows are not very substantial— $- i . e .$ , the choice of τ does not significantly influence the bounds, unless it is too large.

The bounds for zero-noise encoders (Corollaries 2 and 3) can become incorrect and be violated if we choose a τ that is too large. The figure shows that these bounds are influenced more strongly by the choice of τ, and are indeed violated at $\tau = 1 0 ^ { - 2 }$ . On the other hand, τ = 0 is always valid, but appears very loose. Across the four columns, $\tau = 1 0 ^ { - 4 }$ appears to be a reasonable choice, giving the tightest bounds that remain correct. Thus, we use $\tau = 1 0 ^ { - 4 }$ throughout our evaluation.

## 6.5 Visualization of the Reconstruction and Using RTC as a Metric

Figures 10 and 11 visualize some reconstructed images from Figures 2(b–c) (when the encoder has small randomness) and Figures 2(e–f) (when the encoder has no randomness). We also show the encoder’s ratio-to-ceiling (RTC) metric values, an empirical metric we proposed in Section 5.5.

![](images/91b623842c2d47288971bbfe9eb8534c74c388684d771ef32fa9d489a43d7f76.jpg)  
Figure 9: LPIPS bounds with and without noise, under various threshold τ that splits between the range space and the null space (Section 6.4.3). Empirically, $\tau = 1 0 ^ { - 4 }$ seems to be a reasonable choice, where the bounds appear the tightest and never violated.

Smaller embedding dimension m—which leads to both increased MSE and LPIPS bounds (Figures 2 and 3)—also leads to perceptually worse reconstruction. More interestingly, reconstruction hardness is well captured by the RTC metric: smaller RTC correspond to visually worse reconstruction, and reconstructions with RTC < 0.01 fail to recover much meaningful information. These results highlight the potential of RTC as a proxy for invertibility.

The evaluation also shows that MNIST requires a smaller RTC to be strongly protected: even at RTC=0.005, some columns are reasonably reconstructed. This is because the MNIST dataset lies on a much lower-dimensional manifold, which makes reconstruction possible with only very limited information (discussed further in Section 6.5.1). This implies that the RTC threshold considered safe depends on the dataset and must be empirically determined by the community. This situation parallels popular privacy definitions like differential privacy (DP)—although DP offers stronger worst-case guarantees, what DP parameters (e.g., ε and δ) are considered safe in practice is still being determined empirically [25], and there is no single number that is universally safe.

## 6.5.1 Why is MNIST Hard to Perfectly Protect?

Figures 10 and 11 show that MNIST is sometimes reasonably reconstructed even at a very low RTC (e.g., 0.005). For example, digits 7 and 0 in Figure 10, and digits 7 and 1 in Figure 11, are meaningfully reconstructed even at RTC= 0.005. This is in contrast to CIFAR-10, whose reconstructed images are much less informative even at a much higher RTC of 0.009.

Our additional study revealed why—MNIST can be reconstructed to a reasonable level much more easily, even with very little information. We ran a simple experiment to demonstrate this. We grouped the images in the MNIST training set by their labels and averaged all images sharing the same label, producing a class-mean image for each label. These class-mean images are what an attacker could “reconstruct” by knowing only the label of each image—a mere 3.3 bits $( \log _ { 2 } ( 1 0 ) )$ of information that must survive through the encoder anyway for meaningful downstream classification.

![](images/c83f01bb25fda6103c4063db9ccd65b5ad28eb3b11debf6c95d6c94761896767.jpg)  
Figure 10: Visualization of the reconstructed images from Figures 2(b–c) (σ = 10<sup>−3</sup>), along with each encoder’s RTC.

Figure 12 shows the reference images from Figures 10 and 11, along with their respective class-mean images. The results show that the class-mean image is a surprisingly close perceptual reconstruction of the original image. This explains why instance encoding cannot work well for the MNIST dataset. The minimal 3.3 bits of label information that the encoder must preserve are sufficient to reconstruct the image, espe cially for digits like 7, 1, and 0, which were also reasonably well reconstructed in Figures 10 and 11. This was not the case for CIFAR-10: for example, individual cat images vary enough that the class-mean cat image does not resemble a high-quality reconstruction. In other words, it is impossible to fully prevent reconstruction (beyond the level shown in Figure 12) while still achieving high downstream classification accuracy. If any paper claims such a result, it likely indicates that the attack used is not strong enough.

## 7 Conclusion

We developed a set of bounds that lower-bound the adversary’s ability to reconstruct the original data from the embedding, in the setting of instance encoding. We show that our bounds are tighter than that from prior work [35] in many cases, and are useful in cases where the prior bounds do not apply (e.g., when the encoder has no randomness, or when a norm-based metric other than MSE is considered). We also introduce a new, practical way to estimate the data prior more accurately, using a pre-trained DDPM. Our bounds can serve as an indicative metric for an encoder’s invertibility.

![](images/e9bad07549d80ffa43c12fb960e6d8ce59e282c3bfc68b5a3346336fa1eac75d.jpg)

Figure 11: Visualization of the reconstructed images from Figures 2(e–f) (σ = 0), along with each encoder’s RTC.  
![](images/a48de0d4f2898fed70997d6de0455e65e597487ad956769fffcea21c60dbb16c.jpg)  
Figure 12: Same reference images from Figures 10 and 11, compared with their class-mean images.

## Acknowledgments

We thank Chuan Guo for insightful discussions and pointers on developing the LPIPS bound. This work was partly supported by the U.S. National Science Foundation under award No. CNS-2349610 and CCF-2529883. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation. The authors used AI for code, text, and figures, but all outputs were manually reviewed by the authors to ensure correctness.

## References

[1] Yasmeena Akhter, Richa Singh, and Mayank Vatsa. AIbased radiodiagnosis using chest x-rays: A review. Fron-

tiers Big Data, 2023. URL: https://doi.org/10.3 389/fdata.2023.1120989, doi:10.3389/FDATA.20 23.1120989.

[2] Florent Bouchard, Alexandre Renaux, Guillaume Ginolhac, and Arnaud Breloy. Intrinsic bayesian cramér-rao bound with an application to covariance matrix estimation. IEEE Transactions on Information Theory, 2024. URL: https://univ-grenoble-alpes.hal.scien ce/hal-04770231/document.

[3] Nicholas Carlini, Samuel Deng, Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Abhradeep Thakurta, and Florian Tramèr. Is private learning possible with instance encoding? In IEEE Symposium on Security and Privacy (SP), 2021. URL: h t t p s : //ieeexplore.ieee.org/document/9519450.

[4] Nicholas Carlini, Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, and Florian Tramer. Neuracrypt is not private. arXiv preprint arXiv:2108.07256, 2021. URL: https://arxiv.org/ abs/2108.07256.

[5] Jeremy M. Cohen, Elan Rosenfeld, and J. Zico Kolter. Certified adversarial robustness via randomized smoothing. In Proceedings of the International Conference on Machine Learning (ICML), 2019. URL: http: //proceedings.mlr.press/v97/cohen19c.html.

[6] Amir Dembo, Thomas M Cover, and Joy A Thomas. Information theoretic inequalities. IEEE Transactions on Information theory, 1991. URL: https://ieeexp lore.ieee.org/document/104312.

[7] Li Deng. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 2012. URL: https://ieeexplore.ieee. org/document/6296535.

[8] Shiwei Ding, Lan Zhang, Miao Pan, and Xiaoyong Yuan. Patrol: Privacy-oriented pruning for collaborative inference against model inversion attacks. In 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024. URL: https: //ieeexplore.ieee.org/document/10483973.

[9] Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014. URL: https://arxi v.org/abs/1410.8516.

[10] Xin Dong, Barbara De Salvo, Meng Li, Chiao Liu, Zhongnan Qu, HT Kung, and Ziyun Li. Splitnets: Designing neural architectures for efficient distributed computing on head-mounted systems. In Proceedings ofthe

Conference on Computer Vision and Pattern Recognition (CVPR), 2022. URL: https://ieeexplore.iee e.org/document/9879512.

[11] Minxin Du, Xiang Yue, Sherman SM Chow, Tianhao Wang, Chenyu Huang, and Huan Sun. Dp-forward: Fine-tuning and inference on language models with differential privacy in forward pass. In Proceedings of the ACM SIGSAC Conference on Computer and Communications Security (CCS), 2023. URL: https: //dl.acm.org/doi/10.1145/3576915.3616592.

[12] Laurent Fainsin. 1aurent/ddpm-mnist. https://hugg ingface.co/1aurent/ddpm-mnist, 2023.

[13] Liyue Fan. Image pixelization with differential privacy. In Data and Applications Security and Privacy (DBSec), Lecture Notes in Computer Science, 2018. doi:10.1 007/978-3-319-95729-6\_10.

[14] Liyue Fan. Differential privacy for image publication. In Theory and Practice ofDifferential Privacy (TPDP) Workshop, 2019. URL: https://webpages.charlot te.edu/lfan4/pdf/TPDP2019.pdf.

[15] Arpita Ghosh and Robert Kleinberg. Inferential privacy guarantees for differentially private mechanisms. arXiv preprint arXiv:1603.01508, 2016. URL: https://ar xiv.org/abs/1603.01508.

[16] Richard D Gill and Boris Y Levit. Applications of the van trees inequality: a bayesian cramér-rao bound. 1995. URL: https://projecteuclid.org/journals/b ernoulli/volume-1/issue-1-2/Applications-o f-the-van-Trees-inequality--a-Bayesian-Cra m%C3%A9r/bj/1186078362.full.

[17] Chuan Guo, Brian Karrer, Kamalika Chaudhuri, and Laurens Van der Maaten. Bounding training data reconstruction in private (deep) learning. In Proceedings of the International Conference on Machine Learning (ICML), 2022. URL: https://proceedings.mlr.pr ess/v162/guo22c.html.

[18] Awni Hannun, Chuan Guo, and Laurens van der Maaten. Measuring data leakage in machine-learning models with fisher information. In Conference on Uncertainty in Artificial Intelligence (UAI), 2021. URL: https: //arxiv.org/abs/2102.11673.

[19] Qijian He, Wei Yang, Bingren Chen, Yangyang Geng, and Liusheng Huang. Transnet: Training privacypreserving neural network over transformed layer. Proceedings of the VLDB Endowment, 2020. URL: https: //dl.acm.org/doi/abs/10.14778/3407790.3407 794.

[20] Zecheng He, Tianwei Zhang, and Ruby B Lee. Attacking and protecting data privacy in edge–cloud collaborative inference systems. IEEE Internet ofThings Journal, 2020. URL: https://ieeexplore.ieee.org/docu ment/9187880.

[21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), 2020. URL: https://arxiv.org/abs/2006.11239.

[22] Yangsibo Huang, Zhao Song, Kai Li, and Sanjeev Arora. Instahide: Instance-hiding schemes for private distributed learning. In Proceedings of the International Conference on Machine Learning (ICML), 2020. URL: http://proceedings.mlr.press/v119/huang20i. html.

[23] Michael F Hutchinson. A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines. Communications in Statistics-Simulation and Computation, 1989. URL: https://www.tandfonlin e.com/doi/abs/10.1080/03610919008812866.

[24] Jacob Imola, Shiva Prasad Kasiviswanathan, Stephen White, Abhinav Aggarwal, and Nathanael Teissier. Balancing utility and scalability in metric differential privacy. In Conference on Uncertainty in Artificial Intelligence (UAI), 2022. URL: https://proceedings.ml r.press/v180/imola22a.html.

[25] Bargav Jayaraman and David Evans. Evaluating differentially private machine learning in practice. In USENIX Security Symposium, 2019. URL: https://www.usen ix.org/system/files/sec19-jayaraman.pdf.

[26] Yiping Kang, Johann Hauswald, Cao Gao, Austin Rovinski, Trevor Mudge, Jason Mars, and Lingjia Tang. Neurosurgeon: Collaborative intelligence between the cloud and mobile edge. ACM SIGARCH Computer Architecture News, 2017. URL: https://dl.acm.org/doi/1 0.1145/3037697.3037698.

[27] Sanjay Kariyappa, Ousmane Dia, and Moinuddin K Qureshi. Enabling inference privacy with adaptive noise injection. arXiv preprint arXiv:2104.02261, 2021. URL: https://arxiv.org/abs/2104.02261.

[28] Daniel Kifer and Ashwin Machanavajjhala. Pufferfish: A framework for mathematical privacy definitions. ACM Transactions on Database Systems (TODS), 2014. URL: https://dl.acm.org/doi/10.1145/2514689.

[29] Alex Krizhevsky, Geoffrey Hinton, et al. Learning mul tiple layers of features from tiny images. 2009. URL: https://cave.cs.toronto.edu/kriz/learnin g-features-2009-TR.pdf.

[30] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledgeintensive nlp tasks. In Advances in Neural Information Processing Systems (NeurIPS), 2020. URL: https: //arxiv.org/abs/2005.11401.

[31] Ang Li, Jiayi Guo, Huanrui Yang, Flora D Salim, and Yiran Chen. Deepobfuscator: Obfuscating intermediate representations with privacy-preserving adversarial learning on smartphones. In Proceedings of the International Conference on Internet-of-Things Design and Implementation, 2021. URL: https://dl.acm.org/d oi/10.1145/3450268.3453519.

[32] Jingtao Li, Adnan Siraj Rakin, Xing Chen, Zhezhi He, Deliang Fan, and Chaitali Chakrabarti. Ressfl: A resistance transfer framework for defending model inversion attack in split federated learning. In Proceedings ofthe Conference on Computer Vision and Pattern Recognition (CVPR), 2022. doi:10.1109/CVPR52688.2022 .00995.

[33] Meng Li, Liangzhen Lai, Naveen Suda, Vikas Chandra, and David Z Pan. Privynet: A flexible framework for privacy-preserving deep neural network training. arXiv preprint arXiv:1709.06161, 2017. URL: https://ar xiv.org/abs/1709.06161.

[34] Zhijian Liu, Zhanghao Wu, Chuang Gan, Ligeng Zhu, and Song Han. Datamix: Efficient privacy-preserving edge-cloud inference. In European Conference on Computer Vision, 2020. URL: https://www.ecva.net/p apers/eccv\_2020/papers\_ECCV/papers/1235605 62.pdf.

[35] Kiwan Maeng, Chuan Guo, Sanjay Kariyappa, and G Edward Suh. Bounding the invertibility of privacypreserving instance encoding using fisher information. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL: http://papers.nips.cc/p aper\_files/paper/2023/hash/a344f7f474958cc 0775be7e46bc94309-Abstract-Conference.html.

[36] Aravindh Mahendran and Andrea Vedaldi. Understanding deep image representations by inverting them. In Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR), 2015. URL: https://doi.org/10.1109/CVPR.2015.7299155.

[37] Peihua Mai, Ran Yan, Zhe Huang, Youjia Yang, and Yan Pang. Split-and-denoise: Protect large language model inference with local differential privacy. In Proceedings ofthe International Conference on Machine Learning (ICML), 2024. URL: https://dl.acm.org/doi/10.

5555/3692070.3693465, doi:10.5555/3692070.36 93465.

[38] Fatemehsadat Mireshghallah, Mohammadkazem Taram, Ali Jalali, Ahmed Taha Taha Elthakeb, Dean Tullsen, and Hadi Esmaeilzadeh. Not all features are equal: Discovering essential features for preserving prediction privacy. In Proceedings of the International Conference on World Wide Web (WWW), 2021. URL: https://do i.org/10.1145/3442381.3449965.

[39] Fatemehsadat Mireshghallah, Mohammadkazem Taram, Prakash Ramrakhyani, Ali Jalali, Dean Tullsen, and Hadi Esmaeilzadeh. Shredder: Learning noise distributions to protect inference privacy. In Proceedings of the International Conference on Architectural Support for Programming Languages and Operation Systems (ASPLOS), 2020. URL: https://doi.org/10.1145/ 3373376.3378522.

[40] Seyed Ali Osia, Ali Taheri, Ali Shahin Shamsabadi, Kleomenis Katevas, Hamed Haddadi, and Hamid R Rabiee. Deep private-feature extraction. IEEE Transactions on Knowledge and Data Engineering, 2018. URL: https://ieeexplore.ieee.org/document/85150 92.

[41] Huy Phan. Pytorch models trained on cifar-10 dataset. https://github.com/huyvnphan/PyTorch\_CIFAR 10, 2013.

[42] Maarten G Poirot, Praneeth Vepakomma, Ken Chang, Jayashree Kalpathy-Cramer, Rajiv Gupta, and Ramesh Raskar. Split learning for collaborative deep learning in healthcare. arXiv preprint arXiv:1912.12115, 2019. URL: https://doi.org/10.48550/arXiv.1912.12 115.

[43] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings ofthe International Conference on Machine Learning (ICML), 2021. URL: https://dblp.org/rec/conf/icml/RadfordKHRG ASAM21.

[44] Muhammad Usama Saleem, Dominick Reilly, and Liyue Fan. Dp-shield: Face obfuscation with differential privacy. 2022. URL: https://par.nsf.gov/servlets /purl/10332300.

[45] Abhishek Singh, Ayush Chopra, Ethan Garza, Emily Zhang, Praneeth Vepakomma, Vivek Sharma, and Ramesh Raskar. Disco: Dynamic and invariant sensitive channel obfuscation for deep neural networks.

In Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR), 2021. URL: https://api.semanticscholar.org/CorpusID: 229340265.

[46] Abhishek Singh, Praneeth Vepakomma, Vivek Sharma, and Ramesh Raskar. Posthoc privacy guarantees for collaborative inference with modified propose-test-release. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL: https://dl.acm.org/doi/1 0.5555/3666122.3667272.

[47] Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon. Sliced score matching: A scalable approach to density and score estimation. In Conference on Uncertainty in Artificial Intelligence (UAI), 2019. URL: http://proc eedings.mlr.press/v115/song20a.html.

[48] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. In Proceedings of the International Conference on Learning Representations (ICLR), 2020. URL: https://arxiv.org/pdf/2011.13456.

[49] Chandra Thapa, Mahawaga Arachchige Pathum Chamikara, Seyit Camtepe, and Lichao Sun. Splitfed: When federated learning meets split learning. In Proceedings of the AAAI Conference on Artificial Intelligence, 2022. URL: https://ojs.aaai.org/i ndex.php/AAAI/article/view/20825.

[50] Tom Titcombe, Adam J Hall, Pavlos Papadopoulos, and Daniele Romanini. Practical defences against model inversion attacks for split neural networks. arXiv preprint arXiv:2104.05743, 2021. URL: https://doi.org/10 .48550/arXiv.2104.05743.

[51] Dmitry Ulyanov, Andrea Vedaldi, and Victor S. Lempitsky. Deep image prior. In Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR), 2018. URL: http://openaccess .thecvf.com/content\_cvpr\_2018/html/Ulyan ov\_Deep\_Image\_Prior\_CVPR\_2018\_paper.html, doi:10.1109/CVPR.2018.00984.

[52] Praneeth Vepakomma, Otkrist Gupta, Tristan Swedish, and Ramesh Raskar. Split learning for health: Distributed deep learning without sharing raw patient data. arXiv preprint arXiv:1812.00564, 2018. URL: https: //doi.org/10.48550/arXiv.1812.00564.

[53] Praneeth Vepakomma, Abhishek Singh, Otkrist Gupta, and Ramesh Raskar. Nopeek: Information leakage reduction to share activations in distributed deep learning. In 2020 International Conference on Data Mining Workshops (ICDMW), 2020. URL: https://api.semant icscholar.org/CorpusID:221246011.

[54] Praneeth Vepakomma, Abhishek Singh, Emily Zhang, Otkrist Gupta, and Ramesh Raskar. Nopeek-infer: Preventing face reconstruction attacks in distributed inference after on-premise training. In IEEE International Conference on Automatic Face and Gesture Recognition (FG), 2021. URL: https://ieeexplore.ieee.org/ document/9667085.

[55] Patrick von Platen. google/ddpm-cifar10-32, 2022. URL: https://huggingface.co/google/ddpm-cif ar10-32.

[56] Ji Wang, Jianguo Zhang, Weidong Bao, Xiaomin Zhu, Bokai Cao, and Philip S Yu. Not just privacy: Improving performance of private deep learning in mobile cloud. In Proceedings of the ACM International Conference on Knowledge Discovery & Data Mining (KDD), 2018. URL: https://doi.org/10.1145/3219819.322010 6.

[57] Lily Weng, Huan Zhang, Hongge Chen, Zhao Song, Cho-Jui Hsieh, Luca Daniel, Duane Boning, and Inderjit Dhillon. Towards fast computation of certified robustness for relu networks. In Proceedings of the International Conference on Machine Learning (ICML), 2018. URL: https://proceedings.mlr.press/v80/weng 18a.html.

[58] Liyao Xiang, Hao Zhang, Haotian Ma, Yifan Zhang, Jie Ren, and Quanshi Zhang. Interpretable complex-valued neural networks for privacy protection. In Proceedings of the International Conference on Learning Representations (ICLR), 2020. URL: https://openreview.n et/forum?id=S1xFl64tDr.

[59] Hanshen Xiao and Srinivas Devadas. Dauntless: Data augmentation and uniform transformation for learning with scalability and security. IACR Cryptol. ePrint Arch., 2021. URL: https://eprint.iacr.org/2021/201.

[60] Hanshen Xiao and Srinivas Devadas. Pac privacy: Automatic privacy measurement and control of data processing. In Annual International Cryptology Conference, 2023. URL: https://doi.org/10.1007/978-3-031 -38545-2\_20.

[61] Hanshen Xiao, G Edward Suh, and Srinivas Devadas. Formal privacy proof of data encoding: The possibility and impossibility of learnable encryption. In Proceedings of the ACM SIGSAC Conference on Computer and Communications Security (CCS), 2024. URL: https://doi.org/10.1145/3658644.3670277.

[62] Adam Yala, Homa Esfahanizadeh, Rafael G. L. D Oliveira, Ken R. Duffy, Manya Ghobadi, Tommi S. Jaakkola, Vinod Vaikuntanathan, Regina Barzilay, and Muriel Medard. Neuracrypt: Hiding private health

data via random neural networks for public training. arXiv preprint arXiv:2106.02484, 2021. URL: https: //arxiv.org/abs/2106.02484.

[63] Adam Yala, Victor Quach, Homa Esfahanizadeh, Rafael GL D’Oliveira, Ken R Duffy, Muriel Médard, Tommi S Jaakkola, and Regina Barzilay. Syfer: Neural obfuscation for private data release. arXiv preprint arXiv:2201.12406, 2022. URL: https://doi.org/10 .48550/arXiv.2201.12406.

[64] Bin Yang, Issei Sato, and Hiroshi Nakagawa. Bayesian differential privacy on correlated data. In Proceedings ofthe ACM SIGMOD international conference on Management ofData, 2015. URL: https://doi.org/10 .1145/2723372.2747643.

[65] Qiang Yang, Yang Liu, Tianjian Chen, and Yongxin Tong. Federated machine learning: Concept and applications. ACM Transactions on Intelligent Systems and Technology (TIST), 2019. URL: https://doi.or g/10.1145/3298981.

[66] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR), 2018. URL: https: //doi.ieeecomputersociety.org/10.1109/CV PR.2018.00068.

## A Appendix

## A.1 Evaluation Details

## A.1.1 Encoder Architecture

For DNN encoders, we used the following backbone models. We split the backbone models at different split points and added a compression block, as explained in Section 6.1. Backbone CNN model for MNIST consist of the following three blocks:

• Block 0: Conv2d(out\_dim=32, in\_dim=1, k=5, stride=2, pad=2) + GELU

• Block 1: Conv2d(out\_dim=64, in\_dim=32, k=3, stride=2, pad=1) + GELU + AvgPool2d(2)

• Block 2: Conv2d(out\_dim=128, in\_dim=64, k=3, stride=1, pad=1) + GELU

• Linear(out\_dim=10, in\_dim=128×7×7)

Backbone model for CIFAR-10 used the ResNet-18 model from [41], with MaxPool2d replaced with AvgPool2d and ReLU replaced with GELU.

The compression block controls the output of the encoder m by first changing the channel dimension from $C _ { i n }$ to $C _ { o u t }$ If this is insufficient (still too large even with $C _ { o u t } = 1 )$ to meet m, it reduces the H and W dimensions of the embedding through an additional AdaptiveAvgPool2d layer. Concretely, to achieve $\begin{array} { r } { m = C _ { o u t } \times s , } \end{array}$

• Compress: Conv2d(out\_dim=C<sub>out</sub>, in $\scriptstyle { \mathrm { d i m } } = C _ { i n } , \ \ker 3 ) \ +$ GELU + AdaptiveAvgPool2d(s)

• Decompress: Upsample(H, W) + Conv2d(out\_dim=C<sub>in</sub>, in $\scriptstyle \mathbf { \underline { { d i m } } } = C _ { o u t } , \mathbf { k } = 3 ) + \mathbf { G E L U }$

## A.1.2 Attack Details

For the Gaussian inputs, each markers are averaged from 64 inputs. For $\sigma > 0$ , each input was encoded 8 times and tested. For $\sigma = 0$ , each input was encoded and tested once, as everything is fully deterministic. For MNST and CIFAR-10, we tested 16 inputs, with each encoded and tested once, and averaged for each marker.

For Opt. attack, we use an Adam optimizer. For $\sigma > 0$ case, we set $\ B { = } 2 \sigma ^ { 2 } \mathrm { ( M A P }$ estimator). For $\sigma = 0$ , we sweep $\beta \in \{ 0 , 2 e ^ { - \dot { 6 } } , 2 e ^ { - 4 } , 2 e ^ { - 3 } , 2 e ^ { - 2 } , 2 e ^ { - 1 } \}$ . We initialize xˆ with every pixel being 0.5. We run the optimizer for 500 steps with learning rate 0.05, and CosineAnnealingLR scheduler with $T _ { \mathrm { m a x } } = 5 0 0$ and $\eta _ { \mathrm { m i n } } = 0$ . After each step, the image is clamped to $\hat { x } \in [ - 0 . 5 , 1 . 5 ]$ to refrain from diverging too far from [0, 1].

For DNN attack, we use the following architecture from [32], modified to cover arbitrary m:

• First layer: Linear(out\_dim=1024, in\_dim=m) + LeakyReLU(0.2) + Linear(out\_dim=128×h<sup>2</sup>, in\_dim=1024) + Reshape to (128, h, h)

• Body: 3×[Conv2d(out\_dim=128, in\_dim=128, k=3, stride=1, pad=1) + LeakyReLU(0.2)]

• Final layer: ConvTranspose2d(out\_dim=128, in\_dim=128, k=3, stride=2, pad=1, out\_pad=1) + LeakyReLU(0.2) + ConvTranspose2d(out\_dim=C, in\_dim=128, k=3, stride=2, pad=1, out\_pad=1)

where C is the channel dimension of the image dataset. We used $h = 7$ for MNIST and $h = 9$ for CIFAR-10. For training, we used the entire training set with the following hyperparameters: batch size 256, 40 epochs, Adam with OneCycleLR (max\_lr=2e-3, pct\_start=0.3), plain MSELoss, and clip\_grad\_norm(0.1). We scaled down the embedding as the scale varies with m, by the embeddings’ root-mean-square (RMS) value from a sampled 512 embeddings from the training set.

## A.2 Van Trees Inequality (in the form from Bouchard et al. [2])

Below, we restate the matrix van Trees inequality from Bouchard et al. [2] (with notational changes) used to derive our Corollary 1. Note that this form is not original to Bouchard et al. [2] and similar forms were presented in other work as well, but we use the notation from Bouchard et al. [2] as it most closely resemble our need.

Lemma 1 (Matrix van Trees inequality [2]) Let $\hat { \boldsymbol { \theta } }$ be an estimator ofθ, and y be a noisy observation. Let

$$
I _ { y } ( \boldsymbol { \Theta } ) = \mathbb { E } _ { y | \boldsymbol { \Theta } } \big [ \nabla _ { \boldsymbol { \Theta } } \log p ( y | \boldsymbol { \Theta } ) \nabla _ { \boldsymbol { \Theta } } \log p ( y | \boldsymbol { \Theta } ) ^ { \top } \big ] ,
$$

and $S = \mathbb { E } _ { \pmb { \theta } } \left[ \nabla _ { \pmb { \theta } } \log \pi ( \pmb { \theta } ) \nabla _ { \pmb { \theta } } \log \pi ( \pmb { \theta } ) ^ { \top } \right]$ , where $\pi ( \Theta )$ is the probability densityfunction ofθ. $H p ( y , \theta )$ is absolutely continuous with respect to θ a.e. y, $i f { \theta } p ( y ,       \theta )  0$ as $\| { \boldsymbol { \Theta } } \| \to \infty ,$ and if $\mathbb { E } _ { \boldsymbol { \Theta } } \big [ I _ { \boldsymbol { y } } ( \boldsymbol { \Theta } ) \big ] + \boldsymbol { S }$ exists and is non-singular, For a d-dimensional θ, the following holds for any ${ \hat { \mathsf { \theta } } } .$

$$
\mathbb { E } \left[ | | \hat { \boldsymbol { \Theta } } - \boldsymbol { \Theta } | | _ { 2 } ^ { 2 } / d \right] \geq \frac { 1 } { d } \operatorname { T r } \left( \left( \mathbb { E } \left[ I _ { y } ( \boldsymbol { \Theta } ) \right] + S \right) ^ { - 1 } \right) .\tag{10}
$$

## A.3 Generalized Multivariate van Trees Inequality (in the form from Gill & Levit [16])

Below, we restate the generalized multivariate van Trees Inequality in the form presented in Gill & Levit [16], which we use to derive Theorem 2 and Corollary 3. Specifically, below is their Theorem 1 with notational changes and simplifications (we set $B = I _ { k }$ and $n = 1 )$ to fit our purpose.

Lemma 2 (Generalized van Trees with Ψ [16]) Let $\hat { \boldsymbol { \theta } }$ be an estimator of θ, let y be a noisy observation, and let $\Psi : \mathbb { R } ^ { d }  \mathbb { R } ^ { k }$ be differentiable with $J _ { \Psi } = \partial \Psi / \partial \Theta \in \mathbb { R } ^ { k \times d }$ . Suppose $p ( y \mid { \boldsymbol { \Theta } } )$ and $\pi ( \Theta )$ are absolutely continuous in θ, and that $I _ { \mathrm { y } } ( \boldsymbol { \Theta } )$ exists with diag( $I _ { y } ( \Theta ) ) ^ { 1 / 2 }$ locally integrable in θ. Let $s _ { \pi } ( \Theta ) = \nabla _ { \Theta } \log \pi ( \Theta )$ be the scorefunction. Let $C : \mathbb { R } ^ { d }  \mathbb { R } ^ { k \times }$ d be a differentiable matrix weightfield with rows $c _ { i } ( { \boldsymbol { \theta } } ) ^ { \top }$ , with all moments below finite, and suppose $C ( \boldsymbol { \Theta } ) p ( \boldsymbol { y } , \boldsymbol { \Theta } )  0$ as $\| { \boldsymbol { \Theta } } \| \to \infty .$ Then,

$$
\begin{array} { r l } & { \mathbb { E } \big [ \| \Psi ( \hat { { \boldsymbol { \Theta } } } ) - \Psi ( { \boldsymbol { \Theta } } ) \| ^ { 2 } \big ] } \\ & { \qquad \quad \geq \frac { \big ( \mathbb { E } \big [ \operatorname { T r } \big ( J _ { \Psi } ( { \boldsymbol { \Theta } } ) C ( { \boldsymbol { \Theta } } ) ^ { \top } \big ) \big ] \Big ) ^ { 2 } } { \mathbb { E } \big [ \operatorname { T r } \big ( C ( { \boldsymbol { \Theta } } ) I _ { y } ( { \boldsymbol { \Theta } } ) C ( { \boldsymbol { \Theta } } ) ^ { \top } \big ) \big ] + \mathbb { E } \big [ \| \operatorname { d i v } C + C s _ { \pi } \| ^ { 2 } \big ] } , } \end{array}
$$

where $( \mathrm { d i v } C ) _ { i } = \boldsymbol { \nabla } \cdot \boldsymbol { c } _ { i }$

## A.4 Proofs

## A.5 Proofs of the Bounds

Corollary 1 (Improved MSE bound) Under Assumption 1, suppose further that x p(e,x) → 0 as ∥x∥ → ∞ for a.e. e, and that $\mathbb { E } \lceil I _ { e } ( x ) \rceil + S$ is non-singular. Then the following holds for any attack ${ \hat { x } } = \operatorname { A t t } ( e )$

$$
\mathbb { E } \big [ \| \hat { x } - x \| _ { 2 } ^ { 2 } / d \big ] \geq \frac { 1 } { d } \operatorname { T r } \Big ( \big ( \mathbb { E } \big [ I _ { e } ( x ) \big ] + S \big ) ^ { - 1 } \Big ) .\tag{3}
$$

Proof 1 (Proof of Corollary 1) Replacing θ with x and y with efrom Lemma 1 immediately gives the result. Note that for our randomized DNN encoder,

$$
\mathbb { E } _ { e | x } \big [ \nabla _ { x } \log p ( e | x ) \nabla _ { \theta } \log p ( e | x ) ^ { \top } \big ] = \frac { 1 } { \sigma ^ { 2 } } J _ { \mathrm { E n c } _ { \mathrm { D } } } ( x ) ^ { \top } J _ { \mathrm { E n c } _ { \mathrm { D } } } ( x ) ,
$$

which makes the two definitions $I _ { y } ( \Theta )$ from Lemma 1 and $I _ { e } ( x )$ from Equation 1 to collapse to the same matrix under notational changes (hence, we use the same notation I for both).

Corollary 2 (Noise-free MSE bound) Let $r = \operatorname { r a n k } ( \mathbb { E } \left[ G \right] )$ and let $U \in \mathbb { R } ^ { d \times ( d - r ) }$ be an orthonormal basisfor nul $\left\lfloor \left[ \mathbb { E } \left[ G \right] \right) \right.$ . Then,

$$
\mathbb { E } [ | | \hat { x } - x | | _ { 2 } ^ { 2 } / d ] \geq \operatorname* { l i m } _ { \sigma  0 } \frac { 1 } { d } \operatorname { T r } ( ( \mathbb { E } [ I _ { e } ( x ) ] + S ) ^ { - 1 } )\tag{4}
$$

$$
= \operatorname* { l i m } _ { \sigma \to 0 } { \frac { 1 } { d } } \operatorname { T r } \left( { \bigl ( } { \frac { 1 } { \sigma ^ { 2 } } } \mathbb { E } { \bigl [ } G { \bigr ] } + S { \bigr ) } ^ { - 1 } \right)\tag{5}
$$

$$
= { \frac { 1 } { d } } \operatorname { T r } \left( ( U ^ { \top } S U ) ^ { - 1 } \right) .\tag{6}
$$

When $\mathbb { E } \left[ G \right]$ is full rank $( r = d )$ , the bound collapses to zero.

Proof 2 (Proof of Corollary 2) Consider an orthonormal basis $\begin{array} { r l } { [ R } & { { } U ] _ { \ast } } \end{array}$ , where R is an orthonormal basis for range $\left( \mathbb { E } [ G ] \right)$ ), and U is an orthonormal basis for null(E-G). In this basis,

$$
\begin{array} { r l } & { \left[ R \ U \right] ^ { \top } \left( \frac { 1 } { \mathfrak { O } ^ { 2 } } \mathbb { E } \big [ G \big ] + S \right) \left[ R \ U \right] } \\ & { \quad = \left( \frac { 1 } { \mathfrak { O } ^ { 2 } } A \quad 0 \right) + \left( S _ { 1 1 } \quad S _ { 1 2 } \right) = \left( \frac { 1 } { \mathfrak { O } ^ { 2 } } A + S _ { 1 1 } \quad S _ { 1 2 } \right) , } \\ &  \quad = \left( \begin{array} { c c } { 0 } & { 0 \right) + \left( S _ { 2 1 } \quad S _ { 2 2 } \right) = \left( \begin{array} { c c } { S _ { 2 1 } } & { S _ { 2 2 } } \end{array} \right) , } \end{array} \end{array}
$$

where the top-left block A is $r \times r$ and positive definite (invertible), and $\begin{array} { r } { S _ { 1 1 } = R ^ { \top } S R , S _ { 1 2 } = R ^ { \top } S U , S _ { 2 1 } = U ^ { \top } S R , } \end{array}$ , and $S _ { 2 2 } = U ^ { \top } S U$ . By the block-inverse formula, the inverse is again a block matrix with:

$$
\begin{array} { c } { { \bullet ( { \cal I } , { \cal I } ) \ b l o c { k } { : \ } ( \frac { 1 } { \sigma ^ { 2 } } A + S _ { 1 1 } ) ^ { - 1 } + ( \frac { 1 } { \sigma ^ { 2 } } A + S _ { 1 1 } ) ^ { - 1 } S _ { 1 2 } ( S _ { 2 2 } - } } \\ { { S _ { 2 1 } ( \frac { 1 } { \sigma ^ { 2 } } A + S _ { 1 1 } ) ^ { - 1 } S _ { 1 2 } ) ^ { - 1 } S _ { 2 1 } ( \frac { 1 } { \sigma ^ { 2 } } A + S _ { 1 1 } ) ^ { - 1 } } } \end{array}
$$

$$
\begin{array} { r l } { \bullet ~ ( I , 2 ) ~ b l o c k { : } ~ - ( \frac { 1 } { \sigma ^ { 2 } } A + S _ { 1 1 } ) ^ { - 1 } S _ { 1 2 } ( S _ { 2 2 } - S _ { 2 1 } ( \frac { 1 } { \sigma ^ { 2 } } A + } & { { } } \\ { S _ { 1 1 } ) ^ { - 1 } S _ { 1 2 } ) ^ { - 1 } } \end{array}
$$

$$
\begin{array} { c } { { \bullet ( 2 , l ) \ b l o c { k : } \ - ( S _ { 2 2 } - S _ { 2 1 } ( \frac { 1 } { \mathfrak { c } ^ { 2 } } A + S _ { 1 1 } ) ^ { - 1 } S _ { 1 2 } ) ^ { - 1 } S _ { 2 1 } ( \frac { 1 } { \mathfrak { c } ^ { 2 } } A + } } \\ { { S _ { 1 1 } ) ^ { - 1 } } } \end{array}
$$

$$
\bullet \ ( 2 , 2 ) b l o c k \colon ( S _ { 2 2 } - S _ { 2 1 } \big ( \frac { 1 } { \sigma ^ { 2 } } A + S _ { 1 1 } \big ) ^ { - 1 } S _ { 1 2 } \big ) ^ { - 1 }
$$

A $\mathfrak { r } \mathfrak { o } \to 0 , \frac { 1 } { \sigma ^ { 2 } } A$ blows up as A is positive definite, so $\left( { \frac { 1 } { \sigma ^ { 2 } } } A + \right.$ $S _ { 1 1 } ) ^ { - 1 } \to 0$ , and only the (2,2) block survives as $S _ { 2 2 } ^ { - 1 }$ . Hence,

$$
\operatorname* { l i m } _ { \sigma \to 0 } \operatorname { T r } \Big ( \left( \mathbb { E } \big [ I _ { e } ( x ) \big ] + S \right) ^ { - 1 } \Big ) = \operatorname { T r } ( S _ { 2 2 } ^ { - 1 } ) = \operatorname { T r } \Big ( ( U ^ { \top } S U ) ^ { - 1 } \Big ) .
$$

Theorem 2 (Ψ-bound) Let $\Psi : \mathbb { R } ^ { d }  \mathbb { R } ^ { k }$ be a twicedifferentiable function, with $\begin{array} { r } { J _ { \Psi } = \partial \Psi / \partial x . } \end{array}$ Let $P _ { 1 }$ and $P _ { 2 }$ be two orthogonal projectors that are orthogonal to each other and $P _ { 1 } + P _ { 2 } = I _ { d }$ (it can be generalized to arbitrary number ofprojectors, but we keep it two for simplicity). Define for $g \in \{ 1 , 2 \}$

$$
n _ { g } = \mathbb { E } \big [ \big \| J _ { \Psi } P _ { g } \big \| _ { F } ^ { 2 } \big ] , \qquad t _ { g } = \Delta _ { g } \Psi + J _ { \Psi } P _ { g } s _ { \pi } ,
$$

where $\Psi _ { i } : \mathbb { R } ^ { d } $ R is the i-th component ofΨ, and the i-th row of $\Delta _ { g } \Psi$ is $( \Delta _ { g } \Psi ) _ { i } = \mathrm { T r } \left( P _ { g } \nabla ^ { 2 } \Psi _ { i } \right)$ . Next, we define a $2 \times 2$ matrix $D ,$ , with the following (i, j)-th elements:

$$
D _ { i j } = \frac { 1 } { \sigma ^ { 2 } } \mathbb { E } \left[ \langle J _ { \mathrm { E n c } _ { \mathrm { D } } } P _ { i } J _ { \Psi } ^ { \top } , J _ { \mathrm { E n c } _ { \mathrm { D } } } P _ { j } J _ { \Psi } ^ { \top } \rangle _ { F } \right] + \mathbb { E } \left[ \langle t _ { i } , t _ { j } \rangle \right] ,
$$

where $\langle \cdot , \cdot \rangle _ { F }$ is a Frobenius inner product. Suppose Assumption 1 holds, that $J _ { \Psi } ( x ) P _ { g } p ( e , x ) \to 0$ as $\| x \| $ ∞ for a.e. e and each g, that the moments $n _ { g }$ and $D _ { i j }$ arefinite, and that $D \succ 0 .$ . Thenfor any attacker $\hat { x } = \operatorname { A t t } ( e )$

$$
\begin{array} { r } { \mathbb { E } \| \Psi ( \hat { x } ) - \Psi ( x ) \| ^ { 2 } \geq n ^ { \top } D ^ { - 1 } n , \qquad n = [ n _ { 1 } \ n _ { 2 } ] ^ { \top } . } \end{array}\tag{7}
$$

Proof 3 (Proof of Theorem 2) Apply Lemma 2 with

$$
C ( x ) = \sum _ { g \in \{ 1 , 2 \} } w _ { g } J _ { \Psi } ( x ) P _ { g } , \qquad w _ { g } \in \mathbb { R } .
$$

for arbitrary w<sub>1</sub>, $w _ { 2 } \in \mathbb { R }$ . Since $P _ { g }$ is a projector, the numerator ofLemma 2 becomes

$$
\begin{array} { r l } { \mathbb { E } \big [ \mathrm { T r } ( J _ { \Psi } C ^ { \top } ) \big ] = } & { \displaystyle \sum _ { g \in \{ 1 , 2 \} } w _ { g } \mathbb { E } \big [ \mathrm { T r } ( J _ { \Psi } ( x ) P _ { g } J _ { \Psi } ( x ) ^ { \top } ) \big ] } \\ & { = \displaystyle \sum _ { g \in \{ 1 , 2 \} } w _ { g } \mathbb { E } \big [ \mathrm { T r } ( J _ { \Psi } ( x ) P _ { g } P _ { g } J _ { \Psi } ( x ) ^ { \top } ) \big ] } \\ & { = \displaystyle \sum _ { g \in \{ 1 , 2 \} } w _ { g } \mathbb { E } \big [ \mathrm { T r } ( P _ { g } J _ { \Psi } ( x ) ^ { \top } J _ { \Psi } ( x ) P _ { g } ) \big ] } \\ & { = \displaystyle \sum _ { g \in \{ 1 , 2 \} } w _ { g } \mathbb { E } \big [ \mathrm { T r } ( P _ { g } J _ { \Psi } ( x ) ^ { \top } J _ { \Psi } ( x ) P _ { g } ) \big ] } \\ & { = \mathbb { E } \big [ \displaystyle \sum _ { g \in \{ 1 , 2 \} } w _ { g } \| J _ { \Psi } ( x ) P _ { g } \| _ { F } ^ { 2 } \big ] } \\ & { = \displaystyle \sum _ { g \in \{ 1 , 2 \} } w _ { g } n _ { g } = w ^ { \top } n , \qquad w = \big [ w _ { 1 } ~ w _ { 2 } \big ] ^ { \top } . } \end{array}
$$

Next, we consider thefirst denominator term ofLemma 2. The i-th row ofC is $\begin{array} { r } { \sum _ { g \in \{ 1 , 2 \} } w _ { g } \nabla \Psi _ { i } ^ { \top } P _ { g } , } \end{array}$ so

$$
\begin{array} { r l } & { \mathbb { E } \big [ \mathrm { T r } \big ( C ( x ) I _ { e } ( x ) C ( x ) ^ { \top } \big ) \big ] } \\ & { \quad = \frac { 1 } { \mathfrak { C } ^ { 2 } } \mathbb { E } \big [ \mathrm { T r } \big ( C ( x ) J _ { \mathrm { E n c } D } ( x ) ^ { \top } J _ { \mathrm { E n c } D } ( x ) C ( x ) ^ { \top } \big ) \big ] } \\ & { \quad = \frac { 1 } { \mathfrak { C } ^ { 2 } } \mathbb { E } \Big [ \big \| J _ { \mathrm { E n c } \Omega } C ^ { \top } \big \| _ { F } ^ { 2 } \Big ] } \\ & { \quad = \frac { 1 } { \mathfrak { C } ^ { 2 } } \mathbb { E } \Big [ \underset { i } { \overset { . . . } { \prod } } \big \| J _ { \mathrm { E n c } \Omega } \big ( \underset { g \in \{ 1 , 2 \} } { \overset { . . . } { \sum } } w _ { g } \nabla \Psi _ { i } ^ { \top } P _ { g } \big ) ^ { \top } \big \| _ { F } ^ { 2 } \Big ] } \\ & { \quad = \frac { 1 } { \mathfrak { C } ^ { 2 } } \mathbb { E } \Big [ \underset { i } { \overset { . . . } { \prod } } \big \| \underset { g \in \{ 1 , 2 \} } { \overset { . . . } { \sum } } w _ { g } J _ { \mathrm { E n c } \Omega } P _ { g } \nabla \Psi _ { i } \big \| _ { F } ^ { 2 } \Big ] . } \end{array}
$$

For notational simplicity, if we let $K _ { g } = J _ { \mathrm { E n c _ { D } } } P _ { g } J _ { \Psi } ^ { \top } .$

$$
\begin{array} { r l } & { = \displaystyle \frac { 1 } { \sigma ^ { 2 } } \mathbb { E } [ \sum _ { i } \langle \sum _ { \substack { g \in \{ 1 , 2 \} } } w _ { g } J _ { \mathrm { E n c _ { D } } } P _ { g } \nabla \Psi _ { i } , \sum _ { g \in \{ 1 , 2 \} } w _ { g } J _ { \mathrm { E n c _ { D } } } P _ { g } \nabla \Psi _ { i } \rangle ] } \\ & { = \displaystyle \frac { 1 } { \sigma ^ { 2 } } w ^ { \top } ( \mathbb { E } [ \langle K _ { 1 } , K _ { 1 } \rangle _ { F } ]  \quad \mathbb { E } [ \langle K _ { 1 } , K _ { 2 } \rangle _ { F } ] ) w . } \end{array}
$$

Finally, let’s consider the second denominator term of Lemma 2. The i-th row of the divergence of C is:

$$
\begin{array} { r l } { \displaystyle \mathrm { d i v } \Big ( \displaystyle \sum _ { g \in \{ 1 , 2 \} } w _ { g } \nabla \Psi _ { i } ^ { \top } P _ { g } \Big ) = \displaystyle \sum _ { g \in \{ 1 , 2 \} } w _ { g } \mathrm { d i v } \Big ( \nabla \Psi _ { i } ^ { \top } P _ { g } \Big ) } & { } \\ { = \displaystyle \sum _ { g \in \{ 1 , 2 \} } w _ { g } \nabla \cdot \Big ( P _ { g } \nabla \Psi _ { i } \Big ) } & { } \\ { = \displaystyle \sum _ { g \in \{ 1 , 2 \} } w _ { g } \mathrm { T r } \Big ( P _ { g } \nabla ^ { 2 } \Psi _ { i } \Big ) } & { } \\ { = \displaystyle \sum _ { g \in \{ 1 , 2 \} } w _ { g } \Big ( \Delta _ { G } \Psi \Big ) _ { i } . } & { } \end{array}
$$

Hence,

$$
\begin{array} { r l } { \mathbb { E } \left[ \| \operatorname { d i v } C + C s _ { \pi } \| ^ { 2 } \right] = \mathbb { E } \left[ \| \displaystyle \sum _ { g \in \{ 1 , 2 \} } w _ { g } t _ { g } \| ^ { 2 } \right] } & { } \\ { = \displaystyle \sum _ { g \in \{ 1 , 2 \} } \sum _ { g ^ { \prime } \in \{ 1 , 2 \} } w _ { g } w _ { g ^ { \prime } } \langle t _ { g } , t _ { g ^ { \prime } } \rangle } & { } \\ { = w ^ { \top } \left( \mathbb { E } [ \left. t _ { 1 } , t _ { 1 } \right. ] \quad \mathbb { E } [ \left. t _ { 1 } , t _ { 2 } \right. ] \right) w . } \end{array}
$$

Substituting all the components back into Lemma $^ { 2 , }$ , every nonzero w gives a valid bound $( w ^ { \top } n ) ^ { 2 } / ( w ^ { \top } D w )$ . To get its supremum, we define an innerproduct $\langle u , \nu \rangle _ { D } = u ^ { \top } D \nu ( D \succ 0$ required). Then, $w ^ { \top } n = \langle w , D ^ { - 1 } n \rangle _ { D }$ . From Cauchy-Schwarz:

$$
\begin{array} { r } { ( \boldsymbol { w } ^ { \top } \boldsymbol { n } ) ^ { 2 } \leq \langle \boldsymbol { w } , \boldsymbol { w } \rangle _ { D } \langle D ^ { - 1 } \boldsymbol { n } , D ^ { - 1 } \boldsymbol { n } \rangle _ { D } = ( \boldsymbol { w } ^ { \top } D \boldsymbol { w } ) ( n ^ { \top } D ^ { - 1 } \boldsymbol { n } ) . } \end{array}
$$

The final step leverages the fact that D is symmetric. Thus,

$$
\operatorname* { s u p } _ { w \neq 0 } \frac { ( w ^ { \top } n ) ^ { 2 } } { w ^ { \top } D w } = n ^ { \top } D ^ { - 1 } n ,
$$

with equality at w ∝ $D ^ { - 1 } n$

Corollary 3 (Zero-noise Ψ-bound) From Theorem 2, let $P _ { \mathrm { r } }$ be the projector onto the range space range $\left( \mathbb { E } \left\lceil G \right\rceil \right)$ , and let $P _ { \mathrm { n } }$ be the projector onto the null space null $( \mathbb { E } [ G ] )$ ). Let $t _ { \mathrm { r } } , t _ { \mathrm { n } } , n _ { \mathrm { r } } ,$ and $n _ { \mathrm { n } }$ be the respective t and n vectorsfor P and $P _ { \mathrm { n } } .$ . Then, as $\sigma  0$

$$
\begin{array} { r } { n ^ { \top } D ^ { - 1 } n \longrightarrow \frac { n _ { \mathrm { n } } ^ { 2 } } { \mathbb { E } \left[ \left. t _ { \mathrm { n } } \right. ^ { 2 } \right] } = \frac { \left( \mathbb { E } \left[ \left. J _ { \Psi } P _ { \mathrm { n } } \right. _ { F } ^ { 2 } \right] \right) ^ { 2 } } { \mathbb { E } \left[ \left. \Delta _ { \mathrm { n } } \Psi + J _ { \Psi } P _ { \mathrm { n } } s _ { \pi } \right. ^ { 2 } \right] } . } \end{array}
$$

Proof 4 (Proof of Corollary 3) If $P _ { 1 } = P _ { \mathrm { r } }$ and $P _ { 2 } = P _ { \mathrm { n } } , D$ can befurther simplified with thefact that $J _ { \mathrm { E n c } _ { \mathrm { D } } } P _ { \mathrm { r } } = J _ { \mathrm { E n c } _ { \mathrm { D } } }$ and $J _ { \mathrm { E n c } _ { \mathrm { D } } } P _ { \mathrm { n } } = 0 f o r$ π-a.e. x. Then, any term involving $J _ { \mathrm { E n c } _ { \mathrm { D } } } P _ { \mathrm { n } } J _ { \Psi } ^ { \top }$ becomes zero, and D becomes:

$$
\begin{array} { r } { D = \left( \begin{array} { c c } { \frac { 1 } { \sigma ^ { 2 } } \mathbb { E } \left[ \left. J _ { \mathrm { E n c } _ { \mathrm { D } } } J _ { \Psi } ^ { \top } \right. _ { F } ^ { 2 } \right] + \mathbb { E } \left[ \left. t _ { \mathrm { r } } , t _ { \mathrm { r } } \right. \right] } & { \mathbb { E } \left[ \left. t _ { \mathrm { r } } , t _ { \mathrm { n } } \right. \right] } \\ { \mathbb { E } \left[ \left. t _ { \mathrm { n } } , t _ { \mathrm { r } } \right. \right] } & { \mathbb { E } \left[ \left. t _ { \mathrm { n } } , t _ { \mathrm { n } } \right. \right] } \end{array} \right) . } \end{array}
$$

Write

$$
T = \left( \mathbb { E } \left[ \left. t _ { \mathrm { r } } , t _ { \mathrm { r } } \right. \right] \quad \mathbb { E } \left[ \left. t _ { \mathrm { r } } , t _ { \mathrm { n } } \right. \right] \right) , \qquad k = \frac { \mathbb { E } \left[ \left\| J _ { \mathrm { E n c } _ { \mathrm { D } } } J _ { \Psi } ^ { \top } \right\| _ { F } ^ { 2 } \right] } { \sigma ^ { 2 } } .
$$

Then, we can rewrite D as:

$$
\boldsymbol { D } = \boldsymbol { T } + k e _ { 1 } e _ { 1 } ^ { \intercal } ,
$$

where $e _ { 1 } = [ 1 0 ] ^ { \top }$ . From Sherman-Morrisonformula,

$$
D ^ { - 1 } = T ^ { - 1 } - \frac { k T ^ { - 1 } e _ { 1 } e _ { 1 } ^ { \top } T ^ { - 1 } } { 1 + k e _ { 1 } ^ { \top } T ^ { - 1 } e _ { 1 } } .
$$

As $\sigma  0 , k  \infty ,$ , so this becomes:

$$
T ^ { - 1 } - { \frac { T ^ { - 1 } e _ { 1 } e _ { 1 } ^ { \top } T ^ { - 1 } } { e _ { 1 } ^ { \top } T ^ { - 1 } e _ { 1 } } } = { \binom { 0 } { 0 } } \quad { 0 } = { \binom { 0 } { 0 } } \quad 1 / \mathbb { E } \left[ \| t _ { \mathrm { n } } \| ^ { 2 } \right] \mathrm { . }
$$

Plugging this into $n ^ { \top } D ^ { - 1 } n$ concludes the proof.

## A.6 Proofs of the MMSE Attack

Here, we prove that Equations 8 and 9 are the MMSE estimators when reconstructing Gaussian inputs $x \sim \mathcal { N } ( \mu , \boldsymbol { \eta ^ { 2 } } I _ { d } )$ from a linear encoder output $e = W x + \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { d } )$

$\sigma > 0$ case. First, recall that the MMSE estimator is the posterior mean $\hat { x } = \mathbb { E } [ x \mid e ]$ . For $\sigma > 0$

$$
e \mid x \sim { \mathcal { N } } ( W x , \sigma ^ { 2 } I _ { m } ) ,
$$

so the likelihood is

$$
p ( e \mid x ) \propto \exp \left( - \frac { 1 } { 2 { \sigma } ^ { 2 } } \| e - W x \| _ { 2 } ^ { 2 } \right) .
$$

The prior is

$$
\pi ( x ) \propto \exp \left( - \frac { 1 } { 2 \eta ^ { 2 } } \| x - \mu \mathbf { 1 } \| _ { 2 } ^ { 2 } \right) .
$$

By Bayes’ rule,

$$
p ( x \mid e ) \propto p ( e \mid x ) \pi ( x ) ,
$$

and hence

$$
p ( x \mid e ) \propto \exp \left[ - \frac { 1 } { 2 \sigma ^ { 2 } } \| e - W x \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 \eta ^ { 2 } } \| x - \mu { \bf 1 } \| _ { 2 } ^ { 2 } \right] .
$$

Expand the two quadratic terms:

$$
\| e - W x \| _ { 2 } ^ { 2 } = e ^ { \top } e - 2 x ^ { \top } W ^ { \top } e + x ^ { \top } W ^ { \top } W x ,
$$

and

$$
\| x - \mu \mathbf { 1 } \| _ { 2 } ^ { 2 } = x ^ { \top } x - 2 \mu \mathbf { 1 } ^ { \top } x + \mathrm { c o n s t a n t } .
$$

Ignoring terms that do not depend on x,

$$
\begin{array} { l } { { \displaystyle p ( x \mid e ) \propto } \ ~ } \\ { { \displaystyle ~ \exp \left[ - \frac { 1 } { 2 } \left( x ^ { \top } \left( \frac { W ^ { \top } W } { \mathfrak {sigma } ^ { 2 } } + \frac { I _ { d } } { \mathfrak { \eta } ^ { 2 } } \right) x - 2 x ^ { \top } \left( \frac { W ^ { \top } e } { \mathfrak {sigma } ^ { 2 } } + \frac { \mu \mathbf { 1 } } { \mathfrak { \eta } ^ { 2 } } \right) \right) \right] . } } \end{array}
$$

Define

$$
R = \frac { W ^ { \top } W } { \sigma ^ { 2 } } + \frac { I _ { d } } { \eta ^ { 2 } }
$$

and

$$
b = \frac { W ^ { \top } e } { \sigma ^ { 2 } } + \frac { \mu \mathbf { 1 } } { \eta ^ { 2 } } .
$$

Then

$$
p ( x \mid e ) \propto \exp \left[ - { \frac { 1 } { 2 } } \left( x ^ { \top } R x - 2 x ^ { \top } b \right) \right] .
$$

As R is symmetric,

$$
\boldsymbol { x } ^ { \top } \boldsymbol { R } \boldsymbol { x } - 2 \boldsymbol { x } ^ { \top } \boldsymbol { b } = ( \boldsymbol { x } - \boldsymbol { R } ^ { - 1 } \boldsymbol { b } ) ^ { \top } \boldsymbol { R } ( \boldsymbol { x } - \boldsymbol { R } ^ { - 1 } \boldsymbol { b } ) + \mathrm { c o n s t a n t } ,
$$

so

$$
p ( x \mid e ) \propto \exp \left[ - \frac 1 2 \left( ( x - R ^ { - 1 } b ) ^ { \top } R ( x - R ^ { - 1 } b ) \right) \right] .
$$

This is a Gaussian with mean $R ^ { - 1 } b$ and covariance $R ^ { - 1 }$ :

$$
\begin{array} { r } { \boldsymbol { x } \mid e \sim \mathcal { N } ( R ^ { - 1 } b , R ^ { - 1 } ) . } \end{array}
$$

Thus, the posterior mean (MMSE) is

$$
\hat { x } = R ^ { - 1 } \left( \frac { W ^ { \top } e } { \sigma ^ { 2 } } + \frac { \mu \mathbf { I } } { \eta ^ { 2 } } \right) .
$$

Using

$$
R \mu { \bf 1 } = \frac { W ^ { \top } W \mu { \bf 1 } } { \sigma ^ { 2 } } + \frac { \mu { \bf 1 } } { \eta ^ { 2 } } ,
$$

this can equivalently be written as

$$
\hat { x } = \mu \mathbf { 1 } + R ^ { - 1 } \frac { W ^ { \top } ( e - W \mu \mathbf { 1 } ) } { \sigma ^ { 2 } } ,
$$

which is Equation 8.

$\sigma = 0$ case. When $\sigma = 0 .$ , the above equation cannot be used because of the $1 / \sigma ^ { 2 }$ term. Instead, note that the observation is noiseless: $e = W x$ . Define $y = x - \mu \mathbf { 1 }$ . Then,

$$
\begin{array} { r } { y \sim \mathcal { N } ( 0 , \boldsymbol { \eta } ^ { 2 } I _ { d } ) , } \end{array}
$$

and

$$
e - W \mu \mathbf { l } = W y = z .
$$

Our goal is to find

$$
\hat { x } = \mathbb { E } \big [ x | e \big ] = \mathbb { E } \big [ y | e \big ] + \mu \mathbf { 1 } = \mathbb { E } \big [ y | z \big ] + \mu \mathbf { 1 } .
$$

Let $P _ { \mathrm { r o w } }$ and $P _ { \mathrm { n u l l } }$ be the orthogonal projectors to row and null spaces of W. Decompose y into its row-space and null-space components:

$$
y = P _ { \mathrm { r o w } } y + P _ { \mathrm { n u l l } } y .
$$

The null-space component is completely invisible to the observation, since

$$
{ \cal W P } _ { \mathrm { n u l l } } y = 0 .
$$

Moreover, because $y \sim \mathcal { N } ( 0 , \tau ^ { 2 } I )$ , the row-space and nullspace components are independent. Therefore, observing z provides no information about $P _ { \mathrm { n u l l } } )$ , and

$$
\mathbb { E } \big [ P _ { \mathrm { n u l l } } y \mid z \big ] = \mathbb { E } \big [ P _ { \mathrm { n u l l } } y \big ] = 0 .
$$

For the row-space component, we use

$$
P _ { \mathrm { r o w } } = W ^ { + } W ,
$$

where $W ^ { + }$ is a Moore-Penrose pseudoinverse of W. Thus,

$$
\begin{array} { r l } & { \mathbb { E } \big [ y \mid z \big ] + \mu \mathbf { 1 } = \mathbb { E } \big [ P _ { \mathrm { r o w } } y \mid z \big ] + \mu \mathbf { 1 } } \\ & { \qquad = \mathbb { E } \big [ W ^ { + } W y \mid z \big ] + \mu \mathbf { 1 } } \\ & { \qquad = \mathbb { E } \big [ W ^ { + } z \mid z \big ] + \mu \mathbf { 1 } } \\ & { \qquad = W ^ { + } z + \mu \mathbf { 1 } } \\ & { \qquad = W ^ { + } ( e - W \mu \mathbf { 1 } ) + \mu \mathbf { 1 } . } \end{array}
$$

This is Equation 9, thus we conclude our proof.