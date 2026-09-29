# EXPLAINING HYPERBOLIC NEURAL NETWORKS VIA GEOMETRY-AWARE RELEVANCE PROPAGATION

Ping Xiong<sup>1,2</sup>, Shanglin Li<sup>1,2</sup>, Yi Ding<sup>3</sup>, Thomas Schnake<sup>4,5</sup>, Shinichi Nakajima<sup>1,2,6∗</sup> <sup>1</sup>BIFOLD, Germany, <sup>2</sup>Technische Universitat Berlin, Germany¨

<sup>3</sup>Nanyang Technological University, Singapore, <sup>4</sup>University of Toronto, Canada

<sup>5</sup>Vector Institute for Artificial Intelligence, Canada, <sup>6</sup>RIKEN Center for AIP, Japan

## ABSTRACT

Hyperbolic neural networks introduce geometric operations that require explicit treatment in relevance propagation. Equivalent geometric realizations can produce different feature attributions, even when local relevance is conserved. We study this problem through Geometric Representation Invariance (GRI), a specialization of Implementation Invariance, and zero-curvature consistency, which requires identity relevance propagation when a geometric module approaches the identity. We propose LRP-radial-all for origin-centered radial modules, treating geometric scaling as modulation and assigning relevance entirely to the signal branch. The rule conserves relevance, is invariant to equivalent radial factorizations, and satisfies zero-curvature consistency, yielding GRI for a specified Poincare–Lorentz´ logarithmic-map construction. In contrast, a conservative LRP-half baseline can violate both consistency criteria. Experiments on hyperbolic MNIST, sEEG, and CIFAR-10 classifiers assess attribution fidelity, qualitative explanations, and runtime. LRP-radial-all achieves competitive attribution fidelity across datasets with runtime comparable to Gradient×Input and substantially lower than Integrated Gradients. These findings motivate geometry-aware propagation rules that distinguish relevance conservation from consistency across equivalent computations.

## 1 INTRODUCTION

Explainable artificial intelligence (XAI) seeks to clarify how models arrive at their predictions (Gunning et al., 2019; Samek et al., 2021; Holzinger et al., 2020; Samek et al., 2019). However, most existing XAI methods have been developed primarily for conventional Euclidean neural networks. How these methods should be adapted to models whose computations are governed by non-Euclidean geometry remains much less understood.

This question becomes increasingly important for hyperbolic neural networks, which perform neural network operations in hyperbolic space and have attracted increasing attention (Peng et al., 2021; He et al., 2025). This interest is partly motivated by the mismatch between the exponential growth of tree-like hierarchies and the polynomial volume growth of Euclidean space. Hyperbolic space, by contrast, exhibits exponential volume growth with radius, making it well suited to representing hierarchical structures (Krioukov et al., 2010). Leveraging this geometric advantage, hyperbolic representations have shown promising performance across a range of tasks involving hierarchical structure, including computer vision (Ganea et al., 2018; Khrulkov et al., 2020; Mettes et al., 2024), natural language processing (Tifrea et al., 2019), foundation models (Desai et al., 2023; He et al., 2026), and biosignal analysis (Zhou et al., 2026; Li et al., 2026).

However, hyperbolic geometry poses distinct challenges for feature attribution: geometric maps couple feature coordinates, and equivalent representations can expose different computational factorizations of the same intrinsic function. An explanation method must therefore account for these operations while maintaining consistency across equivalent implementations. Layer-wise relevance propagation (LRP) offers a particularly suitable framework for studying this problem, as it redistributes a prediction backward through individual network modules using explicit local rules (Bach et al., 2015). This modular structure allows us to examine how relevance should propagate through hyperbolic operations and whether local conservation is compatible with Geometric Representation Invariance.

![](images/49e31f49ccf8b8bbe33c4a00ae33cea5496daa278ecab6a0bc6968ff956712c1.jpg)  
Figure 1: Geometric Representation Invariance (GRI) requires explanations at the same input after equivalent internal coordinate realizations to be the same.

In this work, we propose a relevance propagation framework for origin-centered hyperbolic opera tions. The origin provides a natural fixed reference point whose tangent space admits a Euclidean representation and is mathematically simple, facilitating efficient computation (Liu et al., 2019). We formulate LRP-radial-all rule for logarithmic maps and exponential maps, treating geometric scaling as modulation and assigning relevance to the signal branch. We study Geometric Representation Invariance (illustrated in Figure 1) as a specialization of Implementation Invariance (Sundararajan et al., 2017), prove consistency for the specified logarithmic-map construction, and show that a conservative LRP-half baseline can violate this property under equivalent radial factorizations. We evaluate the framework through controlled simulations and extensive experiments on various datasets and model architectures, including image and time series. Among the five equivalent hyperbolic models (Cannon et al., 1997), Poincare and Lorentz are the most popular choices and are ´ covered in this paper (Peng et al., 2021; He et al., 2025).

We make three main contributions.

• We propose LRP-radial-all for origin-centered hyperbolic modules, assigning relevance to the signal while treating geometric scaling as modulation.

• We specialize Implementation Invariance to GRI and introduce zero-curvature consistency. We prove conservation, radial factorization invariance, and zero-curvature consistency for radial-all, including GRI for the specified logarithmic-map test, and show that conservative LRP-half can violate both consistency criteria.

• We evaluate attribution fidelity and runtime on MNIST, sEEG, and CIFAR-10 hyperbolic classifiers, alongside controlled consistency tests.

## 2 RELATED WORK

Hyperbolic neural networks. Early work on hyperbolic neural networks extended standard neural operations to hyperbolic geometry. Ganea et al. (2018) introduced hyperbolic formulations of multinomial logistic regression and feed-forward layers using the Poincare ball. Subsequent work ´ broadened the architectural scope: Shimizu et al. (2021) developed hyperbolic fully connected, convolutional, and attention layers, while Bdeir et al. (2024) introduced fully hyperbolic convolutional networks in the Lorentz model for computer vision. Beyond vision, hyperbolic architectures have also been applied to intracranial recordings. For example, Li et al. (2026) incorporated hyperbolic layers into an EEGNet-based model for working-memory load classification from sEEG signals. Together, these developments have made geometry-specific operations such as exponential maps, logarithmic maps, and manifold-valued feature transformations fundamental building blocks of modern hyperbolic architectures.

Explainability for HNN. In contrast to the rapid development of hyperbolic architectures, their explainability has received comparatively limited attention. Most feature-attribution methods were originally formulated for models operating in Euclidean spaces, and therefore do not explicitly account for the geometric operations introduced by HNNs. Recent work has begun to incorporate non-Euclidean geometry into explainability. Manifold Integrated Gradients (MIG) (Zaher et al., 2024) replaces the straight-line path of Integrated Gradients with geodesic paths on a learned Riemannian data manifold, thereby aligning attribution with the intrinsic geometry of the data. Diffeomorphic counterfactual methods use generative models to construct coordinate systems in which gradientbased search produces semantically meaningful changes to the input (Dombrowski et al., 2024). Such approaches demonstrate the relevance of geometry to explainability, but focus primarily on the geometry of the data manifold, rather than on propagating explanations through the internal geometric operations of a hyperbolic neural network. Our work addresses the complementary problem. We study how relevance should be propagated through the geometric primitives of HNNs themselves, with particular attention to logarithmic and exponential maps and transformations between equivalent hyperbolic representations.

Layer-wise relevance propagation (LRP). Layer-wise relevance propagation (LRP) (Bach et al., 2015) decomposes a model prediction into input relevance scores through architecture-specific backward redistribution rules, typically at a computational cost comparable to a backward computation. Deep Taylor Decomposition provides a theoretical foundation for certain LRP rules through local Taylor expansions of relevance functions (Montavon et al., 2017; Samek et al., 2021). Our work is particularly related to the rules for multiplicative interactions. For gated recurrent networks, Arras et al. (2017) assign all relevance to the source branch and none to the gate, interpreting the latter as a modulation of the signal. Similarly, Ali et al. (2022) treat attention weights as fixed during relevance propagation and route relevance through the value branch, following an LRP-all-style allocation for the attention-value product. Alternatively, LRP-half for LSTM (Arras et al., 2019) propagates half of the relevance to weight and half to signal. AttnLRP (Achtibat et al., 2024) derives a uniform redistribution rule for multiplicative interactions. These approaches motivate different treatments of hyperbolic radial maps.

## 3 GEOMETRY-AWARE CRITERIA

## 3.1 HYPERBOLIC NEURAL NETWORKS

Hyperbolic neural networks (HNNs) generalize neural operations to hyperbolic space (Ganea et al., 2018). We describe a representative architecture in the Poincare ball´ $\bar { \mathbb { D } } _ { c } ^ { \mathbf { \hat { d } } } = \{ \pmb { x } \in \dot { \mathbb { R } } ^ { d } : c \| \pmb { x } \| _ { 2 } ^ { 2 } < 1 \}$ with curvature −c, where $c > 0$ . An input feature vector $\mathbf { \pmb { a } } \in \mathbb { R } ^ { d }$ is first interpreted as an origin tangent vector and mapped to the ball: $\pmb { h } ^ { ( 0 ) } = \exp _ { \mathbf { 0 } } ^ { c } ( \pmb { a } )$ . Each hidden layer performs a hyperbolic linear transformation, bias addition, and activation:

$$
{ \pmb m } ^ { ( \ell ) } = \mathrm { e x p } _ { { \bf 0 } } ^ { c } ( { \pmb W } ^ { ( \ell ) } \log _ { { \bf 0 } } ^ { c } ( { \pmb h } ^ { ( \ell - 1 ) } ) ) \oplus _ { c } \mathrm { e x p } _ { { \bf 0 } } ^ { c } ( { \pmb b } ^ { ( \ell ) } ) ,\tag{1}
$$

$$
\pmb { h } ^ { ( \ell ) } = \mathrm { e x p } _ { \mathbf 0 } ^ { c } ( \sigma ( \mathrm { l o g } _ { \mathbf 0 } ^ { c } ( \pmb { m } ^ { ( \ell ) } ) ) ) ,\tag{2}
$$

where $W ^ { ( \ell ) }$ and $\pmb { b } ^ { ( \ell ) }$ are trainable parameters, $\oplus _ { c }$ denotes Mobius addition (definition see Eq.65)¨ and σ is an element-wise activation applied in the tangent space. We show the equations of exp<sup>c</sup> (·) and $\log _ { 0 } ^ { c } ( \cdot )$ in Figure 2. For a tangent-space readout, the final representation is mapped back to Eu clidean coordinates and passed to a prediction head g: $: \hat { \pmb { y } } = g \big ( \log _ { \mathbf { 0 } } ^ { c } ( { \pmb { h } } ^ { ( L ) } ) \big )$  . Appendix A summarizes the Lorentz model and its correspondence with the Poincare representation.´

## 3.2 GEOMETRIC REPRESENTATION INVARIANCE

The Poincare ball and Lorentz hyperboloid are isometric representations of the same hyperbolic ´ geometry. By transforming intermediate representations and the associated operations consistently, a model can be expressed in either space without changing its input-output function. Ideally, explanations in the original input space should also remain unchanged under such a conversion. This requirement is motivated by Implementation Invariance (Sundararajan et al., 2017), which states that functionally equivalent models should yield identical input attributions. We specialize this principle to equivalent hyperbolic representations and refer to the resulting criterion as Geometric Representation Invariance (GRI).

Definition 1 (GRI as a restricted Implementation Invariance criterion). Let $\mathbf { \pmb { a } } \in \mathbb { R } ^ { p }$ be the original feature vector. Let $F _ { P }$ and $F _ { L }$ implement the same scalar targetfunction on a common input domain, with internal hyperbolic computations related by the specified isometries. For a fixed explanation procedure, GRI requires

$$
\mathcal { E } ( F _ { P } , \pmb { a } ) = \mathcal { E } ( F _ { L } , \pmb { a } ) ,\tag{3}
$$

where E denotes an explanation method.

For gradient-based methods defined solely through evaluations of the input-output function such as Gradient×Input (Ancona et al., 2018) and Integrated Gradients (Sundararajan et al., 2017), GRI follows directly from functional equivalence. For propagation-based methods such as LRP, however, GRI is non-trivial because relevance redistribution depends on intermediate computations and local propagation rules. Equivalent geometric representations may expose different computational factorizations, so local relevance conservation alone does not guarantee identical input attributions.

A controlled test of GRI. Figure 1 illustrates a controlled test of whether relevance propagation depends on the internal representation of a hyperbolic logarithmic map. Starting from the same original input $^ { a , }$ a shared encoder produces a Poincare representation´ $\pmb { x } _ { P } \in \mathbb { D } _ { c } ^ { d }$ . We compare two computational paths: the direct path applies the Poincare logarithmic map, whereas the alternative ´ path first converts $\scriptstyle { \mathbf { \boldsymbol { x } } } _ { P }$ to the corresponding Lorentz point ${ \pmb x } _ { L } = \phi ( { \pmb x } _ { P } )$ and then applies the Lorentz logarithmic map.

After aligning the tangent coordinates, both paths produce the same vector $\pmb { v } \in \mathbb { R } ^ { d } ;$

$$
\begin{array} { r } { \pmb { v } = \alpha _ { P } \big ( \pmb { x } _ { P } \big ) \pmb { x } _ { P } = \alpha _ { L } \big ( \pmb { x } _ { L } \big ) \bar { \pmb { x } } _ { L } , \qquad \bar { \pmb { x } } _ { L } = \gamma \big ( \pmb { x } _ { P } \big ) \pmb { x } _ { P } . } \end{array}\tag{4}
$$

Here ${ \pmb x } _ { L } = ( x _ { L , 0 } , \bar { { \pmb x } } _ { L } )$ , and $\alpha _ { L }$ includes the factor $1 / 2$ that converts the ambient Lorentz tangent vector $( 0 , 2 v )$ to the common coordinates ${ \pmb v } .$ . The coordinate conversion and aligned logarithmic maps are derived in Appendix D. Both paths then use the same downstream computation $y = G ( \pmb { v } )$ so they implement the same input-output function.

To test GRI, we initialize relevance at the same scalar output y and propagate it backward along each path, as indicated by the red arrows in Figure 1. The downstream propagation is identical and therefore gives the same relevance $\scriptstyle { R _ { v } }$ to both realizations. The direct path propagates this relevance through the Poincare logarithmic map and the alternative path propagates it through the ´ aligned Lorentz logarithmic map and the coordinate conversion $\phi .$ Using the same relevance rule through the shared encoder, we compare the resulting explanations at the original input:

$$
R _ { a } ^ { \mathrm { d i r e c t } } \overset { ? } { = } R _ { a } ^ { \mathrm { v i a } L } .\tag{5}
$$

This construction isolates the effect of an equivalent geometric realization while keeping the input, prediction, and surrounding computation unchanged.

Appendix E presents a controlled exponential-map test with a shared Lorentz downstream network and an explicitly matched treatment of time-coordinate relevance.

## 3.3 ZERO-CURVATURE CONSISTENCY

A natural consistency requirement arises when hyperbolic operations approach their Euclidean counterparts as curvature vanishes (Ganea et al., 2018). In particular, the origin-centered Poincare expo-´ nential and logarithmic maps converge to the identity in the coordinate convention used here. Once such a geometric module becomes an identity transformation, it should no longer redistribute relevance among input coordinates. This motivates requiring its backward relevance rule to approach identity propagation as well. We formalize this local requirement as zero-curvature consistency.

Definition 2 (Zero-curvature consistency). Let $f _ { c }$ be a family of geometric modules expressed in fixed coordinates such that $f _ { c } ( { \pmb x } ) $ kx as $c \to 0 .$ , where $k > 0$ is fixed and independent of x. We use identity relevance propagationfor the limiting constant scaling, treating k as afixed modulation factor. $L e t \mathcal { R } _ { c } ( { \bf x } , R )$ denote the input relevance obtained by propagating afixed incoming relevance vector R through $f _ { c } .$ . The propagation rule is zero-curvature consistent $i f$

$$
\operatorname* { l i m } _ { c  0 } \mathcal { R } _ { c } ( x , R ) = R\tag{6}
$$

## for every input x and incoming relevance R.

This definition includes both identity limits and fixed coordinate scalings. The origin Poincare´ maps approach the identity, whereas the aligned Lorentz spatial logarithmic and exponential maps approach scaling by $1 / 2$ and 2, respectively. These constant factors reflect the tangent-coordinate convention and receive no separate relevance.

## 4 RELEVANCE PROPAGATION RULES

In this section, we define relevance propagation rules for origin-centered hyperbolic modules. We introduce LRP-radial-all and contrast it with a conservative LRP-half baseline, showing that conservation alone does not ensure consistency across equivalent radial factorizations. For tangent-space linear layers, we use standard Euclidean LRP rules and their propagation formulas and conservation conditions are summarized in Appendix H. For Mobius bias addition, we adopt a separate signal-¨ only convention (see Appendix G).

## 4.1 CONSISTENT RELEVANCE PROPAGATION THROUGH GEOMETRIC MODULATION

Conservation alone does not determine how relevance should be allocated between multiplicative branches. Symmetric splitting can depend on the grouping of multiplication operations, while grouping-invariant symmetric rules require additional assumptions and may have restricted domains (Appendix B). For our geometric modules, we instead distinguish the feature signal from its scalar modulation and adopt an all-to-signal allocation.

Geometry as a modulation branch. For a radial map ${ \pmb y } = \alpha ( { \pmb x } ) ;$ x, the two multiplicative branches have different roles: x carries the feature coordinates, while α(x) applies a shared geometric scaling. We adopt a signal-based attribution convention that preserves the incoming feature-wise relevance allocation across this modulation. This choice is motivated by two properties. First, splitting the geometric scaling into successive modulation factors should not change the relevance propagated along the designated signal path. Second, when an origin map approaches the identity in the zerocurvature limit, its propagation rule should approach identity propagation as well.

These requirements motivate an all-to-signal allocation, consistent with earlier treatments of sourcegate interactions (Arras et al., 2017) and attention-value products (Ali et al., 2022).

Definition 3 (LRP-radial-all). For a scalar-signal module ${ \pmb y } = \alpha ( { \pmb x } ) s$ , LRP-radial-all assigns all incoming relevance to the designated signal s and none to the modulation factor:

$$
R _ { \alpha } = 0 , \qquad R _ { s } = R _ { y } .\tag{7}
$$

For a radial map ofa single vector, $\scriptstyle { s = x }$ and α depends only on its radius.

Proposition 1 (Conservation and signal-path factorization invariance). Considerfunctionally equivalent scalar-signal realizations that differ only in the splitting or grouping of scalar modulation factors along a designated signal path. Suppose the signal dimension and coordinate order are preserved, and the incoming relevance is identical. Applying the all-to-signal rule at every multiplication yields $R _ { x } = R _ { y } ,$ , independently of the number or grouping of modulation factors, and conserves signed total relevance.

Proof. Each multiplication has backward operator $I _ { d }$ on its designated signal and assigns zero rel evance to its modulation branch. A chain of $m \geq 1$ such modules therefore gives $R _ { x } = I _ { d } ^ { m } R _ { y } =$ $R _ { y }$ , and consequently $\mathbf { 1 } ^ { \top } R _ { x } = \mathbf { 1 } ^ { \top } R _ { y }$ □

The proposition provides the local consistency used in our controlled GRI test. More generally, necessary conditions for GRI and zero-curvature consistency can be derived (see Appendix C) for the fixed-proportion radial rules

$$
\begin{array} { r } { R _ { x _ { i } } = \eta R _ { y _ { i } } + ( 1 - \eta ) \frac { x _ { i } ^ { 2 } } { \left. x \right. _ { 2 } ^ { 2 } } \sum _ { j } R _ { y _ { j } } , \qquad \eta \in [ 0 , 1 ] , } \end{array}\tag{8}
$$

where the same constant η is used at every module and the modulation relevance is redistributed through the squared norm. This family includes LRP-half at $\eta = 1 / 2$ and LRP-radial-all at $\eta = 1$ For $\bar { d \geq 2 }$ , invariance under equivalent nonzero radial factorizations permits only $\eta = 0 \mathrm { o r } \eta = 1$ Requiring zero-curvature consistency as defined in Definition 2 further selects $\eta = 1$ . Thus, within this specified family, radial-all is the unique rule satisfying both criteria.

## 4.2 LRP-HALF AS AN ALTERNATIVE

Motivated by equal-split rules for multiplicative interactions (Arras et al., 2019; Achtibat et al., 2024), we consider an LRP-half baseline that assigns half of the relevance to the signal and half

![](images/44bd3b3e8f37f5528a8bc2c1083273b8445ea486a20c1d530c0c779b9b1f3a2a.jpg)  
Figure 2: Top: factorization of log and exp map with origin point in Poincare space. Bottom: our´ proposed radial-all rule and half rule for LRP applied to such layer.

to the radial factor, redistributing the latter through the squared norm (Figures 2 and 7). In the logarithmic-map test of Eq. 4, the direct Poincare path contains one radial module, whereas the ´ equivalent Lorentz path contains two. For $\mathbf { \Delta } { \pmb x } _ { P } \neq { \bf 0 }$ , their backward allocations are

$$
\begin{array} { r } { { \cal R } _ { x _ { P , i } } ^ { P } = \frac 1 2 R _ { v _ { i } } + \frac 1 2 \frac { x _ { P , i } ^ { 2 } } { \| { \boldsymbol x } _ { P } \| _ { 2 } ^ { 2 } } \sum _ { j } R _ { v _ { j } } , \qquad { \cal R } _ { x _ { P , i } } ^ { L } = \frac 1 4 R _ { v _ { i } } + \frac 3 4 \frac { x _ { P , i } ^ { 2 } } { \| { \boldsymbol x } _ { P } \| _ { 2 } ^ { 2 } } \sum _ { j } R _ { v _ { j } } . } \end{array}\tag{9}
$$

Both conserve the incoming total relevance, but generally disagree coordinate-wise. With an identity encoder, this difference directly violates GRI at the common original input. Appendix F provides the derivation and an explicit counterexample. Appendix I reports a numerical test.

Failure of zero-curvature consistency. Although $\log _ { \mathbf { 0 } } ^ { P , c } ( { \pmb x } _ { P } )  { \pmb x } _ { P } \mathrm { a s } c  0 ,$ Eq. 9 is independent of curvature for fixed input and incoming relevance, and generally differs from identity propagation. Thus, LRP-half also fails zero-curvature consistency and the same argument also applies to the origin exponential map. LRP-radial-all, by contrast, has identity backward action for every curvature. Conservation alone therefore guarantees neither of the two consistency criteria.

## 5 EXPERIMENTS

We compare attribution methods on hyperbolic classifiers for MNIST, sEEG, and CIFAR-10 using sparsity-fidelity metrics and qualitative visualizations. LRP-radial-all achieves competitive attribution fidelity. Runtime comparisons further show that LRP-radial-all is comparable in efficiency to Gradient×Input and substantially faster than Integrated Gradients. We also present a controlled GRI numerical test in Appendix I.

## 5.1 MNIST

Model and training. We use a Poincare hyperbolic MLP with dimensions´ 784-128-10, fixed curvature −1, Mobius biases, and a tangent-space ReLU between the two hyperbolic linear layers. Images¨ have pixel values in [0, 1] and flattened inputs are scaled by 0.1 and mapped to the manifold using the origin exponential map. An origin logarithmic map produces the output logits. The training configuration uses 55000 training and 5000 validation images, Adam with learning rate $1 0 ^ { - 3 }$ , batch size 128 and cross-entropy loss. Training runs for 100 epochs, with checkpoint selection by validation accuracy. The model has 97.44% test accuracy.

Explanation methods. All quantitative comparisons explain the original predicted-class logit of the model. We compare radial-all and the half rule for hyperbolic mapping layers using identical linear $\mathrm { L R P - } \gamma$ rules $( \gamma = 0 . 2 5 , \epsilon = 1 0 ^ { - 9 } )$ for other layers. Fixed bias branches receive zero relevance. Baselines are Gradient×Input, Integrated Gradients (IG), Patch Occlusion, and random. IG uses a black-image baseline and 512 midpoint integration steps along a straight path in pixel space. Patch Occlusion uses black $4 \times 4$ windows with stride 4, assigning the target-logit drop to each pixel in the window.

Sparsity-fidelity protocol. We evaluate randomly sampled 1000 images from test set. In one image, pixels are ranked by their signed attribution scores in descending order. Let $S _ { k }$ contain the top k pixels and let $m _ { S _ { k } }$ be their binary pixel mask. For the fixed original predicted class $t ,$ we define

$$
\mathrm { S p a r s i t y } ( k ) = 1 - k / K , \qquad s _ { t } ( x ) = \mathrm { s o f t m a x } ( f ( x ) ) _ { t } ,\tag{10}
$$

$$
\mathrm { F i d e l i t y } ^ { + } ( k ) = s _ { t } ( x ) - s _ { t } ( x \odot ( 1 - m _ { S _ { k } } ) ) , \quad \mathrm { F i d e l i t y } ^ { - } ( k ) = s _ { t } ( x ) - s _ { t } ( x \odot m _ { S _ { k } } ) ,\tag{11}
$$

Table 1: Sparsity-fidelity AUC on MNIST, sEEG and CIFAR-10. $F ^ { + }$ and $F ^ { - }$ denote $\mathrm { F i d e l i t y ^ { + } }$ and Fidelity<sup>−</sup> AUC. Subscripts report ±h, where $h = \operatorname* { m a x } ( \mu - L , U - \mu )$ for the 95% bootstrap interval $[ L , U ] .$ Bold indicates the best mean within each dataset and metric.
<table><tr><td rowspan="2">Method</td><td colspan="2">MNIST</td><td colspan="2">sEEG</td><td colspan="2">CIFAR-10</td></tr><tr><td> $F ^ { + } \uparrow$ </td><td> $F ^ { - } \downarrow$ </td><td> $F ^ { + } \uparrow$ </td><td> $F ^ { - } \downarrow$ </td><td> $F ^ { + } \uparrow$ </td><td> $F ^ { - } \downarrow$ </td></tr><tr><td> $\mathrm { L R P } _ { \mathrm { r a d i a l - a l l } }$ </td><td> $\mathbf { 0 . 6 7 0 { \scriptstyle \pm 0 . 0 0 5 } }$ </td><td> $\mathbf { - 0 . 0 0 3 _ { \pm 0 . 0 0 5 } }$ </td><td> $\mathbf { 0 . 3 1 3 _ { \pm 0 . 0 9 6 } }$ </td><td> $- 0 . 1 1 1 _ { \pm 0 . 0 1 7 }$ </td><td> $0 . 6 3 7 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $\mathbf { 0 . 2 7 7 _ { \pm 0 . 0 1 9 } }$ </td></tr><tr><td> $\mathrm { L R P _ { h a l f } }$ </td><td> $0 . 5 6 2 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 0 6 3 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 3 1 0 { \scriptstyle \pm 0 . 0 9 5 }$ </td><td> $- 0 . 1 0 6 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $0 . 5 8 8 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 5 0 5 { \scriptstyle \pm 0 . 0 2 2 }$ </td></tr><tr><td>Gradient×Input</td><td> $\overline { { 0 . 4 1 0 _ { \pm 0 . 0 1 7 } } }$ </td><td> $\overline { { 0 . 2 0 2 _ { \pm 0 . 0 1 4 } } }$ </td><td> $\overline { { 0 . 3 0 3 _ { \pm 0 . 0 9 6 } } }$ </td><td> $\mathbf { \overline { { - 0 . 1 1 3 _ { \pm 0 . 0 1 7 } } } }$ </td><td> $\overline { { \mathbf { 0 . 7 0 3 _ { \pm 0 . 0 2 0 } } } }$ </td><td> $\overline { { 0 . 5 4 7 _ { \pm 0 . 0 2 3 } } }$ </td></tr><tr><td>Integrated Grad.</td><td> $0 . 6 6 4 _ { \pm 0 . 0 0 5 }$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 3 0 9 { \scriptstyle \pm 0 . 0 9 6 }$ </td><td> $\mathbf { - 0 . 1 1 3 _ { \pm 0 . 0 1 7 } }$ </td><td> $0 . 6 8 3 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $0 . 4 9 5 { \scriptstyle \pm 0 . 0 2 2 }$ </td></tr><tr><td>Patch Occlusion</td><td> $0 . 5 0 4 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 0 6 7 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>Random</td><td> $0 . 1 9 3 _ { \pm 0 . 0 0 5 }$ </td><td> $0 . 1 9 3 _ { \pm 0 . 0 0 5 }$ </td><td> $0 . 0 5 7 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td> $0 . 0 5 8 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 6 6 4 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 6 6 3 _ { \pm 0 . 0 2 2 }$ </td></tr></table>

Table 2: Runtime comparison of explanation methods. We average the runtime on 3 runs.
<table><tr><td>Dataset</td><td>LRP-radial-all</td><td>Gradient×Input</td><td>Occlusion</td><td>Integrated Gradients</td></tr><tr><td>MNIST (ms/image)</td><td>0.505</td><td>0.960</td><td>12.676</td><td>41.511</td></tr><tr><td>sEEG (ms/sample)</td><td>5.236</td><td>3.813</td><td></td><td>94.431</td></tr><tr><td>CIFAR-10 (ms/image)</td><td>15.688</td><td>12.730</td><td></td><td>66.375</td></tr></table>

where K means the total number of pixels. Higher $\mathrm { F i d e l i t y ^ { + } }$ indicates greater necessity of selected regions and lower Fidelity<sup>−</sup> indicates greater sufficiency. We use 50 regions in sparsity, and compute per-image AUC by integration over actual sparsity values. Random rankings are averaged over 10 repetitions per image. We report mean AUCs and percentile 95% confidence intervals from 1000 image-level bootstrap resamples.

Results. As shown in Table 1 and Figure 9, LRP-radial-all achieves the highest $\mathrm { F i d e l i t y ^ { + } }$ AUC and the lowest Fidelity<sup>−</sup> AUC among the evaluated methods. Although the improvement from IG is marginal, LRP-radial-all is substantially faster than IG as in Table 2. Figure 3 shows two example heatmaps by our method and baselines. LRP-radial-all highlights the important curves and intersections in the number pictures. Integrated Gradients can capture the important parts as well, but the heatmaps appear noisier.

## 5.2 SEEG

sEEG captures interactions across electrodes and temporal scales, giving rise to hierarchical structure that can be naturally represented in hyperbolic space. Recent hyperbolic approaches to sEEG modeling have achieved state-of-the-art performance and substantially outperformed their Euclidean counterparts (Guillemaud et al., 2025; Li et al., 2026). Following Li et al. (2026), we evaluate on the working-memory sEEG dataset introduced by Boran et al. (2020). The dataset contains 9 subjects and many trials for each subject. Each trial is represented as a multichannel time series $\mathbf { X } \in \mathbb { R } ^ { \check { C } \times T }$ where C denotes the number of electrodes and T the number of temporal samples, and is associated with a binary label indicating whether the working-memory load is high or low. Following Li et al. (2026), we trained a variant of EEGNet in which the multi-layer perceptron is replaced with its hyperbolic counterpart. On each subject we trained the model using 5-fold cross-validation.

Qualitative results. We take subject 2 as an example and plot the signals and LRP-radial-all heatmaps averaged on all samples of it by the label in Figure 4. In the low-load condition, a prominent positive relevance region is concentrated within a relatively limited subset of channels, indicating a localized contribution to the model prediction. In contrast, the high-load condition exhibits more distributed relevance across multiple channels during the middle portion of the time window. Stronger relevance concentrations also emerge toward the later portion of the high-load window, indicating a greater contribution from these temporal segments. We validate the importance of the relevant time windows to the prediction in Appendix J. In addition, matched upper-channel regions show opposite relevance polarities between the two conditions, indicating a class-dependent difference in how the model uses the same spatiotemporal region. The annotated boxes highlight these representative attribution patterns. The more distributed relevance observed under the high-load condition is consistent with recent intracranial evidence (Yang et al., 2025) showing increased inter regional information sharing as working-memory demands increase.

![](images/4863788b6237f1173c7f400882709c83e705498f56bc049cb6174629ce193671.jpg)  
Figure 3: Heatmaps on images of MNIST.

![](images/1b3c6d06b3da410d3eb1808fa53cbdd43cd0db1f2e9edba1cac567e75599d41f.jpg)  
Figure 4: Averaged signals and relevance heatmaps for class 0 (low working-memory load) and class 1 (high working-memory load). Boxes highlight localized relevance, distributed multichannel relevance, stronger late-window relevance, and class-dependent attribution differences. Each time series stands for a sensor. Time range is 3-second with 3000 points. Red and blue indicate positive and negative relevance for the explained class score, respectively. The full figure is in Figure 11.

Applying DFT-LRP (Vielhaben et al., 2024), we propagate relevance to the frequency space and visualize the relevance in Figure 5. The results show that frequency-domain relevance is mainly concentrated in the lower-frequency range, particularly the delta band (0.5 - 4 Hz). Besides, the magnitude of delta-band relevance differs between the two working-memory load classes. These observations suggest the classifier mainly exploits low-frequency activity, with such activity contributing differently to the decision of the model under two working-memory load conditions.

Quantitative evaluation. We evaluate sensor-level attribution fidelity using all trials from all subjects in the dataset. We follow the setting in (Li et al., 2026) and use subject-specific HEEGNet0 checkpoints from stratified five-fold within-subject cross-validation. For an input $\boldsymbol { x } \in \mathbb { R } ^ { C \times T }$ , we fix the target class to the model’s prediction and compute attributions for the target logit. The relevance of sensor c is $\begin{array} { r } { R _ { c } = \sum _ { t = 1 } ^ { T } R _ { c , t } } \end{array}$ . Sensors are ranked in descending order of $R _ { c }$ . We compare LRP-radial-all, LRP-half, Gradient×Input, Integrated Gradients (IG), and random rankings. Both LRP variants use the ϵ-rule with $\epsilon = 1 \dot { 0 } ^ { - 9 }$ for layers apart from hyperbolic layers. IG uses a zero baseline and integration over 128 steps. The random baseline averages five random rankings.

![](images/f19c6667a39920e7b747ecd36c2105e7edaa6505a7c87d0a0f0ce3ae696b4255.jpg)

![](images/8613461b101bdd7478db3398eb616ca289e9480c942cc71acc8f8cd9fe1c4bc3.jpg)  
Figure 5: Relevance of sEEG prediction by class in frequency space. The relevance score is averaged on all trials of subject 2 in the dataset. We grouped the frequency-domain relevance into conventional electrophysiological frequency bands: delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–70 Hz).

![](images/8490a083e6c58f4e25da8cf9d09f244251d29660c13e1fbfb61ad433ff52aa8a.jpg)  
Figure 6: Qualitative attribution comparison on a correctly classified CIFAR-10 image of ship. Columns show the input, heatmaps for LRP-radial-all, LRP-half, Gradient×Input, and Integrated Gradients. RGB-channel attributions are summed, with red and blue indicating positive and negative values, respectively. More examples in Figure 10.

We replace perturbed sensors with the timepoint-wise mean across all sensors in the original trial. We perturb the input by sparsity from 0% to 100% in 5% increments, and calculate the fidelity as defined in Eq.10,11, as well as the area under the sparsity-fidelity curves. We average trial-level AUCs across the held-out folds within each subject and then average the nine subject means, and the results are summarized in Table 1. LRP-radial-all achieves the best average AUC of Fidelity<sup>+</sup> and comparable AUC of Fidelity<sup>−</sup> to the best baseline.

## 5.3 CIFAR-10

We evaluate our relevance propagation method on CIFAR-10 using a lightweight classifier with six Lorentz convolutional layers inspired by Bdeir et al. (2024). The model architecture and training details are in Appendix K. We assess the resulting explanations through qualitative heatmaps and quantitative fidelity and sparsity metrics.

Qualitative results. Figure 6 compares explanations for a CIFAR-10 image of ship. With the same z<sup>B</sup>-γ-ϵ configuration (Montavon et al., 2017; Samek et al., 2019) for layers apart from hyperbolic layers, LRP-radial-all produces broader, spatially coherent attribution patterns that overlap visible object regions, whereas LRP-half concentrates attribution into smaller regions with positive and negative contributions. Gradient×Input and Integrated Gradients exhibit much noisier patterns. These examples illustrate how the radial propagation rule affects attribution structure.

Quantitative evaluation. We evaluate attribution fidelity on 512 randomly sampled CIFAR-10 test images using pixel-wise mean-color replacement and predicted-class probabilities. As shown in Table 1, LRP-radial-all achieves the lowest Fidelity<sup>−</sup> AUC, outperforming other methods, indicating stronger prediction preservation when retaining highly ranked pixels. However, its Fidelity<sup>+</sup> AUC falls even below Random. These results indicate stronger retention sufficiency but weaker deletion sensitivity under this perturbation protocol.

## 5.4 RUNTIME EVALUATION

We compare attribution runtimes for MNIST on an AMD EPYC 7453, sEEG and CIFAR-10 on a NVIDIA A100. As in Table 2, LRP-radial-all incurs substantially lower computational cost than Integrated Gradients, and comparable with Gradient×Input that only includes one forward and backward computation of the model.

## 6 CONCLUSION

We developed a relevance propagation framework for origin-centered hyperbolic operations, guided by Geometric Representation Invariance and zero-curvature consistency. Our analysis shows that relevance conservation alone does not ensure either property, as the specified LRP-half baseline dependent on equivalent radial factorizations fails to approach identity propagation as curvature vanishes. By treating geometric scaling as modulation, LRP-radial-all conserves relevance locally, preserves the incoming allocation across radial factorizations, and satisfies zero-curvature consistency. These properties guarantee GRI for the specified Poincare–Lorentz logarithmic-map construction.´ Experiments on MNIST, sEEG, and CIFAR-10 demonstrate competitive attribution fidelity with runtime comparable to Gradient×Input and substantially lower than Integrated Gradients, while revealing a trade-off between retention and deletion performance on CIFAR-10.

Limitations and future work. Our theoretical guarantees apply to the specified module realizations and propagation conventions. The controlled exponential map test additionally requires a treatment of time-coordinate relevance, and invariance under alternative treatments remains an open question. Future work includes extending the framework to non-origin base points and structured explanations of message-passing paths and higher-order interactions as in Xiong et al. (2026).

## ACKNOWLEDGMENTS

This work was funded by the German Ministry for Education and Research as BIFOLD - Berlin Institute for the Foundations of Learning and Data (ref. BIFOLD25B). Thomas Schnake is a postdoctoral fellow at the University of Toronto in the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship Program, a program of Schmidt Sciences.

## REFERENCES

Reduan Achtibat, Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, Aakriti Jain, Thomas Wiegand, Sebastian Lapuschkin, and Wojciech Samek. Attnlrp: Attention-aware layer-wise relevance propagation for transformers. In ICML, pp. 135–168. PMLR / OpenReview.net, 2024.

Ameen Ali, Thomas Schnake, Oliver Eberle, Gregoire Montavon, Klaus-Robert M´ uller, and Lior¨ Wolf. XAI for transformers: Better explanations through conservative propagation. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162, pp. 435–451. PMLR, 2022.

Marco Ancona, Enea Ceolini, Cengiz Oztireli, and Markus Gross. Towards better understanding of<sup>¨</sup> gradient-based attribution methods for deep neural networks. In ICLR (Poster). OpenReview.net, 2018.

Leila Arras, Gregoire Montavon, Klaus-Robert M´ uller, and Wojciech Samek. Explaining recurrent¨ neural network predictions in sentiment analysis. In WASSA@EMNLP, pp. 159–168. Association for Computational Linguistics, 2017.

Leila Arras, Jose A. Arjona-Medina, Michael Widrich, Gregoire Montavon, Michael Gillhofer,´ Klaus-Robert Muller, Sepp Hochreiter, and Wojciech Samek. Explaining and interpreting LSTMs.¨ In Explainable AI: Interpreting, Explaining and Visualizing Deep Learning, volume 11700, pp. 211–238. Springer, 2019.

Sebastian Bach, Alexander Binder, Gregoire Montavon, Frederick Klauschen, Klaus-Robert M´ uller,¨ and Wojciech Samek. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. PloS one, 10(7):e0130140, 2015.

Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr. Fully hyperbolic convolutional neural networks for computer vision. In ICLR. OpenReview.net, 2024.

Ece Boran, Tommaso Fedele, Adrian Steiner, Peter Hilfiker, Lennart Stieglitz, Thomas Grunwald, and Johannes Sarnthein. Dataset of human medial temporal lobe neurons, scalp and intracranial eeg during a verbal working memory task. Scientific data, 7(1):30, 2020. doi: 10.1038/s41597-020-0364-3.

James W Cannon, William J Floyd, Richard Kenyon, and Walter R Parry. Hyperbolic geometry. Flavors ofgeometry, 31:59–115, 1997.

Karan Desai, Maximilian Nickel, Tanmay Rajpurohit, Justin Johnson, and Shanmukha Ramakrishna Vedantam. Hyperbolic image-text representations. In International Conference on Machine Learning, pp. 7694–7731. PMLR, 2023.

Ann-Kathrin Dombrowski, Jan E. Gerken, Klaus-Robert Muller, and Pan Kessel. Diffeomorphic¨ counterfactuals with generative models. IEEE Trans. Pattern Anal. Mach. Intell., 46(5):3257– 3274, 2024.

Octavian-Eugen Ganea, Gary Becigneul, and Thomas Hofmann. Hyperbolic neural networks. In´ NeurIPS, pp. 5350–5360, 2018.

Martin Guillemaud, Louis Cousyn, Vincent Navarro, and Mario Chavez. Hyperbolic embedding of brain networks as a tool for epileptic seizures forecasting. Physical Review Research, 7(2): 023182, 2025.

David Gunning, Mark Stefik, Jaesik Choi, Timothy Miller, Simone Stumpf, and Guang-Zhong Yang. Xai—explainable artificial intelligence. Science Robotics, 4(37), 2019.

Neil He, Hiren Madhu, Ngoc Bui, Menglin Yang, and Rex Ying. Hyperbolic deep learning for foundation models: A survey. In KDD (2), pp. 6021–6031. ACM, 2025.

Neil He, Rishabh Anand, Hiren Madhu, Ali Maatouk, Smita Krishnaswamy, Leandros Tassiulas, Menglin Yang, and Rex Ying. Helm: Hyperbolic large language models via mixture-of-curvature experts. Advances in Neural Information Processing Systems, 38:142604–142635, 2026.

Andreas Holzinger, Anna Saranti, Christoph Molnar, Przemyslaw Biecek, and Wojciech Samek. Explainable AI methods - A brief overview. In xxAI@ICML, pp. 13–38. Springer, 2020.

Valentin Khrulkov, Leyla Mirvakhabova, Evgeniya Ustinova, Ivan Oseledets, and Victor Lempitsky. Hyperbolic image embeddings. In 2020 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 6417–6427. IEEE, 2020.

Dmitri Krioukov, Fragkiskos Papadopoulos, Maksim Kitsak, Amin Vahdat, and Marian Bogun´ a.´ Hyperbolic geometry of complex networks. Physical Review E—Statistical, Nonlinear, and Soft Matter Physics, 82(3):036106, 2010. doi: 10.1103/PhysRevE.82.036106.

Shanglin Li, Chu Shiwen, Okan Koc¸, Yi Ding, Qibin Zhao, Motoaki Kawanabe, and Ziheng Chen. HEEGNet: Hyperbolic embeddings for EEG. In The Fourteenth International Conference on Learning Representations, 2026.

Qi Liu, Maximilian Nickel, and Douwe Kiela. Hyperbolic graph neural networks. Advances in neural information processing systems, 32, 2019.

Pascal Mettes, Mina Ghadimi Atigh, Martin Keller-Ressel, Jeffrey Gu, and Serena Yeung. Hyperbolic deep learning in computer vision: A survey. International Journal of Computer Vision, 132 (9):3484–3508, 2024. doi: 10.1007/s11263-024-02043-5.

Gregoire Montavon, Sebastian Lapuschkin, Alexander Binder, Wojciech Samek, and Klaus-Robert´ Muller. Explaining nonlinear classification decisions with deep taylor decomposition. ¨ Pattern Recognit., 65:211–222, 2017.

Maximilian Nickel and Douwe Kiela. Learning continuous hierarchies in the lorentz model of hyperbolic geometry. In ICML, volume 80 of Proceedings of Machine Learning Research, pp. 3776–3785. PMLR, 2018.

Wei Peng, Tuomas Varanka, Abdelrahman Mostafa, Henglin Shi, and Guoying Zhao. Hyperbolic deep neural networks: A survey. IEEE Transactions on pattern analysis and machine intelligence, 44(12):10023–10044, 2021. doi: 10.1109/TPAMI.2021.3136921.

Wojciech Samek, Gregoire Montavon, Andrea Vedaldi, Lars Kai Hansen, and Klaus-Robert M´ uller¨ (eds.). Explainable AI: Interpreting, Explaining and Visualizing Deep Learning, volume 11700 of Lecture Notes in Computer Science. Springer, 2019.

Wojciech Samek, Gregoire Montavon, Sebastian Lapuschkin, Christopher J. Anders, and Klaus-´ Robert Muller. Explaining deep neural networks and beyond: A review of methods and applica-¨ tions. Proc. IEEE, 109(3):247–278, 2021.

Ryohei Shimizu, Yusuke Mukuta, and Tatsuya Harada. Hyperbolic neural networks++. In ICLR. OpenReview.net, 2021.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In ICML, volume 70 of Proceedings ofMachine Learning Research, pp. 3319–3328. PMLR, 2017.

Alexandru Tifrea, Gary Becigneul, and Octavian-Eugen Ganea. Poincar´ e glove: Hyperbolic word´ embeddings. In ICLR (Poster). OpenReview.net, 2019.

Johanna Vielhaben, Sebastian Lapuschkin, Gregoire Montavon, and Wojciech Samek. Explainable´ AI for time series via virtual inspection layers. Pattern Recognit., 150:110309, 2024.

Ping Xiong, Thomas Schnake, Gregoire Montavon, Klaus-Robert M´ uller, and Shinichi Nakajima.¨ Normalized relevance measure as a unifying framework to explain neural network latent structures. CoRR, abs/2606.00557, 2026.

Jiayi Yang, Dan Cao, Chunyan Guo, Lennart Stieglitz, Debora Ledergerber, Johannes Sarnthein, and Jin Li. Enhanced role of the entorhinal cortex in adapting to increased working memory load. Nature Communications, 16(1):5798, Jul 2025. ISSN 2041-1723.

Eslam Zaher, Maciej Trzaskowski, Quan Nguyen, and Fred Roosta. Manifold integrated gradients: Riemannian geometry for feature attribution. In ICML, volume 235 of Proceedings of Machine Learning Research, pp. 58090–58104. PMLR / OpenReview.net, 2024.

Runhe Zhou, Shanglin Li, Guanxiang Huang, Xinliang Zhou, Qibin Zhao, Motoaki Kawanabe, Yi Ding, and Cuntai Guan. Eeg-based multimodal learning via hyperbolic mixture-of-curvature experts. International Conference on Machine Learning, 2026.

## A POINCARE AND´ LORENTZ REPRESENTATIONS

For curvature −c with $c > 0$ , the Poincare ball and Lorentz hyperboloid are´

$$
\mathbb { D } _ { c } ^ { d } = \{ \pmb { x } _ { P } \in \mathbb { R } ^ { d } : c \| \pmb { x } _ { P } \| _ { 2 } ^ { 2 } < 1 \} ,\tag{12}
$$

$$
\mathbb { L } _ { c } ^ { d } = \{ ( x _ { L , 0 } , \bar { x } _ { L } ) \in \mathbb { R } ^ { d + 1 } : - x _ { L , 0 } ^ { 2 } + \| \bar { x } _ { L } \| _ { 2 } ^ { 2 } = - 1 / c , \ x _ { L , 0 } > 0 \} .\tag{13}
$$

Their origins are ${ \bf o } _ { P } = { \bf 0 }$ and ${ \pmb { o } } _ { L } = ( c ^ { - 1 / 2 } , \mathbf { 0 } )$ . The standard Poincare-Lorentz isometry and its´ inverse (Nickel & Kiela, 2018), rescaled here to curvature −c, are

$$
\phi ( \pmb { x } _ { P } ) = \left( \frac { 1 + c \| \pmb { x } _ { P } \| _ { 2 } ^ { 2 } } { \sqrt { c } ( 1 - c \| \pmb { x } _ { P } \| _ { 2 } ^ { 2 } ) } , \frac { 2 \pmb { x } _ { P } } { 1 - c \| \pmb { x } _ { P } \| _ { 2 } ^ { 2 } } \right) ,\tag{14}
$$

$$
\phi ^ { - 1 } ( \pmb { x } _ { L } ) = \frac { \bar { \pmb { x } } _ { L } } { \sqrt { c } \ b { x } _ { L , 0 } + 1 } .\tag{15}
$$

Since $x _ { L , 0 } = \sqrt { c ^ { - 1 } + \| \bar { \mathbf { x } } _ { L } \| _ { 2 } ^ { 2 } }$ , both spatial conversions are radial:

$$
\bar { \ b x } _ { L } = \gamma ( { \pmb x } _ { P } ) { \pmb x } _ { P } , \qquad \gamma ( { \pmb x } _ { P } ) = \frac { 2 } { 1 - c \| { \pmb x } _ { P } \| _ { 2 } ^ { 2 } } , \qquad { \pmb x } _ { P } = \frac { \bar { \ b x } _ { L } } { \sqrt { 1 + c \| \bar { \ b x } _ { L } \| _ { 2 } ^ { 2 } } + 1 } .\tag{16}
$$

## B MULTIPLICATIVE ATTRIBUTION AND GROUPING INVARIANCE

Origin-centered hyperbolic maps and spatial coordinate conversions admit a scalar–signal structure, as in Eq. 4. This raises an attribution question: should relevance be shared between the geometric factor and the signal, or assigned entirely to the signal? Conservation alone does not determine this choice. To clarify the role of multiplicative structure, we first study attribution to scalar factors with equal explanatory status. This analysis complements the signal-only formulation used by LRPradial-all.

## B.1 LOCAL RULES AND GROUPING INVARIANCE

For a scalar product $y = a b$ with $a , b > 1$ , consider a local rule that is linear in incoming relevance and whose coefficients depend only on the current factor values. Every conservative rule in this class has the form

$$
R _ { a } = w ( a , b ) R _ { y } , \qquad R _ { b } = [ 1 - w ( a , b ) ] R _ { y } .\tag{17}
$$

The same coefficient function w is used at every multiplication. Symmetric treatment of the factors additionally requires $w ( a , b ) = 1 - w ( b , a )$

Definition 4 (Multiplicative grouping invariance). A local relevance rule is multiplicatively grouping-invariant if, for every fixed ordered list of factors and incoming relevance, all binary parenthesizations oftheir product yield identical relevance at each originalfactor.

This criterion depends on associative regrouping with the original explanatory factors held fixed. Equal splitting is symmetric and conservative but fails this criterion: for $y = a b c$ , the parenthesizations (ab)c and a(bc) assign relevance proportions $( 1 / 4 , 1 / 4 , 1 / 2 )$ and $( 1 / 2 , 1 / 4 , 1 / 4 )$ , respectively.

More generally, propagation through (ab)c assigns the coefficients

$$
\left( w ( a b , c ) w ( a , b ) , \ w ( a b , c ) [ 1 - w ( a , b ) ] , \ 1 - w ( a b , c ) \right)\tag{18}
$$

to $( R _ { a } , R _ { b } , R _ { c } )$ relative to $R _ { y }$ , whereas propagation through a(bc) assigns

$$
\begin{array} { r } { \big ( w ( a , b c ) , ~ [ 1 - w ( a , b c ) ] w ( b , c ) , ~ [ 1 - w ( a , b c ) ] [ 1 - w ( b , c ) ] \big ) . } \end{array}\tag{19}
$$

Grouping invariance therefore requires, for every $a , b , c > 1$

$$
w ( a b , c ) w ( a , b ) = w ( a , b c ) ,\tag{20}
$$

$$
w ( a b , c ) [ 1 - w ( a , b ) ] = [ 1 - w ( a , b c ) ] w ( b , c ) ,\tag{21}
$$

$$
1 - w ( a b , c ) = [ 1 - w ( a , b c ) ] [ 1 - w ( b , c ) ] .\tag{22}
$$

These identities are also sufficient: any two binary parenthesizations are connected by elementary reassociations, each of which preserves the relevance entering its three sub-expressions.

## B.2 CHARACTERIZATION OF SYMMETRIC FACTOR ATTRIBUTION

Theorem 1 (Continuous symmetric grouping-invariant attribution). Let $w : ( 1 , \infty ) ^ { 2 } $ R be continuous. The conservative local rule in $E q$ . 17 is symmetric and multiplicatively grouping-invariant if and only if

$$
R _ { a } = { \frac { \log a } { \log a + \log b } } R _ { y } , \qquad R _ { b } = { \frac { \log b } { \log a + \log b } } R _ { y } .\tag{23}
$$

Proof. Take incoming relevance $R _ { y } ~ = ~ 1$ and consider a product of N identical factors $t \ > \ 1$ Any adjacent pair can be made siblings by changing the parenthesization. Since symmetry gives $w ( t , t ) = 1 / 2$ , the two factors receive equal relevance in that parenthesization. Grouping invariance transfers this equality to every parenthesization. All adjacent factors therefore receive equal relevance, and conservation implies an allocation of $1 / N$ to each factor.

Now consider $m + n$ identical factors, where $m ,$ n are positive integers, and choose a tree whose root separates the first m factors from the remaining n. Because the rule depends only on the two current factor values, the root assigns relevance $w ( t ^ { \bar { m } } , t ^ { n } )$ to the first subtree. Conservation within this subtree gives

$$
w ( t ^ { m } , t ^ { n } ) = \frac { m } { m + n } .\tag{24}
$$

Consequently, whenever log a/ log $b = m / n .$ , choosing $t = a ^ { 1 / m } = b ^ { 1 / n }$ yields

$$
w ( a , b ) = { \frac { \log a } { \log a + \log b } } .\tag{25}
$$

This argument uses grouping invariance on the fixed list of identical factors and locality at the root and does not require a separate assumption of invariance under factor expansion.

For arbitrary $a , b > 1$ , choose positive rational numbers $q _ { r } \to \log a / \log b .$ . Since $b ^ { q _ { r } }  a$ , continuity extends the identity $w ( b ^ { q _ { r } } , \bar { b } ) = q _ { r } / ( 1 + q _ { r } ) \mathrm { t o } w ( a , \bar { b } ) = \log a / ( \log a + \log b )$

Conversely, the logarithmic rule is continuous, symmetric, and conservative on the stated domain. For any product $\begin{array} { r } { y = \prod _ { i = 1 } ^ { N } a _ { i } . } \end{array}$ , propagation through an arbitrary binary tree yields

$$
R _ { a _ { i } } = \frac { \log a _ { i } } { \sum _ { j = 1 } ^ { N } \log a _ { j } } R _ { y } ,\tag{26}
$$

because intermediate logarithmic sums cancel along each root-to-leaf path. The allocation is therefore independent of parenthesization. □

## B.3 CONNECTION TO RADIAL ATTRIBUTION

This characterization applies to continuous, symmetric, value-based rules on $( 1 , \infty ) ^ { 2 }$ . The logarithmic formula does not directly cover general neural activations, since it is undefined for nonpositive factors and singular when positive factors have product one. Our geometric modules instead distinguish a vector signal from its scalar modulation, motivating the asymmetric LRP-radial-all convention. Its invariance under splitting or merging radial modules differs from regrouping a fixed list of explanatory factors. Appendix C analyzes this setting within the fixed-proportion radial family.

## C CHARACTERIZATION OF FIXED-PROPORTION RADIAL RULES

## C.1 RULE FAMILY AND REPEATED PROPAGATION

Consider nonzero radial modules ${ \pmb y } = \alpha ( \| { \pmb x } \| _ { 2 } ) { \pmb x }$ in dimension $d \geq 2$ , with nonzero intermediate signals. We restrict attention to the rule family in Eq. 8, where $\eta \in [ 0 , 1 ]$ is a fixed constant shared across modules. The signal branch receives proportion η of each output relevance, while the scalar branch receives the remaining total relevance and redistributes it according to squared input coordinates.

For compactness, define the matrix

$$
{ P } _ { \pmb { x } } = \frac { \pmb { x } \odot \pmb { x } } { \| \pmb { x } \| _ { 2 } ^ { 2 } } \mathbf { 1 } ^ { \top } , \qquad T _ { \eta , \pmb { x } } = \eta I _ { d } + ( 1 - \eta ) { P } _ { \pmb { x } } ,\tag{27}
$$

where $\odot$ denotes element-wise multiplication. The rule is $R _ { x } = T _ { \eta , x } R _ { y }$ . Since the normalized squared coordinates sum to one,

$$
P _ { \mathbf { x } } ^ { 2 } = P _ { \mathbf { x } } , \qquad \mathbf { 1 } ^ { \top } P _ { \mathbf { x } } = \mathbf { 1 } ^ { \top } .\tag{28}
$$

Furthermore, $P _ { a x } = P _ { x }$ for every nonzero scalar a. Thus every module along a nonzero radial chain has the same squared-coordinate redistribution matrix.

Proposition 2 (Factorization invariance within the fixed-proportion family). For $d \geq 2 ,$ a rule in Eq. 8 is invariant, for every nonzero input and incoming relevance, under replacement of one radial module by an equivalent chain ofnonzero radial modules ifand only $i f \eta \in \{ 0 , 1 \}$

Proof. Since $P _ { x }$ is idempotent,

$$
T _ { \eta , { \pmb x } } ^ { m } = \eta ^ { m } I _ { d } + ( 1 - \eta ^ { m } ) P _ { \pmb x }\tag{29}
$$

for every $m \geq 1$ . In particular, invariance between one and two modules requires

$$
T _ { \eta , x } ^ { 2 } - T _ { \eta , x } = ( \eta ^ { 2 } - \eta ) ( I _ { d } - P _ { x } ) = 0 .\tag{30}
$$

For $d \ge 2 , P _ { x }$ has rank one and cannot equal $I _ { d } .$ . Therefore $\eta ^ { 2 } = \eta ,$ , giving $\eta = 0$ or $\eta = 1$ Conversely, both choices yield idempotent backward operators, so repeated application does not change the propagated relevance. □

The case $\eta = 1$ is radial-all. The case $\eta = 0$ assigns all relevance to the scalar branch and then redistributes it according to squared input coordinates. Both rules conserve relevance and satisfy the stated radial factorization invariance, demonstrating that these two properties alone do not uniquely identify radial-all.

## C.2 ADDING ZERO-CURVATURE CONSISTENCY

Corollary 1 (Unique joint consistency within the fixed-proportion family). For $d \geq 2 ,$ , LRP-radialall is the unique member ofthe fixed-proportionfamily in $E q .$ . 8 satisfying both radial factorization invariance and zero-curvature consistency for the origin Poincare maps in the fixed coordinates of ´ Definition 2.

Proof. Proposition 2 restricts the candidates to $\eta = 0$ and $\eta = 1$ . For fixed nonzero input and incoming relevance, $T _ { \eta , { \pmb x } }$ is independent of curvature. Zero-curvature (as in Definition 2) consistency therefore requires $T _ { \eta , x } R = R$ for every R. The choice $\eta = 1$ satisfies this requirement. The choice $\eta = 0$ does not, since $P _ { \pm } \neq I _ { d }$ for $d \geq 2$ . Hence only $\eta = 1$ satisfies both criteria. □

## D ORIGIN LOGARITHMIC MAPS AND THE GRI TEST

## D.1 POINCARE BRANCH´

Let $r _ { P } = \| { \pmb x } _ { P } \| _ { 2 }$ . The origin logarithmic map has the radial form

$$
v = \log _ { o _ { P } } ^ { P , c } ( { \pmb x } _ { P } ) = \alpha _ { P } ( { \pmb x } _ { P } ) { \pmb x } _ { P } , \qquad \alpha _ { P } ( { \pmb x } _ { P } ) = \frac { \mathrm { a r t a n h } ( \sqrt { c } r _ { P } ) } { \sqrt { c } r _ { P } } .\tag{31}
$$

The continuous value at the origin is $\alpha _ { P } ( { \bf 0 } ) = 1$

## D.2 LORENTZ BRANCH AND TANGENT ALIGNMENT

Let $\theta _ { L } = \mathrm { a r c o s h } ( \sqrt { c } x _ { L , 0 } )$ . The Lorentz logarithmic map is

$$
\log _ { \sigma _ { L } } ^ { L , c } ( { \pmb x } _ { L } ) = \frac { \theta _ { L } } { \sinh \theta _ { L } } \left( { \pmb x } _ { L } - \sqrt { c } { x } _ { L , 0 } { \pmb o } _ { L } \right) = \left( 0 , \frac { \theta _ { L } } { \sinh \theta _ { L } } \bar { \pmb x } _ { L } \right) .\tag{32}
$$

To compare this output with the Poincare branch, we express both in common tangent coordinates. ´ The Jacobian of the isometry in Eq. 14 satisfies

$$
\left. \frac { \partial \phi ( { \pmb x } _ { P } ) } { \partial { \pmb x } _ { P } } \right| _ { { \pmb x } _ { P } = { \bf 0 } } { \pmb v } = \left( \begin{array} { l } { { \pmb 0 } ^ { \top } } \\ { 2 I _ { d } } \end{array} \right) { \pmb v } = ( 0 , 2 { \pmb v } ) .\tag{33}
$$

Thus, the Lorentz spatial tangent component must be divided by two to recover the common coordinates:

$$
\boldsymbol { v } = \frac { 1 } { 2 } \left[ \mathrm { l o g } _ { \sigma _ { L } } ^ { L , c } ( \boldsymbol { x } _ { L } ) \right] _ { \mathrm { s p } } = \alpha _ { L } ( \boldsymbol { x } _ { L } ) \bar { \boldsymbol { x } } _ { L } , \qquad \alpha _ { L } ( \boldsymbol { x } _ { L } ) = \frac { \theta _ { L } } { 2 \sinh \theta _ { L } } .\tag{34}
$$

Here $[ \cdot ] _ { \mathrm { s p } }$ extracts the spatial components. Let $r _ { L } ~ = ~ \| \bar { \pmb { x } } _ { L } \| _ { 2 }$ , the hyperboloid constraint gives $\sqrt { c } x _ { L , 0 } = \sqrt { 1 + c r _ { L } ^ { 2 } }$ and sinh $\theta _ { L } = \sqrt { c } r _ { L }$ . Consequently,

$$
\alpha _ { L } ( { \pmb x } _ { L } ) = \frac { \mathrm { a r s i n h } ( \sqrt { c } r _ { L } ) } { 2 \sqrt { c } r _ { L } } ,\tag{35}
$$

so the aligned Lorentz logarithmic map is radial in the spatial coordinates, with $\alpha _ { L } ( o _ { L } ) = 1 / 2$

## D.3 FORWARD EQUIVALENCE

For ${ \pmb x } _ { L } = \phi ( { \pmb x } _ { P } )$ and $t = \sqrt { c } r _ { P } \in [ 0 , 1 )$ , the conversion formulas give

$$
\sqrt { c } x _ { L , 0 } = \frac { 1 + t ^ { 2 } } { 1 - t ^ { 2 } } , \qquad \theta _ { L } = 2 \operatorname { a r t a n h } ( t ) , \qquad \sinh \theta _ { L } = \frac { 2 t } { 1 - t ^ { 2 } } .\tag{36}
$$

Together with $\bar { \pmb x } _ { L } = \gamma ( { \pmb x } _ { P } ) { \pmb x } _ { P } $

$$
\alpha _ { L } ( \phi ( \mathbf { x } _ { P } ) ) \gamma ( \mathbf { x } _ { P } ) = { \frac { 2 \operatorname { a r t a n h } ( t ) } { 2 [ 2 t / ( 1 - t ^ { 2 } ) ] } } { \frac { 2 } { 1 - t ^ { 2 } } } = { \frac { \operatorname { a r t a n h } ( t ) } { t } } = \alpha _ { P } ( \mathbf { x } _ { P } ) .\tag{37}
$$

All expressions at $t = 0$ are interpreted by continuity. The two paths therefore produce identical common tangent coordinates:

$$
\pmb { v } = \log _ { \pmb { o } _ { P } } ^ { P , c } ( \pmb { x } _ { P } ) = \frac { 1 } { 2 } \left[ \log _ { \pmb { o } _ { L } } ^ { L , c } ( \phi ( \pmb { x } _ { P } ) ) \right] _ { \mathrm { s p } } .\tag{38}
$$

## D.4 RELEVANCE EQUALITY IN THE CONTROLLED TEST

Under the shared encoder, downstream computation, and relevance procedure specified in Section 3.2, both paths receive the same $\scriptstyle { R _ { v } }$ . LRP-radial-all propagates this relevance unchanged through each spatial radial module. The direct path contains the Poincare logarithmic map, while ´ the Lorentz path contains the aligned logarithmic map and the spatial coordinate conversion. Hence,

$$
\begin{array} { r } { R _ { x _ { P } } ^ { P } = R _ { v } , \qquad R _ { x _ { P } } ^ { L } = R _ { \bar { x } _ { L } } = R _ { v } . } \end{array}\tag{39}
$$

The Lorentz time coordinate is treated as a dependent geometric quantity within these modules. Identical relevance at $\scriptstyle { \mathbf { \boldsymbol { x } } } _ { P }$ , followed by the same deterministic propagation through the shared encoder, gives

$$
R _ { a } ^ { \mathrm { { d i r e c t } } } = R _ { a } ^ { \mathrm { { v i a } } L } .\tag{40}
$$

LRP-radial-all therefore satisfies GRI for this specified logarithmic-map test.

## E EXPONENTIAL MAPS AND THE TESTS

## E.1 POINCARE EXPONENTIAL MAP´

Let $r _ { z } = \| z \| _ { 2 }$ . The origin exponential map has the radial form

$$
y _ { P } = \exp _ { o _ { P } } ^ { P , c } ( z ) = \beta _ { P } ( z ) z , \qquad \beta _ { P } ( z ) = { \frac { \operatorname { t a n h } ( \sqrt { c } r _ { z } ) } { \sqrt { c } r _ { z } } } .\tag{41}
$$

The continuous value at the origin is $\beta _ { P } ( { \bf 0 } ) = 1$

## E.2 LORENTZ EXPONENTIAL MAP AND ALIGNMENT

Under the tangent alignment in Eq. 33, the common coordinates z correspond to the ambient Lorentz tangent vector (0, 2z). For an origin tangent vector $\pmb { u } = ( 0 , \bar { \pmb { u } } )$ , with $\lVert \mathbf { \bar { u } } \rVert _ { L } = \lVert \bar { \mathbf { u } } \rVert _ { 2 }$ , the exponential map is

$$
\mathrm { e x p } _ { o _ { L } } ^ { L , c } ( { \pmb u } ) = \mathrm { c o s h } ( \sqrt { c } \| { \pmb u } \| _ { L } ) { \pmb o } _ { L } + \frac { \mathrm { s i n h } ( \sqrt { c } \| { \pmb u } \| _ { L } ) } { \sqrt { c } \| { \pmb u } \| _ { L } } { \pmb u } .\tag{42}
$$

Substituting $\pmb { u } = ( 0 , 2 z )$ gives

$$
y _ { L } = \exp _ { o _ { L } } ^ { L , c } ( ( 0 , 2 z ) ) = \left( \frac { \cosh ( 2 \sqrt { c } r _ { z } ) } { \sqrt { c } } , \beta _ { L } ( z ) z \right) , \qquad \beta _ { L } ( z ) = \frac { \sinh ( 2 \sqrt { c } r _ { z } ) } { \sqrt { c } r _ { z } } .\tag{43}
$$

Thus, the spatial output $\bar { \pmb { y } } _ { L } = \beta _ { L } ( \pmb { z } ) \pmb { z }$ is radial, with $\beta _ { L } ( \mathbf { 0 } ) = 2$

To verify forward equivalence, set $\mathit { t } ~ = ~ \sqrt { c } r _ { z }$ Taking the norm of Eq. 41 gives $\| { \pmb y } _ { P } \| _ { 2 } =$ tanh $( t ) { \dot { / } } { \sqrt { c } }$ . Substitution into Eq. 14 then yields

$$
\begin{array} { l } { { \displaystyle \phi ( { \pmb y } _ { P } ) = \left( \frac { 1 + \operatorname { t a n h } ^ { 2 } t } { \sqrt { c } ( 1 - \operatorname { t a n h } ^ { 2 } t ) } , \frac { 2 \operatorname { t a n h } t } { t ( 1 - \operatorname { t a n h } ^ { 2 } t ) } z \right) } } \\ { { \displaystyle ~ = \left( \frac { \cosh ( 2 t ) } { \sqrt { c } } , \frac { \sinh ( 2 t ) } { t } z \right) = { \pmb y } _ { L } } , } \end{array}\tag{44}
$$

where the expressions at $t = 0$ are interpreted by continuity. The two exponential maps therefore produce corresponding geometric points when their tangent inputs are aligned.

## E.3 RELEVANCE RULES FOR THE SPATIAL RADIAL MODULES

Both $\pmb { y } _ { P } = \beta _ { P } ( z ) ,$ z and $\bar { \pmb { y } } _ { L } = \beta _ { L } ( \pmb { z } ) \pmb { z }$ have the scalar–signal structure of Eq. 7. We apply the same propagation rules as for the logarithmic maps, with z as the signal. The constant tangent alignment is included in $\beta _ { L }$ . LRP-radial-all and LRP-half are applicable here.

## E.4 A CONTROLLED EXPONENTIAL-MAP TEST

The exponential-map test compares two paths from the same tangent coordinates z to the same Lorentz point. The direct path applies the aligned Lorentz exponential map, whereas the indirect path applies the Poincare exponential map followed by the coordinate conversion:´

$$
\begin{array} { r } { \pmb { y } _ { L } = \exp _ { \pmb { o } _ { L } } ^ { L , c } ( ( 0 , 2 z ) ) = \phi \big ( \exp _ { \pmb { o } _ { P } } ^ { P , c } ( z ) \big ) . } \end{array}\tag{45}
$$

Both paths then use the same downstream computation $G ( y _ { L } )$ . With identical output relevance initialization and the same deterministic downstream propagation, they receive the same relevance $R _ { y _ { L } } = ( R _ { y _ { L , 0 } } , R _ { \bar { y } _ { L } } )$

Shared treatment of the time coordinate. In both realizations, we express the time coordinate through the same spatial constraint,

$$
y _ { L , 0 } = \sqrt { c ^ { - 1 } + \| \bar { \pmb { y } } _ { L } \| _ { 2 } ^ { 2 } } .\tag{46}
$$

Any relevance assigned to this coordinate is propagated through the constraint using the same rule in both paths. For example, for $\bar { \pmb { y } } _ { L } \neq \mathbf { 0 }$ , squared-coordinate redistribution gives

$$
\widetilde { R } _ { { \bar { y } } _ { L , i } } = R _ { { \bar { y } } _ { L , i } } + \frac { { \bar { y } } _ { L , i } ^ { 2 } } { \| { \bar { y } } _ { L } \| _ { 2 } ^ { 2 } } R _ { y _ { L , 0 } } .\tag{47}
$$

This convention conserves the combined spatial and time-coordinate relevance. At zero spatial input, a shared fallback must be specified. The equality argument below requires only that both paths use the same deterministic treatment of the time coordinate.

![](images/3d7bea3784907ff0cfee4f6ef069e0176e9115e273c1e67af4c3076d9c7917d5.jpg)  
Figure 7: Redistribution of scalar-branch relevance from $\alpha ( \| \pmb { x } \| _ { 2 } )$ to x through the squared norm.

Relevance equality under LRP-radial-all. After the shared time-coordinate propagation, both paths receive the same effective spatial relevance $\widetilde { R } _ { \bar { y } _ { L } }$ . The direct spatial map is $\bar { \pmb { y } } _ { L } = \beta _ { L } ( \pmb { z } ) \pmb { z }$ while the indirect path contains two radial modules:

$$
{ \pmb y } _ { P } = \beta _ { P } ( z ) z , \qquad { \bar { \pmb y } } _ { L } = \gamma ( { \pmb y } _ { P } ) { \pmb y } _ { P } .\tag{48}
$$

Applying LRP-radial-all to each spatial radial module gives

$$
R _ { z } ^ { \mathrm { d i r e c t } } = \widetilde { R } _ { \bar { y } _ { L } } , \qquad R _ { z } ^ { \mathrm { v i a } P } = R _ { y _ { P } } = \widetilde { R } _ { \bar { y } _ { L } } .\tag{49}
$$

Thus, the two paths yield identical relevance at the common input z. $\operatorname { I f } z$ is produced by a shared encoder, identical deterministic propagation through that encoder also gives identical relevance at the original input.

## F LRP-HALF: CONSERVATION AND CONSISTENCY COUNTEREXAMPLES

This section analyzes the specified LRP-half baseline, which combines equal splitting at scalar– signal products with squared-norm redistribution of the scalar relevance. Although this rule conserves total relevance, it can violate GRI and zero-curvature consistency.

## F.1 BRANCH SPLIT, SQUARED-NORM REDISTRIBUTION, AND CONSERVATION

Consider a radial module ${ \pmb y } = \alpha ( \| { \pmb x } \| _ { 2 } ) { \pmb x }$ with $\textbf { \em x } \neq \textbf { 0 }$ . LRP-half assigns half of the incoming relevance to the signal branch and half to the scalar factor:

$$
R _ { x _ { i } } ^ { \mathrm { s i g } } = \frac { 1 } { 2 } R _ { y _ { i } } , \qquad R _ { \alpha } = \frac { 1 } { 2 } \sum _ { j } R _ { y _ { j } } .\tag{50}
$$

Let $\begin{array} { r } { s = \| \pmb { x } \| _ { 2 } ^ { 2 } = \sum _ { j } x _ { j } ^ { 2 } } \end{array}$ and $r = { \sqrt { s } } .$ We treat the scalar transformations $s \mapsto r \mapsto \alpha ( r )$ as relevance-preserving, so $\mathbf { \bar { \boldsymbol { R } } } _ { s } = \boldsymbol { R _ { r } } = \boldsymbol { R _ { \alpha } }$ . Contribution-proportional redistribution through the sum, followed by relevance-preserving propagation through each scalar square, gives

$$
R _ { x _ { i } } ^ { \mathrm { r a d } } = \frac { x _ { i } ^ { 2 } } { \| \pmb { x } \| _ { 2 } ^ { 2 } } R _ { s } .\tag{51}
$$

Combining the two branches yields

$$
R _ { x _ { i } } = \frac { 1 } { 2 } R _ { y _ { i } } + \frac { 1 } { 2 } \frac { x _ { i } ^ { 2 } } { \| \pmb { x } \| _ { 2 } ^ { 2 } } \sum _ { j } R _ { y _ { j } } .\tag{52}
$$

Since the squared-coordinate weights sum to one, the rule conserves signed total relevance:

$$
\sum _ { i } R _ { x _ { i } } = \frac { 1 } { 2 } \sum _ { i } R _ { y _ { i } } + \frac { 1 } { 2 } \sum _ { j } R _ { y _ { j } } = \sum _ { j } R _ { y _ { j } } .\tag{53}
$$

At $\textbf { \em x } = \textbf { 0 }$ , the redistribution weights are undefined and require a separate convention, such as identity propagation. The counterexamples below use nonzero inputs and are independent of this choice. Numerical stabilization is discussed in Appendix H.

## F.2 FACTORIZATION DEPENDENCE AND THE GRI COUNTEREXAMPLE

Consider the aligned logarithmic-map paths in Eq. 4, with $\mathbf { \Delta } { \mathbf { x } } _ { P } \neq \mathbf { 0 }$ and identical incoming relevance $\scriptstyle { R _ { v } }$ . The direct Poincare path contains one radial module and gives´

$$
R _ { x _ { P , i } } ^ { P } = \frac { 1 } { 2 } R _ { v _ { i } } + \frac { 1 } { 2 } \frac { x _ { P , i } ^ { 2 } } { \| \pmb { x } _ { P } \| _ { 2 } ^ { 2 } } \sum _ { j } R _ { v _ { j } } .\tag{54}
$$

The Lorentz path contains two radial modules: the spatial conversion ${ \bar { \pmb x } } _ { L } = \gamma ( { \pmb x } _ { P } ) { \pmb x } _ { P }$ and the aligned logarithmic map. Because the conversion scales every coordinate by the same nonzero scalar,

$$
\frac { \bar { x } _ { L , i } ^ { 2 } } { \| \bar { \pmb x } _ { L } \| _ { 2 } ^ { 2 } } = \frac { x _ { P , i } ^ { 2 } } { \| \pmb x _ { P } \| _ { 2 } ^ { 2 } } .\tag{55}
$$

Propagation through the aligned Lorentz logarithmic map first gives

$$
R _ { { \bar { x } } _ { L , i } } = \frac { 1 } { 2 } R _ { v _ { i } } + \frac { 1 } { 2 } \frac { { \bar { x } } _ { L , i } ^ { 2 } } { \| { \bar { x } } _ { L } \| _ { 2 } ^ { 2 } } \sum _ { j } R _ { v _ { j } } , \qquad \sum _ { j } R _ { { \bar { x } } _ { L , j } } = \sum _ { j } R _ { v _ { j } } .\tag{56}
$$

Applying the rule again through the spatial conversion yields

$$
\begin{array} { l } { { \displaystyle R _ { x _ { P , i } } ^ { L } = \frac { 1 } { 2 } R _ { \bar { x } _ { L , i } } + \frac { 1 } { 2 } \frac { x _ { P , i } ^ { 2 } } { | | x _ { P } | | _ { 2 } ^ { 2 } } \sum _ { j } R _ { \bar { x } _ { L , j } } } } \\ { ~ = \frac { 1 } { 2 } \left( \frac { 1 } { 2 } R _ { v _ { i } } + \frac { 1 } { 2 } \frac { x _ { P , i } ^ { 2 } } { | | x _ { P } | | _ { 2 } ^ { 2 } } \sum _ { j } R _ { v _ { j } } \right) + \frac { 1 } { 2 } \frac { x _ { P , i } ^ { 2 } } { | | x _ { P } | | _ { 2 } ^ { 2 } } \sum _ { j } R _ { v _ { j } } } \\ { ~ = \frac { 1 } { 4 } R _ { v _ { i } } + \frac { 3 } { 4 } \frac { x _ { P , i } ^ { 2 } } { | | x _ { P } | | _ { 2 } ^ { 2 } } \sum _ { j } R _ { v _ { j } } . } \end{array}\tag{57}
$$

The constant tangent alignment is included in $\alpha _ { L }$ and is not treated as an additional half-splitting node. Both paths conserve the same total relevance, but their coordinate-wise difference is

$$
R _ { x _ { P , i } } ^ { L } - R _ { x _ { P , i } } ^ { P } = \frac { 1 } { 4 } \left( \frac { x _ { P , i } ^ { 2 } } { \| \pmb { x } _ { P } \| _ { 2 } ^ { 2 } } \sum _ { j } R _ { v _ { j } } - R _ { v _ { i } } \right) .\tag{58}
$$

The allocations coincide if and only if $\begin{array} { r } { R _ { v _ { i } } = ( x _ { P , i } ^ { 2 } / \| \pmb { x } _ { P } \| _ { 2 } ^ { 2 } ) \sum _ { j } R _ { v _ { j } } } \end{array}$ for every i.

An explicit counterexample at the original input. Choose the shared encoder to be the identity on the Poincare ball, so that´ ${ \bf { a } } = { \bf { x } } _ { P }$ . Let $c = 1 , \boldsymbol { x } _ { P } ^ { * } = ( 1 / 4 , 1 / 4 ) ^ { \top }$ , and ${ \pmb v } ^ { * } = \log _ { { \pmb \sigma } _ { P } } ^ { P , 1 } ( { \pmb x } _ { P } ^ { * } )$ . Use the shared linear head $G ( \pmb { v } ) = v _ { 1 } / v _ { 1 } ^ { * }$ , where $v _ { 1 } ^ { * }$ is fixed. At this input, the output score is one, and LRP-0 through the head with unit relevance initialization gives $\boldsymbol { R _ { v } } = ( 1 , 0 ) ^ { \top }$ . The two paths yield

$$
\pmb { R } _ { a } ^ { \mathrm { d i r e c t } } = ( 3 / 4 , 1 / 4 ) ^ { \top } , \qquad \pmb { R } _ { a } ^ { \mathrm { v i a } L } = ( 5 / 8 , 3 / 8 ) ^ { \top } .\tag{59}
$$

Both allocations sum to one but differ at the same original input. Since the aligned paths implement the same function throughout the common domain, this is a counterexample to GRI for the specified LRP-half baseline.

## F.3 FAILURE OF ZERO-CURVATURE CONSISTENCY

For fixed inputs, the origin Poincare logarithmic and exponential maps satisfy´

$$
\log _ { o _ { P } } ^ { P , c } ( { \pmb x } ) = \left( 1 + \frac { c } { 3 } \| { \pmb x } \| _ { 2 } ^ { 2 } + O ( c ^ { 2 } ) \right) { \pmb x } ,\tag{60}
$$

$$
\exp _ { o _ { P } } ^ { P , c } ( z ) = \left( 1 - \frac { c } { 3 } \| z \| _ { 2 } ^ { 2 } + O ( c ^ { 2 } ) \right) z .\tag{61}
$$

Both converge to the identity as $c \to 0$ . However, for fixed nonzero input and incoming relevance, the LRP-half allocation in Eq. 52 is independent of curvature. Consequently,

$$
\operatorname* { l i m } _ { c \to 0 } \left( R _ { x _ { i } } - R _ { y _ { i } } \right) = \frac { 1 } { 2 } \left( \frac { x _ { i } ^ { 2 } } { \| \pmb { x } \| _ { 2 } ^ { 2 } } \sum _ { j } R _ { y _ { j } } - R _ { y _ { i } } \right) ,\tag{62}
$$

which is generally nonzero. Although the radial factor approaches one, the rule continues to redistribute half of the incoming relevance through the radial branch. The same argument applies to the exponential map with z as the input signal. Thus, the specified LRP-half baseline fails zerocurvature consistency for both Poincare origin maps.´

## G BIAS, MOBIUS ¨ ADDITION, AND MATCHED LORENTZ REALIZATION

## G.1 POINCARE BIAS OPERATION´

Let b be a fixed tangent-space parameter and $\pmb { b } _ { P } = \exp _ { { \pmb { o } } _ { P } } ^ { P , c } ( { \pmb { b } } )$ . For ${ \pmb m } _ { P } = { \pmb W } \otimes _ { c } { \pmb x } _ { P }$ , define the bias operation as

$$
{ \pmb y } _ { P } = T _ { { \pmb b } _ { P } } ^ { P } ( { \pmb m } _ { P } ) = { \pmb m } _ { P } \oplus _ { c } { \pmb b } _ { P } .\tag{63}
$$

Mobius addition can also be represented in the scalar-vector product form¨

$$
{ \pmb m } _ { P } \oplus _ { c } { \pmb b } _ { P } = \alpha _ { c } ^ { \oplus } ( { \pmb m } _ { P } , { \pmb b } _ { P } ) { \pmb m } _ { P } + \beta _ { c } ^ { \oplus } ( { \pmb m } _ { P } , { \pmb b } _ { P } ) { \pmb b } _ { P } ,\tag{64}
$$

where

$$
\begin{array} { l } { \displaystyle \alpha _ { c } ^ { \oplus } ( m _ { P } , b _ { P } ) = \frac { 1 + 2 c \langle m _ { P } , b _ { P } \rangle + c \| b _ { P } \| _ { 2 } ^ { 2 } } { 1 + 2 c \langle m _ { P } , b _ { P } \rangle + c ^ { 2 } \| m _ { P } \| _ { 2 } ^ { 2 } \| b _ { P } \| _ { 2 } ^ { 2 } } , } \\ { \displaystyle \beta _ { c } ^ { \oplus } ( m _ { P } , b _ { P } ) = \frac { 1 - c \| m _ { P } \| _ { 2 } ^ { 2 } } { 1 + 2 c \langle m _ { P } , b _ { P } \rangle + c ^ { 2 } \| m _ { P } \| _ { 2 } ^ { 2 } \| b _ { P } \| _ { 2 } ^ { 2 } } . } \end{array}\tag{65}
$$

Although $b _ { P }$ is fixed, both coefficients depend on the input m<sub>P</sub>. The bias contribution generally changes the output direction, so Mobius bias addition is not an origin-centered radial scaling of the¨ signal.

## G.2 SIGNAL-ONLY RELEVANCE CONVENTION

We explain the prediction in terms of input features rather than fixed model parameters. In this spirit, we suggest one way to propagate relevance conservatively:

$$
R _ { b _ { P } } = 0 , \qquad R _ { m _ { P } } = R _ { y _ { P } } .\tag{66}
$$

Note it is a separate attribution convention and not a consequence of the radial factorization result.

## G.3 MATCHED LORENTZ REALIZATION

An exactly equivalent forward bias operation can be constructed through the Poincare–Lorentz isom-´ etry ϕ. Set ${ m _ { L } = \phi ( m _ { P } ) }$ and $b _ { \cal L } = \phi ( b _ { P } )$ , and define

$$
T _ { b _ { L } } ^ { L } = \phi \circ T _ { b _ { P } } ^ { P } \circ \phi ^ { - 1 } ,\tag{67}
$$

$$
T _ { { b } _ { L } } ^ { L } ( { \bf m } _ { L } ) = \phi \big ( \phi ^ { - 1 } ( { \bf m } _ { L } ) \oplus _ { c } { \ b } _ { P } \big ) .\tag{68}
$$

The resulting computation is

$$
{ \pmb m } _ { L } \stackrel { \phi ^ { - 1 } } { \longrightarrow } { \pmb m } _ { P } \stackrel { \oplus _ { c } { \pmb b } _ { P } } { \longrightarrow } { \pmb y } _ { P } \stackrel { \phi } { \longrightarrow } { \pmb y } _ { L } .\tag{69}
$$

By construction, $T _ { \pmb { b } _ { P } } ^ { L } ( \phi ( \pmb { m } _ { P } ) ) = \phi ( T _ { \pmb { b } _ { P } } ^ { P } ( \pmb { m } _ { P } ) )$ , so the two realizations implement the same operation in different geometric representations.

## H TANGENT-SPACE PROPAGATION AND NUMERICAL STABILIZATION

For a bias-free tangent-space linear transformation $z = W v$ , the unstabilized LRP-0 rule is

$$
R _ { v _ { i } } = \sum _ { j } \frac { v _ { i } W _ { j i } } { z _ { j } } R _ { z _ { j } } , \qquad z _ { j } = \sum _ { k } v _ { k } W _ { j k } .\tag{70}
$$

Assuming $z _ { j } \neq 0$ for every output receiving nonzero relevance, and omitting zero-relevance outputs, summation over the inputs gives exact conservation:

$$
\sum _ { i } R _ { v _ { i } } = \sum _ { j } \frac { \sum _ { i } v _ { i } W _ { j i } } { z _ { j } } R _ { z _ { j } } = \sum _ { j } R _ { z _ { j } } .\tag{71}
$$

Numerical stabilization generally introduces a conservation residual. $\mathrm { F o r \ L R P  – } \epsilon$ with $\epsilon > 0 ,$ , replace the denominator by $\widetilde { z } _ { j } = z _ { j } + \epsilon \mathrm { s g n } _ { + } ( z _ { j } )$ , where $\mathrm { s g n } _ { + } ( z ) = 1$ for $z \geq 0$ and −1 otherwise. The propagated total becomes

$$
\sum _ { i } R _ { v _ { i } } = \sum _ { j } \frac { z _ { j } } { z _ { j } + \epsilon \mathrm { s g n } _ { + } ( z _ { j } ) } R _ { z _ { j } } .\tag{72}
$$

Consequently, the signed conservation residual is

$$
\sum _ { j } R _ { z _ { j } } - \sum _ { i } R _ { v _ { i } } = \sum _ { j } { \frac { \epsilon } { | z _ { j } | + \epsilon } } R _ { z _ { j } } .\tag{73}
$$

Exact conservation is therefore not guaranteed unless the residual vanishes or is explicitly handled, for example as in NRM (Xiong et al., 2026).

## I GEOMETRIC REPRESENTATION INVARIANCE EXPERIMENT

Setup. We consider curvature −c with $c = 1$ and choose the shared encoder to be the identity on the Poincare ball, so that the original input is´ ${ \bf { a } } = { \bf { x } } _ { P }$ . We evaluate at $\pmb { x } _ { P } = ( 0 . 2 , 0 . 3 , - 0 . 1 ) ^ { \top }$ <sup>⊤</sup>. Its Lorentz representation is

$$
\pmb { x } _ { L } = \phi ( \pmb { x } _ { P } ) = \left( \frac { 1 + c \| \pmb { x } _ { P } \| _ { 2 } ^ { 2 } } { \sqrt { c } ( 1 - c \| \pmb { x } _ { P } \| _ { 2 } ^ { 2 } ) } , \frac { 2 \pmb { x } _ { P } } { 1 - c \| \pmb { x } _ { P } \| _ { 2 } ^ { 2 } } \right) .\tag{74}
$$

The direct Poincare path and the aligned Lorentz path reach the same tangent coordinates:´

$$
v = \log _ { o _ { P } } ^ { P , c } ( { \pmb x } _ { P } ) = \frac { 1 } { 2 } \left[ \log _ { o _ { L } } ^ { L , c } ( \phi ( { \pmb x } _ { P } ) ) \right] _ { \mathrm { s p } } , \qquad o _ { P } = \mathbf { 0 } , \quad o _ { L } = ( c ^ { - 1 / 2 } , \mathbf { 0 } ) .\tag{75}
$$

Here $[ \cdot ] _ { \mathrm { s p } }$ extracts the spatial components, and the factor $1 / 2$ converts the ambient Lorentz tangent vector to the common coordinates, as specified in Eq. 33. Both paths use the shared scalar head $G ( \pmb { v } ) ~ = ~ v _ { 1 }$ and implement the same function throughout the Poincare ball. Initializing output´ relevance with the target score and applying LRP-0 through this head gives $\boldsymbol { R _ { v } } = ( v _ { 1 } , 0 , 0 ) ^ { \top }$

Relevance propagation. For a radial module ${ \pmb y } = \alpha ( \| { \pmb x } \| _ { 2 } ) { \pmb x }$ , LRP-radial-all propagates $\scriptstyle R _ { x } =$ $R _ { y }$ , whereas the specified LRP-half baseline uses

$$
R _ { x _ { i } } = { \frac { 1 } { 2 } } R _ { y _ { i } } + { \frac { 1 } { 2 } } { \frac { x _ { i } ^ { 2 } } { \| { \pmb x } \| _ { 2 } ^ { 2 } } } \sum _ { j } R _ { y _ { j } } , \qquad { \pmb x } \neq \mathbf { 0 } .\tag{76}
$$

Both rules conserve signed total relevance locally. The direct path contains one radial module, ${ \pmb v } = \alpha _ { P } ( { \pmb x } _ { P } ) { \pmb x } _ { P }$ . The Lorentz path contains two: the spatial conversion $\bar { \pmb x } _ { L } = \gamma ( { \pmb x } _ { P } ) { \pmb x } _ { P }$ and the aligned logarithmic map ${ \pmb v } = \alpha _ { L } ( { \pmb x } _ { L } ) { \bar { \pmb x } } _ { L }$ . We apply each rule consistently to these modules and compare the resulting relevance at the common original input ${ \textbf { \em a } } = { \textbf { \em x } } _ { P }$ The tangent alignment factor is included in $\alpha _ { L } ,$ , rather than treated as an additional multiplication node. The Lorentz time coordinate is treated as a dependent geometric quantity within the radial modules.

Table 3: Relevance at the common original input ${ \pmb a } = { \pmb x } _ { P }$ for the two equivalent logarithmic-map paths. Values are rounded to six decimal places.
<table><tr><td>Rule</td><td>Path</td><td> $R _ { a _ { 1 } }$ </td><td> $R _ { a _ { 2 } }$ </td><td> $R _ { a _ { 3 } }$ </td></tr><tr><td>LRP-radial-all</td><td>Direct</td><td>0.210205</td><td>0</td><td>0</td></tr><tr><td></td><td>Via Lorentz</td><td>0.210205</td><td>0</td><td>0</td></tr><tr><td>LRP-half</td><td>Direct</td><td>0.135132</td><td>0.067566</td><td>0.007507</td></tr><tr><td></td><td>Via Lorentz</td><td>0.097595</td><td>0.101349</td><td>0.011261</td></tr></table>

![](images/39bd34927d083909ea759e34154213dffe7adc4fcf2acad4dbf0ed978cc6ee38.jpg)

![](images/acb95966d4c7c61a6e2d4d3c398a2a3e8beace22db59c003ed473b1f0483574f.jpg)

![](images/e4162bf56cd064681b761e3a71f86ecbf3bd53a93bf5783c2050872a314ac73f.jpg)

![](images/7e58a29f942ad74fd6aca754060c53e0ea2c4d2c850e76c3d8989a6d7c3a75bc.jpg)  
Figure 8: Temporal perturbation validation of LRP for low-load (top) and high-load (bottom) trials. Left: window relevance and the five highest-ranked windows. Right: mean target probability drops under LRP-guided versus random perturbation. Shading indicates the 2.5th-97.5th percentiles across 20 random selections.

Results. Table 3 shows identical input relevance for LRP-radial-all across the two paths. In contrast, LRP-half produces different relevances, although both allocations sum to the same target score $v _ { 1 }$ . This numerical example illustrates the counterexample derived in Appendix F: local conservation does not guarantee GRI, because equivalent radial factorizations can yield different relevance allocations at the same original input.

## J VALIDATE SEEG RELEVANT WINDOW EXPLANATION BY PERTURBATION

To assess whether the relevant time windows in the LRP heatmaps identify input regions relevant to the network’s predictions, we conduct a window perturbation test. We compute trial-wise relevance for the target class logit and averaged the maps within each class. The average input time series (3 seconds long) is divided into non-overlapping 100-ms windows, ranked by their signed relevance summed across channels and time. We perturb the top $k \in \{ 1 , 2 , 3 , 5 \}$ windows in each original trial by replacing the signal within these windows, across all channels, with each channel’s original temporal mean. Randomly selected windows serve as a control, averaged over 20 repetitions. The results are summarized in Figure 8. LRP-guided perturbation produces larger mean reductions in target probabilities than random perturbation for both classes. These results indicate that the temporal relevance estimated by LRP is aligned with the model’s predictive behavior and can effectively identify time periods that are important for the classification decision. This further supports the practical utility of the relevance heatmaps.

## K MODEL CONFIGURATION FOR CIFAR-10

Model architecture. We use a lightweight Lorentz convolutional network inspired by HyperbolicCV (Bdeir et al., 2024), with explicit origin-centered exponential and logarithmic maps for evaluating relevance propagation through radial transformations. We fix the sectional curvature to −c with c = 1 and represent each hyperbolic feature by its spatial coordinates $\boldsymbol { s } \in \mathbb { R } ^ { d }$ , with the time coordinate implicitly given by $t ( \pmb { \mathscr { s } } ) = \sqrt { c ^ { - 1 } + \| \pmb { \mathscr { s } } \| _ { 2 } ^ { 2 } }$ . Let ${ \pmb { o } } _ { L } = ( c ^ { - 1 / 2 } , \mathbf { 0 } )$ . For convenience, we denote the spatial components of the origin exponential and logarithmic maps by

$$
\begin{array} { r l } & { \displaystyle \exp _ { o _ { L } , \mathrm { s p } } ^ { L , c } ( \pmb { v } ) : = \left[ \exp _ { o _ { L } } ^ { L , c } ( ( 0 , \pmb { v } ) ) \right] _ { \mathrm { s p } } = \frac { \sinh ( \sqrt { c } \| \pmb { v } \| _ { 2 } ) } { \sqrt { c } \| \pmb { v } \| _ { 2 } } \pmb { v } , } \\ & { \displaystyle \log _ { o _ { L } , \mathrm { s p } } ^ { L , c } ( \pmb { s } ) : = \left[ \log _ { o _ { L } } ^ { L , c } ( ( t ( \pmb { s } ) , \pmb { s } ) ) \right] _ { \mathrm { s p } } = \frac { \operatorname { a r s i n h } ( \sqrt { c } \| \pmb { s } \| _ { 2 } ) } { \sqrt { c } \| \pmb { s } \| _ { 2 } } \pmb { s } , } \end{array}\tag{77}
$$

where both scalar factors are defined as one at zero. Here v denotes the spatial component of the ambient Lorentz tangent vector $( 0 , v )$ , rather than the aligned Poincare tangent coordinates used in the´ GRI test. These maps operate independently at each spatial location across the feature channels. The normalized RGB input x is first mapped to spatial Lorentz coordinates as $\begin{array} { r } { \pmb { s } ^ { ( 0 ) } = \exp _ { o _ { r , \mathrm { s p } } } ^ { L , c } ( 0 . 2 5 \pmb { x } ) } \end{array}$ We use the factor of 0.25 to moderate the initial tangent-space radius and the resulting hyperbolic coordinate magnitudes.

The backbone contains six $3 \times 3$ convolutional layers with channel widths (32, 32, 64, 64, 128, 128), strides (1, 1, 2, 1, 2, 1), and padding of one. For the flattened spatial patch $\mathbf { \mathbf { \mathbf { \mathbf { p } } } } _ { u } ^ { ( \ell - 1 ) }$ at location u, the ℓth convolution computes

$$
\begin{array} { r } { \tau _ { u } ^ { ( \ell ) } = \sqrt { c ^ { - 1 } + \| \boldsymbol { p } _ { u } ^ { ( \ell - 1 ) } \| _ { 2 } ^ { 2 } } , \qquad z _ { u } ^ { ( \ell ) } = W _ { s } ^ { ( \ell ) } \boldsymbol { p } _ { u } ^ { ( \ell - 1 ) } + \boldsymbol { w } _ { t } ^ { ( \ell ) } \tau _ { u } ^ { ( \ell ) } + \boldsymbol { b } ^ { ( \ell ) } , } \end{array}\tag{78}
$$

where $W _ { s } ^ { ( \ell ) }$ and ${ \pmb w } _ { t } ^ { ( \ell ) }$ are learnable spatial and time-coordinate weights, respectively. Each convolution is followed by a tangent-space activation,

$$
\begin{array} { r } { \pmb { s } _ { u } ^ { ( \ell ) } = \exp _ { \pmb { o } _ { L } , \mathrm { s p } } ^ { L , c } \left( \mathrm { R e L U } \left( \log _ { \pmb { o } _ { L } , \mathrm { s p } } ^ { L , c } ( \pmb { z } _ { u } ^ { ( \ell ) } ) \right) \right) . } \end{array}\tag{79}
$$

The resulting feature resolutions are $3 2 \times 3 2$ $1 6 \times 1 6$ , and $8 \times 8$ , with two convolutional layers at each resolution. After the final layer, we average the spatial coordinates over the $8 \times 8$ feature grid, apply the logarithmic map, and obtain the ten class logits through a linear classifier:

$$
\bar { s } = \frac { 1 } { 6 4 } \sum _ { u = 1 } ^ { 6 4 } s _ { u } ^ { ( 6 ) } , \qquad f ( x ) = W _ { \mathrm { c l s } } \log _ { o _ { L } , \mathrm { s p } } ^ { L , c } ( \bar { s } ) + b _ { \mathrm { c l s } } .\tag{80}
$$

Thus, pooling is an arithmetic mean of spatial coordinates rather than a Lorentz centroid, and classification is performed in the tangent space. Note that this architecture is a custom lightweight variant rather than the L-ResNet18 in HyperbolicCV.

Training configuration. We split the 50000 CIFAR-10 training images into 45000 training and 5000 validation samples, and reserve the official 10,000-image test set for final evaluation. Training augmentation consists of random $3 2 \times 3 2$ crops with four-pixel padding and random horizontal flips. Inputs are normalized using channel-wise means (0.4914, 0.4822, 0.4465) and standard deviations (0.2470, 0.2435, 0.2616). We optimize cross-entropy loss for 200 epochs using AdamW with batch size 128, initial learning rate $\mathrm { 1 0 ^ { - 3 } }$ , and weight decay $1 0 ^ { - 4 }$ The learning rate follows a cosine schedule to zero, and the global gradient norm is clipped to 5. The curvature and input scaling factor remain fixed throughout training. The checkpoint with the highest validation accuracy (85.80%) is selected for test evaluation (test accuracy is 85.70%) and attribution experiments.

Relevance propagation through Lorentz convolutions. We treat each convolution as an affine mapping of the augmented patch coordinates $( p _ { u } , \tau _ { u } )$ and first redistribute output relevance to the spatial and time branches using the $z ^ { B }$ rule in the first layer and the γ-rule in subsequent layers. For a γ-rule layer, suppressing the layer index, the relevance assigned to the patch time coordinate is

$$
R _ { \tau _ { u } } = \sum _ { j } \frac { \tau _ { u } \rho _ { \gamma } ( w _ { t , j } ) } { \mathrm { s t a b } _ { \epsilon } ( \sum _ { i } p _ { u , i } \rho _ { \gamma } ( W _ { s , j i } ) + \tau _ { u } \rho _ { \gamma } ( w _ { t , j } ) + b _ { j } ) } R _ { u , j } ,\tag{81}
$$

where $\rho _ { \gamma } ( w ) = w + \gamma$ max $( w , 0 )$ and sta $\mathrm { b } _ { \epsilon } ( z ) = z + \epsilon \mathrm { s i g n } _ { + } ( z )$ , with $\mathrm { s i g n } _ { + } ( 0 ) = 1$ . Since $\tau _ { u }$ is determined by the spatial patch, we subsequently redistribute its relevance using squared-coordinate proportions:

$$
R _ { u , i } ^ { \mathrm { t i m e } } = \frac { p _ { u , i } ^ { 2 } } { \sum _ { m } p _ { u , m } ^ { 2 } } R _ { \tau _ { u } } , \qquad R _ { u , i } = R _ { u , i } ^ { \mathrm { s p a c e } } + R _ { u , i } ^ { \mathrm { t i m e } } .\tag{82}
$$

![](images/bdad16820963fa273e3b61c12aeff2052e79c73056f8292500cdc84250a6d740.jpg)  
Figure 9: Fidelity-Sparsity curve for MNIST experiment with 95% confidence interval. For Fidelity+ higher is better and for Fidelity- lower is better. Lower sparsity corresponds to a larger selected pixel set, which is removed for Fidelity+ and retained for Fidelity-.

For a zero patch, where $\begin{array} { r } { \sum _ { m } p _ { u , m } ^ { 2 } = 0 } \end{array}$ , we set $R _ { u , i } ^ { \mathrm { t i m e } } = 0$ for every coordinate and record $R _ { \tau _ { u } }$ as unassigned relevance. The recorded zero-patch contribution, together with residuals from affine biases and numerical stabilization, are summed up as the total relevance conservation residual.

The first-layer $z ^ { B }$ rule instead uses bounded contributions for both spatial and time coordinates, followed by the same squared-proportion redistribution. Contributions from overlapping patches are summed at each input coordinate. The fixed curvature term receives no relevance. This time coordinate rule is shared by LRP-radial-all and LRP-half, which differ only in their treatment of the exponential and logarithmic radial maps.

We use a fixed $\gamma = 0 . 2 5$ in convolutional layers 2–6 and $\epsilon = 1 0 ^ { - 6 }$ for denominator stabilization throughout relevance propagation, including the ϵ-rule for classifier and average pooling. The first convolution uses the $z ^ { \mathbf { \hat { \boldsymbol { B } } } }$ rule without $\gamma$ modification, for which we derive a fixed bounding box from the valid RGB range [0, 1], accounting for normalization, input scaling, and the exponential map. Time-coordinate bounds are computed from these spatial bounds using the Lorentz constraint. The same box is used for all images. These settings are identical for LRP-radial-all and LRP-half.

Baseline configurations. Integrated Gradients uses a zero baseline in normalized input space, corresponding to the RGB normalization mean, and integration over 128 steps. Attributions are summed across RGB channels to obtain pixel scores. The random baseline averages three independent pixel permutations per image.

Perturbations replace all three channels of selected pixels with the mean-color baseline. We evaluate sparsity with 5% increment steps. Evaluation uses 512 test images sampled uniformly without replacement. We estimate 95% percentile bootstrap confidence intervals using 10,000 image-level resamples with replacement, sharing resampling indices across methods and metrics.

## L ADDITIONAL FIGURES FOR EXPERIMENTS

![](images/0fd9625939547d870b4d1f1a269baf0c690af21550d887efccea4b65bca1edab.jpg)  
Figure 10: Qualitative attribution comparison on four CIFAR-10 examples. Columns show the input, heatmaps for LRP-radial-all, LRP-half, Gradient×Input, and Integrated Gradients. RGB-channel attributions are summed, with red and blue indicating positive and negative values, respectively.

![](images/3990e8fb0e1dd226ecc67d25e4bcb80a0b13b0cc31446d0ad71c8456955fb644.jpg)  
(a) Low working-memory load class.

![](images/33fab6600a36f4a69e6dbedbfafc64565f5bb7156a9f37a0dbe764507675bd7b.jpg)  
(b) High working-memory load class.  
Figure 11: Averaged signals and relevance heatmaps for class 0 (low working-memory load) and class 1 (high working-memory load). Boxes highlight localized relevance, distributed multichannel relevance, stronger late-window relevance, and class-dependent attribution differences. Each time series stands for a sensor. Time range from left to right is 3-second with 3000 points. Red and blue indicate positive and negative relevance for the explained class score, respectively.