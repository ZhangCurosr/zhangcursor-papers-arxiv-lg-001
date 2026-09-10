# Muon-C: Operator-Aligned Muon for Convolutional Kernels

Jiaxin Qing University of California, Berkeley Berkeley, CA 94720-1776, USA

Lexin Li University of California, Berkeley Berkeley, CA 94720-1776, USA

jxqing@berkeley.edu

lexinli@berkeley.edu

## Abstract

Muon replaces matrix momentum with an approximately orthogonal polar direction, but its geometry depends on the matrix representation. For convolution, standard unfolding describes a local patch map rather than the convolution operator. We introduce Muon-C, an operator-aligned optimizer that represents kernel momentum as frequency-wise channeltransfer matrices, polarizes these blocks independently, and uses a critical Fourier grid to return updates exactly to the original finite kernel support. We show that the new geometry arises from combining the block partition and Fourier coordinates. The exactpolar direction is a linear minimization oracle under the critically sampled convolution norm. Its worst-case guarantee relative to the continuous convolution-operator norm is never weaker than unfolding and is strictly stronger for 3 × 3 kernels. On CIFAR-10 flow matching with matched applied-update RMS, Muon-C reaches 9.87 FID at 40k iterations, compared with 22.26 for unfolded Muon and 51.31 for Adam. It reaches their final quality using 0.62× and 0.64× their model FLOPs, respectively. Under equal tuning budgets, Muon-C achieves 3.42 FID. Gains persist across data scales and transfer to classification across convolutional architectures.

Keywords: Muon, Optimization, Convolutional kernels, Flow matching, Fourier analysis.

## 1 Introduction

Since its introduction in late 2024 (Jordan et al., 2024), Muon has rapidly emerged as an important alternative to coordinate-wise optimizers such as Adam (Kingma and Ba, 2015). Rather than adapting individual coordinates, Muon reads the momentum of a weight as a matrix and replaces it with an approximately orthogonal polar direction. Its practical relevance is already visible at scale. After reducing training time in language-model speedruns, Muon was used to train a 16B mixture-of-experts model that matched an AdamW run at roughly half the compute (Liu et al., 2025). More fundamentally, recent analyses show that the exact polar update minimizes the linearized loss over a spectral-norm ball (Bernstein and Newhouse, 2024; Pethick et al., 2025; Chen et al., 2025). Follow-up work further studies the role of this spectral geometry (Huang et al., 2026; Shumaylov et al., 2026). This interpretation makes a central fact explicit: Muon’s geometry depends on which quantities are treated as the rows and columns of the matrix it orthogonalizes.

This dependence, however, creates a fundamental ambiguity for convolutional kernels. A kernel $\bar { W } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } } \times k _ { h } \times k _ { w } }$ has no canonical matrix representation, and the standard practice is to flatten it into a $C _ { \mathrm { o u t } } \times \left( C _ { \mathrm { i n } } k _ { h } k _ { w } \right)$ matrix before applying Muon. The unfolding preserves every coeficient, but it is not neutral bookkeeping. It makes Muon act on a local patch-to-output map and thereby selects a particular optimizer geometry. Resolving this ambiguity matters because convolution remains a central component of modern learning systems, including the U-Nets used in difusion and flow matching (Ronneberger et al., 2015; Lipman et al., 2023), ResNets (He et al., 2016), and ConvNeXt-style models (Liu et al., 2022; Woo et al., 2023). More broadly, how Muon should act on higher-order parameters has only recently become an explicit optimizer-design question (Bogachev et al., 2026).

We take an operator-aligned view of this problem. A convolutional kernel parameterizes a translation-equivariant linear operator. Translation equivariance identifies Fourier modes as the natural invariant subspaces of this operator. In the Fourier basis, convolution is block diagonal, with one $C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } }$ channel-transfer matrix at each spatial frequency. These transfer matrices are the linear maps implemented by convolution on its translation modes. This observation leads to our central design principle: For a structured parameter, Muon should act on matrices that reflect the natural linear maps of the underlying operator rather than on an arbitrary tensor unfolding.

Adopting this principle, in this article, we introduce Muon-C, an operator-aligned extension of Muon for convolutional kernels. Muon-C involves three key components. The first is Fourier/operator alignment. Muon-C transforms kernel momentum along its spatial axes into frequency-indexed channel maps, so the matrices seen by Muon correspond directly to the maps implemented by convolution on individual translation modes. The resulting representation is independent of feature-map resolution and follows from the operator’s symmetry. The second is independent blockwise polarization. Muon-C applies the polar operation separately to each frequency-wise channel map, instead of applying one global polar operation to the transformed kernel. This distinction is essential because a Fourier change of basis alone does not define a new optimizer. The polar map is unitarily equivariant, so transforming the unfolded momentum, applying one global polar operation, and transforming back returns exactly the ordinary unfolded-Muon direction. The new geometry therefore comes from changing the independently constrained block partition, not from Fourier coordinates by themselves. The third is critical frequency sampling. Muon-C uses exactly a $k _ { h } \times k _ { w }$ Fourier grid, with one frequency sample per stored spatial ofset, to make independent block updates compatible with finite kernel support. The dimensionmatched DFT is unitary and bijective over the stored coeficients. It permits independent frequency-wise polarization and maps every resulting update exactly back to the original $k _ { h } \times k _ { w }$ support without cropping. The critical grid is therefore the smallest frequency representation that remains bijective over every stored coeficient matrix, and its size is independent of feature-map resolution. Figure 1 summarizes why the latter two components are essential.

Theoretically, we provide a rigorous characterization of Muon-C and its relationship to the convolution operator. First, we show that unitary row and column transformations cannot alter the global Muon direction, which isolates the block partition as the essential optimizer-design choice. Second, we show that the critical Fourier grid supplies a unitary, bijective, and support-preserving representation under which Muon-C is the exact linear minimization oracle for the critically sampled convolution norm. Finally, relative to the continuous convolution-operator norm, we establish a worst-case oracle guarantee for Muon-C, such that it is never weaker than standard unfolding for any finite kernel size, and is strictly stronger for the common 3 × 3 case, with a guaranteed fraction of the continuousoracle optimum improved from 1/3 for unfolding to 9/25 for Muon-C.

![](images/152fb9065ab61176d38776cd6298ee0ceaec7f60a8d9926c313de7e021044db1.jpg)  
Figure 1: Two ingredients are essential for frequency-domain Muon. Independent blockwise polarization changes Muon’s geometry beyond a mere Fourier change of coordinates, whereas critical frequency sampling ensures that the resulting update maps exactly back to the original finite kernel support.

Empirically, we conduct controlled experiments across generative and discriminative learning tasks. On CIFAR-10 flow matching with matched applied-update RMS, Muon-C reaches 9.87 FID at 40k iterations, compared with 22.26 for unfolded Muon and 51.31 for Adam. It reaches the baselines’ 400k-step final quality using only 0.62× and 0.64× their model FLOPs, corresponding to savings of 38% and 36%. Under equal optimizerspecific sweep budgets, Muon-C obtains 3.42 FID on CIFAR-10, compared with 3.54 for unfolded Muon and 3.83 for Adam. On ImageNet-1k-32, it improves both the 40k-step FID and the tuned final FID, reaching 21.14 and 12.02 compared with 28.02 and 13.24 for unfolded Muon. On ImageNet-100, Muon-C attains the highest validation accuracy across ResNet-34, ResNet-50, and ConvNeXtV2-T despite their widely diferent exposure to the convolutional optimizer route. A matched comparison with Muon-S, based on the spatial-block Conv2D duality map of Bernstein and Newhouse (2025), shows a stronger early trajectory for translation-frequency blocks than for spatial-ofset blocks.

Our contributions are fourfold.

• We establish operator-aligned matrix geometry as a design principle for extending Muon to convolution. Translation equivariance identifies frequency-wise channeltransfer matrices as the natural linear maps, while independent blockwise polarization defines the new geometry.

• We develop Muon-C, a finite-support realization of this operator-aligned geometry. It combines Fourier/operator alignment and independent frequency-wise polarization with critical $k _ { h } \times k _ { w }$ sampling, enabling blockwise Muon updates that map exactly back to the original kernel support.

• We provide an operator-level theoretical characterization of Muon-C, showing that it is the exact linear minimization oracle under the critically sampled convolution norm and enjoys a stronger worst-case guarantee relative to standard unfolding.

• We demonstrate that Muon-C consistently improves optimization eficiency across generative and classification settings, reaching comparable loss or sample quality substantially earlier than unfolded Muon and Adam/AdamW. A matched spatial-block ablation identifies the contribution of frequency organization.

The rest of the article is organized as follows. Section 2 reviews related work, and Section 3 introduces notation and the spectral-norm interpretation of Muon. Section 4 develops the operator-aligned view of convolution and motivates frequency-wise block geometry. Section 5 presents Muon-C, including the critical-grid construction, theoretical guarantees, and practical implementation. Section 6 reports the empirical evaluations, along with robustness studies and mechanism ablations. Section 7 concludes with a discussion of future directions. The Appendix collects additional results and proofs.

## 2 Related Work

Muon and norm-aware updates. Muon applies an approximate polar map to matrix momentum, and recent analyses connect this operation to spectral-norm steepest descent and norm-constrained linear minimization (Jordan et al., 2024; Bernstein and Newhouse, 2024; Pethick et al., 2025; Chen et al., 2025). These views make the matrix shape part of the optimizer because the chosen rows, columns, and block boundaries specify the spectral trust region. Adam/AdamW instead adapt coordinates individually and therefore do not require an explicit matrix geometry (Kingma and Ba, 2015; Loshchilov and Hutter, 2019).

Modular duality for convolution. Bernstein and Newhouse (2025) derive layerwise duality maps from norms chosen to reflect layer semantics and compose them across neural architectures. Their Conv2D map polarizes each spatial-ofset channel matrix independently. Muon-S adopts this spatial-block geometry under our matched-update protocol. Muon-C instead polarizes translation-frequency channel maps, with critical sampling preserving finite kernel support.

Tensor and structured optimization. K-FAC and Shampoo exploit Kronecker or tensormode structure to construct preconditioners (Martens and Grosse, 2015; Gupta et al., 2018). Concurrent Tensorion constructs tensor-aware norm oracles through adaptively selected matrix unfoldings (Bogachev et al., 2026). Muon-C takes a diferent, operator-aligned approach. Rather than selecting a matrix representation from the tensor structure alone, it derives the matrices seen by Muon from the linear operator parameterized by the kernel. For convolution, translation equivariance identifies the direct-sum Fourier representation, whose blocks are frequency-indexed channel-transfer matrices. The two approaches therefore encode diferent notions of structure. Tensorion adapts Muon to the tensor organization of the parameters, whereas Muon-C aligns Muon with the operator those parameters implement.

Spectral structure of convolution. Fourier analysis has long been used to compute, constrain, or bound singular values of convolutional layers (Miyato et al., 2018; Sedghi et al., 2019; Singla and Feizi, 2021). We use the same transfer-matrix structure to design an optimizer update. Unlike prior uses of convolution spectra as constraints or diagnostics,

Muon-C uses the transfer blocks to define the matrices that receive independent optimizer updates. Critical sampling provides a dimension-matched representation of these frequencywise channel maps for a finite stored kernel.

## 3 Muon as a Spectral-Norm Oracle

We begin with the interpretation of Muon that motivates our operator-aligned construction. The key observation is that Muon’s polar update is the exact linear minimization oracle under a spectral-norm constraint. This interpretation makes the matrix representation part of the optimizer geometry, because the matrices, and more generally the independently constrained matrix blocks, on which Muon acts determine the geometry of its update.

We adopt the following notation throughout. For a matrix A, let $\| A \| _ { \mathrm { o p } }$ denote its spectral norm, $A ^ { * }$ its conjugate transpose, and $\langle A , B \rangle _ { F } = \operatorname { t r } ( A ^ { * } B )$ the Frobenius inner product; we take its real part in optimization objectives. For real-valued matrices, $A ^ { * }$ reduces to the transpose. For a convolutional kernel, write $[ k ] = \{ 0 , \ldots , k - 1 \} , \Omega _ { k } = [ k _ { h } ] \times$ $[ k _ { w } ]$ , and $n = k _ { h } k _ { w }$ . We represent the kernel as $W = \{ W _ { u v } \} _ { ( u , v ) \in \Omega _ { k } }$ , with $W _ { u v } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } } }$ ， or equivalently $W \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } } \times k _ { h } \times k _ { w } }$

Muon replaces the momentum of a matrix-shaped weight by its polar factor. For $M \in$ $\mathbb { C } ^ { m \times d }$ with compact singular value decomposition $M = P \Sigma Q ^ { * }$ , define the hard-polar map $\operatorname { p o l a r } ( M ) = P Q ^ { * }$ . Given gradient $G ^ { t }$ , momentum coeficient $\beta ,$ learning rate $\eta ,$ and a shape-dependent scale s, a Muon step takes the form

$$
M ^ { t } = \beta M ^ { t - 1 } + ( 1 - \beta ) G ^ { t } , \qquad X ^ { t + 1 } = X ^ { t } - \eta s \mathrm { p o l a r } ( M ^ { t } ) .
$$

In practice, computing an exact singular value decomposition at every iteration is expensive, so the polar factor is approximated using a small number of Newton-Schulz iterations on the normalized momentum (Jordan et al., 2024; Higham, 2008).

The polar direction has a precise optimization interpretation. Given a descent direction budget measured in the spectral norm, one solution of min $\lvert \lvert U \rvert \rvert _ { \mathrm { o p } } { \le } \rho  \langle M , U \rangle _ { F }$ is $U ^ { \star } =$ $- \rho \mathrm { p o l a r } ( M )$ , with the usual nonuniqueness when M is rank deficient. Thus, Muon can be viewed as a linear minimization oracle, or equivalently, a steepest linearized descent direction, under a spectral-norm trust region, rather than merely as a heuristic normalization of the momentum (Bernstein and Newhouse, 2024; Pethick et al., 2025; Chen et al., 2025).

This interpretation makes the matrix representation an intrinsic part of the optimizer. The constraint set $\{ U : \| U \| _ { \mathrm { o p } } \leq \rho \}$ depends on which quantities are assigned to the rows and columns of the matrix. More generally, if a parameter is represented by multiple independently constrained matrix blocks, their partition determines a product of spectralnorm balls and hence a diferent linear minimization oracle. Two invertible representations of the same coeficients can therefore induce diferent Muon updates even though they parameterize the same underlying object. For an ordinary matrix, the representation is usually given. For a structured parameter such as a convolutional kernel, however, which matrices Muon should act on becomes an optimizer-design choice. Section 4 develops this choice from the translation-equivariant operator implemented by convolution.

## 4 From Muon to Operator-Aligned Geometry for Convolution

We now develop the operator-aligned matrix geometry for convolutional kernels. The key question is which matrices should define Muon’s spectral geometry for a convolutional operator. We proceed in three steps. We first show that standard unfolding corresponds to a local patch-to-output map, then use translation equivariance to identify frequency-wise channel-transfer matrices as the natural linear maps implemented by convolution, and finally show that these maps induce a genuinely diferent Muon geometry only when they are polarized as independent blocks.

## 4.1 Muon Geometry Depends on Matrix Representation

The standard representation concatenates the spatial coeficient matrices into

$$
B _ { W } = \left[ W _ { 0 0 } \quad W _ { 0 1 } \quad \cdots \quad W _ { k _ { h } - 1 , k _ { w } - 1 } \right] \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times k _ { h } k _ { w } C _ { \mathrm { i n } } } .
$$

This unfolding represents the linear map from a vectorized local input patch to one output vector. Applying Muon to $B _ { W }$ therefore defines an exact spectral-norm oracle for this local patch-to-output geometry. However, this representation, by choosing the rows and columns on which the polar update acts, selects a particular spectral-norm geometry for the kernel.

The convolution kernel also parameterizes a larger translation-equivariant linear operator over the entire feature map. On an $H \times W$ grid, forming this operator explicitly would produce a $C _ { \mathrm { o u t } } H W \times C _ { \mathrm { i n } } H W$ matrix, which is resolution dependent and impractical to polarize during training. Its translation symmetry, however, decomposes this large operator into small, resolution-independent channel maps. These maps provide the operator-aligned alternative to the local patch geometry, as we develop next.

## 4.2 Translation Equivariance Identifies Fourier Channel Maps

Translation equivariance identifies the Fourier basis as the natural representation of the convolution operator. Because convolution commutes with every spatial shift, it preserves the joint eigenspaces of the shift operators, which are precisely the Fourier modes. The layer operator is therefore block diagonal in this basis, with one $C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } }$ channel map at each spatial frequency. Figure 2 illustrates this decomposition for a circular layer with $C _ { \mathrm { i n } } = C _ { \mathrm { o u t } } = 3 , \mathrm { ~ a ~ 3 \times 3 ~ }$ kernel, and a $6 \times 6$ feature grid.

Let x be an input feature map on the infinite lattice $ { \mathbb { Z } } ^ { 2 }$ . Define the convolutional layer A<sub>W</sub> by

$$
( \mathcal { A } _ { W } x ) ( p ) = \sum _ { ( u , v ) \in \Omega _ { k } } W _ { u v } x \big ( p - ( u , v ) \big ) ,
$$

and let $T _ { s }$ be the shift $( T _ { s } x ) ( p ) = x ( p - s )$ . Direct substitution gives A $T _ { s } = T _ { s } \mathcal { A } _ { W }$ for every $s \in \mathbb { Z } ^ { 2 }$ . Thus convolution cannot mix the distinct joint eigenspaces of the shift family. For a fixed frequency $\theta \in [ 0 , 2 \pi ) ^ { 2 }$ , the complex exponential $e _ { \theta } ( p ) = e ^ { \mathrm { i } \langle p , \theta \rangle }$ satisfies $T _ { s } e _ { \theta } =$ $e ^ { - \mathrm { i } \langle s , \theta \rangle } e _ { \theta }$ . Each $e _ { \theta }$ is a simultaneous eigenvector of every shift, and diferent frequencies carry diferent eigenvalue patterns. Hence $\boldsymbol { \mathcal { A } } _ { W }$ acts independently within each frequency.

## a Shifts share one eigenbasis, so the layer cannot mix frequencies

![](images/f15dba36fee3ef21183f4cebab495cd593951218ba314b299eb9c02fed550a21.jpg)

b One basis works for every convolution, a generic basis for none  
![](images/05d34c87ad4adaee10659ee3f5d4a7b4e95a4b874214db164dc5baccafb1781a.jpg)  
Figure 2: Translation equivariance identifies the Fourier channel maps. (a) The Fourier transform simultaneously diagonalizes spatial shifts, decomposing convolution into resolution-independent frequency-wise $C _ { \mathrm { o u t } } { \times } C _ { \mathrm { i n } }$ channel-transfer blocks. (b) The same Fourier basis block diagonalizes diferent convolutional kernels, whereas a generic orthonormal basis does not.

Only the channel index remains within a frequency. Feeding $x ( p ) = e _ { \theta } ( p ) c$ for $c \in \mathbb { C } ^ { C _ { \mathrm { i r } } }$ into the layer gives

$$
\mathcal { A } _ { W } \big ( e _ { \theta } c \big ) = e _ { \theta } A _ { W } ( \theta ) c , \qquad A _ { W } ( \theta ) = \sum _ { ( u , v ) \in \Omega _ { k } } W _ { u v } e ^ { - \mathrm { i } ( u \theta _ { 1 } + v \theta _ { 2 } ) } .\tag{1}
$$

The $C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } }$ matrix $A _ { W } ( \theta )$ is the channel-transfer matrix at frequency θ. These matrices are determined entirely by the stored kernel coeficients and are independent of feature-map resolution (Sedghi et al., 2019; Singla and Feizi, 2021). They therefore provide a resolutionindependent representation of the linear maps implemented by convolution on its invariant translation modes.

This decomposition has two consequences for the choice of Muon geometry. First, the Fourier basis simultaneously block-diagonalizes every translation-equivariant convolution, whereas a generic orthonormal basis does not preserve this common block structure. Figure 2(b) illustrates this distinction for two kernels. Second, and more importantly, each $A _ { W } ( \theta )$ is the channel map actually implemented by convolution on a translation mode. These invariant channel maps provide an operator-aligned choice on which to define Muon’s spectral geometry.

## 4.3 Changing Basis Is Not Enough: Blocks Define the Geometry

Section 4.2 identifies the frequency-wise channel-transfer matrices as the natural operatoraligned maps for Muon. However, expressing the kernel in the Fourier basis alone does not define a new Muon geometry. If the transformed momentum is treated as a single matrix and polarized globally, the resulting direction is exactly the ordinary unfolded-Muon direction, as shown by the next result.

Proposition 1 (Unitary equivariance of the polar map) Let $\ b { X } \in \mathbb { C } ^ { m \times d }$ , and let $L \in$ $\mathbb { C } ^ { m \times m }$ and $R \in \mathbb { C } ^ { d \times d }$ be unitary. For the canonical partial polar factor,

$$
\operatorname { p o l a r } ( L X R ) = L \operatorname { p o l a r } ( X ) R .
$$

Consequently, $L ^ { * } \operatorname { p o l a r } ( L X R ) R ^ { * } = \operatorname { p o l a r } ( X )$

For a convolutional kernel, concatenating the $n = k _ { h } k _ { w }$ spatial coeficient matrices into the local patch map yields

$$
\boldsymbol { B } _ { M } = \left[ M _ { 0 0 } \quad M _ { 0 1 } \quad \cdot \cdot \cdot \quad M _ { k _ { h } - 1 , k _ { w } - 1 } \right] \in \mathbb { C } ^ { C _ { \mathrm { o u t } } \times n C _ { \mathrm { i n } } } .
$$

Mixing the spatial ofsets by an orthonormal DFT is the particular right-unitary transformation $B _ { M } \ \mapsto \ B _ { M } R _ { \mathrm { D F T } }$ Proposition 1 implies that applying a single polar map and then undoing this transformation returns $\mathrm { p o l a r } ( B _ { M } )$ . Including the descent sign gives $- \operatorname { p o l a r } ( B _ { M } )$ . Thus, this unitary row/column transformation cannot by itself produce a new global Muon geometry. What changes the geometry is the partition into independently polarized blocks.

To see how a distinct Muon geometry arises, we now shift attention from the coordinate representation to the partition into independently polarized blocks. Let R map a kernel to a collection of matrices, and let $\mathcal { R } ^ { - 1 }$ return those matrices to parameter space. Then

$$
\mathcal { U } _ { \mathcal { R } } ( M ) = - \mathcal { R } ^ { - 1 } \big ( \big \{ \mathrm { p o l a r } \big ( [ \mathcal { R } ( M ) ] _ { b } \big ) \big \} _ { b } \big ) .
$$

For the left/right unitary representations considered here, including the spatial $\mathrm { D F T }$ acting on the spatial-ofset block columns, R preserves the coeficient-space inner product and the global polar direction. The essential choice is therefore not this particular coordinate transformation but the partition of $\mathcal { R } ( M )$ into independently polarized blocks.

The partition changes the geometry because it changes the constraint set. A single block imposes one spectral-norm constraint $\| B _ { U } \| _ { \mathrm { o p } } \leq \rho _ { \mathrm { : } }$ , whereas a partition into multiple blocks imposes one constraint per block, producing a product of spectral-norm balls. These constraint sets define genuinely diferent linear minimization oracles even when the underlying coordinate transformation is unitary. Blockwise polarization also provides a useful interpretation of this geometry. Each polarized block is a partial isometry, so the blocks contribute at a common spectral scale rather than retaining the relative magnitudes present in the momentum. Our proposed Muon-C therefore extends Muon’s within-matrix spectral equalization to the operator-aligned decomposition across translation modes.

We focus on three block partitions of the same convolutional kernel. The first is global unfolding, which treats $B _ { M }$ as a single block and is the exact linear minimization oracle for the local patch norm $\| U \| _ { \mathrm { p a t c h } } = \| B _ { U } \| _ { \mathrm { o p } }$ . The second is critical-Fourier partitioning, used by Muon-C, which independently polarizes one $C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } }$ channel map at each critical translation frequency, thereby aligning the blocks with the invariant maps of the convolution operator. The third is spatial partitioning, used by Muon-S, which independently polarizes one $C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } }$ channel matrix at each stored spatial ofset. This is the spatial-block polar geometry of the Conv2D duality map in Bernstein and Newhouse (2025), up to layerwise scaling. Muon-S matches Muon-C in block count and block shape while organizing blocks by spatial ofset, providing a baseline for isolating the role of Fourier organization. Figure 3 gives a minimal two-coeficient example in which these three partitions produce pairwise non-collinear directions, demonstrating that changing the block partition alone can change the Muon update.

![](images/b5db6c129bf0984eb10877c90cb8f69709222ca53523038a54efc0450568ac91.jpg)  
Figure 3: Diferent block partitions induce diferent Muon directions. For the two-coeficient momentum $M = ( 1 , 2 )$ , global unfolding, spatial partitioning, and critical-Fourier partitioning yield the directions $- ( 1 , 2 ) / \sqrt { 5 } , - ( 1 , 1 )$ , and $( 0 , - { \sqrt { 2 } } )$ , respectively. The three directions are pairwise non-collinear, illustrating that changing the independently polarized block partition changes the Muon geometry even when the underlying kernel coeficients are the same.

The operator view therefore identifies independent frequency-wise channel maps, rather than Fourier coordinates alone, as the appropriate Muon geometry for convolution. One challenge remains. A finite $k _ { h } \times k _ { w }$ kernel has only $k _ { h } k _ { w }$ free coeficient matrices, whereas its transfer function is defined over a continuum of frequencies. Section 5 resolves this finitesupport mismatch through critical frequency sampling, completing the Muon-C construction by combining Fourier/operator alignment, independent frequency-wise polarization, and exact preservation of the original kernel support.

## 5 Muon-C: Critical Fourier Geometry for Finite Convolutional Kernels

We next develop Muon-C and show how the operator-aligned geometry can be realized for a finite convolutional kernel. We proceed in four steps. We first introduce the critical Fourier grid that makes independent frequency-wise polarization compatible with finite kernel support. We then define the resulting Muon-C update, and establish its exact linear-oracle characterization as well as its guarantee relative to the continuous convolution-operator geometry. Finally, we present the complete algorithm along with practical implementation.

## 5.1 From Continuous Operator Geometry to Critical Sampling

We first formalize the natural operator geometry that Muon-C seeks to approximate. For a kernel $U ,$ the transfer matrix $A _ { U } ( \theta )$ in (1) describes the channel map implemented by convolution at frequency θ. This motivates the convolution norm

$$
\| U \| _ { \mathrm { c o n v } } = \operatorname* { s u p } _ { \theta \in [ 0 , 2 \pi ) ^ { 2 } } \| A _ { U } ( \theta ) \| _ { \mathrm { o p } } .
$$

This quantity is the induced $\ell _ { 2 }$ norm of multichannel convolution on the infinite spatial lattice. It also upper-bounds the corresponding operator norm on any finite zero-padded grid, while for circular convolution it reduces to a maximum over discrete frequencies. Moreover, changing from corner-indexed to centered kernel ofsets only multiplies $A \upsilon ( \theta )$ by a unit-modulus scalar and therefore leaves its singular values unchanged. Thus, $\lVert \cdot \rVert _ { \mathrm { c o n v } }$ is an intrinsic, resolution-independent property of the convolutional kernel and provides the operator-level analogue of the matrix spectral norm underlying ordinary Muon.

Combining this operator norm with the finite kernel support gives the natural convolutional oracle

$$
\operatorname* { m i n } _ { U \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } } \times k _ { h } \times k _ { w } } } ~ \langle M , U \rangle ~ \mathrm { s u b j e c t ~ t o } ~ \| U \| _ { \mathrm { c o n v } } \le \rho .
$$

This is the convolutional counterpart of the spectral-norm oracle that defines Muon in Section 3. An idealized full-grid construction would treat the channel map at every frequency as an independent variable, even though a finite kernel must generate all of these maps from the same $n = k _ { h } k _ { w }$ stored coeficient matrices. Independently polarizing the full-grid blocks therefore generally yields an inverse transform with nonzero coeficients outside $\Omega _ { k }$ so the result is not a feasible finite-support update without projection. Appendix A gives the formal full-grid oracle and its proof, while Appendix C measures this obstruction in trained U-Net momenta.

Muon-C resolves this mismatch using the critical $k _ { h } \times k _ { w }$ Fourier grid. The grid contains exactly $n = k _ { h } k _ { w }$ frequency samples, matching the number of stored coeficient matrices, and the corresponding spatial DFT is unitary and bijective on those coeficients. Its inverse maps every collection of critical-grid blocks exactly back to the original $k _ { h } \times k _ { w }$ support, while each block remains an evaluation of the convolutional transfer matrix $A _ { U } ( \theta )$ The critical grid therefore provides an operator-aligned, bijective, and support-preserving representation on which the frequency blocks can be polarized independently. Section 5.2 uses this representation to define the Muon-C update.

## 5.2 Muon-C Update

We now define the Muon-C geometry on the critical Fourier grid. Let

$$
\theta _ { p } = \frac { 2 \pi p } { k _ { h } } , \qquad \phi _ { q } = \frac { 2 \pi q } { k _ { w } } , \qquad p \in [ k _ { h } ] , ~ q \in [ k _ { w } ] ,
$$

giving $n = k _ { h } k _ { w }$ critical frequencies in total. The corresponding critically sampled convolution norm is

$$
\left\| U \right\| _ { \mathrm { s a m p } } = \operatorname* { m a x } _ { p \in [ k _ { h } ] , q \in [ k _ { w } ] } \left\| A _ { U } ( \theta _ { p } , \phi _ { q } ) \right\| _ { \mathrm { o p } } .\tag{2}
$$

This norm measures the largest channel-transfer operator norm over the critical grid and defines the trust-region geometry used by Muon-C.

To express this geometry in the coordinates used by the algorithm, let $\mathcal { F } _ { \Omega _ { k } }$ denote the orthonormal two-dimensional DFT over the $k _ { h } \times k _ { w }$ kernel grid, and write

$$
\widehat { U } ( p , q ) = \left[ \mathcal { F } _ { \Omega _ { k } } ( U ) \right] ( p , q ) = \frac { 1 } { \sqrt { n } } A _ { U } ( \theta _ { p } , \phi _ { q } ) .
$$

It follows that

$$
\left. U \right. _ { \mathrm { s a m p } } = \sqrt { n } \operatorname* { m a x } _ { p , q } \left. \widehat { U } ( p , q ) \right. _ { \mathrm { o p } } .\tag{3}
$$

The factor $\sqrt { n }$ results from the orthonormal DFT convention and rescales the corresponding trust-region radius without changing its blockwise polar direction. Importantly, up to this common normalization, the matrices $\widehat { U } ( p , q )$ retain their operator interpretation as evaluations of the same transfer matrix $A _ { U } ( \theta )$ that defines the continuous convolution norm $\| U \| _ { \mathrm { c o n v } } .$ . At the same time, the critical DFT is unitary and bijective on the stored kernel coeficients.

In the representation framework of Section 4.3, Muon-C therefore takes $\mathcal { R } = \mathcal { F } _ { \Omega _ { k } }$ and treats each $C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } }$ frequency-wise channel map as an independent block. Given kernel momentum $M ^ { t }$ , let

$$
\widehat { M } ^ { t } = \mathcal { F } _ { \Omega _ { k } } ( M ^ { t } ) .
$$

Muon-C polarizes each critical-frequency block independently,

$$
\widehat { O } ^ { t } ( p , q ) = \mathrm { p o l a r } \left( \widehat { M } ^ { t } ( p , q ) \right) , \qquad p \in [ k _ { h } ] , q \in [ k _ { w } ] ,
$$

and maps the resulting blocks back to the kernel domain,

$$
O ^ { t } = { \mathcal { F } } _ { \Omega _ { k } } ^ { - 1 } ( \widehat { O } ^ { t } ) .
$$

Figure 4 illustrates this update pipeline, and contrasts it with global unfolded Muon that applies a single polar map to the local patch matrix.

Because the critical DFT is bijective, $O ^ { t }$ has exactly the same $k _ { h } \times k _ { w }$ spatial support as the original kernel and requires no cropping or support projection. Moreover, for realvalued momentum, the Fourier blocks are conjugate symmetric, and the canonical polar map preserves this symmetry. Hence the inverse transform returns a real-valued kernel update. Muon-C therefore combines operator-aligned frequency blocks, independent blockwise polarization, and exact preservation of the finite kernel support. Section 5.3 shows that this direction is the exact linear minimization oracle for the sampled geometry in (2) and quantifies its guarantee relative to the continuous convolution norm.

## 5.3 Theoretical Properties of Muon-C

We next establish two properties of Muon-C. First, its update is the exact linear minimization oracle under the sampled convolution norm. Second, this sampled norm provides a controlled approximation to the continuous convolution norm, with a worst-case approximation factor no larger than that of global unfolding and strictly smaller for $3 \times 3$ kernels.

![](images/6d763436860cf72b3893e5399fc5decd87b1a9d140d247fa155845f77f11df1a.jpg)  
Figure 4: Muon-C update on the critical Fourier grid. Muon-C transforms the kernel momentum into frequency-wise channel maps on the critical grid, polarizes these blocks independently, and applies the inverse transform to obtain an update on the original finite kernel support. In contrast, unfolded Muon treats the local patch map as a single block and applies one global polar operation.

We begin with the exact oracle characterization. Recall from (3) that, under the orthonormal kernel DFT, $\begin{array} { r } { \| U \| _ { \mathrm { s a m p } } = \sqrt { n } \operatorname* { m a x } _ { p , q } \| \widehat { U } ( p , q ) \| _ { \mathrm { o p } } } \end{array}$ , with $n = k _ { h } k _ { w }$ . The samplednorm constraint therefore separates across the critical-frequency blocks, yielding the following result.

Proposition 2 (Exact sampled-convolution oracle) Let $M \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } } \times k _ { h } \times k _ { w } }$ be a kernel momentum and let $\widehat { M } = \mathcal { F } _ { \Omega _ { k } } ( M )$ . Then one solution of

$$
\operatorname* { m i n } _ { U } \langle M , U \rangle \qquad s u b j e c t \ t o \qquad \| U \| _ { \mathrm { s a m p } } \leq \rho
$$

is given in the Fourier domain by

$$
\widehat { U } _ { f } ^ { \star } ( p , q ) = - \frac { \rho } { \sqrt { n } } \mathrm { p o l a r } \Big ( \widehat { M } ( p , q ) \Big ) , \qquad p \in [ k _ { h } ] , q \in [ k _ { w } ] .\tag{4}
$$

Up to the common scale $\rho / { \sqrt { n } }$ , Proposition 2 is exactly the Muon-C direction defined in Section 5.2. The result follows from Parseval’s identity and spectral-nuclear norm duality, applied independently to each critical-frequency block. For real-valued kernels, conjugate symmetry of the Fourier blocks is preserved by the canonical polar map, so the resulting spatial update remains real valued.

Having established exactness for the sampled norm, we next investigate how closely this norm controls the continuous convolution norm. To compare the three block partitions introduced in Section 4.3, recall the local patch norm $\| U \| _ { \mathrm { p a t c h } } = \| B _ { U } \| _ { \mathrm { o p } }$ , associated with global unfolded Muon, and define the spatial-block norm

$$
\| U \| _ { \mathrm { s p } } = \operatorname* { m a x } _ { ( u , v ) \in \Omega _ { k } } \| U _ { u v } \| _ { \mathrm { o p } } ,
$$

associated with Muon-S. Muon-C is similarly associated with the sampled convolution norm $\| \cdot \| _ { \mathrm { s a m p } }$ in (2).

For a square kernel, the Conv2D norm of Bernstein and Newhouse (2025) satisfies

$$
\mathrm { C o n v 2 D . n o r m } ( U ) = n \sqrt { \frac { C _ { \mathrm { i n } } } { C _ { \mathrm { o u t } } } } \Vert U \Vert _ { \mathrm { s p } } , \qquad n = k ^ { 2 } ,
$$

since $\| A \| _ { \mathrm { R M S \to R M S } } = \sqrt { C _ { \mathrm { i n } } / C _ { \mathrm { o u t } } } \| A \| _ { \mathrm { o p } }$ . Its duality map is n $^ { - 1 } \sqrt { C _ { \mathrm { o u t } } / C _ { \mathrm { i n } } }$ polar $( M _ { u v } )$ at each spatial ofset. The kernel-area factor n also appears in the spatial upper bound below.

The remaining issue is how much the transfer matrix can grow between the critical-grid samples. Trigonometric interpolation controls this growth through the Lebesgue constant. Define the one-dimensional cardinal functions

$$
\ell _ { p } ^ { ( k ) } ( \theta ) = \frac { 1 } { k } \sum _ { r = 0 } ^ { k - 1 } e ^ { - \mathrm { i } r ( \theta - 2 \pi p / k ) } , \qquad p \in [ k ] ,
$$

and the corresponding Lebesgue constant

$$
\Lambda _ { k } = \operatorname* { s u p } _ { \theta \in [ 0 , 2 \pi ) } \sum _ { p = 0 } ^ { k - 1 } \left| \ell _ { p } ^ { ( k ) } ( \theta ) \right| .
$$

The next theorem places all three finite-support geometries on a common scale relative to the continuous convolution norm.

Theorem 3 (Unified finite-support norm comparison) Let $n = k _ { h } k _ { w }$ . Every kernel U supported on $\Omega _ { k }$ satisfies

$$
\begin{array} { r } { \| U \| _ { \mathrm { s p } } \leq \| U \| _ { \mathrm { p a t c h } } \leq \| U \| _ { \mathrm { s a m p } } \leq \| U \| _ { \mathrm { c o n v } } , } \end{array}\tag{5}
$$

and

$$
\begin{array} { r } { \| U \| _ { \mathrm { c o n v } } \leq \operatorname* { m i n } \left\{ n \| U \| _ { \mathrm { s p } } , \sqrt { n } \| U \| _ { \mathrm { p a t c h } } , \Lambda _ { k _ { h } } \Lambda _ { k _ { w } } \| U \| _ { \mathrm { s a m p } } \right\} . } \end{array}\tag{6}
$$

Moreover,

$$
\Lambda _ { k } \leq \sqrt { k } , ~ a n d h e n c e ~ \Lambda _ { k _ { h } } \Lambda _ { k _ { w } } \leq \sqrt { n } .\tag{7}
$$

For a $3 \times 3$ kernel, $\Lambda _ { 3 } = 5 / 3$ , so the corresponding worst-case approximation factors $f o r$ Muon-S, global unfolding, and Muon-C are 9, 3, and $2 5 / 9$ , respectively.

Theorem 3 quantifies how tightly each tractable geometry controls the continuous convolution norm. Global unfolding controls the local patch map with distortion $\sqrt { n }$ , whereas Muon-C directly controls the critical-frequency channel maps with distortion $\Lambda _ { k _ { h } } \Lambda _ { k _ { w } }$ . By (7), the Muon-C distortion is no larger than that of global unfolding for any finite kernel size. For the common $3 \times 3$ kernel, the inequality is strict:

$$
\underbrace { \frac { 2 5 } { 9 } } _ { \mathrm { { M u o n - C } } } < \underbrace { 3 } _ { \mathrm { { u n f o l d e d ~ M u o n } } } < \underbrace { 9 } _ { \mathrm { { M u o n - S } } } .
$$

Thus, among the three block geometries we consider, the operator-aligned critical-frequency geometry provides the tightest worst-case control of the continuous convolution norm.

The norm comparison also yields a direct guarantee relative to the ideal continuous convolution oracle. Let $U _ { g } ^ { \star } , U _ { f } ^ { \star }$ , and $U _ { s } ^ { \star }$ denote exact radius-ρ minimization oracles for the patch, sampled Fourier, and spatial-block norms, respectively, and define

$$
\kappa _ { g } = \sqrt { n } , \qquad \kappa _ { f } = \Lambda _ { k _ { h } } \Lambda _ { k _ { w } } , \qquad \kappa _ { s } = n .
$$

Corollary 4 (Continuous-convolution oracle guarantee) For $r \in \{ \mathrm { g } , \mathrm { f } , \mathrm { s } \}$ , let $\widetilde { U } _ { r } =$ $U _ { r } ^ { \star } / \kappa _ { r }$ . Then $\smash { \widetilde { U } _ { r } }$ is feasible for the radius-ρ continuous convolution-norm ball and satisfies

$$
- \langle M , \widetilde { U } _ { r } \rangle \ge \frac { 1 } { \kappa _ { r } } \operatorname* { m a x } _ { \| V \| _ { \mathrm { c o n v } } \le \rho } - \langle M , V \rangle .\tag{8}
$$

Corollary 4 gives $1 / \kappa _ { r }$ as the fraction of the optimal continuous-oracle linear objective guaranteed by each surrogate geometry after rescaling to the same convolution-norm ball. For a $3 \times 3$ kernel, these factors are

$$
\underbrace { \frac { 9 } { 2 5 } } _ { \mathrm { { M u o n - C } } } > \underbrace { \frac { 1 } { 3 } } _ { \mathrm { { u n f o l d e d ~ M u o n } } } > \underbrace { \frac { 1 } { 9 } } _ { \mathrm { { M u o n - S } } } .
$$

More generally, $\Lambda _ { k _ { h } } \Lambda _ { k _ { w } } \leq \sqrt { n }$ implies that the worst-case continuous-oracle guarantee for Muon-C is at least as strong as that of global unfolding for every finite kernel size. The evaluated classification architectures also contain $2 \times 2 , 4 \times 4$ , and $7 \times 7$ kernels. For each of these square kernel sizes, $\Lambda _ { k } ^ { 2 } \le k$ preserves the guarantee that the Muon-C distortion is no larger than the global-unfolding distortion. These guarantees concern exact, single-step linear minimization oracles.

Table 1 summarizes the three geometries, their independently polarized blocks, associated trust norms, distortion factors, and the polar shapes of their exact surrogate directions. Together, Proposition 2, Theorem 3, and Corollary 4 characterize Muon-C as an exact Muon oracle for a tractable finite-support surrogate of the convolution-operator geometry, with a worst-case norm comparison that is no weaker than global unfolding and is strictly tighter for $3 \times 3$ kernels. Proofs are given in Appendix A.

Muon-C does incur additional optimizer-side computation relative to global unfolded Muon because it performs a kernel-axis FFT and multiple channel-block polar operations. This overhead is independent of feature-map resolution but is not included in the model-FLOP accounting used in Section 6.2. We therefore separately report measured wall-clock and peak-memory overhead in Section 6.2.

## 5.4 Practical Muon-C Algorithm

Sections 5.2 and 5.3 define the Muon-C direction and characterize it as an exact linear minimization oracle under the sampled convolution norm in (2). We now turn this direction into a practical optimizer through four steps: eficient computation of the frequency-wise polar factors, calibration and routing of the resulting parameter updates, treatment of computational cost and common convolution variants under the critical-grid construction, and assembly of these elements into a complete optimization step. Together, these components preserve the operator-aligned geometry developed above while making Muon-C directly applicable to standard convolutional networks.

<table><tr><td>Geometry</td><td>Matrices seen by Muon</td><td>Trust norm</td><td>Distortion</td><td>Polar shape of exact LMO</td></tr><tr><td>Global unfolded</td><td>One local patch map  $B _ { M }$ </td><td> $\| U \| _ { \mathrm { p a t c h } }$ </td><td> $\sqrt { n }$ </td><td> $\mathrm { p o l a r } ( B _ { M } )$ </td></tr><tr><td>Muon-C</td><td>Critical-frequency channel maps</td><td> $\| U \| _ { \mathrm { s a m p } }$ </td><td> $\Lambda _ { \boldsymbol { k } _ { h } } \Lambda _ { \boldsymbol { k } _ { w } }$ </td><td> $\mathcal { F } ^ { - 1 } \{ \mathrm { p o l a r } ( \widehat { M } ) \}$ </td></tr><tr><td>Spatial ablation</td><td>One channel map per offset</td><td> $\| U \| _ { \mathrm { s p } }$ </td><td>n</td><td> $\{ \mathrm { p o l a r } ( M _ { u v } ) \} _ { u , v }$ </td></tr></table>

Table 1: Comparison of three Muon geometries under the continuous convolution norm. Muon-C has a worst-case approximation factor no larger than global unfolding, and strictly smaller for $3 \times 3$ kernels. Muon-S serves as the matched spatial-block ablation. The last column shows the polar shape of each exact LMO, omitting signs and scales.

The first step is the eficient computation of the Muon-C direction. Muon-C requires no additional optimizer state beyond the momentum already maintained by Muon. Given the momentum $M ^ { t }$ , we apply the $k _ { h } \times k _ { w }$ orthonormal DFT $\mathcal { F } _ { \Omega _ { k } }$ along the two spatial kernel axes, producing the critical-frequency blocks

$$
\widehat { M } ^ { t } = \mathcal { F } _ { \Omega _ { k } } ( M ^ { t } ) .
$$

The polar factor of each $C _ { \mathrm { o u t } } { \times } C _ { \mathrm { i n } }$ block $\widehat { M } ^ { t } ( p , q )$ is then approximated using a small number of Newton-Schulz iterations, and the resulting blocks are transformed back with $\mathcal { F } _ { \Omega _ { k } } ^ { - 1 }$ . Thus, the practical computation directly implements the blockwise direction defined in Section 5.2, with the exact polar map replaced by its Newton-Schulz approximation. Importantly, the number of frequency blocks is $k _ { h } k _ { w }$ and therefore depends only on the kernel support, not on the spatial resolution of the feature map. Appendix B gives the batching, dtype, and complex Newton-Schulz details.

The second step is to convert this direction into the actual parameter update. The oracle characterization in Proposition 2 determines the Muon-C direction up to a common positive scale. Under a radius-ρ sampled-norm constraint, the exact Fourier-domain oracle carries the common factor $\rho / { \sqrt { n } }$ , where $n = k _ { h } k _ { w }$ . In the practical optimizer, we absorb this trust-region radius and DFT normalization into an optimizer scale $s _ { R }$ and calibrate the realized parameter-space update. This changes the magnitude of the step but not the operator-aligned block geometry established in Sections 5.2 and 5.3.

For a representation R, let $U _ { R } ( M ^ { t } )$ denote its shaped descent direction. The applied parameter update takes the generic form

$$
W ^ { t + 1 } = ( 1 - \eta \lambda ) W ^ { t } + s _ { R } \eta U _ { R } ( M ^ { t } ) ,
$$

where $\eta$ is the base learning rate and λ is the decoupled weight decay. For an eligible convolutional kernel with $n = k _ { h } k _ { w } .$ , we use

$$
s _ { g } = 0 . 2 \sqrt { \operatorname* { m a x } \{ C _ { \mathrm { o u t } } , n C _ { \mathrm { i n } } \} } , \qquad s _ { f } = \frac { 0 . 2 } { \mathrm { R M S } ( O ) + \epsilon } ,
$$

for global unfolded Muon and Muon-C, respectively, where O denotes the unscaled Muon-C direction in the original kernel layout. The global rule is the matched-RMS scaling used in large-scale Muon training, which multiplies the polar direction of an $A \times B$ matrix by $0 . 2 \sqrt { \operatorname* { m a x } \{ A , B \} }$ (Liu et al., 2025). Because Muon-C returns its direction directly in the kernel layout, $s _ { f }$ instead normalizes the realized kernel update. Under exact hard-polar shaping, both rules target directional RMS 0.2 before multiplication by the base learning rate, placing the two Muon geometries on the same AdamW-referenced scale.

The same practical step also requires a routing rule specifying which parameters use the convolution-specific geometry. Conv2d kernels with spatial area $k _ { h } k _ { w } > 1$ use either Muon-C or global unfolded Muon, while $1 \times 1$ convolutions, biases, normalization parameters, and all remaining weights use AdamW. Section 6 evaluates these optimizers both under controlled applied-update scale and under optimizer-specific learning-rate tuning.

The third step is to examine the practical consequence of the critical-grid construction. In addition to preserving the original finite kernel support, critical sampling keeps the computational cost of Muon-C tied to the kernel rather than the activation resolution. A construction based on the full feature grid would allocate one channel-transfer block per feature frequency, whereas Muon-C uses only one block per critical kernel frequency. For instance, for a $3 \times 3$ kernel with $C _ { \mathrm { i n } } ~ = ~ C _ { \mathrm { o u t } } ~ = ~ 2 5 6$ , the frequency blocks occupy approximately 3 MiB on the critical kernel grid, compared with approximately 12.4 GiB on a $2 2 4 \times 2 2 4$ feature grid, even before accounting for Newton-Schulz temporaries. Critical sampling thus resolves both the finite-support obstruction in Section 5.1 and the resolutiondependent cost of full-grid polarization. Appendix B gives the full memory calculation.

Within this third step, we also consider how broadly the same construction applies across convolutional layers. For grouped convolution, the transfer matrix is block diagonal across groups at every frequency, and $\| U \| _ { \mathrm { c o n v } } = \operatorname* { m a x } _ { g } \| U _ { g } \| _ { \mathrm { c o n v } }$ . The trust-region constraint and linear objective therefore separate across groups, so applying Muon-C independently within each group gives the corresponding linear minimization oracle for the complete layer. The norm comparison in Theorem 3 and the continuous-oracle guarantee in Corollary 4 apply groupwise without modification. Depthwise convolution is the scalar-block special case of the same construction. The same kernel-grid update can also be applied to a strided convolution because its stored $k _ { h } \times k _ { w }$ kernel support is unchanged, although an operatornorm characterization that explicitly incorporates subsampling remains open.

The fourth and final step is to assemble the preceding components into one practical Muon-C update. Algorithm 1 combines momentum formation, the critical-grid Fourier transform, independent frequency-wise polar shaping, the inverse transform, and the calibrated parameter update. We distinguish the stored exponential moving-average momentum $M ^ { t }$ from the momentum $\widetilde { M } ^ { t }$ that is actually shaped by Muon-C. With Nesterov momentum, the latter includes the additional look-ahead combination used in the implementation.

The resulting optimizer retains the operator-aligned geometry developed in Section 4 and the critical-grid construction of Sections 5.1 to 5.3, while requiring only kernel-sized Fourier transforms and small $C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } }$ polar operations.

Algorithm 1 One operator-aligned Muon-C update on a finite convolutional kernel   
Require: Weight $W ^ { t }$ , gradient $G ^ { t }$ , momentum $M ^ { t - 1 }$ , learning rate $\eta ,$ momentum coeficient $\beta ,$   
weight decay λ, kernel grid $\Omega _ { k } ,$ , numerical stabilizer ϵ   
Ensure: Updated weight $\bar { W } ^ { t + 1 }$   
1: $M ^ { t } \gets \bar { \beta } M ^ { t - 1 } + ( 1 - \beta ) G ^ { t }$   
2: $\widetilde { M } ^ { t }  \beta M ^ { t } + ( 1 - \beta ) G ^ { t }$ for Nesterov momentum, or $\widetilde { M } ^ { t }  M ^ { t }$   
3: $\widehat { M } ^ { t } \gets \mathcal { F } _ { \Omega _ { k } } ( \widetilde { M } ^ { t } )$ over spatial kernel axes   
4: for all critical-frequency blocks $( p , q ) \in \Omega _ { k }$ do   
5: $\widehat { O } ^ { t } ( p , q ) \gets \mathrm { p o l a r } _ { \mathrm { N S } } \Big ( \widehat { M } ^ { t } ( p , q ) \Big )$   
6: end for   
7: $O ^ { t } \gets \mathcal { F } _ { \Omega _ { k } } ^ { - 1 } ( \widehat { O } ^ { t } )$   
8: $s _ { f } \gets 0 . \ddot { 2 } \ddot { / } ( \mathrm { R M S } ( O ^ { t } ) + \epsilon )$   
9: $\Breve { W } ^ { t + 1 }  ( 1 - \eta \lambda ) W ^ { t } - \eta s _ { f } O ^ { t }$

## 6 Numerical Experiments

We evaluate Muon-C along three complementary dimensions. First, we compare it with global unfolded Muon under controlled applied-update scale to determine whether critical Fourier geometry improves optimization eficiency independently of update magnitude. Second, we test whether the advantage remains robust across stochastic replication, optimizerspecific tuning, larger data scale, diferent learning objectives, and convolutional architectures. Third, we use a matched spatial-block ablation to isolate operator alignment from blockwise polarization alone.

## 6.1 Experimental Setup

We evaluate Muon-C in both generative learning and discriminative learning settings. Table 2 summarizes the evaluation suite. For generative modeling, we train the same U-Net architecture with flow matching (Ronneberger et al., 2015; Lipman et al., 2023) on CIFAR-10 and ImageNet-1k resized to 32×32. Both experiments use FID-50k as the primary quality metric and a 400k-step training budget. For discriminative learning, we train ResNet-34, ResNet-50 (He et al., 2016), and ConvNeXtV2-T (Woo et al., 2023) on ImageNet-100 at 224 × 224 resolution and evaluate full-validation top-1 accuracy.

Table 3 reports how much of each architecture is exposed to the convolutional optimizer route. A parameter is eligible when it is a Conv2d weight with spatial area $k _ { h } k _ { w } > 1$ Eligible tensors use either Muon-C or global unfolded Muon, while all remaining parameters use AdamW. The U-Net routes 29.9M of its 35.7M parameters through the convolutional geometry, making it close to a direct test of the proposed geometry. ResNet-34 provides an even higher-exposure classification setting at 98.9%. ResNet-50 is a mixed case at 47.8% because its bottleneck design places substantial capacity in 1×1 projections. ConvNeXtV2- T provides the lowest-exposure test at 6.7%. Its eligible weights are primarily depthwise spatial kernels, while most parameters lie in pointwise expansion matrices. Together, these architectures test the proposed geometry when it acts on nearly all, roughly half, or only a small minority of model parameters.

All flow-matching comparisons use the same U-Net, data pipeline, batch size, mixed precision, warmup-constant schedule, 400k-step budget, exponential moving average, and parameter-routing rule. Final generative quality is evaluated using 50k generated samples, 100 Dopri5 sampling steps, and fixed real-data statistics. The primary CIFAR-10 comparison matches the applied-update RMS on the eligible convolutional route using the scaling in Section 5.4. The robustness studies examine repeated runs at a common learning rate, equalbudget optimizer-specific sweeps, scaling to ImageNet-1k-32, and transfer to ImageNet-100 classification. Appendix B gives the remaining implementation and evaluation details.

<table><tr><td>Dataset</td><td>Task</td><td>Model</td><td>Primary metric</td><td>Budget</td></tr><tr><td>CIFAR-10</td><td>Flow matching</td><td>U-Net</td><td>FID-50k ↓</td><td>400k steps</td></tr><tr><td>ImageNet-1k-32</td><td>Flow matching</td><td>U-Net</td><td>FID-50k↓</td><td>400k shared steps</td></tr><tr><td>ImageNet-100</td><td>Classification</td><td>ResNet-34</td><td>Top-1 ↑</td><td>100 epochs</td></tr><tr><td>ImageNet-100</td><td>Classification</td><td>ResNet-50</td><td> $\mathrm { T o p - 1 ~ \uparrow }$ </td><td>100 epochs</td></tr><tr><td>ImageNet-100</td><td>Classification</td><td>ConvNeXtV2-T</td><td> $\mathrm { T o p - 1 ~ \uparrow }$ </td><td>100 epochs</td></tr></table>

Table 2: Evaluation suite across generative learning and discriminative learning. ImageNet-1k is resized to 32×32 for flow matching, whereas ImageNet-100 classification uses $2 2 4 \times 2 2 4$ inputs.

<table><tr><td>Model</td><td></td><td>Eligible layers Conv2d layers</td><td>Kernel sizes</td><td>Eligible params</td><td>Share</td></tr><tr><td>U-Net</td><td>52</td><td>65</td><td> $3 \times 3$ </td><td>29.9M / 35.7M</td><td>83.8%</td></tr><tr><td>ResNet-34</td><td>33</td><td>36</td><td> $7 \times 7 , 3 \times 3$ </td><td>21.1M /21.3M</td><td>98.9%</td></tr><tr><td>ResNet-50</td><td>17</td><td>53</td><td> $7 \times 7 , 3 \times 3$ </td><td>11.3M / 23.7M</td><td>47.8%</td></tr><tr><td>ConvNeXtV2-T</td><td>22</td><td>22</td><td> $7 \times 7 ~ \mathrm { d w } , 4 \times 4 , 2 \times 2$ </td><td>1.9M / 27.9M</td><td>6.7%</td></tr></table>

Table 3: Coverage of the convolutional optimizer route across architectures. Conv2d weights with $k _ { h } k _ { w } > 1$ are eligible for Muon-C or global unfolded Muon. All remaining parameters use AdamW.

## 6.2 Optimization Eficiency under Controlled Update Scale

We first examine whether replacing global patch geometry with critical Fourier geometry improves optimization eficiency. CIFAR-10 provides the primary controlled comparison, where applied-update RMS is explicitly matched between the two Muon geometries. We then test whether the same advantage persists when the training distribution is scaled to ImageNet-1k-32 under a common base learning rate. Finally, we express the CIFAR-10 trajectories in terms of model FLOPs to quantify compute-to-quality eficiency. Figures 5–7 report these comparisons.

Figure 5 shows the primary controlled comparison on CIFAR-10. At the base learning rate $2 \times 1 0 ^ { - 4 }$ , the calibration described in Section 5.4 matches applied-update RMS on the eligible convolutional route. This controls overall parameter-space update scale while changing the matrices on which Muon applies its polar shaping. Muon-C has the lowest validation loss at all 19 evaluations between 10k and 100k iterations, reaching 0.184 at 20k and approaching its plateau near 0.146 by 80k. The separation is more pronounced in sample quality. Muon-C has the lowest FID-50k at all seven shared checkpoints. At

![](images/5cecd5ecfa3ed4d10419b1c378bc86e9d0d0a6f960928a8440ce0f7d359df369.jpg)  
Figure 5: CIFAR-10 flow matching under matched applied-update RMS. (a) Validation loss over the first 100k iterations. (b) FID-50k at seven shared checkpoints over the full 400k-step budget. Lower is better.

40k iterations, it reaches 9.87 FID, compared with 22.26 for global unfolded Muon and 51.31 for Adam. At the 400k-step endpoint, the corresponding values are 3.47, 3.83, and 3.81. Appendix C further shows the calibrated Muon-C run has a slightly smaller applied displacement.

Figure 6 tests the same comparison at a substantially larger data scale. The experiment changes the training distribution from CIFAR-10 to ImageNet-1k resized to $3 2 \times 3 2$ while keeping the flow-matching objective, U-Net architecture, parameter routing, and base learning rate $2 \times 1 0 ^ { - 4 }$ fixed. Unlike the primary CIFAR-10 experiment, it uses a common base learning rate rather than explicit RMS calibration. Muon-C has the lowest validation loss at 23 of 25 evaluations between 30k and 150k iterations and the lowest FID-50k at all seven shared checkpoints. Its FID is 21.14 at 40k iterations, compared with 28.02 for global unfolded Muon and 58.88 for AdamW. At 80k iterations, Muon-C reaches 14.71, a quality level that the baselines require approximately 149k and 233k iterations to attain. At 400k iterations, Muon-C reaches 12.93 and remains 0.32 below global unfolded Muon and 0.60 below AdamW. The early and final advantages therefore persist when the training distribution is enlarged by more than an order of magnitude.

Figure 7 shows that the CIFAR-10 trajectory advantage translates directly into computeto-quality eficiency. It replots the seven matched-RMS FID checkpoints against cumulative U-Net forward-backward FLOPs, so lower and further left indicates better sample quality at a smaller model-compute budget. The Muon-C curve remains below both baselines throughout the measured range. Log-log interpolation shows that Muon-C reaches the final 400k-step quality of global unfolded Muon and Adam using approximately 0.62× and 0.64× their model FLOPs, corresponding to savings of 38% and 36%. Its extra per-step work consists of a kernel-axis transform and $k _ { h } k _ { w }$ polar operations on $C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } }$ blocks. As discussed in Section 5.4, this cost depends on kernel size rather than feature-map resolution.

![](images/f33ba920d93913c0340fcd8afb9981331c14494bc00b4799d038848ca1b0febf.jpg)

![](images/103a75a36cb13e9b1b777dc3179bbf6ec87d3f2eb6007fff27a8b4a17974aef1.jpg)

Figure 6: ImageNet-1k-32 flow matching under a common base learning rate of $2 \times 1 0 ^ { - 4 }$ (a) Validation loss over the first 150k iterations. (b) FID-50k at seven shared checkpoints over the full 400k-step budget. Lower is better.  
![](images/672c1c07f9531bdaf70bcfaa36235438ecd75a39e7f5d2357514ca51444b7cc9.jpg)

Figure 7: CIFAR-10 compute-to-quality eficiency under matched applied-update RMS. FID-50k is plotted against cumulative U-Net forward-backward FLOPs. Lower and further left indicates better sample quality with less model compute.
<table><tr><td>Optimizer</td><td></td><td></td><td></td><td>Time / step (ms) ↓ Relative time Peak memory (GiB) ↓ Steps to FID 4.0 (k) ↓ time (h) ↓</td><td></td></tr><tr><td>Adam</td><td>73.19</td><td>1.00×</td><td>6.42</td><td>312.9</td><td>6.36</td></tr><tr><td>Global unfolded Muon</td><td>86.02</td><td>1.18×</td><td>6.31</td><td>311.7</td><td>7.45</td></tr><tr><td>Muon-C</td><td>99.39</td><td>1.36×</td><td>6.31</td><td>197.5</td><td>5.45</td></tr></table>

Table 4: CIFAR-10 U-Net runtime and time to FID 4.0. Evaluation time is excluded.

Table 4 complements the model-FLOP comparison with runtime. Muon-C takes 35.8% longer per step than Adam and 15.5% longer than global unfolded Muon, while requiring substantially fewer steps to reach the common target of FID 4.0. Combining the measured step time with the interpolated target crossing gives a projected training time of 5.45 hours for Muon-C, compared with 6.36 hours for Adam and 7.45 hours for global unfolded

Muon, reductions of 14.3% and 26.8%, respectively. Its 6.31 GiB peak allocated memory is essentially the same as global unfolded Muon and slightly below Adam in this benchmark.

Together, these comparisons provide the primary evidence for improved optimization eficiency. Muon-C produces a substantially stronger early-training trajectory, and the advantage persists to the final training budget. On CIFAR-10, the controlled-update comparison yields substantial compute-to-quality savings and reduces projected training time to a common FID target. On ImageNet-1k-32, the same ordering persists at a larger data scale under a common learning rate. The empirical ordering is consistent with the tighter worst-case control of the continuous convolution geometry in Corollary 4. For a $3 \times 3$ kernel, Muon-C guarantees 9/25 of the continuous-oracle objective, compared with $1 / 3$ for global unfolding.

## 6.3 Robustness across Tuning, Data Scale, and Tasks

We next examine the robustness and transfer of the optimization-eficiency gains along three dimensions. We test reproducibility across independent training runs under a common learning rate, compare the optimizers under equal tuning budgets while allowing each to select its own learning rate, and evaluate transfer from flow matching to classification across architectures with substantially diferent routing coverage. These experiments assess whether the benefit of Muon-C extends beyond a particular random seed, learning-rate choice, data scale, learning objective, or architecture.

## 6.3.1 Common-Learning-Rate Replication

Table 5 shows that the Muon-C advantage is stable across repeated runs under a common base learning rate of $2 \times 1 0 ^ { - 4 }$ and otherwise identical protocols. On CIFAR-10, Muon-C achieves $3 . 6 0 { \pm } 0 . 0 7 \ : \mathrm { F I D }$ , compared with $3 . 8 5 { \pm } 0 . 0 8$ for global unfolded Muon and $3 . 9 4 \pm 0 . 0 7$ for Adam. Appendix C reports the paired results for three training seeds, with Muon-C attaining the best endpoint in every repeat. On ImageNet-1k-32, Muon-C similarly reaches $1 2 . 6 2 { \pm } 0 . 0 5 \mathrm { F I D }$ , compared with $1 3 . 0 4 \pm 0 . 0 6$ for global unfolded Muon and $1 3 . 5 3 \pm 0 . 0 5$ for AdamW. The ordering is therefore reproducible across stochastic training runs and holds on both flow-matching datasets.

<table><tr><td>Optimizer</td><td></td><td>CIFAR-10 FID ↓ ImageNet-1k-32 FID ↓</td></tr><tr><td>Adam / AdamW</td><td> $3 . 9 4 \pm 0 . 0 7$ </td><td> $1 3 . 5 3 \pm 0 . 0 5$ </td></tr><tr><td>Global unfolded Muon</td><td> $3 . 8 5 \pm 0 . 0 8$ </td><td> $1 3 . 0 4 \pm 0 . 0 6$ </td></tr><tr><td>Muon-C</td><td> ${ \bf 3 . 6 0 \pm 0 . 0 7 }$ </td><td> ${ \bf 1 2 . 6 2 \pm 0 . 0 5 }$ </td></tr></table>

Table 5: Final flow-matching performance under a common base learning rate of $2 \times 1 0 ^ { - 4 }$ Entries report FID-50k. The coordinate-wise baseline is Adam on CIFAR-10 and AdamW on ImageNet-1k-32. Lower is better.

<table><tr><td></td><td colspan="2">CIFAR-10</td><td colspan="2">ImageNet-1k-32</td></tr><tr><td>Optimizer</td><td>Selected LR</td><td>FID ↓</td><td>Selected LR</td><td>FID ↓</td></tr><tr><td>Adam / AdamW</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $3 . 8 3 \pm 0 . 0 3$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $1 3 . 5 3 \pm 0 . 0 5$ </td></tr><tr><td>Global unfolded Muon</td><td> $8 \times 1 0 ^ { - 4 }$ </td><td> $3 . 5 4 \pm 0 . 0 4$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $1 3 . 2 4 \pm 0 . 0 4$ </td></tr><tr><td>Muon-C</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> ${ \bf 3 . 4 2 \pm 0 . 0 3 }$ </td><td> $6 \times 1 0 ^ { - 4 }$ </td><td> ${ \bf 1 2 . 0 2 \pm 0 . 0 2 }$ </td></tr></table>

Table 6: Equal-budget optimizer-specific learning-rate tuning. Each optimizer receives the same ten-run sweep, and the selected learning rate and final FID-50k are reported. The coordinate-wise baseline is Adam on CIFAR-10 and AdamW on ImageNet-1k-32.

## 6.3.2 E<sub>q</sub>ual-Budget Optimizer-Specific Tuning

Table 6 shows that the advantage remains when every optimizer receives the same tuning budget but may select its own learning rate. For each optimizer, we sweep ten learning rates from $1 \times 1 0 ^ { - 4 } \mathrm { ~ t o ~ } 1 \times 1 0 ^ { - 3 }$ in increments of $1 \times 1 0 ^ { - 4 }$ and select the value with the lowest final FID-50k. Muon-C reaches 3.42 FID on CIFAR-10, compared with 3.54 for global unfolded Muon and 3.83 for Adam. On ImageNet-1k-32, it reaches 12.02, compared with 13.24 for global unfolded Muon and 13.53 for AdamW. Its improvements over global unfolded Muon are therefore 0.12 and 1.22 FID on the two datasets.

The selected learning rates are $5 \times 1 0 ^ { - 4 }$ for Muon-C and $8 \times 1 0 ^ { - 4 }$ for global unfolded Muon on CIFAR-10, and $6 \times 1 0 ^ { - 4 }$ and $1 \times 1 0 ^ { - 3 }$ on ImageNet-1k-32. Appendix C.2 evaluates the selected runs at 0.1B and 0.5B, where $B = 4 0 0 \mathrm { k \Omega }$ steps. Muon-C has the lowest FID at every reported budget fraction on both datasets, so its advantage appears before the final tuned endpoint.

## 6.3.3 Transfer to ImageNet-100 Classification

Figure 8 and Table 7 show that the Muon-C advantage transfers from generative flow matching to discriminative classification. We train ResNet-34, ResNet-50, and ConvNeXtV2-T on ImageNet-100 using the same parameter-routing and applied-update-RMS protocol as in the controlled CIFAR-10 comparison. The architectures expose 98.9%, 47.8%, and 6.7% of their parameters to the convolutional optimizer route, respectively. This range provides a controlled test of whether the benefit survives across architectures with very diferent reliance on spatial convolution.

Figure 8 shows that Muon-C has the highest validation accuracy at every reported evaluation point on all three architectures, with the ordering emerging within the first few thousand updates. At 2k updates, it reaches 57.06% top-1 accuracy on ResNet-34, compared with 53.10% for global unfolded Muon and 41.54% for AdamW. The corresponding values are 43.00%, 39.80%, and 34.48% on ResNet-50, and 59.06%, 56.66%, and 43.60% on ConvNeXtV2-T. The consistent early separation shows the optimization-eficiency advantage is not specific to flow matching.

Table 7 shows that the early advantage also persists to the end of training. Muon-C reaches 80.22% on ResNet-34, improving by 0.58 percentage points over global unfolded

![](images/01e377fdceb229c5116c14b9edc081b63e293dcad60fbd11ec978fad4fa01d1f.jpg)

Figure 8: Early optimization trajectories on ImageNet-100 classification. Full-validation top-1 accuracy over the first 20k optimizer updates for (a) ResNet-34, (b) ResNet-50, and (c) ConvNeXtV2-T. Higher is better.
<table><tr><td>Setting</td><td>Metric</td><td>AdamW</td><td>Unfolded Muon</td><td>Muon-C</td></tr><tr><td>ResNet-34</td><td> $\mathrm { T o p - 1 ~ \uparrow }$ </td><td> $7 9 . 7 4 \pm 0 . 1 6$ </td><td> $7 9 . 6 4 \pm 0 . 1 3$ </td><td> ${ \bf 8 0 . 2 2 \pm 0 . 1 5 }$ </td></tr><tr><td>ResNet-50</td><td> $\mathrm { T o p - 1 ~ \uparrow }$ </td><td> $7 9 . 2 4 \pm 0 . 1 7$ </td><td> $7 9 . 2 8 \pm 0 . 1 8$ </td><td> ${ \bf 7 9 . 8 8 \pm 0 . 1 6 }$ </td></tr><tr><td>ConvNeXtV2-T</td><td> $\mathrm { T o p - 1 ~ \uparrow }$ </td><td> $8 2 . 8 6 \pm 0 . 0 9$ </td><td> $8 4 . 6 2 \pm 0 . 1 2$ </td><td> ${ \bf 8 4 . 8 8 \pm 0 . 1 0 }$ </td></tr></table>

Table 7: ImageNet-100 full-validation top-1 accuracy after 100 training epochs. Results are reported for architectures with substantially diferent exposure to the convolutional optimizer route. Higher is better.

Muon and 0.48 points over AdamW. On ResNet-50, it reaches 79.88%, improving by 0.60 and 0.64 points, respectively. On ConvNeXtV2-T, Muon-C reaches 84.88%, compared with 84.62% for global unfolded Muon and 82.86% for AdamW. The ConvNeXtV2-T result is particularly informative because only 1.9M of its 27.9M parameters use the convolutional route, yet Muon-C still improves over global unfolded Muon.

Across the three classification architectures, Muon-C shows larger gains early in training and smaller gains at the 100-epoch endpoint. The robustness studies show consistent gains across the evaluated seeds, tuning settings, datasets, and architectures. We next compare frequency-wise and spatial-ofset blocks using the matched Muon-S ablation in Section 6.4.

## 6.4 Does Operator Alignment Drive the Gain?

The preceding experiments establish that Muon-C improves optimization eficiency across training protocols, data scales, tasks, and architectures. We now isolate the mechanism by separating blockwise channel polarization from operator-aligned translation-frequency organization. Muon-S evaluates the Conv2D spatial-block duality direction of Bernstein and

![](images/92dffc60588e97dc975b0544344938d05ee389e66b05e274eb27dedd05507391.jpg)  
Figure 9: CIFAR-10 operator-alignment ablation under matched applied-update RMS. (a) Validation loss over the first 100k iterations. (b) FID-50k at seven shared checkpoints over the full 400k-step budget. Muon-S replaces the kernel DFT with the identity while matching block count, block shape, routing, learning rate, update RMS, and training protocol. Lower is better.

Newhouse (2025) with our shared momentum, routing, and update-scale settings. Operationally, it replaces the kernel DFT in Muon-C with the identity transform and applies the same channel-matrix polar map independently at each spatial ofset. It matches Muon-C in block count, block shape, parameter routing, learning rate, applied-update RMS, and training protocol. Muon-C uses translation-frequency blocks, whereas Muon-S uses spatial-ofset blocks.

Figure 9 shows a clear early optimization advantage from the operator-aligned frequency organization. Muon-C has the lowest validation loss at all 19 evaluations from 10k to 100k iterations. At 20k iterations, its loss is 0.184, compared with 0.195 for Muon-S. The separation is larger in sample quality. At 20k iterations, Muon-C reaches 166.83 FID, compared with 211.25 for Muon-S. At 40k iterations, the values are 9.87 and 22.28. The gap narrows later, but Muon-C remains slightly ahead at the 400k-step endpoint with 3.47 FID compared with 3.49 for Muon-S. Both blockwise variants finish ahead of global unfolded Muon at 3.83 and Adam at 3.81.

Under the matched protocol, frequency-wise blocks yield a stronger early trajectory than spatial-ofset blocks. This result supports the role of translation-frequency organization in Muon-C’s optimization eficiency.

The ablation comparison also connects directly to the theory. Muon-S and Muon-C are the exact linear minimization oracles for the spatial-block and critically sampled convolution geometries. For a 3 × 3 kernel, Theorem 3 gives worst-case approximation factors of 9 and 25/9, while Corollary 4 guarantees 1/9 and 9/25 of the continuous-oracle objective.

## 6.5 Directional Curvature and Finite-Support Geometry

The matched Muon-S experiment in Section 6.4 shows that organizing independently polarized channel blocks by translation frequency improves the training trajectory relative to spatial-ofset blocks. We next examine how this operator-aligned geometry acts locally within a trained network and why the critical Fourier grid is important for realizing it with finite-support kernels. We study shared states from a Muon-C U-Net trajectory at 40k, 200k, and 400k iterations. At each state, we use one fixed 512-example evaluation objective, construct the three exact-polar directions from the same Nesterov momentum signal, and jointly perturb 49 eligible $3 \times 3$ convolution kernels. Each direction is scaled per layer to the same convolution-operator-norm budget, estimated on a $3 2 \times 3 2$ Fourier grid, and a common radius multiplier α controls the perturbation size. Appendix C.3.1 specifies the budgets and evaluation protocol.

To separate the linear benefit of a direction from its local curvature, we use the quadratic approximation

$$
L ( \theta + \alpha V ) - L ( \theta ) \approx - \alpha a + \frac { 1 } { 2 } \alpha ^ { 2 } c _ { H } , \qquad a = - \langle \nabla L ( \theta ) , V \rangle , \qquad c _ { H } = V ^ { \top } \nabla ^ { 2 } L ( \theta ) V .\tag{9}
$$

We estimate c and the MSE Gauss-Newton curvature by central finite diferences. The resulting local-geometry diagnostics show that Muon-C retains 73%-89% of unfolded Muon’s linearized decrease while reducing the Hessian and Gauss-Newton curvature estimates to 45%-60% and 47%-51%, respectively. Its resulting quadratic-model maximum decrease, $a ^ { 2 } / ( 2 c _ { H } )$ , is 12%-38% larger. Muon-S’s Gauss-Newton curvature remains within 4% of unfolded Muon’s, whereas Muon-C’s is approximately half of Muon-S’s.

Figure 10 shows the corresponding finite-step behavior under the matched operatornorm budget. Unfolded Muon gives the largest decrease at $\alpha = 1$ at all three states, consistent with its larger linearized decrease. Muon-C, however, tolerates a substantially larger useful step. Its best tested decrease occurs at $\alpha = 2 , 4 , 8$ at 40k, 200k, and 400k iterations, compared with 1.5, 3, 4 for both controls, and it improves the best tested decrease over unfolded Muon by 12%, 18%, and 67%, respectively. Thus, the frequency-organized direction sacrifices some first-order decrease but substantially reduces directional curvature, allowing a larger operator-norm step and a greater finite-step improvement.

Table 8 examines a complementary question: whether the same frequency-wise polarization could instead be carried out on the full feature-frequency grid and then restricted back to the stored kernel support. For nine representative stride-one layers spanning $4 \times 4$ to $3 2 \times 3 2$ feature grids, the full-grid update places substantial energy outside the $3 \times 3$ stored support at every checkpoint. The median outside-support energy remains approximately 25%, while the median discrepancy between the critical-grid update and the cropped full-grid update is 0.62–0.63; the corresponding full-update discrepancy is 0.73–0.74. These efects are stable from 40k through 400k iterations. Exact-SVD checks confirm that the fullgrid oracle itself is solved accurately. Critical sampling makes independent frequency-wise polarization compatible with the original finite kernel support.

<table><tr><td>Checkpoint</td><td>Layers</td><td>Supported discrepancy</td><td> $d _ { \mathrm { s u p p } }$ </td><td>Outside-support energy l</td><td>Full discrepancy  $d _ { \mathrm { f u l l } }$ </td></tr><tr><td>40k</td><td>9</td><td></td><td>0.398 / 0.621 / 0.869</td><td>0.113 / 0.256 / 0.348</td><td>0.503 / 0.735 / 0.914</td></tr><tr><td>200k</td><td>9</td><td></td><td> $0 . 4 2 6 \ \mathrm { ~ / ~ } 0 . 6 2 3 \ \mathrm { ~ / ~ } 0 . 8 5 0$ </td><td>0.130 / 0.252 / 0.342</td><td>0.536 / 0.729 / 0.900</td></tr><tr><td>400k</td><td>9</td><td></td><td> $0 . 4 5 4 \mathrm { ~ / ~ } 0 . 6 3 0 \mathrm { ~ / ~ } 0 . 8 1 1$ </td><td>0.161 / 0.251 / 0.339</td><td>0.580 / 0.731 / 0.880</td></tr></table>

Table 8: Full-grid finite-support obstruction across nine representative stride-one layers. Entries report minimum $/$ median $/$ maximum at each checkpoint for supported discrepancy $d _ { \mathrm { s u p p } } .$ , outside-support energy ℓ, and full discrepancy $d _ { \mathrm { f u l l } }$

![](images/65c9bf9d1d63d988d8d75f2431ddb5187a93d6aaff8204051efe591fb6504b7a.jpg)  
Figure 10: Finite-step loss decrease under matched convolution-operator-norm budgets at 40k, 200k, and 400k iterations. Annotations compare Muon-C with unfolded Muon; panels use separate vertical scales. More negative is better.

Taken together, Figure 10 and Table 8 clarify two complementary roles of Muon-C’s operator-aligned construction. Frequency organization changes the local optimization geometry, producing substantially lower directional curvature and permitting larger useful operator-norm steps, while critical sampling makes this geometry realizable without violating finite kernel support. These diagnostics provide a network-level mechanism consistent with the stronger early training and FID trajectories observed in Sections 6.2 to 6.4.

## 7 Discussion

Muon is inherently representation dependent because its spectral-norm geometry is determined by the matrices, and more generally the independently constrained blocks, on which the polar update acts. The central lesson of this work is therefore broader than one implementation for convolution. Extending Muon to a structured parameter requires choosing not only a coordinate representation but also the linear maps that define its independently polarized blocks. A unitary change of coordinates alone cannot alter the global Muon direction, while a genuinely diferent geometry arises from changing the block partition. For convolution, translation equivariance provides a principled choice through the frequencywise channel-transfer maps. Muon-C realizes this operator-aligned geometry on a critical Fourier grid, allowing these maps to be polarized independently while returning the update exactly to the original finite kernel support.

The theoretical and empirical results provide complementary support for this construction. Muon-C is the exact linear minimization oracle for the critically sampled convolution norm, and the critical DFT gives a unitary, bijective, and support-preserving representation of the stored coeficients. Relative to the continuous convolution-operator norm, its worst-case guarantee is no weaker than global unfolding for any finite kernel size and is strictly stronger for 3 × 3 kernels. The experiments show that this geometric change translates into improved optimization eficiency. Muon-C produces stronger early trajectories and compute-to-quality performance under controlled update scale, and the advantage persists across stochastic replication, optimizer-specific tuning, larger data scale, discriminative training, and architectures with widely diferent routing coverage. The matched Muon-S comparison supports the contribution of translation-frequency organization to the early trajectory gains.

Several questions remain open. On the theoretical side, establishing conditions under which improved operator-oracle alignment translates into faster convergence remains an important next step. The reduced directional curvature observed in Section 6.5 suggests one possible mechanism: operator-aligned first-order geometry may interact with local loss curvature to permit larger useful steps. Analyzing the finite-iteration Newton-Schulz approximation and extending the operator-norm characterization to strided and dilated convolution are additional directions for future work. On the empirical side, Muon-C introduces additional optimizer-side computation despite its resolution-independent critical-grid construction. The strongest empirical evidence is also concentrated in relatively small-resolution generative models and ImageNet-100 classification, rather than full-scale modern convolutional training. Broader wall-clock evaluations, together with extensions to full-resolution ImageNet-1k, larger kernels, and additional convolution-heavy tasks, would further test its practical scope. More broadly, the operator-aligned viewpoint suggests a route for extending Muon to other structured parameters by identifying symmetry-induced linear maps and defining optimizer geometry on those maps.

## Appendix A. Proofs and Operator-Norm Analysis of Muon-C

All frequency-domain pairings use $\langle A , B \rangle _ { F } = \mathrm { t r } ( A ^ { * } B )$ with the real part understood. For real spatial tensors, conjugate symmetry is preserved because polar $( { \overline { { A } } } ) = { \overline { { \operatorname { p o l a r } ( A ) } } }$ . Paired choices therefore invert to real tensors. At zero or rank-deficient blocks, we use the canonical partial polar factor and omit arbitrary null-space components.

Proof (Proposition 1) If $X = P \Sigma Q ^ { * }$ is a compact singular value decomposition, then $L X R = ( L P ) \Sigma ( R ^ { * } Q ) ^ { * }$ is also one, and hence polar $( L X R ) = ( L P ) ( R ^ { * } Q ) ^ { * } = L ( P Q ^ { * } ) R$ The zero case follows from $\mathrm { p o l a r } ( 0 ) = 0$ . This completes the proof of Proposition 1.

Proof (Proposition 2) Equation (3) makes the constraint equivalent to $\left\| \widehat { U } ( p , q ) \right\| _ { \mathrm { o p } } \leq \rho / \sqrt { n }$ at every critical frequency. Parseval’s identity gives

$$
\langle M , U \rangle = \sum _ { p , q } \mathrm { R e } \left. \widehat { M } ( p , q ) , \widehat { U } ( p , q ) \right. _ { F } ,
$$

so the objective and constraint separate by block. Spectral–nuclear norm duality yields $\widehat { U } _ { \mathrm { f } } ^ { \star } ( p , q ) = - ( \rho / \sqrt { n } ) \operatorname { p o l a r } ( \widehat { M } ( p , q ) )$ . For a real M, conjugate symmetry and polar conjugation make the inverse DFT real. This completes the proof of Proposition 2.

Full-grid circular linear minimization oracle. The idealized full-grid reference allows one free coeficient matrix at every location of an $H \times W$ circular grid. Let $G , U \in$ $\mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } } \times H \times W }$ , let $\Omega = [ H ] \times [ W ]$ , and let ${ \widehat { G } } = { \mathcal { F } } _ { \Omega } ( G )$ and $\widehat { U } = \mathcal { F } _ { \Omega } ( U )$ denote their orthonormal spatial DFTs. Define

$$
\| U \| _ { \mathrm { b l o c k , \Omega } } : = \operatorname* { m a x } _ { \omega \in \Omega } \left\| \widehat { U } ( \omega ) \right\| _ { \mathrm { o p } } .
$$

This norm is proportional to the induced $\ell _ { 2 }$ norm of the corresponding circular-convolution operator. The proportionality constant rescales the trust-region radius but does not change the steepest direction.

Proposition 5 (Idealized full-grid Fourier-block LMO) For full-grid momentum $G ,$ one solution of

$$
\operatorname* { m i n } _ { \| U \| _ { \mathrm { b l o c k } , \Omega } \le \rho } \left. G , U \right.
$$

is specified by

$$
\widehat { U } ^ { \star } ( \omega ) = - \rho \mathrm { \ p o l a r } \bigl ( \widehat { G } ( \omega ) \bigr ) , \qquad \omega \in \Omega .
$$

Proof Parseval’s identity separates the objective as

$$
\left. G , U \right. = \sum _ { \omega \in \Omega } \mathrm { R e } \left. \widehat { G } ( \omega ) , \widehat { U } ( \omega ) \right. _ { F } .
$$

The constraint bounds each $\left\| \widehat { U } ( \omega ) \right\| _ { \mathrm { o p } }$ by $\rho ,$ so spectral–nuclear norm duality gives $\widehat { U } ^ { \star } ( \omega ) =$ $- \rho \operatorname { p o l a r } ( { \widehat { G } } ( \omega ) )$ independently at every frequency. Conjugate-symmetric choices invert to a real spatial tensor. This completes the proof of Proposition 5. ■

Proof (Theorem 3) Write $j = ( p , q )$ and $A _ { j } = A _ { U } ( \theta _ { p } , \phi _ { q } )$ . Since

$$
B _ { U } B _ { U } ^ { * } = \sum _ { u , v } U _ { u v } U _ { u v } ^ { * } \succeq U _ { u v } U _ { u v } ^ { * } ,
$$

each $\| U _ { u v } \| _ { \mathrm { o p } } \leq \| U \| _ { \mathrm { p a t c h } }$ , proving $\| U \| _ { \mathrm { s p } } \leq \| U \| _ { \mathrm { p a t c h } }$ . Discrete Fourier orthogonality gives

$$
\frac { 1 } { n } \sum _ { j } A _ { j } A _ { j } ^ { * } = \sum _ { u , v } U _ { u v } U _ { u v } ^ { * } = B _ { U } B _ { U } ^ { * } .
$$

Therefore

$$
\| U \| _ { \mathrm { p a t c h } } ^ { 2 } \le \frac { 1 } { n } \sum _ { j } \lambda _ { \operatorname* { m a x } } ( A _ { j } A _ { j } ^ { * } ) \le \| U \| _ { \mathrm { s a m p } } ^ { 2 } .
$$

The critical frequencies lie in the continuous domain. Hence $\| U \| _ { \mathrm { s a m p } } \leq \| U \| _ { \mathrm { c o n v } }$ , completing (5).

For the spatial upper bound, the triangle inequality gives

$$
\left\| A _ { U } ( \theta ) \right\| _ { \mathrm { o p } } \leq \sum _ { u , v } \left\| U _ { u v } \right\| _ { \mathrm { o p } } \leq n \ \left\| U \right\| _ { \mathrm { s p } } .
$$

Let $Z _ { \theta }$ vertically stack $e ^ { - \mathrm { i } \langle ( u , v ) , \theta \rangle } I$ over $( u , v ) \in \Omega _ { k }$ . Then $A _ { U } ( \theta ) = B _ { U } Z _ { \theta }$ and $Z _ { \theta } ^ { * } Z _ { \theta } = n I$ so $\left\| A _ { U } ( \theta ) \right\| _ { \mathrm { o p } } \leq { \sqrt { n } } \ \left\| U \right\| _ { \mathrm { p a t c h } } .$

Cardinal interpolation gives

$$
A _ { U } ( \theta _ { 1 } , \theta _ { 2 } ) = \sum _ { p , q } \ell _ { p } ^ { ( k _ { h } ) } ( \theta _ { 1 } ) \ell _ { q } ^ { ( k _ { w } ) } ( \theta _ { 2 } ) A _ { j } .
$$

Taking operator norms and then the supremum yields $\left\| U \right\| _ { \mathrm { c o n v } } \leq \Lambda _ { k _ { h } } \Lambda _ { k _ { w } } \left\| U \right\| _ { \mathrm { s a m p } } ,$ proving (6). Fourier orthogonality also gives $\begin{array} { r } { \sum _ { p } | \ell _ { p } ^ { ( k ) } ( \theta ) | ^ { 2 } = 1 } \end{array}$ . Cauchy–Schwarz therefore gives $\Lambda _ { k } \leq \sqrt { k }$ , proving (7). For $k = 3$

$$
| \ell _ { p } ^ { ( 3 ) } ( \theta ) | = { \frac { 1 } { 3 } } \left| 1 + 2 \cos \left( \theta - { \frac { 2 \pi p } { 3 } } \right) \right| .
$$

The three expressions inside the absolute values sum to $^ { 3 , }$ at most one is negative, and each is at least −1. Their absolute values therefore sum to at most $5 ,$ with equality at $\theta = \pi ,$ , so $\Lambda _ { 3 } = 5 / 3$ . This completes the proof of Theorem 3. ■

Proof (Corollary 4) Spectral–nuclear norm duality gives the global and spatial oracles, and Proposition 2 gives the Fourier oracle. Let $\| \cdot \| .$ <sub>r</sub> be the corresponding surrogate norm and define

$$
\Delta _ { r } = \operatorname* { m a x } _ { \| U \| _ { r } \leq \rho } - \left. M , U \right. , \qquad \Delta _ { \mathrm { c o n v } } = \operatorname* { m a x } _ { \| U \| _ { \mathrm { c o n v } } \leq \rho } - \left. M , U \right. .
$$

The theorem gives $\| U \| _ { r } \leq \| U \| _ { \mathrm { c o n v } } \leq \kappa _ { r } \| U \| _ { r }$ . Thus the continuous ball is contained in the surrogate ball and $\Delta _ { r } \geq \Delta _ { \mathrm { c o n v } }$ , while $\| U _ { r } ^ { \star } / \kappa _ { r } \| _ { \mathrm { c o n v } } \leq \| U _ { r } ^ { \star } \| _ { r } \leq \rho$ . Consequently,

$$
- \left. M , U _ { r } ^ { \star } / \kappa _ { r } \right. = \frac { \Delta _ { r } } { \kappa _ { r } } \geq \frac { \Delta _ { \mathrm { c o n v } } } { \kappa _ { r } } ,
$$

which proves (8). This completes the proof of Corollary 4.

## Appendix B. Implementation and Experimental Details

Muon-C implementation and routing. Muon-C applies a two-dimensional orthonormal FFT on the kernel-sized spatial grid and uses five float32 complex Newton–Schulz iterations to approximate the hard-polar direction. We use momentum 0.95 with Nesterov momentum and no support-constraint iteration. Conv2d kernels with spatial area $k _ { h } k _ { w } > 1$ are routed through Muon-C, while $1 \times 1$ convolutions, biases, normalization parameters, and residual matrices use AdamW. Parameters with matching shapes and settings are batched. The global unfolded-Muon baseline uses the same routing boundary but applies matrix Muon to each eligible kernel’s mode-0 unfolding. For a convolution with $g$ groups, Muon-C independently polarizes each group’s $( C _ { \mathrm { o u t } } / g ) \times ( C _ { \mathrm { i n } } / g )$ transfer matrix at each frequency. Thus, ConvNeXtV2-T’s depthwise convolutions use one scalar block per channel and frequency, without mixing channels.

For each frequency block B, let $\widetilde { B } = B ^ { \ast }$ if B has more rows than columns, and $\widetilde { B } = B$ otherwise. The complex Newton–Schulz routine uses

$$
X _ { 0 } = { \frac { \widetilde { B } } { \| \widetilde { B } \| _ { F } + \epsilon } } , \qquad A _ { j } = X _ { j } X _ { j } ^ { * } , \qquad X _ { j + 1 } = a X _ { j } + ( b A _ { j } + c A _ { j } ^ { 2 } ) X _ { j } ,
$$

with $( a , b , c ) = ( 3 . 4 4 4 5 , - 4 . 7 7 5 , 2 . 0 3 1 5 ) , \epsilon = 1 0 ^ { - 8 }$ , and $j = 0 , \ldots , 4$ . It returns $X _ { 5 } ^ { * }$ for a transposed input and $X _ { 5 }$ otherwise. Computation uses complex64; the same ϵ stabilizes the update-RMS normalization in Algorithm 1.

Training configurations. The CIFAR-10 U-Net uses $3 2 \times 3 2$ images, batch size 128, mixed precision, 400k training steps, a 5k-step warmup followed by a constant learning rate, gradient clipping at 1.0, and EMA decay 0.9999. ImageNet-1k-32 uses the same U-Net architecture, batch size, precision, warmup-constant schedule, 400k-step budget, EMA protocol, data pipeline, and parameter-routing rule. ImageNet-100 uses 224×224 inputs and trains ResNet-34, ResNet-50, and ConvNeXtV2-T for 100 epochs, reporting full-validation top-1 accuracy under the same routing and applied-update-scale protocol. All ImageNet-100 runs use batch size 256 and a 1k-step linear warmup followed by cosine decay to zero. The base learning rate and weight decay are $( 2 \times 1 0 ^ { - 4 } , 0 )$ for both ResNets and $( 2 \times 1 0 ^ { - 3 } , 0 . 0 5 )$ for ConvNeXtV2-T. Training uses random resized crops with area scale [0.08, 1] and horizontal flips with probability 0.5. Validation resizes the shorter side to 256 pixels and center-crops to 224 pixels. Both splits use ImageNet channel normalization. Figure 8 shows one training run per method and architecture. Table 7 reports the mean and standard deviation of three independent runs with seeds 42–44.

Learning-rate selection and replication. Common-learning-rate flow-matching runs use a base learning rate of $2 \times 1 0 ^ { - 4 }$ . CIFAR-10 repetitions use paired training seeds 42–44. Each equal-budget sweep evaluates ten learning rates from $1 0 ^ { - 4 }$ to $1 0 ^ { - 3 }$ in increments of $1 0 ^ { - 4 }$ and selects the value with the lowest final FID-50k. The selected learning rates for Adam, global unfolded Muon, and Muon-C are $( 2 , 8 , 5 ) \times 1 0 ^ { - 4 }$ on CIFAR-10 and $( 3 , 1 0 , 6 ) \times 1 0 ^ { - 4 }$ on ImageNet-1k-32. Weight decay is zero in the CIFAR experiments, so AdamW on the hybrid fallback route is equivalent to the pure Adam baseline there.

Evaluation and model-compute accounting. Every FID-50k evaluation uses 50k generated samples, 100 Dopri5 sampling steps, fixed real-data statistics, and EMA weights. The shared checkpoints are 20k, 40k, 80k, 160k, 200k, 300k, and 400k iterations. The Muon-S control uses the same evaluation protocol. Figure 7 reports cumulative U-Net forward– backward model FLOPs, using $4 . 7 8 \times 1 0 ^ { 1 2 }$ FLOPs per batch; optimizer, evaluation, and data-pipeline computation are excluded. Runtime measurements in Table 4 use one NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition GPU. Each optimizer runs in a separate process with 50 warmup steps followed by 500 measured steps at batch size 128. We report mean step time, with CUDA synchronization at both timing boundaries, including data loading and transfer, forward/backward computation, gradient clipping, optimizer and scheduler steps, gradient clearing, and EMA updates. Peak-memory statistics are reset after warmup; we report maximum allocated memory over the timed window. Evaluation is excluded.

Computational and memory cost. A real-input FFT on an $a \times b$ grid stores $a ( \lfloor b / 2 \rfloor + 1 )$ complex channel-transfer blocks. For $C _ { \mathrm { i n } } = C _ { \mathrm { o u t } } = 2 5 6 , \mathrm { ~ a ~ 3 \times 3 ~ }$ critical grid has six blocks and occupies approximately 3 MiB in complex64. Feature grids of size 8, 16, and 32 use 40,

144, and 544 blocks, respectively; the $3 2 \times 3 2$ tensor occupies approximately 272 MiB and the $2 2 4 \times 2 2 4$ tensor approximately 12.4 GiB, before Newton-Schulz temporaries. These values are storage estimates for the frequency-block tensors rather than end-to-end training memory; measured peak allocated memory is reported in Table 4.

## Appendix C. Additional Empirical Results and Diagnostics

## C.1 Applied-Update RMS Diagnostics

Table 9 reports the actual parameter displacements under the AdamW-referenced RMS scaling described in Section 5.4. The probe uses shared initialization, mini-batches, and stochastic draws and measures post-step RMS on the eligible spatial-Conv2d route.

Muon-C has a smaller global route RMS at every measured step, with a mean ratio of 0.888 relative to global unfolded Muon. The same pattern holds across layers: 43 of 52 layers have mean RMS ratios no greater than one, and the median layerwise ratio is 0.744.

## C.2 Robustness across Seeds and Training Budgets

## C.2.1 Replication across Training Seeds

Table 10 resolves the common-learning-rate CIFAR-10 aggregate in Table 5 into three paired training runs. The training seed is shared across methods within each column, and all runs use base learning rate $2 \times 1 0 ^ { - 4 }$ and the same final FID-50k protocol.

Muon-C achieves the best endpoint in every paired seed. Averaged across the three runs, it improves FID by 0.252 over global unfolded Muon and by 0.338 over Adam, with its seedwise gain over global unfolded Muon ranging from 0.080 to 0.354.

<table><tr><td colspan="3">Measurement global unfolded Muon Muon-C Ratio</td></tr><tr><td colspan="3">Global route RMS  $( \times 1 0 ^ { - 6 } )$ </td></tr><tr><td>Step 2</td><td>1.301</td><td>1.214 0.934</td></tr><tr><td>Step 3</td><td>1.599 1.418</td><td>0.887</td></tr><tr><td>Step 4 1.706</td><td>1.494</td><td>0.875</td></tr><tr><td>Step 5 1.779</td><td>1.547</td><td>0.869</td></tr><tr><td>Mean 1.596</td><td>1.418</td><td>0.888</td></tr><tr><td colspan="3">Layerwise RMS-ratio summary over 52 layers</td></tr><tr><td>Mean of layer ratios</td><td></td><td>0.733</td></tr><tr><td>Median layer ratio</td><td></td><td>0.744</td></tr><tr><td>Range</td><td></td><td>[0.137, 1.100]</td></tr><tr><td>Layers with ratio  $\leq 1$ </td><td></td><td>43/52</td></tr></table>

Table 9: Applied-update RMS on the eligible spatial-Conv2d route. Ratios are Muon-C divided by global unfolded Muon; layer summaries average available nonzero ratios over steps 2 to 5.

<table><tr><td>Optimizer</td><td>Seed 42</td><td>Seed 43</td><td>Seed 44</td><td> ${ \mathrm { M e a n } } \pm { \mathrm { s t d } } .$ </td></tr><tr><td>Adam</td><td>3.999</td><td>3.953</td><td>3.869</td><td> $3 . 9 4 1 \pm 0 . 0 6 6$ </td></tr><tr><td>global unfolded Muon</td><td>3.908</td><td>3.894</td><td>3.761</td><td> $3 . 8 5 4 \pm 0 . 0 8 1$ </td></tr><tr><td>Muon-C</td><td>3.588</td><td>3.540</td><td>3.681</td><td> $\mathbf { 3 . 6 0 3 \pm 0 . 0 7 2 }$ </td></tr></table>

Table 10: CIFAR-10 final FID-50k across three paired common-learning-rate training seeds. The final column reports the mean and sample standard deviation. Lower is better.

<table><tr><td>Optimizer</td><td>Selected LR</td><td> $\mathrm { F I D ~ a t } 0 . 1 B \downarrow$ </td><td> $\mathrm { F I D ~ a t } \ 0 . 5 B \ \downarrow$ </td><td>FID at B↓</td></tr><tr><td>CIFAR-10</td><td></td><td></td><td></td><td></td></tr><tr><td>Adam</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $5 1 . 3 7 \pm 0 . 2 5$ </td><td> $4 . 4 1 \pm 0 . 0 4$ </td><td> $3 . 8 3 \pm 0 . 0 3$ </td></tr><tr><td>Global unfolded Muon</td><td> $8 \times 1 0 ^ { - 4 }$ </td><td> $2 4 . 9 7 \pm 0 . 0 5$ </td><td> $3 . 8 9 \pm 0 . 0 7$ </td><td> $3 . 5 4 \pm 0 . 0 4$ </td></tr><tr><td>Muon-C</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> ${ \bf 1 0 . 3 7 \pm 0 . 0 6 }$ </td><td> ${ \bf 3 . 8 5 \pm 0 . 0 6 }$ </td><td> ${ \bf 3 . 4 2 \pm 0 . 0 3 }$ </td></tr><tr><td> $I m a g e N e t – 1 k – 3 2$ </td><td></td><td></td><td></td><td></td></tr><tr><td>AdamW</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $5 8 . 8 7 \pm 0 . 0 6$ </td><td> $1 5 . 1 1 \pm 0 . 0 3$ </td><td> $1 3 . 5 3 \pm 0 . 0 5$ </td></tr><tr><td>Global unfolded Muon</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $2 8 . 0 2 \pm 0 . 0 6$ </td><td> $1 4 . 1 0 \pm 0 . 0 4$ </td><td> $1 3 . 2 4 \pm 0 . 0 4$ </td></tr><tr><td>Muon-C</td><td> $6 \times 1 0 ^ { - 4 }$ </td><td> ${ \bf 2 1 . 1 4 \pm 0 . 0 5 }$ </td><td> ${ \bf 1 3 . 9 6 \pm 0 . 0 5 }$ </td><td> ${ \bf 1 2 . 0 2 \pm 0 . 0 2 }$ </td></tr></table>

Table 11: Sweep-selected flow-matching performance at fixed fractions of the 400k-step training budget. The selected learning rate is fixed across budget fractions. Lower is better.

## C.2.2 Performance across Training Budgets

Table 11 evaluates the sweep-selected runs at three fractions of the budget B = 400k. For each optimizer, the learning rate selected by final FID is held fixed when evaluating the earlier budget fractions. Muon-C has the lowest FID at every reported fraction on both datasets. Its margin is largest at 0.1B, where it reaches 10.37 on CIFAR-10 compared with 24.97 for global unfolded Muon and 51.37 for Adam, and 21.14 on ImageNet-1k-32 compared with 28.02 and 58.87, respectively.

These results complement the final tuned comparison in Table 6. The selected runs retain the same optimizer ordering at 0.1B and 0.5B.

## C.3 Local-Geometry and Finite-Support Diagnostics

## C.3.1 Local-Geometry Protocol

This diagnostic provides the detailed protocol for Figure 10 in Section 6.5. We examine shared states from a Muon-C CIFAR-10 U-Net trajectory at 40k, 200k, and 400k iterations. At each state, we use one fixed 512-example evaluation objective and construct the global unfolded-Muon, Muon-S, and Muon-C directions from the same Nesterov momentum signal. For this diagnostic, all three directions use exact SVD-based polar factors rather than the Newton-Schulz approximation used during training. We jointly perturb the 49 eligible $3 \times 3$ stride-one convolution kernels shared by the comparison.

Each candidate direction is scaled per layer to the same convolution-operator-norm budget, estimated on a $3 2 \times 3 2$ Fourier grid. Let $D _ { \ell , \mathrm { u n f } }$ denote the raw unfolded-Muon direction for layer $\ell ,$ and let $\| \cdot \| _ { \mathrm { c o n v , 3 2 } }$ denote this grid estimate. We set

$$
\rho _ { \ell } = 4 \times 1 0 ^ { - 5 } \frac { \| D _ { \ell , \mathrm { u n f } } \| _ { \mathrm { c o n v } , 3 2 } } { \mathrm { R M S } ( D _ { \ell , \mathrm { u n f } } ) } , \qquad V _ { \ell , r } = \rho _ { \ell } \frac { D _ { \ell , r } } { \| D _ { \ell , r } \| _ { \mathrm { c o n v } , 3 2 } } ,
$$

where $\mathrm { R M S } ( D ) = \| D \| _ { F } / \sqrt { | D | }$ and r indexes the three methods. Thus, the unfolded-Muon perturbation has coeficient RMS $4 \times 1 0 ^ { - 5 }$ at unit radius. Figure 10 evaluates the joint perturbation $\alpha V _ { r }$ for $\alpha \in \{ 0 , 0 . 2 5 , 0 . 5 , 1 , 1 . 5 , 2 , 3 , 4 , 6 , 8 \}$

For a candidate direction V , Section 6.5 uses the local quadratic approximation (9). We use central finite diferences at $\theta \pm h V$ with $h = 0 . 2 5$ in the radius-multiplier units above. The Hessian estimate is

$$
{ \widehat { c } } _ { H } = { \frac { L ( \theta + h V ) - 2 L ( \theta ) + L ( \theta - h V ) } { h ^ { 2 } } } .
$$

The MSE Gauss–Newton estimate is twice the mean squared central diference of model outputs, with each output diference divided by 2h. Both estimates use the same fixed inputs and stochastic draws at the perturbed states.

## C.3.2 Finite-Support Diagnostic

Table 8 in Section 6.5 examines whether independent frequency-wise polarization on the full feature-frequency grid can be restricted back to the stored finite kernel support. Using stored Muon-C momenta from a completed CIFAR-10 run, let $U _ { k }$ denote the update obtained by polarizing on the $3 \times 3$ critical grid and let $U _ { \Omega }$ denote the update obtained from the same momentum on the layer’s full feature grid. Let P crop a full-grid update to the stored support and let E zero-embed a cropped kernel back into the feature grid. We measure

$$
d _ { \mathrm { s u p p } } = \frac { \lVert U _ { k } - P U _ { \Omega } \rVert _ { F } } { \lVert P U _ { \Omega } \rVert _ { F } } , \qquad \ell = \frac { \lVert U _ { \Omega } - E P U _ { \Omega } \rVert _ { F } ^ { 2 } } { \lVert U _ { \Omega } \rVert _ { F } ^ { 2 } } , \qquad d _ { \mathrm { f u l l } } = \frac { \lVert E U _ { k } - U _ { \Omega } \rVert _ { F } } { \lVert U _ { \Omega } \rVert _ { F } } .
$$

Here, $d _ { \mathrm { s u p p } }$ measures the discrepancy between the critical-grid update and the coeficients retained after cropping the full-grid update, ℓ is the fraction of full-grid energy outside the stored support, and $d _ { \mathrm { f u l l } }$ measures their discrepancy on the complete feature grid.

The diagnostic uses nine representative eligible stride-one layers spanning $4 \times 4$ to $3 2 \times 3 2$ feature grids at 40k, 200k, and 400k iterations. As reported in Table 8, the obstruction is substantial and stable throughout training. As numerical checks, the finite-support identity

$$
d _ { \mathrm { f u l l } } ^ { 2 } = ( 1 - \ell ) d _ { \mathrm { s u p p } } ^ { 2 } + \ell
$$

has maximum residual $1 . 7 9 \times 1 0 ^ { - 7 }$ . Exact-SVD checks of the idealized full-grid LMO on one layer at each checkpoint attain maximum relative objective gap $9 . 2 7 \times 1 0 ^ { - 8 }$ and feasibility error $7 . 1 5 \times 1 0 ^ { - 7 }$ . Thus, the full-grid oracle is solved accurately; the observed discrepancy reflects the incompatibility between independent full-grid frequency polarization and the original finite kernel support.

## References

J. Bernstein and L. Newhouse. Old optimizer, new norm: An anthology, 2024.

J. Bernstein and L. Newhouse. Modular duality in deep learning. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 3920–3930. PMLR, 2025. URL https://proceedings.mlr. press/v267/bernstein25a.html.

V. Bogachev, V. Aletov, A. Molozhavenko, S. Kudriashov, and M. Rakhuba. Tensorion: A tensor-aware generalization of the muon optimizer, 2026.

L. Chen, J. Li, and Q. Liu. Muon optimizes under spectral norm constraints, 2025.

V. Gupta, T. Koren, and Y. Singer. Shampoo: Preconditioned stochastic tensor optimization. In International Conference on Machine Learning, pages 1842–1850, 2018.

K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

N. J. Higham. Functions of Matrices: Theory and Computation. SIAM, 2008.

Z. Huang, H. Cao, F. Dong, R. Huang, M. Chen, Y. Yang, X. Zhang, A. Chen, M. Dong, Y. Wang, J. Hou, Q. Lv, R. P. Dick, Y. Cheng, F. Yang, T. Lu, and L. Shang. Spectra: Rethinking optimizers for llms under spectral anisotropy, 2026.

K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, and J. Bernstein. Muon: An optimizer for hidden layers in neural networks. https://kellerjordan.github.io/ posts/muon/, 2024. Accessed: 2026-05-31.

D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

J. Liu, J. Su, X. Yao, Z. Jiang, G. Lai, Y. Du, Y. Qin, W. Xu, E. Lu, J. Yan, Y. Chen, H. Zheng, Y. Liu, S. Liu, B. Yin, W. He, H. Zhu, Y. Wang, J. Wang, M. Dong, Z. Zhang, Y. Kang, H. Zhang, X. Xu, Y. Zhang, Y. Wu, X. Zhou, and Z. Yang. Muon is scalable for LLM training, 2025.

Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11976–11986, 2022.

I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

J. Martens and R. Grosse. Optimizing neural networks with kronecker-factored approximate curvature. In International Conference on Machine Learning, pages 2408–2417, 2015.

T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida. Spectral normalization for generative adversarial networks. In International Conference on Learning Representations, 2018.

T. Pethick, W. Xie, K. Antonakopoulos, Z. Zhu, A. Silveti-Falls, and V. Cevher. Training deep learning models with norm-constrained LMOs, 2025.

O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention, pages 234–241, 2015.

H. Sedghi, V. Gupta, and P. M. Long. The singular values of convolutional layers. In International Conference on Learning Representations, 2019.

Z. Shumaylov, N. Da Costa, P. Zaika, B. Mucsanyi, A. Massucco, Y. Gelberg, C.-B. Schonlieb, Y. Gal, and P. Hennig. Muon is not that special: Random or inverted spectra work just as well, 2026.

S. Singla and S. Feizi. Fantastic four: Diferentiable and eficient bounds on singular values of convolution layers. In International Conference on Learning Representations, 2021.

S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I. S. Kweon, and S. Xie. ConvNeXt V2: Co-designing and scaling convnets with masked autoencoders. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16133– 16142, 2023.