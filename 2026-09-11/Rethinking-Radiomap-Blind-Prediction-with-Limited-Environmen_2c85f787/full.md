# Rethinking Radiomap Blind Prediction with Limited Environment and Configuration Representations

Xiaojie Li<sup>∗†</sup>, Yu Han<sup>∗†</sup>, Han Fang<sup>‡</sup>, Shangqing Liu<sup>§</sup>, Shi Jin<sup>∗†</sup>, and Chao-Kai Wen<sup>¶</sup>

<sup>∗</sup>School of Information Science and Engineering, Southeast University, Nanjing 210096, China

<sup>†</sup>National Mobile Communications Research Laboratory, Southeast University, Nanjing 210096, China

<sup>‡</sup> School of Cyber Science and Engineering, Southeast University, Nanjing 210096, China

<sup>§</sup> State Key Laboratory of Novel Software Technology, Nanjing University, Nanjing 210023, China

<sup>¶</sup>Institute of Communications Engineering, National Sun Yat-sen University, Kaohsiung 80424, Taiwan Email: xiaojieli@seu.edu.cn, hanyu@seu.edu.cn, h fang@seu.edu.cn,

shangqingliu@nju.edu.cn, jinshi@seu.edu.cn and chaokai.wen@mail.nsysu.edu.tw

Abstract—Radiomap blind prediction infers radiomaps from observable representations of the propagation environment and base station (BS) configuration without field measurements. These representations are inherently incomplete and cannot uniquely determine the target radiomap. Under squared loss, we identify the conditional-mean radiomap as the populationoptimal deterministic target and decompose domain risk into target-approximation error and irreducible uncertainty. The train-test risk gap motivates propagation priors as cross-domain guidance, although their partial or simplified forms may bias the attainable predictor. We therefore propose RadioDecomp, which treats a prior-guided predictor as a correctable base and uses deterministic residual refinement to learn its remaining predictable discrepancy. We instantiate RadioDecomp as RadioLSR (LoS-Shadow-Residual). Experiments under cross-configuration and cross-environment settings show that RadioLSR is especially effective for cross-configuration generalization and provides overall gains over a controlled monolithic counterpart under crossenvironment generalization.

Index Terms—radiomap, incomplete observation, generalization, residual learning, wireless propagation

## I. INTRODUCTION

Future 6G networks are expected to rely on environmentaware radio intelligence, creating strong demand for radiomaps that characterize the spatial distribution of radio attributes such as received power and angle of arrival [1]. Such radiomaps are valuable for wireless network planning, optimization, and predictive radio management. In many practical scenarios, the target radiomap must be inferred without field measurements. This gives rise to radiomap blind prediction, which predicts radiomaps only from observable representations of the environment and base station (BS) configuration [2], [3].

Existing studies have mainly approached this task by learning direct predictive mappings, including environment-driven methods such as RadioUNet [4], RME-GAN [5], and RadioDiff [6], configuration-aware methods that incorporate BSside operational parameters [7], and more recent settings with jointly varying environments and BS configurations [3], [8]. Beyond these learning-based approaches, ray-tracing-based simulation [9] can be viewed as a physics-driven counterpart when the environment and BS configuration are fully specified. However, only partial environment and configuration representations are available in practical blind prediction.

A related but distinct line of work studies radiomap estimation from sparse radiomap measurements [10]. Although noisy environment information is considered in [11], such methods still rely on measurement support and therefore do not directly characterize blind radiomap prediction under incomplete observation.

This raises a fundamental question: what does a deterministic blind predictor learn under incomplete observation? Under squared loss, we identify the conditional-mean radiomap as its population-optimal target and decompose domain risk into target-approximation error and irreducible uncertainty, whose domain-wise changes determine the train-test risk gap. Propagation priors provide natural cross-domain guidance for limiting the predictor-dependent approximation gap, but their partial or simplified forms may bias the attainable predictor. We therefore propose RadioDecomp, which retains a prior-guided TIC base and uses a deterministic RIC to correct its predictable discrepancy, and instantiate it as RadioLSR. Experiments on a multi-configuration and multi-environment radiomap dataset [3] show consistent gains under cross-configuration prediction and overall benefits under cross-environment prediction.

## II. PROBLEM FORMULATION AND ANALYSIS

## A. Problem Formulation under Incomplete Observation

Let E denote the complete physical environment and C the complete BS configuration. The former includes propagationrelevant scene information, such as geometry and material properties, while the latter specifies the BS location and transmission setup, including the antenna array, carrier frequency, and beamforming configuration. A radiomap is a discretized spatial representation of a radio attribute, such as received power, angle of arrival, or angle of departure. In the 2D setting considered here, a radiomap realization is represented by $\mathbf { y } ^ { \mathsf { ^ { - } } } \in \mathbb { R } ^ { H \times W }$ ; the same formulation can be extended to voxelized 3D radiomaps [12], [13].

When E and C are fully specified, radiomap generation can be expressed as

$$
\mathbf { y } = F ( E , C ) ,\tag{1}
$$

where $F$ denotes the deterministic physical propagation mechanism.

Radiomap blind prediction differs because the predictor receives only incomplete representations of the environment and BS configuration. Let $\textbf { Z } = \ ( E _ { \mathrm { o b s } } , C _ { \mathrm { o b s } } )$ denote the observable representation, and let U collect the remaining hidden propagation factors. The complete physical state can then be partitioned into (Z, U), such that

$$
\mathbf { Y } = F ( \mathbf { Z } , \mathbf { U } ) .\tag{2}
$$

For a given observable input z, multiple hidden states may be compatible with it and produce different radiomaps. Accordingly,

$$
\mathbf { Y } \mid ( \mathbf { Z } = \mathbf { z } ) = F ( \mathbf { z } , \mathbf { U } ) , \qquad \mathbf { U } \sim p ( \cdot \mid \mathbf { z } ) .\tag{3}
$$

Although U is unobserved, a deterministic blind predictor must return one radiomap for each z. Under squared loss, its population-optimal output is

$$
\begin{array} { r l } & { m ^ { \star } ( \mathbf { z } ) : = \arg \operatorname* { m i n } _ { \hat { \mathbf { y } } } \mathbb { E } \Big [ \| \mathbf { Y } - \widehat { \mathbf { y } } \| _ { \mathrm { F } } ^ { 2 } \mid \mathbf { Z } = \mathbf { z } \Big ] } \\ & { \qquad = \mathbb { E } [ \mathbf { Y } \mid \mathbf { Z } = \mathbf { z } ] = \mathbb { E } _ { \mathbf { U } \sim p ( \cdot \mid \mathbf { z } ) } \left[ F ( \mathbf { z } , \mathbf { U } ) \right] . } \end{array}\tag{4}
$$

Thus, deterministic blind prediction under squared loss seeks to approximate the conditional-mean radiomap $m ^ { \star } ( \mathbf { z } )$ rather than the radiomap associated with any particular realization of the hidden physical factors.

## B. Domain-Wise Risk Decomposition and Train-Test Gap

Let $d \ \in \ \{ \mathrm { t r } , \mathrm { t e } \}$ index the training and test domains, respectively. The population-optimal deterministic predictor in domain d is

$$
m _ { d } ^ { \star } ( \mathbf { z } ) : = \mathbb { E } _ { d } [ \mathbf { Y } \mid \mathbf { Z } = \mathbf { z } ] = \mathbb { E } _ { \mathbf { U } \sim p _ { d } ( \cdot \mid \mathbf { z } ) } \left[ F ( \mathbf { z } , \mathbf { U } ) \right] .\tag{5}
$$

Define the corresponding conditional uncertainty as

$$
v _ { d } ( \mathbf { z } ) : = \mathbb { E } _ { d } \Big [ \| \mathbf { Y } - \boldsymbol { m } _ { d } ^ { \star } ( \mathbf { z } ) \| _ { \mathrm { F } } ^ { 2 } \mid \mathbf { Z } = \mathbf { z } \Big ] .\tag{6}
$$

Theorem 1 (Domain-Wise Risk Decomposition). For any deterministic predictor f with finite second moment and any $d \in \{ \mathrm { t r } , \mathrm { t e } \}$ , the population risk satisfies

$$
\begin{array} { r l } & { R _ { d } ( f ) : = \mathbb { E } _ { d } \Big [ \| \mathbf { Y } - f ( \mathbf { Z } ) \| _ { \mathrm { F } } ^ { 2 } \Big ] } \\ & { \qquad = \underbrace { \mathbb { E } _ { d } \Big [ \| f ( \mathbf { Z } ) - m _ { d } ^ { \star } ( \mathbf { Z } ) \| _ { \mathrm { F } } ^ { 2 } \Big ] } _ { A _ { d } ( f ) } + \underbrace { \mathbb { E } _ { d } [ v _ { d } ( \mathbf { Z } ) ] } _ { \mathcal { U } _ { d } } , } \end{array}\tag{7}
$$

where $\textstyle { \mathcal { A } } _ { d } ( f )$ is the domain-specific target-approximation error and U<sub>d</sub> is the irreducible uncertainty induced by incomplete observation.

Proof. Adding and subtracting $m _ { d } ^ { \star } ( \mathbf { Z } )$ in the squared error yields an approximation term, an uncertainty term, and a cross term. The cross term vanishes because

$$
\mathbb { E } _ { d } [ \mathbf { Y } - m _ { d } ^ { \star } ( \mathbf { Z } ) \mid \mathbf { Z } ] = 0 .
$$

The remaining two terms are exactly $\textstyle { \mathcal { A } } _ { d } ( f )$ and $\mathcal { U } _ { d } .$ , which proves (7). □

Since the same fixed predictor can be evaluated under both domains, subtracting their risk decompositions gives the following result.

Corollary 1 (Train-Test Risk Gap). For any fixed deterministic predictor f,

$$
\begin{array} { r } { R _ { \mathrm { t e } } ( f ) - R _ { \mathrm { t r } } ( f ) = \underbrace { \mathcal { A } _ { \mathrm { t e } } ( f ) - \mathcal { A } _ { \mathrm { t r } } ( f ) } _ { t a r g e t - a p p r o x i m a t i o n ~ g a p } } \\ { + \underbrace { \mathcal { U } _ { \mathrm { t e } } - \mathcal { U } _ { \mathrm { t r } } } _ { c o n d i t i o n a l - u n c e r t a i n t y ~ g a p } . } \end{array}\tag{8}
$$

Theorem 1 separates prediction risk into a predictordependent approximation term and predictor-independent uncertainty. Corollary 1 further shows that the train-test risk difference is determined by domain-dependent changes in these two terms. If a predictor achieves a small training-domain approximation error, good generalization requires its approximation error not to increase substantially from the training domain to the test domain. By contrast, the conditionaluncertainty gap is determined by the observation and domain distributions and cannot be reduced through predictor design.

Remark 1. The squared loss is used to identify the conditionalmean target and obtain the exact risk decomposition. It does not require f to be trained using squared loss; the analysis applies to any resulting predictor with finite second moment, including one obtained using an $\ell _ { 1 }$ training objective.

Reducing the predictor-dependent train-test gap requires learning relations that remain useful beyond the training samples. The physical propagation mechanism F provides a natural source of such guidance because its underlying propagation regularities are shared across environments and BS configurations. Accordingly, incorporating propagation knowledge into the predictor may reduce the finite-sample difficulty of discovering transferable relations and help maintain a small approximation gap across domains.

However, under incomplete observation, an implementable propagation prior can describe only selected or simplified aspects of F. Let Φ denote such a prior and let ${ \mathcal { H } } _ { \Phi }$ denote the effective hypothesis class induced by it. Its domain-specific structural approximation bias can be expressed as

$$
\mathcal { B } _ { \Phi , d } : = \operatorname* { i n f } _ { b \in \mathcal { H } _ { \Phi } } \mathbb { E } _ { d } \Big [ \| b ( \mathbf { Z } ) - m _ { d } ^ { \star } ( \mathbf { Z } ) \| _ { \mathrm { F } } ^ { 2 } \Big ] .\tag{9}
$$

When $B _ { \Phi , d } ~ > ~ 0 .$ , even the best predictor permitted by the prior-guided base class cannot attain the conditional target. Thus, propagation priors may improve finite-sample generalization while introducing an approximation bias that limits the attainable predictor. This benefit-bias tradeoff motivates a correctable prior-guided predictor.

## C. RadioDecomp as a Correctable Prior-Guided Predictor

Accordingly, let $f _ { \mathrm { T I C } } \in \mathcal { H } _ { \Phi }$ denote a propagation-priorguided base predictor, referred to as the transferable interaction component (TIC). It is intended to capture propagation relations that remain useful across domains, but may deviate from the conditional target because of prior-induced bias, finite-sample learning, or imperfect optimization. To retain this propagation guidance while correcting its predictable discrepancy, we introduce RadioDecomp:

$$
\hat { \mathbf { y } } = f ( \mathbf { z } ) = f _ { \mathrm { T I C } } ( \mathbf { z } ) + f _ { \mathrm { R I C } } ( \mathbf { z } , f _ { \mathrm { T I C } } ( \mathbf { z } ) ) ,\tag{10}
$$

where $f _ { \mathrm { R I C } } ( \mathbf { z } , f _ { \mathrm { T I C } } ( \mathbf { z } ) )$ denotes the residual interaction component (RIC). The residual predictor exploits observable relations not captured by TIC and uses $f _ { \mathrm { T I C } } ( \mathbf { z } )$ as a propagationguided prediction anchor.

To formalize its correctability, consider a fixed structured base $f _ { \mathrm { T I C } }$ and a residual hypothesis class $\mathcal { H } _ { R }$ . Define

$$
\begin{array} { r l } & { \mathcal { H } _ { \mathrm { R D } } ( f _ { \mathrm { T I C } } ) = \Big \{ f : f ( \mathbf { z } ) = f _ { \mathrm { T I C } } ( \mathbf { z } ) } \\ & { \qquad + r ( \mathbf { z } , f _ { \mathrm { T I C } } ( \mathbf { z } ) ) , \quad r \in \mathcal { H } _ { R } \Big \} . } \end{array}\tag{11}
$$

Proposition 1 (Deterministic Residual Correctability). If the zero function belongs to the residual class, $0 \in \mathcal { H } _ { R }$ , then, for any $d \in \{ \mathrm { t r } , \mathrm { t e } \}$ ,

$$
\operatorname* { i n f } _ { f \in \mathcal { H } _ { \mathrm { R D } } ( f _ { \mathrm { T I C } } ) } R _ { d } ( f ) \leq R _ { d } ( f _ { \mathrm { T I C } } ) .\tag{12}
$$

Moreover, under squared loss, the pointwise populationoptimal deterministic residual in domain d is

$$
f _ { \mathrm { R I C } , d } ^ { \star } ( \mathbf { z } , f _ { \mathrm { T I C } } ( \mathbf { z } ) ) = m _ { d } ^ { \star } ( \mathbf { z } ) - f _ { \mathrm { T I C } } ( \mathbf { z } ) .\tag{13}
$$

Proof. Because $0 \in \mathcal { H } _ { R } .$ choosing the zero residual recovers $f _ { \mathrm { T I C } }$ , which proves (12). For a fixed base, $f _ { \mathrm { T I C } } ( \mathbf { Z } )$ is deterministic given Z, and hence

$$
\begin{array} { r } { \mathbb { E } _ { d } [ { \mathbf { Y } } - f _ { \mathrm { T I C } } ( { \mathbf { Z } } ) \mid { \mathbf { Z } } = { \mathbf { z } } ] } \\ { = m _ { d } ^ { \star } ( { \mathbf { z } } ) - f _ { \mathrm { T I C } } ( { \mathbf { z } } ) . } \end{array}\tag{14}
$$

Since the conditional mean minimizes conditional squared loss, (13) follows. □

Thus, RadioDecomp retains the prior-guided base as a special case while providing a deterministic correction path toward the domain-specific conditional target. The residual branch can correct the predictable approximation bias of the base.

## III. A RADIODECOMP INSTANTIATION: RADIOLSR

Following this principle, we instantiate RadioDecomp as RadioLSR, where LSR stands for LoS-Shadow-Residual. RadioLSR uses LoS-dominant and blockage-dominant shadowing effects as an observable-supported base estimate, rather than as a complete radiomap decomposition, and refines the remaining discrepancy through a residual part. Accordingly, RadioLSR is written as

$$
\hat { \mathbf { y } } = f ( \mathbf { z } ) = f _ { \mathrm { b a s e } } ( \mathbf { z } ) + f _ { \mathrm { r e s } } ( \mathbf { z } , f _ { \mathrm { b a s e } } ( \mathbf { z } ) ) ,\tag{15}
$$

where $f _ { \mathrm { b a s e } }$ and $f _ { \mathrm { r e s } }$ are the RadioLSR instantiations of $f _ { \mathrm { T I C } }$ and f , respectively. In RadioLSR, both predictors are realized at the prediction level with the corresponding spatial masks incorporated into their outputs.

## A. Physics-Aware Input Representation

We instantiate RadioLSR on the U6G XL-MIMO Radiomap dataset [3]. For each sample, the observable input consists of a beam map $\mathbf { B } \in \mathbb { R } ^ { H \times \dot { W } }$ , a height map $\mathbf { H } \in \mathbb { R } ^ { H \times W }$ , two directional edge maps ${ \bf E } ^ { ( 1 ) } , { \bf E } ^ { ( 2 ) } \in \mathbb { R } ^ { H \times } \dot { \bf W }$ extracted from the height map along the two spatial axes, and a blockage score map $\mathbf { S } \in \dot { \mathbb { R } } ^ { H \times W }$ . Let $k \in \{ 1 , \ldots , H W \}$ denote the index of a spatial grid point in the $H \times W$ observation domain. Then the k-th element of the beam map is defined as [3]

$$
B _ { k } = \frac { \lambda ^ { 2 } } { ( 4 \pi ) ^ { 2 } } P _ { t } \left| \mathbf { w } ^ { H } \mathbf { H } _ { k } ^ { \mathrm { L o S } } \right| ^ { 2 } ,\tag{16}
$$

which represents the analytically computed LoS beamforming power at grid point k under the given BS configuration and beamforming setting. The height map records building heights, while the two directional edge maps provide boundary cues along the horizontal and vertical grid directions.

Prediction is performed only over the valid non-building region. We therefore define a binary valid-region mask $\mathbf { M } _ { \mathrm { v a l i d } } \in$ $\mathsf { \bar { \{ 0 , 1 \} } } H \times W$ , whose (i, j)-th entry equals 1 if the corresponding grid cell belongs to the valid prediction region and 0 otherwise.

To facilitate the modeling of blockage-aware attenuation, we further construct a blockage score for each grid point k by sampling M points along the BS-grid line segment and comparing the direct-ray height with the corresponding building height:

$$
S _ { k } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \Big [ h _ { \mathrm { b l d } } \big ( \mathbf { p } _ { \mathrm { B S } } + t _ { m } \big ( \mathbf { p } _ { k } - \mathbf { p } _ { \mathrm { B S } } \big ) \big ) - h _ { \mathrm { r a y } } \big ( t _ { m } \big ) \Big ] _ { + } ,\tag{17}
$$

where p<sub>BS</sub> and $\mathbf { p } _ { k }$ denote the BS position and the horizontal position of grid point k, respectively, $\{ t _ { m } \} _ { m = 1 } ^ { M } \subset ( 0 , 1 )$ are uniformly sampled interpolation factors, $h _ { \mathrm { b l d } } ( \cdot )$ denotes the building height, $h _ { \mathrm { r a y } } ( t _ { m } )$ is the ray height at interpolation point $t _ { m }$ , and $[ x ] _ { + } = \operatorname* { m a x } ( x , 0 )$ . Thus, $S _ { k }$ characterizes the blockage severity along the BS-grid path. Based on S, we define two binary masks $\mathbf { M } _ { \mathrm { L o S } } , \bar { \mathbf { M } } _ { \mathrm { S h d } } \mathbf { \bar { \Omega } } \in \{ 0 , 1 \} ^ { H \times W }$ as

$$
\mathbf { M } _ { \mathrm { L o S } } = \mathbb { I } ( \mathbf { S } \leq \epsilon ) \odot \mathbf { M } _ { \mathrm { v a l i d } } , \qquad \mathbf { M } _ { \mathrm { S h d } } = \mathbb { I } ( \mathbf { S } > \epsilon ) \odot \mathbf { M } _ { \mathrm { v a l i d } } ,\tag{18}
$$

where ϵ is a blockage-score threshold for separating LoSdominant and blockage-dominant regions, and I(·) denotes the element-wise indicator function. Thus, $\mathbf { M } _ { \mathrm { L o S } }$ and $\mathbf { M } _ { \mathrm { S h d } }$ partition the valid region according to the blockage score.

## B. RadioLSR Architecture

RadioLSR consists of three U-Net branches for LoS prediction, shadow prediction, and residual refinement. The LoS and Shadow branches jointly realize the structured base predictor $f _ { \mathrm { b a s e } } ,$ while the residual branch realizes $f _ { \mathrm { r e s } }$ . This design reflects the available input representation: coarse height map errors mainly affect LoS/shadow boundaries, whereas reflection-related patterns are more sensitive to fine geometry and surface orientation and are therefore deferred to residual refinement.

All three branches adopt the same U-Net template summarized in Table I, but differ in their input channels, prediction roles, and base channel width $C _ { b }$ . The base branches use restricted inputs so that the base stage focuses on relatively stable and explicitly supported structure, while harder effects are deferred to refinement.

1) Base Branch (LoS and Shadow): The base stage contains two U-Net branches. The LoS branch is intended to capture the more direct coverage trend induced by the BS configuration. Since this component is already strongly reflected by the beam map, its prediction is defined as

$$
\hat { \mathbf { y } } _ { \mathrm { L o S } } = f _ { \mathrm { L o S } } ( \mathbf { B } ) \odot \mathbf { M } _ { \mathrm { L o S } } .\tag{19}
$$

This restriction helps keep the branch focused on the most stable part of the structured base predictor.

The Shadow branch is intended to capture blockage-aware attenuation, which remains structured but depends more explicitly on obstruction cues. Its prediction is defined as

$$
\begin{array} { r } { \hat { \mathbf { y } } _ { \mathrm { S h d } } = f _ { \mathrm { S h d } } ( [ \mathbf { B } , \mathbf { S } ] ) \odot \mathbf { M } _ { \mathrm { S h d } } . } \end{array}\tag{20}
$$

Here, the blockage score provides a compact cue for obstruction severity, while the beam map preserves the configurationdependent coverage tendency. This restricted input design encourages the branch to focus on blockage-related attenuation.

The structured base predictor is then defined as

$$
f _ { \mathrm { b a s e } } ( \mathbf { z } ) : = \hat { \mathbf { y } } _ { \mathrm { L o S } } + \hat { \mathbf { y } } _ { \mathrm { S h d } } .\tag{21}
$$

Equivalently,

$$
f _ { \mathrm { b a s e } } ( \mathbf { z } ) = f _ { \mathrm { L o S } } ( \mathbf { B } ) \odot \mathbf { M } _ { \mathrm { L o S } } + f _ { \mathrm { S h d } } ( [ \mathbf { B } , \mathbf { S } ] ) \odot \mathbf { M } _ { \mathrm { S h d } } .\tag{22}
$$

Thus, $f _ { \mathrm { b a s e } }$ realizes the TIC stage in RadioLSR through masked aggregation of the two base-branch predictions.

2) Residual Refinement: The residual stage contains one additional U-Net branch that refines the remaining discrepancy in the dB domain. Let the residual U-Net branch be denoted by $g _ { \mathrm { r e s } } ( \cdot )$ . We define the residual predictor as

$$
f _ { \mathrm { r e s } } ( \mathbf { z } , f _ { \mathrm { b a s e } } ( \mathbf { z } ) ) : = g _ { \mathrm { r e s } } ( [ \mathbf { B } , \mathbf { H } , \mathbf { E } ^ { ( 1 ) } , \mathbf { E } ^ { ( 2 ) } , f _ { \mathrm { b a s e } } ( \mathbf { z } ) ] ) { \odot } \mathbf { M } _ { \mathrm { v a l i d } } .\tag{23}
$$

Accordingly, the final prediction is

$$
\hat { \mathbf { y } } = f _ { \mathrm { b a s e } } ( \mathbf { z } ) + f _ { \mathrm { r e s } } ( \mathbf { z } , f _ { \mathrm { b a s e } } ( \mathbf { z } ) ) .\tag{24}
$$

Conditioning the residual U-Net on $f _ { \mathrm { b a s e } } ( \mathbf { z } )$ is consistent with the additive composition in the dB domain, so refinement is learned relative to the current attenuation level rather than as an independent full-map prediction. The residual branch is not additionally fed with blockage score, since blockage-aware effects have already been assigned to the base stage.

TABLE I  
Unified U-Net backbone of the RadioLSR branches.
<table><tr><td>Stage</td><td>Specification</td></tr><tr><td>Encoder</td><td>DoubleConv (Cb), then 3×[MaxPool + DoubleConv] with channels 2Cb, 4Cb, 8Cb</td></tr><tr><td>Bottleneck Decoder</td><td>DoubleConv  $( 8 C _ { b } )$  3×[Up + skip concat + DoubleConv] with channels  $4 C _ { b } , \mathsf { \bar { 2 } } C _ { b } , C _ { b } ,$  then 1×1 Conv (1)</td></tr></table>

## C. Training Objective

Let y denote the ground-truth normalized radiomap. We train RadioLSR with a masked $\ell _ { 1 }$ loss on the final prediction together with auxiliary supervision on the two base branches (LoS and Shadow) and the residual branch:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { m a i n } } + \lambda _ { \mathrm { L o S } } \mathcal { L } _ { \mathrm { L o S } } + \lambda _ { \mathrm { S h d } } \mathcal { L } _ { \mathrm { S h d } } + \lambda _ { \mathrm { r e s } } \mathcal { L } _ { \mathrm { r e s } } .\tag{25}
$$

where<sup>1</sup> $\hat { \mathbf { y } } _ { \mathrm { r e s } } = f _ { \mathrm { r e s } } ( \mathbf { z } , f _ { \mathrm { b a s e } } ( \mathbf { z } ) )$ , and

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { m a i n } } = \left\| \left( \hat { \mathbf { y } } - \mathbf { y } \right) \odot \mathbf { M } _ { \mathrm { v a l i d } } \right\| _ { 1 } / \left\| \mathbf { M } _ { \mathrm { v a l i d } } \right\| _ { 1 } , } \\ & { \mathcal { L } _ { \mathrm { L o S } } = \left\| \left( \hat { \mathbf { y } } _ { \mathrm { L o S } } - \mathbf { y } \right) \odot \mathbf { M } _ { \mathrm { L o S } } \right\| _ { 1 } / \left\| \mathbf { M } _ { \mathrm { L o S } } \right\| _ { 1 } , } \\ & { \mathcal { L } _ { \mathrm { S h d } } = \left\| \left( \hat { \mathbf { y } } _ { \mathrm { S h d } } - \mathbf { y } \right) \odot \mathbf { M } _ { \mathrm { S h d } } \right\| _ { 1 } / \left\| \mathbf { M } _ { \mathrm { S h d } } \right\| _ { 1 } , } \\ & { \mathcal { L } _ { \mathrm { r e s } } = \left\| \left( \hat { \mathbf { y } } _ { \mathrm { r e s } } - \mathbf { y } _ { \mathrm { r e s } } ^ { \star } \right) \odot \mathbf { M } _ { \mathrm { v a l i d } } \right\| _ { 1 } / \left\| \mathbf { M } _ { \mathrm { v a l i d } } \right\| _ { 1 } . } \end{array}\tag{26}
$$

The residual target is defined as

$$
\mathbf { y } _ { \mathrm { r e s } } ^ { \star } = \left( \mathbf { y } - \mathrm { s g } [ f _ { \mathrm { b a s e } } ( \mathbf { z } ) ] \right) \odot \mathbf { M } _ { \mathrm { v a l i d } } ,\tag{27}
$$

where $\mathrm { s g } [ \cdot ]$ denotes the stop-gradient operation. Here, ${ \mathcal { L } } _ { \mathrm { m a i n } }$ supervises the final composed prediction, while the other three terms encourage LoS, shadow, and residual specialization. Optimization is performed using AdamW with a validationbased ReduceLROnPlateau scheduler [14].

## IV. EXPERIMENTS

The experiments evaluate RadioLSR and examine whether a RadioDecomp-guided structured predictor generalizes better than a monolithic counterpart under incomplete observation.

## A. Experimental Setup

Experiments are conducted on the U6G XL-MIMO Radiomap dataset [3], which contains 78,400 radiomaps from 800 urban scenes and 98 BS configurations. We consider two blindprediction protocols: cross-config, which splits data by BS configuration, and cross-env, which splits data by environment. For both protocols, six train/validation/test ratios are used: $2 / 1 / 7 , 3 / 1 / 6 , 4 / 1 / 5 , 5 / 1 / 4 , 6 / 1 / 3 ,$ and $7 / 1 / 2$

Alongside RadioLSR, we also construct a monolithic counterpart, termed MonoUNet, to provide a controlled reference under the same input information and closely related backbone family. Both models use the same observable inputs, including the beam map, height map, edge features, and blockage score. For RadioLSR, the LoS and Shadow branches use a base channel width $C _ { b }$ in Table I of 16, while the residualbranch width follows the model suffix; thus, RadioLSR-32 and RadioLSR-48 use residual widths of 32 and 48, respectively.

TABLE II  
Train MAE and test MAE/RMSE (dB) of RadioLSR-32 and MonoUNet-32 under the cross-config and cross-env splits.
<table><tr><td rowspan="3">Train/Val/Test</td><td colspan="6">Cross-config split</td><td colspan="6">Cross-env split</td></tr><tr><td colspan="3">RadioLSR-32 Test</td><td colspan="3">MonoUNet-32</td><td colspan="3">RadioLSR-32</td><td colspan="3">MonoUNet-32</td></tr><tr><td>Train MAE</td><td>MAE</td><td>Test RMSE</td><td>Train MAE</td><td>Test MAE</td><td>Test RMSE</td><td>Train MAE</td><td>Test MAE</td><td>Test RMSE</td><td>Train MAE</td><td>Test MAE</td><td>Test RMSE</td></tr><tr><td>2/1/7</td><td>1.8044</td><td>3.5673</td><td>5.8391</td><td>2.2042</td><td>3.8999</td><td>6.4388</td><td>2.3412</td><td>4.3533</td><td>8.4919</td><td>2.4091</td><td>4.4039</td><td>8.5521</td></tr><tr><td>3/1/6</td><td>1.8543</td><td>3.5525</td><td>5.6258</td><td>2.0091</td><td>4.2193</td><td>6.6179</td><td>2.3224</td><td>4.1862</td><td>8.5254</td><td>2.3499</td><td>4.3337</td><td>8.4838</td></tr><tr><td>4/1/5</td><td>1.6079</td><td>3.6721</td><td>5.5804</td><td>1.9079</td><td>4.0620</td><td>6.2850</td><td>2.6196</td><td>4.1241</td><td>8.1609</td><td>2.4224</td><td>4.1252</td><td>8.2251</td></tr><tr><td>5/1/4</td><td>1.5868</td><td>3.7785</td><td>5.6960</td><td>1.8445</td><td>4.3276</td><td>6.3687</td><td>2.4463</td><td>4.0494</td><td>8.0615</td><td>2.0895</td><td>4.2097</td><td>8.4318</td></tr><tr><td>6/1/3</td><td>1.5893</td><td>4.1106</td><td>5.8316</td><td>1.7914</td><td>4.3287</td><td>6.3111</td><td>2.3853</td><td>3.9533</td><td>7.9311</td><td>2.4924</td><td>4.0207</td><td>8.0607</td></tr><tr><td>7/1/2</td><td>1.6089</td><td>4.8613</td><td>6.4511</td><td></td><td>1.9006 5.0672</td><td>6.9664</td><td>2.1774</td><td>4.0771</td><td>8.1701</td><td>1.9845</td><td>4.2538</td><td>8.6370</td></tr></table>

TABLE III

Aggregated test-sample RMSE statistics (dB) over all considered train-ratio settings under the cross-config and cross-env splits, where P50, P90, and P95 denote the 50th, 90th, and 95th percentiles, respectively.
<table><tr><td rowspan="2">Method (Parameter Num)</td><td colspan="5">Cross-config split (test 217,600 radiomaps)</td><td colspan="5">Cross-env split (test 211,680 radiomaps)</td></tr><tr><td>Mean</td><td>Std</td><td>P50</td><td>P90</td><td>P95</td><td>Mean</td><td>Std</td><td>P50</td><td>P90</td><td>P95</td></tr><tr><td>RadioLSR-32 (4.6960M)</td><td>5.5438</td><td>1.8636</td><td>5.5328</td><td>7.8895</td><td>8.5605</td><td>7.4484</td><td>4.1423</td><td>6.7295</td><td>12.9398</td><td>14.6668</td></tr><tr><td>RadioLSR-48 (8.6039M)</td><td>5.6205</td><td>1.8905</td><td>5.5497</td><td>7.9348</td><td>8.7384</td><td>7.4001</td><td>4.0835</td><td>6.7244</td><td>12.8139</td><td>14.5418</td></tr><tr><td>MonoUNet-32 (3.1296M)</td><td>6.2092</td><td>2.1128</td><td>6.1795</td><td>8.9076</td><td>9.7229</td><td>7.6014</td><td>4.1401</td><td>6.8462</td><td>13.4127</td><td>15.0610</td></tr><tr><td>MonoUNet-48 (7.0375M)</td><td>5.7783</td><td>1.9786</td><td>5.7812</td><td>8.2649</td><td>9.0100</td><td>7.5080</td><td>4.0797</td><td>6.7620</td><td>13.2766</td><td>14.9046</td></tr><tr><td>MonoUNet-64 (12.5076M)</td><td>5.6770</td><td>1.9618</td><td>5.6744</td><td>8.1225</td><td>8.8935</td><td>7.4873</td><td>4.1230</td><td>6.7381</td><td>13.2967</td><td>14.9911</td></tr></table>

![](images/33206fa260bb4e8b086be736780da2a7a406c573e63680729834cd298329f091.jpg)  
Fig. 1: Sample-level win rate of RadioLSR-32 over MonoUNet-32 under six train-ratio settings for the crossconfig and cross-env splits.

## TABLE IV

Ablation under 6/1/3 split, comparing full RadioLSR with its TIC-only variant in the cross-config and cross-env settings.
<table><tr><td rowspan="2">Method</td><td colspan="2">Cross-config split</td><td colspan="2">Cross-env split</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td rowspan="2">RadioLSR-32 TIC only</td><td>4.1106</td><td>5.8316</td><td>3.9533</td><td>7.9311</td></tr><tr><td>5.5327</td><td>8.8174</td><td>4.9422</td><td>10.4024</td></tr><tr><td>Error reduction</td><td>1.4221</td><td>2.9858</td><td>0.9889</td><td>2.4713</td></tr></table>

For MonoUNet, the suffix denotes the base channel width of the monolithic U-Net, yielding MonoUNet-32, MonoUNet-48, and MonoUNet-64. MonoUNet is trained with the same masked $\ell _ { 1 }$ radiomap loss together with an additional Sobelbased structural loss to preserve spatial structure.

The blockage threshold is set to $\epsilon = 1 0 ^ { - 6 }$ . For RadioLSR, $\lambda _ { \mathrm { L o S } } = \lambda _ { \mathrm { S h d } } = \lambda _ { \mathrm { r e s } } = 0 . 5 .$ All models are trained for 60 epochs with initial learning rate $1 0 ^ { - 3 }$ , and the best validation checkpoint is used for evaluation. This does not conflict with Section II: the squared loss there defines the conditional mean predictor, while the training losses here only determine the learned predictor f(z). Accordingly, Mean Absolute Error (MAE) is emphasized for optimization consistency, while Root Mean Square Error (RMSE) is also reported to reflect the squared-loss perspective in the theoretical analysis.

1) RadioDecomp-Guided Main Comparison: We first compare the RadioDecomp-guided model RadioLSR-32 with its monolithic counterpart MonoUNet-32 under matched inputs and closely related backbones. Table II shows that RadioLSR-32 maintains comparable or lower training MAE in most settings, indicating that structured reparameterization does not weaken train-domain fitting. On the test sets, it consistently achieves lower MAE and RMSE under cross-config and overall better performance under cross-env, with only limited exceptions. Fig. 1 further supports this trend: RadioLSR wins on a clear majority of samples in all cross-config settings and remains above 50% in all cross-env settings.

2) Capacity and Reparameterization Analysis: We next examine whether the gain can be explained solely by increasing monolithic capacity. Table III shows that widening MonoUNet from 32 to 48 and 64 generally improves RMSE, especially under cross-config. However, the best overall results are still achieved by RadioLSR-32 or RadioLSR-48 rather than by wider MonoUNet baselines. The structured models maintain lower mean RMSE under both splits, with advantages also in P90 and P95. Although the gaps are modest, the comparison uses identical observable inputs and a large number of test radiomaps, supporting the benefit of structured reparameterization beyond monolithic scaling.

![](images/b409da1a6adc8ccaf89b9ff27b8500c7b4611b888567c7151f9285a921484b9f.jpg)  
Fig. 2: Qualitative results of one radiomap sample for RadioLSR under the cross-configuration and cross-environment splits, including the ground truth, base prediction, base error, final prediction, and final error.

3) Ablation on the Role of the RIC Refinement: To examine the refinement role implied by RadioDecomp, we compare the full RadioLSR-32 model with its TIC-only variant under the 6/1/3 split. As shown in Table IV, removing RIC causes clear degradation under both split protocols: the full model reduces MAE/RMSE by 1.4221/2.9858 dB under cross-config and by 0.9889/2.4713 dB under cross-env. This indicates that, within the RadioDecomp view, structured base estimation alone is insufficient, while the RIC stage provides an essential refinement path for the remaining discrepancy.

4) Qualitative Analysis of the Structured Prediction Path: Fig. 2 provides qualitative evidence for the structured prediction path motivated by RadioDecomp. In both cross-config and cross-env examples, the LoS-Shadow base captures the dominant large-scale coverage pattern, while the residual branch contributes localized corrections in more difficult regions, including reflection-related variations supported by the available geometric cues. This behavior is consistent with the intended roles of base estimation and residual refinement in RadioLSR.

## V. CONCLUSION

This paper revisited deterministic radiomap blind prediction under incomplete observation. Under squared loss, we identified the conditional-mean radiomap as its populationoptimal target, decomposed domain risk into approximation error and irreducible uncertainty, and characterized the traintest risk gap through their domain-wise changes. Since propagation priors offer cross-domain guidance but may bias the attainable predictor, we proposed RadioDecomp, which combines a prior-guided TIC base with a deterministic RIC, and instantiated it as RadioLSR. Experiments showed that RadioLSR is especially effective for cross-config generalization and provides overall benefits under cross-env generalization over a controlled monolithic reference. Future work may include generative RIC modeling: its conditional mean could retain base-to-target correction while its samples represent the predictive variability induced by incomplete observation.

## REFERENCES

[1] Y. Zeng, J. Chen, J. Xu, D. Wu, X. Xu, S. Jin, X. Gao, D. Gesbert, S. Cui, and R. Zhang, “A Tutorial on Environment-Aware Communications via

Channel Knowledge Map for 6G,” IEEE Communications Surveys & Tutorials, vol. 26, no. 3, pp. 1478–1519, 2024.

[2] S. Bi, J. Lyu, Z. Ding, and R. Zhang, “Engineering Radio Maps for Wireless Resource Management,” IEEE Wireless Communications, vol. 26, no. 2, pp. 133–141, 2019.

[3] X. Li, Y. Han, Z. Lu, S. Jin, and C.-K. Wen, “U6G XL-MIMO Radiomap Prediction: Multi-Config Dataset and Beam Map Approach,” 2026. [Online]. Available: https://arxiv.org/abs/2603.06401

[4] R. Levie, c. Yapar, G. Kutyniok, and G. Caire, “RadioUNet: Fast Radio Map Estimation With Convolutional Neural Networks,” IEEE Transactions on Wireless Communications, vol. 20, no. 6, pp. 4001– 4015, 2021.

[5] S. Zhang, A. Wijesinghe, and Z. Ding, “RME-GAN: A Learning Framework for Radio Map Estimation Based on Conditional Generative Adversarial Network,” IEEE Internet of Things Journal, vol. 10, no. 20, pp. 18 016–18 027, 2023.

[6] X. Wang, K. Tao, N. Cheng, Z. Yin, Z. Li, Y. Zhang, and X. Shen, “RadioDiff: An Effective Generative Diffusion Model for Sampling-Free Dynamic Radio Map Construction,” IEEE Transactions on Cognitive Communications and Networking, vol. 11, no. 2, pp. 738–750, 2025.

[7] X. Li, Z. Cai, N. Qi, C. Dong, G. Zhu, H. Ma, Q. Wu, and S. Jin, “A Disentangled Representation Learning Framework for Low-Altitude Network Coverage Prediction,” IEEE Transactions on Mobile Computing, vol. 25, no. 5, pp. 6261–6276, 2026.

[8] F. Jaensch, G. Caire, and B. Demir, “Radio Map Prediction From Aerial Images and Application to Coverage Optimization,” IEEE Transactions on Wireless Communications, vol. 25, pp. 308–320, 2026.

[9] N. Vaara, P. Sangi, M. B. Lopez, and J. Heikkil´ a, “Differentiable High-¨ Performance Ray Tracing-Based Simulation of Radio Propagation With Point Clouds,” IEEE Antennas and Wireless Propagation Letters, pp. 1–5, 2026.

[10] D. Romero, T. N. Ha, R. Shrestha, and M. Franceschetti, “Theoretical Analysis of the Radio Map Estimation Problem,” IEEE Transactions on Wireless Communications, vol. 23, no. 10, pp. 13 722–13 737, 2024.

[11] F. Jaensch, C¸ agkan Yapar, G. Caire, and B. Demir, “Radio Map Predic-˘ tion from Noisy Environment Information and Sparse Observations,” 2026. [Online]. Available: https://arxiv.org/abs/2602.11950

[12] L. Zhao, Z. Fei, X. Wang, J. Luo, and Z. Zheng, “3D-RadioDiff: An Altitude-Conditioned Diffusion Model for 3D Radio Map Construction,” IEEE Wireless Communications Letters, vol. 14, no. 7, pp. 1969–1973, 2025.

[13] Z. Liu, Q. Liu, S. Zhang, H. Zhang, and L. Song, “A Fine-Grained 3D Radio Map Construction Paradigm With Ultra-Low Sampling Rates by Large Generative Models,” IEEE Journal on Selected Areas in Communications, vol. 44, pp. 4397–4413, 2026.

[14] A. Al-Kababji, F. Bensaali, and S. P. Dakua, “Scheduling Techniques for Liver Segmentation: ReduceLRonPlateau vs OneCycleLR,” in International conference on intelligent systems and pattern recognition. Springer, 2022, pp. 204–212.