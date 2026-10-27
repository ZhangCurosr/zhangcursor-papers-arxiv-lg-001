# IsoAct: Geometry-Aware Post-hoc Debiasing via Isometric Actions

Sumin Park<sup>1</sup>, Taero Kim<sup>2,\*</sup>, Minhyeong Cho<sup>2</sup>,

Subeen Park<sup>3</sup>, Jong Hwan Lee<sup>4</sup>, Kyungwoo Song<sup>2,\*</sup>

<sup>1</sup>University of Wisconsin–Madison, <sup>2</sup>Yonsei University, <sup>3</sup>SK Shieldus, <sup>4</sup>Arist Inc.

park749@wisc.edu, taero.kim@yonsei.ac.kr, mhcho0226@yonsei.ac.kr

sbpark22@skshieldus.com, jonghwan.lee@arist.io, kyungwoo.song@yonsei.ac.kr

## Abstract

Learning representations that are robust to nuisance factors is essential for reliable deployment, but remains challenging when the representations lie on non-Euclidean manifolds. Existing debiasing methods typically assume Euclidean representations or apply linear interventions, which can distort the geometry of structured latent spaces. We propose IsoAct, a post-hoc framework that accounts for geometry in both nuisance identification and representation editing. It identifies nuisance-predictive directions using Fisher discriminant analysis (FDA) in tangent coordinates and applies corrections through manifold-native actions that preserve manifold membership. Controlled and real-world experiments on hyperspherical, hyperbolic, and SE(3) representations evaluate nuisance reduction, utility preservation, and manifold validity jointly. IsoAct reduces nuisance recoverability while retaining competitive utility. Its clearest joint result is on KLIFS, combining reduced kinase-group recoverability, strong structural prediction, and valid rigid poses. Code: https://github.com/MLAI-Yonsei/IsoAct

## 1 Introduction

Representations robust to nuisance factors are important for reliable deployment, fairness, and domain robustness. Post-hoc debiasing and concept erasure methods address this problem by editing pretrained representations after feature extraction, avoiding the cost of retraining the pretrained encoder [1, 2, 3, 4]. However, this setting becomes more challenging when representations lie on non-Euclidean manifolds rather than unconstrained Euclidean spaces. CLIP-style vision-language embeddings are commonly normalized onto hyperspheres [5], hyperbolic image-text models such as MERU place embeddings on Lorentz hyperbolic manifolds [6], and kinase-ligand structural datasets such as KLIFS contain ligand poses that can be represented as rigid motions in SE(3) [7]. For such representations, debiasing requires reducing nuisance information while preserving downstream utility and manifold validity, measured through residuals of the defining constraints (Table 1).

Existing post-hoc methods typically realize debiasing through Euclidean editing operations, including projection-based subspace removal [1, 2], linear concept erasure [3, 4], feature intervention [8], and ambient-space edits [9, 10]. These approaches are simple and effective baselines, and several preserve selected constraints in specific settings. A key limitation for manifold-valued representations is that the editing operation is not generally defined as a native transformation of the target manifold. As a result, an edit that is valid in Euclidean feature space does not generally preserve unit norm on the hypersphere, Lorentz validity in hyperbolic space, or the orthogonality and determinant constraints of a rotation matrix of SE(3). Manifold-aware and geometry-inspired debiasing methods have therefore been proposed [11, 12, 13]. Still, they are often tied to a particular geometry, generative setting, representation space, or training-time objective, leaving open a general post-hoc approach that respects geometry both when identifying nuisance structure and carrying out the resulting edit.

Geometry matters at both stages of post-hoc debiasing. Local motion is described by tangent spaces, whereas an arbitrary ambient displacement need not respect the manifold’s constraints [14]. This motivates identifying nuisance structure in coordinates that reflect local geometry and realizing the correction through a valid motion on the manifold.

Building on this principle, we propose IsoAct, a post-hoc framework for editing manifold-valued representations. IsoAct uses Fisher discriminant analysis (FDA) in geometry-specific tangent coordinates to identify nuisance-predictive directions. A neutral target constructed from low-confidence training examples supplies the correction reference. The resulting sample-dependent actions preserve manifold validity: angular updates on the hypersphere, angular and translation actions on the Lorentz hyperboloid, and rotation, translation, or joint updates on SE(3).

We evaluate IsoAct on controlled geometric benchmarks and real-world representations. Alongside KLIFS kinase-ligand poses, we examine protected attributes on CLIP and MERU embeddings of FairFace [15] and use Flickr30K text-to-image retrieval [16] to assess their utility. The clearest joint gains occur on KLIFS, where IsoAct combines reduced kinase-group recoverability, strong structural utility, and valid rigid poses. The experiments support evaluating nuisance reduction, utility, and manifold validity together, rather than relying on any one metric alone.

## 2 Related Work

Post-hoc debiasing and concept erasure. Post-hoc debiasing and concept erasure have been extensively studied as practical approaches to removing protected or nuisance information from pretrained representations [1, 17]. INLP [2] removes linearly decodable protected-attribute information by iteratively training linear probes and projecting representations onto the intersection of their nullspaces. RLACE [3] and LEACE [4] further formulate linear concept erasure through a constrained adversarial game and a closed-form oblique projection, respectively. SFID [8] mitigates bias through selective feature imputation over bias-relevant embedding coordinates, whereas SPD [9] models bias as a linear subspace and removes it via subspace projection with neutral mean reinjection. These methods operate in Euclidean feature spaces. IsoAct instead uses FDA in tangent coordinates and applies corrections through actions native to the target manifold.

Geometry-aware debiasing for manifold-valued representations. Euclidean projection can cause information loss by erasing valid associations [18] and may leave residual bias in the embedding geometry beyond the targeted bias direction [19], motivating geometry-aware debiasing methods. PGD [11] mitigates gender bias in Poincaré word embeddings through per-word Riemannian optimization, while FAIRT2V [12] uses anchor-based geodesic transformations on the hypersphere for training-free text-to-video debiasing. PGD and FAIRT2V specialize to their respective geometries and applications. More closely related, WRING [10] replaces projection with rotations in VLM embedding space. IsoAct instead couples tangent-space nuisance identification with native actions in a common framework for hyperspherical, hyperbolic, and SE(3) representations.

Geometric fairness and training-time invariance. Fairness and debiasing have also been studied from the perspectives of representation geometry and invariance [20]. CMF [13] enforces preservation of local Riemannian geometry across factual and counterfactual representations induced by sensitiveattribute interventions, through training-time regularization of the decoder Jacobian and Hessian. In contrast to CMF, IsoAct operates post hoc on fixed pretrained representations, complementing settings where retraining is costly or model internals are inaccessible.

## 3 Preliminaries

## 3.1 Problem Setting

We study post-hoc debiasing of pretrained representations on a known manifold M. Let $\mathcal { D } =$ $\{ ( x _ { i } , s _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ contain N samples, where $x _ { i } \in { \mathcal { M } }$ is a representation, $s _ { i }$ is a categorical nuisance label, and $y _ { i }$ denotes task-relevant information. The nuisance label may encode a protected attribute, such as gender or race, or another undesired attribute. We seek a transformation $T : { \mathcal { M } } \to { \mathcal { M } }$ that makes $s _ { i }$ less linearly predictable from $T ( x _ { i } )$ while preserving utility associated with $y _ { i }$ . The edit is applied after feature extraction, without retraining the pretrained encoder, and must retain membership in $\mathcal { M }$

Table 1: Target manifolds and the membership constraints that admissible edits must preserve. The Lorentz model has curvature $- c < 0 .$ , with $\begin{array} { r } { \langle x , x \rangle _ { L } = - x _ { 0 } ^ { 2 } + \sum _ { j = 1 } ^ { d } x _ { j } ^ { 2 } } \end{array}$ and time coordinate $x _ { 0 }$
<table><tr><td>Manifold</td><td>Representation</td><td>Constraints</td></tr><tr><td>Hypersphere</td><td> $x \in \mathbb { S } ^ { d - 1 }$ </td><td> $1 ) \| x \| _ { 2 } = 1$ </td></tr><tr><td>Lorentz hyperboloid</td><td> $\boldsymbol { x } \in \mathbb { H } _ { c } ^ { d }$ </td><td> $1 ) \ \langle x , x \rangle _ { L } = - 1 / c 2 ) \ x _ { 0 } > 0$ </td></tr><tr><td>SE(3)</td><td> $x = ( R , t ) \in \mathrm { S E } ( 3 )$ </td><td>1) R† R = I3  $2 ) \operatorname* { d e t } ( R ) = 1$ </td></tr></table>

## 3.2 Target Manifolds

Table 1 lists the three geometries and their membership constraints. The hypersphere $\mathbb { S } ^ { d - 1 } \subset \mathbb { R } ^ { d }$ contains unit vectors, with d ambient coordinates and $d - 1$ degrees of freedom, as used for normalized vision-language embeddings [5]. A Lorentz-hyperbolic representation $x = ( x _ { 0 } , v ) \in \mathbb { H } _ { c } ^ { d }$ has d spatial coordinates $v \in \mathbb { R } ^ { d }$ and a positive geometric time coordinate $x _ { 0 } = \sqrt { c ^ { - 1 } + \| v \| _ { 2 } ^ { 2 } } .$ . This time coordinate is part of the representation, not a temporal variable. The model has curvature $- c , c > 0$ and is used by MERU [6]. A rigid pose $x = ( R , \bar { t ) } \in \mathrm { S E } ( 3 )$ consists of a rotation matrix $R \in \mathrm { S O ( 3 ) }$ and a translation $t \in \mathbb { R } ^ { 3 }$ . Its six degrees of freedom describe orientation and position, although storing R and t uses twelve entries [21, 22, 7].

## 3.3 Tangent Coordinates

At a reference point $r \in \mathcal { M }$ , the tangent space describes admissible infinitesimal motion on the manifold [14]. Geometry-specific coordinate maps express representations in a common linear space: $\xi _ { i } = \phi _ { r } ( \bar { x _ { i } } ) \bar { \in \mathbb { R } ^ { D } }$ , where D counts the coordinates used by the map and need not equal the intrinsic dimension. Such coordinates enable linear estimation while accounting for local geometric structure.

## 4 IsoAct: Geometry-Aware Identification and Manifold-Native Actions

IsoAct identifies nuisance-predictive directions, constructs a neutral target, and applies corrections that remain on the manifold (Figure 1). The first two steps use training data; the resulting directions and target guide edits without retraining the representation encoder.

## 4.1 Tangent-Space Nuisance Identification and Neutral Target

Reference and coordinates. We first construct a coordinate reference $r \in \mathcal { M }$ from training representations: a normalized mean on the sphere, a mean spatial vector with its Lorentz time coordinate, or a mean pose with its rotation projected onto $\mathrm { S O ( 3 ) }$ This reference anchors the linearization, giving training coordinates $\xi _ { i } = \dot { \phi _ { r } } ( \dot { x _ { i } } ) \in \mathbb { R } ^ { D }$ . The coordinate maps use the manifold’s logarithm, with transport or a basis change where required. Construction details are in Appendix B.3 through B.6.

Nuisance directions via FDA. Within these coordinates, $S _ { W }$ measures spread among examples with the same nuisance label, whereas $S _ { B }$ measures separation of their class means. FDA [23, 24] seeks directions along which this separation is large relative to within-class variation:

$$
\operatorname* { m a x } _ { u \in \mathbb { R } ^ { D } \setminus \{ 0 \} } \frac { u ^ { \top } S _ { B } u } { u ^ { \top } ( S _ { W } + \rho I _ { D } ) u } ,\tag{1}
$$

where $\rho > 0$ regularizes the within-class scatter and $I _ { D }$ is the D-dimensional identity. We retain K directions $u _ { 1 } , \ldots , u _ { K } \in \mathbb { R } ^ { D }$ from the resulting discriminant subspace to describe nuisance-predictive variation. Appendix B.2 gives the scatter and basis conventions.

![](images/673fa7ec0abd3968667c08a5578b835f05acde4db8d45e51a1fdc62cafb85965.jpg)  
Figure 1: Overview of IsoAct on the hypersphere. FDA identifies nuisance directions in tangent coordinates around a coordinate reference r constructed from training data. A neutral target n from low-confidence training examples supplies the correction reference. The right panel shows one update along $u _ { k }$ while preserving unit norm. Lorentz and SE(3) instances use the same pipeline with geometry-specific actions.

Neutral target. The neutral target $n \in \mathcal { M }$ supplies the reference state for correction along the learned directions. An auxiliary nuisance classifier fitted on the training representations identifies examples whose nuisance labels it predicts with low confidence. We aggregate these examples using the corresponding manifold construction to form n (Appendix B.3). Thus r anchors the analysis coordinates, while n supplies the nuisance-related values used for correction. An action uses these components to compute its correction; it does not move every representation to n itself.

## 4.2 Geometry-Specific Action Primitives

For each retained direction, IsoAct measures a nuisance-related discrepancy from n and realizes the correction through a valid motion on the target manifold. Below, x is the current representation and $x ^ { + }$ is the result of one action associated with $u _ { k }$ . The output becomes the input for the next direction; r, n, and the learned directions remain fixed. Nonnegative edit-strength hyperparameters control the correction magnitude and are selected by the validation rule in Section 4.3.

Hypersphere. Changing a component of a unit vector generally changes its norm. IsoAct instead adjusts the nuisance projection and reconstructs the remaining component to stay on the sphere. Let $x , \overset { \cdot } { n } \in \mathbb { S } ^ { d - 1 }$ and represent the learned tangent direction as a unit vector $u _ { k } \in \dot { \mathbb { R } } ^ { d }$ . For $| x ^ { \top } u _ { k } | < 1$ define the scalar projections of x and n onto $u _ { k }$ and the normalized orthogonal component:

$$
p : = x ^ { \top } u _ { k } , \qquad p _ { n } : = n ^ { \top } u _ { k } , \qquad h : = { \frac { x - p u _ { k } } { \sqrt { 1 - p ^ { 2 } } } } .
$$

With strength $\alpha \geq 0$ , the update is

$$
p ^ { + } = \mathrm { c l i p } ( ( 1 - \alpha ) p + \alpha p _ { n } , - 1 , 1 ) , \qquad x ^ { + } = p ^ { + } u _ { k } + \sqrt { 1 - ( p ^ { + } ) ^ { 2 } } h .\tag{2}
$$

At $\alpha = 0$ the input is unchanged; $\alpha = 1$ matches the neutral projection, and $\alpha > 1$ permits extrapolation. Clipping keeps the requested projection feasible. Since $u _ { k }$ and $h$ are orthonormal, reconstruction preserves unit norm while retaining the orthogonal direction $h$ . This is a sampledependent rotation in their span. A vanishing orthogonal component gives the identity action.

Lorentz hyperboloid. Here $\boldsymbol { x } ~ = ~ ( x _ { 0 } , v )$ couples spatial coordinates v to the geometric time coordinate $x _ { 0 }$ . Write $n = ( n _ { 0 } , n _ { \mathrm { s p } } )$ for the neutral target, and let $u _ { k } \in \mathbb { R } ^ { d }$ be the unit spatial action direction supplied by FDA in the common tangent frame. IsoAct offers two motions. An angular action adjusts $v ^ { \top } u _ { k }$ relative to $n _ { \mathrm { s p } } ^ { \top } u _ { k }$ at fixed spatial radius $\lVert \boldsymbol { v } \rVert _ { 2 }$ , using the spherical reconstruction above. Preserving this radius also preserves $x _ { 0 }$ and the Lorentz constraint.

A hyperbolic translation can change spatial radius, but must update $x _ { 0 }$ and the axis projection together. With $p = v ^ { \top } u _ { k }$ and strength $\alpha _ { \mathrm { t r } } \geq 0 .$ , it requests $q _ { \mathrm { t r } } = ( 1 - \alpha _ { \mathrm { t r } } ) p + \alpha _ { \mathrm { t r } } n _ { \mathrm { s p } } ^ { \top } u _ { k }$ through a

Lorentz boost:

$$
\left[ x _ { 0 } ^ { + } \right] = \left[ \cosh \tau \quad \sinh \tau \right] \left[ x _ { 0 } \right] , \qquad v ^ { + } = v + ( p ^ { + } - p ) u _ { k } .\tag{3}
$$

The boost parameter τ controls this coupled motion, preserving $- x _ { 0 } ^ { 2 } + p ^ { 2 }$ and the orthogonal spatial component. Appendix B.5 gives the target-to-τ calculation, the angular reconstruction domain, and numerical safeguards. The angular strength $\alpha _ { \mathrm { a n g } } \geq 0$ and translation strength are independent; when combined, the second primitive acts on the first’s output within each axis.

SE(3). For a pose $\boldsymbol { x } = ( R , t )$ , translation may move in $\mathbb { R } ^ { 3 }$ , whereas rotation must remain in $\mathrm { S O ( 3 ) }$ IsoAct therefore adds a translation increment to t and multiplies R by a valid rotation increment. The mapping in Appendix B.6 converts each FDA direction $u _ { k } \in \bar { \mathbb { R } } ^ { 6 }$ into a unit pose direction $\bar { u } _ { k } \in \mathbb { R } ^ { 1 2 }$ and native directions $a _ { k } , d _ { k } \in \mathbb { R } ^ { 3 }$ for rotation and translation. Nonzero native directions are normalized; a zero component disables that branch.

Let $\boldsymbol { n } = ( R _ { n } , t _ { n } )$ and let $\mathrm { p a c k } ( x ) \in \mathbb { R } ^ { 1 2 }$ concatenate the row-wise entries of R with t. Translation is driven by the full pose discrepancy along ${ \bar { u } } _ { k } \colon$

$$
\delta ^ { T } = \langle \mathrm { p a c k } ( n ) - \mathrm { p a c k } ( x ) , { \bar { u } } _ { k } \rangle , \qquad t ^ { + } = t + \alpha \delta ^ { T } d _ { k } .\tag{4}
$$

For rotation, $R ^ { \top } R _ { n }$ compares the current and neutral orientations. Its rotation logarithm, converted to an axis-angle vector by $\vee ,$ yields the correction component along $a _ { k } { \mathrm { i } }$

$$
\begin{array} { r } { \delta ^ { R } = \Big \langle \big ( \mathrm { L o g } _ { \mathrm { S O ( 3 ) } } ( R ^ { \top } R _ { n } ) \big ) ^ { \vee } , a _ { k } \Big \rangle , \qquad R ^ { + } = R \mathrm { E x p } _ { \mathrm { S O ( 3 ) } } \big ( \alpha \delta ^ { R } \widehat { a } _ { k } \big ) . } \end{array}\tag{5}
$$

The hat map forms a skew-symmetric matrix from a vector, and ∨ is its inverse. Exponentiating this matrix produces a proper rotation, so multiplication keeps $R ^ { + } \in \mathrm { S O } ( 3 )$ . Rotation-only or translation-only edits keep the other component fixed; a joint step computes both discrepancies from the same incoming pose and returns $( R ^ { + } , t ^ { + } )$

Theorem 4.1 (Pointwise manifold validity). In exact arithmetic, each well-defined IsoAct update above with finite action parameters maps a valid input to the same target manifold. Any finite sequence ofsuch updates preserves manifold membership, provided every step is defined.

Proof. The hyperspherical construction preserves unit norm; Lorentz spatial rotations and boosts preserve the quadratic constraint and upper sheet; and the $\operatorname { S E } ( 3 )$ updates preserve $R \in \mathrm { S O ( 3 ) }$ and $\bar { t } \in \mathbb { R } ^ { 3 }$ . Applying these properties at each step gives the result. A detailed derivation of the SE(3) case is provided in Appendix B.7. □

Scope of the guarantee. Theorem 4.1 guarantees membership of each edited point, not preservation of pairwise distances: correction magnitudes depend on the individual input.

## 4.3 Validation-Based Configuration Selection

For each real-world setting, we select one configuration per method using a common validation rule. Let P be the nuisance-probe metric set and $V _ { m } \bar { ( \theta ) }$ the validation value of metric m for candidate θ. Let $O _ { m }$ and $N _ { m }$ be the corresponding validation values for the unedited Original and a shuffled-label Null reference. Null provides the target level of nuisance predictability under label randomization. We summarize nuisance reduction by

$$
P ( \theta ) = \frac { 1 } { | \mathcal { P } | } \sum _ { m \in \mathcal { P } } \left( | O _ { m } - N _ { m } | - | V _ { m } ( \theta ) - N _ { m } | \right) ,
$$

so that larger values indicate closer agreement with Null relative to Original, including a penalty for overshooting Null. Let $U ( \theta )$ denote the aggregate validation utility. We select the configuration maximizing $S ( \mathbf { \bar { \theta } } ) = [ P ( \theta ) + \mathbf { \dot { \cal U } } ( \theta ) ] / 2$ and use it for test reporting. The experiment-specific metrics and utility aggregation are specified in Appendix B.8.

![](images/eb2c6ed8e29ad6d48fde0267404bdad4a8a6c72415d5a5c8c1dfe2d73d984bc5.jpg)  
Figure 2: Controlled debiasing on three manifolds. IsoAct combines Oracle-matched nuisance-probe scores with preservation of the designated utility proxy and near-zero constraint residuals. All axes improve outward; constraint scales are panel-specific. † marks the learning-based CMF adaptation.

## 5 Experiments

We evaluate post-hoc debiasing along three axes: nuisance reduction, measured by linear-probe Accuracy and macro-F1; utility preservation, measured on the designated tasks; and manifold validity, measured by geometry-specific constraint residuals. We compare IsoAct with feature-intervention SFID [8], subspace-projection SPD [9], and a CMF adaptation that learns an encoder-decoder with geometric regularization [13].

The controlled benchmarks examine editing behavior under known nuisance and utility coordinates. We then evaluate real-world representations, beginning with KLIFS ligand poses and followed by CLIP and MERU embeddings. In the real-world benchmarks, transformations and evaluation predictors are fitted on training data, configurations are selected by the common validation rule in Section 4.3, and selected results are reported on test data. Predictors are refitted to each transformed training set; retrieval instead uses fixed text embeddings.

## 5.1 Controlled synthetic benchmarks

## 5.1.1 Setup

We construct mechanism checks on the hypersphere $\mathbb { S } ^ { 2 }$ , the Lorentz hyperboloid $\mathbb { H } ^ { 2 }$ with curvature −1, and SE(3), with 400 samples split equally between two nuisance groups in each setting. These synthetic data have no semantic downstream task. Instead, a separate geometric variable, generated over the same range for both nuisance groups, serves as a utility proxy. Nuisance and utility occupy separate angular coordinates on the sphere; on Lorentz they occupy angular and radial coordinates. On SE(3), nuisance is injected into the rotation angle $\varphi ,$ while translation magnitude ∥t∥ is the shared utility proxy. The goal is to remove group separation without disrupting this shared structure. Original and the known neutralized Oracle provide evaluation references, not competing methods.

Figure 2 reports normalized proximity of linear-probe Accuracy and macro-F1 to the Oracle, with Original at zero and Oracle at one. Probes use separate training and evaluation subsets. Utility measures the fraction satisfying a preservation criterion: a small polar-angle change on the sphere, or remaining within the original group-shared radial or translation-magnitude range on Lorentz and SE(3). Constraint preservation linearly rescales and reverses the mean residual, so smaller violations appear farther outward. Exact criteria, evaluation splits, panel scales, and additional results are in Appendix C.1.

## 5.1.2 Results

The controlled experiments make the central trade-off visible: strong nuisance reduction alone does not imply a successful manifold-valued edit. On the sphere, all three comparison methods nearly match the Oracle probe scores, yet at most 30.5% of their outputs satisfy the polar-angle preservation criterion, compared with 100% for IsoAct. On Lorentz, SFID and SPD preserve the task band and reduce nuisance predictability but substantially violate the hyperboloid constraint. On SE(3), SPD matches the Oracle probe scores and nearly preserves the task band while violating rotation constraints; SFID has smaller residuals but loses more utility. Across these constructions, IsoAct matches both Oracle probe metrics, satisfies the specified utility condition for all samples, and keeps the measured residuals at numerical precision.

![](images/07dedb5f1545496e07f7f7b39e660a6221becafeb81ec1dffc91a59a0e5495cc.jpg)  
Figure 3: SE(3) mechanism check. Colors denote nuisance groups; the dashed line marks the neutral angle. The light-green band is the shared translation-magnitude range used as a utility proxy. IsoAct removes angular group separation while keeping all samples in this band, matching the Oracle pattern. In contrast, CMF and SFID move many samples outside the band. Rotation residuals and detailed results are given in Figure 2 and Appendix C.1.

Figure 3 provides a complementary point-level view: IsoAct collapses the injected group separation along the nuisance rotation coordinate while retaining the task-relevant translation band. The coordinate view and starplots together show why both task preservation and manifold residuals must accompany nuisance-probe evaluation. These are mechanism checks under controlled geometric structure, motivating the real-world tests that follow.

## 5.2 Real-world SE(3) ligand-pose experiment on KLIFS

Setup. We first evaluate IsoAct on KLIFS kinase-ligand structural data [7]. Each example is a ligand pose $x = ( R , t ) \in \mathrm { S E } ( 3 )$ , with orientation R and position t in the binding site [25, 26]. The nuisance label is one of nine kinase groups, a coarse grouping related to kinase-family identity [27]. We use it as a nuisance proxy to discourage reliance on family-specific cues while retaining structural information about conformation and ligand placement. Kinase IDs are disjoint across the training, validation, and test splits.

Utility comprises αC-helix state classification and pocket-occupancy classification [7]. Linear probes measure kinase-group recoverability; separate classifiers fitted on the edited representations evaluate the structural tasks. We report nuisance Accuracy and macro-F1, utility Recall and AUROC, and mean rotation orthogonality and determinant residuals. Recall targets the training-set minority class; Pocket metrics average the front, gate, and back tasks. Original denotes unedited poses, and Null supplies the shuffled-label baseline targeted by nuisance reduction. Evaluation details and selected settings are given in Appendix C.2.

Results. The common validation rule balances nuisance reduction with utility, rather than selecting for nuisance reduction alone. At the selected operating points in Table 2, IsoAct is the only compared

Table 2: KLIFS results. IsoAct combines reduced kinase-group predictability with strong structural utility and numerical-precision rotation residuals. Constraints 1 and 2 report mean rotation orthogonality and determinant errors, respectively. Bold/underline mark best/second-best values among debiasing methods; Original and Null are reference rows excluded from ranking.
<table><tr><td rowspan="3">Method</td><td colspan="2">Debiasing Capability</td><td colspan="4">Utility Preservation</td><td colspan="2">Manifold Constraint Violation</td></tr><tr><td colspan="2">Kinase Group</td><td colspan="2">αC-helix State</td><td colspan="2">Pocket Occupancy</td><td colspan="2">Constraint 1 Constraint 2</td></tr><tr><td>Acc↓</td><td>F1↓</td><td>Recall↑</td><td>AUROC↑</td><td>Recall↑</td><td>AUROC↑</td><td>Mean↓</td><td>Mean↓</td></tr><tr><td rowspan="4">Original Null KLIES CMF SFID</td><td>0.1398</td><td>0.1409</td><td>0.3362</td><td>0.7163</td><td>0.7357</td><td>0.9708</td><td>6.5065e-08</td><td>2.2077e-08</td></tr><tr><td>0.0827</td><td>0.0679</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>0.1988</td><td>0.1858</td><td>0.3046</td><td>0.7087</td><td>0.8173</td><td>0.9732</td><td>3.6900e-01</td><td>1.1368e-01</td></tr><tr><td>0.1398</td><td>0.1409</td><td>0.3247</td><td>0.7176</td><td>0.7388</td><td>0.9712</td><td>6.5065e-08</td><td>2.2077e-08</td></tr><tr><td>SPD</td><td>0.1425</td><td>0.1415</td><td>0.2414</td><td>0.7061</td><td>0.7797</td><td>0.9729</td><td>9.4948e-01</td><td>3.7450e-01</td></tr><tr><td>IsoAct</td><td>0.1117</td><td>0.1022</td><td>0.3822</td><td>0.7333</td><td>0.8174</td><td>0.9664</td><td>6.9510e-16</td><td>3.5723e-16</td></tr></table>

![](images/44cb6b66b7573e3a75bc1bef2f74ee3292b2d99b20b452e13e7281457b5a96ac.jpg)

Figure 4: KLIFS debiasing-utility trade-offs. Each point is one configuration’s test result; higher and farther right is better. Debiasing gain D measures progress from Original (0) toward the shuffled-label Null (1). Mean utility retention quantifies utility preservation by averaging Recall and AUROC ratios to Original across both structural tasks; 1 denotes unchanged aggregate utility. Outlined diamonds mark the IsoAct Pareto frontier. Its high-utility point combines $D \approx 0 . 5 1$ with mean retention of 1.07, while other frontier points achieve stronger debiasing at lower retention. Definitions and configuration pools are detailed in Appendix C.3.

method to reduce both kinase-group probe metrics relative to Original, while improving three of the four reported structural-utility metrics. SFID leaves nuisance recoverability essentially unchanged, whereas CMF and SPD incur appreciable rotation-constraint violations without reducing either nuisance metric. IsoAct combines these gains with numerical-precision rotation residuals, consistent with its manifold-native action design. The small decrease in pocket AUROC is its remaining utility trade-off. Together, these results highlight the value of reducing nuisance predictability through geometry-valid actions while retaining useful structural information.

Figure 4 extends the single-configuration comparison in Table 2. For each nuisance metric $m \in \mathcal { P } =$ {Acc, macro-F1}, let $M _ { m } , O _ { m }$ , and $N _ { m }$ be the configuration, Original, and Null scores. Debiasing gain is $\begin{array} { r } { D = 1 - \frac { 1 } { 2 } \sum _ { m \in \mathcal { P } } | M _ { m } - N _ { m } | / | O _ { m } - N _ { m } | } \end{array}$ |. The method-specific pools cover overlapping edit-control ranges where applicable. IsoAct’s frontier extends to approximately $D = 0 . 7 0$ , showing additional operating points beyond the table’s selected setting and making the utility cost of stronger debiasing explicit.

## 5.3 Real-world vision-language representations: CLIP and MERU

Setup. We next evaluate the framework on unit-normalized CLIP ViT-B/32 image embeddings on the hypersphere [5] and MERU ViT-B/16 image embeddings on a Lorentz hyperboloid [6]. Each FairFace [15] attribute, age, gender, or race, defines a separate experiment with fixed subsets of 3,500 training, 1,500 validation, and 1,500 test examples, identical across the two models. Linear probes measure protected-attribute recoverability. The FairFace-fitted transformation is also applied to Flickr30K image embeddings to evaluate text-to-image retrieval [16], with text embeddings fixed. Test retrieval uses 5,000 text queries and a gallery of 1,000 images.

In Table 3, Debias averages the normalized proximity of Accuracy and macro-F1 to the labelpermutation Null reference using the Original-to-Null gap, so Original is zero and Null is one. Negative scores indicate greater average normalized distance from Null than Original. Utility averages Original-normalized R@1, R@5, and R@10, where R@k measures how often a text query retrieves its target image among the top k results. Protocols, normalization formulas, and selected settings are in Appendix C.4. Constraint residuals are measured on the FairFace test outputs, separately from retrieval: unit-norm error for CLIP and the mean of Lorentz quadratic and upper-sheet residuals for MERU (Appendix C.5).

Table 3: CLIP and MERU results across protected attributes. IsoAct combines high retrieval retention with the strongest MERU debiasing and attribute-dependent CLIP gains. Debias averages normalized Accuracy/macro-F1 proximity to Null; Utility averages Original-normalized R@1/5/10. Constraint reports CLIP unit-norm or aggregated MERU Lorentz residuals. Bold/underline mark best/secondbest method values, excluding reference rows and numerical-precision MERU residuals.
<table><tr><td></td><td></td><td colspan="3">Age</td><td colspan="3">Gender</td><td colspan="3">Race</td></tr><tr><td>Model</td><td>Method</td><td>Debias↑</td><td>Utility↑</td><td>Constraint↓</td><td>Debias↑</td><td>Utility↑</td><td>Constraint↓</td><td>Debias↑</td><td>Utility↑</td><td>Constraint↓</td></tr><tr><td rowspan="6">CLIP</td><td>Original</td><td>0.0000</td><td>1.0000</td><td>2.2612e-08</td><td>0.0000</td><td>1.0000</td><td>2.2935e-08</td><td>0.0000</td><td>1.0000</td><td>2.2181e-08</td></tr><tr><td>Null</td><td>1.0000</td><td></td><td></td><td>1.0000</td><td></td><td></td><td>1.0000</td><td></td><td></td></tr><tr><td>CMF</td><td>0.6339</td><td>0.0096</td><td>8.8207e-02</td><td>0.4481</td><td>0.0130</td><td>7.0113e-02</td><td>0.6890</td><td>0.0115</td><td>1.1123e-01</td></tr><tr><td>SFID</td><td>0.0117</td><td>1.0005</td><td>3.8070e-03</td><td>0.0389</td><td>0.9788</td><td>8.1287e-03</td><td>-0.0085</td><td>0.9975</td><td>8.3426e-03</td></tr><tr><td>SPD</td><td>0.7536</td><td>0.9867</td><td>1.8327e-02</td><td>0.8243</td><td>0.9938</td><td>1.0768e-02</td><td>0.9656</td><td>0.9891</td><td>5.3183e-02</td></tr><tr><td>IsoAct</td><td>0.9234</td><td>0.9890</td><td>6.3230e-08</td><td>0.4254</td><td>1.0003</td><td>3.5303e-08</td><td>0.8022</td><td>0.9932</td><td>6.1190e-08</td></tr><tr><td rowspan="6">MERU</td><td>Original</td><td>0.0000</td><td>1.0000</td><td>3.9909e-16</td><td>0.0000</td><td>1.0000</td><td>4.0619e-16</td><td>0.0000</td><td>1.0000</td><td>3.9435e-16</td></tr><tr><td>Null</td><td>1.0000</td><td></td><td></td><td>1.0000</td><td></td><td></td><td>1.0000</td><td></td><td></td></tr><tr><td>CMF</td><td>0.1039</td><td>0.2661</td><td>3.8310e-16</td><td>0.2495</td><td>0.2936</td><td>3.9139e-16</td><td>0.8001</td><td>0.0204</td><td>4.0323e-16</td></tr><tr><td>SFID</td><td>0.0759</td><td>0.9822</td><td>3.9257e-16</td><td>0.0001</td><td>0.9944</td><td>3.8902e-16</td><td>0.0466</td><td>0.9623</td><td>3.8192e-16</td></tr><tr><td>SPD</td><td>0.9108</td><td>0.9672</td><td>3.9198e-16</td><td>0.5836</td><td>0.9869</td><td>3.8192e-16</td><td>0.8547</td><td>0.9321</td><td>3.9494e-16</td></tr><tr><td>IsoAct</td><td>0.9392</td><td>0.9502</td><td>3.8902e-16</td><td>0.8891</td><td>0.9692</td><td>3.8192e-16</td><td>0.9825</td><td>0.9449</td><td>3.9257e-16</td></tr></table>

Results. On CLIP, IsoAct retains at least 98.9% mean retrieval utility across all attributes while maintaining unit norm to numerical precision. It achieves the strongest age debiasing and the secondhighest race debiasing. SPD yields stronger gender and race debiasing, but with mean unit-norm errors of order 10<sup>−2</sup>. IsoAct thus offers nuisance reduction with minimal retrieval loss and valid hyperspherical outputs, a favorable combination not captured by ranking debiasing scores alone.

On MERU, IsoAct achieves the highest debiasing score for all three attributes while retaining approximately 94.5% to 96.9% of Original retrieval utility. Relative to SPD, stronger age and gender debiasing comes with lower retrieval retention, while race improves on both scores. SFID retains more retrieval utility but achieves little debiasing. Because MERU’s time coordinate is reconstructed from its stored spatial vector to satisfy the hyperboloid constraint, all methods have numerical-precision residuals; nuisance reduction and retrieval provide the informative comparison here.

## 5.4 Empirical summary and scope

The experiments support evaluating manifold-valued debiasing jointly through nuisance predictability, utility, and validity. The controlled checks illustrate the editing mechanism. KLIFS provides the clearest real-world joint result on a structured space of rotations and translations, combining reduced nuisance predictability with strong structural utility and valid poses. CLIP and MERU extend the evaluation to pretrained embedding spaces, with representation- and attribute-dependent tradeoffs. Together, these results support the geometry-aware identification and manifold-native action framework across the three studied geometries. Reduced linear recoverability and small constraint residuals do not establish complete information removal, guaranteed fairness, or preservation of all pairwise distances.

## 6 Conclusion

We introduced IsoAct, a post-hoc debiasing framework that identifies nuisance-predictive directions using FDA in geometry-appropriate tangent coordinates and applies corrections through manifoldnative actions that preserve manifold membership. Controlled experiments expose cases where strong nuisance reduction coincides with lost task utility or invalid representations. On KLIFS, IsoAct combines reduced linear kinase-group recoverability with strong structural utility and valid SE(3) poses. CLIP and MERU further demonstrate its applicability to pretrained embeddings, with high retrieval retention and debiasing trade-offs that vary by model and protected attribute. Together, these findings support treating nuisance reduction, utility preservation, and manifold validity as joint requirements for post-hoc representation editing. Future work can extend this framework to learned or approximate manifolds and richer nuisance models.

## References

[1] Tolga Bolukbasi, Kai-Wei Chang, James Y Zou, Venkatesh Saligrama, and Adam T Kalai. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. Advances in neural information processing systems, 29, 2016.

[2] Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. Null it out: Guarding protected attributes by iterative nullspace projection. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 7237–7256, 2020.

[3] Shauli Ravfogel, Michael Twiton, Yoav Goldberg, and Ryan D Cotterell. Linear adversarial concept erasure. In International Conference on Machine Learning, pages 18400–18421. PMLR, 2022.

[4] Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, and Stella Biderman. Leace: Perfect linear concept erasure in closed form. Advances in Neural Information Processing Systems, 36:66044–66063, 2023.

[5] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[6] Karan Desai, Maximilian Nickel, Tanmay Rajpurohit, Justin Johnson, and Shanmukha Ramakrishna Vedantam. Hyperbolic image-text representations. In International Conference on Machine Learning, pages 7694–7731. PMLR, 2023.

[7] Albert J Kooistra, Georgi K Kanev, Oscar PJ van Linden, Rob Leurs, Iwan JP de Esch, and Chris de Graaf. Klifs: a structural kinase-ligand interaction database. Nucleic acids research, 44(D1):D365–D371, 2016.

[8] Hoin Jung, Taeuk Jang, and Xiaoqian Wang. A unified debiasing approach for vision-language models across modalities and tasks. Advances in Neural Information Processing Systems, 37:21034–21058, 2024.

[9] Dachuan Zhao, Weiyue Li, Zhenda Shen, Yushu Qiu, Bowen Xu, Haoyu Chen, and Yongchao Chen. Bias is a subspace, not a coordinate: A geometric rethinking of post-hoc debiasing in vision-language models. arXiv preprint arXiv:2511.18123, 2025.

[10] Walter Gerych, Cassandra Parent, Quinn Perian, Rafiya Javed, Justin Solomon, and Marzyeh Ghassemi. Wring out the bias: A rotation-based alternative to projection debiasing. In The Fourteenth International Conference on Learning Representations, 2026.

[11] Vaibhav Kumar, Tenzin Singhay Bhotia, and Tanmoy Chakraborty. Identifying and mitigating gender bias in hyperbolic word embeddings. arXiv preprint arXiv:2109.13767, 2021.

[12] Haonan Zhong, Wei Song, Tingxu Han, Maurice Pagnucco, Jingling Xue, and Yang Song. Fairt2v: Training-free debiasing for text-to-video diffusion models. arXiv preprint arXiv:2601.20791, 2026.

[13] Vidhi Rathore. Causal manifold fairness: Enforcing geometric invariance in representation learning. arXiv preprint arXiv:2601.03032, 2026.

[14] P-A Absil, Robert Mahony, and Rodolphe Sepulchre. Optimization algorithms on matrix manifolds. Princeton University Press, 2008.

[15] Kimmo Kärkkäinen and Jungseock Joo. Fairface: Face attribute dataset for balanced race, gender, and age. arXiv preprint arXiv:1908.04913, 2019.

[16] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings ofthe IEEE international conference on computer vision, pages 2641–2649, 2015.

[17] Nicholas Meade, Elinor Poole-Dayan, and Siva Reddy. An empirical survey of the effectiveness of debiasing techniques for pre-trained language models. In Proceedings ofthe 60th annual meeting ofthe association for computational linguistics (volume 1: long papers), pages 1878–1898, 2022.

[18] Sunipa Dev, Tao Li, Jeff M Phillips, and Vivek Srikumar. Oscar: Orthogonal subspace correction and rectification of biases in word embeddings. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 5034–5050, 2021.

[19] Hila Gonen and Yoav Goldberg. Lipstick on a pig: Debiasing methods cover up systematic gender biases in word embeddings but do not remove them. In Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 609–614, 2019.

[20] Yuzi He, Keith Burghardt, and Kristina Lerman. A geometric solution to fair representations. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, pages 279–285, 2020.

[21] Richard M Murray, Zexiang Li, and S Shankar Sastry. A mathematical introduction to robotic manipulation. CRC press, 2017.

[22] Richard Hartley, Jochen Trumpf, Yuchao Dai, and Hongdong Li. Rotation averaging. International journal of computer vision, 103(3):267–305, 2013.

[23] Ronald A Fisher. The use of multiple measurements in taxonomic problems. Annals ofeugenics, 7(2):179– 188, 1936.

[24] C Radhakrishna Rao. The utilization of multiple measurements in problems of biological classification. Journal ofthe Royal Statistical Society. Series B (Methodological), 10(2):159–203, 1948.

[25] Chao Shen, Xueping Hu, Junbo Gao, Xujun Zhang, Haiyang Zhong, Zhe Wang, Lei Xu, Yu Kang, Dongsheng Cao, and Tingjun Hou. The impact of cross-docked poses on performance of machine learning classifier for protein–ligand binding pose prediction. Journal ofcheminformatics, 13(1):81, 2021.

[26] Yangtian Zhang, Huiyu Cai, Chence Shi, Bozitao Zhong, and Jian Tang. E3bind: An end-to-end equivarian network for protein-ligand docking. arXiv preprint arXiv:2210.06069, 2022.

[27] Vivek Modi and Roland L Dunbrack Jr. A structurally-validated multiple sequence alignment of 497 human protein kinase domains. Scientific reports, 9(1):19790, 2019.

## A Limitations

IsoAct assumes a known target manifold and geometry-specific coordinate maps and action primitives. Extending the framework to learned or approximate geometries requires corresponding constructions. FDA identifies class-separating linear structure in tangent coordinates; nuisance information may remain accessible to nonlinear predictors or under distribution shift. The neutral target provides an editing reference rather than a guarantee of statistical independence or fairness.

The validity guarantee concerns manifold membership. Sample-dependent actions need not preserve pairwise distances, and valid poses do not by themselves establish physical feasibility or binding affinity. Our experiments assess designated structural and retrieval tasks in controlled, KLIFS, and vision-language settings; the observed trade-offs depend on the representation and nuisance attribute.

## B Additional Method Details

## B.1 Procedure and notation

Let $\mathcal { T } _ { \mathrm { t r } }$ index the $N _ { \mathrm { t r } }$ training examples, with representations $x _ { i } \in { \mathcal { M } }$ and nuisance labels $s _ { i }$ . The training-derived reference r defines coordinates $\xi _ { i } = \phi _ { r } ( x _ { i } ) \in \mathbb { R } ^ { D }$ , and $n \in \mathcal { M }$ is the neutral target. FDA yields retained directions $u _ { 1 } , \ldots , u _ { K }$ . For $\mathrm { S E } ( 3 ) , u _ { k } \in \mathbb { R } ^ { 6 }$ is mapped to a packed-pose direction $\bar { u } _ { k } \in \mathbb { R } ^ { 1 2 }$ before constructing the native actions.

Each real-world candidate uses training data to construct $r ,$ estimate FDA directions, and form n from low-confidence examples. These quantities remain fixed while editing all splits. Directions act sequentially on the current representation; additional passes repeat the same direction sequence. Evaluation predictors are fitted on the transformed training data, and the validation rule in Section B.8 selects the configuration used for test reporting. Retrieval uses fixed text embeddings in place of a fitted utility predictor.

## B.2 FDA nuisance-direction estimation

Let C be the nuisance-label set, $C = | \mathcal { C } |$ , and $\mathcal { T } _ { c } = \{ i \in \mathcal { I } _ { \mathrm { t r } } : s _ { i } = c \}$ , with $N _ { c } = | \mathcal { T } _ { c } |$ . Define class means $\mu _ { c } = N _ { c } ^ { - 1 } \sum _ { i \in \mathbb { Z } _ { c } } \xi _ { i }$ and the overall mean $\mu = N _ { \mathrm { t r } } ^ { - 1 } \sum _ { i \in \mathcal { I } _ { \mathrm { t r } } } \xi _ { i }$ . The unnormalized scatter matrices are

$$
W _ { 0 } = \sum _ { c \in \mathcal { C } } \sum _ { i \in \mathcal { I } _ { c } } ( \xi _ { i } - \mu _ { c } ) { ( \xi _ { i } - \mu _ { c } ) } ^ { \top } ,\tag{6}
$$

$$
B _ { 0 } = \sum _ { c \in \mathcal { C } } N _ { c } ( \mu _ { c } - \mu ) ( \mu _ { c } - \mu ) ^ { \top } .\tag{7}
$$

Their normalized forms define the regularized Fisher criterion in Eq. (1).

For CLIP and MERU, $S _ { W } = W _ { 0 } / ( N _ { \mathrm { t r } } - C )$ and $S _ { B } = B _ { 0 } / ( C - 1 )$ . With $A _ { \rho } = S _ { W } + \rho I _ { D }$ and $\rho > 0$ , binary FDA solves $A _ { \rho } w = \mu _ { 1 } - \mu _ { 0 }$ for class labels 0, 1. Multiclass FDA diagonalizes $A _ { \rho } ^ { - 1 / 2 } S _ { B } A _ { \rho } ^ { - 1 / 2 }$ and maps the resulting eigenvectors back through $A _ { \rho } ^ { - 1 / 2 }$ . The ridge $\rho$ is on the scale of the normalized within-class covariance.

For each ridge value, singular value decomposition converts the discriminant basis fitted at the largest requested rank to an orthonormal row basis. The transposes of its first K rows define $u _ { 1 } , \ldots , u _ { K }$ with deterministic signs and numerical rank. The effective rank is bounded by $C - 1$ and the tangent dimension. Each nuisance class has at least two training examples.

For SE(3), the six-dimensional coordinates instead use $S _ { W } = W _ { 0 } / N _ { \mathrm { t r } }$ and $S _ { B } = B _ { 0 } / N _ { \mathrm { t r } }$ . We solve

$$
S _ { B } u = \lambda ( S _ { W } + 1 0 ^ { - 6 } I _ { 6 } ) u
$$

in descending eigenvalue order, with a deterministic basis within tied eigenspaces. Six directions are constructed before selecting the action prefix. Section B.6 specifies their lift to pose directions.

## B.3 Coordinate references and neutral targets

The reference r defines the linearization for FDA, whereas n supplies the nuisance state used to compute corrections. Both are obtained from training data through separate constructions.

For CLIP, r is the normalized arithmetic mean of the training unit vectors. For MERU, the reference retains the mean spatial vector and restores the positive time coordinate to satisfy the Lorentz constraint. For $\mathrm { S E } ( \bar { 3 } )$ ), use $\boldsymbol { r } = \left( \boldsymbol { R } _ { r } , t _ { r } \right)$ with

$$
R _ { r } = \mathrm { P r o j } _ { \mathrm { S O ( 3 ) } } \left( \frac { 1 } { N _ { \mathrm { t r } } } \sum _ { i \in \mathcal { I } _ { \mathrm { t r } } } R _ { i } \right) , \qquad t _ { r } = \frac { 1 } { N _ { \mathrm { t r } } } \sum _ { i \in \mathcal { I } _ { \mathrm { t r } } } t _ { i } .
$$

The determinant-corrected polar projection of $M = L \Sigma V ^ { \top }$ is $\begin{array} { r l } { { \cal L } \operatorname { d i a g } ( 1 , 1 , \operatorname* { d e t } ( { \cal L } V ^ { \top } ) ) V ^ { \top } } & { { } } \end{array}$

An auxiliary nuisance classifier fitted on training data supplies class probabilities $q _ { c } ( x )$ . Its lowconfidence subset is

$$
\mathcal { T } _ { \mathrm { l o w } } = \{ i \in \mathcal { T } _ { \mathrm { t r } } : \operatorname* { m a x } _ { c } q _ { c } ( x _ { i } ) < 0 . 7 \} .
$$

For CLIP, n is the normalized mean of this subset. For MERU, it is the mean spatial vector with the corresponding positive time coordinate. If fewer than ten examples qualify, these two constructions use the full-training mean. The $\operatorname { S E } ( 3 )$ target averages the low-confidence poses and applies the polar projection to the mean rotation.

The MERU and KLIFS auxiliary classifiers use standardized logistic regression with inverse regularization strength 1, at most 2,000 iterations, and unweighted classes. They are fitted to the original training representations to construct the neutral target. These auxiliary classifiers are distinct from the balanced nuisance probe used for KLIFS evaluation.

## B.4 Hyperspherical coordinates and actions

For unit vectors $x , r ,$ let $b = r ^ { \top }$ x and $\vartheta = \operatorname { a r c c o s } { b } .$ Away from the antipodal singularity,

$$
\operatorname { L o g } _ { r } ( x ) = { \frac { \vartheta } { \sin \vartheta } } ( x - b r ) ,
$$

with the continuous limit at $x \ = \ r$ . Inner products are clipped for numerical evaluation, and coordinates and learned directions are projected onto $T _ { r } \mathbb { S } ^ { d - 1 }$ before direction normalization and orthogonalization. CLIP represents these tangent vectors in 512 ambient coordinates, with tangent dimension 511.

For a current unit representation and unit action direction $u _ { k }$ , define the current projection $p = x ^ { \top } u _ { k }$ the neutral projection $p _ { n } = n ^ { \top } u _ { k }$ , and the orthogonal unit direction $h = ( x - p u _ { k } ) / \sqrt { 1 - p ^ { 2 } }$ when $| p | < 1$ . Equation (2) adjusts the projection and reconstructs the point along h. When clipping is inactive, $\alpha = 1$ matches $p _ { n }$ , while α $> 1$ extrapolates beyond it.

A vanishing orthogonal component gives the identity action. Numerical stabilization clips projections just inside the unit interval and renormalizes after an axis update only when unit-norm drift exceeds $\mathrm { \bar { 1 0 } ^ { - 5 } }$ . Additional passes repeat the fixed axis sequence.

## B.5 Lorentz-hyperbolic coordinates and actions

Use $\langle x , y \rangle _ { L } = - x _ { 0 } y _ { 0 } + x _ { \mathrm { s p } } ^ { \top } y _ { \mathrm { s p } }$ on $\mathbb { H } _ { c } ^ { d }$ , with curvature −c for $c > 0$ . A stored spatial vector $v \in \mathbb { R } ^ { d }$ represents $x = ( \sqrt { c ^ { - 1 } + \| v \| ^ { 2 } } , v )$ . Given r, let $\beta = - c \langle r , x \rangle _ { L }$ . The logarithmic tangent vector is

$$
\ell _ { r } ( x ) = { \frac { \operatorname { a r c o s h } \beta } { \sqrt { \beta ^ { 2 } - 1 } } } ( x - \beta r ) ,
$$

using the continuous limit at $x = r$ , numerical clipping, and tangent projection. To compare all samples in a common tangent frame, transport $\ell _ { r } ( x )$ to $\bar { o } = ( 1 / \sqrt { c } , 0 , \dots , \bar { 0 } )$ . For $\eta \in T _ { r } \mathbb { H } _ { c } ^ { \bar { d } }$

$$
\mathrm { P T } _ { r  o } ( \eta ) = \eta + \frac { \langle o , \eta \rangle _ { L } } { c ^ { - 1 } - \langle r , o \rangle _ { L } } ( r + o ) .
$$

The FDA coordinate is $\xi _ { i } = [ \mathrm { P T } _ { r  o } ( \ell _ { r } ( x _ { i } ) ) ] _ { \mathrm { s p } } $ , where $[ \cdot ] _ { \mathrm { s p } }$ retains the spatial components. A tangent vector at o has zero time component, yielding 512 FDA coordinates for MERU.

For a unit action direction $u _ { k } \in \mathbb { R } ^ { d }$ and neutral target $n = ( n _ { 0 } , n _ { \mathrm { s p } } )$ , set $p = { \boldsymbol { v } } ^ { \top } { \boldsymbol { u } } _ { k } , z = { \boldsymbol { v } } - p { \boldsymbol { u } } _ { k }$ $q = n _ { \mathrm { s p } } ^ { \top } u _ { k }$ , and $R _ { v } = \| v \|$ . When $\| z \| > 0$ , the angular action with strength $\alpha _ { \mathrm { a n g } } \geq 0$ is

$$
p _ { \mathrm { a n g } } = \mathrm { c l i p } ( ( 1 - \alpha _ { \mathrm { a n g } } ) p + \alpha _ { \mathrm { a n g } } q , - R _ { v } , R _ { v } ) , \qquad v ^ { + } = p _ { \mathrm { a n g } } u _ { k } + \sqrt { R _ { v } ^ { 2 } - p _ { \mathrm { a n g } } ^ { 2 } } \frac { z } { \| z \| } .
$$

It preserves spatial radius and the time coordinate.

For translation strength $\alpha _ { \mathrm { t r } } \geq 0 .$ , let $q _ { \mathrm { t r } } = ( 1 - \alpha _ { \mathrm { t r } } ) p + \alpha _ { \mathrm { t r } } q .$ A Lorentz boost gives $p ^ { + } = { }$ $x _ { 0 }$ sinh $\tau + p$ cosh τ. With $a _ { + } = x _ { 0 } + p , a _ { - } = x _ { 0 } - p$ , and $w = e ^ { \tau }$ , the equation for the requested projection is

$$
a _ { + } w ^ { 2 } - 2 q _ { \mathrm { t r } } w - a _ { - } = 0 , \qquad w = \frac { q _ { \mathrm { t r } } + \sqrt { q _ { \mathrm { t r } } ^ { 2 } + a _ { + } a _ { - } } } { a _ { + } } .
$$

Valid Lorentz inputs have $a _ { + } , a _ { - } > 0 ,$ , so the displayed root is the positive one and $\tau = \log { w }$ Numerical evaluation bounds rapidity to $[ - 8 , 8 ]$ and uses $\tau = 0$ if no finite positive solution is obtained. The boost in Eq. (3) updates time and the axis projection while leaving z unchanged.

The angular and translation strengths are independent. Their configured order is applied within each axis, with the second primitive receiving the first primitive’s output.

## B.6 SE(3) coordinates, direction lift, and actions

For $\boldsymbol { x } = ( R , t )$ , use $\mathrm { p a c k } ( x ) = \lceil \mathrm { v e c } _ { \mathrm { r o w } } ( R ) ; t \rceil \in \mathbb { R } ^ { 1 2 }$ with row-major vectorization and native translation units. Relative to $\boldsymbol { r } ~ = ~ ( R _ { r } , t _ { r } )$ , the right-body pose logarithm has six-component coordinates $\zeta _ { i } = ( \omega _ { i } , v _ { i } ) \in \mathbb { R } ^ { 6 }$ . Its rotation component is $\omega _ { i } = ( \mathrm { L o g } _ { \mathrm { S O ( 3 ) } } ( R _ { r } ^ { \top } R _ { i } ) ) ^ { \vee }$ , and its translation component is $v _ { i } = J _ { l } ( \omega _ { i } ) ^ { - 1 } R _ { r } ^ { \top } ( t _ { i } - t _ { r } )$

The left Jacobian $J _ { l }$ couples rotation and translation: a twist $( \omega , v )$ exponentiates to rotation $\mathrm { E x p } _ { \mathrm { S O ( 3 ) } } ( \widehat { \omega } )$ ) and translation $J _ { l } ( \omega ) v$ . With $\vartheta = \| \omega \|$

$$
J _ { l } ( \omega ) = I _ { 3 } + \frac { 1 - \cos \vartheta } { \vartheta ^ { 2 } } \widehat { \omega } + \frac { \vartheta - \sin \vartheta } { \vartheta ^ { 3 } } \widehat { \omega } ^ { 2 } .
$$

Near zero, J<sub>l</sub> and $J _ { l } ^ { - 1 }$ use the expansions $I _ { 3 } + \widehat { \omega } / 2 + \widehat { \omega } ^ { 2 } / 6$ and $I _ { 3 } - \widehat { \omega } / 2 + \widehat { \omega } ^ { 2 } / 1 2$ , respectively.

The matrix $J _ { r } \in \mathbb { R } ^ { 1 2 \times 6 }$ maps infinitesimal right-body perturbations at r to packed-pose coordinates. For standard basis vectors $e _ { i } \in \mathbb { R } ^ { 3 }$ , its rotation columns are $[ \mathrm { v e c } _ { \mathrm { r o w } } ( R _ { r } \hat { e } _ { j } ) ; 0 ]$ and its translation columns are $[ 0 ; R _ { r } e _ { j } ]$ . Let $\boldsymbol { B } _ { r } ^ { \check { } } \in \mathbb { R } ^ { 1 2 \times 6 }$ be a deterministic orthonormal basis of $\mathrm { r a n g e } ( J _ { r } )$ . The FDA coordinates and lifted unit directions are

$$
\xi _ { i } = B _ { r } ^ { \top } J _ { r } \zeta _ { i } , \qquad \bar { u } _ { k } = \frac { B _ { r } u _ { k } } { \left\| B _ { r } u _ { k } \right\| } , \qquad u _ { k } \in \mathbb { R } ^ { 6 } .
$$

Thus $\zeta _ { i }$ is the pose-logarithm coordinate and $\xi _ { i }$ is the coordinate used by FDA. The native rotation direction $a _ { k } \in \mathbb { R } ^ { 3 }$ is the normalized rotation component of $J _ { r } ^ { + } \bar { u } _ { k }$ , where $J _ { r } ^ { + }$ is the pseudoinverse. The translation direction $d _ { k } \in \mathbb { R } ^ { 3 }$ is the normalized translation block of $\bar { u } _ { k }$ . A zero corresponding component disables that branch.

The translation action uses the full packed-pose discrepancy in Eq. (4); the rotation action uses the relative rotation to $R _ { n }$ in Eq. (5). Rotation-only and translation-only actions keep the other component fixed. A joint action computes both corrections from the same incoming pose, while successive axis steps use the preceding output. Numerical rotation updates use the polar projection defined in Section B.3 for stabilization.

## B.7 Detailed proof of SE(3) validity

For the $\operatorname { S E } ( 3 )$ case of Theorem 4.1, fix an axis and a valid incoming pose $\boldsymbol { x } = ( R , t )$ . With the axis fixed, write the discrepancies in Eqs. (4) and (5) as functions $\delta ^ { T } ( x )$ and $\delta ^ { R } ( x )$ . Let $\dot { \eta _ { R } } ( x ) = \alpha \delta ^ { R } ( x )$ and $\eta _ { T } ( x ) = \alpha \delta ^ { T } ( x )$ be their finite correction magnitudes. With action directions $a , d \in \mathbb { R } ^ { 3 }$ , the three maps are

$$
\Phi _ { R } ( x ) = ( R Q ( x ) , t ) ,\tag{8}
$$

$$
\Phi _ { T } ( x ) = ( R , t + \eta _ { T } ( x ) d ) ,\tag{9}
$$

$$
\Phi _ { R T } ( x ) = ( R Q ( x ) , t + \eta _ { T } ( x ) d ) , \qquad Q ( x ) = \exp ( \eta _ { R } ( x ) { \widehat { a } } ) .\tag{10}
$$

Proof. For $\boldsymbol { a } = ( a _ { 1 } , a _ { 2 } , a _ { 3 } ) ^ { \top }$ , the matrix

$$
\widehat { a } = \left[ \begin{array} { c c c } { 0 } & { - a _ { 3 } } & { a _ { 2 } } \\ { a _ { 3 } } & { 0 } & { - a _ { 1 } } \\ { - a _ { 2 } } & { a _ { 1 } } & { 0 } \end{array} \right]
$$

is skew-symmetric. Set $A = \eta _ { R } ( x ) { \widehat { a } }$ . Then

$$
Q ( x ) ^ { \top } Q ( x ) = \exp ( A ^ { \top } ) \exp ( A ) = \exp ( - A ) \exp ( A ) = I , \qquad \operatorname* { d e t } Q ( x ) = \exp ( \operatorname { t r } A ) = 1 .
$$

Thus $Q ( x ) \in \mathrm { S O } ( 3 )$ . Since R ∈ SO(3),

$$
( R Q ) ^ { \mathsf { T } } ( R Q ) = Q ^ { \mathsf { T } } R ^ { \mathsf { T } } R Q = I , \qquad \operatorname* { d e t } ( R Q ) = \operatorname* { d e t } R \operatorname* { d e t } Q = 1 .
$$

Also $t + \eta _ { T } ( x ) d \in \mathbb { R } ^ { 3 }$ . Consequently, the rotation-only, translation-only, and joint maps all return an element of $\mathrm { S E } ( 3 )$ . Disabled branches are identities and satisfy the same property. Induction over any finite sequence of these maps proves the result, even when correction magnitudes are recomputed from the current pose at each step. □

## B.8 Validation-based configuration selection

For each real-world setting, let $\mathcal { P } = \{ \mathrm { A c c } , \mathrm { m a c r o } { - } \mathrm { F } 1 \}$ and let $V _ { m } ( \theta )$ denote candidate $\theta ^ { \prime } \mathbf { s }$ validation nuisance metric. The corresponding Original and shuffled-label Null values are $O _ { m }$ and $N _ { m }$ . Null supplies the target level of nuisance predictability under label randomization. The nuisance component is

$$
P ( \theta ) = \frac { 1 } { 2 } \sum _ { m \in \mathcal { P } } \big ( | O _ { m } - N _ { m } | - | V _ { m } ( \theta ) - N _ { m } | \big ) .
$$

For CLIP and MERU, $U ( \theta )$ is the arithmetic mean of validation R@1, R@5, and R@10. For KLIFS, it is the harmonic mean $H ( F 1 _ { \alpha C } , F 1 _ { \mathrm { P o c k e t } } )$ of the two tasks’ validation F1 scores, where $H ( a , b ) = 2 a b / ( a + b )$ and $H ( 0 , 0 ) = 0$ . We maximize $S ( \theta ) = [ P ( \theta ) + U ( \theta ) ] / 2$ among candidates with finite scores, breaking ties by a fixed deterministic ordering.

This rule balances nuisance reduction and utility within each method’s candidate pool. Sections C.2 and C.4 give the selected settings; Section C.3 describes the additional configuration sweep.

## C Experimental Details and Additional Results

## C.1 Controlled synthetic benchmarks

Construction and evaluation. Each benchmark has 400 examples, equally divided between two nuisance groups, with seed 42. Nuisance is injected into an angular coordinate; a separate geometric variable generated over the same range for both groups serves as a utility proxy. The proxy is polar angle on $\mathbb { S } ^ { 2 }$ , a radial coordinate on the curvature-−1 Lorentz hyperboloid $\mathbb { H } ^ { 2 }$ , and translation magnitude on $\operatorname { S E } ( 3 )$ . Original and the known neutralized Oracle provide the evaluation references.

A stratified split with seed 42 gives 280 probe-training and 120 probe-evaluation examples. Logisticregression nuisance probes use at most 1,000 iterations on the sphere and 500 with the lbfgs solver on Lorentz and $\mathrm { S E } ( { \bar { 3 } } )$ ; the $\operatorname { S E } ( 3 )$ probe has inverse regularization strength 0.1. Probe Accuracy and macro-F1 use the 120 evaluation examples, while utility and constraint residuals are measured over all 400 edited outputs.

Debiasing coordinate. Let $M _ { m } , O _ { m }$ , and $Q _ { m }$ denote nuisance metric m for an edited representation, Original, and Oracle, respectively. The starplot uses

$$
D _ { \mathrm { t o y } } = 1 - \frac { 1 } { 2 } \sum _ { m \in \{ \mathrm { A c c } , \mathrm { m a c r o } ^ { - } \mathrm { F 1 } \} } \frac { | M _ { m } - Q _ { m } | } { | O _ { m } - Q _ { m } | } .\tag{11}
$$

Each metric is normalized by its own Original-to-Oracle gap before averaging. Original maps to zero and Oracle to one; a score of one matches the Oracle’s two probe metrics.

Utility coordinate. The utility score U is the fraction of outputs satisfying the designated proxy-preservation criterion. On the sphere, let $\vartheta ( x )$ be the polar angle in degrees, computed as $( \mathrm { { 1 8 0 } } / \pi ) \operatorname { a r c c o s } ( \mathrm { { c l i p } } ( x _ { 3 } / \| x \| , - 1 , 1 ) )$ for nonzero x. For norms below $\mathrm { \overline { { 1 0 } } ^ { - 1 0 } }$ , the denominator is set to one. The criterion is $| \vartheta ( x _ { i } ^ { \mathrm { e d i t e d } } ) - \vartheta ( x _ { i } ^ { \mathrm { o r i g i n a l } } ) | \leq 2 ^ { \circ }$

For Lorentz outputs, define the extracted radial coordinate $\varrho ( x ) = \mathrm { a r c o s h } ( \mathrm { m a x } ( x _ { 0 } , 1 ) )$ and require $1 . 5 \leq \varrho ( x _ { i } ^ { \mathrm { e d i t e d } } ) \leq 2 . 5$ . For $\operatorname { S E } ( 3 )$ , require $1 . 5 \leq \| t _ { i } ^ { \mathrm { e d i t e d } } \| \leq 2 . 5$ . These intervals are the utilityproxy ranges shared by the two groups. Thus the starplot summarizes original-sample tolerance on the sphere and interval retention in the other two settings.

Table 4: Detailed results underlying the toy starplots. Acc and macro-F1 are ranked by proximity to Oracle; higher $D _ { \mathrm { t o y } }$ and U and lower residuals are better. Bold and underline mark the best and second-best edited methods within each geometry; ties share a rank.
<table><tr><td>Method</td><td>Acc</td><td>F1</td><td> $D _ { \mathrm { t o y } }$ </td><td>U</td><td>Mean  $C _ { 1 }$ </td><td>Mean  $C _ { 2 }$ </td></tr><tr><td colspan="7">Hypersphere</td></tr><tr><td>CMF</td><td></td><td></td><td>0.500000000.499443830.982803400.10500000</td><td></td><td>0.02109016</td><td></td></tr><tr><td>SFID</td><td>0.50000000</td><td>0.497767860.98110009 0.05250000</td><td></td><td></td><td>0.08129425</td><td></td></tr><tr><td>SPD</td><td>0.50000000</td><td>0.49986107</td><td>0.98322745</td><td>0.30500000</td><td>0.02415661</td><td></td></tr><tr><td>IsoAct</td><td>0.50833333</td><td>0.50802585</td><td>1.00000000 1.00000000</td><td></td><td>2.33146835e-17</td><td></td></tr><tr><td colspan="7">Lorentz</td></tr><tr><td>CMF</td><td></td><td></td><td>1.00000000 1.00000000 0.00000000 0.97750000</td><td></td><td>3.66718367</td><td>0.00000000</td></tr><tr><td>SFID</td><td></td><td>0.50000000 0.49986107 1.00000000 1.00000000</td><td></td><td></td><td>15.20179613</td><td>0.00000000</td></tr><tr><td>SPD</td><td></td><td>0.48333333 0.47972028 0.96319814 1.00000000</td><td></td><td></td><td>10.98409801</td><td>0.00000000</td></tr><tr><td>IsoAct</td><td>0.50000000 0.49986107</td><td></td><td></td><td></td><td>1.00000000 1.00000000 3.83915122e-15</td><td>0.00000000</td></tr><tr><td colspan="7">SE(3)</td></tr><tr><td>CMF</td><td></td><td></td><td>0.541666670.541380030.886881030.63000000</td><td></td><td>1.22492562</td><td>0.86775255</td></tr><tr><td>SFID</td><td>0.45833333</td><td>0.45829571</td><td>0.95215898 0.58250000</td><td></td><td>0.17973038</td><td>0.00859062</td></tr><tr><td>SPD</td><td></td><td></td><td>0.483333330.482758621.000000000.99250000</td><td></td><td>1.03584875</td><td>0.72990093</td></tr><tr><td>IsoAct</td><td></td><td></td><td></td><td></td><td>0.48333333 0.48275862 1.00000000 1.00000000 2.44253558e-16 1.28785871e-16</td><td></td></tr></table>

![](images/ce68f7b51eb0a3fde8539b447a0117c79eb2e6210a8879c39ec051cd7ce34985.jpg)

![](images/869c6a45df300acbc50342a168d7649b490ab7ba6fa5eafd969ba15e028efae4.jpg)

![](images/376f44dc0eb94470317a86778035130b348586012c5ad562529e4478765988af.jpg)

![](images/5e884745888676da5ab7e325e4f14e53239240ba6e9207285657c221359f7920.jpg)

![](images/00d463b7b061fd2bd338b9c58286b61f2d840b0e0f55e776a82b8edd346c68f6.jpg)

![](images/4331d3093e26446ab858c619b5d62f44788b08d847348d6ad427a4ab469e5138.jpg)  
Figure 5: Hyperspherical mechanism check. The horizontal angle carries nuisance and the polar angle is the shared utility proxy. IsoAct collapses group separation while preserving the proxy. Shading is a visual guide; utility uses each sample’s original-angle $2 ^ { \circ }$ tolerance.

Constraint coordinate. Let $\overline { { C } } _ { j }$ be the sample mean of residual $C _ { j }$ , defined in Section C.5. Set $E = \overline { { C } } _ { 1 }$ on the sphere and ${ \cal E } = ( \overline { { C } } _ { 1 } + \overline { { C } } _ { 2 } ) / 2$ on Lorentz and SE(3). The displayed radius is $r _ { C } = 1 - E / B$ , with panel scales $B = 0 . 1 0$ , 8.00, and 1.20, respectively. Larger radii correspond to smaller violations. The scales support within-panel comparisons rather than absolute residual comparisons across manifolds.

Table 4 gives the probe scores and residuals underlying the starplots; Table 5 adds Oracle discrepancies and utility-proxy errors. IsoAct matches the Oracle probe metrics while satisfying the proxy criterion for every example. Figures 5 and 6 complement the main SE(3) scatter with views of nuisance separation and shared utility structure.

Table 5: Additional controlled results. Nuisance columns show raw scores for Original/Oracle and absolute Oracle discrepancies for edited methods. Lower discrepancies, MAE, and residuals and higher U are better. Bold/underline mark best/second-best edited methods within each geometry; ties share a rank at displayed precision. Table 4 gives higher-precision residuals.
<table><tr><td rowspan="2">Method</td><td colspan="4">Nuisance metric / Oracle discrepancy</td><td colspan="2">Task</td><td colspan="2">Residual mean</td></tr><tr><td>Acc</td><td>Recall</td><td>F1</td><td>AUROC</td><td>MAE</td><td>U</td><td>C1</td><td>C2</td></tr><tr><td colspan="7">Hypersphere</td><td></td><td></td></tr><tr><td></td><td>Original 1.0000</td><td>1.00001.0000</td><td></td><td>1.0000</td><td>0.00001.0000</td><td></td><td>0.0000</td><td></td></tr><tr><td>Oracle</td><td>0.5083</td><td>0.50830.5080</td><td></td><td>0.4956</td><td>0.0000</td><td>1.0000</td><td>0.0000</td><td></td></tr><tr><td>CMF</td><td>0.0083</td><td>0.0083 0.0086</td><td></td><td>0.0023</td><td>3.2653</td><td>0.1050</td><td>0.0211</td><td></td></tr><tr><td>SFID</td><td>0.0083</td><td>0.0083</td><td>0.0103</td><td>0.0000</td><td>13.1276</td><td>0.0525</td><td>0.0813</td><td></td></tr><tr><td>SPD</td><td>0.0083</td><td>0.0083</td><td>0.0082</td><td>0.0109</td><td>2.9891</td><td>0.3050</td><td>0.0242</td><td></td></tr><tr><td>IsoAct</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>1.0000</td><td>0.0000</td><td></td></tr><tr><td colspan="7">Lorentz Original 1.0000 1.0000 1.0000</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>1.0000</td><td></td><td>0.00001.0000</td><td></td><td>0.0000 0.0000</td></tr><tr><td>Oracle</td><td></td><td>0.50000.50000.4999</td><td></td><td>0.5008</td><td>0.0000</td><td>1.0000</td><td></td><td>0.00000.0000</td></tr><tr><td>CMF</td><td></td><td>0.50000.50000.5001</td><td></td><td>0.4992</td><td>0.0161</td><td>0.9775</td><td>3.6672 0.0000</td><td></td></tr><tr><td>SFID</td><td></td><td>0.0000 0.0000 0.0000</td><td></td><td>0.0000</td><td>0.0000</td><td>1.0000</td><td>15.2018 0.0000</td><td></td></tr><tr><td>SPD</td><td>0.0167</td><td>0.0167</td><td>0.0201</td><td>0.0125</td><td>0.0009</td><td>1.0000</td><td>10.9841 0.0000</td><td></td></tr><tr><td>IsoAct</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>1.0000</td><td>0.0000 0.0000</td><td></td></tr><tr><td colspan="7">SE(3)</td><td></td><td></td></tr><tr><td>Original 1.0000 1.0000 1.0000</td><td></td><td></td><td></td><td>1.0000</td><td></td><td>0.00001.0000</td><td>0.0000 0.0000</td><td></td></tr><tr><td>Oracle</td><td>0.4833</td><td>0.48330.4828</td><td></td><td>0.5133</td><td>0.0000</td><td>1.0000</td><td>0.00000.0000</td><td></td></tr><tr><td>CMF</td><td>0.05830.05840.0586</td><td></td><td></td><td>0.0067</td><td>0.4223</td><td>0.6300</td><td>1.22490.8678</td><td></td></tr><tr><td>SFID</td><td>0.02500.02500.0245</td><td></td><td></td><td>0.0377</td><td></td><td>0.46360.5825</td><td>0.17970.0086</td><td></td></tr><tr><td>SPD</td><td>0.0000 0.0000 0.0000</td><td></td><td></td><td>0.0009</td><td>0.0104</td><td>0.9925</td><td>1.03580.7299</td><td></td></tr><tr><td>IsoAct</td><td></td><td></td><td></td><td>0.0000</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.0000 0.0000 0.0000</td><td></td><td></td><td></td><td></td><td>0.0000 1.0000</td><td></td><td>0.0000 0.0000</td></tr></table>

## C.2 KLIFS ligand-pose evaluation

Data and tasks. The train, validation, and test splits contain 7,064, 2,189, and 2,596 poses, respectively, with seed 48 and disjoint kinase IDs. The nine nuisance groups are AGC, Atypical, CAMK, CK1, CMGC, Other, STE, TK, and TKL. Kinase group is a nuisance proxy for familyspecific cues, while utility concerns conformation and ligand placement. Each input pose uses the packing in Section B.6.

The αC-helix task encodes in/out states as 0/1 and excludes other states. Pocket metrics average three separately evaluated binary tasks: front, gate, and back occupancy. Structural-task F1 and Recall treat the minority class identified from training labels as positive; AUROC uses class-1 probabilities.

Predictors and selected settings. Nuisance prediction uses standardization and balanced logistic regression with at most 2,000 iterations. Each utility task uses standardization and a one-hidden-layer multilayer perceptron with 128 units, at most 300 iterations, and seed 42. Predictors are refitted on each transformed training set and evaluated on the associated validation and test sets.

The selected IsoAct configuration uses translation-only actions with K = 5 and α = 2. At each direction, the full packed-pose discrepancy determines the translation update. The selected CMF adaptation has latent dimension 8, geometry weight 0.1, and reconstruction weight 5. SFID and SPD use strengths 2 and 0.75, respectively; SPD requests ten directions.

Table 2 reports the selected configurations’ test nuisance Accuracy and macro-F1, structural-task Recall and AUROC, and mean rotation-constraint residuals. The shuffled-label Null supplies the nuisance-predictability target. Section C.5 defines the residuals.

![](images/c19be598eb0e4e3103596501c400beddd630622f641e1d4ad9b06c9760bf40b2.jpg)  
Figure 6: Lorentz mechanism check. IsoAct removes angular group separation while retaining the shared radial utility range. The dashed line marks the neutral angle and shading marks the shared range; the accompanying residuals assess manifold validity.

## C.3 SE(3) Pareto display

Each point in Figure 4 reports the test performance of one configuration. With nuisance metrics $M _ { m }$ Original values $O _ { m }$ , and Null values ${ \bar { N } } _ { m } .$ , the horizontal coordinate is

$$
D _ { \mathrm { P a r e t o } } = \frac { 1 } { 2 } \sum _ { m \in \{ \mathrm { A c c } , \mathrm { m a c r o - F 1 } \} } \left( 1 - \frac { \left| M _ { m } - N _ { m } \right| } { \left| O _ { m } - N _ { m } \right| } \right) .
$$

The vertical coordinate averages four utility-retention ratios,

$$
U _ { \mathrm { P a r e t o } } = { \frac { 1 } { 4 } } \sum _ { \substack { t \in \{ \alpha C , \mathrm { P o c k e t } \} } } \left( { \frac { \mathrm { R e c a l l } _ { t } } { \mathrm { R e c a l l } _ { t } ^ { \mathrm { O r i g i n a l } } } } + { \frac { \mathrm { A U R O C } _ { t } } { \mathrm { A U R O C } _ { t } ^ { \mathrm { O r i g i n a l } } } } \right) .
$$

Pocket component metrics are averaged before forming the ratios. The plot uses linear axes over $0 \le D \le 0 . { \overset { \_ } { 8 } }$ and $0 . 8 \leq U \leq 1 . 1$

The pools vary IsoAct’s action family, retained rank, and strength; SPD’s rank and strength; SFID’s strength; and CMF’s latent dimension and geometry/reconstruction weights. Edit-strength ranges overlap where applicable. The dashed line connects the highlighted IsoAct configurations to summarize their observed trade-off.

## C.4 CLIP, MERU, FairFace, and Flickr30K

Representations and subsets. CLIP ViT-B/32 supplies 512-dimensional unit-normalized image vectors. MERU ViT-B/16, with input resolution 224, supplies 512 spatial coordinates on the Lorentz model with curvature $- c \approx - 0 . 1$

For each protected attribute, fixed FairFace subsets contain 3,500 training, 1,500 validation, and 1,500 test examples from 86,744 embeddings per model. Sampling without replacement approximately preserves class proportions using rounded class allocations. The respective subset seeds are 42, 43, and 44. Each attribute uses identical subsets for CLIP and MERU.

Age has nine classes: 0 to 2, 3 to 9, 10 to 19, 20 to 29, 30 to 39, 40 to 49, 50 to 59, 60 to 69, and more than 70. Gender labels are Female and Male. Race labels are East Asian, Indian, Black, White, Middle Eastern, Latino/Hispanic, and Southeast Asian. Each set of dataset annotations defines a separate nuisance-prediction task.

Table 6: Selected IsoAct settings for CLIP and MERU. K is the retained rank and $\rho$ the FDA ridge. Ang. and Trans. denote angular and translation actions composed within each direction; a dash indicates a single spherical action.
<table><tr><td></td><td>Model Attribute K</td><td></td><td>ρ</td><td></td><td>Passes Strengths</td><td>Within-axis order</td></tr><tr><td>CLIP</td><td>Age</td><td>7</td><td>0.1</td><td></td><td>1 α = 1</td><td></td></tr><tr><td></td><td>Gender</td><td>1</td><td>0.01</td><td></td><td>1 α = 1</td><td></td></tr><tr><td></td><td>Race</td><td>6</td><td>0.1</td><td></td><td> $2 \alpha = 1$ </td><td></td></tr><tr><td>MERU</td><td>Age</td><td>8</td><td>0.01</td><td>1</td><td> $\alpha _ { \mathrm { a n g } } = 0 . 7 5 , \alpha _ { \mathrm { t r } } = 1$ </td><td>Ang. → Trans.</td></tr><tr><td></td><td>Gender</td><td></td><td>1 0.03</td><td>1</td><td> $\alpha _ { \mathrm { a n g } } = 0 . 5 , \alpha _ { \mathrm { t r } } = 1$ </td><td> ${ \mathrm { T r a n s . } } \to \mathbf { A n g . }$ </td></tr><tr><td></td><td>Race</td><td>6</td><td>0.1</td><td>1</td><td> $\alpha _ { \mathrm { a n g } } = 0 . 5 , \alpha _ { \mathrm { t r } } = 1$ </td><td> ${ \mathrm { T r a n s . } } \to \mathbf { A n g . }$ </td></tr></table>

Nuisance probes and Null. Each transformed training set is used to fit standardization and an L2-regularized logistic-regression probe with lbfgs, balanced class weights, inverse regularization strength 1, at most 2,000 iterations, and seed 42. The same fitted probe supplies validation and test Accuracy and macro-F1. Null is estimated by averaging scores over deterministic label permutations against the Original predictions.

Retrieval utility. Flickr30K’s Karpathy validation split contains 1,014 images and 5,070 text queries; its test split contains 1,000 images and 5,000 queries. The learned transformation edits image representations, while text representations remain fixed. Each query retrieves its corresponding image from the full gallery. CLIP uses dot-product similarity. For MERU, scores take the form softmax $( 1 0 0 V T ^ { \top } )$ ), with image vectors as rows of V and text vectors as rows of T.

R@k measures top-k retrieval success. Exact score ties use an expected-uniform policy: if g images score above the target and m images, including the target, tie with it, the query contributes cli $\mathrm { p } ( k - g , 0 , m ) / m$ . Averaging over all queries gives R@1, R@5, and R@10.

Reported coordinates. For edited, Original, and Null nuisance metrics $M _ { m } , O _ { m } , N _ { m }$ , define $\bar { D _ { m } } = 1 - | M _ { m } - N _ { m } | / | O _ { m } - N _ { m } |$ . Debias averages $D _ { \mathrm { A c c } }$ and $D _ { \mathrm { m a c r o ^ { - } F 1 } }$ . Retrieval retention is

$$
U _ { \mathrm { r e t } } = \frac { 1 } { 3 } \sum _ { k \in \{ 1 , 5 , 1 0 \} } \frac { \mathrm { R @ } k } { \mathrm { R @ } k _ { \mathrm { O r i g i n a l } } } .
$$

Ratios are formed separately before averaging, using the corresponding Original and Null references and finite, nonzero denominators. Negative Debias values indicate greater normalized distance from Null than Original; retention above one indicates a gain relative to Original.

Selected configurations. Table 6 lists the selected IsoAct rank, ridge, strengths, and composition order. All use forward direction order. CLIP race repeats its six-direction sequence twice; MERU composes angular and translation actions within each direction.

Tables 7 and 8 provide the raw test metrics underlying the main table’s normalized summaries.

## C.5 Manifold-constraint diagnostics

For a unit-sphere output, $C _ { 1 } ( x ) = | \| x \| _ { 2 } - 1 |$ . On the Lorentz model with curvature −c, where $c > 0 .$ a point $\boldsymbol { x } = \left( \boldsymbol { x } _ { 0 } , \boldsymbol { v } \right)$ has residuals

$$
C _ { 1 } ( x ) = | - x _ { 0 } ^ { 2 } + \| v \| _ { 2 } ^ { 2 } + c ^ { - 1 } | , \qquad C _ { 2 } ( x ) = \operatorname* { m a x } ( 0 , - x _ { 0 } ) .
$$

For a pose with rotation block R,

$$
C _ { 1 } ( R ) = \Vert R ^ { \top } R - I _ { 3 } \Vert _ { F } , \qquad C _ { 2 } ( R ) = | \operatorname* { d e t } R - 1 | .
$$

We report sample means $\overline { { C } } _ { 1 } , \overline { { C } } _ { 2 } .$ . Real-world diagnostics use the selected FairFace test embeddings (1,500 per attribute) and KLIFS test poses (2,596). The Lorentz terms measure quadratic-constraint error and negative-time magnitude; the rotation terms measure orthogonality and determinant error.

Table 7: CLIP raw test probe and retrieval metrics. F1 denotes macro-F1. Bold/underline mark best/second-best methods within each attribute: smaller absolute distance to Null for Acc/F1, higher values for retrieval. Original and Null are unranked; ties share a rank. Table 3 reports normalized summaries.
<table><tr><td>Method</td><td>Acc</td><td>F1</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>Age</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Original Null</td><td>0.2987 0.1477</td><td>0.2614 0.1060</td><td>0.5876</td><td>0.8350</td><td>0.9004</td></tr><tr><td>CMF SFID SPD</td><td>0.1920 0.2967 0.1860</td><td>0.1742 0.2599 0.1431</td><td>0.0012 0.5892 0.5764</td><td>0.0070 0.8354 0.8268</td><td>0.0166 0.8990</td></tr><tr><td>IsoAct Gender Original</td><td>0.1320</td><td>0.1137</td><td>0.5770</td><td>0.8290</td><td>0.8906 0.8934</td></tr><tr><td></td><td>0.7720</td><td>0.7714</td><td>0.5876</td><td>0.8350</td><td>0.9004</td></tr><tr><td>Null</td><td>0.5008</td><td>0.4995</td><td></td><td></td><td></td></tr><tr><td>CMF</td><td>0.6507</td><td>0.6494</td><td>0.0028</td><td>0.0102</td><td>0.0198</td></tr><tr><td>SFID</td><td>0.7613</td><td>0.7610</td><td>0.5706</td><td>0.8198</td><td>0.8856</td></tr><tr><td>SPD</td><td>0.5480</td><td></td><td></td><td></td><td></td></tr><tr><td>IsoAct</td><td></td><td>0.5478</td><td>0.5810</td><td>0.8322</td><td>0.8968</td></tr><tr><td></td><td>0.6567</td><td>0.6558</td><td>0.5874</td><td>0.8350</td><td>0.9016</td></tr><tr><td>Race</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Original</td><td>0.3973</td><td>0.3910</td><td>0.5876</td><td>0.8350</td><td></td></tr><tr><td>Null</td><td>0.1457</td><td></td><td></td><td></td><td>0.9004</td></tr><tr><td>CMF</td><td>0.2327</td><td>0.1430</td><td></td><td></td><td></td></tr><tr><td>SFID</td><td></td><td>0.2116</td><td>0.0022</td><td>0.0098</td><td>0.0170</td></tr><tr><td></td><td>0.3993</td><td>0.3932</td><td>0.5840</td><td>0.8354</td><td>0.8986</td></tr><tr><td>SPD</td><td>0.1607</td><td>0.1454</td><td>0.5800</td><td>0.8274</td><td>0.8908</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>IsoAct</td><td>0.2033</td><td>0.1844</td><td>0.5818</td><td>0.8310</td><td>0.8952</td></tr></table>

Table 8: MERU raw test probe and retrieval metrics. F1 denotes macro-F1. Bold/underline mark best/second-best methods within each attribute: smaller absolute distance to Null for Acc/F1, higher values for retrieval. Original and Null are unranked; ties share a rank. Table 3 reports normalized summaries.
<table><tr><td>Method</td><td>Acc</td><td>F1</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>Age</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Original Null</td><td>0.2747</td><td>0.2465</td><td>0.2096</td><td>0.4242</td><td>0.5216</td></tr><tr><td>CMF</td><td>0.1439 0.2573</td><td>0.1055 0.2359</td><td>0.0372</td><td>0.1188</td><td>0.1778</td></tr><tr><td>SFID SPD</td><td>0.2587 0.1373</td><td>0.2424</td><td>0.2030</td><td>0.4194 0.4146</td><td>0.5160 0.5094</td></tr><tr><td>IsoAct</td><td>0.1413</td><td>0.1236 0.1199</td><td>0.1986 0.1900</td><td>0.4114</td><td>0.5082</td></tr><tr><td>Gender</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Original</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Null</td><td>0.7440</td><td>0.7435</td><td>0.2096</td><td>0.4242</td><td>0.5216</td></tr><tr><td>CMF</td><td>0.4976</td><td>0.4966</td><td></td><td></td><td></td></tr><tr><td></td><td>0.6827</td><td>0.6818</td><td>0.0428</td><td>0.1270</td><td>0.1968</td></tr><tr><td>SFID</td><td>0.7440</td><td>0.7435</td><td>0.2070</td><td>0.4244</td><td>0.5190</td></tr><tr><td>SPD</td><td>0.6000</td><td>0.5996</td><td>0.2048</td><td>0.4180</td><td>0.5206</td></tr><tr><td>IsoAct</td><td>0.5247</td><td>0.5242</td><td>0.1992</td><td>0.4108</td><td>0.5158</td></tr><tr><td>Race</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Original</td><td>0.3987</td><td></td><td>0.2096</td><td>0.4242</td><td>0.5216</td></tr><tr><td>Null</td><td></td><td>0.3900</td><td></td><td></td><td></td></tr><tr><td></td><td>0.1443</td><td>0.1421</td><td></td><td></td><td></td></tr><tr><td>CMF</td><td>0.1967</td><td>0.1902</td><td>0.0024</td><td>0.0088</td><td>0.0152</td></tr><tr><td>SFID</td><td>0.3860</td><td>0.3793</td><td>0.1994</td><td>0.4082</td><td>0.5076</td></tr><tr><td>SPD</td><td>0.1860</td><td>0.1735</td><td>0.1892</td><td>0.3978</td><td>0.4986</td></tr><tr><td>IsoAct</td><td>0.1507</td><td>0.1445</td><td>0.1970</td><td>0.3940</td><td>0.5038</td></tr></table>

Table 9: Mean manifold-constraint residuals on selected test representations. IsoAct maintains unit-norm and rotation constraints to numerical precision. Lower is better. Bold/underline mark best/second-best methods within each setting, excluding Original and all MERU residuals. MERU’s Lorentz quadratic and upper-sheet residuals are reported separately.
<table><tr><td>Setting</td><td>Method</td><td>Mean  $C _ { 1 }$ </td><td>Mean  $C _ { 2 }$ </td></tr><tr><td>KLIFS</td><td>Original</td><td>6.5065e-08</td><td>2.2077e-08</td></tr><tr><td></td><td>CMF</td><td>3.6900e-01</td><td>1.1368e-01</td></tr><tr><td></td><td>SFID</td><td>6.5065e-08</td><td>2.2077e-08</td></tr><tr><td></td><td>SPD</td><td>9.4948e-01</td><td>3.7450e-01</td></tr><tr><td></td><td>IsoAct</td><td>6.9510e-16</td><td>3.5723e-16</td></tr><tr><td>CLIP age</td><td>Original</td><td>2.2612e-08</td><td></td></tr><tr><td></td><td>CMF</td><td>8.8207e-02</td><td></td></tr><tr><td></td><td>SFID</td><td>3.8070e-03</td><td></td></tr><tr><td></td><td>SPD</td><td>1.8327e-02</td><td></td></tr><tr><td></td><td>IsoAct</td><td>6.3230e-08</td><td></td></tr><tr><td>CLIP gender</td><td>Original</td><td>2.2935e-08</td><td></td></tr><tr><td></td><td>CMF</td><td>7.0113e-02</td><td></td></tr><tr><td></td><td>SFID</td><td>8.1287e-03</td><td></td></tr><tr><td></td><td>SPD</td><td>1.0768e-02</td><td></td></tr><tr><td></td><td>IsoAct</td><td>3.5303e-08</td><td></td></tr><tr><td>CLIP race</td><td>Original</td><td>2.2181e-08</td><td></td></tr><tr><td></td><td>CMF</td><td>1.1123e-01</td><td></td></tr><tr><td></td><td>SFID</td><td>8.3426e-03</td><td></td></tr><tr><td></td><td>SPD</td><td>5.3183e-02</td><td></td></tr><tr><td></td><td>IsoAct</td><td>6.1190e-08</td><td></td></tr><tr><td>MERU age</td><td>Original</td><td>7.9818e-16</td><td>0.0000e+00</td></tr><tr><td></td><td>CMF</td><td>7.6620e-16</td><td>0.0000e+00</td></tr><tr><td></td><td>SFID</td><td>7.8515e-16</td><td>0.0000e+00</td></tr><tr><td></td><td>SPD</td><td>7.8397e-16</td><td>0.0000e+00</td></tr><tr><td></td><td>IsoAct</td><td>7.7804e-16</td><td>0.0000e+00</td></tr><tr><td>MERU gender</td><td>Original</td><td>8.1239e-16</td><td>0.0000e+00</td></tr><tr><td></td><td>CMF</td><td>7.8278e-16</td><td>0.0000e+00</td></tr><tr><td></td><td>SFID</td><td>7.7804e-16</td><td>0.0000e+00</td></tr><tr><td></td><td>SPD</td><td>7.6383e-16</td><td>0.0000e+00</td></tr><tr><td></td><td>IsoAct</td><td>7.6383e-16</td><td>0.0000e+00</td></tr><tr><td>MERU race</td><td></td><td></td><td></td></tr><tr><td></td><td>Original CMF</td><td>7.8870e-16</td><td>0.0000e+00 0.0000e+00</td></tr><tr><td></td><td>SFID</td><td>8.0647e-16 7.6383e-16</td><td>0.0000e+00</td></tr><tr><td></td><td>SPD</td><td>7.8989e-16</td><td>0.0000e+00</td></tr><tr><td></td><td>IsoAct</td><td>7.8515e-16</td><td>0.0000e+00</td></tr></table>

MERU represents each spatial vector v with dependent time coordinate $x _ { 0 } = \sqrt { \| v \| ^ { 2 } + c ^ { - 1 } } > 0 ,$ This satisfies the hyperboloid constraint in exact arithmetic, explaining the numerical-precision $C _ { 1 }$ and zero $C _ { 2 }$ across methods. The main table averages $\overline { { C } } _ { 1 }$ and $\bar { \overline { { C } } } _ { 2 } .$ , while Table 9 lists both residuals separately.

## D Compute Resources

Experiments on frozen representations used an Ubuntu CPU node with an Intel Xeon Gold 6326 processor at 2.90 GHz, 64 CPU threads, and 251 GiB RAM. The final evaluation environment included Python 3.10.19, NumPy 1.26.4, SciPy 1.15.3, and scikit-learn 1.7.2. Dataset splits, evaluation predictors, and selected settings are specified in Appendix C.