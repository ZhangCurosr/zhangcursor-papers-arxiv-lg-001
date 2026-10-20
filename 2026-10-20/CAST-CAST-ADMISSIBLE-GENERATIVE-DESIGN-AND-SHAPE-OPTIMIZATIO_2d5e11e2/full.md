# CAST: CAST-ADMISSIBLE GENERATIVE DESIGN AND SHAPE OPTIMIZATION WITH PAIRED DEPTH IMAGES

Jihoon Kim<sup>1,3,†</sup> Yongmin Kwon<sup>2,3,†</sup> Namwoo Kang<sup>1,3,∗</sup>

<sup>1</sup> Department of Mechanical Engineering,

Korea Advanced Institute of Science and Technology, Daejeon 34141, Republic of Korea <sup>2</sup> Cho Chun Shik Graduate School of Mobility,

Korea Advanced Institute of Science and Technology, Daejeon 34051, Republic of Korea <sup>3</sup> Narnia Labs, Daejeon 34051, Republic of Korea

jihoon.kim@kaist.ac.kr kymin1002@kaist.ac.kr nwkang@kaist.ac.kr

<sup>†</sup> Jihoon Kim and Yongmin Kwon contributed equally to this work.

∗ Corresponding author: Namwoo Kang

## ABSTRACT

Deep generative design can produce complex geometries, but unconstrained representations may create undercuts that prevent release from a two-part mold. We present CAST, which reformulates three-dimensional shape generation under a selected mold-opening constraint as a two-dimensional image generation problem using paired top and bottom depth images. Each valid depth pair defines one material interval along a selected mold-opening direction, so reconstructions exclude undercuts in that direction by construction. Diffusion models generate candidate shapes in this representation, and a finite element analysis (FEA) guided optimizer refines the same two boundaries. We evaluate diffusion-based generation with limited training data comprising 211 SimJEB bracket designs. On SimJEB brackets, FEA-guided refinement with a volume penalty reduced compliance in 193 of 200 generated candidates. The median reduction across all candidates was 36.8 percent in the optimization FEA model. Reconstruction of existing brackets revealed a mass–stiffness trade-off: filling occluded regions increased material volume and reduced displacement under the evaluated loads. Paired depth images thus connect generative exploration and structural refinement in a shared design space that preserves the selected pull-direction restriction.

Keywords deep generative design, design for manufacturing, die casting, shape representation

## 1 Introduction

Deep generative models are increasingly used in engineering design to synthesize large and diverse sets of candidate geometries from limited prior designs [1–3]. Recent applications span native three-dimensional shape synthesis, topology-optimized structural components, and architected material systems [4–7]. For these models to contribute to mass production, however, generated candidates must satisfy not only functional objectives but also the geometric requirements of the intended manufacturing process. Although additive manufacturing can accommodate many complex proposals, its production rate and unit cost do not always suit large production volumes [8, 9]. At high volumes, die casting and injection molding remain economical because tooling and setup costs are spread across many parts [10–12]. Components produced by these processes must permit mold separation along a feasible draw direction, avoid undercuts, and provide sufficient draft for ejection [13–15]. Functional and assembly interfaces impose additional, part-specific geometric constraints. When these requirements are introduced only after generation, geometric repair can alter both the candidate shape and its mechanical behavior.

The central challenge is therefore not simply to generate complex geometry, but to make relevant manufacturing constraints native to the design space. Geometric representations chosen for expressive flexibility do not necessarily preserve such constraints as designs move between generation, reconstruction, analysis, and refinement. Constraint checks and learned design repair provide ways to handle these violations [1, 16]. A process-aware representation offers a more direct alternative: if every valid state already satisfies a critical geometric condition, generated candidates inherit that condition before screening or modification. For two-part molds, one such condition is the absence of undercuts along the selected pull direction. An undercut traps mold material and obstructs withdrawal along that direction (Fig. 1(a,b)).

![](images/0bb8a7bf15766b98d19c19c6d7a3f5105d96ca7f73deca472cc80386379452d6.jpg)  
Figure 1: Undercuts and mold separation along the selected pull direction. (a) Mold material occupies a side recess in the part. (b) Attempted upward motion of the upper mold produces interference with the part, highlighted in red. (c) Filling the interval between the opposing boundaries removes the recess and the pull-direction interference. Sage-green hatching marks the added material.

![](images/9bc4a1c57b6606c702ad89fbd93ae70a7a6783fecbbedf1354b74d99856aef1e.jpg)  
Figure 2: Depth calculation and reconstruction for paired top and bottom fields. Orthographic rays record first-surface distances from opposite camera planes, and the interval between the paired surfaces is filled to reconstruct a solid with at most one material interval along the selected pull direction.

To embed this condition in the shape representation, we use two depth images viewed from opposite sides along the selected mold-opening direction. These images record the upper and lower surfaces, and the space between them is filled with material (Fig. 2). This construction leaves no hidden gaps beneath overhangs that would obstruct mold withdrawal along that direction (Fig. 1(c)). We call shapes satisfying this geometric condition cast-admissible. The representation therefore excludes undercuts by construction.

This two-image representation enables two-dimensional generative models to propose candidate shapes without learning a dense volumetric field [17]. Appropriate inductive biases can improve learning from limited data by incorporating relevant structure [18]. By combining a compact representation with an explicit pull-direction constraint, CAST may therefore reduce training data requirements relative to unconstrained volumetric generation. The same upper and lower boundaries also serve as design variables for finite element analysis (FEA) guided shape optimization, allowing generated candidates to be refined while retaining the selected pull-direction restriction. We parameterize these boundary updates using cubic B-spline free-form deformation (FFD). Together, the paired-depth representation, generation, and shape optimization form the CAST framework shown in Fig. 3.

We evaluate CAST on the SimJEB bracket family [19]. Generation experiments demonstrate the creation of new candidates within the paired-depth design space, and FEA-guided optimization demonstrates structural improvement through updates to the same boundaries under a volume penalty. Reconstruction experiments complement these results by quantifying the geometric and mass–stiffness trade-offs involved in mapping existing designs into this space.

![](images/96f96865186da1326dea4f408ae49fd2d12c55030933467afa988cf3910b0af9.jpg)  
Figure 3: The CAST framework. (a) Valid paired depth fields reconstruct a solid without undercuts along the selected pull direction. (b) A diffusion model generates new paired depth fields. (c) FEA-guided optimization refines the same upper and lower boundaries along that direction while keeping the in-plane footprint fixed.

## The contributions of this work are:

• a paired-depth representation that reformulates pull-direction-constrained 3D shape generation as a twodimensional image generation problem, with valid reconstructions excluding undercuts by construction;

• a generation and FEA-guided shape optimization framework that refines the same depth boundaries used to generate candidates while retaining the selected pull-direction restriction; and

• a structural assessment of paired-depth reconstruction that establishes its mass–stiffness trade-off on SimJEB brackets, showing how filling occluded regions increases material volume and reduces displacement across the evaluated load cases.

Section 2 reviews related work. Section 3 presents the paired-depth representation, generative models, and FEA-guided refinement. Section 4 evaluates reconstruction, generation, and shape optimization, followed by a discussion of the findings and limitations in Section 5 and conclusions in Section 6.

## 2 Related Work

## 2.1 Generative Design

Generative design uses optimization and learned models to explore candidate geometries that satisfy performance objectives [1, 20, 21]. Common deep generative models include variational autoencoders (VAEs), generative adversarial networks (GANs), and diffusion models [22–24]. Deep learning approaches have generated topology-optimized structures [2, 25–28], lattice structures [29], and nanophotonic designs [30]. Conditional GANs have also generated shapes within specified attribute ranges [31], predicted topologies using transfer learning [32], and incorporated layout and mechanical constraints through multiple discriminators [33]. Deep generative and optimization methods have also produced three-dimensional shapes from limited data [34], but such outputs target performance or diversity and are not restricted to geometry that a two-part mold can release. Manufacturing constraints are often introduced through training data, conditioning, or post-processing [35, 36]. The present work instead makes one mold-release restriction explicit in the representation on which both generation and shape optimization operate.

## 2.2 Design for Manufacturing

Design for Manufacturing introduces process constraints while design freedom is still available, reducing later redesign and production cost [37–39]. Patterson et al. [40] surveyed methods for generating and enforcing process-driven manufacturability constraints across design domains. For injection molding and die casting, relevant geometric checks include mold-opening direction, undercuts, draft, wall thickness, and interfaces [15, 41]. These checks are coupled but not interchangeable: a shape can be free of undercuts in one pull direction and still violate draft or wall thickness guidance.

Casting constraints have also been incorporated into structural optimization. Explicit parameterizations have represented solid–void boundaries along a casting direction [42] and optimized stiffener layout and height through directional growth [43]. Other approaches have imposed manufacturing constraints through density projections [44], shadow density filters [45], and density-gradient measures of undercuts [46]. Parting lines have been incorporated into multicomponent topology optimization [47], and recent work jointly optimized cast geometry, draw directions, nonplanar parting surfaces, and side dies [48].

Generative design research has more often treated additive manufacturing limits, where the feasible geometry differs from that of two-part molds [49–51]. Neural networks have also been used to jointly optimize build direction, part segmentation, and topology for additive manufacturing [52]. Manufacturing-aware generation has used synthetic machinable shapes to train GANs for latent-space optimization [53] and domain rules to generate sheet metal parts [54]. Recent work has incorporated injection molding rules into learned repair of existing designs [16], and a survey of diffusion models in design and manufacturing identifies demoldability as an open constraint [55]. CAST incorporates the selected pull-direction restriction into the representation used to generate and refine shapes.

## 2.3 Depth Images

Depth images store the distance from a viewing plane to the first observed surface and are widely used for threedimensional perception and reconstruction [56–59]. Layered depth images store multiple surfaces along viewing rays [60], while layered depth-normal images support solid modeling operations [61]. Height fields have also been used to approximate free-form geometry for manufacturing [62], and multi-view depth maps have supported three-dimensional shape generation with VAEs [17] and diffusion models [63]. In CAST, two opposing orthographic depth images define the upper and lower boundaries of a solid, and filling the space between them produces a single material interval along each line parallel to the selected pull direction.

## 2.4 Physics-Based Shape Optimization

Topology optimization and learned generative models have been combined by training on optimized designs, predicting near-optimal layouts, or guiding search with learned distributions [2, 51, 64]. Wang et al. [65] embedded compliance and volume fraction terms into conditional GAN training to generate functional designs. Shape optimization can instead begin from a generated candidate and optimize a lower-dimensional boundary parameterization. Cubic Bspline FFD supplies smooth boundary control through a surrounding lattice [66], and has been used to optimize the engineering performance of three-dimensional generated and augmented geometries [67]. In CAST, FFD updates the upper and lower depth boundaries directly, enabling structural refinement within the same design representation used for generation.

![](images/68f97927e7cda24e22f6aeceb0ea612bb08d9310a8bb5e56e358cb9f48c309ef.jpg)  
Figure 4: Paired top and bottom depth fields for six SimJEB brackets. Depth is measured in the normalized coordinate system.

CAST thus connects shape generation and FEA-guided refinement within a shared paired-depth design space that preserves the selected pull-direction restriction.

## 3 Methods

CAST uses paired depth images to represent, generate, and refine three-dimensional shapes (Fig. 3). We first describe how the two depth boundaries define a solid and how they are obtained from existing shapes. Generative models then learn to produce new depth pairs, and FEA-guided optimization refines selected candidates by moving the same boundaries.

## 3.1 Two-Depth Representation

Let the selected pull direction be the z axis, and let $P _ { i j }$ denote the in-plane cell associated with pixel $( i , j )$ on an $H \times W$ orthographic grid. The two depth channels derive, rather than store separately, a foreground mask $m _ { i j }$ and boundaries $z _ { i j } ^ { + }$ and $z _ { i j } ^ { - }$ . The represented body is

$$
\Omega _ { D } = \bigcup _ { i , j : m _ { i j } = 1 } \left\{ ( x , y , z ) : ( x , y ) \in P _ { i j } , z _ { i j } ^ { - } \leq z \leq z _ { i j } ^ { + } \right\} ,\tag{1}
$$

where $m _ { i j } ~ = ~ 0$ omits the corresponding column. A paired foreground column has valid depth ordering when $D _ { \mathrm { t o p } } + D _ { \mathrm { b o t t o m } } \leq 0 . 8 $ , equivalently $z _ { i j } ^ { - } \leq z _ { i j } ^ { + }$ . Every line parallel to z then intersects $\Omega _ { D }$ in either an empty set or one connected interval. The no-undercut property used in this paper refers to this reconstructed body and this selected pull direction.

## 3.2 Depth Calculation

Opposing orthographic cameras record the distance to the first surface encountered along each inward ray $( \mathrm { F i g } . 2 )$ Parallel projection preserves a direct mapping between each pixel and a column along the selected pull direction. The paired distances determine the upper and lower surface coordinates, while rays that miss the shape identify empty columns.

For the SimJEB engine brackets used in this study [19], we selected the z axis as the pull direction because the registered family presents its principal bracket face to that axis. Input computer-aided design (CAD) coordinates in millimeters were mapped to a fixed normalized frame computed once over the retained 264 designs. The exact affine center and scale are provided [See Supplemental Material]. Two 256 × 256 grids use $( \widetilde { x } _ { i } , \widetilde { y } _ { j } ) \stackrel { - } { = } ( 2 i / 2 5 6 - 1 , 2 j / 2 5 6 - 1 )$ and camera planes $\widetilde { z } = \pm 0 . 4$ . The first-surface distances are stored as $D _ { \mathrm { t o p } }$ and $\begin{array} { r l } {  { D _ { \mathrm { b o t t o m } } . } } \end{array}$ with 0.8 marking a missed ray. After range mapping, $m _ { i j } = 1$ only when both channel values are below 0.78. Otherwise, the column is empty. For foreground pixels, $z ^ { + } = \bar { 0 } . 4 - D _ { \mathrm { t o p } }$ and $z ^ { - } = - 0 . 4 + D _ { \mathrm { b o t t o m } }$ . Representative paired fields are shown in Fig. 4. The full pseudocode is provided [See Supplemental Material].

## 3.3 Shape Reconstruction

At each in-plane pixel, the reconstruction algorithm forms a sub-voxel scalar field over 128 samples along z, fills the single interval between the top and bottom surfaces, and extracts its zero level set with marching cubes (Fig. 2). The algorithm maps depths below 0 to 0 and depths at least 0.78 to the empty-ray value 0.8 before reconstruction. Under valid depth ordering, the single-interval column construction excludes undercuts along the selected pull direction by construction. It does not preserve internal cavities or features that are occluded along that direction. The implementation equivalent pseudocode is provided [See Supplemental Material].

![](images/f156ed345bc12eed7fff8fae4696c86f5dc295f0918a4bf51b00f18f09c80640.jpg)

![](images/1b6d8b8fd854a554d67256ab1a8272d7d5d5c18b03e3942cbcaa631242b6095d.jpg)  
Figure 5: Restoration of transverse mounting holes after depth reconstruction. Boolean subtraction reopens the filled holes (circled).

## 3.4 Manufacturing-Oriented Post-processing

## 3.4.1 Positive Draft Angle Enforcement

A chosen positive draft angle can also be formulated as a constraint on depth field slopes. An illustrative projection is provided [See Supplemental Material]. It was not used in the reported experiments.

## 3.4.2 Interface Feature Restoration

Depth reconstruction fills mounting holes transverse to the selected pull direction. We restore these specified holes using Boolean mesh operations, as illustrated in Fig. 5. In manufacturing, these transverse holes can be formed using side-action core pins that retract along the hole axes before part ejection, or machined after molding. Their restoration is therefore treated as a separate operation beyond the selected pull-direction guarantee.

## 3.5 Shape Generation from Paired Depth Images

Each training shape is converted into a pair of depth images using the procedure above. A diffusion model learns the distribution of these image pairs and samples new pairs, which are reconstructed into three-dimensional solids by filling the intervals between their boundaries.

The generative-model evaluation used pixel-space diffusion to sample paired depth maps, which were then reconstructed into three-dimensional shapes. The subsequent FEA-guided optimization study used a separate set of candidates generated by latent-space diffusion. Both pipelines produce the same paired-depth representation, whose upper and lower boundaries initialize the optimizer.

The pixel-space model operates directly on the $2 \times 2 5 6 \times 2 5 6$ depth tensor, while the latent-space model operates on a $4 \times 6 4 \times 6 4$ representation produced by a fixed deterministic autoencoder. Both use a U-Net denoising model with v prediction and a sigmoid noise schedule, followed by 250-step deterministic DDIM sampling [24, 68]. The latent autoencoder was trained from scratch and selected using the validation split. Its compact spatial representation reduces the workload of diffusion training. Model settings and computational costs are detailed [See Supplemental Material].

## 3.6 FEA-Guided B-Spline Shape Optimization

A generated depth pair initializes the FEA-guided refinement in Fig. 3(c). The optimizer changes thickness by moving the upper and lower boundaries along the selected pull direction while holding the in-plane footprint fixed. Before each FEA evaluation, the boundaries are projected into the design envelope to maintain minimum thickness and prescribed interfaces. Generation and refinement are sequential: the generative model supplies the initial depth pair, and FEA compliance sensitivities guide subsequent boundary updates. We refine each boundary by adding a smooth displacement field to the generated depth map,

$$
\begin{array} { r } { { \bf z } ^ { + } = { \bf z } _ { 0 } ^ { + } + B { \bf c } ^ { + } , \qquad { \bf z } ^ { - } = { \bf z } _ { 0 } ^ { - } + B { \bf c } ^ { - } , } \end{array}\tag{2}
$$

where ${ \bf z } _ { 0 } ^ { \pm }$ are the generated depth maps and $\mathbf { c } ^ { \pm }$ contain the vertical displacements of an $n _ { c } \times n _ { c }$ grid of control points for each boundary. The matrix $B \in \mathbb { R } ^ { H W \times n _ { c } ^ { 2 } }$ maps these control point displacements to the depth map grid using products of clamped-uniform cubic B-spline basis functions in the two in-plane directions. In this study, we used $n _ { c } = 2 0$ , giving 800 scalar control displacements across the two boundaries before fixing the interface controls described below. Only displacement along z is permitted, and the in-plane sampling grid does not move.

Before each FEA evaluation, the depth boundaries were projected into the design envelope while imposing a prescribed minimum thickness in columns identified as initial material. The fixed and load interfaces were restored to their prescribed envelope boundaries, and nearby control points were held fixed to preserve these connections. The same projection settings and 60-iteration optimization budget were used for all candidates. Thresholds, interface mask construction, and the learning rate schedule are specified [See Supplemental Material].

For analysis, the hard interval in Eq. (1) was replaced by a differentiable density field sampled along z,

$$
\rho _ { i j k } = \sigma \big ( \beta ( z _ { k } - z _ { i j } ^ { - } ) \big ) \sigma \big ( \beta ( z _ { i j } ^ { + } - z _ { k } ) \big ) ,\tag{3}
$$

where $\sigma$ is the logistic sigmoid and β controls the sharpness of the material boundary. The density field was transferred to a tetrahedral mesh of the fixed design envelope for linear-elastic FEA. The mounting interfaces were constrained and a single diagonal load, oriented along $( 1 , - 1 , - 1 ) / \sqrt { 3 }$ , was applied at the load interface. This optimization load is distinct from the four load cases used to evaluate the reconstructions (Sec. 3.7), including their SimJEB Diagonal case, which acts along a different direction. Because the material modulus is normalized and the load is expressed in analysis units, compliance is interpreted as a within-case relative measure. Density resolution, boundary sharpness, material parameters, mesh and interface definitions, and solver settings are specified [See Supplemental Material].

Compliance sensitivity was propagated through the transpose of the forward density transfer, Eq. (3), and $B ^ { \mathsf { T } }$ . The FEA density sensitivity and the linear density-transfer transpose were checked against centered finite differences. This check does not include the hard geometry projections. Across five random directions at each of three base densities, the adjoint directional derivative matched the centered difference to a relative error below $1 0 ^ { - 3 }$ [See Supplemental Material].

Boundary updates combine compliance sensitivities with a volume penalty that discourages material change and a control displacement penalty that limits deformation. The update is a normalized-gradient heuristic rather than descent on one scalar weighted objective. Let $g ^ { + }$ and $g ^ { - }$ be compliance sensitivities with respect to the projected depth boundaries, M the indicator of columns with current positive thickness, $\mathcal { N } ( g ) = g / ( \| g \| _ { 2 } + 1 0 ^ { - 1 2 } )$ , $w _ { V }$ a volume penalty weight (default 3), $w _ { c }$ a control displacement weight (default 0.3), and

$$
V _ { f } = \frac { 1 } { H W ( z _ { \mathrm { h i } } - z _ { \mathrm { l o } } ) } \sum _ { i , j } \operatorname* { m a x } ( z _ { i j } ^ { + } - z _ { i j } ^ { - } , 0 ) ,\tag{4}
$$

$$
s = \operatorname { t a n h } ( 1 0 ( V _ { f } - V _ { f , 0 } ) ) .\tag{5}
$$

Here $V _ { f , 0 }$ is computed from the envelope-clipped generated depth pair before the per-iteration minimum thickness and interface projections. The reported initial $V _ { f }$ denotes the step-0 projected state and can therefore differ from $V _ { f , 0 } ,$ . The control updates are

$$
\begin{array} { r } { q ^ { + } = \mathcal { N } ( B ^ { \mathsf { T } } g ^ { + } ) + w _ { V } \mathcal { N } ( B ^ { \mathsf { T } } ( s M ) ) + w _ { c } c ^ { + } , } \end{array}\tag{6}
$$

$$
\begin{array} { r } { q ^ { - } = \mathcal { N } ( B ^ { \mathsf { T } } g ^ { - } ) + w _ { V } \mathcal { N } ( B ^ { \mathsf { T } } ( - s M ) ) + w _ { c } c ^ { - } , } \end{array}\tag{7}
$$

$$
c ^ { + }  c ^ { + } - \eta q ^ { + } , \qquad c ^ { - }  c ^ { - } - \eta q ^ { - } .\tag{8}
$$

The volume and control terms discourage material change and large deformations but do not impose exact constant volume. Compliance changes are therefore interpreted together with changes in geometric volume, measured from the projected depth boundaries.

The optimizer retains the lowest-compliance projected depth state encountered, including the initial state. Reported compliance and volume refer to this selected state, which is saved and reanalyzed using the documented checks [See Supplemental Material]. Export meshes undergo separate smoothing and interface restoration and are not reanalyzed by FEA.

## 3.7 Evaluation Protocol

Dataset We used the SimJEB engine bracket dataset [19]. A fixed cohort of 264 of the 381 public designs shared the study’s mounting interface layout. Reconstruction costs were evaluated on 230 original–reconstruction pairs matched by shape intersection-over-union (IoU), with a separate 150-pair identifier-verified check. Cohort selection, matching criteria, and the limits of the historical records are detailed [See Supplemental Material]. The diffusion experiments used 211 training, 26 validation, and 27 held-out designs. The held-out set was not used in the generative comparison. The conversion itself accepts arbitrary surface meshes, but all reported experiments concern this bracket family and the selected coordinate convention.

Table 1: Maximum displacement changes after reconstruction for 230 shape-matched pairs. Values are median [interquartile range] in percent.
<table><tr><td>Load case Maximum displacement (%)</td></tr><tr><td>Vertical -16.8 [-25.5, -11.5]</td></tr><tr><td>Horizontal -18.3 [−26.9, -12.4]</td></tr><tr><td>Diagonal -13.6 [−22.7, -7.9]</td></tr><tr><td>Torsion -24.4 [−28.6, -19.8]</td></tr></table>

Representation Evaluation The representation analysis used the 230 shape-matched pairs described above. Its primary mass estimate used material volume on the common 2.5-mm voxel grid; exact mesh volume supplied a discretization cross-check. Structural response was evaluated using C3D8 voxel models in CalculiX under the four published SimJEB load cases and Ti–6Al–4V material properties. On an independent set of 1,114 DeepJEB brackets, the same voxel scorer reproduced commercial-solver stiffness rankings with Spearman correlations of 0.88–0.98 across load cases [69, 70]. This comparison supports ranking use but is not a conforming-mesh validation of every SimJEB reconstruction. Exact material constants, load vectors, boundary conditions, and interface coupling are provided [See Supplemental Material].

Generative Model Evaluation Generation was evaluated across three diffusion seeds, with 256 samples per run, using raw depth validity, reconstructed component counts, and balanced depth distances to the training set and other generated samples. Depth validity was evaluated before the reconstruction oracle applied its range mapping. Metric definitions and the exploratory comparison are provided [See Supplemental Material], including a major-volume fragmentation threshold defined after visual inspection.

Shape Optimization Study The shape optimization study used 200 candidates generated by latent-space diffusion. Generated depths were converted to boundary coordinates and clipped to the design envelope before the minimum thickness and interface projections used for FEA. These prepared states differ from the raw outputs assessed in the generative model evaluation. A single fixed configuration was used for every candidate, with no per-candidate tuning: identical control lattice size, minimum thickness, iteration count, learning rate schedule, and penalty weights. Results include all 200 candidates, including those that retained their initial projected states.

OpenAI Codex (OpenAI, desktop application, September 2026) was used for language editing. The authors reviewed and approved all revisions.

## 4 Results

## 4.1 Reconstruction and Structural Cost

Two-depth reconstruction retains the exposed top and bottom surfaces while filling occluded intervals along the selected pull direction (Fig. 6). The reconstructed brackets therefore preserve their visible outer features but lose internal hollows and recesses that cannot be represented by the two boundaries.

Across the 230 shape-matched pairs, material volume on the common voxel grid increased by a median of 7.8% (interquartile range 4.8–11.3%). Under the four SimJEB load cases, median maximum displacement decreased by 13.6–24.4% relative to the original designs (Table 1). The interquartile range of displacement changes was entirely negative in every load case. These results indicate that reconstruction generally increased stiffness while adding material. Complementary mesh-volume estimates, an identifier-verified subset comparison, and a voxel-resolution check are provided [See Supplemental Material].

The resulting mass–stiffness trade-off reflects the filling of occluded regions required by the representation. For this bracket family, the selected pull-direction restriction was achieved with a median material increase of approximately 8%. The accompanying displacement reductions describe the response of the reconstructed designs under the evaluated loads, rather than an improvement in stiffness at equal mass.

![](images/f7a4c70c1f739ced474789ce3eb5f7f10f60e5bc13b2172fee5d837cf00cb024.jpg)  
Figure 6: Original brackets and their two-depth reconstructions in front, back, side, and isometric views.

## 4.2 Generated Designs

The generated depth pairs reconstruct into three-dimensional bracket candidates with different connecting arms, openings, and rib patterns (Fig. 7). These examples retain recognizable features of the bracket family while varying the material distribution between the mounting regions. The paired images describe the shape’s outline and its upper and lower surfaces, allowing a two-dimensional generative model to produce three-dimensional design candidates.

The generated depth boundaries also provide the starting variables for structural refinement, linking shape generation to the FEA-guided optimization examined next. Figure 7 shows pixel-space diffusion examples, whereas the optimization study uses candidates from latent-space diffusion. A separate comparison of their depth validity, connectivity, and sample distances is provided [See Supplemental Material].

## 4.3 FEA-Guided Shape Optimization of Generated Designs

FEA-guided refinement reduced compliance in 193 of 200 generated candidates. Across all 200 candidates, the median reduction was 36.8% (interquartile range 23.7–50.8%, range 0–89%), as summarized in Fig. 8. The remaining seven candidates retained their initial designs. Figure 9 illustrates the associated shape changes. Subgroup summaries are reported [See Supplemental Material].

Compliance fluctuated during refinement, while the running minimum tracked the lowest compliance reached (Fig. 10a).

![](images/f828d1b5b5818d9c372527704613980d4ab1fc2f14d76e5de76ad46bb154574b.jpg)  
Figure 7: Examples generated by pixel-space diffusion. (a) Paired top and bottom depth fields. (b) Corresponding three-dimensional reconstructions in multiple views. Matching numbers identify the same sample in all views.

![](images/ef53783aed021124bfa0b9545aa1492dd64bd74cefa1ff214745aebc47d8316c.jpg)  
Figure 8: Compliance reduction across 200 generated candidates. Bars show the interquartile range, vertical ticks the median, and diamonds the mean. The thin line for all candidates shows the full observed range.

To examine the volume–compliance trade-off, we varied the volume target on three additional designs (Cases 4–6) from 0.7 to 1.15 times the reference volume fraction $V _ { f , 0 }$ (Fig. 10b). Higher volume targets yielded lower or unchanged saved compliance for each design. At factor 0.7, all three runs retained the initial design. This sweep is an illustrative three-design probe of the trade-off rather than a full frontier estimate.

Removing the FEA gradient while retaining the volume and displacement penalties produced no compliance improvement in Cases 1–3 (Fig. 11a). With FEA guidance and one setting varied at a time, compliance decreased at each tested control lattice resolution $( n _ { c } \in \{ 1 2 , 2 0 , 2 8 \}$ : mean 44%, 36%, 36%) and volume penalty weight $( w _ { V } \in \{ 1 . 5 , 3 , 6 \}$ mean 36%, 36%, 37%). The coarsest lattice gave the largest mean reduction in these three cases. Separate optimizations under y- and z-direction loads also reduced compliance, by means of 33% and 41%, respectively.

![](images/a4eca18d0b1ee32b4ebc7771f0b1c6eb36987d7c6f0672dbd911edd969e95ffb.jpg)  
Figure 9: Shape refinement for Cases 1–3. Before shows reconstructions of generated depth fields clipped to the design envelope. After shows the saved optimized designs. Overlays show material added or removed during input preparation and refinement. All surfaces are smoothed for visualization.

## 5 Discussion

## 5.1 Two-Depth Fields as Shared Design Variables

Paired depth fields provide a common design space for shape generation and structural refinement. Equation (1) defines a restricted geometric class that can be generated as a two-channel image, reconstructed as a solid, and refined by moving the same upper and lower boundaries in Eq. (2). Valid boundary ordering preserves the selected pull-direction restriction throughout these stages, while the quality and variety of candidate shapes depend on the generative model.

Generation and refinement exploit different degrees of freedom within this shared representation. Generation proposes candidates with different footprints and connecting layouts, whereas FEA-guided refinement adjusts the upper and lower boundaries within a fixed footprint. The initial candidate therefore determines the in-plane layout available to subsequent refinement. For the designer, this separates the exploration of alternative layouts from structural improvement through changes in their thickness distribution.

Compliance reductions in 193 of 200 candidates show that a single refinement configuration can improve a range of generated shapes without per-candidate tuning. In the three ablation cases, removing the FEA gradient eliminated the improvement, supporting the contribution of structural sensitivities to refinement. Because the volume penalty does not enforce constant volume, the reported reductions reflect changes in both geometry and material volume and do not establish improved performance at equal mass.

![](images/8cf50760a1086ef2f96d52814a5b57910220b017f5cf08149d35bb02a4bd578b.jpg)

![](images/9d05bf8183feaca7723122247e3daee18d009f2b8e12c3a5df33f74151f130ae.jpg)  
Figure 10: Optimization behavior of three additional candidates (Cases 4–6). (a) Compliance histories (thin) and running minima (thick) at the default volume target. (b) Volume–compliance outcomes for four volume targets. Two targets retain the same initial design for Case 4.

(a) Optimizer settings and FEA guidance  
![](images/e206e6d16253d7c324cc63ba954d6dab898bc2b1b057b804595a4f62c0a56c2f.jpg)

(b) Load direction  
![](images/266c1e6056c55eb1697f788f2a4fca7647a5c3667dbc042c81b7669f944b1939.jpg)  
Figure 11: Effects of optimizer settings, FEA guidance, and load direction on compliance reduction for Cases 1–3, the same candidates as in Fig. 9. (a) Effects of control lattice resolution $n _ { c } ,$ volume penalty weight w<sub>V</sub>, and removing the FEA gradient (FEA-off). (b) Effects of load direction. Bars show means, and symbols identify individual candidates. Green bars and the dashed line mark the default configuration.

## 5.2 Manufacturing Scope

When existing designs are reconstructed in the paired-depth representation, filling occluded intervals adds material and changes their structural response. The representation therefore exposes a deliberate mass–stiffness trade-off rather than preserving all internal geometry without cost. It is best viewed as an early-stage design space for parts governed by a selected pull direction, with later process checks still required. In use, the designer first selects the pull direction and decides whether occluded voids may be filled.

The selected pull-direction restriction addresses geometric interference during mold separation. Positive draft remains a separate requirement. The illustrative projection provided in the Supplemental Material was not applied in the reported experiments. Wall and rib rules, dimensional tolerances, mold filling and solidification, and structural durability require further process-specific assessment. Features perpendicular to the pull direction, such as a horizontal mounting bore, require secondary machining, a side action, or another manufacturing provision and are not counted as evidence of two-part demolding.

## 5.3 Limitations

The study uses fewer than 300 filtered designs from one bracket family. The generative evaluation includes three diffusion seeds, with the latent runs sharing one autoencoder. Together with the observed clustering near training shapes, this limits claims about broad generative diversity. Whether the representation reduces training data requirements relative to volumetric generation remains to be established through comparisons across training set sizes.

Structural evaluation of the reconstructions uses voxel-based finite element models. Reported compliance improvements refer to the smoothed-density FEA model used for optimization [See Supplemental Material]. The corresponding sharp boundary solids were not independently reanalyzed. The one-factor comparisons cover three candidates and separate load directions rather than simultaneous loading. Comparisons with classical topology optimization or alternative initialization methods remain outside this study.

The in-plane grid limits the smallest resolved feature. Independently of resolution, the single-interval representation cannot preserve internal cavities or features occluded along the selected pull direction.

## 6 Conclusion

CAST reformulates three-dimensional shape generation under a selected pull-direction constraint as a two-dimensional image generation problem using paired top and bottom depth fields. Valid depth ordering restricts each in-plane column to a single material interval and excludes undercuts along the selected pull direction.

Two-dimensional generative models proposed bracket geometries, and FEA-guided updates refined the same upper and lower boundaries while retaining the selected pull-direction restriction. Using one fixed configuration, refinement reduced compliance in 193 of 200 generated candidates, with a median reduction of 36.8% in the optimization FEA model under a volume penalty. Reconstruction of existing SimJEB brackets also exposed a mass–stiffness trade-off: filling occluded intervals added material and was associated with lower displacement across the evaluated load cases

The geometric guarantee is limited to undercut exclusion along the selected pull direction. Broader manufacturing feasibility requires additional process-specific assessment. CAST thus demonstrates how a manufacturing-oriented representation can simplify generative modeling while retaining a common design space for physics-guided structural refinement.

## Acknowledgment

This work was supported by grants from the Ministry of Science and ICT (GTL24033-000, N10250154, and No. 2022- 0-00986), the Ministry of Trade, Industry and Energy (RS-2025-02317327 and RS-2025-25444634), the Ministry of Oceans and Fisheries (PET0050), and Korea Hydro & Nuclear Power Co., Ltd. (No. 8-Tech-07).

## Conflict of Interest

The authors declare no conflicts of interest.

## Data Availability Statement

The SimJEB dataset is publicly available from its original authors. Data supporting the findings of this study are available from the corresponding author upon request.

## References

[1] Lyle Regenwetter, Amin Heyrani Nobari, and Faez Ahmed. Deep Generative Models in Engineering Design: A Review. Journal of Mechanical Design, 144(7):071704, 03 2022. ISSN 1050-0472. doi:10.1115/1.4053859.

[2] Sangeun Oh, Yongsu Jung, Seongsin Kim, Ikjin Lee, and Namwoo Kang. Deep Generative Design: Integration of Topology Optimization and Generative Models. Journal ofMechanical Design, 141(11):111405, 09 2019. ISSN 1050-0472. doi:10.1115/1.4044229.

[3] Wei Chen and Faez Ahmed. PaDGAN: Learning to Generate High-Quality Novel Designs. Journal ofMechanical Design, 143(3):031703, 11 2020. ISSN 1050-0472. doi:10.1115/1.4048626.

[4] Dule Shu, James Cunningham, Gary Stump, Simon W. Miller, Michael A. Yukish, Timothy W. Simpson, and Conrad S. Tucker. 3D Design Using Generative Adversarial Networks and Physics-Based Validation. Journal of Mechanical Design, 142(7):071701, 2020. ISSN 1050-0472. doi:10.1115/1.4045419.

[5] Xingang Li, Charles Xie, and Zhenghui Sha. A Predictive and Generative Design Approach for Three-Dimensional Mesh Shapes Using Target-Embedding Variational Autoencoder. Journal ofMechanical Design, 144(11):114501, 2022. ISSN 1050-0472. doi:10.1115/1.4054906.

[6] Ethan Herron, Jaydeep Rade, Anushrut Jignasu, Baskar Ganapathysubramanian, Aditya Balu, Soumik Sarkar, and Adarsh Krishnamurthy. Latent Diffusion Models for Structural Component Design. Computer-Aided Design, 171: 103707, 2024. ISSN 0010-4485. doi:10.1016/j.cad.2024.103707.

[7] Jiangce Chen, Martha Baldwin, Sneha Prabha Narra, and Christopher McComb. Multi-Lattice Topology Optimization Via Generative Lattice Modeling. Journal ofMechanical Design, 147(5):051707, 2025. ISSN 1050-0472. doi:10.1115/1.4067528.

[8] Martin Leary. Design for additive manufacturing. Elsevier, 2019.

[9] Ian Gibson, David Rosen, Brent Stucker, and Mahyar Khorasani. Designfor Additive Manufacturing. Springer International Publishing, Cham, 2021. ISBN 978-3-030-56127-7. doi:10.1007/978-3-030-56127-7-19.

[10] P. Dewhurst and C. Blum. Supporting analyses for the economic assessment of diecasting in product design. CIRP Annals, 38(1):161–164, 1989. doi:10.1016/S0007-8506(07)62675-4.

[11] Ruchi Karania and David Kazmer. Low volume plastics manufacturing strategies. Journal of Mechanical Design, 129(12):1225–1233, 2007. doi:10.1115/1.2790978.

[12] Claudio Favi, Michele Germani, and Marco Mandolini. Analytical cost estimation model in high pressure die casting. Procedia Manufacturing, 11:526–535, 2017. doi:10.1016/j.promfg.2017.07.146.

[13] B. Ravi and M. N. Srinivasan. Decision Criteria for Computer-Aided Parting Surface Design. Computer-Aided Design, 22(1):11–18, 1990. ISSN 0010-4485. doi:10.1016/0010-4485(90)90024-7.

[14] M. Weinstein and S. Manoochehri. Geometric Influence of a Molded Part on the Draw Direction Range and Parting Line Locations. Journal ofMechanical Design, 118(1):29–39, 1996. ISSN 1050-0472. doi:10.1115/1.2826853.

[15] T. Alderighi, L. Malomo, T. Auzinger, B. Bickel, P. Cignoni, and N. Pietroni. State of the art in computational mould design. Computer Graphics Forum, 41(6):435–452, 2022. doi:10.1111/cgf.14581.

[16] Zhichao Wang, Xiaoliang Yan, Shreyes Melkote, and David Rosen. McGAN: Generating manufacturable designs by embedding manufacturing rules into conditional generative adversarial network. arXiv preprint arXiv:2407.16943, 2024.

[17] Amir Arsalan Soltani, Haibin Huang, Jiajun Wu, Tejas D. Kulkarni, and Joshua B. Tenenbaum. Synthesizing 3D shapes via modeling multi-view depth maps and silhouettes with deep generative networks. In 2017 IEEE Confer ence on Computer Vision and Pattern Recognition (CVPR), pages 2511–2519, 2017. doi:10.1109/CVPR.2017.269.

[18] Peter W. Battaglia, Jessica B. Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, Caglar Gulcehre, Francis Song, Andrew Ballard, Justin Gilmer, George Dahl, Ashish Vaswani, Kelsey Allen, Charles Nash, Victoria Langston, Chris Dyer, Nicolas Heess, Daan Wierstra, Pushmeet Kohli, Matt Botvinick, Oriol Vinyals, Yujia Li, and Razvan Pascanu. Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261, 2018. URL https://arxiv.org/abs/1806.01261.

[19] Eamon Whalen, Azariah Beyene, and Caitlin Mueller. Simjeb: simulated jet engine bracket dataset. In Computer Graphics Forum, volume 40, pages 9–17. Wiley Online Library, 2021.

[20] Sivam Krish. A practical generative design method. Computer-Aided Design, 43(1):88–100, 2011. ISSN 0010-4485. doi:10.1016/j.cad.2010.09.009.

[21] Gábor Renner and Anikó Ekárt. Genetic algorithms in computer aided design. Computer-Aided Design, 35(8): 709–726, 2003. ISSN 0010-4485. doi:10.1016/S0010-4485(03)00003-4. Genetic Algorithms.

[22] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

[23] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

[24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 6840–6851. Curran Associates, Inc., 2020.

[25] Tinghao Guo, Danny J Lohan, Ruijin Cang, Max Yi Ren, and James T Allison. An indirect design representation for topology optimization using variational autoencoder and style transfer. In 2018 AIAA/ASCE/AHS/ASC structures, structural dynamics, and materials conference, page 0804, 2018.

[26] Zhenguo Nie, Tong Lin, Haoliang Jiang, and Levent Burak Kara. TopologyGAN: Topology Optimization Using Generative Adversarial Networks Based on Physical Fields Over the Initial Domain. Journal of Mechanical Design, 143(3):031715, 02 2021. ISSN 1050-0472. doi:10.1115/1.4049533.

[27] François Mazé and Faez Ahmed. Diffusion models beat gans on topology optimization. Proceedings of the AAAI Conference on Artificial Intelligence, 37(8):9108–9116, Jun. 2023. doi:10.1609/aaai.v37i8.26093.

[28] Giorgio Giannone, Akash Srivastava, Ole Winther, and Faez Ahmed. Aligning optimization trajectories with diffusion models for constrained design generation. In A. Oh, T. Neumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 51830–51861. Curran Associates, Inc., 2023.

[29] Ogulcan Eren, Nurullah Yüksel, Hüseyin Rıza Börklü, Hüseyin Kür¸sad Sezer, and Olcay Ersel Canyurt.˘ Deep learning-enabled design for tailored mechanical properties of slm-manufactured metallic lattice structures. Engineering Applications of Artificial Intelligence, 130:107685, 2024. ISSN 0952-1976. doi:10.1016/j.engappai.2023.107685.

[30] Wonsuk Kim, Soojeong Kim, Minhyeok Lee, and Junhee Seok. Inverse design of nanophotonic devices using generative adversarial networks. Engineering Applications ofArtificial Intelligence, 115:105259, 2022. ISSN 0952-1976. doi:10.1016/j.engappai.2022.105259.

[31] Amin Heyrani Nobari, Wei Chen, and Faez Ahmed. Range-Constrained Generative Adversarial Network: Design synthesis under constraints using conditional generative adversarial networks. Journal of Mechanical Design, 144 (2):021708, 2022. doi:10.1115/1.4052442.

[32] Mohammad Mahdi Behzadi and Horea T. Ilie¸s. GANTL: Toward practical and real-time topology optimization with conditional generative adversarial networks and transfer learning. Journal of Mechanical Design, 144(2): 021711, 2022. doi:10.1115/1.4052757.

[33] Waad Almasri, Dimitri Bettebghor, Faouzi Adjed, Florence Danglade, and Fakhreddine Ababsa. Geometricallydriven generation of mechanical designs through deep convolutional GANs. Engineering Optimization, 56(1): 18–35, 2024. doi:10.1080/0305215X.2022.2144847.

[34] Yongmin Kwon and Namwoo Kang. Three-dimensional deep shape optimization with a limited dataset. Engineering Applications ofArtificial Intelligence, 159:111504, 2025.

[35] Jaydeep Rade, Anushrut Jignasu, Ethan Herron, Ashton Corpuz, Baskar Ganapathysubramanian, Soumik Sarkar, Aditya Balu, and Adarsh Krishnamurthy. Deep learning-based 3d multigrid topology optimization of manufacturable designs. Engineering Applications ofArtificial Intelligence, 126:107033, 2023. ISSN 0952-1976. doi:10.1016/j.engappai.2023.107033.

[36] Wei Chen and Mark Fuge. Synthesizing Designs With Interpart Dependencies Using Hierarchical Generative Adversarial Networks. Journal of Mechanical Design, 141(11):111403, 09 2019. ISSN 1050-0472. doi:10.1115/1.4044076.

[37] Geoffrey Boothroyd, Peter Dewhurst, and Winston A Knight. Product designfor manufacture and assembly. CRC press, 2010.

[38] Tsai-C. Kuo, Samuel H. Huang, and Hong-C. Zhang. Design for manufacture and design for ‘x’: concepts, applications, and perspectives. Computers & Industrial Engineering, 41(3):241–260, 2001. ISSN 0360-8352. doi:10.1016/S0360-8352(01)00045-6.

[39] Giovanni Formentini, Núria Boix Rodríguez, and Claudio Favi. Design for manufacturing and assembly methods in the product development process of mechanical products: a systematic literature review. The International Journal of Advanced Manufacturing Technology, 120(7):4307–4334, 2022.

[40] Albert E. Patterson, Yong Hoon Lee, and James T. Allison. Generation and enforcement of process-driven manufacturability constraints: A survey of methods and perspectives for product design. Journal ofMechanical Design, 143(11):110801, 2021. doi:10.1115/1.4050740.

[41] Geoffrey Boothroyd. Product design for manufacture and assembly. Computer-Aided Design, 26(7):505–520, 1994.

[42] Allan Roulund Gersborg and Casper Schousboe Andreasen. An explicit parameterization for casting constraints in gradient driven topology optimization. Structural and Multidisciplinary Optimization, 44(6):875–881, 2011. doi:10.1007/s00158-011-0632-0.

[43] Shutian Liu, Quhao Li, Wenjiong Chen, Rui Hu, and Liyong Tong. H-DGTP—a Heaviside-function based directional growth topology parameterization for design optimization of stiffener layout and height of thin-walled structures. Structural and Multidisciplinary Optimization, 52(5):903–913, 2015. doi:10.1007/s00158-015-1281-5.

[44] Sandro L. Vatanabe, Tiago N. Lippi, Cícero R. de Lima, Glaucio H. Paulino, and Emílio C. N. Silva. Topology optimization with manufacturing constraints: A unified projection-based approach. Advances in Engineering Software, 100:97–112, 2016. doi:10.1016/j.advengsoft.2016.07.002.

[45] Gil Ho Yoon and Seon Il Ha. A new development of a shadow density filter for manufacturing constraint and its applications to multiphysics topology optimization. Journal of Mechanical Design, 143(6):061703, 2021. doi:10.1115/1.4048818.

[46] Xiaoping Qian. Undercut and overhang angle control in topology optimization: A density gradient based integral approach. International Journal for Numerical Methods in Engineering, 111(3):247–272, 2017. doi:10.1002/nme.5461.

[47] Hao Zhou, Junyuan Zhang, Yuqing Zhou, and Kazuhiro Saitou. Multi-component topology optimization for die casting (MTO-D). Structural and Multidisciplinary Optimization, 60(6):2265–2279, 2019. doi:10.1007/s00158- 019-02317-4.

[48] Heting Fu and Kazuhiro Saitou. Topology optimization for die casting with nonplanar parting surfaces and side dies. Journal ofMechanical Design, 148(11):111703, 2026. doi:10.1115/1.4071713.

[49] Tomás Zegard and Glaucio H Paulino. Bridging topology optimization and additive manufacturing. Structural and Multidisciplinary Optimization, 53:175–192, 2016.

[50] Matthijs Langelaar. Topology optimization of 3d self-supporting structures for additive manufacturing. Additive manufacturing, 12:60–70, 2016.

[51] Nathan Hertlein, Philip R. Buskohl, Andrew Gillman, Kumar Vemaganti, and Sam Anand. Generative adversarial network for early-stage design flexibility in topology optimization for additive manufacturing. Journal of Manufacturing Systems, 59:675–685, 2021. ISSN 0278-6125. doi:10.1016/j.jmsy.2021.04.007. URL https: //www.sciencedirect.com/science/article/pii/S027861252100087X.

[52] Hongrui Chen, Aditya Joglekar, Kate S. Whitefoot, and Levent Burak Kara. Concurrent Build Direction, Part Segmentation, and Topology Optimization for Additive Manufacturing Using Neural Networks. Journal of Mechanical Design, 145(9):091702, 06 2023. ISSN 1050-0472. doi:10.1115/1.4062663.

[53] Michael Greminger. Generative adversarial networks with synthetic training data for enforcing manufacturing constraints on topology optimization. In ASME 2020 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference, volume 11A: 46th Design Automation Conference, 2020. doi:10.1115/DETC2020-22399.

[54] Christoph Wittig Adão, Jiahang Li, Markus Döllken, Maxim Möllhoff, Xingang Li, and Sven Matthiesen. SheetGen: A rule-based generative design algorithm for manufacturability-focused sheet metal parts based on domain-specific design knowledge. Journal ofMechanical Design, 148(12):121704, 2026. doi:10.1115/1.4072234.

[55] Jiewu Leng, Xiaoli Su, Zean Liu, Lianhong Zhou, Chong Chen, Xin Guo, Yiwei Wang, Ru Wang, Chao Zhang, Qiang Liu, Xin Chen, Weiming Shen, and Lihui Wang. Diffusion model-driven smart design and manufacturing: Prospects and challenges. Journal ofManufacturing Systems, 82:561–577, 2025. doi:10.1016/j.jmsy.2025.07.011.

[56] Richard Szeliski. Computer vision: algorithms and applications. Springer Nature, 2022.

[57] Daniel Scharstein and Richard Szeliski. A taxonomy and evaluation of dense two-frame stereo correspondence algorithms. International journal ofcomputer vision, 47:7–42, 2002.

[58] Jason Geng. Structured-light 3d surface imaging: a tutorial. Adv. Opt. Photon., 3(2):128–160, Jun 2011. doi:10.1364/AOP.3.000128.

[59] Jungong Han, Ling Shao, Dong Xu, and Jamie Shotton. Enhanced computer vision with microsoft kinect sensor: A review. IEEE Transactions on Cybernetics, 43(5):1318–1334, 2013. doi:10.1109/TCYB.2013.2265378.

[60] Jonathan Shade, Steven Gortler, Li-wei He, and Richard Szeliski. Layered depth images. In Proceedings of the 25th Annual Conference on Computer Graphics and Interactive Techniques, pages 231–242, 1998. doi:10.1145/280814.280882.

[61] Charlie C. L. Wang, Yuen-Shan Leung, and Yong Chen. Solid modeling of polyhedral objects by layered depth-normal images on the GPU. Computer-Aided Design, 42(6):535–544, 2010. doi:10.1016/j.cad.2010.02.001.

[62] Philipp Herholz, Wojciech Matusik, and Marc Alexa. Approximating free-form geometry with height fields for manufacturing. Computer Graphics Forum, 34(2):239–251, 2015. doi:10.1111/cgf.12556.

[63] Zhen Wang, Qiangeng Xu, Feitong Tan, Menglei Chai, Shichen Liu, Rohit Pandey, Sean Fanello, Achuta Kadambi, and Yinda Zhang. MVDD: Multi-view depth diffusion models. In Computer Vision – ECCV 2024, pages 236–253, 2024. doi:10.1007/978-3-031-72624-8\_14.

[64] Yonggyun Yu, Taeil Hur, Jaeho Jung, and In Gwun Jang. Deep learning for determining a near-optimal topological design without any iteration. Structural and Multidisciplinary Optimization, 59(3):787–799, 2019.

[65] Zhichao Wang, Shreyes Melkote, and David W. Rosen. Generative design by embedding topology optimization into conditional generative adversarial network. Journal of Mechanical Design, 145(11):111702, 2023. doi:10.1115/1.4062980.

[66] Thomas W. Sederberg and Scott R. Parry. Free-form deformation of solid geometric models. In Proceedings of the 13th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH ’86, pages 151–160. Association for Computing Machinery, 1986. doi:10.1145/15886.15903.

[67] Yongmin Kwon and Namwoo Kang. Three-dimensional geometric augmentation guided by engineering uncertainty. Engineering Applications of Artificial Intelligence, 175:114648, 2026.

[68] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

[69] Seong-Jun Hong, Yongmin Kwon, Dongju Shin, Jangseop Park, and Namwoo Kang. DeepJEB: 3D deep learning-based synthetic jet engine bracket dataset. Journal ofMechanical Design, 2025. doi:10.1115/1.4067089.

[70] Guido Dhondt. The Finite Element Method for Three-Dimensional Thermomechanical Applications. John Wiley & Sons, 2004.

# SUPPLEMENTAL MATERIAL FOR CAST: CAST-ADMISSIBLE GENERATIVE DESIGN AND SHAPE OPTIMIZATION WITH PAIRED DEPTH IMAGES

Jihoon Kim<sup>1,3,†</sup> Yongmin Kwon<sup>2,3,†</sup> Namwoo Kang<sup>1,3,∗</sup>

<sup>1</sup> Department of Mechanical Engineering,

Korea Advanced Institute of Science and Technology, Daejeon 34141, Republic of Korea <sup>2</sup> Cho Chun Shik Graduate School of Mobility,

Korea Advanced Institute of Science and Technology, Daejeon 34051, Republic of Korea <sup>3</sup> Narnia Labs, Daejeon 34051, Republic of Korea jihoon.kim@kaist.ac.kr kymin1002@kaist.ac.kr nwkang@kaist.ac.kr <sup>†</sup> Jihoon Kim and Yongmin Kwon contributed equally to this work. Corresponding author: Namwoo Kang

## ABSTRACT

This Supplemental Material provides algorithms, data processing details, model settings, and additional analyses for the paired-depth representation and FEA-guided shape optimization. Additional analyses compare pixel-space and latent-space diffusion and illustrate a positive draft projection.

Keywords deep generative design, design for manufacturing, die casting, shape representation

## S1 Depth Field Construction and Reconstruction

## S1.1 Orthographic Depth Calculation

The top and bottom channels store ray distances rather than global z coordinates. The foreground mask is derived from these channels. Input CAD coordinates in millimeters were mapped by $\widetilde { \mathbf { x } } = s ( \mathbf { x } - \mathbf { c } )$ , with $\mathbf { \dot { c } } = ( 1 7 . 7 6 , - 7 3 . 2 7 , 3 2 . 0 6 )$ mm and $s = 9 . 6 \dot { 3 } \dot { 5 } \times 1 0 ^ { - 3 } \mathrm { { m m } ^ { - 1 } }$ computed once over the retained 264 designs. Algorithm S1 makes the viewing origins and directions explicit. A missed ray receives 0.8. After range mapping, $m _ { i j } = 1$ only when both depth are below 0.78. A paired foreground column has valid ordering when $D _ { \mathrm { t o p } } + \mathsf { \bar { D } } _ { \mathrm { b o t t o m } } \leq 0 . 8$ . Its boundaries are $z ^ { + } = 0 . 4 - D _ { \mathrm { t o p } } \mathrm { a } \mathrm { \ ' n d } z ^ { - } = - \bar { 0 } . 4 + D _ { \mathrm { b o t t o m } } .$

Algorithm S1 Opposing orthographic depth calculation   
1: Input: registered surface mesh $S ; 2 5 6 \times 2 5 6$ in-plane grid   
2: Output: $\breve { D } _ { \mathrm { t o p } } , D _ { \mathrm { b o t t o m } } \in \mathbb { R } ^ { 2 5 6 \times 2 5 6 }$   
3: for each grid index $( i , j )$ do   
4: $( x _ { i } , \bar { y _ { j } } ) \gets ( 2 i / 2 5 \delta - 1 , 2 j / 2 5 6 - 1 )$   
5: $R _ { t } \gets ( ( x _ { i } , y _ { j } , 0 . 4 ) , ( 0 , 0 , - 1 ) )$   
6: $R _ { b }  ( ( x _ { i } , \bar { y _ { j } } , - 0 . 4 ) , ( 0 , 0 , 1 ) )$   
7: $D _ { \mathrm { t o p } } [ i , \dot { j } ] $ nearest nonnegative hit distance of $R _ { t }$ with $S ,$ or 0.8 if no hit   
8: $D _ { \mathrm { b o t t o m } } [ i , j ]$ ← nearest nonnegative hit distance of $R _ { b }$ with S, or 0.8 if no hit   
9: end for   
10: return $D _ { \mathrm { t o p } } , D _ { \mathrm { b o t t o m } }$

```latex
Algorithm S2 Depth-to-mesh reconstruction
1: Input: $D _ { t } , D _ { b } \in \mathbb { R } ^ { 2 5 6 \times 2 5 6 }$
2: Replace values $D \geq 0 . 7 8$ by 0.8 and values $D < 0$ by 0
3: $\bar { N _ { z } }  1 2 8 ; \delta  0 . 8 / N _ { z } ;$ initialize $S \gets - 0 . 0 1$ with shape $2 5 6 \times 2 5 6 \times N _ { z }$
4: for each pixel $( i , j )$ do
5: $( q _ { b } , \dot { r _ { b } } ) \gets \ddot { \mathrm { d i v m o d } } ( D _ { b } [ i , j ] , \delta ) ; q _ { b } \gets \operatorname* { m i n } ( N _ { z } - 1 , q _ { b } - 1 )$
6: $( q _ { t } , r _ { t } ) \gets \mathrm { d i v m o d } ( D _ { t } [ i , j ] , \delta ) ; q _ { t } \gets \operatorname* { m i n } ( N _ { z } - 1 , q _ { t } - 1 )$
7: if $q _ { b } = N _ { z } - 1 \mathrm { o r } q _ { t } = \ ' N _ { z } - 1$ then
8: $S [ i , j , : ] \gets 0 . 0 0 1$ ▷ empty-ray column
9: continue
10: end if
11: for $\ell = 0 , \ldots , q _ { b }$ do
12: $S [ i , j , \ell ] \longleftarrow r _ { b } + q _ { b } \delta - \ell \delta$
13: end for
14: for $\ell = 0 , \ldots , q _ { t }$ do
15: $S [ i , j , N _ { z } - q _ { t } - 1 + \ell ]  r _ { t } - \delta + \ell \delta$
16: end for
17: end for
18: $M \gets \mathbf { M }$ ARCHINGCUBES(S, level = 0, spacing = (1, 1, 0.8), step = 1)
19: return M
```

## S1.2 Depth-to-Mesh Reconstruction

The depth-to-mesh implementation used for the generative experiments is summarized below. Algorithm S2 mirrors its range mapping and sub-voxel scalar field construction. Here divmod $( D , \delta )$ returns integer quotient q and remainder r. The default call uses no Laplacian smoothing. The scalar values encode the two sub-voxel surface crossings, and the zero level set gives the reconstructed boundary.

## S1.3 Interface Feature Restoration

Boolean mesh operations restore specified mounting interfaces after reconstruction. Features transverse to the selected pull direction require separate tooling or secondary machining.

## S2 Dataset Cohorts and Structural Evaluation

Preprocessing retained a fixed cohort of 264 of the 381 public SimJEB designs with the study’s interface layout. The exact retained-ID list will be released with the public artifacts, but the executable rule that excluded the other 117 meshes was not recovered. Reconstruction indices did not encode SimJEB identifiers, so candidates were matched to originals on normalized $3 2 ^ { 3 }$ voxel occupancy and then checked after alignment in the common physical frame. Eligibility required normalized $\mathrm { I o U } \geq 0 . 7 0$ and 2.5-mm world-frame $\mathrm { I o U } \geq 0 . 5 5$ . When multiple candidates claimed one original, the candidate with the highest normalized IoU was retained. The resulting one-to-one cohort covered 230 distinct originals. The other 34 registered originals had no retained match. A separate 150-pair subset matched each reconstruction to the same-identifier original and required IoU at least 0.6. The primary and identifier-verified cohorts gave similar medians: mass change +7.8% versus +7.7% and vertical-load maximum-displacement change −16.8% versus −17.0%.

Both members of each pair were evaluated on the same 2.5-mm voxel grid of eight-node hexahedral elements (C3D8) in CalculiX using Ti–6Al–4V properties $( E = 1 1 3 8 0 0 \mathrm { N / m m ^ { 2 } } , \nu = \bar { 0 } . 3 4 2 )$ . The four bolt regions were joined with \*RIGID BODY constraints and all six degrees of freedom of their reference nodes were fixed. The load interface surface was connected to its reference node using \*COUPLING with \*DISTRIBUTING(1,6), and the load was applied at that node. The same registered interface regions were used for the full cohort. The four published SimJEB interface load cases are listed in Supp. Table 1. Refining the voxel pitch from 2.5 to 1.25 mm moved the median changes summarized across the four load cases by approximately 3.6 percentage points. Exact mesh volumes gave a median material increase of 12.2%, compared with 7.8% on the common voxel grid, indicating that discretization affects the estimated magnitude but not the direction of the material change. Structural comparisons used the common-grid volume.

Supp. Table 1: SimJEB load cases used for original–reconstruction comparison. Forces are in N and the torsional moment is in Nmm.
<table><tr><td>Case</td><td>Interface load</td></tr><tr><td>Vertical</td><td> $\mathbf { F } = ( 0 , 0 , 3 5 5 8 6 )$ </td></tr><tr><td>Horizontal</td><td> $\mathbf { F } = \bigl ( - 3 7 8 1 0 , 0 , 0 \bigr )$ </td></tr><tr><td>Diagonal</td><td> $\mathbf { F } = ( - 2 8 2 7 6 , 0 , 3 1 4 0 4 )$ </td></tr><tr><td>Torsion</td><td> $\mathbf { M } = ( 0 , 0 , 5 6 4 9 2 4 )$ </td></tr></table>

Supp. Table 2: Configuration of the deterministic spatial autoencoder used by all three latent diffusion runs.  
Item Setting   
Implementation diffusers 0.36.0 AutoencoderKL, trained from scratch   
Input/output 2 × 256 × 256 paired depths; normalization $2 x / 0 . 8 - 1$   
Encoder/decoder widths (64, 128, 256); two residual layers per block   
Activation/normalization SiLU; GroupNorm with 32 groups; mid-block attention   
Latent $4 \times 6 4 \times 6 4$ posterior mean; KL weight 0   
Optimization AdamW, lr $1 \bar { 0 } ^ { - 4 } .$ , betas (0.9, 0.999), weight decay 0   
Batch/precision physical batch 8, accumulation 4, effective batch 32; BF16; gradient clip 1   
Training/selection 20,000 optimizer steps; validation every 250; lowest foreground-balanced validation loss   
Selected checkpoint seed 29, step 19,750

## S3 Generative Model Configuration

## S3.1 Pixel-Space and Latent-Space Diffusion Configurations

The fixed data split contained 211 training, 26 validation, and 27 held-out designs. The held-out set remained unopened during this comparison. Both diffusion arms used a U-Net base width of 64 and multipliers (1, 2, 4, 8), v prediction, a sigmoid schedule, 1000 training timesteps, 250-step DDIM sampling with η = 0, Adam with learning rate $1 0 ^ { - 4 }$ and betas (0.9, 0.99), gradient clipping at 1, no augmentation, 130,000 optimizer steps, and model seeds 17, 29, and 43. Each run generated 256 samples with the same list of sample-seed identities. The comparison used raw weights at step 130,000. EMA outputs were used only for diagnostics.

The arms were matched in data order, update exposure, schedule, seeds, and sample count, but they were not a pure encoding ablation. The pixel-space model operated on $2 \times 2 5 6 \times 2 5 6$ tensors with trajectory clipping enabled. The latent-space model operated on channel-wise normalized $4 \times 6 4 \times 6 4$ posterior means with trajectory clipping disabled and included the prior cost of selecting and training the autoencoder. Input resolution, memory, wall time, and total end-to-end compute therefore differed. The three latent runs shared one validation-selected autoencoder and therefore represent diffusion-seed variation rather than independent repetitions of autoencoder selection and training. No matched two-dimensional–three-dimensional comparison was conducted, so this study does not establish a computational advantage over three-dimensional generators.

## S3.2 Spatial Autoencoder

Supp. Table 2 gives the deterministic spatial autoencoder configuration. It used neither a discriminator nor a perceptual network, and its posterior mean served as the reconstruction and diffusion input. Three models were trained from scratch at seeds 17, 29, and 43. The seed-29 model was selected on the 26-design validation split without using the held-out test set.

## S3.3 Training Resource Profile

Training resources were measured on one NVIDIA A100 80GB GPU with the precision and batch settings in Supp.   
Table 3. PyTorch 2.10 FlopCounterMode counted registered forward and backward operations across all microbatches in one optimizer update, excluding optimizer steps and other training overhead. Two repeated counts agreed exactly.   
Peak allocated memory was measured over three updates after one warm-up update initialized the optimizer state.   
For pixel diffusion, the original two rank-local batches of 16 were combined into one single-process batch of 32.   
Communication was excluded, so this memory measurement is not the original per-GPU distributed-training peak.

Supp. Table 3: Single-A100 training resource measurements. Batch/update lists physical batch size multiplied by gradient accumulation.
<table><tr><td>Component</td><td>Tensor</td><td>Precision</td><td>Batch/update</td><td>Params [M]</td><td>TFLOP/update</td><td>Peak [GiB]</td></tr><tr><td>Pixel diffusion</td><td> $2 \times 2 5 6 \times 2 5 6$ </td><td>FP32</td><td> $3 2 \times 2 = 6 4$ </td><td>35.716</td><td>44.976</td><td>58.466</td></tr><tr><td>Latent diffusion</td><td> $4 \times 6 4 \times 6 4$ </td><td>FP32</td><td> $3 2 \times 2 = 6 4$ </td><td>35.723</td><td>2.800</td><td>4.769</td></tr><tr><td>Autoencoder</td><td> $2 \times 2 5 6 \times 2 5 6$ </td><td>BF16</td><td> $8 \times 4 = 3 2$ </td><td>13.858</td><td>26.597</td><td>8.128</td></tr></table>

Including 20,000 autoencoder updates and 130,000 diffusion updates, the latent pipeline accumulated 0.896 EFLOP versus 5.847 EFLOP for pixel diffusion (15.3%). These are profiler-attributed operations, combining BF16 autoencoder and FP32 diffusion counts, rather than hardware FLOPs or wall time. The autoencoder and diffusion model train sequentially, so their peak memory values are reported separately.

## S4 FEA-Guided B-Spline Shape Optimization

The settings below supplement the boundary update equations in the main article. Each boundary uses a $2 0 \times 2 0$ clamped-uniform cubic B-spline control lattice.

## S4.1 Geometry Projections and Interfaces

For optimization, generated depths are converted to boundary coordinates with their range clipped to the normalized span. Columns with converted thickness below 0.01 are collapsed to zero thickness, and the boundaries are clipped to the design envelope. Minimum thickness and interface projections then define the initial FEA state. Raw-depth metrics in Sec. S5.1 are evaluated before this preparation and concern the separate generative comparison. Before every FEA call, the candidate field is clipped to the fixed envelope. Initial material columns with envelope-clipped thickness above 0.006 are retained, a minimum thickness of 0.016 is imposed, and fixed and loaded interface columns are set to the envelope. The geometry-protection mask is the z projection of fixed.stl and load.stl occupancy voxelized on a $2 5 6 \times 2 5 6 \times 4 0$ grid. This voxelizer samples in-plane cell centers at $( 2 ( i + 0 . 5 ) / 2 5 6 - 1 , 2 ( j + 0 . 5 ) / 2 5 6 - 1 )$ , unlike the depth calculation ray grid. The undilated mask restores the interface columns, while an eight-pixel in-plane dilation selects fixed spline controls. The fixed configuration uses no separate background mask, 60 updates, initial learning rate 0.02 with linear decay to 0.004, normalized volume update weight 3.0, and control displacement weight 0.3.

## S4.2 FEA and Sensitivity Transfer

The differentiable density uses 40 samples through the normalized z span and sigmoid sharpness 48. The two-sigmoid product is not multiplied by a foreground mask. When $z ^ { + } = z ^ { - } = a .$ , its continuous value at z = a is $1 / 4 ,$ although the geometric volume of that column is zero. The influence of this residual density on compliance has not been quantified by a separate sharp-boundary analysis. The fixed transform maps this density grid to a physical-meter tetrahedral envelope mesh with 0.006-m characteristic size. The material model uses $E _ { 0 } = 1 , \nu = 0 . 3 ,$ , SIMP exponent 3, and $E _ { \mathrm { m i n } } \dot { = } 1 0 ^ { - 6 }$ . With the default vertex-distance setting, mesh nodes whose nearest fixed.stl vertex lies within 0.006 m receive Dirichlet constraints. A P1 nodal indicator over mesh nodes within 0.006 m of the nearest load.stl vertex is integrated over the domain and scaled to a total load magnitude of 42,300 analysis units in direction $( 1 , - 1 , - 1 ) / \sqrt { 3 }$ These solver regions are distinct from the voxel-projected geometry masks. The conjugate gradient solve uses algebraic multigrid, relative tolerance $1 0 ^ { - 9 }$ , and at most 3000 iterations.

The forward transfer lets each FEA node select its nearest external density sample and assigns each tetrahedron the mean of its four nodal densities. Its backward pass is the exact transpose of that linear transfer. The FEA density sensitivity and linear density transfer were checked with centered finite differences along five random directions at each of three base densities (0.3, 0.5, and 0.7). The maximum relative error was below $1 0 ^ { - 4 }$ at every base density. Hard envelope, interface, and minimum thickness projections are evaluated before FEA but are not included in that derivative check.

## S4.3 Saved Designs and Export

The projected arrays with the lowest finite compliance are retained, including the initial state. The selected arrays are saved, reloaded with elementwise exact identity, and reanalyzed. A run fails if compliance differs by relative tolerance more than $1 0 ^ { - 6 }$ . Compliance and $V _ { f }$ therefore describe this projected depth state. Exported meshes are separate visualization and geometry check artifacts: they use 200 hard occupancy layers, Gaussian signed distance smoothing with σ = 1.4, removal of components with at most 200 faces, ten Taubin-smoothing iterations, and analytic restoration of the axial bolt holes and the secondary horizontal bore. Watertightness and component count describe that export. Its structural response is not reanalyzed.

Supp. Table 4: Reported optimization results for 200 generated candidates under one fixed configuration. Reduction refers to compliance of the selected projected depth state. “Single-connected” counts projected depth footprints with one component.
<table><tr><td>Group</td><td>n</td><td>Improved</td><td>Mean [%]</td><td>Median [%]</td><td>IQR [%]</td><td>Single- connected</td></tr><tr><td>All candidates</td><td>200</td><td>193</td><td>36.1</td><td>36.8</td><td>23.7–50.8</td><td>195</td></tr><tr><td>moderate topology</td><td>116</td><td>112</td><td>37.1</td><td>39.1</td><td>23.7–53.5</td><td>116</td></tr><tr><td>porous / lattice</td><td>84</td><td>81</td><td>34.7</td><td>34.5</td><td>23.9-43.3</td><td>79</td></tr></table>

Supp. Table 5: Pixel-space and latent-space diffusion results, with 256 samples per run. Out-of-range fractions, crossing fractions, and nearest-training distances are run means. Metrics are defined in Sec. S5.1.
<table><tr><td>Method</td><td>Seed</td><td>Major-volume flags</td><td>Median components</td><td>Out of range [%]</td><td>Crossing [%]</td><td>Nearest-train distance</td></tr><tr><td>Pixel space</td><td>17</td><td>10/256</td><td>5</td><td>0.00</td><td>0.146</td><td>0.02248</td></tr><tr><td>Pixel space</td><td>29</td><td>8/256</td><td>9</td><td>0.00</td><td>0.117</td><td>0.02498</td></tr><tr><td>Pixel space</td><td>43</td><td>34/256</td><td>4</td><td>0.00</td><td>0.171</td><td>0.03046</td></tr><tr><td>Latent space</td><td>17</td><td>0/256</td><td>15</td><td>2.93</td><td>0.450</td><td>0.00216</td></tr><tr><td>Latent space</td><td>29</td><td>0/256</td><td>11</td><td>3.17</td><td>0.326</td><td>0.00197</td></tr><tr><td>Latent space</td><td>43</td><td>0/256</td><td>10</td><td>3.74</td><td>0.313</td><td>0.00187</td></tr></table>

Supp. Table 4 provides the optimization results by candidate group.

## S5 Additional Analyses and Examples

## S5.1 Geometry Metrics

Raw depths were scored before external repair. The out-of-range fraction is the fraction of all values below 0 or above 0.8. A channel pixel is foreground when its depth is below 0.78. Crossing is evaluated only where both channels are foreground and records the fraction satisfying $D _ { \mathrm { t o p } } + D _ { \mathrm { b o t t o m } } > 0 . 8$ . The fixed oracle then applies its own range mapping before reconstruction.

For two depth pairs A and B, the balanced absolute distance first averages absolute error separately over foreground and background pixels for each channel and each direction, then symmetrizes the A-masked and B-masked values. A comparison is null if either pair lacks foreground or background in any channel. Nearest-training distance is the minimum of this symmetric distance over the 211 training designs; smaller values mean greater proximity to the training set. The post-hoc within-run nearest-neighbor distance uses the same metric and excludes the sample itself; smaller values indicate denser local clusters.

Raw oracle meshes were split after degenerate faces were removed. Component count includes every resulting component. The post-hoc major-volume flag is one when the component with the most faces contains less than 99% of total absolute component volume. Largest-component renderings are diagnostic extracts and do not replace the raw generator output.

## S5.2 Pixel-Space and Latent-Space Comparison

Supp. Fig. 1 and Supp. Table 5 report the three-seed comparison. Latent outputs triggered no major-volume fragmentation flags, but had more small components, more depth-range and crossing violations, and smaller distances to training designs and other generated samples. No run contained exact duplicate depth arrays. The fragmentation threshold was defined after inspection, and the difference was concentrated in one pixel-space seed. These results describe the evaluated configurations and do not establish general superiority of either pipeline.

Latent diffusion supplied the optimization candidates because of its lower measured training cost (Sec. S3.3), despite these geometry trade-offs.

![](images/b59f759deaf799e32c4c0f47015d44db8ca68a1744493cb4ae9057b01bc291c8.jpg)

![](images/0c6fd7efa1a61697f322774694d61e8f1e849f41785abd4b2a2c740c4f265710.jpg)

![](images/dc1c2b79dd72cc4ff528c0caf5d5f0740e09e18f71cac5857a77f910b33727eb.jpg)  
Supp. Fig. 1: Pixel-space and latent-space diffusion across three seeds, with 256 samples per run. Balanced depth distance to (a) the nearest training design and (b) the nearest other generated sample. Boxes show medians and interquartile ranges, whiskers extend to 1.5 times the interquartile range, and points show samples. (c) Major-volume fragmentation, with flagged counts. Metrics are defined in Sec. S5.1.

Algorithm S3 Illustrative discrete positive draft projec  
tion for one depth surface   
1: Input: depth surface $D _ { 1 }$ , opposing surface $D _ { 2 } .$ , draft   
angle $\theta ,$ resolution $R$   
2: $\Delta \bar { x } \gets 2 / R ; \delta \gets \cot ( \theta ) \Delta x ;$ k ← 0   
3: repeat   
4: $k \gets k + 1 ;$ ; compute finite difference gradients   
of $D _ { 1 }$   
5: $\bar { S } \gets \{ p : \| \nabla D _ { 1 } ( p ) \| > \mathrm { c o t } ( \theta ) \}$   
6: $( D _ { 1 } ^ { \prime } , D _ { 2 } ^ { \prime } , n _ { \mathrm { u p d a t e } } )  ( D _ { 1 } , D _ { 2 } , 0 )$   
7: for each $p \in S$ do   
8: q ← valid cardinal neighbor in the direction   
of steepest ascent   
9: if $q$ is foreground and the available thickness   
at $q$ is at least $\delta$ then   
10: $D _ { 1 } ^ { \prime } ( p )  D _ { 1 } ( q ) - \delta ; D _ { 2 } ^ { \prime } ( p )  D _ { 2 } ( q )$   
11: n<sub>update</sub> ← n<sub>update</sub> + 1   
12: end if   
13: end for   
14: $( D _ { 1 } , D _ { 2 } )  ( D _ { 1 } ^ { \prime } , D _ { 2 } ^ { \prime } )$   
15: until $n _ { \mathrm { u p d a t e } } = 0 \mathrm { o r } k = 1 0 0 0$   
16: return $\hat { D } _ { 1 } , D _ { 2 }$

![](images/92d851ad5db41355a5028837e37b1ab7e44d265ebe46a043836ea05182893a50.jpg)  
Supp. Fig. 2: Before and after an illustrative draft projection with an exaggerated angle. This operation was not applied to the reported experiments.

## S5.3 Illustrative Positive Draft Projection

For a depth field sampled with in-plane pitch $\Delta x ,$ a draft angle θ measured from the pull axis corresponds to the local finite difference bound

$$
\frac { | D _ { p } - D _ { q } | } { \Delta x } \leq \cot ( \theta )\tag{S1}
$$

for neighboring pixels $p$ and $q .$ Algorithm S3 adjusts neighboring depth values toward this bound while checking the opposing field for available thickness. Supp. Fig. 2 shows the geometric effect with an exaggerated angle. This illustrative operation was not applied to the reported experiments.