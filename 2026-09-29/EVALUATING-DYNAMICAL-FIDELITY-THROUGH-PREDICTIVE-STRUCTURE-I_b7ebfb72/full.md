# EVALUATING DYNAMICAL FIDELITY THROUGH PREDICTIVE STRUCTURE IN PHYSICAL REPRESENTATIONS

A PREPRINT

Oskar Bohn Lassen Department of Technology, Management, and Economics Technical University of Denmark obola@dtu.dk

Stephen I. Thomson   
Department of Mathematics   
and Statistics   
University of Exeter   
s.i.thomson@exeter.ac.uk

João Paulo de Souza Böger Department of Technology, Management, and Economics Technical University of Denmark jpade@dtu.dk

Sebastian Schemm Department of Applied Mathematics and Theoretical Physics University of Cambridge ss3299@cam.ac.uk

Francisco C. Pereira Department of Technology, Management, and Economics Technical University of Denmark camara@dtu.dk

Simon Driscoll Department of Applied Mathematics and Theoretical Physics University of Cambridge sd2136@cam.ac.uk

Filipe Rodrigues Department of Technology, Management, and Economics Technical University of Denmark rodr@dtu.dk

September 29, 2026

## ABSTRACT

Machine-learning models for physical systems are currently evaluated primarily through errors between predicted and reference states and, increasingly, through tests of physical consistency. These metrics assess whether predictions are accurate and satisfy selected physical requirements, but provide limited insight into whether learned trajectories reproduce the underlying dynamics. Domain experts examine such relationships through physical representations that expose relevant processes, interactions, and responses, but these analyses are often separated from typical machinelearning evaluation. We introduce a practical framework for evaluating dynamical fidelity through predictive structure in physical representation spaces. Experts define the representations, while reference trajectories determine which relationships are predictive and retained as evaluation tests. We demonstrate the approach in atmospheric forecasting using ERA5 representations of planetarywave activity and Northern Annular Mode evolution, and evaluate Pangu-Weather, GraphCast, and FengWu. The models exhibit distinct departures from reference predictive structure that are not reflected by conventional forecast errors. The framework thereby turns domain-expert representations into systematic tests of learned physical dynamics without prescribing the relationships in advance.

Keywords Dynamical fidelity · Model evaluation · Physical consistency · Process-oriented diagnostics · Scientific machine learning · Machine learning weather prediction · Stratospheric dynamics

## 1 Introduction

Machine-learning (ML) models are increasingly used to predict the evolution of complex physical systems across materials science, molecular dynamics, and climate science [Noé et al., 2020, Pfaff et al., 2021, Reichstein et al., 2019, Batzner et al., 2022]. These domains are also described by decades of physical theory and numerical modelling, giving us substantial prior knowledge of how the dynamics should behave. While physical constraints and inductive biases can be incorporated into such models [Karniadakis et al., 2021], their success has been mixed, and many leading models remain highly flexible and largely data-driven. This creates a need to use physical knowledge not only to shape models, but to assess whether their predictions reproduce known dynamics.

Evaluation of these models remains dominated by errors between predicted and reference states, which quantify whether a model reaches the correct state but say little about dynamical fidelity: whether it follows the correct dynamical evolution. Recent work therefore complements predictive accuracy with tests of physical consistency, conservation laws, symmetries, and balance relationships [Hansen et al., 2023, Liu et al., 2024]. In principle, a complete set of physical constraints, sufficient to characterize the underlying dynamics, would also establish complete dynamical fidelity. In practice, however, only a limited subset of such constraints can typically be specified and evaluated, so physical consistency restricts the class of admissible dynamics without establishing that the learned evolution matches the underlying system. Across the physical sciences, model validation has therefore also long relied on domain-specific diagnostics of dynamical behaviour, from energy-transfer statistics in turbulence to process-oriented diagnostics in climate models [Ortali et al., 2022, Mohanty et al., 2023, Maloney et al., 2019, Nowack et al., 2020, Lassen et al., 2026, Wu, 2026]. These diagnostics probe whether particular processes, interactions, or response relationships realized by the reference system are preserved by the model. Yet they are usually developed within individual domains and evaluated separately from standard ML metrics, decoupling process-level assessment from routine ML evaluation and making it difficult to determine whether gains in predictive accuracy correspond to more faithful dynamics.

We operationalize process-level dynamical fidelity by identifying predictive relationships in reference trajectories within expert-defined physical representations and testing whether learned models preserve them. These representations define a restricted physical description of the system and therefore another partial view of its dynamics, complementary to testing a finite set of physical-consistency constraints. We assume only that this representation space contains physically relevant sources of predictability for the process under investigation, not that it characterizes the full dynamics. Rather than prescribing which relationships a learned model should satisfy, we identify predictive relationships directly from the reference trajectories within this space, freeze them, and test whether model-generated trajectories preserve the same conditional structure. Experts thereby determine the physical vocabulary in which dynamical behaviour is expressed, while the reference trajectories determine which relationships within that vocabulary are actually predictive.

We instantiate this framework in machine-learning weather forecasting, where models such as Pangu-Weather, Graph-Cast, and FengWu now produce skillful global forecasts at a fraction of the cost of conventional numerical weather prediction [Bi et al., 2023, Lam et al., 2023, Chen et al., 2025]. Evaluation remains largely state-based through benchmarks such as WeatherBench 2 [Rasp et al., 2024], while PhysMetrics.Weather extends this evaluation to conservation laws, spectra, and balance relationships [Kasteleyn et al., 2026]. At the same time, atmospheric science has a long tradition of process-oriented diagnostics for evaluating wave propagation, forcing, and circulation response [Maloney et al., 2019, Kim et al., 2014, Edmon et al., 1980]. This combination makes weather forecasting a natural test bed for evaluating dynamical fidelity beyond state-space accuracy and physical consistency alone. We focus on stratospheric wave–mean-flow dynamics and the subsequent evolution of the Northern Annular Mode (NAM). Using the ERA5 atmo spheric reanalysis as the reference system, we construct an expert-defined representation space spanning planetary-wave activity, vertical propagation, wave forcing, background circulation, and circulation response. Within this space, we identify predictive relationships in ERA5 and test whether ML weather models preserve their conditional consequences, beyond reproducing the atmospheric state itself.

Our contributions are:

• We formalize state-space accuracy, physical consistency, and dynamical fidelity, show that physical consistency does not imply dynamical fidelity, and introduce a framework that discovers process-level dynamical fidelity in expert-defined representations.

• We instantiate the framework for stratospheric wave–mean-flow dynamics and NAM evolution, using ERA5 to discover robust conditional relationships and show that process-level dynamical fidelity reveals model differences across GraphCast, FengWu, and Pangu-Weather not captured by conventional forecast error.

## 2 From Physical Consistency to Dynamical Fidelity

Let $x _ { t } \in \mathcal { X }$ denote the state of a physical dynamical system at time t. A machine-learning model defines a learned state-transition operator $\widehat { x } _ { t + \Delta t } \ = \ \dot { f } _ { \theta } ( \widehat { x } _ { t } )$ , where $f _ { \theta }$ is applied repeatedly to produce an autoregressive trajectory. Predictions are initialised from a reference state $x _ { t _ { 0 } } = \widehat { x } _ { t _ { 0 } }$ , and subsequent hatted states are generated by the model. We use a one-step autoregressive model for notational simplicity, but the formalization below applies to models conditioned on multiple preceding states or predicting multiple future states jointly.

Prediction from data. Machine-learning models of physical dynamical systems are commonly trained by minimising prediction error over observed state transitions,

$$
\theta ^ { \star } = \arg \operatorname* { m i n } _ { \theta } \ \mathbb { E } _ { ( x _ { t } , x _ { t + \Delta t } ) \sim p _ { \mathrm { t r a i n } } } \left[ \ell { \left( f _ { \theta } { \left( x _ { t } \right) } , x _ { t + \Delta t } \right) } \right] ,
$$

where $p _ { \mathrm { t r a i n } }$ is the training distribution and ℓ measures discrepancy between predicted and reference states, possibly with added regularisation. Such training rewards any relationship that improves prediction within $p _ { \mathrm { t r a i n } } ,$ whether it reflects a known physical mechanism, a previously unresolved regularity, or a statistical shortcut [Geirhos et al., 2020]. This flexibility is a strength, the learned dynamics are not restricted to what a numerical scheme already encodes, but low prediction error does not identify which relationships were learned, and two models can perform similarly while relying on different ones [D’Amour et al., 2022]. If those behave differently outside $p _ { \mathrm { t r a i n } }$ or over long rollouts, errors accumulate and trajectories diverge [Sun et al., 2025, Sanchez-Gonzalez et al., 2020].

Physical consistency. Physical-consistency diagnostics assess whether learned dynamics satisfy known requirements of the underlying physical system. Such requirements may also be imposed during training through constrained architectures, physical losses, or hybrid models, but here we consider them purely as evaluation criteria [Karniadakis et al., 2021, Greydanus et al., 2019, Batzner et al., 2022, Hansen et al., 2023, Kochkov et al., 2024]; (Appendix B). To rigorously examine the limitations of physical consistency evaluations, consider a dynamical system

$$
{ \dot { x } } ( t ) = F ( x ( t ) ) ,
$$

where $F$ denotes the vector field mapping each state to its instantaneous time derivative, and let $\mathfrak { F }$ denote the hypothesis class. A sufficiently rich collection of physical relations can itself be viewed as a characterization of the dynamics [Landau and Lifshitz, 1976]. Formally, let

$$
{ \mathcal { Q } } : { \mathfrak { F } } \to { \mathcal { V } } , \qquad F \mapsto { \mathcal { Q } } ( F ) = \left( Q _ { \alpha } [ F ] \right) _ { \alpha \in { \mathcal { D } } }
$$

where α indexes individual physical relations, I is the index set of the complete collection, and $\mathcal { V }$ denotes the space of all possible collections of values taken by these physical relations. Thus, $\mathcal { Q } ( F )$ represents the full set of physical constraints associated with the dynamics defined by $F ,$ , including governing-equation relations, conservation laws, symmetries, balance relations, constitutive constraints, or other physical requirements.

In the complete case, the collection of physical relations $\mathcal { Q } ( F )$ separates admissible dynamics,

$$
\begin{array} { r } { \mathcal { Q } ( F _ { 1 } ) = \mathcal { Q } ( F _ { 2 } ) \quad \Longrightarrow \quad F _ { 1 } = F _ { 2 } . } \end{array}
$$

Thus, when $\mathcal { Q }$ is sufficiently rich to be injective, the complete physical characterization $\mathcal { Q } ( F )$ and the vector field F contain equivalent information about the dynamics.

Physical-consistency evaluation generally has access to only a subset of this characterization. Let $J \subset \mathcal { Z }$ index the physical relations that are known and practically testable. We write each $Q _ { j }$ as a residual such that

$$
Q _ { j } [ F ] = 0
$$

when the underlying dynamics satisfy the corresponding physical relation. The same requirement can then be evaluated for any candidate vector field.

Let $\tilde { F } \in \mathfrak { F }$ denote an arbitrary candidate vector field. The known constraints restrict the possible dynamics to

$$
\mathfrak { F } _ { J } = \left\{ \tilde { F } \in \mathfrak { F } : Q _ { j } [ \tilde { F } ] = 0 \mathrm { f o r } j \in J \right\} .
$$

Unless these constraints are sufficient to identify the dynamics, $\mathfrak { F } _ { J }$ may contain many vector fields ${ \tilde { F } } \neq F$ . Adding another valid physical constraint $j ^ { \ast } \notin J$ can only further restrict this set, such that

$$
F \in { \mathfrak { F } } _ { J \cup \{ j ^ { * } \} } \subseteq { \mathfrak { F } } _ { J } .
$$

However, this restriction alone provides no guarantee that the remaining admissible vector fields are close to the true dynamics. For a chosen norm on vector fields, the identifiability radius is defined as

$$
R _ { J } ( F ) = \operatorname* { s u p } _ { \tilde { F } \in \mathfrak { F } _ { J } } \| \tilde { F } - F \| , \quad R _ { J } ( F ) \in [ 0 , \infty ] ,
$$

which satisfies $R _ { J ^ { \prime } } ( F ) \leq R _ { J } ( F )$ whenever $J \subseteq J ^ { \prime }$ , but need not be small for any $J \subsetneq \mathcal { T }$ that does not identify $F .$

Empirical evaluation introduces a further loss of information. Even when a physical relation is known exactly, it can generally be evaluated only at finite spatial and temporal resolution. For example, the continuous relation ${ \dot { x } } ( t ) = { \bar { F } } ( x ( t ) )$ can only be tested through a temporal discretization such as

$$
\frac { \boldsymbol { x } _ { t + \Delta t } - \boldsymbol { x } _ { t } } { \Delta t } \approx \boldsymbol { F } ( \boldsymbol { x } _ { t } ) ,
$$

together with an analogous discretization in space. Thus, even the known constraints in $\mathcal { Q } _ { J }$ are generally tested only through spatiotemporally coarsened approximations of their continuous counterparts. A learned system may consequently satisfy every evaluated conservation law, symmetry, or balance relation while still realizing different transport, propagation, interaction, or response behaviour [Bonavita, 2024]. Appendix A further formalizes and proves the complementarity between physical-consistency and process-level evaluation introduced next.

Process-level evaluation of dynamical fidelity. Because complete dynamical fidelity is generally not directly testable, it can instead be assessed through process-level diagnostics that expose particular aspects of the system’s evolution. Let

$$
\mathcal { P } _ { k } ( x _ { t : t + L } ) , \qquad k = 1 , \dots , K ,
$$

denote the k-th process diagnostic applied to a trajectory of length $L .$ Such diagnostics are well established across the physical sciences and are typically specified individually for a particular process [Ortali et al., 2022, Mohanty et al., 2023, Maloney et al., 2019]. Depending on the application, $\mathcal { P } _ { k }$ may characterize a flux, transport, propagation, forcing, or response, returning a scalar, a field, or a full space–time diagnostic. Agreement in a given diagnostic can be measured as

$$
\begin{array} { r } { \mathcal { E } _ { k } ^ { \mathrm { d y n } } = { d _ { k } } ( \mathcal { P } _ { k } ( \widehat { x } _ { t : t + L } ) , \mathcal { P } _ { k } ( x _ { t : t + L } ) ) , } \end{array}
$$

where $d _ { k }$ compares predicted and reference diagnostics. Small $\mathcal { E } _ { k } ^ { \mathrm { d y n } }$ therefore indicates agreement in the particular aspect of the dynamics exposed by $\mathcal { P } _ { k }$

## 3 Evaluating dynamical fidelity through expert-defined representations and relationship discovery

Process-level evaluation in the physical sciences has traditionally relied on diagnostics specified individually for a particular process, as represented by $\mathcal { P } _ { k }$ above. Designing and interpreting such tests requires substantial domain expertise and is therefore often separated from the standard machine-learning evaluation pipeline, with important model deficiencies becoming apparent only through subsequent domain-specific analysis. To make process-level evaluation more systematic and easier to integrate into model development, we separate the definition of the physical representation space from the relationships evaluated within it. This framework proposes that experts define a physically meaningful representation space, while the predictive relationships to be evaluated are discovered from reference trajectories and subsequently frozen for model evaluation.

## 3.1 Expert-defined physical representations

The expert-defined representations consist of physically meaningful transformations of the state, chosen to expose relevant sources, transports, interactions, and responses. Their spatial and temporal evolution provides the physical vocabulary within which predictive relationships are subsequently discovered. We make this assumption explicit:

Assumption 1 (Physically informative representation) The expert-defined representation space contains physically relevant sources ofpredictabilityfor the processes under investigation.

This does not require the representation space to be sufficient for the complete system dynamics. It only requires that some combinations of represented quantities, spatial regions, and preceding times contain information about the subsequent evolution of other physically relevant quantities.

Let $\mathcal { X }$ denote the physical state space, with $x _ { t } \in \mathcal { X }$ denoting the system state at time t. We define R expert-specified representations by

$$
\mathcal { G } = \{ g _ { 1 } , \hdots , g _ { R } \} , \qquad g _ { j } : \mathcal { X } \to \mathcal { Z } _ { j } , \quad j = 1 , \hdots , R ,
$$

where ${ \mathcal { Z } } _ { j }$ is the representation space associated with $g _ { j }$ and may be scalar-, vector-, or field-valued. The joint representation map is defined as

$$
G : { \mathcal { X } } \to { \mathcal { Z } } _ { 1 } \times \cdots \times { \mathcal { Z } } _ { R } , \qquad G ( x _ { t } ) = \left( g _ { 1 } ( x _ { t } ) , \ldots , g _ { R } ( x _ { t } ) \right) .
$$

For each representation $g _ { j } ,$ , let $A \in A _ { j }$ denote an admissible spatial region and let $\rho _ { A }$ denote a corresponding spatial operator. We then define the spatially aggregated representation as $g _ { j } ^ { A } ( x _ { t } ) = \rho _ { A } ( g _ { j } ( x _ { t } ) )$

## 3.2 Discovering predictive relationships and evaluating dynamical fidelity

Let $t _ { c }$ denote a cut time separating conditioning history from the future trajectory of interest. Let $W \in { \mathcal { W } }$ denote a temporal window relative to $t _ { c } ,$ , with $W \subseteq ( - \infty , { \bar { 0 } } ]$ ], and let $\psi \in \Psi$ denote an admissible temporal aggregation operator.

A candidate source variable is specified by a representation $j ,$ spatial region A, temporal window $W$ , and operator $\psi$

$$
X _ { t _ { c } } ^ { j , A , W , \psi } = \psi \left( \left( g _ { j } ^ { A } ( x _ { t _ { c } + \tau } ) \right) _ { \tau \in W } \right) .
$$

A conditioning vector consists of M such source variables, which may span different representations, spatial regions, temporal windows, and aggregation operators

$$
\mathbf { X } _ { t _ { c } } = \left( X _ { t _ { c } } ^ { j _ { 1 } , A _ { 1 } , W _ { 1 } , \psi _ { 1 } } , \dots , X _ { t _ { c } } ^ { j _ { M } , A _ { M } , W _ { M } , \psi _ { M } } \right) .
$$

For a target representation k and spatial region $B \in \mathcal A _ { k }$ , its future trajectory after the cut time $t _ { c }$ is defined as

$$
\mathbf { Y } _ { > t _ { c } } ^ { k , B } = \left( g _ { k } ^ { B } ( x _ { t _ { c } + \tau _ { 1 } } ) , \dots , g _ { k } ^ { B } ( x _ { t _ { c } + \tau _ { K } } ) \right) , \qquad 0 < \tau _ { 1 } < \cdot \cdot \cdot < \tau _ { K } .
$$

The relationship-discovery problem is to identify source–target pairs for which the source strongly conditions the target distribution in the reference trajectories. Let C denote the admissible catalogue of candidate pairs $r = ( \mathbf { X } _ { t _ { c } } ^ { ( r ) } , \mathbf { Y } _ { > t _ { c } } ^ { ( r ) } )$ Our method evaluates these candidates on reference trajectories with a discovery and validation procedure ${ \mathcal { D } } _ { \mathrm { r e f } }$ that maps the catalogue to a retained set,

$$
\mathcal { R } _ { \mathrm { r e f } } ^ { \star } = \mathcal { D } _ { \mathrm { r e f } } ( \mathcal { C } ) , \qquad \mathcal { R } _ { \mathrm { r e f } } ^ { \star } \subseteq \mathcal { C } .
$$

The procedure ${ \mathcal { D } } _ { \mathrm { r e f } }$ may be model-based or model-free, causal or purely predictive, and may search candidates individually, greedily, or jointly. The candidate space can grow rapidly if all admissible representations are considered as targets and paired with all possible sources. Established physical theory, prior literature, and literature-informed AI agents can therefore propose and refine ${ \mathcal { C } } ,$ while reference trajectories determine which relationships are retained. Once discovered, the retained relationships are frozen before model evaluation and remain expressed in the expert-defined physical representation space.

For each statistically validated relationship $r \in \mathcal { R } _ { \mathrm { r e f } } ^ { \star }$ , we consider both the joint conditional trajectory distribution and its lead-time marginals,

$$
p _ { \mathrm { r e f } } \left( \mathbf { Y } _ { > t _ { c } } ^ { k , B } \mid \mathbf { X } _ { t _ { c } } \right) , \qquad p _ { \mathrm { r e f } } \left( g _ { k } ^ { B } ( x _ { t _ { c } + \tau _ { \ell } } ) \mid \mathbf { X } _ { t _ { c } } \right) , \quad \ell = 1 , \ldots , K .
$$

We can then evaluate ML model performance by quantifying agreement with the reference conditional relationship using an application-appropriate discrepancy

$$
\mathcal { E } _ { k , B } ^ { \mathrm { d y n } } = S \Big ( p _ { \mathrm { r e f } } \big ( \mathbf { Y } _ { > t _ { c } } ^ { k , B } \mid \mathbf { X } _ { t _ { c } } \big ) , p _ { \mathrm { m o d e l } } \big ( \widehat { \mathbf { Y } } _ { > t _ { c } } ^ { k , B } \mid \mathbf { X } _ { t _ { c } } \big ) \Big ) ,
$$

where $S$ may compare the full distributions or selected statistics such as their conditional means and variances. Each retained relationship thus acts as a process diagnostic $\mathcal { P } _ { k }$ in the sense of Section 2, discovered from the reference rather than specified in advance, and can therefore detect errors that invariants and budgets miss (Appendix A).

## 4 Instantiating Dynamical Fidelity in Stratospheric Circulation

We instantiate the framework in atmospheric forecasting, focusing on the Northern Hemisphere stratospheric circulation. This setting is particularly suitable because established diagnostics of planetary-wave activity and wave–mean-flow interaction provide an expert-defined representation space for predicting subsequent Northern Annular Mode (NAM) evolution.

## 4.1 Scientific target: Northern Annular Mode evolution

We use the Northern Annular Mode (NAM) as a continuous measure of the large-scale circulation and stratosphere– troposphere coupling [Baldwin and Dunkerton, 2001, Baldwin et al., 2021]. Stratospheric NAM anomalies can persist and influence the tropospheric circulation for several weeks [Domeisen et al., 2020, Baldwin et al., 2021], making their early evolution relevant for longer-range prediction.

At pressure level p and time $t ,$ we define the NAM index $N ( p , t )$ as the standardized projection of the geopotential-height anomaly $Z ^ { \prime } ( \lambda , \dot { \phi , } p , t )$ north of 20<sup>◦</sup>N onto its leading empirical orthogonal function (EOF) $e _ { p } ( \lambda , \phi )$ ，

$$
N ( p , t ) = { \frac { c ( p , t ) - \mu _ { c , p } } { \sigma _ { c , p } } } , \qquad c ( p , t ) = \left. \sqrt { \cos \phi } Z ^ { \prime } ( \cdot , \cdot , p , t ) , e _ { p } \right. ,
$$

where $c ( p , t )$ is the area-weighted EOF projection coefficient, λ and ϕ denote longitude and latitude, $\langle \cdot , \cdot \rangle$ denotes the inner product over the latitude–longitude grid, and $\mu _ { c , p }$ and $\sigma _ { c , p }$ are the reference mean and standard deviation of c at pressure level $p .$

![](images/d55d14807bc8059a4a2e85feeef04e47eb91f08b2e045e747c7696eaae038001.jpg)  
Figure 1: First-level predictive structure in ERA5 before robustness and permutation-FDR filtering. (a) Maximum FVE across the candidate space by latitude–pressure region. (b) Representation/wavenumber deep-dive for the three highest-FVE regions. (c) Window/operator deep-dive for their strongest configurations. Outlined cells indicate the successive maxima.

We focus on the NAM averaged over the 50–100 hPa layer, denoted $N _ { 5 0 : 1 0 0 } ( t )$ , and restrict the reference cohort to approximately neutral initial states,

$$
\mathcal { T } = \left\{ t _ { c } : | N _ { 5 0 : 1 0 0 } ( t _ { c } ) | \le 0 . 3 \right\} .
$$

Across winters 1979/80–2024/25 in the ERA5 global atmospheric reanalysis, we obtain $n = 7 7 2$ admissible initialization times. Conditioning on a comparable initial NAM state shifts the question from persistence of the circulation itself to which preceding physical conditions distinguish its subsequent evolution. In this application, all candidate relationships $\boldsymbol { r } \in \mathcal { C }$ share the same target, hence, for each initialization $t _ { c } \in \mathcal { T }$

$$
\mathbf { Y } _ { > t _ { c } } ^ { ( r ) } = \mathbf { Y } _ { > t _ { c } } = ( N _ { 5 0 : 1 0 0 } ( t _ { c } + 1 ) , \ldots , N _ { 5 0 : 1 0 0 } ( t _ { c } + 1 0 ) ) , \qquad r \in \mathcal { C } .
$$

Details of the reference climatology, EOF fitting, and normalization are given in Appendix C.4.

## 4.2 Physical representations and candidate precursor space

Having defined future NAM evolution as the target, we construct a physically meaningful space of candidate precursors from established diagnostics of planetary-wave propagation and wave–mean-flow interaction [Edmon et al., 1980, Baldwin and Dunkerton, 2001]. For $q \in \{ u , v , T \}$ , denoting zonal wind, meridional wind, and temperature, we decompose the flow into its zonal mean $\overline { { q } } ( \phi , p , t )$ and eddy departure $q ^ { \prime } = q - \overline { { q } }$ , separating the background circulation from wave-related variability. The eddy covariances $\overline { { v ^ { \prime } T ^ { \prime } } }$ and $\overline { { u ^ { \prime } v ^ { \prime } } }$ measure the meridional eddy transport of heat and zonal momentum, and enter the quasi-geostrophic Eliassen–Palm (EP) flux [Edmon et al., 1980],

$$
{ \bf F } = ( F _ { \phi } , F _ { p } ) , \qquad F _ { \phi } \propto - \overline { { u ^ { \prime } v ^ { \prime } } } , \qquad F _ { p } \propto \overline { { v ^ { \prime } T ^ { \prime } } } ,
$$

whose components diagnose meridional and vertical propagation of wave activity, and whose divergence $D \equiv \nabla \cdot \mathbf { F }$ diagnoses the resulting wave forcing of the zonal-mean circulation. Our candidate representation space includes these quantities together with the $m = 1$ and $m = 2$ zonal-wavenumber contributions $\mathbf { F } ^ { ( m ) }$ and $D ^ { ( m ) }$ , obtained by Fourier decomposition in longitude, which dominate stratospheric variability. Together with the NAM defined above, the expert representation is

$$
G ( x _ { t } ) = \left\{ \overline { { { u } } } , \overline { { { T } } } , F _ { \phi } , F _ { p } , D , F _ { \phi } ^ { ( 1 ) } , F _ { p } ^ { ( 1 ) } , D ^ { ( 1 ) } , F _ { \phi } ^ { ( 2 ) } , F _ { p } ^ { ( 2 ) } , D ^ { ( 2 ) } , N \right\} .
$$

These representations span the background circulation, planetary-wave propagation and forcing, and the resulting circulation response.

For discovery, the field representations g are converted to climatologically standardized anomalies on the $g _ { j }$ $5 ^ { \circ }$ latitude bands and pressure levels before spatial aggregation into $g _ { j } ^ { A }$ , where $\rho _ { A }$ denotes an unweighted mean over the $5 ^ { \circ }$ latitude bands and pressure levels. A candidate source is defined as

$$
X _ { t _ { c } } ^ { j , A , W _ { d } , \psi } = \psi \left( \left( g _ { j } ^ { A } ( x _ { t _ { c } + \tau } ) \right) _ { \tau \in W _ { d } } \right) .
$$

![](images/25ded431c409d677f66cfa82f22e327d9060c578d547233c03e05b693284da46.jpg)  
(a) First-level candidates

![](images/1ffc0b87da59b3dd15bb2c707373604a007fb80ce71404268d7b3972f4ff9414.jpg)  
(b) Second-level candidates

![](images/7bd0fca49841266a8b5384795ee9bc8561d90c6f526199d16a9088b504d6d145.jpg)  
(c) Example conditional trajectory  
Figure 2: Selection of robust predictive relationships. (a) Of 7,875 first-level candidates, 295 satisfy the 95% directional stability criterion; permutation-FDR calibration retains 283 at $\mathrm { F V E } \geq 0 . 0 5 0 4 .$ . (b) Second-level search within their frozen branches evaluates 4,358,681 conditional candidates, of which 71,494 satisfy 95% stability and 672 remain after permutation-FDR calibration (conditional $\mathrm { F V E } \geq 0 . 3 1 0 2 )$ . (c) Example of leaf that deviates from the neutral mean, isolating initially neutral states that evolve strongly. Lines show winter-weighted medians and shading 10–90% ranges.

where temporal operators ψ ∈ {mean, max, min, positive impulse, negative impulse} are applied over trailing windows $W _ { d } ,$ with $d \in \{ 2 , 3 , 5 , 7 , 1 0 , 2 0 , 3 0 \}$ days and $W _ { d } = \bar { \{ - d , - d + 6 \mathrm { h } , \ldots , - 6 \mathrm { h } \} }$ }. We index each admissible combination $( j , A , W _ { d } , \psi )$ by r, yielding a scalar candidate source. Paired with the common future-NAM trajectory defined above, each scalar source defines a candidate relationship $\boldsymbol { r } \in \mathcal { C }$ . Enumerating all candidate sources yields 7,875 candidate relationships. Additional details are provided in Appendix C-D

## 4.3 Relationship discovery and frozen model evaluation

Using the $n = 7 7 2$ neutral-NAM initializations, we consider each candidate relationship $r \in { \mathcal { C } }$ and threshold split $X ^ { ( r ) } \le c$ where each split partitions $\tau$ into $\mathcal { T } _ { 0 }$ and $\mathcal { T } _ { 1 }$ . Separation is measured by the winter-weighted fraction of variance explained,

$$
\mathrm { F V E } ( r , c ; \mathcal { T } ) = 1 - \frac { \sum _ { b \in \{ 0 , 1 \} } \frac { \eta ( \mathcal { T } _ { b } ) } { \eta ( \mathcal { T } ) } V ( \mathcal { T } _ { b } ) } { V ( \mathcal { T } ) } ,
$$

where each initialization is weighted inversely to the number of dates its winter contributes to $\tau , \eta$ denotes the summed weight, and $V$ is the weighted trajectory variance averaged over NAM lead days 6–10. For each candidate relationship r, we select the eligible threshold

$$
c _ { r } ^ { \star } = \arg \operatorname* { m a x } _ { c } \mathrm { F V E } ( r , c ; \mathcal { T } ) .
$$

To retain reproducible relationships, we use 1,000 winter-level 70/30 train–held-out splits, refitting the threshold on training winters and requiring the separation direction to reproduce in at least 95% of held-out splits [Meinshausen and Bühlmann, 2010]. We then rerun the complete search under 100 cross-winter trajectory permutations and retain the smallest FVE threshold whose upper bootstrap bound (97.5th percentile) on the empirical false discovery rate (FDR) is at most 0.1% [Westfall and Young, 1993].

This gives 283 first-level relationships and thresholds that are frozen, defining 566 branches. An exhaustive second-level search within these branches applies the same stability and permutation-FDR criteria, yielding 672 retained conditional relationships. For each retained split, we select the terminal condition whose days-6–10 NAM mean deviates furthest from its parent branch mean and freeze its ERA5 initialization subset for model evaluation. The same frozen first-level branches are used in every second-level null search. Full resampling and FDR definitions are given in Appendix D.

GraphCast, FengWu, and Pangu-Weather are evaluated on the same initialization subsets. For relationship r, model $m ,$ and lead time $\tau ,$ we measure

$$
E _ { \mu } ^ { ( r , m ) } ( \tau ) = | \mu _ { m , r } ( \tau ) - \mu _ { \mathrm { E R A 5 } , r } ( \tau ) | , \qquad E _ { \sigma ^ { 2 } } ^ { ( r , m ) } ( \tau ) = \big | \sigma _ { m , r } ^ { 2 } ( \tau ) - \sigma _ { \mathrm { E R A 5 } , r } ^ { 2 } ( \tau ) \big | .
$$

Here $\mu _ { m , r } ( \tau )$ and $\sigma _ { m , r } ^ { 2 } ( \tau )$ are the winter-weighted mean and variance of model $m \mathrm { { s } }$ forecast $N _ { 5 0 : 1 0 0 } ( t _ { c } + \tau )$ . Conventional $T , u ,$ and v forecast errors are evaluated on the same subsets, providing a direct comparison between state-space forecast accuracy and preservation of the ERA5 conditional dynamics.

![](images/491fed62b2665c5e1dd6cb05c36b7577e0a8e31c411402f99bbbd630dcb8e202.jpg)  
(a) Conditional-mean error.

![](images/014e946106fe3557ac93d1134c37e2a1538a40ce5a8df9eed710942beb6b79a5.jpg)  
(b) Conditional-variance error.

![](images/21fd6f68405269b30a3e2f2d9f6430d8f8d791256b602d674cf4439983c4c0a5.jpg)  
(c) Conventional forecast error.  
Figure 3: Model evaluation across the 672 selected terminal conditions associated with the retained second-level relationships. (a–b) Absolute error in the conditional mean and variance of future 50–100 hPa NAM within each frozen ERA5-defined terminal leaf; lines show the mean across relationships and shading ±1 standard deviation across relationships. (c) Conventional T, u, and v forecast MAE on the same subsets (unweighted global mean over the 1<sup>◦</sup> grid and 13 pressure levels).

## 5 ERA5 reveals sparse and conditional predictive structure

The ERA5 search reveals a highly structured predictive landscape for the subsequent evolution of initially neutral NAM states (Figure 1). Predictability is not distributed uniformly across the expert-defined representation space: it is concentrated in particular combinations of physical representation, spatial region, temporal window, and aggregation operator. Representations of the background circulation and planetary-wave activity contain the strongest first-level signals, while the successive representation–wavenumber and temporal deep-dives show that these signals are localized to specific physical and temporal configurations rather than to broad diagnostic families as a whole. This structure is also sparse when tested for reproducibility. Only 295 of the 7,875 first-level candidates reproduce their separation direction across winters, with 283 surviving the permutation calibration (Figure 2a). More importantly, conditioning on these first-level states exposes substantial additional structure: exhaustive search within their frozen branches identifies 672 robust second-level relationships from more than 4.3 million conditional candidates (Figure 2b). Thus, much of the reference predictability is not captured by a single precursor in isolation, but emerges only after conditioning on another aspect of the preceding circulation.

The retained relationships therefore describe multiple physically distinct routes from comparable initial NAM states toward different subsequent circulation evolutions. Figure 2c illustrates one such conditional configuration leading to strongly negative subsequent NAM. Taken together, the results depict the reference system not as a small set of prescribed process diagnostics, but as a sparse, hierarchical landscape of predictive relationships distributed across wave activity, wave forcing, background circulation, and prior circulation evolution.

## 6 Dynamical fidelity in ML forecasts

We next test whether GraphCast, FengWu, and Pangu-Weather preserve the ERA5-discovered conditional evolution when initialized from the same states and evaluated on the same frozen subsets. Figure 3 shows that model performance differs depending on whether it is measured by state-space error or preservation of the reference predictive structure. FengWu reproduces the conditional NAM mean and variance most closely over the first three lead days and achieves the lowest T, u, and v MAE at every lead time, but its conditional errors grow fastest and exceed those of both other models by day 10. From day 4 onward, Pangu-Weather most closely reproduces the conditional NAM mean and variance. Thus, lower state-space forecast error does not imply closer agreement with the conditional dynamics identified in ERA5.

More importantly, dynamical fidelity is not uniform across predictive relationships. Table 1 shows that Pangu-Weather has the lowest conditional-mean error across the reported representation and evolution subsets, while FengWu retains the lowest conventional forecast error. The relative fidelity of GraphCast and FengWu, however, changes substantially with the predictive structure being evaluated. For strong negative NAM shifts, GraphCast has lower conditional-mean error than FengWu (0.27 versus 0.33), reversing their ordering for weak negative shifts (0.13 versus 0.09), although FengWu has the lower NAM forecast error on both subsets (Appendix E). Similar variation occurs across individua wave-propagation and wave-forcing representations. These results show that dynamical fidelity is not a single model property: models can preserve different parts of the reference predictive structure even when their aggregate forecast errors are similar. Conventional forecast skill can therefore conceal where within the physical dynamics model behaviour agrees with, or departs from, the reference system.

Table 1: Model evaluation over lead days 6–10 for selected subsets of the 672 selected terminal conditions. Rows are defined by representations used in at least one split and may therefore overlap. $E _ { \mu }$ and $E _ { \sigma ^ { \ 2 } }$ denote conditional NAM mean and variance errors; $T ,$ , u, and v are conventional forecast MAE (unweighted global mean over the $1 ^ { \circ }$ grid and 13 pressure levels). Dark green and light green indicate the lowest and second-lowest model errors in each column, respectively.
<table><tr><td></td><td></td><td></td><td></td><td colspan="4">GraphCast</td><td colspan="6">FengWu</td><td colspan="4">Pangu-Weather</td></tr><tr><td></td><td></td><td></td><td></td><td colspan="2">Fidelity</td><td colspan="2">Forecast</td><td colspan="2">Fidelity</td><td colspan="2">Forecast</td><td colspan="2"></td><td colspan="2">Fidelity</td><td colspan="2">Forecast</td></tr><tr><td></td><td>Predictive structure</td><td>N</td><td>%</td><td> $E _ { \mu }$ </td><td> $E _ { \sigma ^ { 2 } }$ </td><td>T</td><td>u</td><td>v</td><td> $E _ { \mu }$ </td><td> $E _ { \sigma ^ { 2 } }$ </td><td>T u</td><td>v</td><td></td><td> $E _ { \mu }$   $E _ { \sigma ^ { 2 } }$ </td><td>T</td><td>u</td><td>v</td></tr><tr><td></td><td>All</td><td>672</td><td>100.0</td><td>0.17</td><td>0.20</td><td>2.29</td><td>5.73</td><td>5.77</td><td>0.15 0.21</td><td>1.76</td><td>4.49</td><td>4.56</td><td>0.09</td><td>0.11</td><td>2.17</td><td>5.49</td><td>5.58</td></tr><tr><td rowspan="10"></td><td>Total EP1</td><td>102</td><td>15.2</td><td>0.21</td><td>0.18</td><td>2.30</td><td>5.74</td><td>5.76 0.18</td><td>0.20</td><td>1.77</td><td>4.50</td><td>4.57</td><td>0.09</td><td>0.10</td><td>2.19</td><td>5.53</td><td>5.61</td></tr><tr><td>Wave-1 EP1</td><td>96</td><td>14.3</td><td>0.15</td><td>0.26</td><td>2.29</td><td>5.71</td><td>5.77</td><td>0.11 0.21</td><td>1.77</td><td>4.50</td><td>4.57</td><td>0.09</td><td>0.11 0.07</td><td>2.16</td><td>5.45</td><td>5.55</td></tr><tr><td>Wave-2 EP1</td><td>12</td><td>1.8</td><td>0.26</td><td>0.20</td><td>2.27</td><td>5.68</td><td>5.69</td><td>0.18</td><td>0.17</td><td>1.75 4.45</td><td>4.52</td><td></td><td>0.11</td><td>2.17</td><td>5.47</td><td>5.55</td></tr><tr><td>Total EP2</td><td>12</td><td>1.8</td><td>0.18</td><td>0.20</td><td>2.30</td><td>5.75</td><td>5.78</td><td>0.20 0.18</td><td></td><td>1.78 4.51</td><td>4.59</td><td></td><td>0.11 0.12</td><td>2.20</td><td>5.54</td><td>5.61</td></tr><tr><td>Wave-1 EP2</td><td>239 27</td><td>35.6</td><td>0.17</td><td>0.19</td><td>2.30</td><td>5.75</td><td>5.79</td><td>0.17 0.23</td><td>1.76</td><td>4.49</td><td>4.55</td><td>0.11</td><td>0.10</td><td>2.17</td><td>5.49</td><td>5.57</td></tr><tr><td>Wave-2 EP2</td><td>48</td><td>4.0 7.1</td><td>0.21</td><td>0.10</td><td>2.29</td><td>5.71</td><td>5.73</td><td>0.19</td><td>0.14 1.78</td><td>4.51</td><td>4.60</td><td>0.11</td><td>0.10</td><td>2.18</td><td>5.49</td><td>5.59</td></tr><tr><td>Total Div</td><td>146</td><td>21.7</td><td>0.17 0.20</td><td>0.19</td><td>2.30</td><td>5.74</td><td>5.76</td><td>0.20 0.27</td><td>1.78 1.78</td><td>4.51 4.51</td><td>4.57 4.57</td><td>0.12</td><td>0.09 0.11</td><td>2.20</td><td>5.52</td><td>5.60</td></tr><tr><td>Wave-1 Div</td><td>57</td><td>8.5</td><td>0.12</td><td>0.15 0.15</td><td>2.30 2.26</td><td>5.74 5.67</td><td>5.77 5.71</td><td>0.25 0.09</td><td>0.28 0.14 1.76</td><td>4.48</td><td>4.55</td><td>0.12 0.07</td><td>0.10</td><td>2.20</td><td>5.53</td><td>5.59 5.59</td></tr><tr><td>Wave-2 Div</td><td>135</td><td>20.1</td><td>0.17</td><td>0.20</td><td>2.27</td><td>5.69</td><td>5.73</td><td>0.10</td><td>0.18</td><td>1.74 4.46</td><td>4.54</td><td>0.07</td><td>0.14</td><td>2.17 2.18</td><td>5.48 5.50</td><td>5.60</td></tr><tr><td>Zonal-mean T Zonal-mean u</td><td>410</td><td>61.0</td><td>0.15</td><td>0.22</td><td>2.29</td><td>5.73</td><td>5.78</td><td></td><td>1.75</td><td></td><td>4.55</td><td></td><td></td><td>2.16</td><td>5.48</td><td>5.57</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.10</td><td>0.19</td><td></td><td>4.48 4.46</td><td>4.53</td><td>0.08</td><td>0.11 0.09</td><td></td><td></td><td></td></tr><tr><td rowspan="4">Evolution</td><td>Weak positive shift</td><td>61</td><td>9.1</td><td>0.14</td><td>0.14</td><td>2.28</td><td>5.71</td><td>5.76</td><td>0.09 0.09</td><td>0.11 0.16</td><td>1.75 1.75</td><td></td><td></td><td>0.08</td><td>2.15</td><td>5.47</td><td>5.56</td></tr><tr><td>Weak negative shift</td><td>107 19</td><td>15.9 2.8</td><td>0.13 0.12</td><td>0.17 0.14</td><td>2.29 2.23</td><td>5.73 5.59</td><td>5.79 5.59</td><td>0.06 0.08</td><td>1.72</td><td>4.47 4.40</td><td>4.54 4.48</td><td>0.08 0.05</td><td>0.09 0.07</td><td>2.14 2.12</td><td>5.45</td><td>5.54 5.47</td></tr><tr><td>Strong positive shift</td><td>150</td><td>22.3</td><td>0.27</td><td>0.16</td><td>2.32</td><td>5.77</td><td>5.77</td><td>0.33 0.34</td><td>1.79</td><td>4.53</td><td>4.59</td><td>0.15</td><td>0.12</td><td>2.21</td><td>5.39 5.53</td><td>5.58</td></tr><tr><td>Strong negative shift</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 7 Conclusion

We introduced a framework for evaluating learned physical dynamics through predictive structure in expert-defined representation spaces. Experts define the physical vocabulary, while robust relationships are discovered from reference trajectories and frozen as evaluation tests. Applied to stratospheric circulation, the framework reveals differences in how Pangu-Weather, GraphCast, and FengWu preserve ERA5 wave–mean-flow relationships and subsequent NAM evolution that are not captured by conventional forecast errors. Accurate state prediction therefore does not by itself establish preservation of reference predictive structure. These relationships should not be interpreted as identified physical mechanisms: ERA5 serves as the reference rather than ground truth, and the framework evaluates predictive rather than causal structure. Because some precursors summarize histories longer than those directly available to the forecast models, disagreement may reflect either missing initialization information or incorrect subsequent evolution. Finally, stability filtering and permutation calibration reduce sensitivity to chance relationships within the finite ERA5 winter cohort but do not establish transfer to other periods, reanalyses, or physical systems.

## Reproducibility Statement

The relationship catalogue, discovery procedure, and dynamical-fidelity and conventional forecast diagnostics are specified in Appendices C- E. Anonymized code to reproduce the ERA5 inputs, relationship discovery, model forecasts, evaluation, and figures is available here.

## AI use statement

Generative AI tools, including OpenAI ChatGPT and Anthropic Claude, were used throughout this work to support methodological development, mathematical formulation, code development and review, interpretation of results, literature-oriented brainstorming and search, and drafting and editing of the manuscript. AI-assisted code and analyses were tested and verified by the authors, and literature and scientific claims were checked against the underlying sources. The authors made all final decisions regarding the methodology, experiments, interpretation, and presentation of results, and take responsibility for the full content of the work.

## References

Frank Noé, Alexandre Tkatchenko, Klaus-Robert Müller, and Cecilia Clementi. Machine Learning for Molecular Simulation. Annual Review of Physical Chemistry, 71:361–390, 2020. ISSN 1545- 1593. doi:10.1146/annurev-physchem-042018-052331. URL https://www.annualreviews.org/content/ journals/10.1146/annurev-physchem-042018-052331.

Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter Battaglia. Learning Mesh-Based Simulation with Graph Networks. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=roNqYL0\_XP.

Markus Reichstein, Gustau Camps-Valls, Bjorn Stevens, Martin Jung, Joachim Denzler, Nuno Carvalhais, and Prabhat. Deep learning and process understanding for data-driven Earth system science. Nature, 566(7743):195–204, February 2019. ISSN 1476-4687. doi:10.1038/s41586-019-0912-1. URL https://www.nature.com/articles/ s41586-019-0912-1.

Simon Batzner, Albert Musaelian, Lixin Sun, Mario Geiger, Jonathan P. Mailoa, Mordechai Kornbluth, Nicola Molinari, Tess E. Smidt, and Boris Kozinsky. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nature Communications, 13(1):2453, May 2022. ISSN 2041-1723. doi:10.1038/s41467-022-29939-5. URL https://www.nature.com/articles/s41467-022-29939-5.

George Em Karniadakis, Ioannis G. Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics, 3(6):422–440, June 2021. ISSN 2522-5820. doi:10.1038/s42254-021- 00314-5. URL https://www.nature.com/articles/s42254-021-00314-5.

Derek Hansen, Danielle C. Maddix, Shima Alizadeh, Gaurav Gupta, and Michael W. Mahoney. Learning Physical Models that Can Respect Conservation Laws. In Proceedings of the 40th International Conference on Machine Learning, pages 12469–12510. PMLR, July 2023. URL https://proceedings.mlr.press/v202/hansen23b. html.

Ning Liu, Yiming Fan, Xianyi Zeng, Milan Klöwer, Lu Zhang, and Yue Yu. Harnessing the Power of Neural Operators with Automatically Encoded Conservation Laws. In Proceedings ofthe 41st International Conference on Machine Learning, pages 30965–30997. PMLR, July 2024. URL https://proceedings.mlr.press/v235/liu24p. html.

Giulio Ortali, Alessandro Corbetta, Gianluigi Rozza, and Federico Toschi. Numerical proof of shell model turbulence closure. Physical Review Fluids, 7(8):L082401, August 2022. doi:10.1103/PhysRevFluids.7.L082401. URL https://link.aps.org/doi/10.1103/PhysRevFluids.7.L082401.

Shaswat Mohanty, SangHyuk Yoo, Keonwook Kang, and Wei Cai. Evaluating the transferability of machine-learned force fields for material property modeling. Computer Physics Communications, 288:108723, July 2023. ISSN 0010-4655. doi:10.1016/j.cpc.2023.108723. URL https://www.sciencedirect.com/science/article/pii/ S0010465523000681.

Eric D. Maloney, Andrew Gettelman, Yi Ming, J. David Neelin, Daniel Barrie, Annarita Mariotti, C.-C. Chen, Danielle R. B. Coleman, Yi-Hung Kuo, Bohar Singh, H. Annamalai, Alexis Berg, James F. Booth, Suzana J. Camargo, Aiguo Dai, Alex Gonzalez, Jan Hafner, Xianan Jiang, Xianwen Jing, Daehyun Kim, Arun Kumar, Yumin Moon, Catherine M. Naud, Adam H. Sobel, Kentaroh Suzuki, Fuchang Wang, Junhong Wang, Allison A. Wing, Xiaobiao Xu, and Ming Zhao. Process-Oriented Evaluation of Climate and Weather Forecasting Models. Bulletin ofthe American Meteorological Society, 100(9):1665–1686, September 2019. ISSN 0003-0007, 1520-0477. doi:10.1175/BAMS-D 18-0042.1. URL https://journals.ametsoc.org/view/journals/bams/100/9/bams-d-18-0042.1.xml.

Peer Nowack, Jakob Runge, Veronika Eyring, and Joanna D. Haigh. Causal networks for climate model evaluation and constrained projections. Nature Communications, 11(1):1415, March 2020. ISSN 2041-1723. doi:10.1038/s41467- 020-15195-y. URL https://www.nature.com/articles/s41467-020-15195-y.

Oskar Bohn Lassen, Simon Driscoll, Stephen I. Thomson, Sebastian Schemm, and Francisco C. Pereira. Investigating Inductive Biases for Machine Learning Emulation of Sudden Stratospheric Warmings in Idealised Isca Simulations, June 2026. URL http://arxiv.org/abs/2606.18857. arXiv:2606.18857 [cs.LG].

Zheng Wu. Assessing Subseasonal Predictions of Stratosphere-Troposphere Coupling of GraphCast. Journal of Geophysical Research: Atmospheres, 131(6):e2025JD044852, 2026. ISSN 2169-8996. doi:10.1029/2025JD044852. URL https://onlinelibrary.wiley.com/doi/abs/10.1029/2025JD044852.

Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. Accurate medium-range global weather forecasting with 3D neural networks. Nature, 619(7970):533–538, July 2023. ISSN 1476-4687. doi:10.1038/s41586- 023-06185-3. URL https://www.nature.com/articles/s41586-023-06185-3.

Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, Weihua Hu, Alexander Merose, Stephan Hoyer, George Holland, Oriol Vinyals, Jacklynn Stott, Alexander Pritzel, Shakir Mohamed, and Peter Battaglia. Learning skillful medium-range global weather forecasting. Science, 382(6677):1416–1421, December 2023. doi:10.1126/science.adi2336. URL https://www.science.org/doi/10.1126/science.adi2336.

Kang Chen, Tao Han, Fenghua Ling, Junchao Gong, Lei Bai, Xinyu Wang, Jing-Jia Luo, Ben Fei, Wenlong Zhang, Xi Chen, Leiming Ma, Tianning Zhang, Rui Su, Yuanzheng Ci, Bin Li, Xiaokang Yang, and Wanli Ouyang. The operational medium-range deterministic weather forecasting can be extended beyond a 10-day lead time. Communications Earth & Environment, 6(1):518, July 2025. ISSN 2662-4435. doi:10.1038/s43247-025-02502-y. URL https://www.nature.com/articles/s43247-025-02502-y.

Stephan Rasp, Stephan Hoyer, Alexander Merose, Ian Langmore, Peter Battaglia, Tyler Russell, Alvaro Sanchez-Gonzalez, Vivian Yang, Rob Carver, Shreya Agrawal, Matthew Chantry, Zied Ben Bouallegue, Peter Dueben, Carla Bromberg, Jared Sisk, Luke Barrington, Aaron Bell, and Fei Sha. WeatherBench 2: A Benchmark for the Next Generation of Data-Driven Global Weather Models. Journal ofAdvances in Modeling Earth Systems, 16(6):e2023MS004019, June 2024. ISSN 1942-2466, 1942-2466. doi:10.1029/2023MS004019. URL https: //agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS004019.

Emma Kasteleyn, Timo Maier, Axel Lauer, Veronika Eyring, Pierre Gentine, and Ana Lucic. PhysMetrics.Weather: An Evaluation Framework for Physical Consistency in ML Weather Models, June 2026. URL http://arxiv.org/ abs/2606.10642. arXiv:2606.10642 [cs.LG].

Daehyun Kim, Prince Xavier, Eric Maloney, Matthew Wheeler, Duane Waliser, Kenneth Sperber, Harry Hendon, Chidong Zhang, Richard Neale, Yen-Ting Hwang, and Haibo Liu. Process-Oriented MJO Simulation Diagnostic: Moisture Sensitivity of Simulated Convection. Journal ofClimate, 27(14):5379–5395, July 2014. ISSN 0894-8755, 1520-0442. doi:10.1175/JCLI-D-13-00497.1. URL https://journals.ametsoc.org/view/journals/clim/ 27/14/jcli-d-13-00497.1.xml.

H. J. Edmon, B. J. Hoskins, and M. E. McIntyre. Eliassen-Palm Cross Sections for the Troposphere. Journal of the Atmospheric Sciences, 37(12):2600–2616, December 1980. ISSN 0022-4928, 1520-0469. doi:10.1175/1520- 0469(1980)037<2600:EPCSFT>2.0.CO;2. URL https://journals.ametsoc.org/view/journals/atsc/37/ 12/1520-0469\_1980\_037\_2600\_epcsft\_2\_0\_co\_2.xml.

Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, November 2020. ISSN 2522-5839. doi:10.1038/s42256-020-00257-z. URL https://www.nature.com/articles/ s42256-020-00257-z.

Alexander D’Amour, Katherine Heller, Dan Moldovan, Ben Adlam, Babak Alipanahi, Alex Beutel, Christina Chen, Jonathan Deaton, Jacob Eisenstein, Matthew D. Hoffman, Farhad Hormozdiari, Neil Houlsby, Shaobo Hou, Ghassen Jerfel, Alan Karthikesalingam, Mario Lucic, Yian Ma, Cory McLean, Diana Mincu, Akinori Mitani, Andrea Montanari, Zachary Nado, Vivek Natarajan, Christopher Nielson, Thomas F. Osborne, Rajiv Raman, Kim Ramasamy, Rory Sayres, Jessica Schrouff, Martin Seneviratne, Shannon Sequeira, Harini Suresh, Victor Veitch, Max Vladymyrov, Xuezhi Wang, Kellie Webster, Steve Yadlowsky, Taedong Yun, Xiaohua Zhai, and D. Sculley. Underspecification Presents Challenges for Credibility in Modern Machine Learning. Journal ofMachine Learning Research, 23(226): 1–61, 2022. ISSN 1533-7928. URL http://jmlr.org/papers/v23/20-1335.html.

Y. Qiang Sun, Pedram Hassanzadeh, Mohsen Zand, Ashesh Chattopadhyay, Jonathan Weare, and Dorian S. Abbot. Can AI weather models predict out-of-distribution gray swan tropical cyclones? Proceedings of the National Academy of Sciences, 122(21):e2420914122, May 2025. doi:10.1073/pnas.2420914122. URL https://www.pnas.org/doi/ 10.1073/pnas.2420914122.

Alvaro Sanchez-Gonzalez, Jonathan Godwin, Tobias Pfaff, Rex Ying, Jure Leskovec, and Peter Battaglia. Learning to Simulate Complex Physics with Graph Networks. In Proceedings of the 37th International Conference on Machine Learning, pages 8459–8468. PMLR, November 2020. URL https://proceedings.mlr.press/v119/ sanchez-gonzalez20a.html.

Samuel Greydanus, Misko Dzamba, and Jason Yosinski. Hamiltonian Neural Networks. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://papers.nips.cc/ paper\_files/paper/2019/hash/26cd8ecadce0d4efd6cc8a8725cbd1f8-Abstract.html.

Dmitrii Kochkov, Janni Yuval, Ian Langmore, Peter Norgaard, Jamie Smith, Griffin Mooers, Milan Klöwer, James Lottes, Stephan Rasp, Peter Düben, Sam Hatfield, Peter Battaglia, Alvaro Sanchez-Gonzalez, Matthew Willson, Michael P. Brenner, and Stephan Hoyer. Neural general circulation models for weather and climate. Nature, 632 (8027):1060–1066, August 2024. ISSN 1476-4687. doi:10.1038/s41586-024-07744-y. URL https://www.nature. com/articles/s41586-024-07744-y.

L. D. Landau and E. M. Lifshitz. CHAPTER I - THE EQUATIONS OF MOTION. In Mechanics (Third Edition), pages 1–12. Butterworth-Heinemann, Oxford, January 1976. ISBN 978-0-7506-2896-9. doi:10.1016/B978-0-08-050347- 9.50006-X. URL https://www.sciencedirect.com/science/article/pii/B978008050347950006X.

Massimo Bonavita. On Some Limitations of Current Machine Learning Weather Prediction Models. Geophysical Research Letters, 51(12):e2023GL107377, 2024. ISSN 1944-8007. doi:10.1029/2023GL107377. URL https: //onlinelibrary.wiley.com/doi/abs/10.1029/2023GL107377.

Mark P. Baldwin and Timothy J. Dunkerton. Stratospheric Harbingers of Anomalous Weather Regimes. Science, 294 (5542):581–584, October 2001. doi:10.1126/science.1063315. URL https://www.science.org/doi/10.1126/ science.1063315.

Mark P. Baldwin, Blanca Ayarzagüena, Thomas Birner, Neal Butchart, Amy H. Butler, Andrew J. Charlton-Perez, Daniela I. V. Domeisen, Chaim I. Garfinkel, Hella Garny, Edwin P. Gerber, Michaela I. Hegglin, Ulrike Langematz, and Nicholas M. Pedatella. Sudden Stratospheric Warmings. Reviews of Geophysics, 59(1):e2020RG000708, 2021. ISSN 1944-9208. doi:10.1029/2020RG000708. URL https://onlinelibrary.wiley.com/doi/abs/10. 1029/2020RG000708.

Daniela I. V. Domeisen, Amy H. Butler, Andrew J. Charlton-Perez, Blanca Ayarzagüena, Mark P. Baldwin, Etienne Dunn-Sigouin, Jason C. Furtado, Chaim I. Garfinkel, Peter Hitchcock, Alexey Yu. Karpechko, Hera Kim, Jeff Knight, Andrea L. Lang, Eun-Pa Lim, Andrew Marshall, Greg Roff, Chen Schwartz, Isla R. Simpson, Seok-Woo Son, and Masakazu Taguchi. The Role of the Stratosphere in Subseasonal to Seasonal Prediction: 2. Predictability Arising From Stratosphere-Troposphere Coupling. Journal ofGeophysical Research: Atmospheres, 125(2):e2019JD030923, 2020. ISSN 2169-8996. doi:10.1029/2019JD030923. URL https://onlinelibrary.wiley.com/doi/abs/10. 1029/2019JD030923.

Nicolai Meinshausen and Peter Bühlmann. Stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4):417–473, 2010. ISSN 1467-9868. doi:10.1111/j.1467-9868.2010.00740.x. URL https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2010.00740.x.

Peter H. Westfall and S. Stanley Young. Resampling-Based Multiple Testing: Examples and Methods for p-Value Adjustment. John Wiley & Sons, January 1993. ISBN 978-0-471-55761-6. Google-Books-ID: nuQXORVGI1QC.

Ernst Hairer, Christian Lubich, and Gerhard Wanner. Geometric numerical integration. Structure-preserving algorithms for ordinary differential equations. 2nd ed, volume 31. Springer Berlin Heidelberg, January 2006. ISBN 978-3-540- 30663-4. doi:10.1007/3-540-30666-8. Journal Abbreviation: vol. 31. Springer Science & Business Media, 2006. Publication Title: vol. 31. Springer Science & Business Media, 2006.

José Pinto Peixoto. Physics of Climate. American Institute of Physics, 1992. ISBN 978-0-88318-712-8. Google-Books-ID: 3tjKa0YzFRMC.

Gregory J. Hakim and Sanjit Masanam. Dynamical Tests of a Deep Learning Weather Prediction Model. Artificial Intelligence for the Earth Systems, 3(3), July 2024. ISSN 2769-7525. doi:10.1175/AIES-D-23-0090.1. URL https://journals.ametsoc.org/view/journals/aies/3/3/AIES-D-23-0090.1.xml.

M. Raissi, P. Perdikaris, and G. E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal ofComputational Physics, 378:686–707, February 2019. ISSN 0021-9991. doi:10.1016/j.jcp.2018.10.045. URL https://www. sciencedirect.com/science/article/pii/S0021999118307125.

J. Palis. Vector fields generate few diffeomorphisms. Bulletin of the American Mathematical Society, 80(3):503–505, May 1974. ISSN 0002-9904, 1936-881X. URL https://projecteuclid. org/journals/bulletin-of-the-american-mathematical-society/volume-80/issue-3/ Vector-fields-generate-few-diffeomorphisms/bams/1183535529.full.

## A Budget and algebraic constraints as coarsenings of the differential residual

This appendix makes the evaluation axes of Section 2 precise and shows how they relate. We express every test through a single quantity, the differential residual, which measures how far a trajectory departs from the reference dynamics. This lets us order the tests by what they can detect, (3), and a worked example then shows that physical consistency and process-level dynamical fidelity capture different aspects of the dynamics and are therefore complementary.

Let the reference dynamics be ${ \dot { x } } = F ( x )$ . The learned model is instead a discrete map $f _ { \theta }$ that advances the state $x ( t )$ by one step $\Delta t .$ . A model trained on states does not represent F directly, but its integral over $\Delta t$

The issue of not directly modeling $F$ concerns the discrete-time form of the model, not the resolution of the state, and for numerical one-step methods with a small step, backward error analysis shows that the agreement between the discrete and continuous model can be made up to an error exponentially small in $1 / \Delta t$ [Hairer et al., 2006].

The purpose of the discrete-time dynamics assumption is to give the predicted trajectory a meaning between grid times, so that reference and prediction can be tested with the same quantity. We then define $Q _ { 0 }$ , one of the $Q _ { \alpha } ,$ , abusing notation here to include trajectories in its domain when it took the vector field originally, due to the discretization relation discussed above, as the differential residua

$$
Q _ { 0 } [ x ] ( t ) = \dot { x } ( t ) - F \big ( x ( t ) \big ) ,\tag{1}
$$

which compares the actual rate of change of a trajectory x with the rate that the reference dynamics prescribes at the same state. For a trajectory of the reference system the two coincide by definition, so $Q _ { 0 } [ x ] \equiv 0 ,$ , and a non-zero residual signals a trajectory that the reference system could not produce. Along a predicted trajectory $\dot { \widehat { x } } = \widehat { F } ( \widehat { { x } } )$ , so the residual equals the error of the learned vector field, $\delta { \cal F } = \widehat { \cal F } - { \cal F } .$ , at the visited states,

$$
Q _ { 0 } [ \widehat { x } ] ( t ) = \delta F \big ( \widehat { x } ( t ) \big ) .\tag{2}
$$

Tests evaluated on predicted trajectories therefore see $\delta F$ only at the states the model visits, much as a test set probes a model only on the inputs it contains.

In practice, the relations $Q _ { j }$ of Section 2 are checked on trajectories, often sampled coarsely in space and time. We could write $Q _ { j } [ x ]$ for relation j evaluated on a trajectory x, with $Q _ { j } [ x ] = 0$ whenever x is a reference trajectory. The residual $Q _ { 0 }$ is the vector field relation in trajectories form, and the relations below are all expressed through it. A set of relations J is weaker than $J ^ { \prime } .$ , written $J \preceq \bar { J } ^ { \prime }$ , if every trajectory satisfying the relations in ${ \bf { \bar { \boldsymbol { J } } } } ^ { \prime }$ also satisfies those in J, as $\mathfrak { F } _ { J ^ { \prime } } \subseteq \mathfrak { F } _ { J }$ in Section $2 ,$ , and strictly weaker, $J \prec J ^ { \prime }$ , if some trajectory satisfies $J$ but not $J ^ { \prime }$ . Since a trajectory with $Q _ { 0 } [ x ] \equiv 0$ is a reference trajectory, $Q _ { 0 }$ is stronger than every other relation.

Assumption 2 (Flow representation) The learned map is the one-step flow of a vector field $\widehat F \in \mathfrak { F } ,$ , that is, $f _ { \theta } ( \widehat { x } _ { t } ) =$ $\widehat { x } _ { t + \Delta t }$ with $\dot { \widehat { x } } = \widehat { F } ( \widehat { x } )$ in between.

Proposition 1 (Ordering of evaluation tests) Under Assumption 2, let F be Lipschitz, and consider finitely many invariants $\{ C _ { j } \}$ ,finitely many budgets $\{ B _ { j } \}$ over consecutive intervals offixed length $\Delta t$ that include the budgets of the invariants, andfinitely many process diagnostics $\{ \mathcal { E } _ { k } ^ { \mathrm { d y n } } \}$ evaluatedfrom the reference initial state. Then

$$
\underbrace { \left\{ C _ { j } \right\} } _ { i n v a r i a n t s } \preceq \underbrace { \left\{ B _ { j } \right\} } _ { b u d g e t s } \preceq \underbrace { Q _ { 0 } } _ { d i f f e r e n t i a l r e s i d u a l } \succeq \underbrace { \left\{ \mathcal { E } _ { k } ^ { \mathrm { d y n } } \right\} } _ { p r o c e s s d i a g n o s t i c s } ,\tag{3}
$$

where each relation can be strict under the following conditions:

1. Invariants to budget, is strict whenever the budgets include a quantity that the dynamics changes.

2. Process diagnostics and invariants are not ordered: a trajectory can pass every invariant andfail a process diagnostic, or pass a process diagnostic andfail an invariant, and therefore, since an invariant is a sum of budgets, at least one budget of that quantity.

Proof. Sections A.1 and A.2 establish the chain on the left and Section A.3 the relation on the right. For strictness and non-comparability, the scaled oscillator below passes every invariant but fails a budget and the period diagnostic, the damped oscillator passes the period diagnostic but fails the energy invariant, and a residual whose weighted integrals vanish on every interval passes all budgets while $Q _ { 0 } \neq 0$ (Section A.1).

## A.1 Budgets

A budget states that the change of some quantity over a time interval equals what the dynamics adds or removes during that interval, as when the change in heat content of an ocean region equals the net heat flux through its boundaries [Peixoto, 1992]. Our aim here is to write this statement as a relation in the sense of Section 2, a residual that vanishes on reference trajectories, and to express it through $Q _ { 0 }$ . This form shows exactly what a budget checks, and what it cannot see. The budgeted quantity is a functional $E _ { j }$ , a number computed from the whole state, such as the mass $\begin{array} { r } { E _ { j } ( u ) = \int _ { \Omega ^ { \prime } } } \end{array}$ u dz contained in a region $\Omega ^ { \prime }$ of the domain $\Omega ,$ or the kinetic energy $\begin{array} { r } { E _ { j } ( u ) = \frac 1 2 \int _ { \Omega } | u | ^ { 2 } \mathrm { d } z . \ E _ { j } } \end{array}$ is the quantity being tracked, not a relation; the relation is the budget $B _ { j }$ defined below, which plays the role of $Q _ { j } [ x ]$

To describe how $E _ { j }$ changes, we use the spatial inner product $\begin{array} { r } { \langle \varphi , v \rangle = \int _ { \Omega } \varphi \cdot v \mathrm { d } z } \end{array}$ , which returns the φ-weighted total of a field v over the domain and reduces to the Euclidean dot product for finite-dimensional states. The gradient $\partial _ { x } E _ { j } ( x )$ is defined by ${ { E } _ { j } } ( x + \varepsilon v ) = { { E } _ { j } } ( x ) + \varepsilon \langle \partial _ { x } { { E } _ { j } } ( x ) , v \rangle + o ( \bar { \varepsilon } )$ , so that $\langle \partial _ { x } E _ { j } ( x ) , v \rangle$ is the change of $E _ { j }$ caused by a small change v of the state; it is $\mathbf { 1 } _ { \Omega ^ { \prime } }$ for the regional mass and u for the kinetic energy. Along any trajectory, the chain rule gives $\begin{array} { r } { \overline { { \mathrm { d } } } E _ { j } ( x ) = \langle \partial _ { x } E _ { j } ( x ) , \dot { x } \rangle } \end{array}$ , and integrating the residual weighted by this gradient over $[ t , t + \Delta t ]$ gives

$$
\begin{array} { l } { \displaystyle B _ { j } : = \int _ { t } ^ { t + \Delta t } \left. \partial _ { x } E _ { j } ( x ( t ^ { \prime } ) ) , Q _ { 0 } [ x ] ( t ^ { \prime } ) \right. \mathrm { d } t ^ { \prime } } \\ { \displaystyle \ = E _ { j } \big ( x _ { t + \Delta t } \big ) - E _ { j } ( x _ { t } ) - \int _ { t } ^ { t + \Delta t } \left. \partial _ { x } E _ { j } ( x ( t ^ { \prime } ) ) , F ( x ( t ^ { \prime } ) ) \right. \mathrm { d } t ^ { \prime } . } \end{array}\tag{4}
$$

The second line is the budget as it is computed in practice: the change in storage minus the tendency accumulated under the reference dynamics. For conservation-form dynamics $F ( u ) \stackrel { - } { = } - \nabla \cdot f ( \stackrel { - } { u } )$ and the regional mass, the tendency is the net inflow $\dot { \mathbf { \Omega } } - \oint _ { \partial \Omega ^ { \prime } } \boldsymbol { f } \cdot \boldsymbol { n } \mathrm { d } \boldsymbol { S }$ , which recovers the familiar control-volume budget, storage change equals net inflow. Evaluating it requires $F$ along the whole path within the interval, so from grid states alone it is computed by quadrature and vanishes only up to quadrature error. The first line is what places budgets relative to $Q _ { 0 } { \mathrm { : } }$ : a budget is the residual averaged against the weight $\partial _ { x } E _ { j }$ over the interval.

Budgets are therefore necessary conditions for the reference dynamics, but not sufficient ones. They are necessary because $B _ { j }$ is built from $Q _ { 0 } \colon$ if a trajectory follows the reference dynamics, then $Q _ { 0 } [ x ] \equiv 0$ and every budget holds, so $\{ B _ { j } \} \preceq \dot { Q _ { 0 } }$ . They are not sufficient because a budget sees the residual only after it has been summed, over the time interval and over the part of the state that $E _ { j }$ measures. Errors that cancel in this sum ${ \bf g 0 }$ unnoticed. A transport that is too fast in the first half of a step and too slow in the second leaves the budgets over that step satisfied, and mass that moves to the wrong place inside $\Omega ^ { \prime }$ leaves the mass budget of $\Omega ^ { \prime }$ satisfied, although in both cases $Q _ { 0 } \neq 0$ . Only checking every possible quantity $E _ { j }$ over arbitrarily short intervals would recover $Q _ { 0 }$ , since a function whose weighted integrals over every interval all vanish is itself zero. With finitely many budgets at a fixed $\Delta t .$ , as in any practical evaluation, $\{ B _ { j } \} \prec Q _ { 0 }$

## A.2 Invariants

An invariant is a quantity $I _ { j }$ that the reference dynamics cannot change, such as the total mass, energy or momentum of a closed, unforced system. In terms of the gradient introduced for budgets, its tendency vanishes at every admissible state, $\langle \partial _ { x } I _ { j } ( x ) , F ( \dot { x } ) \rangle = 0 :$ the dynamics moves the state only in directions that leave $I _ { j }$ unchanged. An invariant is therefore a budget whose tendency term is identically zero. Taking $E _ { j } = I _ { j }$ in Eq. (4) over $[ t _ { 0 } , t ]$ , the budget reduces to a comparison of two snapshots,

$$
C _ { j } ( x _ { t } ) : = I _ { j } ( x _ { t } ) - I _ { j } ( x _ { t _ { 0 } } ) = \int _ { t _ { 0 } } ^ { t } \langle \partial _ { x } I _ { j } ( x ( s ) ) , Q _ { 0 } [ x ] ( s ) \rangle { \mathrm { d } } s .\tag{5}
$$

This is why invariants are cheap to check: they need only the states at grid times, not the reference tendency $F$ . The global mass budget of a closed domain is the invariant $\begin{array} { r } { I _ { j } ( u ) = \int _ { \Omega } } \end{array}$ u dz. Instantaneous constraints such as $\nabla \cdot u = 0$ fit the same form: in incompressible dynamics the divergence is itself conserved, so it stays zero if it is zero initially.

Invariants are weaker than budgets. Since an invariant is a budget, a trajectory that satisfies the budgets of $I _ { j }$ over consecutive intervals covering $\left[ t _ { 0 } , t \right]$ also satisfies $C _ { j } .$ , their sum, so $\{ { \check { C _ { j } } } \} \preceq \tilde { \{ B _ { j } \} }$ . The converse fails because an invariant sees only the part of the residual that changes $I _ { i } , \mathbf { A }$ model that follows the reference dynamics at the wrong speed, ${ \dot { x } } = \lambda F ( x )$ with $\lambda \neq 1$ , has residual $Q _ { 0 } = { \bar { ( } } \lambda - \mathrm { { \bar { 1 } } } ) F$ , which changes no invariant, so every $C _ { j }$ stays zero. $\mathbf { A }$ budget of a quantity that the dynamics does change still registers the error, since along this trajectory

$$
B _ { j } = ( \lambda - 1 ) \int _ { t } ^ { t + \Delta t } \langle \partial _ { x } E _ { j } , F \rangle \mathrm { d } t ^ { \prime } = \frac { \lambda - 1 } { \lambda } \big [ E _ { j } ( x _ { t + \Delta t } ) - E _ { j } ( x _ { t } ) \big ] ,
$$

which is non-zero whenever $E _ { j }$ changes over the interval. Hence $\{ C _ { j } \} \prec \{ B _ { j } \}$ as soon as the budgets include such a quantity; the harmonic oscillator below makes this concrete.

## A.3 Process diagnostics

A process diagnostic $\mathcal { P } _ { k }$ extracts one aspect of how the system evolves over a time window of length $L _ { ☉ }$ , such as the propagation speed of a wave, the rate at which a tracer is carried across a region, or the response to a change in forcing. Such diagnostics are standard in the evaluation of climate and weather models [Maloney et al., 2019] and have been applied to machine-learning weather models [Hakim and Masanam, 2024]. The corresponding relation compares this aspect with that of the reference trajectory from the same initial state,

$$
\begin{array} { r } { \mathcal { E } _ { k } ^ { \mathrm { d y n } } = { d _ { k } } \big ( \mathcal { P } _ { k } ( \widehat { x } _ { t : t + L } ) , \mathcal { P } _ { k } ( x _ { t : t + L } ) \big ) , } \end{array}
$$

which is zero whenever the two trajectories agree in the aspect that $\mathcal { P } _ { k }$ measures, and in particular whenever they coincide.

Process diagnostics are implied by $Q _ { 0 } . \mathrm { I f } Q _ { 0 } [ \widehat { x } ] \equiv 0$ and $\widehat { x } _ { t _ { 0 } } = x _ { t _ { 0 } } .$ , the prediction follows the reference dynamics from the reference initial state. When this determines a unique trajectory, as it does for Lipschitz $F ,$ whose tendency changes at most proportionally to a change of state, then ${ \widehat { x } } = x$ and every $\mathcal { E } _ { k } ^ { \mathrm { d y n } }$ vanishes, so $\{ \mathcal { E } _ { k } ^ { \mathrm { d y n } } \} \preceq Q _ { 0 }$ . The converse fails because each diagnostic keeps only one aspect of the trajectory and discards the rest, so a model can match every diagnostic in the family while $Q _ { 0 } \neq 0$ . A diagnostic of oscillation period, for instance, is blind to errors in amplitude, as the damped oscillator of the example below shows. Hence $\{ \mathcal { E } _ { k } ^ { \mathrm { d y n } } \} \prec Q _ { 0 }$

Process diagnostics also differ in kind from budgets and invariants. Budgets and invariants check that weighted sums of the residual $Q _ { 0 }$ are zero, whereas a process diagnostic checks an outcome of the dynamics, such as when a wave arrives. A residual can change that outcome while every weighted sum stays zero, and a residual that breaks a budget can leave the outcome unchanged. Neither kind of relation therefore implies the other, and the example below shows both cases.

## A.4 Example: harmonic oscillator

Consider the oscillator with state $x = ( q , p ) , F = ( p , - q )$ and energy $H = { \textstyle \frac { 1 } { 9 } } ( q ^ { 2 } + p ^ { 2 } )$ . From $x _ { t _ { 0 } } = ( 1 , 0 )$ the reference trajectory is $x ( t ) = ( \cos \tau , - \sin \tau ) , \tau = t - t _ { 0 }$ , a circle traversed with period 2π, and we take this period as the process diagnostic.

Suppose a surrogate model $\widehat { F } _ { \mathrm { s c a l e d } } ~ = ~ \lambda F , ~ \lambda ~ \neq ~ 1$ , traverses the same circle at a different speed, $\begin{array} { r l } { { \widehat x } ( t ) } & { { } = } \end{array}$ $( \cos \lambda \tau , - \sin \lambda \tau )$ . It preserves energy, phase-space volume and rotational symmetry exactly, and so passes every invariant, yet its period is $2 \pi / \lambda ,$ and the state error $\| \widehat { x } ( t ) - x ( t ) \| = 2 | \sin ( ( \lambda - 1 ) \tau / 2 )$ | reaches its maximum of 2 at $\tau = \pi / | \lambda - 1 |$ , when the prediction lies on the opposite side of the circle. The budget of $E _ { j } ( x ) = q$ detects this error, $\begin{array} { r } { B _ { j } = ( \lambda - 1 ) \int _ { t } ^ { t + \Delta t } \widehat { p } \mathrm { d } t ^ { \prime } \neq 0 } \end{array}$ for generic slabs.

Consider now a second surrogate, the damped model $\widehat { F } _ { \mathrm { d a m p e d } } ( x ) \ = \ F ( x ) - \gamma x , \gamma \ > \ 0$ , which gives ${ \widehat x } ( t ) =$ $e ^ { - \gamma \tau } ( \cos \tau , - \sin \tau )$ . It oscillates with exactly the reference period and passes the diagnostic, but its energy decays as $\begin{array} { r } { H ( \widehat { x } ( t ) ) = \frac { 1 } { 2 } e ^ { - 2 \gamma \tau } } \end{array}$ , so it fails the energy invariant and every budget built on it. The first model passes the invariants and fails the diagnostic, the second does the opposite, and neither satisfies $Q _ { 0 } \equiv 0$ . The single invariant H thus fixes the circle on which the trajectory lies but not the speed at which it is traversed.

More generally, m functionally independent invariants restrict $\delta F$ only to the tangent space of their joint level set, and even $n - 1$ of them leave $\delta F$ free along $F ,$ so invariants restrict where a trajectory lies but not how it moves along it. The residual penalised by a physics-informed loss [Raissi et al., 2019, Karniadakis et al., 2021] is in this sense the maximal element of the ladder, although it certifies ${ \widehat { F } } = F$ only on the states where it is evaluated.

## A.5 Remark on vector fields and flow maps.

The analysis above assumes that the learned map is the one-step flow of some vector field, $f _ { \theta } ( \widehat { x } _ { t } ) = \widehat { x } _ { t + \Delta t }$ with $\dot { \widehat { x } } = \widehat { F } ( \widehat { x } )$ in between. This is an idealisation: not every map arises in this way, and among diffeomorphisms of a compact manifold those that do are, in a precise sense, few [Palis, 1974]. A simple counterexample is the map $x \mapsto - x$ on the real line. Trajectories of a flow in one dimension cannot cross, so its one-step map preserves the order of points, whereas $x \mapsto - x$ reverses it. The assumption is mild when the map is close to the identity. For numerical one-step methods with a small step, backward error analysis shows that the method is the exact flow of a modified vector field up to an error exponentially small in $1 / \Delta t$ [Hairer et al., 2006]. Machine-learning emulators, however, often use large steps, for which no such guarantee is available. When the assumption fails, the relations above can still be evaluated with the finite difference $( \widehat { x } _ { t + \Delta t } - \widehat { x } _ { t } ) / \Delta t$ in place of ${ \dot { \widehat { x } } } ,$ at the cost of the discretisation error discussed in Section 2.

## B Hybrid and Physics-Informed Architectures

The tests of equation (3) are evaluated on realised trajectories, whereas architectural and training-time constraints act on the learned vector field $\widehat F$ or on the loss. Since a test sees the model only through $\delta F$ at visited states, equation (2), a constraint guarantees it by construction only if it forces the corresponding weighted error to vanish at every state, as $\langle \partial _ { x } I _ { j } ( x ) , \bar { \delta } F ( x ) \rangle = 0$ does for an invariant.

## B.1 Hybrid models

A differentiable-solver model [Kochkov et al., 2024] sets $\widehat { F } = F _ { \mathrm { r e s } } + N _ { \theta }$ , where $F _ { \mathrm { r e s } }$ discretises the known equations on the resolved scales and $N _ { \theta }$ is a learned closure for the unresolved ones. Splitting the reference as $F = F _ { \mathrm { r e s } } { \mathrm { \bar { + } } } F _ { \mathrm { s u b } } ,$ with $F _ { \mathrm { s u b } }$ the unresolved tendency plus the discretisation error, the model error is the closure error, $\delta F = N _ { \theta } - F _ { \mathrm { s u b } }$ If $F _ { \mathrm { r e s } }$ conserves an invariant $I _ { j } ,$ so does $F _ { \mathrm { s u b } } .$ and $\begin{array} { r } { C _ { j } = \int _ { t _ { 0 } } ^ { t } \langle \partial _ { x } I _ { j } , N _ { \theta } \rangle } \end{array}$ ds vanishes by construction whenever the closure cannot change $I _ { j } ,$ , as for total mass under a closure in flux form. A budget, $\begin{array} { r } { B _ { j } = \int _ { t } ^ { t + \Delta t } \langle \partial _ { x } E _ { j } , N _ { \theta } - F _ { \mathrm { s u b } } \rangle \mathrm d t ^ { \prime } } \end{array}$ vanishes only if the closure reproduces the unresolved tendency seen by $E _ { j }$ , such as the subgrid flux through the boundary of a region, which is learned rather than structural. This is the sense in which Section 2 cites hybrid models as physically consistent. By the argument of Appendix A, these invariants constrain $\delta F$ only along $\partial _ { x } I _ { j }$ , leaving the rest of the closure error, and with it the trajectory, free. Process diagnostics typically target the coupling between resolved dynamics and closure, as in wave–mean-flow interaction and the circulation response it drives, and so probe precisely the component no solver fixes. A hybrid model is thus expected to pass the constraints its structure enforces while remaining open at the process level, and the framework makes that expectation testable rather than assumed.

## B.2 Constrained architectures

Hard-constraint layers correct each predicted state so that chosen invariants or budgets hold, and pass those $C _ { j }$ or $B _ { j }$ exactly at grid times [Hansen et al., 2023], leaving every untested direction of $\delta F$ free. Equivariant architectures [Batzner et al., 2022] impose ${ \widehat F } ( g x ) = g { \widehat F } ( x )$ for a symmetry group, which excludes errors that break the symmetry but not those that respect it: in the oscillator of Appendix A, ${ \widehat F } = \lambda F$ is rotation-equivariant for every λ. Hamiltonian networks [Greydanus et al., 2019] write ${ \widehat { F } } = J \nabla { \widehat { H } }$ , with J the canonical symplectic matrix, and conserve the learned energy $\widehat { H }$ exactly, but the reference energy only if $\nabla H ^ { \top } J \nabla \widehat { H } = 0$ . Even then the trajectory is not fixed, since $\lambda F \overset { \cong } { = } J \nabla ( \lambda H )$ is Hamiltonian and conserves H.

## B.3 Physics-informed losses

These losses penalise the residual $Q _ { 0 }$ itself, the strongest test of equation (3), but only where they are evaluated [Karniadakis et al., 2021]. At a training state the model residual is $\delta F \bar { ( x ) }$ , so a small loss bounds $\| \delta \mathbf { \dot { F } } \| ^ { 2 }$ on average over $p _ { \mathrm { t r a i n } }$ , not on the states of a rollout, which drift away from $p _ { \mathrm { t r a i n } }$ as errors accumulate. Computed from grid states, the residual is moreover itself a coarsening of $Q _ { 0 }$

## C Planetary-Wave Diagnostics for Dynamical Fidelity

This appendix elaborates on the definitions of the zonal-mean and eddy quantities, then the Eliassen–Palm (EP) fluxes and lastly the Northern Annular Mode (NAM). Each diagnostic is well defined independently, however, during a sudden stratospheric warming (SSW), their expected temporal ordering links anomalous wave activity, disruption of the polar vortex, and the subsequent evolution of the circulation through the atmospheric column.

## C.1 Zonal Means and Eddy Covariances

The diagnostics are constructed from zonal wind u, meridional wind v, temperature T, and geopotential height Z, defined on a longitude–latitude–pressure grid. Longitude, latitude, pressure, and time are denoted by $\lambda , \phi , p ,$ and t, respectively. Pressure is used as the vertical coordinate, so decreasing p corresponds to increasing altitude.

Planetary waves appear as longitude-dependent disturbances on a longitude-independent background circulation. We therefore average over longitude while retaining latitude, pressure, and time. This decomposition gives the wave pattern around every latitude circle and atmospheric level at every time. Examining the wave-change across pressure can reveal whether a pattern remains confined to the troposphere or extends into the stratosphere whereas examining the wave-change across time can reveal how the wave propagates through time.

![](images/ec500f3588e94cdd86f6291b4cd840949d5ab20cb5f605f4f7a680c1562751c0.jpg)  
Figure 4: Schematic illustration of the zonal-mean decomposition and zonal eddy covariances. At a fixed latitude, pressure, and time, each atmospheric variable forms a longitude-dependent field around an east–west circle. Dashed lines show the zonal means, while the shaded departures are the eddies $u ^ { \prime } , v ^ { \prime } ,$ and $T ^ { \prime }$ . Multiplying the relevant departures longitude by longitude and averaging the products around the circle gives $\overline { { u ^ { \prime } v ^ { \prime } } }$ and $\overline { { v ^ { \prime } T ^ { \prime } } }$ . The faded globes indicate that the same calculation is repeated independently across latitude and pressure. The calculation is also repeated for every time-step but this is not included in the illustration above.

For $q \in \{ u , v , T \}$ , the discretized zonal mean with $N _ { \lambda }$ equally spaced longitude points becomes

$$
\overline { { { q } } } ( \phi _ { j } , p _ { k } , t _ { n } ) = \frac { 1 } { N _ { \lambda } } \sum _ { i = 1 } ^ { N _ { \lambda } } q ( \lambda _ { i } , \phi _ { j } , p _ { k } , t _ { n } ) .
$$

The remaining longitude-dependent component is

$$
q ^ { \prime } ( \lambda , \phi , p , t ) = q ( \lambda , \phi , p , t ) - \overline { { { q } } } ( \phi , p , t ) ,
$$

where $q ^ { \prime }$ collects all longitude-dependent departures from the zonal mean, from planetary-scale disturbances to smaller synoptic structures. Eddy refers specifically to this zonal-mean departure; no other mean–eddy decompositions are considered.

The dynamical importance of these eddies lies not only in their individual amplitudes or spatial scales, but in the transport of heat and momentum produced by their joint organization across variables. These transports redistribute heat between latitudes, help maintain and shift the jet streams and storm tracks, and mediate the wave forcing of the stratospheric polar vortex, making their faithful representation consequential for circulation changes such as SSWs. Crucially, no single-variable eddy field determines a transport: one variable specifies the anomalous motion, while another specifies the quantity carried by that motion.

We consider the two zonal eddy covariances

$$
\overline { { v ^ { \prime } T ^ { \prime } } } ( \phi , p , t ) = \frac { 1 } { N _ { \lambda } } \sum _ { i = 1 } ^ { N _ { \lambda } } v ^ { \prime } ( \lambda _ { i } , \phi , p , t ) T ^ { \prime } ( \lambda _ { i } , \phi , p , t ) ,
$$

and

$$
\overline { { u ^ { \prime } v ^ { \prime } } } ( \phi , p , t ) = \frac { 1 } { N _ { \lambda } } \sum _ { i = 1 } ^ { N _ { \lambda } } u ^ { \prime } ( \lambda _ { i } , \phi , p , t ) v ^ { \prime } ( \lambda _ { i } , \phi , p , t ) .
$$

These quantities describe the meridional eddy transport of heat and zonal momentum, respectively. Their values depend jointly on the amplitudes and relative longitudinal positions of the disturbances in each variable. Consequently, a model may produce plausible wave structure in $u , v ,$ and $T$ separately while suppressing or reversing the transport arising from their interaction.

## C.2 Fourier Interpretation

Figure 4 illustrates the zonal-mean decomposition and eddy covariances directly in longitude space. The same operations can be interpreted in wavenumber space using a Fourier transform. A Fourier transform does not alter the underlying field; it represents the longitude-dependent pattern as a sum of waves with different spatial scales, amplitudes, and longitudinal positions.

At fixed latitude $\phi ,$ pressure $p ,$ and time $t ,$ the eddy field $q ^ { \prime } ( \lambda , \phi , p , t )$ is periodic in longitude and has zero zonal mean. It can therefore be written as

$$
q ^ { \prime } ( \lambda , \phi , p , t ) = \sum _ { m = 1 } ^ { M } A _ { q , m } ( \phi , p , t ) \cos [ m \lambda - \alpha _ { q , m } ( \phi , p , t ) ] ,
$$

where longitude λ is expressed in radians. The zonal wavenumber m counts the number of complete wave cycles around a latitude circle: $m = 1$ represents one cycle around the globe, $m = 2$ represents two, and progressively larger values represent smaller longitudinal scales. The $m = 0$ component is absent because it is precisely the zonal mean removed in the definition of $\check { q ^ { \prime } }$ . The upper limit M is the largest wavenumber represented by the discrete longitude grid.

The coefficient $A _ { q , m } ( \phi , p , t )$ is the amplitude of wavenumber m and has the same physical units as $q .$ The phase $\alpha _ { q , m } ( \phi , p , t )$ determines the longitude of its crests and troughs. For the convention above, a crest occurs at

$$
\lambda _ { \mathrm { m a x } } = \frac { \alpha _ { q , m } } { m } \quad \left( \mathrm { m o d } \frac { 2 \pi } { m } \right) .
$$

Both amplitude and phase may vary with latitude, pressure, and time, allowing the strength and position of a given wave component to change throughout the atmospheric column and forecast trajectory.

The angular wavelength of wavenumber m is 2π $/ m ,$ while its physical wavelength along a latitude circle is

$$
L _ { m } ( \phi ) = \frac { 2 \pi a \cos \phi } { m } ,
$$

where a is Earth’s radius. Low zonal wavenumbers therefore correspond to planetary-scale structures, while higher wavenumbers correspond to progressively smaller disturbances. Low-wavenumber waves often provide the dominant contribution to stratospheric wave propagation and polar-vortex forcing, whereas higher-wavenumber disturbances are commonly associated with tropospheric weather systems, fronts, and storm tracks.

The Fourier representation also makes the role of cross-variable alignment in the eddy covariances explicit. Consider two eddy fields $a ^ { \prime }$ and $b ^ { \prime }$ containing the same zonal wavenumber m:

$$
a _ { m } ^ { \prime } = A _ { a , m } \cos ( m \lambda - \alpha _ { a , m } ) , \qquad b _ { m } ^ { \prime } = A _ { b , m } \cos ( m \lambda - \alpha _ { b , m } ) .
$$

Their zonally averaged product is

$$
\overline { { a _ { m } ^ { \prime } b _ { m } ^ { \prime } } } = \frac { 1 } { 2 } A _ { a , m } A _ { b , m } \cos ( \alpha _ { a , m } - \alpha _ { b , m } ) .
$$

The resulting covariance is controlled jointly by the amplitudes of the two waves and their relative phase. Equal phases give the largest positive covariance, a phase difference of $\pi / 2$ causes positive and negative products to cancel around the latitude circle, and a phase difference of π reverses the sign of the covariance. Two individually strong waves can therefore produce little or no net transport when their longitudinal structures are incorrectly aligned.

Fourier modes with different wavenumbers are orthogonal around a complete latitude circle, so their cross-products vanish under the zonal mean. For eddy fields containing several wavenumbers, the total covariance is therefore the sum of the matching-wavenumber contributions,

$$
\overline { { a ^ { \prime } b ^ { \prime } } } = \frac { 1 } { 2 } \sum _ { m = 1 } ^ { M } A _ { a , m } A _ { b , m } \cos ( \alpha _ { a , m } - \alpha _ { b , m } ) .
$$

This decomposition can be applied to $\overline { { v ^ { \prime } T ^ { \prime } } }$ or $\overline { { u ^ { \prime } v ^ { \prime } } }$ to determine which zonal scales contribute to the meridional transport of heat or momentum. It can therefore distinguish an error in wave amplitude from an error in cross-variable phase, and identify whether a forecast failure originates primarily from planetary-scale or smaller-scale disturbances. Unless stated otherwise, the diagnostics in the main text use the total covariance summed across all resolved zonal wavenumbers.

## C.3 Eliassen–Palm Flux and Wave Forcing

We combine the two eddy covariances into the quasi-geostrophic Eliassen–Palm (EP) flux [Edmon et al., 1980],

$$
\mathbf { F } ( \phi , p , t ) = ( F _ { \phi } , F _ { p } ) , \qquad F _ { \phi } \propto - \overline { { u ^ { \prime } v ^ { \prime } } } , \qquad F _ { p } \propto \overline { { v ^ { \prime } T ^ { \prime } } } .
$$

The connection between heat transport and vertical propagation follows from the geometry of pressure surfaces: warm layers increase the distance between neighbouring pressure levels, whereas cold layers decrease it. The alignment of these thickness anomalies with north–south wind anomalies therefore reveals how the wave pattern tilts between pressure levels, which determines its vertical direction of propagation.

The EP flux produces a vector at every latitude, pressure level, and time. Under the quasi-geostrophic Rossby-wave interpretation, $F _ { \phi }$ describes meridional propagation, while $F _ { p }$ describes propagation in pressure. Because pressure decreases with altitude, flux directed toward lower pressure corresponds to upward propagation.

The EP-flux vector shows where wave activity is directed, but not whether it changes the circulation through which it travels. To measure this interaction, we compare how much wave activity enters and leaves each small region of latitude–pressure space using the EP-flux divergence. Taking $F _ { p }$ positive upward, i.e. toward lower pressure,

$$
\mathbf { F } = a \cos \phi \left( - \overline { { u ^ { \prime } v ^ { \prime } } } , \ - \frac { f \overline { { v ^ { \prime } \theta ^ { \prime } } } } { \partial \overline { { \theta } } / \partial p } \right) , \qquad \nabla \cdot \mathbf { F } = \frac { 1 } { a \cos \phi } \frac { \partial ( F _ { \phi } \cos \phi ) } { \partial \phi } - \frac { \partial F _ { p } } { \partial p } ,
$$

where a is Earth’s radius, $f$ the Coriolis parameter, and θ potential temperature. Because $\partial \overline { { \theta } } / \partial p < 0 .$ , poleward heat transport gives $F _ { p } > 0 ,$ , consistent with $F _ { p } \propto \overline { { v ^ { \prime } T ^ { \prime } } }$ . The representations use $F _ { \phi } / ( a \cos \phi ) , F _ { p } / ( a \cos \phi )$ , and $D = ( a \cos \phi ) ^ { - 1 } \nabla \cdot \mathbf { F }$ , the implied acceleration of the zonal-mean zonal wind in m $\mathrm { { s } ^ { - 1 } \mathrm { { d a y } ^ { - 1 } } }$ , which is undefined at the pole. A non-zero divergence indicates that wave activity is deposited or removed and therefore transfers momentum to the zonal-mean wind. During an SSW, upward-propagating waves converge in the polar stratosphere, decelerating and potentially reversing the westerly polar-vortex winds.

## C.4 Northern Annular Mode

EP flux diagnoses how waves propagate and force the zonal-mean circulation. We use the Northern Annular Mode (NAM) to measure the large-scale circulation state that accompanies this forcing across the atmospheric column [Baldwin and Dunkerton, 2001].

Geopotential height $Z ( \lambda , \phi , p , t )$ is the physical height of a constant-pressure surface. We define its anomaly as

$$
Z ^ { \prime } ( \lambda , \phi , p , t ) = Z ( \lambda , \phi , p , t ) - Z _ { \mathrm { c l i m } } ( \lambda , \phi , p , d ( t ) ) ,
$$

where $Z _ { \mathrm { c l i m } }$ is the seasonally varying reference geopotential height and $d ( t )$ is the calendar day corresponding to time t. Thus, $Z ^ { \prime } > 0$ means that the pressure surface lies higher than expected for that location and time of year, whil $Z ^ { \prime } < 0$ means that it lies lower.

At each pressure level $p ,$ we identify the NAM pattern using principal component analysis. The resulting spatial component, $e _ { p } ( \lambda , \phi )$ , is conventionally called an empirical orthogonal function (EOF) in atmospheric science. We then measure how strongly the geopotential-height anomaly at time t resembles this pattern through the projection

$$
c ( p , t ) = \left. \sqrt { \cos \phi } Z ^ { \prime } ( \cdot , \cdot , p , t ) , e _ { p } \right. .
$$

Here, $\langle \cdot , \cdot \rangle$ is a dot product over the latitude–longitude grid, and $\scriptstyle { \sqrt { \cos \phi } }$ accounts for the smaller area represented by grid points closer to the pole. A large positive projection means that the anomaly resembles $e _ { p } ;$ a large negative projection means that it resembles the opposite circulation pattern.

The dimensionless NAM index is the standardized projection,

$$
N ( p , t ) = \frac { c ( p , t ) - \mu _ { c , p } } { \sigma _ { c , p } } ,
$$

where $\mu _ { c , p }$ and $\sigma _ { c , p }$ are the reference mean and standard deviation at pressure level $p .$ The sign convention is chosen so that positive NAM represents a strong polar vortex with anomalously low polar geopotential height, while negative NAM represents a weakened or disrupted vortex with anomalously high polar geopotential height.

The reference is fitted separately at each of the 37 ERA5 pressure levels on a $2 . 5 ^ { \circ }$ grid between $2 0 ^ { \circ }$ and $9 0 ^ { \circ } \mathrm { N } ,$ obtained by averaging daily-mean (00–18 UTC) geopotential height over blocks of $1 0 \times 1 0$ native grid points, using 1979–2023 with 29 February excluded. $Z _ { \mathrm { c l i m } }$ is the calendar-day mean, low-pass filtered with a 90-day second-order Butterworth filter applied forward and backward. $e _ { p }$ is the leading EOF of the cos ϕ-weighted, 90-day low-pass-filtered anomalies of November–April days, and $\mu _ { c , p }$ and $\sigma _ { c , p }$ are the mean and standard deviation of the projections of the unfiltered anomalies over all days of 1979–2023. The sign of $e _ { p }$ is chosen so that positive NAM corresponds to negative polar-cap $\mathrm { ( \geq 6 0 ^ { \circ } N ) }$ height anomalies. The index for 2024–2025 is obtained by projecting onto the same reference, and $N _ { 5 0 : 1 0 0 }$ is the mean of the 50, 70, and 100 hPa indices.

Computing $N ( p , t )$ at every pressure level and time produces a pressure–time section of the annular circulation state. Following an SSW, negative NAM values often first occur at low pressure (high altitude) in the stratosphere and subsequently appear at progressively higher pressure, corresponding to lower altitude.

## D Relationship Discovery and Selection

This appendix specifies the relationship-discovery procedure summarized in Section 4.

## D.1 Data and cohort

Field diagnostics are computed from the ARCO-ERA5 archive at $0 . 2 5 ^ { \circ }$ resolution on 13 pressure levels (50–1000 hPa) at 00, 06, 12, and 18 UTC, and averaged over twelve $5 ^ { \circ }$ latitude bands between $3 0 ^ { \circ }$ and ${ \bf { \bar { 9 0 } } } ^ { \circ } { \bf N } .$ . They are standardized separately for each calendar day and synoptic hour with a 1979–2017 climatology, whose mean is smoothed with a 31- day running mean and whose standard deviation is not smoothed. The NAM uses all 37 pressure levels (Appendix C.4). The field diagnostics are extracted for November–April of 1979–2025.

An initialization t<sub>c</sub> (00 UTC on a day from November to March) is admissible when its complete 40-day precursor history $( t _ { c } - 4 0 \mathrm { d }$ to $t _ { c } - 6 \mathrm { h }$ ) lies within the extracted months and its subsequent 10-day NAM trajectory is available. Admissible initializations therefore fall between 11 December and 31 March. Winters are indexed by the year of their January, giving 46 complete winters from 1980 to 2025 and $5 , 1 1 8$ admissible dates. Restricting these dates to $| N _ { 5 0 : 1 0 0 } ( t _ { c } ) | \le 0 . 3$ yields 772 initializations (15.1%), with 2 to 47 per winter (mean 16.8).

## D.2 Candidate source space

Field representations are aggregated over five latitude regions,

$$
3 0 - 6 0 ^ { \circ } \mathrm { N } , \quad 3 8 - 6 8 ^ { \circ } \mathrm { N } , \quad 4 5 - 7 5 ^ { \circ } \mathrm { N } , \quad 5 2 - 8 2 ^ { \circ } \mathrm { N } , \quad 6 0 - 9 0 ^ { \circ } \mathrm { N } ,
$$

using the constituent $5 ^ { \circ }$ latitude bands falling within each region. For all non-divergence field representations, the four pressure regions are

$$
\{ 5 0 , 1 0 0 , 1 5 0 \} , \quad \{ 2 0 0 , 2 5 0 , 3 0 0 \} , \quad \{ 4 0 0 , 5 0 0 \} , \quad \{ 6 0 0 , 7 0 0 , 8 5 0 \} \ : \ : \mathrm { h P a } ,
$$

while EP-divergence representations use {600, 700} hPa in the lowest region. Past NAM uses five pressure regions defined by the edges {1, 50, 150, 400, 700, 1000} hPa. Temporal windows are $W \in \{ 2 , 3 , 5 , 7 , \bar { 1 0 } , 2 0 , 3 0 \}$ days, and the five operators are

$$
\Psi = \{ \mathrm { m e a n , m a x i m u m , m i n i m u m , p o s i t i v e i m p u l s e , n e g a t i v e i m p u l s e } \} .
$$

For samples $z _ { s }$ at sampling rate ν per day, the impulse operators are

$$
\psi _ { W } ^ { + } ( z ) = \frac { 1 } { \nu } \sum _ { s \in W } \operatorname* { m a x } ( z _ { s } , 0 ) , \qquad \psi _ { W } ^ { - } ( z ) = \frac { 1 } { \nu } \sum _ { s \in W } \operatorname* { m a x } ( - z _ { s } , 0 ) .
$$

All windows terminate strictly before $t _ { c } .$ . Combining 11 field representations, five latitude regions, four pressure regions, seven windows, and five operators gives 7,700 field candidates; the five NAM pressure regions contribute a further 175, for a total of 7,875.

## D.3 Split criterion and eligibility

For each candidate source variable X, we search exhaustively over all threshold splits of the initializations being split: the neutral cohort $\tau$ at the first level, and a fixed first-level branch at the second; below, $\tau$ denotes this set in either case. Let $x _ { ( 1 ) } \leq \cdots \leq x _ { ( n ) }$ denote the sorted source values. The candidate threshold set is

$$
\Gamma ( X ) = \left\{ \frac { x _ { ( i ) } + x _ { ( i + 1 ) } } { 2 } : x _ { ( i ) } < x _ { ( i + 1 ) } \right\} ,\tag{6}
$$

so thresholds are placed only between distinct adjacent source values. Each $c \in \Gamma ( X )$ partitions the initialization dates into two leaves,

$$
\begin{array} { r } { \mathcal { T } _ { 0 } ( c ) = \{ i : X _ { i } \leq c \} , \qquad \mathcal { T } _ { 1 } ( c ) = \{ i : X _ { i } > c \} . } \end{array}\tag{7}
$$

Each initialization $i \in \mathcal { T }$ carries a weight $w _ { i } \propto 1 / n _ { T } ( \omega _ { i } )$ , normalized so that $\textstyle \sum _ { i \in T } w _ { i } = 1$ , where $\omega _ { i }$ is its winter and $n _ { T } ( \omega )$ is the number of dates of $\tau$ in winter ω. Every winter therefore contributes the same total weight, however many of its dates enter the cohort (between 2 and 47). For $\tau ^ { \prime } \subseteq \tau$ , let $\begin{array} { r } { \eta ( \mathcal { T } ^ { \prime } ) = \sum _ { i \in \mathcal { T } } } \end{array}$ <sub>′</sub> w<sub>i</sub> denote its summed weight and $\begin{array} { r } { \mu _ { \tau } ( \tau ^ { \prime } ) = \eta ( \tau ^ { \prime } ) ^ { - 1 } \sum _ { i \in \mathcal { T } ^ { \prime } } w _ { i } Y _ { i , \tau } } \end{array}$ its weighted mean NAM at lead day τ, and let

$$
\mathcal { V } ( \mathcal { T } ^ { \prime } ) = \frac { 1 } { 5 } \sum _ { \tau = 6 } ^ { 1 0 } \frac { 1 } { \eta ( \mathcal { T } ^ { \prime } ) } \sum _ { i \in \mathcal { T } ^ { \prime } } w _ { i } \big ( Y _ { i , \tau } - \mu _ { \tau } ( \mathcal { T } ^ { \prime } ) \big ) ^ { 2 }\tag{8}
$$

be the weighted variance of the future 50–100 hPa NAM, averaged over the scoring period (lead days 6–10). The fraction of trajectory variance explained (FVE) by threshold c is

$$
\mathrm { F V E } ( c ) = 1 - \frac { \sum _ { b \in \{ 0 , 1 \} } \frac { \eta ( \mathcal T _ { b } ( c ) ) } { \eta ( T ) } \mathcal V ( \mathcal T _ { b } ( c ) ) } { \mathcal V ( T ) } .\tag{9}
$$

The weights are computed for the node being split: the full cohort at the first level, and the fixed branch at the second level. They depend only on the initialization dates, so they are identical in every permutation null universe. In the stability refits, which add or remove whole winters, each remaining winter also keeps equal weight.

The selected threshold maximizes FVE over the midpoints that leave at least $n _ { \mathrm { m i n } }$ dates in each child,

$$
c ^ { \star } = \arg \operatorname* { m a x } _ { c \in \Gamma ( X ) : \ : | \mathcal { T } _ { 0 } ( c ) | , | \mathcal { T } _ { 1 } ( c ) | \geq n _ { \operatorname* { m i n } } } \mathrm { F V E } ( c ) ,\tag{10}
$$

with $n _ { \mathrm { m i n } } = 5 0$ at the first level and $n _ { \mathrm { m i n } } = 3 0$ at the second. Of the two children of $c ^ { \star }$ , the selected child $\mathcal { T } ^ { \star }$ is the one whose weighted day 6–10 mean

$$
{ \bar { \mu } } ( { \mathcal { T } } ^ { \prime } ) = \frac { 1 } { 5 } \sum _ { \tau = 6 } ^ { 1 0 } \mu _ { \tau } ( { \mathcal { T } } ^ { \prime } )\tag{11}
$$

lies furthest from $\bar { \mu } ( \tau )$ . Because

$$
\eta ( \mathcal { T } _ { 0 } ) \bar { \mu } ( \mathcal { T } _ { 0 } ) + \eta ( \mathcal { T } _ { 1 } ) \bar { \mu } ( \mathcal { T } _ { 1 } ) = \eta ( \mathcal { T } ) \bar { \mu } ( \mathcal { T } ) ,\tag{12}
$$

this is the child with the smaller summed weight.

Two winter-coverage requirements prevent a high FVE from being obtained by isolating behaviour confined to one or a few winters. They are applied to the optimized split, and a candidate that fails them is discarded rather than refitted. At the first level, the selected child must span at least five distinct winters; both children of every retained first-level split then define branches for the second-level search. At the second level, the selected child must again span at least five winters, and both children must each span at least eight.

The split search is exhaustive: every eligible midpoint is evaluated, rather than selecting thresholds from a fixed grid or a predefined set of quantiles. Consequently, the reported FVE for a candidate source is the maximum trajectory separation attainable by a single threshold subject to the minimum child size.

## D.4 Winter-level stability filtering

Maximizing FVE on the complete cohort can favor relationships whose apparent separation depends on the particular winters used for discovery. We therefore apply a second-stage stability filter that asks whether the same candidate source repeatedly produces a separation in the same direction when thresholds are estimated from one set of winters and evaluated on different winters.

We generate 1,000 repeated 70/30 train–held-out splits of the winters represented in the node (the 46 winters of the cohort at the first level, or those of the branch at the second), shared by all candidates in that node. The resampling unit is the winter: all initialization dates belonging to a given winter are assigned jointly to either the training or held-out subset. This preserves the temporal dependence among initialization dates from the same winter and prevents densely sampled winters from being treated as collections of independent observations.

The candidate source specification itself is fixed throughout this procedure: its representation, spatial region, temporal window, and temporal operator are not reselected. For resample $s ,$ only its threshold is re-estimated. Using the training winters, we repeat the threshold search described above, with the same minimum child size and five-winter requirement on the selected child,

$$
c _ { s } ^ { \star } = \arg \operatorname* { m a x } _ { c } \mathrm { F V E } _ { \mathrm { t r a i n } , s } ( c ) .\tag{13}
$$

The resulting threshold $c _ { s } ^ { \star }$ is then transferred unchanged to the held-out winters. Thus, the held-out data play no role in choosing either the candidate source variable or its threshold.

Let

$$
d _ { \mathrm { f u l l } } = \mathrm { s i g n } \big ( \bar { \mu } ( T ^ { \star } ) - \bar { \mu } ( T ) \big )\tag{14}
$$

be the direction in which the selected child of the full fit shifts the day 6–10 NAM. In resample s, the training refit determines both $c _ { s } ^ { \star }$ and which of its sides is selected. Applying this rule unchanged to the held-out dates $\mathcal { T } _ { s } ^ { \mathrm { h e l d } }$ gives their selected child $\mathcal { T } _ { s } ^ { \mathrm { h e l d } , \star }$ and

$$
d _ { s } ^ { \mathrm { h e l d } } = \mathrm { s i g n } \big ( \bar { \mu } ( \mathcal { T } _ { s } ^ { \mathrm { h e l d , \star } } ) - \bar { \mu } ( \mathcal { T } _ { s } ^ { \mathrm { h e l d } } ) \big ) ,\tag{15}
$$

with weights recomputed on $\mathcal { T } _ { s } ^ { \mathrm { h e l d } }$ . A resample is valid when the training refit has an admissible split and both held-out children are non-empty. With $ { S _ { \mathrm { v a l i d } } }$ the set of valid resamples, we define directional stability as

$$
S _ { \mathrm { d i r } } = \frac { 1 } { | S _ { \mathrm { v a l i d } } | } \sum _ { s \in S _ { \mathrm { v a l i d } } } \mathbb { I } \left[ d _ { s } ^ { \mathrm { h e l d } } = d _ { \mathrm { f u l l } } \right] .\tag{16}
$$

This criterion requires the direction of the relationship to reproduce in at least 95% of the repeated held-out-winter evaluations, despite re-estimation of the threshold from different training winters. Importantly, the 95% threshold is an operational reproducibility criterion, not a p-value, confidence level, or test of statistical significance. Its purpose is to remove relationships whose fitted effect is highly sensitive to which winters are available for estimation. Chance discovery and multiplicity across the large candidate catalogue are handled separately by the cross-winter permutation-FDR procedure described below. All 283 retained first-level relationships have at least 999 valid resamples. Small second-level branches admit fewer valid refits: the median over the 672 retained second-level relationships is 998, but 61 have fewer than 500 and 23 fewer than 20.

## D.5 Permutation-FDR calibration

Directional stability removes relationships that fail to reproduce across held-out winters, but it does not account for the multiplicity induced by searching a large catalogue of candidate relationships. Even under no genuine precursor–future association, an exhaustive search over thousands of representations, spatial regions, temporal windows, operators, and thresholds may produce apparently strong and stable relationships by chance. We therefore calibrate the full discovery procedure against $B = 1 0 0$ permutation null universes.

Each null universe is constructed by reassigning complete future NAM trajectories across winters. Let $Y _ { i } \in \mathbb { R } ^ { K }$ denote the full future NAM trajectory associated with initialization i. In permutation universe $b ,$ the precursor variables and initialization dates remain unchanged, while the target becomes

$$
Y _ { i } ^ { ( b ) } = Y _ { \pi _ { b } ( i ) } ,\tag{17}
$$

where $\pi _ { b }$ is constrained to assign trajectories from different winters. Each $\pi _ { b }$ is a random permutation of the 772 initializations in which no initialization receives a trajectory from its own winter; the second-level search uses the same 100 permutations, restricted to each branch. The complete trajectory is reassigned as a single vector rather than permuting individual lead times. This preserves the temporal dependence within each future trajectory while breaking its association with the precursor conditions from which it originally evolved.

For every permutation universe, we repeat the same discovery procedure applied to the real ERA5 data: all candidate sources are searched, thresholds are optimized subject to the same eligibility criteria, and the same 1,000 winter-level stability procedure and $S _ { \mathrm { d i r } } \geq 0 . 9 5$ criterion are applied. A permutation universe therefore represents one complete realization of the relationship-discovery pipeline under a null in which precursor and future trajectories are disconnected.

For an FVE threshold t, let

$$
R ( t ) = \# \left\{ { \mathrm { r e a l ~ s t a b i l i t y - p a s s i n g ~ r e l a t i o n s h i p s ~ w i t h ~ F V E } } \geq t \right\} ,\tag{18}
$$

and let

$$
N _ { b } ( t ) = \# \left\{ \mathrm { s t a b i l i t y - p a s s i n g ~ r e l a t i o n s h i p s ~ i n ~ n u l l ~ u n i v e r s e } b \mathrm  ~ w i t h ~ F V E \geq \it t \right\} .\tag{19}
$$

The expected number of discoveries produced by the same search under the null is estimated by averaging across permutation universes,

$$
\overline { { N } } ( t ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } N _ { b } ( t ) ,\tag{20}
$$

giving the empirical false-discovery-rate (FDR) estimate

$$
{ \widehat { \mathrm { F D R } } } ( t ) = { \frac { { \overline { { N } } } ( t ) } { R ( t ) } } = { \frac { B ^ { - 1 } \sum _ { b = 1 } ^ { B } N _ { b } ( t ) } { R ( t ) } } .\tag{21}
$$

Thus, $\widehat { \mathrm { F D R } } ( t )$ compares the number of relationships retained from the real catalogue with the number expected to survive the complete search-and-stability pipeline when the precursor–future association has been destroyed.

Because only $B = 1 0 0$ null universes are available, we additionally quantify uncertainty in this estimate by bootstrapping the permutation universes. Each bootstrap replicate resamples the B complete null universes with replacement, recomputes $\overline { { N } } ( t )$ , and hence recomputes $\widehat { \mathrm { F D R } } ( t )$ . We use the upper end of the 95% bootstrap percentile interval, the 97.5th percentile over 20,000 bootstrap replicates, as a conservative calibration criterion and retain the smallest FVE threshold $t ^ { \star }$ satisfying

$$
q _ { 0 . 9 7 5 } \biggl [ \widehat { \mathrm { F D R } } ( t ^ { \star } ) \biggr ] \leq 1 0 ^ { - 3 } ,\tag{22}
$$

where $q _ { 0 . 9 7 5 }$ denotes the 97.5th percentile over the bootstrap replicates.

The permutation universes, rather than individual null candidates, are the resampling units in this calibration: each universe represents one complete null realization of the entire catalogue search. The resulting threshold therefore bounds the estimated proportion of retained relationships that the full search produces under the cross-winter null, conditional on having already passed the winter-level reproducibility filter.

## D.6 First- and second-level discovery

At the first level, each of the 7,875 candidate source variables is evaluated independently against the same future NAM target. For every source, the best eligible threshold is selected by maximizing FVE as described above, after which the relationship is subjected to the 1,000-resample winter-level stability procedure. Of the 7,875 candidates, 295 satisfy

$$
S _ { \mathrm { d i r } } \geq 0 . 9 5 .\tag{23}
$$

These stability-passing relationships are then calibrated against the first-level permutation null. The smallest FVE threshold whose upper bootstrap bound on the empirical FDR (the 97.5th percentile over 20,000 replicates) is at most $1 0 ^ { - 3 }$ is approximately

$$
t _ { 1 } ^ { \star } \simeq 0 . 0 5 0 4 ,\tag{24}
$$

leaving 283 retained first-level relationships.

Each retained first-level relationship consists of a source specification, an optimized threshold, and the resulting partition of the 772 initialization dates into two child leaves. These quantities are frozen after first-level selection. Since every retained split defines two branches, the 283 relationships produce

$$
2 \times 2 8 3 = 5 6 6\tag{25}
$$

fixed first-level branches.

Second-level discovery asks whether additional precursor information distinguishes future NAM evolution conditional on membership in one of these first-level branches. Within each of the 566 fixed branches, we therefore search exhaustively over all second-source variables from the same 7,875-variable catalogue, subject to the second-level eligibility requirements. The first-level source, threshold, and branch membership remain fixed throughout; only the second-level source and its threshold are optimized.

Across all branches, this exhaustive conditional search evaluates 4,358,681 eligible second-level candidate relationships. Each candidate is then subjected to the same winter-level threshold re-estimation and directional stability procedure used at the first level. Of these candidates, 71,494 satisfy $S _ { \mathrm { d i r } } \geq 0 . 9 5$

The second-level permutation analysis is conditional on the first-level structure discovered in ERA5. In every null universe, we therefore retain exactly the same 283 first-level relationships, their thresholds, and the resulting 566 branch memberships. The first-level search is not repeated under the null. Future NAM trajectories are permuted as described above, and the complete second-level search and stability filtering are rerun within these fixed branches. The resulting null therefore asks whether apparent additional predictive structure of comparable strength could arise by chance once the first-level partition has already been fixed.

Applying the same permutation-FDR criterion to the second-level search gives a conditional-FVE threshold of approxi mately

$$
t _ { 2 } ^ { \star } \simeq 0 . 3 1 0 2 ,\tag{26}
$$

retaining 672 second-level relationships. These relationships define the conditional predictive structures used for subsequent model evaluation.

## D.7 Retained relationship panel

The 672 statistically retained second-level relationships form the primary evaluation panel. Discovery is completed entirely from the reference ERA5 trajectories before any forecast model is evaluated. For each retained relationship, the first-level source and threshold, first-level branch, second-level source and threshold, selected terminal leaf, and initialization dates belonging to that leaf are therefore frozen.

When evaluating GraphCast, FengWu, and Pangu-Weather, no representation, spatial region, temporal window, operator, threshold, branch assignment, or case membership is re-estimated from the model forecasts. Each model is instead evaluated on the same ERA5-defined conditional subset. Differences between the reference and model-generated future NAM distributions therefore measure whether the model reproduces the consequence of a relationship identified independently in the reference system, rather than whether the model can define an alternative partition that better fit its own dynamics.

The exhaustive catalogue nevertheless produces some closely related retained relationships. For example, two relationships may describe the same underlying configuration while differing only in latitude or pressure region, temporal window, or aggregation operator. To assess sensitivity to this redundancy, we group the retained second-level relationships within each fixed first-level branch. Two relationships in the same branch are linked when their second-level sources share a representation and differ in exactly one of latitude region, pressure region, temporal window, or operator. Motifs are the connected components of these links, so relationships joined by a chain of such single-dimension differences share a motif, and relationships in different branches are never grouped. This rule reduces the 672 retained relationships to 287 motifs; 179 contain a single relationship, and the largest contains 39.

The 672 raw relationships remain the primary evaluation panel because they are the direct output of the statistical selection procedure. As a sensitivity analysis, we first average results among relationships belonging to the same motif and then weight the resulting 287 motifs equally. This prevents parent branches containing many closely related catalogue variants from receiving disproportionate weight. The collapse should be interpreted only as a catalogue-level redundancy correction: the resulting 287 motifs are not assumed to be statistically independent or to correspond to 287 distinct physical mechanisms.

Table 2 compares the two weightings over lead days 6–10. The model ranking is unchanged under both: Pangu-Weather has the lowest conditional-mean and conditional-variance errors, and FengWu the lowest T, u, and v MAE; the conventional errors change by at most 0.02. The main difference is FengWu’s conditional-mean error, which rises from 0.148 to 0.169 and nearly reaches GraphCast’s 0.171. FengWu’s relationship-level advantage over GraphCast is concentrated in the ten largest motifs. These hold 191 relationships, 180 of them in branches of 600–850 hPa zonal-mean u at $3 8 \mathrm { - } 6 8 ^ { \circ } \mathrm { N } ,$ and there FengWu’s conditional-mean error is 0.10 against GraphCast’s 0.16. Among the 179 single-relationship motifs, GraphCast has the lower error (0.16 against 0.18).

Table 2: Lead-day 6–10 errors averaged over the 672 retained relationships and over the 287 motifs (relationships averaged within each motif, motifs weighted equally). T, u, and v are forecast MAE in K and m s<sup>−1</sup>.
<table><tr><td>Model</td><td>Average over</td><td> $E _ { \mu }$ </td><td> $E _ { \sigma ^ { 2 } }$ </td><td> $T$ </td><td>u</td><td>v</td></tr><tr><td rowspan="2">GraphCast</td><td>672 relationships</td><td>0.173</td><td>0.195</td><td>2.290</td><td>5.729</td><td>5.768</td></tr><tr><td>287 motifs</td><td>0.171</td><td>0.179</td><td>2.285</td><td>5.717</td><td>5.748</td></tr><tr><td rowspan="2">FengWu</td><td>672 relationships</td><td>0.148</td><td>0.210</td><td>1.762</td><td>4.488</td><td>4.558</td></tr><tr><td>287 motifs</td><td>0.169</td><td>0.205</td><td>1.765</td><td>4.489</td><td>4.556</td></tr><tr><td rowspan="2">Pangu-Weather</td><td>672 relationships</td><td>0.091</td><td>0.107</td><td>2.175</td><td>5.494</td><td>5.581</td></tr><tr><td>287 motifs</td><td>0.090</td><td>0.104</td><td>2.180</td><td>5.498</td><td>5.579</td></tr></table>

## E Machine learning weather models

We evaluate three published, pretrained machine-learning weather forecasting systems: GraphCast (operational checkpoint), Pangu-Weather (6-hour configuration), and FengWu (operational checkpoint). All forecasts are generated through NVIDIA’s earth2studio (v0.13.0) inference framework and initialized from the same ERA5 source. The models retain their native input requirements and inference procedures, but are evaluated using a common forecast horizon, output cadence, variable set, and verification pipeline.

## E.1 GraphCast (operational)

GraphCast is a graph-neural-network weather forecasting model that represents global atmospheric interactions using an internal multimesh graph while ingesting and producing fields on a regular latitude–longitude grid [Lam et al., 2023]. We use the published operational checkpoint at $0 . 2 5 ^ { \circ }$ horizontal resolution with 13 pressure levels and a native 6-hour forecast step. This checkpoint was pretrained on ERA5 over 1979–2017 and subsequently fine-tuned on operational IFS HRES analyses over 2016–2021.

## E.2 Pangu-Weather (6-hour)

Pangu-Weather is a three-dimensional Earth-Specific Transformer designed for global medium-range weather prediction [Bi et al., 2023]. The published system comprises networks associated with different forecast intervals. We use the earth2studio 6-hour configuration, which combines the published 24-hour and 6-hour Pangu-Weather networks to produce forecasts at 6-hour intervals throughout the rollout. The model operates at $0 . 2 5 ^ { \circ }$ horizontal resolution on 13 pressure levels. The published Pangu-Weather models were trained using ERA5 data from 1979–2017.

## E.3 FengWu (operational)

FengWu is a multimodal, multitask transformer for global medium-range weather forecasting [Chen et al., 2025]. We use the fengwu\_v1 checkpoint released by the authors and distributed through earth2studio: a single autoregressive model with a native 6-hour forecast step, operating at $0 . 2 5 ^ { \circ }$ resolution on 13 pressure levels with 69 input variables. It was trained on ERA5 over 1979–2017 and does not include the subsequent transfer learning on ECMWF operational analyses (2017–2021) used for the operationally deployed FengWu.

## E.4 Common forecast protocol

All three models are initialized from the Analysis-Ready, Cloud-Optimized (ARCO) ERA5 archive, which provides hourly fields at 0.25<sup>◦</sup> resolution on 37 pressure levels. Forecasts are initialized independently for each of the 772 neutral-NAM dates defined in Section 4.1. Only variables and pressure levels required by the respective native model inputs are extracted from ERA5.

Each initialization is propagated autoregressively for 11 days (264 h), producing forecasts at

$$
\{ 0 , 6 , 1 2 , \ldots , 2 6 4 \} \mathrm { h } .
$$

The resulting 6-hourly outputs are subsequently grouped by forecast day for the analyses reported in this work.

Evaluation uses the 13 pressure levels shared by all three forecast systems,

$$
\{ 5 0 , ~ 1 0 0 , ~ 1 5 0 , ~ 2 0 0 , ~ 2 5 0 , ~ 3 0 0 , ~ 4 0 0 , ~ 5 0 0 , ~ 6 0 0 , ~ 7 0 0 , ~ 8 5 0 , ~ 9 2 5 , ~ 1 0 0 0 \} ~ \mathrm { h P a } ,
$$

and the implementation checks that every model outputs $T , u , v ,$ , and z on all of them. Forecast lead day d is the calendar day $t _ { c } + d ,$ i.e. the four forecasts valid at 00, 06, 12, and 18 UTC on that day; lead days 1–10 are evaluated.

Two quantities are computed from every rollout. First, conventional forecast errors of T, u, and v are evaluated against a $1 ^ { \circ }$ ERA5 verification archive. Model output on the native $0 . 2 5 ^ { \circ }$ grid is subsampled to every fourth grid point, so verification points coincide with native model grid points rather than being interpolated. The daily MAE is the unweighted mean absolute error over all $1 ^ { \circ }$ grid points (without area weighting), the 13 pressure levels, and the four 6-hourly forecasts of the day. Within a terminal leaf it is averaged over initializations with the same winter weights as $\mu _ { m , r }$ and $\sigma _ { m , r } ^ { 2 }$ (Section 4.3).

Second, the forecast NAM is obtained by averaging the four 6-hourly geopotential fields of each day, converting to height, averaging to the $2 . 5 ^ { \circ }$ reference grid over blocks of $1 0 \times 1 0$ grid points, and projecting onto the ERA5 reference EOFs using the ERA5 climatology and normalization (Appendix C.4). Because 70 hPa is not among the models’ output levels, the model $N _ { 5 0 : 1 0 0 }$ is the mean of the 50 and 100 hPa indices, whereas the ERA5 reference averages 50, 70, and 100 hPa. Using the 50/100 hPa mean for ERA5 as well changes the lead-day 6–10 $E _ { \mu }$ and $E _ { \sigma ^ { 2 } }$ of Table 1 by at most 0.004 and 0.007, respectively, and leaves the model ranking unchanged. The NAM reference contains no 29 February. Model lead days falling on that date therefore have no NAM value and are omitted from that day’s statistics, while the ERA5 target on 29 February is interpolated from the neighbouring days.

All precursor sources are computed from ERA5: every relationship is defined by ERA5 histories before $t _ { c } ,$ and the models enter the evaluation only through their NAM forecasts and forecast errors.

## E.5 Forecast coverage

All $7 7 2 \times 3 = 2 { , } 3 1 6$ model–initialization rollouts are complete for lead days 1–10, and no forecast value is masked, clipped, or imputed. The largest absolute forecast $N _ { 5 0 : 1 0 0 }$ over the cohort is 3.51 for GraphCast, 3.71 for FengWu, and 3.89 for Pangu-Weather.

## E.6 NAM forecast error

To separate conditional fidelity from forecast accuracy on the target variable itself, we also compute the plain NAM forecast error, the winter-weighted mean absolute difference between forecast and ERA5 $N _ { 5 0 : 1 0 0 }$ over lead days 6–10. Over all 772 initializations it is 0.453 for GraphCast, 0.294 for FengWu, and 0.275 for Pangu-Weather (0.494, 0.337, and 0.290 averaged over the 672 terminal leaves). Pangu-Weather therefore also has the lowest NAM forecast error, but its advantage over FengWu is far smaller than in conditional-mean error (0.091 against 0.148) or conditional-variance error (0.107 against 0.210). NAM forecast error also does not reproduce the relationship-level differences: FengWu has a lower NAM forecast error than GraphCast in every row of Table 1, yet GraphCast has the lower conditional-mean error in five of them, including strong negative shifts (NAM forecast error 0.590 against 0.456; conditional-mean error 0.275 against 0.329). GraphCast also has the smallest unconditional mean error over all initializations (0.010, against 0.067 for FengWu and 0.032 for Pangu-Weather) but the largest conditional-mean error, so its errors appear only once the initializations are conditioned on the discovered precursors. Using the 50/100 hPa mean for ERA5 changes these values by at most 0.004. This is an empirical counterpart of Appendix A: agreement in aggregate statistics does not imply agreement in the conditional evolution.

To give these errors a scale, we score two reference forecasts that carry no information about the precursors in the same way: climatology, which predicts the cohort-mean ERA5 trajectory for every initialization, and persistence of the initial ERA5 $N _ { 5 0 : 1 0 0 }$ . Their NAM forecast errors over all initializations are 0.954 and 0.937, and averaged over the 672 terminal leaves their conditional-mean and conditional-variance errors are 0.714 and 0.668, and 1.172 and 1.144, several times those of all three models. The exceptions are the weak-shift rows of Table 1, whose leaves by construction remain close to the cohort mean: there the references reach conditional-mean errors comparable to the models’ (for weak negative shifts, 0.099 for climatology and 0.075 for persistence against 0.078–0.128), while their conditional-variance errors remain 5–10 times larger, so these rows mainly test the conditional variance.

## E.7 Training-period overlap with the evaluation cohort

The reference cohort spans winters from 1979–2025 and therefore overlaps the training periods of the published forecast systems. GraphCast’s operational checkpoint was pretrained on ERA5 over 1979–2017 and subsequently fine-tuned on IFS HRES operational analyses over 2016–2021. The published Pangu-Weather and FengWu models were trained on ERA5 over 1979–2017. Thus, a substantial fraction of the 772 initialization dates lies within periods used in the development or training of these models, whereas later years provide temporally out-of-sample cases. Of the 772 initializations, 648 fall in 2017 or earlier, within the ERA5 training period of all three models; 66 fall in 2018–2021, within GraphCast’s HRES fine-tuning period; and 58 fall in 2022–2025.

The purpose of the present experiment is therefore not to construct a strictly held-out benchmark of general forecast skill. Instead, all three models are evaluated on the same reference-defined dynamical relationships across the full available cohort. The overlap should nevertheless be kept in mind when interpreting absolute forecast performance.