# FIDE: FIBER-INFORMED DIFFERENTIABLE EXPERI-MENTAL DESIGN FOR REFERENCE-RELATIVE LAW DYNAMICS

Pietro Zanotta & Jan Drgo´ naˇ   
Department of Civil and Systems Engineering   
Johns Hopkins University   
Baltimore, MD, USA   
pzanott1@jh.edu

## ABSTRACT

Aggregate measurements generally leave an evolving population law underdetermined, so different measurement systems can be equally adequate for a scientific task while constraining different aspects of the law. We study how to resolve this residual design freedom in baseline-preserving settings, where an existing dynamical model remains the working baseline outside quantities explicitly valued by the task. Fiber-Informed Differentiable Experimental Design (FIDE) completes each candidate’s aggregate constraints by information projection relative to a common frozen reference, then ranks scientifically adequate designs by the dynamical revision required to realize the resulting completed paths. The resulting Full action is the minimum mean-squared velocity correction needed to realize a completed path; equivalently, its square root is the sharp reference-relative expectation-rate discrepancy over a gradient-normalized Sobolev class. This defines a conditional minimum-revision principle for baseline-preserving design rather than a generic truth-recovery objective. Experiments show that this preference changes sensor selection, can help or harm withheld downstream predictions depending on baseline reliability, can be used prospectively from aggregate predictions, and remains computationally usable for a 32-dimensional configuration law.

## 1 INTRODUCTION

Consider an experiment that deploys a small number of aggregate sensors to answer a prespecified scientific question about an evolving population. Several sensor layouts may achieve essentially the same task performance while constraining different aspects of the intermediate law. Because aggregate measurements generally do not identify the full law, these task-equivalent designs can imply different conclusions outside the quantities used to evaluate the task. The primary objective tells us which designs are scientifically adequate, but not how to choose among these remaining law-level differences. Resolving that choice requires an explicit secondary design preference.

Classical and modern optimal experimental design typically chooses measurements to improve information gain, posterior uncertainty, reconstruction quality, or a task-specific utility (Chaloner & Verdinelli, 1995; Foster et al., 2019; Attia et al., 2018). Our question begins after such a primary objective has already identified several scientifically adequate designs. We focus on the baselinepreserving regime: an existing dynamical model is the declared working baseline, and the present experiment is scoped to improve or assess prescribed quantities. Among designs that satisfy that task, we prefer not to introduce additional departures from the baseline unless those departures are valued by another objective or supported by independent evidence.

This regime is natural when new sensors augment an existing modeling or decision pipeline rather than replace it. Related model-credibility and observing-system frameworks likewise evaluate mod els and measurements within a specified context of use (Morrison et al., 2019; Viceconti et al., 2021; Hoffman & Atlas, 2016; Mourre & Alvarez, 2012). Once the scientific task is already satisfied, leaving the residual choice unspecified means that it may be determined by incidental details of the parameterization, optimization procedure, or sensor family rather than by a declared design preference. Our formulation makes that preference explicit: preserve the working baseline in directions left unresolved by the task unless another objective provides a reason to depart from it. Departures that are themselves scientifically valuable should therefore enter the primary risk.

Within this baseline-preserving setting, we propose Fiber-Informed Differentiable Experimental Design (FIDE) to resolve the residual design choice. Aggregate measurements do not determine a unique population law, so FIDE first information-projects each candidate’s constraints relative to the same frozen reference (Kullback & Leibler, 1951; Jaynes, 1957; Csiszar, 1975). This gives every´ candidate a completed law path under the same reference-relative rule before any dynamical score is computed.

The completed path need not follow the frozen reference dynamics, so FIDE asks how much those dynamics must change to realize it. The Full action is the minimum mean-squared velocity correction required to do so. Equivalently, its square root is the largest reference-relative error in expectation rate, per unit gradient energy, over a normalized class of smooth observables. Thus, after scientific adequacy has been enforced, Full measures dynamical revision of the declared baseline across a broad observable class.

Our contributions are:

• Residual experimental design. We formulate baseline-preserving measurement selection as a hierarchical ε-constraint problem in which scientific risk determines adequacy and minimum dynamical revision resolves only the remaining design freedom.

• Reference-relative dynamical revision. We characterize Full as the minimum velocitycorrection energy required to realize the completed law path, with its square root equivalently giving a sharp expectation-rate discrepancy over a normalized class of smooth observables, and derive its Full–Tangent–Hidden decomposition and matched-initial stability bound.

• Differentiable design optimization. For continuous designs, we establish local end-toend differentiability of the finite-dimensional FIDE objective under explicit nonsingularity conditions and differentiate through both information-projection calibration and the Full correction solve.

• Empirical characterization and scope. We study criterion-dependent sensor selection, downstream effects of baseline preservation, prospective selection from aggregate predictions, and a 32-dimensional configuration-law application.

## 2 PROBLEM FORMULATION

Let $t \in [ 0 , 1 ]$ denote normalized experimental time and let $X _ { t } \in \mathcal { X }$ have unknown population law $P _ { t } ,$ , where $\chi$ is a domain in $\mathbb { R } ^ { d }$ or a flat periodic domain. A measurement design $\pmb { \eta } \in \bar { \mathcal { D } }$ specifies a time-independent observable map $\Phi _ { \eta }$ and is paired, for design purposes, with an aggregate moment trajectory $\boldsymbol { c } _ { \eta } ( t ) \in \mathbb { R } ^ { R }$ . At each time, these moments define the fiber

$$
\Phi _ { \eta } : \mathcal { X } \to \mathbb { R } ^ { R } , \qquad \mathcal { F } _ { \eta } ( c ) : = \left\{ Q \in \mathcal { P } ( \mathcal { X } ) : \mathbb { E } _ { Q } [ \Phi _ { \eta } ] = c \right\} .\tag{1}
$$

In the ideal population formulation, $c _ { \eta } ( t ) = \mathbb { E } _ { P _ { t } } [ \Phi _ { \eta } ( X _ { t } ) ]$ . In applications, $c _ { \eta } ( t )$ denotes the candidate-specific aggregate trajectory supplied to FIDE; Section 3 specifies how these inputs are obtained retrospectively or prospectively. The observable coordinates may represent sensor responses, projections, localized windows, or coefficients of an observable subspace. Because finitely many expectations generally do not identify a full law, changing the design changes both what is constrained and which law directions remain unresolved. Here $\bar { \mathcal { P } } ( \mathcal { X } )$ denotes probability laws for which the displayed expectations exist.

Before design optimization, we fix a reference law path $\widetilde { Q } _ { t }$ with density $\widetilde { q } _ { t }$ and velocity $u _ { t }$ satisfying

$$
\begin{array} { r } { \partial _ { t } \widetilde { q } _ { t } + \nabla \cdot \left( \widetilde { q } _ { t } \boldsymbol { u } _ { t } \right) = 0 . } \end{array}\tag{2}
$$

The equation is understood weakly under the boundary or decay conditions of Section 4. The pair $( \widetilde { Q } _ { t } , u _ { t } )$ is held fixed as η varies and is not assumed to reproduce the unknown intermediate population dynamics. Full is defined relative to this entire law–velocity pair, since different velocity fields realizing the same reference marginals can induce different dynamical revision costs.

Scientific adequacy is specified independently by an application-dependent design-stage risk $\mathcal { R } \mathrm { ~ : ~ }$ $\mathcal { D }  \mathbb { R }$ , fixed before design optimization. This risk determines which designs are adequate; Full is used only to rank the design freedom that remains after adequacy has been enforced. Evaluating $\mathcal { R } ( \eta )$ may require additional candidate-specific summaries beyond $c _ { \eta } ( t )$ , such as response covariances or predicted scientific quantities, which are treated as design-stage inputs. Any departure that should affect the design choice must therefore enter the scientific risk or an additional objective. Transfer of scientific adequacy to post-freeze observations is evaluated separately.

## 3 METHODOLOGY

FIDE takes two conceptually distinct inputs. First, each candidate design η is paired with an aggregate trajectory $c _ { \eta } ( t )$ and the summaries needed to evaluate its scientific risk $\mathcal { R } ( \eta )$ . Retrospectively these may come from existing data or simulation, while prospectively they are supplied by a predictive interface fixed before optimization. Second, the frozen pair $( \widetilde { Q } _ { t } , u _ { t } )$ supplies the common dynamical baseline against which every candidate is evaluated. The predictive interface therefore answers what would this measurement system constrain?, whereas the frozen reference answers how much dynamical revision would those constraints require? Given these inputs, FIDE applies the same three steps to every candidate: complete its aggregate constraints, measure the resulting dynamical revision, and minimize that revision among scientifically adequate designs.

Step 1: complete the moment constraints. For each $( t , \eta )$ admitting the projection, define

$$
\begin{array} { r } { Q _ { t } ^ { \eta } \in \arg \operatorname* { m i n } \left\{ D _ { \mathrm { K L } } ( Q \| \widetilde { Q } _ { t } ) : \mathbb { E } _ { Q } [ \Phi _ { \eta } ] = c _ { \eta } ( t ) \right\} . } \\ { Q { \ll } \widetilde { Q } _ { t } } \end{array}\tag{3}
$$

The aggregate trajectory fixes the measured expectations, while information projection fills in the remaining law directions relative to the common reference. Thus $\widetilde { Q } _ { t }$ is fixed across designs and $Q _ { t } ^ { \eta }$ is the completed law associated with design η. The ordering of these operations is deliberate: FIDE first completes every candidate by the same rule and only then asks how dynamically costly that completed path is. This keeps the completion rule common across designs rather than allowing the dynamical score itself to choose unresolved law directions. A path-level minimum-action completion defines the corresponding coupled alternative; Appendix D.2.1 quantifies the difference on a controlled example. Theorem 1 gives the corresponding existence, uniqueness, differentiability, and exponential-tilt results.

Step 2: measure dynamical revision. Let $q _ { t } ^ { \eta }$ denote the projected density. The information projection fixes a law path, but the frozen reference velocity $u _ { t }$ need not transport that path. FIDE therefore asks for the smallest velocity correction $\delta _ { t }$ such that $u _ { t } + \delta _ { t }$ does. We define

$$
a _ { \mathrm { f u l l } } ( t , \eta ) : = \operatorname* { i n f } _ { \delta \in L ^ { 2 } ( Q _ { t } ^ { \eta } ; \mathbb { R } ^ { d } ) } \left\{ \mathbb { E } _ { Q _ { t } ^ { \eta } } [ \left\| \delta \right\| ^ { 2 } ] : \partial _ { t } q _ { t } ^ { \eta } + \nabla \cdot \left( q _ { t } ^ { \eta } ( u _ { t } + \delta ) \right) = 0 \right\} ,\tag{4}
$$

$$
\mathcal { A } ( \pmb { \eta } ) : = \int _ { 0 } ^ { 1 } a _ { \mathrm { f u l l } } ( t , \pmb { \eta } ) \rho ( d t ) ,\tag{5}
$$

where $\rho$ is a fixed probability measure on $[ 0 , 1 ]$ , the continuity constraint is understood weakly under the prescribed boundary conditions, and inf $\varnothing = + \infty$ . We call $a _ { \mathrm { f u l l } } ( t , \eta )$ the instantaneous Full action density and $\mathcal { A } ( \pmb { \eta } )$ the time-integrated Full action. Only the correction $\delta _ { t }$ is optimized; the reference and completed path remain fixed. For later analysis, whenever the required density derivatives exist on the positive support of $q _ { t } ^ { \eta }$ , define the normalized continuity forcing $h _ { t , \eta } : =$ $[ \partial _ { t } q _ { t } ^ { \eta } + \nabla \cdot ( q _ { t } ^ { \eta } u _ { t } ) ] / q _ { t } ^ { \eta }$ . The continuity constraint is then equivalent to $\nabla \cdot ( \dot { q _ { t } ^ { \eta } } \delta _ { t } ) = - \dot { q _ { t } ^ { \eta } } \dot { h } _ { t , \eta }$ Theorem 2 gives the corresponding weak and weighted-Poisson characterizations.

Step 3: select among scientifically adequate designs. The observable family $\mathcal { D } ,$ scientific risk $\mathcal { R } _ { : }$ , tolerance $\varepsilon ,$ , frozen reference $( \dot { Q } _ { t } , \boldsymbol { u } _ { t } )$ , information-projection rule, and Full correction geometry are fixed before optimizing η. Let $\begin{array} { r } { \mathcal { R } ^ { \star } : = \operatorname* { i n f } _ { \eta \in \mathcal { D } } \mathcal { R } ( \eta ) } \end{array}$ and assume $0 < \mathcal { R } ^ { \star } < \infty$ . For $\varepsilon \geq 0 .$ define

$$
\mathcal { D } _ { \varepsilon } : = \left\{ \eta \in \mathcal { D } : \mathcal { R } ( \eta ) \leq ( 1 + \varepsilon ) \mathcal { R } ^ { \star } \right\} , \qquad \eta _ { \mathrm { F u l l } } \in \underset { \eta \in \mathcal { D } _ { \varepsilon } } { \arg \operatorname* { m i n } } A ( \eta ) .\tag{6}
$$

Scientific risk therefore determines feasibility, and Full ranks only admissible designs. This gives FIDE a hierarchical structure: scientific adequacy is enforced first through the ε-constraint, and Full then resolves the remaining design freedom. The following elementary property makes the reference-relative meaning of this secondary criterion explicit.

Proposition 1 (Reference consistency and scope). Fix a design η for which the information projection is well defined. $I f c _ { \eta } ( t ) = \mathbb { E } _ { \widetilde { Q } _ { t } } [ \Phi _ { \eta } ]$ for every t, then $Q _ { t } ^ { \eta } = \widetilde { Q } _ { t }$ and $a _ { \mathrm { f u l l } } ( t , \eta ) = 0$ for every t, hence $\begin{array} { r } { A ( \eta ) = 0 . } \end{array}$ . If an alternative population law $P _ { t } \neq \widetilde { Q } _ { t }$ also satisfies $\mathbb { E } _ { P _ { t } } [ \Phi _ { \eta } ] = c _ { \eta } ( t )$ for every t, this zero action does not imply ${ Q } _ { { t } } ^ { \dot { \eta } } = { P } _ { { t } }$ . Thus zero Full certifies consistency with the declared reference under the supplied constraints; it does not identify among alternative laws that share those constraints.

The first claim follows because the reference itself satisfies the moment constraints and attains zero relative entropy, so its frozen velocity realizes the projected path with zero correction. Intuitively, the candidate measurements then ask for nothing that the reference does not already predict, so neither the completed law nor its dynamics need to change. Existence qualifications for the outer minimizer and the finite-sample Law baseline are collected in Appendix B.3.

Comparator criteria. For empirical comparison, we also select designs using Tangent, residual $L ^ { 2 } .$ , forward $\mathrm { K L , }$ and static $W _ { 2 }$ as secondary criteria, always on the same information-projected law path and within the same admissible set $\mathcal { D } _ { \varepsilon }$ . These criteria probe different properties of the completed path and their definitions are collected in Appendix C.1, while Section 4 summarizes how they differ from Full.

Numerical implementation and design gradients. In finite-sample experiments, we use Law to denote selection by scientific risk alone. We replace the population optimum $\mathcal { R } ^ { \star }$ by the selection risk $R _ { \mathrm { L a w } }$ of the numerically optimized Law design and use the operational admissible set $R _ { \mathrm { s e l } } ( \eta ) \leq ( 1 + \varepsilon ) R _ { \mathrm { L a w } }$ This is a finite-search approximation rather than a certificate of global risk optimality. For each candidate, the numerical pipeline calibrates the empirical information projection, evaluates scientific risk and the chosen secondary criterion, and optimizes over the resulting admissible designs. All candidate-specific design-stage inputs are fixed before optimization; post-freeze target observations are used only for validation and are never fed back into candidate scoring. For an empirical reference $\textstyle \sum _ { i = 1 } ^ { N } { \bar { w } } _ { i } \delta _ { x _ { i } }$ with $z _ { i } ~ = ~ \Phi _ { \eta } ( x _ { i } )$ , exact calibration requires $\widehat { c } \in \mathrm { c o n v } \{ z _ { 1 } , \ldots , z _ { N } \}$ We distinguish this mathematical calibration validity from support certification: a feasible calibrated law may still be poorly supported for estimating Full action. We therefore monitor effective sample size and covariance conditioning separately. Proposition 3 gives the precise feasibility result, while benchmark-specific discretizations and numerical audits are deferred to the corresponding experimental appendices.

FIDE allows both discrete and continuous designs. For continuous designs, the fixed finitedimensional objective is locally differentiable end to end whenever the calibrated covariance and reduced Full systems remain nonsingular and vary smoothly with the design. In particular, if the reduced Full solve is $\mathsf { K } _ { \eta } \beta _ { \eta } = b _ { \eta }$ , then for any design perturbation ξ the instantaneous action satisfies $D _ { \eta } \widehat { a } _ { \mathrm { f u l l } } [ \xi ] = 2 \beta _ { \eta } ^ { \top } D _ { \eta } \widehat { b } _ { \eta } [ \xi ] - \beta _ { \eta } ^ { \top } D _ { \eta } { \sf K } _ { \eta } [ \xi ] \beta _ { \eta }$ . Thus gradients propagate through the informationprojection calibration, physical-time multiplier dynamics, and Full correction solve without differentiating through the iterations of the underlying solvers. Under these local conditions, continuous sensor locations or other differentiable design parameters can therefore be optimized directly with gradient-based methods. Proposition 9 gives the full conditions and integrated-action result.

## 4 THEORETICAL ANALYSIS

This section develops the properties of Full needed for the residual-design interpretation. We first derive the information-projected path and its reference-relative continuity forcing. We then show that the minimum quadratic correction used in Section 3 is exactly a sharp Sobolev expectation-rate discrepancy, identify Tangent as its restriction to the measured observable span, and finally relate Full to the remaining comparators and to matched-initial stability.

Information-projected path and continuity forcing For a fixed design, write $\phi = \Phi _ { \eta } , Q _ { t } =$ $Q _ { t } ^ { \eta } , q _ { t } = q _ { t } ^ { \eta } , c _ { t } = c _ { \eta } ( t ) , \lambda _ { t } = \lambda _ { t , \eta }$ , and $h _ { t } = h _ { t , \eta }$ , and define $A _ { t } ( \lambda ) = \log \mathbb { E } _ { \widetilde { Q } _ { t } } [ e ^ { \lambda ^ { \top } \phi } ] , m _ { t } =$ $J _ { \phi } u _ { t }$ , and $C _ { t } = \mathrm { C o v } _ { Q _ { t } } ( \phi , \phi )$

Theorem 1 (Information-projected lift and forcing). Assume that $c _ { t }$ lies in the relative interior of the feasible moment set, that the dual calibration objective $A _ { t } ( \lambda ) - \lambda ^ { \top } c _ { t }$ attains a minimizer $\lambda _ { t }$ and that $A _ { t }$ is finite in a neighborhood of $\lambda _ { t } .$ . Under the conditions of Appendices B.4–B.6, the information projection exists, is unique at the law level, and, in a minimal observable representation with $C _ { t } \succ 0$ , satisfies

$$
q _ { t } ( x ) = \widetilde { q } _ { t } ( x ) \exp \bigl ( \lambda _ { t } ^ { \top } \phi ( x ) - A _ { t } ( \lambda _ { t } ) \bigr ) , \qquad \mathbb { E } _ { Q _ { t } } [ \phi ] = c _ { t } ,\tag{7}
$$

$$
C _ { t } \dot { \lambda } _ { t } = \dot { c } _ { t } - \mathbb { E } _ { Q _ { t } } [ m _ { t } ] - { \mathrm { C o v } } _ { Q _ { t } } ( \phi , \lambda _ { t } ^ { \top } m _ { t } ) ,\tag{8}
$$

$$
h _ { t } = \dot { \lambda } _ { t } ^ { \top } ( \phi - c _ { t } ) + \lambda _ { t } ^ { \top } ( m _ { t } - \mathbb { E } _ { Q _ { t } } [ m _ { t } ] ) , \qquad \mathbb { E } _ { Q _ { t } } [ h _ { t } ] = 0 .\tag{9}
$$

Thus $\partial _ { t } q _ { t } + \nabla \cdot ( q _ { t } q _ { t } ) = q _ { t } h _ { t }$ , and the forcing can be evaluated from tilted expectations and an $R \times R$ covariance solve without differentiating the reference density in physical time.

Reference-relative expectation-rate robustness For a sufficiently regular time-independent observable $f ,$ define

$$
\mathcal { E } _ { t } ( f ) : = \frac { d } { d t } \mathbb { E } _ { Q _ { t } } [ f ] - \mathbb { E } _ { Q _ { t } } [ \nabla f \cdot u _ { t } ] = \mathbb { E } _ { Q _ { t } } [ h _ { t } f ] .\tag{10}
$$

Because $\mathbb { E } _ { Q _ { t } } [ h _ { t } ] = 0$ , this functional is invariant to additive constants. Let $H _ { q _ { t } }$ denote the mean zero weighted Sobolev energy space defined in Appendix B.1.

Theorem 2 (Sharp expectation-rate characterization of instantaneous Full action). Assume the weighted Dirichletform is coercive on $H _ { q _ { t } }$ and that $f \mapsto \mathbb { E } _ { Q _ { t } } [ h _ { t } f ]$ defines a bounded linearfunctional on $H _ { q _ { t } }$ , under the boundary conventions of Appendix B.1. Then

$$
\begin{array} { r l r l } { a _ { \mathrm { f u l l } } ( t , \pmb { \eta } ) = } & { { } } & { \underset { f \in { \cal H } _ { q _ { \star } } } { \operatorname* { s u p } } } & { } & { { } | \mathcal { E } _ { t } ( f ) | ^ { 2 } , } \end{array}\tag{11}
$$

$$
\mathbb { E } _ { Q _ { t } } [ \| \nabla f \| ^ { 2 } ] { \leq } 1
$$

$$
\nabla \cdot ( q _ { t } \nabla \psi _ { t } ) = q _ { t } h _ { t } , \qquad \mathbb { E } _ { Q _ { t } } [ \psi _ { t } ] = 0 , \qquad \delta _ { t } ^ { \star } = - \nabla \psi _ { t } ,\tag{12}
$$

$$
a _ { \mathrm { f u l l } } ( t , \pmb { \eta } ) = \mathbb { E } _ { Q _ { t } } [ \| \nabla \psi _ { t } \| ^ { 2 } ] = \| h _ { t } \| _ { H ^ { - 1 } ( q _ { t } ) } ^ { 2 } .\tag{13}
$$

Here ${ \delta } _ { t } ^ { \star }$ is the unique minimum-energy correction in equation 4.

In this sense, Full asks not how far the completed law is from the reference at one time, but how difficult it is for the reference dynamics to realize its evolution. The $H ^ { - 1 }$ representation gives a useful transport interpretation of this cost. Full does not penalize continuity mismatch only by its pointwise amplitude; it weights that mismatch by the spatial effort required to correct it. Rapidly varying local discrepancies can often be repaired by moving mass over short distances and therefore have relatively small Full cost, whereas coherent low-frequency discrepancies requiring transport over longer distances are more expensive. Equivalently, in flux variables the energy weights a required correction flux by $1 / q _ { t }$ , making transport of a fixed flux through poorly populated regions costly. This transport-scale sensitivity is precisely what distinguishes Full from a plain $L ^ { 2 }$ norm of the same continuity forcing; Appendix C.4 gives exact Fourier examples.

The three equivalent views of Full play complementary roles. The minimum-correction form gives it a physical interpretation as transport effort, the $H ^ { - 1 }$ form exposes the spatial geometry of that effort, and the dual form gives the interpretation most directly relevant to experimental design. In particular, $\sqrt { a _ { \mathrm { f u l l } } ( t , \eta ) }$ is the smallest uniform constant controlling reference-relative expectationrate discrepancy per unit Sobolev gradient energy. Consequently, once scientific adequacy has been enforced, minimizing $\boldsymbol { \mathcal { A } } ( \boldsymbol { \eta } )$ minimizes the time-integrated squared worst-case discrepancy over this smooth observable class rather than only over the quantities explicitly protected by the primary risk. Equivalent sharp-constant and operator formulations are given in Appendices B.8 and C.2 (Ambrosio & Serfaty, 2008; Mroueh, 2018; Neklyudov et al., 2023; Chemseddine et al., 2025).

Full transports the entire completed law, whereas the measured observables constrain only some of its expectation rates. This suggests separating the required correction into the part already forced by those measured rates and the additional law-level correction they leave unresolved.

Proposition 2 (Full–Tangent–Hidden decomposition). Assume the conditions of Theorem 2, $J _ { \phi } \in$ $L ^ { 2 } ( Q _ { t } ) , J _ { \phi } u _ { t } \in L ^ { 1 } ( Q _ { t } )$ , and $\phi _ { j } - c _ { t , j } \in H _ { q _ { t } }$ for each observable coordinate, with moment $d i f f e r -$ entiation and weak testing justified. Define

$$
r _ { t } = \mathbb { E } _ { Q _ { t } } [ J _ { \phi } u _ { t } ] - \dot { c } _ { t } , \qquad G _ { t } = \mathbb { E } _ { Q _ { t } } [ J _ { \phi } J _ { \phi } ^ { \top } ] .\tag{14}
$$

The Tangent correction is $\delta _ { t } ^ { \mathrm { t a n } } = - J _ { \phi } ^ { \top } G _ { t } ^ { \dagger } r _ { t }$ , with $a _ { \mathrm { t a n } } ( t , \pmb { \eta } ) = r _ { t } ^ { \top } G _ { t } ^ { \dagger } r _ { t }$ . Writing $\delta _ { t } ^ { \mathrm { h i d } } = \delta _ { t } ^ { \star } - \delta _ { t } ^ { \mathrm { t a n } }$ and $a _ { \mathrm { h i d } } ( t , \pmb { \eta } ) = \mathbb { E } _ { Q _ { t } } [ \| \delta _ { t } ^ { \mathrm { h i d } } \| ^ { 2 } ]$ , the Tangent and Hidden components are orthogonal and

$$
a _ { \mathrm { f u l l } } ( t , \pmb { \eta } ) = a _ { \mathrm { t a n } } ( t , \pmb { \eta } ) + a _ { \mathrm { h i d } } ( t , \pmb { \eta } ) .\tag{15}
$$

Integrating against $\rho$ gives the corresponding design-level decomposition equation 150, and hence $\bar { \mathcal { A } } _ { \mathrm { t a n } } ( \pmb { \eta } ) \stackrel { - } { \le } \bar { \mathcal { A } } ( \pmb { \eta } )$

Tangent is therefore the restriction of the Full dual problem to the finite-dimensional span of the centered measured observables, equivalently a Galerkin restriction in the weighted Sobolev energy space, while Hidden is the corresponding Galerkin energy gap (Mroueh, 2018). For experimental design, this means that two sensor systems can be equally adequate for the scientific task while inducing very different dynamical corrections elsewhere in the completed law. Full distinguishes precisely this residual difference. The exact Tangent–Full equality condition and examples showing that Tangent does not generally control Full are given in Appendices B.11 and C.3.

Criterion distinctions and matched-initial stability. The remaining secondary criteria probe different properties of the same information-projected law path. Residual $L ^ { 2 }$ measures the amplitude of the complete normalized continuity forcing, whereas Full weights that forcing through inverse transport geometry. Forward KL and static $W _ { 2 }$ instead measure marginal displacement from the frozen reference and do not encode the temporal realization cost of the prescribed path. Appendix C establishes these distinctions through operator identities, exact information-projected separations, and ranking reversals.

Full also yields a positive stability statement in a stronger regular-flow regime. Let $\mathsf { T } _ { s , t } ^ { u }$ denote the characteristic flow of $u _ { t }$ and define the matched-initial reference rollout $R _ { t } ^ { \eta } : = ( \mathsf { T } _ { 0 , t } ^ { u } ) _ { \# } Q _ { 0 } ^ { \eta }$ Assuming finite second moments and Full action, classical characteristic flows for $u _ { t }$ and ${ u } _ { t } + \delta _ { t } ^ { \star }$ and an integrable spatial Lipschitz bound $\| u _ { t } ( x ) - u _ { t } ( y ) \| \leq L ( t ) \| x - y \|$ , Proposition 14 gives, for $\rho ( d t ) = d t$ and $\textstyle \Lambda = \int _ { 0 } ^ { 1 } L ( t ) d t$

$$
W _ { 2 } ( Q _ { t } ^ { \eta } , R _ { t } ^ { \eta } ) \leq \int _ { 0 } ^ { t } \exp \left( \int _ { s } ^ { t } L ( \tau ) d \tau \right) \sqrt { a _ { \mathrm { f u l l } } ( s , \eta ) } d s \leq e ^ { \Lambda } \sqrt { A ( \eta ) } , \qquad t \in [ 0 , 1 ] .\tag{16}
$$

Hence every globally 1-Lipschitz observable $f$ satisfies

$$
\left| \mathbb { E } _ { Q _ { t } ^ { \eta } } [ f ] - \mathbb { E } _ { R _ { t } ^ { \eta } } [ f ] \right| \leq e ^ { \Lambda } \sqrt { \mathcal { A } ( \eta ) } .
$$

When $Q _ { 0 } ^ { \eta } = \widetilde { Q } _ { 0 } ,$ , one has $R _ { t } ^ { \eta } = \widetilde { Q } _ { t }$ and $\mathcal { A } _ { W _ { 2 } } ( \pmb { \eta } ) \leq e ^ { 2 \Lambda } \mathcal { A } ( \pmb { \eta } )$ . More generally, the preceding bound gives a downstream-agnostic stability interpretation: after initial-law mismatch is removed, small Full action controls the departure of every later-chosen globally 1-Lipschitz observable from the corresponding frozen-reference rollout, even if that observable was not part of the scientific risk used to select the design. This is a one-sided reference-relative guarantee, not a guarantee of recovery of the unknown intermediate population law. Appendix C.6 shows that no reverse bound holds in general. Proofs, technical conditions, and additional analytical results are collected in Appendices B and C.

## 5 EXPERIMENTS

The experiments follow a progression from mechanism to deployment. Gaussian-mixture transport first asks whether the choice of secondary criterion actually changes the measurement design once scientific adequacy is fixed. A controlled downstream experiment then asks when the baselinepreserving preference improves prediction of a quantity that was never used for selection and when that benefit fails. Prospective double-gyre transport removes access to intermediate microscopic target states during design, and interacting skyrmions finally test the same construction on a higherdimensional joint law. Across these studies we evaluate three distinct quantities when available: held-out task risk tests whether the scientific requirement transfers, held-out Full tests whether the selected minimum-revision preference persists on fresh data, and truth-relative downstream diagnostics test consequences outside the selection objective. Section 5.1 collects supporting analyses of the modeling assumptions and numerical approximations behind these results. The source code is available on GitHub.

<table><tr><td>Selector</td><td>Selection risk increase (%)</td><td>Held-out Full (cross-eval.)</td><td>Truth-relative hidden-law  $\mathrm { S W _ { 2 } ^ { 2 } }$ </td></tr><tr><td>Law</td><td>0</td><td>29.4178</td><td>0.096460</td></tr><tr><td>Tangent</td><td>0.491</td><td>37.3317</td><td>0.094144</td></tr><tr><td>Residual  $L ^ { 2 }$ </td><td>0.441</td><td>36.0548</td><td>0.094173</td></tr><tr><td>Forward  $\mathrm { K L }$ </td><td>0.303</td><td>33.0647</td><td>0.094352</td></tr><tr><td>Static  $W _ { 2 }$ </td><td>0.294</td><td>32.9924</td><td>0.094357</td></tr><tr><td>Full</td><td>0.604</td><td>19.5359</td><td>0.102205</td></tr></table>

Table 1: Criterion-specific Gaussian designs at the 1% allowance, selected under the same scientificrisk restriction and cross-evaluated using held-out Full action and truth-relative hidden-law squared sliced-Wasserstein distance. The two diagnostics induce different orderings; values are comparable within columns but not across columns.

(1) Secondary criteria induce different designs: Gaussian-mixture transport The Gaussian benchmark isolates the effect of the secondary criterion in a transparent two-dimensional setting: the endpoint laws are fixed, the orientation of the intermediate mixture varies, and two localized Gaussian sensors are placed on a ring. Law, Tangent, residual $L ^ { 2 }$ , forward KL, static $W _ { 2 } ,$ , and Full select different sensor geometries despite closely matched scientific risk. Table 1 cross-evaluates these designs using reference-relative Full action and truth-relative hidden-law discrepancy. The orderings disagree: the Full-selected geometry has the smallest held-out Full action but the largest hidden-law discrepancy. Thus the secondary criteria resolve the same scientifically admissible design freedom differently rather than serving as interchangeable approximations to a common objective. This disagreement motivates the next question: if minimum revision is not a generic truth-recovery principle, under what conditions does preserving unresolved baseline structure nevertheless improve a quantity that was never used to select the design? Complete criterion comparisons and allowance sweeps are reported in Appendices D.2.4 and E.2.

(2) When does baseline preservation help and when does it hurt? We next ask when the baseline-preserving preference has a measurable consequence for a quantity never used in design. We construct a product-law problem with one task-relevant coordinate and one task-unprotected coordinate. Each candidate sensor contains the same task-relevant signal but mixes it with a different component of the unprotected coordinate, so the primary scientific risk is exactly tied across five sensor geometries. The frozen baseline is initially correct in the unprotected coordinate. As an independent downstream test, we measure truth-relative local-occupancy RMSE on 16 smooth observables of that coordinate, none of which enters either the scientific risk or the secondary design criteria. Because Law is indifferent among all five risk-optimal geometries, we report its expected downstream error under uniform tie-breaking. When the baseline is correct, Full and static $W _ { 2 }$ select the same geometry, reducing expected occupancy error by about half relative to Law, whereas the geometry selected by Tangent, residual $L ^ { 2 }$ , and forward KL performs worse.

We then freeze these selections and the reference and introduce an increasingly strong true departure in the unprotected coordinate. Figure 1 shows the resulting transition: the advantage of the Full/W<sub>2</sub> geometry steadily erodes and eventually reverses relative to Law. This isolates when the baselinepreserving inductive bias is useful: preserving unresolved baseline structure improves prediction of downstream quantities withheld from design when that structure is reliable, but becomes harmful

A Downstream error under a correct baseline

![](images/f25bc4e01327099b8a057aa3294fb8b0edcdf1144097aa9bd752605847e2f9ae.jpg)

![](images/7193cd75bb5b1380a907bf69eda8b9129694eaf49166e9cd88e83bcd17a14472.jpg)  
Figure 1: Downstream consequences of baseline-preserving selection. A: At signal amplitude $\epsilon = 0 . 1 0$ and occupancy localization scale $\kappa = 4 ,$ we evaluate truth-relative local-occupancy RMSE on 16 smooth observables that are withheld from both the scientific risk and all secondary design criteria. Scientific risk is identical across the five candidate geometries. Full and static $\dot { W } _ { 2 }$ select $\alpha = 0 ,$ , while Tangent, residual $L ^ { 2 }$ , and forward KL select $\alpha = 0 . 7 5$ . The Law bar is the expected RMSE under uniform tie-breaking over its five risk-minimizing geometries; open circles show the corresponding geometry-specific errors. B: The selected geometries are then frozen while a true departure of strength $\gamma$ is introduced in the task-unprotected coordinate. The ordinate is occupancy RMSE minus the expected Law RMSE, so negative values indicate improvement over Law and positive values indicate harm. The Full/static- $- \bar { W _ { 2 } }$ geometry changes from lower to higher error than Law as misspecification grows, while the Tangent/residual-L<sup>2</sup>/forward-KL geometry reverses in the opposite direction; both crossings occur between $\gamma = 0 . 5$ and 0.6 in this representative setting.

once the baseline is sufficiently wrong. In this construction static $W _ { 2 }$ makes the same selection, so the result identifies the value of baseline preservation rather than an advantage unique to Full. This controlled setting lets us vary baseline reliability while retaining access to ground truth; we next remove that luxury and ask whether the residual design choice can be made prospectively before intermediate microscopic target states are available. Appendix D.2.2 gives the complete construction, robustness checks, and full misspecification sweep.

(3) Prospective selection without intermediate target states: double-gyre transport The double-gyre provides a nonautonomous transport benchmark with stretching, exchange, and nontrivial passive-tracer motion (Shadden et al., 2005). We choose four localized sensors for a heterogeneous tracer population while withholding all intermediate microscopic target states from the optimizer. Selection uses only endpoint ensembles, a frozen endpoint-trained reference, and a frozen aggregate predictive interface constructed from a separate simulation bank, which supplies the candidate moment trajectories and summaries needed to evaluate scientific risk. After the sensor geometries are frozen, we introduce a separately seeded validation reference and fresh microscopic validation trials. On these held-out data, the $2 \% - 5 \%$ Full-selected designs satisfy their corresponding scientific-risk caps, while the 1% point remains unresolved; across the tested frontier, the Fullselected geometries also retain lower Full action than Law. Thus both scientific adequacy, where statistically resolved, and the minimum-revision ordering transfer after design freeze without access to intermediate target states during selection. Complete results and uncertainty analyses are in Appendix E.3, and Figure 2 summarizes the resulting risk–action tradeoff.

(4) Configuration-space design in 32 dimensions: interacting skyrmions The prospective study changes the information available at design time; our final experiment instead changes the computational regime. Motivated by interacting magnetic skyrmions (Nagaosa & Tokura, 2013; Reichhardt et al., 2015), we use a reduced stochastic model of 16 interacting planar particles, giving a 32- dimensional microscopic state. Four aggregate sensors observe averages over 512 configurations, while scientific risk is defined through nine permutation-invariant features of pair structure and collective order. Full is approximated directly in configuration space using a fixed K = 280 Galerkin potential dictionary, so the reported values assess the finite-dimensional construction rather than continuum convergence. On fresh paired validation trials, the Full-selected designs satisfy their scientific-risk caps, differ from the corresponding Law selections, and retain lower fitted action. The experiment therefore shows that the same residual-design construction remains operational on a 32-dimensional joint law; complete results and numerical audits are in Appendix E.4.

![](images/917d2a6f7136fe8bc0fb6218451552902a34c7cafb86c2b29ac9ffb4e8048a71.jpg)  
Figure 2: Empirical overview. Left: Held-out scientific risk and Full action across the tested risk allowances in the prospective double-gyre study. Right: Interacting-skyrmion setup, where aggregate sensor views constrain a 32-dimensional configuration law without determining it.

## 5.1 SUPPORTING ANALYSES

Supporting studies examine the assumptions and numerical approximations behind the main experiments. Baseline-correctness and reference-perturbation analyses characterize when the minimum revision preference helps or fails under misspecification (Appendices D.2.2 and D.2.5). Allowing an unresolved mode to adapt dynamically reduces Full action by roughly 10–26%, quantifying the effect of fixing completion before scoring, while retrospective double-gyre and prospec tive predictor perturbations assess repeatability and sensitivity to the aggregate interface (Appendices D.2.1, D.1.1, and D.2.8). Numerical checks further test raster, smoothing, and solver choices, manufactured-problem convergence, richer skyrmion potential spaces, and leave-one-out stability (Appendices D.2.6, D.2.7, and D.2.9). Together, these analyses distinguish sensitivity to the modeling assumptions from sensitivity to the numerical implementation.

## 6 CONCLUSION

FIDE separates two questions in experimental design: what must a measurement system accomplish for the scientific task, and how should the remaining law-level freedom be resolved once that task is satisfied? Scientific risk answers the first; FIDE addresses the second by completing each design’s aggregate constraints relative to a common frozen reference and measuring the dynamical revision required to realize the resulting path. Full action makes this revision precise through equivalent minimum-correction and expectation-rate interpretations, while the Full–Tangent–Hidden decomposition identifies the part of the required correction that is invisible to the measured moment rates.

The experiments show that this residual choice matters. Different secondary criteria select different scientifically admissible measurements; preserving unresolved baseline structure can improve prediction of downstream quantities withheld from design when that structure is reliable and can hurt when it is sufficiently misspecified; prospective selection can transfer after design freeze without intermediate microscopic target states; and the finite-dimensional construction can be instantiated on a 32-dimensional joint law. Minimum Full is therefore most naturally viewed as a design principle for settings in which preserving a declared baseline outside task-valued departures is itself part of the experimental specification. Extending FIDE to propagate uncertainty over that baseline is a natural next step. A complementary direction is scalable optimization with discrete or combinatorial experimental decisions, where smooth surrogate objectives or continuous relaxations could retain gradient-based optimization while ultimately selecting feasible discrete designs.

## AI USE STATEMENT

Generative AI tools were used to assist with code development and debugging and with languagelevel polishing of the manuscript. The authors reviewed and validated the resulting code and manuscript text, and take full responsibility for the final submission.

## REPRODUCIBILITY STATEMENT

The code required to reproduce all experiments will be made publicly available on GitHub upon publication. Detailed experimental protocols, information boundaries, numerical procedures, implementation details, hardware and software specifications, and benchmark-specific settings are pro vided in Appendix E. The appendix also reports numerical audits, held-out validation procedures, and sensitivity analyses used to assess reproducibility and stability of the reported results.

## REFERENCES

Michael S. Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic in terpolants. In International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=li7qeBbCR1t.

Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. Journal of Machine Learning Research, 26(209):1–80, 2025. URL https://www.jmlr.org/papers/v26/23-1605.html.

Alen Alexanderian, Noemi Petra, Georg Stadler, and Omar Ghattas. A-optimal design of experiments for infinite-dimensional bayesian linear inverse problems with regularized \ell 0- sparsification. SIAM Journal on Scientific Computing, 36(5):A2122–A2148, 2014.

Luigi Ambrosio and Sylvia Serfaty. A gradient flow approach to an evolution problem arising in superconductivity. Communications on Pure and Applied Mathematics: A Journal Issued by the Courant Institute ofMathematical Sciences, 61(11):1495–1539, 2008.

Anthony C Atkinson and VV370955 Fedorov. The design of experiments for discriminating between two rival models. Biometrika, 62(1):57–70, 1975.

Ahmed Attia, Alen Alexanderian, and Arvind K. Saibaba. Goal-oriented optimal design of experiments for large-scale bayesian linear inverse problems. Inverse Problems, 34(9):095009, 2018. doi: 10.1088/1361-6420/aad210.

Ahmed Attia, Sven Leyffer, and Todd S Munson. Robust a-optimal experimental design for sensor placement in bayesian linear inverse problems. SIAM/ASA Journal on Uncertainty Quantification, 13(2):744–774, 2025.

Jean-David Benamou and Yann Brenier. A computational fluid mechanics solution to the Monge– Kantorovich mass transfer problem. Numerische Mathematik, 84(3):375–393, 2000. doi: 10. 1007/s002110050002.

Raktim Bhattacharya. Sensor design for accuracy-bounded estimation via maximum-entropy likelihood synthesis. arXiv preprint arXiv:2605.11120, 2026.

George EP Box and WILLIAM J Hill. Discrimination among mechanistic models. Technometrics, 9(1):57–71, 1967.

Kristian Bredies and Silvio Fanzon. An optimal transport approach for solving dynamic inverse problems in spaces of measures. ESAIM: Mathematical Modelling and Numerical Analysis, 54 (6):2351–2382, 2020.

Damiano Brigo and Giovanni Pistone. Optimal approximations of the Fokker–Planck–Kolmogorov equation: projection, maximum likelihood eigenfunctions and Galerkin methods, 2017. URL https://arxiv.org/abs/1603.04348v2. Version 2; first posted in 2016.

Damiano Brigo, Bernard Hanzon, and Franc¸ois Le Gland. Approximate nonlinear filtering by projection on exponential manifolds of densities. Bernoulli, 5(3):495–534, 1999. doi: 10.2307/ 3318714.

Kathryn Chaloner and Isabella Verdinelli. Bayesian experimental design: A review. Statistical Science, 10(3):273–304, 1995. doi: 10.1214/ss/1177009939.

Jannis Chemseddine, Christian Wald, Richard Duong, and Gabriele Steidl. Neural sampling from Boltzmann densities: Fisher–Rao curves in the Wasserstein geometry. In The Thirteenth International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 1d1cea122b9ec9f78acc21510659e500-Abstract-Conference.html.

Yifan Chen and Wuchen Li. Optimal transport natural gradient for statistical manifolds with continuous sample space. Information Geometry, 3(1):1–32, 2020.

Yongxin Chen, Tryphon T. Georgiou, and Michele Pavon. Optimal steering of a linear stochastic system to a final probability distribution, part i. IEEE Transactions on Automatic Control, 61(5): 1158–1169, 2016. doi: 10.1109/TAC.2015.2457784.

Yongxin Chen, Tryphon T. Georgiou, and Michele Pavon. Optimal transport over a linear dynamical system. IEEE Transactions on Automatic Control, 62(5):2137–2152, 2017. doi: 10.1109/TAC. 2016.2602103.

Austin H. Cheng, Alston Lo, Kin Long Kelvin Lee, Santiago Miret, and Alan Aspuru-Guzik. Stiefel´ flow matching for moment-constrained structure elucidation. In International Conference on Learning Representations, 2025.

Imre Csiszar. ´ I-Divergence Geometry of Probability Distributions and Minimization Problems. The Annals ofProbability, 3(1):146–158, 1975. doi: 10.1214/aop/1176996454.

Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion Schrodinger ¨ bridge with applications to score-based generative modeling. In Advances in Neural Information Processing Systems, volume 34, 2021. URL https://proceedings.neurips.cc/ paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html.

Purushottam D Dixit, Jason Wagoner, Corey Weistuch, Steve Presse, Kingshuk Ghosh, and Ken A´ Dill. Perspective: Maximum caliber is a general variational principle for dynamical systems. The Journal ofchemical physics, 148(1), 2018.

Adam Foster, Martin Jankowiak, Eli Bingham, Paul Horsfall, Yee Whye Teh, Tom Rainforth, and Noah Goodman. Variational bayesian optimal experimental design. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://proceedings.neurips. cc/paper/2019/hash/d55cbf210f175f4a37916eafe6c04f0d-Abstract. html.

Adam Foster, Desi R Ivanova, Ilyas Malik, and Tom Rainforth. Deep adaptive design: Amortizing sequential bayesian experimental design. In International conference on machine learning, pp. 3384–3395. PMLR, 2021.

Samar Hadou, Yigit Berkay Uslu, and Alejandro Ribeiro. Constrained diffusion models with primaldual inference. arXiv preprint arXiv:2606.17192, 2026.

Dion Hafner and Alexander Lavin. Tesseract core: Universal, autodiff-native software components¨ for simulation intelligence. Journal ofOpen Source Software, 10(111):8385, 2025.

Yacov Haimes. On a bicriterion formulation of the problems of integrated system identification and system optimization. IEEE transactions on systems, man, and cybernetics, (3):296–297, 1971.

Markus Hainy, David J Price, Olivier Restif, and Christopher Drovandi. Optimal bayesian design for model discrimination via classification. Statistics and Computing, 32(2):25, 2022.

Ross N Hoffman and Robert Atlas. Future observing system simulation experiments. Bulletin ofthe American Meteorological Society, 97(9):1601–1616, 2016.

Edwin T. Jaynes. Information theory and statistical mechanics. Physical Review, 106(4):620–630, 1957. doi: 10.1103/PhysRev.106.620.

Ruhui Jin, Martin Guerra, Qin Li, and Stephen J. Wright. Optimal design for linear models via gradient flow. Communications on Pure and Applied Analysis, 32:282–311, 2026. doi: 10.3934/ cpaa.2026041.

Shervin Khalafi, Dongsheng Ding, and Alejandro Ribeiro. Constrained diffusion models via dual training. In Advances in Neural Information Processing Systems, volume 37, 2024.

Andreas Krause, Ajit Singh, and Carlos Guestrin. Near-optimal sensor placements in gaussian processes: Theory, efficient algorithms and empirical studies. Journal of Machine Learning Research, 9:235–284, 2008. URL https://jmlr.org/papers/v9/krause08a.html.

Solomon Kullback and Richard A. Leibler. On information and sufficiency. The Annals ofMathematical Statistics, 22(1):79–86, 1951. doi: 10.1214/aoms/1177729694.

Richard S Laugesen, Prashant G Mehta, Sean P Meyn, and Maxim Raginsky. Poisson’s equation in nonlinear filtering. SIAM Journal on Control and Optimization, 53(1):501–525, 2015.

Etienne Lempereur, Nathanael Cuvelle-Magar, Florentin Coeurdoux, St¨ ephane Mallat, and Eric´ Vanden-Eijnden. MGD: Moment guided diffusion for maximum entropy generation. arXiv preprint arXiv:2602.17211, 2026.

Wuchen Li and Jiaxi Zhao. Wasserstein information matrix: W. li, j. zhao. Information Geometry, 6(1):203–255, 2023.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Liyao Lyu, Xinyue Yu, and Hayden Schaeffer. Multiscale nudging: From macroscopic observations to microscopic dynamics. arXiv preprint arXiv:2606.06809, 2026a.

Liyao Lyu, Xinyue Yu, David Schneidinger, and Hayden Schaeffer. Wasserstein moment nudging for vlasov-poisson data assimilation. arXiv preprint arXiv:2608.21725, 2026b.

Yuezhou Ma, Haixu Wu, Hang Zhou, Huikun Weng, Jianmin Wang, and Mingsheng Long. PhySense: Sensor placement optimization for accurate physics sensing. In Advances in Neural Information Processing Systems, volume 38, pp. 40146–40174, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/ hash/332b4fbe322e11a71fa39d91c664d8fa-Abstract-Conference.html.

Madhusudan Madhavan, Alen Alexanderian, Arvind K Saibaba, Bart van Bloemen Waanders, and Rebekah D White. A control-oriented approach to optimal sensor placement. SIAM/ASA Journal on Uncertainty Quantification, 14(2):711–741, 2026.

Krithika Manohar, Bingni W. Brunton, J. Nathan Kutz, and Steven L. Brunton. Data-driven sparse sensor placement for reconstruction: Demonstrating the benefits of exploiting known patterns. IEEE Control Systems Magazine, 38(3):63–86, 2018. doi: 10.1109/MCS.2018.2810460.

Aimee Maurais and Youssef Marzouk. Sampling in unit time with kernel fisher-rao flow. arXiv preprint arXiv:2401.03892, 2024.

Aimee Maurais, Bamdad Hosseini, and Youssef Marzouk. Learning paths for dynamic measure transport: A control perspective. arXiv preprint arXiv:2511.03797, 2025.

Tina M Morrison, Prasanna Hariharan, Chloe M Funkhouser, Payman Afshari, Mark Goodin, and Marc Horner. Assessing computational model credibility using a risk-based framework: application to hemolysis in centrifugal blood pumps. Asaio Journal, 65(4):349, 2019.

Baptiste Mourre and Alberto Alvarez. Benefit assessment of glider adaptive sampling in the ligurian sea. Deep Sea Research Part I: Oceanographic Research Papers, 68:68–78, 2012.

Olga Movilla Miangolarra, Asmaa Eldesoukey, and Tryphon T Georgiou. Inferring potential landscapes: A schrodinger bridge approach to maximum caliber.¨ Physical Review Research, 6(3): 033070, 2024.

Youssef Mroueh. Regularized finite dimensional kernel sobolev discrepancy. arXiv preprint arXiv:1805.06441, 2018.

Youssef Mroueh and Mattia Rigotti. Unbalanced sobolev descent. Advances in Neural Information Processing Systems, 33:17034–17043, 2020.

Youssef Mroueh, Tom Sercu, and Anant Raj. Sobolev descent. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 2976–2985. PMLR, 2019.

Naoto Nagaosa and Yoshinori Tokura. Topological properties and dynamics of magnetic skyrmions. Nature nanotechnology, 8(12):899–911, 2013.

Kirill Neklyudov, Rob Brekelmans, Daniel Severo, and Alireza Makhzani. Action matching: Learning stochastic dynamics from samples. In International conference on machine learning, pp. 25858–25889. PMLR, 2023.

Kirill Neklyudov, Rob Brekelmans, Alexander Tong, Lazar Atanackovic, Qiang Liu, and Alireza Makhzani. A computational framework for solving Wasserstein lagrangian flows. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 37461–37485. PMLR, 2024. URL https://proceedings. mlr.press/v235/neklyudov24a.html.

Michele Pavon and Augusto Ferrante. On the geometry of maximum entropy problems. SIAM Review, 55(3):415–439, 2013. doi: 10.1137/120862843.

C Reichhardt, D Ray, and CJ Olson Reichhardt. Collective transport properties of driven skyrmions with random disorder. Physical review letters, 114(21):217202, 2015.

Shawn C Shadden, Francois Lekien, and Jerrold E Marsden. Definition and properties of lagrangian coherent structures from finite-time lyapunov exponents in two-dimensional aperiodic flows. Physica D: Nonlinear Phenomena, 212(3-4):271–304, 2005.

Louis Sharrock. Wasserstein gradient flows for batch Bayesian optimal experimental design, 2026. URL https://arxiv.org/abs/2603.12102. arXiv preprint.

Louis Sharrock and Nikolas Kantas. Joint online parameter estimation and optimal sensor placement for the partially observed stochastic advection-diffusion equation. SIAM/ASA Journal on Uncertainty Quantification, 10(1):55–95, 2022. doi: 10.1137/20M1375073.

Alexander Tong, Jessie Huang, Guy Wolf, David van Dijk, and Smita Krishnaswamy. TrajectoryNet: A dynamic optimal transport network for modeling cellular dynamics. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 9526–9536. PMLR, 2020. URL https://proceedings.mlr.press/ v119/tong20a.html.

Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum?id=CD9Snc73AW.

Bruce Turkington. An optimization principle for deriving nonequilibrium statistical models of hamiltonian dynamics. Journal ofStatistical Physics, 152(3):569–597, 2013.

Umesh Vaidya, Rajeev Rajaram, and Sambarta Dasgupta. Actuator and sensor placement in linear advection PDE with building system application. Journal of Mathematical Analysis and Applications, 394(1):213–224, 2012. doi: 10.1016/j.jmaa.2012.03.046.

Marco Viceconti, Francesco Pappalardo, Blanca Rodriguez, Marc Horner, Jeff Bischoff, and Flora Musuamba Tshinanu. In silico trials: Verification, validation and uncertainty quantification of predictive models used in the regulatory evaluation of biomedical products. Methods, 185: 120–127, 2021.

Di Wu, Ling Liang, and Haizhao Yang. Beyond expected information gain: Stable bayesian optimal experimental design with integral probability metrics and plug-and-play extensions. arXiv preprint arXiv:2604.21849, 2026.

## A RELATED WORK

FIDE sits at the intersection of experimental design under partial information, statistical completion of aggregate constraints, and dynamical transport relative to a reference model. The individual ingredients have substantial histories; the distinction here is how they are assembled at the experimentaldesign level. A candidate measurement system determines a finite set of observable constraints and therefore a design-dependent family of compatible laws. FIDE first completes that ambiguity relative to a common frozen reference, then asks how much the resulting completed dynamics must depart from the reference, but only after the declared scientific objective has determined which designs are acceptable.

Experimental design and sensor placement Bayesian optimal experimental design provides a decision-theoretic framework for selecting measurements by their expected utility (Chaloner & Verdinelli, 1995). Modern developments include variational estimators of expected information gain (Foster et al., 2019) and amortized sequential policies (Foster et al., 2021). Goal-oriented design instead concentrates measurement effort on prescribed quantities of interest (Attia et al., 2018), while control-oriented formulations target uncertainty that matters for subsequent control decisions (Madhavan et al., 2026). Closely related sensor-placement objectives include mutual information (Krause et al., 2008), posterior covariance (Alexanderian et al., 2014), and data-driven reconstruction quality (Manohar et al., 2018). Continuous sensor locations can also be learned from reconstruction feedback (Ma et al., 2025), and robust OED explicitly accounts for misspecification in priors or measurement uncertainty (Attia et al., 2025). In transport settings, placement has been studied through observability and controllability Gramians for linear advection (Vaidya et al., 2012) and jointly with state and parameter estimation for stochastic advection–diffusion systems (Sharrock & Kantas, 2022).

Recent work has also used transport geometry directly in the outer design optimization. Wasserstein gradient flows can optimize probability measures over experimental designs for classical linearmodel objectives (Jin et al., 2026) and for batch expected information gain (Sharrock, 2026), while alternative BOED objectives replace KL-based information gain by integral probability metrics such as Wasserstein distance, maximum mean discrepancy, and energy distance (Wu et al., 2026). These approaches differ in the utility they optimize, but they share the usual objective of making the measurement system itself maximally informative, predictive, or useful for a specified downstream task. A particularly relevant contrast is optimal model-discrimination design, where disagreement among candidate models is itself informative and experiments are chosen to distinguish rival mechanisms or reduce uncertainty about model identity (Box & Hill, 1967; Atkinson & Fedorov, 1975; Hainy et al., 2022). FIDE addresses the complementary baseline-preserving regime: the scientific risk first determines an admissible set of comparably adequate designs, and residual disagreement with the frozen reference is not assigned additional scientific utility unless it enters that risk. Only within thi declared scope does FIDE use a second criterion to resolve residual law-level freedom by minimum dynamical revision. The resulting ε-constraint ordering is classical in multiobjective optimization (Haimes, 1971), the contribution here is the reference-relative dynamical criterion used to compare the admissible designs.

Risk-informed model credibility A complementary literature treats model credibility as conditional on a specified context of use rather than as a global property of a model. Risk-informed frameworks make the required credibility evidence commensurate with the consequences of relying on a computational model for that use, and organize verification, validation, and uncertainty assessment around the question the model is intended to support (Morrison et al., 2019; Viceconti et al., 2021). This literature motivates the scoped-use setting considered here but does not prescribe FIDE’s secondary criterion: baseline preservation is an additional design specification, and FIDE provides a dynamical rule for resolving the residual measurement-design freedom left by the primary task.

Aggregate information and statistical completion A separate literature asks what can be inferred when observations constrain only aggregate statistics rather than microscopic states. Maximum entropy and information projection provide classical ways to complete such partial statistical information (Kullback & Leibler, 1951; Jaynes, 1957; Csiszar, 1975). Related work develops geometric´ views of entropy optimization under linear constraints (Pavon & Ferrante, 2013) and generative procedures for sampling moment-constrained maximum-entropy laws (Lempereur et al., 2026). Par ticularly close in sensor-design motivation, Bhattacharya (2026) constructs a posterior and induced likelihood by minimizing KL divergence from a dynamical prior subject to distributional-accuracy constraints, including moment constraints. FIDE also uses KL projection relative to a dynamical baseline, but the role of the projection is different: for each candidate observable map, it provides a canonical completion of that design’s moment fiber before the design is scored dynamically.

There are also dynamic entropy- and projection-based approaches to incomplete information. Maximum Caliber extends entropy principles to trajectory distributions (Dixit et al., 2018), and Schrodinger-bridge formulations can incorporate distributional moments and average currents¨ (Movilla Miangolarra et al., 2024). Projection filters approximate nonlinear filtering dynamics on finite-dimensional exponential families (Brigo et al., 1999), with related Fokker–Planck approximations connecting tangent projection, moment matching, and Galerkin methods (Brigo & Pistone, 2017). Best-fit statistical closure takes another route, choosing paths in quasi-equilibrium families by minimizing a time-integrated weighted squared Liouville residual with a resolved/unresolved decomposition (Turkington, 2013). FIDE does not assume that the measured moments close the dynamics or identify the underlying law. Instead, the observable map itself is a design variable, so different candidate designs induce different moment fibers and different reference-relative completed paths. The completion is fixed before its dynamical correction cost is evaluated.

Reference-relative probability dynamics The frozen reference used by FIDE is closely related to modern learned probability-flow models. Flow matching and stochastic interpolants learn probability paths together with velocity fields (Lipman et al., 2023; Albergo & Vanden-Eijnden, 2023; Albergo et al., 2025), and optimal-transport couplings can be used to simplify the learned flows (Tong et al., 2024). In our experiments, endpoint-trained flow matching is one way to instantiate the frozen pair $( \widetilde { Q } _ { t } , u _ { t } )$ . The theoretical construction itself is more general: it requires only a reference law path and velocity satisfying the continuity equation, and neither is assumed to coincide with the unknown intermediate population dynamics.

Other approaches modify or infer probability dynamics relative to prescribed constraints or prior processes. Schrodinger bridges construct path laws close to a reference stochastic process under¨ marginal constraints (De Bortoli et al., 2021); TrajectoryNet infers continuous dynamics from crosssectional populations (Tong et al., 2020); and constrained generative methods impose distributional (Khalafi et al., 2024), average (Hadou et al., 2026), or exact structural constraints through constrained flow geometry (Cheng et al., 2025). Dynamic optimal transport supplies the continuityconstrained kinetic-energy geometry underlying many such constructions (Benamou & Brenier, 2000; Ambrosio & Serfaty, 2008), with related formulations in minimum-energy distribution steering (Chen et al., 2016), optimal transport with prior dynamics (Chen et al., 2017), and dynamictransport regularization of measure-valued reconstruction from indirect time-dependent observations (Bredies & Fanzon, 2020).

Measure-based nudging provides another related perspective. Lyu et al. (2026a) map macroscopic observations to microscopic mean-field particle corrections through Wasserstein-gradient feedback, while Lyu et al. (2026b) construct Wasserstein feedback from observed hydrodynamic moments for Vlasov–Poisson dynamics, where the moments leave the underlying phase-space law underdetermined. These methods seek corrections for a specified observation mechanism. In FIDE, the observation mechanism itself varies across candidate experimental designs, and the question is which scientifically adequate design induces the least reference-relative dynamical revision after its aggregate information has been canonically completed. More general Wasserstein Lagrangian formulations optimize density paths through combinations of kinetic and potential energies (Neklyudov et al., 2024); by contrast, FIDE does not reoptimize the projected path to reduce its action.

Minimum-energy realization and Sobolev geometry The dynamical criterion used by FIDE is most directly connected to work on minimum-energy realization of prescribed law paths. Action Matching learns minimum-kinetic-energy gradient velocities from temporally unpaired samples of a given marginal evolution (Neklyudov et al., 2023). For Boltzmann density curves, Chemseddine et al. (2025) establish weighted-Poisson conditions for Wasserstein absolute continuity and minimum-norm realization and analyze velocity blow-up for particular interpolations. Related weighted-Poisson constructions connect variational Bayesian updating to feedback particle filtering (Laugesen et al., 2015), transport samples along geometric reference–target density interpolations in kernel Fisher–Rao flow (Maurais & Marzouk, 2024), and jointly optimize tilted density paths and their transport dynamics to obtain smoother velocities (Maurais et al., 2025). FIDE uses the same general realization geometry in a different outer problem: the moment-constrained path is determined by the measurement design and the common frozen reference before its realization cost is computed. Neither that path nor the reference velocity is subsequently optimized to make the action small.

The dual characterization of this correction is also related to Sobolev discrepancies. Finite-feature Sobolev discrepancy uses derivative Gram matrices and identifies the finite-feature approximation gap as an energy-norm error (Mroueh, 2018), while Sobolev Descent uses closely related geometry for particle transport (Mroueh et al., 2019). In FIDE, restricting the full expectation-rate discrepancy to the span of the measured observables produces the Tangent criterion, and the remaining energy defines the Hidden component. Wasserstein natural gradients and Wasserstein information matrices likewise pull transport geometry back to parametrized statistical families (Chen & Li, 2020; Li & Zhao, 2023), which is closely related to the inverse-transport Gram structure that appears in our comparisons.

Taken together, these literatures provide the main ingredients used by FIDE: information projection to complete finite statistical constraints, reference-relative transport geometry to quantify the effort required to realize a prescribed law path, and Sobolev duality to interpret that effort through observable expectation rates. Our contribution is not a new information-projection principle or a new underlying transport metric. It is to place these ingredients inside a residual experimental-design problem in which each candidate measurement map induces its own moment fiber and completed law path, scientific risk determines admissibility, and the remaining design freedom is resolved by the dynamical revision required relative to a common frozen reference.

## B TECHNICAL STATEMENTS AND PROOFS

This appendix develops the technical results underlying Section 4. We first state the notation, weakform conventions, and functional-analytic preliminaries used throughout the proofs. We then give the assumptions and proofs for the information-projected lift, physical-time multiplier dynamics, density-free continuity forcing, the minimum-energy and weighted- $H ^ { - 1 }$ characterizations of Full, the sharp uniform certificate, Tangent, and the Full–Tangent–Hidden decomposition and equality condition. We also establish finite-reference information-projection feasibility, calibration sensitivity, and end-to-end differentiability of the fixed finite-dimensional Full objective. The finite-measure transport–reaction extension is developed alongside the balanced theory and evaluated separately in Appendix D.1.2. Operator-level comparisons and exact information-projected examples separating Full from Tangent, residual $L ^ { 2 }$ , forward KL, and static Wasserstein displacement, together with the matched-initial stability result, are collected in Appendix C.

## B.1 PRELIMINARIES AND STANDING CONVENTIONS

State space, laws, and boundary conventions. Whenever spatial derivatives are used, we identify the state space with a domain $\Omega \subseteq \mathbb { R } ^ { d }$ , a flat torus, or $\mathbb { R } ^ { d }$ . Bounded-domain statements use boundary conditions compatible with the corresponding continuity equation and variational problem; throughout the balanced theory these are periodic, sufficient-decay, or no-flux conditions chosen so that the stated integration-by-parts identities have no boundary contribution. Additional geometric hypotheses, such as connectedness or strict positivity of a density, are imposed only where explicitly required. Probability laws are denoted by capital letters and, when absolutely continuous with respect to Lebesgue measure, their densities by the corresponding lower-case letters. In particular, $\widetilde { Q } _ { t }$ and $\widetilde { q } _ { t }$ denote the frozen reference law and density, while $Q _ { t } ^ { \tilde { \eta } }$ and $q _ { t } ^ { \eta }$ denote the information-projected law and density associated with design η. Expressions involving division by $q _ { t } ^ { \eta }$ are understood on its positive support. Expectations are written $\bar { \mathbb { E } } _ { Q } [ \cdot ]$ or $\mathbb { E } _ { q } [ \cdot ]$ interchangeably when q is the density of $Q .$

Weak continuity equations. Unless the displayed densities and vector fields are regular enough to justify a strong formulation, all continuity equations are understood in the distributional sense. Specifically, for

$$
\begin{array} { r } { \partial _ { t } q _ { t } + \nabla \cdot ( q _ { t } \boldsymbol { v } _ { t } ) = s _ { t } , } \end{array}\tag{17}
$$

the weak formulation is

$$
\int _ { 0 } ^ { 1 } \int _ { \Omega } \left( q _ { t } \partial _ { t } \zeta + q _ { t } v _ { t } \cdot \nabla \zeta + s _ { t } \zeta \right) d x d t = 0\tag{18}
$$

for every admissible smooth test function $\zeta$ with compact support in physical time and satisfying the corresponding spatial boundary convention. Whenever the map $t \mapsto \int f q _ { t }$ dx is absolutely continuous, this implies

$$
\frac { d } { d t } \mathbb { E } _ { q _ { t } } [ f ] = \mathbb { E } _ { q _ { t } } [ \nabla f \cdot v _ { t } ] + \int _ { \Omega } f ( x ) s _ { t } ( x ) d x\tag{19}
$$

for admissible time-independent scalar observables $f .$ In the conservative case $s _ { t } = 0 ,$ applying this identity coordinatewise to a sufficiently regular time-independent observable vector $\Phi _ { \eta } : \dot { \Omega } \stackrel { \smile } { \to } \mathbb { R } ^ { R }$ gives

$$
\frac { d } { d t } \mathbb { E } _ { q _ { t } } [ \Phi _ { \pmb { \eta } } ] = \mathbb { E } _ { q _ { t } } [ J _ { \Phi _ { \pmb { \eta } } } v _ { t } ] .\tag{20}
$$

Weighted $L ^ { 2 }$ geometry. For scalar or vector fields $f$ and $g$ with the required square integrability, write

$$
\langle f , g \rangle _ { L ^ { 2 } ( q ) } : = \mathbb { E } _ { q } [ f ^ { \top } g ] , \qquad \| f \| _ { L ^ { 2 } ( q ) } ^ { 2 } : = \mathbb { E } _ { q } [ \| f \| ^ { 2 } ] .\tag{21}
$$

For an admissible scalar function $\varphi ,$ , define the weighted gradient-energy seminorm

$$
\lVert \varphi \rVert _ { \dot { H } ^ { 1 } ( q ) } ^ { 2 } : = \mathbb { E } _ { q } [ \lVert \nabla \varphi \rVert ^ { 2 } ] .\tag{22}
$$

Whenever the weighted Dirichlet form is coercive modulo constants, as assumed explicitly in $\mathsf { A p - }$ pendix B.7, we use the mean-zero representative $\mathbb { E } _ { q } [ \varphi ] = 0$ and denote by $H _ { q }$ the corresponding completed Hilbert energy space, equipped with the norm $\| \varphi \| _ { H _ { q } } : = \| \varphi \| _ { \dot { H } ^ { 1 } ( q ) }$ . Thus the mean-zero condition fixes the constant gauge in the balanced Poisson problem; it is not a global assumption on every function appearing in the appendix. The screened space $H _ { q , \kappa }$ used in the finite-measure extension is defined separately in Corollary 4, where its zeroth-order term removes the constant null direction.

Observable and Jacobian conventions. For a design η, the observable map is $\Phi _ { \eta } : \Omega \to \mathbb { R } ^ { R }$ . Its Jacobian is taken in the row convention

$$
J _ { \Phi _ { \eta } } ( x ) \in \mathbb { R } ^ { R \times d } ,\tag{23}
$$

so that $J _ { \Phi _ { \eta } } v \in \mathbb { R } ^ { R }$ for $v \in \mathbb { R } ^ { d }$ and $J _ { \Phi _ { \eta } } ^ { \top } a \in \mathbb { R } ^ { d }$ for $a \in \mathbb { R } ^ { R }$ . Covariance matrices are

$$
\operatorname { C o v } _ { Q } ( Y , Z ) : = \mathbb { E } _ { Q } \left[ ( Y - \mathbb { E } _ { Q } [ Y ] ) ( Z - \mathbb { E } _ { Q } [ Z ] ) ^ { \top } \right] .\tag{24}
$$

Linear-algebra conventions. For a matrix M, Range(M) denotes its column space, ker $( M )$ its nullspace, and $M ^ { \dagger }$ its Moore–Penrose pseudoinverse. All covariance and Gram matrices appearing below are symmetric positive semidefinite. Positive definiteness is imposed only when invertibility is required. In particular, positive definiteness of the projected covariance is needed for local multiplier identifiability and implicit differentiation, but not for uniqueness of the information-projected law itself.

Time integration and equality conventions. The design-level actions are integrated against the same fixed probability measure $\rho$ on $[ 0 , 1 ] ,$ , unless a later comparison explicitly specializes to $\rho ( d t ) = d t$ . Equalities between densities, scalar potentials, and vector fields are understood almost everywhere with respect to the relevant base measure. Integrated identities involving nonnegative action densities are valid in the extended-valued sense unless a finite-action assumption is explicitly needed; in particular, conclusions obtained by subtracting two integrated actions are invoked only when the corresponding quantities are finite.

Compact notation. For compactness, when $( t , \eta )$ is fixed we write

$$
\widetilde { q } = \widetilde { q } _ { t } , \qquad q = q _ { t } ^ { \eta } , \qquad \phi = \Phi _ { \eta } , \qquad c = c _ { \eta } ( t ) , \qquad \lambda = \lambda _ { t , \eta } , \qquad u = u _ { t } .\tag{25}
$$

Expectations without an explicit subscript in a proof are taken under the law indicated by the surrounding expression. The conventions above are deliberately weaker than the hypotheses of individual results: existence of an information projection, differentiability of the calibrated multiplier, coercivity of the weighted Poisson problem, positivity of the projected density, and rank conditions are stated locally where they are used rather than imposed globally.

## B.2 FINITE-MEASURE TRANSPORT–REACTION SETUP

The finite-measure results below require additional notation beyond the balanced probability-law construction. We write a finite positive measure as

$$
\mu _ { t } ( d x ) = M _ { t } q _ { t } ( x ) d x , \qquad M _ { t } > 0 , \qquad \int q _ { t } ( x ) d x = 1 ,\tag{26}
$$

where $M _ { t }$ is the total mass and $q _ { t }$ is its normalized shape. The normalized shape is informationprojected relative to the frozen normalized reference shape $\widetilde { q } _ { t }$ , while the mass is treated as a separate scalar channel. We assume that the frozen reference shape satisfies

$$
\begin{array} { r } { \partial _ { t } \widetilde { q } _ { t } + \nabla \cdot \left( \widetilde { q } _ { t } \boldsymbol { u } _ { t } \right) = 0 , } \end{array}\tag{27}
$$

and that the reference mass evolution has a spatially constant logarithmic source rate $g _ { \mathrm { r e f } } ( t )$ . For a time-independent observable vector $\phi ,$ define the raw and normalized moments by

$$
\bar { c } _ { t } : = \int \phi d \mu _ { t } = M _ { t } \mathbb { E } _ { q _ { t } } [ \phi ] , \qquad c _ { t } : = \frac { \bar { c } _ { t } } { M _ { t } } = \mathbb { E } _ { q _ { t } } [ \phi ] .\tag{28}
$$

A corrected finite-measure evolution is taken to satisfy

$$
\partial _ { t } ( M _ { t } q _ { t } ) + \nabla \cdot ( M _ { t } q _ { t } ( u _ { t } + \delta _ { t } ) ) = M _ { t } q _ { t } \left( g _ { \mathrm { r e f } } ( t ) + \alpha _ { t } \right) ,\tag{29}
$$

where $\delta _ { t }$ is the transport correction and $\alpha _ { t }$ is the reaction-rate correction. Define

$$
h _ { \mathrm { s h a p e } } : = \frac { \partial _ { t } q + \nabla \cdot ( q u ) } { q } , \qquad h _ { \mathrm { u b } } : = h _ { \mathrm { s h a p e } } + \frac { \dot { M } } { M } - g _ { \mathrm { r e f } } .\tag{30}
$$

Equation equation 29 is then equivalent to

$$
- \nabla \cdot ( q \delta ) + q \alpha = q h _ { \mathrm { u b } } .\tag{31}
$$

For a fixed reaction penalty $\kappa > 0$ , define the instantaneous unbalanced Full action by

$$
a _ { \mathrm { f u l l } } ^ { \mathrm { u b } } ( t , \eta ) : = M _ { t } \operatorname* { i n f } _ { \delta , \alpha } \left\{ \mathbb { E } _ { q _ { t } } \big [ \| \delta \| ^ { 2 } + \kappa \alpha ^ { 2 } \big ] : - \nabla \cdot ( q _ { t } \delta ) + q _ { t } \alpha = q _ { t } h _ { \mathrm { u b } } \right\} .\tag{32}
$$

The separation of this forcing and action into normalized-shape and total-mass components is established in Corollary 5.

Remark 1 (Mass is separate from the normalized-shape exponential family). The mass $M _ { t }$ should not be encoded by appending the constant observable 1 to the normalized-shape information projection. Normalization already imposes $\mathbb { E } _ { q } [ 1 ] = 1$ , so doing so introduces an exactly redundant coordinate and makes the corresponding covariance matrix singular. The constant observable may, however, be included in the unbalanced Tangent observable vector after the shape has been cal ibrated. In that setting its spatial Jacobian vanishes, and its moment-rate equation isolates the total-mass reaction channel.

## B.3 FINITE-SAMPLE LAW BASELINE

Equation equation 6 defines the population ideal using the exact risk infimum. In the experiments, this value is approximated by the selection risk of a numerically optimized Law design. The resulting baseline defines a finite-search approximation to the admissible set rather than a certificate of global risk optimality. Risk tolerances apply to the selection distribution; their transfer to fresh observations is evaluated separately.

## B.4 CANONICAL INFORMATION PROJECTION

Technical conditions. Fix $( t , \eta )$ . Assume that $c _ { \eta } ( t )$ lies in the relative interior of the feasible moment set, that the dual objective $A _ { t , \eta } ( \lambda ) { - } \lambda ^ { \top } c _ { \eta } ( t )$ attains a minimizer $\lambda _ { t , \eta } .$ , and that $A _ { t , \eta }$ is finite on an open neighborhood of that minimizer. If $C _ { t , \eta } \succ 0$ in a minimal observable representation, the calibrated multiplier is locally unique; without this rank condition, the projected law remains unique but its coordinates need not be.

Proof of the information-projection part of Theorem 1. Let

$$
\mathcal { Q } _ { c } : = \left\{ Q \ll \widetilde { Q } _ { t } : \mathbb { E } _ { Q } [ \phi ] = c \right\}\tag{33}
$$

with the implicit finiteness conditions from equation 3. For any λ at which $A _ { t , \eta } ( \lambda ) < \infty$ , define the exponentially tilted law

$$
\frac { d Q _ { \lambda } } { d \widetilde { Q } _ { t } } ( x ) = \exp \bigl ( \lambda ^ { \top } \phi ( x ) - A _ { t , \pmb { \eta } } ( \lambda ) \bigr ) .\tag{34}
$$

The normalization follows directly from the definition of $A _ { t , \eta }$

By assumption, the dual objective

$$
D ( \boldsymbol { \lambda } ) : = A _ { t , \eta } ( \boldsymbol { \lambda } ) - \boldsymbol { \lambda } ^ { \top } \boldsymbol { c }\tag{35}
$$

attains its minimum at $\lambda = \lambda _ { t , \eta } ,$ , and $A _ { t , \eta }$ is finite on an open neighborhood of that point. On this neighborhood the standard differentiation formula for a log moment-generating function gives

$$
\nabla _ { \lambda } A _ { t , \eta } ( \lambda ) = \mathbb { E } _ { Q _ { \lambda } } [ \phi ] ,\tag{36}
$$

and

$$
\nabla _ { \lambda } ^ { 2 } A _ { t , \eta } ( \lambda ) = \operatorname { C o v } _ { Q _ { \lambda } } ( \phi , \phi ) .\tag{37}
$$

Since D is differentiable at an unconstrained minimizer,

$$
0 = \nabla D ( \lambda _ { t , \eta } ) = \mathbb { E } _ { Q _ { \lambda _ { t , \eta } } } [ \phi ] - c .\tag{38}
$$

Hence the tilted law $Q _ { \lambda _ { t , \eta } }$ belongs to $\mathcal { Q } _ { c }$ . In the shorthand of equation 25, the calibration identity is

$$
\mathbb { E } _ { Q _ { t } ^ { \eta } } [ \phi ] = c .\tag{39}
$$

Set $Q ^ { \star } = Q _ { \lambda _ { t , \eta } }$ . For any feasible $Q \in { \mathcal { Q } } _ { c }$ <sub>c</sub> for which the displayed relative entropies are finite, the Radon–Nikodym chain rule and equation 34 give

$$
\begin{array} { r l } & { { \cal D } _ { \mathrm { K L } } ( { \cal Q } \| \widetilde { Q } _ { t } ) = { \cal D } _ { \mathrm { K L } } ( { \cal Q } \| { \cal Q } ^ { \star } ) + { \mathbb E } _ { { \cal Q } } \left[ \log \frac { d { \cal Q } ^ { \star } } { d \widetilde { Q } _ { t } } \right] } \\ & { \quad \quad \quad \quad = { \cal D } _ { \mathrm { K L } } ( { \cal Q } \| { \cal Q } ^ { \star } ) + \lambda _ { t , \eta } ^ { \top } { \mathbb E } _ { { \cal Q } } [ \phi ] - { \cal A } _ { t , \eta } ( \lambda _ { t , \eta } ) } \\ & { \quad \quad \quad = { \cal D } _ { \mathrm { K L } } ( { \cal Q } \| { \cal Q } ^ { \star } ) + \lambda _ { t , \eta } ^ { \top } c - { \cal A } _ { t , \eta } ( \lambda _ { t , \eta } ) . } \end{array}\tag{40}
$$

For $Q = Q ^ { \star }$ the final two terms equal $D _ { \mathrm { K L } } ( Q ^ { \star } \| \widetilde { Q } _ { t } )$ . Therefore

$$
D _ { \mathrm { K L } } ( Q \lVert \widetilde { Q } _ { t } ) - D _ { \mathrm { K L } } ( Q ^ { \star } \lVert \widetilde { Q } _ { t } ) = D _ { \mathrm { K L } } ( Q \lVert Q ^ { \star } ) \geq 0 .\tag{41}
$$

Equality holds only when $Q = Q ^ { \star }$ almost everywhere. Thus the I-projection is unique at the law level and has the density in equation 7.

Finally, equation 37 evaluated at the calibrated multiplier gives

$$
\nabla _ { \lambda } ^ { 2 } A _ { t , \eta } ( \lambda _ { t , \eta } ) = C _ { t , \eta } .\tag{42}
$$

If $C _ { t , \eta }$ is positive definite in a minimal observable representation, the dual Hessian is nonsingular at the optimizer. By continuity of the Hessian, the dual objective is strictly convex on a sufficiently small neighborhood, and the multiplier is locally unique. If the observable coordinates are redundant, different multipliers can generate the same log-density up to an additive constant on the support of $\widetilde { Q } _ { t }$ , so multiplier non-uniqueness does not contradict law-level uniqueness. □

Proposition 3 (Finite-reference information-projection feasibility). $F i x \left( t , \eta \right)$ and let the empirical reference be $\begin{array} { r } { \widehat { \tilde { Q } } _ { N } = \sum _ { i = 1 } ^ { N } \bar { w } _ { i } \delta _ { x _ { i } } } \end{array}$ with $\bar { w } _ { i } > 0$ and $\begin{array} { r } { \sum _ { i } \bar { w } _ { i } = 1 } \end{array}$ . Write $z _ { i } = \Phi _ { \eta } ( x _ { i } )$ and $\kappa _ { N } =$ conv $\{ z _ { 1 } , \dots , z _ { N } \}$ . For a target moment bc, the empirical constraint $\textstyle \sum _ { i } p _ { i } z _ { i } = { \widehat { c } } , p _ { i } \geq 0 , \sum _ { i } p _ { i } = 1$ is feasible if and only $i f { \widehat { c } } \in { \widehat { K } } _ { N }$ , and whenever it is feasible the empirical KL projection relative to $( \dot { w } _ { i } ) _ { i = 1 } ^ { N }$ is unique. $H \widehat { c } \in$ relint $( \mathcal { K } _ { N } )$ , the minimizer has strictly positive weights oftheform

$$
p _ { i } ( \lambda ) = \frac { \bar { w } _ { i } \exp ( \lambda ^ { \top } z _ { i } ) } { \sum _ { j = 1 } ^ { N } \bar { w } _ { j } \exp ( \lambda ^ { \top } z _ { j } ) } , \qquad \sum _ { i = 1 } ^ { N } p _ { i } ( \lambda ) z _ { i } = \widehat { c } .\tag{43}
$$

After removal of affine redundancies in the observable coordinates, the corresponding finite multi plier is unique and the projected covariance $\begin{array} { r } { \widehat { C } = \sum _ { i } p _ { i } ( \lambda ) ( z _ { i } - \widehat { c } ) ( z _ { i } - \widehat { c } ) ^ { \top } } \end{array}$ is positive definite. If bc lies on the relative boundary of $\kappa _ { N }$ , the unique empirical KL projection still exists but cannot be represented by a finite full-support exponential tilt; it is obtained only as a limiting tilt with vanishing weights on part of the empirical support. $I f { \widehat { c } } \notin { \cal K } _ { N }$ , no exact empirical information projection supported on the reference bank exists.

Proof. The moments attainable by probability weights on the empirical support are exactly the convex combinations of $z _ { 1 } , \dots , z _ { N }$ , which proves the feasibility statement. On the feasible simplex the objective $\textstyle \sum _ { i } p _ { i } \log ( p _ { i } / \bar { w } _ { i } )$ is strictly convex, so its minimizer is unique. For a target in relint $( \ d \mathcal { K } _ { N } )$ the equality-constrained first-order conditions give the exponential weights in equation $ 4 3 ;$ conversely, every finite multiplier gives strictly positive weights and therefore a moment in relint $( \ d \mathcal { K } _ { N } )$ In a minimal affine representation, zero variance of a nonzero linear combination of the features would make that combination constant on the full empirical support, contradicting minimality, so $\widehat C \succ 0 .$ . A target on a proper relative face requires the minimizing distribution to place zero mass outside that face and hence cannot be produced by finite strictly positive exponential weights. Outside $\kappa _ { N }$ the moment constraint is infeasible. □

Corollary 1 (Finite-measure information projection with prescribed mass). Fix $M > 0$ and a raw finite-measure moment vector c¯. Let $c = \bar { c } / M$ , and suppose the conditions of Theorem 1 hold for this normalized moment vector. Then the finite measure $\mu ^ { \star } = M Q ^ { \star }$ , where $\dot { Q } ^ { \star }$ is the information projection in Theorem 1, is the uniquefinite measure ofmass M whose normalized shape minimizes $D _ { \mathrm { K L } } ( Q \lVert \widetilde { Q } _ { t } )$ subject to $\begin{array} { r } { \int \phi d ( M Q ) = \bar { c } . } \end{array}$ Equivalently, $\mu ^ { \star }$ uniquely minimizes $M D _ { \mathrm { K L } } ( Q \| \widetilde { Q } _ { t } )$ over the same fixed-mass fiber, and its normalized density has the same exponential-tilt representation equation 7.

Proof. Every finite measure µ of mass M can be written uniquely as $\mu = M Q$ with Q a probability law, and the raw constraint $\textstyle \int \phi d \mu = { \bar { c } }$ is equivalent to $\mathbb { E } _ { Q } [ \dot { \phi } ] = \dot { c } / \dot { M } = c .$ . Since multiplication of the objective by the positive constant M does not change its minimizer, Theorem 1 applies directly to the normalized shape. □

## B.5 PHYSICAL-TIME MULTIPLIER DYNAMICS

Proposition 4 (Multiplier dynamics). Under the regularity assumptions stated below and in a minimal observable representation with $C _ { t , \eta } \succ 0$ , the calibrated multiplier satisfies equation 8.

Technical conditions. Fix η and a time t. Assume that $c _ { \eta } ( s )$ is continuously differentiable in s near t, that the reference density and velocity are regular enough for the reference continuity equation and the integrations by parts below, and that the normalized exponential family $q _ { \lambda , s }$ defined below is continuously differentiable in $( \lambda , s )$ near $( \lambda _ { t , \eta } , t )$ with differentiation and integration interchangeable. In a minimal observable representation, assume $C _ { t , \eta } \succ 0$

Proof of Proposition $^ { 4 . }$ Fix η and suppress it from the notation. For λ and t in the neighborhood specified above, define

$$
q _ { \lambda , t } ( x ) : = \widetilde { q } _ { t } ( x ) \exp \bigl ( \lambda ^ { \top } \phi ( x ) - A _ { t } ( \lambda ) \bigr ) ,\tag{44}
$$

and the calibration residual

$$
\mathcal { F } ( \lambda , t ) : = \mathbb { E } _ { q _ { \lambda , t } } [ \phi ] - c _ { t } .\tag{45}
$$

By the stated regularity assumptions, $\mathcal { F }$ is continuously differentiable near $( \lambda _ { t } , t )$ , and

$$
\partial _ { \lambda } \mathcal { F } ( \lambda _ { t } , t ) = \mathrm { C o v } _ { Q _ { t } ^ { \eta } } ( \phi , \phi ) = C _ { t , \eta } .\tag{46}
$$

Since $C _ { t , \eta } \succ 0$ , the implicit-function theorem gives a locally unique continuously differentiable calibrated path $t \mapsto \lambda _ { t }$ satisfying $\mathcal { F } ( \lambda _ { t } , t ) = 0$ . In particular, $\dot { \lambda } _ { t }$ exists, so the differentiations below are justified. Let

$$
s _ { t } ( x ) : = \partial _ { t } \log \widetilde { q } _ { t } ( x )\tag{47}
$$

where the reference density is positive. From the exponential representation,

$$
\log q _ { t } ( x ) = \log \widetilde { q } _ { t } ( x ) + \lambda _ { t } ^ { \top } \phi ( x ) - A _ { t } ( \lambda _ { t } ) .\tag{48}
$$

At fixed $\lambda ,$ differentiation of the log-partition function in physical time gives

$$
\partial _ { t } A _ { t } ( \lambda ) = \mathbb { E } _ { q _ { \lambda , t } } [ s _ { t } ] .\tag{49}
$$

Along the calibrated path $t \mapsto \lambda _ { t }$

$$
\frac { d } { d t } A _ { t } ( \lambda _ { t } ) = \mathbb { E } _ { Q _ { t } ^ { \eta } } [ s _ { t } ] + \dot { \lambda } _ { t } ^ { \top } c _ { t } ,\tag{50}
$$

where $c _ { t } = \mathbb { E } _ { Q _ { t } ^ { \eta } } [ \phi ]$

Differentiating the calibration identity $c _ { t } = \mathbb { E } _ { Q _ { t } ^ { \eta } } [ \phi ]$ and using that $\phi$ has no explicit physical-time dependence yields

$$
\begin{array} { r l } & { \dot { c } _ { t } = \mathbb { E } _ { Q _ { t } ^ { \eta } } [ \phi \partial _ { t } \log q _ { t } ] } \\ & { \quad = \operatorname { C o v } _ { Q _ { t } ^ { \eta } } ( \phi , s _ { t } ) + C _ { t , \eta } \dot { \lambda } _ { t } . } \end{array}\tag{51}
$$

It remains to express the covariance with $s _ { t }$ without differentiating the reference density explicitly. Let

$$
w _ { t } ( x ) : = \frac { q _ { t } ( x ) } { \widetilde { q } _ { t } ( x ) } = \exp \bigl ( \lambda _ { t } ^ { \top } \phi ( x ) - A _ { t } ( \lambda _ { t } ) \bigr ) .\tag{52}
$$

The reference continuity equation gives $\partial _ { t } \widetilde { q } _ { t } = - \nabla \cdot ( \widetilde { q } _ { t } \boldsymbol { u } _ { t } )$ . Therefore, using the assumed boundary conditions,

$$
\begin{array} { r l } {  { \mathbb { E } _ { Q _ { t } ^ { \eta } } \bigl [ s _ { t } \bigr ] = \int w _ { t } \partial _ { t } \widetilde { q } _ { t } d x } } \\ & { = - \int w _ { t } \nabla \cdot ( \widetilde { q } _ { t } \boldsymbol { u } _ { t } ) d x } \\ & { = \int \widetilde { q } _ { t } \boldsymbol { u } _ { t } \cdot \nabla w _ { t } d x . } \end{array}\tag{53}
$$

Since

$$
\nabla w _ { t } = w _ { t } J _ { \phi } ^ { \top } \lambda _ { t } ,\tag{54}
$$

we obtain

$$
\mathbb { E } _ { Q _ { t } ^ { \eta } } [ s _ { t } ] = \mathbb { E } _ { Q _ { t } ^ { \eta } } \big [ \lambda _ { t } ^ { \top } J _ { \phi } u _ { t } \big ] = \mathbb { E } _ { Q _ { t } ^ { \eta } } [ \lambda _ { t } ^ { \top } m _ { t } ] .\tag{55}
$$

Similarly, componentwise integration by parts gives

$$
\begin{array} { r l } & { \mathbb { E } _ { Q _ { t } ^ { \eta } } [ \phi s _ { t } ] = \displaystyle \int w _ { t } \phi \partial _ { t } \widetilde { q } _ { t } d x } \\ & { \quad \quad \quad = \displaystyle \int \widetilde { q } _ { t } u _ { t } \cdot \nabla ( w _ { t } \phi ) d x } \\ & { \quad \quad \quad = \mathbb { E } _ { Q _ { t } ^ { \eta } } [ m _ { t } ] + \mathbb { E } _ { Q _ { t } ^ { \eta } } [ \phi \lambda _ { t } ^ { \top } m _ { t } ] . } \end{array}\tag{56}
$$

Subtracting $\mathbb { E } [ \phi ] \mathbb { E } [ s _ { t } ] = c _ { t } \mathbb { E } [ \lambda _ { t } ^ { \top } m _ { t } ]$ yields

$$
\mathrm { C o v } _ { Q _ { t } ^ { \eta } } ( \phi , s _ { t } ) = \mathbb { E } _ { Q _ { t } ^ { \eta } } [ m _ { t } ] + \mathrm { C o v } _ { Q _ { t } ^ { \eta } } ( \phi , \lambda _ { t } ^ { \top } m _ { t } ) .\tag{57}
$$

Substitution into equation 51 gives

$$
C _ { t , \eta } \dot { \lambda } _ { t } = \dot { c } _ { t } - \mathbb { E } _ { Q _ { t } ^ { \eta } } [ m _ { t } ] - \operatorname { C o v } _ { Q _ { t } ^ { \eta } } ( \phi , \lambda _ { t } ^ { \top } m _ { t } ) ,\tag{58}
$$

which is equation 8. Positive definiteness of $C _ { t , \eta }$ in the stated minimal representation makes this a uniquely solvable $R \times R$ linear system. This proves the multiplier-dynamics clause of Theorem 1. □

Corollary 2 (Multiplier dynamics for a finite-measure path). Suppose the assumptions of Proposition 4 hold for the normalized shape $q _ { t }$ , and suppose ${ M } _ { t } ~ > ~ 0$ and the raw moment path $\bar { c } _ { t }$ in equation 28 are continuously differentiable. Then the same covariance equation holds with the normalized target

$$
c _ { t } = { \frac { \bar { c } _ { t } } { M _ { t } } } , \qquad \dot { c } _ { t } = { \frac { \dot { \bar { c } } _ { t } } { M _ { t } } } - { \frac { \dot { M } _ { t } } { M _ { t } } } c _ { t } ,\tag{59}
$$

namely

$$
C _ { t , \eta } \dot { \lambda } _ { t , \eta } = \frac { \dot { \bar { c } } _ { t } } { M _ { t } } - \frac { \dot { M } _ { t } } { M _ { t } } c _ { t } - \mathbb { E } _ { Q _ { t } ^ { \eta } } [ m _ { t , \eta } ] - \operatorname { C o v } _ { Q _ { t } ^ { \eta } } \bigl ( \Phi _ { \eta } , \boldsymbol { \lambda } _ { t , \eta } ^ { \top } m _ { t , \eta } \bigr ) .\tag{60}
$$

Proof. The information projection acts only on the normalized shape, so Proposition 4 applies with target $c _ { t } = \bar { c } _ { t } / M _ { t }$ . Equation equation 59 is the quotient rule, and substitution into equation 8 gives equation 60. □

## B.6 DENSITY-FREE CONTINUITY FORCING

Proposition 5 (Density-free continuity forcing). Under the assumptions of Proposition 4, the reference-relative continuity residual $h _ { t , \eta }$ defined in Section 3 is given by equation 9 and has zero projected-law mean:

$$
\mathbb { E } _ { Q _ { t } ^ { \eta } } [ h _ { t , \eta } ] = 0 .\tag{61}
$$

ProofofProposition 5. Again fix $\eta$ and write $q _ { t } = q _ { t } ^ { \eta }$ . For any positive sufficiently regular density q and vector field $u ,$

$$
\partial _ { t } q + \nabla \cdot ( q u ) = q \left[ ( \partial _ { t } + u \cdot \nabla ) \log q + \nabla \cdot u \right] .\tag{62}
$$

The reference continuity equation implies

$$
\begin{array} { r } { ( \partial _ { t } + u _ { t } \cdot \nabla ) \log \widetilde { q } _ { t } + \nabla \cdot u _ { t } = 0 . } \end{array}\tag{63}
$$

Using equation 48, the time independence of $\phi ,$ and $m _ { t } = J _ { \phi } u _ { t }$

$$
\left( \partial _ { t } + \boldsymbol { u } _ { t } \cdot \nabla \right) \log q _ { t } + \nabla \cdot \boldsymbol { u } _ { t } = \dot { \lambda } _ { t } ^ { \top } \boldsymbol { \phi } + \lambda _ { t } ^ { \top } \boldsymbol { m } _ { t } - \frac { d } { d t } \boldsymbol { A } _ { t } ( \lambda _ { t } ) .\tag{64}
$$

Equations equation 50 and equation 55 give

$$
\frac { d } { d t } A _ { t } ( \lambda _ { t } ) = \dot { \lambda } _ { t } ^ { \top } c _ { t } + \mathbb { E } _ { Q _ { t } ^ { \eta } } [ \lambda _ { t } ^ { \top } m _ { t } ] .\tag{65}
$$

Hence the bracket in equation 62 equals

$$
\dot { \lambda } _ { t } ^ { \top } ( \phi - c _ { t } ) + \lambda _ { t } ^ { \top } \left( m _ { t } - \mathbb { E } _ { Q _ { t } ^ { \eta } } [ m _ { t } ] \right) = h _ { t , \eta } .\tag{66}
$$

Substituting into equation 62 gives $\partial _ { t } q _ { t } + \nabla \cdot \left( q _ { t } u _ { t } \right) = q _ { t } h _ { t , \eta }$ and proves the explicit formula equation 9.

Finally, calibration gives $\mathbb { E } _ { Q _ { \star } ^ { \eta } } [ \phi - c _ { t } ] = 0$ , and the second centered term also has zero expectation, proving equation 61. Equivalently, the zero-mean property follows by integrating the referencerelative continuity identity over space and using conservation of total mass. Together with Proposition 4 and the information-projection argument above, this completes the proof of Theorem 1.

Corollary 3 (Density-free unbalanced forcing). Under the assumptions of Corollary 2, the normalized-shape residual satisfies

$$
h _ { \mathrm { s h a p e } } ( x ) = \dot { \lambda } _ { t } ^ { \top } ( \phi ( x ) - c _ { t } ) + \lambda _ { t } ^ { \top } \left( m _ { t } ( x ) - \mathbb { E } _ { q _ { t } } [ m _ { t } ] \right) ,\tag{67}
$$

and therefore the finite-measure reference-relative forcing in equation 30 is

$$
h _ { \mathrm { u b } } ( x ) = \dot { \lambda } _ { t } ^ { \top } ( \phi ( x ) - c _ { t } ) + \lambda _ { t } ^ { \top } \left( m _ { t } ( x ) - \mathbb { E } _ { q _ { t } } [ m _ { t } ] \right) + \frac { \dot { M } _ { t } } { M _ { t } } - g _ { \mathrm { r e f } } ( t ) .\tag{68}
$$

Moreover,

$$
\mathbb { E } _ { q _ { t } } [ h _ { \mathrm { s h a p e } } ] = 0 , \qquad \mathbb { E } _ { q _ { t } } [ h _ { \mathrm { u b } } ] = \frac { \dot { M } _ { t } } { M _ { t } } - g _ { \mathrm { r e f } } ( t ) .\tag{69}
$$

Proof. Equation equation $^ { 6 7 }$ is Proposition 5 applied to the normalized shape. Adding the scalar mass-rate discrepancy from equation 30 gives equation 68. Taking expectation and using the centering in equation 67 gives equation 69. □

## B.7 MINIMUM-ENERGY FULL CORRECTION

Fix $( t , \eta )$ and write $q = q _ { t } ^ { \eta }$ and $h = h _ { t , \eta } .$ . Let $H _ { q }$ be the mean-zero weighted energy space defined in Appendix B.1. Assume that the weighted Dirichlet form is coercive on this space, for example through a weighted Poincare inequality, and that ´ $\varphi \mapsto \mathbb { E } _ { q } [ h \varphi ]$ defines a bounded linear functional on $\bar { H _ { q } } .$ . The weak Poisson equation is

$$
\mathbb { E } _ { q } [ \nabla \psi \cdot \nabla \varphi ] = - \mathbb { E } _ { q } [ h \varphi ] \qquad { \mathrm { f o r ~ e v e r y ~ } } \varphi \in H _ { q } .\tag{70}
$$

Proof of the minimum-energy and Poisson parts of Theorem 2. Define the bilinear form

$$
a ( \psi , \varphi ) : = \mathbb { E } _ { q } [ \nabla \psi \cdot \nabla \varphi ]\tag{71}
$$

on the mean-zero energy space $H _ { q } .$ . By construction of the energy norm,

$$
| a ( \psi , \varphi ) | \leq \| \psi \| _ { H _ { q } } \left\| \varphi \right\| _ { H _ { q } } , \qquad a ( \psi , \psi ) = \| \psi \| _ { H _ { q } } ^ { 2 } .\tag{72}
$$

The weighted Poincare/coercivity assumption ensures that the mean-zero gauge removes the con- ´ stant null direction and that this energy space is the appropriate Hilbert space for the problem. By assumption,

$$
\ell ( \varphi ) : = - \mathbb { E } _ { q } [ h \varphi ]\tag{73}
$$

is a bounded linear functional on $H _ { q }$ . Lax–Milgram therefore gives a unique $\psi \in H _ { q }$ satisfying

$$
a ( \psi , \varphi ) = \ell ( \varphi ) \qquad { \mathrm { f o r ~ e v e r y ~ } } \varphi \in H _ { q } ,\tag{74}
$$

which is exactly equation 70. Since $\mathbb { E } _ { q } [ h ] = 0$ , the restriction to mean-zero test functions causes no loss: for any admissible $\varphi ,$ , setting $\overset { \cdot } { \varphi _ { 0 } } : = \varphi - \mathbb { E } _ { q } [ \varphi ]$ gives $\nabla \varphi _ { 0 } = \nabla \varphi$ and $\mathbb { E } _ { q } [ h \varphi _ { 0 } ] = \mathbb { E } _ { q } [ h \varphi ]$ so the same weak identity holds for arbitrary admissible test functions. Under the stated periodic, decay, or no-flux boundary conditions, integration by parts converts this weak identity into

$$
\nabla \cdot ( q \nabla \psi ) = q h\tag{75}
$$

in the distributional sense, together with the gauge $\mathbb { E } _ { q } [ \psi ] = 0$

Set

$$
\delta ^ { \star } : = - \nabla \psi .\tag{76}
$$

Then $\delta ^ { \star } \in L ^ { 2 } ( q ;  { \mathbb { R } } ^ { d } )$ . The weak Poisson equation gives, for every $\varphi \in H _ { q }$

$$
\mathbb { E } _ { q } [ \delta ^ { \star } \cdot \nabla \varphi ] = \mathbb { E } _ { q } [ h \varphi ] .\tag{77}
$$

This is precisely the weak form of

$$
\nabla \cdot \left( q \delta ^ { \star } \right) = - q h ,\tag{78}
$$

so $u _ { t } + \delta ^ { \star }$ satisfies the projected-law continuity equation.

Now let $\delta \in L ^ { 2 } ( q ; \mathbb { R } ^ { d } )$ be any other feasible correction and define $w : = \delta - \delta ^ { \star }$ . Subtracting the two weak constraints gives the identity first on the admissible test class. Because $v \in L ^ { 2 } ( q ; \mathbb { R } ^ { d } )$ , the

map $\varphi \mapsto \mathbb { E } _ { q } [ w \cdot \nabla \varphi ]$ is continuous in the $H _ { q }$ energy norm; density therefore extends the identity to all of $H _ { q } \colon$

$$
\mathbb { E } _ { q } [ w \cdot \nabla \varphi ] = 0 \qquad { \mathrm { f o r ~ e v e r y ~ } } \varphi \in H _ { q } .\tag{79}
$$

Taking $\varphi = \psi$ yields

$$
\mathbb { E } _ { q } [ w \cdot \nabla \psi ] = 0 ,\tag{80}
$$

and therefore, because $\delta ^ { \star } = - \nabla \psi$

$$
\mathbb { E } _ { q } [ \delta ^ { \star } \cdot w ] = 0 .\tag{81}
$$

Consequently

$$
\begin{array} { r l } & { \mathbb { E } _ { q } [ \left\| \delta \right\| ^ { 2 } ] = \mathbb { E } _ { q } [ \left\| \delta ^ { \star } + w \right\| ^ { 2 } ] } \\ & { \quad \quad \quad = \mathbb { E } _ { q } [ \left\| \delta ^ { \star } \right\| ^ { 2 } ] + \mathbb { E } _ { q } [ \left\| w \right\| ^ { 2 } ] } \\ & { \quad \quad \quad \geq \mathbb { E } _ { q } [ \left\| \delta ^ { \star } \right\| ^ { 2 } ] . } \end{array}\tag{82}
$$

Equality holds if and only if $w = 0$ q-almost everywhere. Thus $\delta ^ { \star }$ is the unique minimum-energy feasible correction, and

$$
a _ { \mathrm { f u l l } } ( t , \pmb { \eta } ) = \mathbb { E } _ { q } [ \| \delta ^ { \star } \| ^ { 2 } ] = \mathbb { E } _ { q } [ \| \nabla \psi \| ^ { 2 } ] .\tag{83}
$$

This proves the minimum-energy and Poisson assertions in Theorem 2. The weighted $H ^ { - 1 }$ identity in equation 13 is proved in Appendix B.8. □

The screened operator below is the reference-relative analogue of the transport–reaction operator in Sobolev–Fisher discrepancy and Unbalanced Sobolev Descent (Mroueh & Rigotti, 2020). Its gradient and zeroth-order terms correspond to transport and reaction costs, respectively; the present forcing comes from the prescribed finite-measure path, with the boundary conditions stated in this appendix.

Corollary 4 (Minimum-energy unbalanced move–reaction correction). Fix $( t , \eta )$ , write $q = q _ { t } ^ { \eta }$ $M = M _ { t }$ , and $h _ { \mathrm { u b } } = h _ { \mathrm { u b } , t , \eta } ,$ and let $\kappa > 0 .$ . Let $H _ { q , \kappa }$ be the completion of the admissible test functions under

$$
\left\| \psi \right\| _ { H _ { q , \kappa } } ^ { 2 } : = \mathbb { E } _ { q } \left[ \left\| \nabla \psi \right\| ^ { 2 } + \frac { \psi ^ { 2 } } { \kappa } \right] .\tag{84}
$$

Assume that $\varphi \mapsto \mathbb { E } _ { q } [ h _ { \mathrm { u b } } \varphi ]$ is a bounded linear functional on $H _ { q , \kappa } .$ . Then there exists a unique $\psi \in H _ { q , \kappa }$ satisfying

$$
\mathbb { E } _ { q } \bigg [ \nabla \psi \cdot \nabla \varphi + \frac { 1 } { \kappa } \psi \varphi \bigg ] = \mathbb { E } _ { q } [ h _ { \mathrm { u b } } \varphi ] \qquad f o r e \nu e r y \varphi \in H _ { q , \kappa } .\tag{85}
$$

Under the stated boundary and regularity assumptions, this is the weakform of

$$
- \nabla \cdot ( q \nabla \psi ) + \frac { q } { \kappa } \psi = q h _ { \mathrm { u b } } .\tag{86}
$$

The unique minimum-energy correction in equation 32 is

$$
\delta ^ { \star } = \nabla \psi , \qquad \alpha ^ { \star } = \frac { \psi } { \kappa } ,\tag{87}
$$

and

$$
a _ { \mathrm { f u l l } } ^ { \mathrm { u b } } ( t , \eta ) = M \mathbb { E } _ { q } \left[ \left. \nabla \psi \right. ^ { 2 } + \frac { \psi ^ { 2 } } { \kappa } \right] .\tag{88}
$$

Proof. Define

$$
a _ { \kappa } ( \psi , \varphi ) : = \mathbb { E } _ { q } \left[ \nabla \psi \cdot \nabla \varphi + \frac { 1 } { \kappa } \psi \varphi \right] .\tag{89}
$$

This bilinear form is continuous and coercive on $H _ { q , \kappa }$ because $a _ { \kappa } ( \psi , \psi ) = \left\| \psi \right\| _ { H _ { q , \kappa } } ^ { 2 }$ . Lax–Milgram therefore gives a unique solution of equation 85. Integration by parts yields equation 86 whenever the strong form is justified.

Set $( \delta ^ { \star } , \alpha ^ { \star } ) = ( \nabla \psi , \psi / \kappa )$ . Equation equation 85 gives

$$
\mathbb { E } _ { q } [ \delta ^ { \star } \cdot \nabla \varphi + \alpha ^ { \star } \varphi ] = \mathbb { E } _ { q } [ h _ { \mathrm { u b } } \varphi ] ,\tag{90}
$$

which is the weak form of equation 31, so $( \delta ^ { \star } , \alpha ^ { \star } )$ is feasible. Let $( \delta , \alpha )$ be any other feasible pair and write $w = \delta - \delta ^ { \star }$ and $\gamma = \alpha - \alpha ^ { \star }$ . Subtracting the two weak constraints gives

$$
\mathbb { E } _ { q } [ w \cdot \nabla \varphi + \gamma \varphi ] = 0\tag{91}
$$

for every $\varphi \in H _ { q , \kappa }$ . Taking $\varphi = \psi$ yields

$$
\mathbb { E } _ { q } [ w \cdot \delta ^ { \star } + \kappa \gamma \alpha ^ { \star } ] = \mathbb { E } _ { q } [ w \cdot \nabla \psi + \gamma \psi ] = 0 .\tag{92}
$$

Hence, in the product norm induced by $\mathbb { E } _ { q } [ \| \delta \| ^ { 2 } + \kappa \alpha ^ { 2 } ]$ ，

$$
\begin{array} { r l } & { \mathbb { E } _ { q } [ \| \delta \| ^ { 2 } + \kappa \alpha ^ { 2 } ] = \mathbb { E } _ { q } [ \| \delta ^ { \star } \| ^ { 2 } + \kappa ( \alpha ^ { \star } ) ^ { 2 } ] + \mathbb { E } _ { q } [ \| w \| ^ { 2 } + \kappa \gamma ^ { 2 } ] } \\ & { \qquad \geq \mathbb { E } _ { q } [ \| \delta ^ { \star } \| ^ { 2 } + \kappa ( \alpha ^ { \star } ) ^ { 2 } ] . } \end{array}\tag{93}
$$

Equality holds only when $w = 0$ and $\gamma = 0$ in the corresponding $L ^ { 2 } ( q )$ spaces. Multiplication by $\bar { M } > 0$ proves both uniqueness and equation 88. □

Corollary 5 (Mass–shape decomposition of unbalanced Full action). Under the assumptions of Corollary 4, define the spatially constant logarithmic mass-rate discrepancy

$$
b _ { t } : = \dot { \frac { M _ { t } } { M _ { t } } } - g _ { \mathrm { r e f } } ( t ) ,\tag{94}
$$

so that

$$
h _ { \mathrm { u b } } = h _ { \mathrm { s h a p e } } + b _ { t } , \qquad \mathbb { E } _ { q _ { t } } [ h _ { \mathrm { s h a p e } } ] = 0 .\tag{95}
$$

Let $\psi _ { \mathrm { s h a p e } } \in H _ { q _ { t } , \kappa }$ denote the unique solution of

$$
\mathbb { E } _ { q _ { t } } \bigg [ \nabla \psi _ { \mathrm { s h a p e } } \cdot \nabla \varphi + \frac { 1 } { \kappa } \psi _ { \mathrm { s h a p e } } \varphi \bigg ] = \mathbb { E } _ { q _ { t } } \big [ h _ { \mathrm { s h a p e } } \varphi \big ] \qquad f o r e \nu e r y \varphi \in H _ { q _ { t } , \kappa } .\tag{96}
$$

Then

$$
\mathbb { E } _ { q _ { t } } [ \psi _ { \mathrm { s h a p e } } ] = 0 , \qquad \psi = \psi _ { \mathrm { s h a p e } } + \kappa b _ { t } ,\tag{97}
$$

where ψ is the Full screened potentialfrom Corollary 4. Consequently,

$$
\delta ^ { \star } = \nabla \psi _ { \mathrm { s h a p e } } , \qquad \alpha ^ { \star } = b _ { t } + \frac { \psi _ { \mathrm { s h a p e } } } { \kappa } ,\tag{98}
$$

and the instantaneous unbalanced Full action decomposes orthogonally as

$$
a _ { \mathrm { f u l l } } ^ { \mathrm { u b } } ( t , \eta ) = M _ { t } \kappa b _ { t } ^ { 2 } + M _ { t } \mathbb { E } _ { q _ { t } } \left[ \| \nabla \psi _ { \mathrm { s h a p e } } \| ^ { 2 } + \frac { \psi _ { \mathrm { s h a p e } } ^ { 2 } } { \kappa } \right] .\tag{99}
$$

The first term is the cost of the spatially uniform reaction required to correct the total-mass rate, while the second is the screened correction cost of the mean-zero shape forcing.

Proof. Because $\mathbb { E } _ { q _ { t } } [ h _ { \mathrm { s h a p e } } ] = 0 .$ , testing equation 96 with $\varphi = 1$ gives

$$
\frac { 1 } { \kappa } \mathbb { E } _ { q _ { t } } [ \psi _ { \mathrm { s h a p e } } ] = 0 ,\tag{100}
$$

and hence $\mathbb { E } _ { q _ { t } } [ \psi _ { \mathrm { s h a p e } } ] = 0$

Since constants have zero spatial gradient,

$$
- \nabla \cdot ( q _ { t } \nabla ( \kappa b _ { t } ) ) + \frac { q _ { t } } { \kappa } ( \kappa b _ { t } ) = q _ { t } b _ { t } .\tag{101}
$$

Adding this identity to the screened equation for $\psi _ { \mathrm { s h a p e } }$ shows that

$$
\psi = \psi _ { \mathrm { s h a p e } } + \kappa b _ { t }\tag{102}
$$

solves

$$
- \nabla \cdot \left( q _ { t } \nabla \psi \right) + \frac { q _ { t } } { \kappa } \psi = q _ { t } ( h _ { \mathrm { s h a p e } } + b _ { t } ) = q _ { t } h _ { \mathrm { u b } } .\tag{103}
$$

Uniqueness in Corollary 4 therefore identifies this sum with the Full screened potential. Equation equation 98 follows immediately from equation 87.

Finally,

$$
\begin{array} { r l } & { \mathbb { E } _ { q _ { t } } \bigg [ \| \nabla \psi \| ^ { 2 } + \frac { \psi ^ { 2 } } { \kappa } \bigg ] = \mathbb { E } _ { q _ { t } } \bigg [ \| \nabla \psi _ { \mathrm { s h a p e } } \| ^ { 2 } + \frac { ( \psi _ { \mathrm { s h a p e } } + \kappa b _ { t } ) ^ { 2 } } { \kappa } \bigg ] } \\ & { \qquad = \mathbb { E } _ { q _ { t } } \bigg [ \| \nabla \psi _ { \mathrm { s h a p e } } \| ^ { 2 } + \frac { \psi _ { \mathrm { s h a p e } } ^ { 2 } } { \kappa } \bigg ] + 2 b _ { t } \mathbb { E } _ { q _ { t } } [ \psi _ { \mathrm { s h a p e } } ] + \kappa b _ { t } ^ { 2 } } \\ & { \qquad = \mathbb { E } _ { q _ { t } } \bigg [ \| \nabla \psi _ { \mathrm { s h a p e } } \| ^ { 2 } + \frac { \psi _ { \mathrm { s h a p e } } ^ { 2 } } { \kappa } \bigg ] + \kappa b _ { t } ^ { 2 } . } \end{array}\tag{104}
$$

Multiplying by $M _ { t }$ and using equation 88 proves equation 99.

Remark 2 (Operator form of the mass–shape decomposition). Suppose additionally that the weighted Dirichlet form admits the self-adjoint operator representation

$$
\mathsf { L } _ { q _ { t } } = - q _ { t } ^ { - 1 } \nabla \cdot ( q _ { t } \nabla )\tag{105}
$$

on $L ^ { 2 } ( q _ { t } )$ , so that $\mathsf { L } _ { q _ { t } } + \kappa ^ { - 1 } I$ is invertible. Then

$$
\psi _ { \mathrm { s h a p e } } = \left( \mathsf { L } _ { q _ { t } } + \kappa ^ { - 1 } I \right) ^ { - 1 } h _ { \mathrm { s h a p e } } ,\tag{106}
$$

and Corollary 5 becomes

$$
a _ { \mathrm { f u l l } } ^ { \mathrm { u b } } ( t , \eta ) = M _ { t } \kappa \left( \frac { \dot { M } _ { t } } { M _ { t } } - g _ { \mathrm { r e f } } ( t ) \right) ^ { 2 } + M _ { t } \left. h _ { \mathrm { s h a p e } } , \left( \mathsf { L } _ { q _ { t } } + \kappa ^ { - 1 } I \right) ^ { - 1 } h _ { \mathrm { s h a p e } } \right. _ { L ^ { 2 } ( q _ { t } ) } .\tag{107}
$$

Thus the spatially constant mass-rate mismatch and the mean-zero shape forcing are orthogonal in the screened energy geometry.

Remark 3 (Interpretation of the reaction penalty). The parameter $\kappa > 0 .$ fixes the relative cost of reaction versus transport and is chosen before measurement-design optimization. Since δ has units ofstate-distance per unit time and α has units ofinverse time, κ has units ofsquared state distance, or is dimensionless after the corresponding nondimensionalization. Larger κ makes reaction more expensive, while smaller κ allows the forcing to be realized more cheaply through reaction. On the mean-zero shape subspace, $( \mathsf { L } _ { q } + \bar { \kappa } ^ { - 1 } I ) ^ { - 1 } \to \mathsf { L } _ { q } ^ { - 1 }$ as κ → ∞ whenever the balanced inverse is well defined. Thus, $i f b _ { t } = \dot { M } _ { t } / M _ { t } - g _ { \mathrm { r e f } } ( t ) = 0 ,$ , the unbalanced shape cost approaches the balanced transport cost; $i f b _ { t } \ne 0$ , the mass term $M _ { t } \kappa b _ { t } ^ { 2 }$ diverges, so no finite balanced limit exists. We therefore treat κ as a fixed modeling choice rather than tuning it separately across competing designs.

Remark 4 (Screening removes the balanced compatibility and gauge conditions). The reaction term removes the constant null direction present in the balanced Dirichlet seminorm. In the balanced problem one therefore imposes a mean-zero gauge and requires $\mathbb { E } _ { q } [ h ] = 0 .$ . For finite $\kappa ,$ the term $\kappa ^ { - 1 } \mathbb { E } _ { q } [ \psi ^ { 2 } ]$ controls constants directly, so neither a gauge condition nor zero-mean forcing is required. Testing equation 85 with $\varphi = 1$ gives $\mathbb { E } _ { q } [ \psi ] / \bar { \kappa } = \mathbb { E } _ { q } [ h _ { \mathrm { u b } } ]$ . By Corollary 5, this mean component corresponds exactly to the spatially uniform reaction $b _ { t } = \dot { M } _ { t } / M _ { t } - g _ { \mathrm { r e f } } ( t )$ , while the remaining screened potential has zero $q _ { t }$ -mean.

Remark 5 (Scope of regularized density–source evaluations). At afixed time, the weighted-Poisson construction applies to a probability density q and a density-level source $s = q h$ satisfying $\mathbb { E } _ { q } [ h ] =$ 0, whether or not this pair has been derived from a prescribed law path. Under the assumptions of Appendix B.7 for $h = s / q$ , define

$$
\mathfrak { a } ( q , s ) : = \operatorname* { i n f } _ { \delta \in L ^ { 2 } ( q ; \mathbb { R } ^ { d } ) } \left\{ \mathbb { E } _ { q } [ \Vert \delta \Vert ^ { 2 } ] : \nabla \cdot ( q \delta ) = - s \right\} = \Vert s / q \Vert _ { H ^ { - 1 } ( q ) } ^ { 2 } .\tag{108}
$$

This recovers the Full action of a specified path $q _ { t }$ relative to $u _ { t }$ when $s _ { t } = \partial _ { t } q _ { t } + \nabla \cdot \left( q _ { t } u _ { t } \right)$ . The Poisson variational identity alone does not establish this source identity.

For example, let S be afixed convolution with a probability kernel on $\mathbb { R } ^ { d }$ or aflat torus, and assume the derivatives below commute with convolution. $I f s _ { t } = \partial _ { t } q _ { t } + \nabla \cdot \left( q _ { t } u _ { t } \right)$ , then

$$
\partial _ { t } ( S q _ { t } ) + \nabla \cdot ( ( S q _ { t } ) u _ { t } ) = S s _ { t } + \nabla \cdot \left( ( S q _ { t } ) u _ { t } - S ( q _ { t } u _ { t } ) \right) .\tag{109}
$$

Thus separately smoothing the density and source generally produces the represented cost ${ \mathfrak { a } } ( S q _ { t } , S s _ { t } )$ , not automatically the Full action of $S q _ { t }$ under the unchanged velocity $u _ { t } .$ . Moreover, with S<sup>∗</sup> denoting the adjoint and the integrals understood coordinatewise,

$$
\int \Phi _ { \eta } ( x ) ( S q _ { t } ) ( x ) d x = \int ( S ^ { * } \Phi _ { \eta } ) ( x ) q _ { t } ( x ) d x ,
$$

so smoothing need not preserve calibration against the original observables. The continuum theorems therefore do not by themselves establish consistency or convergence of an empirical or spatially regularized evaluator. Discrete minimum-norm and orthogonality identities concern the specified discrete source, operators, and inner product; identifying their value with the population Full action requires a separate consistency argument.

## B.8 WEIGHTED NEGATIVE-SOBOLEV INTERPRETATION

Define the weighted negative-Sobolev norm

$$
\Vert h \Vert _ { H ^ { - 1 } ( q ) } : = \operatorname* { s u p } _ { \varphi \in H _ { q } : \mathbb { E } _ { q } [ \Vert \nabla \varphi \Vert ^ { 2 } ] \leq 1 } | \mathbb { E } _ { q } [ h \varphi ] | .\tag{110}
$$

Lemma 1 (Weighted negative-Sobolev identity). Under the coercivity and regularity assumptions of Appendix B.7,

$$
a _ { \mathrm { f u l l } } ( t , \pmb { \eta } ) = \left. h _ { t , \pmb { \eta } } \right. _ { H ^ { - 1 } ( q _ { t } ^ { \pmb { \eta } } ) } ^ { 2 } .\tag{111}
$$

Proof of Lemma 1. On $H _ { q }$ use the inner product

$$
\langle \psi , \varphi \rangle _ { H _ { q } } : = \mathbb { E } _ { q } [ \nabla \psi \cdot \nabla \varphi ] .\tag{112}
$$

Theorem 2 gives

$$
\mathbb { E } _ { q } [ h \varphi ] = - \langle \psi , \varphi \rangle _ { H _ { q } }\tag{113}
$$

for every $\varphi \in H _ { q }$ . Hence, by Cauchy–Schwarz,

$$
| \mathbb { E } _ { q } [ h \varphi ] | \leq \left\| \psi \right\| _ { H _ { q } } \left\| \varphi \right\| _ { H _ { q } } ,\tag{114}
$$

so $\| h \| _ { H ^ { - 1 } ( q ) } \leq \| \psi \| _ { H _ { q } } . \mathrm { I f } \psi \neq 0$ , choosing $\varphi = - \psi / \left\| \psi \right\| _ { H _ { q } }$ attains equality; if $\psi = 0$ , both sides vanish. Therefore

$$
\begin{array} { r } { \| h \| _ { H ^ { - 1 } ( q ) } = \| \psi \| _ { H _ { q } } = \left( \mathbb { E } _ { q } [ \| \nabla \psi \| ^ { 2 } ] \right) ^ { 1 / 2 } . } \end{array}\tag{115}
$$

Squaring and using the minimum-energy identity proved in Appendix B.7 gives equation 111, which is the second equality in equation 13 and completes the proof of Theorem 2. □

Corollary 6 (Sharp uniform certificate). Under the assumptions ofTheorem 2,

$$
\begin{array} { r } { a _ { \mathrm { f u l l } } ( t , \eta ) = \operatorname* { i n f } \left\{ B \geq 0 : | { \mathcal E } _ { t } ( f ) | ^ { 2 } \leq B \mathbb { E } _ { Q _ { t } } [ \| \nabla f \| ^ { 2 } ] f o r e \nu e r y f \in H _ { q _ { t } } \right\} . } \end{array}\tag{116}
$$

Hence Full is the smallest uniform constant controlling squared reference-relative expectation-rate discrepancy over the admissible Sobolev observable class.

Proof. By Theorem 2 and Lemma $1 , a _ { \mathrm { f u l l } } ( t , \eta )$ is the squared operator norm of the linear functional $f \mapsto { \mathcal { E } } _ { t } ( f )$ on $H _ { q _ { t } }$ equipped with the gradient-energy norm. The displayed infimum is precisely that squared norm. □

Proposition 6 (Galerkin lower bound, monotonicity, and convergence). Fix $( t , \eta )$ under the assumptions ofTheorem 2, write $q = q _ { t } ^ { \eta }$ and $h = h _ { t , \eta } ,$ and let $V \subset H _ { q }$ be afinite-dimensional linear subspace. Define

$$
a _ { V } : = \operatorname* { s u p } _ { f \in V } \left\{ 2 \mathbb { E } _ { q } [ h f ] - \mathbb { E } _ { q } [ \left. \nabla f \right. ^ { 2 } ] \right\} .\tag{117}
$$

$I f \psi = \psi _ { t , \eta }$ is the Full Poisson potential, then

$$
a _ { V } = a _ { \mathrm { f u l l } } - \mathrm { d i s t } _ { H _ { q } } ^ { 2 } ( - \psi , V ) , \qquad 0 \leq a _ { V } \leq a _ { \mathrm { f u l l } } .\tag{118}
$$

Hence $V _ { 1 } \subseteq V _ { 2 }$ implies $a _ { V _ { 1 } } \leq a _ { V _ { 2 } } \leq a _ { \mathrm { f u l l } } . \ I f V _ { 1 } \subseteq V _ { 2 } \subseteq \cdot \cdot \cdot a n d \overline { { \bigcup _ { n } V _ { n } } } ^ { H _ { q } } = H _ { q } ,$ then $a _ { V _ { n } } \uparrow a _ { \mathrm { f u l l } } .$

Proof. The weak Poisson identity gives $\mathbb { E } _ { q } [ h f ] = - \langle \psi , f \rangle _ { H _ { q } }$ , and therefore

$$
2 \mathbb { E } _ { q } [ h f ] - \| f \| _ { H _ { q } } ^ { 2 } = \| \psi \| _ { H _ { q } } ^ { 2 } - \| f + \psi \| _ { H _ { q } } ^ { 2 } .
$$

Since $\| \psi \| _ { H _ { a } } ^ { 2 } = a _ { \mathrm { f u l l } }$ , taking the supremum over $f \in V$ gives equation 118. Enlarging V can only decrease the distance $\mathbf { t o } \mathrm { ~ - ~ } \boldsymbol { \psi }$ , which proves monotonicity; density of the nested union makes this distance converge to zero. □

Remark 6 (Tangent as a Galerkin restriction). Fix $( t , \eta )$ , write $r = r _ { t , \eta } , G = G _ { t , \eta } ,$ and $\overline { { { \phi } } } =$ $\Phi _ { \eta } - c _ { \eta } ( t )$ , and assume that each coordinate of ϕ belongs to $H _ { q }$ . Testing the reference-relative continuity identity gives $\mathbb { E } _ { q } [ h \overline { { \phi } } ] = - r ,$ while $\| \beta ^ { \top } \overline { { \phi } } \| _ { H _ { a } } ^ { 2 } = \beta ^ { \top } G \beta$ . Taking $V = \operatorname { s p a n } \{ { \overline { { \phi } } } _ { 1 } , \dots , { \overline { { \phi } } } _ { R } \}$ in Proposition 6 therefore gives $a _ { V } = r ^ { \top } G ^ { \dag } r = a _ { \mathrm { t a n } }$ <sup>q</sup> and

$$
a _ { \mathrm { f u l l } } - a _ { \mathrm { t a n } } = \mathrm { d i s t } _ { H _ { q } } ^ { 2 } ( - \psi , V ) = a _ { \mathrm { h i d } } .
$$

Thus Tangent is the measured-observable Galerkin restriction of Full, while Proposition 6 applies to arbitrary finite potential dictionaries such as those used in the skyrmion evaluator. This is the samefinite-feature energy-gap viewpoint underlying Sobolev discrepancy (Mroueh, 2018).

Corollary 7 (Screened negative-Sobolev interpretation). Define

$$
\| h \| _ { H _ { \kappa } ^ { - 1 } ( q ) } : = \operatorname* { s u p } _ { \varphi \in H _ { q , \kappa } : \mathbb { E } _ { q } [ \| \nabla \varphi \| ^ { 2 } + \kappa ^ { - 1 } \varphi ^ { 2 } ] \leq 1 } | \mathbb { E } _ { q } [ h \varphi ] | .\tag{119}
$$

Under the conditions of Corollary $^ { 4 , }$

$$
a _ { \mathrm { f u l l } } ^ { \mathrm { u b } } ( t , \eta ) = M _ { t } \left. h _ { \mathrm { u b } , t , \eta } \right. _ { H _ { \kappa } ^ { - 1 } ( q _ { t } ^ { \eta } ) } ^ { 2 } .\tag{120}
$$

Proof. Equation equation 85 states that the functional $\varphi \mapsto \mathbb { E } _ { q } [ h _ { \mathrm { u b } } \varphi ]$ is represented in the Hilbert space $H _ { q , \kappa }$ by $\psi ^ { \star }$ . Cauchy–Schwarz gives an upper bound by $\| \psi ^ { \star } \| _ { H _ { q , \kappa } }$ , and choosing $\varphi = \psi ^ { \star } / \left\| \psi ^ { \star } \right\| _ { H _ { q , \kappa } }$ when $\psi ^ { \star } \neq 0$ attains equality. Corollary 4 then gives equation 120. □

## B.9 MINIMUM-ENERGY TANGENT CORRECTION

For the minimum-energy Tangent problem, exact feasibility is equivalent to

$$
r _ { t , \eta } \in \operatorname { R a n g e } ( G _ { t , \eta } ) .\tag{121}
$$

Proposition 7 (General minimum-energy tangent correction). Assume $J _ { \Phi _ { \eta } } \in L ^ { 2 } ( Q _ { t } ^ { \eta } ; \mathbb { R } ^ { R \times d } )$ . Exact moment-rate matching is feasible if and only if equation 121 holds. In that case the unique minimum-energy correction is

$$
\delta _ { t , \eta } ^ { \mathrm { t a n } } ( x ) = - J _ { \Phi _ { \eta } } ( x ) ^ { \top } G _ { t , \eta } ^ { \dag } r _ { t , \eta } ,\tag{122}
$$

and

$$
a _ { \tan } ( t , \pmb { \eta } ) = r _ { t , \pmb { \eta } } ^ { \top } G _ { t , \pmb { \eta } } ^ { \dagger } r _ { t , \pmb { \eta } } .\tag{123}
$$

If equation 121 fails, the exact moment-rate constraint is infeasible; the pseudoinverse then has only a least-squares interpretation.

Proof of Proposition 7. Fix $( t , \eta )$ and let

$$
\mathcal { H } : = L ^ { 2 } ( Q _ { t } ^ { \eta } ; \mathbb { R } ^ { d } )\tag{124}
$$

with its usual inner product. Define the bounded linear operator

$$
L : \mathcal { H } \to \mathbb { R } ^ { R } , \qquad L \delta : = \mathbb { E } _ { Q _ { t } ^ { \eta } } [ J _ { \Phi _ { \eta } } \delta ] .\tag{125}
$$

Boundedness follows from Cauchy–Schwarz and the assumption $J _ { \Phi _ { \eta } } \in L ^ { 2 } ( Q _ { t } ^ { \eta } )$ ). Its adjoint is

$$
L ^ { * } a = J _ { \Phi _ { \eta } } ^ { \top } a , \qquad a \in \mathbb { R } ^ { R } ,\tag{126}
$$

because

$$
\langle L \delta , a \rangle _ { \mathbb { R } ^ { R } } = \mathbb { E } _ { Q _ { t } ^ { \eta } } \Big [ \delta ^ { \top } J _ { \Phi _ { \eta } } ^ { \top } a \Big ] .\tag{127}
$$

Moreover,

$$
L L ^ { * } a = G _ { t , \eta } a .\tag{128}
$$

Since the codomain is finite dimensional,

$$
\operatorname { R a n g e } ( L ) = \operatorname { R a n g e } ( L L ^ { * } ) = \operatorname { R a n g e } ( G _ { t , \eta } ) .\tag{129}
$$

Thus the condition $r _ { t , \eta } \in \mathrm { R a n g e } ( G _ { t , \eta } )$ is exactly the feasibility condition for $L \delta = - r _ { t , \eta }$

The feasible set is a closed affine subspace of the Hilbert space $\mathcal { H } ,$ so it contains a unique element of minimum norm. This minimum-norm element is orthogonal to ker L and therefore belongs to

$$
( \ker L ) ^ { \perp } = { \overline { { \operatorname { R a n g e } ( L ^ { * } ) } } } .\tag{130}
$$

Because $L ^ { * }$ has finite-dimensional domain, its range is closed, so the minimizer has the form $\delta =$ $L ^ { * } a$ . The constraint becomes

$$
G _ { t , \eta } a = - r _ { t , \eta } .\tag{131}
$$

The minimum-norm coefficient may be chosen as

$$
a = - G _ { t , \eta } ^ { \dag } r _ { t , \eta } ,\tag{132}
$$

which gives equation 122. Its squared norm is

$$
\begin{array} { r l } & { \mathbb { E } _ { Q _ { t } ^ { \eta } } [ \left| \left| \delta _ { t , \eta } ^ { \mathrm { t a n } } \right| \right| ^ { 2 } ] = ( G _ { t , \eta } ^ { \dag } r _ { t , \eta } ) ^ { \top } G _ { t , \eta } ( G _ { t , \eta } ^ { \dag } r _ { t , \eta } ) } \\ & { \quad \quad \quad = r _ { t , \eta } ^ { \top } G _ { t , \eta } ^ { \dag } r _ { t , \eta } , } \end{array}\tag{133}
$$

where the last equality uses symmetry of $G _ { t , \eta }$ and $r _ { t , \eta } \in \mathrm { R a n g e } ( G _ { t , \eta } )$ . This proves equation 123. $\operatorname { I f } r _ { t , \eta } \notin \operatorname { R a n g e } ( G _ { t , \eta } ) = \operatorname { R a n g e } ( L )$ , no exact solution exists, establishing the remaining feasibility statement. 口

Corollary 8 (Minimum-energy unbalanced tangent correction). Let $\textstyle { \bar { c } } _ { t } = \int \phi d \mu _ { t }$ denote the raw finite-measure moment vector, and assume $\phi \in L ^ { 2 } ( q _ { t } ; \mathbb { R } ^ { R } )$ and $J _ { \phi } \in L ^ { 2 } ( q _ { t } ; \mathbb { R } ^ { R \times d } )$ . Define

$$
r _ { t , \eta } ^ { \mathrm { u b } } : = \mathbb { E } _ { q _ { t } } [ J _ { \phi } u _ { t } + g _ { \mathrm { r e f } } \phi ] - \frac { \dot { \bar { c } } _ { t } } { M _ { t } }\tag{134}
$$

and

$$
G _ { t , \eta } ^ { \mathrm { u b } } : = \mathbb { E } _ { q _ { t } } \bigg [ J _ { \phi } J _ { \phi } ^ { \top } + \frac { 1 } { \kappa } \phi \phi ^ { \top } \bigg ] .\tag{135}
$$

$I f r _ { t , \eta } ^ { \mathrm { u b } } \in \mathrm { R a n g e } ( G _ { t , \eta } ^ { \mathrm { u b } } )$ , then the minimum of

$$
M _ { t } \mathbb { E } _ { q _ { t } } [ \| \delta \| ^ { 2 } + \kappa \alpha ^ { 2 } ]\tag{136}
$$

subject to exact raw-moment-rate matching

$$
\mathbb { E } _ { q _ { t } } [ J _ { \phi } \delta + \phi \alpha ] = - r _ { t , \eta } ^ { \mathrm { u b } }\tag{137}
$$

is attained uniquely by

$$
\delta _ { \mathrm { u b } } ^ { \mathrm { t a n } } = - J _ { \phi } ^ { \top } ( G ^ { \mathrm { u b } } ) ^ { \dagger } \boldsymbol { r } ^ { \mathrm { u b } } , \qquad \alpha _ { \mathrm { u b } } ^ { \mathrm { t a n } } = - \frac { 1 } { \kappa } \phi ^ { \top } ( G ^ { \mathrm { u b } } ) ^ { \dagger } \boldsymbol { r } ^ { \mathrm { u b } } ,\tag{138}
$$

and

$$
a _ { \mathrm { t a n } } ^ { \mathrm { u b } } ( t , \eta ) = M _ { t } ( r _ { t , \eta } ^ { \mathrm { u b } } ) ^ { \top } ( G _ { t , \eta } ^ { \mathrm { u b } } ) ^ { \dagger } r _ { t , \eta } ^ { \mathrm { u b } } .\tag{139}
$$

$I f r ^ { \mathrm { u b } } \not \in \mathrm { R a n g e } ( G ^ { \mathrm { u b } } )$ , exact raw-moment-rate matching is infeasible.

Proof. For any sufficiently regular finite-measure path satisfying

$$
\begin{array} { r } { \partial _ { t } \mu + \nabla \cdot ( \mu v ) = \mu g , } \end{array}\tag{140}
$$

integration by parts gives

$$
\frac { d } { d t } \int \phi d \mu = \int ( J _ { \phi } v + \phi g ) d \mu .\tag{141}
$$

Applying this with $v = u + \delta , g = g _ { \mathrm { r e f } } + \alpha$ , and $\mu = M q$ yields equation 137. Now equip

$$
\mathcal { Z } : = L ^ { 2 } ( q ; \mathbb { R } ^ { d } ) \times L ^ { 2 } ( q )\tag{142}
$$

with

$$
\begin{array} { r } { \langle ( \delta , \alpha ) , ( \delta ^ { \prime } , \alpha ^ { \prime } ) \rangle _ { \mathcal { Z } } : = \mathbb { E } _ { q } [ \delta \cdot \delta ^ { \prime } + \kappa \alpha \alpha ^ { \prime } ] . } \end{array}\tag{143}
$$

Define $L _ { \mathrm { u b } } : \mathcal { Z } \to \mathbb { R } ^ { R }$ by $L _ { \mathrm { u b } } ( \delta , \alpha ) = \mathbb { E } _ { q } [ J _ { \phi } \delta + \phi \alpha ]$ . Its adjoint is

$$
{ \cal L } _ { \mathrm { u b } } ^ { * } a = \left( J _ { \phi } ^ { \top } a , \frac { 1 } { \kappa } \phi ^ { \top } a \right) ,\tag{144}
$$

since

$$
\langle ( \delta , \alpha ) , L _ { \mathrm { u b } } ^ { * } a \rangle _ { \mathcal { Z } } = a ^ { \top } \mathbb { E } _ { q } [ J _ { \phi } \delta + \phi \alpha ] .\tag{145}
$$

Consequently,

$$
L _ { \mathrm { u b } } L _ { \mathrm { u b } } ^ { * } a = G ^ { \mathrm { u b } } a .\tag{146}
$$

The Hilbert-space minimum-norm argument from Proposition 7 therefore applies verbatim, giving $( \delta _ { \mathrm { u b } } ^ { \mathrm { t a n } } , \alpha _ { \mathrm { u b } } ^ { \mathrm { t a n } } ) \stackrel { \bullet } { = } - L _ { \mathrm { u b } } ^ { * } ( G ^ { \mathrm { u b } } ) ^ { \dagger } r ^ { \mathrm { u b } }$ and squared product norm $( r ^ { \mathrm { u b } } ) ^ { \top } ( G ^ { \mathrm { u b } } ) ^ { \dagger } r ^ { \mathrm { u b } }$ . Multiplication by $M _ { t }$ gives equation 139. □

Remark 7 (Making total-mass dynamics tangent-visible). If the raw tangent observable vector is augmented by the constant coordinate $\phi _ { 0 } \equiv 1$ , then its transport Jacobian vanishes and its tangent equation reduces to the mass-rate condition. The corresponding Gram entry is $\kappa ^ { - 1 }$ , so total-mass mismatch is represented entirely through reaction. This augmentation is appropriate for the tangent criterion even though, by Remark 1, the constant coordinate should not be inserted into the normalized-shape information-projection covariance.

## B.10 FULL–TANGENT–HIDDEN DECOMPOSITION

Fix $( t , \eta )$ and assume the hypotheses of Theorem 2. Assume additionally that $J _ { \Phi _ { \eta } } \in L ^ { 2 } ( Q _ { t } ^ { \eta } )$ $J _ { \Phi _ { \eta } } u _ { t } \in L ^ { 1 } ( Q _ { t } ^ { \eta } )$ , and $\Phi _ { \eta , j } - c _ { \eta , j } ( t ) \in H _ { q _ { t } ^ { \eta } }$ for every observable coordinate. We further assume that differentiation of the calibrated moments and weak testing with these coordinates are justified and that the relevant boundary terms vanish. These conditions ensure that the Full correction satisfies the Tangent moment-rate constraints and that the Hilbert-space projections below are well defined.

The pointwise identities underlying the balanced decomposition are

$$
\mathbb { E } _ { Q _ { t } ^ { \eta } } \big [ J _ { \Phi _ { \eta } } \delta _ { t , \eta } ^ { \mathrm { h i d } } \big ] = 0 ,\tag{147}
$$

and

$$
\mathbb { E } _ { Q _ { t } ^ { \eta } } \big [ \delta _ { t , \eta } ^ { \mathrm { t a n } } \cdot \delta _ { t , \eta } ^ { \mathrm { h i d } } \big ] = 0 .\tag{148}
$$

The hidden action density is

$$
a _ { \mathrm { h i d } } ( t , \pmb { \eta } ) : = \mathbb { E } _ { Q _ { t } ^ { \eta } } \Big [ \big \| \delta _ { t , \pmb { \eta } } ^ { \mathrm { h i d } } \big \| ^ { 2 } \Big ] \geq 0 .\tag{149}
$$

For measurable action densities, define $\begin{array} { r } { \mathcal { A } _ { \mathrm { h i d } } ( \pmb { \eta } ) : = \int _ { 0 } ^ { 1 } a _ { \mathrm { h i d } } ( t , \pmb { \eta } ) \ d \rho ( d t ) } \end{array}$ . The design-level counterpart of equation 15 is

$$
\begin{array} { r } { A ( \eta ) = A _ { \mathrm { t a n } } ( \eta ) + A _ { \mathrm { h i d } } ( \eta ) , \qquad A _ { \mathrm { t a n } } ( \eta ) \leq A ( \eta ) . } \end{array}\tag{150}
$$

ProofofProposition 2. Fix $( t , \eta )$ and write $Q = Q _ { t } ^ { \eta }$ . Let

$$
L \delta : = \mathbb { E } _ { Q } [ J _ { \Phi _ { \eta } } \delta ]\tag{151}
$$

be the moment-rate operator introduced implicitly in the proof of Proposition 7. Let $\delta _ { t } ^ { \star }$ denote the minimum-energy Full correction. Because $u _ { t } + \delta _ { t } ^ { \star }$ transports the projected law and the calibration identity gives $\check { \mathbb { E } _ { Q } } [ \Phi _ { \pmb { \eta } } ] = \pmb { c } _ { \pmb { \eta } } ( t )$ , the moment-rate identity equation 20 yields

$$
L \delta _ { t } ^ { \star } = - \left( \mathbb { E } _ { Q } [ J _ { \Phi _ { \eta } } u _ { t } ] - \dot { \pmb { c } } _ { \eta } ( t ) \right) = - r _ { t , \eta } .\tag{152}
$$

Hence the tangent constraint is feasible. By Proposition 7, its unique minimum-norm solution is $\delta _ { t , \eta } ^ { \mathrm { t a n } }$

Define

$$
\delta _ { t , \eta } ^ { \mathrm { h i d } } : = \delta _ { t } ^ { \star } - \delta _ { t , \eta } ^ { \mathrm { t a n } } .\tag{153}
$$

Both ${ \delta } _ { t } ^ { \star }$ and $\delta _ { t , \eta } ^ { \mathrm { t a n } }$ satisfy $L \delta = - r _ { t , \eta } .$ , so

$$
L \delta _ { t , \eta } ^ { \mathrm { h i d } } = 0 .\tag{154}
$$

Thus $\delta _ { t , \eta } ^ { \mathrm { h i d } } \in$ ker L, which proves equation 147.

The proof of Proposition 7 also shows that the minimum-norm tangent solution belongs to (ker $L ) ^ { \perp }$ Therefore

$$
\mathbb { E } _ { Q } \big [ \delta _ { t , \eta } ^ { \mathrm { t a n } } \cdot \delta _ { t , \eta } ^ { \mathrm { h i d } } \big ] = 0 ,\tag{155}
$$

which is equation 148. Applying the Pythagorean identity in $L ^ { 2 } ( Q ; \mathbb { R } ^ { d } )$ gives

$$
\begin{array} { r l } & { a _ { \mathrm { f u l l } } ( t , \eta ) = \mathbb { E } _ { Q } [ \left. \delta _ { t } ^ { \star } \right. ^ { 2 } ] } \\ & { \qquad = \mathbb { E } _ { Q } \left[ \left. \delta _ { t , \eta } ^ { \mathrm { t a n } } + \delta _ { t , \eta } ^ { \mathrm { h i d } } \right. ^ { 2 } \right] } \\ & { \qquad = \mathbb { E } _ { Q } [ \left. \delta _ { t , \eta } ^ { \mathrm { t a n } } \right. ^ { 2 } ] + \mathbb { E } _ { Q } [ \left. \delta _ { t , \eta } ^ { \mathrm { h i d } } \right. ^ { 2 } ] } \\ & { \qquad = a _ { \mathrm { t a n } } ( t , \eta ) + a _ { \mathrm { h i d } } ( t , \eta ) , } \end{array}\tag{156}
$$

proving equation 15. Integrating the pointwise identity against the common nonnegative measure $\rho$ gives equation 150 whenever the terms are measurable; the stated pointwise and integrated inequalities follow from nonnegativity of the hidden action. At fixed t, equality $a _ { \mathrm { t a n } } ( t , \pmb { \eta } ) = a _ { \mathrm { f u l l } } ( t , \pmb { \eta } )$ holds if and only if $\delta _ { t , \eta } ^ { \mathrm { h i d } } = 0$ in $L ^ { 2 } ( Q ; \mathbb { R } ^ { d } )$ , equivalently $\delta _ { t } ^ { \star } = \delta _ { t , \eta } ^ { \mathrm { t a n } }$ Q-almost everywhere. For the integrated actions, if $\dot { A } ( \eta ) < \infty$ , then $\mathcal { A } _ { \mathrm { t a n } } ( \pmb { \eta } ) = \mathcal { A } ( \pmb { \eta } )$ holds if and only if $a _ { \mathrm { h i d } } ( t , \pmb { \eta } ) = 0$ for ρ-almost every t, equivalently $\delta _ { t , \eta } ^ { \mathrm { h i d } } = 0$ in $L ^ { 2 } ( \dot { Q } _ { t } ^ { \eta } ; \mathbb R ^ { d } )$ for ρ-almost every t. The decomposition remains valid for measurable nonnegative extended-valued actions, but equality of two infinite actions does not imply that the hidden action vanishes. □

Corollary 9 (Unbalanced Full–Tangent–Hidden decomposition). Under the assumptions ofCorollaries 4 and $\delta ,$ assume additionally that each coordinate of $\phi ~ = ~ \Phi _ { \eta }$ belongs to $H _ { q _ { t } , \kappa } ,$ , that $J _ { \phi } u _ { t } \in L ^ { 1 } ( q _ { t } ; \mathbb { R } ^ { R } )$ , and that raw-moment differentiation and weak testing with these coordinates are justified, with vanishing boundary terms. Let

$$
\begin{array} { r } { z _ { \mathrm { u b } } ^ { \star } : = ( \delta ^ { \star } , \alpha ^ { \star } ) , \qquad z _ { \mathrm { u b } } ^ { \mathrm { t a n } } : = ( \delta _ { \mathrm { u b } } ^ { \mathrm { t a n } } , \alpha _ { \mathrm { u b } } ^ { \mathrm { t a n } } ) , \qquad z _ { \mathrm { u b } } ^ { \mathrm { h i d } } : = z _ { \mathrm { u b } } ^ { \star } - z _ { \mathrm { u b } } ^ { \mathrm { t a n } } . } \end{array}\tag{157}
$$

Then

$$
L _ { \mathrm { u b } } z _ { \mathrm { u b } } ^ { \mathrm { h i d } } = 0\tag{158}
$$

and

$$
\mathbb { E } _ { q } \big [ \delta _ { \mathrm { u b } } ^ { \mathrm { t a n } } \cdot \delta _ { \mathrm { u b } } ^ { \mathrm { h i d } } + \kappa \alpha _ { \mathrm { u b } } ^ { \mathrm { t a n } } \alpha _ { \mathrm { u b } } ^ { \mathrm { h i d } } \big ] = 0 .
$$

Defining

(159)

$$
a _ { \mathrm { h i d } } ^ { \mathrm { u b } } ( t , \eta ) : = M _ { t } \mathbb { E } _ { q } \big [ \| \delta _ { \mathrm { u b } } ^ { \mathrm { h i d } } \| ^ { 2 } + \kappa ( \alpha _ { \mathrm { u b } } ^ { \mathrm { h i d } } ) ^ { 2 } \big ] ,\tag{160}
$$

one has

$$
a _ { \mathrm { f u l l } } ^ { \mathrm { u b } } ( t , \eta ) = a _ { \mathrm { t a n } } ^ { \mathrm { u b } } ( t , \eta ) + a _ { \mathrm { h i d } } ^ { \mathrm { u b } } ( t , \eta ) , \qquad a _ { \mathrm { t a n } } ^ { \mathrm { u b } } ( t , \eta ) \leq a _ { \mathrm { f u l l } } ^ { \mathrm { u b } } ( t , \eta ) .\tag{161}
$$

Whenever the three terms are integrable against the common nonnegative measure $\rho ,$ the same decomposition and inequality hold after integration in time.

Proof. The Full correction satisfies the complete finite-measure continuity equation, so the raw moment identity equation 141 implies that $z _ { \mathrm { u b } } ^ { \star }$ satisfies the tangent constraint. The tangent correction is the unique minimum-norm point of that affine constraint in the product Hilbert space equation 143; therefore $z _ { \mathrm { u b } } ^ { \mathrm { t a n } } \in ( \mathrm { k e r } L _ { \mathrm { u b } } ) ^ { \perp }$ . Since both $z _ { \mathrm { u b } } ^ { \star }$ and $z _ { \mathrm { u b } } ^ { \mathrm { t a n } }$ have the same image under $L _ { \mathrm { u b } }$ , their difference lies in ker $L _ { \mathrm { u b } }$ . This proves equation 158 and equation 159. Pythagoras in the product Hilbert space, followed by multiplication by $M _ { t } .$ , gives equation 161; integration against $\rho$ preserves the identity and inequality. □

When $0 < \mathcal { A } ( \eta ) < \infty$ , we summarize the relative size of the tangent-invisible component by the hidden-action fraction

$$
\Gamma ( \pmb \eta ) : = \frac { \mathcal { A } _ { \mathrm { h i d } } ( \pmb \eta ) } { \mathcal { A } ( \pmb \eta ) } = 1 - \frac { \mathcal { A } _ { \mathrm { t a n } } ( \pmb \eta ) } { \mathcal { A } ( \pmb \eta ) } \in [ 0 , 1 ] .\tag{162}
$$

Thus $\Gamma ( \eta )$ near one indicates that most of the minimum law-level correction lies outside the directions visible through the measured moment rates. In the discretized experimental audits we denote the corresponding common-raster quantity by $\Gamma _ { h }$

## B.11 EXACT CONDITION FOR TANGENT–FULL EQUALITY

Corollary 10 (When Tangent and Full corrections coincide). Under the conditions ofProposition 2, suppose additionally that $\mathrm { \bar { ~ } } Q _ { t } ^ { \eta }$ has a density positive almost everywhere on a connected open domain Ω, and that $\psi _ { t , \eta }$ and the coordinates of $\Phi _ { \eta }$ belong to $H _ { \mathrm { l o c } } ^ { 1 } ( \Omega )$ . Define

$$
\beta _ { t , \eta } : = G _ { t , \eta } ^ { \dag } r _ { t , \eta } .\tag{163}
$$

Then

$$
a _ { \mathrm { t a n } } ( t , \pmb { \eta } ) = a _ { \mathrm { f u l l } } ( t , \pmb { \eta } )\tag{164}
$$

if and only if

$$
\psi _ { t , \eta } ( x ) = \beta _ { t , \eta } ^ { \top } ( \Phi _ { \eta } ( x ) - \pmb { c _ { \eta } ( t ) } ) \qquad Q _ { t } ^ { \eta } - a . e .\tag{165}
$$

Proof of Corollary 10. Let

$$
\beta _ { t , \eta } = G _ { t , \eta } ^ { \dagger } r _ { t , \eta } .\tag{166}
$$

By Proposition 7,

$$
\delta _ { t , \eta } ^ { \mathrm { t a n } } = - J _ { \Phi _ { \eta } } ^ { \top } \beta _ { t , \eta } = - \nabla \big ( \beta _ { t , \eta } ^ { \top } \Phi _ { \eta } \big )\tag{167}
$$

where the final identity is understood in the weak sense. Theorem 2 gives

$$
\delta _ { t } ^ { \star } = - \nabla \psi _ { t , \eta } .\tag{168}
$$

By Proposition 2, equality of tangent and full action is equivalent to $\delta _ { t } ^ { \star } = \delta _ { t , \eta } ^ { \mathrm { t a n } } Q _ { t } ^ { \eta }$ -almost everywhere. Because the density of $Q _ { t } ^ { \eta }$ is positive almost everywhere on Ω, this is equivalent to

$$
\nabla \left( \psi _ { t , \eta } - \beta _ { t , \eta } ^ { \top } \Phi _ { \eta } \right) = 0\tag{169}
$$

almost everywhere on Ω. A Sobolev function with zero weak gradient on a connected open domain is almost everywhere constant. Hence

$$
\psi _ { t , \eta } ( x ) = \beta _ { t , \eta } ^ { \top } \Phi _ { \eta } ( x ) + C\tag{170}
$$

for some constant C. The mean-zero gauge for the Poisson potential and the calibration identity imply

$$
0 = \mathbb { E } _ { Q _ { t } ^ { \eta } } [ \psi _ { t , \eta } ] = \beta _ { t , \eta } ^ { \top } \pmb { c } _ { \pmb { \eta } } ( t ) + C ,\tag{171}
$$

so $C = - \beta _ { t , \eta } ^ { \top } c _ { \eta } ( t )$ , giving equation 165.

Conversely, if equation 165 holds, then

$$
- \nabla \psi _ { t , \eta } = - J _ { \Phi _ { \eta } } ^ { \top } \beta _ { t , \eta } = \delta _ { t , \eta } ^ { \mathrm { t a n } }\tag{172}
$$

$Q _ { t } ^ { \eta }$ -almost everywhere. Thus the hidden component vanishes, and Proposition 2 gives $a _ { \mathrm { f u l l } } ( t , \eta ) =$ $a _ { \mathrm { t a n } } ( t , \eta )$ □

Corollary 11 (Exact unbalanced tangent–Full equality condition). Under the assumptions of Corollary 9, assume that the screened-Poisson potential ψ and the coordinates of ϕ belong to $H _ { q , \kappa } ,$ and define

$$
\beta ^ { \mathrm { u b } } : = ( G ^ { \mathrm { u b } } ) ^ { \dag } r ^ { \mathrm { u b } } .\tag{173}
$$

Then

$$
a _ { \mathrm { t a n } } ^ { \mathrm { u b } } ( t , \pmb { \eta } ) = a _ { \mathrm { f u l l } } ^ { \mathrm { u b } } ( t , \pmb { \eta } )\tag{174}
$$

if and only if

$$
\psi = - ( \beta ^ { \mathrm { u b } } ) ^ { \top } \phi \quad \quad i n H _ { q , \kappa } ,\tag{175}
$$

equivalently, the screened Full potential belongs to the uncentered linear span of the tangent observables with the coefficient fixed by the tangent normal equations.

Proof. Corollary 9 gives equality if and only if the Full and Tangent correction pairs coincide. By Corollaries 4 and 8,

$$
z _ { \mathrm { u b } } ^ { \star } = \left( \nabla \psi , \frac { \psi } { \kappa } \right) , \qquad z _ { \mathrm { u b } } ^ { \mathrm { t a n } } = \left( - J _ { \phi } ^ { \top } \beta ^ { \mathrm { u b } } , - \frac { 1 } { \kappa } \phi ^ { \top } \beta ^ { \mathrm { u b } } \right) .\tag{176}
$$

Equality of the reaction components gives $\psi = - ( \beta ^ { \mathrm { u b } } ) ^ { \top } \phi$ in $L ^ { 2 } ( q )$ ; because both sides lie in $H _ { q , \kappa } ,$ the corresponding weak gradients then agree as well, giving equality of the transport components. Conversely, equation 175 implies equality of both components and hence zero hidden action. Unlike the balanced case, no additive constant remains undetermined because the reaction component contains the potential itself. □

## B.12 DIFFERENTIABILITY WITH RESPECT TO THE MEASUREMENT DESIGN

For continuous designs, the numerical objective depends on the design through the calibrated information projection, its physical-time multiplier derivative, the resulting continuity forcing, and finally the finite-dimensional Full correction solve. We first record the sensitivity of the calibration multiplier and then use it to establish local differentiability of the complete fixed numerical objective.

Define the calibration residual

$$
\begin{array} { r } { F ( \lambda , \eta , t ) : = \mathbb { E } _ { q _ { \lambda , \eta , t } } [ \Phi _ { \eta } ] - c _ { \eta } ( t ) , } \end{array}\tag{177}
$$

where

$$
\begin{array} { r } { q _ { \lambda , \eta , t } ( x ) \propto \widetilde { q } _ { t } ( x ) \exp \bigl ( \lambda ^ { \top } \Phi _ { \eta } ( x ) \bigr ) . } \end{array}\tag{178}
$$

Proposition 8 (Design sensitivity of the information projection). Fix t and express the design in local coordinates on an open set. Suppose F is continuously differentiable in $( \lambda , \eta )$ on an open neighborhood of a calibrated point satisfying $F ( \lambda _ { t , \eta } , \eta , t ) \stackrel { \cdot } { = } 0 .$ Assume that the observable and target maps are continuously differentiable in the design coordinates and that the differentiations of the tilted expectations below can be interchanged with integration throughout this neighborhood. If $C _ { t , \eta } = \partial _ { \lambda } F ( \lambda _ { t , \eta } , \eta , t )$ is positive definite, then the calibrated multiplier defines a locally unique continuously differentiable map ofthe design. For every design perturbation ξ in these coordinates,

$$
D _ { \eta } \lambda _ { t , \eta } [ \xi ] = - C _ { t , \eta } ^ { - 1 } D _ { \eta } F [ \xi ] ,\tag{179}
$$

where

$$
\begin{array} { r l } & { D _ { \eta } F [ \xi ] = \mathbb { E } _ { Q _ { t } ^ { \eta } } [ D _ { \eta } \Phi _ { \eta } [ \xi ] ] + \operatorname { C o v } _ { Q _ { t } ^ { \eta } } \left( \Phi _ { \eta } , \lambda _ { t , \eta } ^ { \top } D _ { \eta } \Phi _ { \eta } [ \xi ] \right) } \\ & { \qquad - D _ { \eta } c _ { \eta } ( t ) [ \xi ] . } \end{array}\tag{180}
$$

ProofofProposition 8. Fix t and abbreviate

$$
q _ { \lambda , \pmb { \eta } } ( x ) = \widetilde { q } _ { t } ( x ) \exp \bigl ( \lambda ^ { \top } \pmb { \Phi } _ { \pmb { \eta } } ( x ) - A _ { t } ( \lambda , \pmb { \eta } ) \bigr ) ,\tag{181}
$$

where

$$
A _ { t } ( \lambda , \pmb { \eta } ) = \log \mathbb { E } _ { \widetilde { Q } _ { t } } \left[ \exp ( \lambda ^ { \top } \pmb { \Phi } _ { \pmb { \eta } } ( X ) ) \right] .\tag{182}
$$

The reference law is frozen with respect to η. At a calibrated solution,

$$
F ( \lambda _ { t , \eta } , \eta , t ) = 0 .\tag{183}
$$

Differentiation with respect to λ gives

$$
\begin{array} { r } { \partial _ { \lambda } F = \mathrm { C o v } _ { q _ { \lambda , \eta } } ( \pmb { \Phi } _ { \eta } , \pmb { \Phi } _ { \eta } ) , } \end{array}\tag{184}
$$

so at the calibrated point

$$
\partial _ { \lambda } F = C _ { t , \eta } .\tag{185}
$$

By assumption this matrix is positive definite and hence invertible. The implicit-function theorem therefore gives a locally differentiable multiplier map $\eta \mapsto \lambda _ { t , \eta }$ and, for any perturbation direction ξ,

$$
D _ { \eta } \lambda _ { t , \eta } [ \xi ] = - C _ { t , \eta } ^ { - 1 } D _ { \eta } F [ \xi ] .\tag{186}
$$

It remains to compute the directional derivative of F while holding λ fixed.

For any scalar or vector test function $f _ { \eta }$ , differentiation of a normalized exponential tilt gives the identity

$$
D _ { \eta } \mathbb { E } _ { q _ { \lambda , \eta } } [ f _ { \eta } ] [ \xi ] = \mathbb { E } _ { q _ { \lambda , \eta } } [ D _ { \eta } f _ { \eta } [ \xi ] ] + \mathrm { C o v } _ { q _ { \lambda , \eta } } \left( f _ { \eta } , \lambda ^ { \top } D _ { \eta } \Phi _ { \eta } [ \xi ] \right) ,\tag{187}
$$

provided differentiation and integration can be interchanged. Applying equation 187 with $f _ { \eta } = \Phi _ { \eta }$ and subtracting $D _ { \eta } \pmb { c } _ { \eta } ( t ) [ \xi ]$ gives

$$
\begin{array} { r l } & { D _ { \eta } F [ \xi ] = \mathbb { E } _ { Q _ { t } ^ { \eta } } [ D _ { \eta } \Phi _ { \eta } [ \xi ] ] + \operatorname { C o v } _ { Q _ { t } ^ { \eta } } \left( \Phi _ { \eta } , \lambda _ { t , \eta } ^ { \top } D _ { \eta } \Phi _ { \eta } [ \xi ] \right) } \\ & { \qquad - D _ { \eta } c _ { \eta } ( t ) [ \xi ] , } \end{array}\tag{188}
$$

which is equation 180. Substitution into equation 186 proves equation 179.

Proposition 9 (End-to-end differentiability of the finite-dimensional Full objective). Let the continuous design η vary in local coordinates on an open set, andfix numerical time nodes $t _ { 1 } , \dots , t _ { M }$ with nonnegative quadrature weights $\omega _ { 1 } , \ldots , \omega _ { M }$ . At each $t _ { j } ,$ , suppose the assumptions ofProposition 8 hold, and write the physical-time multiplier equationfrom equation 8 as

$$
C _ { j , \eta } \dot { \lambda } _ { j , \eta } = g _ { j , \eta } , \qquad g _ { j , \eta } : = \dot { c } _ { \eta } ( t _ { j } ) - \mathbb { E } _ { Q _ { t _ { j } } ^ { \eta } } [ m _ { j , \eta } ] - \mathrm { C o v } _ { Q _ { t _ { j } } ^ { \eta } } \left( \Phi _ { \eta } , \lambda _ { j , \eta } ^ { \top } m _ { j , \eta } \right) .\tag{189}
$$

Assume that $C _ { j , \eta }$ and $g _ { j , \eta }$ are continuously differentiable in η and that $C _ { j , \eta }$ remains nonsingular. Suppose further that, after the fixed gauge choice or basis restriction used by the numerical evaluator, the Full correction at $t _ { j }$ is represented by a reduced linear system

$$
\mathsf { K } _ { j , \pmb { \eta } } \beta _ { j , \pmb { \eta } } = b _ { j , \pmb { \eta } } ,\tag{190}
$$

where $\mathsf { K } _ { j , \eta }$ is symmetric positive definite and both $\mathsf { K } _ { j , \eta }$ and $b _ { j , \eta }$ are continuously differentiable in η. Define the corresponding numerical action by

$$
\widehat { a } _ { \mathrm { f u l l } } ( t _ { j } , \pmb { \eta } ) : = \beta _ { j , \pmb { \eta } } ^ { \top } \mathsf { K } _ { j , \pmb { \eta } } \beta _ { j , \pmb { \eta } } = b _ { j , \pmb { \eta } } ^ { \top } \mathsf { K } _ { j , \pmb { \eta } } ^ { - 1 } b _ { j , \pmb { \eta } } , \qquad \widehat { A } ( \pmb { \eta } ) : = \sum _ { j = 1 } ^ { M } \omega _ { j } \widehat { a } _ { \mathrm { f u l l } } ( t _ { j } , \pmb { \eta } ) .\tag{191}
$$

Then $\lambda _ { j , \eta } , \dot { \lambda } _ { j , \eta } , \beta _ { j , \eta } , \widehat { a } _ { \mathrm { f u l l } } ( t _ { j } , \eta )$ , and $\widehat { A } ( \pmb { \eta } )$ are locally continuously differentiable in η. For every design perturbation $\xi ,$

$$
\begin{array} { r } { D _ { \eta } \dot { \lambda } _ { j , \eta } [ \xi ] = C _ { j , \eta } ^ { - 1 } \Big ( D _ { \eta } g _ { j , \eta } [ \xi ] - D _ { \eta } C _ { j , \eta } [ \xi ] \dot { \lambda } _ { j , \eta } \Big ) , } \end{array}\tag{192}
$$

and the instantaneous action derivative is

$$
D _ { \eta } \widehat { a } _ { \mathrm { f u l l } } ( t _ { j } , \eta ) [ \xi ] = 2 \beta _ { j , \eta } ^ { \top } D _ { \eta } b _ { j , \eta } [ \xi ] - \beta _ { j , \eta } ^ { \top } D _ { \eta } { \sf K } _ { j , \eta } [ \xi ] \beta _ { j , \eta } .\tag{193}
$$

Consequently,

$$
D _ { \eta } \widehat { \mathcal { A } } ( \eta ) [ \xi ] = \sum _ { j = 1 } ^ { M } \omega _ { j } D _ { \eta } \widehat { a } _ { \mathrm { f u l l } } ( t _ { j } , \eta ) [ \xi ] .\tag{194}
$$

Proof. Proposition 8 gives local continuous differentiability of $\lambda _ { j , \eta } .$ . Since $C _ { j , \eta }$ remains nonsingular, equation 189 gives $\dot { \lambda } _ { j , \eta } = C _ { j , \eta } ^ { - 1 } g _ { j , \eta } .$ which is locally continuously differentiable, and differentiation of this identity yields equation 192. The assumed smooth dependence of the remaining numerical quantities therefore makes $\mathsf { K } _ { j , \eta }$ and $b _ { j , \eta }$ continuously differentiable functions of the design. Positive definiteness of $\mathsf { K } _ { j , \eta }$ gives $\beta _ { j , \eta } = \mathsf { K } _ { j , \eta } ^ { - 1 } b _ { j , \eta }$ and hence local continuous differentiability of the reduced correction. Finally,

$$
\widehat { a } _ { \mathrm { f u l l } } = \boldsymbol { b } ^ { \intercal } \mathsf { K } ^ { - 1 } \boldsymbol { b } .
$$

Using $D ( { \mathsf { K } } ^ { - 1 } ) [ \xi ] = - { \mathsf { K } } ^ { - 1 } D { \mathsf { K } } [ \xi ] { \mathsf { K } } ^ { - 1 }$ and β = K<sup>−1</sup>b gives

$$
\begin{array} { r } { D \widehat { a } _ { \mathrm { f u l l } } [ \xi ] = 2 \beta ^ { \top } D b [ \xi ] - \beta ^ { \top } D { \sf K } [ \xi ] \beta , } \end{array}
$$

which proves equation 193. The finite weighted sum in equation 191 then gives equation 194.

Remark 8 (Scope of the differentiability claim). Proposition 9 concerns thefixedfinite-dimensional objective differentiated during continuous-design optimization. It does not assert differentiability of the continuum action under arbitrary perturbations ofthe weighted Sobolev problem, nor ofdiscrete decisions such as calibration-feasibility rejection, support-certification thresholds, changes of numerical basis or gauge, or membership in the risk-constrained admissible set. Within afixed numerical representation and away from such transitions, the proposition gives the end-to-end derivative through calibration, physical-time multiplier dynamics, and the Full correction solve.

Corollary 12 (Design sensitivity in the finite-measure unbalanced extension). Suppose the normalized-shape information projection in Corollary 1 is calibrated to

$$
c _ { \pmb { \eta } } ( t ) = \frac { \bar { c } _ { \pmb { \eta } } ( t ) } { M _ { \pmb { \eta } } ( t ) } ,\tag{195}
$$

and suppose the assumptions ofProposition 8 hold. Then

$$
D _ { \eta } \lambda _ { t , \eta } [ \xi ] = - C _ { t , \eta } ^ { - 1 } D _ { \eta } F [ \xi ]\tag{196}
$$

with the same expression equation 180, but with

$$
D _ { \eta } c _ { \eta } ( t ) [ \xi ] = \frac { 1 } { M _ { \eta } ( t ) } D _ { \eta } \bar { c } _ { \eta } ( t ) [ \xi ] - \frac { c _ { \eta } ( t ) } { M _ { \eta } ( t ) } D _ { \eta } M _ { \eta } ( t ) [ \xi ] .\tag{197}
$$

In particular, if the mass schedule is fixed with respect to the measurement design, only the first term in equation 197 remains.

## C ANALYTICAL COMPARISONS AND EXACT INFORMATION-PROJECTEDEXAMPLES

The criterion distinctions summarized in Section 4 are developed here at the operator level and through exact information-projected examples. We first collect the comparator definitions used in the experiments, then interpret Full through the inverse weighted transport operator, characterize the Tangent-invisible component, compare residual $L ^ { 2 }$ with Full through the transport spectrum, and establish separations from forward KL and static $W _ { 2 }$ . The final subsection proves the matchedinitial stability result used in the main text. The analytical comparisons need not assume equal scientific risk. Unless stated otherwise, $\mathbb { T } = \mathbb { R } / ( 2 \pi \mathbb { Z } )$ , spatial boundary conditions are periodic, Wasserstein distance uses the geodesic ground metric, and integrated comparisons use $\rho ( d t ) = d t$ on [0, 1].

## C.1 COMPARATOR DEFINITIONS

In the experiments, all comparator selectors use the same information-projected path $Q _ { t } ^ { \eta }$ and the same admissible set $\mathcal { D } _ { \varepsilon }$ as Full; only the secondary score changes. With $J _ { \Phi _ { \eta } }$ denoting the observable Jacobian, we define

$$
a _ { \mathrm { t a n } } ( t , \eta ) : = \operatorname* { i n f } _ { \substack { \delta \in L ^ { 2 } ( Q _ { t } ^ { \eta } ; \mathbb { R } ^ { d } ) } } \left\{ \mathbb { E } _ { Q _ { t } ^ { \eta } } [ \| \delta \| ^ { 2 } ] : \mathbb { E } _ { Q _ { t } ^ { \eta } } [ J _ { \Phi _ { \eta } } ( u _ { t } + \delta ) ] = \dot { \epsilon } _ { \eta } ( t ) \right\} ,\tag{198}
$$

$$
a _ { \mathrm { r e s } } ( t , \eta ) : = \mathbb { E } _ { Q _ { t } ^ { \eta } } [ h _ { t , \eta } ^ { 2 } ] ,\tag{199}
$$

$$
d _ { \mathrm { K L } } ( t , \pmb { \eta } ) : = D _ { \mathrm { K L } } ( Q _ { t } ^ { \pmb { \eta } } | | \widetilde { Q } _ { t } ) ,\tag{200}
$$

$$
\mathcal { A } _ { W _ { 2 } } ( \pmb { \eta } ) : = \int _ { 0 } ^ { 1 } W _ { 2 } ^ { 2 } ( Q _ { t } ^ { \pmb { \eta } } , \widetilde { Q } _ { t } ) \rho ( d t ) .\tag{201}
$$

The corresponding integrated scores are $\begin{array} { r l r } { A _ { \mathrm { t a n } } ( \pmb { \eta } ) } & { { } : = } & { \int _ { 0 } ^ { 1 } a _ { \mathrm { t a n } } ( t , \pmb { \eta } ) \rho ( d t ) , ~ A _ { \mathrm { r e s } } ( \pmb { \eta } ) ~ : = } \end{array}$ $\begin{array} { r } { \int _ { 0 } ^ { 1 } a _ { \mathrm { r e s } } ( t , \pmb { \eta } ) \rho ( d t ) } \end{array}$ , and $\begin{array} { r } { J _ { \mathrm { K L } } ( \pmb { \eta } ) : = \int _ { 0 } ^ { 1 } d _ { \mathrm { K L } } ( t , \pmb { \eta } ) \rho ( d t ) } \end{array}$ . Residual $L ^ { 2 }$ is used when $h _ { t , \eta }$ is square integrable, and static $W _ { 2 }$ when the marginals have finite second moments.

## C.2 WEIGHTED TRANSPORT OPERATOR

Fix $( t , \eta )$ and write $Q = Q _ { t } ^ { \eta } , q = q _ { t } ^ { \eta }$ , and $h = h _ { t , \eta }$ . Let $L _ { 0 } ^ { 2 } ( Q ) = \{ f \in L ^ { 2 } ( Q ) : \mathbb { E } _ { Q } [ f ] = 0 \}$ For the $\dot { L } ^ { 2 }$ operator representation, assume $h \in L _ { 0 } ^ { 2 } ( Q )$ , a densely defined closed weighted Dirichlet form, and the weighted Poincare inequality of Appendix B.7. The associated self-adjoint operator´ ${ \mathsf { L } } _ { q }$ on $L _ { 0 } ^ { 2 } ( Q )$ is characterized by

$$
\langle \varphi , \mathsf { L } _ { q } \psi \rangle _ { L ^ { 2 } ( Q ) } = \mathbb { E } _ { Q } [ \nabla \varphi \cdot \nabla \psi ] , \qquad \psi \in \mathrm { D o m } ( \mathsf { L } _ { q } ) , \quad \varphi \in H _ { q } .\tag{202}
$$

When the strong representation is justified,

$$
{ \mathsf { L } } _ { q } \psi = - q ^ { - 1 } \nabla \cdot ( q \nabla \psi ) .\tag{203}
$$

The mean-zero Poisson potential satisfies $\psi = - \mathsf { L } _ { q } ^ { - 1 } h$ . Testing the weak Poisson equation with $\psi$ gives

$$
a _ { \mathrm { f u l l } } ( t , \pmb { \eta } ) = \big \langle h _ { t } , \mathsf { L } _ { q _ { t } } ^ { - 1 } h _ { t } \big \rangle _ { L ^ { 2 } ( Q _ { t } ) } .\tag{204}
$$

Thus Full applies the inverse weighted transport operator to the continuity forcing before measuring its quadratic size. This representation requires the stated $L ^ { 2 }$ operator assumptions; the more general variational characterization of Theorem 2 only requires the forcing to define a bounded functional on $H _ { q _ { t } }$ , in which case the inverse is understood from $H _ { q _ { t } } ^ { * }$ to $H _ { q _ { t } }$ and the pairing is dual rather than an $L ^ { 2 }$ inner product.

## C.3 TANGENT-INVISIBLE CORRECTION AND LACK OF UNIVERSAL CONTROL

Remark 9 (Finite-rank origin of hidden action). Let $\mathcal { H } = L ^ { 2 } ( Q ; \mathbb { R } ^ { d } )$ and $L \delta = \mathbb { E } _ { Q } [ J _ { \Phi _ { \eta } } \delta ]$ . Then rank $( L ) \ \leq \ R$ and Range $( L ^ { * } ) \ = \ \{ J _ { \Phi _ { n } } ^ { \top } b : b \in \mathbb { R } ^ { R } \}$ . Proposition 2 identifies Tangent as the orthogonal projection of the Full correction onto this range and places Hidden in ker L. If H is infinite dimensional,finite R leaves an infinite-dimensional kernel; with D discrete velocity degrees offreedom, dim ker $L \ge D - R$ . This dimension count identifies unresolved directions but does not imply nonzero hidden action in every instance. The Galerkin interpretation and exact equality conditions are given in Remark 6 and Corollary 10.

Theorem 3 (No universal tangent control). There is no universal finite K such that $a _ { \mathrm { f u l l } } ( t , \eta ) \leq$ $K a _ { \mathrm { t a n } } ( t , \eta )$ over all regular instances ofthe information-projected construction. There are smooth instances with $a _ { \mathrm { t a n } } ~ = ~ 0 ~ < ~ a _ { \mathrm { f u l l } }$ , and smooth families with positive Tangent action for which $a _ { \mathrm { f u l l } } / a _ { \mathrm { t a n } } \to \infty$ . The zero-versus-positive construction can be stationary, so no such universal bound holdsfor the integrated actions either.

Example 1 (A stationary law with entirely hidden correction). Take $\Phi ( x ) = \cos x , a > 0 , \vartheta \neq 0 ,$ and

$$
I _ { 0 } ( z ) = \frac { 1 } { 2 \pi } \int _ { 0 } ^ { 2 \pi } e ^ { z \cos x } d x , \qquad \widetilde { q } ( x ) = \frac { e ^ { - \vartheta \cos x } } { 2 \pi I _ { 0 } ( \vartheta ) } , \qquad u ( x ) = a e ^ { \vartheta \cos x } .\tag{205}
$$

The reference flux $\widetilde { q } u = a / ( 2 \pi I _ { 0 } ( \vartheta ) )$ is constant. Information projection onto $\mathbb { E } [ \cos X ] = 0$ gives the uniform law $q = ( 2 \pi ) ^ { - 1 }$ , with

$$
a _ { \mathrm { t a n } } = 0 , \qquad a _ { \mathrm { f u l l } } = a ^ { 2 } \big [ I _ { 0 } ( 2 \vartheta ) - I _ { 0 } ( \vartheta ) ^ { 2 } \big ] > 0 .\tag{206}
$$

Verification and proof of Theorem 3. To include the positive-Tangent perturbation, fix $t _ { 0 } \in ( 0 , 1 )$ and set $\lambda _ { t } = \vartheta \dot { + } 2 \varepsilon \dot { ( t - t _ { 0 } ) }$ . With $A ( \alpha ) = \log \mathbb { E } _ { \widetilde { Q } } [ e ^ { \overset { . } { \alpha } \cos X } ]$ , define

$$
q _ { t } ( x ) = \widetilde { q } ( x ) \exp ( \lambda _ { t } \cos x - A ( \lambda _ { t } ) ) , \qquad c ( t ) = \mathbb { E } _ { q _ { t } } [ \cos X ] .\tag{207}
$$

This is the exact information projection for its target moment. All densities are smooth and positive, and the scalar observable covariance is positive. At $t _ { 0 } , A ( \vartheta ) = - \log I _ { 0 } ( \vartheta )$ , so $q _ { t _ { 0 } } \ = \ ( 2 \pi ) ^ { - 1 }$ $c ( t _ { 0 } ) = 0 , C _ { t _ { 0 } } = G _ { t _ { 0 } } = 1 / 2$ , and $\dot { c } ( t _ { 0 } ) = \varepsilon$ . Since $\mathbb { E } _ { q _ { t _ { 0 } } } [ - \sin X u ( X ) ] { \stackrel { - } { = } } 0$ , one has $r _ { t _ { 0 } } = - \varepsilon$ and $\delta _ { t _ { 0 } } ^ { \mathrm { t a n } } = - 2 \varepsilon$ sin x. The continuity residual and Poisson equation are

$$
h _ { t _ { 0 } } ( x ) = 2 \varepsilon \cos x + u ^ { \prime } ( x ) , \qquad \psi ^ { \prime \prime } ( x ) = 2 \varepsilon \cos x + u ^ { \prime } ( x ) .\tag{208}
$$

Periodicity fixes the integration constant in

$$
\psi ^ { \prime } ( x ) = 2 \varepsilon \sin x + u ( x ) - \mathbb { E } _ { q _ { t _ { 0 } } } [ u ] , \qquad \delta _ { t _ { 0 } } ^ { \star } = - \psi ^ { \prime } ( x ) .\tag{209}
$$

The sine term is odd and $u - \mathbb { E } [ u ]$ is even under $x \mapsto 2 \pi - x .$ , so their inner product vanishes. Consequently,

$$
a _ { \mathrm { t a n } } ( t _ { 0 } ) = 2 \varepsilon ^ { 2 } , \qquad a _ { \mathrm { f u l l } } ( t _ { 0 } ) = 2 \varepsilon ^ { 2 } + a ^ { 2 } \bigl [ I _ { 0 } ( 2 \vartheta ) - I _ { 0 } ( \vartheta ) ^ { 2 } \bigr ] .\tag{210}
$$

The bracket is the uniform-law variance of $e ^ { \vartheta \cos X }$ and is strictly positive because $\vartheta \neq 0$ . Setting $\varepsilon = 0$ gives Example 1 at every time. For ε $\neq 0$

$$
{ \frac { a _ { \mathrm { f u l l } } ( t _ { 0 } ) } { a _ { \mathrm { t a n } } ( t _ { 0 } ) } } = 1 + { \frac { a ^ { 2 } [ I _ { 0 } ( 2 \vartheta ) - I _ { 0 } ( \vartheta ) ^ { 2 } ] } { 2 \varepsilon ^ { 2 } } } \longrightarrow \infty \qquad \mathrm { a s ~ } \varepsilon  0 .\tag{211}
$$

## C.3.1 NO UNIVERSAL TANGENT CONTROL IN THE UNBALANCED GEOMETRY

Corollary 13 (No universal tangent control in the unbalanced geometry). Fix any $\kappa > 0$ . There is no universal finite constant K such that

$$
a _ { \mathrm { f u l l } } ^ { \mathrm { u b } } ( t , \pmb { \eta } ) \leq K a _ { \mathrm { t a n } } ^ { \mathrm { u b } } ( t , \pmb { \eta } )\tag{212}
$$

for all regular finite-measure instances of the unbalanced construction. In fact, there are smooth instances with

$$
a _ { \mathrm { t a n } } ^ { \mathrm { u b } } ( t , \pmb { \eta } ) = 0 \qquad a n d \qquad a _ { \mathrm { f u l l } } ^ { \mathrm { u b } } ( t , \pmb { \eta } ) > 0 .\tag{213}
$$

Proof. Use the construction in the proof of Theorem $^ { 3 , }$ set $M _ { t } \equiv 1$ and $g _ { \mathrm { r e f } } \equiv 0$ , and evaluate at $t = t _ { 0 }$ , where $q _ { t _ { 0 } }$ is uniform and

$$
h _ { \mathrm { u b } } ( x ) = 2 \varepsilon \cos x + u ^ { \prime } ( x ) .\tag{214}
$$

For the single observable $\Phi ( x ) = \cos x$ , Corollary 8 gives

$$
G ^ { \mathrm { u b } } = \mathbb { E } [ \sin ^ { 2 } X ] + \frac { 1 } { \kappa } \mathbb { E } [ \cos ^ { 2 } X ] = \frac { 1 } { 2 } \left( 1 + \frac { 1 } { \kappa } \right) ,\tag{215}
$$

and therefore

$$
a _ { \mathrm { t a n } } ^ { \mathrm { u b } } ( t _ { 0 } ) = \frac { 2 \kappa } { \kappa + 1 } \varepsilon ^ { 2 } .\tag{216}
$$

For the Full problem, the screened operator on the uniform torus is

$$
\mathcal { L } _ { \kappa } : = - \partial _ { x x } + \kappa ^ { - 1 } .\tag{217}
$$

The forcing 2ε cos x is even, whereas $u ^ { \prime } ( x )$ is odd, and $\boldsymbol { \mathcal { L } } _ { \kappa } ^ { - 1 }$ preserves parity. Hence the two forcing components are orthogonal in the screened energy geometry. Since

$$
\mathcal { L } _ { \kappa } ( \cos x ) = \left( 1 + \frac { 1 } { \kappa } \right) \cos x ,\tag{218}
$$

the even component contributes exactly

$$
{ \frac { 2 \kappa } { \kappa + 1 } } \varepsilon ^ { 2 } .\tag{219}
$$

The odd component contributes

$$
c _ { \kappa } : = \mathbb { E } _ { q _ { t _ { 0 } } } \left[ u ^ { \prime } \mathcal { L } _ { \kappa } ^ { - 1 } u ^ { \prime } \right] > 0 ,\tag{220}
$$

because $\mathcal { L } _ { \kappa }$ is strictly positive and $u ^ { \prime } \not \equiv 0$ . Thus

$$
a _ { \mathrm { f u l l } } ^ { \mathrm { u b } } ( t _ { 0 } ) = \frac { 2 \kappa } { \kappa + 1 } \varepsilon ^ { 2 } + c _ { \kappa } .\tag{221}
$$

$\mathbf { A } \mathfrak { t } \varepsilon = 0$ , equation 216–equation 221 give equation 213. For $\varepsilon \neq 0$ their ratio diverges as $\varepsilon  0 ,$ ruling out any universal finite K. □

## C.4 RESIDUAL AMPLITUDE VERSUS TRANSPORT EFFORT

The residual comparator uses the full normalized residual, $a _ { \mathrm { r e s } } = \| h \| _ { L ^ { 2 } ( Q ) } ^ { 2 }$ , rather than a finitedimensional projection. Its difference from Full is the missing inverse transport operator. Weighted squared Liouville residuals also appear in best-fit statistical closure (Turkington, 2013); the present comparator is the plain $L ^ { 2 } ( Q )$ norm on an observation-prescribed path, not a weighted objective that selects the path itself.

Proposition 10 (Spectral comparison). Under Appendix C.2, suppose ${ \mathsf { L } } _ { q }$ admits a complete orthonormal eigenbasis $( e _ { j } ) _ { j \geq 1 }$ on $L _ { 0 } ^ { 2 } ( Q )$ , with ${ \mathsf { L } } _ { q } e _ { j } ~ = ~ \nu _ { j } e _ { j }$ and $0 ~ < ~ \nu _ { 1 } ~ \leq ~ \nu _ { 2 } ~ \leq ~ \cdot \cdot \cdot$ . If $\begin{array} { r } { h = \sum _ { j } c _ { j } e _ { j } } \end{array}$ , then

$$
a _ { \mathrm { r e s } } = \sum _ { j } c _ { j } ^ { 2 } , \qquad a _ { \mathrm { f u l l } } = \sum _ { j } \frac { c _ { j } ^ { 2 } } { \nu _ { j } } , \qquad a _ { \mathrm { f u l l } } \leq \nu _ { 1 } ^ { - 1 } a _ { \mathrm { r e s } } .\tag{222}
$$

For residuals supported in a spectral band $\nu _ { j } \leq \Lambda$

$$
\nu _ { 1 } a _ { \mathrm { f u l l } } \leq a _ { \mathrm { r e s } } \leq \Lambda a _ { \mathrm { f u l l } } .\tag{223}
$$

$I f \nu _ { j }  \infty ,$ , nofinite C satisfies $a _ { \mathrm { r e s } } \leq C a _ { \mathrm { f u l l } }$ for every $h \in L _ { 0 } ^ { 2 } ( Q )$ .

Proof. Parseval’s identity gives the residual norm. Expanding ${ \mathsf { L } } _ { q } ^ { - 1 } h$ in the same eigenbasis gives the Full quadratic form. The inequalities follow by bounding the active eigenvalues above and below. A universal reverse bound would imply $1 \leq C / \nu _ { j }$ upon taking $h = e _ { j }$ , contradicting $\nu _ { j } \to \infty . \quad \sqcup$

Corollary 14 (Pointwise ranking reversal). $I f \nu _ { j } > \nu _ { \ell }$ and $1 < a _ { A } / a _ { B } < \sqrt { \nu _ { j } / \nu _ { \ell } } ,$ , then $h _ { A } = a _ { A } e _ { j }$ and $h _ { B } = a _ { B } e _ { \ell }$ satisfy $a _ { \mathrm { r e s } } ( h _ { A } { \bar { ) } } > a _ { \mathrm { r e s } } ( h _ { B } { \bar { ) } }$ but $a _ { \mathrm { f u l l } } ( h _ { A } ) < a _ { \mathrm { f u l l } } ( h _ { B } )$

Proof. The two comparisons are respectively $a _ { A } ^ { 2 } > a _ { B } ^ { 2 }$ and $a _ { A } ^ { 2 } / \nu _ { j } < a _ { B } ^ { 2 } / \nu _ { \ell }$

Example 2 (Exact information-projected Fourier family). Let $\widetilde { q } = q _ { 0 } = ( 2 \pi ) ^ { - 1 }$ and $u = 0 ,$ . For an integer $k \geq 1$ , define

$$
\Phi _ { k } ( x ) = \cos ( k x ) , \qquad q _ { \lambda , k } ( x ) = { \frac { e ^ { \lambda \cos ( k x ) } } { 2 \pi I _ { 0 } ( \lambda ) } } , \qquad c ( \lambda ) = { \frac { d } { d \lambda } } \log I _ { 0 } ( \lambda ) .\tag{224}
$$

Periodicity makes the normalization and the moment $\mathbb { E } _ { q _ { \lambda , k } } [ \Phi _ { k } ] = c ( \lambda )$ independent of k. Hence ${ { q } _ { \lambda , k } }$ is exactly the information projection onto this momentfiber. Along $q _ { k , t } = q _ { \lambda _ { k } ( t ) , k } ,$

$$
h _ { k , t } ( x ) = \dot { \lambda } _ { k } ( t ) \big ( \cos ( k x ) - c ( \lambda _ { k } ( t ) ) \big ) .\tag{225}
$$

At $\lambda _ { k } ( t _ { 0 } ) = 0$ and $\dot { \lambda } _ { k } ( t _ { 0 } ) = a ;$ , the density is uniform, $h _ { k , t _ { 0 } } = a \cos ( k x )$ , and the mean-zero Poisson potential is $\psi ( x ) = - a \cos ( k x ) / k ^ { 2 }$ . Therefore

$$
a _ { \mathrm { r e s } } ( t _ { 0 } , k ) = { \frac { a ^ { 2 } } { 2 } } , \qquad a _ { \mathrm { f u l l } } ( t _ { 0 } , k ) = { \frac { a ^ { 2 } } { 2 k ^ { 2 } } } .\tag{226}
$$

The pair $h _ { A } = 2 \cos ( 3 x )$ and $h _ { B } = \cos x$ gives the exact instantaneous ranking reversal $2 > 1 / 2$ for residual $L ^ { 2 }$ and $2 / 9 < 1 / 2$ for Full.

Transport metric along the Fourier family. Write $q _ { \lambda } = q _ { \lambda , 1 }$ and define

$$
g ( \lambda ) = \| \cos x - c ( \lambda ) \| _ { H ^ { - 1 } ( q _ { \lambda } ) } ^ { 2 } .\tag{227}
$$

This function is continuous, strictly positive, and satisfies $g ( 0 ) = 1 / 2$ . An explicit verification avoids any assumption about a limiting discretization. Set $f _ { \lambda } ( x ) = \cos x - c ( \lambda )$ and

$$
F _ { \lambda } ( x ) = \int _ { 0 } ^ { x } q _ { \lambda } ( y ) f _ { \lambda } ( y ) d y , \qquad b _ { \lambda } = { \frac { \int _ { 0 } ^ { 2 \pi } F _ { \lambda } ( x ) / q _ { \lambda } ( x ) d x } { \int _ { 0 } ^ { 2 \pi } 1 / q _ { \lambda } ( x ) d x } } .\tag{228}
$$

For a unit multiplier rate, $v _ { \lambda } ~ = ~ ( b _ { \lambda } - F _ { \lambda } ) / q _ { \lambda }$ satisfies $( q _ { \lambda } v _ { \lambda } ) ^ { \prime } = - q _ { \lambda } f _ { \lambda }$ . Its spatial integral vanishes by the definition of $b _ { \lambda }$ , so it is a periodic gradient and the minimum-energy correction. Consequently,

$$
g ( \lambda ) = \int _ { 0 } ^ { 2 \pi } { \frac { ( b _ { \lambda } - F _ { \lambda } ( x ) ) ^ { 2 } } { q _ { \lambda } ( x ) } } d x .\tag{229}
$$

On every compact multiplier interval, $q _ { \lambda }$ is uniformly positive and all terms vary smoothly with $\lambda ,$ proving continuity. The energy cannot vanish because $f _ { \lambda }$ is not identically zero. At $\lambda = 0$ $v _ { 0 } = - \sin x ,$ , giving $g ( 0 ) = 1 / 2$ . Finally, if $\chi _ { \lambda }$ solves $\mathsf { L } _ { q _ { \lambda , 1 } \chi _ { \lambda } } = f _ { \lambda }$ , then $k ^ { - 2 } \chi _ { \lambda } ( k x )$ solves the frequency-k problem. Thus the exact action along any differentiable multiplier path is

$$
a _ { \mathrm { f u l l } } ( t , k ) = \frac { \dot { \lambda } _ { k } ( t ) ^ { 2 } } { k ^ { 2 } } g ( \lambda _ { k } ( t ) ) .\tag{230}
$$

## C.5 FORWARD KL: LOCAL GEOMETRY AND EXACT SEPARATIONS

Proposition 11 (Exact dual form and local information geometry). At fixed $( t , \eta )$ , abbreviate $A =$ $A _ { t , \eta }$ and $\phi = \Phi _ { \eta }$ , and let $Q _ { \lambda }$ be the exponential tilt $o f { \widetilde Q } _ { t }$ . Write $\widetilde { c } = \mathbb { E } _ { \widetilde { Q } _ { t } } [ \phi ]$ and $\widetilde { C } = \mathrm { C o v } _ { \widetilde { Q } _ { t } } ( \phi , \phi )$ Then

$$
D _ { \mathrm { K L } } ( Q _ { \lambda } | | \widetilde { Q } _ { t } ) = \lambda ^ { \top } \nabla A ( \lambda ) - A ( \lambda ) .\tag{231}
$$

At the calibrated multiplier, $\nabla A ( \lambda ) = c _ { \pmb { \eta } } ( t ) . \ I f A i s C ^ { 3 }$ near zero, then

$$
D _ { \mathrm { K L } } ( Q _ { \lambda } | | \widetilde { Q } _ { t } ) = \frac { 1 } { 2 } \lambda ^ { \top } \widetilde { C } \lambda + O ( | | \lambda | | ^ { 3 } ) .\tag{232}
$$

If also $\widetilde { C } \succ 0 ,$ then, for $\Delta c = \mathbb { E } _ { Q _ { \lambda } } [ \phi ] - \widetilde { c } ,$

$$
D _ { \mathrm { K L } } ( Q _ { \lambda } \| \widetilde { Q } _ { t } ) = \frac { 1 } { 2 } \Delta c ^ { \top } \widetilde { C } ^ { - 1 } \Delta c + O ( \| \Delta c \| ^ { 3 } ) .\tag{233}
$$

Proof. Integrating log $( d Q _ { \lambda } / d \widetilde Q _ { t } ) = \lambda ^ { \top } \phi - A ( \lambda )$ ) gives the dual identity. Since $A ( 0 ) = 0$ , Taylor expansion yields $A ( \lambda ) = \widetilde { c } ^ { \top } \lambda + \frac { 1 } { 2 } \lambda ^ { \top } \widetilde { C } \lambda + O ( \Vert \lambda \Vert ^ { 3 } )$ and $\nabla A ( \lambda ) = \widetilde c + \widetilde C \lambda + O ( \| \lambda \| ^ { 2 } )$ . Substitution cancels the linear term. When $\widetilde C \succ 0$ , the inverse-function theorem gives $\lambda = \widetilde { C } ^ { - 1 } \Delta c + O ( \| \Delta c \| ^ { 2 } )$ proving the moment-coordinate expansion. □

Marginal tilt versus multiplier velocity. For a stationary reference with $u = 0$ , let $\overline { { { \phi } } } _ { \lambda } = \phi -$ $\mathbb { E } _ { Q _ { \lambda } } [ \phi ]$ . When the centered coordinates and inverse operator are well defined, set

$$
[ \mathsf { H } ( \lambda ) ] _ { a b } = \langle \overline { { \phi } } _ { \lambda , a } , \mathsf { L } _ { q _ { \lambda } } ^ { - 1 } \overline { { \phi } } _ { \lambda , b } \rangle _ { L ^ { 2 } ( Q _ { \lambda } ) } .\tag{234}
$$

The identity $h _ { t } = \dot { \lambda } _ { t } ^ { \top } \overline { { \phi } } _ { \lambda _ { t } }$ implies

$$
a _ { \mathrm { f u l l } } ( t ) = \dot { \lambda } _ { t } ^ { \top } { \sf H } ( \lambda _ { t } ) \dot { \lambda } _ { t } .\tag{235}
$$

In a regular parametrization admitting a Wasserstein pullback, H is the transport metric in natural parameter coordinates (Chen & Li, 2020; Li & Zhao, 2023). It differs from both the covariance/Fisher matrix and the observable derivative Gram matrix. Marginal KL acts on tilt displacement, whereas Full acts on the velocity of the prescribed path in transport geometry.

Proposition 12 (No universal comparison between forward KL and Full). Even with the state space T and time interval [0, 1] fixed, neither $d _ { \mathrm { K L } } \leq C a _ { \mathrm { f u l l } }$ nor $a _ { \mathrm { f u l l } } \leq C d _ { \mathrm { K I } }$ holds with an instanceindependentfinite C over all regular instances ofthe information-projected construction. Likewise, $f o r \bar { \rho } ( d t ) = \dot { d t }$ , neither $J _ { \mathrm { K L } } \leq \breve { C } \mathcal { A }$ nor $\mathcal { A } \le C \mathcal { j } _ { \mathrm { K L } }$ holds universally.

Example 3 (Pointwise separations). Under the uniform reference with $u = 0 ;$ , take the Fourier family in equation 224. A constant nonzero multiplier λ<sub>0</sub> gives a stationary projected law distinct from the reference: $h _ { t } = 0 ,$ , so $a _ { \mathrm { f u l l } } = 0 ,$ , while $d _ { \mathrm { K L } } > 0$ . Conversely, at $\lambda _ { k } ( t _ { 0 } ) = 0$ and $\dot { \lambda } _ { k } ( t _ { 0 } ) = a \neq 0 ;$ , the two marginals coincide but their time derivatives need not, and

$$
d _ { \mathrm { K L } } ( t _ { 0 } , k ) = 0 , \qquad a _ { \mathrm { f u l l } } ( t _ { 0 } , k ) = \frac { a ^ { 2 } } { 2 k ^ { 2 } } > 0 .\tag{236}
$$

The constant-tilt construction also gives $J _ { \mathrm { K L } } > 0 = \mathcal { A }$ after integration.

Example 4 (Rapid marginal oscillation). Fix $\epsilon > 0 ,$ , set $\lambda _ { n } ( t ) = \epsilon \sin ( 2 \pi n t )$ , and take

$$
q _ { n , t } ( x ) = \frac { \exp ( \lambda _ { n } ( t ) \cos x ) } { 2 \pi I _ { 0 } ( \lambda _ { n } ( t ) ) } , \qquad n \in \mathbb { N } , \quad n \geq 1 .\tag{237}
$$

Every path is an exact information projection for its cos x moment trajectory and satisfies $q _ { n , 0 } =$ $q _ { n , 1 } = q _ { 0 }$ . For integer n,

$$
J _ { \mathrm { K L } } ( n ) = J _ { \mathrm { K L } } ( 1 ) , \qquad \ A ( n ) = n ^ { 2 } A ( 1 ) > 0 .\tag{238}
$$

There exist constants $C _ { \epsilon } < \infty$ and $m _ { \epsilon } > 0 ,$ , independent of $n ,$ such that

$$
J _ { \mathrm { K L } } ( n ) \leq C _ { \epsilon } \epsilon ^ { 2 } , \qquad A ( n ) \geq 2 \pi ^ { 2 } m _ { \epsilon } \epsilon ^ { 2 } n ^ { 2 } .\tag{239}
$$

As a tie-breaking thought experiment, suppose the primary task does not distinguish these paths beyond constraints that they all satisfy, such as their common endpoints. Their time-averaged forward KL displacement from the reference is identical for every integer n, yet their required dynamical revision grows as $\mathit { n } ^ { 2 }$ . Under the minimum-additional-commitment preference specified in the main text, Full therefore distinguishes a path that makes a modest reference-relative excursion from one that repeatedly moves back andforth through the same range, even though a static marginal criterion and the declared endpoint task do not.

Verification and completion of Proposition 12. Let $D ( \boldsymbol { \lambda } )$ be the forward KL of $q _ { \lambda , 1 }$ relative to q<sub>0</sub>. This function is smooth and vanishes to second order at zero, so $D ( \lambda ) \ \leq \ C _ { \epsilon } \lambda ^ { 2 }$ on $[ - \epsilon , \epsilon ]$ . The function $g$ in equation 227 has a positive minimum $m _ { \epsilon }$ on this interval. Thus

$$
J _ { \mathrm { K L } } ( n ) = \int _ { 0 } ^ { 1 } D ( \epsilon \sin ( 2 \pi n t ) ) d t , \qquad A ( n ) = ( 2 \pi n \epsilon ) ^ { 2 } \int _ { 0 } ^ { 1 } \cos ^ { 2 } ( 2 \pi n t ) g ( \epsilon \sin ( 2 \pi n t ) ) d t .\tag{240}
$$

The substitution $\theta = \pm$ πnt integrates n identical periods, proving equation 238. Bounding D and g as above and using $\begin{array} { r } { \int _ { 0 } ^ { 1 } \cos ^ { 2 } ( 2 \pi n t ) d t = 1 / 2 } \end{array}$ proves equation 239. Constants may be chosen uniformly for $0 < \epsilon \le \epsilon _ { 0 }$ by taking bounds on a fixed compact interval $[ - \epsilon _ { 0 } , \epsilon _ { 0 } ]$ . The marginal score is positive for $\epsilon > 0 , \mathrm { s o } \dot { \mathcal { A } } ( n ) / \mathsf { \bar { J } } _ { \mathrm { K L } } ( n ) \to \infty$ . Together with Example 3, this proves all claims of Proposition 12. □

Corollary 15 (Local KL/Full ranking reversal). Let $s \in C ^ { 1 } ( [ 0 , 1 ] )$ ) be nonconstant and compare $q _ { A , t } = q _ { 2 \epsilon s ( t ) , 3 }$ with $q _ { B , t } = q _ { \epsilon s ( t ) , 1 }$ . Then $J _ { \mathrm { K L } } ( A ) > J _ { \mathrm { K L } } ( B )$ for every $\epsilon > 0$ , whereas $\mathcal { A } ( A ) <$ $\mathcal A ( B )$ for all sufficiently small $\epsilon > 0 .$ . More precisely,

$$
J _ { \mathrm { K L } } ( A ) = \epsilon ^ { 2 } \int _ { 0 } ^ { 1 } s ( t ) ^ { 2 } d t + o ( \epsilon ^ { 2 } ) , \qquad J _ { \mathrm { K L } } ( B ) = \frac { \epsilon ^ { 2 } } { 4 } \int _ { 0 } ^ { 1 } s ( t ) ^ { 2 } d t + o ( \epsilon ^ { 2 } ) ,\tag{241}
$$

while

$$
\mathcal { A } ( A ) = \frac { 2 \epsilon ^ { 2 } } { 9 } \int _ { 0 } ^ { 1 } \dot { s } ( t ) ^ { 2 } d t + o ( \epsilon ^ { 2 } ) , \qquad \mathcal { A } ( B ) = \frac { \epsilon ^ { 2 } } { 2 } \int _ { 0 } ^ { 1 } \dot { s } ( t ) ^ { 2 } d t + o ( \epsilon ^ { 2 } ) .\tag{242}
$$

Proof. For $A ( \lambda ) = \log I _ { 0 } ( \lambda ) , D ( \lambda ) = \lambda A ^ { \prime } ( \lambda ) - A ( \lambda )$ is even and $D ^ { \prime } ( \lambda ) = \lambda \mathrm { { V a r } } _ { q _ { \lambda , 1 } } ( \cos X )$ Hence D is strictly increasing in $| \lambda |$ away from zero. Since s is nonconstant, it is nonzero on a set of positive measure, proving the strict KL ordering. The expansion $D ( \lambda ) = \lambda ^ { 2 } / 4 + o ( \lambda ^ { 2 } )$ gives equation 241. Equation equation 230 gives the exact actions $4 \epsilon ^ { 2 } \dot { s } ( t ) ^ { 2 } \dot { g ( 2 \epsilon s ( t ) ) / 9 }$ and $\dot { \epsilon } ^ { 2 } \dot { s } ( \dot { t } ) ^ { 2 } g ( \epsilon s ( \dot { t } ) )$ Continuity of g at zero, $g ( 0 ) = 1 / 2$ , and boundedness of s give equation 242 after integration. Because $\begin{array} { r } { \int _ { 0 } ^ { 1 } \dot { s } ( t ) ^ { 2 } d t > 0 } \end{array}$ , the leading coefficients yield the opposite strict Full ranking for sufficiently small ϵ. □

## C.6 STATIC WASSERSTEIN DISPLACEMENT AND INITIAL MATCHING

The comparator $\mathcal { A } _ { W _ { 2 } }$ in equation 201 averages static distances between projected and reference marginals. It is not a dynamic endpoint-transport objective and does not optimize or directly assess the time derivative of the prescribed projected path.

Proposition 13 (Static Wasserstein displacement and Full are non-equivalent). Over regular instances of the information-projected construction on T, neither $\mathcal { A } _ { W _ { 2 } } \leq C \mathcal { A }$ nor $\mathcal { A } \le C \mathcal { A } _ { W _ { 2 } }$ holds with an instance-independent finite $C$ when $\rho ( d t ) = d t$ . The analogous pointwise inequalities for $W _ { 2 } ^ { 2 } ( Q _ { t } ^ { \eta } , \widetilde { Q } _ { t } )$ and $a _ { \mathrm { f u l l } } ( t , \eta )$ alsofail. In the rapid-oscillationfamily ofExample 4,

$$
\mathcal { A } _ { W _ { 2 } } ( n ) = \mathcal { A } _ { W _ { 2 } } ( 1 ) , \qquad \mathcal { A } ( n ) = n ^ { 2 } \mathcal { A } ( 1 ) ,\tag{243}
$$

and there is $M _ { \epsilon } < \infty ,$ , independent of $n ,$ such that

$$
\mathcal { A } _ { W _ { 2 } } ( n ) \leq M _ { \epsilon } \epsilon ^ { 2 } , \qquad \mathcal { A } ( n ) \geq 2 \pi ^ { 2 } m _ { \epsilon } \epsilon ^ { 2 } n ^ { 2 } .\tag{244}
$$

Proof. The constant nonzero tilt in Example 3 has zero Full action but positive Wasserstein distance from the reference, both pointwise and after integration. At a zero multiplier with nonzero multiplier derivative, the Wasserstein distance is zero while Full is positive. These give the pointwise failures and rule out $\mathcal { A } _ { W _ { 2 } } \leq C \mathcal { A }$ without initial matching. For the reverse integrated bound, the time average of $W _ { 2 } ^ { 2 } ( q _ { \epsilon \sin ( 2 \pi n t ) , 1 } , q _ { 0 } )$ is independent of integer n by periodicity, whereas Full has the exact ${ \overset { \smile } { n } } ^ { 2 }$ scaling already proved.

For the small-amplitude bound, set $M _ { \epsilon } = \operatorname* { m a x } _ { | \lambda | \leq \epsilon } g ( \lambda )$ . Given $| \lambda | \leq \epsilon ,$ , the path $r \mapsto q _ { r \lambda , 1 }$ $r \in [ 0 , 1 ]$ , has minimum kinetic energy density $\lambda ^ { 2 } g ( r \lambda )$ . Using this path as an admissible competitor in dynamic optimal transport gives

$$
W _ { 2 } ^ { 2 } ( q _ { \lambda , 1 } , q _ { 0 } ) \leq \int _ { 0 } ^ { 1 } \lambda ^ { 2 } g ( r \lambda ) d r \leq M _ { \epsilon } \lambda ^ { 2 } .\tag{245}
$$

Substitution of $\lambda = \lambda _ { n } ( t )$ and integration prove equation 244. The constant can be chosen uniformly for $0 < \epsilon \le \epsilon _ { 0 }$ by using the maximum over $[ - \epsilon _ { 0 } , \epsilon _ { 0 } ]$ □

Proposition 14 (Wasserstein stability relative to a matched-initial reference rollout). Let $\mathcal { X } = \mathbb { R } ^ { d }$ or a flat torus, take $\rho ( d t ) = d t$ , and assume $Q _ { t } ^ { \eta } , R _ { t } ^ { \eta } \in \mathcal P _ { 2 } ( \mathcal X )$ and $\mathcal { A } ( \eta ) < \infty$ . Let $\mathsf { T } _ { 0 , t } ^ { u }$ denote the characteristic flow generated by the frozen reference velocity $u _ { t } ,$ , and define the matched-initial reference rollout

$$
R _ { t } ^ { \eta } : = ( \mathsf { T } _ { 0 , t } ^ { u } ) _ { \# } Q _ { 0 } ^ { \eta } .\tag{246}
$$

Suppose $u _ { t }$ and $u _ { t } + \delta _ { t } ^ { \star }$ admit classical characteristicflows realizing $R _ { t } ^ { \eta }$ and $Q _ { t } ^ { \eta }$ , respectively, and assume $\| u _ { t } ( x ) - u _ { t } ( y ) \| \leq L ( t ) \| x - y \|$ with $L \geq 0$ integrable, imposed on periodic liftsfor a torus. With $\begin{array} { r } { \Lambda = \int _ { 0 } ^ { 1 } L ( t ) d t } \end{array}$

$$
W _ { 2 } ( Q _ { t } ^ { \eta } , R _ { t } ^ { \eta } ) \leq \int _ { 0 } ^ { t } \exp \left( \int _ { s } ^ { t } L ( \tau ) d \tau \right) \sqrt { a _ { \mathrm { f u l l } } ( s , \eta ) } d s ,\tag{247}
$$

and consequently

$$
\int _ { 0 } ^ { 1 } W _ { 2 } ^ { 2 } ( Q _ { t } ^ { \eta } , R _ { t } ^ { \eta } ) d t \leq e ^ { 2 \Lambda } \mathcal { A } ( \eta ) .\tag{248}
$$

If additionally $Q _ { 0 } ^ { \eta } \ = \ \widetilde { Q } _ { 0 }$ , then $R _ { t } ^ { \eta } = \widetilde { Q } _ { i }$ <sub>t</sub> and the left-hand side equals $\mathcal { A } _ { W _ { 2 } } ( \pmb { \eta } )$ as defined in equation 201.

Moreover, for every globally 1-Lipschitz observable $f ,$

$$
\left| \mathbb { E } _ { Q _ { t } ^ { \eta } } [ f ] - \mathbb { E } _ { R _ { t } ^ { \eta } } [ f ] \right| \leq W _ { 1 } ( Q _ { t } ^ { \eta } , R _ { t } ^ { \eta } ) \leq W _ { 2 } ( Q _ { t } ^ { \eta } , R _ { t } ^ { \eta } ) .\tag{249}
$$

Since $\begin{array} { r } { \int _ { 0 } ^ { t } a _ { \mathrm { f u l l } } ( s , \eta ) d s \leq \mathcal { A } ( \eta ) } \end{array}$ and $\textstyle \int _ { s } ^ { t } L ( \tau ) d \tau \leq \Lambda$ , the pointwise bound also gives

$$
\operatorname* { s u p } _ { \operatorname { L i p } ( f ) \leq 1 } \left. \mathbb { E } _ { Q _ { t } ^ { \eta } } [ f ] - \mathbb { E } _ { R _ { t } ^ { \eta } } [ f ] \right. \leq e ^ { \Lambda } \sqrt { \mathcal { A } ( \eta ) } , \qquad t \in [ 0 , 1 ] .\tag{250}
$$

Thus accumulated Full action controls a one-sided reference-relative stability envelope for Lipschitz observables after removing initial-law mismatch.

Proof. Couple the two characteristic flows from a common initial random state $X _ { 0 } = Y _ { 0 } \sim Q _ { 0 } ^ { \eta }$ with $\dot { X } _ { t } = u _ { t } ( X _ { t } ) + \delta _ { t } ^ { \star } ( X _ { t } )$ and $\dot { Y } _ { t } = u _ { t } ( Y _ { t } )$ . Then $\mathcal { L } ( X _ { t } ) = Q _ { t } ^ { \eta }$ and $\mathcal { L } ( Y _ { t } ) = R _ { t } ^ { \eta }$ . On a torus, use lifts with the same initial representative; lifted Euclidean distance bounds geodesic distance. Gronwall’s inequality gives

$$
\| X _ { t } - Y _ { t } \| \leq \int _ { 0 } ^ { t } \exp \biggl ( \int _ { s } ^ { t } L ( \tau ) d \tau \biggr ) \| \delta _ { s } ^ { \star } ( X _ { s } ) \| d s .\tag{251}
$$

The $L ^ { 2 }$ integral Minkowski inequality and $\mathbb { E } [ \| \delta _ { s } ^ { \star } ( X _ { s } ) \| ^ { 2 } ] = a _ { \mathrm { f u l l } } ( s , \pmb { \eta } )$ give equation 247. Cauchy– Schwarz yields $\begin{array} { r } { W _ { 2 } ^ { 2 } ( Q _ { t } ^ { \eta } , R _ { t } ^ { \eta } ) \leq e ^ { 2 \Lambda } t \int _ { 0 } ^ { t } a _ { \mathrm { f u l l } } ( s , \eta ) d s \leq e ^ { 2 \Lambda } A ( \eta ) } \end{array}$ , and integration in t proves equation 248. Finally, the Kantorovich–Rubinstein bound gives the first inequality in equation 249, while $W _ { 1 } \leq W _ { 2 }$ follows from the finite second-moment assumption. □

The constant-tilt example has $Q _ { 0 } ^ { \eta } \neq { \widetilde Q } _ { 0 } ,$ , so it does not contradict the special-case bound relative to the original frozen reference marginals. Relative to the matched-initial rollout $R _ { t } ^ { \eta }$ , the constant-tilt path has zero Full action and zero matched-initial Wasserstein departure, consistently with Proposition 14. When $Q _ { 0 } ^ { \eta } = \widetilde { Q } _ { 0 }$ , one has $R _ { t } ^ { \eta } = \widetilde { Q } _ { t }$ , and the proposition specializes to a bound on static Wasserstein displacement from the original reference path. The oscillatory family satisfies this initial equality and still rules out a reverse estimate. Likewise, the spectral equivalence in equation 223 requires common positive lower and finite upper spectral bounds to obtain constants uniform across designs. These statements are independent of the scientific-risk constraint.

## D ADDITIONAL EXPERIMENTS AND ABLATIONS

Shared experimental conventions and reproducibility details are provided in Appendix E.   
Supporting-study-specific protocols are given below where needed.

## D.1 ADDITIONAL EXPERIMENTS

We report two additional supporting experiments. The retrospective double-gyre study tests repeatability of the Full-versus-Law contrast across fresh observations and independently trained fits within the same declared reference-learning regime. The finite-measure study extends FIDE beyond mass-preserving probability laws by allowing reference-relative correction through both transport and reaction.

<table><tr><td>Allowance</td><td>Full holdout risk</td><td>Risk increase (%)</td><td>Excess over allowance (pp)</td></tr><tr><td>0.5%</td><td>0.038777</td><td>0.4404</td><td>-0.0596</td></tr><tr><td>1%</td><td>0.039016</td><td>1.0598</td><td>+0.0598</td></tr><tr><td>2%</td><td>0.039289</td><td>1.7686</td><td>-0.2314</td></tr><tr><td>3%</td><td>0.039531</td><td>2.3945</td><td>-0.6055</td></tr><tr><td>4%</td><td>0.039950</td><td>3.4794</td><td>-0.5206</td></tr><tr><td>5%</td><td>0.040538</td><td>5.0023</td><td>+0.0023</td></tr></table>

Table 2: Held-out scientific-risk transfer for the retrospective frontier. The holdout Law risk is 0.038607. Negative excess indicates a point estimate below the nominal allowance. Values are descriptive because the retained summaries do not contain paired per-trial scientific losses needed for matching simultaneous risk-feasibility inference.

## D.1.1 RETROSPECTIVE DOUBLE-GYRE REPEATABILITY STUDY

System and principal results. The study uses the double-gyre dynamics defined in Appendix E.3, with four Gaussian sensors observing aggregate particle responses at nine acquisition times. Selection uses a frozen microscopic target simulation bank together with three independently seeded endpoint-trained references fitted from the same physical endpoint dataset. The study therefore ask a narrow question: does the Full-versus-Law action reduction repeat across fresh observations and across independent fits within the same declared reference-learning regime? On 64 shared fresh trials per reference, the mean reference-wise reduction ranges from 9.96% at the 0.5% allowance to 34.34% at 5%, and simultaneous 95% paired-bootstrap intervals remain above zero across all 18 reference-by-allowance contrasts. Held-out scientific-risk point estimates remain close to their nom inal allowances, although matching simultaneous risk-feasibility inference is unavailable because the retained evaluation summaries do not contain paired per-trial Law and Full scientific losses. These results support finite-sample and within-regime reference-fit repeatability; they do not establish transfer to an unseen reference family, arbitrary reference misspecification, or independent physical endpoint datasets.

Protocol and numerical evaluation. The truth bank contains 50,000 particles. Each of the three endpoint-trained flow-matching references uses four width-128 hidden layers, 12,000 Adam steps, and 32,768 rollout particles. Moment reconstruction uses endpoint-anchored bounded cubic penalized least squares with three internal knots, smoothing $1 0 ^ { - 4 }$ , and relative ridge $1 0 ^ { - 1 0 }$ . Scientific risk averages four MMD bandwidths across the three references; the frozen Law baseline is 0.0386 and the population-risk ceiling is 0.0634. The shared selection bank contains 128 trials, and the final numerical Full evaluator uses a 256 × 128 raster at all 21 scientific times. After the six-point frontier is frozen, evaluation uses 64 shared fresh trials per reference. Simultaneous 95% action intervals use 100,000 paired bootstrap resamples over all 18 reference-by-allowance contrasts.

## D.1.2 FINITE-MEASURE TRANSPORT–REACTION EXPERIMENT

We consider an unbalanced Gaussian transport problem in which the evolving object is a finite measure rather than a probability law. FIDE acts on the normalized shape through the same informationprojection construction used in the balanced experiments, while discrepancies in total mass are represented through the reaction channel of the screened transport–reaction correction. As in the balanced setting, Law provides the scientific-risk reference design and Tangent optimizes only the correction visible through the measured raw moment rates. Full instead optimizes the unrestricted transport–reaction correction in equation 32. We additionally introduce GlobalReaction as an intermediate structural comparator. It retains an unrestricted transport correction but restricts the reaction field to a single spatially constant rate at each time:

$$
a _ { \mathrm { G R } } ^ { \mathrm { u b } } ( t , \eta ) : = M _ { t } \operatorname* { i n f } _ { \delta , a \in \mathbb { R } } \left\{ \mathbb { E } _ { q _ { t } } \big [ \| \delta \| ^ { 2 } + \kappa a ^ { 2 } \big ] : - \nabla \cdot ( q _ { t } \delta ) + q _ { t } a = q _ { t } h _ { \mathrm { u b } } \right\} .\tag{252}
$$

Integrating the constraint in equation 252 over space fixes $a = \mathbb { E } _ { q _ { t } } [ h _ { \mathrm { u b } } ] = \dot { M } _ { t } / M _ { t } - g _ { \mathrm { r e f } } ( t )$ GlobalReaction therefore permits reaction to correct the global mass-rate discrepancy, while any spatially heterogeneous mean-zero forcing must be realized through transport. Full removes this restriction by allowing the optimal reaction field to vary over space, whereas Tangent imposes the distinct restriction of matching only the measured raw moment rates.

<table><tr><td colspan="5">Mean Full action</td><td colspan="3">Reduction vs. Law (%)</td><td rowspan="2"></td></tr><tr><td>Allowance</td><td>Law</td><td>Tangent</td><td>GlobalReaction</td><td>Full</td><td>Tangent</td><td>GlobalReaction</td><td>Full Full interval (%)</td></tr><tr><td>1%</td><td>0.77523</td><td>0.76973</td><td>0.76652</td><td>0.74994</td><td>0.71</td><td>1.12</td><td>3.26</td><td>[1.50, 5.03]</td></tr><tr><td>2%</td><td>0.77523</td><td>0.76502</td><td>0.76205</td><td>0.73475</td><td>1.32</td><td>1.70</td><td>5.22</td><td>[3.46, 6.99]</td></tr><tr><td>3%</td><td>0.77523</td><td>0.75927</td><td>0.75660</td><td>0.72193</td><td>2.06</td><td>2.40</td><td>6.88</td><td>[5.11, 8.64]</td></tr><tr><td>4%</td><td>0.77523</td><td>0.75296</td><td>0.75043</td><td>0.71029</td><td>2.87</td><td>3.20</td><td>8.38</td><td>[6.61, 10.14]</td></tr><tr><td>5%</td><td>0.77523</td><td>0.74635</td><td>0.74397</td><td>0.69936</td><td>3.72</td><td>4.03</td><td>9.79</td><td>[8.02, 11.55]</td></tr></table>

Table 3: Fresh finite-measure comparison at $\kappa = 1$ . Each frozen selector is evaluated using the same numerical Full-action functional on 128 paired validation trials. Reductions are computed relative to the Law action at the same κ. Tangent and GlobalReaction are descriptive structural comparators; the final column reports the paired interval for the prespecified Full-versus-Law action reduction.

The primary experiment fixes κ = 1 before validation and evaluates the frozen Law, Tangent, GlobalReaction, and Full geometries on the same 128 fresh paired trials across the five prespecified risk allowances. Table 3 cross-evaluates every selected geometry using the common Full-action functional. Tangent and GlobalReaction both reduce the common Full cross-score relative to Law, while the Full-selected geometry yields the lowest Full cross-score at every operating point. The Fullversus-Law reduction increases from 3.26% at the 1% allowance to 9.79% at the 5% allowance, with positive lower endpoints of the paired action intervals throughout. At the 5% operating point, the Tangent- and GlobalReaction-selected geometries reduce the same Full cross-score by 3.72% and 4.03%, respectively, compared with 9.79% for the Full-selected geometry. These cross-scores quantify the unrestricted transport–reaction robustness consequence of the different structural restrictions; they do not rank Tangent or GlobalReaction on their native objectives. Tangent and GlobalReaction are used here as descriptive structural comparators, and the reported action inference is for the prespecified Full-versus-Law contrast.

The mechanism ablations below examine the structural restrictions imposed by Tangent and Global-Reaction. Tangent retains only correction visible through the measured raw moment rates, whereas GlobalReaction allows unrestricted transport but restricts reaction to the global mass-rate discrepancy. Their common-Full cross-scores exceed those of the Full-selected geometries in Table 3. Section D.2.11 examines the corresponding Move–Reaction and Tangent–Hidden decompositions directly.

## D.2 ABLATIONS

We begin with two method-level ablations. The first examines FIDE’s separation of completion from scoring by comparing independent per-time information projection with a restricted path-level completion at exactly fixed measured moments. The second asks when the baseline-preserving preference itself has a truth-relative downstream benefit: a controlled product-law construction makes the reference exactly correct in an unprotected factor, evaluates a held-out local-occupancy loss that is independent of Full’s Sobolev geometry, and then deliberately violates baseline correctness. The remaining studies are benchmark-specific. The Gaussian common-pool analysis compares all secondary criteria on the same promoted candidates and separately perturbs the learned reference; held-out risk analyses assess scientific-risk transfer; prospective studies test the numerical evaluator and aggregate predictive interface; a manufactured solution tests continuum convergence of the periodic Full evaluator; skyrmion analyses examine richer potential spaces and support sensitivity; and the finite-measure analyses examine the transport–reaction extension. Unless stated otherwise, the relevant sensor geometries remain frozen throughout these diagnostics.

## D.2.1 STATIC INFORMATION PROJECTION VERSUS A RESTRICTED PATH-LEVEL COMPLETION

FIDE deliberately separates completion from scoring: each moment fiber is first completed by information projection, after which Full measures the dynamical revision required by the resulting path. A natural alternative is to let unresolved law directions vary jointly over time so as to reduce Full itself. These constructions answer different questions, but the distinction raises a quantitative issue: how much of the Full action assigned to an information-projected path is attributable to the measured moment trajectory, and how much can be changed by choosing a different completion of the same fibers? We examine this in a controlled one-dimensional example in which the measured trajectory is held exactly fixed while one otherwise unmeasured law direction is allowed to adapt dynamically.

Let the state space be the flat torus $\mathbb { T } = \mathbb { R } / ( 2 \pi \mathbb { Z } )$ , with uniform frozen reference $\widetilde { q } = ( 2 \pi ) ^ { - 1 }$ and reference velocity $u = 0$ . The measured observable is $\phi ( x ) = \cos x ,$ and we prescribe

$$
c _ { n } ( t ) = 0 . 4 \sin ( 2 \pi n t ) , \qquad n \in \{ 1 , 2 , 4 \} .\tag{253}
$$

The standard FIDE completion is the pointwise information projection

$$
q _ { t } ^ { \mathrm { I P } } ( x ) = \frac { \exp ( \lambda _ { t } \cos x ) } { \int _ { \mathbb { T } } \exp ( \lambda _ { t } \cos y ) d y } , \qquad \mathbb { E } _ { q _ { t } ^ { \mathrm { I P } } } [ \cos X ] = c _ { n } ( t ) .\tag{254}
$$

To construct a restricted path-level alternative, we enlarge the exponential family by one unmeasured Fourier coordinate,

$$
q _ { t } ^ { \mathrm { d y n } } ( x ) = \frac { \exp ( \lambda _ { t } \cos x + \gamma _ { t } \cos 2 x ) } { \int _ { \mathbb { T } } \exp ( \lambda _ { t } \cos y + \gamma _ { t } \cos 2 y ) d y } , \qquad \mathbb { E } _ { q _ { t } ^ { \mathrm { d y n } } } [ \cos X ] = c _ { n } ( t ) ,\tag{255}
$$

and parameterize $\begin{array} { r } { \gamma _ { t } = \sum _ { k = 1 } ^ { 5 } a _ { k } \sin ( k \pi t ) } \end{array}$ . Thus $\gamma _ { 0 } = \gamma _ { 1 } = 0$ , and both completions have the same measured moment trajectory and the same endpoint densities. For every candidate $\gamma _ { t }$ , the multiplier $\lambda _ { t }$ is recalibrated so that equation 253 holds exactly. We then optimize $a _ { 1 } , \dotsc , a _ { 5 } \in [ - 2 , 2 ]$ to minimize the time-integrated Full action of this enlarged path. The static information projection is the feasible special case $a _ { 1 } = \cdots = a _ { 5 } = 0$ , so this ablation changes only the completion of the unmeasured law direction; it does not change the observed moment trajectory.

Because $u = 0$ , differentiation of the calibrated exponential family gives

$$
\begin{array} { r } { \frac { \partial _ { t } q _ { t } } { q _ { t } } = \dot { \lambda } _ { t } \big ( \cos \ b x - c _ { n } ( t ) \big ) + \dot { \gamma } _ { t } \big ( \cos 2 \ b x - \mathbb { E } _ { q _ { t } } [ \cos 2 \ b X ] \big ) , \qquad \dot { \lambda } _ { t } = \frac { \dot { c } _ { n } ( t ) - \operatorname { C o v } _ { q _ { t } } \big ( \cos \ b X , \cos 2 \ b X \big ) \dot { \gamma } _ { t } } { \operatorname { V a r } _ { q _ { t } } \big ( \cos \ b X \big ) } . } \end{array}\tag{256}
$$

For a periodic one-dimensional density path, let $\begin{array} { r } { F _ { t } ( x ) = \int _ { 0 } ^ { x } \partial _ { t } q _ { t } ( y ) d y } \end{array}$ . The minimum-energy continuity flux has $q _ { t } ( x ) \delta _ { t } ( x ) = b _ { t } - F _ { t } ( x )$ , where

$$
b _ { t } = \frac { \int _ { \mathbb { T } } F _ { t } ( x ) / q _ { t } ( x ) d x } { \int _ { \mathbb { T } } 1 / q _ { t } ( x ) d x } , \qquad a _ { \mathrm { f u l l } } ( t ) = \int _ { \mathbb { T } } \frac { ( b _ { t } - F _ { t } ( x ) ) ^ { 2 } } { q _ { t } ( x ) } d x .\tag{257}
$$

This gives the Full action for both paths without finite-differencing the density in physical time.

The measured trajectory itself imposes a nonzero action independently of the completion rule. Indeed,

$$
\dot { c } _ { n } ( t ) = \mathbb { E } _ { q _ { t } } [ - \sin ( X ) \delta _ { t } ( X ) ] ,
$$

so Cauchy–Schwarz and $\mathbb { E } [ \sin ^ { 2 } X ] \leq 1 - c _ { n } ( t ) ^ { 2 }$ imply

$$
a _ { \mathrm { f u l l } } ( t ) \geq \frac { \dot { c } _ { n } ( t ) ^ { 2 } } { 1 - c _ { n } ( t ) ^ { 2 } } .\tag{258}
$$

We use the integral of equation 258 only as a universal moment-imposed lower bound; it need not be attainable by a feasible law path.

The completion choice is therefore consequential but does not account for all of the dynamical cost. $\mathbf { A } \mathbf { t } { \boldsymbol { n } } = 1$ , allowing the additional unmeasured mode reduces action from 6.52 to 5.89, a 9.67% reduction, while both values remain well above the universal moment-imposed lower bound of 3.30. The reduction grows to 20.00% and 26.15% over the two more rapidly varying trajectories tested. This shows that part of the Full action assigned to the static information-projected path depends on how the unresolved fiber directions are completed rather than being forced solely by the measured moment trajectory.

This result does not make the path-level construction a replacement for the FIDE completion rule. The restricted dynamic path is optimized using the same Full criterion by which it is subsequently evaluated, whereas FIDE intentionally fixes the completion independently of Full and then asks how much dynamical revision that prescribed completion requires. The two procedures therefore encode different modeling choices: the former uses unmeasured law directions to reduce dynamical action, while the latter uses a common local minimum-information-revision rule to resolve those directions before dynamical scoring. The ablation quantifies the price of that separation in this toy family rather than establishing that one completion is scientifically superior.

<table><tr><td>Frequency n</td><td>Moment lower bound</td><td>Static I-projection</td><td>Restricted dynamic</td><td>Reduction (%)</td></tr><tr><td>1</td><td>3.295850</td><td>6.521546</td><td>5.890663</td><td>9.67</td></tr><tr><td>2</td><td>13.183401</td><td>26.086185</td><td>20.867915</td><td>20.00</td></tr><tr><td>4</td><td>52.733603</td><td>104.344741</td><td>77.063457</td><td>26.15</td></tr></table>

Table 4: Effect of the completion rule at fixed measured moments. The static column evaluates Full on the independent per-time information projections in equation 254. The restricted dynamic column optimizes one additional unmeasured Fourier coordinate over time while preserving exactly the same trajectory $c _ { n } ( t )$ . “Reduction” is $1 0 0 ( \mathcal { A } _ { \mathrm { I P } } - \mathcal { A } _ { \mathrm { d y n } } ) / \mathcal { A } _ { \mathrm { I P } }$ . The moment lower bound is the time integral of equation 258 and is not asserted to be attainable.

The numerical calculation uses 512 periodic spatial points and 121 physical-time nodes. Moment calibration, mass compatibility, and differentiated-moment compatibility hold to below $6 \times 1 0 ^ { - 1 5 }$ in the reported runs, and setting all dynamic coefficients to zero reproduces the independently evaluated static action to machine precision. Re-evaluation of the frozen solutions on successively finer spatial and temporal grids changes the reported completion gaps by less than 0.006 percentage points. For $n = 2$ and $n = 4$ , the first coefficient reaches the prespecified lower bound −2; the increase in relative gap across the three tested frequencies should therefore be read descriptively rather than as a converged or universal frequency law. The dynamic construction is restricted to the family in equation 255 and is not claimed to attain the globally minimum-action path over all laws satisfying the moment constraints.

## D.2.2 WHEN DOES BASELINE PRESERVATION HELP? A CONTROLLED DOWNSTREAM ABLATION

The baseline-preserving specification is useful only if preserving the reference in directions left outside the scientific task has a meaningful consequence. We therefore construct a population-level ablation in which this assumption is exactly true and then violate it in a controlled way. The purpose is not to show that Full recovers truth in general (Proposition 1 rules out such a claim), but to ask two narrower questions: when the reference is correct in an unprotected part of the state, which secondary selectors protect a downstream quantity that is independent of their native objectives, and how do those consequences change as the same unprotected direction becomes genuinely misspecified?

A baseline-correct residual design problem. Let $( Y , Z ) \in \mathbb { T } ^ { 2 }$ with uniform stationary reference $\widetilde { q } = ( 2 \pi ) ^ { - 2 }$ and $u = 0$ , and write $e _ { k } ( x ) = { \sqrt { 2 } } \cos ( k x )$ . At baseline correctness the true population is

$$
p _ { t , 0 } ( y , z ) = \frac { 1 + \epsilon s ( t ) e _ { 1 } ( y ) } { ( 2 \pi ) ^ { 2 } } , \qquad s ( t ) = \sin ( 2 \pi t ) ,\tag{259}
$$

so the scientific signal evolves only in $Y$ while the unprotected Z marginal is exactly the stationary reference. Candidate sensors deliberately mix the protected signal with different unresolved Z directions:

$$
\phi _ { \alpha } ( y , z ) = e _ { 1 } ( y ) + \psi _ { \alpha } ( z ) , \qquad \psi _ { \alpha } ( z ) = \sum _ { k \in \{ 1 , 3 , 1 0 \} } \sqrt { w _ { \alpha , k } } e _ { k } ( z ) , \qquad \alpha \in \{ 0 , \frac { 1 } { 4 } , \frac { 1 } { 2 } , \frac { 3 } { 4 } , 1 \} ,\tag{260}
$$

where $w _ { \alpha } = ( 1 - \alpha ) ( 0 , 9 0 / 9 1 , 1 / 9 1 ) + \alpha ( 1 0 / 1 1 , 0 , 1 / 1 1 )$ . Every $\psi _ { \alpha }$ has the same $L ^ { 2 }$ energy and the same gradient energy,

$$
\sum _ { k } w _ { \alpha , k } = 1 , \qquad \sum _ { k } k ^ { 2 } w _ { \alpha , k } = 1 0 ,
$$

but different inverse-transport energy

$$
H _ { \alpha } : = \sum _ { k \in \{ 1 , 3 , 1 0 \} } { \frac { w _ { \alpha , k } } { k ^ { 2 } } } \in \{ 0 . 1 1 , 0 . 3 1 , 0 . 5 1 , 0 . 7 1 , 0 . 9 1 \} .\tag{261}
$$

$\mathbf { A } \mathbf { t } \gamma = 0$ , every sensor has the same target moment E ${ \bf \nabla } _ { P _ { t } } [ \phi _ { \alpha } ] = \epsilon s ( t )$ and exactly the same integrated sampling risk, $\mathrm { \Delta 2 - \epsilon ^ { 2 } / 2 }$ . Scientific risk therefore leaves all five designs tied by construction; the ablation isolates only how a secondary criterion resolves that residual freedom.

The information projection factorizes as $Q _ { t } ^ { \alpha } = Q _ { Y , t } ^ { \alpha } \otimes Q _ { Z , t } ^ { \alpha }$ . For $C _ { \epsilon } : = 2 \pi ^ { 2 } \epsilon ^ { 2 }$ , the small-amplitude expansion gives

$$
\mathcal { A } ( \alpha ) = \frac { C _ { \epsilon } } { 4 } ( 1 + H _ { \alpha } ) + o ( \epsilon ^ { 2 } ) .\tag{262}
$$

To expose the corresponding hidden mechanism directly, we also evaluate a fixed-reference Sobolev rate error on the unprotected factor,

$$
D _ { Z } ( \alpha ) : = \int _ { 0 } ^ { 1 } \sum _ { m > 1 } \frac { | \dot { m } _ { m , c } ^ { \alpha } ( t ) | ^ { 2 } + | \dot { m } _ { m , s } ^ { \alpha } ( t ) | ^ { 2 } } { m ^ { 2 } } d t , \qquad D _ { Z } ( \alpha ) = \frac { C _ { \epsilon } } { 4 } H _ { \alpha } + o ( \epsilon ^ { 2 } ) ,\tag{263}
$$

where $m _ { m , c } ^ { \alpha }$ and $m _ { m , s } ^ { \alpha }$ are the cosine and sine moments of $Q _ { Z , t } ^ { \alpha }$ . Because the true $Z$ marginal is stationary at baseline correctness, these are truth-relative spurious rates. Full and $D _ { Z }$ therefore have the same predicted strict ordering, with $\alpha = 0$ preferred. Tangent, residual $L ^ { 2 }$ , and forward $\mathrm { K L }$ are independent of $\alpha$ at leading order because the corresponding $\dot { L } ^ { 2 }$ and $H ^ { 1 }$ quantities were matched; static $W _ { 2 }$ shares the local inverse-transport ordering and is therefore expected to agree with Full in this construction.

The numerical experiment uses the prespecified amplitudes $\epsilon \in \{ 0 . 1 0 , 0 . 0 5 , 0 . 0 2 5 \}$ without parameter search. At every amplitude, Full and $D _ { Z }$ are strictly increasing in α and have Spearman rank correlation one. The maximum relative errors of the leading Full and $D _ { Z }$ expansions are respectively $2 . 8 8 \times 1 0 ^ { - 4 }$ and $9 . 8 3 \times 1 0 ^ { - 4 } \mathrm { ~ a t ~ } \epsilon = 0 . 1 0$ , and decrease to $1 . 8 \times 1 0 ^ { - 5 }$ and $6 . 1 4 \times 1 0 ^ { - 5 }$ at $\epsilon = 0 . 0 2 5$ . Tangent, residual $L ^ { 2 }$ , and KL have small finite-amplitude preferences for $\alpha = 0 . 7 5$ , but their relative score ranges shrink with $\epsilon ,$ consistently with their leading-order indifference. Static $W _ { 2 }$ selects the same $\alpha = 0$ geometry as Full.

Held-out local occupancy: an independent downstream loss. The preceding $D _ { Z }$ diagnostic is intentionally aligned with the inverse-transport mechanism underlying Full, so it is not by itself an independent downstream test. We therefore freeze the five sensors and evaluate ordinary local population predictions that never enter scientific risk, information projection, or any secondary criterion. For 16 evenly spaced centers $\theta _ { j } = 2 \pi j / 1 6$ and three prespecified localization scales $\kappa \in \{ 2 , 4 , 8 \}$ define

$$
g _ { \kappa , j } ( z ) = \exp \bigl ( \kappa [ \cos ( z - \theta _ { j } ) - 1 ] \bigr ) ,\tag{264}
$$

and measure

$$
\mathrm { O c c R M S E } _ { \kappa } ( \alpha ) : = \left[ \frac { 1 } { 1 6 } \sum _ { j = 0 } ^ { 1 5 } \int _ { 0 } ^ { 1 } \left( \mathbb { E } _ { Q _ { Z , t } ^ { \alpha } } [ g _ { \kappa , j } ] - \mathbb { E } _ { P _ { Z , t } } [ g _ { \kappa , j } ] \right) ^ { 2 } d t \right] ^ { 1 / 2 } .\tag{265}
$$

This loss is evaluated directly against the known true $Z$ marginal. It contains no referencerelative rate discrepancy, $H ^ { - 1 }$ norm, inverse Laplacian, correction energy, gradient normalization, or design-dependent weighting.

At baseline correctness, Full and static $W _ { 2 }$ select $\alpha = 0 ,$ , while Tangent, residual $L ^ { 2 }$ , and forward KL select $\alpha = 0 . 7 5$ in the finite-amplitude calculation. Law remains exactly indifferent among all five geometries. For the selector-level comparison, we therefore use an outcome-independent Law policy that draws uniformly from the complete set of five risk minimizers and then freezes the sampled geometry; its expected occupancy RMSE is the arithmetic mean of the five per-geometry RMSEs, not the RMSE of a mixture distribution and not a tie broken using the downstream loss. At $\epsilon = 0 . 1 0$ and $\kappa = 4$ , the resulting RMSEs are $2 . 1 4 9 2 \times 1 0 ^ { - 3 }$ for Full/static $W _ { 2 } , 4 . 3 1 4 7 \times 1 0 ^ { - 3 }$ for the expected Law policy, and $5 . 3 2 9 6 \times 1 0 ^ { - 3 }$ for Tangent/residual $L ^ { 2 } .$ /forward KL, corresponding to differences $0 \mathrm { f - 5 0 . 1 9 \% }$ and $+ 2 3 . 5 2 \%$ relative to expected Law. Across all three tested amplitudes and all three localization scales, the correct-baseline ordering remains Full/static $W _ { 2 } <$ expected Law < Tangent/residual $L ^ { 2 } / \mathrm { f o r w a r d ~ K I }$ . Figure 3 retains the more detailed candidate-family view: occupancy error increases monotonically across the five frozen geometries, with $\alpha = 0 < 0 . 2 5 <$ $0 . 5 \dot { < } 0 . \dot { 7 } 5 < 1$ at all three localization scales.

![](images/63734b60bd134fabeba50b3773e60bc92aca99ee57aa64be7ec560db3c122f67.jpg)  
Figure 3: Candidate-family occupancy error under baseline correctness. $\mathrm { ~ A t ~ } \epsilon ~ = ~ 0 . 1 0$ and $\gamma = 0$ , truth-relative local-occupancy RMSE is shown for all five frozen geometries and all three prespecified localization scales $\kappa \in \{ 2 , 4 , 8 \}$ . Scientific risk is identical across the five designs. Full and static $W _ { 2 }$ select $\alpha = 0$ , Tangent, residual $L ^ { 2 } .$ , and forward KL select $\alpha = 0 . 7 5$ , and Law is indifferent among all five geometries. The occupancy loss is withheld from design and is not used by any selector.

The occupancy result is also predicted independently of Full. Writing $I _ { k }$ for the modified Bessel function,

$$
\mathrm { O c c M S E } _ { \kappa } ( \alpha ) = \frac { \epsilon ^ { 2 } e ^ { - 2 \kappa } } { 8 } \sum _ { k \in \{ 1 , 3 , 1 0 \} } w _ { \alpha , k } I _ { k } ( \kappa ) ^ { 2 } + o ( \epsilon ^ { 2 } ) .\tag{266}
$$

This is a different spectral weighting from $\begin{array} { r } { H _ { \alpha } = \sum _ { k } w _ { \alpha , k } / k ^ { 2 } } \end{array}$ . The predicted squared-error ratios between $\alpha = 1$ and $\alpha = 0$ are 51.3866, 7.8609, and 2.6372 for $\kappa = 2 , 4 , 8 ; \mathrm { a t } \epsilon = 0 . 0 2 5$ the numerical ratios are 51.3868, 7.8610, and 2.6373. Thus the downstream agreement is not obtained by evaluating the Full objective under another name.

Deliberately violating baseline correctness. We next add a genuine unprotected departure to the true $Z$ marginal while leaving the reference and sensor family unchanged:

$$
p _ { t , \gamma } ( y , z ) = \frac { 1 + \epsilon s ( t ) e _ { 1 } ( y ) + \gamma \epsilon r ( t ) e _ { 1 } ( z ) } { ( 2 \pi ) ^ { 2 } } , \qquad r ( t ) = \sin ( 4 \pi t ) .\tag{267}
$$

The $\alpha = 0$ sensor has no $k = 1$ component in $Z$ and is therefore blind to this new departure, whereas $\alpha = 1$ is strongly sensitive to it. No design is reselected as γ changes. The local expansion predicts that the $\alpha = 0$ versus $\alpha = 1$ ordering in $D _ { Z }$ reverses at $\gamma _ { \star } = \sqrt { 2 2 / 3 0 9 } \approx 0 . 2 6 6 8$ ; the numerical crossover estimates converge to this value as ϵ decreases.

The independently defined occupancy loss has its own, later, scale-dependent reversal. Its smallamplitude expansion is

$$
\mathrm { O c c M S E } _ { \kappa } ( w ; \gamma ) = \epsilon ^ { 2 } e ^ { - 2 \kappa } \left[ \frac { A _ { \kappa } ( w ) } { 8 } + \frac { \gamma ^ { 2 } B _ { \kappa } ( w ) } { 2 } \right] + o ( \epsilon ^ { 2 } ) ,\tag{268}
$$

with $\begin{array} { r } { A _ { \kappa } ( w ) ~ = ~ \sum _ { k \in \{ 1 , 3 , 1 0 \} } w _ { k } I _ { k } ( \kappa ) ^ { 2 } } \end{array}$ and $B _ { \kappa } ( w ) ~ = ~ I _ { 1 } ( \kappa ) ^ { 2 } ( w _ { 1 } / 2 - 1 ) ^ { 2 } + { \textstyle \frac { w _ { 1 } } { 4 } } \{ w _ { 3 } I _ { 3 } ( \kappa ) ^ { 2 } + \kappa \llap / v _ { 1 } \}$ $w _ { 1 0 } I _ { 1 0 } ( \kappa ) ^ { 2 } \}$ . This gives precomputed $\alpha = 0$ versus $\alpha = 1$ crossover predictions 0.5632, 0.5314, and 0.4482 for $\kappa = 2 , 4 , 8$ Using only the prespecified $\gamma \ { \mathrm { g r i d } } .$ , the corresponding estimates at $\epsilon = 0 . 0 2 5$ are 0.56324, 0.53139, and 0.44817. Figure 4 shows the complete candidate-family sweep at $\epsilon = 0 . 1 0 \colon$ the baseline-preserving $\alpha = 0$ geometry is best near $\gamma = 0$ , but its advantage erodes as the unprotected departure grows. Other frozen geometries eventually overtake it, and the $\alpha = 0$ versus $\alpha = 1$ ordering reverses at the separately predicted, scale-dependent thresholds.

![](images/52ef0b32831300a919f5807028f36d80b73821a269f3571881fe5b20336730b5.jpg)

![](images/8504f275008c9ef7ffa7d798d84bacee5fe0dbdc0f9f02f487d84d8c51af0528.jpg)

![](images/3fd4c774b5f4cc9743073d1000d24357b4a93c0b0e36f1923816a72fcb5df657.jpg)  
Figure 4: Candidate-family occupancy error under misspecification. ${ \mathrm { A t ~ } } \epsilon \ = \ 0 . 1 0 .$ , truthrelative local-occupancy RMSE is shown across all five frozen geometries as the unprotected departure strength γ increases. Blue denotes $\alpha \ = \ 0 .$ , orange denotes $\alpha \ = \ 1$ , and gray denotes $\alpha \in \{ 0 . 2 5 , 0 . 5 , 0 . 7 5 \}$ . The baseline-preserving $\alpha = 0$ geometry is best near $\gamma = 0$ but eventually becomes worse than more departure-sensitive geometries. Dashed vertical lines mark the independently predicted small-amplitude $\alpha = 0$ versus $\alpha = 1$ occupancy crossovers from equation 268: $\gamma _ { \star } = 0 . 5 6 3 2 , 0 . 5 3 1 4 ,$ and 0.4482 for $\kappa = 2 , 4 , 8 ,$ respectively. No geometry is reselected as γ varies; Figure 1 gives the selector-level comparison emphasized in the main text.

These occupancy crossovers are substantially different from the $D _ { Z }$ crossover, confirming that the held-out occupancy loss probes a different downstream geometry. The main-text selector-level comparison in Figure 1 freezes the actual finite-amplitude selections, namely $\alpha = 0$ for Full/static $W _ { 2 }$ and $\alpha = 0 . 7 5$ for Tangent/residual L<sup>2</sup>/forward KL, and compares each with the expected Law tiebreaking policy. $\mathrm { A t } \epsilon = 0 . 1 0$ and $\kappa = 4$ , both differences change sign between the evaluated points $\gamma = 0 . 5$ and 0.6; linear interpolation of the plotted segments gives 0.5156 for Full/static $\bar { W _ { 2 } }$ and 0.5160 for Tangent/residual L<sup>2</sup>/forward KL. These interpolants are descriptive crossing locations rather than separately evaluated roots.

The same reversal directions hold for all nine combinations of $\epsilon \in \lbrace 0 . 1 0 , 0 . 0 5 , 0 . 0 2 5 \rbrace$ and $\kappa \in$ $\{ 2 , 4 , 8 \}$ : Full/static $W _ { 2 }$ change from lower to higher occupancy error than expected Law, while Tangent/residual L<sup>2</sup>/forward KL change from higher to lower error. The crossing bracket is [0.5, 0.6] for $\kappa \in \{ 2 , 4 \}$ and $[ 0 . 4 , 0 . 5 ]$ for $\kappa = 8$ at every tested amplitude.

The main-text selector-level summary in Figure 1, together with the detailed candidate-family views in Figures 3–4, isolates the scope condition behind baseline-preserving residual design. When the task leaves a direction unprotected and the declared baseline is correct there, the geometry selected by Full and static $W _ { 2 }$ preserves that structure and yields the smallest occupancy error among the selector groups considered here. As a genuine departure in the same unprotected direction grows, that advantage erodes and eventually reverses, while the geometry selected by Tangent, residual $L ^ { 2 }$ , and forward KL exhibits the opposite change relative to the Law tie policy. No selector is therefore uniformly favored by this truth-relative downstream loss across baseline regimes. The result supports a conditional role for baseline preservation rather than a robustness guarantee under misspecification, and it does not establish a Full-specific advantage over static $W _ { 2 }$ in this construction.

No parameter search is used. The three amplitudes, five sensors, three occupancy scales, and misspecification grid are fixed before evaluation. Across all nine amplitude–localization-scale combinations, the correct-baseline selector ordering Full/static $W _ { 2 } <$ expected Law $<$ Tangent/residual L<sup>2</sup>/forward KL and the corresponding reversal directions under increasing misspecification are unchanged, although effect magnitudes and crossing locations depend on localization scale. All candidate-family rankings also persist over nine spatial–temporal refinements; the largest relative occupancy-metric change is $3 . \dot { 1 } \times 1 0 ^ { - 7 }$ , direct quadrature and Fourier/Bessel evaluations agree within $1 0 ^ { \dot { - } 1 2 }$ , and the original baseline-ablation outputs are unchanged.

## D.2.3 DO THE HELD-OUT DESIGNS SATISFY THEIR SCIENTIFIC-RISK ALLOWANCES UNDER SAMPLING UNCERTAINTY?

Selection feasibility does not imply that the same relative risk allowance transfers to fresh observations. For studies whose current validation artifacts retain paired per-trial scientific losses, we assess

![](images/97d5c4c69b8189bee76590b0996792ec7c8303b0f7534bb68554249075481e60.jpg)  
Figure 5: Do the held-out designs satisfy their scientific-risk allowances under sampling uncertainty? Descriptive held-out relative risk increase divided by the nominal allowance for the Gaussian-mixture, prospective double-gyre, and skyrmion validation studies. Gaussian and prospective double-gyre use their respective 64-trial independent holdouts, while skyrmion uses 32 fresh paired validation trials. The dashed line at one marks equality with the nominal allowance. Filled markers denote operating points whose simultaneous one-sided $9 5 \%$ upper bound for $\mathbb { E } [ Z _ { i } ( \boldsymbol { p } ) ]$ is below zero; open markers are unresolved. The plotted ratio is descriptive: inferential status is determined by the paired excess-loss statistic in equation 269, not by the ratio alone.

the held-out constraint directly. For paired evaluation losses, define

$$
Z _ { i } ( p ) = \ell _ { i } ( \eta _ { \mathrm { F u l l } , p } ) - \left( 1 + \frac { p } { 1 0 0 } \right) \ell _ { i } ( \eta _ { \mathrm { L a w } } ) ,\tag{269}
$$

so that $\mathbb { E } [ Z _ { i } ( p ) ] ~ < ~ 0$ corresponds to satisfaction of the $p \%$ relative allowance on the evaluation distribution. For each benchmark family, we use 100,000 paired bootstrap resamples with common resampled trial indices across allowances. Writing $\bar { Z } _ { p }$ for the observed paired mean and $\bar { Z } _ { p } ^ { * }$ for its bootstrap replicate, the pointwise one-sided upper bound is $\bar { Z } _ { p } + q _ { 0 . 9 5 } \bigl ( \bar { Z } _ { p } ^ { * } - \bar { Z } _ { p } \bigr )$ , while the simultaneous bound replaces the scalar bootstrap deviation by its maximum over all allowances in that family. An operating point is supported below its held-out cap only when this simultaneous upper bound is negative; otherwise we report it as unresolved. Multiplicity control is applied separately within each benchmark family. The current paired-loss analysis covers the Gaussian, prospective double-gyre, and skyrmion validation studies. The current retrospective six-allowance frontier is treated separately because its retained evaluation summaries do not contain the paired per-trial Law and Full scientific-risk losses required for a matching bootstrap analysis.

Table 5 sharpens the held-out scientific-risk assessment for the Gaussian, prospective double-gyre, and skyrmion validation studies. In the Gaussian study, the 0.5% operating point is unresolved under simultaneous one-sided inference, whereas the $1 \% - 5 \%$ points are supported below their Lawrelative held-out caps. In the prospective double-gyre study, the 1% point likewise remains unresolved, with $\bar { Z } = \bar { 7 } . 3 6 \times 1 0 ^ { - \bar { 4 } }$ and simultaneous upper bound $8 . 4 0 \times 1 0 ^ { - 3 }$ , whereas the $2 \% - 5 \%$ operating points are supported below their held-out caps. All five fresh skyrmion operating points are supported below their corresponding caps. The current retrospective six-allowance frontier is not included in this bootstrap analysis because the retained evaluation summaries do not contain the paired per-trial scientific-risk losses required to construct matching excess-loss statistics; its verified holdout point estimates are reported in Appendix D.1.1.

The concentration of uncertainty at the tightest allowance is expected from the nested feasibility thresholds: for a fixed held-out loss pair, increasing $p$ shifts $Z _ { i } ( p )$ downward by $( p / 1 0 0 ) \ell _ { i } ( \eta _ { \mathrm { L a w } } )$ without comparably increasing its sampling variability. This is especially transparent for the prospective $2 \% - 4 \%$ points, which use the same frozen Full geometry and have the same 1.022% held-out risk increase; the increasingly negative feasibility margin therefore reflects additional tolerance rather than a change in the validated design.

<table><tr><td>Study</td><td>Allowance</td><td>Risk increase (%)</td><td>Simultaneous upper</td><td>Conclusion</td></tr><tr><td>Gaussian</td><td>0.5%</td><td>0.516</td><td> $1 . 6 1 \times 1 0 ^ { - 4 }$ </td><td>Unresolved</td></tr><tr><td>Gaussian</td><td>1%</td><td>0.656</td><td> $- 7 . 5 8 \times 1 0 ^ { - 5 }$ </td><td>Below cap</td></tr><tr><td>Gaussian</td><td>2%</td><td>0.656</td><td> $- 7 . 3 3 \times 1 0 ^ { - 4 }$ </td><td>Below cap</td></tr><tr><td>Gaussian</td><td>3%</td><td>0.656</td><td> $- 1 . 3 9 \times 1 0 ^ { - 3 }$ </td><td>Below cap</td></tr><tr><td>Gaussian</td><td>4%</td><td>0.656</td><td> $- 2 . 0 5 \times 1 0 ^ { - 3 }$ </td><td>Below cap</td></tr><tr><td>Gaussian</td><td>5%</td><td>0.656</td><td> $- 2 . 7 0 \times 1 0 ^ { - 3 }$ </td><td>Below cap</td></tr><tr><td>Prospective</td><td>1%</td><td>1.060</td><td> $8 . 4 0 \times 1 0 ^ { - 3 }$ </td><td>Unresolved</td></tr><tr><td>Prospective</td><td>2%</td><td>1.022</td><td> $- 4 . 3 1 \times 1 0 ^ { - 3 }$ </td><td>Below cap</td></tr><tr><td>Prospective</td><td>3%</td><td>1.022</td><td> $- 1 . 6 6 \times 1 0 ^ { - 2 }$ </td><td>Below cap</td></tr><tr><td>Prospective</td><td>4%</td><td>1.022</td><td> $- 2 . 8 8 \times 1 0 ^ { - 2 }$ </td><td>Below cap</td></tr><tr><td>Prospective</td><td>5%</td><td>1.711</td><td> $- 3 . 2 6 \times 1 0 ^ { - 2 }$ </td><td>Below cap</td></tr><tr><td>Skyrmion</td><td>1%</td><td>0.050</td><td> $- 2 . 6 2 \times 1 0 ^ { - 3 }$ </td><td>Below cap</td></tr><tr><td>Skyrmion</td><td>2%</td><td>0.304</td><td> $- 6 . 5 2 \times 1 0 ^ { - 3 }$ </td><td>Below cap</td></tr><tr><td>Skyrmion</td><td>3%</td><td>0.636</td><td> $- 1 . 0 0 \times 1 0 ^ { - 2 }$ </td><td>Below cap</td></tr><tr><td>Skyrmion</td><td>4%</td><td>0.636</td><td> $- 1 . 5 2 \times 1 0 ^ { - 2 }$ </td><td>Below cap</td></tr><tr><td>Skyrmion</td><td>5%</td><td>1.653</td><td> $- 1 . 5 2 \times 1 0 ^ { - 2 }$ </td><td>Below cap</td></tr></table>

Table 5: Held-out risk-feasibility inference. Relative risk increase is $1 0 0 ( \bar { \ell } _ { \mathrm { F u l l } } / \bar { \ell } _ { \mathrm { L a w } } - 1 )$ . The fourth column gives the simultaneous one-sided 95% upper bound for the paired excess-loss statistic $Z _ { i } ( \boldsymbol { p } )$ within the corresponding benchmark family. A point is labeled below cap only when this bound is negative.

## D.2.4 COMMON-POOL COMPARISON OF GAUSSIAN DESIGN CRITERIA

To complement the criterion-specific optimization results, we compare all six objectives on a shared set of scientifically competitive designs. Let C be the union of the geometrically admissible candidates that reached the final evaluation stage under Law, Tangent, Full, residual $\dot { L } ^ { 2 }$ , forward KL, or static $W _ { 2 } .$ . Across the six selectors and risk allowances, this gives 17 unique promoted candidates. The common-pool Full scores use the 64-trial selection bank, whereas Table 1 evaluates the same Law and Full geometries with the same numerical Full evaluator on an independent 64-trial holdout bank. Lower-fidelity prescreen candidates and raw optimization iterates are excluded so that every design in C has undergone the same final evaluation protocol.

At the 1% allowance, the operational feasibility threshold is

$$
R _ { \mathrm { s e l } } ( \eta ) \leq 1 . 0 1 R _ { \mathrm { L a w } } = 0 . 0 6 6 8 4 7 5 9 ,\tag{270}
$$

together with the population, support, and numerical requirements of the Gaussian experiment. All $1 \bar { 7 }$ candidates in C satisfy these conditions. For each criterion $J ,$ we define its common-pool selection by

$$
\pmb { \eta } _ { J } ^ { \mathcal { C } } \in \operatorname * { a r g m i n } _ { \pmb { \eta } \in \mathcal { C } : \pmb { \eta } } J ( \pmb { \eta } ) .\tag{271}
$$

Table 6 reports these selections and their cross-evaluated Full actions. Figure 6 provides the corresponding geometric view: panel (a) compares scientific risk and Full action across the common pool, while panel (b) locates the same designs in the two-angle sensor space.

Five distinct criterion-specific selections arise from the common pool, with only forward KL and static $W _ { 2 }$ selecting the same candidate. Law and Full recover the geometries from the original comparison, while static $W _ { 2 }$ moves to the candidate selected by forward KL. Under common Full cross-evaluation, the selected actions are 19.7193 for Full, 30.2016 for Law, 38.0653 for Tangent, 36.5877 for residual $L ^ { 2 }$ , and 33.1209 for the shared KL/static- $. W _ { 2 }$ selection. These are selectionbank cross-scores, not values on a common native objective scale, and they should not be identified with the independent held-out scores in Table 1. The scientific risks remain closely matched acros the promoted pool, while both the selected geometries and their Full cross-scores differ substantially.

## D.2.5 CONTROLLED REFERENCE MISSPECIFICATION IN THE GAUSSIAN COMMON POOL

Appendix D.2.2 isolates the baseline-correctness mechanism in a controlled population setting with known truth and a predeclared failure threshold. Here we ask a complementary benchmark-specific question: does the same reference sensitivity appear when the learned Gaussian reference itself is perturbed while the original scientific task and promoted candidate pool are held fixed? Starting from the frozen reference $( \widetilde { Q } _ { t } , u _ { t } )$ , we construct five endpoint-preserving reference conditions by applying a smooth time-dependent rotation with $\gamma \in \{ - \pi \mathrm { { / 6 , - } } \bar { \pi } / 1 2 , 0 , \pi / 1 2 , \pi / 6 \}$ , corresponding $\mathrm { t o - 3 0 ^ { \circ } , - 1 5 ^ { \circ } , 0 , + 1 5 ^ { \circ } , + 3 0 ^ { \circ } }$ . Let $\theta _ { \gamma } ( t ) = \gamma \sin ^ { 2 } ( \pi t )$ and let $R _ { \theta }$ denote the planar rotation matrix. We define

<table><tr><td>Criterion</td><td>Scientific risk</td><td>Own objective</td><td>Full action</td></tr><tr><td>Law</td><td>0.066186</td><td>0.066186</td><td>30.2016</td></tr><tr><td>Tangent</td><td>0.066511</td><td>2.10920</td><td>38.0653</td></tr><tr><td>Full</td><td>0.066586</td><td>19.71935</td><td>19.7193</td></tr><tr><td>Residual  $L ^ { 2 }$ </td><td>0.066478</td><td>52.50258</td><td>36.5877</td></tr><tr><td>Forward KL</td><td>0.066387</td><td>0.235235</td><td>33.1209</td></tr><tr><td>Static W2</td><td>0.066387</td><td>0.134317</td><td>33.1209</td></tr></table>

Table 6: Gaussian criterion-specific selections on a shared promoted candidate pool. Each row gives the minimizer of one criterion over the same set of 17 scientifically feasible designs. The “own objective” column is reported on each criterion’s native scale and is therefore not comparable across rows; the final column cross-evaluates every selected design using the common Full-action functional. Forward KL and static $W _ { 2 }$ select the same candidate, whereas Law, Tangent, Full, and residual $L ^ { 2 }$ select distinct candidates.

![](images/7b0f8d6e00c66c1abb904ec5d0c08f01806e8bd51ff19341a30bbc0abdd7ae70.jpg)  
(a) Scientific risk versus Full action.

![](images/3853af6003ccbc07449be46431da92064fa2964950dd4ef053a7d6ea426023a3.jpg)  
(b) Sensor geometries in the two-angle design space.  
Figure 6: Gaussian criterion-specific selections on the common promoted candidate pool. Gray points denote the remaining promoted candidates and labeled markers denote the six criterionspecific minimizers obtained by reselecting each objective over the shared pool. Left: scientific risks occupy a narrow range while Full action varies substantially across candidate designs; the dashed line marks the frozen $1 \%$ operational risk ceiling. Forward KL and static $W _ { 2 }$ select the same candidate and therefore overlap. Right: the corresponding sensor geometries occupy distinct regions of the design space, again with the forward-KL and static- $. W _ { 2 }$ selections coincident.

$$
\widetilde { Q } _ { t } ^ { \gamma } = ( R _ { \theta _ { \gamma } ( t ) } ) _ { \# } \widetilde { Q } _ { t } , \qquad u _ { t } ^ { \gamma } ( y ) = \dot { \theta } _ { \gamma } ( t ) J y + R _ { \theta _ { \gamma } ( t ) } u _ { t } \big ( R _ { - \theta _ { \gamma } ( t ) } y \big ) , \qquad J = \left( { 0 } _ { 1 } \begin{array} { l l } { - 1 } \\ { 0 } \end{array} \right) .
$$

The transformed law–velocity pair therefore remains dynamically consistent, and because $\theta _ { \gamma } ( 0 ) =$ $\theta _ { \gamma } ( 1 ) = 0$ , every perturbation has the same endpoint reference laws as the original condition.

The purpose is to isolate reference dependence of the secondary selection rather than redefine the scientific task. We therefore keep the original 1% scientific-risk threshold, population screen, and 17 promoted candidates fixed across all five conditions. Every candidate remains admissible and numerically certified. Tangent, residual $L ^ { 2 }$ , forward KL, static $\mathbf { \bar { \boldsymbol { W } } } _ { 2 }$ , and Full re-rank this same pool after recalibration against the perturbed reference; Law remains the original action-independent risk minimizer. $\mathrm { A t } \gamma = 0$ , all six selectors reproduce their original common-pool selections.

![](images/714c514b8c2c13176d74a8f0f024df71655fdc71a29e9f167a27906c8f1cbdb8.jpg)

![](images/3308bfcf85b646e3f8796159074c027e26bab4cc4b9fcfb6230d0445be443512.jpg)  
Figure 7: Controlled Gaussian reference misspecification on the fixed 1% admissible pool. Left: covariance-normalized detectability of the discrepancy between the analytic target and the perturbed reference; larger values mean that the selected sensors expose the reference error more strongly. Right: truth-relative hidden-law sliced-Wasserstein error of the corresponding information-projected completion; smaller values mean closer agreement with the analytic intermediate law. The two diagnostics need not induce the same ordering.

We measure sensitivity to the imposed reference error separately from each selector’s native objective. For candidate η, define

$$
D _ { \mathrm { d e t e c t } } ( \eta , \gamma ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \sum _ { q } \omega _ { q } \Delta \mu _ { b q } ^ { \top } V _ { b q , \eta } ^ { - 1 } \Delta \mu _ { b q } , \qquad \Delta \mu _ { b q } : = \mathbb { E } _ { P _ { b , t _ { q } } } [ \Phi _ { \eta } ] - \mathbb { E } _ { \widetilde { Q } _ { t _ { q } } ^ { \gamma } } [ \Phi _ { \eta } ] ,
$$

where $V _ { b q , \eta }$ is the existing measurement covariance and $\omega _ { q }$ are the scientific-time quadrature weights. Larger $D _ { \mathrm { d e t e c t } }$ means that the candidate measurements distinguish the analytic target from the perturbed reference more strongly. This diagnostic is distinct from truth-relative completion error, which we continue to measure using the original hidden-law sliced Wasserstein metric with 128 fixed directions.

The selectors separate clearly under misspecification. At all four nonzero rotations, the Full-selected geometry has lower $D _ { \mathrm { d e t e c t } }$ than Law and ranks between 14th and 17th among the 17 admissible candidates by detectability. Forward KL and static $W _ { 2 }$ are also less detectable than Law in three of the four conditions, with nonzero-condition ranks between 11th and 17th. Tangent and residual $L ^ { 2 }$ , by contrast, are never less detectable than Law in this sweep and occupy ranks 4–8 and $1 { - } 8 ,$ respectively. Reduced sensitivity to a misspecified reference is therefore neither unique to Full nor a generic consequence of using the reference anywhere in the construction: it appears consistently for Full and often for the static reference-proximity criteria, but not for the two residual-rate criteria in this finite pool.

Detectability and truth fidelity remain separate. Forward KL and static $W _ { 2 }$ have lower hidden-law $\mathrm { S W _ { 2 } ^ { 2 } }$ than Law at all four nonzero perturbations despite often selecting less detectable measurements. Full has higher hidden-law error than Law at the two $\pm 1 5 ^ { \circ }$ conditions and lower error at the two $\pm 3 0 ^ { \circ }$ conditions. Thus lower sensitivity to reference error does not by itself imply worse reconstruction of the hidden law, nor does higher detectability guarantee better truth-relative completion.

Finally, the rank changes should not be read as evidence that increasing misspecification causes Full to progressively move toward “blind” sensor geometries. Full selects the same geometry at $0 ^ { \circ } , \pm 1 5 ^ { \circ }$ , and $- 3 0 ^ { \circ }$ , switching only to a nearby geometry at $+ 3 0 ^ { \circ }$ ; much of the detectability-rank change occurs because the reference being compared against changes while the geometry does not. The ablation therefore supports a narrower conclusion: under the baseline-preserving preference, Full can favor scientifically admissible measurements that expose less of a misspecified baseline. This behavior is a limitation of the declared reference-preserving scope and should not be interpreted as a truth-recovery guarantee. The analysis is post hoc and conditional on this frozen reference, fixed promoted pool, and perturbation family, so it does not establish a universal robustness threshold or behavior under arbitrary reference misspecification.

<table><tr><td>Evaluator</td><td>Law action</td><td>Full action</td><td>Reduction (%)</td><td>Paired Full—Law interval</td></tr><tr><td> $1 2 8 \times 6 4$ </td><td>1.50697</td><td>1.16113</td><td>22.95</td><td>[-0.388, -0.307]</td></tr><tr><td> $1 9 2 \times 9 6$ </td><td>1.50973</td><td>1.16290</td><td>22.97</td><td>[-0.389, -0.308]</td></tr><tr><td> $2 5 6 \times 1 2 8$ </td><td>1.51070</td><td>1.16352</td><td>22.98</td><td>[-0.389, -0.308]</td></tr><tr><td>Bandwidth  $0 . 8 \times$ </td><td>1.63096</td><td>1.24506</td><td>23.66</td><td> $\left[ - 0 . 4 3 0 , - 0 . 3 4 4 \right]$ </td></tr><tr><td>Bandwidth  $1 . 2 \times$ </td><td>1.38646</td><td>1.08121</td><td>22.02</td><td>[−0.344, −0.269]</td></tr><tr><td>PCG tolerance  $1 0 ^ { - 1 0 }$ </td><td>1.50697</td><td>1.16113</td><td>22.95</td><td> $\left[ - 0 . 3 8 8 , - 0 . 3 0 7 \right]$ </td></tr></table>

Table 7: Is the prospective Full-action reduction stable to the numerical evaluator? Mean numerical Full actions for the frozen Law and 5% Full geometries under spatial refinement, density– source bandwidth perturbation, and solver-tolerance perturbation. The reported paired intervals are for the Full-minus-Law action difference. No setting produces a numerical failure, and the action reduction remains between 22.02% and 23.66%.

## D.2.6 IS THE PROSPECTIVE FULL-ACTION REDUCTION STABLE TO THE NUMERICAL EVALUATOR?

The prospective result uses a spatially discretized density–source evaluator, so we test whether its Law–Full contrast depends materially on the spatial raster, smoothing bandwidth, or linear-solver tolerance. The Law and 5% Full geometries, observations, reference dynamics, moment reconstruction, and scientific-risk calculation remain frozen throughout; only the numerical action evaluator is changed.

As shown in Table 7, spatial refinement changes the Law–Full reduction only from 22.95% to 22.97% and 22.98% at $\mathrm { 1 2 8 \times 6 4 , 1 9 2 \times 9 6 }$ , and $2 5 6 \times 1 2 8 .$ , respectively. Across the bandwidth and solver perturbations, the reduction remains between $2 2 . 0 2 \%$ and 23.66%, and every paired action-difference interval remains strictly below zero. No numerical failures occur, the minimum ESS fraction is 0.168, and the maximum Poisson residual is $1 . 0 2 \times 1 0 ^ { - 1 0 }$ . The reported Law–Full contrast is therefore stable across the tested numerical choices, although these checks do not by themselves establish convergence of the prospective density–source evaluator to the continuum Full functional. The manufactured-solution audit below separately tests continuum convergence for a periodic Full evaluator on an analytically tractable family without density estimation, smoothing, or observation noise.

## D.2.7 DOES THE PERIODIC FULL EVALUATOR CONVERGE TO ANALYTICALLY KNOWN CONTINUUM ACTION?

The preceding refinement study tests stability of an empirical Law–Full contrast but does not isolate discretization error from density–source preprocessing. We therefore perform a deterministic manufactured-solution audit on a periodic domain using the exact information-projected Fourier family of Appendix C.4. Let the frozen reference be uniform with $u = 0 ,$ , and for $k \in \{ 1 , 2 , 4 \}$ define

$$
q _ { \lambda , k } ( x ) = \frac { \exp ( \lambda \cos ( k x ) ) } { 2 \pi I _ { 0 } ( \lambda ) } , \qquad \lambda ( t ) = \frac { 1 } { 2 } \sin ( 2 \pi t ) .\tag{272}
$$

The corresponding normalized continuity forcing is

$$
h _ { k , t } ( x ) = \dot { \lambda } ( t ) \left( \cos ( k x ) - \frac { I _ { 1 } ( \lambda ( t ) ) } { I _ { 0 } ( \lambda ( t ) ) } \right) .\tag{273}
$$

Writing $g ( \lambda )$ for the exact transport metric in equation 227–equation 229, the continuum instantaneous action is

$$
a _ { \mathrm { f u l l } } ( t , k ) = \frac { \dot { \lambda } ( t ) ^ { 2 } } { k ^ { 2 } } g ( \lambda ( t ) ) .\tag{274}
$$

The continuum benchmark is evaluated independently by adaptive one-dimensional quadrature and therefore does not use the discrete Poisson solver under test. Equation equation 274 also gives the exact integrated-action ratios $\mathcal { A } ( 1 ) / \mathcal { A } ( 2 ) = 4$ and $\mathcal { A } ( 1 ) / \mathcal { A } ( 4 ) \overset { ^ { \circ } } { = } 1 6$

<table><tr><td>Frequency k</td><td>Relative error at  $N = 5 1 2$ </td><td>Empirical spatial order</td></tr><tr><td>1</td><td> $1 . 1 5 6 \times 1 0 ^ { - 5 }$ </td><td>2.00004</td></tr><tr><td>2</td><td> $4 . 6 2 3 \times 1 0 ^ { - 5 }$ </td><td>2.00015</td></tr><tr><td>4</td><td> $1 . 8 5 0 \times 1 0 ^ { - 4 }$ </td><td>2.00061</td></tr></table>

Table 8: Manufactured-solution convergence of the periodic Full evaluator. Relative errors compare the discrete time-integrated action with an independently quadrature-evaluated continuum benchmark. The empirical order is computed from the final two spatial refinements. At fixed N, the larger relative error at higher frequency is consistent with the expected dependence of a secondorder spatial discretization on the dimensionless resolution kh.

The available periodic weighted-Poisson implementation is tested by extruding the one-dimensional family to the square torus as $q ( x , y ) \ = \ \bar { q } ( x ) / ( 2 \pi )$ and $s ( x , y ) \stackrel { . } { = } \ q ( x ) \bar { h } ( x ) / ( 2 \pi )$ . Because the solution is independent of $y ,$ this extrusion leaves the continuum action unchanged. Exact cell-centered densities and sources are supplied directly, with no particle sampling, smoothing, density floor, source recentering, or observation noise. Spatial convergence is evaluated at $N \in \{ 3 2 , 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \}$ using 257 time nodes, while temporal convergence is evaluated separately on the finest spatial grid.

The evaluator exhibits approximately second-order spatial convergence for all three frequencies. At $N = 5 1 2$ , the numerical action ratios are 3.999861 for $k = 1$ versus $k = 2$ and 15.997226 for $k = 1$ versus $k = 4$ , compared with the exact values 4 and 16. The isolated temporal-quadrature error is approximately $6 . 7 \times 1 0 ^ { - 1 3 }$ already at 17 nodes and reaches floating-point roundoff under further refinement, so the finest-grid continuum errors are dominated by spatial discretization. The largest physical Poisson residual, $1 . 7 9 \times 1 0 ^ { - 8 }$ , occurs for the coarsest $\bar { N = 3 2 , k = 4 \mathrm { c a s e } }$ , where only eight cells resolve each wavelength; for $N \geq 6 4$ , the residual remains below $2 . 3 \times 1 0 ^ { - 1 1 }$ and no solver failures occur.

This manufactured-solution audit shows that the tested periodic Full evaluator converges to analytically known continuum action and reproduces the inverse-frequency transport scaling predicted by the theory.

## D.2.8 ABLATION: WHICH PARTS OF THE PROSPECTIVE PREDICTIVE INTERFACE MATTER?

Prospective selection is conditional on three classes of predicted inputs: sensor-response means, single-particle response covariances, and scientific quantities used by the primary risk. This ablation isolates their roles using the 541 labeled sensor geometries discovered by the original prospective search, of which 406 satisfy the original geometric constraints, while holding the candidate pool fixed across perturbations. Fixing the pool removes variation from continuous design optimization and isolates how errors in the predictive interface change the ranking and admissibility of already discovered measurement systems. The unperturbed condition is therefore a new selection restricted to this fixed pool rather than a reevaluation of the originally reported Law–Full pair, and its heldout action reduction need not equal that of the original prospective frontier. Under each perturbed interface, Law minimizes the corresponding scientific risk over the fixed pool and Full minimizes action within the condition-specific 5% Law-relative risk band and numerical-support constraints; all selected geometries are then frozen before evaluation with the common unperturbed post-freeze validation reference.

We consider three perturbation families. First, we add a fixed bias direction to the predicted sensorresponse means at amplitudes 0.25 and 0.50 times the local response standard deviation. Second, we rescale the predicted single-particle sampling covariance by factors 0.75 and 1.25, leaving detector noise unchanged. Third, we perturb the predicted scientific-QoI targets by 0.25 and 0.50 frozen QoI scales along a fixed direction. Table 9 summarizes the resulting finite-pool selections and their heldout validation performance; the reported rank correlation is computed over the same 406 statically admissible geometries in every condition. Figure 8 complements this summary by showing how each perturbation reorganizes the design-time scientific-risk landscape over the common candidate pool as a function of minimum sensor separation.

<table><tr><td colspan="4"></td><td rowspan="2">Held-out risk Held-out action reduction (%)</td></tr><tr><td>Predictor condition</td><td>Feasible</td><td>Risk-rank  $\rho$ </td><td>change (%)</td></tr><tr><td>Unperturbed</td><td>246</td><td>1.000</td><td>2.030</td><td>24.324</td></tr><tr><td>Mean bias, 0.25 local SD</td><td>16</td><td>-0.294</td><td>-0.141</td><td>0.327</td></tr><tr><td>Mean bias, 0.50 local SD</td><td>1</td><td>-0.355</td><td>0.000</td><td>0.000</td></tr><tr><td>Sampling covariance ×0.75</td><td>246</td><td>1.000</td><td>2.030</td><td>24.324</td></tr><tr><td>Sampling covariance ×1.25</td><td>246</td><td>1.000</td><td>2.279</td><td>23.133</td></tr><tr><td>QoI-target bias, 0.25 scale</td><td>224</td><td>0.800</td><td>-1.749</td><td>26.747</td></tr><tr><td>QoI-target bias, 0.50 scale</td><td>46</td><td>0.746</td><td>0.641</td><td>25.642</td></tr></table>

Table 9: Sensitivity of prospective selection to the aggregate predictive interface. All conditions rank the same fixed pool of 541 candidate geometries. “Feasible” is the number of candidates satisfying the original numerical/support requirements and the condition-specific 5% Law-relative scientific-risk band. Risk-rank $\rho$ is the Spearman correlation with the unperturbed condition over the 406 statically admissible geometries. Held-out quantities compare the Full and Law designs selected under the same predictor condition on the common post-freeze validation experiment.

As summarized in Table 9 and visualized in Figure 8, the covariance perturbations leave the design landscape essentially unchanged. At both covariance scales, the feasible set remains at 246 candidates, the scientific-risk rankings have correlation above 0.9999 with the unperturbed predictor, and the Full-selected geometry is unchanged. The corresponding held-out action reductions are 24.32% and 23.13%. Thus, within the tested range, moderate covariance rescaling has little effect on either admissibility or the secondary action ranking.

Systematic errors in the predicted sensor means have a markedly different effect. At the 0.25-SD perturbation, all 406 statically admissible candidates continue to pass the non-risk numerical and support requirements, but only 16 remain within the 5% scientific-risk band. The scientific-risk ranking over these valid geometries is negatively correlated with the unperturbed ranking, ρ = −0.294, and the selected Law and Full geometries become nearly identical, leaving only a 0.33% held-out action reduction. At the 0.50-SD perturbation, only one candidate remains feasible, so Law and Full necessarily coincide. This stronger perturbation also affects support: only 67 statically valid candidates satisfy the non-risk gates in both the perturbed and unperturbed conditions, yet their scientific-risk rankings are still strongly reversed, with $\rho = - 0 . 7 6 1$ . The sensitivity therefore persists after excluding both geometrically invalid designs and condition-dependent support failures.

The same mean perturbations also shift the apparent low-risk region of the fixed pool toward closely spaced sensors. Before applying the geometric constraint, none of the 246 unperturbed below-cap candidates violates the minimum-separation rule, compared with 15 of 31 candidates under the 0.25- SD perturbation and 30 of 31 under the 0.50-SD perturbation. Candidate-level decompositions show that the effect is not a uniform offset of the risk values: the perturbed sensor means change reconstructed moment trajectories and projected scientific quantities in a geometry-dependent way. For example, the original Law geometry changes from design-time risk 1.13 to 6.63 and 19.35 under the two mean perturbations, whereas the geometry selected under the larger perturbation changes from risk 8.52 in the unperturbed condition to 2.53. The dominant changes in the inspected candidate arise from squared mean QoI errors rather than trial-to-trial variance.

Perturbing the scientific-QoI targets produces a third, more localized mechanism. For a fixed geometry, these perturbations leave predicted sensor responses, reconstructed moments, projected law trajectories, and Full actions unchanged; only the scientific admissibility criterion changes. Consequently, the Full-action ranking is identical to the unperturbed ranking on the common pre-riskvalid pool, while the scientific-risk rank correlation decreases to 0.800 and 0.746 and the feasible set contracts to 224 and 46 candidates. The selected Full designs nevertheless retain held-out action reductions of 26.75% and 25.64%. This separates sensitivity of the scientific-risk constraint from sensitivity of the law-level action itself.

The held-out results also illustrate the scope of a relative risk constraint. Every Full design remains within 5% of the Law design selected under the same perturbed predictor, but this within-condition comparison does not constrain absolute performance relative to the unperturbed design problem. In particular, the held-out Full risks under the two sensor-mean perturbations are 4.70 and 8.94, compared with 1.22 for the unperturbed Law design. Thus, the relative constraint continues to control the Full–Law tradeoff under a given predictive model, while systematic predictor error can shift the scientific-risk landscape shared by both methods.

![](images/69d8cc8d7041cab436072a6918efc02da349aed132d1bef385807ed0af75ffdf.jpg)  
Figure 8: Predictor perturbations reshape the prospective design-time risk landscape in different ways. Each panel shows the same fixed pool of 541 candidate sensor geometries, with minimum pairwise sensor separation on the horizontal axis and scientific-risk margin $R _ { \mathrm { s e l } } / R _ { \mathrm { L a w } } ^ { \mathcal { C } } - 1$ relative to the condition-specific candidate-pool Law anchor on the vertical axis. The vertical dashed line marks the minimum allowed separation 0.24, and the horizontal dashed line marks the 5% Lawrelative risk allowance. Highlighted markers denote the Law and Full designs selected from the fixed pool. The tested covariance rescalings leave the landscape nearly unchanged, whereas sensormean perturbations substantially reorganize the relative risk ordering and move an increasing fraction of the below-cap candidates into the separation-invalid region. QoI-target perturbations alter the scientific-risk landscape without changing the underlying reconstructed trajectories. Because each panel is normalized by its own Law anchor, vertical positions describe within-condition relative risk rather than absolute risk across perturbations. The two dashed thresholds alone do not define final feasibility, which also requires the condition-dependent numerical and support criteria.

The fixed-pool study identifies three forms of predictor sensitivity. The tested covariance rescalings leave both scientific and action rankings nearly unchanged; QoI-target shifts alter scientific admissibility while preserving the law-level action ranking; and sensor-mean bias changes the reconstructed moments themselves, substantially reorganizing both the scientific-risk landscape and the available Law–Full tradeoff. These results are conditional on the fixed discovered candidate pool, one selection reference and one separately trained post-freeze validation reference, and the specified perturbation directions.

## D.2.9 ARE THE SKYRMION ACTION REDUCTIONS ROBUST TO NUMERICAL APPROXIMATION?

We test two possible explanations for the skyrmion action reductions: sensitivity to the potential dictionary used during design optimization, and disproportionate influence of a trial with concentrated importance weights. Sensor geometries remain frozen throughout, and all 32 paired validation trials remain in the primary analysis.

<table><tr><td>Risk allowance</td><td>Base</td><td>Fourier-400</td><td>Fourier-560</td><td>Interaction-416</td></tr><tr><td>1%</td><td>3.51</td><td>3.51</td><td>3.52</td><td>3.53</td></tr><tr><td>2%</td><td>6.91</td><td>6.92</td><td>6.93</td><td>6.95</td></tr><tr><td>3%</td><td>9.38</td><td>9.40</td><td>9.41</td><td>9.43</td></tr><tr><td>4%</td><td>9.38</td><td>9.40</td><td>9.41</td><td>9.43</td></tr><tr><td>5%</td><td>12.22</td><td>12.25</td><td>12.28</td><td>12.31</td></tr></table>

Table 10: Does the action reduction persist under a richer potential space? Percentage reductions in mean fitted action relative to Law under each evaluator. All 32 trials are retained, including failed-certification cases; the 3%/4% row represents one shared geometry. Fourier-400 and Fourier-560 refine the Fourier dictionary, while Interaction-416 adds interaction features. Fourier-560 has six held-out energy-check failures, and ESS support failures remain under every evaluator.
<table><tr><td>Design</td><td>Base (K = 280)</td><td>Fourier-560 (K = 560)</td><td>Increase (%)</td></tr><tr><td>Law</td><td>0.024433</td><td>0.024581</td><td>0.61</td></tr><tr><td>Full (5% allowance)</td><td>0.021449</td><td>0.021564</td><td>0.54</td></tr></table>

Table 11: Absolute fitted action under nested Galerkin enrichment. Mean fitted actions over the same 32 validation trials. In agreement with the monotonicity of the underlying restricted variational problem in Proposition 6, enlarging the Fourier potential space raises the fitted action for both frozen designs. The corresponding Full-versus-Law reduction changes only from 12.22% to 12.28%.

Does the action reduction persist under a richer potential space? A design optimized against a restricted Galerkin evaluator could appear favorable because the evaluator misses correction directions that matter more for that design. Proposition 6 makes this issue explicit: for a fixed projected law and continuity forcing, the exact restricted variational action is a lower bound on Full, and nested potential-space enrichments can only tighten that bound. This property alone does not show that the Base dictionary captures enough of the Full correction for reliable design comparison, so we re-evaluate the frozen frontier using four potential spaces: Base is the original dictionary of 248 one-particle Fourier functions and 32 radial pair functions, Fourier-400 and Fourier-560 retain these coordinates and append 120 and 280 Fourier functions respectively, while Interaction-416 instead appends 136 pre-specified distinct-particle products of low-frequency Fourier channels, testing pairdependent potential directions rather than additional one-particle resolution. Within each trial and geometry, reconstructed moments, projected weights and continuity forcing are held fixed across evaluators, with fitting and audit banks kept separate. Original coordinates, normalization, solver settings and certification thresholds are unchanged; only the potential space is enriched.

For evaluator E, let $\overline { { A } } _ { E } ( \pmb { \eta } )$ denote mean fitted action over the 32 trials. We compare

$$
\widehat { D } _ { p , E } = 1 - \frac { \overline { { A } } _ { E } ( \eta _ { \mathrm { F u l l } , p } ) } { \overline { { A } } _ { E } ( \eta _ { \mathrm { L a w } } ) } , \qquad \Delta _ { p , E } = 1 0 0 \big ( \widehat { D } _ { p , E } - \widehat { D } _ { p , \mathrm { B a s e } } \big ) ,\tag{275}
$$

where $1 0 0 \widehat { D } _ { p , E }$ is the percentage action reduction and $\Delta _ { p , E }$ is its change in percentage points relative to Base. Reductions are ratios of mean actions, not means of trial-wise ratios. The 3% and 4% allowances select identical coordinates and are evaluated once.

The action reduction persists at every operating point under all four evaluators (Table 10), with a largest change of only 0.096 percentage points relative to Base. Full retains lower fitted action than Law in every individual trial under every evaluator, and the ordering across allowances is unchanged. Table 11 additionally shows the absolute-action change under the nested Base-to-Fourier-560 enrichment: both frozen designs acquire a small amount of additional fitted action while their relative contrast remains essentially unchanged.

Thus, the observed contrasts are stable under the tested Fourier and interaction enrichments, although these checks do not bound correction energy in untested potential directions.

Contrast stability is distinct from numerical certification. The positive, ordered frontier is reproduced by Fourier-400 and Interaction-416, both of which pass the reported held-out residual checks, so the enrichment result does not rely on the largest Fourier-560 system. Fourier-560 exceeds the held-out energy threshold in six of 160 unique trial/design evaluations, identifying a numerical-certification limit of that larger basis rather than a reversal of the design ordering. Separately, the low-support validation pair fails one ESS support check for Law and all four distinct Full geometries under every evaluator. We therefore do not claim full numerical certification or continuum convergence; Proposition 6 characterizes what convergence of the underlying restricted variational problem would require, while the present experiment supports only that the observed Full-versus-Law contrast is stable under the tested one-particle and interaction-space enrichments.

<table><tr><td>Allowance</td><td>κ = 0.25</td><td>κ = 1</td><td>κ = 4</td></tr><tr><td>1%</td><td>3.02</td><td>3.26</td><td>2.97</td></tr><tr><td>2%</td><td>4.79</td><td>5.22</td><td>4.95</td></tr><tr><td>3%</td><td>6.27</td><td>6.88</td><td>6.59</td></tr><tr><td>4%</td><td>7.61</td><td>8.38</td><td>8.09</td></tr><tr><td>5%</td><td>8.85</td><td>9.79</td><td>9.49</td></tr></table>

Table 12: Sensitivity of the Full-versus-Law contrast to the transport–reaction metric. Entries are percentage reductions in mean Full action relative to Law, computed separately within each value of κ. Because κ changes the correction metric, absolute action values are not compared across columns.

Does the low-support trial drive the aggregate action reduction? We examine the low-support validation pair’s influence using Base at the 5% allowance. Retaining all 32 pairs gives a 12.215% action reduction; excluding only this prespecified pair gives 12.380%, a change of +0.165 percentage points. Across all leave-one-pair-out calculations, the reduction ranges from 12.083% to 12.442%. The low-support validation pair ranks third in absolute deletion influence, but removing any single pair preserves the aggregate action reduction. Thus the low-support trial does not drive the reported contrast: every leave-one-pair-out analysis preserves an action reduction above 12%. These deletions are sensitivity checks rather than replacements for the primary analysis, and they do not resolve the trial’s separate support-certification issue.

We additionally evaluate both frozen geometries on independent reference banks, preserving their saved moment trajectories and Base dictionary while recalibrating projected weights and forcing on each bank. Increasing the fitting bank from 65,536 to 131,072 and 262,144 configurations changes the low-support validation pair’s fitted-action reduction from 9.685% to 8.421% and 8.877%, respectively. The trial-specific estimate varies with reference-bank size, but its influence on the aggregate contrast is small under this diagnostic: replacing only this pair’s actions by the largest-bank estimates changes the overall reduction by −0.048 percentage points.

## D.2.10 SENSITIVITY TO THE TRANSPORT–REACTION METRIC

In the unbalanced setup, the reaction penalty κ determines the relative geometry assigned to move ment and reaction and is therefore a modeling choice rather than a parameter whose absolute action values can be compared across settings. We test whether the within-metric Full-versus-Law contrast depends strongly on this choice by evaluating the frozen construction at $\kappa \in \{ 0 . 2 5 , 1 , 4 \}$ on the same shared physical validation bank. As shown in Table 12 and Figure 9, Full retains lower mean action than Law at every allowance under all three metrics. The reduction ranges from 2.97% to 3.26% at the 1% allowance and from 8.85% to 9.79% at the 5% allowance, with the intermediate allowances showing a similar pattern, indicating that the within-metric Full-versus-Law contrast is not tied to the default reaction penalty.

## D.2.11 UNBALANCED MECHANISM DECOMPOSITIONS

The screened Full correction admits two complementary decompositions established in the finitemeasure theory. First, the Full energy in equation 88 separates into Move and Reaction contributions, corresponding respectively to the transport and reaction terms of the optimal correction. Second, the orthogonal decomposition in equation 161 writes Full as Tangent plus Hidden, where Tangent is the minimum correction visible through the measured raw moment rates and Hidden is the remaining law-level component. GlobalReaction provides a separate restriction diagnostic: unlike Tangent, which restricts the correction through the measured moment-rate equations, GlobalReaction restricts the admissible reaction field while retaining transport. Table 13 and Figure 10 show that the Tangentinvisible component remains substantial throughout the tested regime, while the common-Full crossscore of the Full-selected geometry is lower than those of both restricted comparators in every tested κ–allowance combination.

![](images/5b48930a52a5112806e49f16385f37f1b6edcdc66a040b7a5685061d6ecbb9af.jpg)

Figure 9: The within-metric Full-versus-Law action reduction persists across reaction penalties. Each curve reports the percentage reduction in mean Full action relative to Law at a fixed κ. Scientific observations are shared across the sweep, while κ changes only the transport–reaction correction metric; vertical comparisons of absolute action across κ are therefore not meaningful.
<table><tr><td>κ</td><td>Move / Full (%)</td><td>Reaction / Full (%)</td><td>Hidden / Full (%)</td><td>Full vs. Tangent (%)</td><td>Full vs. GlobalReaction (%)</td></tr><tr><td>0.25</td><td>21.99–22.65</td><td>77.35–78.01</td><td>43.15-43.80</td><td>1.95–6.12</td><td>2.23-6.28</td></tr><tr><td>1</td><td>23.97–24.21</td><td>75.79–76.03</td><td>64.58–65.51</td><td>2.57–6.30</td><td>2.16–6.00</td></tr><tr><td>4</td><td>38.79-39.33</td><td>60.67–61.21</td><td>73.51-74.62</td><td>2.27–5.62</td><td>1.76–5.20</td></tr></table>

Table 13: Unbalanced mechanism summary across the 1%–5% allowances. The first three quantities describe the Full-selected geometry and are reported as ranges over the five allowances. Move and Reaction form one decomposition of Full, while Tangent and Hidden form a separate orthogonal decomposition. The final two columns report the range of reductions in the common Full cross-score obtained by Full relative to the Tangent- and GlobalReaction-selected geometries.

## E EXPERIMENTAL DETAILS AND REPRODUCIBILITY

Shared experimental conventions. For particle-based studies, a trial observes $\begin{array} { r l } { { \mathbf { \nabla } } y _ { k } } & { { } = } \end{array}$ $\begin{array} { r } { n ^ { - 1 } \sum _ { i = 1 } ^ { n } \Phi _ { \eta } ( X _ { t _ { k } } ^ { ( i ) } ) + \sigma _ { \mathrm { o b s } } \varepsilon _ { k } } \end{array}$ with $\varepsilon _ { k } \sim \mathcal { N } ( 0 , I _ { R } )$ ; one microscopic state is a tracer position in the transport studies and a complete interacting configuration in the skyrmion study. Reconstruction produces a differentiable moment trajectory that is information-projected relative to the frozen reference. Except for the stricter skyrmion development constraint stated below, allowance $p \%$ means $R _ { \mathrm { s e l } } ( \pmb { \eta } ) \leq ( \bar { 1 } + p / 1 0 0 ) R _ { \mathrm { L a w } }$ , where $R _ { \mathrm { L a w } }$ is the selection risk of the numerically optimized Law design. For a fixed reference, $\widehat { D } _ { p } = 1 - \overline { { A } } ( \eta _ { \mathrm { F u l l } , p } ) / \overline { { A } } ( \eta _ { \mathrm { L a w } } )$ , with $\overline { { A } }$ the trial-mean numerical Full action; when several frozen references are used, reference-wise effects are computed before aggregation as specified by the study. Selection feasibility, held-out scientific risk, numerical certification, and action improvement are reported separately. Numerical Full denotes the declared raster or Galerkin approximation, and absolute action values are not compared across systems or reference regimes. For normalized empirical projection weights $p _ { 1 } , \dotsc , p _ { N } , \dot { \mathrm { r E S S } } = ( N \dot { \sum _ { i } } p _ { i } ^ { 2 } ) ^ { - 1 }$ ; the Gaussian and double-gyre studies require $\mathrm { r \mathrm { \bar { E } S S \geq 0 . 0 \bar { 3 } } }$ for their declared selection-time support checks

$$
--- - \kappa = 0 . 2 5 \mathrm { ~  ~ { ~ \tau ~ } ~ } - \Omega - \kappa = 1 \mathrm { ~  ~ { ~ \tau ~ } ~ } \cdots \kappa = 4
$$

![](images/f851541cfbf93d37efb2c977bf94e4677e47e6ec27d8550876d67e05c194e27b.jpg)

(b) Reaction fraction  
![](images/5bc6209e2c6abb63054cc393186b0bfd5c3bcda8404253e8460dbfc01d70af24.jpg)  
Figure 10: Mechanism decomposition of the unbalanced Full correction. Left: fraction of Full action lying in the Tangent-invisible Hidden component. Right: fraction of Full action allocated to reaction rather than spatial movement. Fractions are computed within each fixed transport–reaction metric; the figure does not compare absolute action values across κ.

and the skyrmion study requires 0.05. These thresholds diagnose support concentration rather than empirical moment feasibility, which is characterized separately by Proposition 3.

Reference flow matching. The Gaussian, retrospective and prospective double-gyre, and skyrmion studies use reference velocities trained by flow matching on endpoint pairs. For an experiment-specific endpoint coupling $( A _ { 0 } , A _ { 1 } ) \sim \Pi$ , bridge $X _ { t } ^ { \mathrm { b r } }$ , and $T \sim \mathrm { U n i f } [ 0 , 1 )$ , we minimize

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } \left[ \left. f _ { \theta } ( X _ { T } ^ { \mathrm { b r } } , T ) - \left. \partial _ { t } X _ { t } ^ { \mathrm { b r } } \right| _ { t = T } \right. _ { 2 } ^ { 2 } \right] . } \end{array}\tag{276}
$$

For the Gaussian benchmark, $X _ { t } ^ { \mathrm { b r } } = ( 1 - t ) A _ { 0 } + t A _ { 1 } + 0 . 1 5 \sin ( \pi t ) Z$ with independently sampled endpoint states and $Z \sim { \mathcal { N } } ( 0 , { \dot { I } } )$ For both double-gyre studies, independently sampled empirical endpoints are first mapped to box-logit coordinates and connected by the noiseless linear bridge $X _ { t } ^ { \mathrm { b \bar { r } } } = ( 1 - t ) A _ { 0 } + \bar { t } \bar { A } _ { 1 } ;$ the learned latent velocity is mapped back to physical coordinates by the logistic Jacobian. For the skyrmion reference, endpoint configurations use the frozen particlematched coupling and the periodic bridge $X _ { t } ^ { \mathrm { b r } } = [ A _ { 0 } \dot { + } t \Delta + 0 . \dot { 0 } 1 \sin ( \pi t ) Z ]$ mod $L ,$ where $\Delta$ is the minimum-image displacement. After training, the reference is frozen and $\widetilde { Q } _ { t }$ is represented by deterministic forward rollout of the learned velocity from the initial law; the training bridge itself is not used as the reference marginal path. The unbalanced experiment instead uses the fixed analytical reference described in Appendix E.5.

## E.1 INFORMATION BOUNDARIES

The studies differ in the target information available during measurement design. Table 14 summarizes these regimes. In particular, the prospective double-gyre optimizer accesses intermediate target information only through a frozen aggregate interface: the interface is constructed offline from a separate microscopic simulation bank, but the underlying intermediate particles and target simulator are not available during sensor optimization.

## E.2 GAUSSIAN-MIXTURE CRITERION COMPARISON

Gaussian system and sensor family. For $d ( \beta ) = ( \cos { \beta } , \sin { \beta } ) ^ { \top }$ , let

$$
\begin{array} { r } { g _ { \beta } ( x ) = \frac { 1 } { 2 } \mathcal { N } ( x ; 1 . 5 d ( \beta ) , 0 . 3 ^ { 2 } I _ { 2 } ) + \frac { 1 } { 2 } \mathcal { N } ( x ; - 1 . 5 d ( \beta ) , 0 . 3 ^ { 2 } I _ { 2 } ) } \end{array}
$$

and

$$
\bar { p } _ { t } ^ { \alpha } = ( 1 - t ) ^ { 2 } g _ { 0 } + 2 t ( 1 - t ) g _ { \alpha } + t ^ { 2 } g _ { \pi / 2 }
$$

<table><tr><td>Study</td><td>Information available during selection</td><td>Information introduced after design freeze</td></tr><tr><td></td><td>Gaussian mixture Analytic target law, target-derived moments and covariances, noisy finite-sample observations, population-risk screen, and one frozen endpoint-trained reference.</td><td>Fresh target draws, particle sampling, and detector noise from the same analytic family. The reference is unchanged.</td></tr><tr><td>Retrospective double-gyre</td><td>Frozen microscopic target simulation bank and three endpoint-trained references; candidate observations and scientific-risk quantities are evaluated directly from the target bank.</td><td>Fresh particle resampling and detector noise, conditional on the same target bank and reference ensemble.</td></tr><tr><td>Prospective double-gyre</td><td>Endpoint ensembles, one frozen endpoint-trained reference, and fixed aggregate sensor-response means, predictions. These aggregate fields are precomputed from a separate microscopic simulation bank and its intermediate</td><td>A separately seeded evaluation reference trained from the same endpoint dataset, together with new microscopic validation cross-sensor covariances, and scientific-QoI trajectories, sampling, and detector noise for action and scientific-risk validation.</td></tr><tr><td>Skyrmions</td><td>particles are not exposed to the optimizer. A fixed development set of interacting configurations, one frozen endpoint-trained permutation-equivariant reference, aggregate sensor observations, and the nine permutation-invariant scientific features used to evaluate development risk.</td><td>A separate set of 32 fresh paired 1 measurement trials for scientific-risk and action validation. The reference model and selected sensor geometries remain frozen. Richer potential dictionaries, leave-one-pair-out analyses, and larger reference banks are introduced only as</td></tr><tr><td>Unbalanced Gaussian</td><td>A fixed analytical reference and finite-measure simulator, together with development observations and simulator truth used for scientific-risk evaluation.</td><td>post-selection numerical sensitivity checks. A fresh bank of 128 paired trials with new physical nuisances, particle observations, detector noise, and log-mass noise. The same frozen bank is reused across the κ sensitivity analysis.</td></tr></table>

Table 14: Information boundaries of the reported studies. “After design freeze” denotes information unavailable when optimizing the reported sensor geometries. Fresh evaluation is conditional on the quantities explicitly retained in the corresponding row.

for $\alpha \in [ \pi / 6 , \pi / 3 ]$ . Benchmark truth is the normalized mass of $\bar { p } _ { t } ^ { \alpha }$ on a $5 1 \times 5 1$ grid in $[ - 3 . 2 , 3 . 2 ] ^ { 2 }$ Two sensors have centers $z _ { j } = 1 . 5 d ( \theta _ { j } )$ on the radius-1.5 ring, angular separation at least $2 0 ^ { \circ }$ , and responses $\Phi _ { \eta , j } ( x ) = \mathrm { e x p } \bar { ( } - \| x - z _ { j } \| ^ { 2 } / ( 2 \ell ^ { 2 } ) )$ ) with $\ell = 0 . 4 5$ . Each trial observes 100 particles at eleven acquisition times with $\sigma _ { \mathrm { o b s } } = 0 . 0 1$ , and endpoint-anchored quadratic generalized least squares reconstructs moments on 21 scientific times.

Reference, selection, and evaluation protocol. The reference is trained using equation 276 with four width-128 SiLU hidden layers for 12,000 Adam steps, batch size 2,048, initial learning rate $1 0 ^ { - 3 }$ with cosine decay to $5 \times 1 0 ^ { - 5 }$ , and gradient clipping at 10. Scientific risk is Gaussiankernel $\mathrm { { \bf M M D } ^ { 2 } }$ with bandwidth $0 . 5 5$ on the unobserved time nodes. The frozen Law anchor is $R _ { \mathrm { L a w } } = 0 . 0 6 6 2$ with population-risk ceiling 0.0692. Law, Tangent, Full, residual $L ^ { 2 }$ , forward KL, and static $W _ { 2 }$ are optimized at allowances 0.5, 1, 2, 3, 4, 5% using the same information-projected construction; each receives seven common starts and 40 Adam steps per allowance with learning rate 0.012 and constraint penalty $1 0 ^ { 4 }$ , together with the same number of final candidate evaluations. Selection uses 4 trials for gradient estimation, 32 for scientific-risk evaluation, 16 for preliminary action evaluation, and 64 for final objective evaluation. Full uses a $4 1 ^ { 2 }$ optimization raster and a $1 0 1 ^ { 2 }$ final evaluation raster with no added operator-density floor. Frozen geometries are evaluated on 64 paired holdout trials. Static $W _ { 2 }$ was added after the original five-selector holdout had been inspected. It received the same search opportunity but reused that holdout and is therefore post hoc.

Allowance sweep and criterion-specific selections. Across the prescribed risk allowances, the secondary targets select different sensor geometries despite operating under the same scientific-risk restriction. At the 0.5% allowance, the Full-selected design reduces held-out Full action by 30.67% relative to Law; from 1% through 5%, Full selects the same geometry and gives a 33.59% reduction. The 1% cross-objective results are reported in Table 1. Appendix D.2.4 repeats the selection over a common promoted candidate pool and again produces criterion-specific geometries, showing that the separation is not solely a consequence of criterion-specific optimization trajectories. All reported designs satisfy their selection-time scientific and numerical requirements; held-out scientific-risk transfer is assessed separately in Appendix D.2.3, where the 0.5% point remains unresolved under simultaneous one-sided inference and the 1%–5% points are supported below their corresponding held-out caps.

## E.3 PROSPECTIVE DOUBLE-GYRE STUDY

Double-gyre system. Passive tracers evolve on $\Omega \ = \ [ 0 , 2 ] \times [ 0 , 1 ]$ under the time-dependent double-gyre with $a ( \tau ) ~ = ~ 0 . 2 5 \sin ( 2 \pi \tau / 1 0 ) , ~ f ( x _ { 1 } , \tau ) ~ \stackrel { . } { = } ~ a ( \tau ) x _ { 1 } ^ { 2 } + \stackrel { . } { ( 1 - 2 a ( \tau ) ) } x _ { 1 } , ~ v ( x , \tau ) ~ = ~ 0 .$ $0 . 1 \pi ( - \mathrm { \bar { s i n } } ( \pi f )$ cos(πx<sub>2</sub>), cos(πf) sin(πx<sub>2</sub>)∂<sub>x</sub> f)<sup>⊤</sup>, and $d X _ { t } / d t \overset { \cdot } { = } 1 0 \overset { \cdot } { \boldsymbol { v } } ( X _ { t } , 1 0 \dot { t } )$ . The initial population is a mixture of four truncated Gaussian components with coordinate standard deviation 0.07 and a 10% uniform background. The retrospective and prospective studies use this common transport system; their information boundaries and selection procedures differ as described below and in Appendix D.1.1.

Information boundary and scientific objective. The optimizer receives endpoint ensembles, one frozen endpoint-trained reference, and a frozen aggregate predictive interface constructed offline from a separate microscopic simulation bank; the underlying intermediate particles, hidden trajectories, and target simulator are not exposed during sensor optimization. On a $1 2 9 \times 6 5$ aggregate $\mathrm { g r i d } .$ , the interface supplies the sensor-response mean ${ \widehat { \mu } _ { \eta } ( t ) }$ , which provides the designstage moment trajectory $c _ { \eta } ( t )$ used by FIDE, together with the full single-particle response covariance $\widehat { \Sigma } _ { \eta } ( t )$ and predicted trajectories of the scientific quantities used by the design-stage risk. For 2,000 sampled particles and detector-noise standard deviation 0.005, selection uses $\mathbb { E } [ { \pmb { y } } _ { t } \ | \ { \pmb { \eta } } ] = \widehat { \pmb { \mu } } _ { \pmb { \eta } } ( t )$ and $\smash {  { \mathrm { C o v } } ( y _ { t } \mid \pmb { \eta } ) = \widehat { \Sigma } _ { \pmb { \eta } } ( t ) / 2 , 0 0 0 + 0 . 0 0 5 ^ { 2 } I _ { 4 } } ,$ , retaining cross-sensor sampling dependence. The scientific features are $\pmb { \xi } ( x _ { 1 } , x _ { 2 } ) = ( x _ { 1 } / 2 , x _ { 2 } , ( x _ { 1 } / 2 ) ^ { 2 } , x _ { 2 } ^ { 2 } , x _ { 1 } x _ { 2 } / 2 ) ^ { \top }$ . Writing $m _ { j } ( t )$ for their frozen aggregate predictions and $s _ { j } = \operatorname* { m a x } \{ \mathrm { s t d } _ { t } m _ { j } ( t ) , 0 . 0 2 \}$ , selection risk is $\begin{array} { r } { R _ { \mathrm { s e l } } ( \eta ) = B ^ { - 1 } \sum _ { b = 1 } ^ { B } \sum _ { q = 0 } ^ { 2 0 } \omega _ { q } \sum _ { j = 1 } ^ { 5 } [ ( \mathbb { E } _ { Q _ { b , t _ { a } } ^ { \eta } } [ \xi _ { j } ] - m _ { j } ( t _ { q } ) ) / s _ { j } ] ^ { 2 } } \end{array}$ , where $\omega _ { q }$ are normalized trapezoidal weights on the 21 scientific times. The final Law anchor is $R _ { \mathrm { L a w } } = 1 . 1 3 4 4$ and is obtained entirely from selection-stage information.

Optimization, freeze, and validation. The selection reference uses 32,768 rollout particles; Law and Tangent use 16 starts and 60 Adam steps, while Full uses 16 starts and 50 steps followed by higher-resolution evaluation of eight candidates and local refinement of three. Final selection uses 32 common trials and a $1 2 8 \times 6 \bar { 4 }$ Full raster at all 21 scientific times. After the frontier for $p \in$ {1, 2, 3, 4, 5} is frozen, a separately seeded validation reference trained from the same endpoint dataset and 64 paired trials with fresh microscopic observations and hidden scientific targets are introduced, with no validation result used to alter the geometries or Law anchor. The Full-versus Law action reductions are 13.71%, 19.05%, 19.05%, 19.05%, and 22.95%; the 2%–4% allowances select the same Full geometry and have simultaneous 95% interval [16.72%, 21.37%], while the 5% interval is [20.63%, 25.27%]. At 5%, Full uses 2.834% additional selection risk and has mean Full action 1.16113, compared with 1.50697 for Law and descriptive Tangent cross-score 1.32591. Held-out scientific-risk increases are 1.060%, 1.022%, 1.022%, 1.022%, and 1.711%; simultaneous one-sided inference leaves the 1% point unresolved and supports the 2%–5% points below their corresponding caps.

## E.4 INTERACTING-SKYRMION STUDY

System, observations, and scientific objective. One microscopic state is a configuration $X _ { t } =$ $( \bar { r } _ { 1 , t } , \ldots , r _ { 1 6 , t } ) \in \mathcal { X } _ { \mathrm { s k y } } : = \Omega _ { \mathrm { p e r } } ^ { 1 6 }$ , where $\Omega _ { \mathrm { p e r } } \ = \ ( \mathbb { R } / 2 \mathbb { Z } ) \times \mathbf { \bar { \Omega } } ( \mathbb { R } / \mathbb { Z } )$ , so dim $\mathcal { X } _ { \mathrm { s k y } } ~ = ~ 3 2$ and ${ \dot { r } } _ { i , t } = V _ { i } ( X _ { t } , t ) { \mathrm { f o r } } i = 1 , \dots , { \dot { 1 6 } }$ . The implemented simulator uses 24 Euler–Maruyama substeps per scientific interval, interaction strength 0.035 and range 0.16, pinning strength 0.055 and width 0.10, Magnus coefficient 0.32, and diffusion amplitude 0.006. Four width-0.12 aggregate sensors have periodic pairwise separation at least 0.20; each measurement averages 512 configurations and adds detector noise with standard deviation 0.002 at seven acquisition times, while reconstruction is evaluated at 13 scientific nodes. Each fresh validation truth contains 5,000 configurations. Scientific risk compares nine permutation-invariant structural features—four radial-pair summaries, four structure factors, and one hexatic-order summary—using fixed covariance whitening and normalized trapezoidal time weights.

Reference, configuration-space approximation, and development protocol. The frozen reference is an endpoint-only permutation-equivariant flow trained for 6,000 Adam steps with batch size 512; validation does not retrain or replace this reference. Full is approximated in configuration space by $\begin{array} { r } { \psi _ { t , K } ( X ) = \sum _ { a = 1 } ^ { K } \beta _ { a } ( t ) \chi _ { a } ( X ) } \end{array}$ and $\delta _ { t , K } ( X ) = - \nabla _ { X } \psi _ { t , K } ( X )$ with an immutable $K = 2 8 0$ dictionary containing 248 one-particle Fourier functions and 32 radial-pair functions. Risk, Galerkin fitting, and independent audit banks contain 32,768, 65,536, and 262,144 complete configurations, respectively. The solver uses float64 arithmetic with Jacobi equilibration and relative eigenvalue cutoff $1 0 ^ { - 1 2 }$ , and the prespecified support requirement is $\mathrm { r E S S } \geq 0 . 0 5$ The reported frontier is optimized on 17 previously inspected development draws using the stricter paired development-risk ceiling $R _ { \mathrm { d e v } } ( \eta ) \leq ( 1 + 0 . 8 p / 1 0 0 ) R _ { \mathrm { L a w } }$ for nominal allowances $p \in \{ 1 , 2 , 3 , 4 , 5 \}$ . Development action reductions are 4.04%, 7.94%, 10.70%, 10.70%, and 12.87%; the 3% and 4% operating points use the same selected geometry. These values describe the development frontier only and are separated from the fresh validation analysis.

Fresh validation and numerical sensitivity. After selection, the frozen frontier is evaluated on 32 new paired trials, with the reference model and sensor geometries unchanged. Full reduces mean fitted Galerkin action relative to Law by $3 . 5 1 \% , 6 . 9 1 \% , 9 . 3 8 \% , 9 . 3 8 \% .$ , and 12.22% for nominal allowances 1% through 5%, while the corresponding mean scientific-risk increases are 0.050%, 0.304%, 0.636%, 0.636%, and 1.653%. Simultaneous one-sided held-out risk inference supports all five operating points below their nominal caps. Action uncertainty uses 100,000 paired bootstrap resamples conditional on the fixed reference and shared quadrature samples; exploratory simultaneous 95% intervals have positive lower bounds throughout, including $[ 2 . 4 0 \% , \bar { 4 } . 6 1 \% ]$ at 1% and [11.11%, 13.32%] at 5%. One validation pair falls below the prespecified $\mathrm { r E S S } = \mathrm { 0 . 0 5 }$ support threshold, so complete numerical support certification is not claimed; all 32 pairs are retained, and every leave-one-pair-out analysis at 5% preserves an action reduction above 12%. Post-selection re-evaluation with richer Fourier and interaction potential spaces changes the reported reduction by at most 0.096 percentage points relative to the base evaluator; these checks assess stability under the specified numerical perturbations and do not establish Galerkin convergence.

## E.5 UNBALANCED GAUSSIAN STUDY

The finite-measure study uses a two-dimensional three-component Gaussian mixture on $\mathbb { R } ^ { 2 }$ over $t ~ \in ~ [ 0 , 1 ]$ Writing $\dot { s } ( t ) ~ = ~ \sin ^ { 2 } ( \pi t )$ and $\theta _ { j } ( t ) ~ = ~ 2 \pi \bar { j } / 3 + \pi t / 2$ , the component centers are $u _ { j } ( t ) \ = \ 1 . 3 5 ( \cos \theta _ { j } , \sin \theta _ { j } ) + a s ( t ) ( - \sin \theta _ { j } , \cos \theta _ { j } )$ and the common standard deviation is $\sigma ( t ) \stackrel { } { = } 0 . 3 8 + 0 . 0 6 s ( t )$ . Trial-level nuisances $a \stackrel { \cdot } { \sim } U [ - 0 . 3 0 , 0 . 3 0 ] , b _ { 1 } , b _ { 2 } \sim U [ - 0 . 6 5 , 0 . 6 5 ]$ , and $c \sim U [ - 0 . 2 0 , 0 . 2 0 ]$ perturb the intermediate shape and mass while vanishing at the endpoints. With $\beta = b _ { 1 } ( 1 , - 1 , 0 ) + b _ { 2 } ( 1 , 1 , - 2 ) / \sqrt { 3 }$ , the component masses are $m _ { j } ( t ) = e ^ { c s ( t ) }$ softmax $( s ( t ) \beta ) _ { j }$ so the total mass is $M _ { t } = e ^ { c \sin ^ { 2 } ( \pi t ) }$ . The frozen analytical reference sets all four nuisances to zero, giving equal component weights, unit mass, zero reference source, and the same rotating centers and time-dependent variance. No reference model is trained.

Two labeled Gaussian sensors of width 0.45 have centers in $[ - 2 . 2 , 2 . 2 ] ^ { 2 }$ with minimum pairwise separation 0.45. Each trial observes 100 particles from the normalized shape at eleven acquisition times $j / 1 0$ and adds independent detector noise with standard deviation 0.01; log mass is observed separately with noise standard deviation 0.02. Shape moments and log mass are reconstructed independently by endpoint-anchored generalized least squares using three sin $\lfloor ^ { 2 } ( \pi t )$ -weighted polynomial basis functions, after which the normalized shape is information-projected relative to the analytical reference and recombined with the reconstructed mass. Scientific risk is finite-measure Gaussiankernel $\mathrm { M M D ^ { 2 } }$ with bandwidth 0.55, averaged over the ten non-acquisition interior times, so both shape and total-mass errors contribute to the admissibility criterion.

Law, Tangent, GlobalReaction, and Full are selected on 32 common development trials using constrained SLSQP with differentiable objectives and the full 1%–5% Law-relative scientific-risk allowances. Tangent uses the transport–reaction Galerkin correction generated by the two measured observables together with the constant mass coordinate. GlobalReaction, defined in equation 252, retains unrestricted transport but restricts reaction to the spatially uniform mass-rate correction, whereas Full solves the unrestricted screened transport–reaction problem. The authoritative ac tion evaluator uses a cell-centered finite-volume discretization with no density smoothing or floor, float64 arithmetic, diagonally preconditioned conjugate gradients at relative tolerance $1 \bar { 0 } ^ { - 1 0 }$ , and 193 trapezoidal action nodes. The primary $\kappa = 1$ evaluation uses a $9 7 ^ { 2 }$ spatial grid; the $\kappa = 0 . 2 5$ sensitivity evaluation uses $1 2 9 ^ { 2 }$ , while $\kappa = 4$ again uses $9 7 ^ { 2 }$ . Numerical certification checks moment calibration, projected effective sample size, captured Gaussian mass, projected covariance conditioning, screened and balanced PDE residuals, forcing centering, and the Move–Reaction and Tangent–Hidden energy identities.

After the five-point frontiers are frozen, validation uses one new bank of 128 paired trials shared across all methods, allowances, and $\kappa \in \{ 0 . 2 5 , 1 , 4 \}$ , with no pooling of development observations. The primary confirmatory analysis is the $\kappa = 1$ Full-versus-Law comparison. Action uncertainty uses 100,000 paired bootstrap resamples with simultaneous intervals over the five allowances; heldout scientific-risk feasibility is assessed separately using simultaneous one-sided bounds for the paired excess-loss statistic. Tangent and GlobalReaction are retained as separate descriptive fivepoint comparator families. The κ sweep reuses the same physical validation bank so that changes across κ reflect the transport–reaction metric rather than different scientific observations; absolute action values are therefore compared only within a fixed κ.

## E.6 COMPUTING ENVIRONMENT

Experiments were run locally on a system with an Intel Core Ultra 9 275HX processor, comprising 24 physical cores with maximum turbo frequency 5.4 GHz, and an NVIDIA GeForce RTX 5090 Laptop GPU with 24 GB of dedicated GDDR7 memory.

Execution routes differed across studies. JAX-based simulation, reference training, and selected optimization or evaluation stages used GPU acceleration where applicable; empirical information projection and several differentiable proxy calculations used native C++/OpenMP implementations.

The computational environment was hosted in Windows Subsystem for Linux 2 (WSL2), running Ubuntu 24.04.3 LTS with Linux kernel 6.6.87.2-microsoft-standard-WSL2. The principal implementation used Python 3.12.3 and JAX 0.8.3 with 64-bit numerical precision enabled. GPU acceleration was provided through the JAX CUDA 12 backend.

Empirical information projection used a C++ backend where specified by the study protocol. At a given trajectory node, for empirical reference particles $x _ { i }$ with base weights $\bar { w } _ { i }$ and reconstructed target moment bc, the double-precision Newton solver finds a finite multiplier λ satisfying $F ( \lambda ) = 0 ,$ where $p _ { i } ( \lambda )$ is given by equation 43 and

$$
F ( \lambda ) = \sum _ { i = 1 } ^ { N } p _ { i } ( \lambda ) \Phi _ { \eta } ( x _ { i } ) - \widehat { c } , \qquad \nabla _ { \lambda } F ( \lambda ) = \operatorname { C o v } _ { p ( \lambda ) } ( \Phi _ { \eta } , \Phi _ { \eta } ) .\tag{277}
$$

Thus failure to attain the requested moment and ill-conditioning of an otherwise calibrated interior solution are distinct numerical events: the former concerns empirical feasibility or solver convergence, whereas the latter concerns local stability of the multiplier coordinates and implicit derivatives. Multiplier estimates are warm-started between consecutive time nodes, while independent trajectories and linear-algebra operations are parallelized using OpenMP. The native solver is exposed to Python through pybind11 and integrated with JAX through Tesseract (Hafner & Lavin, 2025);¨ Jacobian–vector and vector–Jacobian products use implicit solves with the moment covariance in equation 277 rather than differentiation through the individual Newton iterations.

The C++ extension was compiled in release mode using the GNU C++ 13.3 compiler, the $\mathrm { C } { + } { + } 1 7$ language standard, and the optimization flags -O3 and -march=native. Fast-math optimizations were not enabled in order to preserve the numerical behavior of the double-precision projection solver. The complete software environment is summarized in Table 15.

Table 15: Hardware and software environment used for the experiments.
<table><tr><td>Component</td><td>Configuration</td></tr><tr><td>Operating system</td><td>Ubuntu 24.04.3 LTS under WSL2</td></tr><tr><td>Linux kernel</td><td>6.6.87.2-microsoft-standard-WSL2</td></tr><tr><td>Processor</td><td>Intel Core Ultra 9 275HX (24 cores, up to 5.4 GHz)</td></tr><tr><td>GPU</td><td>NVIDIA GeForce RTX 5090 Laptop GPU</td></tr><tr><td>GPU memory</td><td>24 GB GDDR7</td></tr><tr><td>NVIDIA driver</td><td>581.57</td></tr><tr><td>Python</td><td>3.12.3</td></tr><tr><td>JÁX</td><td>0.8.3</td></tr><tr><td>jaxlib</td><td>0.8.3</td></tr><tr><td>Accelerator backend</td><td>CUDA 12 via the JAX CUDA plugin 0.8.3</td></tr><tr><td>CUDA runtime</td><td>12.9 (JAX-packaged runtime)</td></tr><tr><td>NumPy</td><td>2.5.1</td></tr><tr><td>SciPy</td><td>1.18.0</td></tr><tr><td>Tesseract Core</td><td>tesseract-core 1.11.0</td></tr><tr><td>Tesseract JAX interface</td><td>tesseract-jax0.2.3</td></tr><tr><td>Projection backend</td><td>Native Tesseract C++/OpenMP batched trajectory solver</td></tr><tr><td>C++ language standard</td><td>C++17</td></tr><tr><td>C/C++ compiler</td><td>GCC/G++13.3.0</td></tr><tr><td>Build system</td><td>CMake 4.4.2, release configuration</td></tr><tr><td>Python/C++ bindings</td><td>pybind11 3.1.0</td></tr><tr><td>CPU parallelization</td><td>OpenMP 4.5</td></tr><tr><td>Native compilation</td><td>-03,-march=native,without-ffast-math</td></tr><tr><td>Native numerical precision</td><td>IEEE 754 double precision (f1oat 64)</td></tr><tr><td>Neural optimization</td><td>Custom JAX implementation of Adam</td></tr><tr><td>JAX numerical precision</td><td>64-bit floating point enabled</td></tr><tr><td>Job scheduling</td><td>Local execution; no external scheduler</td></tr></table>