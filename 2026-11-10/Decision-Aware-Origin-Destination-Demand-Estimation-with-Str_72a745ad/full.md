Authors are encouraged to submit new papers to INFORMS journals by means of a style file template, which includes the journal title. However, use of a template does not certify that the paper has been accepted for publication in the named journal. INFORMS journal templates are for the exclusive purpose of submitting to an INFORMS journal and should not be used to distribute the papers in print or online or to submit the papers to another publication.

# Decision Aware Origin-Destination Demand Estimation with Structurally Constrained Data

Xianzhe Wang

Institute of Transportation Studies, University of California, Davis, CA, USA, xzhwang@ucdavis.edu

Yueyue Fan

Department of Civil and Environmental Engineering, University of California, Davis, CA, USA, yyfan@ucdavis.edu (corresponding author)

Johannes O. Royset

Daniel J. Epstein Department of Industrial and Systems Engineering, University of Southern California, CA, USA, royset@usc.edu

Origin-destination (OD) demand plays an important role in transportation system design and manage ment applications. Conventional OD estimation methods typically judge estimation quality based on how estimated demand aligns with observed trafic data and historical demand patterns. We argue that data alignment alone is insuficient; how errors in OD estimates influence downstream decision quality should also be considered. This paper develops a decision aware OD demand estimation framework that leverages structurally constrained trafic data to characterize an admissible demand region, within which we identify OD estimates that would lead to least regret in the downstream decision. Therefore, our OD estimates are both data-consistent and decision-relevant. Methodologically, we approximate the admissible demand region induced by physical network constraints through a geometric representation, which regularizes the search over feasible OD estimates. Within this regularized region, we use a convex surrogate of downstream regret to obtain a tractable decision aware optimization problem. Numerical experiments show that the proposed framework consistently reduces decision regret relative to conventional estimation benchmarks, demonstrat ing the value of aligning OD demand estimation with downstream decision quality.

Key words : OD demand estimation; Decision aware; Structurally constrained data History :

## 1. Introduction

Travel demand serves as a critical parameter in almost all transportation system design and management applications, including trafic assignment, network design, congestion pricing, and infrastructure investment (Patriksson 2015, Yang and H. Bell 1998, De Palma and Lindsey 2011, Rokicki and

Stępniak 2018). In the transportation science literature, the characterization and quantification of travel demand have long been central research objectives. Much of the existing literature on demand estimation focuses primarily on data alignment, where success is measured by how well estimated origin-destination (OD) flows conform to observed measurements or long-run covariates (Bierlaire and Crittin 2004, Krishnakumari et al. 2019, Demissie and Kattan 2022). In most transportation applications, OD demands are not ends in themselves, but rather intermediate inputs to downstream optimization, control, or policy design problems. Despite the central role of travel demand played in downstream applications, established methods for demand estimation remain decision agnostic. In this paper, we begin from the premise that the value of travel demand estimates lies in the quality of the decisions they support, not merely in their fidelity to observed data. Building on this premise, we introduce a new modeling framework for estimating the OD demand that is consistent with observed trafic conditions as well as its impact on the downstream decision-making objectives.

Existing travel demand estimation methods can be broadly categorized into two complementary paradigms: demand forecasting and demand reconstruction. Demand forecasting is for longer-term prediction and is typically grounded in the modeling of socioeconomic factors, such as household characteristics, economic activities, land use and accessibility. that give rise to travel demand (Jin et al. 2014, Pourebrahim et al. 2019, Yao et al. 2020, Yin et al. 2023). Demand reconstruction typically aims to infer an OD demand that best explains observed trafic conditions, thus often explicitly builds on network-based models. In this regard, our study fits better within the category of demand reconstruction, though our estimation framework leaves some flexibility to incorporate predictions from established behavioral models as prior information.

A major challenge in demand reconstruction lies in the underdetermined nature of the inverse network problem (Wang et al. 2023). Data consistency alone provides an incomplete basis for evaluating OD estimates, as multiple demand patterns may be equally compatible with observed measurements. A common approach to address the non-uniqueness of the inferred demand is to incorporate prior OD information, either for the complete set of OD pairs in a network (Cascetta 1984, Castillo et al. 2008) or for a partially observed subset of OD demands (Yang and Fan 2015). More broadly, uniqueness has been achieved by embedding additional structural preferences into the estimation problem, such as entropy maximization (Xie, Kockelman, and Waller 2011, Bar-Gera 2006), sparsity regularization (Wang et al. 2023), and statistical or dynamical assumptions (Maher 1983, Ros-Roca et al. 2022), which efectively shrink the feasible OD set and select a single admissible solution consistent with observed trafic data. All of these approaches are driven primarily by data alignment, with little consideration given to the quality of the downstream decisions that the resulting OD estimates are intended to support.

This paper presents a decision aware framework for OD estimation, shifting the focus from aligning estimates with data to aligning them with the performance of downstream decision-making tasks. To operationalize this decision awareness, we introduce decision regret as a bridge between OD estimation and downstream decision-making. Regret quantifies the loss in downstream objective value, had the estimated demand missed the “true” value. The concept of regret has been widely adopted in operations research as a principled criterion for assessing decision quality, particularly in evaluating the robustness and economic consequences of imperfect information (Bell 1983, Donti, Amos, and Kolter 2017, Elmachtoub and Grigas 2022).

We consider a downstream optimization problem that depends on the OD demand $f .$ Let $\mathcal { F }$ denote the feasible OD demand space, consisting of OD demand vectors that are consistent with the observed network data and satisfy basic feasibility constraints, such as nonnegativity and prescribed lower and upper bounds on OD demand. For any $f \in { \mathcal { F } } .$ , let $w ( f ) \in \arg \operatorname* { m i n } _ { w \in \mathcal { W } } f ^ { \intercal }$ w denote an optimal downstream decision. Given the true OD demand $f _ { \mathrm { t r u e } }$ , the decision regret of an estimated demand $f$ is defined as

$$
R ( f _ { \mathrm { t r u e } } , f ) = { f _ { \mathrm { t r u e } } } ^ { \top } w ( f ) - \operatorname* { m i n } _ { w \in \mathcal { W } } { f _ { \mathrm { t r u e } } } ^ { \top } w .\tag{1}
$$

Since $f _ { \mathrm { t r u e } }$ is unobserved in practice, we use the available prior demand as a proxy for $f _ { \mathrm { t r u e } }$ when constructing the regret criterion. The resulting formulation combines three key elements: link flow observations and feasibility constraints define the admissible OD space ${ \mathcal { F } } ,$ the prior demand provides the reference for regret evaluation, and the downstream decision objective measures the performance of candidate OD estimates.

This research design departs from classical OD estimation in three ways. First, it replaces data fit as the primary criterion with downstream decision performance, directly linking OD estimation with downstream optimization. Second, it recognizes that OD estimation errors difer in their decision impact, giving greater weight to errors that are more consequential for downstream decisions. Third, this perspective treats the non-uniqueness of OD reconstruction not as a deficiency, but as an opportunity to select among feasible OD solutions based on their downstream decision performance.

Certainly, the new estimation framework also raises several challenges. The feasible OD demand space $\mathcal { F }$ induced by structural constraints on the data can be high-dimensional and geometrically complex, while the regret function is generally nonconvex and thus computationally challenging to optimize. Moreover, directly optimizing a linear objective over $\mathcal { F }$ tends to yield an extremepoint solution, which may be overly sensitive to sample-specific noise and thereby increase the risk of overfitting. To address these challenges, we develop a tractable decision aware estimation framework that uses geometric approximations of the feasible OD space to regularize estimation and mitigate sensitivity to extreme-point solutions, and replaces the nonconvex regret function with a convex surrogate. Exploiting duality, we reformulate the resulting problem as a convex quadratically constrained program, enabling scalable identification of structurally admissible OD estimates that support high-quality downstream decisions.

The main contributions of this study are as follows. From a conceptual perspective, we formally introduce a new problem formulation that systematically integrates structurally constrained OD reconstruction and downstream decision evaluation within a unified optimization framework. From a computational perspective, we develop a tractable and scalable decision aware estimation framework. The proposed approach comprises two key components: (i) a geometric ellipsoidal approximation of the structurally admissible OD demand space to address underdetermination and mitigate overfitting; (ii) a convex surrogate of the regret objective that preserves sensitivity to downstream decisions. From an empirical perspective, we develop a computationally eficient and scalable solution method and conduct extensive numerical experiments to validate the proposed framework. The results demonstrate that the decision aware OD estimates consistently yield lower regret in downstream decision problems compared with conventional methods, highlighting the practical value of aligning OD estimation objectives with decision performance.

The remainder of the paper is organized as follows. Section 2 summarizes the current state of knowledge in the broader decision aware estimation literature, and identifies remaining gaps arising from unique challenges of transportation applications. Section 3 develops the decision aware OD estimation framework. It constructs a geometric approximation of the admissible OD demand space from structurally constrained data, formulates the corresponding decision aware estimation problem, and develops surrogate regret formulations that recast this problem as a tractable convex program. Section 4 presents numerical experiments on a range of network settings to evaluate the proposed approach in terms of downstream decision performance. Section 5 concludes with a discussion of implications and directions for future research.

## 2. Relation to the Broader Literature on Decision Aware Estimation

The use of downstream decision models, typically formulated as optimization problems, to inform estimation and learning has a long conceptual history and has attracted renewed attention in recent decades. As early as 1983, Bell (1983) introduces decision regret as a decision level concept based on counterfactual comparisons, laying the conceptual foundation for later decision-based evaluation frameworks, though not explicitly for estimation or prediction. Bengio (1997) articulates a decision centric learning paradigm, arguing that when downstream decision performance is the ultimate objective, minimizing prediction error is inherently suboptimal. Building on this insight, Kao, Roy, and Yan (2009) proposes directed regression, which explicitly embeds directional sensitivity induced by downstream objectives into the estimation process, thereby shifting the evaluation of estimation from standalone predictive accuracy to its consequences for downstream decision quality. Subsequent studies have shown the practical relevance of decision aware estimation across diverse domains, including operational planning, capacity management, and inventory control (Liyanage and Shanthikumar 2005, Bassamboo and Zeevi 2009, Akcay, Biller, and Tayur 2011, Li et al. 2026), consistently finding that gains in predictive accuracy do not necessarily translate into improved decision quality.

In recent years, there has been a surge of interest in systematically incorporating downstream decision models into estimation and learning procedures. A canonical example is the Smart "Predict, then Optimize" (SPO) framework introduced by Elmachtoub and Grigas (2022). For optimization problems with uncertain linear objective coeficients, the framework introduces the SPO loss, which uses decision regret rather than traditional prediction losses (e.g., mean squared error) to penalize the downstream optimality gap, and admits a convex surrogate known as the SPO+ loss. In a sequence of subsequent studies, El Balghiti et al. (2023) establishes finite-sample generalization bounds for the SPO loss function. Building on this line of work, Loke, Tang, and Xiao (2022) proposes a decision-driven regularization framework that provides a unifying perspective on balancing prediction accuracy and decision quality, with the SPO+ loss arising as a special case under specific choices of the regularization parameters. Mandi et al. (2020) demonstrates the empirical efectiveness of the SPO+ loss across a range of combinatorial optimization problems. Beyond seeking surrogate losses for decision regret, a broad stream of work aims to directly approximate the sensitivity of downstream decisions with respect to predicted parameters. Specifically, this line of research focuses on estimating the gradient $\frac { \partial w } { \partial f }$ and leveraging this information to capture how changes in the predictive model afect downstream decisions. The resulting sensitivity information is then used to guide the parameter updates of the estimation model. Some studies restrict attention to structured classes of optimization problems. For example, Amos and Kolter (2017) consider convex quadratic programs and obtain analytical expressions for the decision gradients by diferentiating through the Karush–Kuhn–Tucker (KKT) conditions via the implicit function theorem. Related approaches have been developed for conic optimization problems (Agrawal et al. 2019). However, for general optimization problems, this gradient-based paradigm faces a fundamental challenge because the mapping from problem parameters to optimal decisions can be nonsmooth or unstable; see, for example, Section 5.A of Royset and Wets (2021) for a discussion of the instability of minimizers. In many discrete optimization problems, this mapping is piecewise constant, implying that the corresponding derivatives are zero almost everywhere and undefined at the remaining points. To address this issue, existing approaches introduce regularization to construct a surrogate for the decision gradient (Wilder, Dilkina, and Tambe 2019, Mandi and Guns 2020). Beyond explicitly regularizing the optimization objective, several studies adopt a probabilistic perspective and introduce implicit regularization through randomized decision mappings. In this line of work, the mapping from predictions $f$ to decisions $w$ is modeled as a function that induces a probability distribution over the feasible region W, so that $w ( f )$ can be interpreted as a random variable conditional on $f .$ This formulation enables gradient estimation via perturbation-based sampling schemes (Pogančić et al. 2019, Blondel, Martins, and Niculae 2020). A substantial body of subsequent work has examined the statistical properties, surrogate loss constructions, and computational tractability of decision aware objectives, with demonstrated applicability to a broad class of downstream decision problems. We refer the reader to recent survey papers in this area for a comprehensive overview (Sadana et al. 2025, Mandi et al. 2024, Qi and Shen 2022).

Limitations and gaps in transportation applications. While these paradigms have substantially advanced the integration of estimation and optimization, they remain fundamentally misaligned with the structure of transportation network-level OD demand estimation problems. Here, OD demand estimation is characterized by high-dimensional feasible demand sets induced by physical network constraints, flow conservation, and behavioral assumptions, together with limited and indirect observations. In such settings, OD demand is generically non-identifiable: many distinct demand patterns are observationally equivalent, yet can give rise to sharply diferent downstream decisions. Existing decision aware approaches, however, largely retain a point estimation perspective, implicitly framing estimation as the recovery of a single latent “true” parameter vector and interpreting decision losses as the consequence of small, local deviations around this point. This viewpoint is ill-suited to OD estimation, where uncertainty is not concentrated around a neighborhood of a unique solution, but is distributed over a large, structured feasible region. The proposed framework integrates structurally constrained OD estimation with decision aware optimization to fill the gaps.

## 3. Decision Aware OD Estimation Framework

This section develops a decision aware framework for OD demand estimation in transportation networks. In Section 3.1, we characterize the structurally underdetermined OD estimation problem induced by network flow constraints and construct a tractable ellipsoidal approximation of the admissible OD space. Building on this geometric representation, in Section 3.2, we then formulate a decision aware OD estimation framework that explicitly incorporates downstream optimization problems and defines estimation objectives in terms of expected downstream decision regret under a given error distribution of the true demand. Section 3.3 introduces a convex surrogate for the generally nonconvex regret objective, yielding tractable surrogate regret formulations. Finally, Section 3.4 derives deterministic equivalent reformulations of the resulting models, showing that the formulation over the original feasible OD region reduces to a linear program, whereas its ellipsoidal counterpart admits an equivalent convex quadratically constrained programming formulation.

## 3.1. Geometric Approximation from Structurally Constrained Data

A defining feature of transportation systems is that many observed quantities are governed by physical conservation laws imposed by the network structure. In particular, when trafic moves through a fixed set of links and originates from a collection of OD pairs, the relationship between link flows and OD flows is fully encoded by the link-OD incidence matrix. Let $X \in \mathbb { R } ^ { m \times n }$ denote this matrix, where $m$ is the number of directed links and n is the number of OD pairs. Given a vector of reliable link-flow measurements $L \in \mathbb { R } ^ { m }$ , the underlying OD flow vector $f \in \mathbb { R } ^ { n }$ must satisfy the fundamental physical constraint

$$
L = X f .\tag{2}
$$

Equation (2) captures the conservation of vehicles on every link of the network: the total number of vehicles that traverse a link must equal the aggregate contributions of all OD pairs that utilize that link.

For realistic networks, the number of OD pairs far exceeds the number of links, $\mathrm { i . e . , ~ } n \gg m$ Consequently, the system of equations in (2) is underdetermined. For any observed link-flow vector $L ,$ there exist infinitely many estimators for $f ,$ but they must respect the feasible set defined by (2). Put diferently, even though the OD flows are unobserved and often highly uncertain, they are never arbitrary: they are restricted to lie within the afine subspace of $\mathbb { R } ^ { n }$ determined by the measured link flows and the physical network structure.

For a linear system $X f = L$ where X is rank deficient $( n \gg m )$ , we highlight two remarks that are directly relevant to this study. First, when $X ^ { \top } X$ is nonsingular (the full rank case), Moore-Penrose pseudoinverse $X ^ { + }$ is equivalent to $X ^ { + } = ( X ^ { \top } X ) ^ { - 1 } X ^ { \top }$ , where $X ^ { - 1 }$ is the ordinary inverse of matrix X. Second, the solution $X ^ { + } L$ , calculated based on the Moore-Penrose pseudoinverse, has the minimum length among all feasible solutions to $X f = L$ (see Fan and Kalaba (2003), Rao and Mitra (1971) for more comprehensive discussions). Within the broader feasible OD demand space ${ \mathcal { F } } ,$ we consider the subset S characterized by consistency with the observed link flows. With this notation, the full solution set to the normal equation is

$$
S = X ^ { + } L + \ker ( X ^ { \top } X ) = ( X ^ { + } L + \{ v \in \mathbb { R } ^ { n } : X ^ { \top } X v = 0 \} .\tag{3}
$$

This decomposition reveals that all feasible OD flow vectors consist of a canonical minimum norm solution plus an arbitrary vector in the nullspace of $X ^ { \top } X$ , which captures all OD flow directions that do not afect the observed link flows.

Let $q = \mathrm { r a n k } ( X )$ and let ${ \bar { f } } = X ^ { + } L$ denote the particular least squares solution. We express the feasible set for the demand estimator as

$$
S ^ { \prime } = \{ \bar { f } + V z \ | \ 0 \leq \bar { f } + V z \leq U , z \in \mathbb { R } ^ { n - q } \ \} ,
$$

where $V \in \mathbb { R } ^ { n \times ( n - q ) }$ is the matrix whose columns form an orthonormal basis for ker $( X ^ { \top } X )$ , and $U$ sets an upper bound. This representation follows directly from the fact that the nullspace of $X ^ { \top } X$ contains all directions that leave the link flows unchanged; hence, any feasible $f$ difers from $\bar { f }$ only through motions in this subspace.

In a meaningful size of network problem, the feasible region $S ^ { \prime }$ constitutes a high-dimensional polyhedral set. Because direct characterization of its size and shape is computationally intractable, we approximate this feasible region by an ellipsoid, which provides a tractable representation of its geometric extent and uncertainty. Following the approach of Renegar (2001) and Chen, Ordónez, and Palmer (2006), we construct an ellipsoidal approximation centered at the analytic-center of $S ^ { \prime }$ . Let $z ^ { \star }$ denote the analytic-center parameter, defined by the solution to the logarithmic barrier problem

$$
\operatorname* { m i n } _ { z } - \sum _ { i = 1 } ^ { n } \Bigl [ \log ( U _ { i } - \bar { f } _ { i } - ( V z ) _ { i } ) + \log ( \bar { f } _ { i } + ( V z ) _ { i } ) \Bigr ] .
$$

For notational simplicity, define the vector $g ( z ) = \bar { f } + V z$ , so that the objective becomes $\phi ( z ) =$ $\begin{array} { r } { { } \cdot \sum _ { i = 1 } ^ { n } \Big \lceil \log ( U _ { i } - g _ { i } ( z ) ) + \log ( g _ { i } ( z ) ) \Big \rceil } \end{array}$

Let $H$ be the Hessian of $\phi ( z )$ at $z ^ { \star }$ . Since $g ( z ) = \bar { f } + V z$ is afine, its Jacobian is $V ,$ and the Hessian takes the explicit form

$$
H = V ^ { \top } \left[ \mathrm { d i a g } \left( \frac { 1 } { g _ { i } ( z ^ { \star } ) ^ { 2 } } + \frac { 1 } { ( U _ { i } - g _ { i } ( z ^ { \star } ) ) ^ { 2 } } \right) \right] V .
$$

This Hessian defines the local curvature of the logarithmic barrier and yields a natural ellipsoidal approximation of $S ^ { \prime }$ centered at the analytic-center $f ^ { \star } = g ( z ^ { \star } )$ . Following Renegar (2001), define the H-norm ball $B ( z ^ { \star } , r ) = \big \{ z ~ | ~ ( z - z ^ { \star } ) ^ { \top } H ( z - z ^ { \star } ) \leq r ^ { 2 } \big \}$ . Renegar’s analysis establishes that, for self-concordant logarithmic barriers,

$$
B ( z ^ { \star } , 1 ) \subseteq \{ z | 0 \leq \bar { f } + V z \leq U \} \subseteq B ( z ^ { \star } , 4 \nu + 1 ) ,
$$

where $\nu$ is the complexity parameter of the barrier function. In our case, the barrier consists of $2 n$ logarithmic terms, and therefore $\nu = 2 n$

Mapping these sets back to the OD flow space via the afine transformation $f = \bar { f } + V z$ yields $\bar { f } + V B ( z ^ { \star } , 1 ) \ \subseteq \ S ^ { \prime } \ \subseteq \ \bar { f } + V B ( z ^ { \star } , 4 \nu + 1 )$ . Since V acts only on the displacement variable, we also have the equivalent representation $\bar { f } + V B ( z ^ { \star } , r ) = f ^ { \star } + V B ( 0 , r )$ . This ellipsoidal sandwiched approximation provides a computationally tractable geometric surrogate for the true feasible set $S ^ { \prime }$ Figure 1 geometrically illustrates how the OD flow uncertainty set emerges from the nullspace of the link–OD incidence matrix and how the ellipsoidal approximation captures the feasible variation directions.

![](images/f123c59622561cd02687235019d9f290011bc8f94a985ab358de1377cbb47503.jpg)  
Figure 1 Schematic representation of the geometric structure induced by the link-OD incidence matrix. The outer region depicts the nullspace of $X ^ { T } X$ . The rectangular region represents the bounded feasible set $S ^ { \prime } =$ $S \cap [ 0 , U ]$ . The ellipses illustrate the inscribed and circumscribed ellipsoidal approximations $\bar { f } + V B ( z ^ { \star } , 1 )$ and $\bar { f } + V B ( z ^ { \star } , 4 \nu + 1 )$ . The figure also highlights the true OD flow $f _ { \mathrm { t r u e } }$ , the least squares estimate ${ \bar { f } } ,$ and the analytic-center estimate $f ^ { \star }$

The representation $f = \bar { f } + V z$ highlights that all admissible OD flows in $S ^ { \prime }$ difer only through movements in the nullspace directions encoded by $z .$ These directions do not afect the physical constraint $L = X f ,$ , since $X V = 0$ , and, when z remains within the ball $B ( 0 , r )$ , the resulting f automatically respects the box constraints $0 \leq f \leq U$ . Thus, z captures the “degrees of freedom” available for adjusting $f$ without violating either the physical conservation constraints or the OD bounds. This flexibility allows downstream decision models to choose z in a way that optimizes performance while remaining within the feasible ellipsoidal region.

In particular, the ellipsoidal constraint

$$
z ^ { \top } H z \leq r ^ { 2 } ,\tag{4}
$$

where $r \geq 1$ controls the radius of the search region, provides a compact and tractable characterization of the controllable degrees of freedom available for downstream optimization. The choice $r = 1$ guarantees that the ellipsoid is contained in $S ^ { \prime }$ . As the nullspace dimension grows, however, the unit ellipsoid may capture only a small portion of the admissible OD variation and thus become overly conservative. We therefore treat r as a tuning parameter: increasing r expands the ellipsoidal search region and allows a broader range of decision-relevant demand patterns to be explored. Although $r > 1$ relinquishes the automatic containment guarantee, the nullspace construction continues to preserve flow conservation because $X V = 0 ;$ ; only the lower and upper bounds on OD demand may be violated. Such violations can be tolerated when modest, while the original bounds can be imposed explicitly whenever strict feasibility is required. Appendix A analyzes the efect of nullspace dimension and discusses the calibration of r.

This controlled expansion is useful for decision aware estimation because the goal is to reduce downstream regret rather than preserve statistical fidelity alone. The resulting estimate therefore need not be unbiased, and small violations of the data-derived OD bounds may be justified if they improve downstream decisions. For example, suppose an intervention is triggered when estimated demand exceeds 25. If failing to intervene when actual demand exceeds this threshold is more costly than intervening unnecessarily, a modest upward bias near the threshold may be preferable to a comparable downward bias. A larger r provides the flexibility for such decision driven adjustments, while the regret objective determines their direction and magnitude.

## 3.2. Optimization Formulation for Decision Aware OD Estimation

Consider the following decision problem that takes OD flow $f$ as an input:

$$
\operatorname* { m i n } _ { w \in \mathcal { W } } f ^ { \top } w ,\tag{5}
$$

where w denotes the downstream decision and W is a nonempty compact polyhedron defined by the linear operational constraints of the downstream problem. We denote the optimal objective value of this problem by $\begin{array} { r } { c ( f ) = \operatorname* { m i n } _ { w \in \mathcal { W } } f ^ { \top } w } \end{array}$ , and denote by $w ( f ) \in \arg \operatorname* { m i n } _ { w \in \mathcal { W } } f ^ { \intercal }$ w an optimal decision under $f .$ Note that, by duality, $f$ can equivalently appear either as coeficients in the objective function or as right-hand-side constants in the constraints. This formulation captures many transportation applications in which OD flows drive payofs, costs, or service levels, including assignment problems (e.g., allocating vehicles to OD pairs), matching problems (e.g., pairing demand and supply across zones), and transportation or freight flow problems with fixed capacity constraints.

If the true OD flow is $f _ { \mathrm { t r u e } }$ but the downstream optimization problem is solved using an estimated OD flow vector $f ,$ the resulting decision generally yields a suboptimal objective value. The associated loss is measured by the regret defined in (1), which captures the degradation in decision quality induced by OD estimation error. In practice, $f _ { \mathrm { t r u e } }$ is fixed but unobserved. Suppose that prior information provides an estimate $f _ { 0 }$ of the true OD flow. We model the unknown discrepancy between $f _ { 0 }$ and $f _ { \mathrm { t r u e } }$ by a random error ε with distribution ${ \mathcal { P } } _ { : }$ and represent the uncertain true OD flow as

$$
\begin{array} { r } { F = f _ { 0 } + \varepsilon , \qquad \varepsilon \sim \mathcal { P } . } \end{array}
$$

Here, $f _ { 0 }$ is fixed and observed, whereas ε and hence $F$ are random. For now, we assume that the error distribution $\mathcal { P }$ is known. Appendix B describes how $\mathcal { P }$ can be constructed from historical prediction residuals and contextual information, including historical demand and socioeconomic features.

Decision aWare Estimation (DWE) therefore seeks to minimize the expected regret under the uncertainty in the true OD flow:

$$
\operatorname* { m i n } _ { g ( z ) \in S ^ { \prime } } \mathbb { E } _ { \varepsilon \sim \mathcal { P } } \left[ R \big ( f _ { 0 } + \varepsilon , g ( z ) \big ) \right] .\tag{DWE-original}
$$

Recall that $S ^ { \prime }$ is the feasible OD set induced by link-flow conservation and OD bounds, and that $g ( z ) = \bar { f } + V z$ parameterizes a candidate OD estimate in $S ^ { \prime }$

Within the ellipsoidal approximation of the feasible region, the DWE formulation becomes

$$
\operatorname* { m i n } _ { z : z ^ { \top } H z \leq r ^ { 2 } } \mathbb { E } _ { \varepsilon \sim \mathcal { P } } \left[ R \big ( f _ { 0 } + \varepsilon , g ( z ) \big ) \right] .\tag{DWE-Ellipsoid}
$$

For applications where tail risk plays an important role, a more conservative formulation replaces the expectation with the superquantile, also known as Conditional Value at Risk (CVaR), where $\beta \in ( 0 , 1 )$ denotes the confidence level:

$$
\operatorname* { m i n } _ { \substack { z : z ^ { \top } H z \leq r ^ { 2 } } } \mathrm { C V a R } _ { \beta , \varepsilon \sim \mathcal { P } } \left[ R \big ( f _ { 0 } + \varepsilon , g ( z ) \big ) \right] .\tag{6}
$$

## 3.3. Surrogate Regret Formulations

The regret defined in (1) is generally nonconvex in the estimated OD flow $f .$ Therefore, directly minimizing its expectation or a risk measure with respect to $f$ (or equivalently with respect to z through $f = \bar { f } + V z )$ leads to a nonconvex optimization problem.

Elmachtoub and Grigas (2022) shows that, for linear optimization problems, the regret admits a tractable convex upper bound known as the $S P O +$ loss. We follow their construction and retain the auxiliary scaling parameter $\alpha > 0$ that arises in the derivation of the $\mathrm { S P O + }$ upper bound, yielding a parameterized surrogate denoted by $\ell _ { \alpha }$ . We discuss the role and choice of α later in this section. Let $f ^ { * }$ denote a realization of the uncertain true OD flow $F = f _ { 0 } + \varepsilon$ . Given $f ^ { * }$ and a candidate OD estimate $f ,$ the parameterized surrogate is defined as

$$
\ell _ { \alpha } ( f ^ { * } , f ) : = \operatorname* { m a x } _ { w \in \mathcal { W } } ( f ^ { * } - \alpha f ) ^ { \top } w + \alpha f ^ { \top } w ( f ^ { * } ) - { f ^ { * } } ^ { \top } w ( f ^ { * } ) .\tag{7}
$$

The following proposition summarizes the properties of $\ell _ { \alpha }$ in our notation. Appendix C provides the proof.

Proposition 1 (Properties of $\ell _ { \alpha } )$ . For any $\alpha > 0$ and any realized OD flow vector $f ^ { * }$ , the surrogate $\ell _ { \alpha }$ satisfies:

1. Upper bound on regret:

$$
R ( f ^ { * } , f ) \leq \ell _ { \alpha } ( f ^ { * } , f ) \qquad f o r \ a l l \ f \in \mathbb { R } ^ { n } ;
$$

2. Convexity: $\ell _ { \alpha } ( f ^ { * } , f )$ is a convex function of $f ,$

## 3. Subgradient: for any $f ,$

$$
\alpha \big ( w ( f ^ { * } ) - w ( \alpha f - f ^ { * } ) \big ) \in \partial _ { f } \ell _ { \alpha } ( f ^ { * } , f ) .
$$

Proposition 1 establishes several fundamental properties of the $\ell _ { \alpha }$ surrogate, including its tractability and its ability to provide an upper bound on the true regret associated with a downstream linear decision problem.

The classical $\mathrm { S P O + }$ construction of Elmachtoub and Grigas (2022) sets $\alpha = 2$ because, within the empirical risk minimization (ERM) framework, this choice yields a convex surrogate that is Fisher consistent, meaning that, under the stated conditions, minimizing the population $\mathrm { S P O + }$ risk also minimizes the true SPO risk. However, our approach does not rely on such distributional consistency. Rather than learning a direct prediction f through ERM, we optimize the parameterized surrogate over a physically constrained feasible OD space. Under this paradigm, the desirable value of α is not dictated by Fisher consistency of an ERM estimator, but by the quality of the resulting surrogate and its downstream decision performance, thereby allowing greater flexibility in its choice.

The surrogate objective $\ell _ { \alpha } ( f ^ { * } , f )$ is afected by two distinct sources of discrepancy: (i) surrogate approximation, arising from replacing the original regret with a tractable convex upper bound; and (ii) uncertainty perturbation, arising because the surrogate is evaluated at a realization $f ^ { * }$ of the uncertain OD flow rather than at the unobserved true flow $f _ { \mathrm { t r u e } }$ . These two efects interact with the choice of α. The following proposition provides an explicit bound that characterizes this tradeof.

Proposition 2 (Tradeof Bound for the $\ell _ { \alpha }$ Surrogate). Let $f ^ { * }$ be a realization of the uncertain OD flow F. Suppose that W is compact and that there exist constants $U > 0$ and $B _ { w } > 0$ such that $\| f \| , \| f ^ { * } \| , \| f _ { \mathrm { t r u e } } \| \leq U$ and $\| \boldsymbol { w } \| \le B _ { w }$ for all $w \in \mathcal { W }$ . Further suppose that, for the candidate OD estimate f, there exists $\kappa > 0$ such that $f ^ { \top } ( w - w ( f ) ) \geq \kappa \Vert w - w ( f ) \Vert$ for all $w \in \mathcal { W }$

For any $\alpha > 0$ , the discrepancy between the surrogate and the true regret admits the decomposition

$$
\ell _ { \alpha } ( f ^ { * } , f ) - R ( f _ { \mathrm { t r u e } } , f ) = \underbrace { \ell _ { \alpha } ( f _ { \mathrm { t r u e } } , f ) - R ( f _ { \mathrm { t r u e } } , f ) } _ { s u r r o g a t e \ a p p r o x i m a t i o n } + \underbrace { \ell _ { \alpha } ( f ^ { * } , f ) - \ell _ { \alpha } ( f _ { \mathrm { t r u e } } , f ) } _ { u n c e r t a i n t y \ p e r t u r b a t i o n } .
$$

Moreover,

$$
\ell _ { \alpha } ( f ^ { * } , f ) - R ( f _ { \mathrm { t r u e } } , f ) \leq \frac { 2 U ^ { 2 } B _ { w } } { \kappa \alpha } + 2 \alpha B _ { w } \left( \| f - f _ { \mathrm { t r u e } } \| + \| f ^ { * } - f \| \right) + 2 B _ { w } \| f ^ { * } - f _ { \mathrm { t r u e } } \| .\tag{8}
$$

$I f \parallel f - f _ { \mathrm { t r u e } } \parallel + \parallel f ^ { * } - f \parallel > 0$ , the α-dependent part of the bound is minimized at

$$
\alpha ^ { * } = \frac { U } { \sqrt { \kappa \left( \left\| f - f _ { \mathrm { t r u e } } \right\| + \left\| f ^ { * } - f \right\| \right) } } .\tag{9}
$$

The detailed proof is given in Appendix D.

Discussion. Proposition 2 shows that α balances surrogate approximation against sensitivity to OD flow discrepancies. In (8), the approximation term decreases as $1 / \alpha$ , whereas the discrepancydependent term increases linearly with α. Hence, a small α yields a looser regret approximation, while a large α makes the surrogate more sensitive to diferences among the candidate, realized, and true OD flows. The remaining uncertainty term is independent of α. Equation (9) summarizes this balance. Larger OD-flow discrepancies favor a smaller α because their contribution to the bound is amplified linearly by $\alpha .$ . In contrast, a smaller κ favors a larger α: when the downstream optimum is less sharply separated, a larger scaling factor is needed to control the surrogate-approximation term. Since $f _ { \mathrm { t r u e } }$ is unobserved, $\alpha ^ { * }$ serves as an oracle benchmark; in practice, α is selected based on downstream decision performance in Section 4.

By replacing $R ( f _ { 0 } + \varepsilon , g ( z ) )$ in the objectives of (DWE-original) and (DWE-Ellipsoid) with its convex surrogate $\ell _ { \alpha } ( f _ { 0 } + \varepsilon , g ( z ) )$ , and noting that $g ( z ) = \bar { f } + V z$ is afine in $z ,$ the resulting optimization problems are convex in z and can be solved using standard convex optimization methods.

Theorem 1 (Convexity and Subgradient of the Expectation Formulation). Let $H \in$ $\mathbb { R } ^ { d \times d }$ be positive definite, and consider

$$
\operatorname* { m i n } _ { z \in \mathbb { R } ^ { d } : z ^ { \top } H { z } \le r ^ { 2 } } \phi ( z ) : = \mathbb { E } _ { \varepsilon \sim \mathcal { P } } \left[ \ell _ { \alpha } ( f _ { 0 } + \varepsilon , g ( z ) ) \right] .
$$

Assume that $\ell _ { \alpha } ( f _ { 0 } + \varepsilon , g ( z ) )$ is integrable for every feasible z, and that the subgradient expression below is integrable with respect to $\varepsilon \sim \mathcal { P }$ . Then:

1. $\phi ( z )$ is convex in z, and the optimization problem is a convex optimization problem;

2. a subgradient of ϕ at z is given by

$$
\begin{array} { r } { \mathbb { E } _ { \varepsilon \sim \mathcal { P } } \left[ \alpha V ^ { \top } \left( w ( f _ { 0 } + \varepsilon ) - w \big ( \alpha g ( z ) - f _ { 0 } - \varepsilon \big ) \right) \right] \in \partial \phi ( z ) . } \end{array}
$$

Theorem 1 shows that the expectation formulation is a convex optimization problem. Since $\ell _ { \alpha } ( f ^ { * } , f )$ is convex but generally nonsmooth in its second argument, $\phi ( z )$ is convex and may also be nondiferentiable. By Proposition 1, for each realization $f _ { 0 } + \varepsilon$ , a subgradient with respect to the second argument is $\alpha [ w ( f _ { 0 } + \varepsilon ) - w ( \alpha g ( z ) - f _ { 0 } - \varepsilon ) ]$ ]. Since $g ( z )$ is afine, multiplication by $V ^ { \top }$ gives the corresponding subgradient with respect to z, and taking expectations yields the expression in Theorem 1. An analogous argument applies to $\mathrm { C V a R ; }$ see Appendix F for a formal statement and proof.

## 3.4. Equivalent Reformulations and Computational Solution

We next show that the surrogate DWE formulations admit standard deterministic optimization reformulations and can therefore be solved directly using of-the-shelf optimization software. To obtain a finite-dimensional computational representation, we approximate the prior distribution of the true OD flow by a discrete scenario distribution. Specifically, let $\tau _ { k } : = f _ { 0 } + \varepsilon _ { k } , k = 1 , \ldots , K$ denote the representative OD-flow realizations, with associated probabilities $p _ { k }$ , where $\begin{array} { r } { \sum _ { k = 1 } ^ { K } p _ { k } = 1 } \end{array}$ This yields the discrete approximation $\begin{array} { r } { \widehat { \mathcal { P } } = \sum _ { k = 1 } ^ { K } p _ { k } \delta _ { \tau _ { k } } } \end{array}$ , where $\delta _ { \tau _ { k } }$ denotes a point mass at $\tau _ { k } .$ . We further define $\begin{array} { r } { \bar { \boldsymbol { w } } : = \mathbb { E } _ { \tau \sim \widehat { \mathcal { P } } } [ \boldsymbol { w } ( \tau ) ] = \sum _ { k = 1 } ^ { K } p _ { k } \boldsymbol { w } ( \tau _ { k } ) } \end{array}$ ,, which is the probability-weighted average of the optimal downstream decisions induced by the prior OD-flow realizations.

Under the finite-scenario representation of the prior distribution, substituting the definition of $\ell _ { \alpha } ( \tau _ { k } , f )$ in (7) yields

$$
\sum _ { k = 1 } ^ { K } p _ { k } \ell _ { \alpha } ( \tau _ { k } , f ) = \sum _ { k = 1 } ^ { K } p _ { k } \operatorname* { m a x } _ { w \in \mathcal { W } } ( \tau _ { k } - \alpha f ) ^ { \top } w + \alpha f ^ { \top } \bar { w } - \sum _ { k = 1 } ^ { K } p _ { k } \tau _ { k } ^ { \top } w ( \tau _ { k } ) .\tag{10}
$$

The last term is independent of the estimation decision f and can therefore be omitted without afecting the optimizer.

We now exploit the linear structure of the downstream decision problem. Suppose that its feasible region admits the general polyhedral representation

$$
\mathcal { W } = \left\{ w \in \mathbb { R } ^ { n } : A w \leq b , B w = d \right\} ,
$$

where the equality constraints capture, for example, flow-conservation or balance requirements, while capacity constraints, resource restrictions, and variable bounds can be incorporated into $A w \leq b$

For each prior scenario $k ,$ the inner optimization problem is therefore

$$
\operatorname* { m a x } _ { w } \left\{ ( \tau _ { k } - \alpha f ) ^ { \top } w : A w \leq b , B w = d \right\} .
$$

Let $\lambda _ { k } \geq 0$ denote the dual variables associated with the inequality constraints and let $\mu _ { k }$ denote the unrestricted dual variables associated with the equality constraints. Assuming the downstream problem is feasible and bounded, strong linear programming duality gives

$$
\operatorname* { m a x } _ { \boldsymbol { w } } \left\{ \left( \tau _ { k } - \alpha f \right) ^ { \top } \boldsymbol { w } : \boldsymbol { A } \boldsymbol { w } \leq \boldsymbol { b } , \boldsymbol { B } \boldsymbol { w } = \boldsymbol { d } \right\} = \operatorname* { m i n } _ { \boldsymbol { \lambda } _ { k } \geq 0 , \mu _ { k } } \left\{ \boldsymbol { b } ^ { \top } \boldsymbol { \lambda } _ { k } + \boldsymbol { d } ^ { \top } \boldsymbol { \mu } _ { k } : \boldsymbol { A } ^ { \top } \boldsymbol { \lambda } _ { k } + \boldsymbol { B } ^ { \top } \boldsymbol { \mu } _ { k } = \tau _ { k } - \alpha f \right\} .
$$

Thus, the nested optimization in the surrogate loss can be eliminated, yielding a single-level deterministic equivalent formulation.

For the original feasible OD region $S ^ { \prime } { } _ { ; }$ DWE-original is equivalent, up to an additive constant independent of the decision variables, to

$$
\begin{array} { r l l } { \displaystyle \operatorname* { m i n } _ { f , \{ \lambda _ { k } , \mu _ { k } \} _ { k = 1 } ^ { K } } } & { \alpha f ^ { \top } \bar { w } + \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } \left( b ^ { \top } \lambda _ { k } + d ^ { \top } \mu _ { k } \right) } \\ { \mathrm { s . t . } } & { A ^ { \top } \lambda _ { k } + B ^ { \top } \mu _ { k } = \tau _ { k } - \alpha f , } & { k = 1 , \ldots , K , } \\ & { \lambda _ { k } \geq 0 , } & { k = 1 , \ldots , K , } \\ & { X f = L , } \\ & { 0 \leq f \leq U . } \end{array}\tag{DWE-original-DE}
$$

The objective and all constraints in DWE-original-DE are linear. Hence, the finite-scenario surrogate formulation of DWE-original is a linear program.

The same dual reformulation applies to DWE-Ellipsoid. Substituting $f = g ( z ) = { \bar { f } } + V z$ yields

$$
\begin{array} { r l r } { \underset { z , \{ \lambda _ { k } , \mu _ { k } \} _ { k = 1 } ^ { K } } { \operatorname* { m i n } } } & { \alpha ( \bar { f } + V z ) ^ { \top } \bar { w } + \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } \left( \boldsymbol { b } ^ { \top } \boldsymbol { \lambda } _ { k } + d ^ { \top } \boldsymbol { \mu } _ { k } \right) } \\ { \mathrm { s . t . } } & { A ^ { \top } \boldsymbol { \lambda } _ { k } + B ^ { \top } \boldsymbol { \mu } _ { k } = \tau _ { k } - \alpha ( \bar { f } + V z ) , } & { k = 1 , \ldots , K , } \\ & { \boldsymbol { \lambda } _ { k } \geq 0 , } & { k = 1 , \ldots , K , } \\ & { \quad \quad \quad z ^ { \top } H z < r ^ { 2 } . } \end{array}\tag{DWE-Ellipsoid-DE}
$$

Since $H \succ 0$ , the ellipsoidal constraint is convex, while the objective and all remaining constraints are linear. Thus, DWE-Ellipsoid-DE can be viewed as a special case of a convex quadratically constrained quadratic program (QCQP), with a linear objective function and a single convex quadratic constraint. Moreover, letting $H = R ^ { \top } R$ denote a Cholesky factorization, the ellipsoidal constraint can equivalently be expressed as $\| R z \| _ { 2 } \leq r$ . Hence, DWE-Ellipsoid-DE also admits an equivalent second-order cone programming (SOCP) representation.

The resulting formulations belong to standard classes of convex optimization problems. DWEoriginal-DE reduces to an LP, whereas DWE-Ellipsoid-DE can be formulated as either a convex QCQP or an SOCP. Both formulations can be handled directly by modern optimization solvers, thereby avoiding the need for a customized nonsmooth optimization algorithm.

## 4. Numerical Experiments

Numerical experiments based on three test networks of varying sizes are conducted to test and validate the proposed decision aware OD estimation framework. Besides checking the accuracy and scalability of our method, we also examine how decision regret varies with model misspecification and data noise, and show that the proposed framework efectively reduces decision error and adapts the surrogate parameter α as the error structure changes.

## 4.1. An Illustrative Toy Example for Motivating Decision-awareness

This simple numerical example is designed to motivate the need for decision awareness and to illustrate the interaction between various pulling sources, including prior information presented by historical samples, physical feasibility based on current sensor observations, and downstream decision quality, in an underdetermined OD estimation problem.

Consider a toy network with three nodes, three OD flows, and two directed links, as shown in Figure 2. Let $f = ( f _ { 1 2 } , f _ { 2 3 } , f _ { 1 3 } ) ^ { \intercal }$ denote the OD demand vector. Suppose that only link (1, 2) is instrumented with a sensor, and the observed link flow is $L _ { 1 2 } = 1 0 0$ . Under the path structure in Figure 2, this observation implies the linear constraint $f _ { 1 2 } + f _ { 1 3 } = 1 0 0$ . In addition, assume simple capacity bounds $0 \leq f _ { 1 2 } , f _ { 2 3 } , f _ { 1 3 } \leq 2 0 0$ . These constraints form the feasible OD set, and any point within this feasible set aligns with the observed data.

![](images/4efa5d94d70ce74eb6a480db2d3d933c37fe064933d5133d372a23d8a8deaa66.jpg)

![](images/5de2689caf6b49a771f6dd33a36b3a2bd8695fd177840917f2fb3d62955b7676.jpg)  
Physical Constraints  
Figure 2 Illustrative three-node network with three OD flows, a single sensor observation, and the induced physical constraint.

Now let us illustrate how OD estimation afects downstream decisions. Consider the following simple decision problem:

$$
\begin{array} { r l } { \underset { \mathbf { w } \in \mathbb { R } ^ { 3 } } { \operatorname* { m i n } } } & { f _ { 1 2 } w _ { 1 } + f _ { 2 3 } w _ { 2 } + f _ { 1 3 } w _ { 3 } } \\ { \mathrm { ~ s . t . ~ } } & { w _ { 1 } + w _ { 2 } + w _ { 3 } = 1 , } \\ & { w _ { 1 } - w _ { 3 } = 0 , } \\ & { 0 \leq w _ { 1 } , w _ { 2 } , w _ { 3 } \leq 1 . } \end{array}\tag{11}
$$

With symmetry between $w _ { 1 }$ and $w _ { 3 }$ , problem (11) reduces to comparing two efective actions. The derived decision rule is

$$
\left\{ \begin{array} { l l } { w = ( 0 , 1 , 0 ) , } & { \mathrm { i f ~ } f _ { 2 3 } \leq \displaystyle \frac { f _ { 1 2 } + f _ { 1 3 } } { 2 } , } \\ { w = \left( \displaystyle \frac { 1 } { 2 } , 0 , \displaystyle \frac { 1 } { 2 } \right) , } & { \mathrm { i f ~ } f _ { 2 3 } > \displaystyle \frac { f _ { 1 2 } + f _ { 1 3 } } { 2 } . } \end{array} \right.
$$

For OD estimates that satisfy the sensor information $f _ { 1 2 } + f _ { 1 3 } = 1 0 0$ , this decision boundary reduces to $f _ { 2 3 } = 5 0$ . Therefore, the downstream action depends only on whether $f _ { 2 3 }$ lies below or above the threshold value 50. This is precisely the type of setting in which a decision aware estimator can difer sharply from a prediction-oriented one: not all directions of OD error are equally consequential, and errors that move the estimate across the decision boundary are much more important than errors that remain decision-neutral.

We set the true OD vector to $f ^ { \star } = ( 3 5 , 4 9 , 6 5 ) ^ { \top }$ , which satisfies the physical constraint $f _ { 1 2 } + f _ { 1 3 } =$ 100. Under the true OD vector, min $\{ f _ { 2 3 } ^ { \star } , 5 0 \} = \operatorname* { m i n } \{ 4 9 , 5 0 \} = 4 9$ , so the optimal solution is the $f _ { \mathrm { 2 3 } } \mathrm { - d r i v e n }$ action, i.e., $w = ( 0 , 1 , 0 )$ . We then construct a dataset of 300 historical OD samples and consider the sample mean as the simplest benchmark predictor. To make the distinction between prediction error and decision impact transparent, we construct two subgroups of historical samples with comparable Euclidean deviation from the true OD vector but sharply diferent decision impacts:

• The first subgroup contains $2 0 0 s a f e$ historical samples generated around the center $f ^ { \star } +$ $( 1 0 , 0 , - 1 0 ) ^ { \top }$ . Specifically, for each sample we add an integer-valued perturbation $\left( \varepsilon _ { 1 } , \varepsilon _ { 2 } , \varepsilon _ { 3 } \right)$ , where $\varepsilon _ { 1 }$ and $\varepsilon _ { 3 }$ are constructed so that $\varepsilon _ { 3 } = - \varepsilon _ { 1 }$ , and $\varepsilon _ { 2 }$ is a relatively small perturbation around zero. These samples are intentionally displaced in a direction that mainly redistributes mass between $f _ { 1 2 }$ and $f _ { 1 3 }$ while leaving the decision-relevant component $f _ { 2 3 }$ nearly unchanged. As a result, although their mean squared deviation from the truth is non-negligible, their efect on the downstream decision is limited.

• The second subgroup contains 100 risky historical samples generated around the center $f ^ { \star } +$ $( 0 , 1 0 \sqrt { 2 } , 0 ) ^ { \top }$ , with moderate perturbations around that point. This group is constructed so that its Euclidean distance from the truth is of a similar magnitude to that of the safe group, but the displacement occurs primarily in the sensitive direction associated with $f _ { 2 3 }$ . Consequently, these samples are much more likely to move the estimate across the decision threshold and induce the wrong action.

This construction is deliberate. The safe and risky samples have nearly identical average Mean Squared Error (MSE) relative to the true OD vector—67.27 for the safe group and 65.35 for the risky group—and are therefore dificult to distinguish under a standard prediction metric. Yet their decision implications are very diferent. Most samples remain consistent with the correct downstream action, whereas a smaller subset is systematically shifted toward the wrong side of the decision boundary and thus behaves like decision-distorting outliers. These samples do not stand out in terms of MSE and do not dominate the dataset numerically, yet they are suficient to shift a conventional predictor across the decision threshold and induce the wrong decision.

This distinction is reflected in both the aggregate prediction results and the geometry shown in Figure 3. Using the sample mean of the entire group of historical samples as a benchmark predictor, yields $\widehat { f } ^ { \mathrm { m e a n } } = ( 4 1 . 6 9 , 5 3 . 6 2 , 5 8 . 3 0 ) ^ { \top }$ . By contrast, the proposed Decision Aware Estimation (DWE) method returns $\widehat { f } ^ { \mathrm { D W E } } = ( 5 0 . 0 0 , 4 9 . 9 0 , 5 0 . 0 0 ) ^ { \top }$ . In prediction terms, the sample mean is substantially closer to the ground truth: its MSE is 36.96, whereas the DWE estimate has a much larger MSE of 150.27. However, the performance reverses once decision quality is taken into account. Because $\widehat { f } _ { 2 3 } ^ { \mathrm { m e a n } } > 5 0$ , the sample mean induces the wrong decision; in contrast, because $\widehat { f } _ { 2 3 } ^ { \mathrm { D W E } } < 5 0$ , the DWE estimate recovers the correct decision.

Figure 3 illustrates the example from two complementary perspectives. In the reduced 2-D view in panel (a), the downstream decision depends only on whether $f _ { 2 3 }$ lies above or below the threshold of

![](images/1111904343e5b4fa634d29245e4c8a37fa84b3770048a152132a24e7c597106d.jpg)  
(a)

![](images/3d09e63b7c10e7d0ba9a4e1a75f25635152307e0baed30e87b37cf3803d7e4bf.jpg)  
(b)  
Figure 3 Comparison of the mean predictor and the DWE solution under decision-distorting samples: (a) reduced two-dimensional view on the feasible manifold; (b) corresponding three-dimensional geometric illustration.

50. Although most samples lie in the decision-consistent region, a small cluster of decision-distorting samples pulls the historical mean (blue point) across the boundary, despite its smaller MSE relative to the true OD vector (black star). The DWE solution (green triangle), in contrast, remains in the decision-consistent region. Panel (b) reveals the underlying geometry: the historical mean is pulled toward the center of the data cloud on the sensor-induced feasible manifold, improving data fidelity but crossing the projected decision boundary, whereas DWE optimizes the decision aware objective within the geometrically approximated feasible region and shifts the estimate toward a decisionrobust region. Thus, lower prediction error does not necessarily translate into better operationa performance.

Next, we examine a situation where historical samples violate the physical feasibility induced by hard data. We generate 300 historical OD samples:

• 200 regular samples satisfying the physical relation $f _ { 1 2 } + f _ { 1 3 } = 1 0 0$ , with $f _ { 1 2 } \in [ 2 5 , 4 5 ] , f _ { 2 3 } \in$ [45, 50], $f _ { 1 3 } = 1 0 0 - f _ { 1 2 }$

• 100 infeasible samples for which $f _ { 1 2 }$ and $f _ { 2 3 }$ remain in the same range, but $f _ { 1 3 }$ is shifted downward so that $f _ { 1 3 } = ( 1 0 0 - f _ { 1 2 } ) - \xi , \xi \in [ 2 0 , 3 0 ]$ . As a result, these contaminated historical samples no longer lie on the physically feasible manifold.

This construction captures an important failure mode in OD learning: in practice, historical data may contain observations that are inconsistent with the current physical state of the network.

Figure 4 illustrates the efect of physically infeasible samples from two complementary perspectives. In the reduced 2-D view in panel (a), $f _ { 1 2 } + f _ { 1 3 } = 1 0 0$ defines the sensor-implied feasibility condition, while $f _ { 2 3 } = ( f _ { 1 2 } + f _ { 1 3 } ) / 2$ defines the decision boundary. The infeasible samples (red points) pull the historical mean away from the physically feasible region, whereas DWE optimizes the decision aware objective directly over the feasible manifold, yielding a solution close to the true value while satisfying the physical constraints by construction. Panel (b) reveals the corresponding 3-D geometry: infeasible samples lie systematically of the manifold and distort the raw data cloud, making the historical mean an unreliable summary, while DWE remains anchored to the physically feasible region. Thus, physical constraints protect the estimator from sample errors that would otherwise pull it away from the structure of the current system.

![](images/72037933ed19c9331dc808873323c9b009b216ef358b19d4b51aa3e6b397ceec.jpg)  
(a)

![](images/4563d28d52d2129eb08364591cb36ddd577845c13e063f42a59d04e29e336579.jpg)  
(b)  
Figure 4 Comparison of the mean predictor and the DWE solution under infeasible samples: (a) reduced twodimensional view showing physical feasibility and the decision boundary; (b) corresponding three-dimensional geometric illustration.

This example illustrates the limitation of purely data-driven estimation. Historical observations are generated under heterogeneous network states, demand conditions, and behavioral responses, and thus populate a broad, high-dimensional space. However, once current observations and network conditions are imposed, the true OD vector is confined to a smaller feasible region lying on a smaller admissible manifold. A predictor may therefore fit the historical data well in a global sense and yet remain misaligned with the structure of the current system. This misalignment is both geometric and operational: some deviations are innocuous, while others are amplified through the downstream optimization, leading to substantial decision regret. The objective is not to reproduce historical patterns, but to extract an estimate that is physically grounded in current conditions and decision-relevant.

![](images/4592b29bb5da8bd1c7dea3b40c20e216c35409e7c56595a8ef3e576566d06b90.jpg)  
Figure 5 Small network schematic

$$
\begin{array} { r l r l r } & { \mathrm { L ~ = ~ } X f } & { \quad \quad } & { \mathrm { O D ~ F l o w } } \\ & { \quad } & { \quad } & { \quad } \\ & { \quad } & { \quad [ \mathrm { [ 1 ~ 1 ~ 0 ~ ~ 0 ~ ~ 0 ~ ~ 0 ~ ~ 0 ~ ~ 0 ~ ~ 0 ~ ~ 0 ~ ~ 0 ~ ~ 0 ~ ] ~ } \quad } & { \mathrm { ( 1 , 2 ) } , } \\ & { \quad } & { \quad } & { \quad \mathrm { ( 1 , 3 ) } , } \\ & { \quad \mathrm { e 2 } , } & { \quad \quad } & { \quad [ \mathrm { 0 ~ 0 ~ 0 ~ 1 ~ 0 ~ 0 ~ 1 ~ 0 ~ 0 ~ 1 ~ 0 ~ 0 ~ 1 ~ } \quad } & { \mathrm { ( 1 , 4 ) } , } \\ & { \quad \mathrm { e 2 } , } & { \quad \quad } & { \quad \mathrm { ( 0 ~ 0 ~ 0 ~ 1 ~ 1 ~ 1 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ ) ~ } \quad } & { \mathrm { ( 2 , 3 ) } , } \\ & { \quad \mathrm { e 3 } , } & { \quad \mathrm { [ 1 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 1 ~ 0 ~ 0 ~ 1 ~ 0 ~ 0 ~ ] ~ } \quad } & { \mathrm { ( 2 , 4 ) } , } \\ & { \quad \mathrm { e 4 } , } & { \quad \quad } & { \quad \mathrm { ( 1 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 1 ~ 1 ~ 0 ~ 0 ~ 0 ~ 0 ~ ) ~ } \quad } & { \mathrm { ( 2 , 4 ) } , } \\ & { \quad \mathrm { e 5 } , } & { \quad \quad } & { \quad \mathrm { 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 1 ~ 1 ~ 1 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 1 ~ } \quad } & { \mathrm { ( 3 , 1 ) } , } \\ & { \quad \mathrm { e 6 } , } & { \quad \quad } & { \quad \mathrm { [ 0 ~ 1 ~ 0 ~ 0 ~ 0 ~ 1 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 0 ~ 1 ~ 1 ~ } \quad } & { \mathrm { ( 3 , 2 ) } , } \\ & { \quad \mathrm { e 7 } , } & { \quad \quad } &  \quad \mathrm   \end{array}
$$

## 4.2. A Small Network for Benchmarking and Sensitivity Analysis

4.2.1. Network Configuration Consider a network consisting of four terminal nodes connected through a single central junction, resulting in eight directed links. Each terminal node can serve as both an origin and a destination, yielding a total of 12 OD pairs. The network topology and link directions are illustrated in Figure 5. Let $f \in \mathbb { R } _ { + } ^ { 1 2 }$ denote the OD flow vector and $L \in \mathbb { R } _ { + } ^ { 8 }$ the corresponding link flow vector. The physical relationship between OD flows and link flows is encoded by a fixed link-OD incidence matrix $X \in \{ 0 , 1 \} ^ { 8 \times 1 2 }$ , such that $L = X f$

4.2.2. Fabrication of True OD and Historical OD Samples We use a polynomial function based on four features to generate historical OD samples and the true OD flows. The case $P = 1$ serves as the lowest-complexity benchmark, in which the latent OD index is linear in the features. Larger values of P introduce increasing nonlinear structure in the feature-to-OD mapping. Specifically, for each polynomial order $P \in \{ 1 , \ldots , 6 \}$ , latent OD flows are generated from a polynomial transformation of the feature vector and then passed through a softplus transformation, $Q = \log ( 1 + \exp ( Z ) )$ , which guarantees non-negativity while preserving smoothness. Thus, varying P creates OD samples with diferent levels of complexity and diferent degrees of model misspecification for linear prediction models. This allows us to evaluate how prediction-oriented and decision aware estimators perform as the historical OD data become increasingly nonlinear and harder to fit with a linear model.

For each value of $P ,$ we generate separate training and testing samples. The training samples are used to construct prediction-oriented and decision aware OD estimators, while the testing samples are used to evaluate downstream decision quality. See Appendix H for details of the OD datageneration process.

4.2.3. Downstream Decision Problem We use the demonstration investment allocation problem described in Appendix G as the downstream decision problem. This problem is formulated as a budget-constrained linear program, in which a transportation authority allocates a fixed investment budget across candidate road segments, and the resulting link improvements reduce OD-level travel times. For a given OD-flow estimate $f ,$ the downstream problem returns an optimal investment allocation $x ^ { * } ( f )$ ; the corresponding post-investment OD-level travel times are denoted by $w ^ { * } ( f )$

To avoid confounding the comparison with random variation from the downstream optimization problem, we evaluate all OD-estimation methods under the same investment environment within each experimental instance. Specifically, the OD-to-segment incidence matrix, baseline travel times, marginal improvement costs, and segment-level improvement upper bounds are generated once and then held fixed throughout the method comparison. The investment budget is also fixed and is set to 35% of the maximum feasible investment level. Thus, the downstream LP is identical across methods except for the OD-flow vector used in the objective. This design ensures that diferences in downstream performance are driven by the estimated OD vectors rather than by random variation in the underlying investment problem.

4.2.4. Benchmark Methods We organize the benchmark methods according to two dimensions: the source of OD-demand information and the way this information is incorporated into the downstream decision problem. The first dimension distinguishes diferent forms of prior OD information available before solving the downstream optimization problem. In particular, we consider three historical-demand priors, denoted by $\mathcal { R } = \{ \mathrm { M , L R , L o c } \}$ . Here, M denotes the global historical mean, which uses the average OD vector over the historical sample. LR denotes a model-based prediction obtained from a linear regression model trained to predict OD flows by minimizing mean squared error; this produces a single point estimate for the current instance. Loc denotes a local historical-demand prior, which uses multiple historical OD samples associated with covariate values similar to the current instance. In our implementation, this local prior is constructed by averaging the OD vectors of the k-nearest historical samples in the covariate space.<sup>1</sup>

We first evaluate these priors under a standard predict-then-optimize (PTO) pipeline. In this class of methods, an OD vector is first estimated without using the downstream decision objective, and the resulting point estimate is then passed directly to the investment allocation problem. This gives three PTO baselines: PTO-M, PTO-LR, PTO-Loc, corresponding respectively to the global historical mean, the linear-regression prediction, and the local historical-demand prior. These baselines represent the conventional sequential approach in which OD estimation quality and downstream decision quality are optimized separately.

In addition to these demand-side priors, we include an OLS benchmark based on observed link-flow information. This method reconstructs OD flows from observed link flows through a least-squares formulation and then applies the downstream investment model to the reconstructed OD vector. The OLS benchmark therefore represents the information available from the link-flow observation side, in contrast to the historical-demand priors used by the PTO baselines.

![](images/b43d9b2f3fbe698b2917e739e7e8ceace1a71abb07eb525b57a9196024dfcbef.jpg)  
(a)

![](images/8cff5ecd4c190a84127562bd1707a6bc16d9c79e92830c2e19348dd9ae0a6580.jpg)  
(b)

![](images/80c798b5b3b1ec0acab7afc03daecf4f1eaf8c6bac272a5090fd4425ba7ff172.jpg)  
(c)  
Figure 6 Decision regret under increasing nonlinear demand generation. Panel (a) reports the average test regret, Panel (b) reports the 80th percentile regret, and Panel (c) reports the 90th percentile regret.

We then compare these baselines with two direct-reformulation DWE methods. For each priorinformation source $r \in \{ \mathrm { M , L R , L o c } \}$ , we denote the DWE-original and DWE-Ellipsoid formulations introduced in Section 3.4 by D-Full-r and D-Geo-r, respectively. The former optimizes the SPO+ surrogate over the original link-flow consistent OD feasible region, whereas the latter uses its ellipsoidal approximation. This paired comparison isolates the efect of the feasible-region approximation while holding the prior-information source and decision aware loss fixed.

4.2.5. Regret Behavior under Increasing Demand Generation Nonlinearity This experiment examines how downstream decision regret changes as the polynomial order increases from 1 to 6, thereby introducing progressively stronger nonlinearity into the true demand generation process. This setting allows us to evaluate whether each method remains decision reliable as the actual demand generation mechanism becomes more complex and prediction-centered estimators are increasingly exposed to model misspecification. The comparison is motivated by a central curiosity: Whether and how does model misspecification translate prediction errors into unreliable downstream decisions? Specifically, we will answer three empirical questions. First, does increasing demand-generation nonlinearity undermine the downstream reliability of decisions produced by prediction-centered methods? Second, conditional on using the same historical demand information, does decision aware estimation provide additional regret reduction beyond standard PTO? Third, are these gains concentrated in average performance, or do they also translate into lower upper-tai regret and greater stability across training samples?

Figure 6 summarizes the efect of model misspecification (indicated by increasing demand generation nonlinearity) on downstream decision quality. We first consider Panel (a), which reports the average test-set regret. The OLS benchmark provides a useful contrast because it relies only on observed link-flow consistency and does not use any historical OD-demand prior. Its regret varies substantially across polynomial orders. This behavior reflects the ill-posed nature of OD reconstruction merely from aggregate link flows: many OD vectors can be consistent with the same link-flow observations, but they may induce very diferent investment decisions, thus leading to unstable downstream regret.

Among the PTO baselines, PTO-M performs poorly because it relies only on a coarse global mean prior and ignores the covariate information of the current instance. Incorporating covariate information substantially improves performance in the low-order regime, reflected by the low regrets produced by PTO-LR and PTO-Loc. For example, PTO-LR produced 4.17 at order 1, which reflects the advantage of the linear predictor when the true demand-generation process is close to the assumed model class. However, as the polynomial order increases and model misspecification becomes more severe, both PTO-LR and PTO-Loc deteriorate rapidly.

We next compare the DWE methods in Panel (a). The D-Full-r methods provide clear improvements over their PTO counterparts at low and intermediate orders, confirming the value of incorporating downstream decision structure into the OD-estimation step. However, their performance becomes less stable in the high-order regime. For instance, at order 6, regret remains at 284.11 for D-Full-M, 226.08 for D-Full-LR, and 224.47 for D-Full-Loc. In contrast, the D-Geo-r methods form the most stable group across orders and prior sources. At order 6, D-Geo-LR and D-Geo-Loc achieve regrets of 79.80 and 73.76, respectively. The gap between D-Full-r and D-Geo-r is particularly informative. Since both methods use the same decision aware objective, their diference is driven primarily by the feasible-region representation. The superior stability of D-Geo-r suggests that the geometric approximation plays a role similar to regularization: it constrains the OD estimate to a more stable and decision-relevant region, thereby preventing the direct decision aware fit over the full feasible set from becoming overly sensitive in high-complexity settings.

Panels (b) and (c) further show that the main benefit of the regularized DWE methods is not limited to the average regret. Panel (b) reports the 80th percentile test-set regret. At this level, diferences among the stronger methods become much smaller. In particular, the D-Geo-r methods attain values that are essentially zero across nearly all orders and prior sources, indicating that they incur little or no regret for a large majority of test instances. Thus, much of the average-regret diference in Panel (a) is not driven by the bulk of ordinary test cases, but rather by a smaller set of more dificult instances. This interpretation is confirmed by Panel (c), where the 90th percentile regret reveals much sharper separation across methods. Once attention shifts to the upper tail of the regret distribution, the risk-control advantage of the regularized DWE methods becomes clear. For example, at order 6 under the local prior, the 90th percentile regret is 336.17 for PTO-Loc,

![](images/99175a326237c848a184944ee2e4a27a9e053c8c03a72b61533fa84125d1f7be.jpg)  
Figure 7 Regret distributions across methods as the polynomial order increases.

173.13 for D-Full-Loc, and 37.53 for D-Geo-Loc. Under the linear-regression prior at the same order, the corresponding values are 634.22 for PTO-LR, 194.11 for D-Full-LR, and 55.73 for D-Geo-LR. Therefore, the main advantage of the D-Geo-r methods lies not only in reducing average regret, but also in suppressing rare but severe decision failures. This type of tail robustness becomes increasingly important as the true demand-generation mechanism grows more nonlinear.

Figure 7 complements the heatmap by showing the within-prior distribution of mean test-set regret across training seeds. This figure addresses whether the improvement of the DWE methods is driven by a few favorable training samples or is systematic across training realizations. In the highorder regimes, PTO-LR and PTO-Loc have not only higher median regret, but also wider boxes, longer upper whiskers, and more extreme outliers. This indicates that, under stronger nonlinearity, PTO becomes highly sensitive to the particular training sample: some seeds lead to especially poor downstream decisions. In contrast, the D-Geo variants shift the entire seed-level distribution downward and keep it substantially more concentrated. Thus, their improvement is not driven by a small number of favorable training seeds. Rather, they systematically reduce variability and suppress high-regret training realizations.

Finally, Figure 8 decomposes the reduction in regret into three incremental sources. The first is the gain from using a covariate-aligned historical prior. Moving from PTO-M to PTO-Loc lowers regret across all polynomial orders, indicating that matching historical OD information to the current covariate realization is valuable. However, this improvement alone is not suficient: in the nonlinear regime, PTO-Loc still exhibits substantial regret. The second gain comes from adding the decision aware objective and physical OD constraints. Relative to PTO-Loc, D-Full-Loc further reduces regret, showing that decision aware refinement provides value beyond predictioncentered use of the same prior. The third and largest high-order gain comes from the geometric approximation. Compared with the full-constraint decision aware estimator, D-Geo-Loc achieves a much lower regret in the nonlinear regime, suggesting that restricting the update to a stable, decision-relevant region is crucial.

![](images/b942585aa0f14836672edda04f7aa5ac6c72bcbd41beb30b7ba70a2448fffba0.jpg)  
Figure 8 Regret reduction from local prior and decision aware refinement across polynomial orders.

Overall, the results provide consistent answers to the three empirical questions posed above. First, increasing level of model misspecification (indicated by demand-generation nonlinearity) does undermine the downstream reliability of prediction-centered methods. Although historical priors that exploit current-instance covariate information improve substantially upon a crude global mean, PTO methods still deteriorate markedly in the highly nonlinear regime. Second, conditional on using the same historical demand information, decision aware estimation provides additional regret reduction beyond standard PTO, showing that downstream decision structure contains useful information for OD estimation. However, direct decision aware fitting over the full feasible set can remain unstable when the demand-generation process becomes suficiently complex. Third, the benefits of introducing the geometric approximation in the decision aware estimation problem yields clear benefits, reducing both the average and upper-tail regret across training samples.

4.2.6. Robustness to Noisy Historical OD Labels The previous experiments varied the structural complexity of the OD generation process. We next examine a complementary source of misspecification: noisy historical OD labels. The central question is: holding the data-generating process fixed, how sensitive are prediction centered and decision centered estimators to increasing noise in the historical OD information used to construct their priors?

Figure 9 reports the resulting regret sensitivity. In this experiment, we fix the polynomial order at three and add independent nonnegative uniform perturbations to the historical OD demand. The perturbation magnitude increases with the noise level, and the noisy OD entries are clipped to remain within the feasible demand range. The test set and the underlying data-generating process are held fixed, so the experiment isolates the efect of corrupted historical outcome information on downstream decision quality. Since the OLS benchmark is not afected by this historical-label perturbation, we use it as the normalization baseline and report

![](images/c8b4f488611234a9d49270dbd755981586401039687830ead1fc4e9ade58a324.jpg)  
Figure 9 Percentage reduction in decision regret relative to OLS under increasing historical OD label noise.

$$
{ \mathrm { R e d u c t i o n ~ v s . ~ O L S } } = 1 0 0 \cdot { \frac { { \mathrm { R e g r e t } } _ { \mathrm { O L S } } - { \mathrm { R e g r e t } } _ { m } } { \mathrm { R e g r e t } _ { \mathrm { O L S } } } } .
$$

The global mean historical prior is omitted because its performance is uniformly poor, allowing the comparison to focus on the LR and Loc. Under both prior constructions, the PTO methods deteriorate as the historical OD labels become noisier. Both PTO-LR and PTO-Loc initially achieve substantial regret reductions relative to OLS, but their advantages vanish and eventually become negative as the noise level increases. The two PTO variants, however, exhibit diferent deterioration patterns. PTO-Loc declines more quickly and more smoothly, indicating that the local prior is immediately afected by noisy neighboring historical observations. PTO-LR is more stable at moderate noise levels, but after the perturbation exceeds a critical range, its regret reduction drops sharply and falls below the OLS baseline. This behavior indicates that PTO is highly sensitive to corrupted historical labels, because the downstream optimization is driven by a single predictive reference that becomes increasingly unreliable.

The decision aware estimator with the full structural constraints also becomes less stable under noise. In both panels, D-FULL starts with a sizable reduction relative to OLS but declines sharply as the perturbation level increases. Nevertheless, it continues to improve upon the corresponding PTO baseline over most of the noise range. This suggests that incorporating the decision objective remains beneficial, but an unrestricted decision aware fit can overreact to noisy historical signals. By contrast, the ellipsoidal approximation is substantially more robust. The D-Geo-r methods maintain a large and nearly flat regret reduction across the entire noise range in both the LR and Loc panels. Although their performance also decreases mildly as noise increases, the decline is much smaller than that of PTO or D-FULL. This stability suggests that the ellipsoidal restriction filters out noisy directions in the historical OD information while preserving the decision-relevant region of the feasible set.

![](images/a1af489acb51c63f7d6fe2ca79b4546f75d2274cf829d33874646906b6e9067e.jpg)

![](images/989f9291b38e9a76f908fe9ccd000aa8391b6028f2211efe3ecde4289d1de6de.jpg)  
Figure 10 Sensitivity of test-set decision regret to the tuning parameter α under noisy historical OD labels.

Overall, this experiment reinforces the main robustness message of the paper. Historical OD information is useful only if it is incorporated in a way that is stable under label corruption. Methods that prioritize data alignment alone quickly lose their downstream advantage as historical labels become noisy, whereas decision aware methods with geometric regularization remain robust, preserving meaningful regret reductions even under high noise.

4.2.7. Sensitivity of the Tuning Parameter α to Label Noise We next examine how sensitive the decision aware estimator is to the $\mathrm { S P O + }$ surrogate parameter α when the historical OD prior are noisy. Following the same noisy setting as above, we perturb the historical OD samples used to construct the local prior and evaluate the test-set decision regret of D-Geo-Loc over a grid of positive $\alpha$ values. For each noise level η, let $R _ { \eta } ( \alpha )$ denote the average test-set regret obtained using tuning parameter α. To measure sensitivity relative to the best value in the grid, we define the excess-regret sensitivity function

$$
S _ { \eta } ( \alpha ) = R _ { \eta } ( \alpha ) - \operatorname* { m i n } _ { \alpha ^ { \prime } \in \mathcal { A } } R _ { \eta } ( \alpha ^ { \prime } ) ,
$$

where A is the set of positive α values considered in the experiment. A smaller value of $S _ { \eta } ( \alpha )$ indicates that the performance at α is close to the best achievable performance for that noise level.

Figure 10 reports the sensitivity of D-Geo-Loc to the tuning parameter α under diferent levels of label noise. The left panel shows a clear decreasing pattern in average regret as α moves away from very small values. When $\alpha = 0 . 1$ or 0.5, the regret remains relatively high across all noise levels, indicating that too small a value of α provides insuficient decision guidance. As α increases to a moderate range, especially around $\alpha = 2$ , the regret drops substantially and then becomes relatively flat for larger values. This empirical pattern is broadly consistent with the standard $\mathrm { S P O + }$ formulation, where $\alpha = 2$ is the canonical scaling used in the Fisher-consistency analysis of the SPO+ surrogate (Elmachtoub and Grigas 2022). The right panel reports the excess-regret sensitivity function $S _ { \eta } ( \alpha )$ . Consistent with the left panel, small values of α incur a noticeable excess regret, while moderate and large values produce performance close to the best value within each noise level. This suggests that DWE does not require fine tuning of α under mild to moderate label noise, which is practically useful because the method does not depend on a highly delicate hyperparameter choice. Rather, the main requirement is to avoid overly small values of $\alpha ;$ once $\alpha$ is suficiently positive, the estimator achieves relatively stable decision performance across the noise levels considered.

## 4.3. Sioux Falls network

We further evaluate the proposed decision aware estimation framework on a larger and more realistic transportation network, the Sioux Falls network. This network consists of 24 nodes, 76 directed links, and 552 OD pairs. The physical relationship between OD flows and link flows is encoded by a fixed link-OD incidence matrix $X \in \mathbb { R } ^ { 7 6 \times 5 5 2 }$ in $L = X f$ , leading to a substantially higherdimensional and more severely underdetermined estimation problem than the synthetic intersection network considered earlier.

4.3.1. OD Demand Patterns To assess robustness across heterogeneous demand structures, we construct five distinct OD demand patterns on the Sioux Falls network. These patterns are designed to capture qualitatively diferent spatial and structural characteristics commonly observed in real transportation systems: distance decay, hub-spoke concentration, community block structure, directional corridors, and low-rank latent demand. Figure 11 illustrates the resulting OD matrices together with their induced link-flow patterns. While these demand modes exhibit markedly diferent spatial organizations, all satisfy the same physical flow conservation constraints. Detailed descriptions of the demand construction and their statistical properties are provided in Appendix I.1.

4.3.2. Experimental Design For all five demand patterns, we generate OD-flow priors using a gravity model with imperfect features, providing a common prior across all experiments. We consider three total-demand levels: low (1,000), medium (10,000), and high (20,000).

We compare OLS, PTO, and two DWE variants. OLS reconstructs OD flows from observed link flows using least squares, while PTO directly optimizes the downstream problem using the gravitymodel prediction. Following the previous name convention, DWE methods are denoted by D-Full-G and D-Geo-G, where G denotes the gravity-model prior. D-Full-G directly solves the full decision aware estimation formulation, whereas D-Geo-G uses the proposed geometry-based formulation; for D-Geo-G, we evaluate multiple radii and report the best-performing radius $r ^ { * }$

Table 1 summarizes the performances across all demand patterns and demand levels, evaluated using downstream decision regret.

![](images/9e073f310db95c42264716d61c9d5d860eea8730214787ad47c2bd5e6297039c.jpg)  
Figure 11 OD patterns (left) and corresponding link flows (right).

4.3.3. Results Table 1 documents a clear advantage of decision aware estimation on the Sioux Falls network. OLS yields the largest decision regret in all 15 demand-level–mode combinations, highlighting the limitations of recovering OD demand from link-flow consistency alone in a severely underdetermined system. PTO substantially improves upon OLS by incorporating the gravity-model prediction, but considerable decision regret remains in several settings. Both DWE variants deliver further gains: D-Full-G achieves strictly lower decision regret than PTO in 10 of the 15 cases and matches it in the remaining five, while D-Geo-G improves upon PTO in 11 cases and matches it in the other four. Among the cases with strict improvement, the reduction relative to PTO ranges from 52.2% to 100% for D-Full-G and from 68.7% to 100% for D-Geo-G. Across all 15 cases, the simple average regret decreases from 184.00 under PTO to 64.40 under D-Full-G and 38.19 under D-Geo-G, corresponding to reductions of approximately 65.0% and 79.2%, respectively. To assess whether these conclusions are robust to variation across experimental realizations, we conduct an additional repeated-sampling experiment reported in Appendix Table 3. Each setting is repeated 20 times, and the mean decision regret is reported. D-Full-G and D-Geo-G strictly outperform PTO and OLS in all 15 settings, confirming the robustness of decision aware estimation. Thus, explicitly accounting for the downstream decision produces substantial improvements without deteriorating decision quality in any of the baseline scenarios. A direct comparison of the two DWE variants provides evidence of an additional regularization efect from the geometric formulation. For D-Geo-G, Table 1 reports the best result over the tested radii. D-Geo-G achieves decision regret that is lower than or equal to that of D-Full-G in all 15 cases: it performs strictly better in five cases and matches D-Full-G in the remaining ten. Its average regret is also 40.7% lower than that of D-Full-G (38.19 versus 64.40).

Table 1 Performance across demand levels and demand modes.
<table><tr><td rowspan="2">Demand Level</td><td rowspan="2">Demand Mode</td><td rowspan="2">RMSE</td><td rowspan="2">OLS</td><td rowspan="2">PTO</td><td colspan="2">DWE Variants</td></tr><tr><td>D-FULL-G</td><td>D-GEO-G</td></tr><tr><td rowspan="5">Low</td><td>Distance Decay</td><td>0.55</td><td>25.77</td><td>2.59</td><td>2.59</td><td>2.59 (r* = 5)</td></tr><tr><td>Low Rank</td><td>1.74</td><td>59.84</td><td>12.74</td><td>3.99</td><td>3.99 (r* = 10)</td></tr><tr><td>Community Blocks</td><td>2.57</td><td>55.15</td><td>42.18</td><td>5.45</td><td>5.45 (r* = 10)</td></tr><tr><td>Directional Corridor</td><td>1.59</td><td>52.97</td><td>26.91</td><td>6.09</td><td>0.00 (r* = 3)</td></tr><tr><td>Hub Spoke</td><td>1.75</td><td>92.92</td><td>3.77</td><td>0.15</td><td>0.15 (r* = 10)</td></tr><tr><td rowspan="5">Medium</td><td>Distance Decay</td><td>5.33</td><td>320.14</td><td>36.19</td><td>36.19</td><td>3.36 (r* = 3)</td></tr><tr><td>Low Rank</td><td>17.11</td><td>795.97</td><td>257.26</td><td>12.81</td><td>12.81 (r* = 10)</td></tr><tr><td>Community Blocks</td><td>25.13</td><td>560.63</td><td>430.56</td><td>73.84</td><td>28.59 (r* = 5)</td></tr><tr><td>Directional Corridor</td><td>16.14</td><td>513.44</td><td>198.08</td><td>94.74</td><td>0.00 (r* = 3)</td></tr><tr><td>Hub Spoke</td><td>17.55</td><td>1018.62</td><td>45.33</td><td>0.00</td><td>0.00 (r* = 10)</td></tr><tr><td rowspan="5">High</td><td>Distance Decay</td><td>11.42</td><td>228.24</td><td>3.84</td><td>3.84</td><td>3.84 (r* = 10)</td></tr><tr><td>Low Rank</td><td>35.78</td><td>1420.49</td><td>195.95</td><td>195.95</td><td>195.95 (r* = 5)</td></tr><tr><td>Community Blocks</td><td>52.50</td><td>949.19</td><td>873.83</td><td>271.38</td><td>271.38 (r* = 10)</td></tr><tr><td>Directional Corridor</td><td>31.52</td><td>1067.51</td><td>585.95</td><td>214.22</td><td>0.00 (r* = 3)</td></tr><tr><td>Hub Spoke</td><td>34.93</td><td>1609.66</td><td>44.75</td><td>44.75</td><td>44.75 (r* = 10)</td></tr></table>

Notes: The RMSE column reports the prediction RMSE of the prior baseline. All OLS, PTO, and DWE columns report decision regret divided by $\mathrm { 1 0 ^ { 3 } }$ . For D-GEO-G, the best result over all tested radii is reported, with the selected radius shown as $r ^ { * }$ . Bold values indicate the best decision regret within each setting.

One may notice that the magnitude of decision regrets varies sharply across demand patterns. It would be interesting to explore how demand-prediction error propagates through downstream problems of diferent network structures, but it is beyond the scope of this work.

![](images/64d9181ac4fa45752581f1690ee2112a1825fc0ab2f5b04288071d22990917fd.jpg)  
Figure 12 Performance–computational trade-of across diferent OD patterns and solution methods, measured by decision regret and solve time.

On the other hand, the gain of D-Geo-G in performance robustness comes at a substantial computational cost. As shown in Figure 12, D-Full-G solves in 0.922 seconds on average, with a median of 0.921 seconds and a standard deviation of only 0.011 seconds. D-Geo-G, by contrast, requires 19.455 seconds on average, with a median of 17.845 seconds and a standard deviation of 5.835 seconds. Thus, D-Geo-G requires approximately 21.1 times the mean solve time of D-Full-G. The runtime of D-Full-G across all instances is similar, while D-Geo-G‘s runtime seem to vary by instances. D-Full-G is therefore preferable when computational eficiency is paramount, while D-Geo-G ofers a more conservative alternative when the additional protection against over-adaptation justifies the computational premium.

## 5. Conclusions and Future Research

As we conclude this study, we reflect on a broader lesson that extends beyond OD estimation. OD demand is not directly observed and must instead be inferred from indirect information, such as observed link flows, physical network constraints, and prior demand patterns. These sources help characterize which OD estimates are plausible, but plausibility alone does not determine which estimate is most useful. A key premise we emphasize is: not all estimation errors are consequential in the same way. A good estimate should be faithful enough to the world we observe, but attentive to the consequences of the decisions it informs. Through analytical and numerical results, we have shown that accounting for how estimates are ultimately used can substantially improve their value for downstream decision-making.

Beyond knowledge from transportation science, our research design draws on two complementary methodological perspectives: geometry and stochastic optimization. A geometric perspective shapes how we view and represent the demand space. OD demand vectors are typically modeled in a high-dimensional space whose dimension equals the number of OD pairs in the network. A central message from this study is that a realistic OD demand does not occupy the entire ambient space. Instead, it is implicitly restricted by a variety of physical and behavioral constraints. These structural restrictions imply that feasible OD demand vectors lie on a much lower-dimensional manifold embedded in the high-dimensional OD space. Ignoring this geometric structure can lead to misleading conclusions in data-driven prediction. A stochastic optimization perspective provides a flexible framework for integrating data, prior knowledge, physical constraints, and decision regret within a unified model. Equally important, advances in stochastic optimization provide the theoretical and computational tools needed to solve the resulting models eficiently and reliably. Our experience with these complementary perspectives also points to a broader lesson: new research challenges often benefit from ideas that lie outside the traditional boundaries of a discipline. Particularly for young scholars, the broader the set of intellectual building blocks one acquires, the better prepared one may be to construct solutions to challenges not yet anticipated.

The present study opens several promising directions for future research, spanning modeling generality, data reliability, theoretical analysis, and algorithmic development. First and foremost, while the linear constraints considered in this study capture fundamental flow conservation laws, real transportation systems are governed by richer physical mechanisms, including nonlinear constraints and dynamic relations that may be expressed through diferential or diference equations. Accommodating such general physical constraints would substantially broaden its applicability, while also posing new analytical and computational challenges. Second, the reliability of link-flow observations themselves deserves explicit attention. In this paper, link flows are treated as accurate measurements that define a consistent constraint system. Developing principled methods to handle imperfect link-flow observations, such as through soft constraints, feasibility robust formulations, or joint estimation of OD flows and link-flow errors, remains an important and challenging direction for future research. In additional, a natural extension is to consider more complex decision problems, including those with nonlinear objectives, integer decisions, equilibrium constraints, or multistage structures. Such extensions would allow the framework to capture a broader class of transportation decisions, but would also require new tools to analyze regret, tractability, and solution stability. Finally, how to characterize regret presents future research opportunities. For example, while the current results provide initial bounds, sharper regret guarantees—such as tighter upper bounds, lower bounds, or

instance-dependent characterizations—are needed to better understand the fundamental limits of decision aware OD estimation. These directions are not simply extensions of the present framework; they point toward a broader opportunity to rethink how estimation and decision-making should interact. As increasingly rich data and increasingly consequential decisions become intertwined, we believe that learning not only from the data, but also from how the data will ultimately be used, ofers a promising path forward.

## Acknowledgments

This work is supported in part by the National Science Foundation under grant CMMI-2432336/CMMI-2432337.

## References

Agrawal A, Barratt S, Boyd S, Busseti E, Moursi WM, 2019 Diferentiating through a cone program. arXiv preprint arXiv:1904.09043 .

Akcay A, Biller B, Tayur S, 2011 Improved inventory targets in the presence of limited historical demand data. Manufacturing & Service Operations Management 13(3):297–309.

Amos B, Kolter JZ, 2017 OptNet: Diferentiable optimization as a layer in neural networks. International Conference on Machine Learning, 136–145 (PMLR).

Bar-Gera H, 2006 Primal method for determining the most likely route flows in large road networks. Transportation Science 40:269–286, URL http://dx.doi.org/10.1287/trsc.1050.0142.

Bassamboo A, Zeevi A, 2009 On a data-driven method for stafing large call centers. Operations Research 57(3):714–726.

Bell DE, 1983 Risk premiums for decision regret. Management Science 29(10):1156–1166.

Bengio Y, 1997 Using a financial training criterion rather than a prediction criterion. International Journal of Neural Systems 8(04):433–443.

Bertsimas D, Kallus N, 2020 From predictive to prescriptive analytics. Management Science 66(3):1025–1044.

Bierlaire M, Crittin F, 2004 An eficient algorithm for real-time estimation and prediction of dynamic OD tables. Operations Research 52(1):116–127.

Blondel M, Martins AF, Niculae V, 2020 Learning with Fenchel-Young losses. Journal of Machine Learning Research 21(35):1–69.

Cascetta E, 1984 Estimation of trip matrices from trafic counts and survey data: a generalized least squares estimator. Transportation Research Part B: Methodological 18(4-5):289–299.

Castillo E, Conejo AJ, Menéndez JM, Jiménez P, 2008 The observability problem in trafic network models. Computer-Aided Civil and Infrastructure Engineering 23(3):208–222.

Chen Y, Ordónez F, Palmer K, 2006 Confidence intervals for OD demand estimation. USC-ISE Working Paper 1.

De Palma A, Lindsey R, 2011 Trafic congestion pricing methodologies and technologies. Transportation Research Part C: Emerging Technologies 19(6):1377–1399.

Demissie MG, Kattan L, 2022 Estimation of truck origin-destination flows using GPS data. Transportation Research Part E: Logistics and Transportation Review 159:102621.

Donti P, Amos B, Kolter JZ, 2017 Task-based end-to-end model learning in stochastic optimization. Advances in Neural Information Processing Systems 30.

El Balghiti O, Elmachtoub AN, Grigas P, Tewari A, 2023 Generalization bounds in the predict-then-optimize framework. Mathematics of Operations Research 48(4):2043–2065, URL http://dx.doi.org/10.1287/ moor.2022.1330.

Elmachtoub AN, Grigas P, 2022 Smart “predict, then optimize”. Management Science 68(1):9–26.

Fan Y, Kalaba R, 2003 Dynamic programming and pseudo-inverses. Applied Mathematics and Computation 139(2-3):323–342.

Jin PJ, Cebelak M, Yang F, Zhang J, Walton CM, Ran B, 2014 Location-based social networking data: exploration into use of doubly constrained gravity model for origin–destination estimation. Transportation Research Record 2430(1):72–82.

Kao Yh, Roy B, Yan X, 2009 Directed regression. Advances in Neural Information Processing Systems 22.

Krishnakumari P, van Lint H, Djukic T, Cats O, 2019 A data driven method for OD matrix estimation. Transportation Research Procedia 38:139–159.

Li X, Jin K, Ma H, Feng Z, Xia Y, 2026 Data-driven capacity management of ride-hailing service with prescriptive analytics and deep learning. INFORMS Journal on Computing URL http://dx.doi.org/ 10.1287/ijoc.2024.0595.

Liyanage LH, Shanthikumar JG, 2005 A practical inventory control policy using operational statistics. Operations Research Letters 33(4):341–348.

Loke GG, Tang Q, Xiao Y, 2022 Decision-driven regularization: A blended model for predict-then-optimize. Available at SSRN 3623006 .

Maher MJ, 1983 Inferences on trip matrices from observations on link volumes: A Bayesian statistical approach. Transportation Research Part B: Methodological 17(6):435–447.

Mandi J, Guns T, 2020 Interior point solving for LP-based prediction+optimisation. Advances in Neural Information Processing Systems 33:7272–7282.

Mandi J, Kotary J, Berden S, Mulamba M, Bucarey V, Guns T, Fioretto F, 2024 Decision-focused learning: Foundations, state of the art, benchmark and future opportunities. Journal of Artificial Intelligence Research 80:1623–1701.

Mandi J, Stuckey PJ, Guns T, et al., 2020 Smart predict-and-optimize for hard combinatorial optimization problems. Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, 1603–1610.

Patriksson M, 2015 The trafic assignment problem: models and methods (Courier Dover Publications).

Pogančić MV, Paulus A, Musil V, Martius G, Rolinek M, 2019 Diferentiation of blackbox combinatorial solvers. International Conference on Learning Representations.

Pourebrahim N, Sultana S, Niakanlahiji A, Thill JC, 2019 Trip distribution modeling with Twitter data. Computers, Environment and Urban Systems 77:101354.

Qi M, Shen ZJ, 2022 Integrating prediction/estimation and optimization with applications in operations management. Tutorials in operations research: emerging and impactful topics in operations, 36–58 (INFORMS).

Rao CR, Mitra SK, 1971 Generalized Inverse of Matrices and its Applications (New York: John Wiley & Sons).

Renegar J, 2001 A mathematical view of interior-point methods in convex optimization (SIAM).

Rokicki B, Stępniak M, 2018 Major transport infrastructure investment and regional economic development– an accessibility-based approach. Journal of Transport Geography 72:36–49.

Ros-Roca X, Montero L, Barceló J, Nökel K, Gentile G, 2022 A practical approach to assignment-free dynamic origin-destination matrix estimation problem. Transportation Research Part C: Emerging Technologies 134:103477.

Royset JO, Wets RJB, 2021 An optimization primer, volume 440 (Springer).

Sadana U, Chenreddy A, Delage E, Forel A, Frejinger E, Vidal T, 2025 A survey of contextual optimization methods for decision-making under uncertainty. European Journal of Operational Research 320(2):271– 289.

Wang J, Lu S, Liu H, Ban X, 2023 Transportation origin-destination demand estimation with quasi-sparsity. Transportation Science 57(2):289–312.

Wilder B, Dilkina B, Tambe M, 2019 Melding the data-decisions pipeline: Decision-focused learning for combinatorial optimization. Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, 1658–1665.

Xie C, Kockelman KM, Waller ST, 2011 A maximum entropy-least squares estimator for elastic origindestination trip matrix estimation. Procedia-Social and Behavioral Sciences 17:189–212.

Yang H, H Bell MG, 1998 Models and algorithms for road network design: a review and some new developments. Transport Reviews 18(3):257–278.

Yang Y, Fan Y, 2015 Data dependent input control for origin–destination demand estimation using observability analysis. Transportation Research Part B: Methodological 78:385–403.

Yao X, Gao Y, Zhu D, Manley E, Wang J, Liu Y, 2020 Spatial origin-destination flow imputation using graph convolutional networks. IEEE Transactions on Intelligent Transportation Systems 22(12):7474–7484.

Yin G, Huang Z, Bao Y, Wang H, Li L, Ma X, Zhang Y, 2023 ConvGCN-RF: A hybrid learning model for commuting flow prediction considering geographical semantics and neighborhood efects. GeoInformatica 27(2):137–157.

## Appendix A: Ellipsoidal Radius and Calibration

Let $d = \dim ( \ker X ) = n - q$ and consider the ellipsoidal search region ${ \boldsymbol { \mathcal { B } } } _ { H } ( { \boldsymbol { r } } ) = \{ u \in \mathbb { R } ^ { d } \mid u ^ { \top } H u \leq r ^ { 2 } \}$ , where $H \succ 0$ is the Hessian of the logarithmic barrier evaluated at the analytic-center. Let $H = Q \Lambda Q ^ { \top } , \Lambda =$ d $\mathrm { i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { d } )$ . Under the coordinate transformation $y = Q ^ { \top } u ,$ the ellipsoid becomes $\textstyle \sum _ { i = 1 } ^ { d } \lambda _ { i } y _ { i } ^ { 2 } \leq r ^ { 2 }$ , or equivalently $\textstyle \sum _ { i = 1 } ^ { d } y _ { i } ^ { 2 } / ( r ^ { 2 } / \lambda _ { i } ) \leq 1$ . Comparing this with the standard ellipsoid form $\textstyle \sum _ { i = 1 } ^ { d } y _ { i } ^ { 2 } / a _ { i } ^ { 2 } \leq 1$ , the principal semiaxis along the ith eigenvector of H is $a _ { i } ( \boldsymbol r ) = \boldsymbol r / \sqrt { \lambda _ { i } }$ . Thus, r controls the overall scale of the ellipsoid, while the eigenvalues $\lambda _ { i }$ determine its relative shape across directions. Increasing r by a factor α expands every principal semiaxis by the same factor α.

The efect of dimension can also be seen from the volume of the ellipsoid. Its d-dimensional volume is

$$
\mathrm { V o l } _ { d } \big ( \mathcal { B } _ { H } ( r ) \big ) = \frac { \pi ^ { d / 2 } } { \Gamma ( d / 2 + 1 ) } \frac { r ^ { d } } { \sqrt { \operatorname* { d e t } H } } .
$$

Defining $g _ { H } = ( \operatorname* { d e t } H ) ^ { 1 / d }$ , and applying Stirling’s approximation gives

$$
\mathrm { V o l } _ { d } \big ( \mathcal { B } _ { H } ( r ) \big ) \sim \frac { 1 } { \sqrt { \pi d } } \left( \frac { 2 \pi e r ^ { 2 } } { d g _ { H } } \right) ^ { d / 2 } .
$$

Therefore, under a fixed coordinate normalization, the volume represented by a fixed radius r can change substantially with the nullspace dimension d. For example, when H = I and $r = 1$ , the ellipsoid reduces to the Euclidean unit ball. Its volume decreases from approximately 2.55 in 10 dimensions to $2 . 5 8 \times 1 0 ^ { - 2 }$ in 20 dimensions, and further to 1 $. 7 3 \times 1 0 ^ { - 1 3 }$ in 50 dimensions. Thus, the same numerical value of r can correspond to very diferent volumetric coverage as the dimension of the nullspace changes.

This observation suggests a simple heuristic for choosing the radius as the dimension changes. From the leading term in the volume approximation, the quantity $\frac { r ^ { 2 } } { d g _ { H } }$ determines the exponential scale of the ellipsoid volume, suggesting the scaling $r \propto \sqrt { d g _ { H } }$ . We first select a representative reference instance and choose $r _ { 0 }$ from a set of candidate radii by minimizing the downstream regret on a validation set. Let $d _ { 0 }$ and $H _ { 0 }$ denote the nullspace dimension and Hessian of this reference instance, and let $g _ { H , 0 } = ( \operatorname* { d e t } H _ { 0 } ) ^ { 1 / d _ { 0 } }$ denote the corresponding Hessian geometric mean. We scale the validated radius $r _ { 0 }$ to a new instance according to

$$
r _ { \mathrm { h e u r } } = \operatorname* { m a x } \left\{ 1 , { r _ { 0 } } \sqrt { \frac { d g _ { H } } { d _ { 0 } g _ { H , 0 } } } \right\} .
$$

For example, if $g _ { H } \approx g _ { H , 0 }$ and validation selects $r _ { 0 } = 1$ for a reference instance with $d _ { 0 } = 1 0$ , the rule gives approximately $r _ { \mathrm { h e u r } } = 2$ for $d = 4 0$ and $r _ { \mathrm { h e u r } } = 3$ for $d = 9 0$ . In this case, the rule reduces to $r \propto { \sqrt { d } } .$ The heuristic transfers the radius calibrated on the reference instance to problems with diferent dimensions and Hessian geometries, and the resulting value can be further refined using validation data if needed.

## Appendix B: Construction of the Local Error Distribution

This appendix describes how the error distribution $\mathcal { P }$ can be constructed from historical data and contextual information. Let $\mathcal { D } = \{ ( \boldsymbol { x } _ { d } , \boldsymbol { r } _ { d } ) \} _ { d = 1 } ^ { D }$ denote a historical dataset, where $x _ { d }$ contains contextual information associated with observation $d ,$ such as historical demand, socioeconomic characteristics, land use, and other factors related to travel demand. The residual $r _ { d }$ measures the error of the prior OD estimate for that observation. In particular, if $f _ { 0 , d }$ denotes the prior estimate and $f _ { d } ^ { \mathrm { o b s } }$ the corresponding historical OD observation, we define $r _ { d } = f _ { d } ^ { \mathrm { o b s } } - f _ { 0 , d }$

For a target instance with contextual features $x ^ { \star }$ , we use historical observations with similar features to characterize the uncertainty in the prior estimate $f _ { 0 } .$ . Following the local weighting perspective of Bertsimas and Kallus (2020), we define a neighborhood $\mathcal { N } ( x ^ { \star } ) \subseteq \{ 1 , \ldots , D \}$ that determines which historical residuals contribute to the local error distribution. A simple empirical construction is

$$
\widehat { \mathcal { P } } ( x ^ { \star } ) = \frac { 1 } { | \mathcal { N } ( x ^ { \star } ) | } \sum _ { d \in \mathcal { N } ( x ^ { \star } ) } \delta _ { r _ { d } } ,
$$

where $\delta _ { r _ { d } }$ denotes a point mass at residual $r _ { d } .$ . Diferent methods for defining $\mathcal { N } ( x ^ { \star } )$ correspond to diferent notions of similarity in the feature space. We consider several standard constructions below.

## B.1. k-nearest neighbors (kNN)

The k-nearest-neighbor neighborhood consists of the k historical observations whose feature vectors are closest to $x ^ { \star }$ under a chosen distance metric $d i s ( \cdot , \cdot )$

$$
\mathcal N _ { \mathrm { k N N } } ( x ^ { \star } ) = \{ d : x _ { d } \mathrm { ~ i s ~ a m o n g ~ t h e ~ } k \mathrm { ~ c l o s e s t ~ p o i n t s ~ t o ~ } x ^ { \star } \mathrm { ~ u n d e r ~ } d i s ( \cdot , \cdot ) \} .\tag{12}
$$

This construction fixes the number of residual observations used to estimate the local error distribution.

## B.2. ε-ball neighborhood

An ε-ball neighborhood includes all historical observations within distance ε of $x ^ { \star }$

$$
\begin{array} { r } { \mathcal { N } _ { \varepsilon } ( x ^ { \star } ) = \left\{ d : \| x _ { d } - x ^ { \star } \| \leq \varepsilon \right\} . } \end{array}\tag{13}
$$

The resulting neighborhood size adapts to the local density of the feature space.

## B.3. Decision tree-based neighborhood

A decision tree trained on the historical data partitions the feature space into axis-aligned regions. The neighborhood contains observations that fall in the same terminal leaf as $x ^ { \star }$ :

$$
\mathcal { N } _ { \mathrm { t r e e } } ( x ^ { \star } ) = \{ d \colon x _ { d } { \mathrm { ~ a n d ~ } } x ^ { \star } { \mathrm { ~ l i e ~ i n ~ t h e ~ s a m e ~ t e r m i n a l ~ l e a f } } \} .\tag{14}
$$

This construction allows the neighborhood to reflect feature relevance and interactions learned from the historical data.

## B.4. Kernel-based neighborhood

A kernel-based construction assigns each historical residual a weight

$$
w _ { d } = K \left( { \frac { \| x _ { d } - x ^ { \star } \| } { h } } \right) ,\tag{15}
$$

where $K ( \cdot )$ is a kernel function and $h$ is a bandwidth parameter. After normalization, $\begin{array} { r } { \tilde { w } _ { d } = w _ { d } / \sum _ { j = 1 } ^ { D } w _ { j } } \end{array}$ , the local error distribution can be written as

$$
\widehat { \mathcal { P } } ( x ^ { \star } ) = \sum _ { d = 1 } ^ { D } \tilde { w } _ { d } \delta _ { r _ { d } } .
$$

This weighted construction avoids a discrete neighborhood boundary and gives greater influence to historical observations that are more similar to the target instance.

## Appendix C: Proof of Proposition 1

We prove the three stated properties of $\ell _ { \alpha }$ for any realized OD-flow vector $f ^ { * }$ , candidate OD estimate $f ,$ and $\alpha > 0$

Proof. We begin by recalling the definition of regret. For a given decision rule $w ( f )$ , the regret associated with a realized OD flow $f ^ { * }$ is

$$
R ( f ^ { * } , f ) = f ^ { * } { } ^ { \top } w ( f ) - c ( f ^ { * } ) ,
$$

where $\begin{array} { r } { c ( f ^ { * } ) : = \operatorname* { m i n } _ { w \in \mathcal { W } } { f ^ { * } } ^ { \top } w = { f ^ { * } } ^ { \top } w ( f ^ { * } ) } \end{array}$ and $w ( f ) \in \arg \operatorname* { m i n } _ { w \in \mathcal { W } } f ^ { \intercal }$ w denotes an optimal solution to the downstream optimization problem under $f .$

Step 1 (Upper bound). Fix any $\alpha > 0$ . Using $f ^ { \top } w ( f ) = c ( f )$ , we have

$$
\begin{array} { r l } & { R ( f ^ { * } , f ) = f ^ { * \top } w ( f ) - c ( f ^ { * } ) } \\ & { \qquad = ( f ^ { * } - \alpha f ) ^ { \top } w ( f ) + \alpha c ( f ) - c ( f ^ { * } ) } \\ & { \qquad \le \underset { w \in \mathcal { W } } { \operatorname* { m a x } } ( f ^ { * } - \alpha f ) ^ { \top } w + \alpha c ( f ) - c ( f ^ { * } ) . } \end{array}
$$

Since $w ( f ^ { * } ) \in \mathcal { W }$ is feasible for the downstream problem under $f ,$

$$
c ( f ) = f ^ { \top } w ( f ) \leq f ^ { \top } w ( f ^ { * } ) .
$$

Because $\alpha > 0$ , it follows that

$$
\begin{array} { r l } & { R ( f ^ { * } , f ) \leq \displaystyle \operatorname* { m a x } _ { w \in \mathcal { W } } ( f ^ { * } - \alpha f ) ^ { \top } w + \alpha f ^ { \top } w ( f ^ { * } ) - c ( f ^ { * } ) } \\ & { \quad \quad \quad = \ell _ { \alpha } ( f ^ { * } , f ) . } \end{array}
$$

Hence,

$$
R ( f ^ { * } , f ) \leq \ell _ { \alpha } ( f ^ { * } , f ) ,
$$

which proves the first property in Proposition 1.

Step 2 (Convexity). For fixed $f ^ { * }$ , the surrogate can be written as

$$
\ell _ { \alpha } ( f ^ { * } , f ) = \operatorname* { m a x } _ { w \in \mathcal { W } } \left\{ ( f ^ { * } - \alpha f ) ^ { \top } w + \alpha f ^ { \top } w ( f ^ { * } ) - c ( f ^ { * } ) \right\} .
$$

For each fixed $w \in \mathcal { W }$ , the expression inside the maximum is afine, and hence convex, in $f .$ Therefore, by the supremum rule for convex functions (Royset and Wets 2021, Proposition 1.18(a)), $\ell _ { \alpha } ( f ^ { * } , \cdot )$ is convex. This proves the second property in Proposition 1.

Step 3 (Subgradient characterization).

For fixed $f ^ { * }$ and each $w \in \mathcal { W }$ , the expression inside the maximum in (7) is afine in $f ,$ with gradient $\alpha \big ( w ( f ^ { * } ) - w \big )$ . Since W is compact, Proposition 6.30 of Royset and Wets (2021) implies

$$
\partial _ { f } \ell _ { \alpha } ( f ^ { * } , f ) = \mathrm { c o n } \left\{ { \alpha } \big ( w ( f ^ { * } ) - w \big ) : w \in \arg \operatorname* { m a x } _ { \tilde { w } \in W } ( f ^ { * } - { \alpha } f ) ^ { \top } \tilde { w } \right\} ,
$$

where $\mathrm { c o n } ( \cdot )$ denotes the convex hull. Moreover,

$$
\arg \operatorname* { m a x } _ { w \in \mathcal { W } } ( f ^ { * } - \alpha f ) ^ { \top } w = \arg \operatorname* { m i n } _ { w \in \mathcal { W } } ( \alpha f - f ^ { * } ) ^ { \top } w .
$$

Therefore, choosing $w ( \alpha f - f ^ { * } )$ from this optimal set yields

$$
\alpha \big ( w ( f ^ { * } ) - w ( \alpha f - f ^ { * } ) \big ) \in \partial _ { f } \ell _ { \alpha } ( f ^ { * } , f ) ,
$$

which proves the third property in Proposition 1.

## Appendix D: Proof of Proposition 2

Proof. We first decompose the discrepancy between the surrogate evaluated at the realized OD flow $f ^ { * }$ and the regret under the true OD flow $f _ { \mathrm { t r u e } }$ as

$$
\ell _ { \alpha } ( f ^ { * } , f ) - R ( f _ { \mathrm { t r u e } } , f ) = \left[ \ell _ { \alpha } ( f _ { \mathrm { t r u e } } , f ) - R ( f _ { \mathrm { t r u e } } , f ) \right] + \left[ \ell _ { \alpha } ( f ^ { * } , f ) - \ell _ { \alpha } ( f _ { \mathrm { t r u e } } , f ) \right] .
$$

We bound these two terms separately.

Step 1 (Surrogate approximation). From the definitions of $\ell _ { \alpha }$ and $R ,$ we have

$$
\ell _ { \alpha } ( f _ { \mathrm { t r u e } } , f ) - R ( f _ { \mathrm { t r u e } } , f ) = \operatorname* { m a x } _ { w \in \mathcal W } ( f _ { \mathrm { t r u e } } - \alpha f ) ^ { \top } \big ( w - w ( f ) \big ) + \alpha f ^ { \top } \big ( w ( f _ { \mathrm { t r u e } } ) - w ( f ) \big ) .
$$

For any $w \in \mathcal { W }$ , the boundedness of $f _ { \mathrm { t r u e } }$ and the sharpness condition imply

$$
\begin{array} { r l } & { \left( f _ { \mathrm { t r u e } } - \alpha f \right) ^ { \top } \left( w - w ( f ) \right) = f _ { \mathrm { t r u e } } ^ { \top } \big ( w - w ( f ) \big ) - \alpha f ^ { \top } \big ( w - w ( f ) \big ) } \\ & { \qquad \leq \big ( U - \alpha \kappa \big ) \| w - w ( f ) \| . } \end{array}
$$

If $\alpha \kappa \geq U$ , the right-hand side is nonpositive for every $w \in \mathcal { W }$ , while choosing $w = w ( f )$ gives zero. Hence, the maximum is zero. If $\alpha \kappa < U$ , then $\| w - w ( f ) \| \le 2 B _ { w }$ and $U < U ^ { 2 } / ( \kappa \alpha )$ , so

$$
\operatorname* { m a x } _ { w \in \mathcal { W } } ( f _ { \mathrm { t r u e } } - \alpha f ) ^ { \top } \big ( w - w ( f ) \big ) \leq \frac { 2 U ^ { 2 } B _ { w } } { \kappa \alpha } .
$$

Therefore, the same bound holds for every $\alpha > 0$

For the second term, the optimality of $w ( f _ { \mathrm { t r u e } } )$ under $f _ { \mathrm { t r u e } }$ gives $f _ { \mathrm { t r u e } } ^ { \top } ( w ( f _ { \mathrm { t r u e } } ) - w ( f ) ) \leq 0$ . Thus,

$$
f ^ { \top } \big ( w ( f _ { \mathrm { t r u e } } ) - w ( f ) \big ) = ( f - f _ { \mathrm { t r u e } } ) ^ { \top } \big ( w ( f _ { \mathrm { t r u e } } ) - w ( f ) \big ) + f _ { \mathrm { t r u e } } ^ { \top } \big ( w ( f _ { \mathrm { t r u e } } ) - w ( f ) \big )
$$

$$
\leq 2 B _ { w } \| f - f _ { \mathrm { t r u e } } \| .
$$

Combining the two bounds yields

$$
\ell _ { \alpha } ( f _ { \mathrm { t r u e } } , f ) - R ( f _ { \mathrm { t r u e } } , f ) \leq \frac { 2 U ^ { 2 } B _ { w } } { \kappa \alpha } + 2 \alpha B _ { w } \| f - f _ { \mathrm { t r u e } } \| .
$$

Step 2 (Uncertainty perturbation). Using the definition of $\ell _ { \alpha }$

$$
\begin{array} { r l } & { \ell _ { \alpha } ( f ^ { * } , f ) - \ell _ { \alpha } ( f _ { \mathrm { t r u e } } , f ) = \displaystyle \operatorname* { m a x } _ { w \in \mathcal { W } } ( f ^ { * } - \alpha f ) ^ { \top } w - \displaystyle \operatorname* { m a x } _ { w \in \mathcal { W } } ( f _ { \mathrm { t r u e } } - \alpha f ) ^ { \top } w } \\ & { \qquad + \alpha f ^ { \top } \big ( w ( f ^ { * } ) - w ( f _ { \mathrm { t r u e } } ) \big ) + c ( f _ { \mathrm { t r u e } } ) - c ( f ^ { * } ) . } \end{array}
$$

The first diference is bounded by

$$
\operatorname* { m a x } _ { w \in \mathcal { W } } ( f ^ { * } - \alpha f ) ^ { \top } w - \operatorname* { m a x } _ { w \in \mathcal { W } } ( f _ { \mathrm { t r u e } } - \alpha f ) ^ { \top } w \leq \operatorname* { m a x } _ { w \in \mathcal { W } } ( f ^ { * } - f _ { \mathrm { t r u e } } ) ^ { \top } w \leq B _ { w } \| f ^ { * } - f _ { \mathrm { t r u e } } \| .
$$

Similarly, since $w ( f ^ { * } )$ is feasible under $f _ { \mathrm { t r u e } }$

$$
c ( f _ { \mathrm { t r u e } } ) - c ( f ^ { * } ) \leq ( f _ { \mathrm { t r u e } } - f ^ { * } ) ^ { \top } w ( f ^ { * } ) \leq B _ { w } \| f ^ { * } - f _ { \mathrm { t r u e } } \| .
$$

It remains to bound the term involving the two optimal decisions. Since $w ( f ^ { * } )$ is optimal under $f ^ { * }$ ${ f ^ { * } } ^ { \top } ( w ( f ^ { * } ) - w ( f _ { \mathrm { t r u e } } ) ) \le 0$ . Hence,

$$
f ^ { \top } \left( w ( f ^ { * } ) - w ( f _ { \mathrm { t r u e } } ) \right) = ( f - f ^ { * } ) ^ { \top } \left( w ( f ^ { * } ) - w ( f _ { \mathrm { t r u e } } ) \right) + f ^ { * } { } ^ { \top } \left( w ( f ^ { * } ) - w ( f _ { \mathrm { t r u e } } ) \right) \leq 2 B _ { w } \| f ^ { * } - f \| .
$$

Therefore,

$$
\ell _ { \alpha } ( f ^ { * } , f ) - \ell _ { \alpha } ( f _ { \mathrm { t r u e } } , f ) \leq 2 \alpha B _ { w } \| f ^ { * } - f \| + 2 B _ { w } \| f ^ { * } - f _ { \mathrm { t r u e } } \| .
$$

Step 3 (Combining the bounds). Combining Steps 1 and 2 gives

$$
\ell _ { \alpha } ( f ^ { * } , f ) - R ( f _ { \mathrm { t r u e } } , f ) \leq \frac { 2 U ^ { 2 } B _ { w } } { \kappa \alpha } + 2 \alpha B _ { w } \left( \| f - f _ { \mathrm { t r u e } } \| + \| f ^ { * } - f \| \right) + 2 B _ { w } \| f ^ { * } - f _ { \mathrm { t r u e } } \| ,
$$

which proves (8).

Finally, suppose that $\| f - f _ { \mathrm { t r u e } } \| + \| f ^ { * } - f \| > 0$ . The part of the bound that depends on α is

$$
\frac { 2 U ^ { 2 } B _ { w } } { \kappa \alpha } + 2 \alpha B _ { w } \left( \| f - f _ { \mathrm { t r u e } } \| + \| f ^ { * } - f \| \right) .
$$

Diferentiating with respect to α and setting the derivative equal to zero gives

$$
- \frac { 2 U ^ { 2 } B _ { w } } { \kappa \alpha ^ { 2 } } + 2 B _ { w } \left( \| f - f _ { \mathrm { t r u e } } \| + \| f ^ { * } - f \| \right) = 0 .
$$

Solving for α yields

$$
\alpha ^ { * } = \frac { U } { \sqrt { \kappa \left( \left\| f - f _ { \mathrm { t r u e } } \right\| + \left\| f ^ { * } - f \right\| \right) } } .
$$

Since the second derivative is positive for every $\alpha > 0$ , this is the unique minimizer of the α-dependent part of the bound. This proves (9). □

## Appendix E: Proof of Theorem 1

Proof. For a fixed realization ε, define

$$
h _ { \varepsilon } ( z ) : = \ell _ { \alpha } ( f _ { 0 } + \varepsilon , g ( z ) ) , \qquad g ( z ) = \bar { f } + V z .
$$

By Proposition 1, $\ell _ { \alpha } ( f _ { 0 } + \varepsilon , \cdot )$ is convex. Since $g ( z )$ is afine, the afine composition rule implies that $h _ { \varepsilon } ( z )$ is convex in z. Therefore,

$$
\begin{array} { r } { \phi ( z ) = \mathbb { E } _ { \varepsilon \sim \mathcal { P } } \left[ h _ { \varepsilon } ( z ) \right] , } \end{array}
$$

is convex, since expectation preserves convexity. Moreover, the feasible set $\{ z : z ^ { \top } H z \leq r ^ { 2 } \}$ is convex because $H \succ 0$ . Hence, the expectation formulation is a convex optimization problem.

We next establish the subgradient expression. By Proposition 1, for each realization $f _ { 0 } + \varepsilon$

$$
\alpha \left[ w ( f _ { 0 } + \varepsilon ) - w \big ( \alpha g ( z ) - f _ { 0 } - \varepsilon \big ) \right] \in \partial _ { f } \ell _ { \alpha } \big ( f _ { 0 } + \varepsilon , g ( z ) \big ) .
$$

Since $g ( z ) = \bar { f } + V z$ is afine, the afine subgradient chain rule gives

$$
\alpha V ^ { \top } \left[ w ( f _ { 0 } + \varepsilon ) - w \big ( \alpha g ( z ) - f _ { 0 } - \varepsilon \big ) \right] \in \partial _ { z } h _ { \varepsilon } ( z ) .
$$

Under the stated integrability conditions, Proposition 3.23 of Royset and Wets (2021) allows the expectation and subgradient operations to be interchanged. Therefore,

$$
\begin{array} { r } { \mathbb { E } _ { \varepsilon \sim \mathcal { P } } \left[ \alpha V ^ { \top } \left( w ( f _ { 0 } + \varepsilon ) - w \big ( \alpha g ( z ) - f _ { 0 } - \varepsilon \big ) \right) \right] \in \partial \phi ( z ) , } \end{array}
$$

which is the subgradient expression stated in Theorem 1.

## Appendix F: Extension to Alternative Risk Measures

This appendix provides the formal statement and proof of the extension mentioned in the main text, showing that the convexity and subgradient characterization also applies to the Conditional Value-at-Risk (CVaR) formulation. Throughout this appendix, $\beta \in ( 0 , 1 )$ denotes the CVaR confidence level, while $\alpha > 0$ remains the scaling parameter in the surrogate loss.

Theorem 2 (Convexity and Subgradient of the CVaR Formulation). Let $H \in \mathbb { R } ^ { d \times d }$ be positive definite and consider

$$
\operatorname* { m i n } _ { \substack { z \in \mathbb { R } ^ { d } : z ^ { \top } H z \leq r ^ { 2 } } } \psi ( z ) : = \mathrm { C V a R } _ { \beta } \left[ \ell _ { \alpha } ( f _ { 0 } + \varepsilon , g ( z ) ) \right] , \qquad \varepsilon \sim \mathcal { P } ,
$$

where $g ( z ) = \bar { f } + V z$ . Assume that the loss and the subgradient expression below are integrable with respect to $\varepsilon \sim \mathcal { P }$ . Then:

1. $\psi ( z )$ is convex in z, and the optimization problem is a convex optimization problem;

2. let $\eta ^ { * } ( z )$ be an optimal threshold in the Rockafellar–Uryasev representation of $\mathrm { C V a R } _ { \beta }$ . For each realization ε, let $\xi _ { \eta ^ { * } ( z ) } ( \varepsilon ) \in [ 0 , 1 ]$ satisfy $\xi _ { \eta ^ { \ast } ( z ) } ( \varepsilon ) \in \partial _ { u } \bigl ( u - \eta ^ { \ast } ( z ) \bigr ) _ { + } \Big | _ { u = \ell _ { \alpha } ( f _ { 0 } + \varepsilon , g ( z ) ) }$ and $\mathbb { E } _ { \varepsilon \sim \mathcal { P } } \left[ \xi _ { \eta ^ { \ast } \left( z \right) } \mathopen { } \mathclose \bgroup \left( \varepsilon \right) \right] = 1 - \beta$ Then a subgradient of ψ at z is

$$
\frac { \alpha } { 1 - \beta } \mathbb { E } _ { \varepsilon \sim \mathcal { P } } \left[ \xi _ { \eta ^ { * } ( z ) } ( \varepsilon ) V ^ { \top } \left( w ( f _ { 0 } + \varepsilon ) - w \big ( \alpha g ( z ) - f _ { 0 } - \varepsilon \big ) \right) \right] \in \partial \psi ( z ) .
$$

Proof. For a fixed realization ε, define

$$
Y _ { z } ( \varepsilon ) : = \ell _ { \alpha } ( f _ { 0 } + \varepsilon , g ( z ) ) .
$$

By Proposition 1, $\ell _ { \alpha } ( f _ { 0 } + \varepsilon , \cdot )$ is convex. Since $g ( z ) = \bar { f } + V z$ is afine, $Y _ { z } ( \varepsilon )$ is convex in z.

By the Rockafellar-Uryasev representation,

$$
\psi ( z ) = \operatorname* { m i n } _ { \eta \in \mathbb { R } } \left\{ \eta + \frac { 1 } { 1 - \beta } \mathbb { E } _ { \varepsilon \sim \mathcal { P } } \left[ \big ( Y _ { z } ( \varepsilon ) - \eta \big ) _ { + } \right] \right\} .
$$

Define

$$
\varphi ( z , \eta ) : = \eta + \frac { 1 } { 1 - \beta } \mathbb { E } _ { \varepsilon \sim \mathcal { P } } \left[ \left( Y _ { z } ( \varepsilon ) - \eta \right) _ { + } \right] .
$$

For each realization $\varepsilon , ( Y _ { z } ( \varepsilon ) - \eta ) _ { + }$ is jointly convex in $( z , \eta )$ because it is the maximum of the convex function $Y _ { z } ( \varepsilon ) - \eta$ and zero. Expectation preserves convexity, and hence $\varphi ( z , \eta )$ is jointly convex in $( z , \eta )$ . Therefore, partial minimization over η preserves convexity, so $\begin{array} { r } { \psi ( z ) = \operatorname* { m i n } _ { \eta \in \mathbb { R } } \varphi ( z , \eta ) ~ \{ z : z ^ { \top } H z \leq r ^ { 2 } \} } \end{array}$ is convex. Thus, the CVaR formulation is a convex optimization problem.

We next establish the subgradient characterization. By Proposition 1, for each realization $\varepsilon ,$

$$
\alpha \left[ w ( f _ { 0 } + \varepsilon ) - w \big ( \alpha g ( z ) - f _ { 0 } - \varepsilon \big ) \right] \in \partial _ { f } \ell _ { \alpha } \big ( f _ { 0 } + \varepsilon , g ( z ) \big ) .
$$

Since $g ( z ) = \bar { f } + V z$ is afine,

$$
\alpha V ^ { \top } \left[ w ( f _ { 0 } + \varepsilon ) - w \big ( \alpha g ( z ) - f _ { 0 } - \varepsilon \big ) \right] \in \partial _ { z } Y _ { z } ( \varepsilon ) .
$$

Now fix an optimal threshold $\begin{array} { r } { \eta ^ { \ast } ( z ) \in \arg \operatorname* { m i n } _ { \eta } \varphi ( z , \eta ) } \end{array}$ . For each realization, choose

$$
\begin{array} { r } { \xi _ { \eta ^ { \ast } \left( z \right) } ( \varepsilon ) \in \partial _ { u } \left( u - \eta ^ { \ast } ( z ) \right) _ { + } \bigg \rvert _ { u = Y _ { z } \left( \varepsilon \right) } , } \end{array}
$$

where

$$
\xi _ { \eta ^ { \ast } ( z ) } ( \varepsilon ) = \left\{ \begin{array} { l l } { 0 , } & { Y _ { z } ( \varepsilon ) < \eta ^ { \ast } ( z ) , } \\ { [ 0 , 1 ] , } & { Y _ { z } ( \varepsilon ) = \eta ^ { \ast } ( z ) , } \\ { 1 , } & { Y _ { z } ( \varepsilon ) > \eta ^ { \ast } ( z ) . } \end{array} \right.
$$

The optimality of $\eta ^ { * } ( z )$ implies that the selection at the threshold can be chosen so that

$$
\begin{array} { r } { \mathbb { E } _ { \varepsilon \sim \mathcal { P } } \left[ \xi _ { \eta ^ { \ast } \left( z \right) } \mathopen { } \mathclose \bgroup \left( \varepsilon \right) \right] = 1 - \beta . } \end{array}
$$

Applying the subgradient chain rule and taking expectations therefore gives

$$
\frac { \alpha } { 1 - \beta } \mathbb { E } _ { \varepsilon \sim \mathcal { P } } \left[ \xi _ { \eta ^ { * } ( z ) } ( \varepsilon ) V ^ { \top } \left( w ( f _ { 0 } + \varepsilon ) - w \big ( \alpha g ( z ) - f _ { 0 } - \varepsilon \big ) \right) \right] \in \partial \psi ( z ) ,
$$

which proves the stated subgradient expression.

## Appendix G: Demonstration Decision Problem

We consider a stylized investment allocation problem faced by an urban transportation authority. The authority is endowed with a fixed capital budget B and seeks to allocate this budget across a set of candidate road segments in order to reduce aggregate commuter travel time in the city. Typical interventions include road widening, signal timing optimization, and intersection redesign, each of which improves link-level travel eficiency.

Let $J = \{ 1 , \ldots , m \}$ denote the set of candidate road segments eligible for investment. Prior to improvement, segment $j \in J$ has a baseline per-traveler travel time $t _ { j } ^ { 0 } .$ . The city’s travel demand is represented by OD demand vector $f \in \mathbb { R } _ { + } ^ { n }$ , where $f _ { i }$ denotes the daily travel demand associated with OD pair $i \in N : = \{ 1 , \dots , n \}$ The demand vector $f$ is generated by an upstream prediction model and serves as the key informational input to the downstream investment allocation decision. Network connectivity between OD pairs and road segments is described by a binary incidence matrix $S \in \{ 0 , 1 \} ^ { n \times m }$ , where $S _ { i j } = 1 \mathrm { i f } \mathrm { O D }$ pair i traverses segment $j ,$ , and $S _ { i j } = 0$ otherwise. We abstract from endogenous route choice and network equilibrium efects. Each OD pair is assumed to follow a representative baseline path, and link-level improvements are aggregated along this fixed path to obtain OD-level travel time. For each road segment $j \in J ,$ , let $x _ { j }$ denote the expected reduction (in minutes) in per-traveler travel time achieved through investment on that segment. We impose an upper bound $u _ { j }$ reflecting engineering or physical feasibility constraints, here $u _ { j } \leq t _ { j } ^ { 0 }$ . Let $w _ { i }$ denote the per trip travel time for OD pair i as a result of the investments. Let $c _ { j }$ denote the marginal cost of reducing one unit (minute) of travel time on segment $j .$ Thus, the total investment cost on segment $j$ equals $c _ { j } x _ { j }$ , and the aggregate expenditure across all segments must not exceed the available budget B. The sets, parameters, and decision variables are summarized in Table 2

Table 2 Notation Summary for the Demonstration Decision Problem
<table><tr><td colspan="2">Sets</td></tr><tr><td> $\overline { { J = \{ 1 , \dots , m \} } }$   $N = \{ 1 , \ldots , n \}$ </td><td>Set of candidate road segments eligible for investment Set of OD pairs</td></tr><tr><td colspan="2">Parameters</td></tr><tr><td> $\overline { { f _ { i } } }$   $S _ { i j }$   $c _ { j }$ </td><td>Daily travel demand associated with OD pair  $\overline { { i } }$  Binary incidence indicator;  $S _ { i j } = 1$  if OD pair i traverses segment  $j$  Marginal cost of reducing one unit (minute) of travel time on segment  $j$   $j$ </td></tr><tr><td colspan="2"> $u _ { j }$  Maximum feasible travel time reduction on segment  $B$  Total available investment budget</td></tr><tr><td colspan="2">Decision Variables  $x _ { j }$  Per-trip travel time reduction (minutes) achieved on segment</td></tr></table>

The transportation authority seeks to minimize the total system-wide travel time experienced by al commuters. Because $f _ { i }$ represents the demand for OD pair $i ,$ the aggregate daily travel time equals $\textstyle \sum _ { i \in N } f _ { i } w _ { i }$ The resulting linear program is:

$$
\operatorname* { m i n } _ { x , w } \quad \sum _ { i \in N } f _ { i } w _ { i }\tag{16}
$$

$$
\mathrm { s . t . } \quad w _ { i } = \sum _ { j \in J } S _ { i j } ( t _ { j } ^ { 0 } - x _ { j } ) ,
$$

$$
\forall i \in N ,\tag{17}
$$

$$
\sum _ { j \in J } c _ { j } x _ { j } \leq B ,\tag{18}
$$

$$
0 \leq x _ { j } \leq u _ { j } ,
$$

$$
\forall j \in J .\tag{19}
$$

Constraint (17) defines the post investment travel time for each OD pair as the sum of the improved link travel times along its route. Constraint (18) enforces the budget limitation. Constraint (19) ensures feasibility of link-level improvements. This formulation captures a fundamental trade-of in transportation planning: investments on heavily utilized corridors yield large aggregate benefits, yet marginal improvement costs and physical constraints limit achievable reductions.

## Appendix H: Supplementary Details for Experiment 4.2

## H.1. Feature and OD Flow Generation

For each experimental instance, we generate synthetic OD-flow samples from a nonlinear feature-to-demand mapping. Each sample is associated with a feature vector $x _ { i } \in \mathbb { R } ^ { d _ { \mathrm { f e a t } } }$ , where $d _ { \mathrm { f e a t } } = 4$ . The first three feature

dimensions are independently drawn from a standard normal distribution, while the fourth dimension is a binary indicator:

$$
x _ { i 1 } , x _ { i 2 } , x _ { i 3 } \sim \mathcal { N } ( 0 , 1 ) , \qquad x _ { i 4 } \sim \mathrm { B e r n o u l l i } ( 0 . 5 ) .
$$

Training and testing feature vectors are generated independently using diferent random seeds.

Let $n _ { \mathrm { O D } }$ denote the number of OD pairs. For a given polynomial order $P \in \{ 1 , \ldots , 6 \}$ , the latent OD-flow vector is generated by

$$
Z _ { i } = b + x _ { i } W _ { 1 } ^ { \top } + \sum _ { p = 2 } ^ { P } x _ { i } ^ { \circ p } W _ { p } ^ { \top } ,
$$

where $x _ { i } ^ { \circ p }$ denotes the elementwise p-th power of the feature vector.

The nonlinear mapping is parameterized so that lower-order terms capture the dominant feature efects, while higher-order terms introduce progressively smaller nonlinear perturbations. Specifically, for each experimental instance, we first draw an intercept vector $b \in \mathbb { R } ^ { n _ { \mathrm { O D } } }$ and a collection of coeficient matrices $\{ W _ { p } \} _ { p = 1 } ^ { P }$ where $W _ { p } \in \mathbb { R } ^ { n _ { \mathrm { O D } } \times d _ { \mathrm { f e a t } } }$ . The entries are sampled independently according to

$$
b _ { j } \sim U ( 0 , 3 ) , \qquad ( W _ { p } ) _ { j k } \sim U ( - \rho _ { p } , \rho _ { p } ) ,
$$

with

$$
( \rho _ { 1 } , \rho _ { 2 } , \rho _ { 3 } , \rho _ { 4 } , \rho _ { 5 } , \rho _ { 6 } ) = ( 1 , 0 . 5 , 0 . 2 , 0 . 1 , 0 . 0 5 , 0 . 0 2 ) .
$$

Only the matrices $W _ { 1 } , \ldots , W _ { P }$ are used for a polynomial order P. This scaling prevents high-order polynomia terms from dominating the generated OD flows, while still allowing the true demand function to become increasingly nonlinear as P increases.

Given the latent value $Z _ { i } ,$ the clean OD-flow vector is constructed through a softplus transformation, $\widetilde { Q } _ { i } = \log ( 1 + \exp ( Z _ { i } ) )$ , which ensures nonnegativity. The transformed values are then rescaled to the target OD-flow magnitude and truncated to a bounded interval:

$$
Q _ { i } = \Pi _ { [ 0 , 2 0 0 0 ] } \left( s \widetilde { Q } _ { i } \right) ,
$$

where $s > 0$ is a fixed scaling constant and $\Pi _ { [ 0 , 2 0 0 0 ] } ( \cdot )$ denotes componentwise projection onto the interval [0, 2000]. This step keeps the generated OD flows on a stable numerical scale and avoids unrealistically large demands induced by rare high-order polynomial values. The scaling factor places the OD flows on a realistic numerical scale for the downstream investment problem, while the clipping step avoids unrealistically large demands caused by rare high-order polynomial values.

## Appendix I: Supplementary Details for Experiment 4.3

## I.1. OD Demand Patterns Generation

Common construction. Let $m \in \{ 1 , \ldots , 5 \}$ index the five demand-generation modes. For each OD pair $( i , j )$ , let $d _ { i j }$ denote the shortest-path distance, normalized by the median of all positive shortest-path distances in the network. For each mode m, we first construct a base OD pattern

$$
\widetilde { F } ^ { ( m ) } = ( 1 - \lambda ) B + \lambda S ^ { ( m ) } , \qquad \lambda = 0 . 8 5 ,
$$

where B is a common nonnegative gravity-type background and $S ^ { ( m ) }$ is the nonnegative mode-specific component.

The common background B incorporates heterogeneous origin-production and destination-attraction efects, moderate distance decay, and mild cell-level variation. It introduces a shared level of spatial heterogeneity across all modes and prevents the resulting OD patterns from becoming artificially sparse. The larger weight on $S ^ { ( m ) }$ preserves the distinctive structure of each demand-generation mode. Intrazonal demand is set to zero.

Pattern variants. As illustrated in Figure 11, the five mode-specific components $S ^ { ( m ) }$ are constructed as follows.

1. Distance decay. Demand is generated from heterogeneous origin and destination propensities and decreases exponentially with network distance. This pattern represents a conventional gravity-type structure in which nearby OD pairs interact more strongly.

2. Low-rank latent demand. The demand signal is obtained from the nonnegative part of a rank-three bilinear factor model and is moderately attenuated by distance. This construction captures correlated OD movements driven by a small number of latent factors.

3. Community blocks. Nodes are randomly assigned to three communities, with substantially stronger demand within communities than across communities. Cell-level variation is retained so that flows remain heterogeneous within each block.

4. Directional corridors. Disjoint source and sink sets, each containing approximately one quarter of the nodes, are selected to form dominant source-to-sink corridors. Corridor demand decreases with distance, while a weak difuse component allows residual demand outside the primary corridors, producing a strongly asymmetric OD structure.

5. Hub–spoke concentration. The node with the highest network degree is designated as the hub. Demand is concentrated on trips originating from or terminating at the hub, with a small residual component representing interactions among nonhub nodes.

Demand scenarios. Let $\ell \in \{ 1 , 2 , 3 \}$ index the low, medium, and high demand levels, with prescribed aggregate demands $Q _ { \ell } \in \{ 1 , 0 0 0 , 1 0 , 0 0 0 , 2 0 , 0 0 0 \}$ , respectively. Combining the five demand-generation modes with the three demand levels yields 15 scenarios.

To ensure that the three levels are not simple scalar copies of the same OD pattern, we introduce levelspecific origin and destination efects. For each level ℓ, define

$$
M _ { i j } ^ { ( \ell ) } = \exp \left( \alpha _ { i } ^ { ( \ell ) } + \beta _ { j } ^ { ( \ell ) } \right) ,
$$

where $\alpha _ { i } ^ { ( \ell ) }$ and $\beta _ { j } ^ { ( \ell ) }$ denote the origin- and destination-specific efects, respectively. The efects are centered, while their dispersion varies across demand levels.

The ground-truth OD matrix for mode m and demand level ℓ is then constructed as

$$
F _ { \mathrm { t r u e } } ^ { ( m , \ell ) } = \mathrm { S c a l e } _ { Q _ { \ell } } \left( \widetilde { F } ^ { ( m ) } \odot M ^ { ( \ell ) } \right) ,
$$

where ⊙ denotes elementwise multiplication and, for any nonnegative matrix A with zero diagonal,

$$
\mathrm { S c a l e } _ { Q } ( A ) = { \frac { Q } { \sum _ { i \neq j } A _ { i j } } } A .
$$

Hence, $\begin{array} { r } { \sum _ { i \neq j } F _ { \mathrm { t r u e } , i j } ^ { ( m , \ell ) } = Q _ { \ell } } \end{array}$

For each demand level $\ell ,$ the same multiplier $M ^ { ( \ell ) }$ is applied across all five modes, and fixed random seeds are used throughout. This design preserves comparability across modes while allowing both aggregate demand and OD margins to vary across demand levels.

Finally, each ground-truth OD matrix is vectorized as $f _ { \mathrm { t r u e } } ^ { ( m , \ell ) }$ and mapped to link flows through the common assignment relation

$$
L ^ { ( m , \ell ) } = X f _ { \mathrm { t r u e } } ^ { ( m , \ell ) } .
$$

Thus, all scenarios share the same network topology and assignment mechanism, while varying only in their OD structure and demand level.

## I.2. Repeated-Sampling Robustness Analysis

To examine whether the conclusions from the Sioux Falls experiment are robust to variation across experimental realizations, we conduct an additional repeated-sampling analysis. Each of the 15 demand-level–mode combinations is independently evaluated over 20 experimental realizations generated using distinct random seeds. Table 3 reports the sample mean and sample standard deviation of decision regret across these realizations.

Table 3 Repeated-sampling decision regret across demand settings on the Sioux Falls network.
<table><tr><td rowspan="2">Demand Level Demand Mode</td><td rowspan="2"></td><td colspan="4">Mean Decision Regret (SD)</td><td rowspan="2"> $\pmb { r } ^ { * }$ </td></tr><tr><td>OLS</td><td>PTO</td><td>D-FULL-G</td><td>D-GEO-G</td></tr><tr><td rowspan="5">Low</td><td>Distance Decay</td><td>18.51 (7.08)</td><td>3.62 (2.69)</td><td>1.37 (1.27)</td><td>1.29 (1.23)</td><td>5</td></tr><tr><td>Low Rank</td><td>51.58 (23.68)</td><td>21.95 (20.49)</td><td>4.09 (5.35)</td><td>4.09 (5.35)</td><td>10</td></tr><tr><td>Community Blocks</td><td>53.70 (18.13)</td><td>29.06 (21.74)</td><td>3.71 (3.27)</td><td>3.71 (3.27)</td><td>10</td></tr><tr><td>Directional Corridor</td><td>48.65 (28.08)</td><td>11.79 (10.98)</td><td>4.59 (5.56)</td><td>5.03 (5.48)</td><td>10</td></tr><tr><td>Hub Spoke</td><td>107.04 (28.83)</td><td>5.12 (4.37)</td><td>1.69 (1.94)</td><td>2.27 (2.65)</td><td>10</td></tr><tr><td rowspan="5">Medium</td><td>Distance Decay</td><td>211.90 (113.20)</td><td>33.70 (24.28)</td><td>14.92 (15.19)</td><td>12.04 (13.62)</td><td>3</td></tr><tr><td>Low Rank</td><td>531.08 (235.01)</td><td>235.79 (163.89)</td><td>34.39 (57.87)</td><td>34.98 (58.72)</td><td>10</td></tr><tr><td>Community Blocks</td><td>596.87 (180.51)</td><td>282.17 (224.14)</td><td>32.04 (35.70)</td><td>32.04 (35.70)</td><td>10</td></tr><tr><td>Directional Corridor</td><td>468.94 (320.83)</td><td>119.44 (143.96)</td><td>53.01 (69.35)</td><td>59.26 (78.96)</td><td>10</td></tr><tr><td>Hub Spoke</td><td>1098.04 (344.61)</td><td>57.66 (42.63)</td><td>17.39 (15.50)</td><td>16.25 (15.10)</td><td>10</td></tr><tr><td rowspan="5">High</td><td>Distance Decay</td><td>345.48 (190.05)</td><td>65.71 (47.21)</td><td>23.94 (22.17)</td><td>22.58 (21.82)</td><td>5</td></tr><tr><td>Low Rank</td><td>984.47 (413.93)</td><td>490.46 (361.87)</td><td>72.98 (61.36)</td><td>72.98 (61.36)</td><td>10</td></tr><tr><td>Community Blocks</td><td>1184.09 (409.70)</td><td>619.09 (430.08)</td><td>98.47 (78.00)</td><td>98.47 (78.00)</td><td>10</td></tr><tr><td>Directional Corridor</td><td>1059.13 (518.97)</td><td>277.58 (229.15)</td><td>92.71 (88.28)</td><td>88.32 (89.99)</td><td>10</td></tr><tr><td>Hub Spoke</td><td>1924.63 (651.64)</td><td>106.03 (91.86)</td><td>43.43 (45.71)</td><td>44.43 (47.86)</td><td>10</td></tr></table>

Notes. Values are reported as mean (sample standard deviation (SD)) over 20 independent experimental realizations. All decision-regret values are divided by $1 0 ^ { 3 } .$ . For D-GEO-G, $r ^ { * }$ denotes the radius attaining the lowest mean decision regret among the tested radii. Bold values indicate the lowest mean decision regret within each demand setting, with both values highlighted in the case of a tie.

The repeated-sampling results provide consistent evidence for the advantage of decision aware estimation. In every one of the 15 demand-level–mode combinations, both D-FULL-G and D-GEO-G achieve strictly lower mean decision regret than PTO, with no exception across the three demand levels or the five demand modes. Relative to PTO, the reduction in mean regret ranges from 55.6% to 88.6% for D-FULL-G and from 50.4% to 88.6% for D-GEO-G. Averaged across all settings, mean regret decreases from 157.28 under PTO to 33.25 under D-FULL-G and 33.18 under D-GEO-G, corresponding to an average reduction of approximately 78.9% for either variant. The improvement is also reflected in the variability across experimental realizations: both DWE variants have a smaller sample standard deviation than PTO in all 15 settings. For example, under high-demand Community Blocks, PTO produces a mean regret of 619.09 with a standard deviation of 430.08, whereas both D-FULL-G and D-GEO-G reduce these values to 98.47 and 78.00, respectively. Similarly, under medium-demand Directional Corridor, D-FULL-G and D-GEO-G reduce the PTO mean regret from 119.44 to 53.01 and 59.26, while their standard deviations decrease from 143.96 to 69.35 and 78.96. Thus, the gains from decision aware estimation are not limited to lower average regret; they are accompanied by consistently lower variation across repeated realizations. The comparison between the two DWE variants does not indicate uniform dominance. D-GEO-G achieves lower mean regret than D-FULL-G in five settings, matches it in five settings, and has higher mean regret in the remaining five, while their overal averages remain nearly identical. Its largest relative improvement occurs under medium-demand Distance Decay, where mean regret decreases from 14.92 to 12.04, a reduction of 19.3% relative to D-FULL-G. Its largest absolute shortfall occurs under medium-demand Directional Corridor, where mean regret increases from 53.01 to 59.26; nevertheless, D-GEO-G still achieves a strictly lower mean regret than PTO by 50.4% in this setting. These diferences reflect the trade-of introduced by the geometric restriction. D-FULL-G permits a more flexible adjustment, whereas D-GEO-G restricts the admissible adjustment through the radius r. This restriction can limit sample-specific over-adjustment, but it may also exclude corrections that are important to the downstream decision and thereby introduce approximation error. Notably, the selected radius reaches $r ^ { * } = 1 0$ , the maximum radius considered, in all five settings where D-GEO-G has higher mean regret than D-FULL-G. This boundary behavior is consistent with the admissible geometric region remaining too restrictive in these cases. The underperformance is concentrated in the Low Rank, Directional Corridor, and Hub Spoke modes, whose global, directional, or hub-centered structures may be less well represented by a finite geometric neighborhood. Although this observation does not establish that increasing $r _ { \mathrm { m a x } }$ would necessarily eliminate the performance gap, it identifies the finite geometric restriction as a plausible source of approximation error. D-GEO-G should therefore be interpreted as a regularized alternative to D-FULL-G rather than as a uniformly dominating variant. Importantly, its occasional disadvantage relative to D-FULL-G does not afect the central robustness conclusion: D-GEO-G, like D-FULL-G, achieves strictly lower mean regret and lower cross-realization variability than PTO in every tested setting.