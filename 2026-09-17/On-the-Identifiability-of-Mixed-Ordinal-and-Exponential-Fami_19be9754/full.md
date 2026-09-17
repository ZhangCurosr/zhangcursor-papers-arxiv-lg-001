# On the Identifiability of Mixed Ordinal and Exponential Family Causal DAGs under Linear Parametric Models

Sambit Mishra

sambitmi@usc.edu

Ming Hsieh Department of Electrical and Computer Engineering   
University of Southern California   
Los Angeles, CA 90089, USA

Urbashi Mitra

Ming Hsieh Department of Electrical and Computer Engineering   
University of Southern California   
Los Angeles, CA 90089, USA

ubli@usc.edu

## Editor:

## Abstract

The problem of identifiability in linear parametric models (LPMs) whose nodes follow either an ordered logit model or a regular one-parameter exponential family is evaluated. The results go beyond classical structural equation models as well as results for nodes with observations from a homogeneous family of distributions. The main result establishes that the orientation of every edge joining an ordinal node to an exponential-family node is identi fiable from the joint distribution alone at every parameter value, provided the ordinal node has at least three categories and the exponential-family node at least three points of sup port, with no restriction on the suficient statistic. Converses show that both requirements are necessary: the three-category requirement is binding only for afine suficient statistics, and the three-point requirement is binding under the canonical link. The guarantee extends to orienting every such mixed ordinal-exponential family edge of a given d-node undirected skeleton. Numerical experiments illustrate the theoretical results by successfully separating orientations within a Markov equivalence class, which are indistinguishable by conditional independence alone.

Keywords: Causal Discovery, Causal Inference, Directed Acyclic Graphs, Identifiability, Distributional Identifiability, Exponential Family, Ordinal Distribution, Structure Learning, Parametric Causal Models, Linear Models

## 1 Introduction

Causal reasoning plays a major role across modern scientific inquiry, in fields as varied as wireless networks (Thomas et al., 2024), oncology (Xue et al., 2019), molecular biology (Triantafillou et al., 2017), financial fraud detection (Ren, 2025), and digital advertising (Hill et al., 2015). Common to all of them is the need to uncover the causal mechanisms driving a stochastic system rather than its correlations. A commonly used structure for modeling causal relationships among variables is the directed acyclic graph (DAG), where nodes represent variables, and directed edges encode the flow of causal influence (Pearl, 2009). The central task of causal discovery is recovering the graph from data. Graph knowledge enables the design of interventions and the separation of cause from correlation, both of which most applications of causal inference rely on.

Current literature divides causal DAG recovery into two broad classes, interventional and observational. Interventional causal discovery refines the set of graphs compatible with the data, thereby improving identifiability (Hauser and B¨uhlmann, 2012). Recent works in interventional causal discovery include the optimized selection and design of soft interventions (Yang et al., 2018; Jaber et al., 2020; Peng et al., 2026; Peng and Mitra, 2025) and causal bandit settings (Lattimore et al., 2016; Varici et al., 2023; Peng et al., 2025). In many scientific settings, however, experimentation is expensive, unethical, or infeasible, and one must recover the graph from observational data alone. Existing works in observational causal discovery explore DAG recovery by constraint-based procedures testing conditional independence (Spirtes et al., 2001), by score-based procedures searching over graphs (Chickering, 2002), or by recent characterizations of the finite-sample and detectiontheoretic limits of recovery (Ghoshal and Honorio, 2017; Gao et al., 2022; Shaska and Mitra, 2025; Lungu et al., 2026; Mishra and Mitra, 2026). Underlying all such procedures is the more basic question of identifiability, namely, whether the observational distribution sufices to determine causal direction at all. Identifiability is the question we address herein for causal graphs with observational data and nodes that admit mixed-ordinal and oneparameter exponential-family distributions.

A key challenge in identifiability theory is the dificulty in identifying a causal DAG beyond its Markov equivalence class (MEC) from observational data alone without additional structural assumptions (Spirtes et al., 2001). The existing literature addresses the challenge using structural equation modeling (SEM), thereby obtaining identifiability across various continuous and discrete families, as reviewed in Section 1.2. Across those results, the causal mechanism often takes one of two restrictive forms: either the child node’s value is a deterministic function of its parents and exogenous noise, or the parents act on the child only through a structural intermediate, rather than directly shaping the parameters of the child’s conditional distribution.

Real observational datasets rarely fit assumptions of either kind. The present work is motivated by epidemiological studies routinely combining ordinal, count, bounded, and continuous variables in a single record. For instance, a study of COVID-19 case dynamics in Germany (Steiger et al., 2021) models a daily count of new cases against ordinal intervention indicators, binary policy variables, continuous weather measurements, bounded mobility percentages, and unbounded socio-demographic shares. The authors had to center and scale continuous predictors and decorrelate collinear mobility variables via principal component analysis, precisely because no single SEM family natively accommodates the heterogeneity. Similar works on global mammalian virus spillover (Johnson et al., 2015, 2020) likewise model a count outcome against a mix of ordinal, binary, and continuous predictors. Forcing such datasets into a continuous additive-noise SEM requires latent thresholding, link transformations, and truncations, all of which risk mis-specifications that can invalidate both the identifiability guarantees and the downstream causal estimates.

A complementary line of work on parametric causal models (Bodik and Chavez-Demoulin, 2025, 2026) lets the parent values directly shape the parameters of the child’s conditional distribution. The support, dispersion, and tail behavior of the child are then inherited from a chosen parametric family rather than from an exogenous noise term, allowing count, bounded, and continuous variables to coexist in a single graph without ad hoc transformations. We shift the formulation to construct the Linear Parametric Model (LPM), in which the parameters of each child’s conditional distribution are functions of a weighted linear combination of the parent values, the weights being the model parameters. This allows for the conditional family to difer across nodes, with parents acting on each child only through the linear predictor defined by the weights. Additionally, the structure of LPMs also allows for the use of more computationally eficient back propagation-based algorithms, which remain out of scope of this paper.

We specialize our focus on identifiability in the mixed setting where an edge joins an ordinal node to a node whose conditional distribution belongs to a regular one-parameter exponential family. Existing parametric models that achieve within-MEC identifiability each assume a single distributional family for all nodes, leaving edges across families untreated. The conditionally parametric causal model (CPCM) framework of Bodik and Chavez-Demoulin (2025) treats the exponential family in generality, but requires both conditional directions to admit the exponential family form. We fix the ordinal nodes to follow the ordered-logit model of McCullagh (1980), and remark on extensions beyond it. This proportional-odds cumulative link model has served as the standard model for ordinal responses across the applied literature (Lall et al., 2002; Williams, 2006; DeSantis et al., 2014; French and Shotwell, 2022), and does not admit a one-parameter exponential family form, as Proposition 1 below establishes, so the CPCM identifiability conditions do not carry over to an edge carrying an ordinal endpoint. The identifiability of edges between ordinal and exponential-family nodes in mixed-distribution DAGs is therefore an open problem that we address in this work. We enumerate the contributions of this work as follows:

1. We prove any edge between an ordinal node and a regular one-parameter exponential family node to be distributionally identifiable at every parameter value of the model, requiring only that the ordinal node has at least three categories and that the exponential-family node has at least three points of support, with no restriction on the suficient statistic.

2. We also prove converses, showing that when the exponential-family node has two points of support, we can construct an explicit family of forward and reverse parameters that induce identical joint distributions for any number of ordinal categories. Similarly, when the ordinal node has two categories, we establish a dichotomy that the orientation is identifiable if and only if the exponential-family suficient statistic is non-afine on its support.

3. We establish a multivariate extension orienting every such edge in a general d-node DAG at every parameter value, under the Markov property, causal suficiency, and the absence of selection. We strengthen this result to the identifiability of the entire orientation of a given skeleton, and prove a converse showing that both requirements remain necessary for an edge whose endpoints carry no neighbors other than each other.

4. We empirically validate the framework on the canonical 3-node MEC, demonstrating a decay of the orientation error rate with increasing sample size, and identifiability within the MEC.

## 1.1 Notation

A causal DAG is a pair $\mathcal { G } = ( \nu , \mathcal { E } )$ on d nodes, where $\mathcal { V } = \{ 1 , \ldots , d \}$ is the vertex set and ${ \mathcal { E } } = \{ ( i , j ) \mid i  j$ edge exists} is the edge set, carrying a deterministic weighted adjacency matrix $\mathbf { W } _ { \mathcal { G } } \in \mathbb { R } ^ { d \times d }$ given by

$$
[ \mathbf { W } _ { \mathcal { G } } ] _ { i , j } = \left\{ \begin{array} { l l } { 0 , } & { ( i , j ) \notin \mathcal { E } , } \\ { w _ { i , j } \neq 0 , } & { ( i , j ) \in \mathcal { E } . } \end{array} \right.\tag{1}
$$

We assume $\mathcal { G }$ is acyclic. The parent set of a node $j$ is $\mathcal { P } \left( j \right)$ defined as

$$
{ \mathcal { P } } \left( j \right) \triangleq \left\{ i \in \mathcal { V } : \left( i , j \right) \in \mathcal { E } \right\} ,\tag{2}
$$

and each node i carries a scalar random variable $X _ { i }$ , with $\mathbf { x } ~ = ~ ( X _ { 1 } , \ldots , X _ { d } )$ denoting the vector of all $d$ variables and $\mathbf { X } \in \mathbb { R } ^ { n \times d }$ is the data matrix collecting n independent realizations of $\mathbf { x } .$ With a slight abuse of notation, we write $\mathbf { x } _ { \mathcal { P } ( i ) }$ for the parent variables and for their observed values interchangeably. When more than one graph is in play, we write $\mathcal { P } _ { \mathcal { G } } \left( j \right)$ for the parent set of $j$ in $\mathcal { G } .$

## 1.2 Related Works

The existing literature has extensively considered additive noise-based SEMs for proving identifiability in causal $\mathrm { D A G s } ,$ restricting the causal mechanism to $X _ { i } = f _ { i } \left( \mathbf { x } _ { \mathcal { P ( i ) } } \right) + \varepsilon _ { i }$ for a deterministic function $f _ { i } ,$ with the noise variables $\varepsilon _ { i }$ jointly independent. For continuous variables, Peters and B¨uhlmann (2014) established identifiability for linear Gaussian models with equal error variance, and Shimizu et al. (2006) established the same for linear non-Gaussian models, with both cases considering the linear model $X _ { i } = \beta _ { i } ^ { \mathsf { T } } \mathbf { x } _ { \mathcal { P } ( i ) } + \varepsilon _ { i }$ with the noise $\varepsilon _ { i }$ being Gaussian and non-Gaussian, respectively. Hoyer et al. (2008) considered non-linear additive noise models, while Zhang and Hyv¨arinen (2009) introduced the postnonlinear model $X _ { i } = g _ { 1 } \left( g _ { 2 } \left( \mathbf { x } _ { \mathcal { P } \left( i \right) } \right) + \varepsilon _ { i } \right)$ with an invertible link function $g _ { 1 }$ . In the discrete setting, similar results hold for additive noise models (Peters et al., 2011) and integer modular additive noise models (Suzuki et al., 2014). At the same time, Cai et al. (2018) exploited a structural asymmetry instead, passing the cause through a low-cardinality deterministic intermediate before it generates the efect. In each case, identifiability was established by fixing the functional form through which the noise enters, and by drawing every node of the graph from a single distributional family.

Subsequent studies also allowed the parents to influence the dispersion of the child as well as its mean, giving the location-scale or heteroscedastic noise models $X _ { i } = f _ { i } \left( \mathbf { x } _ { \mathcal { P } \left( i \right) } \right) +$ $g _ { i } \left( \mathbf { x } _ { \mathcal { P ( i ) } } \right) \varepsilon _ { i }$ for a positive scale function $g _ { i }$ . Immer et al. (2023) established that the causa direction in such models is identifiable outside a set of pathological cases, with related literature considering binned approximations (Xu et al., 2022), patient-specific root-cause estimation (Strobl and Lasko, 2023), autoregressive-flow parametrizations (Khemakhem et al., 2021), and skewed variants (Klippert and Marx, 2026).

A third line of work dropped the noise term entirely and lets the parents parametrize the child’s conditional distribution within a specified family, deriving identifiability from that family’s structural form. Park and Raskutti (2015) treated Poisson DAGs, in which $X _ { i } \mid \mathbf { x } _ { \mathcal { P } ( i ) } \sim$ Poisson $\left( \theta _ { i } \left( \mathbf { x } _ { \mathcal { P } ( i ) } \right) \right)$ , and Park and Park (2019) considered the broader generalized hypergeometric family, in which the mean and variance stand in a polynomial relationship, while Ni and Mallick (2022) analyzed the identifiability in ordinal DAGs. Rajendran et al. (2021) tied exponential-family conditionals to greedy score-based recovery through a Bregman-information characterization, and Wang et al. (2024) employed nodewise generalized linear models under a formulation admitting latent confounders. Each such treatment, however, fixed a single family for the whole graph, leaving an edge between nodes of diferent families out of reach.

Closer to the present setting is a body of work that targeted the mixed continuous– discrete edge directly. Wenjuan et al. (2018) proposed a locally consistent information criterion over a mixed factor space, and Marx and Vreeken (2018) oriented pairs of arbitrary and possibly mixed type by minimum description length. Huang et al. (2018) constructed kernel-based score functions admitting mixed data within greedy equivalence search, while Zeng et al. (2022) gave suficient conditions for a linear mixed model combining LiNGAM with a logistic-type conditional. More recently, Yao et al. (2025) extended additive noise models to mixed types through a general function class, and Maeda et al. (2025) oriented a continuous–discrete pair by testing monotonicity of the conditional density ratio. An older tradition treated such variables through the conditional Gaussian distributions of Lauritzen and Wermuth (1989), with structure learning studied by Andrews et al. (2019). When the discrete variable was the cause, such treatments took the conditionals across its levels either to form a location-shift family or to be independently parametrized, and drew identifiability from non-Gaussianity, additive noise, description length, or a genericity argument placing coincidences in a set of measure zero.

Nearest in formulation is a line of work that replaced the deterministic parent-to-child link with a node-specific conditional distribution whose parameters are themselves functions of the parent values, so that $X _ { i } ~ \mid ~ \mathbf { x } _ { \mathcal { P } ( i ) } \sim ~ F \left( \theta _ { i } \left( \mathbf { x } _ { \mathcal { P } ( i ) } \right) \right)$ for a known parametric family F and a parameter map $\theta _ { i }$ . Janzing et al. (2009) fixed a second-order exponential maximum-entropy form and observed that variables supported on proper subsets of R can induce distinct joint laws even for Markov-equivalent graphs, illustrating the point for a binary–continuous pair. The aforementioned CPCM model of Bodik and Chavez-Demoulin (2025) developed the idea across the exponential family, where the density takes the form $H \left( x \right) \exp \left( \theta ^ { \mathsf { T } } T \left( x \right) - a \left( \theta \right) \right)$ with suficient statistic T, base measure H, log-partition function a, and natural parameter $\boldsymbol { \theta } = \theta _ { i } \left( \mathbf { x } _ { \mathcal { P } ( i ) } \right)$ , reaching multi-parameter cases such as the two-parameter Gaussian, Gamma, and Beta as well as non-linear parameter maps, with identifiability characterized by conditions relating that map to the suficient statistics. The characterization, however, required the exponential family representation to hold in both causal directions.

Our own earlier treatment of mixed Ordinal–Poisson DAGs (Shaska et al., 2025), which motivates this work, showed such an edge between an ordinal and a Poisson distributed node to be orientable outside a Lebesgue-null set of parameters by an analyticity argument. The treatment admits any analytic cumulative link with a nowhere-vanishing derivative and thus covers the probit, but remains confined to the Poisson conditional and the bivariate case, leaves the multivariate extension as a simulation-supported conjecture, and does not address where identifiability fails. Our most recent work on the treatment of Ordinal– Exponential family DAGs (Mishra et al., 2026) explores this further by considering a subset of the regular one-parameter exponential family of distributions, restricted by functional assumptions on the suficient statistics of the exponential family. The work proves distribu tional identifiability for continuous exponential family distributions, subject to the specified assumptions on the suficient statistics and to identifiability outside a Lebesgue-null set of parameters, for their discrete counterparts. However, it omits the necessary conditions, the analysis of non-identifiability, the treatment of the unrestricted suficient-statistic case, the distributional identifiability of discrete cases, and an extensive analysis of the multi-node DAG case, all of which this work addresses in depth.

To the best of our knowledge, no existing treatment addresses the case where an edge has endpoints in diferent families, one of which admits no exponential-family representation. The mixed continuous–discrete results do not apply since an exponential-family conditional is, in general, neither a location-shift family across the levels of the cause nor a collection of independently parametrized laws. The CPCM characterization fails for a diferent reason: it is stated in terms of a representation that an ordinal endpoint does not admit. Where such an edge has been treated at all, the conclusion has held outside a null set of parameters rather than at every point of the parameter space, and where the conclusion has been shown to hold for every point in the parameter set, it has been restricted by strong assumptions on the exponential family, and nothing has been established about where the conclusion fails.

The remainder of the paper is organized as follows. Section 2 introduces the LPM framework and specializes the framework to the mixed ordinal–exponential family setting. Section 3 establishes the bivariate and multivariate identifiability results together with their converses. Section 4 presents the numerical experiments, and Section 5 concludes the paper.

## 2 Preliminaries

In this section, we formalize the modeling framework used throughout the paper. We begin with the standard assumptions for causal DAGs, then introduce the LPM framework as a specialization of the CPCM framework, and finally specialize the LPM to the ordinalexponential setting underlying our identifiability results.

## 2.1 Causal DAGs

We assume throughout that the joint distribution of x is Markov with respect to ${ \mathcal { G } } ,$ factorizing as

$$
p \left( \mathbf { x } \right) = \prod _ { i = 1 } ^ { d } p \big ( X _ { i } \mid \mathbf { x } _ { \mathcal { P } ( i ) } \big ) .\tag{3}
$$

Two DAGs are Markov equivalent when they entail the same conditional independence relations, equivalently when they share a skeleton and a set of v-structures. The collection of all such DAGs equivalent to a given DAG is its Markov equivalence class (MEC).

The classical SEM specifies each node as a deterministic function of its parents and exogenous noise,

$$
X _ { i } = f _ { i } \left( \mathbf { x } _ { \mathcal { P } ( i ) } , \varepsilon _ { i } \right) , \quad i \in \mathcal { V } ,\tag{4}
$$

where $f _ { i } : \mathbb { R } ^ { | \mathcal { P } ( i ) | + 1 } \ :  \ : \mathbb { R }$ is the deterministic function defining the SEM and the noise variables $\{ \varepsilon _ { i } \} _ { i = 1 } ^ { d }$ are mutually independent with pre-defined distributions. In such instanti-

ations, however, the stochasticity of the child enters only through the exogenous noise term $\varepsilon _ { i }$ whose family is fixed in advance, leaving tail and support inherited from $\varepsilon _ { i } .$ restricting the parents from acting on the full conditional distribution of the child.

## 2.2 Linear Parametric Models

The CPCM framework of Bodik and Chavez-Demoulin (2025) replaces the deterministic construction with a node-specific conditional distribution whose parameters are functions of the parent values,

$$
X _ { i } \sim p _ { i } \left( \cdot \mid \pmb { \theta } _ { i } \left( \mathbf { x } _ { \mathcal { P } ( i ) } \right) \right) , \quad i \in \mathcal { V } ,\tag{5}
$$

where $p _ { i }$ is the conditional distribution of $X _ { i }$ . This allows for the support, dispersion, and tail behavior to be inherited from the parametric family rather than an exogenous noise term. Instead of focusing on each individual parent value in $\mathbf { x } _ { \mathcal { P } ( i ) }$ directly as done in CPCM, we turn our focus on a weighted linear combination of the parent values $\mathbf { w } _ { i } ^ { \mathsf { T } } \mathbf { x } _ { \mathcal { P } ( i ) }$ , where the weights $\mathbf { w } _ { i }$ determine the influence of each parent value on the conditional of the child node. This formulation is motivated by artificial neural networks, where the output of a single neuron is estimated as a function of the weighted linear combination of its inputs. Such constructions also facilitate the development of eficient back-propagation-based algorithms currently studied in the literature, which are outside the scope of this paper. We refer to such models as Linear Parametric Models (LPM), formalized in Definition 1 below.

Definition 1 (Linear Parametric Model) A linear parametric model on a $D A G { \mathcal { G } } =$ $( \nu , \mathcal { E } )$ is a tuple

$$
\mathcal { S } = \left( \mathcal { G } , \{ p _ { i } \} _ { i \in \mathcal { V } } , \{ \pmb { \theta } _ { i } \} _ { i \in \mathcal { V } } , \mathbf { W } _ { \mathcal { G } } \right) ,\tag{6}
$$

satisfying the following:

1. For each $i \in \mathcal { V } , p _ { i } \left( \cdot \mid \pmb { \theta } _ { i } \right)$ is a regular parametric family with real parameters $\theta _ { i }$ and real support $\mathcal { X } _ { i } \subseteq \mathbb { R }$

2. For each $i \in \mathcal V$ , the parameters are determined by the parent values through a known parameter link function given as,

$$
\begin{array} { r } { \pmb { \theta } _ { i } \triangleq \pmb { \theta } _ { i } \left( \mathbf { w } _ { i } ^ { \mathsf { T } } \mathbf { x } _ { \mathcal { P } ( i ) } \right) , } \end{array}\tag{7}
$$

where $\mathbf { w } _ { i } \triangleq [ \mathbf { W } _ { \mathcal { G } } ] _ { \mathcal { P } ( i ) , i }$ is the weight vector corresponding to node $i .$

3. The conditional distributions $\{ p _ { i } \left( X _ { i } \mid \mathbf { x } _ { \mathcal { P } ( i ) } \right) \} _ { i \in \mathcal { V } }$ are compatible with the Markov factorization in (3).

It is interesting to note that Definition 1 recovers the canonical linear Gaussian SEM considered by Peters and B¨uhlmann (2014) with additive noise $\varepsilon _ { i } \sim \mathcal { N } \left( 0 , \sigma _ { i } ^ { 2 } \right)$ on setting $p _ { i }$ Gaussian with mean $\mathbf { w } _ { i } ^ { \mathsf { T } } \mathbf { x } _ { \mathcal { P } ( i ) }$ and fixed variance $\sigma _ { i } ^ { 2 }$ . The conditional family $p _ { i }$ may also difer across nodes, allowing a single LPM to combine ordinal, count, bounded, and continuous variables within one graph without ad-hoc transformations. Lastly, note that the conditional distribution $p _ { i }$ rather than an exogenous noise term determines the support of $X _ { i } ,$ respecting discrete and bounded variables by construction. We summarize the contrast between the three constructions — SEM, CPCM, and LPM — at a single node $X _ { i }$ in Figure 1, marking the quantities known in advance and those estimated from data.

![](images/aecc5d3e517bc7739841216a53db71ca0d05a27b7e622f4d66f39633a590a4bd.jpg)  
Figure 1: Generative view of the SEM, CPCM, and LPM frameworks at a single node i with parent set $\mathcal { P } \left( i \right)$

## 2.3 Ordinal-Exponential LPM

We now focus on the ordinal and regular one-parameter exponential family mixed distribution LPM studied in the remainder of this paper. The node set is partitioned as $\mathcal { V } = \mathcal { V } _ { \mathrm { o r d } } \cup \mathcal { V } _ { \mathrm { e x p } }$ , where $\mathcal { V } _ { \mathrm { o r d } }$ represents the set of the ordinal nodes and $\nu _ { \mathrm { e x p } }$ represents that of the exponential-family nodes. We describe both node families below.

## 2.3.1 Ordinal Nodes

Let $i \in \mathcal { V } _ { \mathrm { o r d } }$ correspond to an ordinal random variable $X _ { i }$ with finite ordered support $\mathcal { X } _ { i } = \{ 1 , \ldots , s \}$ for $s \geq 2$ . If the node is a root node, i.e. $\mathcal { P } \left( i \right) = \emptyset$ , the distribution of $X _ { i }$ reduces to a categorical distribution specified by category probabilities $\pi _ { i , x }$ as

$$
p _ { i } \left( X _ { i } = x \right) = \pi _ { i , x } , \quad \pi _ { i , x } > 0 , \quad \forall x \in \mathcal { X } _ { i } , \quad \mathrm { a n d } \quad \sum _ { x \in \mathcal { X } _ { i } } \pi _ { i , x } = 1 .\tag{8}
$$

For the non-root node case, we adopt the ordered logit cumulative link parametrization of McCullagh (1980) given by

$$
p \left( X _ { i } = x \mid \mathbf { x } _ { \mathcal { P } ( i ) } \right) = \sigma { \Big ( } \gamma _ { i , x } - \mathbf { w } _ { i } ^ { \mathsf { T } } \mathbf { x } _ { \mathcal { P } ( i ) } { \Big ) } - \sigma { \Big ( } \gamma _ { i , x - 1 } - \mathbf { w } _ { i } ^ { \mathsf { T } } \mathbf { x } _ { \mathcal { P } ( i ) } { \Big ) } , \quad \forall x \in \pmb { \chi } _ { i }\tag{9}
$$

where $\sigma \left( z \right) = \left( 1 + e ^ { - z } \right) ^ { - 1 }$ is the sigmoid function, and $\gamma _ { i , j }$ are strictly ordered cutpoints such that $- \infty = \gamma _ { i , 0 } < \gamma _ { i , 1 } < \cdot \cdot \cdot < \gamma _ { i , s - 1 } < \gamma _ { i , s } = + \infty$ . The cutpoints are further assumed to be independent of the parent variables $\mathbf { x } _ { \mathcal { P } ( i ) }$ and the weights $\mathbf { w } _ { i }$

## 2.3.2 Exponential-Family Nodes

Let $i \in \mathcal { V } _ { \mathrm { e x p } }$ correspond to a random variable $X _ { i }$ whose conditional distribution belongs to a regular one-parameter exponential family. The support $\mathcal { X } _ { i } \subseteq \mathbb { R }$ may be discrete or continuous. For the root node case with $\mathcal { P } \left( i \right) = \emptyset$ , the marginal distribution of $X _ { i }$ is fixed to be non-degenerate and positive on the support $\mathcal { X } _ { i }$ . Therefore, we have,

$$
p _ { X _ { i } } \left( x \right) > 0 , \quad \forall x \in \mathcal { X } _ { i } , \qquad \mathrm { a n d } \qquad \sum _ { x \in \mathcal { X } _ { i } } p _ { X _ { i } } \left( x \right) = 1\tag{10}
$$

If the node has parent nodes, i.e. $\mathcal { P } \left( i \right) \neq \emptyset$ , then the node follows a regular one-parameter exponential family distribution with known suficient statistic $T _ { i } : \mathcal { X } _ { i }  \mathbb { R }$ , taken to be minimal in the sense that $T _ { i }$ is not constant on $\mathcal { X } _ { i } .$ , so that distinct natural parameters index distinct distributions, and where the natural parameter is fixed to follow a known strictly monotone function of the linear predictor of the parents $\eta _ { i } \left( \mathbf { w } _ { i } ^ { \mathsf { T } } \mathbf { x } _ { \mathcal { P } ( i ) } \right)$ , and the conditional distribution takes the form

$$
p _ { X _ { i } | \mathbf { x } _ { \mathcal { P } ( i ) } } \left( x \mid \mathbf { x } _ { \mathcal { P } ( i ) } \right) = H _ { i } \left( x \right) \exp \left( \eta _ { i } \left( \mathbf { w } _ { i } ^ { \mathsf { T } } \mathbf { x } _ { \mathcal { P } ( i ) } \right) T _ { i } \left( x \right) - a _ { i } \left( \eta _ { i } \left( \mathbf { w } _ { i } ^ { \mathsf { T } } \mathbf { x } _ { \mathcal { P } ( i ) } \right) \right) \right) , \quad \forall x \in \mathcal { X } _ { i } ,\tag{11}
$$

where $H _ { i } \left( x \right) > 0$ is the base measure, and $a _ { i } \left( \cdot \right)$ is the log-partition function.

We now proceed to record a fundamental mismatch between the ordinal and the exponential family nodes described in this section.

Proposition 1 Let X be ordinal with support $\mathcal { X } = \{ 1 , . . . , s \} f o r s \ge 3$ , let Y be supported on a set Y containing at least three points, and let

$$
p \left( X = x \mid Y = y \right) = \sigma ( \gamma _ { x } - w y ) - \sigma ( \gamma _ { x - 1 } - w y ) , \quad \forall x \in \mathcal { X } , y \in \mathcal { Y } ,\tag{12}
$$

for some $w \ne 0$ and strictly ordered cutpoints − $\begin{array} { r } { - \infty = \gamma _ { 0 } < \gamma _ { 1 } < \cdots < \gamma _ { s - 1 } < \gamma _ { s } = + \infty } \end{array}$ Then, regarded as a family of distributions on X indexed by y, this conditional is not a one-parameter exponential family in X.

## Proof See Appendix A.1.

Proposition 1 shows that on the ordinal support, the ordered logit generates a family that does not admit a one-parameter exponential family representation. At the same time, this makes the edge orientable within the framework developed here, as we show in the following section.

## 3 Identifiability

This section provides the theoretical analysis of the population-level identifiability of mixed ordinal-exponential Linear Parametric Models (LPMs) as discussed in Section 2. We begin by analyzing a simple bivariate Directed Acyclic Graph (DAG) setting and then extend our findings to multivariate scenarios.

## 3.1 Identifiability in Bivariate Case

Consider a two-node causal DAG over random variables X and Y, whose single edge admits the two competing orientations $\mathcal { M } _ { X  Y }$ and $\mathcal { M } _ { Y  X }$ , referred to respectively as the forward and the reverse model throughout the paper. To establish identifiability, we need to show that the two model classes yield distinct joint distributions, which we formalize as follows.

Definition 2 (Distributional Identifiability of Edge Direction) The edge direction, in a directed bivariate causal model over random variables X and Y , is distributionally identifiable if the two orientations induce disjoint sets of joint distributions, i.e. $P _ { X , Y } \neq$ $Q _ { X , Y }$ for all $P _ { X , Y } \in \mathcal { M } _ { X  Y }$ and $Q _ { X , Y } \in \mathcal { M } _ { Y  X }$

We start by fixing the conditional distribution of X to follow the ordered-logit ordinal model with support $\mathcal { X } = \{ 1 , \ldots , s \}$ , and the conditional distribution of Y to belong to a regular one-parameter exponential family with known suficient statistic $T \left( y \right)$ and support Y. With a slight abuse of notation, we write $\mathcal { M } _ { X  Y }$ and $\mathcal { M } _ { Y  X }$ also for the sets of joint laws the two models respectively induce on $\mathcal { X } \times \mathcal { V }$ . In the forward model, the edge weight is $w \neq 0$ , and the node corresponding to $X$ is the root node with a categorical marginal:

$$
p _ { X } ( x ; { \mathcal { M } } _ { X  Y } ) = \pi _ { x } , \quad { \mathrm { w i t h ~ } } \pi _ { x } > 0 ~ { \forall x \in { \mathcal { X } } } ~ { \mathrm { a n d } } ~ \sum _ { x \in { \mathcal { X } } } \pi _ { x } = 1 ,\tag{13}
$$

and $Y \mid X$ follows a regular one-parameter exponential family distribution, with its conditional distribution defined as

$$
p _ { Y | X } ( y \mid x ; \mathcal { M } _ { X  Y } ) = H ( y ) \exp ( \eta ( w x ) T ( y ) - a ( \eta ( w x ) ) ) , \quad \forall y \in \mathcal { V } , \ x \in \mathcal { X } ,\tag{14}
$$

where $\eta \left( \cdot \right)$ is a known monotone injective link, $H \left( \cdot \right)$ the base measure, and $a \left( \cdot \right)$ the logpartition function, so the forward class is

$$
p _ { X , Y } ( x , y ; \mathcal { M } _ { X  Y } ) = \pi _ { x } H ( y ) \exp ( \eta ( w x ) T ( y ) - a ( \eta ( w x ) ) ) , \quad \forall y \in \mathcal { Y } , \ x \in \mathcal { X } .\tag{15}
$$

In the reverse model with edge weight $v \neq 0$ , the node corresponding to Y is the root node with its marginal fixed to be non-degenerate and positive on Y,

$$
p _ { Y } ( y ; \mathcal { M } _ { Y  X } ) > 0 , \quad \forall y \in \mathcal { Y } , \qquad \mathrm { a n d } \qquad \sum _ { y \in \mathcal { Y } } p _ { Y } ( y ) = 1 ,\tag{16}
$$

and $X \mid Y$ follows the ordered-logit ordinal model with cutpoints $\gamma _ { 1 } < \dots < \gamma _ { s - 1 }$

$$
p _ { X | Y } ( x \mid y ; \mathcal { M } _ { Y  X } ) = \sigma ( \gamma _ { x } - v y ) - \sigma ( \gamma _ { x - 1 } - v y ) , \quad \forall x \in \mathcal { X } , \ y \in \mathcal { Y } ,\tag{17}
$$

where $\sigma \left( z \right) = \left( 1 + e ^ { - z } \right) ^ { - 1 }$ is the sigmoid function. We adopt the boundary conventions $\gamma _ { 0 } = - \infty$ and $\gamma _ { s } = + \infty , \mathrm { g i v i n g } \sigma ( \gamma _ { 0 } - v y ) = 0$ and $\sigma ( \gamma _ { s } - v y ) = 1$ . The reverse class is thus

$$
p _ { X , Y } ( x , y ; \mathcal { M } _ { Y  X } ) = p _ { Y } ( y ; \mathcal { M } _ { Y  X } ) ( \sigma ( \gamma _ { x } - v y ) - \sigma ( \gamma _ { x - 1 } - v y ) ) , \quad \forall x \in \mathcal { X } , \ y \in \mathcal { Y } .\tag{18}
$$

To compare the two classes, we focus on the conditional distribution of $X \mid Y$ . In the reverse model (17) supplies the conditional directly, while in the forward model, Bayes theorem obtains this conditional from the marginal (13) and the $Y \ | \ X$ conditional (14). Since a categorical law on X is determined uniquely by its log-odds relative to a fixed reference category, comparing conditional log-ratios sufices. For any $u , x \in { \mathcal { X } }$ and any $y \in \mathcal { V }$ , define the conditional log-odds-ratio

$$
R \left( y , u , x ; \mathcal { M } \right) \triangleq \log \left( \frac { p _ { X | Y } \left( u \mid y ; \mathcal { M } \right) } { p _ { X | Y } \left( x \mid y ; \mathcal { M } \right) } \right) ,\tag{19}
$$

where $\mathcal { M } \in \{ \mathcal { M } _ { X  Y } , \mathcal { M } _ { Y  X } \}$ . The following lemma establishes an important restriction of the forward model.

Lemma 1 In the forward model $\mathcal { M } _ { X  Y }$ with edge weight w $\neq ~ 0$ , where the marginal distribution of X follows a categorical distribution on support $\mathcal { X } = \{ 1 , . . . , s \}$ given by (13), and $Y \ | \ X$ follows a regular one-parameter exponential family distribution given by (14), the log-odds-ratio (19) is afine in the suficient statistic $T \left( y \right)$ , i.e.,

$$
R ( y , u , x ; \mathcal { M } _ { X  Y } ) = \alpha T ( y ) + \beta , \quad \forall u , x \in \mathcal { X } , u \neq x \ a n d \ y \in \mathcal { V } ,\tag{20}
$$

where $\alpha \neq 0$ and $\beta$ are independent of y and are given by

$$
\alpha = \eta \left( w u \right) - \eta \left( w x \right) , \qquad \beta = a \left( \eta \left( w x \right) \right) - a \left( \eta \left( w u \right) \right) + \log \left( \frac { \pi _ { u } } { \pi _ { x } } \right) .\tag{21}
$$

Proof See Appendix A.2.

We now turn to the reverse model, whose conditional (17) is strictly positive on X for every $y \in \mathcal { V }$ by the strict ordering of the cutpoints and the strict monotonicity of the sigmoid. It is important to note that $T \left( y \right)$ carries no distinguished role in the reverse model since $Y$ is not constrained to an exponential family. Therefore, $T \left( y \right)$ enters only as a fixed function on $\mathcal { V }$ for the reverse model. However, if the two models were to induce a common joint law, then Lemma 1 would force every reverse log-odds-ratio to be afine in that same $T \left( y \right)$ , although nothing but the sigmoid geometry of the cutpoints governs those ratios. We analyze the reverse model by focusing on the two adjacent category pairs whose log-odds ratios admit closed forms with opposite curvature, an important observation that helps settle the afineness argument for the reverse model.

Lemma 2 For the reverse model $\mathcal { M } _ { Y  X }$ with edge weight $v \neq 0$ , where $X \mid Y$ follows the ordered-logit ordinal conditional given in (17) with cutpoints $- \infty = \gamma _ { 0 } < \gamma _ { 1 } < \cdot \cdot \cdot < \gamma _ { s - 1 } <$ $\gamma _ { s } = + \infty$ and support $\mathcal { X } = \{ 1 , \ldots , s \}$ where $s \geq 3$ , the extreme adjacent log-odds-ratios $B ( y ) \triangleq R ( y , 2 , 1 ; \mathcal { M } _ { Y  X } )$ and $U ( y ) \triangleq R ( y , s , s - 1 ; \mathcal { M } _ { Y  X } )$ admit the closed forms

$$
B \left( y \right) = c _ { B } - \log \left( 1 + e ^ { \gamma _ { 2 } - v y } \right) , \quad U \left( y \right) = c _ { U } + \log \left( 1 + e ^ { v y - \gamma _ { s - 2 } } \right) , \quad \forall y \in \mathcal { V } ,\tag{22}
$$

for finite constants $c _ { B }$ and $c _ { U }$ independent of y given by

$$
c _ { B } = \log \left( \frac { e ^ { \gamma _ { 2 } } - e ^ { \gamma _ { 1 } } } { e ^ { \gamma _ { 1 } } } \right) , \quad c _ { U } = \log \left( \frac { e ^ { \gamma _ { s - 2 } } } { e ^ { \gamma _ { s - 1 } } - e ^ { \gamma _ { s - 2 } } } \right) .\tag{23}
$$

Moreover, the right-hand sides of (22) are defined for every $y \in \mathbb { R }$ , thus, for the real line extensions $\bar { B } , \bar { U } : \mathbb { R }  \mathbb { R }$ such that $B = \left. \bar { B } \right| _ { y }$ and $U = { \bar { U } } | _ { y } ,$ B<sup>¯</sup> is strictly concave and $\bar { U }$ is strictly convex on $\mathbb { R }$

Proof See Appendix A.3.

The two lemmas are found to impose incompatible requirements on a single quantity: Lemma 1 confines every forward log-odds-ratio to a straight line in $T \left( y \right)$ , while Lemma 2 bends the two extreme reverse ones in opposite directions. Eliminating $T \left( y \right)$ between the two extremes, as defined in Lemma 2, leaves a function of $y$ whose shape the cutpoints and the reverse edge weight settle by themselves, and which the elimination forces to be constant on ${ \mathcal { V } } ,$ whereas strict curvature permits a constant value at no more than two points. This creates a contradiction, which leads us to the identifiability result formally stated in Theorem 1.

Theorem 1 Let $\mathcal { M } _ { X  Y }$ be the forward model with edge weight w $\neq 0$ , in which X follows the categorical marginal (13) on $\mathcal { X } = \{ 1 , \ldots , s \}$ with $s \geq 3$ , and $Y \mid X$ follows the regular one-parameter exponential family conditional (14) with suficient statistic $T \left( y \right)$ and support $\mathcal { V }$ . Let $\mathcal { M } _ { Y  X }$ be the reverse model with edge weight $v \neq 0$ , in which Y follows the positive marginal (16) on Y, and $X \ | \ Y$ follows the ordered-logit conditional (17) with cutpoints $- \infty = \gamma _ { 0 } < \gamma _ { 1 } < \dots < \gamma _ { s - 1 } < \gamma _ { s } = + \infty$ . If the support contains at least three points, $| \mathscr { y } | \geq 3$ , then $\mathcal { M } _ { X  Y }$ and $\mathcal { M } _ { Y  X }$ are distributionally identifiable.

Proof Two proofs are given in Appendix $\mathrm { A . 4 } .$ , the first proceeding from the opposing curvature of Lemma 2 and the second from Proposition 1. ■

Theorem 1 imposes no condition on the forward model suficient statistic $T \left( y \right)$ , beyond the minimality already required in Section 2, so a single result governs every regular oneparameter exponential family, over finite, countable, and continuous supports alike. The LPM setting calls for exactly such generality, since a graph whose exponential-family nodes carry distinct families admits no reduction to a canonical case, and a criterion stated in terms of T would have to be verified edge by edge. The guarantee also holds for every parameter value, rather than only outside an exceptional set, in contrast to Shaska et al. (2025). It is also important to note that guarantees based on generic identifiability, which prove identifiability outside an exceptional set, cannot be reliably used on real datasets whose parameters are unknown. Such issues are mitigated under distributional identifiability, which holds for every valid parameter in the parameter set.

Appendix A.4 gives two proofs for Theorem 1, and both of them are important for establishing two distinct ideas. The first proof provides direct intuition for the requirement of $| y | \geq 3 .$ , and more importantly, it also paves the way for a suficient condition on any cumulative link beyond the ordered-logit, as formalized in Proposition 2. The second proof relies directly on Proposition 1 and is the shorter one, exhibiting the asymmetry in its plainest form, at the cost of being tied to the logit through the algebra that establishes that proposition.

Proposition 2 Let the forward model and the remaining hypotheses of Theorem 1 be unchanged, and let the ordered-logit conditional (17) of the reverse model be replaced by

$$
p _ { X | Y } ( x \mid y ; \mathcal { M } _ { Y  X } ) = F ( \gamma _ { x } - v y ) - F ( \gamma _ { x - 1 } - v y ) , \quad \forall x \in \mathcal { X } , \ y \in \mathcal { V } ,\tag{24}
$$

for a continuous and strictly increasing cumulative link $F : \mathbb { R } \to ( 0 , 1 )$ , the boundary conventions $\gamma _ { 0 } = - \infty$ and $\gamma _ { s } = + \infty$ being retained, so that $F \left( \gamma _ { 0 } - v y \right) = 0$ and $F \left( \gamma _ { s } - v y \right) = 1$ Write

$$
\bar { B } \left( y \right) = \log \left( \frac { F \left( \gamma _ { 2 } - v y \right) - F \left( \gamma _ { 1 } - v y \right) } { F \left( \gamma _ { 1 } - v y \right) } \right) , \quad \bar { U } \left( y \right) = \log \left( \frac { 1 - F \left( \gamma _ { s - 1 } - v y \right) } { F \left( \gamma _ { s - 1 } - v y \right) - F \left( \gamma _ { s - 2 } - v y \right) } \right) ,\tag{25}
$$

for the two extreme adjacent log-odds-ratios of (24), regarded as functions on all of R. If, for every $v \ne 0$ and every strictly ordered set of cutpoints, B<sup>¯</sup> is strictly concave and U<sup>¯</sup> strictly convex on $\mathbb { R } ,$ then $\mathcal { M } _ { X  Y }$ and $\mathcal { M } _ { Y  X }$ are distributionally identifiable.

Proof See Appendix A.5.

We note that Proposition 2 provides only a suficient condition, and its necessity remains out of the scope of this paper. Proposition 2’s hypothesis isolates the single property of the logit that the first proof of Theorem 1 uses, and Lemma 2 shows the logit to possess that property. We investigate in the following subsection whether the thresholds $s \geq 3$ and $| \mathscr { y } | \geq 3$ are necessary, or merely suficient for the arguments given.

## 3.2 Analysis of Non-Identifiability

We start by analyzing the necessity of the $| \mathscr { y } | \geq 3$ condition. Intuitively, when the support of Y is only two points, the constraints linking the two models become too few to force a unique orientation. The following result establishes the necessity of the support condition on the exponential-family node for key cases.

Theorem 2 Let $\mathcal { M } _ { X  Y }$ be the forward model with edge weight w $\neq 0$ , in which X follows the categorical marginal (13) on $\mathcal { X } = \{ 1 , \ldots , s \}$ with $s \geq 3$ , and $Y \mid X$ follows the regular one-parameter exponential family conditional (14) with suficient statistic $T \left( y \right)$ and support Y. Let $\mathcal { M } _ { Y  X }$ be the reverse model with edge weight $v \neq 0$ , in which Y follows the positive marginal (16) on Y, and $X \ | \ Y$ follows the ordered-logit conditional (17) with cutpoints $- \infty = \gamma _ { 0 } < \gamma _ { 1 } < \dots < \gamma _ { s - 1 } < \gamma _ { s } = + \infty$ . If the support of Y contains exactly two points, $\mathcal { Y } = \{ y _ { 1 } , y _ { 2 } \}$ with $y _ { 1 } < y _ { 2 }$ , the forward model suficient statistic is afine in $y , \ i . e .$ $T \left( y \right) = c _ { 1 } y + c _ { 0 }$ with $c _ { 0 } , c _ { 1 } \in \mathbb { R }$ and $c _ { 1 } \neq 0$ , and the link is identity, i.e. $\eta \left( z \right) = z$ , then there exists a two-parameter family of forward parameters $( \pi , w )$ with $w \ne 0$ and reverse parameters $( \gamma , v )$ with $v \ne 0$ that induce identical joint distributions on $\mathcal { X } \times \mathcal { V }$ The direction of the edge is therefore not distributionally identifiable.

Proof See Appendix A.6

Theorem 2 shows the requirement $| \mathcal { V } | \ge 3$ cannot be dropped from Theorem 1. Its assumptions are weaker than they appear. Since two distinct points are always collinear, every non-constant suficient statistic is afine on a two-point support. Therefore, the minimality already assumed in Section 2 supplies the afineness, leaving the canonical link as the only genuine restriction added. An exponential-family node supported on two points is a binary variable, so the non-identifiable edges are those joining an ordinal node to a binary one, which is a common pairing in practical datasets, highlighting the importance of this result. We provide a construction for deriving parameters which produce identical joint distributions in either direction under the condition that $| \mathcal { V } | = 2$ in Remark 1. It is important to note that this construction may not be the only way to derive parameters that produce the non-identifiable models.

Remark 1 Given the support $\mathcal { V } = \{ y _ { 1 } , y _ { 2 } \}$ with $y _ { 1 } ~ < ~ y _ { 2 }$ , the afine suficient statistic $T \left( y \right) = c _ { 1 } y + c _ { 0 }$ with $c _ { 1 } \neq 0$ , the canonical link $\eta \left( z \right) = z$ with log-partition $a \left( \cdot \right)$ and base measure H (·), the number of categories $s \geq 3$ , and a forward model edge weight w $\neq 0$ with $w c _ { 1 } > 0$ (the case for $w c _ { 1 } < 0$ follows by reversing every inequality below).

We define the following terms for a reverse model with edge weight v (the value to be determined below),

$$
\Delta y = y _ { 2 } - y _ { 1 } , \qquad \Delta T = c _ { 1 } \Delta y , \qquad L = v \Delta y , \qquad t _ { i } = e ^ { v y _ { i } } , \ i \in \{ 1 , 2 \} ,\tag{26}
$$

and, for a sequence $L = D _ { 0 } > D _ { 1 } > \dots > D _ { s - 1 } > D _ { s } = 0$ , define the cutpoints

$$
\gamma _ { j } = \log \left( \frac { t _ { 2 } - t _ { 1 } e ^ { D _ { j } } } { e ^ { D _ { j } } - 1 } \right) , \qquad j = 1 , \ldots , s - 1 .\tag{27}
$$

Construction of the reverse parameters $( \gamma , v )$

1. If s is even, i. $e , \exists \ m \ > \ 0$ such that $s \ = \ 2 m { : }$ set $v \ = \ m c _ { 1 } w$ , choose any $D _ { 1 } \in$ $\textstyle \left( { \frac { m - 1 } { m } } L , L \right)$ , and set

$$
\begin{array} { r } { D _ { j } = L - \frac { j } { 2 m } L ( j \ e v e n ) , \qquad D _ { j } = D _ { 1 } - \frac { j - 1 } { 2 m } L ( j \ o d d ) , \qquad j = 0 , \ldots , s . } \end{array}\tag{28}
$$

2. If s is odd, i. $. e . \ \exists \ m > 0$ such that $s = 2 m + 1$ : choose any $v \in ( m w c _ { 1 } , ( m + 1 ) w c _ { 1 } )$ ， and set

$$
\begin{array} { r } { D _ { j } = L - \frac { j } { 2 } w \Delta T ( j \ e v e n ) , \qquad D _ { j } = \left( m - \frac { j - 1 } { 2 } \right) w \Delta T ( j \ o d d ) , \qquad j = 0 , \dots , s . } \end{array}\tag{29}
$$

3. In either case, obtain $\gamma _ { 1 } < \dots < \gamma _ { s - 1 }$ from (27).

Construction of the forward marginal π. With $( \gamma , v )$ in hand, set for $x \in \mathcal { X }$

$$
l _ { x } = \log \left( \frac { \sigma \left( \gamma _ { x } - v y _ { 1 } \right) - \sigma \left( \gamma _ { x - 1 } - v y _ { 1 } \right) } { \sigma \left( \gamma _ { 1 } - v y _ { 1 } \right) } \right) , \qquad \beta _ { x } = l _ { x } - w \left( x - 1 \right) T \left( y _ { 1 } \right) ,\tag{30}
$$

with $\gamma _ { 0 } = - \infty$ and $\gamma _ { s } = + \infty$ , and take

$$
\pi _ { x } = \frac { \exp \left( \beta _ { x } + a \left( w x \right) - a \left( w \right) \right) } { \sum _ { x ^ { \prime } \in \mathcal { X } } \exp \left( \beta _ { x ^ { \prime } } + a \left( w x ^ { \prime } \right) - a \left( w \right) \right) } , \qquad x \in \mathcal { X } .\tag{31}
$$

Reverse marginal of Y. Set

$$
p _ { Y } ( y _ { i } ; \mathcal { M } _ { Y  X } ) = \sum _ { x \in \mathcal { X } } \pi _ { x } H ( y _ { i } ) \exp ( w x T ( y _ { i } ) - a ( w x ) ) , \qquad i \in \{ 1 , 2 \} .\tag{32}
$$

The forward model $( \pi , w )$ and the reverse model $( \gamma , v )$ so obtained induce the same joint distribution on $\mathcal { X } \times \mathcal { V }$ The one free scalar $( D _ { 1 }$ when s is even, v when s is odd) ranges over a non-empty open interval, so the construction yields a continuum of coincident pairs. Conversely, a reverse model $( \gamma , v )$ admits a matching forward model if and only if $D _ { j } = \log \left( \left( e ^ { \gamma _ { j } } + t _ { 2 } \right) / \left( e ^ { \gamma _ { j } } + t _ { 1 } \right) \right)$ satisfies $D _ { j } - D _ { j + 2 } = D _ { 0 } - D _ { 2 }$ for all $j = 0 , \ldots , s - 2$ , in which case the forward weight is uniquely $w = \left( L - D _ { 2 } \right) / \Delta T$ and π follows as above.

The strict ordering of the cutpoints is equivalent to the strict decrease of the $D _ { j }$ from $D _ { 0 } = L$ to $D _ { s } = 0$ , and the constant $D _ { j } - D _ { j + 2 }$ for any $j \in \{ 0 , \ldots , s - 2 \}$ splits that sequence into two interleaved arithmetic chains, each pinned at an endpoint. One scalar survives the count and ranges over a non-empty open interval, so the construction returns a continuum of coincident pairs rather than a single one. Only two scalars are free in total, so the coincident forward models form a two-dimensional subset of the s-dimensional forward parameter space, which is Lebesgue-null whenever $s \geq 3$ . Theorem 2 therefore establishes that the two model classes fail to be disjoint at two support points, not that the orientation is unrecoverable at a typical parameter value.

We now move to analyze the necessity of the $s \geq 3$ condition on the ordinal nodes. It is interesting to observe that at $s = 2$ , the reverse model retains a single cutpoint $\gamma _ { 1 }$ , and the resulting two-category conditional is itself a one-parameter exponential family in X, with natural parameter $v y - \gamma _ { 1 }$ . Proposition 1 therefore no longer applies, neither form of the argument establishing Theorem 1 survives, and the outcome turns entirely on the suficient statistic. This intuitively leads to the non-identifiability claim, provided in the following theorem.

Theorem 3 Let $\mathcal { M } _ { X  Y }$ be the forward model with edge weight w $\neq 0$ , in which X follows the categorical marginal (13) on $\mathcal { X } = \{ 1 , 2 \}$ , and $Y \mid X$ follows the regular one-parameter exponential family conditional (14) with suficient statistic $T \left( y \right)$ and support Y with $| \mathscr { y } | \geq 3$ Let $\mathcal { M } _ { Y  X }$ be the reverse model with edge weight $v \ne 0$ , in which Y follows the positive marginal (16) on Y, and $X \mid Y$ follows the ordered-logit conditional (17) with the single cutpoint $\gamma _ { 1 }$ . Then $\mathcal { M } _ { X  Y }$ and $\mathcal { M } _ { Y  X }$ are distributionally identifiable if and only if the suficient statistic is $\_ n o n { - } a f f i n e$ on the support, that is, if and only if there exist $y _ { 1 } , y _ { 2 } , y _ { 3 } \in \mathcal { V }$ for which $\left( T \left( y _ { 2 } \right) - T \left( y _ { 1 } \right) \right) / \left( y _ { 2 } - y _ { 1 } \right) \neq \left( T \left( y _ { 3 } \right) - T \left( y _ { 1 } \right) \right) / \left( y _ { 3 } - y _ { 1 } \right)$ . In particular, if $T \left( y \right) = c _ { 1 } y + c _ { 0 }$ is afine, then for every forward model, there exist reverse parameters $( \gamma _ { 1 } , v )$ with $v \neq 0$ that induce the same joint distribution, and the models are not identifiable.

Theorem 3 shows that instead of a full non-identifiability result, we arrive at a dichotomy. The condition of $s \geq 3$ remains necessary when the suficient statistic is afine and dispensable otherwise, so the requirement of three categories belongs to the statistic rather than to the ordinal variable alone. A third category then supplies a second cutpoint, and with it both the curvature of Lemma 2 and the applicability of Proposition 1, which is why T is unconstrained once $s \geq 3$ . We provide a construction for deriving model parameters such that both the forward and reverse models induce the same joint distribution for afine $T ,$ as noted in Remark 2. We note that Theorem 2 leaves a Lebesgue-null set of coincident forward models, whereas at $s = 2$ with afine T every forward model admits a matching reverse one, so the failure is complete rather than exceptional. For instance, the Poisson statistic is afine, so an Ordinal–Poisson edge is not identifiable at two ordina categories, and a binary ordinal node is the one case in which the guarantees of Section 3.1 require checking the statistic first.

Remark 2 Given the support Y, the afine suficient statistic $T \left( y \right) = c _ { 1 } y + c _ { 0 }$ with $c _ { 1 } \neq 0$ the monotone injective link $\eta \left( \cdot \right)$ with log-partition a (·) and base measure $H \left( \cdot \right)$ , the number of categories $s = 2$ , and forward parameters $( \pi , w )$ with $\pi _ { 1 } , \pi _ { 2 } > 0 , \pi _ { 1 } + \pi _ { 2 } = 1$ , and $w \ne 0$ We define the slope and intercept of the forward log-odds-ratio,

$$
\alpha \triangleq \eta \left( 2 w \right) - \eta \left( w \right) \neq 0 , \qquad \beta \triangleq a \left( \eta \left( w \right) \right) - a \left( \eta \left( 2 w \right) \right) + \log \left( \frac { \pi _ { 2 } } { \pi _ { 1 } } \right) .\tag{33}
$$

Construction of the reverse parameters $( \gamma _ { 1 } , v )$

1. Set $v = \alpha c _ { 1 }$ , which is nonzero since $\alpha \neq 0$ and $c _ { 1 } \neq 0$

2. Set $\gamma _ { 1 } = - \left( \alpha c _ { 0 } + \beta \right)$

Reverse marginal of Y . Set

$$
p _ { Y } ( y ; \mathcal { M } _ { Y  X } ) = \sum _ { x \in \{ 1 , 2 \} } \pi _ { x } H ( y ) \exp ( \eta ( w x ) T ( y ) - a ( \eta ( w x ) ) ) , \qquad y \in \mathcal { Y } .\tag{34}
$$

The forward model $( \pi , w )$ and the reverse model $( \gamma _ { 1 } , v )$ so obtained induce the same joint distribution on $\mathcal { X } \times \mathcal { V }$ . Unlike Remark 1, no condition is imposed on the forward parameters, so every forward model admits a matching reverse model. Conversely, given reverse parameters $( \gamma _ { 1 } , v )$ with v $\neq 0$ and the reverse marginal of Y set as above, the forward parameters are recovered by solving $\eta \left( 2 w \right) - \eta \left( w \right) = v / c _ { 1 }$ for w, which gives $w = v / c _ { 1 }$ under the canonical link, and then setting

$$
\frac { \pi _ { 2 } } { \pi _ { 1 } } = \exp \left( - \gamma _ { 1 } - \alpha c _ { 0 } - a \left( \eta \left( w \right) \right) + a \left( \eta \left( 2 w \right) \right) \right) , \qquad \pi _ { 1 } + \pi _ { 2 } = 1 .\tag{35}
$$

Theorem 1 through Theorem 3 settle the bivariate case, and the classification they give is fixed by the model specification rather than by the parameter values. The number of categories, the size of the support, and the shape of the suficient statistic are all determined once the node families are chosen, so an edge can be placed before any data are seen.

## 3.3 Multivariate Identifiability

The results established so far concern a single edge in isolation, forming the bivariate DAG. We now analyze multivariate mixed ordinal–exponential-family DAGs in which each node follows the LPM model. We begin by stating what identifiability of an edge direction means in the multivariate setting.

Definition 3 (Edge Direction Identifiability in Multivariate DAGs) Let $\mathcal { G } = ( \nu , \mathcal { E } )$ be a causal DAG with undirected skeleton $\mathcal { G } _ { u } = ( \nu , \mathcal { E } _ { u } )$ , and let $( i , j ) \in \mathcal { E } _ { u }$ . Denote by $\mathcal { G } ^ { i  j }$ and $\mathcal { G } ^ { j  i }$ the two directed graphs that agree with $\mathcal { G }$ on every edge of $\mathcal { E } _ { u } \setminus \{ ( i , j ) \}$ and orient the edge $( i , j )$ as $i  j$ and as $j  i$ respectively, both assumed acyclic, and let M $( \mathcal { G } ^ { i  j } )$ and $\mathcal { M } ( \mathcal { G } ^ { j  i } )$ denote the sets of joint distributions over V induced $b y$ them. The direction of the edge (i, j) is distributionally identifiable if

$$
p _ { \mathcal { V } } \neq q _ { \mathcal { V } } , \quad \forall p _ { \mathcal { V } } \in \mathcal { M } ( \mathcal { G } ^ { i  j } ) , q _ { \mathcal { V } } \in \mathcal { M } ( \mathcal { G } ^ { j  i } ) .\tag{36}
$$

We observe that Definition 3 reduces to Definition 2 when $\mathcal { V } = \{ i , j \}$ , with $\mathcal { M } ( \mathcal { G } ^ { i  j } )$ and $\mathcal { M } ( \mathcal { G } ^ { j  i } )$ becoming $\mathcal { M } _ { X  Y }$ and $\mathcal { M } _ { Y  X }$ , so the bivariate results of the preceding subsections are the two-node case of the theory below. It is intuitive to see that within a causal DAG, an ordinal–exponential-family edge is embedded among further nodes, and both endpoints may carry additional parents whose contributions enter the conditional laws. Those contributions shift the cutpoints of the ordinal endpoint and the natural parameter of the other, but leave both conditional forms intact, so the pair is again an instance of the bivariate setting. The reverse class available within a graph is also contained in the bivariate reverse class, since the graph constrains a reverse-oriented exponential-family endpoint that the bivariate model leaves free. We formalize this intuition in the next theorem, which establishes that the conditions that secure identifiability in the bivariate setting also sufice for an arbitrary graph.

Theorem 4 Let $\mathcal { G } = ( \nu , \mathcal { E } )$ be a causal DAG on d nodes with undirected skeleton $\mathcal { G } _ { u } =$ $( \nu , \mathcal { E } _ { u } )$ , in which each node is designated either ordinal or exponential family and every edge weight is nonzero. Let each non-root ordinal node k with $s _ { k }$ categories follow the ordered-logit conditional (17) given its parents with cutpoints $- \infty = \gamma _ { 0 } < \gamma _ { 1 } < \cdot \cdot \cdot < \gamma _ { s _ { k } - 1 } < \gamma _ { s _ { k } } = + \infty$ let each non-root exponential family node k follow the regular one-parameter exponential family conditional (14) given its parents with suficient statistic $T _ { k } ,$ monotone injective link $\eta ,$ and support $\mathcal { X } _ { k }$ , and let each root node k follow an arbitrary strictly positive marginal on its support, categorical on $\{ 1 , \ldots , s _ { k } \}$ when k is ordinal. Let the joint distribution p<sub>V</sub> be Markov with respect to ${ \mathcal { G } } ,$ , factorizing as $\begin{array} { r } { p _ { \mathcal { V } } = \prod _ { k \in \mathcal { V } } p _ { X _ { k } | \mathbf { x } _ { \mathcal { P } ( k ) } } } \end{array}$ , with G causally suficient, containing no latent confounders and no selection variables. Let $( i , j ) \in \mathcal { E } _ { u }$ join an ordinal node i with s categories to an exponential family node j with suficient statistic T and support $\chi _ { j }$ . If the ordinal node has at least three categories, $s \geq 3$ , and the exponential family node has at least three points of support, $| \mathcal { X } _ { j } | \ge 3$ , then the two orientations of the edge $\{ i , j \}$ induce disjoint sets of joint distributions on $\nu _ { \cdot }$ , and the direction of the edge is distributionally identifiable.

Proof See Appendix A.8.

Theorem 4 implicitly assumes the Markov property, causal suficiency, and the absence of selection, but not faithfulness, and the relaxation is substantive. Constraint-based procedures read orientation of conditional independencies and so require the graph to entail every independence the distribution exhibits; the argument here compares joint distributions directly and tests no independence, remaining untouched at parameter values where path cancellation produces an independence $\mathcal { G }$ does not imply. The exemption covers orientation alone, since recovering the skeleton by conditional independence testing still requires adjacency faithfulness (Spirtes et al., 2001; Ramsey et al., 2006) or a similar condition, and the skeleton is taken as given throughout.

The orientation is also local, resting on the conditional law of the two endpoints given their parents, so neither endpoint needs to be a source node, no topological ordering is required, and nothing is propagated from one edge to another. Locality distinguishes the guarantee from orientation rules that follow a v-structure and depend on what the rest of the skeleton contains. It is important to note that locality is achieved precisely because the root marginals of both the ordinal and exponential-family nodes are maximally free as described in (8) and (10), allowing subsequent constraints to produce subsets that inherit the properties derived for the original root node cases. Definition 3, however, compares two graphs agreeing everywhere but on a single edge, so Theorem 4 rules out one reversal at a time, and the following result extends the conclusion to the whole orientation space.

Theorem 5 Let $\mathcal { G } _ { u } = ( \nu , \mathcal { E } _ { u } )$ be an undirected skeleton in which every edge joins an ordinal node to an exponential family node, and let $\mathcal { O } \left( \mathcal { G } _ { u } \right)$ denote its acyclic orientations. Let $\mathcal { G } \in \mathcal { O } \left( \mathcal { G } _ { u } \right)$ , and let $p _ { \mathcal { V } }$ be generated by an ordinal-exponential LPM on $\mathcal { G }$ in which every ordinal node carries at least three categories, every exponential family node at least three points of support, every link $\eta$ is monotone injective, and every edge weight is nonzero. Then for every $\mathcal { G } ^ { \prime } \in \mathcal { O } \left( \mathcal { G } _ { u } \right)$ with $\mathcal { G } ^ { \prime } \neq \mathcal { G }$ , no ordinal-exponential LPM on $\mathcal { G } ^ { \prime }$ generates $p \nu$ , irrespective of how many edges the two orientations difer in.

Proof See Appendix A.9.

Theorem 5 strengthens Theorem 4 from a single reversal to the entire orientation space, and the strengthening does not follow from repeated application of the earlier result: a competing orientation may difer in many edges at once, and the comparisons cannot be chained through intermediate graphs, none of which is the truth. Theorem 5 instead excludes every member of $\mathcal { O } \left( \mathcal { G } _ { u } \right)$ simultaneously.

We note that the result concerns distributions rather than any particular score, although a consequence for score-based estimation is immediate: for a likelihood-based criterion, Theorem 5 gives strictly positive Kullback–Leibler divergence from $p \nu$ at every competing orientation and every parameter value, so the truth is the unique population optimum. Uniqueness is weaker than consistency, which would additionally require that the divergence remain bounded away from zero as the competing parameters range over an unbounded set, a condition that rests on a compactness argument we do not pursue herein. Whether the two conditions of Theorem 4 remain necessary once the edge sits in a graph is settled next.

Theorem 6 Let $\mathcal { G } = ( \nu , \mathcal { E } )$ be a causal DAG on d nodes with undirected skeleton $\mathcal { G } _ { u } =$ $( \nu , \mathcal { E } _ { u } )$ , in which each node is designated either ordinal or exponential family and every edge weight is nonzero. Let each non-root ordinal node k with $s _ { k }$ categories follow the ordered-logit conditional (17) given its parents with cutpoints $- \infty = \gamma _ { 0 } < \gamma _ { 1 } < \cdot \cdot \cdot < \gamma _ { s _ { k } - 1 } < \gamma _ { s _ { k } } = + \infty$ let each non-root exponential family node k follow the regular one-parameter exponential family conditional (14) given its parents with suficient statistic $T _ { k }$ , monotone injective link η, and support $\mathcal { X } _ { k }$ , and let each root node k follow an arbitrary strictly positive marginal on its support, categorical on $\{ 1 , \ldots , s _ { k } \}$ when k is ordinal. Let the joint distribution $p \nu$ be Markov with respect to G, factorizing as $\begin{array} { r } { p _ { \mathcal { V } } = \prod _ { k \in \mathcal { V } } p _ { X _ { k } | \mathbf { x } _ { \mathcal { P } ( k ) } . } } \end{array}$ , with G causally suficient, containing no latent confounders and no selection variables. Let $( i , j ) \in \mathcal { E } _ { u }$ join an ordinal node i with s categories to an exponential family node j with suficient statistic T and support $\chi _ { j }$ , where neither i nor $j$ is adjacent in $\mathcal { G } _ { u }$ to any node other than the other endpoint. If either the exponential family node has exactly two points of support, $| { \mathcal { X } } _ { j } | = 2$ , with afine suficient statistic $T \left( y \right) = c _ { 1 } y + c _ { 0 } , c _ { 0 } , c _ { 1 } \in \mathbb { R } , c _ { 1 } \neq 0$ , and canonical link $\eta \left( z \right) = z$ , or the ordinal node has exactly two categories, $s = 2$ , with T afine on $\chi _ { j }$ in the same sense, then the direction of the edge (i, j) is not distributionally identifiable.

Proof See Appendix A.10.

Theorem 6 shows that neither condition of Theorem 4 can be dropped, under the condition that the true DAG has isolated ordinal–exponential-family edges. When the edge is not isolated, the remaining parents of the endpoints impose constraints that the bivariate construction need not satisfy, and whether identifiability is restored at such an edge is beyond the scope of this paper. The converse regime is correspondingly narrow, requiring a two-node connected component of the skeleton together with either a two-point support or a binary ordinal node, and an afine suficient statistic in either case. The conditions on the number of ordinal categories and the size of the support are therefore suficient in an arbitrary causal DAG, and necessary as well for an edge whose endpoints carry no neighbors other than each other.

## 4 Numerical Results and Discussion

We illustrate the identifiability results of Section 3 on synthetic data generated from the models of Section 2. Since Definition 3 compares graphs sharing a skeleton, the estimation problem the theory poses is the orientation of a known bipartite skeleton rather than its recovery, and no procedure below inserts or deletes an edge. The graph considered here is suficiently small that an exhaustive search yields an exact solution to the score minimization, so the recovery error reflects only finite-sample noise. The large-graph regime, in which greedy search supplies a scalable but heuristic estimator, is reported in Appendix D. Our main aim for this section is to show that DAGs are identifiable beyond the MEC, in contrast to non-identifiable settings such as linear SEMs, where observational discovery is limited to the MEC (Spirtes et al., 2001; Chickering, 2002; Peters and B¨uhlmann, 2014). Additionally, we note that estimation is not a central focus of this work. The estimators used in this section and the corresponding appendices are either exhaustive search-based or greedy heuristics, and are used solely to demonstrate the theoretical population-level identifiability results, which remain the paper’s core focus.

Let $\mathbf { X } \in \mathbb { R } ^ { n \times d }$ be an n-sample dataset generated by an unknown d-node ground truth DAG G, and let $\mathcal { P } _ { \mathcal { G } _ { \mathrm { e s t } } } \left( i \right)$ be the parent set of node i under an estimated graph $\mathcal { G } _ { \mathrm { e s t } }$ with weighted adjacency matrix $\mathbf { W } _ { \mathrm { e s t } }$ . Candidates are scored by the Bayesian Information Criterion (Chickering, 2002),

$$
\operatorname { B I C } \left( \mathcal { G } _ { \mathrm { e s t } } ; \mathbf { X } \right) \triangleq - \sum _ { j = 1 } ^ { n } \sum _ { i = 1 } ^ { d } \ln p \left( \mathbf { X } _ { j , i } \mid \mathbf { X } _ { j , \mathcal { P } _ { \mathcal { G } _ { \mathrm { e s t } } } ( i ) } ; \mathbf { W } _ { \mathrm { e s t } } , \gamma _ { i } \right) + \frac { \log n } { 2 } \sum _ { i = 1 } ^ { d } k _ { i } ,\tag{37}
$$

which is to be minimized over the acyclic orientations $\mathcal { O } \left( \mathcal { G } _ { u } \right)$ of the supplied skeleton, with $k _ { i }$ the number of parameters fitted at node i. Both terms are sums over nodes, so the score decomposes into node contributions, and the parameters of each candidate are estimated per node using conditional maximum likelihood. Additional details on estimation algorithms, parameter values, and the simulation code repository are provided in Appendix B and Appendix $\mathrm { C } ,$ with experimentation details in Appendix E.

Every candidate carries the skeleton $\mathcal { G } _ { u }$ of the truth, so an estimate can difer from the truth only in the direction assigned to an edge, and we measure recovery by the orientation error rate

$$
\rho \left( \mathcal { G } _ { \mathrm { e s t } } , \mathcal { G } \right) \triangleq \frac { 1 } { \left| \mathcal { E } _ { u } \right| } \left| \left\{ ( i , j ) \in \mathcal { E } _ { u } : \mathcal { G } _ { \mathrm { e s t } } \mathrm { a n d } \mathcal { G } \mathrm { o r i e n t } ( i , j ) \mathrm { o p p o s i t e l y } \right\} \right| ,\tag{38}
$$

which measures the fraction of skeleton edges assigned the wrong direction. With the skeleton fixed, every discrepancy is a reversal, so $\rho$ is the structural Hamming distance normalized by edge count, lies in [0, 1], and is comparable across skeletons of diferent size and density. An estimated graph that is Markov-equivalent to the truth but oriented diferently has strictly positive $\rho ,$ so the figures below report identifiability within Markov equivalence classes rather than up to them.

We consider three ground-truth bipartite DAGs over $\{ X _ { 1 } , X _ { 2 } , X _ { 3 } \}$ with $X _ { 1 } , X _ { 3 } \in \mathcal { V } _ { \mathrm { o r d } }$ and $X _ { 2 } \in \mathcal { V } _ { \mathrm { e x p } }$

$$
\begin{array} { r l } & { \mathcal { G } _ { 1 } \mathrm { ~ ( c h a i n ) } \mathrm { : } \quad X _ { 1 } \to X _ { 2 } \to X _ { 3 } , } \\ & { \mathcal { G } _ { 2 } \mathrm { ~ ( i n v e r s e ~ f o r k ) } \mathrm { : } \quad X _ { 1 } \to X _ { 2 }  X _ { 3 } , } \\ & { \mathcal { G } _ { 3 } \mathrm { ~ ( f o r k ) } \mathrm { : } \quad X _ { 1 }  X _ { 2 } \to X _ { 3 } . } \end{array}\tag{39}
$$

The chain and the fork share a skeleton and have no v-structures, so both lie in one Markov equivalence class, while the inverse fork has a v-structure at $X _ { 2 }$ and lies in another. Exhaustive search runs over the four acyclic orientations of $X _ { 1 } - X _ { 2 } - X _ { 3 } .$ , exactly the set Theorem 5 separates, so any deviation from the truth is finite-sample noise rather than optimization error. The four do not carry equal parameter counts: summing $k _ { i }$ gives the edge count and the ordinal cutpoint totals, both fixed across orientations, plus the number of exponential-family nodes left without parents (which are treated as free variables as they can have an arbitrary marginal as given in (10)), so the fork carries one parameter more than the rest and the penalty in (37) charges accordingly.

Figure 2 shows $\rho$ against N for each structure, with $X _ { 2 }$ drawn in turn from each of the seven families of Table 1. Every curve decays with N, the flat asymptotes being the numerical floor of the log plot, at which no trial recovered an incorrect orientation. The three structures difer in the orientation error decay rates the inverse fork is fastest, reaching the floor for every family but Binomial and Pascal; the chain reaches the floor for Poisson and Gamma (β) alone, with Exponential the slowest curve in the figure; and the fork moves least at the smallest sample sizes and accelerates as sample sizes get larger, reaching the floor for Pascal, Gaussian, and Gamma (α). The rate of descent depends on the criterion at finite N and on the generating parameters of Appendix E, which cannot be justified from population-level identifiability alone. We see in Figure 2 that both the fork and the chain structures can be distinguished despite being in the same MEC. That the error decays with the number of observations and is zero for some of the families verifies the result that Theorem 4 establishes in the population law. The rapid convergence of the inverse fork instead reflects the ease of detecting a v-structure, as it is the only member of its MEC, and serves as a check that the search favors no particular structure.

![](images/2be9a4765ce3edc05db6413cf33d191f64767d06e2ea8f9d4f448d9b9ff3024a.jpg)  
(a) G<sub>1</sub>: $X _ { 1 }  X _ { 2 }  X _ { 3 }$ (chain)

![](images/2626fb48767c7c94d9012b8894ac1e6eacb4c6a131d5964a83a7d2a0dc6bc699.jpg)  
(b) G<sub>2</sub>: $X _ { 1 } \right. X _ { 2 } \left. X _ { 3 }$ (inverse fork)

![](images/5e2e6767dec69c1efae6a822bea291376e036926075f2429cb4041c08449d93c.jpg)  
(c) $\mathcal { G } _ { 3 } \colon X _ { 1 } \left. X _ { 2 } \right. X _ { 3 }$ (fork)  
Figure 2: Orientation error rate ρ versus sample size N for the three-node bipartite DAGs, with $X _ { 2 }$ drawn from each family of Table 1, over B = 10000 trials.

## 5 Conclusions

This paper establishes the identifiability of causal DAGs whose nodes follow either the ordered-logit ordinal model or a regular one-parameter exponential family, within the Linear Parametric Model framework defined herein. Two features distinguish the guarantee obtained: it holds at every parameter value rather than outside an exceptional set, and it does not constrain the suficient statistic – governing finite, countable, and continuous supports alike. We also provide converses highlighting the limits of the identifiability results and the necessity of the conditions; a multivariate extension that carries the conclusion to every such edge of a d-node DAG under the Markov property, causal suficiency, and the absence of selection. Our numerical experiments confirm that the orientation error rate decays as the sample size increases within a Markov equivalence class. Future work includes consistency guarantees for likelihood-based estimators under the LPM, edges joining two exponential-family nodes, multi-parameter families with unknown nuisance parameters, and cumulative links beyond the logit.

## Acknowledgments and Disclosure of Funding

We are grateful to Christine K. Johnson and Yingying Wang of the University of California, Davis, for discussions of epidemiological practice, and to Joni Shaska for discussions of ordinal links and for early assistance with the simulation setup.

This work has been funded by one or all of the following grants: ARO W911NF1910269, ARO W911NF2410094, ONR N00014-22-1-2363, NSF CIF-2311653, NSF CIF-2148313, NSF RINGS-2148313, NSF DBI-2412522, and is also supported in part by funds from federal agencies and industry partners as specified in the RINGS program.

## Appendix A. Proofs

## A.1 Proof of Proposition 1

Proof We proceed by contradiction. Assume the conditional distribution $p _ { X | Y } \left( \cdot \mid \cdot \right)$ is a one-parameter exponential family in X. Then there exist a base measure $\{ h _ { x } \} _ { x = 1 } ^ { s }$ with $h _ { x } > 0$ , a suficient statistic $\{ T _ { x } \} _ { x = 1 } ^ { s } ,$ , a scalar natural parameter $\theta \left( y \right)$ , and a log-partition function A such that

$$
p \left( { X } = x \mid { Y } = y \right) = h _ { x } \exp ( \theta \left( { y } \right) T _ { x } - A \left( \theta \left( y \right) \right) ) , \quad \forall x \in \mathcal { X } , \ y \in \mathcal { Y } .\tag{40}
$$

Fix distinct $x , x ^ { \prime } \in { \mathcal { X } }$ . Taking the ratio in (40) cancels the log-partition term, and taking logarithms gives

$$
\log \frac { p \left( x \mid y \right) } { p \left( x ^ { \prime } \mid y \right) } = \log \frac { h _ { x } } { h _ { x ^ { \prime } } } + \theta \left( y \right) \left( T _ { x } - T _ { x ^ { \prime } } \right) , \quad \forall y \in \mathcal { V } ,\tag{41}
$$

so every pairwise log-ratio is afine in the scalar $\theta \left( y \right)$ , with slope $T _ { x } - T _ { x ^ { \prime } }$ independent of y. Let $u , v , r$ be pairwise distinct with $T _ { v } \neq T _ { r }$ . Writing (41) for the pairs $( u , r )$ and $( v , r )$ and eliminating $\theta \left( y \right)$ between them,

$$
\log \frac { p \left( u \mid y \right) } { p \left( r \mid y \right) } = \frac { T _ { u } - T _ { r } } { T _ { v } - T _ { r } } \log \frac { p \left( v \mid y \right) } { p \left( r \mid y \right) } + c , \quad \forall y \in \mathcal { V } ,\tag{42}
$$

with c independent of $y .$ We exhibit three categories for which (42) fails.

Take the categories $\{ 1 , 2 , s \}$ and set $z \triangleq w y$ , strictly monotone in y since w $\neq 0 ,$ , so that distinct points of Y give distinct values of $z .$ From (12) with the boundary conventions $e ^ { \gamma _ { 0 } } = 0$ and $e ^ { \gamma _ { s } } = \infty$ ，

$$
p \left( 1 \mid y \right) = \frac { e ^ { \gamma _ { 1 } } } { e ^ { \gamma _ { 1 } } + e ^ { z } } , \quad p \left( s \mid y \right) = \frac { e ^ { z } } { e ^ { \gamma _ { s - 1 } } + e ^ { z } } , \quad p \left( 2 \mid y \right) = \frac { \left( e ^ { \gamma _ { 2 } } - e ^ { \gamma _ { 1 } } \right) e ^ { z } } { \left( e ^ { \gamma _ { 1 } } + e ^ { z } \right) \left( e ^ { \gamma _ { 2 } } + e ^ { z } \right) } ,\tag{43}
$$

where $e ^ { \gamma _ { 2 } } - e ^ { \gamma _ { 1 } } > 0$ by the strict ordering of the cutpoints. Taking category 1 as reference, define

$$
\ell _ { 2 1 } \left( z \right) \triangleq \log \frac { p \left( 2 \mid y \right) } { p \left( 1 \mid y \right) } , \qquad \ell _ { s 1 } \left( z \right) \triangleq \log \frac { p \left( s \mid y \right) } { p \left( 1 \mid y \right) } .\tag{44}
$$

Substituting (43) into (44),

$$
\begin{array} { r l } & { \ell _ { 2 1 } \left( z \right) = \log ( e ^ { \gamma _ { 2 } } - e ^ { \gamma _ { 1 } } ) - \gamma _ { 1 } + z - \log ( e ^ { \gamma _ { 2 } } + e ^ { z } ) , } \\ & { \ell _ { s 1 } \left( z \right) = z + \log ( e ^ { \gamma _ { 1 } } + e ^ { z } ) - \gamma _ { 1 } - \log ( e ^ { \gamma _ { s - 1 } } + e ^ { z } ) , } \end{array}\tag{45}
$$

both diferentiable on $\mathbb { R } ,$ with

$$
\ell _ { 2 1 } ^ { \prime } \left( z \right) = \sigma ( \gamma _ { 2 } - z ) > 0 , \qquad \ell _ { s 1 } ^ { \prime } \left( z \right) = \sigma ( \gamma _ { s - 1 } - z ) + \sigma ( z - \gamma _ { 1 } ) > 0 .\tag{46}
$$

Since $\ell _ { 2 1 } ^ { \prime } \left( z \right) > 0$ , the map $z \mapsto \ell _ { 2 1 } \left( z \right)$ is strictly increasing, so the planar curve $z \mapsto$ $( \ell _ { 2 1 } \left( z \right) , \ell _ { s 1 } \left( z \right) )$ admits reparametrization by its first coordinate, along which

$$
\frac { d \ell _ { s 1 } } { d \ell _ { 2 1 } } = \frac { \sigma ( \gamma _ { s - 1 } - z ) } { \sigma ( \gamma _ { 2 } - z ) } + \frac { \sigma ( z - \gamma _ { 1 } ) } { \sigma ( \gamma _ { 2 } - z ) } .\tag{47}
$$

Set $t \triangleq e ^ { z }$ , strictly increasing in z. Substituting $\sigma ( \gamma - z ) = e ^ { \gamma } / \left( e ^ { \gamma } + t \right)$ and $\sigma ( z - \gamma _ { 1 } ) =$ $t / \left( e ^ { \gamma _ { 1 } } + t \right)$ into (47),

$$
\frac { d \ell _ { s 1 } } { d \ell _ { 2 1 } } = F _ { 1 } \left( t \right) + F _ { 2 } \left( t \right) , \quad \mathrm { w h e r e ~ } F _ { 1 } \left( t \right) \triangleq \frac { e ^ { \gamma _ { s - 1 } } \left( e ^ { \gamma _ { 2 } } + t \right) } { e ^ { \gamma _ { 2 } } \left( e ^ { \gamma _ { s - 1 } } + t \right) } , \quad F _ { 2 } \left( t \right) \triangleq \frac { t \left( e ^ { \gamma _ { 2 } } + t \right) } { e ^ { \gamma _ { 2 } } \left( e ^ { \gamma _ { 1 } } + t \right) } .\tag{48}
$$

Diferentiating in $t ,$

$$
F _ { 1 } ^ { \prime } \left( t \right) = \frac { e ^ { \gamma _ { s - 1 } } \left( e ^ { \gamma _ { s - 1 } } - e ^ { \gamma _ { 2 } } \right) } { e ^ { \gamma _ { 2 } } \left( e ^ { \gamma _ { s - 1 } } + t \right) ^ { 2 } } \geq 0 , \qquad F _ { 2 } ^ { \prime } \left( t \right) = \frac { t ^ { 2 } + 2 t e ^ { \gamma _ { 1 } } + e ^ { \gamma _ { 1 } + \gamma _ { 2 } } } { e ^ { \gamma _ { 2 } } \left( e ^ { \gamma _ { 1 } } + t \right) ^ { 2 } } > 0 ,\tag{49}
$$

the first inequality holding since $\gamma _ { s - 1 } \geq \gamma _ { 2 }$ , with equality throughout exactly when $s = 3 ,$ and the second since $t > 0$ . Hence $F _ { 1 } ^ { \prime } \left( t \right) + F _ { 2 } ^ { \prime } \left( t \right) > 0$ , and since t is strictly increasing in $z ,$

$$
\begin{array} { r l } & { ~ \frac { d } { d t } \bigg ( \frac { d \ell _ { s 1 } } { d \ell _ { 2 1 } } \bigg ) > 0 , ~ \forall t > 0 } \\ & { \implies \frac { d } { d z } \bigg ( \frac { d \ell _ { s 1 } } { d \ell _ { 2 1 } } \bigg ) > 0 , ~ \forall z \in \mathbb { R } } \\ & { \implies \frac { d ^ { 2 } \ell _ { s 1 } } { d \ell _ { 2 1 } ^ { 2 } } = \frac { 1 } { \ell _ { 2 1 } ^ { \prime } ( z ) } \frac { d } { d z } \bigg ( \frac { d \ell _ { s 1 } } { d \ell _ { 2 1 } } \bigg ) > 0 , ~ \forall z \in \mathbb { R } , } \end{array}\tag{50}
$$

the last implication applying the chain rule to the reparametrization together with $\ell _ { 2 1 } ^ { \prime } \left( z \right) >$ 0 from (46). Hence $\ell _ { s 1 }$ is a strictly convex function of $\ell _ { 2 1 }$ , and no three distinct points of the curve are collinear.

It remains to contradict (42). Taking $x = 2$ and $x ^ { \prime } = 1$ in (41) and comparing with (44),

$$
\ell _ { 2 1 } \left( z \right) = \log \frac { h _ { 2 } } { h _ { 1 } } + \theta \left( y \right) \left( T _ { 2 } - T _ { 1 } \right) .\tag{51}
$$

$\mathrm { I f } T _ { 2 } = T _ { 1 }$ , then $\ell _ { 2 1 }$ would be constant on Y by (51), contradicting $\ell _ { 2 1 } ^ { \prime } \left( z \right) > 0$ in (46). Hence $T _ { 2 } \neq T _ { 1 }$ , so (42) applies to $( u , v , r ) = ( s , 2 , 1 )$ and yields constants $a = \left( T _ { s } - T _ { 1 } \right) / \left( T _ { 2 } - T _ { 1 } \right)$ and $b ,$ both independent of $y ,$ with

$$
\ell _ { s 1 } \left( z \right) = a \ell _ { 2 1 } \left( z \right) + b .\tag{52}
$$

Since $| \mathscr { y } | \geq 3$ and $\ell _ { 2 1 }$ is strictly increasing in z, three distinct points of $\mathcal { V }$ give three distinct points of the curve, which (52) places on a common line, contradicting the strict convexity established in (50). Therefore, we conclude by contradiction that no base measure, suficient statistic, natural parameter, and log-partition function satisfy (40), and the conditional is therefore not a one-parameter exponential family in $X$

## A.2 Proof of Lemma 1

Proof We proceed by direct computation. From (15), the joint distribution under the forward model $\mathcal { M } _ { X  Y }$ is

$$
p _ { X , Y } ( x , y ; \mathcal { M } _ { X  Y } ) = \pi _ { x } H ( y ) \exp ( \eta ( w x ) T ( y ) - a ( \eta ( w x ) ) ) , \quad \forall y \in \mathcal { Y } , \ x \in \mathcal { X } .\tag{53}
$$

Since $\pi _ { x } > 0$ and, by definition of an exponential family, $H \left( y \right) \exp ( \eta ( w x ) T \left( y \right) - a ( \eta ( w x ) ) ) >$ 0 for all $y \in \mathcal { V }$

$$
\begin{array} { r l } & { p _ { X , Y } ( x , y ; \mathcal { M } _ { X  Y } ) > 0 , \quad \forall x \in \mathcal { X } , y \in \mathcal { Y } } \\ { \implies } & { \displaystyle \sum _ { x ^ { \prime } \in \mathcal { X } } p _ { X , Y } ( x ^ { \prime } , y ; \mathcal { M } _ { X  Y } ) > 0 , \quad \forall y \in \mathcal { Y } } \\ { \implies } & { p _ { Y } ( y ; \mathcal { M } _ { X  Y } ) > 0 , \quad \forall y \in \mathcal { Y } . } \end{array}\tag{54}
$$

Fix distinct $u , x \in { \mathcal { X } }$ . Since $p _ { Y } ( y ; \mathcal { M } _ { X  Y } ) > 0$ by (54), dividing numerator and denominator by the marginal equates the posterior ratio with the joint ratio,

$$
\frac { p _ { X | Y } ( u \mid y ; \mathcal { M } _ { X  Y } ) } { p _ { X | Y } ( x \mid y ; \mathcal { M } _ { X  Y } ) } = \frac { p _ { X | Y } ( u \mid y ; \mathcal { M } _ { X  Y } ) p _ { Y } ( y ; \mathcal { M } _ { X  Y } ) } { p _ { X | Y } ( x \mid y ; \mathcal { M } _ { X  Y } ) p _ { Y } ( y ; \mathcal { M } _ { X  Y } ) } = \frac { p _ { X , Y } ( u , y ; \mathcal { M } _ { X  Y } ) } { p _ { X , Y } ( x , y ; \mathcal { M } _ { X  Y } ) } ,\tag{55}
$$

the last equality applying the product rule $p _ { X , Y } \left( x , y \right) = p _ { X | Y } \left( x \mid y \right) p _ { Y } \left( y \right)$ . Taking logarithms and substituting (15),

$$
\begin{array} { r l r } {  { R ( y , u , x ; \mathcal { M } _ { X \to Y } ) = \log \frac { p _ { X , Y } ( u , y ; \mathcal { M } _ { X \to Y } ) } { p _ { X , Y } ( x , y ; \mathcal { M } _ { X \to Y } ) } } } \\ & { } & { = ( \eta ( w u ) - \eta ( w x ) ) T ( y ) + a ( \eta ( w x ) ) - a ( \eta ( w u ) ) + \log ( \frac { \pi _ { u } } { \pi _ { x } } ) . } \end{array}\tag{56}
$$

Define $\alpha$ and $\beta ,$ independent of $y ,$ as

$$
\alpha = \eta \left( w u \right) - \eta \left( w x \right) , \qquad \beta = a \left( \eta \left( w x \right) \right) - a \left( \eta \left( w u \right) \right) + \log \left( \frac { \pi _ { u } } { \pi _ { x } } \right) ,\tag{57}
$$

where $\alpha \neq 0$ since $\eta \left( \cdot \right)$ is monotone injective, $w \ne 0$ , and $u \neq x$ , yielding

$$
R ( y , u , x ; \mathcal { M } _ { X  Y } ) = \alpha T ( y ) + \beta ,\tag{58}
$$

as claimed.

## A.3 Proof of Lemma 2

Proof We compute the two extreme adjacent log-odds-ratios from the conditional (17) and the definition (19). Using $\sigma \left( \gamma _ { k } - v y \right) = e ^ { \gamma _ { k } } / \left( e ^ { \gamma _ { k } } + e ^ { v y } \right)$ together with the boundary conventions $e ^ { \gamma _ { 0 } } = 0$ and $e ^ { \gamma _ { s } } = \infty$ , we obtain

$$
\begin{array} { r l } & { B ( y ) = \log ( \frac { p _ { X \mid Y } ( 2 \mid y ; M _ { Y  X } ) } { p _ { X \mid Y } ( 1 \mid y ; M _ { Y  X } ) } ) } \\ & { \quad = \log ( \frac { \sigma ( \gamma _ { 2 } - v y ) - \sigma ( \gamma _ { 1 } - v y ) } { \sigma ( \gamma _ { 1 } - v y ) - \sigma ( \gamma _ { 0 } - v y ) } ) } \\ & { \quad = \log ( \frac { e ^ { v y } ( e ^ { \gamma _ { 2 } } - e ^ { \gamma _ { 1 } } ) } { e ^ { \gamma _ { 1 } } ( e ^ { \gamma _ { 2 } } + e ^ { v y } ) } ) } \\ & { \quad = \log ( \frac { e ^ { \gamma _ { 2 } } - e ^ { \gamma _ { 1 } } } { e ^ { \gamma _ { 1 } } } ) - \log ( 1 + e ^ { \gamma _ { 2 } - v y } ) . } \end{array}\tag{59}
$$

Similarly, for $U \left( y \right)$ we have

$$
\begin{array} { r l } & { U ( y ) = \log ( \frac { p _ { X \mid Y } ( s \mid y ; \mathcal { M } _ { Y  X } ) } { p _ { X \mid Y } ( s - 1 \mid y ; \mathcal { M } _ { Y  X } ) } ) } \\ & { \quad \quad = \log ( \frac { \sigma ( \gamma _ { s } - v y ) - \sigma ( \gamma _ { s - 1 } - v y ) } { \sigma ( \gamma _ { s - 1 } - v y ) - \sigma ( \gamma _ { s - 2 } - v y ) } ) } \\ & { \quad \quad = \log ( \frac { e ^ { \gamma _ { s - 2 } } + e ^ { v y } } { e ^ { \gamma _ { s - 1 } } - e ^ { \gamma _ { s - 2 } } } ) } \\ & { \quad \quad = \log ( \frac { e ^ { \gamma _ { s - 2 } } } { e ^ { \gamma _ { s - 1 } } - e ^ { \gamma _ { s - 2 } } } ) + \log ( 1 + e ^ { v y - \gamma _ { s - 2 } } ) . } \end{array}\tag{60}
$$

Defining constants $c _ { B }$ and $c _ { U }$ , independent of $y ,$ as

$$
c _ { B } = \log \left( \frac { e ^ { \gamma _ { 2 } } - e ^ { \gamma _ { 1 } } } { e ^ { \gamma _ { 1 } } } \right) , \quad c _ { U } = \log \left( \frac { e ^ { \gamma _ { s - 2 } } } { e ^ { \gamma _ { s - 1 } } - e ^ { \gamma _ { s - 2 } } } \right) ,\tag{61}
$$

where the strict ordering of the cutpoints together with $s \geq 3$ , so that $s - 2 \geq 1$ , ensures that $c _ { B }$ and $c _ { U }$ are finite, yields the closed-form expressions

$$
B \left( y \right) = c _ { B } - \log \left( 1 + e ^ { \gamma _ { 2 } - v y } \right) , \quad U \left( y \right) = c _ { U } + \log \left( 1 + e ^ { v y - \gamma _ { s - 2 } } \right) .\tag{62}
$$

The closed forms involve y only through $e ^ { \pm v y }$ and are therefore twice diferentiable at every $y \in \mathbb { R } ,$ whereas B and U are defined only on ${ \mathcal { V } } ,$ which need not be an interval. Let $\bar { B } , \bar { U } : \mathbb { R } $ R denote the functions the closed forms define on all of R, so $B = \left. \bar { B } \right| _ { y }$ and $U = { \bar { U } } | _ { y }$ . Diferentiating twice with respect to y,

$$
\frac { d ^ { 2 } \bar { B } \left( y \right) } { d y ^ { 2 } } = - \frac { v ^ { 2 } e ^ { \gamma _ { 2 } - v y } } { \left( 1 + e ^ { \gamma _ { 2 } - v y } \right) ^ { 2 } } < 0 , \quad \forall y \in \mathbb { R } ,\tag{63}
$$

$$
\frac { d ^ { 2 } \hat { U } \left( y \right) } { d y ^ { 2 } } = \frac { v ^ { 2 } e ^ { v y - \gamma _ { s - 2 } } } { \left( 1 + e ^ { v y - \gamma _ { s - 2 } } \right) ^ { 2 } } > 0 , \quad \forall y \in \mathbb { R } ,\tag{64}
$$

both inequalities being strict because $v \ne 0$ . Hence B<sup>¯</sup> is strictly concave and U<sup>¯</sup> strictly convex on R, with B and U having their restrictions to Y. ■

## A.4 Proofs of Theorem 1

First proof, by the curvature of Lemma 2. We proceed by contradiction. Assume that the models are not identifiable. $\mathrm { B y }$ Definition 2, there exist joint distributions $p _ { X , Y } \in$ $\mathcal { M } _ { X  Y }$ and $q _ { X , Y } \in \mathcal { M } _ { Y  X }$ such that

$$
p _ { X , Y } \left( x , y \right) = q _ { X , Y } \left( x , y \right) , \quad \forall x \in \mathcal { X } , \ y \in \mathcal { Y } .\tag{65}
$$

Summing over $x \in \mathcal { X }$ gives $p _ { Y } \left( y \right) = q _ { Y } \left( y \right)$ , strictly positive, and dividing the joint by the common marginal yields

$$
p _ { X | Y } \left( x \mid y \right) = q _ { X | Y } \left( x \mid y \right) , \quad \forall x \in \mathcal { X } , y \in \mathcal { Y } .\tag{66}
$$

![](images/313df6055bfa585eb16788c6b98d56dca6d36dc637250e421c0e19481808a561.jpg)  
(a) Lemma 1 straight, Lemma 2 bent

![](images/998e66f144733f55d63959323c976e54a0740e22390940fc06819be2c2a09bc1.jpg)  
(b) at most two crossings, but $| \mathscr { y } | \geq 3$  
Figure 3: The asymmetry behind Theorem 1.

Hence, for distinct $u , x \in { \mathcal { X } }$ and any $y \in \mathcal { V }$ , the log-odds-ratios of the two models coincide,

$$
R ( y , u , x ; \mathcal { M } _ { X  Y } ) = R ( y , u , x ; \mathcal { M } _ { Y  X } ) , \quad \forall y \in \mathcal { V } , \ u , x \in \mathcal { X } , \ u \neq x .\tag{67}
$$

From Lemma 1, there exist real $\alpha _ { u , x } \neq 0$ and $\beta _ { u , x } ,$ , independent of $y ,$ such that

$$
R ( y , u , x ; \mathcal { M } _ { X  Y } ) = \alpha _ { u , x } T ( y ) + \beta _ { u , x } , \quad \forall y \in \mathcal { V } , \ u , x \in \mathcal { X } , \ u \neq x .\tag{68}
$$

Combining this with (67),

$$
R ( y , u , x ; \mathcal { M } _ { Y  X } ) = \alpha _ { u , x } T ( y ) + \beta _ { u , x } , \quad \forall y \in \mathcal { V } , \ u , x \in \mathcal { X } , \ u \neq x .\tag{69}
$$

Choosing $u = 2$ and $x = 1$ , valid since $s \geq 3$ , and substituting into (69),

$$
\begin{array} { r l } & { R ( y , 2 , 1 ; \mathcal { M } _ { Y  X } ) = \alpha _ { 2 , 1 } T ( y ) + \beta _ { 2 , 1 } , \quad \forall y \in \mathcal { V } , } \\ { \implies } & { B ( y ) = \alpha _ { 2 , 1 } T ( y ) + \beta _ { 2 , 1 } , \quad \forall y \in \mathcal { V } , } \end{array}\tag{70}
$$

where $B \left( y \right)$ is the lower extreme log-odds-ratio as defined in (22). Choosing $u = s$ and $x = s - 1$ , also valid since $s \geq 3$ , and substituting into (69),

$$
\begin{array} { c c } { { R ( y , s , s - 1 ; \mathcal { M } _ { Y  X } ) = \alpha _ { s , s - 1 } T ( y ) + \beta _ { s , s - 1 } , } } & { { \forall y \in \mathcal { V } , } } \\ { { \mathrm { } } } & { { } } \\ { { \Longrightarrow ~ U ( y ) = \alpha _ { s , s - 1 } T ( y ) + \beta _ { s , s - 1 } , } } & { { \forall y \in \mathcal { V } , } } \end{array}\tag{71}
$$

where $U \left( y \right)$ is the upper extreme log-odds-ratio also defined in (22). Define

$$
\Psi \left( y \right) \triangleq \alpha _ { s , s - 1 } B \left( y \right) - \alpha _ { 2 , 1 } U \left( y \right) .\tag{72}
$$

Multiplying (70) by $\alpha _ { s , s - 1 }$ and (71) by $\alpha _ { 2 , 1 }$ and subtracting, the term in $T \left( y \right)$ cancels, leaving only the intercepts, so that

$$
\Psi \left( y \right) = \alpha _ { s , s - 1 } \beta _ { 2 , 1 } - \alpha _ { 2 , 1 } \beta _ { s , s - 1 } , \quad \forall y \in \mathcal { V } ,\tag{73}
$$

the right-hand side being independent of y. Set $t \triangleq e ^ { v y }$ , a strictly monotone injective transformation of $y ,$ and let ${ \mathcal { T } } \triangleq \{ e ^ { v y } : y \in { \mathcal { V } } \} \subseteq ( 0 , \infty )$ . Substituting the closed forms of B<sup>¯</sup> and U<sup>¯</sup> from Lemma 2 into (72), which extends Ψ from $\tau$ to all of $( 0 , \infty )$ , and using $e ^ { \gamma _ { 2 } - v y } = e ^ { \gamma _ { 2 } } / t$ and $e ^ { v y - \gamma _ { s - 2 } } = t / e ^ { \gamma _ { s - 2 } }$ we get,

$$
\Psi \left( t \right) = \alpha _ { s , s - 1 } \left( \log t - \log \left( e ^ { \gamma _ { 2 } } + t \right) \right) - \alpha _ { 2 , 1 } \log \left( e ^ { \gamma _ { s - 2 } } + t \right) + k , \quad \forall t \in \left( 0 , \infty \right) ,\tag{74}
$$

where k collects the terms independent of t. Setting the derivative in t to zero,

$$
\begin{array} { c } { { \displaystyle \frac { d \Psi \left( t \right) } { d t } = 0 } } \\ { { \implies \alpha _ { s , s - 1 } \left( \displaystyle \frac { 1 } { t } - \displaystyle \frac { 1 } { e ^ { \gamma _ { 2 } } + t } \right) - \displaystyle \frac { \alpha _ { 2 , 1 } } { e ^ { \gamma _ { s - 2 } } + t } = 0 . } } \end{array}\tag{75}
$$

Since $t > 0$ , both $e ^ { \gamma _ { 2 } } + t > 0$ and $e ^ { \gamma _ { s - 2 } } + t > 0 $ ; clearing these denominators reduces the equation to

$$
\alpha _ { 2 , 1 } t ^ { 2 } + \left( \alpha _ { 2 , 1 } - \alpha _ { s , s - 1 } \right) e ^ { \gamma _ { 2 } } t - \alpha _ { s , s - 1 } e ^ { \gamma _ { 2 } + \gamma _ { s - 2 } } = 0 .\tag{76}
$$

By Lemma 1 and (14), $\eta \left( \cdot \right)$ is monotone injective and $\alpha _ { u , x } = \eta \left( w u \right) - \eta \left( w x \right) \neq 0$ for distinct $u , x$ . Hence

$$
\frac { \alpha _ { s , s - 1 } } { \alpha _ { 2 , 1 } } = \frac { \eta \left( s w \right) - \eta \left( s w - w \right) } { \eta \left( 2 w \right) - \eta \left( w \right) } > 0 , \quad \forall w \neq 0 ,\tag{77}
$$

since a monotone η makes both diferences share the sign of w. Writing $t _ { + }$ and t<sub>−</sub> for the two roots of (76), their product is

$$
t _ { + } t _ { - } = - \frac { \alpha _ { s , s - 1 } } { \alpha _ { 2 , 1 } } e ^ { \gamma _ { 2 } + \gamma _ { s - 2 } } < 0 ,\tag{78}
$$

by (77), so exactly one root is strictly positive. Without loss of generality, fix $t _ { + } > 0$ and $t _ { - } < 0$ . Since $\tau \subseteq ( 0 , \infty )$ , the only admissible critical point of Ψ is $t _ { + }$ . Consequently Ψ has exactly one critical point on $( 0 , \infty )$ , is strictly monotone on each of $( 0 , t _ { + } )$ and $( t _ { + } , \infty )$ , and hence attains any given value at most twice.

Because $t = e ^ { v y }$ is a bijection from R onto $( 0 , \infty )$ , the points of Y map to distinct points of T. But (73) transported to t makes Ψ constant on $\tau$ , and since $| \mathscr { y } | \geq 3$ there exist three pairwise distinct $\tau _ { 1 } , \tau _ { 2 } , \tau _ { 3 } \in \mathcal { T }$ at which Ψ attains the common value $\alpha _ { s , s - 1 } \beta _ { 2 , 1 } - \alpha _ { 2 , 1 } \beta _ { s , s - }$ <sub>1</sub>, contradicting the preceding paragraph.

The contradiction establishes that the afine representations (70) and (71) cannot both hold. By Lemma 1 the two are equivalent to (67), a necessary consequence of the equality of conditionals and in turn of the equality of joints. Hence no $p _ { X , Y } \in \mathcal { M } _ { X  Y }$ and $q _ { X , Y } \in$ $\mathcal { M } _ { Y  X }$ coincide on $\mathcal { X } \times \mathcal { V } ;$ ; the two classes induce disjoint sets of joint distributions, and by Definition 2 the edge direction is distributionally identifiable.

Figure 3 gives a visual summary of the proof. Panel (a) presents the obstruction the afine representations (70) face, a strictly concave $\bar { B }$ meeting any straight line at no more than two points, and panel (b) presents the same obstruction after the elimination, Ψ strictly curved yet held constant on $\mathcal { V }$ by (73), which the three points supplied by $| \mathscr { y } | \geq 3$ contradict. □

Second proof, by Proposition 1. We again argue by contradiction, showing this time that the forward model enforces the conditional distribution of X given Y to be a oneparameter exponential family in X, whereas the reverse model, by Proposition 1, does not. Assume that the models are not identifiable. Then, from Definition 2, there exist joint distributions $p _ { X , Y } \in \mathcal { M } _ { X  Y }$ and $q _ { X , Y } \in \mathcal { M } _ { Y  X }$ such that

$$
p _ { X , Y } \left( x , y \right) = q _ { X , Y } \left( x , y \right) , \quad \forall x \in \mathcal { X } , \ y \in \mathcal { Y } .\tag{79}
$$

Summing over $x \in \mathcal { X }$ , the marginals of $Y$ under the two models satisfy

$$
\begin{array} { r l } & { \quad p _ { X , Y } \left( x , y \right) = q _ { X , Y } \left( x , y \right) , \quad \forall x \in \mathcal { X } , y \in \mathcal { Y } } \\ & { \implies \displaystyle \sum _ { x ^ { \prime } \in \mathcal { X } } p _ { X , Y } \left( x ^ { \prime } , y \right) = \sum _ { x ^ { \prime } \in \mathcal { X } } q _ { X , Y } \left( x ^ { \prime } , y \right) , \quad \forall y \in \mathcal { Y } } \\ & { \implies p _ { Y } \left( y \right) = q _ { Y } \left( y \right) , \quad \forall y \in \mathcal { Y } . } \end{array}\tag{80}
$$

The marginals of Y thus coincide, and since $p _ { Y } \left( y \right) > 0$ for all $y \in \mathcal { V }$ by (54), so too is $q _ { Y } \left( y \right) > 0$ . Combining with (79), the product rule gives

$$
p _ { X | Y } ( x \mid y ; \mathcal { M } _ { X  Y } ) = p _ { X | Y } ( x \mid y ; \mathcal { M } _ { Y  X } ) , \quad \forall x \in \mathcal { X } , \ y \in \mathcal { Y } .\tag{81}
$$

We now compute the left-hand side of (81). Applying Bayes’ rule to the marginal of (13) and the $Y \ | \ X$ conditional given in (14), further canceling the base measure $H \left( y \right)$ ， which is strictly positive and common to numerator and denominator, yields

$$
p _ { X | Y } ( x \mid y ; \mathcal { M } _ { X  Y } ) = \frac { \pi _ { x } \exp ( \eta ( w x ) T ( y ) - a ( \eta ( w x ) ) ) } { \sum _ { x ^ { \prime } \in \mathcal { X } } \pi _ { x ^ { \prime } } \exp ( \eta ( w x ^ { \prime } ) T ( y ) - a ( \eta ( w x ^ { \prime } ) ) ) } , \quad \forall x \in \mathcal { X } , y \in \mathcal { Y } .\tag{82}
$$

Define the quantities

$$
h _ { x } \triangleq \pi _ { x } e ^ { - a \left( \eta \left( w x \right) \right) } , \quad \tilde { T } _ { x } \triangleq \eta \left( w x \right) , \quad \theta \left( y \right) \triangleq T \left( y \right) , \quad A \left( \theta \right) \triangleq \log \sum _ { x ^ { \prime } \in \mathcal { X } } h _ { x ^ { \prime } } e ^ { \theta \tilde { T } _ { x ^ { \prime } } } ,\tag{83}
$$

where $h _ { x } > 0$ for every $x \in \mathcal { X }$ , since $\pi _ { x } > 0$ by (13) and the log-partition function a is finite on the natural parameter space. Substituting (83) into (82) yields

$$
\begin{array} { r l } & { \qquad p _ { X | Y } \left( x \mid y ; \mathcal { M } _ { X \to Y } \right) = \frac { h _ { x } e ^ { \theta \left( y \right) \tilde { T } _ { x } } } { \sum _ { x ^ { \prime } \in \mathcal { X } } h _ { x ^ { \prime } } e ^ { \theta \left( y \right) \tilde { T } _ { x ^ { \prime } } } } , \quad \forall x \in \mathcal { X } , \ y \in \mathcal { Y } } \\ { \implies } & { p _ { X | Y } \left( x \mid y ; \mathcal { M } _ { X \to Y } \right) = h _ { x } \exp \left( \theta \left( y \right) \tilde { T } _ { x } - A \left( \theta \left( y \right) \right) \right) , \quad \forall x \in \mathcal { X } , \ y \in \mathcal { Y } , } \end{array}\tag{84}
$$

which is a one-parameter exponential family on X indexed by $y ,$ with base measure $\{ h _ { x } \}$ suficient statistic $\left\{ \tilde { T } _ { x } \right\}$ , scalar natural parameter $\theta \left( y \right)$ , and log-partition function A. Equation (84) restates Lemma 1 in a diferent form, since taking log-ratios returns the afine representation (20) with $\alpha _ { u , x } = \tilde { T } _ { u } - \tilde { T } _ { x }$

We next consider the right-hand side of (81), which by (17) is the ordered-logit conditional with edge weight $v \ne 0$ and strictly ordered cutpoints $- \infty = \gamma _ { 0 } < \gamma _ { 1 } < \cdots <$ $\gamma _ { s - 1 } < \gamma _ { s } = + \infty$ , on the support $\mathcal { X } = \{ 1 , \ldots , s \}$ with $s \geq 3$ , indexed by y ranging over Y with $| \mathscr { D } | \geq 3$ . These are the conditions of Proposition 1, under which the conditiona admits no representation of the form (84), for any base measure, suficient statistic, natural parameter, and log-partition function whatsoever.

The two sides of (81) therefore cannot be equal, the left-hand side being a one-parameter exponential family in X indexed by $y$ and the right-hand side admitting no such representation. Hence (81) cannot hold, and being a necessary consequence of (79), renders the assumption (79) untenable. No $p _ { X , Y } \in \mathcal { M } _ { X  Y }$ and $q _ { X , Y } \in \mathcal { M } _ { Y  X }$ coincide on $\mathcal { X } \times \mathcal { V } ;$ the two classes induce disjoint sets of joint distributions, and by Definition 2 the edge direction is distributionally identifiable. □

## A.5 Proof of Proposition 2

Proof Assume, to show contradiction, that the two model classes are not disjoint. Then there exist joint distributions $p _ { X , Y } \in \mathcal { M } _ { X  Y }$ and $q _ { X , Y } \in \mathcal { M } _ { Y  X }$ such that

$$
p _ { X , Y } \left( x , y \right) = q _ { X , Y } \left( x , y \right) , \quad \forall x \in \mathcal { X } , \ y \in \mathcal { Y } .\tag{85}
$$

Since $F$ is strictly increasing and the cutpoints are strictly ordered, every diference $F \left( \gamma _ { x } - v y \right) -$ $F \left( \gamma _ { x - 1 } - v y \right)$ in (24) is strictly positive, so the reverse conditional is strictly positive on X for every $y \in \mathcal { V }$ and the quantities (25) are well defined.

Summing (85) over $x \in \mathcal { X }$ , the marginals of $Y$ under the two models satisfy

$$
\begin{array} { r l } & { \quad p _ { X , Y } \left( x , y \right) = q _ { X , Y } \left( x , y \right) , \quad \forall x \in \mathcal { X } , y \in \mathcal { Y } } \\ & { \implies \displaystyle \sum _ { x ^ { \prime } \in \mathcal { X } } p _ { X , Y } \left( x ^ { \prime } , y \right) = \sum _ { x ^ { \prime } \in \mathcal { X } } q _ { X , Y } \left( x ^ { \prime } , y \right) , \quad \forall y \in \mathcal { Y } } \\ & { \implies p _ { Y } \left( y \right) = q _ { Y } \left( y \right) , \quad \forall y \in \mathcal { Y } . } \end{array}\tag{86}
$$

This common marginal is strictly positive, so dividing (85) by it and applying the product rule gives

$$
p _ { X | Y } \left( x \mid y \right) = q _ { X | Y } \left( x \mid y \right) , \quad \forall x \in \mathcal { X } , y \in \mathcal { Y } ,\tag{87}
$$

and hence, by the definition (19) of the log-odds-ratio, for distinct $u , x \in { \mathcal { X } }$

$$
R ( y , u , x ; \mathcal { M } _ { X  Y } ) = R ( y , u , x ; \mathcal { M } _ { Y  X } ) , \quad \forall y \in \mathcal { V } , \ u \neq x .\tag{88}
$$

From Lemma 1 there exist real $\alpha _ { u , x } = \eta \left( w u \right) - \eta \left( w x \right) \neq 0$ and $\beta _ { u , x } ,$ , independent of $y ,$ with $R ( y , u , x ; \mathcal { M } _ { X  Y } ) = \alpha _ { u , x } T ( y ) + \beta _ { u , x } ,$ , so that (88) forces

$$
R ( y , u , x ; \mathcal { M } _ { Y  X } ) = \alpha _ { u , x } T ( y ) + \beta _ { u , x } , \quad \forall y \in \mathcal { V } , \ u \neq x .\tag{89}
$$

Choosing $u = 2 , x = 1$ and then $u = s , x = s - 1$ , both admissible since $s \geq 3$ , and substituting into (89),

$$
\bar { B } \left( y \right) = \alpha _ { 2 , 1 } T \left( y \right) + \beta _ { 2 , 1 } , \qquad \bar { U } \left( y \right) = \alpha _ { s , s - 1 } T \left( y \right) + \beta _ { s , s - 1 } , \qquad \forall y \in \mathcal { V } .\tag{90}
$$

Define

$$
\Psi \left( y \right) \triangleq \alpha _ { s , s - 1 } \bar { B } \left( y \right) - \alpha _ { 2 , 1 } \bar { U } \left( y \right) , \qquad y \in \mathbb { R } .\tag{91}
$$

Multiplying the first identity in (90) by $\alpha _ { s , s - 1 }$ and the second by $\alpha _ { 2 , 1 }$ and subtracting, the term in $T \left( y \right)$ cancels and only the intercepts remain, so that

$$
\Psi \left( y \right) = \alpha _ { s , s - 1 } \beta _ { 2 , 1 } - \alpha _ { 2 , 1 } \beta _ { s , s - 1 } , \quad \forall y \in \mathcal { V } ,\tag{92}
$$

the right-hand side being independent of $y .$

We next fix the signs of the two slopes. Both $\alpha _ { 2 , 1 } = \eta \left( 2 w \right) - \eta \left( w \right)$ and $\alpha _ { s , s - 1 } =$ $\eta \left( s w \right) - \eta \left( s w - w \right)$ compare the values of $\eta$ at two points separated by $w ,$ the larger argument being taken first in each. Since $\eta$ is strictly monotone and $w \neq 0$ , both diferences are nonzero and carry the same sign, that of w when η is increasing and its opposite when η is decreasing. In either case

$$
\mathrm { s i g n } \left( \alpha _ { 2 , 1 } \right) = \mathrm { s i g n } \left( \alpha _ { s , s - 1 } \right) .\tag{93}
$$

Suppose first that both slopes are positive. By hypothesis $\bar { B }$ is strictly concave on $\mathbb { R } ,$ , so $\alpha _ { s , s - 1 } \bar { B }$ is strictly concave, being a positive multiple of $\mathrm { i t } ;$ and $\bar { U }$ is strictly convex on R, so $- \alpha _ { 2 , 1 } U$ is strictly concave, being a negative multiple of it. Their sum Ψ is therefore strictly concave on R. Suppose instead that both slopes are negative. Then $\alpha _ { s , s - 1 } \bar { B }$ is strictly convex and $- \alpha _ { 2 , 1 } \bar { U }$ is strictly convex, so Ψ is strictly convex on R. By (93) no other case arises, and Ψ is strictly concave or strictly convex on R.

A function of either kind attains any given value at most twice. Suppose $\Psi$ attained the value c at three pairwise distinct points, and assume without loss of generality that y<sub>3</sub> $< y _ { 2 } < y _ { 1 }$ . Setting

$$
\lambda = \frac { y _ { 2 } - y _ { 3 } } { y _ { 1 } - y _ { 3 } } \in \left( 0 , 1 \right) ,\tag{94}
$$

which lies in $( 0 , 1 )$ because $0 < y _ { 2 } - y _ { 3 } < y _ { 1 } - y _ { 3 }$ , the middle point admits the convex representation $y _ { 2 } = \lambda y _ { 1 } + ( 1 - \lambda ) y _ { 3 }$ . Strict concavity would then give

$$
\begin{array} { r } { \Psi \left( y _ { 2 } \right) > \lambda \Psi \left( y _ { 1 } \right) + \left( 1 - \lambda \right) \Psi \left( y _ { 3 } \right) = c , } \end{array}\tag{95}
$$

and strict convexity would yield the reverse strict inequality, each contradicting $\Psi \left( y _ { 2 } \right) = c .$ Since $| y | \geq 3$ , there exist three pairwise distinct points of $\mathcal { V }$ at which, by (92), the function Ψ attains the common value $\alpha _ { s , s - 1 } \beta _ { 2 , 1 } - \alpha _ { 2 , 1 } \beta _ { s , s - 1 }$ . This contradicts the preceding paragraph.

The contradiction establishes that (92) cannot hold, and therefore that the afine representations (90) cannot both hold. By Lemma 1 the two are equivalent to (88), a necessary consequence of (87) and in turn of (85), so the assumption (85) is untenable. No $p _ { X , Y } \in \mathcal { M } _ { X  Y }$ and $q _ { X , Y } \in \mathcal { M } _ { Y  X }$ coincide on $\mathcal { X } \times \mathcal { V }$ . Thus, the two classes induce disjoint sets of joint distributions, and by Definition 2, the edge direction is distributionally identifiable. 厂

## A.6 Proof of Theorem 2

Proof We construct parameters inducing identical joint distributions under both models. Set $\Delta y = y _ { 2 } - y _ { 1 } > 0 , t _ { i } \triangleq e ^ { v y _ { i } }$ for $i \in \{ 1 , 2 \}$ , and $L \triangleq v \Delta y$ . The sign of v is not free: the construction below yields a strictly decreasing sequence whose two-step diferences all equal $w \Delta T$ , which must therefore be positive, and since $\Delta y > 0$ and $\Delta T = c _ { 1 } \Delta y$ , the positivity forces sign $( v ) = \mathrm { s i g n } ( w c _ { 1 } )$ . We take $w c _ { 1 } > 0$ and hence $v > 0$ below, the opposite sign being obtained throughout by reversing the inequalities. For each $x \in \mathcal { X }$ write $l _ { x } \left( y ; \mathcal { M } \right) = R \left( y , x , 1 ; \mathcal { M } \right)$ for the log-odds ratio of category x against category 1 under model M.

Now, for $j = 0 , 1 , \ldots , s$ define,

$$
D _ { j } = \log \left( \frac { e ^ { \gamma _ { j } } + t _ { 2 } } { e ^ { \gamma _ { j } } + t _ { 1 } } \right) , \quad \forall j \in \left\{ 0 , 1 , \ldots , s \right\} ,\tag{96}
$$

where by convention $\gamma _ { 0 } = - \infty$ and $\gamma _ { s } = + \infty$ we would get the boundary values as,

$$
D _ { s } = 0 , \quad D _ { 0 } = \log \left( { \frac { t _ { 2 } } { t _ { 1 } } } \right) = v \Delta y = L .\tag{97}
$$

Now, diferentiating $D _ { j }$ w.r.t. $\gamma _ { j }$ for $j \in \{ 1 , \dots , s - 1 \}$ gives us,

$$
\frac { d D _ { j } } { d \gamma _ { j } } = \frac { \left( t _ { 1 } - t _ { 2 } \right) e ^ { \gamma _ { j } } } { \left( e ^ { \gamma _ { j } } + t _ { 1 } \right) \left( e ^ { \gamma _ { j } } + t _ { 2 } \right) } , \quad \forall j \in \left\{ 1 , \ldots , s - 1 \right\} .\tag{98}
$$

$\mathrm { A s } y _ { 2 } > y _ { 1 }$ is given, $t _ { i }$ is an injective transformation of $y _ { i }$ , giving us $t _ { 2 } > t _ { 1 }$ , and by definition $t _ { i } > 0$ for $i \in \{ 1 , 2 \}$ . Thus, we get,

$$
\frac { d D _ { j } } { d \gamma _ { j } } < 0 , \quad \forall j \in \left\{ 1 , \dots , s - 1 \right\} .\tag{99}
$$

Therefore, $D _ { j }$ is a monotone decreasing function in $\gamma _ { j }$ . As we have $- \infty = \gamma _ { 0 } < \gamma _ { 1 } < \cdot \cdot \cdot <$ $\gamma _ { s - 1 } < \gamma _ { s } = + \infty$ , therefore we should also have,

$$
L = D _ { 0 } > D _ { 1 } > \cdots > D _ { s - 1 } > D _ { s } = 0 .\tag{100}
$$

From Lemma 1, we know that there exist $\alpha _ { x , 1 } = \eta \left( w x \right) - \eta \left( w \right) \neq 0$ for all $x \in \{ 2 , \ldots , s \}$ and $\beta _ { x , 1 }$ independent of y such that,

$$
l _ { x } ( y ; \mathcal { M } _ { X  Y } ) = \alpha _ { x , 1 } T ( y ) + \beta _ { x , 1 } , \quad \forall y \in \mathcal { Y } \mathrm { ~ a n d ~ } x \in \{ 2 , \ldots , s \} .\tag{101}
$$

As $\eta \left( z \right) = z$ is given, therefore, we would also have $\alpha _ { x , 1 } = w \left( x - 1 \right)$ . Defining $\Delta T =$ $T \left( y _ { 2 } \right) - T \left( y _ { 1 } \right)$ , we will have,

$$
l _ { x } ( y _ { 2 } ; \mathcal { M } _ { X  Y } ) - l _ { x } ( y _ { 1 } ; \mathcal { M } _ { X  Y } ) = w ( x - 1 ) \Delta T , \quad \forall x \in \{ 2 , \ldots , s \} .\tag{102}
$$

Now, from the reverse model, we get,

$$
\begin{array} { r l } & { l _ { x } ( y _ { 2 } ; M _ { Y  X } ) - l _ { x } ( y _ { 1 } ; M _ { Y  X } ) } \\ { = } & { R ( y _ { 2 } , x , 1 ; M _ { Y  X } ) - R ( y _ { 1 } , x , 1 ; M _ { Y  X } ) } \\ { = } & { \log ( \frac { \sigma ( \gamma _ { x } - v y _ { 2 } ) - \sigma ( \gamma _ { x - 1 } - v y _ { 2 } ) } { \sigma ( \gamma _ { 1 } - v y _ { 2 } ) - \sigma ( \gamma _ { 0 } - v y _ { 2 } ) } ) - \log ( \frac { \sigma ( \gamma _ { x } - v y _ { 1 } ) - \sigma ( \gamma _ { x - 1 } - v y _ { 1 } ) } { \sigma ( \gamma _ { 1 } - v y _ { 1 } ) - \sigma ( \gamma _ { 0 } - v y _ { 1 } ) } ) } \\ { = } & { \log ( \frac { e ^ { v y _ { 2 } } ( e ^ { - v } - e ^ { \gamma x - 1 } ) ( e ^ { \gamma x } + e ^ { v y _ { 2 } } ) } { ( e ^ { \gamma _ { x } } + e ^ { v y _ { 2 } } ) ( e ^ { \gamma _ { x } } - 1 + e ^ { v y _ { 2 } } ) e ^ { \gamma _ { 1 } } } ) - \log ( \frac { e ^ { v y _ { 1 } } ( e ^ { \gamma x } - e ^ { \gamma x - 1 } ) ( e ^ { \gamma _ { 1 } } + e ^ { v y _ { 1 } } ) } { ( e ^ { \gamma _ { x } } + e ^ { v y _ { 1 } } ) ( e ^ { \gamma _ { x } } - 1 + e ^ { v y _ { 1 } } ) ( e ^ { \gamma _ { x } } ) } ) } \\ { = } &  \log ( \frac { e ^ { \gamma _ { 1 } } + e ^ { v y _ { 2 } } } { e ^ { \gamma _ { 1 } } + e ^ { v y _ { 1 } } } ) + \log ( \frac { e ^ { v y _ { 2 } } } \end{array}\tag{103}
$$

Equating (102) and (103) gives us,

$$
\left( D _ { 1 } + D _ { 0 } \right) - \left( D _ { x } + D _ { x - 1 } \right) = \left( x - 1 \right) w \Delta T , \quad \forall x \in \{ 2 , \ldots , s \} .\tag{104}
$$

Now choose some arbitrary $i \in \{ 0 , \ldots , s - 3 \}$ , the upper limit being forced by the requirement that $x = i + 3$ lie in $\{ 2 , \ldots , s \}$ . Substituting $x = i + 2$ and $x = i + 3$ in (104) gives us,

$$
\left( D _ { 1 } + D _ { 0 } \right) - \left( D _ { i + 2 } + D _ { i + 1 } \right) = ( i + 1 ) w \Delta T ,\tag{105}
$$

$$
\left( D _ { 1 } + D _ { 0 } \right) - \left( D _ { i + 3 } + D _ { i + 2 } \right) = \left( i + 2 \right) w \Delta T .\tag{106}
$$

Subtracting (105) from (106) gives us,

$$
D _ { i + 1 } - D _ { i + 3 } = w \Delta T .\tag{107}
$$

Similarly, substituting $x = 2$ in (104) gives us,

$$
D _ { 0 } - D _ { 2 } = w \Delta T .\tag{108}
$$

Since i was arbitrary in $\{ 0 , \ldots , s - 3 \}$ , relation (107) holds with i+1 ranging over $\{ 1 , \ldots , s - 2 \}$ and (108) supplies the one remaining index. Together they give

$$
D _ { i } - D _ { i + 2 } = w \Delta T , \quad \forall i \in \left\{ 0 , \dots , s - 2 \right\} .\tag{109}
$$

We also know that $D _ { 0 } = L$ and $D _ { s } = 0$ . Therefore, the recursion in (109) decouples into even and odd chains as,

$$
D _ { 2 k } = L - k w \Delta T , \quad D _ { 2 k + 1 } = D _ { 1 } - k w \Delta T , \quad \forall 2 k , 2 k + 1 \in \{ 0 , \dots , s \} .\tag{110}
$$

If s is even, i.e. $s = 2 m$ for some $m \in \{ 2 , 3 , \ldots \}$ , then the condition $D _ { s } = 0$ leads us to,

$$
D _ { 2 m } = 0 \implies L = m w \Delta T ,\tag{111}
$$

leaving $D _ { 1 }$ as a free variable. On the contrary, if s is odd, i.e. $s = 2 m + 1$ for some $m \in \{ 1 , 2 , 3 , \ldots \}$ , then the condition $D _ { s } = 0$ leads us to,

$$
D _ { 2 m + 1 } = 0 \implies D _ { 1 } = m w \Delta T ,\tag{112}
$$

leaving L as a free variable. In order to constitute a valid solution, we need to choose the free variable such that the whole sequence of $D _ { j }$ for all $j \in \{ 0 , \ldots , s \}$ follows the strict ordering given in (100).

For even $s = 2 m$ , using (110) with $L = m w \Delta T$ , the strict ordering in (100) reduces to the two conditions $L - w \Delta T < D _ { 1 }$ and $D _ { 1 } < L$ , which lead to the open interval

$$
D _ { 1 } \in \left( \frac { m - 1 } { m } L , L \right) ,\tag{113}
$$

which is non-empty as $m \geq 2$ . By (96), the first cutpoint $\gamma _ { 1 }$ satisfies

$$
e ^ { \gamma _ { 1 } } = \frac { t _ { 2 } - t _ { 1 } e ^ { D _ { 1 } } } { e ^ { D _ { 1 } } - 1 } ,\tag{114}
$$

and since $D _ { 1 }$ is strictly decreasing in $\gamma _ { 1 }$ , the interval (113) corresponds to an open interval for γ . At the endpoint $D _ { 1 } \to L , ( 1 1 4 )$ gives $e ^ { \gamma _ { 1 } } \to 0$ , that is $\gamma _ { 1 }  - \infty$ , and at the endpoint $\begin{array} { r } { D _ { 1 } \to \frac { m - 1 } { m } L } \end{array}$ , it gives the finite value

$$
e ^ { \gamma _ { 1 } }  \frac { t _ { 2 } e ^ { L / m } - t _ { 1 } e ^ { L } } { e ^ { L } - e ^ { L / m } } ,\tag{115}
$$

which is strictly positive, since $t _ { 2 } = t _ { 1 } e ^ { L }$ makes the numerator $t _ { 1 } e ^ { L } \left( e ^ { L / m } - 1 \right) > 0$ and the denominator $e ^ { L } - e ^ { L / m } > 0$ as $L > L / m$ . Therefore, for even $s ,$ the first cutpoint $\gamma _ { 1 }$ ranges over the non-empty open interval

$$
\gamma _ { 1 } \in \left( - \infty , \log \left( \frac { t _ { 2 } e ^ { L / m } - t _ { 1 } e ^ { L } } { e ^ { L } - e ^ { L / m } } \right) \right) .\tag{116}
$$

For odd $s = 2 m + 1$ , using (110) with $D _ { 1 } = m w \Delta T$ , the strict ordering in (100) reduces to the two conditions mw $\Delta T < L$ and $L < ( m + 1 ) w \Delta T$ , which lead to the open interval

$$
L \in \left( m w \Delta T , \ \left( m + 1 \right) w \Delta T \right) ,\tag{117}
$$

which is non-empty as $w \Delta T > 0$ . Since $L = v \Delta y$ and $\Delta T = c _ { 1 } \Delta y$ , the interval (117) is equivalently an open interval for the reverse edge weight $v ,$

$$
v \in \left( m w c _ { 1 } , \ ( m + 1 ) w c _ { 1 } \right) ,\tag{118}
$$

which is non-empty, its length being wc<sub>1</sub>, positive under the sign convention fixed at the start of the proof.

In either parity, a model parameter ranges over a non-empty open interval, the first cutpoint $\gamma _ { 1 }$ in (116) when s is even and the reverse edge weight v in (118) when s is odd. For every admissible value of this parameter, (110) determines all $D _ { j }$ , and (96) returns strictly ordered cutpoints $\gamma _ { j }$ through $e ^ { \gamma _ { j } } = \left( t _ { 2 } - t _ { 1 } e ^ { D _ { j } } \right) / \left( e ^ { D _ { j } } - 1 \right)$ . Setting $\beta _ { x , 1 } =$ $l _ { x } ( y _ { 1 } ; \mathcal { M } _ { Y  X } ) - \alpha _ { x , 1 } T ( y _ { 1 } )$ and applying Lemma 1, the forward marginal is recovered as

$$
\frac { \pi _ { x } } { \pi _ { 1 } } = \exp \left( \beta _ { x , 1 } - a \left( \eta \left( w \right) \right) + a \left( \eta \left( w x \right) \right) \right) > 0 , \quad \forall x \in \left\{ 2 , \ldots , s \right\} ,\tag{119}
$$

which, after normalization, yields a valid categorical marginal. By construction, these forward parameters $( \pi , w )$ with $w \ne 0$ and reverse parameters $( \gamma , v )$ with $v \neq 0$ produce equal increments (102) and (103), and since the log-odds ratios at $y _ { 1 }$ are matched through the choice of $\beta _ { x , 1 }$ , the two models induce identical conditional laws $p _ { X | Y }$ on $\mathcal { V }$ . In the reverse model the marginal of $Y$ is an unconstrained positive distribution on ${ \mathcal { V } } ,$ specified independently of the cutpoints $\gamma$ and the edge weight $v ,$ so we set it equal to the marginal induced by the forward model,

$$
p _ { Y } ( y ; \mathcal { M } _ { Y  X } ) = \sum _ { x ^ { \prime } \in \mathcal { X } } p _ { X , Y } ( x ^ { \prime } , y ; \mathcal { M } _ { X  Y } ) , \quad \forall y \in \mathcal { Y } .\tag{120}
$$

With the conditionals and the marginals of Y now equal, the joint distributions coincide on $\mathcal { X } \times \mathcal { V }$ . As the admissible parameter ranges over a non-empty open interval, the constructed pairs form a family of forward and reverse parameters inducing the same joint law, and hence $\mathcal { M } _ { X  Y }$ and $\mathcal { M } _ { Y  X }$ are not distributionally identifiable.

## A.7 Proof of Theorem 3

Proof We prove both directions of the equivalence. As it is given that $| \mathcal { X } | = 2$ , the reverse model has a single finite cutpoint $\gamma _ { 1 }$ . For each $y \in \mathcal { V }$ the only distinct pair $u , x \in { \mathcal { X } }$ is $u = 2 , x = 1$ , so the only log-odds-ratio is $R \left( y , 2 , 1 ; \mathcal { M } \right)$ for $\mathcal { M } \in \{ \mathcal { M } _ { X  Y } , \mathcal { M } _ { Y  X } \}$ . For brevity, we define $l \left( y ; \mathcal { M } \right) \triangleq R \left( y , 2 , 1 ; \mathcal { M } \right)$

Unlike the case $s \geq 3$ , here the reverse log-odds-ratio $l ( y ; \mathcal { M } _ { Y  X } )$ is afine in y on $\mathcal { V } .$ Indeed,

$$
\begin{array} { r c l } { l ( y ; \mathcal { M } _ { Y  X } ) } & { = } & { \displaystyle \log ( \frac { 1 - \sigma ( \gamma _ { 1 } - v y ) } { \sigma ( \gamma _ { 1 } - v y ) } ) , \quad \forall y \in \mathcal { V } , } \\ & { = } & { \displaystyle \log ( e ^ { v y - \gamma _ { 1 } } ) , \quad \forall y \in \mathcal { V } , } \\ & { = } & { v y - \gamma _ { 1 } , \quad \forall y \in \mathcal { V } . } \end{array}\tag{121}
$$

From Lemma 1, for the forward model there exist $\alpha _ { 2 , 1 } \neq 0$ and $\beta _ { 2 , 1 }$ independent of y such that

$$
l ( y ; \mathcal { M } _ { X  Y } ) = \alpha _ { 2 , 1 } T ( y ) + \beta _ { 2 , 1 } , \quad \forall y \in \mathcal { V } .\tag{122}
$$

We first prove that if $T \left( y \right)$ is afine on the support, then the models are not distributionally identifiable. Suppose $T \left( y \right) = c _ { 1 } y + c _ { 0 }$ for all $y \in \mathcal { V }$ with $c _ { 1 } \neq 0$ . Given any forward model with parameters $( \pi , w )$ , substituting $T \left( y \right) = c _ { 1 } y + c _ { 0 }$ into (122) gives

$$
l ( y ; \mathcal { M } _ { X  Y } ) = \alpha _ { 2 , 1 } c _ { 1 } y + ( \alpha _ { 2 , 1 } c _ { 0 } + \beta _ { 2 , 1 } ) , \quad \forall y \in \mathcal { V } ,\tag{123}
$$

which is afine in $y .$ . Choosing the reverse parameters as

$$
v = \alpha _ { 2 , 1 } c _ { 1 } \neq 0 , \qquad \gamma _ { 1 } = - \left( \alpha _ { 2 , 1 } c _ { 0 } + \beta _ { 2 , 1 } \right) ,\tag{124}
$$

where $v \neq 0$ since $\alpha _ { 2 , 1 } \neq 0$ and $c _ { 1 } \neq 0$ , the reverse log-odds-ratio in (121) becomes

$$
l ( y ; \mathcal { M } _ { Y  X } ) = \alpha _ { 2 , 1 } c _ { 1 } y + ( \alpha _ { 2 , 1 } c _ { 0 } + \beta _ { 2 , 1 } ) = l ( y ; \mathcal { M } _ { X  Y } ) , \quad \forall y \in \mathcal { V } ,\tag{125}
$$

so the two models induce identical conditional laws $p _ { X | Y }$ on $\mathcal { V } .$ . It remains to match the marginals of $Y .$ . In the reverse model, the marginal of $Y$ is an unconstrained positive distribution on $\mathcal { V } .$ , specified independently of $\gamma _ { 1 }$ and $v ,$ so we may set it equal to the marginal of $Y$ induced by the forward model,

$$
p _ { Y } ( y ; \mathcal { M } _ { Y  X } ) = \sum _ { x ^ { \prime } \in \mathcal { X } } p _ { X , Y } ( x ^ { \prime } , y ; \mathcal { M } _ { X  Y } ) , \quad \forall y \in \mathcal { Y } .\tag{126}
$$

With the conditionals and the marginals of $Y$ now equal, the joint distributions coincide on $\mathcal { X } \times \mathcal { V } .$ , and hence the models are not distributionally identifiable.

We next prove that if $T \left( y \right)$ is not afine on $\mathcal { V } ,$ , then the models are distributionally identifiable. By non-afineness on the support, there exist pairwise distinct $y _ { 1 } , y _ { 2 } , y _ { 3 } \in \mathcal { V }$ which is possible since $| \mathscr { y } | \geq 3$ , such that

$$
{ \frac { T \left( y _ { 2 } \right) - T \left( y _ { 1 } \right) } { y _ { 2 } - y _ { 1 } } } \neq { \frac { T \left( y _ { 3 } \right) - T \left( y _ { 1 } \right) } { y _ { 3 } - y _ { 1 } } } .\tag{127}
$$

Suppose, for contradiction, that the models are not distributionally identifiable. Then there exist parameters inducing the same joint distribution for both models. Summing over X gives equal marginals of $Y$ , and dividing gives equal conditionals, hence equal log-oddsratios,

$$
l ( y ; \mathcal { M } _ { X  Y } ) = l ( y ; \mathcal { M } _ { Y  X } ) , \quad \forall y \in \mathcal { Y } .\tag{128}
$$

Substituting (121) and (122),

$$
\alpha _ { 2 , 1 } T \left( y \right) + \beta _ { 2 , 1 } = v y - \gamma _ { 1 } , \quad \forall y \in \mathcal { V } .\tag{129}
$$

Evaluating (129) at $y _ { 1 } , y _ { 2 } , y _ { 3 }$ and taking diferences,

$$
\alpha _ { 2 , 1 } \left( T \left( y _ { 2 } \right) - T \left( y _ { 1 } \right) \right) = \left( y _ { 2 } - y _ { 1 } \right) v , \quad \alpha _ { 2 , 1 } \left( T \left( y _ { 3 } \right) - T \left( y _ { 1 } \right) \right) = \left( y _ { 3 } - y _ { 1 } \right) v ,\tag{130}
$$

and since $y _ { 2 } \neq y _ { 1 } , y _ { 3 } \neq y _ { 1 }$ , and $\alpha _ { 2 , 1 } \neq 0$ , dividing gives

$$
\frac { T \left( y _ { 2 } \right) - T \left( y _ { 1 } \right) } { y _ { 2 } - y _ { 1 } } = \frac { v } { \alpha _ { 2 , 1 } } = \frac { T \left( y _ { 3 } \right) - T \left( y _ { 1 } \right) } { y _ { 3 } - y _ { 1 } } .\tag{131}
$$

This contradicts (127). The contradiction arises solely from the assumption that a common joint distribution exists, so no forward and reverse parameters can induce the same joint distribution on $\mathcal { X } \times \mathcal { V }$ . Equivalently, the model classes $\mathcal { M } _ { X  Y }$ and $\mathcal { M } _ { Y  X }$ induce disjoint sets of joint distributions, and by Definition 2 the edge direction is distributionally identifiable.

## A.8 Proof of Theorem 4

Proof

![](images/a85689256e829063689bdef0fa1b77345d81db863c451e31c6132385d61dbc5a.jpg)  
Figure 4: The reduction behind Theorem 4.

Suppose, for contradiction, that the direction of the edge $\{ i , j \}$ is not distributionally identifiable. Then, from Definition 3, there exist joint distributions $p \nu \in \mathcal { M } ( \mathcal { G } ^ { i  j } )$ and $q \nu \in \mathcal { M } ( \mathcal { G } ^ { j  i } )$ such that

$$
p _ { \mathcal { V } } \left( \mathbf { x } \right) = q _ { \mathcal { V } } \left( \mathbf { x } \right) , \quad \forall \mathbf { x } \in \prod _ { k \in \mathcal { V } } \mathcal { X } _ { k } .\tag{132}
$$

Write ${ \mathcal { Z } } \triangleq \mathcal { V } \setminus \{ i , j \}$ , and define $\mathcal { A } \triangleq \mathcal { P } \left( i \right) \backslash \left\{ j \right\}$ and $B \triangleq { \mathcal { P } } \left( j \right) \backslash \left\{ i \right\}$ . Since $\mathcal { G } ^ { i  j }$ and $\mathcal { G } ^ { j }$ →i difer only in the orientation of $\{ i , j \}$ , the sets $\mathcal { A }$ and $\boldsymbol { B }$ are the same in both graphs, and every $k \in { \mathcal { Z } }$ has the same parent set in both graphs.

Every conditional distribution is determined by the joint distribution, so (132) gives, for each $k \in { \mathcal { Z } }$

$$
p \left( x _ { k } \mid \mathbf { x } _ { \mathcal { P } ( k ) } \right) = q \left( x _ { k } \mid \mathbf { x } _ { \mathcal { P } ( k ) } \right) .\tag{133}
$$

Fix any $\mathbf { x } _ { \mathcal { Z } }$ in the support of $\mathcal { Z }$ , which satisfies $p \left( \mathcal { Z } = \mathbf { x } _ { \mathcal { Z } } \right) > 0$ by the strict positivity of $p \nu$ and write $\mathbf { x } _ { \mathcal { A } }$ and $\mathbf { x } _ { B }$ for its components indexed by A and $\boldsymbol { B }$ . For distinct $u , x \in \{ 1 , \ldots , s \}$ and $y \in \mathcal { X } _ { j }$ , define

$$
R \left( y , u , x \right) \triangleq \log \left( \frac { p \left( X _ { i } = u \mid X _ { j } = y , \mathcal { Z } = \mathbf { x } _ { \mathcal { Z } } \right) } { p \left( X _ { i } = x \mid X _ { j } = y , \mathcal { Z } = \mathbf { x } _ { \mathcal { Z } } \right) } \right) ,\tag{134}
$$

which by (132) takes the same value whether computed from $p \nu$ or from $q _ { \nu }$

Factorizing $p _ { \mathcal { V } }$ according to $\mathcal { G } ^ { i  j }$ , in which $\mathcal { P } \left( i \right) = \mathcal { A }$ and $\mathcal { P } \left( j \right) = \mathcal { B } \cup \{ i \}$ , and canceling the factors that do not depend on $X _ { i }$ , we get

$$
R \left( y , u , x \right) = \log \left( \frac { p \left( u \mid \mathbf { x } _ { \mathcal { A } } \right) } { p \left( x \mid \mathbf { x } _ { \mathcal { A } } \right) } \right) + \log \left( \frac { p \left( y \mid u , \mathbf { x } _ { \mathcal { B } } \right) } { p \left( y \mid x , \mathbf { x } _ { \mathcal { B } } \right) } \right) + \Lambda \left( y , u , x \right) ,\tag{135}
$$

where

$$
\Lambda \left( y , u , x \right) \triangleq \sum _ { k \in \mathcal { Z } } \log \left( \frac { p \left( x _ { k } \mid \mathbf { x } _ { \mathcal { P } \left( k \right) } \right) \big \vert _ { X _ { i } = u } } { p \left( x _ { k } \mid \mathbf { x } _ { \mathcal { P } \left( k \right) } \right) \big \vert _ { X _ { i } = x } } \right)\tag{136}
$$

collects the contributions of the nodes in $\mathcal { Z } .$ , the terms of which vanish for every k with $i \notin \mathcal { P } \left( k \right)$ . Factorizing $q \nu$ according to $\mathcal { G } ^ { j  i }$ , in which $\mathcal { P } \left( i \right) = \mathcal { A } \cup \{ j \}$ and $\mathcal { P } \left( j \right) = B$ , and canceling in the same way,

$$
R \left( y , u , x \right) = \log \left( \frac { q \left( u \mid y , \mathbf { x } _ { A } \right) } { q \left( x \mid y , \mathbf { x } _ { A } \right) } \right) + \Lambda ^ { \prime } \left( y , u , x \right) ,\tag{137}
$$

where $\Lambda ^ { \prime }$ is defined as Λ with $p$ replaced by $q .$ . By (133) the two sums coincide, $\Lambda = \Lambda ^ { \prime }$ , so equating (135) and (137) gives

$$
\log \left( \frac { p \left( u \mid \mathbf { x } _ { A } \right) } { p \left( x \mid \mathbf { x } _ { A } \right) } \right) + \log \left( \frac { p \left( y \mid u , \mathbf { x } _ { B } \right) } { p \left( y \mid x , \mathbf { x } _ { B } \right) } \right) = \log \left( \frac { q \left( u \mid y , \mathbf { x } _ { A } \right) } { q \left( x \mid y , \mathbf { x } _ { A } \right) } \right) , \quad \forall y \in \mathcal { X } _ { j } , \ u \neq x .\tag{138}
$$

We now exhibit two bivariate models to which (138) applies, the reduction being the one drawn in Figure 4. Let $g _ { B }$ denote the contribution of $\mathbf { x } _ { B }$ to the linear predictor of $X _ { j }$ , so that the conditional $p \left( \boldsymbol { y } \mid \boldsymbol { x } , \mathbf { x } _ { B } \right)$ is the regular one-parameter exponential family conditional (14) with suficient statistic $T$ , support $\chi _ { j }$ , and natural parameter $\eta \left( w x + g _ { B } \right)$ Writing $\widetilde { \eta } \left( t \right) \triangleq \eta \left( t + g _ { B } \right)$ , which is monotone injective since $\eta$ is, and taking the categorical marginal $p \left( \cdot \mid \mathbf { x } _ { \mathcal { A } } \right)$ , which is strictly positive, the pair

$$
\tilde { p } \left( x , y \right) \triangleq p \left( x \mid \mathbf { x } _ { \mathcal { A } } \right) p \left( y \mid x , \mathbf { x } _ { \mathcal { B } } \right)\tag{139}
$$

is a forward model of the form $\mathcal { M } _ { X  Y }$ with edge weight $w \neq 0$ , link $\tilde { \eta } ,$ and $s \geq 3$ categories, the support $\chi _ { j }$ of node $j$ playing the role of $\mathcal { V }$ in the bivariate models of Section 3.1. Similarly, let $h _ { \mathbf {  } }$ denote the contribution of $\mathbf { x } _ { \mathcal { A } }$ to the linear predictor of $X _ { i } ,$ so that $q \left( x \mid y , \mathbf { x } _ { \mathcal { A } } \right)$ is the ordered-logit conditional (17) with cutpoints $\gamma _ { 1 } - h _ { { \cal A } } < \cdots < \gamma _ { s - 1 } - h _ { { \cal A } }$ and edge weight $v \neq 0$ , the shift preserving the strict ordering. Taking the marginal of $X _ { j }$ to be the marginal induced by (139), which is admissible since the marginal of $Y$ in $\mathcal { M } _ { Y  X }$ is an unconstrained positive distribution on $\chi _ { j }$ , the pair

$$
\tilde { q } \left( x , y \right) \triangleq \left( \sum _ { x ^ { \prime } \in \left\{ 1 , \ldots , s \right\} } \tilde { p } \left( x ^ { \prime } , y \right) \right) q \left( x \mid y , \mathbf { x } _ { A } \right)\tag{140}
$$

is a reverse model of the form $\mathcal { M } _ { Y  X }$

By (138), the log-odds ratios of $\tilde { p }$ and $\tilde { q }$ against category 1 agree for every $y \in \mathcal { X } _ { j }$ , and a categorical law is determined by its log-odds against a fixed category, so the conditional laws of $X _ { i }$ given $X _ { j }$ coincide under $\tilde { p }$ and ${ \tilde { q } } .$ . The marginals of $X _ { j }$ coincide by the construction in (140), and therefore

$$
\tilde { p } \left( x , y \right) = \tilde { q } \left( x , y \right) , \quad \forall x \in \left\{ 1 , \ldots , s \right\} , \ y \in \mathcal { X } _ { j } .\tag{141}
$$

Since $s \geq 3$ and $| \mathcal { X } _ { j } | \geq 3$ , Theorem 1 states that no forward model of the form $\mathcal { M } _ { X  Y }$ and reverse model of the form $\mathcal { M } _ { Y  X }$ can induce the same joint distribution, which contradicts (141).

The contradiction establishes that the bivariate models (139) and (140) cannot coincide, and therefore that (138) cannot hold. Since (138) was obtained from the two factorizations (135) and (137) of the log-odds ratio (134), which are necessary consequences of the equality of joints (132), the assumption (132) is untenable. We conclude that no $p \nu \in \mathcal { M } ( \mathcal { G } ^ { i  j } )$ and $q _ { \mathcal { V } } \in \mathcal { M } ( \mathcal { G } ^ { j  i } )$ can coincide on $\Pi _ { k \in \mathcal { V } } \mathcal { X } _ { k }$ . Equivalently, the two orientations induce disjoint sets of joint distributions, and by Definition 3 the direction of the edge $\{ i , j \}$ is distributionally identifiable.

## A.9 Proof of Theorem 5

Proof We proceed by contradiction, reducing any competing orientation to a bivariate comparison governed by Theorem 1. Densities are taken throughout with respect to the product of the counting measure on the ordinal coordinates and the dominating measure of the relevant exponential family on the remaining coordinates, and every conditional in the model is strictly positive on its support, so $p \nu$ is strictly positive.

We first record that $p _ { \mathcal { V } }$ is causally minimal with respect to $\mathcal { G } .$ . By Proposition 4 of Peters et al. (2014), causal minimality with respect to a graph H is equivalent to

$$
\begin{array} { r } { X _ { k } \downarrow \downarrow X _ { l } \mid \mathbf { x } _ { \mathcal { P } _ { \mathcal { H } } \left( k \right) \setminus \left\{ l \right\} } , \quad \forall k \in \mathcal { V } , l \in \mathcal { P } _ { \mathcal { H } } \left( k \right) . } \end{array}\tag{142}
$$

Fix such a k and l and hold the remaining parents at any value. Since the weight of the edge from l into k is nonzero, two distinct values of $X _ { l }$ give two distinct values of the linear predictor of $X _ { k }$ . These index distinct conditional laws, by injectivity of $\eta$ together with minimality of $T _ { k }$ when $k$ is an exponential family node, and by strict monotonicity of $\sigma$ when k is ordinal. Both parent configurations carry positive probability, so (142) holds.

Assume, for contradiction, that some $\mathcal { G } ^ { \prime } \in \mathcal { O } \left( \mathcal { G } _ { u } \right)$ with $\mathcal { G } ^ { \prime } \neq \mathcal { G }$ carries an ordinalexponential LPM generating $p \nu$ . Then $p \nu$ is strictly positive and is Markov and causally minimal with respect to both $\mathcal { G }$ and $\mathcal { G } ^ { \prime }$ , which are the hypotheses of part (i) of Proposition 29 of Peters et al. (2014). Two distinct orientations must disagree on some edge, but an arbitrary disagreeing edge need not admit a conditioning set lying outside the descendants of its endpoints in both graphs at once, which is what the reduction below requires. The proposition produces one that does, supplying nodes a and $b ,$ with $b \to a$ in $\mathcal { G }$ and $a \to b$ in $\mathcal { G } ^ { \prime }$ , together with the sets

$$
\begin{array} { r } { A \triangleq \mathcal { P } _ { \mathcal { G } } \left( a \right) \backslash \left\{ b \right\} , \qquad B \triangleq \mathcal { P } _ { \mathcal { G } ^ { \prime } } \left( b \right) \backslash \left\{ a \right\} , \qquad \mathcal { C } \triangleq A \cup \mathcal { B } , } \end{array}\tag{143}
$$

every member of $\mathcal { C }$ being a non-descendant of a in $\mathcal { G }$ other than b and a non-descendant of $b$ in $\mathcal { G } ^ { \prime }$ other than a. Since $\mathcal { G }$ and $\mathcal { G } ^ { \prime }$ orient the common skeleton, $( a , b ) \in \mathcal { E } _ { u }$ , so exactly one of $a$ and b is ordinal and the other an exponential family node.

Fix any $\mathbf { x } _ { \mathcal { C } }$ in the support of ${ \mathcal { C } } ,$ which carries positive probability. By the non-descendant property just recorded, every member of $\mathcal { C } \setminus \mathcal { A }$ is a non-descendant of $a$ in $\mathcal { G }$ that is not a parent of a, so the local Markov property renders $X _ { a }$ independent of them given its parents, and $p \left( x _ { a } \mid x _ { b } , \mathbf { x } _ { \mathcal { C } } \right) = p \left( x _ { a } \mid x _ { b } , \mathbf { x } _ { \mathcal { A } } \right)$ is the parametric conditional of $X _ { a }$ given its parents in ${ \mathcal { G } } .$ Hence

$$
p \left( x _ { a } , x _ { b } \mid \mathbf { x } _ { \mathcal { C } } \right) = p \left( x _ { b } \mid \mathbf { x } _ { \mathcal { C } } \right) p \left( x _ { a } \mid x _ { b } , \mathbf { x } _ { \mathcal { A } } \right) ,\tag{144}
$$

and the same argument applied to $\mathcal { G } ^ { \prime }$ , using $\mathcal { P } _ { \mathcal { G } ^ { \prime } } \left( b \right) = \mathcal { B } \cup \{ a \}$ , gives

$$
q \left( x _ { a } , x _ { b } \mid \mathbf { x } _ { \mathcal { C } } \right) = q \left( x _ { a } \mid \mathbf { x } _ { \mathcal { C } } \right) q \left( x _ { b } \mid x _ { a } , \mathbf { x } _ { \mathcal { B } } \right) .\tag{145}
$$

Suppose b is ordinal and a an exponential family node. In (144) the factor $p \left( \boldsymbol { x } _ { b } \mid \mathbf { x } _ { \mathcal { C } } \right)$ is a strictly positive categorical law, and $p \left( x _ { a } \mid x _ { b } , \mathbf { x } _ { \mathcal { A } } \right)$ is the conditional (14) with natural parameter $\eta \left( w x _ { b } + g _ { \mathcal { A } } \right)$ , where $g _ { \mathcal { A } }$ collects the contribution of $\mathbf { x } _ { \mathbf {  } }$ . Writing $\widetilde { \eta } \left( t \right) \triangleq \eta \left( t + g _ { \mathcal { A } } \right)$ monotone injective since η is, (144) is a forward model of the form $\mathcal { M } _ { X  Y }$ with edge weight $w \ne 0$ . In (145) the factor $q \left( x _ { a } \mid \mathbf { x } _ { \mathcal { C } } \right)$ is a strictly positive law on $\mathcal { X } _ { a } .$ , and $q \left( x _ { b } \mid x _ { a } , \mathbf { x } _ { B } \right)$ is the ordered-logit conditional (17) with cutpoints displaced by the contribution of $_ { \mathbf { x } _ { B } }$ , which preserves their strict ordering, so (145) is a reverse model of the form $\mathcal { M } _ { Y  X }$ with edge weight $v \neq 0$ . Exchanging the two displays covers the case in which a is ordinal.

The ordinal member of the pair carries at least three categories and the exponential family member at least three points of support, so by Theorem 1 the two model classes induce disjoint sets of joint distributions on $( X _ { a } , X _ { b } )$ . But (144) and (145) are both computed from $p _ { \mathcal { V } }$ and therefore coincide. This contradiction establishes that no such $\mathcal { G } ^ { \prime }$ exists.

## A.10 Proof of Theorem 6

Proof We construct joint distributions on $\nu$ coinciding under the two orientations. Write ${ \mathcal { Z } } \triangleq \mathcal { V } \setminus \{ i , j \}$ . Since neither i nor $j$ is adjacent in $\mathcal { G } _ { u }$ to any node other than the other endpoint, both have no parents and no children in $\mathcal { Z }$ in either orientation, and every $k \in { \mathcal { Z } }$

has the same parent set, contained in $\mathcal { Z }$ , in $\mathcal { G } ^ { i  j }$ and $\mathcal { G } ^ { j  i }$ . Factorizing according to each graph therefore gives

$$
\begin{array} { r } { p \nu \left( \mathbf { x } \right) = p _ { i , j } \left( x , y \right) \displaystyle \prod _ { k \in \mathcal { Z } } p \left( x _ { k } \mid \mathbf { x } _ { \mathcal { P } \left( k \right) } \right) , } \\ { q _ { \mathcal { V } } \left( \mathbf { x } \right) = q _ { i , j } \left( x , y \right) \displaystyle \prod _ { k \in \mathcal { Z } } q \left( x _ { k } \mid \mathbf { x } _ { \mathcal { P } \left( k \right) } \right) , } \end{array}\tag{146}
$$

so that the joint distribution over V factorizes into the law of the pair $( X _ { i } , X _ { j } )$ and a factor depending only on $\mathcal { Z }$

In $\mathcal { G } ^ { i  j }$ the node i is a root and therefore follows an arbitrary strictly positive categorical marginal on $\{ 1 , \ldots , s \}$ , and $X _ { j }$ follows the regular one-parameter exponential family conditional (14) given $X _ { i }$ with suficient statistic $T$ , link $\eta ,$ and support $\chi _ { j }$ . Hence $p _ { i , j }$ ranges exactly over the forward models $\mathcal { M } _ { X  Y }$ with edge weight $w \ne 0$ . In $\mathcal { G } ^ { j  i }$ , the node $j$ is a root and therefore follows an arbitrary strictly positive marginal on $\chi _ { j }$ , and $X _ { i }$ follows the ordered-logit conditional (17) given $X _ { j }$ with cutpoints $\gamma _ { 1 } < \dots < \gamma _ { s - 1 }$ . Hence $q _ { i , j }$ ranges exactly over the reverse models $\mathcal { M } _ { Y  X }$ with edge weight $v \neq 0$

Consider first the case $| { \mathcal { X } } _ { j } | = 2$ with an afine suficient statistic and a canonical link. By Theorem 2 there exist forward parameters $( \pi , w )$ with w $\neq 0$ and reverse parameters $( \gamma , v )$ with $v \neq 0$ such that

$$
p _ { i , j } \left( x , y \right) = q _ { i , j } \left( x , y \right) , \quad \forall x \in \left\{ 1 , \ldots , s \right\} , y \in \mathcal { X } _ { j } .\tag{147}
$$

Consider next the case $s = 2$ with a suficient statistic afine on $\chi _ { j }$ . By Theorem 3 there exist forward parameters $( \pi , w )$ with $w \ne 0$ and reverse parameters $( \gamma _ { 1 } , v )$ with $v \ne 0$ satisfying (147). In either case the law of the pair coincides under the two orientations.

Choosing the conditionals of the nodes in $\mathcal { Z }$ to be identical under the two graphs, which is admissible since their parent sets coincide, the second factors in (146) agree, and combining this with (147) gives

$$
p _ { \mathcal { V } } \left( \mathbf { x } \right) = q _ { \mathcal { V } } \left( \mathbf { x } \right) , \quad \forall \mathbf { x } \in \prod _ { k \in \mathcal { V } } \mathcal { X } _ { k } .\tag{148}
$$

The two orientations, therefore, induce joint distributions that coincide on $\Pi _ { k \in \mathcal { V } } \mathcal { X } _ { k }$ , so their sets of induced joint distributions are not disjoint, and by Definition 3 the direction of the edge $( i , j )$ is not distributionally identifiable.

## Appendix B. Algorithms for Mixed DAG Discovery

## B.1 Score and Optimization Problem

Let $\mathbf { X } \in \mathbb { R } ^ { n \times d }$ be an n-sample dataset generated by a d-node DAG $\mathcal { G } = ( \nu , \mathcal { E } )$ with $\nu =$ $\mathcal { V } _ { \mathrm { o r d } } \cup \mathcal { V } _ { \mathrm { e x p } }$ . Edges are permitted only between nodes of unlike type, enforced by the mask

$$
[ \mathbf { M } ] _ { i , j } = \left\{ { 0 , \mathrm { i f } } i , j \in \mathcal { V } _ { \mathrm { o r d } } { \mathrm { o r } } i , j \in \mathcal { V } _ { \mathrm { e x p } } , \right.\tag{149}
$$

so $\mathcal { G }$ is bipartite by construction. Root nodes of a candidate structure are fitted rather than fixed: a root ordinal node retains its $s _ { i } - 1$ cutpoints, giving an unconstrained categorical marginal on $\{ 1 , \ldots , s _ { i } \}$ , and a root exponential-family node has an empty linear predictor with its natural parameter fitted as a single free scalar. Both conventions are narrower than the arbitrary positive marginal of Section 2, so restrict rather than enlarge the model class. Let Let $\mathcal { P } _ { \mathcal { G } _ { \mathrm { e s t } } } \left( i \right)$ be the parent set of node i under a candidate be the parent set of node i under a candidate $\mathcal { G } _ { \mathrm { e s t } }$ with weighted adjacency with weighted adiacency matrix $\mathbf { W } _ { \mathrm { e s t } }$ , and $k _ { i }$ the number of fitted parameters in its conditional,

$$
k _ { i } = \left\{ \begin{array} { l l } { \operatorname* { m a x } \left\{ \left| \mathcal { P } _ { \mathcal { G } _ { \mathrm { e s t } } } \left( i \right) \right| , 1 \right\} , } & { i \in \mathcal { V } _ { \mathrm { e x p } } , } \\ { \left| \mathcal { P } _ { \mathcal { G } _ { \mathrm { e s t } } } \left( i \right) \right| + s _ { i } - 1 , } & { i \in \mathcal { V } _ { \mathrm { o r d } } . } \end{array} \right.\tag{150}
$$

Candidates are scored by the Bayesian Information Criterion (Chickering, 2002),

$$
\mathrm { B I C } \left( \mathcal { G } _ { \mathrm { e s t } } ; \mathbf { X } \right) \triangleq - \sum _ { j = 1 } ^ { n } \sum _ { i = 1 } ^ { d } \ln p \left( \mathbf { X } _ { j , i } \mid \mathbf { X } _ { j , \mathcal { P } _ { \mathcal { G } _ { \mathrm { e s t } } } ( i ) } ; \mathbf { W } _ { \mathrm { e s t } } , \gamma _ { i } \right) + \frac { \log n } { 2 } \sum _ { i = 1 } ^ { d } k _ { i } ,\tag{151}
$$

repeating (37). Both terms are sums over nodes, so the score decomposes into node contributions. Writing $\mathcal { G } _ { u } = ( \nu , \mathcal { E } _ { u } )$ for the skeleton, taken as given, and $\mathcal { O } \left( \mathcal { G } _ { u } \right)$ for its acyclic orientations, each respecting (149), the optimization problem is

$$
\mathcal { G } _ { \mathrm { e s t } } ^ { * } = \arg \operatorname* { m i n } _ { \mathcal { G } _ { \mathrm { e s t } } \in \mathcal { O } ( \mathcal { G } _ { u } ) } \mathrm { B I C } \left( \mathcal { G } _ { \mathrm { e s t } } ; \mathbf { X } \right) ,\tag{152}
$$

with $\left( \mathbf { W } _ { \mathrm { e s t } } , \{ \gamma _ { i } \} _ { i \in \mathcal { V } _ { \mathrm { o r d } } } \right)$ obtained by per-node conditional maximum likelihood estimation (Appendix C). The node conditionals available to $\nu _ { \mathrm { e x p } }$ are listed in Table 1.

Table 1: Regular one-parameter exponential families used as node conditionals.
<table><tr><td>Distribution</td><td>Natural Parameter Link Function η</td><td> $\eta = g ( z _ { i } )$ </td><td>Suff. Stat. Support  $T ( x )$ </td><td>X</td></tr><tr><td>Exponential</td><td> $\overline { { - \lambda , \ ( \lambda > 0 ) } }$ </td><td> $\eta = - e ^ { z _ { i } }$ </td><td>x</td><td> $\overline { { [ 0 , \infty ) } }$ </td></tr><tr><td>Poisson</td><td> $\log \lambda , ( \lambda > 0 )$ </td><td> $\eta = z _ { i }$ </td><td>x</td><td> $\{ 0 , 1 , \ldots \}$ </td></tr><tr><td>Gaussian (fixed variance  $\sigma ^ { 2 } )$ </td><td> ${ \frac { \mu } { \sigma ^ { 2 } } } , \ ( \mu \in \mathbb { R } )$ </td><td> $\eta = z _ { i }$ </td><td>x</td><td>R</td></tr><tr><td>Gamma (fixed shape α)</td><td> $\bar { - } \beta , \ ( \beta > 0 )$ </td><td> $\eta = - e ^ { z _ { i } }$ </td><td>x</td><td> $( 0 , \infty )$ </td></tr><tr><td>Binomial (fixed trials  $n \geq 3 )$ </td><td> $\textstyle \log { \frac { p } { 1 - p } } , \ ( p \in ( 0 , 1 ) )$ </td><td> $\eta = z _ { i }$ </td><td>x</td><td> $\{ 0 , \ldots , n \}$ </td></tr><tr><td>Pascal (fixed successes r) Gamma (fixed rate β)</td><td> $\log ( 1 - p ) , ~ ( p \in ( 0 , 1 ) )$   $\alpha - 1 , \ ( \alpha > 0 )$ </td><td> $\eta = - \log ( 1 + e ^ { z _ { i } } )$   $\eta = e ^ { z _ { i } } - 1$ </td><td>x  $\log ( x )$ </td><td> $\{ 0 , 1 , \ldots \}$   $( 0 , \infty )$ </td></tr></table>

## B.2 Search Procedures

A skeleton with $| \mathcal { E } _ { u } |$ edges admits at most $2 ^ { | \mathcal { E } _ { u } | }$ orientations, the acyclic ones constituting $\mathcal { O } \left( \mathcal { G } _ { u } \right)$ . We solve (152) by two procedures.

Exhaustive search. Enumerate $\mathcal { O } \left( \mathcal { G } _ { u } \right)$ , fit each member, and return the minimizer. The global minimizer is recovered exactly, so any deviation from the truth is finite-sample noise rather than optimization error. Feasible up to a few dozen edges.

Greedy reversal search. Start from an arbitrary acyclic orientation and repeatedly reverse the single edge whose reversal most reduces the score, subject to acyclicity, halting when no reversal improves the score. Insertion and deletion do not arise, the skeleton being fixed. By decomposability of (151) a reversal refits only the two endpoints, giving per-iteration cost $O \left( | \mathcal { E } _ { u } | \right)$ . The procedure is a heuristic and may halt where several joint reversals would improve the score. Algorithm 1 states the procedure.

Algorithm 1 Greedy Orientation Search with BIC Score   
Require: Data matrix X, skeleton $\mathcal { G } _ { u } = ( \nu , \mathcal { E } _ { u } )$ , initial acyclic orientation $\mathcal { G } _ { \mathrm { e s t } }$   
Ensure: Estimated orientation $\mathcal { G } _ { \mathrm { e s t } } ^ { * }$   
1: Compute BIC $( \mathcal { G } _ { \mathrm { e s t } } ; \mathbf { X } )$ via (151)   
2: repeat   
3: $\Delta ^ { * }  0 .$ best edge ← None   
4: for each edge $e \in \mathcal { E } _ { u }$ whose reversal in $\mathcal { G } _ { \mathrm { e s t } }$ preserves acyclicity do   
5: Let $\mathcal { G } _ { e }$ be G<sub>est</sub> with e reversed   
6: Compute $\Delta _ { e } \gets \mathrm { B I C } \left( \mathcal { G } _ { e } ; \mathbf { X } \right) - \mathrm { B I C } \left( \mathcal { G } _ { \mathrm { e s t } } ; \mathbf { X } \right)$ by refitting the two endpoints of e   
7: if $\Delta _ { e } < \Delta ^ { * }$ then   
8: $\Delta ^ { * }  \Delta _ { e } ,$ best edge $ e$   
9: end if   
10: end for   
11: if best edge is not None then   
12: Reverse best edge in $\mathcal { G } _ { \mathrm { e s t } }$   
13: end if   
14: until best edge is None   
15: return $\mathcal { G } _ { \mathrm { e s t } } ^ { * }  \mathcal { G } _ { \mathrm { e s t } }$

## Appendix C. Per-Node Conditional MLE

Both algorithms of Appendix B score a candidate DAG by fitting the per-node conditional parameters by maximum likelihood on a fixed parent set, then computing (151). Since the joint negative log-likelihood and the penalty decompose over nodes, the fitting can be performed one node at a time. We describe the two node types separately.

## C.1 Exponential-Family Nodes

For node $i \in \mathcal { V } _ { \mathrm { e x p } }$ with conditional family $p _ { i }$ and known suficient statistic $T _ { i }$ , the conditional MLE problem is

$$
\hat { \mathbf { w } } _ { i } = \arg \operatorname* { m i n } _ { \mathbf { w } \in \mathbb { R } ^ { | \mathcal { P } ( i ) | } } - \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \log p _ { i } \big ( \mathbf { X } _ { j , i } \mid \mathbf { X } _ { j , \mathcal { P } ( i ) } ; \mathbf { w } \big ) ,\tag{153}
$$

where the conditional log-likelihood takes the exponential-family form with natural parameter $\eta _ { i } = g _ { i } \left( \mathbf { w } ^ { \mathsf { T } } \mathbf { X } _ { j , \mathcal { P } ( i ) } \right)$ . The objective in (153) is convex in w for the families of Table 1 under the canonical link, and we solve (153) via L-BFGS-B (Byrd et al., 1995) initialized at ${ \bf w } = { \bf 0 }$ . For families with an exponential link (Poisson, Exponential, Gamma with fixed shape, Gamma with fixed rate), the linear predictor is constrained within an interval keeping $| \mathbf { w } ^ { \mathsf { T } } \mathbf { X } _ { j , \mathcal { P } ( i ) } |$ bounded for the bulk of the sample, preventing numerical overflow in exp during the line search. The procedure terminates when the relative change in the objective falls below $\epsilon _ { \mathrm { f } } = 1 0 ^ { - 8 }$ or after 200 iterations.

## C.2 Ordinal Nodes

For node $i \in \mathcal { V } _ { \mathrm { o r d } }$ with $s _ { i }$ categories, the conditional MLE problem jointly fits the weight vector $\mathbf { w } _ { i }$ and the cutpoints $\gamma _ { i } = ( \gamma _ { i , 1 } , \dots , \gamma _ { i , s _ { i } - 1 } )$ . The conditional negative log-likelihood is

$$
l _ { i } \left( \mathbf { w } _ { i } , \gamma _ { i } ; \mathbf { X } \right) \ = \ - { \frac { 1 } { n } } \sum _ { j = 1 } ^ { n } \log \left[ \sigma \left( \gamma _ { i } , \mathbf { X } _ { j , i } - \mathbf { w } _ { i } ^ { \mathsf { T } } \mathbf { X } _ { j , { \mathcal { P } } ( i ) } \right) - \sigma \left( \gamma _ { i } , \mathbf { X } _ { j , i } - 1 - \mathbf { w } _ { i } ^ { \mathsf { T } } \mathbf { X } _ { j , { \mathcal { P } } ( i ) } \right) \right] ,\tag{154}
$$

where $\sigma$ is the logistic sigmoid and the sentinel values $\gamma _ { i , 0 } = - \infty , \gamma _ { i , s _ { i } } = + \infty$ are used as needed at the boundary categories.

To enforce the strict ordering constraint $\gamma _ { i , 1 } < \ldots < \gamma _ { i , s _ { i } - 1 }$ without a constrained optimizer, we reparametrize the cutpoints in terms of unconstrained parameters $\pmb { \alpha } _ { i } \in \mathbb { R } ^ { s _ { i } - 1 }$ via

$$
\begin{array} { r c l } { \gamma _ { i , 1 } } & { = } & { \alpha _ { i , 1 } , } \\ { \gamma _ { i , k } } & { = } & { \gamma _ { i , k - 1 } + \log \left( 1 + \exp \left( \alpha _ { i , k } \right) \right) , \quad k = 2 , \ldots , s _ { i } - 1 . } \end{array}\tag{155}
$$

The softplus increment log $\left( 1 + \exp \left( \alpha _ { i , k } \right) \right) \ > \ 0$ guarantees strict ordering for any $\alpha _ { i } \ \in$ $\mathbb { R } ^ { s _ { i } - 1 }$ , and the joint parameter vector $( \mathbf { w } _ { i } , \pmb { \alpha } _ { i } ) \in \mathbb { R } ^ { | \mathcal { P } ( i ) | + s _ { i } - 1 }$ is then optimized over an unconstrained domain. We set $\mathbf { w } _ { i } = \mathbf { 0 }$ and ${ \pmb { \alpha } } _ { i }$ to the inverse of the empirical cumulative distribution function of $X _ { i }$ via the logit link, then convert the resulting cutpoint diferences to the softplus parametrization. The joint optimization is carried out via L-BFGS-B with analytical gradients, using the same convergence tolerance as in the exponential-family case.

## Appendix D. Multi-Node Experiments

Setup as in Appendix E.3: $d = 2 0$ bipartite skeletons, greedy reversal search (Algorithm 1) initialized at a uniformly drawn acyclic orientation, sparse $( p = 0 . 1 )$ and dense $( p = 0 . 9 )$ regimes with 10 and 90 edges, eight distribution modes, $B = 1 0 0 0$ trials per configuration. Figure 5 reports $\rho$ against N.

Sparse regime. All eight modes decay across the range, with most trials at $N =$ 500 orienting every edge correctly. The modes separate rather than cluster, Pascal and Gamma (β) lowest and Poisson and the mixed mode highest, spanning about one order of magnitude at $N = 5 0 0$

Dense regime. Gamma (α), Exponential, and Gaussian fall furthest; the mixed mode and Pascal occupy the middle; Poisson, Binomial, and Gamma (β) plateau highest. No mode reaches the floor, and the curves are flat to within trial-to-trial variation over the final portion of the range. The dense model carries 119 free parameters against 45 for the sparse one, giving roughly four observations per parameter at $N = 5 0 0$ ; the search is moreover NP-hard in the large-sample limit for any consistent scoring criterion (Chickering et al., 2004) and tractable only once the graph is sparse (Claassen et al., 2013). Neither consideration bears on the population-level identifiability of Section 3.

![](images/8a00b5c3e63545057253b0e3c7c46fa4691f629d73ebf8b2257b1a520645e6df.jpg)  
(a) Sparse bipartite DAG, d = 20, 10 edges

![](images/947b884ebe3f27bd0307dba9dc5b27d09dd611e9d35d04d80bd0be9f00fb8691.jpg)  
(b) Dense bipartite DAG, d = 20, 90 edges  
Figure 5: Orientation error rate ρ versus sample size N for d = 20 bipartite DAGs, sparse and dense, over B = 1000 trials per configuration.

The mixed mode lies within the envelope of the homogeneous modes in both regimes.

## Appendix E. Experimentation Details

Implementation in Python 3.13.14 with NumPy 2.3.4 and SciPy 1.16.3, executed on a multi-core CPU cluster. Codebase at https://github.com/SamMathelete/Ordinal\_EF.

## E.1 Generative Parameters

Ordinal cutpoints were generated as s − 1 draws from Uniform (−1, 1), sorted, with $\gamma _ { i , 0 } =$ −∞ and $\gamma _ { i , s } = + \infty$ appended; these were drawn once per configuration and held fixed across sample sizes and trials. The link function was ordered logit throughout, and exponentialfamily hyperparameters were fixed across all experiments: Poisson and Exponential had no free hyperparameters, Gaussian used $\sigma ^ { 2 } = 1 . 0$ , Gamma with fixed shape used $\alpha = 2 . 0$ Binomial used $n _ { \mathrm { t r i a l s } } ~ = ~ 5$ , Pascal used $r \ = \ 3$ , and Gamma with fixed rate used $\beta \ =$ 1.0. For root nodes, the linear predictor is identically zero, so a root exponential-family node has natural parameter $\eta _ { i } \left( 0 \right)$ and a root ordinal node has $\operatorname* { P r } \left( X _ { i } \leq x \right) = \sigma \left( \gamma _ { i , x } \right)$ for $x = 1 , \ldots , s - 1 ;$ ; this gives unit rate (Exponential, Gamma with fixed shape), unit mean (Poisson), zero mean (Gaussian), success probability $1 / 2$ (Binomial, Pascal), and unit shape (Gamma with fixed rate), and no root parameter is held at its generating value during estimation. Seeding used a single integer seed of 7, with per-configuration seeds adding a stable hash of the configuration identifiers, and the b-th data replicate with the configuration seed ofset by b.

## E.2 Three-Node Experiments

The skeleton $X _ { 1 } - X _ { 2 } - X _ { 3 }$ was supplied and never estimated, with $X _ { 1 } , X _ { 3 } \in \mathcal { V } _ { \mathrm { o r d } }$ for $s = 4$ and $X _ { 2 } \in \mathcal { V } _ { \mathrm { e x p } }$ . The candidate set $\mathcal { O } ( \mathcal { G } _ { u } ) = \{ \mathcal { G } _ { 1 } , \mathcal { G } _ { 2 } , \mathcal { G } _ { 3 } , X _ { 1 }  X _ { 2 }  X _ { 3 } \}$ was enumerated exhaustively. Edge weights were set to $w = 0 . 6$ , with N ranging over 20 equally spaced values in [5, 500] and $B = 1 0 0 0 0$ trials per configuration. One configuration was run per family of Table 1.

## E.3 Multi-Node Experiments

The dimension was $d = 2 0$ , with $| \nu _ { \mathrm { o r d } } |$ drawn uniformly from $\{ \lfloor d / 2 \rfloor - 1 , \lfloor d / 2 \rfloor , \lfloor d / 2 \rfloor + 1 \}$ and node identities assigned by random permutation. Each admissible bipartite edge was included independently with probability $p \in \{ 0 . 1 , 0 . 9 \}$ , yielding 10 and 90 edges respectively, with one ground-truth graph per regime. Edge weight magnitudes were drawn from Uniform (0.5, 1.0) with uniform signs, and ordinal nodes had $s \in \{ 3 , 4 \}$ drawn uniformly. The type partition, weights, and cutpoints were held fixed per regime, so trials are data replicates from one model. Eight distribution modes were considered: seven homogeneous modes, one per family of Table 1, and one mixed mode in which each $i \in \mathcal { V } _ { \mathrm { e x p } }$ draws uniformly from the same table. N ranged over 20 equally spaced values in [5, 500], with $B = 1 0 0 0$ trials per configuration.

## E.4 Optimizer Settings

Per-node conditional maximum likelihood estimation (Appendix C) used L-BFGS-B, with at most 200 iterations and objective tolerance $\epsilon _ { \mathrm { f } } = 1 0 ^ { - 8 }$ . The greedy search (Algorithm 1) terminated when no reversal improved the score by more than $\epsilon _ { \mathrm { s } } = 1 0 ^ { - 6 }$ , or after $4 \left. \mathcal { E } _ { u } \right. + 1$ reversals, whichever occurred first.

## References

Bryan Andrews, Joseph Ramsey, and Gregory F. Cooper. Learning high-dimensional directed acyclic graphs with mixed data-types. In Proceedings of Machine Learning Research, volume 104, pages 4–21, 05 Aug 2019.

Juraj Bodik and Val´erie Chavez-Demoulin. Identifiability of causal graphs under nonadditive conditionally parametric causal models. Journal of Machine Learning Research, 26(264):1–55, 2025.

Juraj Bodik and Val´erie Chavez-Demoulin. Structural restrictions in local causal discovery: identifying direct causes of a target variable. Biometrika, 113(1):asaf042, 02 2026.

Richard H. Byrd, Peihuang Lu, Jorge Nocedal, and Ciyou Zhu. A limited memory algorithm for bound constrained optimization. SIAM Journal on Scientific Computing, 16(5):1190– 1208, 1995.

Ruichu Cai, Jie Qiao, Kun Zhang, Zhenjie Zhang, and Zhifeng Hao. Causal discovery from discrete data using hidden compact representation. In Advances in Neural Information Processing Systems, volume 31, 2018.

David Maxwell Chickering. Optimal structure identification with greedy search. Journal of Machine Learning Research, 3(Nov):507–554, 2002.

David Maxwell Chickering, David Heckerman, and Christopher Meek. Large-sample learning of Bayesian networks is NP-hard. Journal of Machine Learning Research, 5(Oct): 1287–1330, 2004.

Tom Claassen, Joris M. Mooij, and Tom Heskes. Learning sparse causal models is not NP-hard. In Proceedings of the 29th Conference on Uncertainty in Artificial Intelligence, page 172–181, 2013.

Stacia M. DeSantis, Christos Lazaridis, Yuko Palesch, and Viswanathan Ramakrishnan. Regression analysis of ordinal stroke clinical trial outcomes: An application to the NINDS t-PA trial. International Journal of Stroke, 9(2):226–231, 2014.

Benjamin French and Matthew S. Shotwell. Regression models for ordinal outcomes. JAMA, 328(8):772–773, 08 2022.

Ming Gao, Wai Ming Tai, and Bryon Aragam. Optimal estimation of gaussian DAG models. In Proceedings of The 25th International Conference on Artificial Intelligence and Statistics, pages 8738–8757, 28–30 Mar 2022.

Asish Ghoshal and Jean Honorio. Learning identifiable gaussian bayesian networks in polynomial time and sample complexity. In Advances in Neural Information Processing Systems, volume 30, 2017.

Alain Hauser and Peter B¨uhlmann. Characterization and greedy learning of interventional markov equivalence classes of directed acyclic graphs. Journal of Machine Learning Research, 13(79):2409–2464, 2012.

Daniel N. Hill, Robert Moakler, Alan E. Hubbard, Vadim Tsemekhman, Foster Provost, and Kiril Tsemekhman. Measuring causal impact of online actions via natural experiments: Application to display advertising. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, page 1839–1847, 2015.

Patrik Hoyer, Dominik Janzing, Joris Mooij, Jonas Peters, and Bernhard Sch¨olkopf. Nonlinear causal discovery with additive noise models. In Advances in Neural Information Processing Systems, volume 21, 2008.

Biwei Huang, Kun Zhang, Yizhu Lin, Bernhard Sch¨olkopf, and Clark Glymour. Generalized score functions for causal discovery. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1551–1560, 2018.

Alexander Immer, Christoph Schultheiss, Julia E. Vogt, Bernhard Sch¨olkopf, Peter B¨uhlmann, and Alexander Marx. On the identifiability and estimation of causal locationscale noise models. In International Conference on Machine Learning, 2023.

Amin Jaber, Murat Kocaoglu, Karthikeyan Shanmugam, and Elias Bareinboim. Causal discovery from soft interventions with unknown targets: Characterization and learning. Advances in neural information processing systems, 33:9551–9561, 2020.

Dominik Janzing, Xiaohai Sun, and Bernhard Sch¨olkopf. Distinguishing cause and efect via second order exponential models. arXiv preprint arXiv:0910.5561, 2009.

Christine K. Johnson, Peta L. Hitchens, Tierra Smiley Evans, Tracey Goldstein, Kate Thomas, Andrew Clements, Damien O. Joly, Nathan D. Wolfe, Peter Daszak, William B Karesh, and Jonna A. K. Mazet. Spillover and pandemic properties of zoonotic viruses with high host plasticity. Scientific Reports, 5, 2015.

Christine K. Johnson, Peta L. Hitchens, Pranav S. Pandit, Julie Rushmore, Tierra Smiley Evans, Cristin C. W. Young, and Megan M. Doyle. Global shifts in mammalian population trends reveal key predictors of virus spillover risk. Proceedings of the Royal Society B: Biological Sciences, 287(1924):20192736, 04 2020.

Ilyes Khemakhem, Ricardo Monti, Robert Leech, and Aapo Hyvarinen. Causal autoregressive flows. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, pages 3520–3528, 13–15 Apr 2021.

Daniel Klippert and Alexander Marx. Skewness-robust causal discovery in location-scale noise models. In International Conference on Machine Learning, 2026.

R Lall, M J Campbell, S J Walters, and K Morgan. A review of ordinal regression models applied on health-related quality of life assessments. Statistical Methods in Medical Research, 11(1):49–67, 2002.

Finnian Lattimore, Tor Lattimore, and Mark D Reid. Causal bandits: Learning good interventions via causal inference. In Advances in Neural Information Processing Systems, volume 29, 2016.

Stefen L. Lauritzen and Nanny Wermuth. Graphical models for associations between variables, some of which are qualitative and some quantitative. Annals of Statistics, 17:31–57, 1989.

Valentinian Lungu, Joni Shaska, Ioannis Kontoyiannis, and Urbashi Mitra. Bayesian structure learning and detection in the linear causal model. In 2026 IEEE International Symposium on Information Theory (ISIT), pages 1–6, 2026.

Takashi Nicholas Maeda, Shohei Shimizu, and Hidetoshi Matsui. Density ratio-based causal discovery from bivariate continuous-discrete data. arXiv preprint arXiv:2505.08371, 2025.

Alexander Marx and Jilles Vreeken. Causal inference on multivariate and mixed-type data. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 655–671, 2018.

Peter McCullagh. Regression models for ordinal data. Journal of the Royal Statistical Society: Series B (Methodological), 42(2):109–127, 01 1980.

Sambit Mishra and Urbashi Mitra. Causal discovery in equal variance linear gaussian DAGs via SURE-tuned ridge regression. arXiv preprint arXiv:2608.17132, 2026.

Sambit Mishra, Yingying Wang, Christine K. Johnson, and Urbashi Mitra. Epidemiological causal graph identification: Challenges, identifiability and algorithms. In 60th Asilomar Conference on Signals, Systems, and Computers, 2026. Accepted for publication.

Yang Ni and Bani Mallick. Ordinal causal discovery. In Proceedings of the 38th Conference on Uncertainty in Artificial Intelligence, pages 1530–1540, 01–05 Aug 2022.

Gunwoong Park and Hyewon Park. Identifiability of generalized hypergeometric distribution (GHD) directed acyclic graphical models. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics, pages 158–166, 16–18 Apr 2019.

Gunwoong Park and Garvesh Raskutti. Learning large-scale poisson DAG models based on overdispersion scoring. In Advances in Neural Information Processing Systems, volume 28, 2015.

Judea Pearl. Causality. Cambridge University Press, 2nd edition, 2009.

Chen Peng and Urbashi Mitra. Causal graph identification under soft intervention. In 2025 IEEE International Symposium on Information Theory (ISIT), pages 1–6, 2025.

Chen Peng, Di Zhang, and Urbashi Mitra. Asymmetric graph error control with low complexity in causal bandits. IEEE Transactions on Signal Processing, 73:1792–1807, 2025.

Chen Peng, Sambit Mishra, and Urbashi Mitra. Learning to intervene: Optimized soft intervention selection for causal discovery. In Proceedings of the 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6196–6200, 2026.

J. Peters and P. B¨uhlmann. Identifiability of gaussian structural equation models with equal error variances. Biometrika, 101(1):219–228, 03 2014.

Jonas Peters, Dominik Janzing, and Bernhard Sch¨olkopf. Causal inference on discrete data using additive noise models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(12):2436–2450, 2011.

Jonas Peters, Joris M. Mooij, Dominik Janzing, and Bernhard Sch¨olkopf. Causal discovery with continuous additive noise models. Journal of Machine Learning Research, 15(58): 2009–2053, 2014.

Goutham Rajendran, Bohdan Kivva, Ming Gao, and Bryon Aragam. Structure learning in polynomial time: Greedy algorithms, bregman information, and exponential families. In Advances in Neural Information Processing Systems, volume 34, pages 18660–18672, 2021.

Joseph Ramsey, Peter Spirtes, and Jiji Zhang. Adjacency-faithfulness and conservative causal inference. In Proceedings of the 22nd Conference on Uncertainty in Artificial Intelligence, page 401–408, 2006.

Luqing Ren. Causal modeling for fraud detection: Enhancing financial security with interpretable AI. European Journal of Business, Economics & Management, 1(4):94–104, Oct 2025.

Joni Shaska and Urbashi Mitra. Causal link discovery with unequal edge error tolerance. IEEE Transactions on Signal Processing, 73:2848–2861, 2025.

Joni Shaska, Yingying Wang, Christine K. Johnson, and Urbashi Mitra. Ordinal-Poisson causal discovery. In Proceedings of the 61st Annual Allerton Conference on Communication, Control, and Computing, 2025.

Shohei Shimizu, Patrik O. Hoyer, Aapo Hyv¨arinen, and Antti Kerminen. A linear nongaussian acyclic model for causal discovery. Journal of Machine Learning Research, 7 (72):2003–2030, 2006.

Peter Spirtes, Clark Glymour, and Richard Scheines. Causation, prediction, and search. The MIT press, 2001.

Edgar Steiger, T. Mussgnug, and Lars Eric Kroll. Causal graph analysis of COVID-19 observational data in german districts reveals efects of determining factors on reported case numbers. PLoS ONE, 16, 2021.

Eric V. Strobl and Thomas A. Lasko. Identifying patient-specific root causes with the heteroscedastic noise model. Journal of Computational Science, 72:102099, 2023.

Joe Suzuki, Takanori Inazumi, Takashi Washio, and Shohei Shimizu. Identifiability of an integer modular acyclic additive noise model and its causal structure discovery. arXiv preprint arXiv:1401.5625, 2014.

Christo Kurisummoottil Thomas, Christina Chaccour, Walid Saad, M´erouane Debbah, and Choong Seon Hong. Causal reasoning: Charting a revolutionary course for nextgeneration AI-native wireless networks. IEEE Vehicular Technology Magazine, 19(1): 16–31, 2024.

Sofia Triantafillou, Vincenzo Lagani, Christina Heinze-Deml, Angelika Schmidt, Jesper N. Tegn´er, and Ioannis Tsamardinos. Predicting causal relationships from biological data: Applying automated causal discovery on mass cytometry data of human immune cells. Scientific Reports, 7, 2017.

Burak Varici, Karthikeyan Shanmugam, Prasanna Sattigeri, and Ali Tajer. Causal bandits for linear structural equation models. Journal of Machine Learning Research, 24(297): 1–59, 2023.

Minjie Wang, Xiaotong Shen, and Wei Pan. Causal discovery with generalized linear models through peeling algorithms. Journal of Machine Learning Research, 25(310):1–49, 2024.

Wei Wenjuan, Feng Lu, and Liu Chunchen. Mixed causal structure discovery with application to prescriptive pricing. In Proceedings of the 27th International Joint Conference on Artificial Intelligence, pages 5126–5134, 7 2018.

Richard Williams. Generalized ordered logit/partial proportional odds models for ordinal dependent variables. The Stata Journal, 6(1):58–82, 2006.

Sascha Xu, Osman A Mian, Alexander Marx, and Jilles Vreeken. Inferring cause and efect in the presence of heteroscedastic noise. In Proceedings of the 39th International Conference on Machine Learning, pages 24615–24630, 17–23 Jul 2022.

Yifan Xue, Gregory F. Cooper, Chunhui Cai, Songjian Lu, Baoli Hu, Xiaojun Ma, and Xinghua Lu. Tumour-specific causal inference discovers distinct disease mechanisms underlying cancer subtypes. Scientific Reports, 9, 2019.

Karren Yang, Abigail Katcof, and Caroline Uhler. Characterizing and learning equivalence classes of causal DAGs under interventions. In Proceedings of the 35th International Conference on Machine Learning, pages 5541–5550, 10–15 Jul 2018.

Ruicong Yao, Tim Verdonck, and Jakob Raymaekers. Causal discovery in mixed additive noise models. In Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, pages 3088–3096, 03–05 May 2025.

Yan Zeng, Shohei Shimizu, Hidetoshi Matsui, and Fuchun Sun. Causal discovery for linear mixed data. In Proceedings of the 1st Conference on Causal Learning and Reasoning, pages 994–1009, 11–13 Apr 2022.

Kun Zhang and Aapo Hyv¨arinen. On the identifiability of the post-nonlinear causal model. In Conference on Uncertainty in Artificial Intelligence, 2009.