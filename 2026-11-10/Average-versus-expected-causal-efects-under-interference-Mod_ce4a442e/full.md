# Average versus expected causal efects under interference: Modeling assumptions and estimation bottlenecks

Vishesh Karwa<sup>∗†</sup> Edoardo M. Airoldi<sup>∗†</sup> Donald B. Rubin<sup>∗†‡</sup>

vishesh.karwa@temple.edu

edoardo.airoldi@temple.edu

rubin@stat.harvard.edu

<sup>∗</sup>Department of Statistics, Operations, and Data Science, Temple University

<sup>†</sup>Data Science Institute, Temple University, Philadelphia, PA 19122, USA

<sup>‡</sup>Department of Statistics, Harvard University, Cambridge, MA 02138, USA

## Abstract

Causal inference under treatment interference requires assumptions to reduce the expo nentially large number of potential outcomes. We develop hierarchical finite-population models for potential outcomes by separating these assumptions into three layers: interference neighborhoods, exposure mappings, and outcome models. Given an interference neighborhood and exposure mapping, we show that potential outcomes admit a nonparametric decomposition into direct, spillover, and treatment–spillover interaction components. This decomposition provides a common coordinate system for nonparametric, semiparametric, and parametric models of finite-population potential outcomes. For a given potential outcome model, we define two classes of estimands. Average causal efects (ACEs), such as direct and total treatment efects, contrast potential outcomes under fixed treatment– exposure conditions. Expected causal efects (ECEs), such as expected direct and indirect efects, contrast expectations of potential outcomes under a specified allocation policy. For both classes, we bound the variance of the canonical Horvitz–Thompson estimator by three terms: a diagonal weight term, a term from overlapping interference neighborhoods, and a residual design-dependence term. Our upper and lower bounds show that under graphunaware designs, nonparametric design-unbiased estimation of ACEs can require sample sizes growing exponentially in the interference degree, whereas polynomial growth can suf fice for design-aligned ECEs. Under neighborhood interference, we construct graph-aware independent-set designs with variance bounds polynomial in the maximum degree for direct and total treatment efects on every graph, whereas graph cluster randomization can incur worst-case variance exponential in the degree.

Keywords: causal inference, interference, potential outcomes, network experiments, randomization inference

## Contents

## 1 Introduction

2 Hierarchical models of potential outcomes 6   
2.1 Interference graph . 7   
2.2 Exposure mapping . 8   
2.3 A non-parametric decomposition of the causal response surface 9   
2.4 Outcome model . 10   
2.5 Canonical model classes 11   
3 Defining causal estimands under network interference 13   
3.1 Average causal efects under network interference 13   
3.2 Expected causal efects under network interference 15   
3.3 Algebraic representation of ACEs and ECEs 16   
3.4 Relevant and Nuisance Potential Outcomes 18   
3.5 Conditional expected efects 19   
4 Estimation of ACEs and ECEs 19   
4.1 Dependence from interference and design 20   
4.2 ACE estimation . 22   
4.3 ECE estimation 25   
5 Design-Based Estimation of Average Causal Efects 27   
5.1 Independent set designs for the direct treatment efect 28   
5.2 Independent set designs for the total treatment efect 31   
5.3 Independent set designs for indirect efects 33   
5.4 Cluster randomized designs 36   
6 Implications for Designs, Estimators, and Models 36   
6.1 Classical designs and estimators target expected causal efects . 36   
6.2 Positivity and overlap for average causal efects 37   
6.3 Model-assisted estimation of average causal efects 37   
7 Numerical illustration 37   
8 Discussion 41   
A Additional examples and extensions for Section 2 43   
A.1 Asymmetric and bipartite interference 43   
A.2 Vector-valued and continuous exposures 43   
A.3 Nonlinear and threshold outcome restrictions 43   
A.4 Non-anonymous exposure 44   
B Proofs for Section 2 44   
B.1 Proof of Theorem 1 . 44   
C Proofs and details for Section 3 45   
C.1 Mixed causal efects: EIC, EIT, and ETE 45   
C.2 Proof of Proposition 3 46   
C.3 Proof of Proposition 5 47   
C.4 Proof of Proposition 25 47   
C.5 Proof of Corollary 4 48   
C.6 Proof of Proposition 2 48   
C.7 Symmetric exposure mappings 49   
D Proofs for Section 4 50   
D.1 Proof of Lemma 8 50   
D.2 A generic design-dependence bound 50   
D.3 Residual design dependence under complete randomization 51   
D.4 Proof of Proposition 10 52   
D.5 Proof of Proposition 11 54   
D.6 Proof of Proposition 12 55   
D.7 Weighted dependence bound for graph-aware ECEs . 56   
D.8 Proof of Proposition 29 57   
D.9 Proof of Proposition 13 58   
D.10 Proof of Corollary 14 . 59   
D.11 Heterogeneous ACE/ECE comparison under classical designs 60   
D.12 Proof of Corollary 30 . 61   
D.13 Auxiliary calculation: DTE and TTE table rates 62   
D.14 Auxiliary calculation: exposure compression and propensity scores 64   
D.15 Auxiliary calculation: outcome-model span condition 65   
E Proofs for Section 5 65   
E.1 Proof of Proposition 15 65   
E.2 Proof of Lemma 16 . 65   
E.3 Proof of Proposition 18 66   
E.4 Proof of Corollary 19 . 66   
E.5 Proof of Lemma 20 . 67   
E.6 Proof of Proposition 21 67   
E.7 Proof of Corollary 22 . 67   
E.8 Proof of Proposition 24 68

## 1 Introduction

Under treatment interference, the potential outcome of one unit may depend on the treatment assignments of other units in the finite population. Consequently, the table of science is exponentially large. Moreover, unlike the classical case of no interference, there is no unique canonical causal estimand like the average treatment efect. Thus causal inference under interference requires assumptions on the structure of the table of science, together with a specification of a causal estimand and a design and estimator used to learn the estimand.

A large literature has emerged for causal inference under interference across settings such as peer efects (Manski, 1993; Bramoull´e et al., 2009; Graham, 2008), vaccination studies (Hudgens and Halloran, 2008), spatial environments (Zigler et al., 2012; Gifin et al., 2023), and online and network experiments (Aral and Walker, 2011). These approaches use a variety of assumptions to reduce the number of potential outcomes per unit and define diferent causal estimands along with designs and estimators to estimate them. However, two layers of the causal inference problem are often specified jointly, sometimes even implicitly. The first concerns the modeling assumptions on potential outcomes. Assumptions under interference involve three distinct choices: which units’ treatment assignments in the finite population may afect a given unit’s potential outcome, how those assignments are summarized or aggregated, and what functional restrictions, if any, are imposed on the resulting potential outcomes. These choices correspond to interference neighborhoods, exposure mappings, and outcome models respectively. These three modeling assumptions are often conflated in the literature, but they play diferent mathematical roles.

The second layer concerns the definition of causal efects and its interaction with design and estimation. Some causal estimands under interference are fixed potential-outcome contrasts: they compare outcomes under two specified treatment allocations or treatment– exposure conditions. Other estimands are allocation-averaged contrasts: they average potential outcomes under a specified allocation distribution, often corresponding to a policy implementation. We treat the allocation distribution that is part of the definition of an estimand as distinct from the design used to estimate it.

Without separating these layers, it can be dificult to determine whether diferences across methods arise from diferent assumptions on the potential-outcome table, diferent causal estimands, diferent allocation policies, diferent experimental designs, or diferent estimators. In this paper, we develop a unified framework for finite population causal inference under interference that makes these distinctions explicit. We consider the Table of Science as the primitive finite-population object. We develop a set of hierarchical modeling assumptions on the Table of Science in the form of interference neighborhoods, exposure mappings, and outcome models. The first two are exclusion restrictions: they reduce the Table of Science by declaring which assignment coordinates can matter and which assignments are equivalent. On the other hand, an outcome model is a functional restriction on the potential outcomes. Once an interference neighborhood and exposure mapping are specified, the full-assignment potential outcome $Y _ { i } ( \mathbf { z } )$ is represented by a lower-dimensional reduced potential outcome $Y _ { i } ( z , e )$ , where z is the vector of treatment allocations, $z _ { i }$ is unit i’s treatment, and $e _ { i }$ is its exposure. A central consequence of this representation is a nonparametric linear decomposition of the reduced potential outcome $Y _ { i } ( z , e )$ given by

$$
Y _ { i } ( z , e ) = A _ { i } ( z ) + B _ { i } ( e ) + z C _ { i } ( e ) ,
$$

where $A _ { i }$ captures the direct treatment component, $B _ { i }$ the spillover component, and $C _ { i }$ the treatment–spillover interaction. This decomposition is a reparameterization of the potential outcomes. It gives a common language for specifying nonparametric, semiparametric, and parametric finite population models of potential outcomes under interference. For example, a nonparametric model can leave the exposure as the full interference neighborhood and leave $A _ { i } , B _ { i } , C _ { i }$ unrestricted. Semiparametric and parametric outcome models impose additional restrictions such as additivity, homogeneity, linearity in exposure, or parametric forms for the spillover and interaction components. All potential-outcome models used in the interference literature, including arbitrary interference, linear-in-means models, stratified interference, binary exposure, spatial decay, and threshold or contagion models, can be represented within this hierarchy by appropriate choices of the interference neighborhood, exposure mapping, and outcome-model restrictions.

The framework also separates two natural classes of causal estimands under interference, assuming that the interference neighborhood and exposure mappings are correctly specified. Average causal efects (ACEs), such as the direct and total treatment efects, are contrasts of fixed potential outcomes under specific treatment allocations or treatment– exposure combinations. They are properties of the finite population and do not involve an allocation distribution. In contrast, expected causal efects (ECEs), such as the expected direct efect (EDE) and expected indirect efect (EIE), are contrasts of expectations of potential outcomes under allocation distributions–sometimes also referred to as policies. The distributions entering the two arms of the contrast may be the same, diferent, or induced by a common allocation rule; in all cases, the allocation distribution used to average potential outcomes is part of the causal question itself. The experimental design remains distinct from this allocation distribution, though it is usually informed by it.

We show that there is a fundamental separation between the two classes for nonparametric estimation. For ACEs, the design must reveal the particular treatment–exposure potential outcomes that define the target estimand. Potential outcomes revealed by the design but outside this set are nuisance for that estimand: without further outcome-model restrictions on the causal response surface, observing them gives no information about the fixed target potential outcomes. For ECEs, by contrast, the exposure states generated by the target allocation distribution enter the estimand. The design does not need to isolate a set of fixed treatment-exposure conditions; it must reproduce the allocation distribution’s averaging weights.

We illustrate this through design-based variance bounds for canonical weighted unbiased Horvitz–Thompson estimators. We show that, for both classes of estimands, the variance of the canonical estimator decomposes into a diagonal weight term, an interference-dependence term from overlapping neighborhoods, and a residual design-dependence term induced by the randomization design. The diference lies in the weight term. ACE weights involve design-induced inverse propensity scores for fixed relevant treatment–exposure conditions. Under classical graph-unaware designs, these propensity scores can be exponentially small in the degree of the interference graph, and matching lower bounds show that this obstruction applies to all unbiased estimators. By contrast, ECE weights are allocation policy-to-design importance weights; when the design is aligned with the allocation policy, the diagonal term is mild and the dominant cost is the dependence among observed weighted outcomes. Consequently we show that, under the same classical designs, fixed treatment-exposure ACE estimation can require sample sizes growing exponentially in the degree of the interference graph, whereas ECE estimation is limited by polynomial dependence on the degree of an appropriate dependence graph.

These results have direct practical implications for design and estimation. Designs for nonparametric ACE estimation must be aligned with the treatment–exposure combinations that define the target estimand. Potential outcomes that do not appear in the definition of the estimand are nuisance potential outcomes for nonparametric estimation. Thus, nonparametric estimation of ACEs requires specialized graph-aware designs that place suficient probability on revealing the relevant potential outcomes. We construct such designs from independent sets of the interference graph: a distance-one independent-set design targets the direct treatment efect and a distance-two independent-set design targets the total treatment efect. We show that in each case, these graph aware designs replace the exponentially small propensity of the relevant potential outcome by a focal probability that is polynomial in the relevant local graph size. Sparse seeding of the distance-one independent set additionally estimates the influencer spillover efect—the average efect of treated influencers on untreated peers under the any-treated exposure map—at constant-order focal propensities on locally tree-like regular graphs (Section 5.3). The resulting estimators have mean squared error polynomial in the graph degree scale for broad graph classes. For the total effect, cluster randomized designs can be sharper than the independent-set construction when closed neighborhoods are contained in few clusters, though this requires the clustering to be aligned with the interference graph. Outcome-model restrictions provide another route for ACE estimation by relating nuisance potential outcomes to target potential outcomes, but this route is no longer fully nonparametric and is subject to misspecification of the outcome model. By contrast, for ECEs, the allocation policy is part of the estimand itself, and all potential outcomes with positive probability under the policy enter the estimand. Thus, ECEs can be estimated nonparametrically under classical Bernoulli designs when the target allocation policy is Bernoulli.

The paper is organized as follows. Section 2 introduces the Table of Science and the modeling hierarchy. Section 2.3 gives the non-parametric decomposition of Potential Outcomes and shows how common model classes arise as special cases. Section 3 defines ACEs and ECEs. Section 4 develops the generic estimation bounds and the ACE/ECE bottleneck comparison. Section 6 draws implications for classical estimators, specialized designs, and model-assisted estimation. Section 5 develops estimand-aligned independent-set designs that recover polynomial rates for average causal efects—including a sparse-seeding design that estimates the influencer spillover efect at constant-order propensities on boundeddegree graphs—together with a comparison against cluster randomized designs, and Section 8 concludes.

## 2 Hierarchical models of potential outcomes

Consider a finite population of n units, indexed by $i \in \{ 1 , \ldots , n \}$ , and a binary treatment assignment $z _ { i } \in \{ 0 , 1 \}$ for each unit i. The vector of treatment assignments is denoted by $\mathbf { z } = ( z _ { 1 } , \ldots , z _ { n } )$ , with $\mathbf { z } \in \{ 0 , 1 \} ^ { n }$ . Under arbitrary treatment interference, the potentia outcome of a unit i depends on the entire assignment vector. We write

$$
Y _ { i } ( \mathbf { z } ) , \qquad \mathbf { z } \in \{ 0 , 1 \} ^ { n } .
$$

This denotes the full-assignment potential outcome of unit i under treatment allocation z. The collection of all full-assignment potential outcomes

$$
\mathbb { T } = \{ Y _ { i } ( \mathbf { z } ) : i = 1 , \ldots , n , \ \mathbf { z } \in \{ 0 , 1 \} ^ { n } \}
$$

is the Table of Science for the study. Crucially, T is a fixed, full $n \times 2 ^ { n }$ table dictated by nature, existing prior to any modeling assumptions. Let Z denote the random treatment assignment generated by a design $p ( \mathbf { Z } )$ . The observed outcome for unit i is

$$
Y _ { i } ^ { \mathrm { o b s } } = Y _ { i } ( { \bf Z } ) ,
$$

and in any realized experiment, exactly one column of $\mathbb { T }$ is observed.

Since each unit has exponentially many counterfactuals relative to what is observable, modeling assumptions are necessary to relate unobserved potential outcomes to observed ones. We specify finite population models of potential outcomes through three distinct hierarchical components: an interference graph, an exposure mapping, and an outcome model.

Such assumptions constrain the entries of $\mathbb { T }$ in two diferent ways. The interference graph and exposure mappings are exclusion restrictions. The interference graph specifies which treatment coordinates can enter a unit’s full-assignment potential outcome and the exposure mapping specifies how those coordinates aggregate. When combined, the interference graph and the exposure mapping partition the $2 ^ { n }$ columns of T into equivalence classes for each row and require $\mathbb { T }$ to be constant within each class. On the other hand, Outcome models are a diferent kind of constraint: they do not merge columns of $\mathbb { T }$ , but instead impose functional constraints on the values $\mathbb { T }$ takes across distinct potential outcomes, potentially linking entries within a row or across rows (units). The classical no-interference setting is the most restrictive exclusion restriction, forcing each row to take only two distinct values across all $2 ^ { n }$ columns. Formally, each exclusion restriction is specified by an equivalence relation $\sim _ { i }$ on $\{ 0 , 1 \} ^ { n }$ that T must respect, meaning $\mathbf { z } \sim _ { i } \mathbf { z } ^ { \prime } \Rightarrow Y _ { i } ( \mathbf { z } ) = Y _ { i } ( \mathbf { z } ^ { \prime } )$ ; T is then fully determined by its values on the class representatives.

## 2.1 Interference graph

The first restriction specifies the set of units in the finite popluation whose treatment assignments efect $Y _ { i } ( \mathbf { z } )$

Assumption 1 (Interference exclusion restriction) For each unit $i ,$ there exists a subset $N _ { i } \subseteq \{ 1 , \dots , n \} \setminus \{ i \}$ such that, for any two treatment allocations z and $\mathbf { z } ^ { \prime }$

$$
z _ { i } = z _ { i } ^ { \prime } \quad a n d \quad z _ { N _ { i } } = z _ { N _ { i } } ^ { \prime } \quad \Longrightarrow \quad Y _ { i } ( { \bf z } ) = Y _ { i } ( { \bf z } ^ { \prime } ) .
$$

Formally, Assumption 1 is the equivalence relation

$$
{ \bf z } \sim _ { i } ^ { G } { \bf z } ^ { \prime } \iff z _ { i } = z _ { i } ^ { \prime } \mathrm { a n d } \ { \bf z } _ { N _ { i } } = { \bf z } _ { N _ { i } } ^ { \prime } ,
$$

which partitions $\{ 0 , 1 \} ^ { n }$ into $2 ^ { d _ { i } + 1 }$ equivalence classes for row $i ,$ where $d _ { i } = | N _ { i } |$ . Row i of T takes at most $2 ^ { d _ { i } + 1 }$ distinct values—one per class—and the potential outcome can be written as

$$
Y _ { i } ( \mathbf { z } ) : = Y _ { i } ( z _ { i } , \mathbf { z } _ { N _ { i } } ) ,\tag{2.1}
$$

depending only on i’s own treatment and the treatments of units in $N _ { i }$ . The neighborhoods $\{ N _ { i } \} _ { i = 1 } ^ { n }$ may overlap, be asymmetric, or vary in size across units. For a fixed Table of Science, there is a minimal neighborhood

$$
N _ { i } ^ { \star } = \left\{ j \neq i : \exists \mathbf { z } , \mathbf { z } ^ { \prime } ~ { \mathrm { d i f f e r i n g ~ o n l y ~ i n ~ } } z _ { j } ~ { \mathrm { s u c h ~ t h a t ~ } } Y _ { i } ( \mathbf { z } ) \neq Y _ { i } ( \mathbf { z } ^ { \prime } ) \right\} .
$$

The collection $\{ N _ { i } ^ { \star } \}$ can be encoded via a minimal functional interference graph $\boldsymbol { G } ^ { \star } =$ $( V , E ^ { \star } )$ on n nodes. The vertices of the graph V correspond to the n units in the finite population. A directed edge from node $j$ to node i exists if the treatment of unit $j$ can afect the potential outcome of unit i. In the remainder of the paper, we work with a specified interference graph $G = ( V , E )$ and the corresponding neighborhoods $\{ N _ { i } \} _ { i = 1 } ^ { n } ,$ under Assumption 1. Issues arising from random or misspecified interference graphs are discussed in the remark below. We use the degree to denote the in-degree of this graph G. Setting $N _ { i } = \{ 1 , \dots , n \} \setminus \{ i \}$ recovers the complete interference case in which all $2 ^ { n }$ entries of each row of T are distinct (the restriction is vacuous), while $N _ { i } = \emptyset$ yields the classical SUTVA case in which each row takes only two distinct values.

It is useful to distinguish the interference graph from an observed social graph. The interference graph is an abstract representation of restrictions on the potential-outcome table. It may or may not coincide with a social graph, be an induced subgraph of it, or be a complete graph with the strength of interference modeled through an exposure function. In social-network settings, a natural choice is to set $N _ { i }$ to i’s neighbors in the social graph (onehop interference), or to all units within graph distance k (multi-hop interference). Partial interference, where units can be partitioned into disjoint groups and interference is assumed to occur only within groups, is another special case (Sobel, 2006; Hudgens and Halloran, 2008). Spatial and bipartite interference models can also be represented by appropriate choices of G; see Appendix A.

Remark 1 (Random and misspecified interference) The minimal functional interference graph $G ^ { \star }$ may not be known in applications, and the specified interference graph G may itself be noisy or random. The graph G used in analysis should therefore be understood as an assumed interference graph. If the assumed interference neighborhoods $N _ { i }$ satisfies $N _ { i } ^ { \star } \subseteq N _ { i }$ , the exclusion restriction remains valid but the representation is nonminimal. If, on the other hand, $N _ { i }$ omits units in $N _ { i } ^ { \star }$ , the interference graph is misspecified. Another approach is to treat the interference graph as random, and model the observed graph using a distribution; see Bhattacharya et al. (2020), Li and Wager (2022), and Ohnishi et al. (2025) for examples. For doubly robust estimators under a misspecified G, see Weinstein and Nevo (2026).

Dependence graphs The interference graph should not be confused with the dependence graph used in large-sample analysis of design based estimators. The interference graph is a restriction on the arguments of the potential-outcomes. A dependence graph is a probabilistic object attached to a particular collection of random variables, such as observed outcomes or estimator summands. Under independent Bernoulli assignment, this dependence may be induced by overlapping interference neighborhoods. Under dependent randomization designs, such as complete or cluster randomization, the assignment mechanism can induce additional dependence even when the interference neighborhoods do not overlap. We define the dependence graph formally in Section 4.1, where it is used for variance and consistency results.

Under Assumption 1, row i of T takes at most $2 ^ { d _ { i } + 1 }$ distinct values; T itself remains an $n \times 2 ^ { n }$ table, but is fully determined by specifying these $2 ^ { d _ { i } + 1 }$ values per row. Even so, $2 ^ { d _ { i } + 1 }$ can still be large; the exposure mapping, introduced next, provides a further exclusion restriction.

## 2.2 Exposure mapping

An exposure mapping (Aronow and Samii, 2017) summarizes neighborhood assignments into an exposure value,

$$
e _ { i } = f _ { i } ( \mathbf { z } _ { N _ { i } } ) \in \mathcal { E } _ { i } .\tag{2.2}
$$

Assumption 2 (Exposure mapping) For each unit $i ,$ there exists an exposure mapping $f _ { i } : \{ 0 , 1 \} ^ { | N _ { i } | }  \mathcal { E } _ { i }$ such that, for any two neighborhood assignments $\mathbf { z } _ { N _ { i } }$ and $\mathbf { z } _ { N _ { i } } ^ { \prime }$

$$
f _ { i } ( { \bf z } _ { N _ { i } } ) = f _ { i } ( { \bf z } _ { N _ { i } } ^ { \prime } ) \quad \Longrightarrow \quad Y _ { i } ( z _ { i } , { \bf z } _ { N _ { i } } ) = Y _ { i } ( z _ { i } , { \bf z } _ { N _ { i } } ^ { \prime } ) \quad f o r z _ { i } \in \{ 0 , 1 \} .
$$

Formally, Assumption 2 is the equivalence relation

$$
\mathbf { z } _ { N _ { i } } \sim _ { i } ^ { f } \mathbf { z } _ { N _ { i } } ^ { \prime } \iff f _ { i } ( \mathbf { z } _ { N _ { i } } ) = f _ { i } ( \mathbf { z } _ { N _ { i } } ^ { \prime } ) ,
$$

which coarsens ${ \sim } _ { i } ^ { G }$ by merging neighborhood assignments that produce the same exposure value. The function $f _ { i }$ encodes how the treatment assignments of unit’s in $N _ { i }$ afect unit $i \mathrm { \ ' } _ { \mathrm { S } }$ potential outcome. In the main text, we focus on the case where exposure values are one-dimensional and discrete, with $| \mathcal { E } _ { i } | = K _ { i }$ exposure levels. Vector-valued and continuous exposures are discussed in Appendix A. We use the symbols $e _ { i } = 0$ to denote no exposure and $e _ { i } = 1$ to denote full exposure, with the exposure mapping specifying what these labels mean in context.

Combined, Assumptions 1 and 2 define the relation

$$
\mathbf { z } \sim _ { i } \mathbf { z } ^ { \prime } \iff z _ { i } = z _ { i } ^ { \prime } \mathrm { a n d } f _ { i } ( \mathbf { z } _ { N _ { i } } ) = f _ { i } ( \mathbf { z } _ { N _ { i } } ^ { \prime } ) ,
$$

with $2 K _ { i }$ equivalence classes, where $K _ { i } = | \mathcal { E } _ { i } |$ . Row i of T is thus fully determined by its $2 K _ { i }$ values $Y _ { i } ( z _ { i } , e _ { i } )$ , one for each $( z _ { i } , e _ { i } )$ pair. We call each such value a reduced potential outcome. The collection of these distinct reduced potential outcomes,

$$
\mathbb { S } = \{ Y _ { i } ( z , e ) : z \in \{ 0 , 1 \} , \ e \in \mathscr { E } _ { i } , \ i = 1 , \ldots , n \} ,
$$

is the causal response surface induced by the specified interference neighborhoods and exposure mappings. Thus $Y _ { i } ( \mathbf { z } )$ denotes a full-assignment potential outcome in $\mathbb { T } .$ whereas $Y _ { i } ( z , e )$ denotes the corresponding reduced potential outcome in S. When there is no interference, $Y _ { i } ( z )$ is the classical own-treatment potential outcome, the special case with no exposure argument. The non-parametric case corresponds to the identity mapping $e _ { i } = { \mathbf { z } } _ { N _ { i } }$ giving $K _ { i } = 2 ^ { d _ { i } }$ and recovering ${ \sim } _ { i } ^ { G }$

Common examples of exposure mappings include symmetric or anonymous exposure, where $f _ { i }$ depends only on the number or fraction of treated neighbors (Hudgens and Halloran, 2008; Manski, 2013); binary exposure, where $\mathcal { E } _ { i } = \{ 0 , 1 \}$ ; additive exposure of the form $\begin{array} { r } { f _ { i } ( \mathbf { z } _ { N _ { i } } ) = \sum _ { j \in N _ { i } } h _ { j } ( z _ { j } ) } \end{array}$ ; and weighted or non-anonymous exposure, where $f _ { i }$ incorporates neighbor identity, covariates, or graph distance.

## 2.3 A non-parametric decomposition of the causal response surface

Assumption 1 concerns who can interfere with unit i, Assumption 2 concerns the form of that interference. Under Assumptions 1 and $2 , \ \mathbb { T }$ is fully determined by the causal response surface $\mathbb { S } ,$ comprising $2 \sum _ { i } K _ { i }$ values. This response surface admits a natural nonparametric linear representation. This representation is an identity, and, without loss of generality, decomposes the potential outcome of each unit into three components: (a) a component depending only on a unit’s own treatment, (b) a component depending only on a unit’s exposure, and (c) the interaction between (a) and (b).

Theorem 1 (Non-parametric decomposition) Under Assumptions 1 and 2, there exist functions $A _ { i } ( z ) , B _ { i } ( e _ { i } )$ and $C _ { i } ( e _ { i } )$ such that, for each unit i and each $( z _ { i } , e _ { i } ) \in \{ 0 , 1 \} \times \mathcal { E } _ { i }$ a potential outcome $Y _ { i } ( z , e ) \in \mathbb { S }$ can be written as

$$
Y _ { i } ( z _ { i } , e _ { i } ) = A _ { i } ( z _ { i } ) + B _ { i } ( e _ { i } ) + z _ { i } C _ { i } ( e _ { i } ) = \alpha _ { i } + \beta _ { i } z _ { i } + B _ { i } ( e _ { i } ) + z _ { i } C _ { i } ( e _ { i } ) ,\tag{2.3}
$$

with the normalization $B _ { i } ( 0 ) = C _ { i } ( 0 ) = 0$

The decomposition has a nice interpretation: the $A _ { i }$ term captures the part of the potential outcome depending only on unit $i \mathrm { \ ' } _ { \mathrm { S } }$ own treatment, $B _ { i }$ term caputres the additive exposure component, and $C _ { i }$ the interaction between own-treatment and exposure (synergy or competition). These components define the corresponding direct, spillover, and interaction causal efects studied in Section 3.

The representation is a full-rank reparameterization of the causal response surface. In particular, it makes no assumptions on the potential outcomes and parametrizes the $2 K _ { i }$ distinct values of row i using $2 K _ { i }$ linearly independent parameters. For example, under the identity exposure mapping, the number of exposure levels is $K _ { i } = 2 ^ { d _ { i } }$ and the number of parameters in the representation $Y _ { i } ( \mathbf { z } ) = \alpha _ { i } + \beta _ { i } z _ { i } + B _ { i } ( z _ { N _ { i } } ) + z _ { i } C _ { i } ( z _ { N _ { i } } )$ for each unit i is $2 ^ { d _ { i } + 1 }$ . Similarly, under arbitrary interference, $K _ { i } = 2 ^ { n - 1 }$ and Theorem 1 gives the non-parametric model $Y _ { i } ( \mathbf { z } ) = \alpha _ { i } + \beta _ { i } z _ { i } + B _ { i } ( \mathbf { z } _ { - i } ) + z _ { i } C _ { i } ( \mathbf { z } _ { - i } )$ , which has $2 ^ { n }$ parameters for unit i.

## 2.4 Outcome model

An outcome model imposes functional constraints on the $2 K _ { i }$ distinct values of the causal response surface via specifying semi-parametric or parametric restrictions on $\{ A _ { i } , B _ { i } , C _ { i } \}$ in the decomposition (2.3); it does not operate by further merging assignment columns of T. Such constraints reduce the number of free parameters and may arise from scientific knowledge $\left( \mathrm { e . g . } , \right.$ a linear response) or be used to express sharp null hypotheses. Unlike exclusion restrictions, which operate within a single row of T before forming S, outcome models constrain the distinct potential outcomes of S and can also link entries across rows — homogeneity assumptions, for instance, require $\beta _ { i }$ to be common across units.

Additive efects. A natural and commonly used restriction is that the direct and indirect efects enter additively, without interaction. In terms of the decomposition in (2.3), this corresponds to

$$
C _ { i } ( e ) = 0 \quad { \mathrm { f o r ~ a l l ~ } } i { \mathrm { ~ a n d ~ a l l ~ } } e \in \mathcal { E } _ { i } .
$$

The potential outcomes then simplify to $Y _ { i } ( z , e ) = A _ { i } ( z ) + B _ { i } ( e )$ , and each unit has $K _ { i } + 2$ free parameters instead of $2 K _ { i }$

Constant (homogeneous) efects. In some settings, it may be reasonable to assume that treatment and exposure efects do not vary across units. This can be written as

$$
A _ { i } ( z ) = A ( z ) , \qquad B _ { i } ( e ) = B ( e ) , \qquad C _ { i } ( e ) = C ( e ) , \quad \mathrm { f o r ~ a l l ~ } i .
$$

This links entries across rows of T: potential outcomes from diferent units are constrained to share the same efect functions, reducing the parameter count from $O ( n K _ { i } )$ to $O ( K )$ ).

Linear efects in exposure. When exposure levels are ordered $( \mathrm { e . g . }$ , counts of treated neighbors), it may be plausible that the interference efect depends linearly on e. One simple example is

$$
B _ { i } ( e ) = \gamma _ { i } e , \qquad C _ { i } ( e ) = \theta _ { i } e ,
$$

which yields $Y _ { i } ( z , e ) = \alpha _ { i } + \beta _ { i } z _ { i } + \gamma _ { i } e + \theta _ { i } z e$ . If $\alpha _ { i } , \beta _ { i } , \gamma _ { i }$ and $\theta _ { i }$ are further assumed constant across units, the model reduces to a four-parameter linear dose-response formulation.

Sharp null hypotheses. Outcome models can also be used to express sharp null hypotheses for randomization testing. The linear decomposition allows these to be stated precisely. For instance, a sharp null of no treatment efect under interference is

$$
\beta _ { i } = 0 { \mathrm { ~ a n d ~ } } B _ { i } ( e _ { i } ) = C _ { i } ( e _ { i } ) = 0 { \mathrm { ~ f o r ~ a l l ~ } } e _ { i } , i
$$

Similarly, the condition

$$
Y _ { i } ( 1 , e ) = Y _ { i } ( 0 , e ) \quad { \mathrm { f o r ~ a l l ~ } } i { \mathrm { ~ a n d ~ a l l ~ } } e \in \mathcal { E } _ { i } ,
$$

is equivalent to the hypothesis $A _ { i } ( 1 ) + C _ { i } ( e _ { i } ) = A _ { i } ( 0 )$ and implies sharp null of no treatment efect only under additivity $\begin{array} { r } { \mathrm { ( i . e . } \ C _ { i } ( e _ { i } ) \ = \ 0 ) } \end{array}$ . More targeted nulls, such as absence of spillover efects or no interaction efects, can be expressed by setting subsets of $\{ B _ { i } ( e ) , C _ { i } ( e ) \}$ to zero. See Rosenbaum (2007); Aronow (2012); Pouget-Abadie et al. (2019); Athey et al. (2018); Basse et al. (2019); Puelz et al. (2022); Rigdon and Hudgens (2015) for construction of statistical tests under various such hypotheses.

Relation to estimation. Outcome modeling assumptions are not required to define causal estimands. However, they can be used for model-assisted design and estimation. In the absence of such assumptions, the parameters $\{ A _ { i } , B _ { i } , C _ { i } \}$ vary freely across units and exposure levels. Imposing them constrains these parameters, so that a response observed under one treatment-exposure combination can inform missing responses under other combinations. If correctly specified, this model-assisted approach can significantly reduce variance at either the design or analysis stage.

## 2.5 Canonical model classes

Diferent choices of the interference neighborhood $N _ { i }$ , the exposure mapping $f _ { i }$ , and the outcome model restrictions imposed on $( A _ { i } , B _ { i } , C _ { i } )$ lead to diferent potential-outcome models. The decomposition in Theorem 1 encompasses the full range of models used in the literature. A non-parametric model imposes no restrictions on $A _ { i } , B _ { i } , C _ { i }$ beyond the interference neighborhood and exposure mapping; the functions themselves are unrestricted. A semi-parametric model imposes qualitative restrictions on $A _ { i } , B _ { i } , C _ { i ^ { - } }$ —such as additivity $( C _ { i } \equiv 0 )$ , symmetry, or monotonicity. A parametric model specifies finite-dimensional functional forms for $A _ { i } , B _ { i } , C _ { i }$ , for example linear functions of exposure or homogeneous efects across units. We present some commonly used examples below.

Example 1 (Arbitrary interference) Let $N _ { i } = \{ 1 , \ldots , n \} \setminus \{ i \}$ , and let $f _ { i } ( { \bf z } _ { N _ { i } } ) = { \bf z } _ { N _ { i } }$ Then

$$
Y _ { i } ( \mathbf { z } ) = Y _ { i } ( z _ { i } , \mathbf { z } _ { - i } ) = \alpha _ { i } + \beta _ { i } z _ { i } + B _ { i } ( \mathbf { z } _ { - i } ) + z _ { i } C _ { i } ( \mathbf { z } _ { - i } ) .
$$

Row i of T takes $2 ^ { n }$ distinct values; Assumption 1 places no constraint.

Example 2 (Neighborhood interference) Let $N _ { i } \subsetneq \{ 1 , \dots , n \} \backslash \{ i \}$ be a specified proper subset, and let $f _ { i } ( { \bf z } _ { N _ { i } } ) = { \bf z } _ { N _ { i } }$ . Then

$$
Y _ { i } ( { \bf z } ) = Y _ { i } ( z _ { i } , { \bf z } _ { N _ { i } } ) = \alpha _ { i } + \beta _ { i } z _ { i } + B _ { i } ( { \bf z } _ { N _ { i } } ) + z _ { i } C _ { i } ( { \bf z } _ { N _ { i } } ) .
$$

Row i of T takes $2 ^ { d _ { i } + 1 }$ distinct values, where $d _ { i } = | N _ { i } | ,$ the exponent reduces from $n - 1$ to the local neighborhood size.

Example 3 (No interference) Let $N _ { i } = \emptyset$ . Then $Y _ { i } ( \mathbf { z } ) = Y _ { i } ( z _ { i } )$ , which is the classical no-interference case. Each unit has two own-treatment potential outcomes.

Example 4 (Count or fractional exposure mappings) Let $\begin{array} { r } { e _ { i } = f _ { i } ( { \bf z } _ { N _ { i } } ) = \sum _ { j \in N _ { i } } z _ { j } } \end{array}$ denote the number of treated neighbors of unit i, or alternatively let $\begin{array} { r } { e _ { i } ~ = ~ \sum _ { j \in N _ { i } } { z _ { j } } / { d _ { i } } } \end{array}$ denote the proportion of treated neighbors of unit i. Then

$$
Y _ { i } ( z _ { i } , e _ { i } ) = \alpha _ { i } + \beta _ { i } z _ { i } + B _ { i } ( e _ { i } ) + z _ { i } C _ { i } ( e _ { i } ) ,\tag{2.4}
$$

where $e _ { i } \in \{ 0 , 1 , \ldots , d _ { i } \}$ in the count case and $e _ { i } \in \{ 0 , 1 / d _ { i } , 2 / d _ { i } , . . . , 1 \}$ in the fractional case. Under this model, the number of free parameters per unit is $2 ( d _ { i } + 1 )$ . If additivity is imposed, $C _ { i } ( e _ { i } ) = 0$ for all $e _ { i } ,$ and the number of free parameters per unit reduces to $d _ { i } + 2$

Example 5 (Linear-in-means model) Starting from equation (2.4), let $B _ { i } ( e _ { i } ) = \gamma _ { i } e _ { i }$ and $C _ { i } ( e _ { i } ) = \theta _ { i } e _ { i }$ . Then

$$
Y _ { i } ( z _ { i } , e _ { i } ) = \alpha _ { i } + \beta _ { i } z _ { i } + \gamma _ { i } e _ { i } + \theta _ { i } z _ { i } e _ { i } .\tag{2.5}
$$

If the interaction term is zero, the model becomes

$$
Y _ { i } ( z _ { i } , e _ { i } ) = \alpha _ { i } + \beta _ { i } z _ { i } + \gamma _ { i } e _ { i } .\tag{2.6}
$$

This additive linear outcome model is the linear-in-means model used in the econometrics literature (Manski, 1993; Bramoull´e et al., 2009; Graham, 2008).

Example ${ \textbf { 6 } } ( 2 \times 2$ exposure mapping) Let $f _ { i } ( { \bf z } _ { N _ { i } } ) = 0$ if all elements $O f { \bf z } _ { N _ { i } }$ are 0 and $f _ { i } ( { \bf z } _ { N _ { i } } ) = 1$ if at least one element $O f \mathbf { z } _ { N _ { i } }$ is 1. This gives the two-by-two response-surface model, or binary exposure mapping,

$$
Y _ { i } ( z _ { i } , e _ { i } ) = \alpha _ { i } + \beta _ { i } z _ { i } + \gamma _ { i } e _ { i } + \theta _ { i } z _ { i } e _ { i } ,\tag{2.7}
$$

where $e _ { i } \in \{ 0 , 1 \}$ and $z _ { i } \in \{ 0 , 1 \}$ . Each unit has four distinct potential outcomes. Setting $\theta _ { i } = 0$ gives the additive binary exposure mapping,

$$
Y _ { i } ( z _ { i } , e _ { i } ) = \alpha _ { i } + \beta _ { i } z _ { i } + \gamma _ { i } e _ { i } ,\tag{2.8}
$$

with three free parameters per unit.

Example 7 (Weighted exposure mappings) In non-anonymous interference, the exposure depends on the identity or covariates of the neighbors. For example, one may specify

$$
e _ { i } = \sum _ { j \in N _ { i } } w _ { i j } z _ { j } ,
$$

and then write

$$
Y _ { i } ( z _ { i } , e _ { i } ) = \alpha _ { i } + \beta _ { i } z _ { i } + B _ { i } ( e _ { i } ) + z _ { i } C _ { i } ( e _ { i } ) .
$$

The additive version of this model has been used for total treatment efects under network interference (Yu et al., 2022). Additional non-anonymous and spatial examples are given in Appendix A.

The three canonical non-parametric representations—arbitrary interference, neighborhood interference, and exposure mapping (illustrated by examples 4–7 above)—are compared in Section 4 with respect to their estimation bottlenecks.

## 3 Defining causal estimands under network interference

Under treatment interference, there are many non-equivalent ways to define causal efects, and the appropriate estimand depends on the scientific question of interest. Causal estimands in the framework of Section 2 are summaries of the causal response surface $Y _ { i } ( z , e )$ We study two broad classes: average causal efects (ACEs), which are contrasts of fixed potential outcomes and do not involve any distribution over treatment allocations; and expected causal efects (ECEs), which are contrasts of expectations of potential outcomes under a specified allocation policy.

## 3.1 Average causal efects under network interference

Average causal efects (ACEs) are defined as contrasts between two fixed potential outcomes. Once an interference neighborhood and exposure mapping have been specified, the treatment–exposure space for unit i is

$$
\mathcal { X } _ { i } = \{ 0 , 1 \} \times \mathcal { E } _ { i } .
$$

For an assignment z, let

$$
D _ { i } ( \mathbf { z } ) = ( z _ { i } , e _ { i } ( \mathbf { z } ) ) \in \mathcal { X } _ { i }
$$

denote the treatment–exposure combination realized by unit i. A generic ACE compares two fixed treatment–exposure combinations

$$
d _ { i } ^ { + } = ( z _ { i } ^ { + } , e _ { i } ^ { + } ) , \qquad d _ { i } ^ { - } = ( z _ { i } ^ { - } , e _ { i } ^ { - } ) , \qquad d _ { i } ^ { + } , d _ { i } ^ { - } \in \mathcal { X } _ { i } .
$$

Let $d ^ { + } = \{ d _ { i } ^ { + } \} _ { i = 1 } ^ { n }$ and $\mathfrak { d } ^ { - } = \{ d _ { i } ^ { - } \} _ { i = 1 } ^ { n }$ . The corresponding ACE is defined as

$$
\tau _ { \mathrm { A C E } } ( d ^ { + } , d ^ { - } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big ( Y _ { i } ( d _ { i } ^ { + } ) - Y _ { i } ( d _ { i } ^ { - } ) \Big ) .\tag{3.1}
$$

This definition directly generalizes the average treatment efect under no-interference assumptions, and does not require assumptions on the functional form of the response surface. Diferent choices of $d _ { i } ^ { + }$ and $d _ { i } ^ { - }$ correspond to diferent causal questions about treatment and interference.

This definition also includes arbitrary interference as a special case: take $N _ { i } = \{ 1 , \dots , n \} \backslash$ {i} and use the identity exposure mapping $e _ { i } = \mathbf { z } _ { - i }$ . Then $D _ { i } ( { \bf z } ) = ( z _ { i } , { \bf z } _ { - i } )$ , and equation (3.1) defines ACEs directly in terms of the full treatment vector.

We now give a few examples of ACEs. Recall that the exposure levels $e _ { i } = 0$ and $e _ { i } = 1$ are special values defined to represent the situations when a unit is not exposed or fully exposed respectively.

Direct treatment efect. The direct treatment efect compares the average outcome under treatment versus control when exposure is held at the baseline level of 0:

$$
\pi _ { \mathrm { D T E } } = \tau _ { \mathrm { A C E } } ( d ^ { + } , d ^ { - } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big ( Y _ { i } ( 1 , 0 ) - Y _ { i } ( 0 , 0 ) \Big ) .\tag{3.2}
$$

where $d _ { i } ^ { + } = ( 1 , 0 )$ and $d _ { i } ^ { - } = ( 0 , 0 )$ for all i. The direct treatment efect is the interference analogue of the average treatment efect in the no-interference case. It measures the average efect of treatment on a unit when the exposure is fixed at the baseline level.

Average indirect efects. For two exposure levels $e _ { a } , e _ { b } \in \mathcal { E }$ and own treatment $z \in$ $\{ 0 , 1 \}$ , define the average indirect efect

$$
\pi \mathrm { E } ( z ; e _ { a } , e _ { b } ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \Big ( Y _ { i } ( z , e _ { b } ) - Y _ { i } ( z , e _ { a } ) \Big ) ,\tag{3.3}
$$

which holds the unit’s own treatment fixed and varies only the exposure regime between the two levels. The pair $( e _ { a } , e _ { b } )$ is part of the estimand: spillover questions can concern any two exposure levels, and the baseline $e _ { a }$ need not be the no-exposure level. The special case $e _ { a } =$ 0 recovers the level-specific indirect efects $\tau _ { \mathrm { I E C } } ( e ) = \tau _ { \mathrm { I E } } ( 0 ; 0 , e )$ and $\tau _ { \mathrm { I E T } } ( e ) = \tau _ { \mathrm { I E } } ( 1 ; 0 , e )$ on control and treated units, respectively. Writing τ<sub>IEC</sub> $= \tau _ { \mathrm { I E C } } ( e _ { i } ^ { \mathbf { 1 } } )$ and $\tau _ { \mathrm { I E T } } = \tau _ { \mathrm { I E T } } ( e _ { i } ^ { 1 } )$ for the full-exposure versions at the all-treated level $e = e _ { i } ^ { \mathbf { 1 } }$

$$
\pi _ { \mathrm { E C } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big ( Y _ { i } ( 0 , 1 ) - Y _ { i } ( 0 , 0 ) \Big ) ,\tag{3.4}
$$

$$
\pi _ { \mathrm { R T } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big ( Y _ { i } ( 1 , 1 ) - Y _ { i } ( 1 , 0 ) \Big ) .\tag{3.5}
$$

A natural two-level exposure pair arises from the any-treated exposure map

$$
\begin{array} { r } { U _ { i } = \mathbb { 1 } \Big \{ \sum _ { j \in N _ { i } } Z _ { j } \geq 1 \Big \} , } \end{array}\tag{3.6}
$$

which records whether at least one neighbor of unit i is treated. With $e _ { a } = 0$ and $e _ { b } = 1$ , the contrast $\tau _ { \mathrm { I E } } ( z ; 0 , 1 )$ ) measures the average efect of moving a unit with own treatment z from an environment free of treated neighbors to one with at least one treated neighbor. Held at $z = 0$ , this is the influencer spillover efect estimated in influencer-seeding experiments in social-network settings: the efect on an average untreated unit of the presence of a treated influencer among its neighbors (Paluck et al., 2016; Aral and Walker, 2012; Bond et al., 2012). Section 5.3 shows that this two-level estimand is estimable at constant-order focal propensities on bounded-degree graphs by a sparse-seeding design, in contrast to the exponentially small propensities of its extreme full-exposure special cases.

By Proposition 3 below, $\begin{array} { r } { \tau _ { \mathrm { I E } } ( 0 ; e _ { a } , e _ { b } ) = n ^ { - 1 } \sum _ { i } \{ B _ { i } ( e _ { b } ) - B _ { i } ( e _ { a } ) \} } \end{array}$ captures the additive exposure component, while $\tau _ { \mathrm { I E } } ( 1 ; e _ { a } , e _ { b } )$ adds the treatment–exposure interaction; the two coincide under additivity $( C _ { i } \equiv 0 )$

Remark 2 We use the term average indirect efect for the fixed-exposure ACE estimands above. This is diferent from the average indirect efect introduced in Hu et al. (2022), which we define as the expected indirect efect as it belongs to the expected causal efects class.

Total treatment efect. The total treatment efect compares the fully treated and fully exposed condition to the fully untreated and unexposed condition:

$$
\pi _ { \mathrm { T T E } } = \tau _ { \mathrm { A C E } } ( d ^ { + } , d ^ { - } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big ( Y _ { i } ( 1 , 1 ) - Y _ { i } ( 0 , 0 ) \Big ) .\tag{3.7}
$$

where $d _ { i } ^ { + } = ( 1 , 1 )$ and $d _ { i } ^ { - } = ( 0 , 0 )$ for all i.

## 3.2 Expected causal efects under network interference

Average causal efects compare two fixed potential outcomes. In many cases, however, the exposure of a unit is random under a policy of scientific interest. Expected causal efects are defined as contrasts of allocation-averaged responses, where entries of the causal response surface are averaged over one or more allocation distributions specified by the estimand. The target allocation policy is part of the causal question itself. A generic expected causal efect contrasts random treatment–exposure combinations induced by an allocation policy. Let $\mathcal { Q } ^ { + }$ and $\mathcal { Q }$ <sup>−</sup> denote two allocation policies. For each unit i, let

$$
D _ { i } ^ { + } ( { \bf Z } ) , \qquad D _ { i } ^ { - } ( { \bf Z } )
$$

be two corresponding random elements of $\mathcal { X } _ { i }$ . A generic ECE is any contrast of the form

$$
\operatorname { \pi } _ { \operatorname { \mathrm { E C E } } } ( D ^ { + } , D ^ { - } ; \mathcal { Q } ^ { + } , \mathcal { Q } ^ { - } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ \mathbb { E } _ { \mathcal { Q } ^ { + } } Y _ { i } \{ D _ { i } ^ { + } ( \mathbf { Z } ) \} - \mathbb { E } _ { \mathcal { Q } ^ { - } } Y _ { i } \{ D _ { i } ^ { - } ( \mathbf { Z } ) \} \right] .\tag{3.8}
$$

The distributions $\mathcal { Q } ^ { + } , \mathcal { Q } ^ { - }$ can be diferent from the experimental design $P ,$ but are often induced by $P .$ . For example, a study may use a randomized saturation design with severa treatment probabilities to estimate the expected efect of a single target rollout policy, such as treating 70 percent of units; the rollout policy defines the estimand, while the saturation design is chosen to provide overlap and precision. For the special case where ${ \mathcal { Q } } ^ { + } = { \mathcal { Q } } ^ { - } = { \mathcal { Q } } .$ we write $\tau _ { \mathrm { E C E } } ( D ^ { + } , D ^ { - } ; \mathcal { Q } )$ . We now consider some examples of expected causal efects used in the literature. Let $P _ { \mathcal { Q } } ( E _ { i } = e )$ denote the marginal distribution of the exposure level $E _ { i }$ under allocation policy Q. The expected response of unit i under own treatment $z \in \{ 0 , 1 \}$ is

$$
\mu _ { i } ( z ; \mathcal { Q } ) = \sum _ { e \in \mathcal { E } } Y _ { i } ( z , e ) P _ { \mathcal { Q } } ( E _ { i } = e ) .\tag{3.9}
$$

Expected (average) direct efect. The expected direct efect compares the average outcome under treatment versus control, averaging over the exposure distribution induced by the allocation policy:

$$
\pi _ { \mathrm { E D E } } ( Q ) = \mathbb { E } _ { Q } \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( Y _ { i } ( 1 , E _ { i } ) - Y _ { i } ( 0 , E _ { i } ) \right) \right] = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big ( \mu _ { i } ( 1 ; Q ) - \mu _ { i } ( 0 ; Q ) \Big ) .\tag{3.10}
$$

This is the one-contrast special case of (3.8) with $D _ { i } ^ { + } ( { \bf Z } ) = ( 1 , E _ { i } ) , D _ { i } ^ { - } ( { \bf Z } ) = ( 0 , E _ { i } )$ , and common policy $\mathcal { Q } .$ . The expected average direct efect is also known as the Average Direct Efect (ADE) in the literature, see S¨avje et al. (2017). We use the term expected direct efect to emphasize that it is an ECE and not an ACE.

Expected (average) indirect efect. The expected indirect efect compares the expected outcome of unit i when unit $j ^ { \dag } \mathrm { s }$ treatment is changed, keeping all other treatment assignments the same. For $j \in N _ { i }$ , define

$$
E _ { i } ^ { + j } = f _ { i } ( \mathbf { Z } _ { N _ { i } \backslash \{ j \} } , Z _ { j } = 1 ) , \qquad E _ { i } ^ { - j } = f _ { i } ( \mathbf { Z } _ { N _ { i } \backslash \{ j \} } , Z _ { j } = 0 ) ,
$$

as the exposure of unit i when neighbor $j$ is treated or untreated, holding all others fixed. Then,

$$
\tau _ { \mathrm { E I E } } ( Q ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { j \in N _ { i } } \mathbb { E } _ { Q } \Bigl [ Y _ { i } \bigl ( Z _ { i } , E _ { i } ^ { + j } \bigr ) \bigr ) - Y _ { i } \bigl ( Z _ { i } , E _ { i } ^ { - j } \bigr ) \Bigr ] .\tag{3.11}
$$

This is an aggregation of neighbor-specific ECE contrasts. For each ordered pair $( i , j )$ with $j \in N _ { i }$ , define

$$
D _ { i j } ^ { + } ( { \bf Z } ) = ( Z _ { i } , E _ { i } ^ { + j } ) , \qquad D _ { i j } ^ { - } ( { \bf Z } ) = ( Z _ { i } , E _ { i } ^ { - j } ) .
$$

The expected indirect efect is also known as the Average Indirect Efect. The arbitrary interference version of this estimand was introduced in Hu et al. (2022).

Proposition 2 (Policy derivative representation) Under a Bernoulli policy $\mathcal { Q } _ { \pi }$ with $\begin{array} { r } { \mathbf Z \sim \prod _ { i } \mathrm { B e r n } ( \pi _ { i } ) } \end{array}$ , writing $\begin{array} { r } { \mu ( { \boldsymbol \pi } ) = n ^ { - 1 } \sum _ { i } \mathbb { E } _ { \boldsymbol \pi } [ Y _ { i } ( Z _ { i } , e _ { i } ( \mathbf { Z } _ { N _ { i } } ) ) ] } \end{array}$ for the expected population outcome, the total policy derivative $\tau _ { \mathrm { T P E } } ( \pi ) \ : = \ : \sum _ { i } \partial \mu ( \pi ) / \partial \pi _ { j }$ decomposes as $\tau _ { \mathrm { T P E } } ( \pi ) =$ $\tau _ { \mathrm { E D E } } ( \pi ) + \tau _ { \mathrm { E I E } } ( \pi )$ , recovering the Average Direct Efect / Average Indirect Efect decomposition of Hu et al. (2022) under arbitrary interference as a special case. Under symmetric exposure mappings, EIE has a simpler finite-diference form; details are in Appendix C.6.

Expected policy efect. The general definition also allows the two arms of the contrast to be averaged under diferent allocation policies. Let $\mathcal { Q } ^ { + }$ and ${ \mathfrak { Q } } ^ { - }$ be two policies, for example Bernoulli policies with treatment probabilities $p _ { + }$ and $p _ { \cdot }$ <sub>−</sub>. The expected policy efect is

$$
\tau _ { \mathrm { p o l i c y } } ( \boldsymbol { Q } ^ { + } , \boldsymbol { Q } ^ { - } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ \mathbb { E } _ { \boldsymbol { Q } ^ { + } } Y _ { i } \{ D _ { i } ( \mathbf { Z } ) \} - \mathbb { E } _ { \boldsymbol { Q } ^ { - } } Y _ { i } \{ D _ { i } ( \mathbf { Z } ) \} \right] .\tag{3.12}
$$

This is the special case of (3.8) with $D _ { i } ^ { + } ( { \bf Z } ) = D _ { i } ^ { - } ( { \bf Z } ) = D _ { i } ( { \bf Z } )$ , but with ${ \mathfrak { Q } } ^ { + } \neq { \mathfrak { Q } } ^ { - }$ It represents questions such as comparing an allocation policy with 50 percent treatment probability to one with 20 percent treatment probability.

Mixed efects that combine policy-averaged and fixed-baseline potential outcomes (EIC, EIT, ETE) are defined in Appendix C.1.

## 3.3 Algebraic representation of ACEs and ECEs

We now represent the two classes of estimands using the non-parametric linear decomposition of the causal response surface.

Proposition 3 (Representation of fixed and expected efects) Under the non-parametric linear decomposition of Theorem $^ { 1 , }$

$$
Y _ { i } ( z , e ) = \alpha _ { i } + \beta _ { i } z + B _ { i } ( e ) + z C _ { i } ( e ) , \qquad B _ { i } ( 0 ) = C _ { i } ( 0 ) = 0 ,
$$

the average causal efects can be written as

$$
\tau _ { \mathrm { { D T E } } } = \frac { 1 } { n } \sum _ { i } \beta _ { i } ,
$$

$$
\pi \mathrm { { / } } \mathrm { { E } } ( z ; e _ { a } , e _ { b } ) = \frac { 1 } { n } \sum _ { i } \{ B _ { i } ( e _ { b } ) - B _ { i } ( e _ { a } ) + z \left( C _ { i } ( e _ { b } ) - C _ { i } ( e _ { a } ) \right) \} ,
$$

$$
\tau _ { \mathrm { I E C } } ( e ) = \frac { 1 } { n } \sum _ { i } B _ { i } ( e ) ,
$$

$$
\pi \mathrm { { r } } \mathrm { { T } } \mathrm { { E } } \mathrm { { T } }  \big ( e \big ) = \frac { 1 } { n } \sum _ { i } \{ B _ { i } ( e ) + C _ { i } ( e ) \} ,
$$

$$
\tau _ { \mathrm { T T E } } = \frac { 1 } { n } \sum _ { i } \{ \beta _ { i } + B _ { i } ( 1 ) + C _ { i } ( 1 ) \} .
$$

For an allocation policy $\mathcal { Q } _ { i }$ , let $\begin{array} { r } { \bar { B } _ { \mathcal { Q } } = n ^ { - 1 } \sum _ { i } \mathbb { E } _ { \mathcal { Q } } [ B _ { i } ( E _ { i } ) ] } \end{array}$ and $\begin{array} { r } { \bar { C } _ { \mathcal { Q } } = n ^ { - 1 } \sum _ { i } \mathbb { E } _ { \mathcal { Q } } [ C _ { i } ( E _ { i } ) ] } \end{array}$ Then

$$
\tau _ { \mathrm { E D E } } ( \mathcal { Q } ) = \frac { 1 } { n } \sum _ { i } \beta _ { i } + \bar { C } _ { \mathcal { Q } } .
$$

The mixed efects EIC, EIT, and ETE satisfy analogous expressions: $\tau _ { \mathrm { E I C } } ( \mathcal { Q } ) ~ = ~ \bar { B } _ { \mathcal { Q } }$ $\tau _ { \mathrm { E I T } } ( \mathcal { Q } ) = \bar { B } _ { \mathcal { Q } } + \bar { C } _ { \mathcal { Q } }$ , and $\begin{array} { r } { \tau _ { \mathrm { E T E } } ( \mathcal { Q } ) = n ^ { - 1 } \sum _ { i } \beta _ { i } + \bar { B } _ { \mathcal { Q } } + \bar { C } _ { \mathcal { Q } } , } \end{array}$ see Appendix C.1.

These representations connect the estimands directly to the components of the nonparametric response-surface decomposition. While fixed-exposure ACEs select values of $B _ { i }$ and $C _ { i }$ at specific exposure levels, ECEs average these same components over the exposure distribution induced by the allocation policy. Specifically, τ isolates the direct efect component $\beta _ { i }$ , whereas τ<sub>IEC</sub> and τ<sub>IET</sub> capture the pure indirect and indirect-plus-interference components, respectively. Notably, the expected direct efect encapsulates both the pure direct component and the treatment–interference interaction component, the latter being 0 under additivity. This yields the following corollary:

Corollary 4 Under additivity of the causal response surface, $\tau _ { \mathrm { E D E } } ( \mathcal { Q } ) = \tau _ { \mathrm { D T E } }$ for every allocation policy $\mathcal { Q }$

Without additivity, the expected direct efect contains the policy-weighted interaction component $\bar { C } _ { \mathcal { Q } }$ and is policy-dependent: estimators that target EDE do not target the fixed-exposure DTE. The total treatment efect also has a decomposition:

Proposition 5 (ACE total efect decomposition) The total treatment efect decomposes as

$$
\tau _ { \mathrm { T T E } } = \tau _ { \mathrm { D T E } } + \tau _ { \mathrm { I E T } } .
$$

Under additivity $( C _ { i } \equiv 0 ) , \tau _ { \mathrm { I E T } } = \tau _ { \mathrm { I E C } } , s o \tau _ { \mathrm { T T E } } = \tau _ { \mathrm { D T E } } + \tau _ { \mathrm { I E C } } .$

Decompositions involving the mixed efects EIC, EIT, and ETE are given in Appendix C.1.

## 3.4 Relevant and Nuisance Potential Outcomes

Another distinction between average causal efects (ACEs) and expected causal efects (ECEs) ties to which entries of S define the estimand. In a non-parametric framework, no outcome model is imposed on the distinct potential outcomes: observing one element of S provides no information about another distinct element of S. Thus, potential outcomes that do not enter the definition of an estimand are uninformative for its non-parametric estimation, regardless of how often an experimental design reveals them. We formalize this distinction by partitioning the causal response surface relative to a target estimand.

Definition 6 (Relevant and nuisance potential outcomes) Fix a causal estimand $\tau ,$ defined as a functional of the causal response surface S. A potential outcome $Y _ { i } ( z , e )$ is relevant $f o r \tau$ if the functional defining τ depends on $Y _ { i } ( z , e )$ . The relevant set is

$$
\mathcal { R } _ { \tau } = \{ Y _ { i } ( z , e ) \in \mathbb { S } : \tau \ d e p e n d s \ o n \ Y _ { i } ( z , e ) \} .
$$

The complement $\mathcal { R } _ { \tau } ^ { c } = \mathbb { S } \setminus \mathcal { R } _ { \tau }$ is the set of nuisance potential outcomes $f o r \tau$

For ACEs, the relevant set $\mathcal { R } _ { \tau }$ is typically a sparse subset of the causal response surface. For example, a direct treatment efect depends only on

$$
\{ Y _ { i } ( 1 , 0 ) , Y _ { i } ( 0 , 0 ) \} _ { i = 1 } ^ { n } ,
$$

whereas a total treatment efect depends only on

$$
\{ Y _ { i } ( 1 , 1 ) , Y _ { i } ( 0 , 0 ) \} _ { i = 1 } ^ { n } .
$$

All other entries of S are nuisance for non-parametric estimation of these estimands. For example, if a design exposes a unit to an intermediate exposure level $e ,$ revealing $Y _ { i } ( 1 , e )$ that observation provides no non-parametric information about target potential outcomes such as $Y _ { i } ( 1 , 1 )$ or $Y _ { i } ( 1 , 0 )$ . Without an outcome model relating distinct potential outcomes of $\mathbb { S }$ across treatment or exposure levels, observations from nuisance potential outcomes cannot reduce variance for the target estimand. They may instead waste probability mass, contribute bias, or require outcome modeling assumptions that go beyond non-parametric estimation.

For ECEs, the allocation policy is part of the definition of the estimand. Any treatment– exposure combination that is assigned positive probability under the allocation policy is included in the relevant set $\mathcal { R } _ { \tau }$ of an ECE. For example, the relevant set for the expected direct efect $\tau _ { \mathrm { E D E } } ( \mathcal { Q } )$ is the set of $Y _ { i } ( 1 , e )$ and $Y _ { i } ( 0 , e )$ for each exposure level e in the support of the exposure distribution induced by policy Q. When the policy induces positive probability on every treatment–exposure combination in the specified representation, the relevant set is the entire causal response surface.

This diference in relevant potential outcomes leads to diferent non-parametric estimation problems, as discussed in Section 4. For ACEs, the design must reveal the relevant potential outcomes. Non-parametric estimation thus requires the propensity scores for those combinations to be bounded away from zero. When the probability of revealing the relevant potential outcomes is rare under the design, inverse-probability weights become unstable and variance can be large. Consequently, non-parametric estimation of ACEs often requires specialized designs that place suficient probability mass on the relevant potential outcomes; such designs are discussed in Section 6. For ECEs, the estimand instead averages over the treatment–exposure combinations induced by the policy. The relevant support is therefore determined by the policy itself. In this case, the main variance bottleneck is not the probability of observing a narrow set of fixed treatment–exposure entries, but the overlap between the observed assignment design and the policy. Importance-weighted HT estimators use outcomes across the support of the policy, with precision governed by policy-to-design weights, network dependence, and the extent of support overlap.

## 3.5 Conditional expected efects

The two classes introduced in Sections 3.1 and 3.2 also clarify conditional estimands that have appeared in the literature on partial interference. Hudgens and Halloran (2008) introduced causal efects under partial interference based on conditional expectations of potential outcomes. In their setting, units are partitioned into disjoint interference groups, and causal efects are defined by conditioning on the treatment assignment within the group. Formally, one considers conditional potential outcomes of the form

$$
\mathbb { E } \big [ Y _ { i } ( z , e ) \big | Z _ { i } = z \big ] ,
$$

where the expectation is taken with respect to the distribution of exposure in the group, conditional on the individual treatment. Estimands are then defined as contrasts between such conditional expectations. These estimands can be viewed as conditional expected causal efects, which difer from the marginal expected efects defined in Section 3.2. Conditional estimands depend on the randomization scheme within groups, and are most natural under two-stage designs with partial interference in which treatment probabilities are assigned at the group level. In arbitrary network interference, or under designs where a unit’s treatment and exposure are dependent, conditioning on $Z _ { i }$ also changes the exposure distribution. The resulting estimand remains well defined, but its interpretation depends on the conditional allocation distribution; see S¨avje et al. (2017) for a discussion.

## 4 Estimation of ACEs and ECEs

In this section, we study design-unbiased estimators of ACEs and ECEs. In particular, under a fixed experimental design P, we derive upper bounds for the variance of canonical Horvitz–Thompson (HT) estimators as well as lower bounds on the design variance of any unbiased estimator. Finally, we establish the conditions under which these estimators achieve design consistency. Under a growing sequence of finite populations indexed by n, an estimator $\hat { \tau } _ { n }$ is design-MSE consistent for $\tau _ { n }$ if

$$
\mathbb { E } _ { P _ { n } } \{ ( \hat { \tau } _ { n } - \tau _ { n } ) ^ { 2 } \}  0 .
$$

The estimation machinery applies to any choice of interference graph and exposure mapping. As examples, we study estimation under the three examples introduced in Section 2: arbitrary interference $( Z _ { i } , \mathbf { Z } _ { - i } )$ , neighborhood interference $( Z _ { i } , { \bf Z } _ { N _ { i } } )$ , and exposure mappings $\left( Z _ { i } , e _ { i } ( { \bf Z } _ { N _ { i } } ) \right)$ ). Throughout this section, no outcome modeling assumptions are imposed.

Once the interference neighborhood and exposure mapping have been fixed, non-parametric estimation treats the distinct potential outcomes of S as unrestricted. Observing one distinct potential outcome $Y _ { i } ( z , e )$ does not, without an outcome model, identify or reduce uncertainty about another distinct potential outcome $Y _ { i } ( z ^ { \prime } , e ^ { \prime } )$

## 4.1 Dependence from interference and design

Consider a generic estimator $\hat { \tau } _ { n }$ that is a function of random variables $X _ { 1 } , \ldots , X _ { n }$ . In the design-based setting, the potential-outcome table is fixed, so any observed-outcome term $Y _ { i } ^ { o b s } = Y _ { i } ( { \bf Z } )$ is a measurable function of the treatment assignment vector. Thus each $X _ { i }$ can be written generally as $X _ { i } = g _ { i } ( \mathbf { Z } )$ . The variance bounds in the sequel use dependence graphs for the random variables {X<sub>i</sub>}.

Definition 7 (Dependence graph and average dependence) Consider random variables $\{ X _ { i } \} _ { i = 1 } ^ { n }$ . A graph $G _ { \mathrm { d e p } } = \left( V , E _ { \mathrm { d e p } } \right)$ is a dependence graph of $\{ X _ { i } \} _ { i = 1 } ^ { n }$ if the following holds: whenever two disjoint sets of vertices A, $B \subseteq V$ have no edge between them, the collections $\{ X _ { i } : i \in A \}$ and $\{ X _ { j } : j \in B \}$ are independent. We write $c _ { i j } = 1$ when i and j are adjacent in this dependence graph and $c _ { i j } = 0$ otherwise. The dependence degree of i is

$$
c _ { i } = \sum _ { j \neq i } c _ { i j } ,
$$

and the average dependence degree is

$$
c _ { \mathrm { a v g } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } c _ { i } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } c _ { i j } .
$$

Remark 3 (Estimator dependence versus interference neighborhoods) The dependence graph is an object associated with the estimator $\hat { t } a u _ { n }$ and the random variables $X _ { i } ,$ not directly with the causal estimand. $I f X _ { i }$ depends only on a subvector of the treatment assignment, we may write $X _ { i } = g _ { i } ( \mathbf { Z } _ { A _ { i } } )$ , where $A _ { i } \subseteq \{ 1 , \dots , n \}$ is an assignment-dependence set for that estimator variable. The set $A _ { i }$ includes the treatment assignment coordinates used by the observed outcome and by any random weights or normalizing factors. When an interference graph with neighborhoods $N _ { i }$ is specified, write $\mathcal { C } _ { i } = \{ i \} \cup N _ { i }$ for the closed interference neighborhood; this includes arbitrary interference as the special case $N _ { i } = \{ 1 , \dots , n \} \setminus \{ i \}$ For the local HT/IPW estimators below, the observed outcome has the form $Y _ { i } ( Z _ { i } , { \bf Z } _ { N _ { i } } )$ and the weights are also a function of $( Z _ { i } , { \bf Z } _ { N _ { i } } )$ . Hence $A _ { i } \subseteq { \mathcal { C } } _ { i }$ . Ratio estimators such as Hajek estimators may introduce additional assignment dependence because their weights depend on the full treatment assignment vector Z.

Typically, the variables $X _ { i }$ and $X _ { j }$ can be dependent for two reasons: their treatmentassignment index sets $A _ { i }$ and $A _ { j }$ can overlap because of interference, and the randomization design can directly induce dependence among treatment assignments, even when $A _ { i }$ and $A _ { j }$ do not overlap. Below, we separate these two sources into interference-overlap dependence and residual design dependence.

Under independent treatment assignment, there is no residual design dependence. The remaining dependence is induced by overlap of the assignment coordinates on which the variables $X _ { i }$ depend: if $X _ { i } = g _ { i } ( \mathbf { Z } _ { A _ { i } } )$ , then $\mathbb { 1 } \{ A _ { i } \cap A _ { j } \ \neq \ \varnothing \}$ gives a valid dependence edge under independent assignment. The following lemma records the closed-neighborhood specialization used for the local HT/IPW estimators studied below.

Lemma 8 (Dependence induced by local estimator variables) Suppose the treatment assignments are mutually independent and $X _ { i }$ is a function of $\mathbf { Z } _ { \mathcal { C } _ { i } }$ , where $\mathcal { C } _ { i } = \{ i \} \cup N _ { i }$ is the closed interference neighborhood of unit i. Then a valid dependence graph of $\{ X _ { i } \}$ is obtained by setting

$$
c _ { i j } = \mathbb { 1 } \{ { \mathcal { C } } _ { i } \cap { \mathcal { C } } _ { j } \neq \emptyset \} , \qquad i \neq j .
$$

If the interference graph is undirected, $d _ { i } = | N _ { i } |$ , and $B _ { 2 } ( i )$ denotes the two-hop interference neighborhood of i, including i, then

$$
c _ { i } = | B _ { 2 } ( i ) | - 1
$$

and

$$
b _ { 2 , \mathrm { a v g } } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \{ | B _ { 2 } ( i ) | - 1 \} = c _ { \mathrm { a v g } } , \qquad { \bar { d } } \leq b _ { 2 , \mathrm { a v g } } \leq \operatorname* { m i n } \left\{ n - 1 , \frac { 1 } { n } \sum _ { i = 1 } ^ { n } d _ { i } ^ { 2 } \right\} , \qquad { \bar { d } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } d _ { i } .
$$

Thus, for independent assignment on an undirected interference graph, the average dependence degree of estimators that use local weights is controlled by the second moment of the degree distribution of the interference graph rather than by the average degree alone. For bounded-degree and sparse Erd˝os–Renyi interference graphs with bounded expected degree, $c _ { \mathrm { a v g } } = O ( 1 )$ . On locally tree-like d-regular graphs, $c _ { \mathrm { a v g } } \asymp d ^ { 2 }$ . On a star graph, $\bar { d } = O ( 1 )$ but $c _ { \mathrm { a v g } } \asymp n$ , because all leaves have overlapping closed neighborhoods through the hub. On complete graphs, $c _ { \mathrm { a v g } } = n - 1$

Under dependent randomization designs, two random variables $X _ { i }$ and $X _ { j }$ can be dependent even when their closed interference neighborhoods do not overlap. We measure this residual design-induced dependence by the following coeficient.

Definition 9 (Design dependence coeficient) For two assignment-index sets $A , B \subseteq$ $\{ 1 , \ldots , n \}$ , define

$$
\rho _ { P } ( A , B ) = \operatorname* { s u p } _ { \| f \| _ { \infty } \leq 1 , \| g \| _ { \infty } \leq 1 } \left| \operatorname { C o v } _ { P } \{ f ( \mathbf { Z } _ { A } ) , g ( \mathbf { Z } _ { B } ) \} \right| ,
$$

where the supremum is over all measurable functions of the indicated subvectors.

For the local HT/IPW variance bounds below, we use

$$
\rho _ { i j } = \rho _ { P } ( \mathcal { C } _ { i } , \mathcal { C } _ { j } ) ,
$$

and

$$
r _ { \mathrm { a v g } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } ( 1 - c _ { i j } ) \rho _ { i j } .
$$

Here $\rho _ { i j }$ is the pairwise design-dependence coeficient between the two closed-neighborhood assignment subvectors, while $r _ { \mathrm { a v g } }$ averages only the residual design dependence among pairs whose closed neighborhoods do not overlap. If the design is independent assignment and $\mathcal { C } _ { i } \cap \mathcal { C } _ { j } = \emptyset$ , then $\rho _ { i j } = 0$ . Complete randomization, cluster randomization, paired designs, and other dependent designs can have $\rho _ { i j } > 0$ even for disjoint closed neighborhoods. These quantities separate overlap of assignment coordinates from dependence induced directly by the randomization design. The variance bounds below therefore separate three sources of variation: the diagonal scale of the HT or importance weights, dependence generated by overlapping closed interference neighborhoods, and residual dependence generated by the randomization design.

## 4.2 ACE estimation

We first consider nonparametric estimation of an ACE

$$
\tau _ { \mathrm { A C E } } ( d ^ { + } , d ^ { - } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \{ Y _ { i } ( d _ { i } ^ { + } ) - Y _ { i } ( d _ { i } ^ { - } ) \} .
$$

Write $\pi _ { i } ( d ) = P \{ D _ { i } ( \mathbf { Z } ) = d \}$ for the design-induced propensity that unit i is observed in treatment–exposure combination d. The generic Horvitz–Thompson estimator is

$$
\hat { \tau } _ { \mathrm { A C E } } ^ { H T } ( d ^ { + } , d ^ { - } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Y _ { i } ^ { o b s } \left\{ \frac { \mathbb { 1 } \{ D _ { i } ( \mathbf { Z } ) = d _ { i } ^ { + } \} } { \pi _ { i } ( d _ { i } ^ { + } ) } - \frac { \mathbb { 1 } \{ D _ { i } ( \mathbf { Z } ) = d _ { i } ^ { - } \} } { \pi _ { i } ( d _ { i } ^ { - } ) } \right\} .
$$

This is the exposure-mapping Horvitz–Thompson estimator of Aronow and Samii (2017), written here for the generic ACE notation. For the variance bound, let

$$
H _ { i } = \frac { 1 } { \pi _ { i } ( d _ { i } ^ { + } ) } + \frac { 1 } { \pi _ { i } ( d _ { i } ^ { - } ) } .
$$

Proposition 10 (ACE Horvitz–Thompson estimator) Assume $\pi _ { i } ( d _ { i } ^ { + } ) > 0$ and $\pi _ { i } ( d _ { i } ^ { - } ) >$ 0 for all i. Then $\hat { \tau } _ { \mathrm { A C E } } ^ { H T } ( d ^ { + } , d ^ { - } )$ is design-unbiased for $\tau _ { \mathrm { A C E } } ( d ^ { + } , d ^ { - } )$ for every potentialoutcome table. If, in addition, $| Y _ { i } ( d ) | \leq M$ and the ith summand depends on the assignment only through $\mathbf { Z } _ { \mathcal { C } _ { i } }$ , then

$$
\mathrm { V a r } \{ \hat { \tau } _ { \mathrm { A C E } } ^ { H T } ( d ^ { + } , d ^ { - } ) \} \leq \frac { M ^ { 2 } } { n ^ { 2 } } \left\{ \sum _ { i = 1 } ^ { n } H _ { i } + \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } c _ { i j } H _ { i } H _ { j } + \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } ( 1 - c _ { i j } ) \rho _ { i j } H _ { i } H _ { j } \right\} ,
$$

and if $H _ { i } \leq H$ for all i, then

$$
\mathrm { V a r } \{ \hat { \tau } _ { \mathrm { A C E } } ^ { H T } ( d ^ { + } , d ^ { - } ) \} \leq M ^ { 2 } \left\{ \frac { H } { n } + H ^ { 2 } \frac { c _ { \mathrm { a v g } } + r _ { \mathrm { a v g } } } { n } \right\} .
$$

Consequently, along a sequence of finite populations, the estimator is design-MSE consistent if $H + H ^ { 2 } ( c _ { \mathrm { a v g } } + r _ { \mathrm { a v g } } ) = o ( n )$

The proof is given in Appendix D. This proposition separates three sources of variation in the HT estimator: the inverse propensity scores of the relevant potential outcomes $Y _ { i } ( d _ { i } ^ { + } )$ and $Y _ { i } ( d _ { i } ^ { - } ) ~ ( H _ { i } )$ , the dependence generated by overlapping closed interference neighborhoods $\left( c _ { \mathrm { a v g } } \right)$ , and residual dependence from the randomization design $( r _ { \mathrm { a v g } } )$ . The next proposition shows that the inverse-propensity part is not merely an artifact of the HT estimator.

Proposition 11 (Inverse propensity scores for ACEs) Fix the design $P _ { n }$ and the induced treatment–exposure combinations $D _ { i } ( \mathbf { Z } )$ . Suppose that $\hat { \tau } _ { n }$ is design-unbiased for $\tau _ { \mathrm { A C E } } ( d ^ { + } , d ^ { - } )$ for every bounded potential-outcome table. Then

$$
\operatorname* { s u p } _ { | Y | \leq M } \operatorname { V a r } _ { P _ { n } } ( \hat { \tau } _ { n } ) \geq \frac { M ^ { 2 } } { n ^ { 2 } } \operatorname* { m a x } _ { i } \left\{ \frac { 1 } { \pi _ { i } ( d _ { i } ^ { + } ) } - 1 , \frac { 1 } { \pi _ { i } ( d _ { i } ^ { - } ) } - 1 \right\} .
$$

The proof is given in Appendix D. This coordinate-wise bound demonstrates that no unbiased estimator of a fixed-exposure ACE can avoid the smallest propensity scores among the relevant treatment–exposure combinations. Below, we show that the diagonal term in the upper bound of Proposition 10 scales identically, and is usually the dominant term when compared to of-diagonal network interference and design dependence terms, giving a matching lower bound for regular graph-unaware designs.

Estimation of DTE and TTE under classical designs. We now examine when the Horvitz–Thompson estimator can consistently estimate the direct and total treatment effects under standard graph-unaware designs. Proposition 11 provides a matching lower bound: no unbiased estimator can avoid the inverse-propensity scale H, so the variance conditions we derive are necessary for any design-unbiased estimator, not just the HT. The analysis covers independent designs (Bernoulli, heterogeneous Bernoulli) and exchangeable designs (complete randomization) alike. Proposition 31 (Appendix D.13) shows they share a key property: any specific treatment configuration on a units has probability exp $\{ - \Theta ( a ) \}$ , so its inverse propensity scales as $\exp \{ \Theta ( a ) \}$ . Here a is the number of units the exposure condition constrains — $d _ { i } + 1$ for neighborhood-interference targets, n for arbitrary-interference targets. This exponential-in-a scaling is the central bottleneck for fixed-exposure ACE estimation and is the only design feature needed below. The variance bound of Proposition 10 formalizes this bottleneck: under independent assignment,

$$
\mathrm { V a r } \{ \hat { \tau } _ { \mathrm { A C E } } ^ { H T } \} \leq M ^ { 2 } \big \{ H / n + H ^ { 2 } c _ { \mathrm { a v g } } / n \big \} ,
$$

where $H = \mathrm { m a x } _ { i } ( 1 / \pi _ { i } ( d _ { i } ^ { + } ) + 1 / \pi _ { i } ( d _ { i } ^ { - } ) )$ and $c _ { \mathrm { a v g } }$ is the average dependence degree (Lemma 8). We write $\mathcal { V } _ { n } ( H , c ) = H / n + H ^ { 2 } c / n$ for this scale, suppressing M. An additional residual dependence term appears under exchangeable designs (Appendix D.3), but it does not change the conclusions.

Tables 1 and 2 summarize H and the matching lower bound for the DTE and TTE under diferent representations. The column “HT $H ^ { \dprime }$ records the inverse-propensity scale and “LB $H ^ { \dprime }$ the matching lower bound. The variance scale column combines H with the dependence degree $b _ { 2 , \mathrm { a v g } }$ (which equals $c _ { \mathrm { a v g } }$ under independent assignment). When H is exponential, the polynomial dependence is negligible by comparison. For the exposure-mapping rows, let $\begin{array} { r } { S _ { i } = \sum _ { j \in N _ { i } } Z _ { j } } \end{array}$ denote the number of treated neighbors and $U _ { i } = \mathbb { 1 } \{ S _ { i } \geq 1 \}$ the binary any-treated indicator.

Direct treatment efect. The relevant potential outcomes for the DTE are $Y _ { i } ( 1 , 0 _ { N _ { i } } )$ and $Y _ { i } ( 0 , 0 _ { N _ { i } } )$ (Definition 6). Under neighborhood interference, the treated arm $( 1 , 0 _ { N _ { i } } )$ requires the unit to be treated and all its neighbors to be untreated. Under Bernoulli assignment,

$$
\pi _ { i } ( 1 , 0 _ { N _ { i } } ) = p ( 1 - p ) ^ { d _ { i } } , \qquad \pi _ { i } ( 0 , 0 _ { N _ { i } } ) = ( 1 - p ) ^ { d _ { i } + 1 } ,
$$

Table 1: Direct-efect ACEs under Bernoulli assignment. The rows show the DTE target under arbitrary interference, neighborhood interference, and count/fraction or binary exposure representations.
<table><tr><td>Representation</td><td>Target conditions</td><td>HT H</td><td>LB H</td><td>HT variance scale</td><td></td></tr><tr><td>Arbitrary interference</td><td> $\overline { { ( 1 , \mathbf { 0 } _ { - i } ) \ \mathrm { v s . } \ ( 0 , \mathbf { 0 } _ { - i } ) } }$ </td><td></td><td> $\overline { { e ^ { \Theta ( n ) } } }$   $\overline { { e ^ { \Theta ( n ) } } }$ </td><td></td><td> $\overline { { \mathcal { V } _ { n } ( e ^ { \Theta ( n ) } , n ) } }$ </td></tr><tr><td>Neighborhood interference</td><td> $( 1 , { \bf 0 } _ { N _ { i } } ) \mathrm { ~ v s . ~ } \left( 0 , { \bf 0 } _ { N _ { i } } \right)$ </td><td> $e ^ { \Theta ( d _ { \mathrm { m a x } } ) }$ </td><td> $e ^ { \Theta ( d _ { \mathrm { m a x } } ) }$ </td><td> $\mathcal { V } _ { n } \big ( e ^ { \Theta ( d _ { \mathrm { m a x } } ) } , b _ { \mathrm { 2 , a v g } } \big )$ </td><td></td></tr><tr><td>Count/fraction exposure</td><td> $( Z _ { i } , S _ { i } ) = ( 1 , 0 ) \ \mathrm { v s } .$ </td><td>(0,0)</td><td> $e ^ { \Theta ( d _ { \mathrm { m a x } } ) }$   $e ^ { \Theta ( d _ { \mathrm { m a x } } ) }$ </td><td></td><td> $\mathcal { V } _ { n } \big ( e ^ { \Theta ( d _ { \mathrm { m a x } } ) } , b _ { \mathrm { 2 , a v g } } \big )$ </td></tr><tr><td>Binary any-treated exposure</td><td> $( Z _ { i } , U _ { i } ) = ( 1 , 0 ) ~ \mathrm { v s . } ~ ( 0 , 0 )$ </td><td></td><td> $e ^ { \Theta ( d _ { \mathrm { m a x } } ) }$ </td><td> $e ^ { \Theta ( d _ { \mathrm { m a x } } ) }$ </td><td> $\mathcal { V } _ { n } \big ( e ^ { \Theta ( d _ { \mathrm { m a x } } ) } , b _ { \mathrm { 2 , a v g } } \big )$ </td></tr></table>

Table 2: Total-efect ACEs under Bernoulli assignment. Exposure-mapping rows show representative treated-exposure versus no-exposure targets.
<table><tr><td>Representation</td><td>Target conditions</td><td>HT H</td><td>LB H</td><td>HT variance scale</td></tr><tr><td>Arbitrary interference</td><td> $( 1 , 1 _ { - i } ) \ \mathrm { v s . } \ ( 0 , \mathbf { 0 } _ { - i } )$ </td><td> $\overline { { e ^ { \Theta ( n ) } } }$ </td><td> $\overline { { e ^ { \Theta ( n ) } } }$ </td><td> $\overline { { \mathcal { V } _ { n } ( e ^ { \Theta ( n ) } , n ) } }$ </td></tr><tr><td>Neighborhood interference</td><td> $( 1 , \mathbf { 1 } _ { N _ { i } } )$  vS.  $( 0 , { \bf 0 } _ { N _ { i } } )$ </td><td> $e ^ { \Theta ( d _ { \mathrm { m a x } } ) }$ </td><td> $e ^ { \Theta ( d _ { \mathrm { m a x } } ) }$ </td><td> $\mathcal { V } _ { n } \big ( e ^ { \Theta ( d _ { \mathrm { m a x } } ) } , b _ { \mathrm { 2 , a v g } } \big )$ </td></tr><tr><td>Count/fraction exposure</td><td> $( Z _ { i } , S _ { i } ) = ( 1 , d _ { i } )$  vs. (0, 0)</td><td> $e ^ { \Theta ( d _ { \mathrm { m a x } } ) }$ </td><td> $e ^ { \Theta ( d _ { \mathrm { m a x } } ) }$ </td><td> $\mathcal { V } _ { n } \big ( e ^ { \Theta ( d _ { \mathrm { m a x } } ) } , b _ { \mathrm { 2 , a v g } } \big )$ </td></tr><tr><td>Binary any-treated exposure</td><td> $( Z _ { i } , U _ { i } ) = ( 1 , 1 )$  vs. (0,0)</td><td> $e ^ { \Theta ( d _ { \mathrm { m a x } } ) } e ^ { \Theta ( d _ { \mathrm { m a x } } ) }$ </td><td></td><td> $\mathcal { V } _ { n } \big ( e ^ { \Theta ( d _ { \mathrm { m a x } } ) } , b _ { \mathrm { 2 , a v g } } \big )$ </td></tr></table>

so $H _ { i } = \Theta ( e ^ { \gamma d _ { i } } )$ with $\gamma = - \log ( 1 - p )$ and $H = e ^ { \Theta ( d _ { \mathrm { m a x } } ) }$ . The same exponential scale applies under count, fraction, and binary any-treated exposure mappings for the DTE, since all encode the no-exposure condition as $S _ { i } = 0 \mathrm { o r } U _ { i } = 0$

Total treatment efect. The relevant potential outcomes for the TTE are $Y _ { i } ( 1 , { \bf 1 } _ { N _ { i } } )$ and $Y _ { i } ( 0 , { \mathbf { 0 } } _ { N _ { i } } )$ (Definition 6). The treated arm requires all neighbors to be treated and the control arm requires all neighbors to be untreated, so H is exponential in $d _ { \mathrm { m a x } }$ for every representation in Table 2. Crucially, even the binary any-treated exposure mapping does not remove this scale: the control arm (0, 0) requires all neighbors to be untreated, with propensity $( 1 - p ) ^ { d _ { i } + 1 }$ decaying exponentially with $d _ { i }$

Implications for practice. Under classical graph-unaware designs, the relevant potential outcomes for the $\mathrm { D T E } \mathrm { - } Y _ { i } ( 1 , 0 )$ and $Y _ { i } ( 0 , 0 )$ (Definition 6)—occur with probability $\exp \{ - \Theta ( d _ { i } ) \}$ , leading to an inverse-propensity barrier exponential in the neighborhood degree. For the DTE,

$$
H \asymp e ^ { \gamma d _ { \mathrm { m a x } } } , \qquad \gamma = - \log ( 1 - p ) ,
$$

and the variance bound vanishes only when $d _ { \operatorname* { m a x } } \lesssim \frac { 1 } { 2 \gamma }$ log n, i.e., when the degree grows at most logarithmically in n. This condition is necessary for any unbiased estimator, not just the HT (Proposition 11). For the TTE, the exponential barrier applies across all exposure mappings (Table 2) because both relevant potential outcomes— $- Y _ { i } ( 1 , { \bf 1 } )$ and $Y _ { i } ( 0 , { \mathbf { 0 } } ) -$ require every neighbor to be in a specific treatment state.

To overcome this exponential barrier induced by classical designs, we need specialized graph aware designs to estimate ACEs. Section 6 develops such designs: independent-set designs isolate the relevant potential outcomes for the DTE on ego units, while cluster randomized designs increase the probability of the all-treated and all-control conditions needed for the TTE.

## 4.3 ECE estimation

We now consider estimation of ECEs. Following the definition in Section 3.2, let ${ \mathcal { Q } } ^ { + } , { \mathcal { Q } }$ be two allocation policies and $D _ { i } ^ { + } ( \mathbf { Z } ) , D _ { i } ^ { - } ( \mathbf { Z } ) \in \mathcal { X } _ { i }$ the corresponding random treatment– exposure combinations. Define the policy propensities scores as

$$
\lambda _ { i } ^ { + } ( d ) = { \mathcal { Q } } ^ { + } \{ D _ { i } ^ { + } ( \mathbf { Z } ) = d \} , \qquad \lambda _ { i } ^ { - } ( d ) = { \mathcal { Q } } ^ { - } \{ D _ { i } ^ { - } ( \mathbf { Z } ) = d \} ,
$$

so that the ECE can be written as

$$
\tau _ { \mathrm { E C E } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { d \in \mathcal { X } _ { i } } Y _ { i } ( d ) \{ \lambda _ { i } ^ { + } ( d ) - \lambda _ { i } ^ { - } ( d ) \} .
$$

Its Horvitz–Thompson estimator has the same form as the ACE estimator of Section 4.2, with the fixed inverse propensity scores of the relevant potential outcomes replaced by importance weights that are ratios of the policy propensity scores to the design propensity scores,

$$
h _ { i } ^ { + } ( { \bf z } ) = \frac { \lambda _ { i } ^ { + } \{ D _ { i } ( { \bf z } ) \} } { \pi _ { i } ^ { P } \{ D _ { i } ( { \bf z } ) \} } , \qquad h _ { i } ^ { - } ( { \bf z } ) = \frac { \lambda _ { i } ^ { - } \{ D _ { i } ( { \bf z } ) \} } { \pi _ { i } ^ { P } \{ D _ { i } ( { \bf z } ) \} } ,
$$

with the convention $0 / 0 = 0$ . Define the diagonal scale

$$
V _ { i } = \sum _ { d : \pi _ { i } ^ { P } ( d ) > 0 } \frac { \{ \lambda _ { i } ^ { + } ( d ) - \lambda _ { i } ^ { - } ( d ) \} ^ { 2 } } { \pi _ { i } ^ { P } ( d ) } ,
$$

and the weight bound $H _ { i } = \operatorname* { s u p } _ { \mathbf { z } } \{ | h _ { i } ^ { + } ( \mathbf { z } ) | + | h _ { i } ^ { - } ( \mathbf { z } ) | \}$

Proposition 12 (ECE importance-weighted estimator) Suppose $\lambda _ { i } ^ { + } ( d ) = \lambda _ { i } ^ { - } ( d ) = 0$ whenever $\pi _ { i } ^ { P } ( d ) = 0$ . Then

$$
\hat { \tau } _ { \mathrm { E C E } } ^ { H T } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \{ h _ { i } ^ { + } ( { \bf Z } ) - h _ { i } ^ { - } ( { \bf Z } ) \} Y _ { i } ^ { o b s }
$$

is design-unbiased for τ<sub>ECE</sub> for every potential-outcome table. $I f \ | Y _ { i } ( \mathbf { z } ) | \leq M$ and the ith summand depends on the assignment only through $\mathbf { Z } _ { \mathcal { C } _ { i } }$ , then

$$
\mathrm { V a r } ( \hat { \tau } _ { \mathrm { E C E } } ^ { H T } ) \leq \frac { M ^ { 2 } } { n ^ { 2 } } \left\{ \sum _ { i = 1 } ^ { n } V _ { i } + \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } c _ { i j } H _ { i } H _ { j } + \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } ( 1 - c _ { i j } ) \rho _ { i j } H _ { i } H _ { j } \right\} .
$$

$I f V _ { i } \leq V$ and $H _ { i } \leq H$ for all $i ,$ then

$$
\mathrm { V a r } ( \hat { \tau } _ { \mathrm { E C E } } ^ { H T } ) \leq M ^ { 2 } \left\{ \frac { V } { n } + H ^ { 2 } \frac { c _ { \mathrm { a v g } } + r _ { \mathrm { a v g } } } { n } \right\} ,
$$

and the estimator is design-MSE consistent if $V + H ^ { 2 } ( c _ { \mathrm { a v g } } + r _ { \mathrm { a v g } } ) = o ( n )$

The proof is given in Appendix D. The bound has the same three terms as the ACE bound of Proposition 10: a diagonal term, an interference-dependence term, and a residual design-dependence term. What changes is the diagonal scale. For an ACE, the policy is a point mass at the relevant potential outcomes $Y _ { i } ( d _ { i } ^ { \pm } )$ . The importance weights reduce to inverse propensities, and $V _ { i }$ and $H _ { i }$ are of the order of $1 / \pi _ { i } ( d _ { i } ^ { \pm } )$ . For an ECE, $V _ { i }$ and $H _ { i }$ are governed by the ratios $\lambda _ { i } ^ { \pm } ( d ) / \pi _ { i } ^ { P } ( d )$ , and these are large only when the allocation policy places probability on combinations that the design rarely produces. When the allocation policy is the design itself, the ratios are bounded, the diagonal term is $O ( 1 / n )$ , and the variance is governed by the dependence terms alone. The next two results make this concrete for the expected direct and indirect efects under Bernoulli policy and design.

Expected direct efect. Under Bernoulli assignment with treatment probabilities $p _ { i } ,$ , consider the EDE whose policy is the same Bernoulli allocation. The importance weight reduces to the classical Horvitz–Thompson weight

$$
h _ { i } ^ { \mathrm { E D E } } ( \mathbf { Z } ) = \frac { Z _ { i } } { p _ { i } } - \frac { 1 - Z _ { i } } { 1 - p _ { i } } ,
$$

so $\hat { \tau } _ { \mathrm { E D E } } ^ { H T }$ is the usual diference in weighted means and does not require the interference graph for computation. The weight scales are $V _ { i } = H _ { i } = 1 / p _ { i } + 1 / ( 1 - p _ { i } )$ , which are $O ( 1 )$ when the treatment probabilities are bounded away from zero and one. No inverse propensity of a rare exposure combination appears in the bound.

Proposition 13 (Variance bound for the EDE estimator) Let $\mathcal { Q } _ { p }$ be a Bernoulli allocation policy, with treatment probabilities bounded away from zero and one. Assume Bernoulli assignment, and sup $_ \ z \left| Y _ { i } ( \mathbf { z } ) \right| \le M$ . Then $\hat { \tau } _ { \mathrm { E D E } } ^ { H T }$ is design-unbiased for $\tau _ { \mathrm { E D E } } ( \mathcal { Q } _ { p } )$ , and

$$
\operatorname { V a r } _ { P _ { n } } ( \hat { \tau } _ { \mathrm { E D E } } ^ { H T } ) \leq C _ { p } M ^ { 2 } \frac { 1 + c _ { \mathrm { a v g } } } { n } ,
$$

where $C _ { p }$ depends only on the bounds on the treatment probabilities. In particular, $\hat { \tau } _ { \mathrm { E D E } } ^ { H T }$ is design-MSE consistent $i f c _ { \mathrm { a v g } } = o ( n )$

The proof is given in Appendix D. This result should be read against the DTE rows of Table 1. The two estimators use the same design and the same interference graph. The DTE estimator must isolate the treatment–exposure combination $( 1 , 0 _ { N _ { i } } )$ , which the design produces with probability $p ( 1 - p ) ^ { d _ { i } }$ , so its inverse-propensity scale is $\dot { H } = e ^ { \Theta ( d _ { \operatorname* { m a x } } ) }$ and its full variance scale from Proposition 10 is $H / n + H ^ { 2 } c _ { \mathrm { a v g } } / n$ . The EDE estimator uses every observed outcome, and its variance scale is $( 1 + c _ { \mathrm { a v g } } ) / n .$ On a d-regular locally tree-like interference graph the comparison is $( e ^ { 2 \gamma d } d ^ { 2 } ) / n$ against $d ^ { 2 } / n \colon$ the DTE is consistent only in the logarithmic-degree regime $e ^ { 2 \gamma d } d ^ { 2 } = o ( n )$ , i.e., $d \lesssim \frac { 1 } { 2 \gamma }$ log n, while the EDE HT estimator is consistent whenever $d = o ( { \sqrt { n } } )$ . The price of interference for the EDE is not the rarity of any treatment-exposure combination but the dependence among the summands, measured by $c _ { \mathrm { a v g } }$

Expected indirect $e f f e c t .$ Graph-aware ECEs that contrast neighbor treatments have larger importance weights. The expected indirect efect sums the weight over neighbors, $\begin{array} { r } { h _ { i } ^ { \mathrm { E I E } , G } ( \mathbf { Z } ) = \sum _ { j \in N _ { i } } \mathring \{ Z _ { j } / } p _ { j } - \mathring ( 1 - Z _ { j } \mathring ) / \mathring ( 1 - p _ { j } ) \}  \end{array}$ , so $H _ { i } \leq C _ { p } d _ { i }$ when the treatment probabilities are bounded away from zero and one. The weights grow with the degree, but linearly rather than exponentially.

Corollary 14 (Variance bound for the graph-aware EIE estimator) Assume the true potential outcomes satisfy neighborhood interference with known neighborhoods $N _ { i }$ . Assume Bernoulli assignment with treatment probabilities bounded away from zero and one, and assume $\operatorname* { s u p } _ { \mathbf { z } } | Y _ { i } ( \mathbf { z } ) | \leq M$ . Then $\hat { \tau } _ { \mathrm { E I E } } ^ { G }$ is design-unbiased for τ<sub>EIE</sub>(π), and

$$
\mathrm { V a r } ( \hat { \tau } _ { \mathrm { E I E } } ^ { G } ) \leq \frac { C _ { p } M ^ { 2 } } { n ^ { 2 } } \left\{ \sum _ { i } d _ { i } ^ { 2 } + \sum _ { i } \sum _ { j \neq i } c _ { i j } d _ { i } d _ { j } \right\} ,
$$

where $d _ { i } = | N _ { i } |$ and $c _ { i j } = \mathbb { 1 } \{ \mathscr { C } _ { i } \cap \mathscr { C } _ { j } \neq \emptyset \}$ . If $d _ { i } \leq d _ { \operatorname* { m a x } }$ for all i, then

$$
\mathrm { V a r } ( \hat { \tau } _ { \mathrm { E I E } } ^ { G } ) \leq C _ { p } M ^ { 2 } d _ { \mathrm { m a x } } ^ { 2 } \frac { 1 + c _ { \mathrm { a v g } } } { n } .
$$

The proof is given in Appendix D. The variance scale $d _ { \mathrm { m a x } } ^ { 2 } ( 1 + c _ { \mathrm { a v g } } ) / n$ is polynomial in the degrees: on a d-regular locally tree-like graph it is of order $d ^ { 4 } / n$ , so the EIE is consistently estimable whenever $d = o ( n ^ { 1 / 4 } )$ . As for the EDE, no exponential barrier due to the propensity score appears, and the dependence term dominates the bound.

Appendix D.11 extends the DTE/EDE comparison to arbitrary heterogeneous degree sequences, where it is governed by two graph functionals: exponential moments of the degree sequence for fixed-exposure ACEs, and the average dependence degree for the EDE.

The variance bounds of Section 4 show that classical graph-unaware designs are matched to expected causal efects and pay an exponential cost for average causal efects. Section 5 constructs designs that remove this cost, and Section 6 collects the resulting implications for classical estimators, model-assisted estimation, and the interpretation of positivity under interference.

## 5 Design-Based Estimation of Average Causal Efects

When classical estimators fail for average causal efects under interference, the failure arises because the realized assignment does not reveal the relevant potential outcomes with sufficient probability. Section 4 shows that under classical graph-unaware designs the target treatment–exposure combinations have inverse propensities that are exponential in the interference degree, and that this cost is unavoidable for any unbiased estimator. One remedy is to modify the experimental design so that those potential outcomes are observed with non-negligible probability. We study designs that control the assignment structure rather than impose an outcome model. The common device is to use the interference graph to create positive correlation within neighborhoods, so that the event that all of a unit’s neighbors are held at a common treatment occurs as a single correlated event rather than as a product of independent events. This replaces the exponential propensity $( 1 - p ) ^ { d _ { i } }$ of a classical design by a propensity that decays only polynomially in the degree $d _ { i } .$ , and the variance is reduced accordingly. Throughout, we assume that the interference neighborhood is known and that an exposure mapping is specified, and we take the potential outcomes to be uniformly bounded, $\begin{array} { r } { \operatorname* { s u p } _ { i , z , e } | Y _ { i } ( z , e ) | \leq M } \end{array}$

Suppose the target average causal efect has the form

$$
\tau = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \{ Y _ { i } ( d _ { i } ^ { + } ) - Y _ { i } ( d _ { i } ^ { - } ) \} ,
$$

where $d _ { i } ^ { + }$ and $d _ { i } ^ { - }$ are two target treatment–exposure combinations. If the design gives positive probability to both combinations,

$$
\pi _ { i } ( d _ { i } ^ { + } ) = \operatorname* { P r } \{ ( Z _ { i } , E _ { i } ) = d _ { i } ^ { + } \} > 0 , \qquad \pi _ { i } ( d _ { i } ^ { - } ) = \operatorname* { P r } \{ ( Z _ { i } , E _ { i } ) = d _ { i } ^ { - } \} > 0 ,
$$

then the treatment–exposure Horvitz–Thompson estimator

$$
\hat { \tau } _ { \mathrm { H T } } ^ { Z E } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left\{ \frac { \mathbb { 1 } \{ ( Z _ { i } , E _ { i } ) = d _ { i } ^ { + } \} Y _ { i } ^ { \mathrm { o b s } } } { \pi _ { i } ( d _ { i } ^ { + } ) } - \frac { \mathbb { 1 } \{ ( Z _ { i } , E _ { i } ) = d _ { i } ^ { - } \} Y _ { i } ^ { \mathrm { o b s } } } { \pi _ { i } ( d _ { i } ^ { - } ) } \right\}
$$

is unbiased for $\tau$ . The role of the design is therefore to make the relevant treatment–exposure combinations occur with positive and reasonably large propensity scores.

Proposition 15 (Treatment–exposure HT unbiasedness for ACEs) Suppose the observed outcome satisfies $Y _ { i } ^ { \mathrm { o b s } } = Y _ { i } ( Z _ { i } , E _ { i } )$ . Let

$$
\tau = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \{ Y _ { i } ( d _ { i } ^ { + } ) - Y _ { i } ( d _ { i } ^ { - } ) \} ,
$$

and suppose $\pi _ { i } ( d _ { i } ^ { + } ) > 0$ and $\pi _ { i } ( d _ { i } ^ { - } ) > 0$ for all i. Then $ { \hat { \tau } } _ { \mathrm { H T } } ^ { Z E }$ is design-unbiased for $\tau .$

The proof is given in Appendix E.

We develop two graph-aware constructions based on independent sets. The first targets the direct treatment efect and uses a distance-one independent set; the second targets the total treatment efect and uses a distance-two independent set. In each case a randomized greedy construction gives focal inclusion probabilities in closed form, and the resulting Horvitz–Thompson estimator has mean squared error that is polynomial, rather than exponential, in the interference degree. We then discuss cluster randomized designs, whose polynomial behavior is conditional on the clustering being aligned with the graph.

## 5.1 Independent set designs for the direct treatment efect

The direct treatment efect

$$
\tau ^ { D } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \{ Y _ { i } ( 1 , \mathbf { 0 } _ { N _ { i } } ) - Y _ { i } ( 0 , \mathbf { 0 } _ { N _ { i } } ) \}
$$

contrasts the treated and control potential outcomes of a unit whose neighbors are all held at control. Both arms require the no-exposure condition $Z _ { N _ { i } } = \mathbf { 0 } ;$ ; they difer only in the unit’s own treatment. Consequently two units i and j that share a neighbor $k \in N _ { i } \cap N _ { j }$ impose the same requirement $Z _ { k } = 0$ , so there is no conflict between their target conditions. A distance-one independent set, in which selected units are mutually non-adjacent but may share neighbors, therefore sufices.

We construct a random independent set of ego units by a randomized greedy rule, and randomize treatment among the egos while holding their neighbors at control. Units outside the independent set act as alter units that bufer the egos from interference.

Design (IS-1).

(1) Draw $U _ { 1 } , \dots , U _ { n }$ independently from the uniform distribution on [0, 1].

(2) Let $\mathcal { T } = \{ i : U _ { i } < U _ { k }$ for all $k \in N _ { i } \}$ be the set of units whose draw is the strict minimum in their closed neighborhood $\{ i \} \cup N _ { i }$ . The set I is an independent set almost surely.

(3) For each ego $i \in \mathcal { Z }$ , draw $B _ { i } \sim$ Bernoulli(q) independently, set $Z _ { i } = B _ { i }$ , and set $Z _ { k } = 0$ for all $k \in N _ { i }$

(4) Assign any unit not fixed by step (3) a treatment by a prespecified rule that does not overwrite the constraints of step (3).

The construction is graph-aware: whether i is an ego depends only on the draws in its closed neighborhood. If two egos i and j share a neighbor $k ,$ step (3) sets $Z _ { k } = 0$ from both, so the assignment is consistent regardless of how many egos claim $k .$ Because the event $\{ i \in \mathcal { I } \}$ is determined within $\{ i \} \cup N _ { i }$ , the inclusion probability is available in closed form.

Lemma 16 (Inclusion and reveal probabilities under IS-1) Under the design $I S \mathrm { - } 1 ,$

$$
\operatorname* { P r } ( i \in \mathcal { I } ) = \frac { 1 } { d _ { i } + 1 } .
$$

Let $F _ { i } ^ { ( 1 ) } = \{ i \in \mathcal { I } , \ B _ { i } = 1 \}$ and $F _ { i } ^ { ( 0 ) } = \{ i \in \mathcal { I } , \ B _ { i } = 0 \}$ be the focal reveal events. On $F _ { i } ^ { ( 1 ) }$ the assignment satisfies $\left( Z _ { i } , Z _ { N _ { i } } \right) = \left( 1 , \mathbf { 0 } \right)$ , and on $F _ { i } ^ { ( 0 ) }$ it satisfies $\left( Z _ { i } , Z _ { N _ { i } } \right) = \left( 0 , \mathbf { 0 } \right)$ , so

$$
\rho _ { i } ^ { ( 1 ) } : = \mathrm { P r } ( F _ { i } ^ { ( 1 ) } ) = \frac { q } { d _ { i } + 1 } , \qquad \rho _ { i } ^ { ( 0 ) } : = \mathrm { P r } ( F _ { i } ^ { ( 0 ) } ) = \frac { 1 - q } { d _ { i } + 1 } .
$$

The proof is given in Appendix E. The inclusion probability follows from the symmetry of the $d _ { i } + 1$ independent draws in the closed neighborhood, and the reveal probabilities follow from the independent treatment coin $B _ { i }$ . The focal reveal probabilities are exact and do not require simulation. They are stated for the focal events rather than for the raw exposure events $\{ Z _ { i } = s , Z _ { N _ { i } } = \mathbf { 0 } \}$ , because non-ego units may also happen to satisfy the no-exposure condition depending on the leftover rule in step (4); the estimator below uses only the focal events, whose probabilities are known exactly. Taking $\begin{array} { r } { q = \frac { 1 } { 2 } } \end{array}$ gives the symmetric values $\rho _ { i } ^ { ( 1 ) } = \rho _ { i } ^ { ( 0 ) } = 1 / \{ 2 ( d _ { i } + 1 ) \}$

The focal Horvitz–Thompson estimator for the direct treatment efect is

$$
\hat { \tau } _ { \mathrm { I S } } ^ { D } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left\{ \frac { \mathbb { 1 } \{ F _ { i } ^ { ( 1 ) } \} Y _ { i } ^ { \mathrm { o b s } } } { \rho _ { i } ^ { ( 1 ) } } - \frac { \mathbb { 1 } \{ F _ { i } ^ { ( 0 ) } \} Y _ { i } ^ { \mathrm { o b s } } } { \rho _ { i } ^ { ( 0 ) } } \right\} ,
$$

which at $\begin{array} { r } { q = \frac { 1 } { 2 } } \end{array}$ reduces to

$$
{ \hat { \tau } } _ { \mathrm { I S } } ^ { D } = { \frac { 2 } { n } } \sum _ { i = 1 } ^ { n } ( d _ { i } + 1 ) \left\{ \mathbb { 1 } \{ F _ { i } ^ { ( 1 ) } \} Y _ { i } ^ { \mathrm { o b s } } - \mathbb { 1 } \{ F _ { i } ^ { ( 0 ) } \} Y _ { i } ^ { \mathrm { o b s } } \right\} .
$$

Proposition 17 (Unbiasedness of the IS-1 estimator) Under the design $I S \ – I$ with $q \in$ (0, 1), the estimator $\hat { \tau } _ { \mathrm { I S } } ^ { D }$ is design-unbiased for the direct treatment efect, $\mathbb { E } [ \hat { \tau } _ { \mathrm { I S } } ^ { D } ] = \tau ^ { D }$

The proof is the same Horvitz–Thompson calculation as Proposition 15, applied to the focal reveal events rather than to the raw treatment–exposure events. On $F _ { i } ^ { ( 1 ) }$ and $F _ { i } ^ { ( 0 ) }$ the observed outcome equals the corresponding direct-efect potential outcome, and the focal probabilities $\rho _ { i } ^ { ( 1 ) } , \rho _ { i } ^ { ( 0 ) }$ are positive and known.

The variance of the estimator is controlled by the closed-neighborhood inclusion probabilities and by the dependence among egos that lie within distance two of one another. Write $\delta _ { i } = Y _ { i } ( 1 , { \bf 0 } _ { N _ { i } } ) - Y _ { i } ( 0 , { \bf 0 } _ { N _ { i } } )$ and $p _ { i } = \mathrm { P r } ( i \in \mathcal { I } ) = 1 / ( d _ { i } + 1 )$

Proposition 18 (Variance of the IS-1 estimator) Under the design IS-1 with $\begin{array} { r } { q = \frac { 1 } { 2 } } \end{array}$ 2 the estimator satisfies the exact identity

$$
n ^ { 2 } \operatorname { V a r } ( \hat { \tau } _ { \mathrm { I S } } ^ { D } ) = \sum _ { i = 1 } ^ { n } \operatorname { V a r } ( W _ { i } ) + \sum _ { \substack { i \neq j \colon \operatorname { d i s t } _ { G } ( i , j ) \leq 2 } } \delta _ { i } \delta _ { j } \left( \frac { \operatorname* { P r } ( i , j \in \mathcal { I } ) } { p _ { i } p _ { j } } - 1 \right) ,
$$

where $W _ { i } = 2 ( d _ { i } + 1 ) \mathbb { 1 } \{ i \in \mathcal { T } \} \{ Y _ { i } ( 1 , \mathbf { 0 } _ { N _ { i } } ) B _ { i } - Y _ { i } ( 0 , \mathbf { 0 } _ { N _ { i } } ) ( 1 - B _ { i } ) \}$ . For non-adjacent $i , j$ sharing $s _ { i j } = | N _ { i } \cap N _ { j } |$ common neighbors, $\operatorname* { P r } ( i , j \in { \mathcal { Z } } ) / ( p _ { i } p _ { j } ) = ( d _ { i } { + } d _ { j } { + } 2 ) / ( d _ { i } { + } d _ { j } { + } 2 { - } s _ { i j } )$ while $\operatorname* { P r } ( i , j \in \mathcal { I } ) = 0$ when i and j are adjacent. Consequently, uniformly over $| Y _ { i } ( \cdot ) | \leq M$

$$
\mathrm { V a r } ( \hat { \tau } _ { \mathrm { I S } } ^ { D } ) \leq \frac { 4 M ^ { 2 } } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } ( d _ { i } + 1 ) + \frac { 4 M ^ { 2 } } { n ^ { 2 } } \left[ 2 | E | + \sum _ { i \neq j \colon i \neq j , ~ s _ { i j } \geq 1 } \frac { s _ { i j } } { d _ { i } + d _ { j } + 2 - s _ { i j } } \right] .
$$

The proof is given in Appendix E. The first term is a diagonal contribution of order $n ^ { - 2 } \textstyle \sum _ { i } ( d _ { i } + 1 )$ ; the second collects the dependence among egos whose closed neighborhoods overlap, that is, egos within graph distance two.

Corollary 19 (Polynomial rate for the direct treatment efect) If G is d-regular and locally tree-like, in the sense that every pair of non-adjacent vertices has at most a bounded number of common neighbors $( s _ { i j } = O ( 1 ) )$ , then

$$
\mathrm { V a r } ( \hat { \tau } _ { \mathrm { I S } } ^ { D } ) = { \cal O } \biggl ( \frac { M ^ { 2 } d } { n } \biggr ) .
$$

For a general d-regular graph, the second-moment bound gives the cruder rate

$$
\mathrm { V a r } ( \hat { \tau } _ { \mathrm { I S } } ^ { D } ) = { \cal O } \biggl ( \frac { M ^ { 2 } d ^ { 2 } } { n } \biggr ) ,
$$

since on any d-regular graph the crude bound evaluates to $4 M ^ { 2 } ( n + n d + n d ^ { 2 } ) / n ^ { 2 }$ . For an arbitrary graph, the crude second-moment bound gives

$$
\mathrm { V a r } ( \hat { \tau } _ { \mathrm { I S } } ^ { D } ) \leq \frac { 4 M ^ { 2 } } { n ^ { 2 } } \left[ n + \sum _ { i } d _ { i } + \sum _ { i } d _ { i } ^ { 2 } \right] = O \left( \frac { M ^ { 2 } \left( 1 + \bar { d } + \overline { { d ^ { 2 } } } \right) } { n } \right) , \qquad \overline { { d ^ { 2 } } } = \frac { 1 } { n } \sum _ { i } d _ { i } ^ { 2 } .
$$

The proof is given in Appendix E. Under a Bernoulli design the same estimand requires the no-exposure condition, whose propensity $p ( 1 - p ) ^ { d _ { i } }$ is exponentially small in $d _ { i } ,$ and the resulting variance is exponential in the average degree by Proposition 11. The IS-1 design replaces this exponential cost by dependence that is polynomial in the degree sequence. The bound of Corollary 19 holds for arbitrary, not necessarily regular, interference graphs: it is of order $M ^ { 2 } ( 1 + \bar { d } + \overline { { d ^ { 2 } } } ) / n$ , so this estimator is consistent whenever ${ \overline { { d ^ { 2 } } } } = o ( n )$ . This is the same second-moment condition on the degree sequence under which the expected direct efect is consistent by the generic dependence-graph bound. On d-regular graphs that are locally tree-like, in the sense that non-adjacent vertices share at most a bounded number of common neighbors, the sharper rate $O ( M ^ { 2 } { \bar { d } } / n )$ holds. On a general d-regular graph the wedge sum $\begin{array} { r } { \sum _ { i \neq j \colon { i \neq j } } s _ { i j } / ( d _ { i } + d _ { j } + 2 - s _ { i j } ) } \end{array}$ can be as large as $\Theta ( n d ^ { 2 } )$ when closed neighborhoods overlap heavily, so the crude rate $O ( M ^ { 2 } d ^ { 2 } / n )$ is the best that the secondmoment bound certifies in general. Figure 3 illustrates the efect: on a fixed family of graphs, the error of the independent set estimator remains nearly flat in the degree, whereas under a Bernoulli design it grows exponentially. For irregular graphs with high-degree hubs the crude bound can be larger, and a hub-aware modification of the focal sampling may control it, though we do not pursue this refinement here. For extreme hub graphs such as a star, this particular randomized local-minimum IS-1 design has a non-vanishing secondmoment bound; this indicates a limitation of the simple construction rather than a universal impossibility theorem.

## 5.2 Independent set designs for the total treatment efect

The total treatment efect

$$
\boldsymbol { \tau } ^ { T } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \{ Y _ { i } ( 1 , e _ { i } ^ { { \bf 1 } } ) - Y _ { i } ( 0 , 0 ) \} , \qquad e _ { i } ^ { { \bf 1 } } = f _ { i } ( { \bf 1 } _ { N _ { i } } ) ,
$$

contrasts the all-treated and all-control regimes on the closed neighborhood $\{ i \} \cup N _ { i }$ . Unlike the direct treatment efect, the two arms now impose opposite requirements on a shared neighbor: if egos i and $j$ share a neighbor k and i is on the treated side while j is on the control side, then i requires $Z _ { k } ~ = ~ 1$ and $j$ requires $Z _ { k } ~ = ~ 0$ , a direct conflict. Avoiding such conflicts requires that egos have disjoint closed neighborhoods, that is, that they be at graph distance at least three. This is a distance-two independent set, obtained by running the randomized greedy rule on the radius-two balls.

For a unit i let $B _ { i } ^ { ( 2 ) } = \{ k : \mathrm { d i s t } _ { G } ( i , k ) \leq 2 \}$ denote its radius-two ball and $b _ { i } = | B _ { i } ^ { ( 2 ) } |$ its size. Since the neighborhood is contained in the ball,

$$
d _ { i } + 1 \le b _ { i } \le 1 + d _ { i } + \sum _ { j \in N _ { i } } ( d _ { j } - 1 ) \le 1 + d _ { i } d _ { \operatorname* { m a x } } .
$$

On d-regular locally tree-like graphs this gives $b _ { i } = 1 + d + d ( d - 1 ) = d ^ { 2 } + 1$

Design (IS-2).

(1) Draw $U _ { 1 } , \dots , U _ { n }$ independently from the uniform distribution on [0, 1].

(2) Let $\mathcal { T } _ { 2 } = \{ i : U _ { i } < U _ { k }$ for all $k \in B _ { i } ^ { ( 2 ) } \backslash \{ i \} \}$ be the set of units whose draw is the strict minimum in their radius-two ball. The set $\mathcal { T } _ { 2 }$ is a distance-two independent set almost surely.

(3) For each ego $i \in \mathcal { T } _ { 2 }$ , draw $R _ { i } \sim \mathrm { B e r n o u l l i } ( \frac { 1 } { 2 } )$ independently, and set $Z _ { k } = R _ { i }$ for all $k \in \{ i \} \cup N _ { i }$

(4) Assign any unit not fixed by step (3) an independent Bernoulli( <sup>1</sup>) treatment.

Because egos are at distance at least three, their closed neighborhoods are disjoint and step (3) is free of conflicts. Since the outcome of an ego depends only on the assignment of its closed neighborhood, which is entirely fixed in step (3), the leftover assignments in step (4) do not afect the revealed outcomes of egos.

Lemma 20 (Inclusion and reveal probabilities under IS-2) Under the design IS-2,

$$
\operatorname* { P r } ( i \in \mathcal { Z } _ { 2 } ) = \frac { 1 } { b _ { i } } , \qquad \rho _ { i } ^ { ( 1 ) } : = \operatorname* { P r } ( i \in \mathcal { Z } _ { 2 } , R _ { i } = 1 ) = \frac { 1 } { 2 b _ { i } } , \qquad \rho _ { i } ^ { ( 0 ) } : = \operatorname* { P r } ( i \in \mathcal { Z } _ { 2 } , R _ { i } = 0 ) = \frac { 1 } { 2 b _ { i } } .
$$

On these focal events the assignment reveals the corresponding all-treated or all-control potential outcome. In particular, if the radius-two balls are polynomially bounded in the interference degree scale, the inverse focal probabilities are polynomial as well.

The proof is given in Appendix E. The focal Horvitz–Thompson estimator for the total treatment efect is

$$
\hat { \tau } _ { \mathrm { I S } } ^ { T } = \frac { 2 } { n } \sum _ { i \in \mathbb { Z } _ { 2 } } b _ { i } \left\{ { \mathbb { 1 } \{ R _ { i } = 1 \} } Y _ { i } ^ { \mathrm { o b s } } - { \mathbb { 1 } \{ R _ { i } = 0 \} } Y _ { i } ^ { \mathrm { o b s } } \right\} .
$$

Proposition 21 (Unbiasedness and variance of the IS-2 estimator) Under the design IS-2, the estimator $\hat { \tau } _ { \mathrm { I S } } ^ { T }$ is design-unbiased for the total treatment efect. The unbiasedness is with respect to the focal reveal events $( i \in \mathcal { T } _ { 2 } , R _ { i } = 1 )$ and $( i \in \mathcal { I } _ { 2 } , R _ { i } = 0 )$ , each of which implies the corresponding treatment–exposure condition. Writing $\Delta _ { i } = Y _ { i } ( 1 , e _ { i } ^ { 1 } ) -$ $Y _ { i } ( 0 , 0 )$ and $q _ { i } = 1 / b _ { i }$ , it satisfies the exact identity

$$
n ^ { 2 } \operatorname { V a r } ( \hat { \tau } _ { \mathrm { I S } } ^ { T } ) = \sum _ { i = 1 } ^ { n } \operatorname { V a r } ( V _ { i } ) + \sum _ { \substack { i \neq j \colon B _ { i } ^ { ( 2 ) } \cap B _ { j } ^ { ( 2 ) } \neq \emptyset } } \Delta _ { i } \Delta _ { j } \left( \frac { \operatorname* { P r } ( i , j \in \mathcal { Z } _ { 2 } ) } { q _ { i } q _ { j } } - 1 \right) ,
$$

where $V _ { i } = 2 b _ { i } \mathbb { 1 } \{ i \in \mathcal { I } _ { 2 } \} \{ Y _ { i } ( 1 , e _ { i } ^ { \mathbf { 1 } } ) R _ { i } - Y _ { i } ( 0 , 0 ) ( 1 - R _ { i } ) \}$ . If dist $_ G ( i , j ) \leq 2$ , then $\operatorname* { P r } ( i , j \in$ $\begin{array} { r } { \mathcal { T } _ { 2 } ) = 0 ; } \end{array}$ if dis $\mathrm { \prime } _ { G } ( i , j ) \in \{ 3 , 4 \}$ and $s = | B _ { i } ^ { ( 2 ) } \cap B _ { i } ^ { ( 2 ) } | \geq 1$ , then the ratio equals $( b _ { i } + b _ { j } ) / ( b _ { i } +$ $b _ { j } - s )$ . Consequently, uniformly over $| Y _ { i } ( \cdot ) | \leq M$

$$
\mathrm { V a r } ( \hat { r } _ { 1 5 } ^ { T } ) \le \frac { 4 M ^ { 2 } } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } b _ { i } + \frac { 4 M ^ { 2 } } { n ^ { 2 } } \sum _ { \substack { i \neq j \neq \mathrm { ~ d i s t } _ { G } ( i , j ) \le 2 } } 1 + \frac { 4 M ^ { 2 } } { n ^ { 2 } } \sum _ { \substack { i \neq j \neq \mathrm { ~ d i s t } _ { G } ( i , j ) \le [ 3 , 4 ] , \ s _ { i } ^ { ( 2 ) } \ge 1 } } \frac { s _ { i j } ^ { ( 2 ) } } { b _ { i } + b _ { j } - s _ { i j } ^ { ( 2 ) } } , \qquad s _ { i j } ^ { ( 2 ) } = | B _ { i } ^ { ( 2 ) } \cap B _ { j } ^ { ( 2 ) } |
$$

The proof is given in Appendix E.

Corollary 22 (Polynomial rate for the total treatment efect) On d-regular graphs that are locally tree-like, the diagonal and distance-two exclusion terms are of order $M ^ { 2 } d ^ { 2 } / n$ and the positive radius-two overlap term contributes at most the same order (distance-three partners contribute $M ^ { 2 } d / n$ and distance-four partners $M ^ { 2 } d ^ { 2 } / n )$ , so

$$
\mathrm { V a r } ( \hat { \tau } _ { \mathrm { I S } } ^ { T } ) = { \cal O } \biggl ( \frac { M ^ { 2 } d ^ { 2 } } { n } \biggr ) .
$$

For an arbitrary graph, the universal bound is Var $\big ( \hat { \tau } _ { \mathrm { I S } } ^ { T } \big ) = O ( M ^ { 2 } \overline { { b ^ { 2 } } } / n )$ with ${ \overline { { b ^ { 2 } } } } = n ^ { - 1 } \sum _ { i } b _ { i } ^ { 2 }$

Since $d _ { i } \leq d _ { \operatorname* { m a x } }$ and $b _ { i } \leq 1 + d _ { i } d _ { \operatorname* { m a x } } .$ , the bounds in Corollaries 19 and 22 give variance bounds of order $O ( M ^ { 2 } ( 1 + d _ { \operatorname* { m a x } } ^ { 2 } ) / n )$ for IS-1 and $O ( M ^ { 2 } ( 1 + d _ { \operatorname* { m a x } } ^ { 4 } ) / n )$ for IS-2 on every interference graph. The proof is given in Appendix E. Under a Bernoulli design the total treatment efect requires the all-treated condition on the closed neighborhood, whose propensity $p ^ { d _ { i } + 1 }$ is again exponentially small in $d _ { i }$ . On locally tree-like d-regular graphs, the IS-2 design attains the polynomial rate $O ( M ^ { 2 } d ^ { 2 } / n )$ , the same degree order as the expected direct efect under a Bernoulli design established in Proposition 13. On arbitrary graphs the rate is governed by the radius-two ball sizes $b _ { i }$ . The exponential separation between average and expected causal efects under classical designs is therefore not intrinsic to the estimands: with a design aligned to the estimand, the total treatment efect can be estimated at polynomial rates. The direct treatment efect is easier still, since Corollary 19 gives the smaller rate $O ( M ^ { 2 } { \bar { d } } / n )$ on locally tree-like regular graphs.

## 5.3 Independent set designs for indirect efects

Indirect (spillover) efects ask how the outcome of a unit is afected by the treatment of others, its own treatment held fixed. A recurring target in network experiments is the efect of treating a selected set of influencers on an average unit: seeding experiments assign treatment to a sparse set of influential units and measure the response of otherwise untreated peers exposed to a treated influencer (Paluck et al., 2016; Aral and Walker, 2012; Aral et al., 2013; Bond et al., 2012). Under the any-treated exposure map (3.6), which records $U _ { i } = 1$ whenever at least one neighbor of i is treated, this target is the two-level average indirect efect $\begin{array} { r } { \tau _ { \mathrm { I E } } ( z ; 0 , 1 ) = n ^ { - 1 } \sum _ { i } \{ Y _ { i } ( z , 1 ) - Y _ { i } ( z , 0 ) \} } \end{array}$ of Section 3.1: a comparison of the no-exposure level $e _ { a } \ = \ 0$ and the any-treated level $e _ { b } \ = \ 1$ with the unit’s own treatment z fixed $( z = 0$ for an untreated peer). Estimators of spillover efects under twostage and independent-set randomization are developed in Baird et al. (2018); Basse and Feller (2018); Cai et al. (2024), and two-level exposure contrasts of this form are the fixedexposure analogue of the allocation-averaged estimand class studied by Viviano (2026). The uncompressed special cases contrast the fully exposed and unexposed neighborhood regimes, with the unit’s own treatment held fixed:

$$
\pi _ { \mathrm { E C } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \{ Y _ { i } ( 0 , e _ { i } ^ { 1 } ) - Y _ { i } ( 0 , 0 ) \} , \qquad \pi _ { \mathrm { E T } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \{ Y _ { i } ( 1 , e _ { i } ^ { 1 } ) - Y _ { i } ( 1 , 0 ) \} .
$$

Both arms of an indirect contrast fix an extreme neighborhood regime, and the exponential barrier of Section 4 applies exactly as it does to the direct and total efects. Under a Bernoulli design with probability $p ,$ the unexposed arm has propensity $\operatorname* { P r } ( Z _ { i } = z ^ { \dagger } ) ( 1 - p ) ^ { d _ { i } }$ the any-treated arm has propensity $\operatorname* { P r } ( Z _ { i } = z ^ { \dagger } ) ( 1 - ( 1 - p ) ^ { d _ { i } } )$ , and the full-exposure arm has propensity $\operatorname* { P r } ( Z _ { i } = z ^ { \dagger } ) p ^ { d _ { i } }$ , where $z ^ { \dagger }$ is the own-treatment level of the estimand and all but the any-treated propensity are exponentially small in the degree. Every indirect efect that fixes a rare exposure regime is therefore exponentially costly under a graph-unaware design.

The distance-two independent set construction of Section 5.2 removes this barrier for the indirect efects with a single modification. On the ego set $\mathcal { T } _ { 2 }$ , hold each ego’s own treatment at the level $z ^ { \dagger }$ required by the estimand $( z ^ { \dagger } = 0$ for $\tau _ { \mathrm { I E C } }$ and for control units; $z ^ { \dagger } = 1$ for $\tau _ { \mathrm { I E T } } )$ , and assign the ego’s neighborhood the common value $R _ { i } \in \{ 0 , 1 \}$ by an independent fair coin. An ego then reveals $( z ^ { \dagger } , e _ { i } ^ { \mathbf { 1 } } )$ when $R i = 1$ and $( z ^ { \dagger } , 0 )$ when $R _ { i } = 0$ , each with focal probability $1 / ( 2 b _ { i } )$

Proposition 23 (Independent set designs for indirect efects) Under the modified distancetwo independent set design above, the focal Horvitz–Thompson estimator is design-unbiased, with focal reveal probabilities $1 / ( 2 b _ { i } )$ for both arms, for the any-treated indirect efects $\tau _ { \mathrm { I E } } ( 0 ; 0 , 1 )$ and $\tau _ { \mathrm { I E } } ( 1 ; 0 , 1 )$ under the exposure map (3.6), and for the full-exposure versions τ<sub>IEC</sub> $( z ^ { \dagger } = 0 )$ and τ<sub>IET</sub> $( z ^ { \dagger } = 1 )$ . Its variance satisfies the bound of Proposition ${ \mathit { 2 1 } } ,$ and on locally tree-like d-regular graphs

$$
\mathrm { V a r } ( \hat { \tau } _ { \mathrm { I S } } ) = { \cal O } \left( \frac { M ^ { 2 } \overline { { { d ^ { 2 } } } } } { n } \right) ,
$$

matching the rate of the total treatment efect and of the expected direct efect on this graph class.

The proof is identical to that of Proposition 21: fixing the ego’s own treatment changes neither the inclusion probabilities nor the covariance structure, and only the pair of revealed treatment–exposure arms difers. No modification of the design is needed for the coarsened two-level estimand. Because the all-treated arm realizes the any-treated level $U _ { i } = 1$ and the all-control arm realizes $U _ { i } = 0$ under (3.6), the polarized design estimates $\tau _ { \mathrm { I E } } ( z ^ { \dagger } ; 0 , 1 )$ with the same focal probabilities $1 / ( 2 b _ { i } )$ . More generally, any monotone exposure map that takes the value 1 on the all-treated vector and 0 on the all-control vector yields the same two realized levels, so the same estimator applies to the corresponding coarsened two-level indirect efect. The coarsening is therefore free for the polarized design: it changes only the exposure interpretation, not the assignment, and it can only shorten the distance between the two levels of the contrast.

Sparse seeding with the distance-one independent set. Influencer-seeding experiments treat a sparse, well-separated set of units and measure the responses of their nontreated alters. The distance-one independent set $\mathcal { T }$ of Section 5.1 is exactly such a set: egos are mutually non-adjacent, and on typical graphs each alter is adjacent to a controlled number of egos. Treating each ego with an independent coin $B _ { i } \sim \mathrm { B e r n o u l l i } ( q )$ and setting all non-ego units to control yields the sparse seeding design: the treated influencers are the egos with $B _ { i } = 1$ , and an alter’s any-treated indicator (3.6) equals 1 precisely when at least one of its neighbors is a treated ego. Let $K _ { i } = | \mathcal { I } \cap N _ { i } |$ be the number of ego neighbors of unit i. Because I is an independent set, an ego has no ego neighbors; the unit i consequently reveals (0, 1) on the event

$$
\begin{array} { r } { F _ { i } ^ { ( 1 ) } = \Big \{ \sum _ { j \in N _ { i } } B _ { j } \mathbb { 1 } \{ j \in \mathbb { Z } \} \geq 1 \Big \} \qquad \mathrm { i . e . } \qquad \{ Z _ { i } = 0 , ~ U _ { i } = 1 \} , } \end{array}
$$

and reveals (0, 0) on $F _ { i } ^ { ( 0 ) } = \{ Z _ { i } = 0 \} \setminus F _ { i } ^ { ( 1 ) }$ , with $F _ { i } ^ { ( 1 ) } \cup F _ { i } ^ { ( 0 ) } = \{ Z _ { i } = 0 \}$ . The focal probabilities are

$$
\rho _ { i } ^ { ( 1 ) } = \mathbb { E } \big [ 1 - ( 1 - q ) ^ { K _ { i } } \big ] , \qquad \rho _ { i } ^ { ( 0 ) } = \mathbb { E } \big [ ( 1 - q ) ^ { K _ { i } } \big ] - \frac { q } { d _ { i } + 1 } .\tag{5.1}
$$

Both are positive for every unit with degree at least one.

Proposition 24 (Sparse seeding estimates the influencer spillover efect) Under the sparse seeding design with $q \in ( 0 , 1 )$ and no isolated units, the focal Horvitz–Thompson estimator

$$
\hat { \tau } _ { \mathrm { s e e d } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big \{ \frac { \mathbb { 1 } \{ F _ { i } ^ { ( 1 ) } \} Y _ { i } ^ { \mathrm { o b s } } } { \rho _ { i } ^ { ( 1 ) } } - \frac { \mathbb { 1 } \{ F _ { i } ^ { ( 0 ) } \} Y _ { i } ^ { \mathrm { o b s } } } { \rho _ { i } ^ { ( 0 ) } } \Big \}
$$

is design-unbiased for the influencer spillover efect $\tau _ { \mathrm { I E } } ( 0 ; 0 , 1 )$ under the any-treated exposure map (3.6). On a locally tree-like d-regular graph, $K _ { i } \Rightarrow$ Poisson(1) and, as $d \to \infty$

$$
\rho _ { i } ^ { ( 1 ) } \to 1 - e ^ { - q } , \qquad \rho _ { i } ^ { ( 0 ) } \to e ^ { - q } :
$$

both arms have constant focal probability, independent of n and of the degree, and the inverse-propensity weights $1 / \rho _ { i } ^ { ( 1 ) } + 1 / \rho _ { i } ^ { ( 0 ) }$ are $O ( 1 )$ . The covariance structure is inherited from the ego draws and has the radius-two range of Proposition 18, so the variance satisfies the generic bound of Proposition $1 0 ;$ on locally tree-like d-regular graphs the diagonal term is $O ( M ^ { 2 } / n )$ , removing the inverse-propensity barrier entirely for this estimand.

The proof is given in Appendix E. The constant-order focal probabilities contrast sharply with the focal probability $1 / ( 2 b _ { i } ) = O ( 1 / d ^ { 2 } )$ of the full-exposure design of Proposition 23: the coarsening to the any-treated level $e _ { b } = 1$ is what converts the exponentially rare alltreated event into a condition–“at least one treated neighbor”–that has constant probability under sparse seeding. The hub caveat is that the guarantee is tied to the ego-neighborhood count $K _ { i }$ being $O ( 1 )$ in expectation. On a star, for example, the center faces up to d simultaneously eligible ego candidates among its leaves and $\textstyle \rho _ { c } ^ { ( 0 ) } = \int _ { 0 } ^ { 1 } ( 1 - q u ) ^ { d } d u = O ( 1 / ( q d ) )$ the unexposed arm of the hub decays polynomially rather than staying constant, so the invertible focal weights grow linearly in the hub degree. The constant-propensity claim is therefore restricted to graph families in which ego-neighborhoods are small in expectation, such as locally tree-like d-regular graphs and, more generally, graphs of bounded degree. Thus the distance-one construction of Section 5.1, run as a seeding experiment, estimates the influencer spillover efect at constant focal propensities on exactly the bounded-degree families where the intervention is well-defined. In every case the exponential separation between average and expected causal efects under classical designs is a consequence of graph-unaware assignment, not a property of the estimand.

## 5.4 Cluster randomized designs

A cluster randomized design partitions the units into disjoint clusters $\mathcal { C } = \{ C _ { 1 } , \ldots , C _ { K } \}$ and assigns a common treatment to all units in a cluster, drawing the cluster-level indicators $W _ { 1 } , \dots , W _ { K }$ according to a specified scheme. A unit i reveals $Y _ { i } ( 1 , e _ { i } ^ { \mathbf { 1 } } )$ when its cluster and all clusters meeting its neighborhood are treated, and reveals $Y _ { i } ( 0 , 0 )$ when they are all controlled. Provided the induced propensities $\pi _ { i } ( 1 , e _ { i } ^ { 1 } )$ and $\pi _ { i } ( 0 , 0 )$ are positive, the treatment–exposure Horvitz–Thompson estimator of Proposition 15 is unbiased for the total treatment efect; explicit expressions for these propensities under graph cluster randomization are given by Ugander et al. (2013).

The polynomial behavior of cluster randomization is, however, conditional on the clustering being aligned with the graph. Let $m _ { i }$ denote the number of distinct clusters met by the closed neighborhood $\{ i \} \cup N _ { i }$ . Under independent cluster-level assignment with probability $\frac { 1 } { 2 }$ , the all-treated condition requires all $m _ { i }$ of these clusters to be treated, so

$$
\begin{array} { r } { \pi _ { i } ( 1 , e _ { i } ^ { 1 } ) = \Big ( \frac { 1 } { 2 } \Big ) ^ { m _ { i } } , } \end{array}
$$

and the inverse propensity is $2 ^ { m _ { i } }$ <sup>i</sup>, exponential in the number of clusters the neighborhood spans rather than in the degree. When the clustering places each closed neighborhood inside a bounded number of clusters, $m _ { i } = { O } \left( 1 \right)$ and the propensity is bounded below by a constant. Low-boundary or low-conductance partitions can have this property in clustered graphs, but the relevant design quantity is the neighborhood cluster count $m _ { i }$ itself. On graphs for which every practical balanced clustering splits many neighborhoods across a growing number of clusters, $m _ { i }$ grows with the degree, and the exponential cost reappears. Cluster randomization therefore trades an exponential dependence on degree for an exponential dependence on the neighborhood boundary, and removes the estimation barrier only when the chosen clustering is aligned with the interference graph. The independent set design IS-2 gives a graph-agnostic polynomial upper bound for the total treatment efect, while cluster designs can be sharper on graphs with strong community structure.

## 6 Implications for Designs, Estimators, and Models

The variance bounds of Section 4 have several implications for the design and analysis of experiments under interference. We collect them here.

## 6.1 Classical designs and estimators target expected causal efects

Bernoulli and completely randomized designs, analyzed with the diference-in-means estimator, are the tools most commonly used in practice. That these classical tools estimate an expected causal efect rather than an average causal efect under interference is well known (Hudgens and Halloran, 2008; S¨avje et al., 2017): the diference-in-means estimator targets the expected direct efect, obtained by averaging the unit-level direct efect over the exposure induced by the design. Under a Bernoulli design, the diference-in-means estimator is design-unbiased for the expected direct efect; by Proposition 13, its variance is of order $( 1 + c _ { \mathrm { a v g } } ) / n$ , and its computation does not require the knowledge of interference graph. However, the expected direct efect is policy dependent, and it mixes direct treatment efect and interference efects, as shown in Proposition 3 which uses the non-parametric linear decomposition.

The same design and estimator are ill-suited to non-parametrically estimate average causal efects. The direct treatment efect is a contrast of the potential outcomes ${ Y _ { i } } ( 1 , { \bf 0 } _ { N _ { i } } )$ and $Y _ { i } ( 0 , { \mathbf { 0 } } _ { N _ { i } } )$ , and under a Bernoulli design the cell $( 1 , \mathbf { 0 } _ { N _ { i } } )$ is observed with probability $p ( 1 - p ) ^ { d _ { i } }$ . By Proposition 11, the variance of any graph unaware design-unbiased estimator of the direct treatment efect is then exponential in the interference degree, a similar result holds for the total treatment efect. Moreover, the diference-in-means estimator is not consistent for the direct treatment efect: its probability limit is the expected direct efect, so it carries a bias equal to the expected treatment–exposure interaction ${ \overline { { C } } } ,$ , which is zero only under additivity. Classical estimators are therefore appropriate for expected causal efects such as the expected direct efect, but not for average causal efects such as the direct and total treatment efects.

## 6.2 Positivity and overlap for average causal efects

Changing the exposure mapping does not provide an escape. As shown in Section 4, the direct treatment efect requires the no-exposure potential outcome under any mapping, so its propensity remains exponentially small in the degree. The dificulty is a failure of positivity for the relevant treatment–exposure combination, the analogue under interference of the overlap condition in observational studies. On interference graphs with high-degree units this failure is severe, and a fixed-exposure average causal efect can be identified only on the subpopulation of units whose neighborhoods are small enough to provide adequate overlap.

## 6.3 Model-assisted estimation of average causal efects

A second strategy is available when one is willing to impose outcome model restrictions on the potential outcomes. The non-parametric decomposition $Y _ { i } ( z , e ) = A _ { i } ( z ) + B _ { i } ( e ) + z C _ { i } ( e )$ suggests a natural class of working models: a restriction on the spillover and interaction components $B _ { i }$ and $C _ { i }$ relates the nuisance potential outcomes to the relevant ones, so that units that reveal nuisance potential outcomes also become informative for the estimand. The simplest such restriction is additivity, $C _ { i } \equiv 0 ,$ , under which the diference-in-means estimator is consistent for the direct treatment efect, since the bias $\overline { C }$ then vanishes. More generally, a working model can be used within a model-assisted estimator, such as a generalized regression estimator, that retains a design-based target while borrowing information across treatment–exposure cells. Such estimators trade the exponential cost of the inverse propensity for sensitivity to misspecification of the working model. Model-assisted estimators of this kind under interference are developed in Karwa and Airoldi (2018).

## 7 Numerical illustration

We illustrate the estimation gap between ACEs and ECEs on simulated experiments. The interference graph, the exposure mapping, and the potential outcomes are fixed in advance, and the only source of randomness is the treatment assignment.

We draw the interference graph from two random graph model families. The first is the Erd˝os–R´enyi graph $G ( n , c / n )$ , which is sparse and has degrees concentrated about their mean; it represents limited and roughly homogeneous interference. The second is the Barab´asi–Albert graph, whose degree distribution is heavy-tailed and contains a few highdegree hubs; it represents the heterogeneous interference common in social and economic networks. We use $n = 2 0 0 0$ units in the Erd˝os–R´enyi experiments and $n = 3 0 0 0$ in the Barab´asi–Albert experiment.

The exposure mapping is the fraction of treated neighbors, $\begin{array} { r } { e _ { i } = d _ { i } ^ { - 1 } \sum _ { j \in N _ { i } } z _ { j } . } \end{array}$ , with $e _ { i } =$ 0 when $d _ { i } = 0$ . Potential outcomes follow the decomposition ${ Y _ { i } ( z , e ) = \mu _ { i } + \bar { A _ { i } } ( z ) + B _ { i } ( e ) + }$ $z C _ { i } ( e )$ , with $A _ { i } ( z ) = \alpha _ { i } z , B _ { i } ( e ) = \beta _ { i } e$ , and $C _ { i } ( e ) = \gamma _ { i } e$ . The unit-level coeficients are drawn independently once, with $\mu _ { i } \sim$ Unif(−0.2, 0.2), $\alpha _ { i } \sim$ Unif(0.3, 0.7), $\beta _ { i } \sim \mathrm { U n i f } ( 0 . 2 , 0 . 6 )$ and $\gamma _ { i } \sim \mathrm { U n i f } ( 0 , 0 . 4 )$ , and then held fixed. Thus the response surface is bounded and heterogeneous across units. The interaction coeficients have positive mean, so the direct and expected direct efects difer. The figures focus on three estimands: the direct treatment efect, which contrasts $Y _ { i } ( 1 , 0 )$ and $Y _ { i } ( 0 , 0 )$ ; the expected direct efect under the Bernoulli allocation policy; and the expected indirect efect under the same policy. Because the treated-neighbor count is binomial conditional on the graph under a Bernoulli design, each target is available in closed form and estimation error is measured against the exact finitepopulation value.

Each efect is estimated by the corresponding Horvitz–Thompson estimator of Section 4: treatment–exposure weights for the direct treatment efect, fixed own-treatment weights for the expected direct efect, and graph-aware weights for the expected indirect efect. The usual diference-in-means estimator appears separately in Figure 2; it is the ratio form of the own-treatment weighted estimator. We consider two designs: a graph-unaware Bernoulli design with treatment probability one half, used in all but one experiment, and an independent-set design aligned with the direct treatment efect. Each reported error is the root-mean-square error over independent draws of the assignment; Figures 1–3 use Monte Carlo repetitions, while Figure 4 reports the exact probability that the relevant potential outcome is observed at least once in $1 0 ^ { 4 }$ Bernoulli assignments. The Horvitz–Thompson estimators are design-unbiased by construction, and we confirmed that the empirical bias of each is zero up to Monte Carlo error.

Figure 1 reports the root-mean-square error against the mean interference degree on the Erd˝os–R´enyi graphs; the vertical axis is logarithmic, so a straight line is exponential growth. The direct treatment efect (red circles) follows the inverse-propensity scale for observing $Y _ { i } ( 1 , 0 )$ or $Y _ { i } ( 0 , 0 )$ (the red dashed line, Proposition 10). The expected direct efect (blue squares) is horizontal, as Proposition 13 predicts, and stays near its dependence scale (the blue dashed line). The expected indirect efect (green triangles) rises only polynomially, well below the direct efect. The three estimands use the same graph, the same design, and the same outcomes. The separation is not only asymptotic. At bounded degree the RMSE of both classes decreases at the parametric rate $n ^ { - 1 / 2 }$ , but the constant difers sharply: for the direct treatment efect the efective sample size is of order $n p ( 1 - p ) ^ { d }$ , since only assignments revealing the relevant potential outcomes contribute, whereas the expected direct efect uses every unit and its constant grows only polynomially in the degree. Computed at a single sample size, Figure 1 shows that this gap in constants is already large at the sizes typical of experiments.

![](images/bc22e5cff0bc9dfc65e5308c3d16c4416881f206c0ea24a063211ae2fe90a7c1.jpg)  
Figure 1: Design root-mean-square error of the Horvitz–Thompson estimator under a Bernoulli design $( p = 1 / 2 )$ against the mean interference degree, on Erd˝os–R´enyi graphs $( n = 2 0 0 0 )$ ; the vertical axis is logarithmic. The direct treatment efect (fixed-exposure ACE, red) follows the exponential inverse-propensity scale (red dashed; Proposition 10); the expected direct efect (aligned ECE, blue) is horizontal; the expected indirect efect (graph-aware ECE, green) grows polynomially.

![](images/9ec602db65dc62222f49692df3ac7ef1e2e5a0b727fbae06c6616c468e6142a5.jpg)  
Figure 2: The diference-in-means estimator against two targets as the sample grows (Erd˝os– R´enyi, mean degree 8, $p = 1 / 2 )$ . Its error for the expected direct efect decays at the parametric rate; its error for the direct treatment efect settles at the policy-averaged interaction floor $\bar { C } _ { \pi }$ set by the treatment–exposure interaction that own-treatment contrasts cannot remove.

Figure 2 follows a single estimator, the diference in means, against two targets as the sample grows. Its error for the expected direct efect (blue) decays as a straight line of slope $- 1 / 2$ on the logarithmic axes, the parametric rate; its error for the direct treatment efect (red) flattens at a positive floor, the horizontal dashed line at $\bar { C } _ { \pi }$ . The estimand class, not the estimator, governs consistency: the same estimator is consistent for the efect the design averages over and inconsistent for the fixed-exposure efect on the same data. The decomposition $Y _ { i } ( z , e ) = A _ { i } ( z ) + B _ { i } ( e ) + z C _ { i } ( e )$ identifies the floor exactly. It is the policy-averaged treatment–exposure interaction $\bar { C } _ { \pi }$ , equal to $p n ^ { - 1 } \textstyle \sum _ { i } \gamma _ { i }$ in this simulation, and is the part of the response surface that own-treatment contrasts cannot remove; under additivity, $C _ { i } \equiv 0$ , the floor vanishes and the diference in means is consistent for the direct efect as well.

![](images/97a427db9c4b1144e2c90c0b8fa49aa68b956891bf28e275ef55a35a26f4397c.jpg)  
Figure 3: Estimand-aligned design for the direct treatment efect (Erd˝os–R´enyi, $n = 2 0 0 0$ $p = 1 / 2 )$ . Under a Bernoulli design the Horvitz–Thompson error grows exponentially in the mean interference degree; under an independent-set design on the same graphs, which reveals $Y _ { i } ( 1 , 0 )$ and $Y _ { i } ( 0 , 0 )$ for ego units, it stays nearly flat.

Figure 3 illustrates the first route around the exponential barrier for estimating ACEs. The independent-set design treats mutually non-adjacent ego units and holds their neighborhoods at control, so each ego reveals one of the two potential outcomes that define the direct treatment efect, $Y _ { i } ( 1 , 0 )$ or $Y _ { i } ( 0 , 0 )$ . On the same Erd˝os–R´enyi graphs as Figure 1, the error under the Bernoulli design (red) grows exponentially in the degree, while the error under the independent-set design (purple) remains nearly flat.

Figure 4 turns to a heavy-tailed Barab´asi–Albert graph. The direct-efect potential outcome for a high-degree unit is observed with probability that decays exponentially in the unit’s degree, so the inverse propensity of the highest-degree unit exceeds $1 0 ^ { 7 0 }$ (left panel) and the probability of seeing the relevant potential outcome at least once in $1 0 ^ { 4 }$ assignment draws is essentially zero (right panel). The fixed-exposure ACE is not merely expensive to estimate on such graphs; at feasible sample sizes it is efectively non-estimable for the hubs, while the expected direct efect uses every unit. Truncating or stabilizing the extreme inverse-propensity weights can reduce variance, but only by changing the estimator and introducing bias for the fixed finite-population target unless additional modeling assumptions are imposed. Heterogeneity in the degree distribution sharpens the separation rather than

Heavy-tailed interference (Barabasi-Albert, n = 3000, max degree = 222): 162 units effectively never reveal the DTE cell

![](images/fd910b81f1d609ce760ff89bf06e8624354e6984144a15b43a66e3ea03f86838.jpg)

![](images/ff7c678233de4d33f18cc203ec3ff931e94dd48654762ff3d2160b5fef19d329.jpg)  
Figure 4: Heavy-tailed interference (Barab´asi–Albert, n = 3000). Left: the inverse propensity of the direct-efect exposure condition grows exponentially in a unit’s degree, exceeding $1 0 ^ { 7 0 }$ for the highest-degree unit. Right: the probability that a unit ever reveals the relevant direct-efect potential outcome, in $1 0 ^ { 4 }$ assignment draws, collapses to zero for high-degree units, whereas the expected direct efect uses every unit.

softening it: the cost of a fixed-exposure ACE is governed by the exponential moment of the degree distribution, which the hubs dominate.

## 8 Discussion

For principled causal inference, the potential-outcomes framework keeps three ingredients separate: restrictions on the potential outcomes, the definition of a causal estimand, and the design and estimator used to learn it. Without interference, the first ingredient is a single clean exclusion restriction: each unit has two potential outcomes. Interference is not the negation of that restriction. It is a family of possible restrictions. One has to specify which units’ treatment assignments can afect a given unit’s potential outcome, how those assignments enter the potential outcome, and whether the response surface has additional structure such as additivity, homogeneity, or linearity. The non-parametric decomposition of potential outcomes given in this paper provides a unified language to specify any finitepopulation model of potential outcomes under interference.

Moreover, without interference there is a single canonical finite-population average treatment efect. With interference there are multiple causal efects of interest, such as direct efects, spillover efects, total efects, and policy efects. We classify estimands under interference into two broad categories: average causal efects, which contrast fixed potential outcomes, and expected causal efects, which average potential outcomes under an allocation policy. This distinction matters because the same name can refer to diferent functionals of the potential-outcome table. For example, a direct efect may be a fixed contrast holding neighbors’ treatment fixed, as in the direct treatment efect. It may also average over the neighbor treatments generated by an allocation policy, as in the expected direct efect. These estimands coincide under additivity, but not in general.

This perspective also clarifies why several results in the literature that appear to point in diferent directions are compatible. Partial-interference analyses specify groups and exposure within groups, and the resulting efects are tied to the group-level allocation policy (Hudgens and Halloran, 2008). Exposure-mapping approaches target average causal efects by contrasting fixed treatment–exposure combinations (Aronow and Samii, 2017). Some estimators remain valid under unknown interference because their targets average over the exposure distribution induced by a policy or design; in the terminology of this paper, the average direct efect studied by S¨avje et al. (2017) is an expected direct efect. Other approaches recover average causal efects under unknown network structure by imposing outcome-model restrictions; for example, Yu et al. (2022) target a total treatment efect using additivity and linearity. Estimation of these efects is not free. The graph assumption is replaced either by a policy-averaged target or by an outcome-model assumption on the potential outcomes.

The same distinction governs design. Classical graph-unaware designs are often well matched to estimate policy-averaged expected efects, because they generate the exposure distribution being averaged over. They are poorly matched to estimate fixed-exposure average efects when the defining treatment–exposure combinations are rare. In observational studies without interference, positivity requires treatment variation within the covariate strata that define the target comparison. Under interference, the analogous strata are treatment–exposure cells. A Bernoulli design gives abundant information about typical exposure states, but a direct treatment efect ACE may require the atypical cell in which a treated unit has no treated neighbors. Similarly, a total treatment efect ACE may require all-treated and all-control neighborhoods. These cells can have exponentially small probability in the interference degree under classical designs. Fixed-exposure ACEs therefore require either estimand-aligned designs or outcome-model restrictions. Coarsening the spillover boundary can collapse this cost: under the any-treated exposure map the influencer spillover efect is estimated with constant focal propensities by sparse seeding on locally tree-like regular graphs (Section 5.3), while the uncompressed full-exposure versions retain the exponential barrier and need the distance-two design.

The exponential-versus-polynomial gap implies that the estimand and the design cannot be chosen separately. An investigator who targets a fixed-exposure average causal efect but implements a classical design will incur large estimation errors. The estimand and a design that reveals it must be chosen together, at the planning stage. When the design cannot be controlled, as in observational studies, the situation is more severe: average causal efects are then essentially out of reach without strong outcome models or bounded or logarithmic interference degrees, whereas expected causal efects aligned with the assignment mechanism remain estimable under the conditions that make them estimable by design.

A central problem is that the graph and exposure mapping are rarely known at the design stage. If the specified graph and exposure mapping are correct, a fixed-exposure average causal efect is a contrast on the reduced potential outcomes. If the exposure mapping is misspecified, the same contrast may average over assignment features that the mapping omits; this distinction between a target defined under a working exposure mapping and the corresponding target in the full Table of Science is emphasized by S¨avje (2024). The same issue arises for the interference graph. If the graph omits relevant treatment coordinates, a fixed-exposure contrast in the reduced representation averages over those omitted coordinates in the full Table of Science. It is then no longer the original fixedexposure ACE, but a coarsened estimand defined by the misspecified representation. This is not only a matter of bias in an estimator. The target itself has changed.

Several problems follow. Experiments should be designed so that conclusions are stable to plausible changes in the interference graph and exposure mapping. When network data, pretreatment covariates, or repeated experiments are used to learn these objects, the uncertainty in that learning step should enter the final inference. Variance estimation must account for randomization dependence and interference dependence at the same time. Model-assisted estimators add another layer: the fitted response model can reduce variance or recover information from nuisance potential outcomes, but its use must be separated from the design-based correction it supports. These are the questions that arise after the estimand, the interference model, and the design have been put in the same language.

## Appendix A. Additional examples and extensions for Section 2

## A.1 Asymmetric and bipartite interference

While many models assume symmetric interference, interference may be directional. For instance, in social hierarchies or upstream–downstream systems, the treatment of one unit can afect another without a reciprocal efect. In these cases, the in-degree represents a unit’s exposure to external interference, while the out-degree measures that unit’s reach within the network.

The framework also extends to bipartite interference. In such cases, the graph $G =$ $( U \cup V , E )$ consists of a set of treatment units U and outcome units V , where edges only exist from U to V. This structure is useful when the units being treated difer from the units whose outcomes are measured, as in environmental and service-provider settings (Zigler and Papadogeorgou, 2021).

## A.2 Vector-valued and continuous exposures

The exposure function may be multidimensional. For example, in a social network one may use

$$
e _ { i } = \mathrm { { \small { \left( n u m b e r ~ o f ~ t r e a t e d ~ f r i e n d s , ~ n u m b e r ~ o f ~ t r e a t e d ~ f r i e n d s - o f - f r i e n d s \right) } } } .
$$

In settings where interference is governed by physical proximity, the exposure mapping can be defined as a continuous function of geographic distance. For example, $N _ { i }$ can represent all units within a relevant radius, with

$$
e _ { i } = \sum _ { j \in N _ { i } } K ( \rho _ { i j } ; \phi ) z _ { j } ,
$$

where $\rho _ { i j }$ is the distance between units i and $j ,$ and $K ( \cdot )$ is a kernel function parameterized by $\phi$ (Gifin et al., 2023).

## A.3 Nonlinear and threshold outcome restrictions

The decomposition in Theorem 1 also accommodates nonlinear restrictions on the spillover function $B _ { i }$ and the interaction function $C _ { i }$ . For example, under additivity $( C _ { i } = 0 )$ , one

may posit a concave interference function such as $B _ { i } ( e _ { i } ) = \gamma \log ( 1 + e _ { i } )$ , a convex function such as $B _ { i } ( e _ { i } ) = \gamma e _ { i } ^ { 2 }$ , or a sigmoidal function

$$
Y _ { i } ( z _ { i } , e _ { i } ) = \alpha _ { i } + \beta _ { i } z _ { i } + \frac { L } { 1 + \exp \{ - k ( e _ { i } - e _ { 0 } ) \} } .
$$

Such restrictions arise in settings where the first few treated neighbors have large efects, or where spillovers occur only after a threshold has been reached (Aral and Walker, 2011; Leskovec et al., 2007; De Giorgi et al., 2010).

Threshold and contagion models can be represented by binary exposure mappings such as

$$
e _ { i } = \mathbb { I } \left( \sum _ { j \in N _ { i } } z _ { j } \geq K \right) ,
$$

which are often used in models of collective behavior and complex contagion (Granovetter, 1978; Centola, 2010). Additional structural restrictions can then be imposed on $B _ { i } ( e _ { i } )$ and $C _ { i } ( e _ { i } )$ depending on whether individual treatment and exposure are substitutes, complements, or interact in some other way.

## A.4 Non-anonymous exposure

In non-anonymous interference, the efect of neighbors’ treatments depends on their identities or covariates. One example is

$$
e _ { i } = \sum _ { j \in N _ { i } } w ( X _ { j } ) z _ { j } ,
$$

where $X _ { j }$ are neighbor attributes. The potential outcome can then be written as

$$
Y _ { i } ( z _ { i } , e _ { i } ) = \alpha _ { i } + \beta _ { i } z _ { i } + B _ { i } \left( \sum _ { j \in N _ { i } } w ( X _ { j } ) z _ { j } \right) + z _ { i } C _ { i } \left( \sum _ { j \in N _ { i } } w ( X _ { j } ) z _ { j } \right) .
$$

This includes weighted and spatial exposure mappings, including environmental settings where exposure depends on distance or direction (Zigler et al., 2012; Patacchini et al., 2017; Adhikari and Zheleva, 2025).

## Appendix B. Proofs for Section 2

## B.1 Proof of Theorem 1

Restatement of Theorem 1. For any interference neighborhood $N _ { i }$ and exposure mapping $e _ { i } = f ( { \bf Z } _ { N _ { i } } )$ , there exist functions $A _ { i } ( z ) , B _ { i } ( e )$ , and $C _ { i } ( e )$ such that, for each unit i and each $( z , e ) \in \{ 0 , 1 \} \times \mathcal { E } _ { i }$ ，

$$
Y _ { i } ( z , e ) = A _ { i } ( z ) + B _ { i } ( e ) + z C _ { i } ( e ) ,
$$

with the normalization $B _ { i } ( 0 ) = C _ { i } ( 0 ) = 0$

Proof Fix a unit i. Once the interference neighborhood and exposure mapping are specified, the potential outcomes for unit i are indexed by $( z , e ) \in \{ 0 , 1 \} \times \mathcal { E } _ { i }$ . Define

$$
A _ { i } ( z ) = Y _ { i } ( z , 0 ) , \qquad B _ { i } ( e ) = Y _ { i } ( 0 , e ) - Y _ { i } ( 0 , 0 ) ,
$$

and

$$
C _ { i } ( e ) = Y _ { i } ( 1 , e ) - Y _ { i } ( 1 , 0 ) - Y _ { i } ( 0 , e ) + Y _ { i } ( 0 , 0 ) .
$$

Then $B _ { i } ( 0 ) = 0$ and $C _ { i } ( 0 ) = 0 . \mathrm { ~ I f ~ } z = 0$ , then

$$
A _ { i } ( 0 ) + B _ { i } ( e ) = Y _ { i } ( 0 , 0 ) + Y _ { i } ( 0 , e ) - Y _ { i } ( 0 , 0 ) = Y _ { i } ( 0 , e ) .
$$

$\operatorname { I f } z = 1$ , then

$$
\begin{array} { c } { A _ { i } ( 1 ) + B _ { i } ( e ) + C _ { i } ( e ) = Y _ { i } ( 1 , 0 ) + Y _ { i } ( 0 , e ) - Y _ { i } ( 0 , 0 ) } \\ { + Y _ { i } ( 1 , e ) - Y _ { i } ( 1 , 0 ) - Y _ { i } ( 0 , e ) + Y _ { i } ( 0 , 0 ) } \\ { = Y _ { i } ( 1 , e ) . } \end{array}
$$

Thus $Y _ { i } ( z , e ) = A _ { i } ( z ) + B _ { i } ( e ) + z C _ { i } ( e )$ for every z and e.

The representation is also a full-rank reparametrization. For each unit $i ,$ there are $2 K _ { i }$ distinct potential outcomes. The parametrization contains

$$
2 + ( K _ { i } - 1 ) + ( K _ { i } - 1 ) = 2 K _ { i }
$$

free parameters:

$$
\{ A _ { i } ( 0 ) , A _ { i } ( 1 ) \} , \qquad \{ B _ { i } ( e ) : e \neq 0 \} , \qquad \{ C _ { i } ( e ) : e \neq 0 \} .
$$

The formulas above give the inverse map from the potential-outcome table to these parameters, so the map is one-to-one and full rank. ■

## Appendix C. Proofs and details for Section 3

## C.1 Mixed causal efects: EIC, EIT, and ETE

Mixed causal efects contrast expected potential outcomes under a policy against fixed baseline responses. They can be viewed as expected analogues of the ACE indirect efects.

Expected exposure efects. Expected exposure efects capture the policy-averaged indirect efect, holding the own treatment fixed. The expected exposure efect on control units is

$$
\pi _ { \mathrm { E I C } } ( \mathcal { Q } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { \epsilon \in \mathcal { E } } \Big ( Y _ { i } ( 0 , e ) - Y _ { i } ( 0 , 0 ) \Big ) P _ { \mathcal Q } ( E _ { i } = e ) = \frac { 1 } { n } \sum _ { i } \mu _ { i } ( 0 ; \mathcal { Q } ) - \frac { 1 } { n } \sum _ { i } Y _ { i } ( 0 , 0 ) ,\tag{C.1}
$$

and on treated units is

$$
\pi _ { \mathrm { E I T } } ( \mathcal { Q } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { \epsilon \in \mathcal { E } } \Big ( Y _ { i } ( 1 , e ) - Y _ { i } ( 1 , 0 ) \Big ) P _ { \mathcal Q } ( E _ { i } = e ) = \frac { 1 } { n } \sum _ { i } \mu _ { i } ( 1 ; \mathcal { Q } ) - \frac { 1 } { n } \sum _ { i } Y _ { i } ( 1 , 0 ) .\tag{C.2}
$$

These satisfy $\tau _ { \mathrm { E I C } } ( \mathcal { Q } ) = \mathbb { E } _ { \mathcal { Q } } [ \tau _ { \mathrm { I E C } } ( E ) ] \ \mathrm { a n d } \ \tau _ { \mathrm { E I T } } ( \mathcal { Q } ) = \mathbb { E } _ { \mathcal { Q } } [ \tau _ { \mathrm { I E T } } ( E ) ]$

Expected total efect. The expected total efect compares the expectation under full treatment to the baseline condition:

$$
\pi _ { \mathrm { E T E } } ( \mathcal { Q } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { \epsilon \in \mathcal { E } } \Big ( Y _ { i } ( 1 , e ) - Y _ { i } ( 0 , 0 ) \Big ) P _ { \mathcal Q } \big ( E _ { i } = e \big ) = \frac { 1 } { n } \sum _ { i } \mu _ { i } ( 1 ; \mathcal { Q } ) - \frac { 1 } { n } \sum _ { i } Y _ { i } ( 0 , 0 ) .\tag{C.3}
$$

By Proposition 3, these satisfy $\tau _ { \mathrm { E I C } } ( \mathcal { Q } ) = \bar { B } _ { \mathcal { Q } } , \tau _ { \mathrm { E I T } } ( \mathcal { Q } ) = \bar { B } _ { \mathcal { Q } } + \bar { C } _ { \mathcal { Q } }$ , and $\tau _ { \mathrm { E T E } } ( \mathcal { Q } ) =$ $\begin{array} { r } { n ^ { - 1 } \sum _ { i } \beta _ { i } + \bar { B } _ { \mathcal { Q } } + \bar { C } _ { \mathcal { Q } } } \end{array}$ , and they decompose as $\tau _ { \mathrm { E T E } } ( \mathcal { Q } ) = \tau _ { \mathrm { E D E } } ( \mathcal { Q } ) + \tau _ { \mathrm { E I C } } ( \mathcal { Q } ) = \tau _ { \mathrm { D T E } } +$ $\pi _ { \mathrm { E I T } } ( \mathcal { Q } )$

Proposition 25 (ECE mixed-efect decomposition) For any allocation policy Q,

$$
\tau _ { \mathrm { E T E } } ( \mathcal { Q } ) = \tau _ { \mathrm { E D E } } ( \mathcal { Q } ) + \tau _ { \mathrm { E I C } } ( \mathcal { Q } ) = \tau _ { \mathrm { D T E } } + \tau _ { \mathrm { E I T } } ( \mathcal { Q } ) .
$$

## C.2 Proof of Proposition 3

Restatement. Under the non-parametric linear representation of Theorem 1,

$$
Y _ { i } ( z , e ) = \alpha _ { i } + \beta _ { i } z + B _ { i } ( e ) + z C _ { i } ( e ) , \qquad B _ { i } ( 0 ) = C _ { i } ( 0 ) = 0 ,
$$

the ACEs and ECEs have the representations stated in Proposition 3.

Proof For the fixed-exposure efects, substitute the representation into each contrast. The direct treatment efect is

$$
Y _ { i } ( 1 , 0 ) - Y _ { i } ( 0 , 0 ) = \{ \alpha _ { i } + \beta _ { i } + B _ { i } ( 0 ) + C _ { i } ( 0 ) \} - \{ \alpha _ { i } + B _ { i } ( 0 ) \} = \beta _ { i } .
$$

Similarly,

$$
Y _ { i } ( 0 , e ) - Y _ { i } ( 0 , 0 ) = B _ { i } ( e ) , \qquad Y _ { i } ( 1 , e ) - Y _ { i } ( 1 , 0 ) = B _ { i } ( e ) + C _ { i } ( e ) ,
$$

and

$$
Y _ { i } ( 1 , 1 ) - Y _ { i } ( 0 , 0 ) = \beta _ { i } + B _ { i } ( 1 ) + C _ { i } ( 1 ) .
$$

Averaging these identities over i gives the ACE formulas.

For the expected efects, take expectations with respect to the exposure distribution induced by the allocation policy. Since

$$
Y _ { i } ( 1 , E _ { i } ) - Y _ { i } ( 0 , E _ { i } ) = \beta _ { i } + C _ { i } ( E _ { i } ) ,
$$

we have

$$
\tau _ { \mathrm { E D E } } ( \mathcal { Q } ) = \frac { 1 } { n } \sum _ { i } \beta _ { i } + \frac { 1 } { n } \sum _ { i } \mathbb { E } _ { \mathcal { Q } } [ C _ { i } ( E _ { i } ) ] .
$$

The remaining three formulas follow from

$$
\begin{array} { c l c r } { { } } & { { Y _ { i } ( 0 , E _ { i } ) - Y _ { i } ( 0 , 0 ) = B _ { i } ( E _ { i } ) , } } \\ { { } } & { { } } \\ { { } } & { { Y _ { i } ( 1 , E _ { i } ) - Y _ { i } ( 1 , 0 ) = B _ { i } ( E _ { i } ) + C _ { i } ( E _ { i } ) , } } \end{array}
$$

and

$$
Y _ { i } ( 1 , E _ { i } ) - Y _ { i } ( 0 , 0 ) = \beta _ { i } + B _ { i } ( E _ { i } ) + C _ { i } ( E _ { i } ) .
$$

## C.3 Proof of Proposition 5

Restatement. The total treatment efect satisfies

$$
\tau _ { \mathrm { T T E } } = \tau _ { \mathrm { D T E } } + \tau _ { \mathrm { I E T } } .
$$

Under additivity, $C _ { i } ( e ) = 0$ for all $i , e ,$ so

$$
\tau _ { \mathrm { T T E } } = \tau _ { \mathrm { D T E } } + \tau _ { \mathrm { I E T } } = \tau _ { \mathrm { D T E } } + \tau _ { \mathrm { I E C } } .
$$

Proof By Proposition 3,

$$
\tau _ { \mathrm { T T E } } = \frac { 1 } { n } \sum _ { i } \{ \beta _ { i } + B _ { i } ( 1 ) + C _ { i } ( 1 ) \} ,
$$

while

$$
\tau _ { \mathrm { D T E } } + \tau _ { \mathrm { I E T } } = \frac { 1 } { n } \sum _ { i } \beta _ { i } + \frac { 1 } { n } \sum _ { i } \{ B _ { i } ( 1 ) + C _ { i } ( 1 ) \} .
$$

The two expressions are equal. If $C _ { i } ( e ) = 0$ for all $i , e ,$ then $\tau _ { \mathrm { I E T } } = \tau _ { \mathrm { I E C } }$ , which gives the additive form. ■

## C.4 Proof of Proposition 25

Restatement. For any allocation policy $\mathcal { Q } .$

$$
\pi \mathrm { r } \mathrm { r } \mathrm { E } \big ( \mathcal { Q } \big ) = \pi \mathrm { D } \mathrm { E } \big ( \mathcal { Q } \big ) + \pi \mathrm { E } \mathrm { I C } \big ( \mathcal { Q } \big ) = \pi \mathrm { T } \mathrm { T } \mathrm { E } + \tau \mathrm { E } \mathrm { I T } \big ( \mathcal { Q } \big ) .
$$

Proof Using the notation of Proposition $^ { 3 , }$

$$
\tau _ { \mathrm { E T E } } ( \mathcal { Q } ) = \frac { 1 } { n } \sum _ { i } \beta _ { i } + \bar { B } _ { \mathcal { Q } } + \bar { C } _ { \mathcal { Q } } .
$$

Also,

$$
\tau _ { \mathrm { E D E } } ( \boldsymbol { \mathcal { Q } } ) + \tau _ { \mathrm { E I C } } ( \boldsymbol { \mathcal { Q } } ) = \left\{ \frac { 1 } { n } \sum _ { i } \beta _ { i } + \bar { C } _ { \boldsymbol { Q } } \right\} + \bar { B } _ { \boldsymbol { Q } } ,
$$

and

$$
\tau _ { \mathrm { D T E } } + \tau _ { \mathrm { E I T } } ( \mathcal { Q } ) = \frac { 1 } { n } \sum _ { i } \beta _ { i } + \bar { B } _ { \mathcal { Q } } + \bar { C } _ { \mathcal { Q } } .
$$

Both expressions equal $\tau _ { \mathrm { E T E } } ( \mathcal { Q } )$

## C.5 Proof of Corollary 4

Restatement. Under additivity of the causal response surface, $\tau _ { \mathrm { E D E } } ( \mathcal { Q } ) = \tau _ { \mathrm { D T E } }$ for every allocation policy Q.

Proof Under additivity, $C _ { i } ( e ) = 0$ for all units and exposure levels. Therefore $\bar { C } _ { \mathcal { Q } } =$ $\begin{array} { r } { n ^ { - 1 } \sum _ { i } \mathbb { E } _ { \mathcal { Q } } [ C _ { i } ( E _ { i } ) ] = 0 } \end{array}$ , and Proposition 3 gives

$$
\tau _ { \mathrm { E D E } } ( \mathcal { Q } ) = \frac { 1 } { n } \sum _ { i } \beta _ { i } = \tau _ { \mathrm { D T E } } .
$$

## C.6 Proof of Proposition 2

Restatement. Under a Bernoulli assignment, the exposure mapping versions of EDE and EIE decompose the derivative of the expected population outcome:

$$
\tau _ { \mathrm { T P E } } ( \pi ) = \tau _ { \mathrm { E D E } } ( \pi ) + \tau _ { \mathrm { E I E } } ( \pi ) .
$$

Proof Let the treatment vector be $\mathbf { Z } = \left( Z _ { 1 } , \ldots , Z _ { n } \right)$ , where $Z _ { i } \sim \mathrm { B e r n o u l l i } ( \pi _ { i } )$ are independent. Let $e _ { i } = f ( { \bf Z } _ { N _ { i } } )$ be the exposure mapping for unit i. The expected population outcome under the policy $\pi = ( \pi _ { 1 } , \ldots , \pi _ { n } )$ is

$$
\mu ( \pi ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { \pi } [ Y _ { i } ( Z _ { i } , e _ { i } ) ] .
$$

By the multivariate chain rule,

$$
\pi _ { \mathrm { T P E } } ( \pi ) = \sum _ { j = 1 } ^ { n } \frac { \partial } { \partial \pi _ { j } } \mu ( \pi ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left\{ \frac { \partial } { \partial \pi _ { i } } \mathbb { E } _ { \pi } [ Y _ { i } ( Z _ { i } , e _ { i } ) ] + \sum _ { j \in N _ { i } } \frac { \partial } { \partial \pi _ { j } } \mathbb { E } _ { \pi } [ Y _ { i } ( Z _ { i } , e _ { i } ) ] \right\} ,
$$

where the partial derivative is zero for $j \notin N _ { i } \cup \{ i \}$

For the own-treatment derivative, the independence of $Z _ { i }$ from $e _ { i }$ under Bernoulli assignment gives

$$
\frac { \partial } { \partial \pi _ { i } } \mathbb { E } _ { \boldsymbol \pi } [ Y _ { i } ( Z _ { i } , e _ { i } ) ] = \mathbb { E } _ { \boldsymbol \pi } [ Y _ { i } ( 1 , e _ { i } ) - Y _ { i } ( 0 , e _ { i } ) ] .
$$

Averaging over i gives τ<sub>EDE</sub>(π).

For $j \in N _ { i }$ , define

$$
e _ { i } ^ { + j } = e _ { i } ( Z _ { N _ { i } \backslash j } , z _ { j } = 1 ) , \qquad e _ { i } ^ { - j } = e _ { i } ( Z _ { N _ { i } \backslash j } , z _ { j } = 0 ) .
$$

Again using the product structure of the Bernoulli design,

$$
\frac { \partial } { \partial \pi _ { j } } \mathbb { E } _ { \pi } [ Y _ { i } ( Z _ { i } , e _ { i } ) ] = \mathbb { E } _ { \pi } \left[ Y _ { i } ( Z _ { i } , e _ { i } ^ { + j } ) - Y _ { i } ( Z _ { i } , e _ { i } ^ { - j } ) \right] .
$$

Summing over $j \in N _ { i }$ and averaging over i gives $\tau _ { \mathrm { E I E } } ( \pi )$ . Combining the own-treatment and neighbor-treatment terms proves

$$
\tau _ { \mathrm { T P E } } ( \pi ) = \tau _ { \mathrm { E D E } } ( \pi ) + \tau _ { \mathrm { E I E } } ( \pi ) .
$$

## C.7 Symmetric exposure mappings

Proposition 26 (EIE under symmetric exposure) Suppose the assignment is Bernoulli with common probability π, and suppose the exposure mapping is symmetric. If

$$
e _ { i } ( { \bf Z } _ { N _ { i } } ) = \sum _ { j \in N _ { i } } Z _ { j } , \qquad d _ { i } = | N _ { i } | ,
$$

then the expected indirect efect can be written as

$$
\tau _ { \mathrm { E I E } } ( \pi ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } d _ { i } \mathbb { E } _ { \pi } \left[ Y _ { i } ( Z _ { i } , \widetilde { E } _ { i } + 1 ) - Y _ { i } ( Z _ { i } , \widetilde { E } _ { i } ) \right] ,
$$

where ${ \widetilde { E } } _ { i }$ is the exposure generated by the remaining $d _ { i } - 1$ neighbors. If instead

$$
e _ { i } ( \mathbf { Z } _ { N _ { i } } ) = \frac { 1 } { d _ { i } } \sum _ { j \in N _ { i } } Z _ { j } ,
$$

then

$$
\tau _ { \mathrm { E I E } } ( \pi ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } d _ { i } \mathbb { E } _ { \pi } \left[ Y _ { i } \left( Z _ { i } , \widetilde { E } _ { i } + \frac { 1 } { d _ { i } } \right) - Y _ { i } \left( Z _ { i } , \widetilde { E } _ { i } \right) \right] ,
$$

where ${ \widetilde { E } } _ { i }$ is the fractional exposure generated by the remaining $d _ { i } - 1$ neighbors.

Proof For the count exposure, fix i and $j \in N _ { i }$ . Since the Bernoulli design has a common treatment probability and the exposure mapping is symmetric, the distribution of the exposure generated by $N _ { i } \setminus \{ j \}$ is the same for every neighbor j. Moreover,

$$
e _ { i } ^ { + j } = \widetilde { E } _ { i } + 1 , \qquad e _ { i } ^ { - j } = \widetilde { E } _ { i } .
$$

Substituting these two expressions into the definition of $\tau _ { \mathrm { E I E } } ( \pi )$ and summing the identical neighbor contributions over $j \in N _ { i }$ gives

$$
\tau _ { \mathrm { E I E } } ( \pi ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } d _ { i } \mathbb { E } _ { \pi } \Big [ Y _ { i } ( Z _ { i } , \widetilde { E } _ { i } + 1 ) - Y _ { i } ( Z _ { i } , \widetilde { E } _ { i } ) \Big ] .
$$

For fractional exposure, changing $Z _ { j }$ from zero to one increases exposure by $1 / d _ { i }$ . Thus

$$
e _ { i } ^ { + j } = \widetilde { E } _ { i } + \frac { 1 } { d _ { i } } , \qquad e _ { i } ^ { - j } = \widetilde { E } _ { i } ,
$$

and the same averaging argument gives the second formula.

## Appendix D. Proofs for Section 4

## D.1 Proof of Lemma 8

Proof If there is no dependence-graph edge between two disjoint vertex sets A and $B ,$ then $\mathcal { C } _ { i } \cap \mathcal { C } _ { j } = \emptyset$ for every $i \in A$ and $j \in B$ . Hence the assignment coordinates in $\cup _ { i \in A } \mathcal { C } _ { i }$ are disjoint from the assignment coordinates in $\cup _ { j \in B } \mathcal { C } _ { j }$ . Since treatment assignments are mutually independent, the two assignment subvectors are independent. Because each $X _ { i }$ is a function of $\mathbf { Z } _ { \mathcal { C } _ { i } }$ , the collections $\{ X _ { i } : i \in A \}$ and $\{ X _ { j } : j \in B \}$ are independent. Thus the graph defined by $c _ { i j } = 1 \{ \mathcal { C } _ { i } \cap \mathcal { C } _ { j } \neq \emptyset \}$ is a valid dependence graph.

Now suppose the interference graph is undirected. Then $\mathcal { C } _ { i } \cap \mathcal { C } _ { j } \neq \emptyset$ if and only if i and $j$ are at graph distance at most two. Hence $c _ { i } = | B _ { 2 } ( i ) | - 1$ . Since every neighbor of i lies in $B _ { 2 } ( i ) , c _ { i } \geq d _ { i }$ , and averaging gives $c _ { \mathrm { a v g } } \geq \bar { d } .$ . For the upper bound, the vertices within graph distance at most two from i, excluding i, consist of the $d _ { i }$ neighbors of i and the additional vertices reached through those neighbors. Thus

$$
c _ { i } \leq d _ { i } + \sum _ { \ell \in N _ { i } } ( d _ { \ell } - 1 ) = \sum _ { \ell \in N _ { i } } d _ { \ell } .
$$

Averaging over i and using the symmetry of the undirected graph gives

$$
c _ { \mathrm { a v g } } \leq \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { \ell \in N _ { i } } d _ { \ell } = \frac { 1 } { n } \sum _ { \ell = 1 } ^ { n } d _ { \ell } ^ { 2 } .
$$

The bound $c _ { \mathrm { a v g } } \leq n - 1$ is immediate because each unit has at most $n - 1$ other units in its dependence neighborhood. ■

## D.2 A generic design-dependence bound

The following auxiliary result separates dependence generated by overlapping assignment coordinates from dependence induced by the randomization design, using the design-dependence coeficient in Definition 9.

Proposition 27 (Interference and design dependence bound) Consider a generic designbased estimator of the form

$$
{ \hat { \Psi } } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \psi _ { i } ( \mathbf { Z } ) ,
$$

where $\psi _ { i } ( \mathbf { Z } )$ is the weighted observed-outcome contribution ofunit i. Suppose that $\psi _ { i }$ depends on the assignment only through $\mathbf { Z } _ { \mathcal { C } _ { i } }$ , and suppose $| \psi _ { i } ( \mathbf { Z } ) | \leq B _ { i }$ . Define

$$
\ell _ { i j } = \mathbb { 1 } \{ \mathcal { C } _ { i } \cap \mathcal { C } _ { j } \neq \emptyset \} , \qquad i \neq j .
$$

Define

$$
B _ { 2 , n } ^ { 2 } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } B _ { i } ^ { 2 } ,
$$

$$
L _ { B } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } \ell _ { i j } B _ { i } B _ { j } ,
$$

and

$$
R _ { B } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } ( 1 - \ell _ { i j } ) \rho _ { i j } ^ { P } B _ { i } B _ { j } .
$$

Then, under an arbitrary design $P$

$$
\mathrm { V a r } _ { P } \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \psi _ { i } ( \mathbf { Z } ) \right) \leq \frac { 1 } { n } \{ B _ { 2 , n } ^ { 2 } + L _ { B } + R _ { B } \} .
$$

Proof Write $\begin{array} { r } { \hat { \Psi } = n ^ { - 1 } \sum _ { i } \psi _ { i } ( \mathbf { Z } ) } \end{array}$ . The variance expansion gives

$$
\mathrm { V a r } _ { P } ( \hat { \Psi } ) = \frac { 1 } { n ^ { 2 } } \left[ \sum _ { i } \mathrm { V a r } _ { P } \{ \psi _ { i } ( { \bf Z } ) \} + \sum _ { i } \sum _ { j \ne i } \mathrm { C o v } _ { P } \{ \psi _ { i } ( { \bf Z } ) , \psi _ { j } ( { \bf Z } ) \} \right] .
$$

Since $| \psi _ { i } | \leq B _ { i }$ , Var $P ( \psi _ { i } ) \leq B _ { i } ^ { 2 }$ . If $\ell _ { i j } = 1$ , then

$$
| \mathrm { C o v } _ { P } \{ \psi _ { i } ( \mathbf { Z } ) , \psi _ { j } ( \mathbf { Z } ) \} | \leq \mathbb { E } _ { P } | \psi _ { i } ( \mathbf { Z } ) \psi _ { j } ( \mathbf { Z } ) | \leq B _ { i } B _ { j } .
$$

If $\ell _ { i j } = 0$ , then $\mathcal { C } _ { i } \cap \mathcal { C } _ { j } = \varnothing$ . When $B _ { i } B _ { j } > 0$ , set $f ( \mathbf { Z } _ { \mathcal { C } _ { i } } ) = \psi _ { i } ( \mathbf { Z } ) / B _ { i }$ and $g ( \mathbf { Z } _ { \mathcal { C } _ { j } } ) = \psi _ { j } ( \mathbf { Z } ) / B _ { j }$ These functions have sup-norm at most one, and therefore

$$
\begin{array} { r } { | \mathrm { C o v } _ { P } \{ \psi _ { i } ( \mathbf { Z } ) , \psi _ { j } ( \mathbf { Z } ) \} | \leq \rho _ { i j } ^ { P } B _ { i } B _ { j } . } \end{array}
$$

If $B _ { i } B _ { j } = 0$ , the same bound is immediate. Substituting these three bounds into the variance expansion proves the result. ■

## D.3 Residual design dependence under complete randomization

The residual design-dependence term can be bounded explicitly for complete randomization. Suppose the design assigns exactly m of the n units to treatment. If $A , B \subseteq \{ 1 , \dots , n \}$ are disjoint assignment-coordinate sets with $| A | = a$ and $| B | = b ,$ then

$$
\rho _ { P } ( A , B ) \leq C \operatorname* { m i n } \left\{ 1 , { \frac { a b } { n - a - b } } \right\} ,
$$

for a universal constant $C _ { i }$ , whenever $a + b < n$ . Consequently, for local HT/IPW estimators whose ith summand depends on $\mathcal { C } _ { i } = \{ i \} \cup N _ { i }$

$$
r _ { \arg \bigstar } \leq \frac { C } { n } \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } ( 1 - c _ { i j } ) \operatorname* { m i n } \left\{ 1 , \frac { | { \mathcal { C } } _ { i } | | { \mathcal { C } } _ { j } | } { n - | { \mathcal { C } } _ { i } | - | { \mathcal { C } } _ { j } | } \right\} .
$$

In particular, if ma $\mathfrak { c } _ { i } \left| \mathcal { C } _ { i } \right| \le n / 4$ , then

$$
r _ { \arg \operatorname { S } } \leq \frac { C } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } ( 1 - c _ { i j } ) | \mathcal { C } _ { i } | | \mathcal { C } _ { j } | \leq C \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | \mathcal { C } _ { i } | \right) ^ { 2 } .
$$

Thus on a d-regular local interference graph, $r _ { \mathrm { a v g } } = O ( d ^ { 2 } )$ , matching the order of $c _ { \mathrm { a v g } } \asymp d ^ { 2 }$ For bounded-degree graphs, $r _ { \mathrm { a v g } } = O ( 1 )$ , so the residual design-dependence contribution $H \sqrt { r _ { \mathrm { a v g } } / n }$ has the same $n ^ { - 1 / 2 }$ scaling as the Bernoulli dependence term, up to the relevant propensity scale H.

Proof It sufices to bound the dependence between two disjoint subvectors. For a fixed configuration $x _ { A }$ with $\begin{array} { r } { s = \sum _ { \ell \in A } x _ { \ell } } \end{array}$ , the conditional distribution of $\mathbf { Z } _ { B }$ under complete randomization is sampling without replacement from a population of size $n - a$ containing $m - s$ treated labels. The marginal distribution of $\mathbf { Z } _ { B }$ is sampling without replacement from a population of size n containing m treated labels.

Couple the two draws sequentially over the b coordinates in B. After $k < b$ coordinates in B have been exposed and r treated labels have appeared among them, the two conditional probabilities that the next coordinate is treated are

$$
{ \frac { m - s - r } { n - a - k } } \qquad { \mathrm { a n d } } \qquad { \frac { m - r } { n - k } } .
$$

Their absolute diference is at most $C a / ( n - a - b )$ , uniformly over admissible $s , r , k$ . The union bound over the b sequential draws gives a total-variation bound of order $a b / ( n - a - b )$ , truncated at one. For any $\| g \| _ { \infty } \leq 1$ , the diference between the conditional and marginal expectations of $g ( \mathbf { Z } _ { B } )$ is at most twice this total variation distance. Taking the supremum also over $\| f \| _ { \infty } \leq 1$ gives the displayed bound for $\rho _ { P } ( A , B )$

Substituting $A = \mathcal { C } _ { i } , B = \mathcal { C } _ { j }$ for pairs with $\mathcal { C } _ { i } \cap \mathcal { C } _ { j } = \emptyset$ gives the first bound on $r _ { \mathrm { a v g } }$ . The simplified bound follows from $| \mathcal { C } _ { i } | + | \mathcal { C } _ { j } | \le n / 2$ , and the final regular and bounded-degree statements follow by setting $| { \mathcal { C } } _ { i } | = d + 1$ or max<sub>i</sub> $| \mathcal { C } _ { i } | = O ( 1 )$

## D.4 Proof of Proposition 10

Restatement. Assume $\pi _ { i } ( d _ { i } ^ { + } ) > 0$ and $\pi _ { i } ( d _ { i } ^ { - } ) > 0$ for all $i ,$ and define

$$
\hat { \tau } _ { \mathrm { A C E } } ^ { H T } ( d ^ { + } , d ^ { - } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Y _ { i } ^ { o b s } \left\{ \frac { \mathbb { 1 } \{ D _ { i } ( \mathbf { Z } ) = d _ { i } ^ { + } \} } { \pi _ { i } ( d _ { i } ^ { + } ) } - \frac { \mathbb { 1 } \{ D _ { i } ( \mathbf { Z } ) = d _ { i } ^ { - } \} } { \pi _ { i } ( d _ { i } ^ { - } ) } \right\} .
$$

Then $\hat { \tau } _ { \mathrm { A C E } } ^ { H T } ( d ^ { + } , d ^ { - } )$ is design-unbiased for $\tau _ { \mathrm { A C E } } ( d ^ { + } , d ^ { - } )$ . $\mathrm { I f ~ } | Y _ { i } ( d ) | \leq M$ and the ith summand depends on the assignment only through $\mathbf { Z } _ { \mathcal { C } _ { i } }$ , then the variance is bounded as stated. Proof Because $Y _ { i } ^ { o b s } = Y _ { i } \{ { D _ { i } ( { \bf Z } ) } \}$ , the expectation of the first weighted term is

$$
\mathbb { E } \left[ Y _ { i } ^ { o b s } \frac { \mathbb { 1 } \{ D _ { i } ( \mathbf { Z } ) = d _ { i } ^ { + } \} } { \pi _ { i } ( d _ { i } ^ { + } ) } \right] = \frac { 1 } { \pi _ { i } ( d _ { i } ^ { + } ) } \mathbb { E } \left[ Y _ { i } \{ D _ { i } ( \mathbf { Z } ) \} \mathbb { 1 } \{ D _ { i } ( \mathbf { Z } ) = d _ { i } ^ { + } \} \right] = Y _ { i } ( d _ { i } ^ { + } ) .
$$

The same calculation gives

$$
\mathbb { E } \left[ Y _ { i } ^ { o b s } \frac { \mathbb { 1 } \left\{ D _ { i } ( \mathbf { Z } ) = d _ { i } ^ { - } \right\} } { \pi _ { i } ( d _ { i } ^ { - } ) } \right] = Y _ { i } ( d _ { i } ^ { - } ) .
$$

Therefore

$$
\mathbb { E } \{ \hat { \tau } _ { \mathrm { A C E } } ^ { H T } ( d ^ { + } , d ^ { - } ) \} = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \{ Y _ { i } ( d _ { i } ^ { + } ) - Y _ { i } ( d _ { i } ^ { - } ) \} = \tau _ { \mathrm { A C E } } ( d ^ { + } , d ^ { - } ) .
$$

For the variance bound, write

$$
X _ { i } ^ { A C E } = Y _ { i } \{ D _ { i } ( \mathbf { Z } ) \} \left\{ \frac { \mathbb { 1 } \{ D _ { i } ( \mathbf { Z } ) = d _ { i } ^ { + } \} } { \pi _ { i } ( d _ { i } ^ { + } ) } - \frac { \mathbb { 1 } \{ D _ { i } ( \mathbf { Z } ) = d _ { i } ^ { - } \} } { \pi _ { i } ( d _ { i } ^ { - } ) } \right\} .
$$

Then $\begin{array} { r } { \hat { \tau } _ { \mathrm { A C E } } ^ { H T } ( d ^ { + } , d ^ { - } ) = n ^ { - 1 } \sum _ { i } X _ { i } ^ { A C E } } \end{array}$ . Since $| Y _ { i } ( d ) | \leq M$ 2

$$
| X _ { i } ^ { A C E } | \leq M \left\{ \frac { 1 } { \pi _ { i } ( d _ { i } ^ { + } ) } + \frac { 1 } { \pi _ { i } ( d _ { i } ^ { - } ) } \right\} = M H _ { i } .
$$

Because the two target events in the ACE contrast are disjoint,

$$
\mathbb { E } \{ ( X _ { i } ^ { A C E } ) ^ { 2 } \} \le M ^ { 2 } \left\{ \frac { 1 } { \pi _ { i } ( d _ { i } ^ { + } ) } + \frac { 1 } { \pi _ { i } ( d _ { i } ^ { - } ) } \right\} = M ^ { 2 } H _ { i } .
$$

Thus Var $( X _ { i } ^ { A C E } ) \leq M ^ { 2 } H _ { i }$ . If $c _ { i j } = 1$ , then

$$
| \operatorname { C o v } ( X _ { i } ^ { A C E } , X _ { j } ^ { A C E } ) | \leq \mathbb { E } | X _ { i } ^ { A C E } X _ { j } ^ { A C E } | \leq M ^ { 2 } H _ { i } H _ { j } .
$$

If $c _ { i j } = 0$ , then the closed interference neighborhoods are disjoint. When $H _ { i } H _ { j } > 0$ , the functions $X _ { i } ^ { A C E } / ( M H _ { i } )$ and $X _ { j } ^ { A C E } / ( M H _ { j } )$ have sup-norm at most one, so Definition 9 gives

$$
| \operatorname { C o v } ( X _ { i } ^ { A C E } , X _ { j } ^ { A C E } ) | \leq M ^ { 2 } \rho _ { i j } H _ { i } H _ { j } .
$$

The same bound is immediate when $H _ { i } H _ { j } = 0$ . Combining the diagonal and of-diagonal terms gives

$$
\begin{array} { r l r } {  { \operatorname { V a r } \{ \hat { \tau } _ { \mathrm { A C E } } ^ { H T } ( d ^ { + } , d ^ { - } ) \} = \frac { 1 } { n ^ { 2 } } \{ \sum _ { i = 1 } ^ { n } \operatorname { V a r } ( X _ { i } ^ { A C E } ) + \sum _ { i = 1 } ^ { n } \sum _ { j \ne i } \operatorname { C o v } ( X _ { i } ^ { A C E } , X _ { j } ^ { A C E } ) \} } } \\ & { } & { \leq \frac { M ^ { 2 } } { n ^ { 2 } } \{ \sum _ { i = 1 } ^ { n } H _ { i } + \sum _ { i = 1 } ^ { n } \sum _ { j \ne i } c _ { i j } H _ { i } H _ { j } + \sum _ { i = 1 } ^ { n } \sum _ { j \ne i } ( 1 - c _ { i j } ) \rho _ { i j } H _ { i } H _ { j } \} . } \end{array}
$$

If $H _ { i } \leq H$ for all $i ,$ this becomes

$$
\begin{array} { l } { { \mathrm { V a r } \{ \hat { \tau } _ { \mathrm { A C E } } ^ { H T } ( d ^ { + } , d ^ { - } ) \} \le \displaystyle \frac { M ^ { 2 } } { n ^ { 2 } } \left\{ n H + H ^ { 2 } \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } c _ { i j } + H ^ { 2 } \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } ( 1 - c _ { i j } ) \rho _ { i j } \right\} } } \\ { { = M ^ { 2 } \left\{ \displaystyle \frac { H } { n } + H ^ { 2 } \frac { c _ { \mathrm { a v g } } + r _ { \mathrm { a v g } } } { n } \right\} . } } \end{array}
$$

## D.5 Proof of Proposition 11

Restatement. Fix the design $P _ { n }$ and the induced treatment–exposure combinations $D _ { i } ( \mathbf { Z } )$ Suppose that $\hat { \tau } _ { n }$ is design-unbiased for $\tau _ { \mathrm { A C E } } ( d ^ { + } , d ^ { - } )$ for every bounded potential-outcome table. Then

$$
\operatorname* { s u p } _ { | Y | \leq M } \operatorname { V a r } _ { P _ { n } } ( \hat { \tau } _ { n } ) \geq \frac { M ^ { 2 } } { n ^ { 2 } } \operatorname* { m a x } _ { i } \left\{ \frac { 1 } { \pi _ { i } ( d _ { i } ^ { + } ) } - 1 , \frac { 1 } { \pi _ { i } ( d _ { i } ^ { - } ) } - 1 \right\} .
$$

Proof Fix a unit i and a target state $d _ { i } ^ { + }$ . Let

$$
A _ { i } = \{ D _ { i } ( { \bf Z } ) = d _ { i } ^ { + } \} , \qquad \pi _ { i } ( d _ { i } ^ { + } ) = P _ { n } ( A _ { i } ) .
$$

Construct two potential-outcome tables $Y ^ { + }$ and $Y ^ { \cdot }$ <sup>−</sup> that agree everywhere except at the coordinate $Y _ { i } ( d _ { i } ^ { + } )$ , where

$$
Y _ { i } ^ { + } ( d _ { i } ^ { + } ) = M , \qquad Y _ { i } ^ { - } ( d _ { i } ^ { + } ) = - M .
$$

All other potential outcomes are the same in the two tables. If this coordinate enters the estimand with coeficient $c _ { i } .$ , then

$$
\Delta = \tau ( Y ^ { + } ) - \tau ( Y ^ { - } ) = 2 M c _ { i } .
$$

Let

$$
W ( Z ) = \hat { \tau } _ { n } ( Z , Y ^ { + , \mathrm { o b s } } ) - \hat { \tau } _ { n } ( Z , Y ^ { - , \mathrm { o b s } } ) .
$$

By design unbiasedness, $\mathbb { E } [ W ] = \Delta$ . On the event $A _ { i } ^ { c }$ , the two tables produce identical observed data, and hence $W = 0$ . Therefore

$$
\Delta = \mathbb { E } \{ W \mathbb { 1 } ( A _ { i } ) \} .
$$

By Cauchy–Schwarz,

$$
\Delta ^ { 2 } \leq \mathbb { E } ( W ^ { 2 } ) P _ { n } ( A _ { i } ) = \mathbb { E } ( W ^ { 2 } ) \pi _ { i } ( d _ { i } ^ { + } ) ,
$$

so $\mathbb { E } ( W ^ { 2 } ) \ge \Delta ^ { 2 } / \pi _ { i } ( d _ { i } ^ { + } )$ . Since $\mathbb { E } ( W ) = \Delta$

$$
\mathrm { V a r } ( W ) = \mathbb { E } ( W ^ { 2 } ) - \Delta ^ { 2 } \ge \Delta ^ { 2 } \left\{ \frac { 1 } { \pi _ { i } ( d _ { i } ^ { + } ) } - 1 \right\} .
$$

On the other hand,

$$
W - \Delta = \{ \hat { \tau } _ { n } ( Y ^ { + } ) - \tau ( Y ^ { + } ) \} - \{ \hat { \tau } _ { n } ( Y ^ { - } ) - \tau ( Y ^ { - } ) \} .
$$

Thus

$$
\operatorname { V a r } ( W ) \leq 4 \operatorname* { m a x } \left\{ \operatorname { V a r } _ { P _ { n } } ( \hat { \tau } _ { n } ; Y ^ { + } ) , \operatorname { V a r } _ { P _ { n } } ( \hat { \tau } _ { n } ; Y ^ { - } ) \right\} .
$$

Combining the last two displays gives

$$
\operatorname* { s u p } _ { | Y | \leq M } \mathrm { V a r } _ { P _ { n } } ( \hat { \tau } _ { n } ; Y ) \geq \frac { \Delta ^ { 2 } } { 4 } \left\{ \frac { 1 } { \pi _ { i } ( d _ { i } ^ { + } ) } - 1 \right\} = c _ { i } ^ { 2 } M ^ { 2 } \left\{ \frac { 1 } { \pi _ { i } ( d _ { i } ^ { + } ) } - 1 \right\} .
$$

For an average ACE, the coeficient of each arm is $c _ { i } = \pm 1 / n$ . Applying the same argument to either $d _ { i } ^ { + }$ or $d _ { i } ^ { - }$ , and then taking the maximum over units and arms, gives the claimed bound. 7

## D.6 Proof of Proposition 12

Restatement. Suppose $\lambda _ { i } ^ { + } ( d ) = \lambda _ { i } ^ { - } ( d ) = 0$ whenever $\pi _ { i } ^ { P } ( d ) = 0$ , and define $h _ { i } ^ { + } ( \mathbf { z } )$ and $h _ { i } ^ { - } ( { \bf z } )$ as in the proposition. Let

$$
V _ { i } = \sum _ { d \in \mathcal { X } _ { i } : \pi _ { i } ^ { P } ( d ) > 0 } \frac { \{ \lambda _ { i } ^ { + } ( d ) - \lambda _ { i } ^ { - } ( d ) \} ^ { 2 } } { \pi _ { i } ^ { P } ( d ) } .
$$

Let

$$
\hat { \tau } _ { \mathrm { E C E } } ^ { H T } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \{ h _ { i } ^ { + } ( { \bf Z } ) - h _ { i } ^ { - } ( { \bf Z } ) \} Y _ { i } ^ { o b s } .
$$

Then $\hat { \tau } _ { \mathrm { E C E } } ^ { H T }$ is design-unbiased for

$$
\tau _ { \mathrm { E C E } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { d \in \mathcal { X } _ { i } } Y _ { i } ( d ) \{ \lambda _ { i } ^ { + } ( d ) - \lambda _ { i } ^ { - } ( d ) \} .
$$

If $| Y _ { i } ( \mathbf { z } ) | \leq M$ and

$$
X _ { i } = \{ h _ { i } ^ { + } ( { \bf Z } ) - h _ { i } ^ { - } ( { \bf Z } ) \} Y _ { i } \{ D _ { i } ( { \bf Z } ) \}
$$

depends on the assignment only through $\mathbf { Z } _ { \mathcal { C } _ { i } }$ , then

$$
\mathrm { V a r } ( \hat { \tau } _ { \mathrm { E C E } } ^ { H T } ) \leq \frac { M ^ { 2 } } { n ^ { 2 } } \left\{ \sum _ { i = 1 } ^ { n } V _ { i } + \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } c _ { i j } H _ { i } H _ { j } + \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } ( 1 - c _ { i j } ) \rho _ { i j } H _ { i } H _ { j } \right\} .
$$

Proof Because $Y _ { i } ^ { o b s } = Y _ { i } ( { \bf Z } )$ , the expectation of the ith summand is

$$
{ \mathbb E } [ \{ h _ { i } ^ { + } ( { \bf Z } ) - h _ { i } ^ { - } ( { \bf Z } ) \} Y _ { i } ^ { o b s } ] = \sum _ { { \bf z } \in \Omega } p ( { \bf z } ) \{ h _ { i } ^ { + } ( { \bf z } ) - h _ { i } ^ { - } ( { \bf z } ) \} Y _ { i } \{ D _ { i } ( { \bf z } ) \} .
$$

Grouping allocations by the treatment–exposure combination they induce gives the plusarm contribution

$$
\begin{array} { l } { { \displaystyle \sum _ { { \bf z } \in \Omega } p ( { \bf z } ) h _ { i } ^ { + } ( { \bf z } ) Y _ { i } \{ D _ { i } ( { \bf z } ) \} } } \\ { { \displaystyle ~ = \sum _ { \scriptstyle d \in \mathcal { X } _ { i } : \pi _ { i } ^ { P } ( d ) > 0 } Y _ { i } ( d ) \lambda _ { i } ^ { + } ( d ) \frac { \sum _ { { \bf z } \in \Omega : D _ { i } ( { \bf z } ) = d } p ( { \bf z } ) } { \pi _ { i } ^ { P } ( d ) } } } \\ { { \displaystyle ~ = \sum _ { \scriptstyle d \in \mathcal { X } _ { i } : \pi _ { i } ^ { P } ( d ) > 0 } Y _ { i } ( d ) \lambda _ { i } ^ { + } ( d ) = \sum _ { \scriptstyle d \in \mathcal { X } _ { i } } Y _ { i } ( d ) \lambda _ { i } ^ { + } ( d ) , } } \end{array}
$$

where the last equality uses the positivity condition. The same calculation for the minus arm gives

$$
\mathbb { E } \{ h _ { i } ^ { - } ( \mathbf { Z } ) Y _ { i } ^ { o b s } \} = \sum _ { d \in \mathcal { X } _ { i } } Y _ { i } ( d ) \lambda _ { i } ^ { - } ( d ) .
$$

Summing over i and dividing by n proves design unbiasedness:

$$
\mathbb { E } \big ( \hat { \tau } _ { \mathrm { E C E } } ^ { H T } \big ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { d \in \mathcal { X } _ { i } } Y _ { i } ( d ) \{ \lambda _ { i } ^ { + } ( d ) - \lambda _ { i } ^ { - } ( d ) \} = \tau _ { \mathrm { E C E } } .
$$

For the variance bound, write

$$
X _ { i } = \{ h _ { i } ^ { + } ( { \bf Z } ) - h _ { i } ^ { - } ( { \bf Z } ) \} Y _ { i } \{ D _ { i } ( { \bf Z } ) \} .
$$

Then

$$
\hat { \tau } _ { \mathrm { E C E } } ^ { H T } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } X _ { i } .
$$

By boundedness, $| X _ { i } | \le M H _ { i }$ , so

$$
\mathrm { V a r } ( X _ { i } ) \leq \mathbb { E } ( X _ { i } ^ { 2 } ) \leq M ^ { 2 } \sum _ { \substack { d \in \mathcal { X } _ { i } : \pi _ { i } ^ { P } ( d ) > 0 } } \frac { \{ \lambda _ { i } ^ { + } ( d ) - \lambda _ { i } ^ { - } ( d ) \} ^ { 2 } } { \pi _ { i } ^ { P } ( d ) } = M ^ { 2 } V _ { i } .
$$

If $c _ { i j } = 1$ , Cauchy–Schwarz gives

$$
| \operatorname { C o v } ( X _ { i } , X _ { j } ) | \leq \mathbb { E } | X _ { i } X _ { j } | \leq M ^ { 2 } H _ { i } H _ { j } .
$$

If $c _ { i j } = 0$ , then the closed interference neighborhoods are disjoint. When $H _ { i } H _ { j } > 0$ , the functions $X _ { i } / ( M H _ { i } )$ and $X _ { j } / ( M H _ { j } )$ have sup-norm at most one, so Definition 9 gives

$$
| \operatorname { C o v } ( X _ { i } , X _ { j } ) | \leq M ^ { 2 } \rho _ { i j } H _ { i } H _ { j } .
$$

The same bound is immediate when $H _ { i } H _ { j } = 0$ . Therefore

$$
\begin{array} { r l r } {  { \mathrm { V a r } ( \hat { \tau } _ { \mathrm { E C E } } ^ { H T } ) = \frac { 1 } { n ^ { 2 } } \{ \sum _ { i = 1 } ^ { n } \mathrm { V a r } ( X _ { i } ) + \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } \mathrm { C o v } ( X _ { i } , X _ { j } ) \} } } \\ & { } & { \leq \frac { M ^ { 2 } } { n ^ { 2 } } \{ \underset { i = 1 } { \overset { n } { \sum } } V _ { i } + \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } c _ { i j } H _ { i } H _ { j } + \underset { i = 1 } { \overset { n } { \sum } } \sum _ { j \neq i } ( 1 - c _ { i j } ) \rho _ { i j } H _ { i } H _ { j } \} . } \end{array}
$$

If $V _ { i } \leq V$ and $H _ { i } \leq H$ for all i, this becomes

$$
\left. \begin{array} { l } { { \displaystyle \mathrm { V a r } ( \hat { \tau } _ { \mathrm { E C E } } ^ { H T } ) \leq \frac { M ^ { 2 } } { n ^ { 2 } } \left\{ n V + H ^ { 2 } \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } c _ { i j } + H ^ { 2 } \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } ( 1 - c _ { i j } ) \rho _ { i j } \right\} } } \\ { { \displaystyle \quad = M ^ { 2 } \left\{ \frac { V } { n } + H ^ { 2 } \frac { c _ { \mathrm { a v g } } + r _ { \mathrm { a v g } } } { n } \right\} . } } \end{array} \right.
$$

## D.7 Weighted dependence bound for graph-aware ECEs

Definition 28 (Weighted summand and dependence complexity) Write $h _ { i } ( \mathbf { Z } ) = h _ { i } ^ { + } ( \mathbf { Z } )$ $h _ { i } ^ { - } ( { \bf Z } )$ and $\psi _ { i } ( \mathbf { Z } ) = h _ { i } ( \mathbf { Z } ) Y _ { i } \{ D _ { i } ( \mathbf { Z } ) \}$ . For the canonical ECE estimator $\begin{array} { r } { \hat { \tau } _ { h } = n ^ { - 1 } \sum _ { i } \psi _ { i } ( \mathbf { Z } ) } \end{array}$ define

$$
\omega _ { i j } ( Y ) = \operatorname { C o v } \{ \psi _ { i } ( \mathbf { Z } ) , \psi _ { j } ( \mathbf { Z } ) \} , \qquad i \neq j ,
$$

and

$$
\Gamma _ { h } ( M ) = \operatorname* { s u p } _ { | Y | \leq M } \left[ \sum _ { i = 1 } ^ { n } \mathrm { V a r } \{ \psi _ { i } ( \mathbf { Z } ) \} + \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } \omega _ { i j } ( Y ) \right] .
$$

The supremum is over all bounded potential-outcome tables satisfying $| Y _ { i } ( \mathbf { z } ) | \leq M$

Proposition 29 (Weighted dependence bound for graph-aware ECEs) Assume independent Bernoulli assignment. Suppose the potential outcomes satisfy neighborhood interference with known neighborhoods $N _ { i }$ , and suppose $h _ { i } ( \mathbf { Z } )$ depends only on $\mathbf { Z } _ { \mathcal { C } _ { i } }$ . Then a valid dependence graph for $\psi _ { i } ( \mathbf { Z } ) = h _ { i } ( \mathbf { Z } ) Y _ { i } ( \mathbf { Z } )$ is given by

$$
c _ { i j } = \mathbb { 1 } \{ { \mathcal { C } } _ { i } \cap { \mathcal { C } } _ { j } \neq \emptyset \} , \qquad i \neq j .
$$

Moreover,

$$
\operatorname* { s u p } _ { | Y | \leq M } \operatorname { V a r } ( \hat { \tau } _ { h } ) = \frac { \Gamma _ { h } ( M ) } { n ^ { 2 } } .
$$

In particular, $i f \left| h _ { i } ( \mathbf { Z } ) \right| \leq H _ { i }$ , then

$$
\mathrm { V a r } ( { \hat { \tau } } _ { h } ) \leq \frac { M ^ { 2 } } { n ^ { 2 } } \left\{ \sum _ { i = 1 } ^ { n } H _ { i } ^ { 2 } + \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } c _ { i j } H _ { i } H _ { j } \right\} .
$$

This result separates two roles of the dependence graph. The unweighted graph gives a conservative upper bound by replacing every nonzero covariance by its maximal possible size. The sharper object is the weighted dependence complexity $\Gamma _ { h } ( M )$ , which keeps the covariance weights induced by the design contrast and the potential-outcome table.

## D.8 Proof of Proposition 29

Restatement. Assume independent Bernoulli assignment. Suppose the potential outcomes satisfy neighborhood interference with known neighborhoods $N _ { i }$ , and suppose $h _ { i } ( \mathbf { Z } )$ depends only on $\mathbf { Z } _ { C _ { i } } ,$ , where $\mathcal { C } _ { i } = \{ i \} \cup N _ { i }$ . Then a valid dependence graph for $\psi _ { i } ( \mathbf { Z } ) =$ $h _ { i } ( \mathbf { Z } ) Y _ { i } ( \mathbf { Z } )$ is given by

$$
c _ { i j } = \mathbb { 1 } \{ { \mathcal { C } } _ { i } \cap { \mathcal { C } } _ { j } \neq \emptyset \} , \qquad i \neq j .
$$

Moreover,

$$
\operatorname* { s u p } _ { | Y | \leq M } \operatorname { V a r } ( \hat { \tau } _ { h } ) = \frac { \Gamma _ { h } ( M ) } { n ^ { 2 } } .
$$

If $| h _ { i } ( \mathbf { Z } ) | \leq H _ { i }$ , then

$$
\mathrm { V a r } ( { \hat { \tau } } _ { h } ) \leq \frac { M ^ { 2 } } { n ^ { 2 } } \left\{ \sum _ { i = 1 } ^ { n } H _ { i } ^ { 2 } + \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } c _ { i j } H _ { i } H _ { j } \right\} .
$$

Proof By neighborhood interference, $Y _ { i } ( \mathbf { Z } )$ is a function only of $\mathbf { Z } _ { \mathcal { C } _ { i } }$ . By assumption, $h _ { i } ( \mathbf { Z } )$ is also a function only of $\mathbf { Z } _ { \mathcal { C } _ { i } }$ . Hence

$$
\psi _ { i } ( \mathbf { Z } ) = h _ { i } ( \mathbf { Z } ) Y _ { i } ( \mathbf { Z } )
$$

is a function only of $\mathbf { Z } _ { \mathcal { C } _ { i } }$ . If $\mathcal { C } _ { i } \cap \mathcal { C } _ { j } = \varnothing$ , then $\mathbf { Z } _ { \mathcal { C } _ { i } }$ and $\mathbf { Z } _ { \mathcal { C } _ { i } }$ are independent under independent Bernoulli assignment. Therefore $\psi _ { i } ( \mathbf { Z } )$ and $\psi _ { j } ( \mathbf { Z } )$ are independent. The graph with edge indicators $c _ { i j } = \mathbb { 1 } \{ \mathscr { C } _ { i } \cap \mathscr { C } _ { j } \neq \emptyset \}$ is therefore a valid dependence graph.

For a fixed potential-outcome table $Y .$

$$
\hat { \tau } _ { h } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \psi _ { i } ( \mathbf { Z } ) ,
$$

so

$$
\mathrm { V a r } ( \hat { \tau } _ { h } ) = \frac { 1 } { n ^ { 2 } } \left[ \sum _ { i = 1 } ^ { n } \mathrm { V a r } \{ \psi _ { i } ( { \bf Z } ) \} + \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } \mathrm { C o v } \{ \psi _ { i } ( { \bf Z } ) , \psi _ { j } ( { \bf Z } ) \} \right] .
$$

Taking the supremum over all bounded potential-outcome tables gives

$$
\operatorname* { s u p } _ { | Y | \leq M } \operatorname { V a r } ( \hat { \tau } _ { h } ) = \frac { \Gamma _ { h } ( M ) } { n ^ { 2 } }
$$

by Definition 28.

Finally, if $| Y _ { i } ( \mathbf { Z } ) | \leq M$ and $| h _ { i } ( \mathbf { Z } ) | \leq H _ { i }$ , then $| \psi _ { i } ( \mathbf { Z } ) | \leq M H _ { i }$ . Thus

$$
\mathrm { V a r } \{ \psi _ { i } ( { \bf Z } ) \} \leq M ^ { 2 } H _ { i } ^ { 2 } .
$$

If $c _ { i j } = 0$ , then $\psi _ { i }$ and $\psi _ { j }$ are independent and the covariance is zero. If $c _ { i j } = 1$ , then Cauchy–Schwarz gives

$$
| \operatorname { C o v } \{ \psi _ { i } ( \mathbf { Z } ) , \psi _ { j } ( \mathbf { Z } ) \} | \leq \{ \operatorname { V a r } ( \psi _ { i } ) \operatorname { V a r } ( \psi _ { j } ) \} ^ { 1 / 2 } \leq M ^ { 2 } H _ { i } H _ { j } .
$$

Substituting these bounds into the variance expansion gives the stated inequality. ■

## D.9 Proof of Proposition 13

Restatement. Assume the true potential outcomes satisfy neighborhood interference with neighborhoods $N _ { i }$ . Assume Bernoulli assignment with treatment probabilities bounded away from zero and one, let $\mathcal { Q } _ { p }$ denote this Bernoulli allocation policy, and assume $\mathrm { s u p } _ { \mathbf { z } } \left| Y _ { i } ( \mathbf { z } ) \right| \leq$ $M$ . Then $\hat { \tau } _ { \mathrm { E D E } } ^ { H T }$ is design-unbiased for $\tau _ { \mathrm { E D E } } ( \mathcal { Q } _ { p } )$ , and

$$
\operatorname { V a r } _ { P _ { n } } ( \hat { \tau } _ { \mathrm { E D E } } ^ { H T } ) \leq C _ { p } M ^ { 2 } \frac { 1 + c _ { \mathrm { a v g } } } { n } .
$$

In particular, $\hat { \tau } _ { \mathrm { E D E } } ^ { H T }$ is design-MSE consistent if $c _ { \mathrm { a v g } } = o ( n )$

Proof Write the estimator as n $^ { - 1 } \sum _ { i } X _ { i }$ , where

$$
X _ { i } = Y _ { i } ^ { o b s } \left\{ Z _ { i } / p _ { i } - ( 1 - Z _ { i } ) / ( 1 - p _ { i } ) \right\} .
$$

First condition on $Z _ { - i }$ . Since the assignment is Bernoulli, $Z _ { i }$ is independent of $Z _ { - i }$ , and hence

$$
\mathbb { E } ( X _ { i } \mid Z _ { - i } ) = Y _ { i } ( 1 , Z _ { - i } ) - Y _ { i } ( 0 , Z _ { - i } ) .
$$

Taking expectations over $Z _ { - i }$ gives

$$
\mathbb { E } \big ( \hat { \tau } _ { \mathrm { E D E } } ^ { H T } \big ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \{ \mathbb { E } [ Y _ { i } ( 1 , Z _ { - i } ) ] - \mathbb { E } [ Y _ { i } ( 0 , Z _ { - i } ) ] \} = \tau _ { \mathrm { E D E } } ( \mathcal { Q } _ { p } ) .
$$

The boundedness of the potential outcomes and the positivity of the treatment probabilities imply $| X _ { i } | \le C _ { p } M$ . If $\mathcal { C } _ { i } \cap \mathcal { C } _ { j } = \emptyset$ , then $X _ { i }$ and $X _ { j }$ depend on disjoint sets of independent treatment assignments and are therefore independent. Thus only pairs with $c _ { i j } = 1$ contribute to the covariance sum. This gives

$$
\mathrm { V a r } ( \hat { \tau } _ { \mathrm { E D E } } ^ { H T } ) \leq \frac { C _ { p } M ^ { 2 } } { n ^ { 2 } } \left( n + \sum _ { i } \sum _ { j \neq i } c _ { i j } \right) = C _ { p } M ^ { 2 } \frac { 1 + c _ { \mathrm { a v g } } } { n } .
$$

## D.10 Proof of Corollary 14

Restatement. Assume the true potential outcomes satisfy neighborhood interference with known neighborhoods $N _ { i }$ . Assume Bernoulli assignment with treatment probabilities bounded away from zero and one, and assume $\operatorname { s u p } _ { \mathbf { z } } \left| Y _ { i } ( \mathbf { z } ) \right| \leq M$ . Then $\hat { \tau } _ { \mathrm { E I E } } ^ { G }$ is designunbiased for $\tau _ { \mathrm { E I E } } ( \pi )$ , and

$$
\mathrm { V a r } ( \hat { \tau } _ { \mathrm { E I E } } ^ { G } ) \leq \frac { C _ { p } M ^ { 2 } } { n ^ { 2 } } \left\{ \sum _ { i } d _ { i } ^ { 2 } + \sum _ { i } \sum _ { j \neq i } c _ { i j } d _ { i } d _ { j } \right\} .
$$

Proof The graph-aware EIE estimator is

$$
\hat { \tau } _ { \mathrm { E I E } } ^ { G } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Y _ { i } ^ { o b s } \sum _ { j \in N _ { i } } \left\{ \frac { Z _ { j } } { p _ { j } } - \frac { 1 - Z _ { j } } { 1 - p _ { j } } \right\} .
$$

Its design unbiasedness follows by applying the ECE importance-weighting argument to each local neighbor contrast and summing over $j \in N _ { i } ;$ equivalently, it follows from the policy-derivative calculation in Proposition 2. To obtain the variance bound, write

$$
h _ { i } ^ { \mathrm { E I E } , G } ( \mathbf { Z } ) = \sum _ { j \in N _ { i } } \left\{ \frac { Z _ { j } } { p _ { j } } - \frac { 1 - Z _ { j } } { 1 - p _ { j } } \right\} .
$$

If the treatment probabilities are bounded away from zero and one, then there is a constant $C _ { p }$ such that

$$
| h _ { i } ^ { \mathrm { E I E } , G } ( \mathbf { Z } ) | \leq C _ { p } d _ { i } .
$$

Moreover $h _ { i } ^ { \operatorname { E I E } , G } ( \mathbf { Z } )$ depends only on $\mathbf { Z } _ { N _ { i } }$ , and hence on $\mathbf { Z } _ { \mathcal { C } _ { i } }$ . Proposition 29, with $H _ { i } =$ $C _ { p } d _ { i }$ , gives

$$
\mathrm { V a r } ( \hat { \tau } _ { \mathrm { E I E } } ^ { G } ) \leq \frac { C _ { p } M ^ { 2 } } { n ^ { 2 } } \left\{ \sum _ { i } d _ { i } ^ { 2 } + \sum _ { i } \sum _ { j \neq i } c _ { i j } d _ { i } d _ { j } \right\} ,
$$

Table 3: Generic comparison of the ACE inverse-propensity scale and the EDE dependence scale under Bernoulli assignment. The ACE column corresponds to a direct-efect target combination such as $( 1 , 0 _ { N _ { i } } )$ . The EDE column gives the order of $c _ { \mathrm { a v g } }$
<table><tr><td>Graph sequence</td><td>ACE scale  $\mathcal { A } _ { n }$ </td><td>EDE dependence  $c _ { \mathrm { a v g } }$ </td></tr><tr><td>Bounded degree</td><td> $\overline { { O ( n ^ { - 1 } ) } }$ </td><td> $\overline { { O ( 1 ) } }$ </td></tr><tr><td>Second-neighborhood regular, degree d</td><td> $e ^ { \gamma d } / n$ </td><td> $d ^ { 2 }$ </td></tr><tr><td>Sparse Erdős-Renyi, mean de- approximately gree λ</td><td> $e ^ { \gamma \lambda } / n$ </td><td>approximately  $\lambda + \lambda ^ { 2 }$ </td></tr><tr><td>Dense block model, block size s approximately Star graph</td><td> $e ^ { \gamma s } / n$  dominated by hub term n</td><td>approximately s</td></tr><tr><td></td><td> $e ^ { \gamma n } / n ^ { 2 }$ </td><td></td></tr><tr><td>Complete graph</td><td> $e ^ { \gamma n } / n$ </td><td>n</td></tr></table>

after absorbing constants depending only on the treatment-probability bounds into $C _ { p } .$ . If $d _ { i } \leq d _ { \operatorname* { m a x } }$ , then

$$
\sum _ { i } d _ { i } ^ { 2 } + \sum _ { i } \sum _ { j \neq i } c _ { i j } d _ { i } d _ { j } \leq d _ { \operatorname* { m a x } } ^ { 2 } \left\{ n + \sum _ { i } \sum _ { j \neq i } c _ { i j } \right\} = n d _ { \operatorname* { m a x } } ^ { 2 } ( 1 + c _ { \operatorname { a v g } } ) ,
$$

which gives the final display.

## D.11 Heterogeneous ACE/ECE comparison under classical designs

The DTE/EDE comparison of Section 4.3 extends to arbitrary heterogeneous degree sequences, where it is governed by two graph functionals. For a fixed-exposure ACE, the relevant scale is the inverse propensity-score moment

$$
\mathcal { A } _ { n } ( d ^ { + } ) = \frac { 1 } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } \pi _ { i } ( d _ { i } ^ { + } ) ^ { - 1 } ;
$$

for the direct-efect entry $( 1 , 0 _ { N _ { i } } )$ under Bernoulli assignment, $\pi _ { i } ( 1 , 0 _ { N _ { i } } ) = p ( 1 - p ) ^ { d _ { i } }$ , so $\begin{array} { r } { \mathcal { A } _ { n } ( 1 , 0 ) \asymp n ^ { - 2 } \sum _ { i } \exp ( \gamma d _ { i } ) } \end{array}$ is an exponential moment of the degree sequence. For the EDE under the same design, the relevant scale is the average dependence degree, $B _ { n } = c _ { \mathrm { a v g } } / n$

The comparison is design-specific. It applies to classical designs such as Bernoulli assignment, and complete randomization with a fixed treatment fraction has the same exponential propensity-score behavior for local fixed-exposure combinations, up to finite-population corrections. It does not rule out designs that are constructed to increase the propensity scores of the ACE target combinations, such as the independent-set designs.

The table is meant to show why a single average degree condition is not adequate for heterogeneous graphs. By Jensen’s inequality,

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \exp ( \gamma d _ { i } ) \geq \exp ( \gamma \bar { d } ) ,
$$

so large average degree is enough to make the ACE inverse-propensity scale large. However, small average degree is not suficient to make ACE estimation easy, because hubs can dominate the exponential moment. For EDE estimation, average degree is also not the right quantity: a star graph has $\bar { d } = O ( 1 )$ but $c _ { \mathrm { a v g } } \asymp n$

Corollary 30 (Heterogeneous ACE/ECE separation under Bernoulli assignment) Consider a sequence of undirected interference graphs with degrees $d _ { i } = | N _ { i } |$ , and suppose treatment is assigned independently with a common probability p bounded away from zero and one. For the direct-efect ACE entry $\left( 1 , 0 _ { N _ { i } } \right)$ , there are constants $0 < \gamma _ { 1 } \leq \gamma _ { 2 } < \infty$ depending only on p, such that

$$
\frac { 1 } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } \exp ( \gamma _ { 1 } d _ { i } ) \lesssim \frac { 1 } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } \pi _ { i } ( 1 , 0 _ { N _ { i } } ) ^ { - 1 } \lesssim \frac { 1 } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } \exp ( \gamma _ { 2 } d _ { i } ) .
$$

The canonical ACE Horvitz–Thompson variance is bounded by

$$
\mathrm { V a r } \{ \hat { \tau } _ { \mathrm { A C E } } ^ { H T } \} \leq \frac { C _ { p } M ^ { 2 } } { n ^ { 2 } } \left\{ \sum _ { i = 1 } ^ { n } \exp ( \gamma _ { 2 } d _ { i } ) + \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } c _ { i j } \exp \{ \gamma _ { 2 } ( d _ { i } + d _ { j } ) \} \right\} ,
$$

where $c _ { i j }$ is the interference-overlap graph for the $A C E$ summands. By contrast, the canonical EDE Horvitz–Thompson estimator satisfies

$$
\mathrm { V a r } \{ \hat { \tau } _ { \mathrm { E D E } } ^ { H T } \} \leq C _ { p } M ^ { 2 } \frac { 1 + c _ { \mathrm { a v g } } } { n } .
$$

Thus, under the same Bernoulli design and for arbitrary heterogeneous graph sequences, the ACE benchmark involves exponential moments of the degree sequence and their dependenceweighted analogues, whereas the EDE benchmark involves only the average degree of the dependence graph.

The regular-graph comparison of Section 4.3 is recovered as a special case. More generally, Table 3 should be read through the two graph functionals appearing in the corollary: exponential degree moments for ACE target combinations and average dependence degree for EDE importance-weighted estimators.

## D.12 Proof of Corollary 30

Restatement. Under Bernoulli assignment with common treatment probability bounded away from zero and one, the direct-efect ACE benchmark involves exponential moments of the degree sequence and their dependence-weighted analogues, whereas the EDE benchmark involves only the average dependence degree.

Proof For the direct-efect entry $\left( 1 , 0 _ { N _ { i } } \right)$ under Bernoulli assignment,

$$
\pi _ { i } ( 1 , 0 _ { N _ { i } } ) = p ( 1 - p ) ^ { d _ { i } } .
$$

Since $p$ and $1 - p$ are bounded away from zero and one, there are constants $0 < \gamma _ { 1 } \leq \gamma _ { 2 } < \infty$ depending only on $p ,$ such that

$$
\exp ( \gamma _ { 1 } d _ { i } ) \leq \pi _ { i } ( 1 , 0 _ { N _ { i } } ) ^ { - 1 } \leq \exp ( \gamma _ { 2 } d _ { i } )
$$

after absorbing fixed multiplicative constants into the comparison constants. Summing over i gives the inverse-propensity display.

For the ACE HT upper bound, Proposition 10 gives

$$
\mathrm { V a r } \{ \hat { \tau } _ { \mathrm { A C E } } ^ { H T } \} \leq \frac { M ^ { 2 } } { n ^ { 2 } } \left\{ \sum _ { i } H _ { i } + \sum _ { i } \sum _ { j \neq i } c _ { i j } H _ { i } H _ { j } \right\} .
$$

Here the residual design-dependence term is zero because treatment is assigned independently. For the direct-efect contrast with no treated neighbors, both arm propensities are bounded below by constants times $\exp ( - \gamma _ { 2 } d _ { i } )$ , so $H _ { i } \leq C _ { p } \exp ( \gamma _ { 2 } d _ { i } )$ . Substitution gives the stated dependence-weighted exponential bound. The EDE bound is exactly Proposition 13.

## D.13 Auxiliary calculation: DTE and TTE table rates

Proposition 31 (DTE and TTE propensities under graph-unaware designs) Let $A _ { i }$ denote the assignment coordinates that define the target treatment–exposure combination for unit i. For arbitrary interference, $A _ { i } = \{ 1 , \ldots , n \}$ , while under neighborhood interference $A _ { i } = \mathcal { C } _ { i } = \{ i \} \cup N _ { i }$ . Write $a _ { i } = | A _ { i } |$

(i) Suppose the design assigns treatments independently, with $\epsilon \leq P ( Z _ { k } = 1 ) \leq 1 - \epsilon f o \tau$ all k. Then every fixed assignment pattern on $A _ { i }$ has probability $\exp \{ - \Theta ( a _ { i } ) \}$ , with constants depending only on ϵ. Consequently, the DTE no-exposure conditions and the TTE full-exposure/no-exposure conditions have inverse propensities $\exp \{ \Theta ( n ) \}$ under arbitrary interference and $\exp \{ \Theta ( d _ { i } ) \}$ under neighborhood interference.

(ii) Suppose the design is exchangeable, so that conditional on $\begin{array} { r } { T _ { n } = \sum _ { k } Z _ { k } = m } \end{array}$ all assignments with m treated units are equally likely. Assume that, for some $\epsilon > 0$ $T _ { n } / n \in [ \epsilon , 1 - \epsilon ]$ almost surely and that $a _ { i } \leq \epsilon n / 2$ . Then every fixed assignment pattern on $A _ { i }$ has probability exp $\{ - \Theta ( a _ { i } ) \}$ , with constants depending only on ϵ. Thus the same $\exp \{ \Theta ( d _ { i } ) \}$ inverse-propensity order holds for neighborhood DTE and TTE conditions, including the no-exposure and full-exposure count/fraction exposure representations. Fixed-size complete randomization is a special case when the treatment fraction is bounded away from zero and one.

In both cases, the inverse-propensity lower bound in Proposition 11 applies to any designunbiased non-parametric estimator of the corresponding DTE or TTE. The exchangeable case covers local treatment–exposure combinations. It does not cover arbitrary-interference TTE/DTE targets under fixed nontrivial treatment fractions, because such designs put zero probability on the global all-treated or all-control assignments.

Proof [Proof of Proposition 31] For the independent-design claim, fix a set A with $| A | = a$ and a pattern $z _ { A } \in \{ 0 , 1 \} ^ { a } . \mathrm { ~ I f ~ } \epsilon \leq P ( Z _ { k } = 1 ) \leq 1 - \epsilon$ for all $k ,$ then

$$
\epsilon ^ { a } \leq P ( \mathbf { Z } _ { A } = z _ { A } ) \leq ( 1 - \epsilon ) ^ { a } .
$$

Thus every fixed pattern on A has probability $\exp \{ - \Theta ( a ) \}$ . For arbitrary interference, the DTE and TTE target conditions impose fixed patterns on all n assignments. Under neighborhood interference, the DTE no-exposure conditions $( 1 , 0 _ { N _ { i } } )$ and $( 0 , 0 _ { N _ { i } } )$ , and the TTE full-exposure/no-exposure conditions $( 1 , 1 _ { N _ { i } } )$ and $( 0 , 0 _ { N _ { i } } )$ , impose fixed patterns on $\mathcal { C } _ { i } = \{ i \} \cup N _ { i }$ , whose size is $d _ { i } + 1$ . The inverse propensities therefore have the stated exponential orders.

For the exchangeable-design claim, condition on $T _ { n } = m$ . If a fixed pattern on A contains t treated units, then

$$
P ( \mathbf { Z } _ { A } = z _ { A } \mid T _ { n } = m ) = \frac { ( m ) _ { t } ( n - m ) _ { a - t } } { ( n ) _ { a } } ,
$$

where $( x ) _ { k } = x ( x - 1 ) \cdot \cdot \cdot ( x - k + 1 )$ . On the event $m / n \in [ \epsilon , 1 - \epsilon ]$ and for $a \leq \epsilon n / 2$ , each factor in this product is bounded below by $\epsilon / 2$ and above by $1 - \epsilon ,$ after adjusting constants depending only on ϵ. Hence

$$
\begin{array} { r } { ( \epsilon / 2 ) ^ { a } \leq P ( \mathbf { Z } _ { A } = z _ { A } \mid T _ { n } = m ) \leq ( 1 - \epsilon ) ^ { a } . } \end{array}
$$

The same bounds hold unconditionally because the support of $T _ { n }$ is contained in $[ \epsilon n , ( 1 -$ $\epsilon ) n ]$ . Applying this with $A = { \mathcal { C } } _ { i }$ gives the neighborhood DTE and TTE orders. Complete randomization with $m / n \in [ \epsilon , 1 - \epsilon ]$ is the special case in which $T _ { n } = m$ is fixed. The last statement follows immediately from Proposition 11.

Restatement. The orders in Tables 1 and 2 follow from the propensities of the target treatment–exposure combinations under Bernoulli assignment. Complete randomization with a fixed treatment fraction has the same exponential order for the no-exposure and full-exposure fixed entries, up to finite-population corrections.

Proof Assume Bernoulli assignment with common treatment probability $p \in ( 0 , 1 )$ bounded away from zero and one, and write $d _ { i } = | N _ { i } |$ . For a fixed target contrast, the HT weight scale $H _ { i }$ is the larger reciprocal propensity among the two target arms, up to constants. The uniform table scale is $H = \operatorname* { m a x } _ { i } H _ { i }$ . The lower-bound column has the same propensity order by Proposition 11, and the HT variance-scale column follows by substituting H and the relevant dependence scale into $\mathcal { V } _ { n } ( H , c ) = H / n + H ^ { 2 } c / n$

For the direct-efect rows, arbitrary interference with no treated units except i has target propensities

$$
p ( 1 - p ) ^ { n - 1 } \qquad \mathrm { a n d } \qquad ( 1 - p ) ^ { n } ,
$$

so $H = e ^ { \Theta ( n ) }$ . Under neighborhood interference, the target conditions $( 1 , 0 _ { N _ { i } } )$ and $( 0 , 0 _ { N _ { i } } )$ have propensities

$$
p ( 1 - p ) ^ { d _ { i } } \qquad \mathrm { a n d } \qquad ( 1 - p ) ^ { d _ { i } + 1 } ,
$$

so $H = e ^ { \Theta ( d _ { \mathrm { m a x } } ) }$ . The count and fractional no-exposure rows impose $S _ { i } = 0$ , equivalently $F _ { i } = 0 .$ and therefore have the same order.

For the binary any-treated direct-efect row, the DTE no-exposure condition is $U _ { i } = 0$ 2 equivalently $S _ { i } = 0$ , so

$$
P \{ Z _ { i } = 1 , U _ { i } = 0 \} = p ( 1 - p ) ^ { d _ { i } } , \qquad P \{ Z _ { i } = 0 , U _ { i } = 0 \} = ( 1 - p ) ^ { d _ { i } + 1 } .
$$

Thus $H = e ^ { \Theta ( d _ { \operatorname* { m a x } } ) }$ . For a typical count exposure $k _ { i } \asymp p d _ { i }$ 2

$$
P \{ S _ { i } = k _ { i } \} = { \binom { d _ { i } } { k _ { i } } } p ^ { k _ { i } } ( 1 - p ) ^ { d _ { i } - k _ { i } } = \Theta ( d _ { i } ^ { - 1 / 2 } )
$$

uniformly for $k _ { i }$ in the central binomial range, by Stirling’s approximation. Multiplying by the constant arm probabilities $p$ or $1 - p$ gives $H = \Theta ( \sqrt { d _ { \operatorname* { m a x } } } )$

For the total-efect rows, arbitrary interference compares all treated with all control, giving propensities $p ^ { n }$ and $( 1 - p ) ^ { n }$ , hence $H = e ^ { \Theta ( n ) }$ . Under neighborhood interference, the target conditions $( 1 , 1 _ { N _ { i } } )$ and $( 0 , 0 _ { N _ { i } } )$ have propensities

$$
p ^ { d _ { i } + 1 } \qquad \mathrm { a n d } \qquad ( 1 - p ) ^ { d _ { i } + 1 } ,
$$

so $H = e ^ { \Theta ( d _ { \mathrm { m a x } } ) }$ . The count and fractional full-versus-none row imposes $S _ { i } = d _ { i }$ on the treated arm and $S _ { i } ~ = ~ 0$ on the control arm, equivalently $F _ { i } ~ = ~ 1$ versus $F _ { i } ~ = ~ 0$ , and has the same order. For the binary any-treated version, the treated arm has probability $p \{ 1 - ( 1 - p ) ^ { d _ { i } } \}$ , but the control arm with no exposure has probability $( 1 - p ) ^ { d _ { i } + 1 }$ . The no-exposure arm therefore still gives $H = e ^ { \Theta ( d _ { \operatorname* { m a x } } ) }$

Finally, under complete randomization with m treated units, the direct no-exposure target propensities are

$$
{ \frac { \binom { n - d _ { i } - 1 } { m - 1 } } { \binom { n } { m } } } , \qquad { \frac { \binom { n - d _ { i } - 1 } { m } } { \binom { n } { m } } } ,
$$

whenever the binomial coeficients are well defined; uniformly over units this requires $1 \leq$ m $\leq n - d _ { \mathrm { m a x } } - 1$ for both direct-efect arms. If $m / n \to p \in ( 0 , 1 )$ , these are the finitepopulation analogues of $p ( 1 - p ) ^ { d _ { i } }$ and $( 1 - p ) ^ { d _ { i } + 1 }$ ; for example, when $d _ { i } = o ( n )$ they are equal to the Bernoulli propensities times exp $\{ O ( d _ { i } ^ { 2 } / n ) \}$ . For the full-exposure treated arm in the TTE row, the corresponding propensity is

$$
{ \frac { { \binom { n - d _ { i } - 1 } { m - d _ { i } - 1 } } } { { \binom { n } { m } } } } ,
$$

which is the finite-population analogue of $p ^ { d _ { i } + 1 }$ . Uniform positivity for the full-versus-none TTE rows requires $d _ { \operatorname* { m a x } } + 1 \leq m \leq n - d _ { \operatorname* { m a x } } - 1$ . Thus complete randomization has the same exponential dependence on $d _ { i }$ for local fixed no-exposure and full-exposure entries, up to the usual finite-population correction, but the variance bound also includes the residual design-dependence term $r _ { \mathrm { a v g } }$ . The arbitrary-interference rows are not covered by fixed nontrivial complete randomization, since such a design does not assign positive probability to both all-treated and all-control global regimes. 7

## D.14 Auxiliary calculation: exposure compression and propensity scores Restatement. For an exposure mapping $e = f _ { i } ( u )$ 2

$$
\pi _ { i } ^ { f } ( z , e ) = \sum _ { u \in f _ { i } ^ { - 1 } ( e ) } \pi _ { i } ^ { \mathrm { i d } } ( z , u ) ,
$$

and hence

$$
\pi _ { i } ^ { f } ( z , e ) \geq \pi _ { i } ^ { \mathrm { i d } } ( z , u ) \qquad \mathrm { f o r ~ e v e r y ~ } u \in f _ { i } ^ { - 1 } ( e ) .
$$

Under balanced Bernoulli assignment,

$$
\pi _ { i } ^ { f } ( z , e ) = 2 ^ { - ( | N _ { i } | + 1 ) } | f _ { i } ^ { - 1 } ( e ) | .
$$

Proof The event $\{ Z _ { i } = z , f _ { i } ( Z _ { N _ { i } } ) = e \}$ is the disjoint union of the events $\{ Z _ { i } = z , Z _ { N _ { i } } = u \}$ over $u \in f _ { i } ^ { - 1 } ( e )$ . Summing their probabilities gives the first equality and the inequality. Under balanced Bernoulli assignment, each full local configuration $( z , u )$ has probability $2 ^ { - ( | N _ { i } | + 1 ) }$

## D.15 Auxiliary calculation: outcome-model span condition

Restatement. Suppose an outcome model restricts $\theta _ { i } = X _ { i } \gamma _ { i }$ . Let $O _ { i }$ be the set of local states with positive propensity score under the design. A linear functional $\ell _ { i } ^ { \top } \theta _ { i }$ is determined by the potential outcomes in the observable states $O _ { i }$ for every $\gamma _ { i }$ if and only if

$$
\ell _ { i } ^ { \top } X _ { i } \in \mathrm { r o w s p a n } ( X _ { i , O _ { i } } ) .
$$

Proof The observable states determine the vector $X _ { i , O _ { i } } \gamma _ { i }$ . The functional $\ell _ { i } ^ { \top } \theta _ { i } = \ell _ { i } ^ { \top } X _ { i } \gamma _ { i }$ is determined by these observable values for every $\gamma _ { i }$ if and only if it is a linear function of $X _ { i , O _ { i } } \gamma _ { i }$ . This is equivalent to the row vector $\ell _ { i } ^ { \top } X _ { i }$ lying in the row span of $X _ { i , O _ { i } }$

## Appendix E. Proofs for Section 5

## E.1 Proof of Proposition 15

Proof $\mathrm { B y }$ stability, $\mathbb { 1 } \{ ( Z _ { i } , E _ { i } ) = d _ { i } ^ { + } \} Y _ { i } ^ { \mathrm { o b s } } = \mathbb { 1 } \{ ( Z _ { i } , E _ { i } ) = d _ { i } ^ { + } \} Y _ { i } ( d _ { i } ^ { + } )$ , and since the potential outcomes are fixed under the design,

$$
\mathbb { E } \bigg [ \frac { \mathbb { 1 } \{ ( Z _ { i } , E _ { i } ) = d _ { i } ^ { + } \} Y _ { i } ^ { \mathrm { o b s } } } { \pi _ { i } ( d _ { i } ^ { + } ) } \bigg ] = \frac { \pi _ { i } ( d _ { i } ^ { + } ) } { \pi _ { i } ( d _ { i } ^ { + } ) } Y _ { i } ( d _ { i } ^ { + } ) = Y _ { i } ( d _ { i } ^ { + } ) .
$$

The same calculation gives $Y _ { i } ( d _ { i } ^ { - } )$ for the second arm. Taking the diference and averaging over i yields $\mathbb { E } [ \hat { \tau } _ { \mathrm { H T } } ^ { Z E } ] = \tau$

## E.2 Proof of Lemma 16

Proof The event $\{ i \in \mathcal { I } \}$ equals $\{ U _ { i } < U _ { k } \ \forall k \in N _ { i } \}$ , that is, $U _ { i }$ is the strict minimum among the $d _ { i } + 1$ independent uniform draws in $\{ i \} \cup N _ { i }$ . By exchangeability each draw is equally likely to be the minimum, so $\operatorname* { P r } ( i \in \mathcal { I } ) = 1 / ( d _ { i } + 1 )$ . The coin $B _ { i }$ is drawn independently of the draws $U$ , so

$$
{ \boldsymbol { \rho } } _ { i } ^ { ( 1 ) } = \operatorname* { P r } ( i \in { \mathcal { Z } } ) \operatorname* { P r } ( B _ { i } = 1 ) = { \frac { q } { d _ { i } + 1 } } , \qquad { \boldsymbol { \rho } } _ { i } ^ { ( 0 ) } = \operatorname* { P r } ( i \in { \mathcal { Z } } ) \operatorname* { P r } ( B _ { i } = 0 ) = { \frac { 1 - q } { d _ { i } + 1 } } .
$$

On $F _ { i } ^ { ( 1 ) }$ , step (3) sets $Z _ { i } = 1$ and $Z _ { k } = 0$ for all $k \in N _ { i }$ , so $\left( Z _ { i } , Z _ { N _ { i } } \right) = \left( 1 , \mathbf { 0 } \right)$ ; on $F _ { i } ^ { ( 0 ) }$ it sets $Z _ { i } = 0$ and $Z _ { k } = 0$ for all $k \in N _ { i }$ , giving $\left( Z _ { i } , Z _ { N _ { i } } \right) = \left( 0 , \mathbf { 0 } \right)$ ■

## E.3 Proof of Proposition 18

Proof Write $\begin{array} { r } { \hat { \tau } _ { \mathrm { I S } } ^ { D } = n ^ { - 1 } \sum _ { i } W _ { i } } \end{array}$ with $W _ { i } = 2 ( d _ { i } + 1 ) \mathbb { 1 } \{ i \in \mathcal { T } \} \{ Y _ { i } ( 1 , \mathbf { 0 } _ { N _ { i } } ) B _ { i } - Y _ { i } ( 0 , \mathbf { 0 } _ { N _ { i } } ) ( 1 -$ $B _ { i } ) \}$ . Conditional on $\mathcal { T } ,$ the coins $B _ { i }$ are independent, and $\mathbb { E } [ W _ { i } \mid \mathcal { T } ] = \mathbb { 1 } \{ i \in \mathcal { T } \} \delta _ { i } / p _ { i } ;$ taking expectations gives $\mathbb { E } [ W _ { i } ] = \delta _ { i }$ . For $i \neq j ,$ , independence of $B _ { i } , B _ { j }$ given I yields

$$
\mathbb { E } [ W _ { i } W _ { j } ] = \frac { \mathrm { P r } ( i , j \in \mathcal { Z } ) } { p _ { i } p _ { j } } \delta _ { i } \delta _ { j } , \qquad \mathrm { C o v } ( W _ { i } , W _ { j } ) = \delta _ { i } \delta _ { j } \left( \frac { \mathrm { P r } ( i , j \in \mathcal { Z } ) } { p _ { i } p _ { j } } - 1 \right) .
$$

If the closed neighborhoods of i and j are disjoint, the events $\{ i \in \mathcal { T } \}$ and $\{ j \in \mathcal { I } \}$ depend on disjoint sets of draws, so $\operatorname* { P r } ( i , j \in { \mathcal { I } } ) = p _ { i } p _ { j }$ and the covariance vanishes; this occurs whenever $\mathrm { d i s t } _ { G } ( i , j ) > 2$ . Summing the diagonal variances and these covariances gives the stated exact identity.

For the pairwise ratio, if $i \sim j$ then i and j cannot both be closed-neighborhood minima, so $\Pr ( i , j \in { \mathcal { I } } ) = 0$ . If $i \not \sim j$ share $s _ { i j } = | N _ { i } \cap N _ { j } |$ | neighbors, set $a = d _ { i } + 1 , b = d _ { i } + 1$ $s = s _ { i j }$ . Conditioning on $U _ { i } = x$ and $U _ { j } = y$ , both are minima if the $a - 1 - s$ private neighbors of i exceed $x ,$ the $b - 1 - s$ private neighbors of $j$ exceed $y ,$ and the s shared neighbors exceed max $( x , y )$

$$
\operatorname* { P r } ( i , j \in \mathcal { T } ) = \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } ( 1 - x ) ^ { a - 1 - s } ( 1 - y ) ^ { b - 1 - s } ( 1 - \operatorname* { m a x } \{ x , y \} ) ^ { s } d x d y = \frac { a + b } { a b ( a + b - s ) } .
$$

Since $p _ { i } p _ { j } = 1 / ( a b )$ , the ratio is $( a + b ) / ( a + b - s ) = ( d _ { i } + d _ { j } + 2 ) / ( d _ { i } + d _ { j } + 2 - s _ { i j } )$ , and the excess over 1 is $s _ { i j } / ( d _ { i } + d _ { j } + 2 - s _ { i j } )$ For the bound, $\begin{array} { r } { \mathbb { E } [ W _ { i } ^ { 2 } ] = 2 ( d _ { i } + 1 ) \{ Y _ { i } ( 1 , \mathbf { 0 } _ { N _ { i } } ) ^ { 2 } + Y _ { i } ( 0 , \mathbf { 0 } _ { N _ { i } } ) ^ { 2 } \} \le 4 M ^ { 2 } ( d _ { i } + 1 ) } \end{array}$ , so Var $( W _ { i } ) \leq$ $4 M ^ { 2 } ( d _ { i } + 1 )$ Taking absolute values in the covariance sum and using $| \delta _ { i } \delta _ { j } | ~ \leq ~ 4 M ^ { 2 }$ with the pairwise ratios (contribution 1 for each of the $2 | E |$ ordered adjacent pairs, and $s _ { i j } / ( d _ { i } + d _ { j } + 2 - s _ { i j } )$ for each non-adjacent overlapping pair) gives the displayed inequality.

## E.4 Proof of Corollary 19

Proof On a d-regular graph, $\textstyle \sum _ { i } ( d _ { i } + 1 ) = n ( d + 1 )$ and $2 | E | = n d .$ . Each adjacent pair contributes at most 1, and each non-adjacent overlapping pair contributes $s _ { i j } / ( 2 d + 2 - s _ { i j } ) \leq 1$ since $s _ { i j } \leq d - 1$ , while each ordered wedge $\begin{array} { r } { \sum _ { k } d _ { k } ( d _ { k } - 1 ) = n d ( d - 1 ) } \end{array}$ counts at most one non-adjacent overlapping pair. The number of non-adjacent overlapping pairs is therefore $O ( n d ^ { 2 } )$ , and the weighted sum of the pairwise ratios is $O ( n d ^ { 2 } )$ ; this yields the cruder rate $O ( M ^ { 2 } d ^ { 2 } / n )$ after dividing by $n ^ { 2 }$ . If the graph is locally tree-like, $s _ { i j } = O ( 1 )$ , the pairwise ratio satisfies $s _ { i j } / ( 2 d + 2 - s _ { i j } ) = O ( 1 / d )$ (since the denominator is at least $d + 3 )$ and the weighted sum over the $O ( n d ^ { 2 } )$ overlapping pairs is $O ( n d )$ ; dividing by $n ^ { 2 }$ gives $O ( M ^ { 2 } d / n )$ . For a general graph, drop the denominators using $d _ { i } + d _ { j } + 2 - s _ { i j } \geq 1$ and count ordered wedges: $\begin{array} { r } { \sum _ { i \neq j \colon i \ne j } s _ { i j } \le \sum _ { k } d _ { k } ( d _ { k } - 1 ) } \end{array}$ . Adding $\textstyle 2 | E | = \sum _ { i } d _ { i }$ and the diagonal $\begin{array} { r } { \sum _ { i } ( d _ { i } + 1 ) = n + \sum _ { i } \ddot { d } _ { i } } \end{array}$ bounds the bracket by $\textstyle n + \sum _ { i } d _ { i } + \sum _ { i } d _ { i } ^ { 2 }$ , giving the stated $O ( M ^ { 2 } ( 1 + \bar { d } + \overline { { { d ^ { 2 } } } } ) / n )$ ■

## E.5 Proof of Lemma 20

Proof The event $\{ i \in \mathcal { I } _ { 2 } \}$ is $\{ U _ { i } < U _ { k } \ \forall k \in B _ { i } ^ { ( 2 ) } \setminus \{ i \} \}$ , i.e. $U _ { i }$ is the strict minimum among the $b _ { i }$ independent draws in $B _ { i } ^ { ( 2 ) }$ ; by exchangeability this has probability $1 / b _ { i }$ . On $\{ i \in \mathcal { T } _ { 2 } , R _ { i } = 1 \}$ , step (3) sets the entire closed neighborhood to 1, giving $\left( Z _ { i } , Z _ { N _ { i } } \right) = \left( 1 , { \bf 1 } \right)$ , and $R _ { i }$ is independent of the draws, so $\rho _ { i } ^ { ( 1 ) } = 1 / ( 2 b _ { i } )$ ; the control side is symmetric. The lower bound $d _ { i } + 1 \le b _ { i }$ holds because the closed neighborhood is contained in the radiustwo ball. The upper bound follows by counting vertices reached from i in zero, one, or two steps: at most $\begin{array} { r } { 1 + d _ { i } + \sum _ { j \in N _ { i } } ( d _ { j } - 1 ) \le 1 + d _ { i } d _ { \operatorname* { m a x } } } \end{array}$ ■

## E.6 Proof of Proposition 21

Proof Unbiasedness follows directly from the focal reveal events. On $\{ i \in \mathcal { T } _ { 2 } , R _ { i } = 1 \}$ the observed outcome equals $Y _ { i } ( 1 , e _ { i } ^ { \mathbf { 1 } } )$ , and on $\{ i \in \mathcal { T } _ { 2 } , R _ { i } = 0 \}$ it equals $Y _ { i } ( 0 , 0 )$ ; each event has probability $1 / ( 2 b _ { i } )$ . Writing $\begin{array} { r } { \hat { \tau } _ { \mathrm { I S } } ^ { T } = n ^ { - 1 } \sum _ { i } V _ { i } } \end{array}$ , the argument of Proposition 18 applies with $p _ { i }$ replaced by $q _ { i } = 1 / b _ { i }$ and $\delta _ { i }$ by $\Delta _ { i } { : }$ conditioning on $\mathcal { T } _ { 2 }$ gives $\mathbb { E } [ V _ { i } \mid \mathcal { T } _ { 2 } ] = \mathbb { 1 } \{ i \in \mathcal { I } _ { 2 } \} \Delta _ { i } / q _ { i }$ and, for $i \neq j$ $\mathrm { C o v } ( V _ { i } , V _ { j } ) = \Delta _ { i } \Delta _ { j } ( \mathrm { P r } ( i , j \in \mathbb { Z } _ { 2 } ) / ( q _ { i } q _ { j } ) - 1 )$ . If $B _ { i } ^ { ( 2 ) } \cap B _ { i } ^ { ( 2 ) } = \varnothing$ the covariance vanishes. If dis $\operatorname { t } _ { G } ( i , j ) \leq 2$ , then i and j cannot both be strict minima in their radius-two balls, so $\mathrm { P r } ( i , j \in \mathcal { I } _ { 2 } ) = 0$ . If $\mathrm { d i s t } _ { G } ( i , j ) \in \{ 3 , 4 \}$ , the same integral as before with ball sizes $( b _ { i } , b _ { j } )$ and overlap $s = s _ { i j } ^ { ( 2 ) }$ gives $\operatorname* { P r } ( i , j \in \mathbb { Z } _ { 2 } ) / ( q _ { i } q _ { j } ) = ( b _ { i } + b _ { j } ) / ( b _ { i } + b _ { j } - s )$ Finally $\mathbb { E } [ V _ { i } ^ { 2 } ] = 2 b _ { i } \{ Y _ { i } ( 1 , e _ { i } ^ { 1 } ) ^ { 2 } + Y _ { i } ( 0 , 0 ) ^ { 2 } \} \le 4 M ^ { 2 } b _ { i }$ , and taking absolute values in the covariance sum gives the bound: distance-one and distance-two pairs contribute at most one each, and distance-three and distance-four overlapping pairs contribute $s _ { i j } ^ { ( 2 ) } / ( b _ { i } + b _ { j } - s _ { i j } ^ { ( 2 ) } )$

## E.7 Proof of Corollary 22

Proof On a d-regular locally tree-like graph, $b _ { i } = 1 + d + d ( d - 1 ) = d ^ { 2 } + 1$ , so the diagonal term is $\textstyle 4 M ^ { 2 } n ^ { - 2 } \sum _ { i } b _ { i } = O ( M ^ { 2 } d ^ { 2 } / n )$ . The number of ordered pairs at distance at most two is $O ( n d ^ { 2 } )$ , so the distance-two exclusion term is also $O ( M ^ { 2 } d ^ { 2 } / n )$ . For the positive radiustwo overlap term, each unit has at most $d ( d - 1 ) ^ { 2 }$ distance-three partners, each with overlap $s = 2$ and denominator of order $d ^ { 2 }$ , contributing $O ( n d )$ ; each unit has $O ( d ^ { 4 } )$ distance-four partners, each with overlap $s = 1$ and denominator of order $d ^ { 2 }$ , contributing $O ( n d ^ { 2 } )$ , which is the same order as the diagonal and exclusion terms. Thus the total is $O ( M ^ { 2 } d ^ { 2 } / n )$ . For a general graph, the diagonal and distance-two exclusion terms are bounded by $O ( \sum _ { i } b _ { i } )$ , and wedge-counting in the radius-two overlap graph bounds the positive overlap sum by $\begin{array} { r } { \sum _ { k } b _ { k } ( b _ { k } - 1 ) } \end{array}$ , giving $O ( M ^ { 2 } \overline { { b ^ { 2 } } } / n )$

## E.8 Proof of Proposition 24

Proof Unbiasedness. On $F _ { i } ^ { ( 1 ) } = \{ Z _ { i } = 0 , U _ { i } = 1 \}$ the observed outcome equals $Y _ { i } ( 0 , 1 )$ , and on $F _ { i } ^ { ( 0 ) } = \{ Z _ { i } = 0 , U _ { i } = 0 \}$ it equals $Y _ { i } ( 0 , 0 )$ . Since $\rho _ { i } ^ { ( b ) } > 0$ for every unit of degree at least one,

$$
\mathbb { E } \Big [ \frac { \mathbb { 1 } \{ F _ { i } ^ { ( 1 ) } \} Y _ { i } ^ { \mathrm { o b s } } } { \rho _ { i } ^ { ( 1 ) } } \Big ] = Y _ { i } ( 0 , 1 ) , \qquad \mathbb { E } \Big [ \frac { \mathbb { 1 } \{ F _ { i } ^ { ( 0 ) } \} Y _ { i } ^ { \mathrm { o b s } } } { \rho _ { i } ^ { ( 0 ) } } \Big ] = Y _ { i } ( 0 , 0 ) ,
$$

so $\begin{array} { r } { \mathbb { E } \big [ \hat { \tau } _ { \mathrm { s e e d } } \big ] = n ^ { - 1 } \sum _ { i } \{ Y _ { i } ( 0 , 1 ) - Y _ { i } ( 0 , 0 ) \} = \tau _ { \mathrm { I E } } ( 0 ; 0 , 1 ) . } \end{array}$

Focal probabilities. Because I is an independent set, no ego has an ego neighbor. Consequently the event $\{ U _ { i } = 1 \}$ —at least one treated neighbor—forces $i \not \in \mathcal { I }$ and hence $Z _ { i } = 0$ . Writing $K _ { i } = | \mathcal { I } \cap N _ { i } |$ , the number of treated ego neighbors is a binomial thinning of $K _ { i }$ by the independent coins $B _ { j }$ , so $\operatorname* { P r } ( U _ { i } = 1 ) = \mathbb { E } [ 1 - ( 1 - q ) ^ { K _ { i } } ]$ , which gives $\rho _ { i } ^ { ( 1 ) }$ Further, $Z _ { i } = 0$ holds when either $i \not \in { \mathcal { I } }$ or $i \in \mathcal { Z }$ with $B _ { i } = 0$ , so

$$
\operatorname* { P r } ( Z _ { i } = 0 ) = 1 - q \operatorname* { P r } ( i \in \mathcal { I } ) = 1 - \frac { q } { d _ { i } + 1 } ,
$$

and $\begin{array} { r } { \rho _ { i } ^ { ( 0 ) } = \operatorname* { P r } ( Z _ { i } = 0 ) - \rho _ { i } ^ { ( 1 ) } = \mathbb { E } [ ( 1 - q ) ^ { K _ { i } } ] - \frac { q } { d _ { i } + 1 } } \end{array}$ , as claimed in (5.1).

Poisson limit on locally tree-like d-regular graphs. Suppose the induced subgraph on the radius-two ball of i is a tree, as on locally tree-like d-regular graphs. Given $U _ { i } = u$ the ego events of the d neighbors of i are determined by disjoint sets of draws and are therefore conditionally independent, each with common probability

$$
p ( u ) = \operatorname* { P r } \Big ( U _ { j } < u , ~ U _ { j } < U _ { \ell } \forall \ell \in N _ { j } \setminus \{ i \} \Big ) = \int _ { 0 } ^ { u } ( 1 - x ) ^ { d - 1 } d x = \frac { 1 - ( 1 - u ) ^ { d } } { d } .
$$

Hence $K _ { i } \mid U _ { i } = u \sim$ Binomial $( d , p ( u ) )$ and, for $s \in [ 0 , 1 ]$ ，

$$
\mathbb { E } [ s ^ { K _ { i } } ] = \int _ { 0 } ^ { 1 } \Big ( 1 - ( 1 - s ) p ( u ) \Big ) ^ { d } d u = \int _ { 0 } ^ { 1 } \Big ( 1 - \frac { 1 - s } { d } \big ( 1 - ( 1 - u ) ^ { d } \big ) \Big ) ^ { d } d u \xrightarrow { d  \infty } e ^ { - ( 1 - s ) } ,
$$

by dominated convergence, since $d p ( u ) = 1 - ( 1 - u ) ^ { d } \to 1$ for $u \in ( 0 , 1 )$ . Thus $K _ { i } \Rightarrow$ Poisson(1). Taking $s = 1 - q$ gives $\mathbb { E } [ ( 1 - q ) ^ { K _ { i } } ] \to e ^ { - q } ;$ ; hence $\rho _ { i } ^ { ( 1 ) } \to 1 - e ^ { - q }$ , and since $q / ( d + 1 )  0 , \rho _ { i } ^ { ( 0 ) }  e ^ { - q }$

Star center. For the center c of a star, the leaves are degree-one units: given $U _ { c } = u$ each leaf is an ego exactly when its single draw falls below u, so the leaf ego events are independent Bernoulli(u) and

$$
\mathbb { E } [ ( 1 - q ) ^ { K _ { c } } ] = \int _ { 0 } ^ { 1 } ( 1 - q u ) ^ { d } d u = \frac { 1 - ( 1 - q ) ^ { d + 1 } } { q ( d + 1 ) } = O \Big ( \frac { 1 } { q d } \Big ) ,
$$

so $\rho _ { c } ^ { ( 0 ) }$ has the same polynomial order and $\rho _ { c } ^ { ( 1 ) }  1$ . In general $K _ { i } \leq d _ { i }$ , so $\mathbb { E } [ ( 1 - q ) ^ { K _ { i } } ] \ge$ $( 1 - q ) ^ { d _ { i } }$ and $\rho _ { i } ^ { ( 0 ) } \geq ( 1 - q ) ^ { d _ { i } } - q / ( d _ { i } + 1 )$ : when the ego-neighborhood count $K _ { i }$ concentrates on values of order $d _ { i }$ the unexposed arm is no larger than the classical $( 1 - q ) ^ { d _ { i } }$ scale. The constant-order guarantee is therefore tied to units whose ego-neighborhood is $O ( 1 )$ in expectation.

Variance. Write $\hat { \tau } _ { \mathrm { s e e d } } = n ^ { - 1 } \sum _ { i } W _ { i }$ . Since $| Y _ { i } | \le M$

$$
\mathbb { E } [ W _ { i } ^ { 2 } ] = \mathbb { E } \Big [ \frac { \mathbb { 1 } \{ F _ { i } ^ { ( 1 ) } \} ( Y _ { i } ^ { \mathrm { o b s } } ) ^ { 2 } } { ( \rho _ { i } ^ { ( 1 ) } ) ^ { 2 } } + \frac { \mathbb { 1 } \{ F _ { i } ^ { ( 0 ) } \} ( Y _ { i } ^ { \mathrm { o b s } } ) ^ { 2 } } { ( \rho _ { i } ^ { ( 0 ) } ) ^ { 2 } } \Big ] \leq M ^ { 2 } \Big ( \frac { 1 } { \rho _ { i } ^ { ( 1 ) } } + \frac { 1 } { \rho _ { i } ^ { ( 0 ) } } \Big ) ,
$$

so the diagonal term is at most $M ^ { 2 } n ^ { - 2 } \sum _ { i } ( 1 / \rho _ { i } ^ { ( 1 ) } + 1 / \rho _ { i } ^ { ( 0 ) } ) = O ( M ^ { 2 } / n )$ on locally tree-like d-regular graphs. The focal event $F _ { i } ^ { ( b ) }$ is a function of the draws U in the radius-two ball of i and of the coins of the egos in $N _ { i } .$ , so for dis $_ G ( i , j ) \geq 5$ the focal events of i and $j$ are independent and $\operatorname { C o v } ( W _ { i } , W _ { j } ) = 0$ . The remaining covariance contributions come from the $O ( n d ^ { 4 } )$ pairs at distance at most four and obey the same radius-two overlap anatomy as in the proof of Proposition 18; combined with the diagonal term this is exactly the structure of the generic ACE variance bound of Proposition 10.

## References

Shishir Adhikari and Elena Zheleva. Inferring individual direct causal efects under heterogeneous peer influence. Machine Learning, 114(4):113, 2025.

Sinan Aral and Dylan Walker. Creating social contagion through viral product design: A randomized trial of peer influence in networks. Management Science, 57(9):1623–1639, 2011.

Sinan Aral and Dylan Walker. Identifying influential and susceptible members of social networks. Science, 337(6092):337–341, 2012.

Sinan Aral, Lev Muchnik, and Arun Sundararajan. Engineering social contagions: Optimal network seeding in the presence of homophily. Network Science, 1(2):125–153, 2013.

Peter M Aronow. A general method for detecting interference between units in randomized experiments. Sociological Methods & Research, 41(1):3–16, 2012.

Peter M. Aronow and Cyrus Samii. Estimating average causal efects under general interference. The Annals of Applied Statistics, 11(4):1912–1947, 2017.

Susan Athey, Dean Eckles, and Guido W Imbens. Exact p-values for network interference. Journal of the American Statistical Association, 113(521):230–240, 2018.

Sarah Baird, J Aislinn Bohren, Craig McIntosh, and Berk Ozler. Optimal design of ex- <sup>¨</sup> periments in the presence of interference. Review of Economics and Statistics, 100(5): 844–860, 2018.

Guillaume W. Basse and Avi Feller. Design of experiments in the presence of interference. Annual Review of Statistics and Its Application, 5:45–69, 2018. doi: 10.1146/ annurev-statistics-031017-100449.

Guillaume W Basse, Avi Feller, and Panos Toulis. Randomization tests of causal efects under interference. Biometrika, 106(2):487–494, 2019.

Rohit Bhattacharya, Daniel Malinsky, and Ilya Shpitser. Causal inference under interference and network uncertainty. In Uncertainty in Artificial Intelligence, pages 1028–1038. PMLR, 2020.

Robert M Bond, Christopher J Fariss, Jason J Jones, Adam DI Kramer, Cameron Marlow, Jaime E Settle, and James H Fowler. A 61-million-person experiment in social influence and political mobilization. Nature, 489(7415):295–298, 2012.

Yann Bramoull´e, Habiba Djebbari, and Bernard Fortin. Identification of peer efects through social networks. Journal of Econometrics, 150(1):41–55, 2009.

Chencheng Cai, Xu Zhang, and Edoardo M. Airoldi. Independent-set design of experiments for estimating treatment and spillover efects under network interference. In International Conference on Learning Representations (ICLR), 2024.

Damon Centola. The spread of behavior in an online social network experiment. Science, 329(5996):1194–1197, 2010.

Giacomo De Giorgi, Michele Pellizzari, and Silvia Redaelli. Identification of social interactions through identifying restrictions. American Economic Journal: Applied Economics, 2(2):275–291, 2010. doi: 10.1257/app.2.2.275.

Andrew Gifin, BJ Reich, Shu Yang, and AG Rappold. Generalized propensity score approach to causal inference with spatial interference. Biometrics, 79(3):2220–2231, 2023.

Bryan S Graham. Identifying social interactions through conditional variance restrictions. Econometrica, 76(3):643–660, 2008.

Mark Granovetter. Threshold models of collective behavior. American Journal of Sociology, 83(6):1420–1443, 1978.

Yuchen Hu, Shuangning Li, and Stefan Wager. Average direct and indirect causal efects under interference. Biometrika, 109(4):1165–1172, 2022.

Michael G Hudgens and M Elizabeth Halloran. Toward causal inference with interference. Journal of the American Statistical Association, (482):832–842, 2008.

Vishesh Karwa and Edoardo M Airoldi. A systematic investigation of classical causal inference strategies under mis-specification due to network interference. arXiv preprint arXiv:1810.08259, 2018.

Jure Leskovec, Lada A Adamic, and Bernardo A Huberman. The dynamics of viral marketing. ACM Transactions on the Web (TWEB), 1(1):5–es, 2007. doi: 10.1145/1232722. 1232727.

Shuangning Li and Stefan Wager. Random graph asymptotics for treatment efect estimation under network interference. The Annals of Statistics, 50(4):2334–2358, 2022.

Charles F Manski. Identification of endogenous social efects: The reflection problem. The Review of Economic Studies, 60(3):531–542, 1993.

Charles F Manski. Identification of treatment response with social interactions. The Econometrics Journal, 16(1):S1–S23, 2013.

Yuki Ohnishi, Bikram Karmakar, and Arman Sabbaghi. Degree of interference: A general framework for causal inference under interference. Journal of Machine Learning Research, 26(120):1–37, 2025.

Elizabeth Levy Paluck, Hana Shepherd, and Peter M. Aronow. Changing climates of conflict: A social network experiment in 56 schools. Proceedings of the National Academy of Sciences, 113(3):566–571, 2016.

Eleonora Patacchini, Edoardo Rainone, and Yves Zenou. Heterogeneous peer efects in education. Journal of Economic Behavior & Organization, 134:190–227, 2017.

Jean Pouget-Abadie, Guillaume Saint-Jacques, Martin Saveski, Weitao Duan, Souvik Ghosh, Ya Xu, and Edoardo M Airoldi. Testing for arbitrary interference on experimentation platforms. Biometrika, 106(4):929–940, 2019.

David Puelz, Guillaume Basse, Avi Feller, and Panos Toulis. A graph-theoretic approach to randomization tests of causal efects under general interference. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(1):174–204, 2022.

Joseph Rigdon and Michael G Hudgens. Exact confidence intervals in the presence of interference. Statistics & probability letters, 105:130–135, 2015.

Paul R Rosenbaum. Interference between units in randomized experiments. Journal of the American Statistical Association, 102(477), 2007.

Fredrik S¨avje. Causal inference with misspecified exposure mappings: separating definitions and assumptions. Biometrika, 111(1):1–15, 2024.

Fredrik S¨avje, Peter M Aronow, and Michael G Hudgens. Average treatment efects in the presence of unknown interference. arXiv preprint arXiv:1711.06399, 2017.

Michael E Sobel. What do randomized studies of housing mobility demonstrate? causal inference in the face of interference. Journal of the American Statistical Association, 101 (476):1398–1407, 2006.

Johan Ugander, Brian Karrer, Lars Backstrom, and Jon Kleinberg. Graph cluster randomization: Network exposure to multiple universes. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 329– 337. ACM, 2013.

Davide Viviano. Experimental design under network interference. Journal of Political Economy: Microeconomics, 2026. First version 2020; arXiv:2003.08421.

Bar Weinstein and Daniel Nevo. Causal inference with misspecified network interference structure. Biometrics, 82(1):ujag023, 2026.

Christina Lee Yu, Edoardo M Airoldi, Christian Borgs, and Jennifer T Chayes. Estimating the total treatment efect in randomized experiments with unknown network structure. Proceedings of the National Academy of Sciences, 119(44):e2208975119, 2022.

Corwin M Zigler and Georgia Papadogeorgou. Bipartite causal inference with interference. Statistical Science, 36(1):109–123, 2021.

Corwin M Zigler et al. Estimating causal efects of air quality regulations. Biostatistics, 13 (2):289–302, 2012.