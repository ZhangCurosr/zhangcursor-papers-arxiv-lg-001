# Hidden Disparities in Ranking: Category-Aware Fairness via Distributional Stability

Gayane Taturyan Charlotte Laclau Stephan Cl´emen¸con

LTCI, T´el´ecom Paris

Institut Polytechnique de Paris, France

## Abstract

We study the problem of fair ranking in recommender systems. We provide a formal probabilistic framework in which ranking policies are distributions over permutations, and the utility of a user is the exposure given to the items they interact with, relative to that of an ideal ranking. We consider utility parity across user groups, as well as its category-aware refinement, conditional utility parity, which requires parity within each category of items. Following a recent view of fairness as distributional stability, we show that both criteria are equivalent to the stability of the utility under changes in the composition of user groups. Optimizing over mixtures of ranking policies leads to a secondorder cone program, for which we provide generalization bounds. A numerical study validates our theoretical findings.

## 1 INTRODUCTION

Recommender Systems (RS), often trained as learningto-rank models, play an increasing role in access to opportunities. On job platforms, the ofers displayed to a candidate are ordered according to their predicted relevance to the candidate’s profile. When trained on historical interaction data, these models can reproduce disparities related to sensitive attributes such as gender. For example, women may see fewer high-paying or high-responsibility positions than men (Imana et al., 2021). Since candidates primarily interact with the highest-ranked ofers, what each user sees at the top of their ranking determines the quality of service they receive, and can ultimately afect their career path.

A natural answer is to require comparable utility across user groups, where the utility of a ranking measures how much exposure it gives to the ofers the user is interested in. We refer to this criterion as utility parity. Existing works on user-side fairness in recommendation compare a single utility, aggregated over all ofers (Li et al., 2021; Wang and Joachims, 2021; Yao and Huang, 2017). Such aggregate parity can be misleading: a ranking model may satisfy it while favouring one group on some categories of ofers, such as administrative positions, and the other group on others, such as managerial positions, so that the gaps cancel out in aggregate. A similar phenomenon is known in classification, where parity on a population can hide disparities within its subgroups (Kearns et al., 2018). Recent work has pointed out this limitation and proposed item category-aware criteria (Kheya et al., 2025; Wang et al., 2024).

Parity, whether aggregate or per category, also has a dynamic reading. On a job platform, the share of women among job seekers may change over time, for instance when the platform opens to new sectors or regions, and as long as a gap remains, the average quality of the recommendations rises or falls with this balance. We build on this reading, drawing inspiration from Taturyan et al. (2026), who view fairness in classification as the stability of a prediction rate under perturbations of group proportions, and learn fair classifiers as mixtures of reweighted predictors. Extending this view to ranking is far from straightforward, due to the global nature of this learning task. For each observation, a classifier predicts a label, which can be modified without changing the other predicted labels. In contrast, a ranking policy organizes the entire catalogue: each item is assigned a specific position in the list, so that its exposure depends on the position of all other items. And changing the rank of one item can afect the rank of (possibly many) other items. We therefore represent ranking policies through the exposure they allocate. The utility of a user then depends linearly on this exposure, so that average utility plays the role of the prediction rate, and the utility of a mixture of policies is exactly the mixture of their utilities. The stability view thus carries over to ranking, and the resulting problem is solved exactly over mixtures of policies, whereas in classification the risk of a mixture is handled through a surrogate.

The framework has several benefits. First, since the utility of a ranking is additive over the ofers it contains, it decomposes according to their category, and category-aware constraints arise naturally in this formulation, leading to what we call conditional utility parity. Second, utility parity is equivalent to the sta bility of utility under changes in the proportion of each user group, so that the guarantees hold uniformly over a range of group compositions. Third, the constraints can include a target of utility level per category, which prevents parity from being reached by lowering the utility that both groups derive from a category.

Contributions First, we provide a formal probabilistic framework for recommendation setup (Section 2). Second, we formalize utility parity and conditional utility parity as stability properties, which yields a constraint controlling both the group gap and the utility level (Section 3). Third, considering mixtures of policies, we write the empirical problem as a secondorder cone program (Section 4) and discuss its feasibility. Fourth, we provide finite-sample guarantees on fairness, deployment stability and utility, with explicit exposure rates for Plackett–Luce policies (Section 5). Finally, we conduct numerical experiments that confirm our theory in practice (Section 6).

Related Works There is a large amount of literature on fair ranking, most of which focus on itemside fairness through allocated exposure. A seminal work of Singh and Joachims (2018) formulates fairness of exposure as a linear program over doubly stochastic matrices, from which stochastic rankings are sampled through a Birkhof–von Neumann decompo sition. Wang and Joachims (2021) build on this approach to enforce user fairness, item fairness and di versity together: they optimize both notions of fairness as a convex program, and derive the ranking policy through a new decomposition algorithm. Exposurebased fairness has also been integrated in the training of ranking policies, by backpropagating through constrained optimization problems (Kotary et al., 2022; Dinh et al., 2024), or through eficient gradient esti mators for Plackett–Luce policies (Oosterhuis, 2021).

Closest to our work are the category-aware criteria of Kheya et al. (2025) and Wang et al. (2024). Kheya et al. (2025) compare the share of each category in the top-K lists of men and women and use the result ing gap as a regularizer during training. Wang et al. (2024) require a recall-based metric of relevant items to be similar across all pairs of user and item groups, and reweight the training losses with a variant of group distributionally robust optimization. Our criteria difer from both. Unlike the former, our utility only rewards exposure to relevant ofers. Unlike the latter, we compare user groups within each category, without asking categories to be treated equally.

Overall, despite the rich literature, most fair ranking methods are algorithmic and empirical, and do not rely on a probabilistic framework that would allow for statistical guarantees. The existing guarantees are mostly of a diferent nature. For example, fair rank aggregation algorithms come with approximation guarantees which bound how far their solution can be from the optimum (Chakraborty et al., 2022, 2026, 2025). Statisti cal guarantees have been provided in other setups, for instance in learning to rank, where a stochastic ranker built on a pretrained model is calibrated post hoc to guarantee a target level of utility, while item-exposure fairness is improved but not guaranteed (Guo et al., 2024), or in recommendation, where an online bandit algorithm decides with high probability whether some user would prefer another user’s recommendations, without degrading the audited system’s performance by more than a fixed fraction (Do et al., 2022). In contrast, we provide finite-sample guarantees on both the user-side category-aware fairness and the utility of a learned ranking policy, with fairness gaps and utility levels controlled uniformly over a range of group compositions. As a result, we observe a clear tradeof curve between utility and fairness, which methods without guarantees cannot ensure.

Notation Random variables are written in uppercase letters $( \mathrm { e . g . , ~ } G )$ , their realizations in corresponding lowercase letters $( \mathrm { e . g . , } g )$ . Vectors are set in boldface (e.g., X and x). For an integer $L \ > \ 0$ , we define $[ L ] \ { \stackrel { \mathrm { d e f } } { = } } \ \left\{ 1 , \cdots , L \right\}$ and write $\Delta _ { L }$ for the probability simplex in $\mathbb { R } ^ { L }$ . We denote by $\mathfrak { S } _ { L }$ the set of all permutations of [L].

## 2 BACKGROUND AND PRELIMINARIES

In this section, we provide a formal probabilistic framework for the recommendation setup.

## 2.1 Users, items, relevance

We consider a recommendation setting where the system ranks a fixed collection of items for a population of users. We denote the fixed finite item catalogue by $\mathcal { T } \stackrel { \mathrm { d e f } } { = } [ m ]$ . In addition, we assume that the item catalogue is partitioned into A disjoint groups, $\mathcal { Z } =$ $\textstyle { \bigcup _ { a = 1 } ^ { A } { \overline { { Z } } } _ { a } }$ , where each $\mathcal { T } _ { a }$ represents a collection of items sharing a common attribute of interest (e.g., category). A user is modeled as a random triplet $\pmb { Z } = ( \pmb { X } , \pmb { G } , \pmb { Y } )$ , taking values in ${ \mathcal { Z } } \ { \stackrel { \mathrm { d e f } } { = } } \ { \mathcal { X } } \times { \mathcal { G } } \times { \mathcal { Y } }$ , and defined on a probability space $( \Omega , \mathcal { F } , \mathbb { P } )$ with joint law $\mathcal { D } \in \mathcal { M } _ { 1 } ( \mathcal { Z } )$ Here, $\boldsymbol { x } \in \mathcal { X }$ is the user–query feature vector, that may contain a static profile of user characteristics, as well as session-level information (e.g., search terms, active filters, and other contextual features). A binary sensitive attribute is denoted by $G \in \{ 0 , 1 \}$ and represents a protected characteristic of a user, such as race or gender, with a marginal $p _ { g } \ { \stackrel { \mathrm { d e f } } { = } } \ \mathbb { P } ( G = g )$ . Finally, $\pmb { Y } = ( Y _ { i } ) _ { i \in [ m ] } \in \{ 0 , 1 \} ^ { m }$ denotes the relevance vector of the user, where $Y _ { i } = 1$ if the user interacted with item i (e.g., clicked, applied), and $Y _ { i } = 0$ otherwise.

## 2.2 Ranking policies, exposure, utility

Definition 1 (Ranking policy). A ranking policy is a Markov kernel $P : \mathcal { X } \to \mathcal { M } _ { 1 } ( \mathfrak { S } _ { m } ) , \pmb { x } \mapsto P ( \cdot \mid \pmb { x } )$ Given $X = x$ the system draws a ranking $\pi \sim P ( \cdot \mid$ x), and $\pi ( j )$ is the item placed at position $j \in [ m ]$

We operate in an unaware setup, where the policy does not have access to sensitive attribute G at prediction time. The policy can also be summarised by its position marginals.

Definition 2 (Marginal rank matrix). For a policy P and $\textbf { \em x } \in { \mathcal { X } }$ , the marginal rank matrix is $\mathbf { M } ^ { P } ( \pmb { x } ) \in$ $[ 0 , 1 ] ^ { m \times m }$ with $\mathbf { M } _ { i j } ^ { P } ( { \pmb x } ) \stackrel { \mathrm { d e f } } { = } \mathbb { P } _ { \pi \sim P ( \cdot | { \pmb x } ) } ( \pi ( j ) = i )$

Each row and column of ${ \bf { M } } ^ { P }$ sums to one, making ${ \bf M } ^ { P }$ doubly stochastic, and thus a point of the Birkhof polytope $\mathfrak { B } _ { m }$ . This formulation accommodates both deterministic and stochastic ranking policies.

Example 1 (Deterministic sorting policy). A scoring function $s _ { \theta } : \mathcal { X } \times \mathcal { T } $ R induces a deterministic sorting policy $\begin{array} { r } { P _ { \theta } ( \pi \mid x ) = \mathbb { I } \{ \pi = \pi _ { \theta } ( x ) \} } \end{array}$ , where $\pi _ { \boldsymbol \theta } ( { \pmb x } ) = \mathrm { a r g } \mathrm { s o r t } _ { i \in [ m ] } s _ { \boldsymbol \theta } ( { \pmb x } , i )$ ranks the items by decreasing score.

Example 2 (Plackett–Luce policy (Plackett, 1975)). A scoring function $s _ { \theta } ~ \because ~ { \mathcal { X } } ~ \times ~ { \mathcal { T } } ~ \to ~ { \mathbb { R } } ~ i n - $ duces the Plackett-Luce policy $\begin{array} { r l } { P _ { \theta } ( \pi } & { { } | \textbf { \textit { x } } ) } \end{array} =$ $\begin{array} { r } { \prod _ { j = 1 } ^ { m } \frac { \exp \left( s _ { \theta } \left( \pmb { x } , \pi ( j ) \right) \right) } { \sum _ { i = j } ^ { m } \exp \left( s _ { \theta } \left( \pmb { x } , \pi ( i ) \right) \right) } } \end{array}$ , from which rankings are sampled by successive Gumbel-max draws.

Considering that higher-ranked items receive more attention, let $\pmb { v } = ( v _ { 1 } , \dots , v _ { m } )$ be a position-discount vector with $v _ { 1 } \geqslant \cdot \cdot \cdot \geqslant v _ { m } \geqslant 0 .$ . Assuming $\mathbf { \nabla } _ { \pm } \mathbf { 0 }$ , without loss of generality, we normalize so that $\| \pmb { v } \| _ { 1 } = 1$ The position-discount vector allows us to quantify the exposure that each item receives under a given policy. Definition 3 (Expected exposure). The expected exposure of item $ { \boldsymbol { { i } } } _ { \mathbf { {  { ~ \left( \varepsilon ~ \left. ~ \varepsilon \right. ~ \varepsilon \right. ~ } } } }  { \boldsymbol { { \tau } } }$ under policy P for a user with features x is $\begin{array} { r l } { \rho _ { i } ( P ; \pmb { x } ) } & { { } \stackrel { \mathrm { d e f } } { = } } \end{array}$ $\begin{array} { r } { \mathbb { E } _ { \pi \sim P ( \cdot | \mathbf { x } ) } \Big [ \sum _ { j = 1 } ^ { m } v _ { j } \mathbb { I } \{ \pi ( j ) = i \} \Big ] } \end{array}$

Equivalently, using the marginal rank matrix notation, we have $\begin{array} { r } { \rho _ { i } ( P ; { \pmb x } ) = \sum _ { j = 1 } ^ { m } v _ { j } { \bf M } _ { i j } ^ { P } ( { \pmb x } ) = \big ( { \bf M } ^ { P } ( { \pmb x } ) { \pmb v } \big ) _ { i } } \end{array}$ and $\begin{array} { r } { \rho ( P ; \pmb { x } ) \stackrel { \mathrm { d e f } } { = } \mathbf { M } ^ { P } ( \pmb { x } ) \pmb { v } } \end{array}$ . The notion of exposure leads to the utility of the ranking policy, which measures the quality of the policy by aligning the allocated exposure with user’s relevant items.

Definition 4 (Utility). The utility of P for a user z is defined as $\begin{array} { r c l l } { U ( P ; z ) } & { \stackrel { \mathrm { d e f } } { = } } & { \sum _ { i \in [ m ] } \rho _ { i } ( P ; { \pmb x } ) y _ { i } } & { = } \end{array}$ $\langle \pmb { \rho } ( P ; \pmb { x } ) , \pmb { y } \rangle$

However, the given notion of the utility is not comparable across users, as users with fewer interactions have lower overall utility.

Definition 5 (Normalized utility). We define the normalized utility of P for a user z as $\begin{array} { r } { \overline { { U } } ( P ; z ) \stackrel { \mathrm { d e f } } { = } \frac { U ( P ; z ) } { V ( z ) } } \end{array}$ where $\begin{array} { r } { V ( z ) \stackrel { \mathrm { d e f } } { = } \underset { \mathbf { M } \in \mathcal { B } } { \operatorname* { m a x } } \left. \mathbf { M } \pmb { v } , \pmb { y } \right. = \sum _ { j = 1 } ^ { \| \pmb { y } \| _ { 1 } } v _ { j } } \end{array}$

Assumption 1. It holds that $\| \mathbfcal { Y } \| _ { 1 } ~ \geqslant ~ 1$ , hence $V ( Z ) \geqslant \underline { { v } } > 0$ , almost surely.

The expected normalized utility is defined as $\mathcal { U } ( P ) \stackrel { \mathrm { d e f } } { = }$ $\mathbb { E } [ \overline { { U } } ( P ; Z ) ]$ , and the group-conditional normalized utility as $\mathcal { U } _ { g } ( P ) \stackrel { \mathrm { d e f } } { = } \mathbb { E } [ \overline { { U } } ( P ; Z ) | G = g ]$

Remark 1 (Connection to NDCG@K). The normalized discounted cumulative gain at cutof K (NDCG@K) is one of the most popular metrics used for evaluating rankings. For a ranking π and labels $y , \textit { \textbf { i t } } \textit { \textbf { i s } }$ defined as $\begin{array} { r l r } { \mathrm { N D C G @ } K ( \pi ; \pmb { y } ) } & { { } = } & { \frac { \mathrm { D C G @ } K ( \pi ; \pmb { y } ) } { \operatorname* { m a x } _ { \pi ^ { \prime } \in \mathfrak { S } _ { m } } \mathrm { D C G @ } K ( \pi ^ { \prime } ; \pmb { y } ) } } \end{array}$ , where DCG@ $\begin{array} { r l r } { K ( \pi ; \pmb { y } ) } & { { } = } & { \sum _ { j = 1 } ^ { K } \frac { \pmb { y } _ { \pi ( j ) } } { \log _ { 2 } ( 1 + j ) } , } \end{array}$ Our normalized utility recovers its binary-relevance case with the truncated logarithmic discount $\begin{array} { r } { v _ { j } ~ = ~ \frac { c \mathbb { I } \{ j \leqslant K \} } { \log _ { 2 } ( 1 + j ) } , } \end{array}$ where $c \quad > \quad 0$ normalizes $\| v \| _ { 1 } ~ = ~ 1$ It yields $\overline { { U } } ( P ; z ) = \mathbb { E } _ { \pi \sim P ( \cdot | \pmb { x } ) } [ \mathrm { N D C G @ } K ( \pi ; \pmb { y } ) ]$

## 2.3 Fairness as stability

In a recent work, Taturyan et al. (2026) define fairness in classification as a property of distributional stability. A classifier $h \ : \ \mathcal { X } \  \ \{ 0 , 1 \}$ is ε–fair w.r.t. class of perturbation operators P, if $\begin{array} { r } { \operatorname* { s u p } _ { ( \mathcal { D } , \mathbf { P } ) \in \mathcal { M } \times \mathfrak { P } } \Delta ( h ; ~ ( \mathcal { D } , \mathbf { P } ) ) \leqslant \varepsilon } \end{array}$ , where $\Delta ( h ; ~ ( \mathcal { D } , \mathbf { P } ) )$ is a statistical distance between the distributions of the prediction $h ( X )$ under $\mathcal { D } \in \mathcal { M }$ and under $\mathbf P ( \mathcal D )$ The authors show, that the individual and group fairness can both be expressed as stability under specific choices of P and ∆. For group fairness, the perturbations change the composition of the sensitive groups.

## 3 UTILITY PARITY AND DISTRIBUTIONAL STABILITY

In this section, we consider recommendation utility based fairness constraints and then express them as stability under distributional changes.

## 3.1 Utility-based parities

First, we introduce a rather natural notion of Utility Parity, that requires the expected recommendation quality to be the same across user groups.

Definition 6 (Utility Parity). A policy P satisfies Utility Parity (UP) $i f f { \mathcal { U } } _ { 0 } ( P ) = { \mathcal { U } } _ { 1 } ( P )$

Next, we introduce Conditional Utility Parity, that requires parity within each category ${ \mathcal { T } } _ { a }$ . For $a \in [ A ]$ denote the normalized utility within the category by

$$
\overline { { U } } _ { a } ( P , z ) \stackrel { \mathrm { d e f } } { = } \frac { \sum _ { i \in \mathbb { Z } _ { a } } \rho _ { i } ( P ; \pmb { x } ) y _ { i } } { V ( z ) } ,
$$

and the respective expectations by $\mathcal { U } ^ { a } ( P )$ <sup>def</sup> = $\mathbb { E } [ \overline { { U } } _ { a } ( P , Z ) ]$ and $\mathcal { U } _ { g } ^ { a } ( P ) \ { \stackrel { \mathrm { d e f } } { = } } \ \mathbb { E } [ \overline { { U } } _ { a } ( P , Z ) \mid G = g ]$

Definition 7 (Conditional Utility Parity). A policy P satisfies Conditional Utility Parity $( C U P ) ~ i f f { \mathcal U } _ { 0 } ^ { a } ( P ) =$ $\mathcal { U } _ { 1 } ^ { a } ( P )$ for every $a \in [ A ]$

Notice that, $\begin{array} { r } { \overline { { U } } ( P , z ) = \sum _ { a \in [ A ] } \overline { { U } } _ { a } ( P , z ) } \end{array}$ and $\mathcal { U } _ { g } ( P ) =$ $\textstyle \sum _ { a \in [ A ] } { \mathcal { U } } _ { g } ^ { a } ( P )$ . Therefore, CUP implies UP, and enforcing just CUP is enough to ensure both.

## 3.2 Group perturbations and utility rates

We consider a distributional perturbation, where we change the sensitive group proportions, while keeping the conditionals the same. For a parameter $\xi \in [ 0 , 1 ]$ we define the following perturbed distribution,

$$
{ \mathcal { D } } _ { \xi } \ { \stackrel { \mathrm { d e f } } { = } } \ \xi { \mathcal { D } } ( \cdot \mid G = 1 ) + ( 1 - \xi ) { \mathcal { D } } ( \cdot \mid G = 0 ) .
$$

To ensure that both groups are always present in initial and perturbed distributions, we enforce the following natural assumption.

Assumption 2 (Bounded proportions). $p _ { g } \in [ \underline { { p } } , 1 - \underline { { p } } ]$ and $\xi \in [ \xi , 1 - \xi ]$ for some $\underline { { \xi } } , \underline { { p } } \in ( 0 , \frac { 1 } { 2 } )$ and $g \in \{ 0 , 1 \}$

Under Assumption 2, we have $\mathcal { D } _ { \xi } \ll \mathcal { D } _ { \xi }$ . Thus, computing Radon–Nikodym derivative $\begin{array} { r } { \begin{array} { r c l } { \displaystyle w _ { \xi } ( g ) \stackrel { \mathrm { d e f } } { = } } & { \frac { \mathrm { d } \mathcal { D } _ { \xi } } { \mathrm { d } \mathcal { D } } ( g ) } \end{array} } \end{array}$ ， we get the explicit importance weights

$$
w _ { \xi } ( g ) = \frac { \xi } { p _ { 1 } } \mathbb { I } \{ g = 1 \} + \frac { 1 - \xi } { p _ { 0 } } \mathbb { I } \{ g = 0 \} .\tag{1}
$$

Moreover, for every integrable $f ,$ it holds that

$$
\begin{array} { r } { \mathbb { E } _ { \mathcal { D } _ { \xi } } [ f ( Z ) ] = \mathbb { E } _ { \mathcal { D } } \left[ w _ { \xi } ( G ) f ( Z ) \right] . } \end{array}
$$

Definition 8 (Utility rate functionals). For a ranking policy P and a perturbation parameter $\xi \in [ 0 , 1 ]$ , we define the following rate functionals

$$
\Phi ( P ; \xi ) \stackrel { \mathrm { d e f } } { = } \mathbb { E } _ { { \mathcal { D } } _ { \xi } } [ \overline { { U } } ( P ; { \pmb Z } ) ] = \mathbb { E } [ w _ { \xi } ( G ) \overline { { U } } ( P ; { \pmb Z } ) ]
$$

$$
\begin{array} { r l } & { \Phi _ { a } ( P ; \xi ) \overset { \mathrm { d e f } } { = } \mathbb { E } _ { \mathcal { D } _ { \xi } } [ \overline { { U } } _ { a } ( P ; Z ) ] } \\ & { \qquad = \mathbb { E } [ w _ { \xi } ( G ) \overline { { U } } _ { a } ( P ; Z ) ] \ f o r \ a l l \ a \in [ A ] . } \end{array}
$$

We notice that $\begin{array} { r } { \Phi ( \cdot ; \cdot ) ~ = ~ \sum _ { a \in [ A ] } \Phi _ { a } ( \cdot ; \cdot ) } \end{array}$ Moreover, both functionals admit an afine decomposition through the law of total expectation and (1). For $\Phi _ { \perp } \in \{ \Phi ; \Phi _ { a } \}$ and $\overline { { U } } _ { \perp } \in \{ \overline { { U } } ; \hat { \overline { { U } } } _ { a } \}$ , it holds that

$$
\Phi _ { \Pi } ( P ; \xi ) = \xi \mathbb { E } [ \overline { { U } } _ { \Pi } \mid G = 1 ] + ( 1 - \xi ) \mathbb { E } [ \overline { { U } } _ { \Pi } \mid G = 0 ]
$$

$$
= \mathbb { E } [ \overline { { U } } _ { \perp } \mid G = 0 ] + \xi ( \mathbb { E } [ \overline { { U } } _ { \perp } \mid G = 1 ] - \mathbb { E } [ \overline { { U } } _ { \perp } \mid G = 0 ] ) .\tag{2}
$$

This allows us to formulate the following proposition. Proposition 1 (Parity and stability). Let P be a ranking policy. Then

1. P satisfies $U P \Leftrightarrow \xi \mapsto \Phi ( P ; \xi )$ is constant on $[ 0 , 1 ] .$ Moreover, the unfairness gap is

$$
\left| \gamma ( P ) \right| \stackrel { \mathrm { d e f } } { = } \operatorname* { s u p } _ { \xi , \xi ^ { \prime } \in [ 0 , 1 ] } \left| \Phi ( P ; \xi ) - \Phi ( P ; \xi ^ { \prime } ) \right| .
$$

2. P satisfies $C U P \Leftrightarrow \xi \mapsto \Phi _ { a } ( P ; \xi )$ is constant on $[ 0 , 1 ]$ for all $a \in [ A ]$ . Moreover, the unfairness gap is

$$
\left| \gamma _ { a } ( P ) \right| \stackrel { \mathrm { d e f } } { = } \operatorname* { s u p } _ { \xi , \xi ^ { \prime } \in [ 0 , 1 ] } \left| \Phi _ { a } ( P ; \xi ) - \Phi _ { a } ( P ; \xi ^ { \prime } ) \right| , \forall a \in [ A ] .
$$

Both criteria are Lipschitz statements about the map $\xi \mapsto \Phi _ { \Pi } ( P ; \xi )$ . Since this map is afine, its Lipschitz constant is exactly the unfairness gap. Parity, however, is satisfied just as well by equally poor rank ings as by equally good ones. Under UP, maximizing the expected utility $\mathcal { U } ( P )$ under the constraint solves this, however for CUP it does not. Since $\begin{array} { r } { \mathcal { U } ( P ) = \sum _ { a \in [ A ] } \mathcal { U } ^ { a } ( P ) } \end{array}$ , parity within a category may be reached at the expense of both groups. Moreover, any residual gap makes utility vary with the composition of users. We therefore introduce a constraint that keeps utility, within each category for conditional utility parity, close to a target across group compositions.

## 3.3 Stability to perturbations

Let $\Xi = U [ \xi , 1 - \xi ]$ be the perturbation parameter distribution. We denote $\mu { \boldsymbol { \Xi } } { \stackrel { \mathrm { d e f } } { = } } { \mathbb { E } } { \boldsymbol { \Xi } } [ \xi ]$ and $\sigma _ { \Xi } ^ { 2 } \ { \stackrel { \mathrm { d e f } } { = } } \ \operatorname { V a r } _ { \Xi } ( \xi )$ Definition 9 (Perturbation variance and t-variance). For $t , t _ { a } \in [ 0 , 1 ]$ and rate functionals Φ and $\Phi _ { a . }$ , we define perturbation variance functionals as

$$
\begin{array} { r l } & { \mathscr { V } _ { \Xi } ( P ) \overset { \mathrm { d e f } } { = } \operatorname { V a r } _ { \xi \sim \Xi } \big ( \Phi ( P ; \xi ) \big ) \quad a n d } \\ & { \mathscr { V } _ { \Xi } ^ { a } ( P ) \overset { \mathrm { d e f } } { = } \operatorname { V a r } _ { \xi \sim \Xi } \big ( \Phi _ { a } ( P ; \xi ) \big ) \quad \forall a \in [ A ] , } \end{array}
$$

and perturbation t-variance functionals as

$$
\begin{array} { r l } & { { \mathscr C } _ { \Xi } ( P ; t ) \stackrel { \mathrm { d e f } } { = } \mathbb { E } _ { \xi \sim \Xi } \big [ ( \Phi ( P ; \xi ) - t ) ^ { 2 } \big ] \quad a n d } \\ & { { \mathscr C } _ { \Xi } ^ { a } ( P ; t ) \stackrel { \mathrm { d e f } } { = } \mathbb { E } _ { \xi \sim \Xi } \big [ ( \Phi _ { a } ( P ; \xi ) - t _ { a } ) ^ { 2 } \big ] \quad \forall a \in [ A ] . } \end{array}
$$

Since $\Phi _ { \perp } ( P ; \cdot )$ is afine, $\mathcal V _ { \Xi } ^ { \perp } ( P ) = \gamma _ { \perp } ( P ) ^ { 2 } \sigma _ { \Xi } ^ { 2 }$ , so that $| \gamma _ { \perp } ( P ) | \leqslant \varepsilon \iff \mathcal { V } _ { \Xi } ^ { \perp } ( P ) \overset { - } { \leqslant } \varepsilon ^ { 2 } \sigma _ { \Xi } ^ { 2 }$ . Adding and subtracting $\mathbb { E } { \Xi } [ \Phi _ { \perp } ( P ; \xi ) ]$ ] gives the following bias-variance decomposition

$$
\mathcal { C } _ { \Xi } ^ { \perp } ( P ; t ) = \mathcal { V } _ { \Xi } ^ { \perp } ( P ) + \big ( \mathbb { E } _ { \Xi } [ \Phi _ { \Xi } ( P ; \xi ) ] - t \big ) ^ { 2 } ,\tag{3}
$$

so that a bound ${ \mathcal C } _ { \Xi } ^ { \perp } ( P ; t ) \ \leqslant \ \delta$ controls both the slope $\gamma _ { \perp } ( P )$ (fairness) and the mean level of $\Phi _ { \perp } ( P ; \cdot )$ around the target t<sub>□</sub> (uniform deployment). Moreover, it holds that $\begin{array} { r } { | \gamma ( P ) | ~ \leqslant ~ \sum _ { a \in [ A ] } | \gamma _ { a } ( P ) | } \end{array}$ and $\begin{array} { r } { ( \mathcal { C } _ { \Xi } ( P ; \sum _ { a \in [ A ] } t _ { a } ) ) ^ { 1 / 2 } \leqslant \sum _ { a \in [ A ] } ( \mathcal { C } _ { \Xi } ^ { a } ( P ; t _ { a } ) ) ^ { 1 / 2 } } \end{array}$

Remark 2 (On normalization). Interestingly, under $C U P , \mathcal { U } ^ { a }$ is normalized by $V ( z )$ and is averaged over all users, so a gap between $\mathcal { U } _ { 0 } ^ { a } ( P )$ and $\mathcal { U } _ { 1 } ^ { a } ( P )$ may partly reflect diferences in how often the two groups are interested in category a. A variant closer to Equal Opportunity would restrict each category to the users interested in it. Writing $\scriptstyle { \boldsymbol { y } } ^ { a }$ for the restriction of y to $\mathcal { T } _ { a }$ , let $S _ { a } = \{ z : { \pmb y } ^ { a } \neq 0 \}$ , normalize by the ideal utility of the category $\begin{array} { r } { V ^ { a } ( z ) = \sum _ { j \leq \| \pmb { y } ^ { a } \| _ { 1 } } v _ { j } } \end{array}$ , and, $f o r z \in S _ { a }$ $\begin{array} { r l r } { \widetilde { U } ^ { a } ( P , z ) } & { { } \stackrel { \mathrm { d e f } } { = } } & { \sum _ { i \in I _ { a } } \rho _ { i } ( P ; \pmb { x } ) y _ { i } / V _ { a } ( z ) } \end{array}$ . The matching perturbation mixes $\overset { \cdot } { \mathcal { D } } ( \cdot \mathrm { ~ \vert ~ } S _ { a } , G = g )$ , where $\xi _ { a }$ is the share of group 1 among users in $S _ { a } . { \ \widetilde \Phi } _ { a } ( P ; \xi _ { a } )$ is then afine in $\xi _ { a }$ with slope $\mathbb { E } [ \widetilde { U } ^ { a } \ | \ S _ { a } , G = 1 ] - \mathbb { E } [ \widetilde { U } ^ { a } \ |$ $S _ { a } , G = 0 ]$ , and the optimization problem keeps the same form with $\widetilde { \mathcal { U } } _ { a }$ in place $o f { \mathcal { U } } _ { a }$ . However, these criteria no longer sum to U, and thus do not imply UP.

## 4 METHODOLOGY : MIXTURE POLICIES

We now propose a methodology for maximizing utility under the stability constraints, by optimizing over mixtures of ranking policies.

## 4.1 Linearity in policies

Definition 10 (Policy mixture). For policies $P _ { 1 } , \cdots , P _ { L }$ and $\pmb { \alpha } \in \Delta _ { L }$ we define policy mixture as $P ^ { \alpha } ( \cdot \mid x ) \stackrel { \mathrm { d e f } } { = } \sum _ { \ell = 1 } ^ { L } \alpha _ { \ell } P _ { \ell } ( \cdot \mid x )$

Note that policy mixture is again a policy, since it is a convex combination of probability measures on ${ \mathfrak { S } } _ { m }$ . Moreover, the linearity transfers to the following terms.

Lemma 1 (Policy linearity). For all $\pmb { \alpha } \in \Delta _ { L }$ , all ${ \pmb x } \in$ X and all $z \in { \mathcal { Z } }$ , it holds that

$$
\mathbf { M } ^ { P ^ { \alpha } } ( \pmb { x } ) = \sum _ { \ell \in [ L ] } \alpha _ { \ell } \mathbf { M } ^ { P _ { \ell } } , \rho ( P ^ { \alpha } ; \pmb { x } ) = \sum _ { \ell \in [ L ] } \alpha _ { \ell } \rho ( P _ { \ell } ; \pmb { x } ) ,
$$

$$
\overline { { { U } } } _ { \perp } ( P ^ { \alpha } ; z ) = \sum _ { \ell \in [ L ] } \alpha _ { \ell } \overline { { { U } } } _ { \perp } ( P _ { \ell } ; z ) , \ w i t h \ \overline { { { U } } } _ { \perp } \in \{ \overline { { { U } } } ; \overline { { { U } } } _ { a } \} .
$$

Note, that $\Phi _ { \perp }$ is jointly bi-afine: it is afine in $\xi$ for fixed $P$ and linear in $P$ for fixed $\xi .$

## 4.2 Problem formulation

To write in a unified manner, let us denote by A the set of active constraints. We will consider $A = \{ 0 \}$ for $\mathrm { U P }$ and $A = \left\lceil A \right\rceil$ for CUP. With this notation, we have $\gamma _ { 0 } \equiv \gamma , { \mathcal C } _ { \Xi } ^ { 0 } \equiv { \bar { { \mathcal C } } } _ { \Xi } , t _ { 0 } \equiv t , \delta _ { 0 } \equiv \delta$ . Given a policy dictionary $P _ { 1 } , \cdots , P _ { L } $ , as well as $t _ { a } \in [ 0 , 1 ]$ and $\delta _ { a } > 0$ for all $a \in A .$ , our objective is to solve the following problem

$$
\alpha ^ { \star } \in \arg \operatorname* { m a x } _ { \alpha \in \Delta _ { L } } \left\{ \mathcal { U } ( P ^ { \alpha } ) : \mathcal { C } _ { \Xi } ^ { a } ( P ^ { \alpha } ; t _ { a } ) \leqslant \delta _ { a } , \forall a \in \mathcal { A } \right\} .\tag{P}
$$

Enforcing UP requires one constraint, and enforcing CUP requires one constraint per category (i.e., A constraints).

Empirical problem Let $\mathcal { D } ^ { n ^ { \prime } } = \{ Z _ { 1 } ^ { \prime } , \cdot \cdot \cdot , Z _ { n ^ { \prime } } ^ { \prime } \}$ be an i.i.d. training sample from D. Let $\widehat { P } _ { 1 } , \cdots , \widehat { P } _ { L }$ be policies trained on ${ \mathcal { D } } ^ { n ^ { \prime } } , \ \mathrm { e . g . }$ , under the importance weights $w _ { \xi _ { \ell } }$ , and denote by $P _ { \ell }$ the population counterpart of $\widehat { P } _ { \ell } .$ . Let $\mathcal { D } ^ { n } ~ = ~ \{ Z _ { 1 } , \cdot \cdot \cdot , Z _ { n } \}$ be independent i.i.d. sample from D, and $\begin{array} { r l } { \Xi ^ { N } } & { { } = } \end{array}$ $\{ \xi _ { 1 } , \cdots , \xi _ { N } \}$ be perturbation parameters (stratified) sampled from $\Xi .$ For simplicity, we assume $p _ { g }$ to be known. For $a \in A .$ define a matrix $\widehat { \pmb { T } } ^ { a } \in \mathbb { R } ^ { N \times L }$ with elements ${ \widehat T } _ { k , \ell } ^ { a } = { \widehat \Phi } _ { n } ^ { a } ( { \widehat P } _ { \ell } , \xi _ { k } )$ , where $\widehat { \Phi } _ { n } ^ { a } \big ( \widehat { P } _ { \ell } , \xi _ { k } \big ) =$ $\begin{array} { r } { \frac { 1 } { n } \sum _ { i \in [ n ] } w _ { \xi _ { k } } ( G _ { i } ) \overline { { U } } _ { a } ( \widehat { P } _ { \ell } ; Z _ { i } ) } \end{array}$ . The empirical t-variance functional is $\begin{array} { r } { \widehat { \mathcal { C } } _ { n , N } ^ { a } ( \pmb { \alpha } ; t _ { a } ) ~ = ~ \frac { 1 } { N } \sum _ { k \in [ N ] } ( \widehat { \Phi } _ { n } ^ { a } ( \widehat { P } ^ { \pmb { \alpha } } ; \xi _ { k } ) ~ - } \end{array}$ $\begin{array} { r } { t _ { a } ) ^ { 2 } = \frac { 1 } { N } \| \widehat { T } _ { a } \pmb { \alpha } - t _ { a } \mathbf { 1 } \| _ { 2 } ^ { 2 } } \end{array}$ . Let us also define a vector $\widehat { \pmb u } \in \mathbb { R } ^ { L }$ , with elements $\begin{array} { r } { \widehat { u } _ { \ell } = \frac { 1 } { n } \sum _ { i \in [ n ] } \overline { { U } } ( \widehat { P } _ { \ell } , Z _ { i } ) } \end{array}$ With this notation, we can write the empirical problem as a second-order cone program (SOCP),

$$
\widehat { \alpha } \in \underset { \alpha \in \Delta _ { L } } { \arg \operatorname* { m a x } } \ \big \{ \widehat { \pmb { u } } ^ { \top } \pmb { \alpha } : \| \widehat { T _ { a } } \pmb { \alpha } - t _ { a } { \bf 1 } \| _ { 2 } \leqslant \sqrt { N \delta _ { a } } , \forall a \in \mathcal { A } \big \} .\tag{Pb}
$$

## 4.3 Existence and feasibility

Existence The feasible sets of both (P) and $( \widehat { \mathcal { P } } )$ are convex and compact, and whenever they are nonempty, an optimizer exists.

Feasibility In classical fairness problems the (strict) feasibility of the constraint is ensured by considering constant functions. When considering mixtures of classifiers, the feasibility can be ensured by adding constant function(s) to the dictionary and choosing appropriate weights. However, in exposure-based fairness problems feasibility isn’t always guaranteed. For example, in the work of Singh and Joachims (2018), the authors present a specific conditions under which the constraint is attainable for their objective and explain how to handle the issue in practice, however they do not guarantee the feasibility in general. Similarly, feasibility cannot be guaranteed in general for our objective and must instead be characterized. These are conditions that can be checked in practice, and we discuss them in Appendix A. From a theoretical perspective, we impose the following simplified assumption.

Assumption 3 (Strict feasibility). There exists $ { \alpha } ^ { 0 } \in$ $\Delta _ { L } s . t . \mathscr { C } _ { \Xi } ^ { a } ( P ^ { \alpha ^ { 0 } } ; t _ { a } ) < \delta _ { a } , \forall a \in \mathcal { A }$

## 5 THEORETICAL GUARANTEES

We now study how the empirical solution $\widehat { P } ^ { \widehat { \alpha } } \ \mathrm { o f } \ ( \widehat { \mathcal { P } } )$ approaches its population counterpart $P ^ { \alpha ^ { \star } }$ of (P). By Lemma 1, the rate functionals and the utility depend on a policy only through its exposure, thus a natural requirement on the dictionary is a rate on exposures.

Assumption 4 (Policy-exposure rates). For any $\eta _ { \delta _ { n ^ { \prime } } } \in ( 0 , 1 )$ , with probability greater than $1 - \eta _ { \delta _ { n } }$ it holds that ma $\begin{array} { r } { \mathbf { \boldsymbol { x } } _ { \ell \in [ L ] } \operatorname { \mathbb { E } } \| \rho ( P _ { \ell } ; \boldsymbol { X } ) - \rho ( \widehat { P } _ { \ell } ; \boldsymbol { X } ) \| _ { 1 } \leqslant \delta _ { n ^ { \prime } } } \end{array}$

In general we want $\delta _ { n ^ { \prime } } \to 0$ , which can be computed for specific policies. For example, the following Lemma states the exposure rates for a Plackett-Luce policies.

Lemma 2 (Plackett-Luce rates). Let $P _ { \ell }$ and $\widehat { P } _ { \ell }$ be Plackett-Luce policies with score functions $s _ { \ell }$ and $\widehat { s } _ { \ell ; }$ for $\ell \in [ L ]$ . For every $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ , it holds that

$$
\begin{array} { r } { \| \rho ( P _ { \ell } ; \pmb { x } ) - \rho ( \widehat { P } _ { \ell } ; \pmb { x } ) \| _ { 1 } \leqslant \sqrt { m - 1 } \| \pmb { s } _ { \ell } ( \pmb { x } , \cdot ) - \widehat { \pmb { s } } _ { \ell } ( \pmb { x } , \cdot ) \| _ { \infty } . } \end{array}
$$

Under mild appropriate noise conditions, fast rates for the statistical recovery of optimal ranking/scoring policies have been obtained in Cl´emen¸con et al. (2018), extending the results for fast empirical ranking aggregation proved in Korba et al. (2017) (see also Vogel and Cl´emen¸con (2020) for one-versus-one approaches).

The error comes from three independent samples, each with its own rate. The dictionary policies are trained on $\mathcal { D } ^ { n ^ { \prime } }$ ; by Lemma 1, the exposure error of Assumption 4 transfers to the rate functionals of any mixture with an error $\delta _ { n ^ { \prime } } ^ { \prime } \stackrel { \mathrm { d e f } } { = } \delta _ { n ^ { \prime } } \big / ( \underline { { p } } \underline { { v } } )$ , where $1 / \underline { { p } }$ bounds the importance weights and $1 / \underline { { v } }$ comes from the normalization of the utility. Given the dictionary, the rate functionals and the utility are estimated on $\mathcal { D } ^ { n }$ , which contributes $\delta _ { n } \asymp n ^ { - \frac { 1 } { 2 } }$ and $\delta _ { n } ( \mathcal { U } ) \asymp n ^ { - \frac { 1 } { 2 } }$ , respectively. Finally, the perturbation law $\Xi$ is discretized by the stratified sample $\Xi ^ { N }$ , and since the rate functionals are afine in $\xi ,$ this contributes $\delta _ { N } \asymp N ^ { - \frac { 3 } { 2 } }$

Theorem 1 (Informal). Suppose Assumptions $1 \mathrm { - } \mathit { 4 }$ hold. Then, conditionally on $\mathcal { D } ^ { n ^ { \prime } }$ and with “high prob-$a b i l i t y ^ { \prime \prime }$ over $( \mathcal { D } ^ { n } , \Xi ^ { N } )$ , every αb feasible for (Pb) satisfies, for all $a \in { \mathcal { A } } .$

1. Stability. $\mathcal { C } _ { \Xi } ^ { a } ( \widehat { P } ^ { \widehat { \alpha } } ; t _ { a } ) \leqslant ( \sqrt { \delta _ { a } } + \delta _ { n } ) ^ { 2 } + \delta _ { N }$

2. Unfairness. $| \gamma _ { a } ( \widehat { P } ^ { \widehat { \alpha } } ) | \leqslant ( \sqrt { \delta _ { a } } + \delta _ { n } + \sqrt { \delta _ { N } } ) \Big / \sigma \equiv \delta$

3. Uniform deployment. For all $\xi ^ { \prime } \in [ \xi , 1 - \xi ]$ , simultaneously,

$$
| \Phi _ { a } ( \widehat { P } ^ { \alpha } ; \xi ^ { \prime } ) - t _ { a } | \leqslant ( \sqrt { \delta _ { a } } + \delta _ { n } + \sqrt { \delta _ { N } } ) ( 1 + | \xi ^ { \prime } - \mu \Xi | / \sigma _ { \Xi } ) .
$$

Moreover, with “high probabil $; t y ^ { , , }$ over $( \mathcal { D } ^ { n ^ { \prime } } , \mathcal { D } ^ { n } , \Xi ^ { N } )$ $( \widehat { \mathcal { P } } )$ is feasible and, for some constant $0 \leqslant C < \infty$ every solution αb satisfies

$$
\begin{array} { r l } { \mathcal { U } . } & { U t i l i t y . ~ \mathcal { U } ( P ^ { \alpha ^ { \star } } ) - \mathcal { U } ( \widehat { P } ^ { \widehat { \alpha } } ) \leqslant 2 \delta _ { n } ( \mathcal { U } ) + \delta _ { n ^ { \prime } } \big / _ { \underline { { v } } } } \\ & { \quad \ + \ C \Big ( 2 \operatorname* { m a x } _ { a \in \mathcal { A } } \sqrt { \delta _ { a } } ( \delta _ { n } + \delta _ { n ^ { \prime } } ^ { \prime } ) + \delta _ { N } \Big ) . } \end{array}
$$

All rates are within $\widetilde { \mathcal { O } } ( n ^ { - \frac { 1 } { 2 } } + N ^ { - \frac { 3 } { 4 } } + \delta _ { n ^ { \prime } } )$ , and the stability and utility bounds enjoy the faster $N ^ { - \frac { 3 } { 2 } }$ rates. The number of policies L and of categories $| { \cal A } |$ only enter $\delta _ { n }$ and $\delta _ { n } ( \mathcal { U } )$ through log L and log |A|, which are absorbed in the Oe, so that enlarging the dictionary or refining the categories does not afect the convergence rates. Formal statements and proofs are presented in Appendix C.

## 6 EXPERIMENTS

Finally, we illustrate our framework on synthetic and real datasets.

## 6.1 Synthetic Data

Setup We simulate a simplified job platform with 8000 users, and a catalogue of 300 ofers divided into three categories: managerial (A), administrative (B) and other (C). The true interests of users do not depend on their group, so an oracle policy has no gap in any category. The policies are however learned from biased historical interactions: with probability $\beta ,$ an interest of a woman in a managerial ofer, or of a man in an administrative ofer, was not recorded. The constraints are estimated, and the policies evaluated, on held-out users labelled with their true interests. To illustrate the deployment guarantee, we also evaluate them on new users drawn with a share of women between 0.1 and 0.9. The results, deferred to Appendix D.2, are in line with Theorem 1. The data generation is detailed in Appendix D.2.

Policies and constraints The dictionary contains the uniform policy, an unweighted LambdaMART ranker $P _ { 0 } .$ , and five LambdaMART rankers trained with the weights $w _ { \xi } , \xi \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ Rankers are trained with LightGBM on PureSVD em beddings of the history, and the sensitive attribute is never used as a feature. Policies rank deterministically, and the utility is the NDCG@20 (Remark 1).

![](images/7f0d2a8b4f8f83d86b5b609947533aad5b67234f2bf522cf023d8c60c9570ee5.jpg)  
Figure 1: Synthetic data. (a, b) Utility gaps (women − men) under UP and CUP with free targets. (c) Utility. (d) CUP with targets $t _ { a } = \kappa \Phi _ { a } ( P _ { 0 } ; \mu _ { \Xi } )$ at $\beta = 0 . 5$ , relative to $P _ { 0 }$ . Means and standard deviations over 3 seeds.

We solve $( \widehat { \mathcal { P } } )$ with Ξ uniform on [0.1, 0.9]. The budgets $\delta _ { a }$ are set so that the tolerated gap in each category is λ times the estimation noise of the gap of $P _ { 0 } ,$ with $\lambda = 1$ unless stated otherwise. We first leave the targets free, which amounts to constraining the gaps only, and then set each target to a fraction κ of the level of $P _ { 0 } , \ t _ { a } = \kappa \Phi _ { a } ( P _ { 0 } ; \mu _ { \Xi } )$ . More details are available in Appendix D.1.

Results. Figure 1 reports, on the test users, the gaps and the utility of the mixture selected under each constraint. Under UP (a), the aggregate gap remains close to zero for every $\beta ,$ so the constraint is never active and the selected mixture reduces to the ranker with the highest estimated utility. Meanwhile, women are increasingly under-served on managerial ofers and men on administrative ones. UP is therefore blind to disparities that cancel out across categories. Under CUP (b), the gaps remain close to zero in every category. We also analyse the price of this correction in (c) and note that with this dictionary, removing the category gaps requires moving towards uniform rankings, which lowers utility for both groups. Finally, we set the targets $( \beta = 0 . 5 , \lambda = 5 )$ . For $\kappa \geqslant 0 . 9$ , the targets forbid lowering the utility of any category, and since this dictionary cannot reduce the gaps otherwise, the selected policy remains a single ranker, $P _ { 0 }$ on two seeds out of three. Lowering κ lets the utility decrease roughly in proportion, while the largest gap shrinks faster. The targets thus make the loss of utility explicit and leave its extent to the practitioner.

## 6.2 MovieLens-1M

Set-up and baselines We use MovieLens-1M with gender as the sensitive attribute (G = 1 for women). Item categories are the six movie genres (Sci-Fi, Adventure, Crime, Romance, Children’s, Horror), and we keep the movies that belong to exactly one of them. The interactions of each user are sorted by time: the first 80% form the history x and the remaining 20% are the relevant items. The candidates of a user are all movies outside their history. Users are split at random into training (50%), estimation (25%), on which the mixture weights are fitted, and test (25%) users, and results are averaged over 10 random splits. Statistics are given in Appendix D.3. We compare with ITFR (Wang et al., 2024) and CatFAIR (Kheya et al., 2025), which, like CUP, consider disparities between (user group, item category) pairs: ITFR through a recallbased measure of relevant items, CatFAIR through the share of each category in the recommendation lists, regardless of relevance. Both modify the training of a matrix factorization model with the BPR loss (Rendle et al., 2009), see the same interactions as the rankers, and use G during training only. Each is compared with its own backbone trained without the fairness compo nent (MF-BPR for ITFR, α = 0 for CatFAIR), as $P _ { 0 }$ is for our approach.

The dictionary of Section 6.1 is enriched with several rankers to improve feasibility. In the end, we consider $L = 1 9$ , detailed in Table 2 (Appendix D.3)). Targets are $t _ { a } = \kappa \Phi _ { a } ( P _ { 0 } ; \mu _ { \Xi } )$ , and κ varies from 1 to 0.8 to trace the trade-of between utility and gaps, with $\lambda =$ 2, the smallest budget of our grid for which the targets with $\kappa \leq 0 . 9$ can be met on every split. CUP without target uses $\lambda \in \{ 1 . 5 , 2 , 3 \}$

On the test users, we report the utility, the largest category gap $\operatorname* { m a x } _ { a } \left| \gamma _ { a } \right|$ , and the worst-case utility loss $\begin{array} { r } { 1 - \operatorname* { m i n } _ { a } \operatorname* { m i n } _ { \xi } \Phi _ { a } ( P ; \xi ) / \operatorname* { m i n } _ { \xi } \Phi _ { a } ( \mathrm { B a c k b o n e } ; \xi ) . } \end{array}$ , which measures how much the less-served population loses in the most afected category.

Utility and fairness. We see from Figure 2 that CUP, ITFR and CatFAIR all reduce the gaps between women and men across genres with respect to their backbones, which start from diferent levels of utility, $P _ { 0 }$ being the most accurate. The three approaches differ in how the trade-of is controlled. First, ITFR gives a single operating point, as none of its hyperparameters sets the trade-of (Wang et al., 2024, App. A.4).

![](images/98b523f27767ed9bbfcbc09d5ed7e4de5e228ccc1ca49f55d64a9899520e6664.jpg)

![](images/d462fcb964f7fd8ec3228661d49be4a8f96664b9cf2a8f79acc1899dc3e5027e.jpg)

![](images/4b295848a1c90952b058feb1eb3b8a30a8741aaccb471ad30e3c17c12e9bb4e3.jpg)  
Figure 2: MovieLens-1M. Each policy is compared with its backbone B. Utility against the largest gap between women and men over genres (a), and against the worst-case utility loss $1 - \operatorname* { m i n } _ { a }$ min<sub>ξ</sub> $\Phi _ { a } ( P ; \xi ) /$ min<sub>ξ</sub> $\Phi _ { a } ( B ; \xi )$ (b), both relative to $B ;$ the best policies lie in the upper left corner. UP and CUP vary the target level κ from 1 (top) to 0.8 (bottom); hollow squares are values feasible on part of the splits only $( 6 / 1 0$ for $\kappa = 1$ , 9/10 for $\kappa = 0 . 9 5 )$ . CUP without target varies the budget λ, and CatFAIR the fairness regularisation parameter. (c) Per genre, for CUP $( \kappa = 0 . 9 )$ and the other baselines next to their backbones: gap between women and men divided by the utility of the genre under B (top), and change of this utility with respect to B (bottom). Means and standard errors over 10 splits.

CatFAIR reaches its smallest gap with the smallest weight α, and larger values only lower utility (a). CUP instead traces a front: lowering κ reduces the largest gap at the cost of utility, and any point of this front is obtained in milliseconds from the same dictionary. Second, the targets prevent levelling down (b). Without targets, CUP reaches the same utility and largest gap, but the less-served group loses much more utility in the most afected genre. So does CatFAIR, which ignores relevance, with the weight used by its authors.

Now turning to the question of which movie genres are afected, we see from Figure 2(c) that the three methods reduce the gaps in diferent ways. CUP narrows the gap of $P _ { 0 }$ in every genre and changes the utility of each genre by less than 10%. ITFR moves utility across genres, from −21% for Sci-Fi to +51% for Horror, where the gap widens. CatFAIR closes the gaps in Romance and Children’s, but by lowering the utility of women in these genres, which is halved in Romance.

Choosing the trade-of in practice. For a practitioner, the level of fairness is a choice that must be made after seeing its cost in utility, and possibly revised: an explicit front is therefore as important as the quality of a single operating point. Our guarantees are what make such a front reliable: the budgets and targets set the gaps and utility levels directly, and Theorem 1 ensures that these settings carry over to new users, whereas ITFR and CatFAIR control fairness only through training hyperparameters, with no guarantee on the resulting gaps. With CUP, the dictionary is trained once (about 2 min per split), and each point of the front, set by interpretable targets, is a second-order cone program solved in less than 10 ms. With ITFR and CatFAIR, each point requires a full training (about 38 and 2 min on the same CPU), and finding the trade-of means cross-validating hyperparameters whose efects on fairness are not monotone.

## 7 CONCLUSION

We studied user-side category-aware fair ranking through the lens of distributional stability. Representing ranking policies through exposure, we showed that utility parity and its category-aware refinement can be viewed as the stability of utility under changes in the composition of the population, and that targets fix the level of service that each category keeps, whatever this composition. Considering mixtures of policies leads to a second-order cone program, for which we provided finite-sample guarantees on stability, fairness, deployment and utility. Since it only combines existing ranking policies, the approach is model-agnostic: it can be applied on top of the most accurate ranker available, and the trade-of between utility and fairness can be set, and revised, after training at negligible cost. Our experiments showed that aggregate parity can hide opposite disparities across categories, which conditional utility parity removes, illustrated the deployment guarantee on new populations, and showed that the targets give an explicit control of the tradeof while limiting levelling down. Our guarantees cover changes in the share of each group; changes in the behaviour of each group are left for future work.

## AI use statement

In this work, we used generative AI tools to provide feedback on research and assist with translation. We also use it for the experiment section, notably to clean and reformat the real-world dataset and to produce figures from the result files obtained. We have not used generative AI tools to generate synthetic data sets, help develop theoretical models or conceptual frameworks, formulate mathematical claims, provide critical ingredients for proving mathematical claims, assist in the writing of proofs, propose or refine hypotheses, implement methods, support qualitative and thematic data analysis, interpret results.

We have reviewed all AI-assisted work. For the code, we checked the correctness of the pre-processing code by cross-comparison with statistics reported by published papers that rely on the same dataset. For the figure, we manually checked that the numbers displayed are coherent with the results from the files. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## References

Chakraborty, D., Das, H., Dey, S., and Yan, A. H. Y. (2025). Improved rank aggregation under fairness constraint. In Kwok, J., editor, Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI-25, pages 330–338. IJ-CAI Organization. Main Track.

Chakraborty, D., Das, S., Khan, A., and Subramanian, A. (2022). Fair rank aggregation. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A., editors, Advances in Neural Information Processing Systems, volume 35, pages 23965–23978. Curran Associates, Inc.

Chakraborty, D., Mazumdar, A., Saha, B., and Yan, A. H. Y. (2026). Fairness in aggregation: Optimal top-\$k\$ and improved full ranking. In Forty-third International Conference on Machine Learning.

Cl´emen¸con, S., Korba, A., and Sibony, E. (2018). Ranking median regression: Learning to order through local consensus. In Janoos, F., Mohri, M., and Sridharan, K., editors, Proceedings of Algorithmic Learning Theory, volume 83 of Proceedings of Machine Learning Research, pages 212–245. PMLR.

Dinh, M. H., Kotary, J., and Fioretto, F. (2024). Learning fair ranking policies via diferentiable optimization of ordered weighted averages. In Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’24, page 2508–2517, New York, NY, USA. Association for Computing Machinery.

Do, V., Corbett-Davies, S., Atif, J., and Usunier, N. (2022). Online certification of preference-based fairness for personalized recommender systems. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 6532–6540.

Guo, R., Ton, J.-F., Liu, Y., and Li, H. (2024). Inference-time stochastic ranking with risk control. arXiv preprint arXiv:2306.07188.

Imana, B., Korolova, A., and Heidemann, J. (2021). Auditing for discrimination in algorithms delivering job ads. In Proceedings of the Web Conference 2021, WWW ’21, page 3767–3778, New York, NY, USA. Association for Computing Machinery.

Kearns, M., Neel, S., Roth, A., and Wu, Z. S. (2018). Preventing fairness gerrymandering: Auditing and learning for subgroup fairness. In Dy, J. and Krause, A., editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 2564– 2572. PMLR.

Kheya, T. A., Bouadjenek, M. R., and Aryal, S. (2025). Unmasking gender bias in recommendation systems and enhancing category-aware fairness. In Proceedings of the ACM on Web Conference 2025, WWW ’25, page 5127–5138, New York, NY, USA. Association for Computing Machinery.

Korba, A., Clemencon, S., and Sibony, E. (2017). A Learning Theory of Ranking Aggregation. In Singh, A. and Zhu, J., editors, Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, volume 54 of Proceedings of Machine Learning Research, pages 1001–1010. PMLR.

Kotary, J., Fioretto, F., Van Hentenryck, P., and Zhu, Z. (2022). End-to-end learning for fair ranking systems. In Proceedings of the ACM Web Conference 2022, WWW ’22, page 3520–3530, New York, NY, USA. Association for Computing Machinery.

Li, Y., Chen, H., Fu, Z., Ge, Y., and Zhang, Y. (2021). User-oriented fairness in recommendation. In Proceedings of the Web Conference 2021, WWW ’21, page 624–632, New York, NY, USA. Association for Computing Machinery.

Oosterhuis, H. (2021). Computationally eficient optimization of plackett-luce ranking models for relevance and fairness. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’21, page 1023–1032, New York, NY, USA. Association for Computing Machinery.

Plackett, R. L. (1975). The analysis of permutations. Applied Statistics, 24(2):193–202.

Rendle, S., Freudenthaler, C., Gantner, Z., and Schmidt-Thieme, L. (2009). Bpr: Bayesian person-

alized ranking from implicit feedback. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, UAI ’09, page 452–461, Arlington, Virginia, USA. AUAI Press.

Singh, A. and Joachims, T. (2018). Fairness of exposure in rankings. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’18, page 2219–2228, New York, NY, USA. Association for Computing Machinery.

Taturyan, G., Laclau, C., and Cl´emencon, S. (2026). A comprehensive view of fairness through distributional stability. arXiv preprint arXiv:2609.37061.

Vogel, R. and Cl´emen¸con, S. (2020). A multiclass classification approach to label ranking. In Chiappa, S. and Calandra, R., editors, Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pages 1421– 1430. PMLR.

Wang, L. and Joachims, T. (2021). User fairness, item fairness, and diversity for rankings in two-sided markets. In Proceedings of the 2021 ACM SIGIR International Conference on Theory of Information Retrieval, ICTIR ’21, page 23–41, New York, NY, USA. Association for Computing Machinery.

Wang, Y., Sun, P., Ma, W., Zhang, M., Zhang, Y., Jiang, P., and Ma, S. (2024). Intersectional twosided fairness in recommendation. In Proceedings of the ACM Web Conference 2024, WWW ’24, page 3609–3620, New York, NY, USA. Association for Computing Machinery.

Yao, S. and Huang, B. (2017). Beyond parity: fairness objectives for collaborative filtering. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, page 2925–2934, Red Hook, NY, USA. Curran Associates Inc.

## CHECKLIST

1. For all models and algorithms presented, check if you include:

(a) A clear description of the mathematical setting, assumptions, algorithm, and/or model. [Yes]

(b) An analysis of the properties and complexity (time, space, sample size) of any algorithm. [Yes]

(c) (Optional) Anonymized source code, with specification of all dependencies, including external libraries. [Yes]

2. For any theoretical claim, check if you include:

(a) Statements of the full set of assumptions of all theoretical results. [Yes]

(b) Complete proofs of all theoretical results. [Yes]

(c) Clear explanations of any assumptions. [Yes]

3. For all figures and tables that present empirical results, check if you include:

(a) The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL). [Yes]

(b) All the training details (e.g., data splits, hyperparameters, how they were chosen). [Yes]

(c) A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times). [Yes]

(d) A description of the computing infrastructure used. (e.g., type of GPUs, internal cluster, or cloud provider). [Yes]

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include:

(a) Citations of the creator If your work uses existing assets. [Yes]

(b) The license information of the assets, if applicable. [Not Applicable]

(c) New assets either in the supplemental material or as a URL, if applicable. [Yes]

(d) Information about consent from data providers/curators. [Not Applicable]

(e) Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. [Not Applicable]

5. If you used crowdsourcing or conducted research with human subjects, check if you include:

(a) The full text of instructions given to participants and screenshots. [Not Applicable]

(b) Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. [Not Applicable]

(c) The estimated hourly wage paid to participants and the total amount spent on partic ipant compensation. [Not Applicable]

# Supplementary Material

## ORGANIZATION

The Supplementary Material is organized as follows.

• In Appendix A, we discuss the characterization of feasiblity.

• Appendix B collects small proofs for Section 3.

• Appendix C provides the proofs for Section 5 regarding the convergence rates of the constraints and the utility, as well as provides rates for a Plackett-Luce policy.

• Appendix D provides additional experimental details.

## A FEASIBILITY

It follows from Lemma 1 that $\begin{array} { r } { \gamma _ { a } ( P ^ { \alpha } ) = \sum _ { \ell \in \mathsf { I } L \vert } \alpha _ { \ell } \gamma _ { a } ( P _ { \ell } ) } \end{array}$ and $\gamma _ { a } ( P ^ { \alpha } )$ is in convex hull of $\{ \gamma _ { a } ( P _ { \ell } ) \} _ { \ell }$ . Therefore, if all of them have the same sign, so will $\gamma _ { a } ( P ^ { \alpha } )$ , and it will hold that $\begin{array} { r } { { \mathcal C } _ { \Xi } ^ { a } ( P ^ { \alpha } ; t _ { a } ) \ \geqslant \ \sigma _ { \Xi } ^ { 2 } \operatorname* { m i n } _ { \ell } \gamma _ { a } ^ { 2 } ( P _ { \ell } ) } \end{array}$ . On the contrary, if $\exists P _ { \ell } , P _ { \ell ^ { \prime } } \ s . t . \ \gamma _ { a } ( P _ { \ell } ) \ < \ 0 \ < \ \gamma _ { a } ( P _ { \ell ^ { \prime } } )$ , then zero unfairness is attainable, and picking a weight $\frac { \left| \gamma _ { a } ( P _ { \ell ^ { \prime } } ) \right| } { \left| \gamma _ { a } ( P _ { \ell } ) | + | \gamma _ { a } ( P _ { \ell ^ { \prime } } ) | \right| }$ for $P _ { \ell }$ and the complement of it for $P _ { \ell ^ { \prime } }$ will yield $\mathcal { C } _ { \Xi } ^ { a } ( P ^ { \alpha } ; t _ { a } ) \leqslant ( \mathbb { E } _ { \Xi } [ \Phi _ { a } ( P ^ { \alpha } ; \xi ) ] - t _ { a } ) ^ { 2 }$ , which is zero for $t _ { a } = \mathbb { E } _ { \Xi } [ \Phi _ { a } ( P ^ { \alpha } ; \xi ) ]$ . Whether such pairs exist depends on the dictionary, which should contain, for each category, policies whose gaps have opposite signs. Such policies can be obtained by diversifying how the dictionary is trained, for instance by training under the importance weights $w _ { \xi }$ with $\xi$ close to $\xi$ and $1 - \xi .$ . When disparities <sup>¯ ¯</sup>originate in the observed relevance rather than in the group proportions, this may not sufice, and one may additionally emphasize, during training, the interactions of a given group with a given category. Overall, for a fixed $t _ { a } .$ one might ensure that $\begin{array} { r } { \delta _ { a } \geqslant \operatorname* { m i n } _ { { \pmb { \alpha } } \in \Delta _ { L } } \mathcal { C } _ { \Xi } ^ { a } ( P ^ { { \pmb { \alpha } } } ; t _ { a } ) , \forall a \in \mathcal { A } } \end{array}$ , where the minimization is a quadratic program. This condition is suficient for UP, and for the CUP it must also hold that min $\mathbf { \delta } ^ { 1 } \alpha \in \Delta _ { \mathrm { j } }$ L $\begin{array} { r } { \operatorname* { m a x } _ { a \in \mathcal { A } } \frac { \mathcal { C } _ { \Xi } ^ { a } ( P ^ { \alpha } ; t _ { a } ) } { \delta _ { a } } \leqslant 1 } \end{array}$ . In practice, these conditions can be checked.

## B PROOFS FOR SECTION 3

Proof of Proposition 1.

Proposition (Parity and stability). Let P be a ranking policy. Then

1. P satisfies UP if and only $i f \xi \mapsto \Phi ( P ; \xi )$ is constant on [0, 1];

2. P satisfies CUP if and only $i f \xi \mapsto \Phi ^ { a } ( P ; \xi )$ is constant on [0, 1] for every $a \in [ A ]$

Moreover, $f o r \Phi ^ { \square } \in \{ \Phi , \Phi ^ { a } \}$ , one has the explicit identity

$$
\Phi ^ { \checkmark } ( P ; \xi ) = \mathcal { U } _ { 0 } ^ { \bigcirc } ( P ) + \xi \gamma ^ { \bigcirc } ( P ) , \qquad \gamma ^ { \bigcirc } ( P ) \overset { \mathrm { d e f } } { = } \mathcal { U } _ { 1 } ^ { \bigcirc } ( P ) - \mathcal { U } _ { 0 } ^ { \bigcirc } ( P ) ,
$$

so that

$$
\operatorname* { s u p } _ { \xi , \xi ^ { \prime } \in [ 0 , 1 ] } \big | \Phi ^ { \boxminus } ( P ; \xi ) - \Phi ^ { \boxplus } ( P ; \xi ^ { \prime } ) \big | = \big | \gamma ^ { \boxplus } ( P ) \big | .
$$

Proof. Fix P and $\Phi ^ { \perp } \in \{ \Phi , \Phi ^ { a } \}$ , with the corresponding $\mathcal { U } ^ { \perp } \in \{ \mathcal { U } , \mathcal { U } ^ { a } \}$ , and write $\mathcal { U } _ { g } ^ { \perp } ( P ) = \mathbb { E } [ \mathcal { U } ^ { \perp } ( P ; Z ) \mid G = g ]$ for $g \in \{ 0 , 1 \}$ . For every $\xi \in [ 0 , 1 ]$ , it holds that

$$
\Phi ^ { \sharp } ( P ; \xi ) = \mathbb { E } _ { \mathcal { D } _ { \xi } } \big [ \mathcal { U } ^ { \sharp } ( P ; Z ) \big ] = \xi \mathcal { U } _ { 1 } ^ { \sharp } ( P ) + ( 1 - \xi ) \mathcal { U } _ { 0 } ^ { \sharp } ( P ) = \mathcal { U } _ { 0 } ^ { \sharp } ( P ) + \xi \gamma ^ { \sharp } ( P ) .
$$

(⇒) First, let us show that parity implies stability. If P satisfies UP, then $\gamma ( P ) = 0$ , and $\Phi ( P ; \xi ) = \mathcal { U } _ { 0 } ( P )$ for every $\xi \in [ 0 , 1 ]$ , which does not depend on ξ. Similarly, if P satisfies CUP, then $\gamma _ { a } ( P ) = 0$ for every $a \in [ A ]$ , and each map $\dot { \xi } \mapsto \bar { \Phi } ^ { a } ( P ; \xi ) = \mathcal { U } _ { 0 } ^ { a } ( P )$ is constant on [0, 1].

(⇐) Now, suppose that $\xi \mapsto \Phi ^ { \sharp } ( P ; \xi )$ is constant on [0, 1]. Then, for all $\xi , \xi ^ { \prime } \in [ 0 , 1 ]$

$$
0 = \Phi ^ { \perp } ( P ; \xi ) - \Phi ^ { \perp } ( P ; \xi ^ { \prime } ) = ( \xi - \xi ^ { \prime } ) \gamma ^ { \perp } ( P ) .
$$

Since we may choose $\xi \neq \xi ^ { \prime }$ , it follows that $\gamma ^ { \sqcup } ( P ) = 0 .$ , that is, $\mathcal { U } _ { 0 } ^ { \bigstar \bigstar } ( P ) = \mathcal { U } _ { 1 } ^ { \bigstar } ( P ) \ ( \mathrm { U P } )$ . If the maps $\xi \mapsto \Phi ^ { a } ( P ; \xi )$ are constant for every $a \in [ A ]$ , we obtain $\mathcal { U } _ { 0 } ^ { a } ( P ) = \mathcal { U } _ { 1 } ^ { a } ( P )$ for every $a \in [ A ] ~ ( \mathrm { C U P } )$

The equivalences are established. Finally, for all $\xi , \xi ^ { \prime } \in [ 0 , 1 ]$ , the identity above yields

$$
\big | \Phi ^ { \flat } ( P ; \xi ) - \Phi ^ { \flat } ( P ; \xi ^ { \prime } ) \big | = | \xi - \xi ^ { \prime } | \big | \gamma ^ { \flat } ( P ) \big | \leqslant \big | \gamma ^ { \flat } ( P ) \big | ,
$$

with equality for $( \xi , \xi ^ { \prime } ) = ( 1 , 0 )$ . Therefore,

$$
\operatorname* { s u p } _ { \xi , \xi ^ { \prime } \in [ 0 , 1 ] } \big | \Phi ^ { \boxdot \mathrm { Z } } ( P ; \xi ) - \Phi ^ { \boxdot \mathrm { Z } } ( P ; \xi ^ { \prime } ) \big | = \big | \gamma ^ { \boxdot \mathrm { Z } } ( P ) \big | = \big | \mathcal { U } _ { 1 } ^ { \boxdot \mathrm { Z } } ( P ) - \mathcal { U } _ { 0 } ^ { \boxdot \mathrm { Z } } ( P ) \big | ,
$$

and the map $\xi \mapsto \Phi ^ { \sharp } ( P ; \xi )$ is Lipschitz with constant $| \gamma ^ { \bigtriangledown } ( P ) |$

The proof is concluded.

Proof of Lemma 1.

Lemma (Policy linearity). For all $\pmb { \alpha } \in \Delta _ { L }$ , all $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ and $a l l z \in { \mathcal { Z } }$ , it holds that

$$
\begin{array} { r l } & { { \mathbf { M } ^ { P ^ { \alpha } } ( { \pmb x } ) } = \displaystyle \sum _ { \ell \in [ L ] } \alpha _ { \ell } { \mathbf { M } ^ { P _ { \ell } } } , \rho ( P ^ { \alpha } ; { \pmb x } ) = \displaystyle \sum _ { \ell \in [ L ] } \alpha _ { \ell } \rho ( P _ { \ell } ; { \pmb x } ) , } \\ & { \overline { { U } } _ { \perp } ( P ^ { \alpha } ; { z } ) = \displaystyle \sum _ { \ell \in [ L ] } \alpha _ { \ell } \overline { { U } } _ { \perp } ( P _ { \ell } ; { z } ) , w i t h \overline { { U } } _ { \perp } \in \{ \overline { { U } } ; \overline { { U } } _ { a } \} . } \end{array}
$$

Proof. The proof follows from linearity of the functional $\begin{array} { r } { \mathbf { M } _ { i j } ^ { P } ( \pmb { x } ) = \int \mathbb { I } \{ \pi ( j ) = i \} P ( \mathrm { d } \pi \mid \pmb { x } ) } \end{array}$ of the measure $P ( \cdot \mid x )$ ; from linearity of $v \mapsto \mathbf { M } v$ , of the inner product, and of E. □

## C PROOFS FOR SECTION 5

First, let us recall some notations and introduce additional ones. We have $\mathcal { T } _ { 0 } \equiv \mathcal { T } , \overline { { U } } _ { 0 } \equiv \overline { { U } }$ and $\Phi _ { 0 } \equiv \Phi$ , and we treat UP $\left( \mathcal { A } = \{ 0 \} \right)$ and CUP $( { \mathcal { A } } = [ A ] )$ simultaneously. We also denote

$$
\begin{array} { r } { \beta _ { \mathfrak { a } } ( P ) \overset { \scriptscriptstyle { \mathrm { d e f } } } { = } \mathbb { E } \big [ \overline { { U } } _ { \mathfrak { a } } ( P ; Z ) \mid G = 0 \big ] \quad \mathrm { a n d } \quad \gamma _ { \mathfrak { a } } ( P ) \overset { \scriptscriptstyle { \mathrm { d e f } } } { = } \mathbb { E } \big [ \overline { { U } } _ { \mathfrak { a } } ( P ; Z ) \mid G = 1 \big ] - \mathbb { E } \big [ \overline { { U } } _ { \mathfrak { a } } ( P ; Z ) \mid G = 0 \big ] , } \end{array}
$$

so that $\Phi _ { a } ( P ; \xi ) = \beta _ { a } ( P ) + \xi \gamma _ { a } ( P )$

## C.1 Auxiliary lemmas

In this section we present several auxiliary lemmas that we will use throughout the proofs.

Lemma 3 (Hoefding’s lemma). Let X be any real-valued random variable s.t. $a \leqslant X \leqslant b$ almost surely. Then, for all $\lambda \in \mathbb { R }$

$$
\mathbb { E } [ e ^ { \lambda ( X - \mathbb { E } [ X ] ) } ] \leqslant e ^ { \frac { \lambda ^ { 2 } ( b - a ) ^ { 2 } } { 8 } } .
$$

Lemma 4 (Hoefding’s inequality). Let $X _ { 1 } , \cdots , X _ { N }$ be independent random variables such that $X _ { i } \in [ a _ { i } , b _ { i } ]$ for every i. Then, for any $t > 0$ , we have

$$
\mathbb { P } \Big ( \sum _ { i = 1 } ^ { N } ( X _ { i } - \mathbb { E } [ X _ { i } ] ) \geqslant t \Big ) \leqslant e ^ { - \frac { 2 t ^ { 2 } } { \sum _ { i = 1 } ^ { N } ( b _ { i } - a _ { i } ) ^ { 2 } } } .
$$

Lemma 5 (Pinsker’s inequality). Let P and $Q$ be two probability distributions on a measurable space $( \mathcal { X } , \Sigma )$ Then it holds that

$$
T V ( P , Q ) \leqslant { \sqrt { { \frac { 1 } { 2 } } \operatorname { K L } ( P \| Q ) } } .
$$

Lemma 6 (Stratified sampling). (Taturyan et al., 2026, Lemma 6) Let $\xi _ { 1 } , \cdots , \xi _ { N }$ be stratified samples $f r o m \Xi ,$ and let $f : [ \xi , 1 - \xi ] \to \mathbb { R }$ be a L–Lipschitz function. For all $\eta \in ( 0 , 1 )$ with probability greater than $1 - \eta$ , it holds that

$$
\Big | \frac { 1 } { N } \sum _ { i = 1 } ^ { N } f ( \xi _ { i } ) - \mathbb { E } _ { \Xi } [ f ( \xi ) ] \Big | \leqslant \frac { L ( 1 - 2 \xi ) } { N ^ { 3 / 2 } } \sqrt { 2 \log ( ^ { 2 } / \eta ) } .
$$

## C.2 Example: Plackett-Luce exposure rate

We discuss the specific example of Plackett-Luce policies and derive its exposure rates.

Proof of Lemma 2.

Lemma (Plackett-Luce rates). Let $P _ { \ell }$ and $\widehat { P } _ { \ell }$ be Plackett-Luce policies with score functions $s _ { \ell }$ and $\widehat { s } _ { \ell } , f o r \ell \in [ L ]$ For every $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ , it holds that

$$
\begin{array} { r } { \| \rho ( P _ { \ell } ; \pmb { x } ) - \rho ( \widehat { P } _ { \ell } ; \pmb { x } ) \| _ { 1 } \leqslant \sqrt { m - 1 } \| s _ { \ell } ( \pmb { x } , \cdot ) - \widehat { s } _ { \ell } ( \pmb { x } , \cdot ) \| _ { \infty } . } \end{array}
$$

Proof. We have that $\begin{array} { r } { \rho _ { i } ( P ; \pmb { x } ) = \sum _ { \pi } P ( \pi ) v _ { \pi ^ { - 1 } ( i ) } } \end{array}$ . By the fact that $\| \pmb { v } \| _ { 1 } = 1$ and by Pinsker’s inequality (Lemma 5) we have

$$
\| \rho ( P _ { \ell } ; x ) - \rho ( \widehat { P } _ { \ell } ; x ) \| _ { 1 } \leqslant \sum _ { \pi } | P ( \pi ) - \widehat { P } ( \pi ) | \sum _ { i } v _ { \pi ^ { - 1 } ( i ) } = 2 \mathrm { T V } ( P ; \widehat { P } ) \leqslant \sqrt { 2 \mathrm { K L } ( P \| \widehat { P } ) } .
$$

From the definition of Plackett-Luce policies, we have

$$
\frac { P ( \pi ) } { \widehat { P } ( \pi ) } = \prod _ { j = 1 } ^ { m } \frac { e ^ { s ( \pi ( j ) ) } \sum _ { i = j } ^ { m } e ^ { \widehat { s } ( \pi ( i ) ) } } { e ^ { \widehat { s } ( \pi ( j ) ) } \sum _ { i = j } ^ { m } e ^ { s ( \pi ( i ) ) } } = \prod _ { j = 1 } ^ { m } e ^ { ( s - \widehat { s } ) ( \pi ( j ) ) } \frac { \sum _ { i = j } ^ { m } e ^ { \widehat { s } ( \pi ( i ) ) } } { \sum _ { i = j } ^ { m } e ^ { s ( \pi ( i ) ) } } .
$$

Taking the logarithm yields

$$
\log \frac { P ( \pi ) } { \widehat { P } ( \pi ) } = \sum _ { j = 1 } ^ { m } \underbrace { \left[ \left( s - \widehat { s } \right) ( \pi ( j ) ) + \log \frac { \sum _ { i = j } ^ { m } e ^ { \widehat { s } ( \pi ( i ) ) } } { \sum _ { i = j } ^ { m } e ^ { s ( \pi ( i ) ) } } \right] } _ { \mathrm { d e f } } ,
$$

and taking the expectation over $\pi \sim P$ yields $\begin{array} { r } { \mathrm { K L } ( P \| \widehat { P } ) = \sum _ { j = 1 } ^ { m } \mathbb { E } _ { \pi \sim P } \left[ S _ { j } \right] } \end{array}$ . Now, the sums $\scriptstyle \sum _ { i = j } ^ { m }$ depend on π only through $\pi ( 1 ) , \cdot \cdot \cdot , \pi ( j - 1 )$ , hence, for any function $f ( \cdot )$ , we have that

$$
\mathbb { E } _ { P } [ f ( \pi ( j ) ) | \pi ( 1 ) , \cdots , \pi ( j - 1 ) ] = \frac { \sum _ { i = j } ^ { m } ( e ^ { s ( \pi ( i ) ) } f ( \pi ( i ) ) ) } { \sum _ { i = j } ^ { m } e ^ { s ( \pi ( i ) ) } } .
$$

Moreover, we have

$$
\frac { \sum _ { i = j } ^ { m } e ^ { \widehat s ( \pi ( i ) ) } } { \sum _ { i = j } ^ { m } e ^ { s ( \pi ( i ) ) } } = \sum _ { i = j } ^ { m } \frac { e ^ { \widehat s ( \pi ( i ) ) } } { \sum _ { k = j } ^ { m } e ^ { s ( \pi ( k ) ) } } = \sum _ { i = j } ^ { m } \frac { e ^ { s ( \pi ( i ) ) } } { \sum _ { k = j } ^ { m } e ^ { s ( \pi ( k ) ) } } e ^ { - ( s - \widehat s ) ( \pi ( i ) ) } .
$$

Taking $f = e ^ { - ( s - \widehat { s } ) }$ , we obtain

$$
\begin{array} { r l } & { \mathbb { E } _ { P } [ S _ { j } | \pi ( 1 ) , \cdots , \pi ( j - 1 ) ] = \mathbb { E } _ { P } [ ( s ( \pi ( j ) ) - \widehat { s } ( \pi ( j ) ) | \cdot ] + \log \mathbb { E } _ { P } [ e ^ { - s ( \pi ( j ) ) - \widehat { s } ( \pi ( j ) ) } | \cdot ] } \\ & { \qquad = \log \mathbb { E } _ { P } [ e ^ { - \big ( s ( \pi ( j ) ) - \widehat { s } ( \pi ( j ) ) - \mathbb { E } [ s ( \pi ( j ) ) - \widehat { s } ( \pi ( j ) ) \mid \cdot ] \big ) } | \cdot ] . } \end{array}
$$

Noticing that $\begin{array} { r } { \operatorname* { m a x } _ { i } ( s - \widehat { s } ) - \operatorname* { m i n } _ { i } ( s - \widehat { s } ) \leqslant 2 \| s - \widehat { s } \| _ { \infty } } \end{array}$ and applying Hoefding’s lemma (Lemma 3), we get

$$
\mathbb { E } _ { P } [ S _ { j } | \pi ( 1 ) , \cdots , \pi ( j - 1 ) ] \leqslant \log e ^ { \frac { \| s - \widehat { s } \| _ { \infty } ^ { 2 } } { 2 } } = \frac { \| s - \widehat { s } \| _ { \infty } ^ { 2 } } { 2 } .
$$

Since $S _ { m } = 0$ , we obtain

$$
\mathrm { K L } ( P \| \widehat { P } ) = \sum _ { j = 1 } ^ { m - 1 } \mathbb { E } _ { \pi \sim P } \left[ S _ { j } \right] \leqslant ( m - 1 ) \frac { \| s - \widehat { s } \| _ { \infty } ^ { 2 } } { 2 } .
$$

The proof is concluded.

## C.3 Error rates by source

Before proving the main theorem, we identify the diferent sources of error and quantify their respective contributions to the overall error. Throughout, the samples $\mathcal { D } ^ { n ^ { \prime } } , \mathcal { D } ^ { n }$ and $\Xi ^ { N }$ are mutually independent. The dictionary $\widehat { P } _ { 1 } , \cdots , \widehat { P } _ { L }$ is a function of $\mathcal { D } ^ { n ^ { \prime } }$ only, while $\widehat { \pmb { T } } _ { a } .$ ub and αb are functions of $( \mathcal { D } ^ { n ^ { \prime } } , \mathcal { D } ^ { n } , \bar { \Xi } ^ { N } )$ . For a policy P depending on the data, $\Phi _ { a } ( P ; \xi )$ and $\mathcal { U } ( P )$ denote expectations over an independent copy $z \sim \mathcal { D } ,$ , conditionally on the data. Since αb depends on $( \mathcal { D } ^ { n } , \dot { \Xi } ^ { N } )$ , all the bounds involving it are established on events that do not depend on ${ \widehat { \mathbf { a } } } ,$ and hold uniformly over $\pmb { \alpha } \in \Delta _ { L }$

Overall, the convergence proofs follow the structure of Taturyan et al. (2026), with necessary modifications specific to our setup.

## C.3.1 Policy estimation rates

Lemma 7. Let Assumptions 1, 2 and 4 hold. Then, with probability greater than $1 - \eta _ { \delta _ { n ^ { \prime } } }$ <sub>′</sub> over $\mathcal { D } ^ { n ^ { \prime } }$ , for every $\alpha \in \Delta _ { L } , a \in \mathcal { A }$ and $\xi \in [ \xi , 1 - \xi ]$ , it holds that

$$
\left| \Phi _ { a } \big ( \widehat { P } ^ { \alpha } ; \xi \big ) - \Phi _ { a } \big ( P ^ { \alpha } ; \xi \big ) \right| \leqslant \frac { \delta _ { n ^ { \prime } } } { p \underline { { v } } } \overset { \mathrm { d e f } } { = } \delta _ { n ^ { \prime } } ^ { \prime } \quad a n d \quad \left| \mathcal { U } \big ( \widehat { P } ^ { \alpha } \big ) - \mathcal { U } \big ( P ^ { \alpha } \big ) \right| \leqslant \frac { \delta _ { n ^ { \prime } } } { \underline { { v } } } .
$$

Proof. For a fixed $\boldsymbol { z } = ( \boldsymbol { x } , g , y )$ , we have

$$
\left| \overline { { U } } _ { a } ( \widehat { P } ^ { \alpha } ; z ) - \overline { { U } } _ { a } ( P ^ { \alpha } ; z ) \right| = \frac { 1 } { V ( z ) } \Big | \sum _ { i \in \mathcal { I } _ { a } } \left( \rho _ { i } ( \widehat { P } ^ { \alpha } ; \boldsymbol { x } ) - \rho _ { i } ( P ^ { \alpha } ; \boldsymbol { x } ) \right) y _ { i } \Big | \leqslant \frac { 1 } { v } \left\| \rho ( \widehat { P } ^ { \alpha } ; \boldsymbol { x } ) - \rho ( P ^ { \alpha } ; \boldsymbol { x } ) \right\| _ { 1 } .
$$

By Lemma 1, the triangle inequality and the fact that $\begin{array} { r } { \alpha _ { \ell } \geqslant 0 , \sum _ { \ell } \alpha _ { \ell } = 1 } \end{array}$ , we have

$$
\mathbb { E } \left\| \rho ( \widehat { P } ^ { \alpha } ; X ) - \rho ( P ^ { \alpha } ; X ) \right\| _ { 1 } = \mathbb { E } \Big \| \sum _ { \ell \in [ L ] } \alpha _ { \ell } \big ( \rho ( \widehat { P } _ { \ell } ; X ) - \rho ( P _ { \ell } ; X ) \big ) \Big \| _ { 1 } \leqslant \operatorname* { m a x } _ { \ell \in [ L ] } \mathbb { E } \left\| \rho ( \widehat { P } _ { \ell } ; X ) - \rho ( P _ { \ell } ; X ) \right\| _ { 1 } .
$$

Finally, from Assumptions 2 and 4, we get

$$
\left| \Phi _ { \alpha } ( \widehat { P } ^ { \alpha } ; \xi ) - \Phi _ { \alpha } ( P ^ { \alpha } ; \xi ) \right| = \left| \mathbb { E } \Big [ w _ { \xi } ( G ) \big ( \overline { { U } } _ { \alpha } ( \widehat { P } ^ { \alpha } ; Z ) - \overline { { U } } _ { \alpha } ( P ^ { \alpha } ; Z ) \big ) \Big ] \right| \leqslant \frac { 1 } { p v } \mathbb { E } \left\| \rho ( \widehat { P } ^ { \alpha } ; X ) - \rho ( P ^ { \alpha } ; X ) \right\| _ { 1 } \leqslant \frac { \delta _ { n ^ { \prime } } } { p v } .
$$

The second inequality follows from the same arguments, with $w _ { \xi } \equiv 1$ and $\mathcal { T } _ { a }$ replaced by I.

The proof is concluded.

## C.3.2 Perturbation error rates

Lemma 8. For all $\eta _ { \delta _ { N } } \in ( 0 , 1 )$ ), with probability greater than $1 - \eta _ { \delta _ { N } } \ o v e r \Xi ^ { N }$ , for every $P , a \in { \mathcal { A } }$ and $t _ { a } \in [ 0 , 1 ]$ it holds that

$$
\left| \mathcal { C } _ { \Xi } ^ { a } ( P ; t _ { a } ) - \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left( \Phi _ { a } ( P ; \xi _ { k } ) - t _ { a } \right) ^ { 2 } \right| \leqslant \left( 2 + 2 ( 1 - \xi ) \right) \frac { 1 - 2 \xi } { N ^ { 3 / 2 } } \sqrt { 2 \log ( 4 / \eta _ { \delta _ { N } } ) } \stackrel { \mathrm { d e f } } { = } \delta _ { N } .
$$

Proof. For fixed $P , a \in { \mathcal { A } }$ and $t _ { a } \in [ 0 , 1 ]$ , we have

$$
\big ( \Phi _ { a } ( P ; \xi ) - t _ { a } \big ) ^ { 2 } = \big ( \beta _ { a } ( P ) - t _ { a } \big ) ^ { 2 } + 2 \big ( \beta _ { a } ( P ) - t _ { a } \big ) \gamma _ { a } ( P ) \xi + \gamma _ { a } ( P ) ^ { 2 } \xi ^ { 2 } .
$$

Therefore,

$$
\mathcal { C } _ { \Xi } ^ { \alpha } ( P ; l _ { a } ) - \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left( \Phi _ { \alpha } ( P ; \xi _ { k } ) - l _ { a } \right) ^ { 2 } = 2 \big ( \beta _ { \alpha } ( P ) - l _ { \alpha } \big ) \gamma _ { \alpha } ( P ) \Big \{ \mathbb { E } _ { \Xi } [ \xi ] - \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \xi _ { k } \Big \} + \gamma _ { \alpha } ( P ) ^ { 2 } \Big \{ \mathbb { E } _ { \Xi } [ \xi ^ { 2 } ] - \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \xi _ { k } ^ { 2 } \Big \} .
$$

We have $| \beta _ { a } ( P ) - t _ { a } | \leqslant 1$ and $| \gamma _ { a } ( P ) | \leqslant 1$ . Since the maps $\xi \mapsto \xi$ and $\xi \mapsto \xi ^ { 2 }$ are respectively 1–Lipschitz and $2 ( 1 - \xi ) \ / - \mathrm { L }$ ipschitz on $[ \xi , 1 - \xi ]$ , applying Lemma 6 to each of them with $\eta = \eta _ { \delta _ { N } } / 2$ , and a union bound, with <sup>¯ ¯</sup>probability greater than $1 - \eta _ { \delta _ { N } }$ we obtain,

$$
\left. \mathcal { C } _ { \Xi } ^ { a } ( P ; t _ { a } ) - \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left( \Phi _ { a } ( P ; \xi _ { k } ) - t _ { a } \right) ^ { 2 } \right. \leqslant \left( 2 + 2 ( 1 - \xi ) \right) \frac { 1 - 2 \xi } { N ^ { 3 / 2 } } \sqrt { 2 \log ( 4 / \eta _ { \delta _ { N } } ) } .
$$

The proof is concluded.

## C.3.3 Sample generalization rates

Lemma 9. Conditionally on $( \mathcal { D } ^ { n ^ { \prime } } , \Xi ^ { N } )$ , for all $\eta _ { \delta _ { n } } \in ( 0 , 1 )$ , with probability greater than $1 - \eta _ { \delta _ { \tau } }$ over $\mathcal { D } ^ { n }$ , it holds that

$$
\operatorname* { m a x } _ { a \in \mathcal { A } , \ell \in [ L ] , k \in [ N ] } \left| \widehat { \Phi } _ { n } ^ { a } ( \widehat { P } _ { \ell } ; \xi _ { k } ) - \Phi _ { a } ( \widehat { P } _ { \ell } ; \xi _ { k } ) \right| \leqslant \frac { 1 } { p } \sqrt { \frac { \log \left( 2 | \mathcal { A } | L N / \eta _ { \delta _ { n } } \right) } { 2 n } } \ \stackrel { \mathrm { d e f } } { = } \ \delta _ { n } .
$$

Moreover, for all $\alpha \in \Delta _ { L } , a \in \mathcal { A }$ and $k \in [ N ]$ , it holds that $| \widehat { \Phi } _ { n } ^ { a } ( \widehat { P } ^ { \alpha } ; \xi _ { k } ) - \Phi _ { a } ( \widehat { P } ^ { \alpha } ; \xi _ { k } ) | \leqslant \delta _ { n }$

Similarly, for all $\eta _ { \delta _ { n } ( \mathcal { U } ) } \in ( 0 , 1 )$ , with probability greater than $1 - \eta _ { \delta _ { n } ( \mathcal { U } ) }$ , it holds that

$$
\operatorname* { s u p } _ { \alpha \in \Delta _ { L } } \big | \widehat { \mathcal { U } } _ { n } ( \widehat { P } ^ { \alpha } ) - \mathcal { U } ( \widehat { P } ^ { \alpha } ) \big | \leqslant \sqrt { \frac { \log ^ { \left( 2 L \big / \eta _ { \delta _ { n } ( \mathcal { U } ) } \right) } } { 2 n } } \stackrel { \mathrm { d e f } } { = } \delta _ { n } ( \mathcal { U } ) .
$$

Proof. Let us fix $a \in \mathcal { A } , \ell \in [ L ]$ and $k \in [ N ]$ , and the policy $\widehat { P } _ { \ell }$ is fixed conditionally on $\mathcal { D } ^ { n ^ { \prime } }$ . Let us define $\zeta _ { i } \ { \stackrel { \mathrm { d e f } } { = } }$ $w _ { \xi _ { k } } ( G _ { i } ) \overline { { { U } } } _ { a } ( \widehat { P } _ { \ell } ; Z _ { i } )$ , for $\pmb { Z } _ { i } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { D }$ . From Assumption 2 we have $\zeta _ { i } \in [ 0 , 1 / \underline { { p } } ]$ , and it holds that $\mathbb { E } [ \zeta _ { i } ] = \Phi _ { a } ( \widehat { P } _ { \ell } ; \xi _ { k } )$ Applying Hoefding’s inequality (Lemma 4), for any $\epsilon > 0 .$ , it holds that

$$
\mathbb { P } _ { \mathcal { D } ^ { n } } \Big ( \Big | \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \zeta _ { i } - \Phi _ { a } \big ( \widehat { P } _ { \ell } ; \xi _ { k } \big ) \Big | \geqslant \epsilon \Big | \ \mathcal { D } ^ { n ^ { \prime } } , \xi _ { k } \Big ) \leqslant 2 \exp \big ( - 2 n \underline { { p } } ^ { 2 } \epsilon ^ { 2 } \big ) .
$$

Setting $\begin{array} { r } { \frac { \eta _ { \delta _ { n } } } { | \ r { A } | { L } N } = 2 \exp ( - 2 n \underline { { p } } ^ { 2 } \epsilon ^ { 2 } ) } \end{array}$ , obtaining $\begin{array} { r } { \epsilon = \frac { 1 } { p } \sqrt { \frac { \log \left( 2 \vert \mathcal { A } \vert L N / \eta _ { \delta _ { n } } \right) } { 2 n } } } \end{array}$ , and applying a union bound over all triples $( a , \ell , k ) \in \mathcal { A } \times [ L ] \times [ N ]$ <sup>¯</sup> concludes the proof of the first inequality.

Moreover, by Lemma 1, the triangle inequality and the fact that $\begin{array} { r } { \alpha _ { \ell } \geqslant 0 , \sum _ { \ell } \alpha _ { \ell } = 1 } \end{array}$ , we obtain

$$
\bigl | \widehat { \Phi } _ { n } ^ { a } ( \widehat { P } ^ { \alpha } ; \xi _ { k } ) - \Phi _ { a } ( \widehat { P } ^ { \alpha } ; \xi _ { k } ) \bigr | = \Big | \sum _ { \ell \in [ I ] } \alpha _ { \ell } \bigl ( \widehat { \Phi } _ { n } ^ { a } ( \widehat { P } _ { \ell } ; \xi _ { k } ) - \Phi _ { a } ( \widehat { P } _ { \ell } ; \xi _ { k } ) \bigr ) \Big | \leqslant \operatorname* { m a x } _ { \ell \in [ I ] } \bigl | \widehat { \Phi } _ { n } ^ { a } ( \widehat { P } _ { \ell } ; \xi _ { k } ) - \Phi _ { a } ( \widehat { P } _ { \ell } ; \xi _ { k } ) \bigr | \leqslant \delta _ { n } .
$$

Finally, for each fixed $\ell \in [ L ]$ , the random variables $\overline { { U } } ( \widehat { P } _ { \ell } ; Z _ { i } )$ are i.i.d. conditionally on $\mathcal { D } ^ { n ^ { \prime } }$ and take values in [0, 1]. Hoefding’s inequality (Lemma 4) and a union bound over $\ell \in [ L ]$ yield max $\phantom { } _ { \ell } | \widehat { \mathcal { U } } _ { n } ( \widehat { P } _ { \ell } ) - \mathcal { U } ( \widehat { P } _ { \ell } ) | \leqslant \delta _ { n } ( \mathcal { U } )$ with probability greater than $1 - \eta _ { \delta _ { n } ( \mathcal { U } ) }$ . Linearity transfers the bound to the mixtures.

The proof is concluded.

## C.4 Proof of Theorem 1

Finally, we are in position to prove the main theorem.

## C.4.1 Constraint rates: stability, unfairness, deployment

Formal version and the proof of the first statement of Theorem 1.

Theorem 2 (Stability). Let $\widehat { \pmb { \alpha } } \in \Delta _ { L }$ be a vector s.t. $\widehat { \mathcal { C } } _ { n , N } ( P ^ { \widehat { \alpha } } ; t _ { a } ) \leqslant \delta _ { a }$ for all $a \in { \mathcal { A } }$ . For every $\eta _ { \delta _ { n } } , \eta _ { \delta _ { N } } \in ( 0 , 1 )$ and conditionally on $\mathcal { D } ^ { n ^ { \prime } }$ , with probability greater than $1 - \eta _ { \delta _ { n } } - \eta _ { \delta _ { N } }$ over $( \mathcal { D } ^ { n } , \Xi ^ { N } )$ , it holds that

$$
\mathcal { C } _ { \Xi } ^ { a } ( \widehat { P } ^ { \widehat { \alpha } } ; t _ { a } ) \leqslant \left( \sqrt { \delta _ { a } } + \delta _ { n } \right) ^ { 2 } + \delta _ { N } , \forall a \in \mathcal { A } ,
$$

where

$$
\delta _ { n } \stackrel { \mathrm { \scriptsize ~ d e f } } { = } \frac { 1 } { p } \sqrt { \frac { \log \left( 2 | A | L N / \eta _ { \delta _ { n } } \right) } { 2 n } } \quad a n d \quad \delta _ { N } \stackrel { \mathrm { \scriptsize ~ d e f } } { = } \left( 2 + 2 ( 1 - \xi ) \right) \frac { 1 - 2 \xi } { N ^ { 3 / 2 } } \sqrt { 2 \log \left( 4 / \eta _ { \delta _ { N } } \right) } .
$$

Proof. For a fixed $k \in [ N ]$ , by the triangle inequality and Lemma 9, we have

$$
\big | \Phi _ { a } \big ( \widehat { P } ^ { \widehat { \alpha } } ; \xi _ { k } \big ) - t _ { a } \big | \leqslant \big | \widehat { \Phi } _ { n } ^ { a } \big ( \widehat { P } ^ { \widehat { \alpha } } ; \xi _ { k } \big ) - t _ { a } \big | + \big | \Phi _ { a } \big ( \widehat { P } ^ { \widehat { \alpha } } ; \xi _ { k } \big ) - \widehat { \Phi } _ { n } ^ { a } \big ( \widehat { P } ^ { \widehat { \alpha } } ; \xi _ { k } \big ) \big | \leqslant \big | \widehat { \Phi } _ { n } ^ { a } \big ( \widehat { P } ^ { \widehat { \alpha } } ; \xi _ { k } \big ) - t _ { a } \big | + \delta _ { n } .
$$

Squaring both sides and averaging over $k = 1 , \cdots , N .$ , we get

$$
\frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left( \Phi _ { a } ( \widehat { P } ^ { \widehat { \alpha } } ; \xi _ { k } ) - t _ { a } \right) ^ { 2 } \leqslant \widehat { \mathcal { C } } _ { n , N } ^ { a } ( \widehat { \alpha } ; t _ { a } ) + 2 \delta _ { n } \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left| \widehat { \Phi } _ { n } ^ { a } ( \widehat { P } ^ { \widehat { \alpha } } ; \xi _ { k } ) - t _ { a } \right| + \delta _ { n } ^ { 2 } .
$$

By the Cauchy–Schwarz inequality and the feasibility of $\widehat { \mathbf { \alpha } } _ { \mathbf { \alpha } } \mathbf { \alpha } _ { \mathbf { \alpha } }$ i.e., $\widehat { \mathcal { C } } _ { n , N } ^ { a } ( \widehat { \pmb { \alpha } } ; t _ { a } ) \leqslant \delta _ { a }$ , we have

$$
\frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left| \widehat { \Phi } _ { n } ^ { a } ( \widehat { P } ^ { \widehat { \alpha } } ; \xi _ { k } ) - t _ { a } \right| \leqslant \Big ( \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \big ( \widehat { \Phi } _ { n } ^ { a } ( \widehat { P } ^ { \widehat { \alpha } } ; \xi _ { k } ) - t _ { a } \big ) ^ { 2 } \Big ) ^ { 1 / 2 } = \widehat { \mathcal { C } } _ { n , N } ^ { a } ( \widehat { \alpha } ; t _ { a } ) ^ { 1 / 2 } \leqslant \sqrt { \delta _ { a } } .
$$

Combining the two inequalities above, we get

$$
\frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left( \Phi _ { a } ( \widehat { P } ^ { \widehat { \alpha } } ; \xi _ { k } ) - t _ { a } \right) ^ { 2 } \leqslant \delta _ { a } + 2 \delta _ { n } \sqrt { \delta _ { a } } + \delta _ { n } ^ { 2 } = \left( \sqrt { \delta _ { a } } + \delta _ { n } \right) ^ { 2 } .\tag{4}
$$

Finally, applying Lemma 8 for $P = \widehat { P } ^ { \widehat { \alpha } }$ , combined with (4), we get

$$
\mathcal { C } _ { \Xi } ^ { a } ( \widehat { P } ^ { \widehat { \alpha } } ; t _ { a } ) \leqslant \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left( \Phi _ { a } ( \widehat { P } ^ { \widehat { \alpha } } ; \xi _ { k } ) - t _ { a } \right) ^ { 2 } + \delta _ { N } \leqslant \left( \sqrt { \delta _ { a } } + \delta _ { n } \right) ^ { 2 } + \delta _ { N } .
$$

The proof is concluded.

□

The second statement of Theorem 1 follows from Theorem 2.

Corollary 1 (Unfairness). Under the conditions of Theorem ${ \mathit { 2 } } ,$ it holds that

$$
\big | \gamma _ { a } \big ( \widehat { P } ^ { \widehat { \alpha } } \big ) \big | \leqslant \frac { \sqrt { \delta _ { a } } + \delta _ { n } + \sqrt { \delta _ { N } } } { \sigma _ { \Xi } } .
$$

Proof. From Theorem 2 and from Definition 9 we have

$$
\sigma _ { \Xi } ^ { 2 } \gamma _ { a } ( \widehat { P } ^ { \widehat { \alpha } } ) ^ { 2 } \leqslant { \mathcal C } _ { \Xi } ^ { a } ( \widehat { P } ^ { \widehat { \alpha } } ; t _ { a } ) \leqslant \left( \sqrt { \delta _ { a } } + \delta _ { n } \right) ^ { 2 } + \delta _ { N } .
$$

Taking the square root and using $\sqrt { u ^ { 2 } + w } \leqslant u + \sqrt { w }$ for $u , w \geqslant 0 .$ , we obtain

$$
\left| \gamma _ { a } ( \widehat { P } ^ { \widehat { \alpha } } ) \right| \leqslant \frac { \sqrt { ( \sqrt { \delta _ { a } } + \delta _ { n } ) ^ { 2 } + \delta _ { N } } } { \sigma _ { \Xi } } \leqslant \frac { \sqrt { \delta _ { a } } + \delta _ { n } + \sqrt { \delta _ { N } } } { \sigma _ { \Xi } } .
$$

The proof is concluded.

Formal version and the proof of the third statement of Theorem 1.

Lemma 10 (Deployment). Under the conditions of Theorem 2, simultaneously for all $a \in { \mathcal { A } }$ and $\xi ^ { \prime } \in [ \xi , 1 - \xi ]$ it holds that

$$
\bigl | \Phi _ { a } \bigl ( \widehat { P } ^ { \widehat { \alpha } } ; \xi ^ { \prime } \bigr ) - t _ { a } \bigr | \leqslant \Bigl ( 1 + \frac { | \xi ^ { \prime } - \mu _ { \Xi } | } { \sigma _ { \Xi } } \Bigr ) \bigl ( \sqrt { \delta _ { a } } + \delta _ { n } + \sqrt { \delta _ { N } } \bigr ) .
$$

Proof. It holds that

$$
\begin{array} { r } { \left( \mathbb { E } _ { \Xi } [ \Phi _ { a } ( \widehat { P } ^ { \widehat { \alpha } } ; \xi ) ] - t _ { a } \right) ^ { 2 } \leqslant { \mathcal C } _ { \Xi } ^ { a } ( \widehat { P } ^ { \widehat { \alpha } } ; t _ { a } ) \leqslant \left( \sqrt { \delta _ { a } } + \delta _ { n } \right) ^ { 2 } + \delta _ { N } . } \end{array}
$$

Taking the square root, we get $| \mathbb { E } _ { \Xi } [ \Phi _ { a } ( \widehat { P } ^ { \widehat { \alpha } } ; \xi ) ] - t _ { a } | \leqslant \sqrt { \delta _ { a } } + \delta _ { n } + \sqrt { \delta _ { N } }$ . For a fixed $\xi ^ { \prime } \in [ \xi , 1 - \xi ]$ , we have

$$
\Phi _ { a } \big ( \widehat { P } ^ { \widehat { \alpha } } ; \xi ^ { \prime } \big ) - t _ { a } = \underbrace { \Phi _ { a } \big ( \widehat { P } ^ { \widehat { \alpha } } ; \xi ^ { \prime } \big ) - \mathbb { E } _ { \Xi } \big [ \Phi _ { a } \big ( \widehat { P } ^ { \widehat { \alpha } } ; \xi \big ) \big ] } _ { \mathrm { f u c t u a t i o n ~ t e r m } } + \underbrace { \mathbb { E } _ { \Xi } \big [ \Phi _ { a } \big ( \widehat { P } ^ { \widehat { \alpha } } ; \xi \big ) \big ] - t _ { a } } _ { \mathrm { b i a s ~ t e r m } } \ .
$$

Since $\Phi _ { a } ( P ; \xi ) = \beta _ { a } ( P ) + \xi \gamma _ { a } ( P )$ and $\mathbb { E } _ { \Xi } [ \Phi _ { a } ( P ; \xi ) ] = \beta _ { a } ( P ) + \mu _ { \Xi } \gamma _ { a } ( P )$ , the fluctuation term equals $( \xi ^ { \prime } -$ $\mu _ { \Xi } ) \gamma _ { a } ( \widehat { P } ^ { \widehat { \alpha } } )$ . Applying the triangle inequality and Corollary 1, we obtain

$$
\big | \Phi _ { a } ( \widehat { P } ^ { \widehat { \alpha } } ; \xi ^ { \prime } ) - t _ { a } \big | \leqslant | \xi ^ { \prime } - \mu \circ | \frac { \sqrt { \delta _ { a } } + \delta _ { n } + \sqrt { \delta _ { N } } } { \sigma \circ } + \sqrt { \delta _ { a } } + \delta _ { n } + \sqrt { \delta _ { N } } = \Big ( 1 + \frac { | \xi ^ { \prime } - \mu \circ | } { \sigma \circ } \Big ) \big ( \sqrt { \delta _ { a } } + \delta _ { n } + \sqrt { \delta _ { N } } \Big ) .
$$

The proof is concluded.

## C.4.2 Utility rates

First, let us define $\boldsymbol { \epsilon } _ { 0 } \ { \stackrel { \mathrm { d e f } } { = } }$ mi $1 _ { a \in \mathcal { A } } \left\{ \delta _ { a } - \mathcal { C } _ { \Xi } ^ { a } ( P ^ { \alpha ^ { 0 } } ; t _ { a } ) \right\}$ . Now, for $0 \leqslant \epsilon \leqslant \epsilon _ { 0 }$ , let us define

$$
\nu ( \epsilon ) \stackrel { \mathrm { d e f } } { = } \operatorname* { m a x } _ { \alpha \in \Delta _ { L } } \left\{ \mathcal { U } ( P ^ { \alpha } ) : \mathcal { C } _ { \Xi } ^ { a } ( P ^ { \alpha } ; t _ { a } ) \leqslant \delta _ { a } - \epsilon , \forall a \in \mathcal { A } \right\} .
$$

Under Assumption $3 , \alpha ^ { 0 }$ is feasible for every $\epsilon \in [ 0 , \epsilon _ { 0 } ]$ , and in particular, $\nu ( 0 ) = \mathcal { U } ( P ^ { \alpha ^ { \star } } )$

Next, let us prove the following two lemmas that we will need later.

Lemma 11 (Reverse feasibility). Let $a \in { \mathcal { A } }$ be such that $\delta _ { n } + \delta _ { n ^ { \prime } } ^ { \prime } \leqslant \sqrt { \delta _ { a } }$ . If $\pmb { \alpha } \in \Delta _ { L }$ satisfies $\mathcal { C } _ { \Xi } ^ { \underline { { a } } } ( P ^ { \alpha } ; t _ { a } ) \ \leqslant$ $\left( \sqrt { \delta _ { a } } - \delta _ { n } - \delta _ { n ^ { \prime } } ^ { \prime } \right) ^ { 2 } - \delta _ { N } , t h e n \widehat { \mathcal { C } } _ { n , N } ^ { a } ( \alpha ; t _ { a } ) \leqslant \delta _ { a }$

Proof. For a fixed $k \in [ N ]$ , by the triangle inequality, Lemma 7 and Lemma 9, it holds that

$$
\begin{array} { r l } & { \left| \widehat { \Phi } _ { n } ^ { a } ( \widehat { P } ^ { \alpha } ; \xi _ { k } ) - t _ { a } \right| \leqslant \left| \widehat { \Phi } _ { n } ^ { a } ( \widehat { P } ^ { \alpha } ; \xi _ { k } ) - \Phi _ { a } ( \widehat { P } ^ { \alpha } ; \xi _ { k } ) \right| + \left| \Phi _ { a } ( \widehat { P } ^ { \alpha } ; \xi _ { k } ) - \Phi _ { a } ( P ^ { \alpha } ; \xi _ { k } ) \right| + \left| \Phi _ { a } ( P ^ { \alpha } ; \xi _ { k } ) - t _ { a } \right| } \\ & { \qquad \leqslant \delta _ { n } + \delta _ { n ^ { \prime } } ^ { \prime } + \left| \Phi _ { a } ( P ^ { \alpha } ; \xi _ { k } ) - t _ { a } \right| . } \end{array}
$$

Squaring and averaging over $k = 1 , \cdots , N$ , we get

$$
\widehat { \mathcal { C } } _ { n , N } ^ { a } ( \alpha ; t _ { a } ) \leqslant \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left( \Phi _ { a } ( P ^ { \alpha } ; \xi _ { k } ) - t _ { a } \right) ^ { 2 } + 2 ( \delta _ { n } + \delta _ { n ^ { \prime } } ^ { \prime } ) \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left| \Phi _ { a } ( P ^ { \alpha } ; \xi _ { k } ) - t _ { a } \right| + ( \delta _ { n } + \delta _ { n ^ { \prime } } ^ { \prime } ) ^ { 2 } .
$$

By Lemma 8 and the assumption on $\alpha ,$ we have

$$
\frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left( \Phi _ { a } ( P ^ { \alpha } ; \xi _ { k } ) - t _ { a } \right) ^ { 2 } \leqslant \mathcal { C } _ { \Xi } ^ { a } ( P ^ { \alpha } ; t _ { a } ) + \delta _ { N } \leqslant \left( \sqrt { \delta _ { a } } - \delta _ { n } - \delta _ { n ^ { \prime } } ^ { \prime } \right) ^ { 2 } .
$$

From Cauchy–Schwarz inequality, $\begin{array} { r } { \frac { 1 } { N } \sum _ { k } | \Phi _ { a } ( P ^ { \alpha } ; \xi _ { k } ) - t _ { a } | \leqslant \sqrt { \delta _ { a } } - \delta _ { n } - \delta _ { n ^ { \prime } } ^ { \prime } } \end{array}$

Finally, combining the above, we get

$$
\begin{array} { r } { \widehat { \mathcal { C } } _ { n , N } ^ { a } ( \alpha ; t _ { a } ) \leqslant \left( \sqrt { \delta _ { a } } - \delta _ { n } - \delta _ { n ^ { \prime } } ^ { \prime } \right) ^ { 2 } + 2 ( \delta _ { n } + \delta _ { n ^ { \prime } } ^ { \prime } ) \big ( \sqrt { \delta _ { a } } - \delta _ { n } - \delta _ { n ^ { \prime } } ^ { \prime } \big ) + ( \delta _ { n } + \delta _ { n ^ { \prime } } ^ { \prime } ) ^ { 2 } = \delta _ { a } . } \end{array}
$$

The proof is concluded.

Lemma 12 (Value continuity). Under Assumption 3, for every $0 \leqslant \epsilon \leqslant \epsilon _ { 0 }$ , it holds that

$$
0 \leqslant \nu ( 0 ) - \nu ( \epsilon ) \leqslant \epsilon C _ { 0 } , \qquad w h e r e \qquad C _ { 0 } \stackrel { \mathrm { d e f } } { = } \frac { \nu ( 0 ) - \mathcal { U } ( P ^ { \alpha ^ { 0 } } ) } { \epsilon _ { 0 } } .
$$

Proof. Let us fix $\epsilon \in [ 0 , \epsilon _ { 0 } ]$ and set

$$
\lambda _ { \epsilon } \stackrel { \mathrm { d e f } } { = } \frac { \epsilon } { \epsilon _ { 0 } } \in [ 0 , 1 ] , \qquad \alpha _ { \epsilon } \stackrel { \mathrm { d e f } } { = } ( 1 - \lambda _ { \epsilon } ) \alpha ^ { \star } + \lambda _ { \epsilon } \alpha ^ { 0 } .
$$

The simplex $\Delta _ { L }$ is convex, hence ${ \pmb { \alpha } } _ { \epsilon } \in \Delta _ { L }$ . Moreover, ${ \alpha } \mapsto \Phi _ { a } ( P ^ { \alpha } ; \xi )$ is linear for every fixed $\xi \ ( \mathrm { L e m m a \ 1 } )$ , hence $\alpha \mapsto ( \Phi _ { a } ( P ^ { \alpha } ; \xi ) - t _ { a } ) ^ { 2 }$ is convex, and taking the expectation over $\xi \sim \Xi$ preserves convexity. Therefore, for every $a \in { \mathcal { A } } .$ , it holds that

$$
\mathcal { C } _ { \Xi } ^ { a } ( P ^ { \alpha _ { \ c } } ; t _ { a } ) \leqslant ( 1 - \lambda _ { \epsilon } ) \mathcal { C } _ { \Xi } ^ { a } ( P ^ { \alpha ^ { * } } ; t _ { a } ) + \lambda _ { \epsilon } \mathcal { C } _ { \Xi } ^ { a } ( P ^ { \alpha ^ { 0 } } ; t _ { a } ) \leqslant ( 1 - \lambda _ { \epsilon } ) \delta _ { a } + \lambda _ { \epsilon } \left( \delta _ { a } - \epsilon _ { 0 } \right) = \delta _ { a } - \epsilon .
$$

Hence, $\pmb { \alpha } _ { \epsilon }$ is feasible at level $\epsilon ,$ and thus $\nu ( \epsilon ) \geqslant \mathcal { U } ( P ^ { \alpha _ { \epsilon } } )$ . Using the linearity of $\alpha \mapsto \mathcal { U } ( P ^ { \alpha } )$ (Lemma 1), we obtain

$$
\begin{array} { r } { \mathcal { U } ( P ^ { \alpha _ { c } } ) = ( 1 - \lambda _ { \epsilon } ) \mathcal { U } ( P ^ { \alpha ^ { \star } } ) + \lambda _ { \epsilon } \mathcal { U } ( P ^ { \alpha ^ { 0 } } ) = \nu ( 0 ) - \lambda _ { \epsilon } \big ( \nu ( 0 ) - \mathcal { U } ( P ^ { \alpha ^ { 0 } } ) \big ) = \nu ( 0 ) - \epsilon C _ { 0 } . } \end{array}
$$

Finally, combining the above yields $\nu ( 0 ) - \nu ( \epsilon ) \leqslant \epsilon C _ { 0 }$ . We conclude the proof by noticing that $\nu ( \epsilon ) \leqslant \nu ( 0 )$ due to the monotonicity of the feasible sets. □

Finally, the formal version and the proof of the fourth statement of Theorem 1.

Theorem 3 (Utility). Let Assumptions $1 , \ 2 , \ 3 , \ 4$ hold. Let

$$
\epsilon _ { 0 } \stackrel { \mathrm { d e f } } { = } \operatorname* { m i n } _ { a \in \mathcal { A } } \left\{ \delta _ { a } - \mathcal { C } _ { \Xi } ^ { a } ( P ^ { \alpha ^ { 0 } } ; t _ { a } ) \right\} , \qquad \delta _ { n ^ { \prime } } ^ { \prime } \stackrel { \mathrm { d e f } } { = } \frac { \delta _ { n ^ { \prime } } } { p v } , \qquad \kappa \stackrel { \mathrm { d e f } } { = } \operatorname* { m a x } _ { a \in \mathcal { A } } 2 \sqrt { \delta _ { a } } \left( \delta _ { n } + \delta _ { n ^ { \prime } } ^ { \prime } \right) + \delta _ { N } ,
$$

where $\delta _ { n }$ and $\delta _ { N }$ are as in Theorem ${ \mathit { 2 } } ,$ and suppose that $n , n ^ { \prime } , N$ are large enough so that $\kappa \leqslant \epsilon _ { 0 }$ . Then, for all $\eta _ { \delta _ { n ^ { \prime } } } , \eta _ { \delta _ { n } } , \eta _ { \delta _ { N } } , \eta _ { \delta _ { n } ( \mathcal { U } ) } \in ( 0 , 1 )$ , with probability greater than $1 - \eta _ { \delta _ { n ^ { \prime } } } - \eta _ { \delta _ { n } } - \eta _ { \delta _ { N } } - \eta _ { \delta _ { n } ( \mathcal { U } ) }$ over $( \mathcal { D } ^ { n ^ { \prime } } , \mathcal { D } ^ { n } , \Xi ^ { N } )$ problem (Pb) is feasible, and every solution αb $o f \ ( { \widehat { \mathcal { P } } } )$ satisfies

$$
\mathcal { U } ( P ^ { \alpha ^ { \star } } ) - \mathcal { U } ( \widehat { P } ^ { \widehat { \alpha } } ) \leqslant C _ { 0 } \kappa + 2 \delta _ { n } ( \mathcal { U } ) + \frac { \delta _ { n ^ { \prime } } } { \underline { { v } } } ,
$$

where

$$
C _ { 0 } \stackrel { \mathrm { d e f } } { = } \frac { \mathcal { U } ( P ^ { \alpha ^ { \star } } ) - \mathcal { U } ( P ^ { \alpha ^ { 0 } } ) } { \epsilon _ { 0 } } \in \Big [ 0 , \frac { 1 } { \epsilon _ { 0 } } \Big ] \qquad a n d \qquad \delta _ { n } ( \mathcal { U } ) = \sqrt { \frac { \log ( 2 L / \eta _ { \delta _ { n } ( \mathcal { U } ) } ) } { 2 n } } .
$$

Proof. Let $\widetilde { \pmb { \alpha } }$ be a maximizer in the definition of $\nu ( \kappa )$ , which exists since $\kappa \leqslant \epsilon _ { 0 } .$ , so that $\mathcal { U } ( P ^ { \widetilde { \alpha } } ) = \nu ( \kappa )$ and, for every $a \in { \mathcal { A } }$

$$
\mathcal { C } _ { \Xi } ^ { a } ( P ^ { \bar { \alpha } } ; t _ { a } ) \leqslant \delta _ { a } - \kappa \leqslant \delta _ { a } - 2 \sqrt { \delta _ { a } } \left( \delta _ { n } + \delta _ { n ^ { \prime } } ^ { \prime } \right) - \delta _ { N } \leqslant \left( \sqrt { \delta _ { a } } - ( \delta _ { n } + \delta _ { n ^ { \prime } } ^ { \prime } ) \right) ^ { 2 } - \delta _ { N } ,
$$

where the second inequality follows from the definition of $\kappa .$ Moreover, since $\mathcal { C } _ { \Xi } ^ { \underline { { a } } } ( P ^ { \alpha ^ { 0 } } ; t _ { a } ) \geqslant 0$ , we have $\epsilon _ { 0 } \leqslant \delta _ { a } .$ hence $2 \sqrt { \delta _ { a } } ( \delta _ { n } + \delta _ { n ^ { \prime } } ^ { \prime } ) \ \leqslant \kappa \ \leqslant \ \epsilon _ { 0 } \ \leqslant \ \delta _ { a }$ , that is $\delta _ { n } + \delta _ { n ^ { \prime } } ^ { \prime } \leqslant \sqrt { \delta _ { a } } / 2$ . Therefore, applying Lemma 11, we get $\widehat { \mathcal { C } } _ { n , N } ^ { a } ( \widetilde { \pmb { \alpha } } ; t _ { a } ) \leqslant \delta _ { a }$ for all $a \in { \mathcal { A } }$ . Thus, αe is feasible for $( \widehat { \mathcal { P } } )$ , which is thus non-empty and admits a solution $\widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } }$

Since $\widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf { \alpha } } \widehat { \mathbf \alpha } \widehat { \mathbf { \alpha } } \widehat { \mathbf \alpha } \widehat { \mathbf { \alpha } } \widehat { \mathbf \alpha } \widehat { \mathbf \alpha } \widehat { \mathbf \alpha } \widehat { \mathbf \alpha } \widehat { \mathbf \alpha } \widehat { \mathbf \alpha } \mathbf { \alpha } \widehat \mathbf { \alpha \alpha } \widehat \mathbf { \alpha } \widehat \mathbf { \alpha \alpha } \widehat \mathbf { \alpha \mathbf }$ is a maximizer of $( \widehat { \mathcal { P } } )$ and $\widetilde { \alpha }$ is feasible for $( \widehat { \mathcal { P } } )$ , we have $\widehat { \mathcal { U } } _ { n } ( \widehat { P } ^ { \widehat { \alpha } } ) \geqslant \widehat { \mathcal { U } } _ { n } ( \widehat { P } ^ { \widetilde { \alpha } } )$ . Combining this with Lemma 9, we get

$$
\begin{array} { r } { \mathcal { U } ( \widehat { P } ^ { \widehat { \alpha } } ) \geqslant \widehat { \mathcal { U } } _ { n } ( \widehat { P } ^ { \widehat { \alpha } } ) - \delta _ { n } ( \mathcal { U } ) \geqslant \widehat { \mathcal { U } } _ { n } ( \widehat { P } ^ { \widehat { \alpha } } ) - \delta _ { n } ( \mathcal { U } ) \geqslant \mathcal { U } ( \widehat { P } ^ { \widehat { \alpha } } ) - 2 \delta _ { n } ( \mathcal { U } ) . } \end{array}
$$

Next, applying Lemma 7, we obtain

$$
\mathcal { U } ( \widehat { P } ^ { \widetilde { \alpha } } ) \geqslant \mathcal { U } ( P ^ { \widetilde { \alpha } } ) - \frac { \delta _ { n ^ { \prime } } } { \underline { { v } } } = \nu ( \kappa ) - \frac { \delta _ { n ^ { \prime } } } { \underline { { v } } } .
$$

Finally, applying Lemma 12 with $\epsilon = \kappa \leqslant \epsilon _ { 0 }$ , we get $\nu ( \kappa ) \geqslant \nu ( 0 ) - \kappa C _ { 0 } = \mathcal { U } ( P ^ { \alpha ^ { \star } } ) - \kappa C _ { 0 }$ . Combining everything above, we obtain

$$
\mathcal { U } ( P ^ { \alpha ^ { \star } } ) - \mathcal { U } ( \widehat { P } ^ { \widehat { \alpha } } ) \leqslant C _ { 0 } \kappa + 2 \delta _ { n } ( \mathcal { U } ) + \frac { \delta _ { n ^ { \prime } } } { \underline { { v } } } ,
$$

where $0 \leqslant C _ { 0 } \leqslant 1 / \epsilon _ { 0 }$ , since $\alpha ^ { 0 }$ is feasible for (P) and $\mathcal { U } ( P ^ { \alpha } ) \in [ 0 , 1 ]$ for every $\pmb { \alpha } \in \Delta _ { L }$ The proof is concluded.

## D EXPERIMENTS: ADDITIONAL DETAILS

## D.1 Implementation Details

Utility. Policies rank the candidates of a user by decreasing score, and the uniform policy places every candidate at every position with equal probability. The position weights are $v _ { j } \ \propto \ 1 / \log _ { 2 } ( j + 1 )$ for $j \le K = 2 0$ and $v _ { j } = 0$ beyond, so that the utility $U ( P , z )$ is the NDCG@20 with binary relevance. The utility of category a is $\begin{array} { r } { \bar { U } ^ { a } ( P , z ) = \sum _ { i \in \mathcal { T } _ { a } } \rho _ { i } ( P ; \pmb { x } ) y _ { i } / V ( z ) } \end{array}$ , with the same normalisation $V ( z )$ as $U$ , so that $\textstyle \sum _ { a } U ^ { a } = U$

Rankers. The history of a user is encoded with PureSVD: Q holds the top d right singular vectors of the binary history matrix of the training users, and $e _ { u } = h _ { u } Q$ for any user with history $h _ { u }$ . The features of a pair $( u , i )$ are $\langle e _ { u } , q _ { i } \rangle , e _ { u } \odot q _ { i } , \log ( 1 + n _ { i } )$ , where $n _ { i }$ is the number of training histories containing $i ,$ and the length of the history of u. The sensitive attribute is never a feature. Rankers are LambdaMART models trained with LightGBM (150 trees, learning rate 0.1, 31 leaves, at least 40 samples per leaf, row and column subsampling 0.9 and 0.8), on all relevant candidates of the training users and $n _ { \mathrm { n { e g } } }$ non-relevant candidates per user drawn at random. The group-reweighted rankers use the weights $\boldsymbol { w } _ { \xi } ( g ) = \xi / p _ { 1 }$ for $g = 1$ and $( 1 - \xi ) / ( 1 - p _ { 1 } )$ for $g = 0$ with $p _ { 1 }$ the share of women among training users.

Problem $( \widehat { \mathcal { P } } ) \sharp$ is uniform on $[ \xi , 1 - \xi ]$ with $\xi = 0 . 1$ , so that $\mu _ { \Xi } = 1 / 2$ and $\sigma _ { \Xi } ^ { 2 } = ( 1 - 2 \xi ) ^ { 2 } / 1 2$ . We use $N = 2 0$ values of $\xi ,$ one drawn uniformly in each of $\bar { N }$ intervals of equal width covering the support. The budgets are $\delta _ { a } = \sigma _ { \Xi } ^ { 2 } ( \lambda \varepsilon _ { a } ) ^ { 2 }$ , where $\varepsilon _ { a }$ is twice the standard error of the gap $\widehat { \gamma } _ { a } ( P _ { 0 } )$ on the estimation users. Since the perturbation variance of category a is $\sigma _ { \Xi } ^ { 2 } \gamma _ { a } ^ { 2 }$ , the constraint without target amounts to $| \gamma _ { a } | \leq \lambda \varepsilon _ { a } \colon$ with $\lambda = 1$ the tolerated gap is the estimation noise of the gap of $P _ { 0 }$ . The aggregate constraint of UP uses the same rule with the aggregate gap. The targets are $t _ { a } = \kappa \widehat { \Phi } _ { a } ( P _ { 0 } ; \mu _ { \Xi } )$ , computed on the estimation users. Problem $( \widehat { \mathcal { P } } )$ is solved with CVXPY and the Clarabel solver.

## D.2 Synthetic Data

Ground truth. Each of the 8000 users belongs to group $G = 1$ (women) with probability 0.4. The 300 ofers are split into three categories of equal size. Users have a latent taste $u \sim \mathcal { N } ( 0 , I _ { 8 } )$ and an afinity $a _ { u , c } \sim \mathcal { N } ( 0 , 1 )$ for each category $c ,$ both independent of $G ,$ and ofers have an embedding $q _ { i } \sim \mathcal { N } ( 0 , I _ { 8 } / 8 )$ . The true interest of user u in ofer i of category $c ( i )$ is $y _ { u i } ^ { \star } \sim$ Bernoulli sigmoid $\left( \left. u , q _ { i } \right. + 1 . 5 a _ { u , c ( i ) } - 3 \right) \big )$ . Since the interests do not depend on $G ,$ an oracle ranking by $y ^ { \star }$ has no gap in any category.

Historical bias. The recorded interactions keep each true interest, except those of women in managerial ofers and of men in administrative ofers, which are kept with probability $1 - \beta$ only. The two biases have the same strength, so that the aggregate gap remains small while the gaps of the two categories have opposite signs.

Histories, candidates and splits. For each user, the catalogue is split at random, independently of $G ,$ into a history pool (60% of the ofers) and a candidate pool. The history is the set of recorded interactions in the history pool, and the candidates are the ofers of the candidate pool. Users are split at random into training (50%), estimation (25%) and test (25%) users. The relevant candidates are the recorded interactions for training users, and the true interests for estimation and test users. Users with fewer than 3 history items or no relevant candidate are removed.

Settings. The dictionary is described in Section 6.1, with $d = 1 6$ and $n _ { \mathrm { n e g } } = 5 0$ . We use $\beta \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 \}$ ， $\lambda \in \{ 1 , 2 , 3 , 5 , 8 \}$ and $\kappa \in \{ 1 , 0 . 9 , \ldots , 0 . 4 \}$ , and repeat each setting with 3 seeds, which change the data, the splits and the rankers.

![](images/ef101fd122b1770e3532ba2907cc509655eb9f5c6c68051d30901d66bd90f340.jpg)  
Figure 3: Synthetic data, deployment on new users $( \beta = 0 . 5 , \lambda = 5 )$ . Utility of the mixtures selected on the estimation users (share of women 0.4), as a function of the share of women among new users, relative to its mean value at 0.5. Means and standard deviations over 3 seeds.

Deployment on new populations. To illustrate the uniform deployment guarantee (Theorem 1.3), we deploy the mixtures selected for $\beta = 0 . 5$ and λ = 5 on new populations of 4000 users, drawn with the same catalogue and a share of women $p ^ { \prime } \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ , and labelled with their true interests. Figure 3 shows that the utility of $P _ { 0 }$ over all ofers does not depend on $p ^ { \prime }$ , whereas the utility of managerial and administrative ofers moves by about 20% in opposite directions. CUP reduces this dependence as κ decreases, at the cost of utility shown in Figure 1(d). For CUP, the distance between the utility of each category and its target stays below $\lambda \varepsilon _ { a } ( \sigma _ { \Xi } + | p ^ { \prime } - \mu _ { \Xi } | )$ , the leading term of the bound of Theorem 1.3, in 85 of the 90 cases (three categories, five shares, three seeds and two values of κ).

## D.3 MovieLens-1M

Data. Every rating counts as an interaction. We keep the movies that belong to exactly one of the six genres, and the users with at least 10 history items and at least one relevant item that is not in their history. Table 1 gives the resulting statistics. The splits of users change with the seed, and so do the rankers.

Table 1: MovieLens-1M after preprocessing: 5018 users, of which 28% women, and 1280 movies. On average, a user has 61 history items (median 39), 16 relevant items (median 10) and 1219 candidates. Share of relevant items: share of the relevant items of each group that belong to the genre.
<table><tr><td>genre</td><td>movies</td><td>relevant items</td><td>women</td><td>share of relevant items men</td></tr><tr><td>Sci-Fi</td><td>138</td><td>13953</td><td>0.14</td><td>0.19</td></tr><tr><td>Adventure</td><td>108</td><td>8155</td><td>0.08</td><td>0.11</td></tr><tr><td>Crime</td><td>182</td><td>11615</td><td>0.12</td><td>0.15</td></tr><tr><td>Romance</td><td>424</td><td>24360</td><td>0.40</td><td>0.28</td></tr><tr><td>Children&#x27;s</td><td>159</td><td>10307</td><td>0.15</td><td>0.12</td></tr><tr><td>Horror</td><td>269</td><td>11134</td><td>0.11</td><td>0.15</td></tr><tr><td>total</td><td>1280</td><td>79524</td><td></td><td></td></tr></table>

Dictionary. The L = 19 policies are listed in Table 2, with $d = 6 4$ and $n _ { \mathrm { n e g } } = 1 0 0$ . A cell is a pair (group, genre), and the under-represented group of a genre is the group for which this genre has the smallest share of its relevant items, among training users. The cell-based rankers only reweight the relevant candidates, so that G enters through the weights and is never a feature. We use $\kappa \in \{ 1 , 0 . 9 5 , 0 . 9 , 0 . 8 5 , 0 . 8 \}$ and $\lambda \in \{ 1 , 1 . 5 , 2 , 3 , 5 , 8 \}$

Table 2: Dictionary on MovieLens-1M. All policies but the first three are LambdaMART models trained on the same rows.
<table><tr><td>policy</td><td>description</td></tr><tr><td>uniform</td><td>every candidate at every position with equal probability</td></tr><tr><td>popularity</td><td> $\log ( 1 + n _ { i } )$  , the same ranking for every user</td></tr><tr><td>PureSVD</td><td> $\left. \boldsymbol { e } _ { u } , \boldsymbol { q } _ { i } \right.$ </td></tr><tr><td> $P _ { 0 }$ </td><td>unweighted</td></tr><tr><td>ξ-reweighted (5)</td><td>weights  $w _ { \xi } ( g ) , \xi \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ </td></tr><tr><td>one group (2)</td><td>trained on the users of one group, deployed on all users</td></tr><tr><td>cell-balanced</td><td>each relevant item weighted by the inverse number of relevant items of its cell</td></tr><tr><td>under-represented (1)</td><td>relevant items of the under-represented group of every genre weighted by 5</td></tr><tr><td>under-represented, genre a (6) the same, for genre a only</td><td></td></tr></table>

ITFR and MF-BPR. We use the code of Wang et al. (2024), with minimal changes that leave the training algorithm untouched (device handling, and calls deprecated in recent versions of NumPy and PyTorch). The user groups are the two genders and the item groups the six genres. The training interactions are the histories of all users and 90% of the relevant items of the training users; the remaining 10% form the validation set. The relevant items of estimation and test users are never used. We keep the hyperparameters of the authors for MovieLens-1M: embeddings of dimension 64, Adam with learning rate $1 0 ^ { - 3 }$ , no $\ell _ { 2 }$ regularisation, batches of 1024, one negative per positive, $\eta = 0 . 0 5$ $\gamma = 0 . 5 ,$ $\rho = 0 . 0 2$ and $\tau = 0 . 2$ . The model of the epoch with the best validation NDCG@20 within 100 epochs is kept. Candidates are scored by the cosine between user and item embeddings, as in ITFR, and ranked deterministically like the rankers of the dictionary. MF-BPR is trained with the same code and hyperparameters, without the sharpness-aware perturbation and with group weights equal to the share of each group in the batch, which gives the BPR loss; it keeps the normalisation of the scores.

CatFAIR. We re-implement in PyTorch the MF model of the code of Kheya et al. (2025), on the same training interactions and validation set as ITFR. The model has 40 factors, user and item biases and a globa bias, initialised as in their code, and is trained with the BPR loss, one negative per positive drawn uniformly, Adam with learning rate $1 0 ^ { - 3 }$ , weight decay 0.02 and batches of 1024. For the users of each batch, a diferentiable top-50 over all items (hard in the forward pass, sigmoid $( s - s _ { 5 0 } ) / 0 . 0 1 )$ in the backward pass) gives the share of each genre in their list; the regularizer is the sum over genres of the absolute diference between the mean shares of women and men (eq. 10 of their paper), passed through sigmoid(0.1 $( g - 0 . 5 ) ,$ ) and weighted by α against the BPR loss, as in their code. Training stops when the validation NDCG@20 has not improved by 0.0005 for 10 epochs, within 100 epochs. We use their grid $\alpha \in \{ 0 , 0 . 1 , \ldots , 0 . 6 \}$ , where $\alpha = 0 . 6 $ , and $\alpha = 0$ gives the backbone. Candidates are scored by the inner product plus biases. The variant of the released code, which divides the share of each genre by its share in the catalogue, gives comparable results.

## D.4 Additional Results on MovieLens-1M

Feasibility. Table 3 gives the number of splits on which $( \widehat { \mathcal { P } } )$ is feasible for CUP. UP is feasible on every split except for $\kappa = 1$ and $\lambda = 1 ~ ( 8 / 1 0 )$ , and the versions without target are feasible on every split. With $\lambda = 2$ , the targets are met on every split for $\kappa \leq 0 . 9$

Computation time. Table 4 details the times of Section 6. Times are measured on a MacBook Pro with an Apple M2 chip and 16 GB of memory, on CPU only. The dictionary times are averaged over 3 splits run with an empty cache. Training times are medians, validation included: over 20 trainings for ITFR (about 23 s per epoch), and over the trainings of splits 1 to 9 for CatFAIR (9 for $\alpha = 0$ , 54 for $\alpha > 0 .$ , about 3 s per epoch).

Table 3: Number of splits (out of 10) on which CUP is feasible.
<table><tr><td colspan="4">κ</td></tr><tr><td>λ</td><td>1</td><td>0.95 0.9</td><td>0.85 0.8</td></tr><tr><td>1</td><td>0</td><td>0 0</td><td>0 1</td></tr><tr><td>1.5</td><td>1 3</td><td>5 7</td><td>7</td></tr><tr><td>2</td><td>6 9</td><td>10 10 10</td><td>10</td></tr><tr><td>≥3</td><td>10 10</td><td>10</td><td>10</td></tr></table>

Table 4: Computation time on MovieLens-1M, per split.
<table><tr><td>step time</td></tr><tr><td>UP and CUP PureSVD and features 0.2s fitting the 16 LightGBM rankers 50s scoring the candidates of estimation and test users 81 s utilities of the 19 policies 1.6s</td></tr><tr><td>one SOCP, UP / CUP 2.5 / 7.4 ms the 72 SOCPs of the grid 0.4s</td></tr><tr><td>Baselines, one training 38 min</td></tr><tr><td>ITFR (100 epochs) CatFAIR, α = 0 26 s CatFAIR, α &gt; 0 2.3 min</td></tr></table>