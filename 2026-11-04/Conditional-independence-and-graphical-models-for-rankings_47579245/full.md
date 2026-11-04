# Conditional independence and graphical models for rankings

Aritz Pérez Basque Center for Applied Mathematics (BCAM) aperez@bcamth.org

## Abstract

Ranking models define probability distributions over rankings and are fundamental in applications such as preference learning and decision-making. Without structural assumptions, these models are intractable due to their factorial complexity. Although independence assumptions are key to reducing complexity and enabling interpretability, standard notions of conditional independence must account for the mutual exclusivity constraints inherent to rankings. We introduce relative rank models, a probabilistic framework based on structured coarsenings of the ranking space that relax these constraints while preserving relative order information. This representation enables a well-defined notion of conditional independence for rankings and allows rankings to be modeled using Bayesian networks. As a result, standard methods for learning and probabilistic inference can be applied to rankings, with interpretable representations based on conditional independence. We further show that two prominent ranking models based on independence assumptions tailored to rankings arise as special cases, providing a unifying framework for independence in ranking models.

## 1 Introduction

Modeling rankings is a fundamental problem in machine learning, with applications including preference learning, recommender systems, and information retrieval. In these settings, rankings are represented by permutations and express preferences over a finite set of items [Regenwetter et al., 2006], where lower ranks correspond to stronger preferences. A ranking model defines a probability distribution over the ranking space and is typically learned from ranking data [Mallows, 1957].

The most general ranking model assigns an arbitrary probability to each ranking, requiring a factorial number of parameters in the number of items. This factorial complexity renders such models impractical, as they cannot be learned from limited data, do not admit efficient inference procedures, and lack interpretability.

To overcome these limitations, ranking models commonly rely on structural assumptions that reduce model complexity [Mallows, 1957, Fligner and Verducci, 1986]. Among these, independence assumptions play a central role. Two representative examples are the repeated insertion model [Doignon et al., 2004, Kenig et al., 2018] and the riffle shuffle model [Huang and Guestrin, 2009, 2010, Huang et al., 2012]. The repeated insertion model imposes a fixed insertion order that governs how items are sequentially incorporated into a ranking, whereas the riffle shuffle model introduces a more flexible notion of independence by allowing rankings over disjoint subsets of items to be interleaved.

In probabilistic modeling, the most fundamental notion of independence is conditional independence [Dawid, 1979], which underlies graphical models such as Bayesian networks [Pearl, 1988, Lauritzen, 1996]. Graphical models provide compact representations of complex distributions, support efficient inference and learning algorithms, and offer a clear interpretation in terms of conditional independence relations [Koller and Friedman, 2009]. However, applying conditional independence to rankings requires accounting for the mutual exclusivity constraints (MEC) inherent to permutations [Huang et al., 2009a], which require distinct items to occupy distinct ranks. These constraints induce strong dependencies among ranking variables and restrict the free parameterization of standard graphical factorizations.

In this work, we address this limitation by shifting the modeling focus from full rankings to structured forms of partial ranking information. Rather than treating partial rankings merely as incomplete observations [Lebanon and Mao, 2007, Lu and Boutilier, 2011, Lan et al., 2013, Lu and Boutilier, 2014], we use them as fundamental modeling primitives [Achab et al., 2019, Chierichetti et al., 2018]. By defining structured families of partial rankings, we construct random variables whose realizations induce partitions of the permutation space. This coarsening relaxes the mutual exclusivity constraints and enables a well-defined notion of conditional independence for ranking data, thereby supporting the construction of graphical models for rankings.

Contributions. We introduce relative rank models, probabilistic models for rankings based on structured coarsenings of the permutation space. Our main contributions are:

• We establish fundamental results for relative rank variables and the coarsenings of the permutation space, relaxing mutual exclusivity constraints and providing the basis for relative rank models.

• We define conditional independence for rankings based on relative rank variables and introduce relative rank networks, graphical models for rankings that enable the use of standard theory and algorithms for learning and probabilistic inference, while providing interpretable representations in terms of conditional independence.

• We highlight the connection between relative rank networks and existing ranking models, showing that models such as repeated insertion and riffle shuffle can be represented within the RRN framework and that their assumptions correspond to specific sets of conditional independence statements.

We provide the proofs of the theoretical results in the main text in Appendix F.

Notation. Let $N = \{ 1 , \ldots , n \}$ be a set of indices, and let $V = \{ v _ { 1 } , \ldots , v _ { n } \}$ be a set of items. Uppercase letters $A , \dot { B , C }$ denote subsets of V , and |A| denotes the cardinality of A. We use $i , j , k \in N$ and $u , v , w \in V$ to denote particular indices and items, respectively. We denote by $S _ { n } ^ { V }$ the set of bijective functions from V to N, called rankings, and by $\it S _ { V } ^ { n }$ the set of bijective functions from N to V, called orderings. Rankings are denoted by Greek letters $\rho , \sigma , \tau , \pi$ , and their respective orderings by $\bar { \rho } , \bar { \sigma } , \bar { \tau } , \bar { \pi }$ . For simplicity, in examples we identify items with their indices, $\mathbf { i . e . , } V = N$ and represent rankings and orderings as tuples using () and [], respectively. For example, the ranking $( 3 , 1 , 2 )$ corresponds to the ordering [2, 3, 1], and vice versa. All directed graphs considered in this work have vertex set V and are identified with their sets of arcs, denoted by uppercase calligraphic letters $\mathcal { V } , \mathcal { G } \subseteq V \times V$

For a ranking $\sigma \in S _ { n } ^ { V }$ and a subset $A \subseteq V .$ , the partial ranking $\sigma ^ { A } \in { \cal S } _ { | A | } ^ { A }$ is the bijective function from A to $\{ 1 , \ldots , | A | \}$ that preserves the relative order of the items in A under σ. Its inverse, $\bar { \sigma } ^ { A } \in { \cal S } _ { A } ^ { | A | }$ <sup>|</sup>, is the corresponding partial ordering.

## 2 Relative rank variables

Relative rank variables represent rankings through events in the ranking space. Each realization of such a variable corresponds to a subset of rankings, and together these events form a disjoint cover of the ranking space. Because rankings are permutations, they are subject to mutual exclusivity constraints (MEC), which imply that some combinations of relative ranks cannot occur. In the representation induced by relative rank variables, these constraints appear as empty events. However, carefully designed relative rank variables can break the mutual exclusivity constraints of permutations, so that every realization corresponds to a non-empty event. In this case, the induced events form a partition<sup>1</sup> of the ranking space. Relative rank variables that break MEC will play a central role in this work, as they allow the definition of ranking models and conditional independence relations without parametric constraints.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>x</td></tr><tr><td rowspan=1 colspan=1>1,2,31,2,31</td><td rowspan=1 colspan=1>,2,2</td></tr><tr><td rowspan=1 colspan=1>1,3,2 1,3,2 |</td><td rowspan=1 colspan=1>1,2,1</td></tr><tr><td rowspan=1 colspan=1>2,1,3 2,1,3</td><td rowspan=1 colspan=1>1,1,2</td></tr><tr><td rowspan=1 colspan=1>2,3,1 3,1,2</td><td rowspan=1 colspan=1>1,2,1</td></tr><tr><td rowspan=1 colspan=1>3,1,2 2,3,1</td><td rowspan=1 colspan=1>1,1,2</td></tr><tr><td rowspan=1 colspan=1>3,2,1 3,2,1</td><td rowspan=1 colspan=1>1,1,1</td></tr></table>

Relative rank variables are based on the notion of relative rank.

Definition 2.1 (Relative rank). Let $v \in V$ be an item and $P \subseteq V \setminus \{ v \}$ be a subset of items called the reference set. The relative rank of v with respect to $P$ is the mapping $r _ { P } ^ { v } : S _ { n } ^ { V }  \{ 1 , . . . , | P | + 1 \}$ , defined for $\sigma \in S _ { n } ^ { V }$ as

$$
r _ { P } ^ { v } ( \sigma ) : = 1 + \sum _ { u \in P } \mathbb { 1 } { \bigl ( } \sigma ( u ) < \sigma ( v ) { \bigr ) } .
$$

The value $r _ { P } ^ { v } ( \sigma ) = i$ indicates that item v has the i-th smallest rank among the items in $P \cup \{ v \}$ under σ. In particular, $r _ { P } ^ { v } ( \sigma ) = 1 \ : ( \mathrm { r e s p . } , r _ { P } ^ { v } ( \sigma ) = | P | + 1 )$ means that v is ranked ahead of (resp., behind) every item in $P$ under σ. Equivalently, $r _ { P } ^ { v } ( \sigma ) = \sigma ^ { P \cup \{ v \} } ( v )$ is the rank of item v in the partial ranking of $P \cup \{ v \}$ under $\sigma , \sigma ^ { P \cup \{ v \} } \in \mathring { S } _ { | P | + 1 } ^ { { P \cup \{ v \} } }$ . For instance, for $\sigma = ( 2 , 1 , 3 )$ we have $\sigma ^ { \{ 2 , 3 \} } = ( 1 , 2 )$ and $r _ { \{ 3 \} } ^ { 2 } ( \sigma ) = 1$ , whereas for $\tau = ( 2 , 3 , 1 )$ we have $\tau ^ { \{ 2 , 3 \} } = ( 2 , 1 )$ and $r _ { \{ 3 \} } ^ { 2 } ( \tau ) = 2$

Another useful interpretation of relative ranks is in terms of insertions into partial orderings. Given a ranking σ, the value $r _ { P } ^ { v } ( \sigma )$ specifies the insertion position of item v into the partial ordering $\bar { \sigma } ^ { P }$ to obtain $\bar { \sigma } ^ { P \cup \{ v \} }$ . For example, for $\sigma = ( 2 , 3 , 1 )$ the relative rank of 1 with respect to $\{ 2 , 3 \}$ is $r _ { \{ 2 , 3 \} } ^ { 1 } ( \sigma ) = 2$ , meaning that inserting item 1 into $\bar { \sigma } ^ { \{ 2 , 3 \} } = [ 3 , 2 ]$ at position 2 yields the ordering $\bar { \sigma } =$ $[ 3 , 1 , 2 ]$ corresponding to the ranking σ. Relative ranks generalize pairwise comparisons [Wauthier et al., 2013], which correspond to reference sets of size 1.

Definition 2.2 (Relative rank variable). The relative rank variable (RRV) for V defined by the reference sets $\{ P _ { v } : v \in V \}$ is the n-dimensional categorical random variable

$$
\begin{array} { r } { \pmb { X } = ( X _ { 1 } , \ldots , X _ { n } ) , \qquad X _ { v } ( \sigma ) : = r _ { P _ { v } } ^ { v } ( \sigma ) , \quad \sigma \in \mathscr { S } _ { n } ^ { V } , } \end{array}
$$

where each component $X _ { v }$ is called the $R R V f o r v$ . The state space of each variable $X _ { v }$ is $\mathcal { X } _ { v } : =$ $\{ 1 , \ldots , | P _ { v } | + 1 \}$ , and the joint state space of $\dot { \boldsymbol { X } }$ is $\pmb { \mathscr { X } } : = \mathscr { X } _ { 1 } \times \overline { { \cdots } } \times \mathscr { X } _ { n }$

For a realization $x _ { v } \in \mathcal { X } _ { v }$ , the associated event is $\langle x _ { v } \rangle : = \{ \sigma \in S _ { n } ^ { V } : X _ { v } ( \sigma ) = x _ { v } \}$ , and for a joint realization $\pmb { x } = ( x _ { 1 } , \dots , x _ { n } ) \in \pmb { \chi }$ , the associated event is

$$
\langle { \pmb x } \rangle : = \{ \sigma \in S _ { n } ^ { V } : { \pmb X } ( \sigma ) = { \pmb x } \} = \bigcap _ { v \in V } \langle { \pmb x } _ { v } \rangle .
$$

We denote by $\langle { \pmb X } \rangle = \{ \langle { \pmb x } \rangle : { \pmb x } \in { \pmb X } \}$ the disjoint cover of $S _ { n } ^ { V }$ induced by X.

An RRV for V maps each ranking to a tuple of relative ranks and induces a disjoint cover of the ranking space through the associated events. Such variables are fully determined by the collection of reference sets $\{ P _ { v } : v \in V \}$ defining the relative ranks of each item. This collection of reference sets can be naturally represented by a directed graph, where edges encode which items belong to the reference set of each variable. This graphical representation provides a convenient way to specify RRVs for V and analyze the structural properties of the covers they induce over the ranking space.

![](images/a2847fa9fa5b6440cb3285dff00e83a0824b4755b5a53366e5c583d1060f7d38.jpg)

Figure 1: The variable graph $\mathcal { V } = \{ ( 1 , 2 ) , ( 2 , 3 ) \}$ defining X for $V = \{ 1 , { \overline { { 2 } } } , { \dot { 3 } } \}$ . The table shows the mapping of X from rankings $\sigma \in S _ { 3 } ^ { V }$ to the corresponding realizations $\pmb { x } \in \pmb { \mathscr { X } }$ . Colors in column x group the rankings belonging to the same event.

Definition 2.3 (Variable graph). Let $\mathcal { V } \subseteq V \times V$ be a directed graph with $( v , v ) \notin \mathcal { V }$ for every $v \in V$ For each $v \in V$ , define

$$
P _ { v } ^ { \nu } : = \{ u \in V \setminus \{ v \} : ( u , v ) \in \mathcal { V } \}
$$

as the set of parents of node v in V. We say that the RRV $\pmb { X } = ( X _ { 1 } , \ldots , X _ { n } )$ is defined by V if the reference set of each component $X _ { v }$ is given by $P _ { v } = P _ { v } ^ { \nu }$ for all $v \in V$ . The directed graph V is called the variable graph defining X.

We denote by $\mathrm { a n c } ( \mathcal V ) = \{ \bar { \pi } \in S _ { V } ^ { n } : \pi ( u ) < \pi ( v )$ for every $( u , v ) \in \mathcal { V } \}$ the set of ancestral (or topological) orderings of V, where $\pi = \bar { \pi } ^ { - 1 }$

Figure 1 shows an example of an RRV $\textbf { \em X } = ~ \left( X _ { 1 } , X _ { 2 } , X _ { 3 } \right)$ for V defined by the variable graph $\nu ~ = ~ \{ ( 1 , 2 ) , ( 2 , 3 ) \}$ . The reference sets defined by V are $P _ { 1 } ~ = ~ \emptyset , ~ P _ { 2 } ^ { \mathbf { \bar { \Delta } } } = \{ 1 \}$ , and

$P _ { 3 } = \{ 2 \}$ . The realizations of X induce the cover of $S _ { n } ^ { V }$ given by the set of events $\langle x \rangle =$ $\{ \{ ( 1 , 2 , 3 ) \} , \{ ( 1 , 3 , 2 ) , ( 2 , 3 , 1 ) \} , \{ ( 2 , 1 , 3 ) , ( 3 , 1 , 2 ) \} , \{ ( 3 , 2 , 1 ) \} \}$

Rankings are subject to mutual exclusivity constraints: distinct items must occupy distinct ranks in a ranking. RRVs defined by arbitrary variable graphs can also be subject to MEC, since certain combinations of relative ranks may lead to contradictions and therefore cannot occur simultaneously in any ranking. In the representation induced by an RRV, these constraints appear as empty events in the cover of the ranking space. This motivates identifying RRVs whose realizations correspond only to non-empty events, effectively removing the constraints imposed by permutations. The following definition formalizes this property.

Definition 2.4 (Mutual Exclusivity Constraints). Let X be an RRV for V with state space X. We say that X is subject to mutual exclusivity constraints (MEC) if there exists a realization $\pmb { x } \in \pmb { \mathscr { X } }$ such that $\langle { \pmb x } \rangle = \emptyset . \mathrm { I f } \left. { \pmb x } \right. \ne \emptyset$ for all $\pmb { x } \in \pmb { \mathcal { X } }$ , we say that X breaks MEC.

Breaking MEC ensures that every realization of the RRV corresponds to a non-empty subset of rankings. Whether this property holds depends on the structure of the reference sets defining the RRV for V, and therefore on the associated variable graph. The following result characterizes when an RRV breaks MEC in terms of the structure of its variable graph.

Theorem 2.5 (Breaking MEC). Let X be an RRV for V defined by a variable graph V. Then X breaks MEC if and only if V is acyclic.

When an RRV breaks MEC, every realization corresponds to a non-empty event, and the induced cover ⟨X ⟩ becomes a partition of the ranking space. This property is fundamental for the developments that follow. In particular, RRVs that break MEC allow the definition of probabilistic models and conditional independence relations without structural constraints imposed by permutations.

The acyclicity of the variable graph not only guarantees that RRVs break MEC, but also provides a constructive way to generate the rankings in the event associated with each realization. In Appendix B, we present Algorithm 1 for generating events associated with RRVs defined by acyclic variable graphs. The procedure incrementally constructs partial rankings by inserting elements according to the realization of the RRV, following a topological ordering induced by the variable graph.

![](images/fbd94a72a550f95ab5b59530a3ce11dd90101cfe5bd3dd13ea47f5bfc13b4d22.jpg)

<table><tr><td rowspan=1 colspan=1>σ ō |</td><td rowspan=1 colspan=1>x</td></tr><tr><td rowspan=1 colspan=1>1,2,31,2,31</td><td rowspan=1 colspan=1>,2,2</td></tr><tr><td rowspan=1 colspan=1>1,3,2 1,3,2 |</td><td rowspan=1 colspan=1>1,2,1</td></tr><tr><td rowspan=1 colspan=1>2,1,32,1,3</td><td rowspan=1 colspan=1>2,1,2</td></tr><tr><td rowspan=1 colspan=1>2,3,1– 3,1,2</td><td rowspan=1 colspan=1>1,2,1</td></tr><tr><td rowspan=1 colspan=1>3,1,2 2,3,1</td><td rowspan=1 colspan=1>2,1,2</td></tr><tr><td rowspan=1 colspan=1>3,2,1 3,2,1</td><td rowspan=1 colspan=1>2,1,1</td></tr></table>

Figure 2: The variable graph V = {(1, 2), (2, 1), (2, 3)} contains the cycle 1, 2, 1. The realizations $\textbf { \textit { x } } \in \mathbf {  { \mathbf { \Theta } } }$ {(1, 1, 1), (1, 1, 2), (2, 2, 1), (2, 2, 2)} also correspond to empty events and do not appear in the table.

Figure 2 shows a cyclic variable graph defining an RRV for V subject to MEC, as predicted by Theorem 2.5. The presence of the cycle 1, 2, 1 induces logical consistency constraints between

the RRVs for items 1 and 2. As a consequence, the realizations $\pmb { x } _ { \{ 1 , 2 \} } \in \{ ( 1 , 1 ) , ( 2 , 2 ) \}$ have associated empty events. Thus the joint realizations $\pmb { x } \in \{ ( 1 , 1 , 1 ) , ( \tilde { 1 , 1 , 2 } ) , ( 2 , 2 , 1 ) , ( 2 , 2 , 2 ) \}$ } are contradictory and therefore correspond to empty events. For additional examples of variable graphs, see Appendix A.

## 3 Relative rank models

Once an RRV for V is defined, probabilistic models over rankings can be constructed by assigning probabilities to its realizations. Such models are called relative rank models. In this section, we formally introduce relative rank models and describe their relationship with ranking models through the pushforward mapping.

Definition 3.1 (Relative rank model). Let X be an RRV for V. A relative rank model (RRM) is a probability distribution

$$
h \in \Delta ( { \mathcal { X } } ) .
$$

An RRM assigns probability mass to the realizations of the RRV. Under conditions concerning empty events specified below, this assignment defines a probability distribution over the non-empty events of the disjoint cover $\langle x \rangle$ induced by X.

Next, we connect RRMs with ranking models through the pushforward relation.

Definition 3.2 (Pushforward RRM). Let X be an RRV for V . The pushforward mapping under X is the function push $\mathbf { \Omega } _ { \mathbf { X } } : \Delta ( S _ { n } ^ { V } )  \Delta ( \pmb { \mathcal { X } } )$ , such that $h = \mathrm { p u s h } _ { X } ( p )$ for any ranking model p, where

$$
h ( { \pmb x } ) : = p ( { \pmb x } ) = \sum _ { \sigma \in \langle { \pmb x } \rangle } p ( \sigma ) , \qquad { \pmb x } \in { \pmb X } .\tag{1}
$$

In addition, the pullback of h under X is the set of ranking models satisfying Eq. 1, denoted by $\mathsf { p u l l } ( h ) \subseteq \bar { \Delta } ( \mathcal { S } _ { n } ^ { \bar { V } } )$

The pushforward relation can be used to characterize the set of RRMs associated with ranking models. Definition 3.3 (Admissible RRM). Let X be an RRV for V. We say that an RRM h for X is admissible if there exists $p \in \Delta ( \hat { S _ { n } ^ { V } } )$ such that push $\mathbf { \boldsymbol { x } } \mathbf { \mathit { ( p ) } } = h$ . Otherwise, it is inadmissible.

Admissible RRMs are those that correspond to at least one ranking model through the pullback relation. By construction, any RRM obtained as the pushforward of a ranking model is admissible. However, RRMs can also be defined directly over $_ { x }$ , and in this case admissibility is not guaranteed.

Admissibility imposes a structural constraint: an RRM must assign zero probability to all realizations associated with empty events. This condition is automatically satisfied when the underlying RRV breaks MEC, since all realizations correspond to non-empty events. In contrast, when the RRV is subject to MEC, admissibility requires enforcing structural zeros. A detailed characterization is provided in Appendix C.

## 4 Conditional independence for rankings

Although RRMs reduce the complexity of ranking models by controlling the size of the reference sets $P _ { v }$ for $v \in V$ , the number of free parameters of a general RRM is given by

$$
\left( \prod _ { v \in V } ( | P _ { v } | + 1 ) \right) - 1 .\tag{2}
$$

As in classical probabilistic modeling, a natural strategy to further reduce model complexity is to impose conditional independence assumptions and factorize the model accordingly. The challenge in the ranking setting is not merely to impose independence, but to define it in a representation that is structurally compatible with rankings. RRVs provide such a representation, allowing conditional independence relations to be defined analogously to multidimensional categorical variables.

Definition 4.1 (Conditional independence). Let $A , B \subseteq V$ be disjoint non-empty subsets, let $C \subseteq V \setminus ( A \cup B )$ , and let X be an RRV for $V$ with an RRM h. We say that $X _ { A }$ is conditionally independent of X<sub>B</sub> given $X _ { C }$ under $h ,$ denoted by $\pmb { X } _ { A } \perp _ { h } \pmb { X } _ { B } \vert \pmb { X } _ { C } , \mathbf { \bar { i } }$

$$
h ( \pmb { x } _ { A , B } \mid \pmb { x } _ { C } ) = h ( \pmb { x } _ { A } \mid \pmb { x } _ { C } ) h ( \pmb { x } _ { B } \mid \pmb { x } _ { C } ) ,\tag{3}
$$

for all $( { \pmb x } _ { A } , { \pmb x } _ { B } , { \pmb x } _ { C } ) \in { \pmb x } _ { A } \times { \pmb x } _ { B } \times { \pmb x } _ { C }$ whenever $h ( \pmb { x } _ { C } ) > 0$ . Let $p \in \Delta ( S _ { n } ^ { V } )$ be a ranking model. We say that $X _ { A }$ is conditionally independent of $\mathbf { \delta } _ { \mathbf { X } _ { B } }$ given $X _ { C }$ under $p ,$ denoted by $X _ { A } \perp _ { p } X _ { B } \mid \mathbf { \bar { X } } _ { C }$ , if the pushforward of $p$ under X satisfies Eq. 3. When $C = \varnothing$ , these relations reduce to (unconditional) independence, denoted by $X _ { A } \perp _ { h } X _ { B }$ and $X _ { A } \perp _ { p } X _ { B }$ , respectively.

In words, conditional independence indicates that knowing a realization of $X _ { B }$ does not provide any additional information about the probabilities of $X _ { A }$ once the realization of $X _ { C }$ is known.

In general, given an ordering of the items π¯, any RRM can be factorized using the chain rule as $\begin{array} { r } { h ( \pmb { x } ) = \prod _ { v \in V } h ( x _ { v } \mid \pmb { x } _ { < \pi ^ { v } } ) } \end{array}$ where $< _ { \pi } v = \{ u \in V : \pi ( u ) < \pi ( v ) \}$ denotes the conditioning set of v. Conditional independence relations can simplify this factorization by reducing the conditioning sets. In probabilistic graphical models, Bayesian networks provide a compact representation of such factorizations through a directed graph whose structure encodes conditional independence relations.

## 5 Relative rank networks

Relative rank networks are graphical models for rankings obtained by combining RRVs with Bayesian network factorizations. In this section, we focus on RRVs defined by acyclic variable graphs, which break MEC and therefore induce partitions of the ranking space without empty events. This restriction avoids the structural zero constraints that arise when RRVs are subject to MEC and allows RRMs to be parameterized freely (see Appendix C). Building on this representation, relative rank networks provide a principled way to represent complex ranking distributions using conditional independence assumptions.

Given an RRV defined by an acyclic variable graph, Eq. 2 shows that an unrestricted RRM can require specifying an exponential number of parameters. Even for sparse variable graphs such as directed trees, this can lead to ${ \mathcal { O } } ( 2 ^ { n } )$ parameters. Moreover, probabilistic inference tasks, such as marginalization over subsets of RRVs, are computationally intractable in general. To address these challenges, we introduce graphical models that exploit conditional independence relations among RRVs.

Definition 5.1 (Relative rank network). A relative rank network (RRN) is a triplet $( \nu , \mathcal { G } , \pmb { \theta } )$ , where V is an acyclic variable graph defining the RRV X for V, G is a directed acyclic graph called the factorization graph, and $\mathbf { \bar { \theta } } = ( \theta _ { 1 } , \ldots , \theta _ { n } )$ denotes the model parameters. The RRN $( \nu , \mathcal { G } , \pmb \theta )$ represents the RRM

$$
h ( \pmb { x } ; \mathcal { G } , \pmb { \theta } ) = \prod _ { v \in V } h ( x _ { v } \mid \pmb { x } _ { P _ { v } ^ { \mathcal { G } } } ; \theta _ { v } ) .\tag{4}
$$

An RRN combines two complementary graphical structures. The variable graph V defines the RRVs and the partition of the ranking space they induce. The factorization graph $\mathcal { G }$ specifies how the joint distribution over these variables factorizes into conditional distributions. In this way, V determines the representation of rankings, while $\mathcal { G }$ determines the probabilistic structure of the model. We call $( \nu , \mathcal { G } )$ the structure of an RRN.

The number of parameters of an RRN depends on its structure. The graph V determines the size of the state space of each relative rank variable, while $\mathcal { G }$ determines the conditioning sets in the factorization. The total number of free parameters of an RRN with structure $( \nu , \mathcal { G } )$ is

$$
\sum _ { v \in V } | P _ { v } ^ { \nu } | \prod _ { u \in P _ { v } ^ { \mathcal { G } } } \left( | P _ { u } ^ { \nu } | + 1 \right) .\tag{5}
$$

This flexibility allows the construction of RRNs with controlled complexity by restricting the maximum number of parents in both the variable graph V and the factorization graph ${ \mathcal { G } } .$ . In particular, the number of free parameters is at most $n k ^ { s + \mathrm { i } }$ , where $k = \operatorname* { m a x } _ { v \in V } | \mathcal { X } _ { v } |$ and $s = \operatorname* { m a x } _ { v \in V } | P _ { v } ^ { \mathcal { G } } | ;$ it grows linearly in n when k and s are bounded.

For a fixed RRN structure, exact probabilistic inference over RRVs takes $\mathcal { O } ( n k ^ { t + 1 } )$ time [Koller and Friedman, 2009], where $k = \operatorname* { m a x } _ { v \in V } | \mathcal { X } _ { v } |$ | is the maximum RRV cardinality and t is the treewidth of the moralized factorization graph $\mathcal { G }$ . For example, when $\mathcal { G }$ is a directed tree, $t = 1$ and the cost reduces to $\mathcal { O } ( n k ^ { 2 } )$ ).

RRNs can be connected to ranking models through a generalization of the pushforward mapping. Given a ranking model $p ,$ the pushforward under $( \bar { \nu _ { , } } \mathcal { G } )$ defines the parameters of an RRN by computing the conditional probabilities of the RRV realizations from $p .$ This mapping allows us to focus on the structure $( \mathcal { V } , \bar { \mathcal { G } } )$ independently of the specific parameter values.

Definition 5.2 (Factorized pushforward). Let X be an RRV defined by an acyclic variable graph $\nu ,$ and let G be a factorization graph. Thefactorized pushforward of a ranking model $p \in \Delta ( { S } _ { n } ^ { V } )$ under $( \nu , \mathcal { G } )$ is the function push $\bar { \nu } , \varsigma : \Delta ( S _ { n } ^ { V } ) \to \Delta \varsigma ( \bar { \pmb { x } } )$ , where $\Delta _ { \mathcal { G } } ( \pmb { \mathscr { x } } ) \subseteq \bar { \Delta ( \pmb { \mathscr { x } } ) }$ denotes the space of RRNs with structure $( \nu , \mathcal { G } )$ . The RRN $h = \mathrm { p u s h } _ { \gamma , \mathcal { G } } ( p )$ is defined as

$$
h ( \pmb { x } ) = \prod _ { v \in V } h ( x _ { v } \mid \pmb { x } _ { P _ { v } ^ { \mathscr { G } } } ) ,
$$

where $\begin{array} { r } { h ( x _ { v } \mid \pmb { x } _ { P _ { v } ^ { \mathcal { G } } } ) = \frac { p ( \pmb { x } _ { v } , \pmb { x } _ { P _ { v } ^ { \mathcal { G } } } ) } { p ( \pmb { x } _ { P _ { v } ^ { \mathcal { G } } } ) } \mathrm { i f } p ( \pmb { x } _ { P _ { v } ^ { \mathcal { G } } } ) > 0 } \end{array}$ , and $\begin{array} { r } { h ( x _ { v } \mid \pmb { x } _ { P _ { v } ^ { \mathcal { G } } } ) = \frac { 1 } { | \mathcal { X } _ { v } | } } \end{array}$ otherwise, for all $v \in V$

The factorized pushforward mapping of a ranking model under $( \nu , \mathcal { G } )$ produces an RRM that factorizes according to G. Since in RRNs the RRV for $V$ breaks MEC, the factorized pushforward mapping yields admissible models, and any RRM can be specified without additional structural constraints. In Appendix D, we discuss the admissibility of factorized models over RRVs subject to MEC and the structural zero constraints they impose.

For observed rankings and a fixed variable graph V, learning $\mathcal { G }$ reduces to Bayesian-network structure learning over the categorical realizations $\mathbf { \bar { \mathbf { X } } } ( \sigma )$ , and the local distributions can be estimated from empirical frequencies. Appendix E outlines this procedure and the additional selection of $\nu$ in Algorithm 2.

The structure of an RRN admits a graphical interpretation in terms of conditional independence. The factorization graph G defines a Bayesian network over the RRV $\pmb { X } = ( X _ { 1 } , \ldots , X _ { n } )$ defined by $\nu ,$ and therefore encodes conditional independence relations among its components $X _ { \imath }$ for $v \in V$ As in standard Bayesian networks, these relations can be derived from $\mathcal { G }$ using the d-separation criterion [Pearl, 1988].

Definition 5.3 (Independence model). Let $( \nu , \mathcal { G } )$ be the structure of an RRN. The independence model induced by $( \bar { \nu _ { , } } \mathcal { G } )$ is the set of conditional independence relations

$$
\operatorname { i n d } ( \mathcal { V } , \mathcal { G } ) = \{ \ X _ { A } \perp X _ { B } \mid X _ { C } \colon A , B , C \subseteq V , X _ { A } \ \perp _ { \mathcal { G } } \ X _ { B } \mid X _ { C } \} ,
$$

where X is the RRV defined by V, the subsets $A , B , C \subseteq V$ are pairwise disjoint with $A , B \neq \emptyset$ and the relation $\pmb { X } _ { A } \perp _ { \mathcal { G } } \pmb { X } _ { B } \perp \mathbf { \bar { X } } _ { C }$ indicates that A is d-separated from B by C in $\mathcal { G } .$

Given the structure $( \nu , \mathcal { G } )$ , the induced independence model is a compositional graphoid satisfying the local Markov property [Studený, 2005].

Any RRN with structure (V, G) satisfies the conditional independence relations in ind(V, G).

Lemma 5.4. Let $h \in \Delta _ { \mathcal { G } } ( \mathcal { X } )$ be an RRN with structure $( \nu , \mathcal { G } )$ , where X is the RRV defined by V, and let $p \in p u l l ( h )$ . Then h and p satisfy the conditional independence relations in $i n d ( \nu , \mathcal { G } )$ , that $i s ,$

$$
X _ { A } \perp _ { \mathcal { G } } X _ { B } \mid X _ { C } \implies \{ X _ { A } \perp _ { h } X _ { B } \mid X _ { C } , \quad X _ { A } \perp _ { p } X _ { B } \mid X _ { C } \} .
$$

## 6 Representing ranking models and independence assumptions

This section has two goals. First, we show how RRMs can be used to represent ranking models in general. Second, we show that two prominent families of ranking models based on alternative independence notions, namely repeated insertion and riffle shuffle models, can be represented within the RRN framework. In this way, the independence assumptions underlying both models correspond to specific independence models over $\mathbf { R R } \mathbf { \bar { V } s }$

## 6.1 Ranking models based on RRMs

An RRV X defined by an acyclic variable graph induces a partition of the ranking space, and RRMs for X represent probability distributions over this partition. A particularly useful situation occurs when an RRV induces a thin partition of the ranking space, that is, a partition whose cells contain a single ranking. In this case, the mapping induced by the RRV from the ranking space $S _ { n } ^ { V }$ to the state space X is bijective, and therefore RRMs for X are ranking models.

For a general acyclic variable graph, an RRM specifies the probabilities of the events $\langle x \rangle$ but does not determine how probability is distributed among rankings within each event. A ranking model can be obtained by specifying a distribution within each event, as described in Appendix C. For example, assigning equal probability to the rankings within each event yields $p ( \sigma ) = \bar { h ( \boldsymbol { X } ( \sigma ) ) } / | \langle \boldsymbol { X } ( \sigma ) \rangle$ |. The cardinality of the corresponding event can be obtained by enumerating its rankings using Algorithm 1.

The following result shows that an RRV defined by a complete acyclic variable graph induces a thin partition of the ranking space.

Proposition 6.1 (Thin partition). Let X be an RRV defined by V. Then, X induces a thin partition of $S _ { n } ^ { V }$ if and only if V is a complete acyclic variable graph.

Intuitively, a complete acyclic variable graph has a unique ancestral ordering π¯, and the reference set defining the RRV for $v = \bar { \pi } ( i )$ is $P _ { v } \bar { = } \{ \bar { \pi } ( 1 ) , \ldots , \bar { \pi } ( i - 1 ) \}$ } for $i = 1 , \ldots , n$ . Thus, a realization x encodes the position of each item with respect to all previously considered items in π¯, yielding a representation equivalent to the permutation itself. Figure 3 illustrates this correspondence for $n = 3$ where the ancestral ordering is $\bar { \bar { \pi } } = [ 1 , 2 , 3 ]$

An immediate consequence is that RRVs for V defined by complete acyclic variable graphs represent bijective mappings between the state space and the ranking space, and therefore ranking models can be represented by RRMs. In the literature, RRVs providing bijective representations of permutations are known as inversion vectors [Malagon et al., 2025], and they form the basis of repeated insertion models [Doignon et al., 2004]. According to Proposition 6.1, there are n! different RRVs corresponding to inversion vectors.

![](images/041efe3086648541957539d2ae102039b201c1940ded3d8e85634dc39faadb01.jpg)  
Figure 3: The complete acyclic variable graph $\nu =$ $\{ ( \mathbf { \bar { 1 } } , 2 ) , ( 2 , 3 ) , ( 1 , \mathbf { \bar { 3 } } ) \}$ , where $\bar { \pi } = [ 1 , \bar { 2 } , \bar { 3 } ]$ is its unique ancestral ordering. The RRV defined by V induces a thin partition of the ranking space.

In the remainder of this section, we use this representation to relate RRNs with two prominent families of ranking models based on independence notions tailored to permutations.

## 6.2 Repeated insertion model

Repeated insertion (RI) independence is a probabilistic independence notion defined with respect to a fixed ordering π¯ of the items, called the insertion ordering [Doignon et al., 2004]. A ranking model p satisfies RI independence for π¯ [Kenig et al., 2018] if it factorizes as

$$
p ( \sigma ; \bar { \pi } ) = \prod _ { i = 1 } ^ { n } p _ { i } ( f _ { \sigma , \bar { \pi } } ( i ) ) ,\tag{6}
$$

for all $\sigma \in S _ { n } ^ { V }$ , where $p _ { i }$ is the insertion distribution of the item $\bar { \pi } ( i )$ and

$$
f _ { \sigma , \bar { \pi } } ( i ) = 1 + \sum _ { j = 1 } ^ { i - 1 } { \bf 1 } \{ \sigma ( \bar { \pi } ( j ) ) < \sigma ( \bar { \pi } ( i ) ) \} .
$$

The function $f _ { \sigma , \bar { \pi } } ( i )$ determines the insertion position of the item $\bar { \pi } ( i )$ in the partial ranking $\sigma ^ { < _ { \bar { \pi } } i }$ where $< _ { \bar { \pi } } i = \{ \bar { \pi } ( 1 ) , \dots , \bar { \pi } ( i - 1 ) \}$ . We call a repeated insertion model (RIM) for π¯ any ranking model that factorizes according to Eq. 6, and we denote the space of RIMs for π¯ by $\Delta _ { \bar { \pi } } ( { \cal S } _ { n } ^ { V } )$

For a fixed insertion ordering and arbitrary categorical insertion distributions, a RIM has $n ( n - 1 ) / 2$ free parameters.

From a generative perspective, a RIM for π¯ constructs rankings by sequentially inserting items according to the ordering π¯. Starting from the trivial ranking over $\{ \bar { \pi } ( 1 \bar { ) } \} , ( 1 )$ , each item $\bar { \pi } ( i )$ is inserted into the current partial ranking at a position sampled from its insertion distribution $p _ { i }$ . After $n - 1$ insertion steps, this procedure produces a full ranking for V. This generative process is a particular case of Algorithm 1 in Appendix B.

Theorem 6.2. Let $\bar { \pi } \in S _ { V } ^ { n }$ be an insertion ordering. Let $\mathcal { V } ^ { R I }$ be the complete acyclic variable graph with ancestral ordering π¯, and let $\mathcal { G } ^ { R I } = \mathcal { O }$ be the empty factorization graph. Let X be an RRV defined by $\mathcal { V } ^ { R I }$ . Then,

$$
\Delta _ { \bar { \pi } } ( S _ { n } ^ { V } ) = \Delta _ { \mathcal G ^ { R I } } ( \pmb { \mathscr X } ) .
$$

As a consequence, the independence notion underlying RIMs can be expressed entirely in terms of conditional independence relations among RRVs.

Corollary 6.3. $L e t \bar { \pi } \in \mathcal S _ { V } ^ { n }$ be the insertion ordering, and let $\mathcal { V } ^ { R I }$ and ${ \mathcal { G } } ^ { R I }$ be the graphs defined in Theorem 6.2. Then, RI independencefor π¯ corresponds to the independence model $i n \bar { d } ( \mathcal { V } ^ { R I } , \mathcal { G } ^ { R I } )$

In particular, since the factorization graph ${ \mathcal { G } } ^ { R I }$ is empty, the independence model ind $( \mathcal { V } ^ { R I } , \mathcal { G } ^ { R I } )$ contains all conditional independence relations $\pmb { X } _ { A } \perp \hat { \pmb { X } } _ { B } \perp \pmb { X } _ { C }$ for disjoint non-empty subsets $A , B \subset V$ and $C \subseteq V \setminus ( A \cup B )$

An additional consequence of Theorem 6.2 is that a RIM for π¯ can be constructed from any ranking model using the factorized pushforward mapping.

Corollary 6.4. Let $\bar { \pi } \in S _ { V } ^ { n }$ be the insertion ordering, and let $\mathcal { V } ^ { R I }$ and $\mathcal { G } ^ { R I }$ be the graphs defined in Theorem 6.2. Let $p \in \Delta ( { \cal S } _ { n } ^ { V } )$ be a ranking model. Then, $h = p u s h _ { \mathcal { V } ^ { R I } , \mathcal { G } ^ { R I } } ( p )$ is the RIMfor π¯ that minimizes the KL divergence with respect to $p ,$ that is, $h = \arg \operatorname* { m i n } _ { \boldsymbol { q } \in \Delta _ { \boldsymbol { \pi } } ( S _ { n } ^ { V } ) } D _ { K L } ( \boldsymbol { p } | | \boldsymbol { q } )$

## 6.3 Riffle shuffle model

Riffle shuffle (RS) independence [Huang and Guestrin, 2009, 2010, Huang et al., 2012] models rankings by independently ranking disjoint subsets of items and then interleaving the resulting partial rankings. The name is inspired by the riffle shuffle used in card games. Let $A = \{ a _ { 1 } , \ldots , a _ { m } \}$ and $B = \{ \bar { b } _ { 1 } , \dots , b _ { n - m } \}$ be a partition of V. A ranking model p satisfies RS independence for A and B if it factorizes as

$$
p ( \sigma ) = p _ { A } ( \sigma ^ { A } ) p _ { A } ^ { B } ( s _ { A } ^ { B } ( \sigma ) ) p _ { B } ( \sigma ^ { B } ) ,\tag{7}
$$

for every $\sigma \in S _ { n } ^ { V }$ , where $\sigma ^ { A } \in { \cal S } _ { | A | } ^ { A }$ and $\sigma ^ { B } \in \mathcal { S } _ { | B | } ^ { B }$ denote the (partial) rankings of the items in A and B induced by σ (see Definition 12 in [Huang and Guestrin, 2012]). The distributions $p _ { A } \in \Delta ( S _ { | A | } ^ { A } )$ and $\dot { p _ { B } } \in \dot { \Delta } ( \mathcal { S } _ { | B | } ^ { B } )$ model the partial rankings within each subset of items. The function $s _ { A } ^ { \dot { B } } : S _ { n } ^ { V } \to \binom { N } { | B | }$ determines how the positions of the items in A and B are interleaved in $\sigma ,$ and the distribution $\dot { p } _ { A } ^ { B }$ models the interleavings. For an interleaving $s , s ( i )$ denotes its i-th smallest element. Given the interleaving $\pmb { s } = s _ { A } ^ { B } ( \sigma )$ we have $\sigma ( v ) = s ( \sigma ^ { B } ( v ) )$ , for $v \in B$ . We call a riffle shuffle model (RSM) for A and B any ranking model that factorizes according to Eq. 7, and we denote the space ofRSMsfor A and B by $\Delta _ { A , B } ( { \cal S } _ { n } ^ { V } )$

An RSM for a fixed partition A, B, with arbitrary partial ranking and interleaving distributions, has $| A | ! + | B | ! + { \binom { n } { | B | } } - 3$ free parameters: $| { \cal { A } } | ! - \bar { 1 }$ for $p _ { A } , | B | ! - 1$ for $p _ { B }$ , and $\binom { \varkappa } { | B | } - 1$ for $p _ { A } ^ { B }$

From a generative perspective, an RSM constructs rankings in two stages. First, the items in A and B are ranked independently according to the distributions $p _ { A }$ and $p _ { B }$ , producing the rankings $\sigma ^ { A }$ and $\sigma ^ { B }$ . Then, the two rankings are interleaved by sampling an interleaving pattern from the distribution $p _ { A } ^ { B }$ , which determines the set of positions occupied by the items of each subset. The interleaving pattern is selected independently of $\sigma ^ { A }$ and $\sigma ^ { B }$

We now show that RSMs can also be represented within the RRN framework by combining appropriate choices of variable graphs and a nontrivial factorization graph. In this representation, two RRN structures are considered, each defined by a different variable graph but sharing the same factorization graph. A ranking model that factorizes according to both structures corresponds to an RSM for A and B.

Theorem 6.5. Let $A = \{ a _ { 1 } , \ldots , a _ { m } \}$ and $B = \{ b _ { 1 } , \ldots , b _ { n - m } \}$ form a partition of V . Let $\mathcal { V } ^ { R S }$ and $\mathcal { W } ^ { R S }$ be complete acyclic variable graphs with ancestral orderings $\bar { \pi } = [ a _ { 1 } , \ldots , a _ { m } , b _ { 1 } , \ldots , b _ { n - m } ]$ and $\bar { \tau } = [ b _ { 1 } , \dots , b _ { n - m } , a _ { 1 } , \dots , a _ { m } ] ,$ , respectively. Let X and Y be the RRVs defined $b y \mathcal { V } ^ { R S }$ and $\mathcal { W } ^ { R S }$ , respectively. Let $\mathcal { G } ^ { R S } = \mathcal { G } _ { A } \dot { \cup } \mathcal { G } _ { B }$ be a directed acyclic graph, where $\mathcal { G } _ { A }$ and $\mathcal { G } _ { B }$ are complete subgraphs over the vertex sets A and $B ,$ , respectively. Then,

$$
\Delta _ { A , B } ( { \cal S } _ { n } ^ { V } ) = \Delta _ { \mathcal { G } ^ { R S } } ( { \pmb { \chi } } ) \cap \Delta _ { \mathcal { G } ^ { R S } } ( { \pmb { \ y } } ) .
$$

RIMs correspond to a particularly simple instance of RRNs, characterized by a complete variable graph and an empty factorization graph. RSMs, in contrast, rely on a richer independence structure. In the RRN representation, this structure is captured by a nontrivial factorization graph over two RRVs for V that provide two different relative-rank encodings of the ranking space, both inducing the same thin partition.

According to Theorem 6.5, the space of RSMs is given by the intersection of the spaces of two RRNs. Thus, RS independence corresponds to the independence models specified by the structures of the two RRNs.

Corollary 6.6. Let A and B be a partition ofV, and let $\nu ^ { R S } , \mathcal { W } ^ { R S }$ , and $\mathcal { G } ^ { R S }$ be the graphs defined in Theorem 6.5. Then, RS independence for A and B corresponds to in $d ( \mathcal { V } ^ { \tilde { R } S } , \mathcal { G } ^ { R S } ) \cup \breve { i } n d ( \mathcal { W } ^ { R \check { S } } , \mathcal { G } ^ { R S } )$

Finally, the next result shows how to obtain an RSM for A and B from any ranking model using the factorized pushforward mapping.

Proposition 6.7. Let p be a ranking model, let A and B be a partition ofV, and let $\nu ^ { R S } , \mathcal { W } ^ { R S } ,$ , and $\mathcal { G } ^ { R \dot { S } }$ be the graphs defined in Theorem 6.5. Then, $h = p u s h _ { \mathcal { V } ^ { R S } , \mathcal { G } ^ { R S } } \left( p u s h _ { \mathcal { W } ^ { R S } , \mathcal { G } ^ { R S } } ( p ) \right)$ is the RSM for A and B that minimizes the Kullback-Leibler divergence with respect to p.

Together, these results show that RIMs and RSMs can be represented within the RRN framework, each corresponding to a specific independence model over relative rank variables.

## 7 Conclusions

This theoretical work introduces relative rank models as a probabilistic framework for rankings based on structured coarsenings of the permutation space. By representing rankings through relative rank variables, the framework replaces the structure of permutations with categorical random variables that can break the mutual exclusivity constraints of rankings.

A central result of the paper is the characterization of acyclic variable graphs as the exact condition under which relative rank variables break mutual exclusivity constraints. When mutual exclusivity constraints are broken, admissibility imposes no further structural constraints on relative rank models, and relative rank models can be simplified by assuming conditional independence among relative rank variables.

Building on this characterization, we define relative rank networks, which combine relative rank variables with Bayesian network factorizations. Relative rank networks provide a principled way to encode conditional independence relations among relative rank variables while retaining a clear semantics in terms of partitions of the ranking space. As a consequence, they enable the application of standard tools from probabilistic graphical models to ranking data.

We further show that well-known ranking models based on specialized independence notions, namely repeated insertion and riffle shuffle models, can be represented within the framework of relative rank networks. From this perspective, these models correspond to specific sets of conditional independences over relative rank variables rather than fundamentally different probabilistic constructions.

More generally, the results of this paper suggest a shift in how independence should be introduced in ranking models: instead of defining independence notions directly on permutations, one should first construct a structured coarsening that breaks mutual exclusivity constraints and then apply standard probabilistic independence relations. Relative rank networks provide a concrete and flexible realization of this principle.

## Acknowledgments and Disclosure of Funding

The author thanks Ekhiñe Irurozki for helpful discussions. This work was supported by MCIN/AEI/10.13039/501100011033 under projects PLEC2024-011247 and CEX2021-001142-S, and by the Basque Government through the BERC 2022–2025 and Elkartek programs.

## References

M. Achab, A. Korba, and S. Clémençon. Dimensionality reduction and (bucket) ranking: a mass transportation approach. In International Conference on Algorithmic Learning Theory, volume 98, pages 64–93, 2019.

F. Chierichetti, A. Dasgupta, S. Haddadan, R. Kumar, and S. Lattanzi. Mallows models for top-k lists. In Advances in Neural Information Processing Systems, volume 31, 2018.

A. P. Dawid. Conditional independence in statistical theory. Journal ofthe Royal Statistical Society Series B: Statistical Methodology, 41(1):1–15, 1979.

J. P. Doignon, A. Pekec, and M. Regenwetter. The repeated insertion model for rankings: Missing ˇ link between two subset choice models. Psychometrika, 69(1):33–54, 2004.

M. A. Fligner and J. S. Verducci. Distance based ranking models. Journal ofthe Royal Statistical Society: Series B (Methodological), 48(3):359–369, 1986.

J. Huang and C. Guestrin. Riffled independence for ranked data. In Advances in Neural Information Processing Systems, volume 22, 2009.

J. Huang and C. Guestrin. Learning hierarchical riffle independent groupings from rankings. In International Conference on Machine Learning, 2010.

J. Huang and C. Guestrin. Uncovering the riffled independence structure of ranked data. Electronic Journal ofStatistics, 6:199–230, 2012.

J. Huang, C. Guestrin, and L. Guibas. Fourier theoretic probabilistic inference over permutations. Journal ofMachine Learning Research, 10(37):997–1070, 2009a.

J. Huang, C. Guestrin, X. Jiang, and L. Guibas. Exploiting probabilistic independence for permutations. In Artificial Intelligence and Statistics, pages 248–255. PMLR, 2009b.

J. Huang, A. Kapoor, and C. Guestrin. Riffled independence for efficient inference with partial rankings. Journal ofArtificial Intelligence Research, 44:491–532, 2012.

B. Kenig, L. Ilijasic, H. Ping, B. Kimelfeld, and J. Stoyanovich. Probabilistic inference over repeated´ insertion models. In AAAI Conference on Artificial Intelligence, 2018.

D. Koller and N. Friedman. Probabilistic Graphical Models: Principles and Techniques. The MIT Press, Cambridge, Massachusetts, 2009.

Y. Lan, S. Niu, J. Guo, and X. Cheng. Is top-k sufficient for ranking? In ACM International Conference on Information and Knowledge Management, pages 1261–1270, 2013.

S. L. Lauritzen. Graphical Models. Oxford University Press, New York, 1996.

G. Lebanon and Y. Mao. Non-parametric modeling of partially ranked data. In Advances in Neural Information Processing Systems, volume 20, pages 857–864, 2007.

T. Lu and C. Boutilier. Learning Mallows models with pairwise preferences. In International Conference on Machine Learning, pages 145–152, 2011.

T. Lu and C. Boutilier. Effective sampling and learning for Mallows models with pairwise-preference data. Journal ofMachine Learning Research, 15(117):3963–4009, 2014.

M. Malagon, A. Barrena, H. Iñigo, E. Irurozki, J. A. Lozano, and J. Ceberio. A unified view of bijective transformations for optimizing permutation problems. In European Conference on Artificial Intelligence, pages 4750–4757, 2025.

C. L. Mallows. Non-null ranking models. I. Biometrika, 44(1/2):114–130, 1957.

J. Pearl. Probabilistic Reasoning in Intelligent Systems: Networks ofPlausible Inference. Morgan Kaufmann, San Mateo, California, 1988.

M. Regenwetter, B. Grofman, A. A. J. Marley, and I. Tsetlin. Behavioral Social Choice: Probabilistic Models, Statistical Inference, and Applications. Cambridge University Press, 2006.

M. Studený. Probabilistic Conditional Independence Structures. Springer, 2005.

F. Wauthier, M. Jordan, and N. Jojic. Efficient ranking from pairwise comparisons. In International Conference on Machine Learning, pages 109–117. PMLR, 2013.

## A Examples of variable graphs and the induced partition

Figure 4 shows five variable graphs with the vertex set $V = \{ 1 , 2 , 3 \}$ . Each variable graph V defines a collection of reference sets $P _ { v }$ for $v \in V$ , where $P _ { v }$ corresponds to the set of parents of v in V. The associated RRV $\pmb { X } = ( X _ { 1 } , X _ { 2 } , X _ { 3 } )$ for V is defined by

$$
X _ { v } ( \sigma ) : = r _ { P _ { v } } ^ { v } ( \sigma ) , \qquad v \in V .
$$

The variable graphs and the reference sets that define the corresponding RRVs are:

$$
\bullet \ \mathcal { V } _ { 1 } = \{ ( 2 , 1 ) , ( 2 , 3 ) \} \colon P _ { 1 } = \{ 2 \} , P _ { 2 } = \emptyset , P _ { 3 } = \{ 2 \} .
$$

$$
\bullet \mathcal { V } _ { 2 } = \{ ( 1 , 2 ) , ( 3 , 2 ) \} \colon P _ { 1 } = \emptyset , P _ { 2 } = \{ 1 , 3 \} , P _ { 3 } = \emptyset .
$$

$$
\bullet \ \mathcal { V } _ { 3 } = \{ ( 1 , 2 ) , ( 1 , 3 ) \} \colon P _ { 1 } = \emptyset , P _ { 2 } = \{ 1 \} , P _ { 3 } = \{ 1 \} .
$$

$$
\bullet \ \mathcal { V } _ { 4 } = \{ ( 1 , 2 ) , ( 2 , 3 ) , ( 3 , 1 ) \} \colon P _ { 1 } = \{ 3 \} , P _ { 2 } = \{ 1 \} , P _ { 3 } = \{ 2 \} .
$$

$$
\bullet \ \mathcal { V } _ { 5 } = \{ ( 2 , 1 ) , ( 3 , 1 ) , ( 3 , 2 ) \} \colon P _ { 1 } = \{ 2 , 3 \} , P _ { 2 } = \{ 3 \} , P _ { 3 } = \emptyset .
$$

![](images/9559e68d37834dc5761d5b043b899d2e95997af4d684b7e16650614171f03fee.jpg)  
Figure 4: Five variable graphs defined over items $V = \{ 1 , 2 , 3 \}$

Table 1 shows the relation between rankings $\sigma \in S _ { n } ^ { V }$ (and orderings $\bar { \sigma } \in S _ { V } ^ { n } )$ and the realizations $\textbf { \em x } \in { \textbf { \em x } }$ of the RRVs defined by these graphs. For each variable graph, the table displays the cover ⟨X ⟩ of the ranking space $\cdot _ { s _ { n } ^ { V } }$ , where each realization $x \in \mathcal { X }$ and its corresponding event ⟨x⟩ is identified by a distinct color. For example, for $\mathcal { V } _ { 1 }$ and realization $\pmb { x } = ( 2 , 1 , 2 )$ , the event $\langle \pmb { x } \rangle = \{ ( 2 , 1 , 3 ) , ( 3 , 1 , 2 ) \}$ is highlighted in blue.

<table><tr><td>σ</td><td>σ</td><td>V1</td><td>V2 x =</td><td>V3 (x1, x2, x3)</td><td>V4</td><td>V5</td></tr><tr><td>1,2,3</td><td>1,2,3</td><td>1,1,2</td><td>1,2,1</td><td>1,2,2</td><td>1,2,2</td><td>1,1,1</td></tr><tr><td>1,3,2</td><td>1,3,2</td><td>1,1,1</td><td>1,3,1</td><td>1,2,2</td><td>1,2,1</td><td>1,2,1</td></tr><tr><td>2,1,3</td><td>2,1,3</td><td>2,1,2</td><td>1,1,1</td><td>1,1,2</td><td>1,1,2</td><td>2,1,1</td></tr><tr><td>2,3,1</td><td>3,1,2</td><td>1,1,1</td><td>1,3,1</td><td>1,2,1</td><td>2,2,1</td><td>2,2,1</td></tr><tr><td>3,1,2</td><td>2,3,1</td><td>2,1,2</td><td>1,1,1</td><td>1,1,1</td><td>2,1,2</td><td>3,1,1</td></tr><tr><td>3,2,1</td><td>3,2,1</td><td>2,1,1</td><td>1,2,1</td><td>1,1,1</td><td>2,1,1</td><td>3,2,1</td></tr></table>

Table 1: Relative rank variable realizations ${ \pmb x } = ( x _ { 1 } , x _ { 2 } , x _ { 3 } )$ for the rankings $\sigma \in S _ { n } ^ { V }$ and orderings $\bar { \sigma } \in S _ { V } ^ { n }$ , defined according to the five variable graphs shown in Figure 4. The graph $\nu _ { 4 }$ is cyclic and the realizations $\pmb { x } \in \{ ( 1 , 1 , 1 ) , ( 2 , 2 , 2 ) \}$ } do not appear in the table because they correspond to empty events.

The graph $\nu _ { 4 }$ is cyclic and therefore its associated RRV is subject to MEC, which implies that some realizations of the associated RRV correspond to empty events. In particular, the cyclic dependencies among the variables impose mutually inconsistent constraints on the relative ranks, so that some configurations cannot be satisfied by any ranking. As a consequence, the realizations ${ \pmb x } = ( 1 , 1 , 1 )$ and $\bar { \pmb { x } } = ( 2 , 2 , 2 )$ correspond to empty events, $\mathfrak { i } . \mathfrak { e } . , \langle \pmb { x } \rangle = \emptyset$ (see Theorem 2.5). This prevents $\nu _ { 4 }$ from inducing a proper partition of the ranking space.

The graph $\nu _ { 5 }$ is a complete acyclic variable graph and therefore induces a thin partition of the ranking space. By Proposition 6.1, every realization ${ \bar { \mathbf { x } } } \in { \mathcal { X } }$ corresponds to a singleton event, so that each ranking $\bar { \sigma } \in \bar { S _ { n } ^ { V } }$ is uniquely identified by its image ${ \pmb x } = { \pmb X } ( \sigma )$ . As a consequence, the RRVs defined by complete acyclic variable graphs define a bijection between the ranking space and their state space, and the RRMs are ranking models, $\Delta ( { \pmb x } ) = \Delta ( S _ { n } ^ { V } )$ . In particular, any ranking model p can be represented exactly as an RRM $h = \mathrm { p u s h } ( p ) , \mathrm { s a t i s f y i n g } p ( \boldsymbol { \sigma } ) = h ( \boldsymbol { X } ( \boldsymbol { \sigma } ) )$ for all $\bar { \sigma } \in S _ { n } ^ { V }$ , and $\begin{array} { r } { p ( \pmb { x } ) \overset { \cdot } { = } \sum _ { \sigma \in \langle \pmb { x } \rangle } p ( \sigma ) \overset { \cdot } { = } h ( \pmb { x } ) } \end{array}$ for all $\pmb { x } \in \pmb { \mathscr { X } }$

## B Generating the events

Thanks to the acyclicity of the variable graph, all the rankings belonging to a given event can be enumerated using Algorithm 1. The algorithm is an insertion-based procedure that follows an ancestral ordering of the variable graph to incrementally construct a set of partial rankings. At each step, a new item is inserted into each ranking from the previous iteration at positions determined by its associated relative rank.

```latex
Algorithm 1 Event construction from an RRV realization
Input: A realization x of an RRV for ${ \overline { { V , X } } } .$ , defined by an acyclic variable graph V with an ancestral
ordering $\bar { \pi } \in \mathrm { a n c } ( \mathcal { V } )$
Output: The non-empty event associated to x, $\langle \pmb { x } \rangle = \mathcal { R } _ { n }$
1: Choose an ordering $\bar { \pi } \in$ anc(V)
2: Initialize $\mathcal { R } _ { 1 }  \{ ( 1 ) \} = { S } _ { 1 } ^ { \{ \bar { \pi } ( 1 ) \} }$
3: for $i = 2 , \ldots , n$ do
4: Initialize $\mathcal { R } _ { i }  \mathcal { O }$
5: Let $v  \bar { \pi } ( i )$
6: for each $\alpha \in \mathcal { R } _ { i - 1 }$ do
7: if $P _ { v } ^ { \nu } = \varnothing$ then
8: $I _ { v } \gets \{ 1 , \ldots , i \}$
9: else $\mathbf { i f } \ x _ { v } = 1$ then
10: $I _ { v }  \{ 1 , . . . , \alpha ( \bar { \alpha } ^ { P _ { v } ^ { v } } ( 1 ) ) \}$
11: else if $\bar { x _ { v } } \in \{ 2 , \ldots , | P _ { v } ^ { \mathcal { V } } | \}$ then
12: $I _ { v } \gets \left\{ \alpha \big ( \bar { \alpha } ^ { P _ { v } ^ { \mathcal { V } } } ( x _ { v } - 1 ) \big ) + 1 , \dots , \alpha \big ( \bar { \alpha } ^ { P _ { v } ^ { \mathcal { V } } } ( x _ { v } ) \big ) \right\}$
13: else
14: $I _ { v } \gets \{ \alpha ( \bar { \alpha } ^ { P _ { v } ^ { \mathcal { V } } } ( | P _ { v } ^ { \mathcal { V } } | ) ) + 1 , \dots , i \}$
15: end if
16: $\mathcal { R } _ { i }  \mathcal { R } _ { i } \cup \{ \mathrm { i n s e r t } ( v , r , \alpha ) : r \in I _ { v } \}$
17: end for
18: end for
19: return $\mathcal { R } _ { n }$
```

Definition B.1 (Insertion). Let $U \subsetneq V$ with $m = | U |$ , let $\alpha \in \mathcal { S } _ { m } ^ { U }$ be a (partial) ranking on U, let $v \in V \setminus U$ , and let $r \in \{ 1 , \ldots , m + 1 \}$ . The insertion operator

$$
\mathrm { i n s e r t } ( v , r , \alpha ) = \beta \in S _ { m + 1 } ^ { U \cup \{ v \} }
$$

is defined by

$$
\beta ( v ) = r , { \mathrm { ~ a n d ~ } } \beta ( u ) = \alpha ( u ) + \mathbf { 1 } \{ \alpha ( u ) \geq r \} { \mathrm { ~ f o r ~ } } u \in U .
$$

The resulting ranking on $U \cup \{ v \} , \beta ,$ corresponds to the ordering

$$
\bar { \beta } = [ \bar { \alpha } ( 1 ) , \ldots , \bar { \alpha } ( r - 1 ) , v , \bar { \alpha } ( r ) , \ldots , \bar { \alpha } ( m ) ] .
$$

We consider the direct extensions of Definitions 2.1 and 2.2 to partial rankings containing the item and its reference set.

Lemma B.2 (Relative rank preservation). Consider the set of items $U \subsetneq V$ with size $m = | U | ,$ the (partial) ranking $\alpha \in S _ { m } ^ { \dot { U } } ,$ the item $v \in V \setminus U ,$ , the rank $r \in \{ 1 , \ldots , m + 1 \}$ , and the ranking $\beta = i n s e r t ( v , r , \alpha ) \in S _ { m + 1 } ^ { U \cup \{ v \} }$ . Then, for any $u \in U$ and any $P \subseteq U \setminus \{ u \}$

$$
r _ { P } ^ { u } ( \beta ) = r _ { P } ^ { u } ( \alpha ) .
$$

Proof. Fix $u \in U$ and $P \subseteq U \setminus \{ u \}$ . Define the sets

$$
A = \{ w \in P : \alpha ( w ) < r \} , \qquad B = \{ w \in P : r \leq \alpha ( w ) \} .
$$

By Definition B.1, we have:

• if w ∈ A, then $\beta ( w ) = \alpha ( w )$

• if $w \in B$ , then $\beta ( w ) = \alpha ( w ) + 1$

We consider two cases for $\alpha ( u )$

i) $\alpha ( u ) < r \colon$

Then $\beta ( u ) = \alpha ( u )$ , and

$$
r _ { P } ^ { u } ( \beta ) = 1 + \sum _ { w \in A } \mathbf { 1 } \{ \beta ( w ) < \beta ( u ) \} + \sum _ { w \in B } \mathbf { 1 } \{ \beta ( w ) < \beta ( u ) \} .
$$

Substituting the definition of $\beta ,$

$$
r _ { P } ^ { u } ( \beta ) = 1 + \sum _ { w \in A } \mathbf { 1 } \{ \alpha ( w ) < \alpha ( u ) \} + \sum _ { w \in B } \mathbf { 1 } \{ \alpha ( w ) + 1 < \alpha ( u ) \} .
$$

Since $\alpha ( u ) < r \leq \alpha ( w )$ for all $w \in B .$ , we have

$$
r _ { P } ^ { u } ( \beta ) = 1 + \sum _ { w \in \cal { A } } { \bf 1 } \{ \alpha ( w ) < \alpha ( u ) \} + \sum _ { w \in \cal { B } } { \bf 1 } \{ \alpha ( w ) < \alpha ( u ) \} = r _ { P } ^ { u } ( \alpha ) .
$$

ii) $\alpha ( u ) \geq r ;$

Then $\beta ( u ) = \alpha ( u ) + 1$ , and

$$
r _ { P } ^ { u } ( \beta ) = 1 + \sum _ { w \in A } \mathbf { 1 } \{ \beta ( w ) < \beta ( u ) \} + \sum _ { w \in B } \mathbf { 1 } \{ \beta ( w ) < \beta ( u ) \} .
$$

Substituting,

$$
r _ { P } ^ { u } ( \beta ) = 1 + \sum _ { w \in A } \mathbf { 1 } \{ \alpha ( w ) < \alpha ( u ) + 1 \} + \sum _ { w \in B } \mathbf { 1 } \{ \alpha ( w ) + 1 < \alpha ( u ) + 1 \} .
$$

Since $\alpha ( w ) < r \leq \alpha ( u )$ for all $w \in A$ , we have

$$
r _ { P } ^ { u } ( \beta ) = 1 + \sum _ { w \in A } \mathbf { 1 } \{ \alpha ( w ) < \alpha ( u ) \} + \sum _ { w \in B } \mathbf { 1 } \{ \alpha ( w ) < \alpha ( u ) \} = r _ { P } ^ { u } ( \alpha ) .
$$

The next result shows that for any $\pmb { x } \in \pmb { \mathcal { X } }$ Algorithm 1 generates all the rankings in the event $\langle x \rangle$

Proposition B.3. Let X be an RRV defined by an acyclic variable graph V. Then, given a realization $\pmb { x } \in \pmb { \mathscr { X } }$ , Algorithm 1 generates the rankings in the event $\langle { \pmb x } \rangle \neq \emptyset .$

Proof. Let X be an RRV defined by V with the reference sets $P _ { v } = P _ { v } ^ { \nu }$ for $v = 1 , \ldots , n$ . We have to prove that $\langle \pmb { x } \rangle = \mathcal { R } _ { n }$ for arbitrary $\pmb { x } \in \pmb { \mathscr { X } }$

V is a DAG and thus there exists $\bar { \pi } \in \operatorname { a n c } ( \mathcal { V } )$ . Without loss of generality, assume that $\bar { \pi } = [ 1 , \ldots , n ]$ By definition of an ancestral ordering, the reference set $P _ { v } \subseteq \{ \overline { { \bar { \pi } } } ( 1 ) , \dots , \bar { \pi } ( v - 1 ) \}$ , for $v = 1 , \ldots , n$ We prove the result by induction over $i = 1 , \ldots , n \colon$ for $v = \bar { \pi } ( i ) = i$ , the event $\langle \pmb { x } _ { \{ 1 , . . . , v \} } \rangle =$ $\textstyle \bigcup _ { \alpha \in { \mathcal { R } } _ { i } } \left. \alpha \right.$ , where $\langle \alpha \rangle = \{ \sigma : \sigma ^ { \{ 1 , \ldots , v \} } = \alpha \}$ and $\left| \mathcal { R } _ { i } \right| \geq 1$

Base case $( i = v = 1 )$

The parents of v in V are $P _ { 1 } = \emptyset$ . Then ${ \mathcal { X } } _ { v } = \{ 1 \}$ and thus $x _ { v } = 1$ with $\langle x _ { v } \rangle = S _ { n } ^ { V }$

Using Algorithm 1, we have $\begin{array} { r } { \mathcal { R } _ { 1 } = \{ ( 1 ) \} = S _ { 1 } ^ { \{ 1 \} } , \mathrm { a n d } \bigcup _ { \alpha \in \mathcal { R } _ { 1 } } \langle \alpha \rangle = \langle ( 1 ) \rangle = S _ { n } ^ { V } = \langle x _ { v } \rangle . } \end{array}$

For $1 \leq i < n ,$ , let us assume that $\begin{array} { r } { \langle \pmb { x } _ { \{ 1 , \dots , i \} } \rangle = \bigcup _ { \alpha \in \mathcal { R } _ { i } } \langle \alpha \rangle } \end{array}$ where $\mathcal { R } _ { i } \subseteq { S _ { i } ^ { \{ 1 , \dots , i \} } }$ with $| \mathcal { R } _ { i } | \geq 1$ . Let $v = \bar { \pi } ( i + 1 ) = i + 1$

The ancestral ordering is $\bar { \pi } ~ = ~ [ 1 , \ldots , n ]$ and $P _ { v } \subseteq \{ 1 , \dotsc , i \}$ . By Definition 2.2 we have $\langle \pmb { x } _ { \{ 1 , . . . , v \} } \rangle = \langle \pmb { x } _ { v } \rangle \cap \langle \pmb { x } _ { \{ 1 , . . . , i \} } \rangle$ and thus

$$
\langle \pmb { x } _ { \{ 1 , \dots , v \} } \rangle = \bigcup _ { \boldsymbol { \alpha } \in \mathcal { R } _ { i } } \langle \boldsymbol { \alpha } \rangle \cap \langle \pmb { x } _ { v } \rangle = \bigcup _ { \boldsymbol { \alpha } \in \mathcal { R } _ { i } } \langle \boldsymbol { \alpha } , \boldsymbol { x } _ { v } \rangle ,
$$

where $\langle x _ { v } \rangle = \{ \sigma : r _ { P _ { v } } ^ { v } ( \sigma ) = x _ { v } \}$ and $\langle \alpha , x _ { v } \rangle = \{ \sigma : \sigma ^ { \{ 1 , \ldots , i \} } = \alpha , r _ { P _ { v } } ^ { v } ( \sigma ) = x _ { v } \}$ . To represent $\langle \pmb { x } _ { \{ 1 , . . . , v \} } \rangle$ , we have to construct the set of partial rankings $\{ \beta \in S _ { v } ^ { \{ 1 , \ldots , v \} } : \beta ^ { \{ 1 , \ldots , i \} } = \alpha , r _ { P _ { v } , } ^ { v } ( \beta ) =$ $\boldsymbol { x } _ { v } \boldsymbol  \}$ for each partial ranking $\alpha \in \mathcal { R } _ { i }$ . By Definition B.1 and Lemma B.2 these partial rankings are obtained by inserting v in α in the ranks $I _ { v } = \{ r \in \{ 1 , \ldots , i + 1 \} : \sum _ { u \in P _ { v } } \mathbb { 1 } \big ( \alpha ( u ) < r \big ) = x _ { v } - 1 \}$

$$
\langle \alpha , x _ { v } \rangle = \bigcup _ { r \in I _ { v } } \langle \operatorname { i n s e r t } ( v , r , \alpha ) \rangle .
$$

Note that for $r \in I _ { v }$ and $\beta = \operatorname { i n s e r t } ( v , r , \alpha )$ we have

$$
X _ { v } ( \beta ) = r _ { P _ { v } } ^ { v } ( \beta ) = 1 + \sum _ { w \in P _ { v } } \mathbb { 1 } \big ( \beta ( w ) < \beta ( v ) \big ) = 1 + \sum _ { w \in P _ { v } } \mathbb { 1 } \big ( \beta ( w ) < r \big ) = x _ { v } .
$$

Given $\alpha \in S _ { i } ^ { \{ 1 , \ldots , i \} }$ , the set of insertion ranks $I _ { v }$ satisfying the constraint $\begin{array} { r } { \sum _ { u \in P _ { v } } \mathbb { 1 } \big ( \alpha ( u ) < r \big ) = } \end{array}$ $x _ { v } - 1 \mathrm { i s } \left\{ 1 , \ldots , i + 1 \right\}$ if $P _ { v } = \mathcal { O }$ . Otherwise:

$$
\bullet I _ { v } = \{ 1 , \ldots , \alpha ( \bar { \alpha } ^ { P _ { v } } ( 1 ) ) \} , \mathrm { i f } x _ { v } = 1
$$

$$
\bullet I _ { v } = \{ \alpha ( \bar { \alpha } ^ { P _ { v } } ( x _ { v } - 1 ) ) + 1 , \ldots , \alpha ( \bar { \alpha } ^ { P _ { v } } ( x _ { v } ) ) \} , \mathrm { i f } \ 1 < x _ { v } \leq | P _ { v } |
$$

• and $I _ { v } = \{ \alpha ( \bar { \alpha } ^ { P _ { v } } ( | P _ { v } | ) ) + 1 , \ldots , i + 1 \} , \mathrm { i f } x _ { v } = | P _ { v } | + 1 .$

Finally, the set of insertion ranks satisfies $\left| I _ { v } \right| \geq 1$ and thus $| \mathcal { R } _ { v } | \geq 1$ , for $v = 2 , \ldots , n$ . Thus, $\langle { \pmb x } \rangle = \mathcal { R } _ { n } \neq \emptyset$ □

## C Admissible models

In this appendix we consider RRVs defined according to arbitrary variable graphs and we characterize admissible RRMs through the pushforward relation with ranking models. Recall that an RRM is admissible if and only if it is the pushforward of at least one ranking model. We first provide a constructive description of the pullback of an admissible RRM, and then derive a simple necessary and sufficient condition for admissibility based on the support of the model. This characterization shows that admissibility imposes a structural constraint on the model: An RRM must assign zero probability to all realizations associated with empty events.

Lemma C.1 (Pullback ranking models). Let X be an RRVfor V and h be an RRMfor X, and let $p \in \Delta ( S _ { n } ^ { V } )$ be a ranking model. Then, $p \in p u l l ( h )$ if and only if for all $\pmb { x } \in \pmb { \mathcal { X } }$ with $h ( { \pmb x } ) > 0$ there exists $p _ { { \pmb x } } \in \Delta ( \langle { \pmb x } \rangle )$ ) such that

$$
p ( \sigma ) = h ( { \pmb x } ) p _ { { \pmb x } } ( \sigma ) ,\tag{8}
$$

for $\sigma \in \langle x \rangle$

Proof. By Definition $3 . 2 , p \in \mathrm { p u l l } ( h )$ if and only if $h = \mathrm { p u s h } _ { X } ( p )$

i) Suppose that for every x with $h ( { \pmb x } ) > 0$ there exists $p _ { \pmb { x } } \in \Delta ( \langle \pmb { x } \rangle )$ satisfying Eq. 8. Summing within each such event gives

$$
\sum _ { \sigma \in \langle { \pmb x } \rangle } p ( \sigma ) = h ( { \pmb x } ) \sum _ { \sigma \in \langle { \pmb x } \rangle } p _ { \pmb x } ( \sigma ) = h ( { \pmb x } ) .
$$

These events are disjoint, and their total probability under p is $\scriptstyle \sum _ { \pmb { x } : h ( \pmb { x } ) > 0 } h ( \pmb { x } ) = 1$ . Since p is a probability distribution, every remaining event has probability zero. Hence $\mathrm { p u s h } _ { X } ( p ) = h$

ii) Suppose that $h = \mathrm { p u s h } _ { X } ( p )$ . For each x with $h ( { \pmb x } ) > 0$ , define $p _ { \pmb { x } } ( \sigma ) = p ( \sigma ) / h ( \pmb { x } )$ for $\sigma \in \langle x \rangle$ This is a probability distribution on ⟨x⟩, because its values are nonnegative and sum to one, and it satisfies Eq. 8.

If $h ( { \pmb x } ) = 0$ , then $p ( \sigma ) = 0$ for every $\sigma \in \langle x \rangle$ . For $\langle { \pmb x } \rangle \neq \emptyset$ , any choice of $p _ { \pmb { x } } \in \Delta ( \langle \pmb { x } \rangle )$ satisfies Eq. 8; if $\langle { \pmb x } \rangle = \emptyset$ , no $p _ { \pmb { x } }$ needs to be defined. □

Using Eq. 8, we can generate ranking models in the pullback of any admissible RRM by choosing a distribution $p _ { { \pmb x } }$ within each event with $h ( { \pmb x } ) > 0 .$ A particularly simple choice that does not require extra parameters assigns equal probability to all rankings within each such event.

The next result provides a simple rule to characterize the admissible models.

Proposition C.2 (Structural zeroes). Let X be an RRVfor V. An RRM $h \in \Delta ( { \mathcal { X } } )$ is admissible if and only $i f h ( { \pmb x } ) = 0 f o r$ all $\pmb { x } \in \pmb { \mathscr { X } }$ with $\langle \pmb { x } \rangle = \emptyset$

Proof. The proof has two parts.

i) If h is admissible then $h ( { \pmb x } ) = 0$ for all $\pmb { x } \in \pmb { \mathscr { X } }$ with $\langle { \pmb x } \rangle = { \pmb \ O } \rangle$

If $h \in \Delta ( { \mathcal { X } } )$ is admissible by Definition 3.3 then there exists $p \in \Delta ( S _ { n } ^ { V } )$ such that $h ( { \pmb x } ) = p ( { \pmb x } ) : =$ $\textstyle \sum _ { \sigma \in \langle { \pmb x } \rangle } { \dot { p } } ( \sigma )$ for all $\pmb { x } \in \pmb { \mathcal { X } }$ . Therefore, for $\pmb { x } ^ { \mathcal { O } } \in \pmb { \mathcal { X } }$ with $\langle \pmb { x } ^ { \tilde { \mathcal { O } } } \rangle = \dot { \mathcal { O } }$ we have $h ( \mathbf { x } ^ { \mathcal { O } } ) \overset { \cdot } { = } p ( \overset { \cdot } { \mathbf { x } } ^ { \mathcal { O } } ) \overset { \cdot } { = } 0$ ii) If $h ( { \pmb x } ) = 0$ for all $\pmb { x } \in \pmb { \mathscr { X } }$ with $\langle { \pmb x } \rangle = \emptyset$ then h is admissible:

By Definition 3.3, h is admissible if there exists $p \in { \mathrm { p u l l } } ( h )$ . According to Lemma C.1 $p \in { \mathrm { p u l l } } ( h )$ if and only if

$$
p ( \sigma ) = h ( { \pmb x } ) p _ { { \pmb x } } ( \sigma ) ,
$$

where $p _ { \pmb { x } } \in \Delta ( \langle \pmb { x } \rangle )$ for $\pmb { x } \in \pmb { \mathcal { X } }$ with $h ( { \pmb x } ) > 0$

We obtain a $p \ \in \ \mathsf { p u l l } ( h )$ by construction. For $\textbf { \em x } \in \textbf { \em x }$ with $h ( { \pmb x } ) \ > \ 0 .$ , define the uniform distribution $p _ { \pmb { x } } ~ \in ~ \Delta ( \langle \pmb { x } \rangle )$ as $p _ { \pmb { x } } ( \sigma ) \ = \ 1 / | \langle \pmb { x } \rangle$ | if $\sigma ~ \in ~ \langle x \rangle$ and $p _ { \mathbf { x } } ( \sigma ) ~ = ~ 0$ otherwise. Set $p ( \sigma ) \ : = \ : h ( X ( \sigma ) ) p _ { X ( \sigma ) } ( \sigma )$ when $h ( X ( \sigma ) ) { \overset { } { > } } { \overset { \partial } { \partial } }$ , and $p ( \sigma ) = 0$ otherwise. Then $p \ \in \Delta ( { \mathcal { S } } _ { n } ^ { V } )$ because $p ( \sigma ) \geq 0$ and

$$
\sum _ { \sigma \in S _ { n } ^ { V } } p ( \sigma ) = \sum _ { \substack { x : ( x ) \neq \emptyset } } \sum _ { \sigma \in \langle x \rangle } p ( \sigma ) = \sum _ { \stackrel { x : ( x ) \neq \emptyset , } { h ( x ) > 0 } } h ( x ) \sum _ { \sigma \in \langle x \rangle } p _ { \alpha } ( \sigma ) = \sum _ { \stackrel { x : ( x ) \neq \emptyset , } { h ( x ) > 0 } } h ( x ) = 1 ,
$$

where the last equality holds because $h ( { \pmb x } ) = 0$ for all x with $\langle { \pmb x } \rangle = \emptyset$ . Therefore $p \in { \mathrm { p u l l } } ( h )$ and $\mathrm { p u l l } ( h ) \neq \emptyset$ □

All RRMs defined over variables that break MEC are admissible, since every realization corresponds to a non-empty event. In contrast, when RRVs are subject to MEC, admissibility requires assigning zero probability to all realizations associated with empty events.

## D Admissible models and conditional independence

When an RRV is defined by an acyclic variable graph, it breaks MEC, and all realizations correspond to non-empty events. In this case, admissibility does not require additional constraints, and conditional independence relations can be imposed without requiring structural zeros in the associated RRMs.

In contrast, when an RRV is defined by a cyclic variable graph, it is subject to MEC, and some realizations correspond to empty events. In this setting, certain conditional independence relations require imposing structural zeros on the associated RRMs in order to ensure admissibility. Moreover, these constraints propagate to ranking models in the pullback, potentially requiring structural zeros over an exponential number of rankings.

The following result illustrates this phenomenon.

Theorem D.1. Let $A , B , C$ be three pairwise disjoint nonempty subsets of V . Let X be an RRV for V with $\pmb { x } ^ { \alpha } \in \pmb { x }$ so that $\langle \pmb { x } _ { A , B , C } ^ { \alpha } \rangle = \mathrm { \hat { \Gamma } }$

A ranking model $p \in \Delta ( S _ { n } ^ { V } )$ ) satisfies $X _ { A } \perp _ { p } X _ { B } \mid X _ { C }$ only if

$$
p ( \pmb { x } _ { C } ^ { \mathcal { O } } ) = 0 \quad o r \quad p ( \pmb { x } _ { A , C } ^ { \mathcal { O } } ) = 0 \quad o r \quad p ( \pmb { x } _ { B , C } ^ { \mathcal { O } } ) = 0 .
$$

Proof. We argue by contradiction. Suppose that there exists $p \ \in \ \Delta ( S _ { n } ^ { V } )$ , with $p ( \pmb { x } _ { C } ^ { \mathcal { O } } ) \ > \ 0$ $p ( \pmb { x } _ { A , C } ^ { \mathcal { O } } ) > 0$ and $p ( \pmb { x } _ { B , C } ^ { \mathcal { O } } ) > 0$

The event associated to the realization of $\pmb { x } _ { A , B , C } ^ { \mathcal { O } }$ is empty, $\langle \pmb { x } _ { A , B , C } ^ { \mathcal { O } } \rangle = \emptyset$ , and thus

$$
p ( \pmb { x } _ { A , B , C } ^ { \mathcal { O } } ) = \sum _ { \sigma \in \langle \pmb { x } _ { A , B , C } ^ { \mathcal { O } } \rangle } p ( \sigma ) = 0 .
$$

In addition, by Definition 4.1, the conditional independence $X _ { A } ~ \bot _ { p } ~ X _ { B } ~ \vert ~ X _ { C }$ implies that $p ( \pmb { x } _ { A , B } | \pmb { x } _ { C } ) \overset { \cdot } { = } p ( \pmb { x } _ { A } | \pmb { x } _ { C } ) p ( \pmb { x } _ { B } | \pmb { x } _ { C } )$ for all $\pmb { x } _ { A , B , C } \in \pmb { \mathcal { X } } _ { A , B , C }$ with $p ( \bar { \mathbfcal { x } } _ { C } ) > 0$

Since $p ( \pmb { x } _ { C } ^ { \mathcal { O } } ) > 0$ , we have

$$
p ( \pmb { x } _ { A , B , C } ^ { \mathcal { O } } ) = \frac { p ( \pmb { x } _ { A , C } ^ { \mathcal { O } } ) p ( \pmb { x } _ { B , C } ^ { \mathcal { O } } ) } { p ( \pmb { x } _ { C } ^ { \mathcal { O } } ) } > 0 .
$$

This contradicts $p ( \pmb { x } _ { A , B , C } ^ { \alpha } ) = 0$

Note that $\langle \pmb { x } _ { A , C } ^ { \alpha } \rangle \subseteq \langle \pmb { x } _ { C } ^ { \alpha } \rangle$ and $\langle \pmb { x } _ { B , C } ^ { \alpha } \rangle \subseteq \langle \pmb { x } _ { C } ^ { \alpha } \rangle$ . This result implies that p must assign zero probability to all rankings in $\langle \pmb { x } _ { A , C } ^ { \alpha } \rangle$ or in $\langle \pmb { x } _ { B , C } ^ { \alpha } \rangle$ . The number of such rankings can be exponential in the size of V, making these constraints highly restrictive.

This phenomenon extends the notion of probabilistic independence for rankings introduced in [Huang et al., 2009b]. In that framework, independence between subsets of items is defined directly in terms of rankings, leading to strong structural constraints. In comparison, RRVs provide a more flexible representation, but when the MEC are present, enforcing conditional independence may still require a large number of structural zeros.

In contrast, for RRVs that break MEC, all realizations correspond to non-empty events, admissibility imposes no structural restrictions, and conditional independence assumptions lead to valid factorizations of RRMs. Consequently, acyclic RRVs provide the appropriate foundation for developing graphical models over rankings. The separation between the construction of the partition and the imposition of independence assumptions is a key advantage of RRMs.

## E Learning relative rank networks

Let $\mathcal { D } = ( \sigma ^ { ( 1 ) } , \dots , \sigma ^ { ( M ) } )$ be a sample of M rankings, and let $\hat { p }$ denote its empirical distribution. For a fixed acyclic variable graph V, transform the sample into the categorical data ${ \mathcal D } _ { \mathbf X } = ( { \pmb X } ( \sigma ^ { ( 1 ) } ) , \dots , { \pmb X } ( \sigma ^ { ( M ) } ) )$ . A factorization graph $\mathcal { G }$ can then be learned using standard score-based, constraint-based, or hybrid Bayesian-network structure-learning methods [Koller and Friedman, 2009]. For fixed $( \nu , \mathcal { G } )$ , estimating the local conditional distributions by empirical frequencies gives $h = \mathrm { p u s h } _ { \mathcal { V } , \mathcal { G } } ( \hat { p } )$ , with the convention for unobserved parent configurations specified in Definition 5.2.

Selecting V requires comparing different coarsenings of the ranking space. The selection criterion should account for the ranking information retained and the number of model parameters. Raw RRM likelihoods alone do not provide this comparison: an empty variable graph maps every observation to the same realization, which can be assigned probability one, giving log-likelihood zero regardless of the sample. Algorithm 2 therefore takes a model-selection criterion as an input. Defining and evaluating such a criterion remains future work.

Algorithm 2 A general procedure for learning an RRN   
Input: Ranking sample D; a finite non-empty collection V of candidate acyclic variable graphs;   
a categorical Bayesian-network structure learner; a model-selection criterion, with higher values   
preferred   
Output: A selected RRN $( \gamma ^ { * } , \mathcal { G } ^ { * } , \pmb \theta ^ { * } )$   
1: for each $\nu \in \mathfrak { V }$ do   
2: Define X from V and form $\mathcal { D } _ { \boldsymbol { X } } = ( \boldsymbol { X } ( \sigma ^ { ( j ) } ) ) _ { j = 1 } ^ { M }$   
3: Learn a factorization graph G from $\mathcal { D } _ { X }$ using the supplied structure learner   
4: Estimate θ from empirical conditional frequencies, as in Definition 5.2   
5: Evaluate $( \nu , \mathcal { G } , \pmb \theta )$ using the supplied model-selection criterion   
6: end for   
7: Select a highest-scoring RRN $( \gamma ^ { * } , \mathcal { G } ^ { * } , \pmb \theta ^ { * } )$ among the fitted candidates   
8: return $( \breve { \nu ^ { * } } , \mathcal { G } ^ { * } , \pmb { \theta } ^ { * } )$   
The candidate variable graphs may be restricted by their maximum number of parents to control RRV   
cardinalities. Likewise, the learner for G may impose indegree or treewidth constraints to control   
parameter counts or inference cost. More elaborate searches could alternate local changes in V and   
G, recomputing the categorical data when V changes. Developing and empirically evaluating these   
searches remains future work.

## F Proofs of Theoretical Results

This appendix provides the proofs of the theoretical results in the main text.

## F.1 Proof of Theorem 2.5 (Breaking MEC)

Proof. i) If X breaks MEC, then V is acyclic (by contradiction):

Assume that X is defined by a cyclic graph V and breaks MEC. Suppose for contradiction that V has a directed cycle

$$
u _ { 1 } , . . . , u _ { l } , u _ { 1 }
$$

for some length $1 \leq l \leq n$ . Let $U = \{ u _ { 1 } , \ldots , u _ { l } \}$ be the set of items in the cycle. Now define a realization $\pmb { x } _ { U } ^ { \emptyset } = ( 1 , 1 , . . . , 1 )$ that forces a strict cyclic pairwise preference. Since $u _ { i } \in P _ { u _ { i + } }$ for $i = 1 , \ldots , l - 1$ and $u _ { l } \in P _ { u _ { 1 } } , \pmb { x } _ { U } ^ { \alpha }$ implies

$$
\sigma ( u _ { 1 } ) > \sigma ( u _ { 2 } ) > \cdots > \sigma ( u _ { l } ) > \sigma ( u _ { 1 } )
$$

for any $\sigma \in \langle \pmb { x } _ { U } ^ { \mathcal { D } } \rangle$ , which is impossible. So $\langle \pmb { x } _ { U } ^ { \mathcal { B } } \rangle = \mathcal { O } .$

Now take a realization $\pmb { x } = ( \pmb { x } _ { V \backslash U } , \pmb { x } _ { U } ^ { \otimes } )$ . We have

$$
\langle \pmb { x } \rangle = \langle \pmb { x } _ { V \backslash U } \rangle \cap \langle \pmb { x } _ { U } ^ { \mathcal { Q } } \rangle \subseteq \langle \pmb { x } _ { U } ^ { \mathcal { Q } } \rangle = \emptyset ,
$$

which contradicts the assumption that X breaks MEC. Therefore, V must be acyclic.

ii) If V is acyclic, then X breaks MEC:

If X is defined by an acyclic variable graph, then by Proposition B.3 for any $\textbf { \em x } \in { \mathcal { X } }$ we have $| \langle { \pmb x } \rangle | \geq 1$ , and thus X breaks MEC. □

## F.2 Proof of Lemma 5.4

Proof. By Theorem 2.5, an RRV X defined by an acyclic variable graph V breaks MEC. Then, by Proposition C.2, any RRN h for X is admissible, that is, pul $( h ) \neq \bar { \infty }$

By Definition 5.1, h factorizes according to ${ \mathcal { G } } .$

On one hand, according to Theorem 3.3 in Koller and Friedman [2009], if a distribution h factorizes according to $\mathcal { G }$ and the sets A and B are d-separated by C in ${ \mathcal { G } } .$ , then $X _ { A \perp _ { h } } X _ { B } | X _ { C }$ . On the other hand, $p \in { \mathrm { p u l l } } ( h )$ , and thus $h = \mathrm { p u s h } _ { X } ( p )$ , and by Definition 4.1 $X _ { A } \perp _ { p } X _ { B } | \dot { X } _ { C }$ □

## F.3 Proof of Proposition 6.1 (Thin partition)

Proof. i) If the partition induced by X is thin, then V is acyclic complete (by contradiction):

First, assume that V is cyclic. Then, according to Theorem 2.5, there exists $\pmb { x } \in \pmb { \mathscr { X } }$ with $\langle { \pmb x } \rangle = \emptyset$ and thus ⟨X ⟩ is not thin. Therefore V is acyclic.

Next, assume that V is not complete. Then the size of the state space of X is smaller than n!. The size of the ranking space $S _ { n } ^ { V }$ is n!, and thus there is at least one realization x with an associated event with size larger than 1. Then the partition $\langle x \rangle$ is not thin. Therefore V is complete.

ii) If V is acyclic complete then the partition induced by X is thin:

The RRV X defined by an acyclic V breaks MEC, and thus $\langle x \rangle$ is a partition of $S _ { n } ^ { V }$ . Besides, the size of the state space of X when V is complete is n!. The only partition of $S _ { n } ^ { V }$ with n! non-empty subsets is a thin partition. □

## F.4 Proof of Theorem 6.2

Proof. Without loss of generality, we assume that $\bar { \pi } = [ 1 , \ldots , n ]$

X is defined by $\mathcal { V } ^ { R I }$ that is a complete DAG, and by Proposition 6.1, X induces a thin partition. Then, for simplicity, for any $\pmb { x } \in \mathcal { X }$ with $\langle { \pmb x } \rangle = \{ \sigma \}$ we write $h ( \sigma ) = h ( \pmb { x } )$ , and $p ( { \pmb x } ) = p ( { \boldsymbol \sigma } )$

i) $\Delta _ { \mathcal { G } ^ { R I } } ( { \pmb x } ) \subseteq \Delta _ { \bar { \pi } } ( S _ { n } ^ { V } ) \mathrm { . }$

Let $h \in \Delta _ { \mathcal { G } ^ { R I } } ( \pmb { \mathscr { X } } )$ . Then, by Definition 5.1

$$
h ( \pmb { x } ) = \prod _ { v = 1 } ^ { n } h ( x _ { v } ) ,
$$

for all $\textbf { \em x } \in \textbf { \em x }$ , where $X _ { v } ( \sigma ) : = r _ { P _ { \sigma } } ^ { v } ( \sigma )$ for $v \in V , \ \mathcal { V } ^ { R I }$ is a complete DAG with the ancestral ordering π¯ and thus $X _ { v } ( \sigma ) : = r _ { \{ 1 , \ldots , v - 1 \} } ^ { v } ( \sigma ) = f _ { \sigma , \bar { \pi } } ( v )$ , for $v \in V$ . Therefore, by Eq. 6, h is a RIM for π¯, and $\Delta _ { \mathcal { G } ^ { R I } } ( \pmb { \mathscr { X } } ) \subseteq \Delta _ { \bar { \pi } } ( S _ { n } ^ { V } )$

ii) $\Delta _ { \bar { \pi } } ( S _ { n } ^ { V } ) \subseteq \Delta _ { \mathcal { G } ^ { R I } } ( { \pmb x } )$

We prove it by showing that, for $p \in \Delta _ { \bar { \pi } } ( S _ { n } ^ { V } )$ , the RRN $h = \mathrm { p u s h } _ { \mathcal { V } ^ { R I } , \mathcal { G } ^ { R I } } ( p )$ is equivalent to p, i.e., $p ( \sigma ) = h ( X ( \sigma ) )$ for all $\sigma \in S _ { n } ^ { V }$

Let $p \in \Delta _ { \bar { \pi } } ( S _ { n } ^ { V } )$ and $h = \mathrm { p u s h } _ { \mathcal { V } ^ { R I } , \mathcal { G } ^ { R I } } ( p )$ . Then by Definition 5.2 and (i)

$$
h ( \pmb { x } ) = \prod _ { v = 1 } ^ { n } h ( x _ { v } ) = \prod _ { v = 1 } ^ { n } p ( x _ { v } ) = \prod _ { i = 1 } ^ { n } p _ { i } ( f _ { \sigma , \bar { \pi } } ( i ) ) = p ( \sigma ) ,
$$

where $\langle { \pmb x } \rangle = \{ \sigma \}$ , for all $\pmb { x } \in \pmb { \mathcal { X } }$

## F.5 Proof of Corollary 6.3

Proof. Let X be an RRV defined by $\mathcal { V } ^ { R I }$ . By Theorem 6.2 we have $\Delta _ { \bar { \pi } } ( S _ { n } ^ { V } ) = \Delta _ { \mathcal { G } ^ { R I } } ( { \pmb x } )$ . Thus, by Lemma 5.4, RI independence is the independence model ind $( \mathcal { V } ^ { R I } , \mathcal { G } ^ { R I } )$ □

## F.6 Proof of Corollary 6.4

To prove this result we use the notation introduced in the proof of Theorem 6.2.

Proof. Let X be an RRV defined by $\mathcal { V } ^ { R I }$ and let $h = \mathrm { p u s h } _ { \mathcal { V } ^ { R I } , \mathcal { G } ^ { R I } } ( p )$ . By Definition 5.2, we have $h \in \Delta _ { \mathcal { G } ^ { R I } } ( \pmb { \mathscr { X } } )$ , where $\begin{array} { r } { h ( \pmb { x } ) = \prod _ { v \in V } p ( x _ { v } ) } \end{array}$ , for all $\pmb { x } \in \pmb { \mathscr { X } }$ . By Theorem 6.2, $h \in \Delta _ { \bar { \pi } } ( S _ { n } ^ { V } )$ .

The Kullback-Leibler divergence between p and g is:

$$
D _ { K L } ( p | | g ) = \sum _ { \sigma \in S _ { n } ^ { V } } p ( \sigma ) \ln \frac { p ( \sigma ) } { g ( \sigma ) } .
$$

We have that the minimization of the Kullback-Leibler divergence is equivalent to the next problems:

$$
\begin{array} { r l } { \underset { g \in \Delta _ { \pi } ( S _ { n } ^ { \nu } ) } { \mathrm { a r g ~ m i n ~ } } D _ { K L } ( p | | g ) = } & { \underset { g \in \Delta _ { \pi } ( S _ { n } ^ { \nu } ) } { \mathrm { a r g ~ m a x ~ } } \underset { \sigma \in S _ { n } ^ { \nu } } { \sum } p ( \sigma ) \ln g ( \sigma ) } \\ & { = \underset { g \in \Delta _ { \sigma } R ^ { \nu } ( \mathcal { X } ) } { \mathrm { a r g ~ m a x ~ } } \underset { x \in \mathcal { X } } { \sum } p ( x ) \ln g ( x ) } \\ & { = \underset { g \in \Delta _ { \sigma } \mathrm { m a x ~ } } { \mathrm { a r g ~ m a x ~ } } \underset { x \in \mathcal { X } } { \sum } p ( x ) \ln \underset { v \in V } { \prod } g ( x _ { v } ) } \\ & { = \underset { g \in \Delta _ { \sigma } R ^ { \nu } ( \mathcal { X } ) } { \mathrm { a r g ~ m a x ~ } } \underset { v \in \mathcal { X } _ { n } \in \mathcal { X } _ { n } } { \sum } p ( x _ { v } ) \ln g ( x _ { v } ) . } \end{array}
$$

The first equality follows from the definition of KL divergence, since $\scriptstyle \sum _ { \sigma } p ( \sigma )$ ln p(σ) does not depend on $g .$ . The second follows from Theorem 6.2 and the bijection between rankings and RRV realizations established by Proposition 6.1. The third follows from Definition 5.1, since $\breve { \mathcal { G } } ^ { R I }$ is empty. The fourth follows by expanding the logarithm of the product and marginalizing over the remaining variables.

Since the marginal distributions $g ( x _ { v } )$ can be chosen independently, the last objective can be maximized separately for each $v \in V$ . Therefore, a RIM that minimizes the Kullback-Leibler divergence with respect to p is

$$
g ^ { * } ( \pmb { x } ) = \prod _ { v \in V } p ( x _ { v } ) = h ( \pmb { x } ) ,
$$

for all $\pmb { x } \in \pmb { \mathscr { X } }$

## F.7 Proof of Theorem 6.5

To prove Theorem 6.5, we first introduce a result that relates RRVs with relative rankings when the variable graph is a complete acyclic graph.

Let A and B be two disjoint non-empty subsets. The relative ranking of A with respect to $B$ under $\sigma _ { \mathrm { { : } } }$ $\sigma _ { B } ^ { A } \in { \cal S } _ { | A | + | B | } ^ { A } ,$ is the injective function from A to $\{ 1 , \ldots , | A | + | \tilde { B | } \}$ mapping each item to its rank in the partial ranking $\sigma ^ { A , B }$ , that is, $\sigma _ { B } ^ { A } ( a ) = \sigma ^ { A , B } ( a )$ for $a \in A$ . Here, $S _ { | A | + | B | } ^ { A }$ denotes the space of these injective functions.

Lemma F.1 (Partial and relative rankings). Let V be a complete acyclic variable graph with ancestral ordering $\bar { \pi } = [ a _ { 1 } , \ldots , a _ { m } , b _ { 1 } , \ldots , b _ { n - m } ] ,$ , let $A = \{ a _ { 1 } , . . . , a _ { m } \}$ and $B = \{ b _ { 1 } , \ldots , b _ { n - m } \}$ be a partition $o f V$ , and let X be defined according to V. Then

$X _ { A }$ induces a thin partition of the (partial) ranking space of $A , S _ { m } ^ { A }$ , and

$X _ { B }$ induces a thin partition ofthe relative ranking space ofB with respect to $A , S _ { n } ^ { B }$

Proof. i) $X _ { A }$ induces a thin partition of $ { \mathcal { S } } _ { m } ^ { A }$

The definition of $X _ { A }$ is equivalent to an RRV for A defined by the complete acyclic variable graph $\mathcal { V } ^ { A }$ with ancestral ordering $\bar { \pi } ^ { A } = [ a _ { 1 } , \ldots , a _ { m } ]$ . Thus by Proposition $6 . 1 , X _ { A }$ induces a thin partition of $ { S _ { | A | } ^ { A } }$

ii) $X _ { B }$ induces a thin partition of $S _ { n } ^ { B }$

For $j = 1 , \dotsc , n - m$ , the reference set of $b _ { j }$ is $A \cup \{ b _ { 1 } , \dotsc , b _ { j - 1 } \}$ . Given $\scriptstyle { \mathbf { { x } } } _ { B }$ , start from any partial ranking $\alpha \in \mathcal { S } _ { m } ^ { A }$ and successively insert each $b _ { j }$ at rank $x _ { b _ { i } }$ using the insertion operator of Definition B.1. Each insertion rank is uniquely specified because the reference set contains all previously inserted items. Changing α changes only the partial ordering of the items in $A ,$ leaving the insertion ranks and the final ranks of the items in $B$ unchanged. Thus $\scriptstyle { \mathbf { { x } } } _ { B }$ uniquely determines an injective function from B to N, independently of α.

Conversely, given such an injective function, choose a ranking of V in which the items in B have the specified ranks and the partial ranking of A is $\alpha .$ . Remove $b _ { n - m } , \ldots , b _ { 1 }$ in this order, recording the rank of each item immediately before removal. These ranks recover $x _ { b _ { n - m } } , \ldots , x _ { b _ { 1 } }$ , respectively, independently of α. Thus the two constructions are inverse, establishing a bijection between $\scriptstyle { \mathcal { X } } _ { B }$ and $S _ { n } ^ { B }$ . In particular, for ${ \pmb x } = { \pmb X } ( \sigma )$

$$
\langle \pmb { x } _ { B } \rangle = \{ \rho \in \pmb { S } _ { n } ^ { V } : \rho _ { A } ^ { B } = \sigma _ { A } ^ { B } \} ,
$$

which proves the claim.

Now, we proceed with the proof of Theorem 6.5.

Proof. We prove that $\Delta _ { A , B } ( S _ { n } ^ { V } ) = \Delta _ { \mathcal G ^ { R S } } ( \pmb { \chi } ) \cap \Delta _ { \mathcal G ^ { R S } } ( \pmb { \ y } )$ , where X and Y are defined by $\nu ^ { R S }$ and $\mathcal { W } ^ { R S }$ , respectively, by showing that $h \in \Delta _ { \mathcal { G } ^ { R S } } ( \pmb { \mathcal { X } } ) \cap \Delta _ { \mathcal { G } ^ { R S } } ( \pmb { \mathcal { Y } } )$ if and only if $h \in \Delta _ { A , B } ( S _ { n } ^ { V } )$ The RRVs X and Y for V are defined by the complete acyclic graphs $\mathcal { V } ^ { R S }$ and $\mathcal { W } ^ { R S }$ , respectively. By Proposition 6.1, X and Y induce thin partitions. Then, for $h \in { \Delta } ( { \mathcal { X } } )$ and $\textbf { \em x } \in { \textbf { \em x } }$ with $\langle { \pmb x } \rangle = \{ \sigma \}$ , we write $h ( \sigma ) = h ( \pmb { x } )$ and $p ( { \pmb x } ) = p ( { \boldsymbol \sigma } )$ for simplicity. Moreover, for any $p \in \Delta ( S _ { n } ^ { V } )$ partial ranking $\sigma ^ { A } \in { \cal S } _ { m } ^ { A }$ , and interleaving $s _ { A } ^ { B } \in \mathsf { \Gamma } ( _ { | B | } ^ { N } )$ , we denote $\begin{array} { r } { p ( \sigma ^ { A } ) = \sum _ { \rho \in \langle \sigma ^ { A } \rangle } p ( \rho ) } \end{array}$ and $\begin{array} { r } { p ( \pmb { \mathscr { s } } _ { A } ^ { B } ) = \sum _ { \rho \in \langle \pmb { \mathscr { s } } _ { A } ^ { B } \rangle } p ( \rho ) } \end{array}$ , where $\langle \sigma ^ { A } \rangle = \{ \rho : \rho ^ { A } = \sigma ^ { \dot { A } } \}$ and $\langle \pmb { s } _ { A } ^ { B } \rangle = \{ \rho \in S _ { n } ^ { V } : s _ { A } ^ { B } ( \rho ) = \overline { { \pmb { s } _ { A } ^ { B } } } \}$

An RRM $h \in \Delta _ { \mathcal { G } ^ { R S } } ( \pmb { \mathscr { x } } )$ if and only if

$$
h ( \sigma ) = h ( \sigma ^ { A } ) h ( \sigma _ { A } ^ { B } ) = h ( \sigma ^ { A } ) h ( s _ { A } ^ { B } ( \sigma ) , \sigma ^ { B } ) ,
$$

for all $\sigma \in S _ { n } ^ { V }$ . The first equality holds by Lemma F.1. The second equality holds because for $v \in B$

$$
\sigma _ { A } ^ { B } ( v ) = s _ { A } ^ { B } ( \sigma ^ { B } ( v ) ) ,
$$

where $s _ { A } ^ { B } = s _ { A } ^ { B } ( \sigma )$ , for $\sigma \in S _ { n } ^ { V }$

An RRM $h \in \Delta _ { \mathcal { G } ^ { R S } } ( \mathcal { y } )$ if and only if

$$
h ( \sigma ) = h ( \sigma _ { B } ^ { A } ) h ( \sigma ^ { B } ) = h ( \sigma ^ { A } , s _ { B } ^ { A } ( \sigma ) ) h ( \sigma ^ { B } ) ,
$$

for all $\sigma \in S _ { n } ^ { V }$ . In addition, $s _ { B } ^ { A } ( \sigma )$ uniquely determines $s _ { A } ^ { B } ( \sigma )$ , for $\sigma \in S _ { n } ^ { V }$ . Thus, we have that $h \in \Delta _ { \mathcal { G } ^ { R S } } ( \mathcal { V } )$ if and only if

$$
h ( \sigma ) = h ( \sigma ^ { A } , s _ { A } ^ { B } ( \sigma ) ) h ( \sigma ^ { B } ) ,
$$

for $\sigma \in S _ { n } ^ { V }$

Then, $h \in \Delta _ { \mathcal { G } ^ { R S } } ( \pmb { \mathcal { X } } ) \cap \Delta _ { \mathcal { G } ^ { R S } } ( \pmb { \mathcal { Y } } )$ if and only if

$$
h ( \sigma ) = h ( \sigma ^ { A } ) h ( s _ { A } ^ { B } ( \sigma ) ) h ( \sigma ^ { B } ) ,
$$

for $\sigma \in S _ { n } ^ { V }$

By $\mathrm { E q . ~ } 7 , h \in \Delta _ { A , B } ( S _ { n } ^ { V } )$ if and only if

$$
h ( \sigma ) = h ( \sigma ^ { A } ) h ( s _ { A } ^ { B } ( \sigma ) ) h ( \sigma ^ { B } ) ,
$$

for all $\sigma \in S _ { n } ^ { V }$ . And therefore, $h \in \Delta _ { \mathcal { G } ^ { R S } } ( \pmb { \mathcal { X } } ) \cap \Delta _ { \mathcal { G } ^ { R S } } ( \pmb { \mathcal { Y } } )$ if and only if $\begin{array} { r } { h \in \Delta _ { A , B } ( S _ { n } ^ { V } ) } \end{array}$ □

## F.8 Proof of Corollary 6.6

Proof. Let X and Y be the RRVs defined by $\mathcal { V } ^ { R S }$ and $\mathcal { W } ^ { R S }$ . By Theorem 6.5 we have $\Delta _ { A , B } ( S _ { n } ^ { V } ) \subseteq$ $\Delta _ { \mathcal { G } ^ { R S } } ( \mathcal { X } )$ and $\Delta _ { A , B } ( S _ { n } ^ { V } ) \subseteq \Delta _ { \mathcal { G } ^ { R S } } ( { \mathfrak { y } } )$ . Thus, by Lemma 5.4, RS independence is given by the union of independence models ind $( \mathcal { V } ^ { R S } , \mathcal { G } ^ { R S } )$ and in $1 ( \mathcal { W } ^ { R S } , \mathcal { G } ^ { R S } )$ □

## F.9 Proof of Proposition 6.7

In this proof we reuse the notation in the proof of Theorem 6.5.

$$
\begin{array} { r } { g = \mathrm { p u s h } _ { \mathcal { W } ^ { R S } , g ^ { R S } } ( \boldsymbol { p } ) \in \Delta _ { \mathcal { G } ^ { R S } } ( \mathfrak { V } ) \mathrm { , a n d } \ h = \mathrm { p u s h } _ { \mathcal { V } ^ { R S } , \mathcal { G } ^ { R S } } ( g ) \in \Delta _ { \mathcal { G } ^ { R S } } ( \pmb { \mathcal { X } } ) . } \end{array}
$$

i) $h \in \Delta _ { A , B } ( S _ { n } ^ { V } )$ :

First, by Definition 5.2, we have

$$
\begin{array} { c } { { g ( \sigma ) = p ( \sigma _ { B } ^ { A } ) p ( \sigma ^ { B } ) } } \\ { { { } } } \\ { { { } = p ( \sigma ^ { A } , \pmb { s } _ { B } ^ { A } ) p ( \sigma ^ { B } ) } } \\ { { { } } } \\ { { { } = p ( \sigma ^ { A } , \pmb { s } _ { A } ^ { B } ) p ( \sigma ^ { B } ) , } } \end{array}
$$

where $s _ { B } ^ { A } = s _ { B } ^ { A } ( \sigma )$ and $s _ { A } ^ { B } = s _ { A } ^ { B } ( \sigma )$ , for all $\sigma \in S _ { n } ^ { V }$ . The second equality holds because $\sigma _ { B } ^ { A } ( v ) =$ $s _ { B } ^ { A } ( \sigma ^ { A } ( \bar { v } ) )$ for all $v \in A$ , and thus $\langle \sigma _ { B } ^ { A } \rangle = \langle \sigma ^ { A } , s _ { B } ^ { A } \rangle$ . The third equality holds because the ranks in $s _ { A } ^ { B }$ are uniquely determined by the ranks in $\begin{array} { l } { \displaystyle { s _ { B } ^ { A } } } \end{array}$ and thus $\langle s _ { B } ^ { A } \rangle = \langle s _ { A } ^ { B } \rangle$

Then, by Definition 5.2, we have

$$
\begin{array} { c } { { h ( \sigma ) = g ( \sigma ^ { A } ) g ( \sigma _ { A } ^ { B } ) } } \\ { { { } } } \\ { { = g ( \sigma ^ { A } ) g ( \sigma ^ { B } , s _ { A } ^ { B } ) } } \\ { { { } } } \\ { { { } = p ( \sigma ^ { A } ) p ( \sigma ^ { B } ) p ( s _ { A } ^ { B } ) , } } \end{array}
$$

where $s _ { A } ^ { B } = s _ { A } ^ { B } ( \sigma )$ , for all $\sigma \in S _ { n } ^ { V }$ . The second equality holds because $\sigma _ { A } ^ { B } ( b ) = s _ { A } ^ { B } ( \sigma ^ { B } ( b ) )$ for all $b \in B$

ii) h = arg min<sub>q∈∆A,B(SV )</sub> D<sub>KL</sub>(p||q):

By (i) $h \in \Delta _ { A , B } ( S _ { n } ^ { V } )$ and $h ( \sigma ) = p ( \sigma ^ { A } ) p ( \sigma ^ { B } ) p ( s _ { A } ^ { B } )$

The Kullback-Leibler divergence between p and g is:

$$
D _ { K L } ( p | | g ) = \sum _ { \sigma \in S _ { n } ^ { V } } p ( \sigma ) \ln \frac { p ( \sigma ) } { g ( \sigma ) } .
$$

Minimizing the divergence is equivalent to maximizing the expected log-likelihood,

$$
\operatorname * { a r g m i n } _ { g \in \Delta _ { A , B } ( S _ { n } ^ { V } ) } D _ { K L } ( p | | g ) = \operatorname * { a r g m a x } _ { g \in \Delta _ { A , B } ( S _ { n } ^ { V } ) } \sum _ { \sigma \in S _ { n } ^ { V } } p ( \sigma ) \ln g ( \sigma ) .
$$

Any $g \in \Delta _ { A , B } ( S _ { n } ^ { V } )$ factorizes as

$$
g ( \sigma ) = g ( \sigma ^ { A } ) g ( \sigma ^ { B } ) g ( s _ { A } ^ { B } ( \sigma ) ) .
$$

Using the factorization, we can decompose the objective function of the maximization problem additively:

$$
\sum _ { \sigma \in S _ { n } ^ { V } } p ( \sigma ) \ln g ( \sigma ) = \sum _ { \sigma \in S _ { n } ^ { V } } p ( \sigma ) \ln g ( \sigma ^ { A } ) + \sum _ { \sigma \in S _ { n } ^ { V } } p ( \sigma ) \ln g ( \sigma ^ { B } ) + \sum _ { \sigma \in S _ { n } ^ { V } } p ( \sigma ) \ln g ( s _ { A } ^ { B } ( \sigma ) ) .
$$

By grouping rankings in the events we have

$$
\begin{array} { r } { { \displaystyle \sum _ { \sigma \in S _ { n } ^ { V } } p ( \sigma ) \ln g ( \sigma ^ { A } ) = \sum _ { \sigma ^ { A } \in S _ { | A | } ^ { A } } p ( \sigma ^ { A } ) \ln g ( \sigma ^ { A } ) } , } \\ { { \displaystyle \sum _ { \sigma \in S _ { n } ^ { V } } p ( \sigma ) \ln g ( \sigma ^ { B } ) = \sum _ { \sigma ^ { B } \in S _ { | B | } ^ { B } } p ( \sigma ^ { B } ) \ln g ( \sigma ^ { B } ) } , } \\ { { \displaystyle \sum _ { \sigma \in S _ { n } ^ { V } } p ( \sigma ) \ln g ( s _ { A } ^ { B } ( \sigma ) ) = \sum _ { s _ { A } ^ { B } \in \binom { N } { | B | } } p ( s _ { A } ^ { B } ) \ln g ( s _ { A } ^ { B } ) } . } \end{array}
$$

Therefore, the RSM that minimizes the Kullback-Leibler divergence with respect to p is h, where

$$
h ( \sigma ) = p ( \sigma ^ { A } ) p ( \sigma ^ { B } ) p ( s _ { A } ^ { B } ( \sigma ) ) ,
$$

for all $\sigma \in S _ { n } ^ { V }$