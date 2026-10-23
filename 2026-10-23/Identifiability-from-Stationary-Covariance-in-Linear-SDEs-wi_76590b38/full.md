# Identifiability from Stationary Covariance in Linear SDEs with Reciprocal Interactions and Latent Variables

Ana Let´ıcia Garcez Vicente

Saber Salehkaleybar

Leiden Institute of Advanced Computer Science (LIACS), Leiden University, Netherlands

## Abstract

Continuous-time stochastic diferential equations (SDEs) provide a natural framework for modeling interacting dynamical systems. A central causal question is whether causal effects among observed variables can be identified from observational data drawn from the stationary distribution, given the underlying causal graph. For linear SDEs, existing identifiability results are largely restricted to fully observed systems whose causal graphs contain no directed 2-cycles. We first establish new graphical conditions for the generic identifiability of causal efects in systems that may contain directed 2-cycles, thereby allowing for reciprocal interactions. Next, we address the previously unexplored setting of latent variables by developing an instrumentalvariable framework for identifying a target direct causal efect under partial observability.

## 1 INTRODUCTION

Understanding causal interactions in dynamical systems is a fundamental problem across biology (Marbach et al., 2012; Lorch et al., 2026), economics (Hamilton, 2020), and engineering (<sup>˚</sup>Astr¨om, 2012). Stochastic diferential equations (SDEs), in which the difusion captures external stochastic perturbations and the drift function describes the deterministic interactions among variables, are a natural model for many such systems that evolve continuously over time (Øksendal, 2003). Recovering the drift coeficients shows the local direct causal efects controlling the system dynamics. However, temporal trajectories are either not available or only partially observed in many applications. Instead, data consist of samples collected from a system operating near equilibrium. In this context, stationary SDEs have emerged as an attractive framework for causal modeling (Varando and Hansen, 2020; Lorch et al., 2024).

Under standard stability assumptions, stationary linear SDEs satisfy the continuous Lyapunov equation, which provides an algebraic relationship between the stationary covariance matrix and the unknown drift matrix. This connection enables causal information to be inferred from equilibrium observations without requiring access to the underlying trajectories. Furthermore, graphical continuous Lyapunov models provide a causal graphical representation analogous to structural causal models while naturally accommodating cycles and self-loops, common features in real dynamical systems (Varando and Hansen, 2020; Hansen, 2025; Fitch, 2019).

In linear SDEs, a fundamental challenge is whether the drift matrix can be recovered from the stationary covariance matrix when its sparsity pattern is known. Recent work has shown that the drift matrix is globally identifiable when the underlying graph is simple, i.e., when it contains no directed cycles of length two (Dettling et al., 2023). Reciprocal interactions, therefore, constitute the principal obstruction to identifiability. Directed 2-cycles naturally arise in many applications. In biological regulatory networks, for example, mutually activating and mutually repressing components form fundamental feedback motifs responsible for bistability and switch-like behavior (Ferrell Jr, 2002). Reciprocal interactions between cortical and subcortical are also ubiquitous in neural circuits (Cruz et al., 2023). Similar bidirectional interactions are also present in engineered systems with coupled controllers and feedback loops (<sup>˚</sup>Astr¨om and Murray, 2021). Whether directed 2-cycles necessarily prevent identification, or whether additional graphical structure can resolve the resulting ambiguity, remains largely unexplored.

Another challenge arises when the system is only partially observed. In many applications, latent variables directly influence the variables of interest, introducing confounding that prevents direct causal efects from being recovered through standard adjustment techniques. Although latent confounding has been extensively studied in structural causal models through instrumental variables, adjustment criteria Perkovi´c et al. (2018), and do-calculus Pearl (2009), comparable findings for stationary SDEs are far less established.

In this paper, we study identification from stationary covariance in non-simple linear SDEs through a common rank-based perspective. Directed 2-cycles are allowed throughout. Under full observability, we study graphical conditions that guarantee identification of the complete drift matrix through a global rank condition derived from the stationary Lyapunov equation. Under partial observability, latent variables prevent full drift recovery, but suitable instruments lead to a target-row rank condition for identifying coeficients of interest. Both settings follow the same principle: using graphical structure to certify rank conditions for covariance-based systems induced by the stationary Lyapunov equation. Our main contributions are:

• We establish new graphical conditions for generic identifiability of graphical continuous Lyapunov models containing directed 2-cycles (Theorem 1). We further characterize several identifiable four-node motifs that fall outside these conditions (Proposition 1) and show how identifiability can be localized to identifiable subgraphs within larger non-identifiable systems (Theorem 2);

• We provide an instrumental-variable framework for identifying direct drift coeficients under latent confounding using only observed covariance matrix (Theorem 3);

• We provide graphical conditions guaranteeing the full-rank requirement of the instrumental covariance matrix. We completely characterize the direct-instrument case via a bipartite matching condition (Theorem 4) and derive a suficient graphical criterion based on vertex-disjoint directed paths when intermediate variables are present (Theorem 5).

## 2 RELATED WORK

Causal modeling with stochastic diferential equations. Classical causal inference is predominantly formulated through structural causal models (Pearl, 2009). Extending causal reasoning to dynamical systems requires accounting for temporal dynamics and feedback. Hansen and Sokol (2014) formalized interventions on stochastic diferential equations, while Bongers and Mooij (2018); Mogensen et al. (2018) developed causal models for general stochastic dynamical systems.

When temporal trajectories are unavailable, inference may rely on stationary observations. Initial work investigated learning directed graphical models from stationary Gaussian observations (Fitch, 2019), while Varando and Hansen (2020) introduced graphical continuous Lyapunov models, in which the sparsity pattern of the drift matrix defines a directed graph. In this representation, a directed edge corresponds to a direct interaction in the drift, while the stationary covariance summarizes the equilibrium dependence induced by the entire dynamical system. Subsequent work established a trek rule providing a graphical characterization of stationary covariance entries (Hansen, 2025), while Lorch et al. (2024, 2026) developed methods for causal learning directly from stationary distributions.

A fundamental question is when the drift matrix is identifiable from stationary covariance information. Dettling et al. (2023) proved that graphical continuous Lyapunov models are globally identifiable if and only if the underlying graph is simple, and further established necessary conditions for generic identifiability. Our work complements this line of research by identifying graphical conditions under which non-simple graphs remain generically identifiable and by characterizing local graphical structures that repair the ambiguity introduced by directed reciprocal interactions.

Causal efects under partial observation Partial observation is a central challenge in causal inference where unobserved common causes can invalidate adjustment, induce spurious associations, and prevent causal efects from being recovered from the observed distribution alone. In structural causal models, valid adjustment set or do-calculus provide solutions to identify causal efects in the presence of latent variables (Pearl, 2009; Richardson and Spirtes, 2002; Shpitser and Pearl, 2006; Shpitser et al., 2012; Perkovi´c et al., 2018). However, these approaches are mainly developed for acyclic structural causal models where no directed cycle is allowed.

In linear structural equation models, generic identifiability from observational covariance has been studied through graphical rank criteria, including the halftrek criterion (Foygel Barber et al., 2022), instrumental cutsets (Kumor et al., 2019), and latent-variable extensions (Foygel et al., 2012). These approaches, like ours, use graphical path structures to certify rank conditions. Our setting difers in that identification is based on stationary covariance constrained by the continuous Lyapunov equation.

For stationary SDEs, considerably less is known. Existing work primarily focused on recovering the complete drift matrix under full observation (Dettling et al., 2023). In many applications, however, recovering the entire dynamical system is unnecessary; instead, the objective is to identify a particular direct causal efect despite latent confounding. More recently, van Seeventer and Salehkaleybar (2026) investigated sign identifiability from stationary second-order information, including settings with latent variables. Additionally, instrumental processes whose identifying moment equations are built from integrated covariance information have been used to extend instrumentalvariable approaches to stochastic processes observed over time (Mogensen, 2023). Our second contribution difers from these approaches. Rather than recovering the entire drift matrix, identifying only the sign of causal efects, or relying on trajectory observations to estimate integrated covariances, we study identification of a target drift coeficient under latent confounding. We derive instrumental equations and establish graphical conditions under which the observed stationary covariance uniquely identifies the target causal effects.

## 3 PRELIMINARIES

## 3.1 Stochastic diferential equations

Stochastic diferential equations (SDEs) provide a natural framework for modeling continuous-time systems evolving both deterministic interactions and random perturbations indexed by node set V, with $| V | = d .$ Let $\mathbf { X } ( t ) \in \mathbb { R } ^ { d }$ denote the state vector at time t. Each component $X _ { v } ( t )$ represents variable $v \in V$ whose evolution is influenced by other variables in the system as well as by external stochastic perturbations. Our analysis focuses on the linear SDEs:

$$
\begin{array} { r } { d \mathbf { X } ( t ) = \mathbf { M } \mathbf { X } ( t ) d t + \mathbf { D } d \mathbf { W } ( t ) , } \end{array}\tag{1}
$$

which corresponds to the multivariate Ornstein-Uhlenbeck (OU) process. D is the difusion matrix and $\mathbf { W } ( t )$ is a d-dimensional standard Brownian motion. The matrix $\mathbf { M } = ( m _ { i j } ) _ { i , j \in V } \in \mathbb { R } ^ { d \times d }$ is the drift matrix and characterizes the deterministic interactions among variables. In particular, the entry $m _ { i j }$ quantifies the direct efect of variable $X _ { j }$ on the evolution of $X _ { i }$ . Therefore, from a causal perspective, $m _ { i j }$ represents the direct causal efect from $X _ { j }$ to $X _ { i }$

Let $\mathbf { C } : = \mathbf { D } \mathbf { D } ^ { \top }$ . Under the stability condition that all eigenvalues of M have strictly negative real part, the process admits an unique stationary Gaussian distribution with covariance matrix Σ satisfying:

$$
\mathbf { M } \mathbf { \boldsymbol { \Sigma } } + \mathbf { \boldsymbol { \Sigma } } \mathbf { M } ^ { \top } + \mathbf { \boldsymbol { C } } = 0 .\tag{2}
$$

Following Dettling et al. (2023), we assume that the difusion matrix C is known.

## 3.1.1 Graphical Representation

The drift matrix can be associated with a directed graph $G = ( V , E )$ , called the drift graph. Each node in V corresponds to one component of the stochastic process $\mathbf { X } = \left( X _ { 1 } , \cdots , X _ { d } \right)$ , and a directed edge $X _ { i } \to X _ { j }$ is present whenever the drift coeficient describing the influence of $X _ { i }$ on $X _ { j }$ is nonzero, $\mathrm { i . e . , } m _ { j i } \neq 0 .$ . The variables are partitioned into $V = O \cup L$ , where O denotes the observed variables, and L are the unobserved/latent ones.

For a node $x \in V ,$ , we denote the set of parents of x $\operatorname { P a } ( x ) = \{ v \in V : v  x \}$ the set of nodes with a direct edge pointing to x. Similarly, the set of children of x is $\operatorname { C h } ( x ) = \{ v \in V : x \to v \}$ . A directed path from a node x to y $( x \sim y )$ is a sequence of distinct nodes $x =$ $v _ { 0 }  \cdots  v _ { k } = y$ , such that every consecutive pair of nodes is connected by a directed edge oriented in the same direction. The ancestors of x, denoted by $\mathrm { A n } ( x )$ are the nodes from which there exists a directed path to x, including x itself. Conversely, the descendants, De(x), are the nodes reachable from x through directed paths, including x.

The graph G may contain directed cycles. A directed cycle of length k is a sequence of distinct nodes $v _ { 1 } \to \cdots \to v _ { k } \to v _ { 1 }$ , with $k \geq 2$ . In particular, a cycle of length two, referenced as $\it { \Delta } a  { 2 - c y c l e }$ throughout this paper, consists of two reciprocal edges $x  y$ and $y  x$ representing mutual feedback. A graph without directed cycles is called a Directed Acyclic Graph (DAG). Our framework does not assume acyclicity.

Under the stability assumption, the stationary distribution of the linear SDE driven by a Wiener process is multivariate Gaussian. Without loss of generality, we assume that the stationary mean is zero. Consequently, the stationary distribution is completely characterized by its covariance matrix, and our identifiability analysis can therefore be formulated entirely in terms of stationary covariances.

A trek between two nodes x and y is a path without colliders, i.e., a path that can be decomposed into two directed paths sharing a common source node $c , x \gets$ $\cdots  c  \cdots  y .$ The node c is called the top of the trek. A trek may degenerate into a purely directed path if $c = x { \mathrm { ~ o r ~ } } c = y$ Treks play a central role in OU models with Wiener process. Each entry $( i , j )$ of stationary covariance admits a decomposition as sums of contributions over treks connecting the pair $( i , j )$ . Therefore, the existence or absence of treks determines whether two variables may exhibit nonzero covariance. In particular, if no trek connects two nodes $X _ { i }$ and $X _ { j } ,$ then $\Sigma _ { i j } = 0$ , implying that $X _ { i }$ and $X _ { j }$ are marginally independent (Boege et al., 2025).

## 3.1.2 Problem Formulation

Given a directed graph $G = ( V , E )$ and a known positive definite diagonal difusion matrix $\mathbf { C } ,$ we consider the graphical continuous Lyapunov model $\mathcal { M } _ { G , { \bf C } } =$ $\{ \boldsymbol { \Sigma } : \mathbf { M } \boldsymbol { \Sigma } + \boldsymbol { \Sigma } \mathbf { M } ^ { \top } + \mathbf { C } = 0 , s u p p ( \mathbf { M } ) \subseteq E , \mathbf { M }$ stable} where supp $( \mathbf { M } ) = \{ ( i , j ) : m _ { i j } \neq 0 \}$ . The paper studies two identification problems over $\mathcal { M } _ { G , \mathbf { C } } \colon \mathrm { ( i ) }$ recovery of the entire drift matrix under full observability, and (ii) recovery of a target direct causal efect $m _ { y x }$ under partial observability.

Definition 1 (Global identifiability). A parameter $\theta ( \mathbf { M } )$ is globally identifiable in $\mathcal { M } _ { G , \mathbf { C } }$ if for any two admissible drift matrices $\begin{array} { r l } { \Sigma ( \mathbf { M } _ { 1 } ) } & { { } = \ \Sigma ( \mathbf { M } _ { 2 } ) \quad \implies } \end{array}$ $\theta ( \mathbf { M } _ { 1 } ) = \theta ( \mathbf { M } _ { 2 } )$

Definition 2 (Generic identifiability). A parameter θ(M) is generically identifiable in $\mathcal { M } _ { G , \mathbf { C } }$ if there exists a proper algebraic subset A of the admissible parameter space such that $\Sigma ( { \bf M } _ { 1 } ) = \Sigma ( { \bf M } _ { 2 } ) \implies \theta ( { \bf M } _ { 1 } ) =$ $\boldsymbol { \theta } ( \mathbf { M } _ { 2 } )$ for all admissible $\mathbf { M } _ { 1 } , \mathbf { M } _ { 2 } \notin { \mathcal { A } } .$

Kernel matrix and rank criterion Our analysis of full observability relies on the algebraic characterization of identifiability Dettling et al. (2023). For $\pmb { \Sigma } \in \mathcal { M } _ { G , \mathbf { C } }$ , define:

$$
\begin{array} { r } { \mathbf { H } ( \Sigma ) _ { i  j , ( k , l ) } = \{ \begin{array} { l l } { - \Sigma _ { l j } , } & { i = k , } \\ { \Sigma _ { k j } , } & { i = l , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array}  } \end{array}
$$

where the rows are indexed by all ordered pairs $( i , j ) \in$ $V \times V$ (corresponding to all possible directed edges, including self-loops), and the columns are indexed by unordered pairs $( k , l )$ with $k < l .$ For a graph $G = ( V , E )$ let $\mathbf { H } ( \Sigma ) _ { E ^ { c } }$ denote the submatrix obtained by restricting the rows of $\mathbf { H } ( \pmb { \Sigma } )$ to those corresponding to the non-edges $E ^ { c }$ . Throughout the paper, we certify generic identifiability by exhibiting square submatrices (minors) of $\mathbf { H } ( \Sigma ) _ { E ^ { c } }$ <sub>,·</sub> whose determinants are not identically zero.

The following characterization is the main algebraic criterion used throughout the paper.

Lemma 1 ((Dettling et al., 2023, Lemma $6 . 4 ( \mathrm { i i } ) )$ Let $G \ = \ ( V , E )$ be a directed graph with $| V | = d .$ The graphical continuous Lyapunov model $\mathcal { M } _ { G , \mathbf { C } }$ is generically identifiable if and only if there exists $\pmb { \Sigma } \in$ $\mathcal { M } _ { G , \mathbf { C } }$ such that rank $\begin{array} { l } { \displaystyle { \left( H ( \Sigma ) _ { E ^ { c } , \cdot } \right) \ = \ \frac { d ( d - 1 ) } { 2 } } } \end{array}$ . Equivalently, generic identifiability can be certified by exhibiting a nonzero $\frac { d ( d - 1 ) } 2 \times \frac { d ( d - 1 ) } 2$ minor of $H ( \Sigma ) _ { E ^ { c } } ,$

Proofs Proofs of lemmas and theorems are given in the Appendix.

![](images/91c3323ba3473d7f7787448c5e5293db588619526b5731a87606889794439e50.jpg)  
(a) Child exten- (b) Parent exsion. tension.  
Figure 1: Generically identifiable 3-node motifs.

## 4 GENERIC IDENTIFIABILITY UNDER FULL OBSERVABILITY

In this section, we investigate generic identifiability of the drift matrix under full observability. We first characterize a class of non-simple graphs that remain generically identifiable despite containing directed 2- cycles (Theorem 1). We then show a class of 4-node motifs generically identifiable (Proposition 1).

## 4.1 Identifiability Beyond Simple Structures

Generic identifiability of linear stochastic diferential equations has been extensively studied through the algebraic characterization of the Lyapunov equation 2. Dettling et al. (2023) showed that the drift matrix is globally identifiable if and only if the graph is simple, i.e., contains no 2-cycles. They also observed that certain non-simple graphs may remain generically identifiable, illustrating this through isolated examples. However, no general graphical characterization of such non-simple identifiable graphs was provided. The goal of this section is to characterize a class of non-simple graphs for which generic identifiability is guaranteed.

Theorem 1. Let $G = ( V , E )$ be a directed graph satisfying: (i) Every directed 2-cycle $( i , j )$ is vertex-disjoint from all other 2-cycles; and (ii) For every directed 2- cycle $( i , j )$ , there exists a vertex $k _ { i j }$ such that, after possibly exchanging i and $j , k _ { i j } \sim i , k _ { i j } \not \sim j$ , where ∼ denotes adjacency, i.e., there is an edge in at least one direction; and such that $k _ { i j }$ does not belong to any directed 2-cycle. Moreover, the vertices $k _ { i j }$ can be chosen pairwise distinct for distinct directed 2-cycles. $\mathcal { M } _ { G , C }$ is generically identifiable.

Proof sketch. By Lemma 1, it sufices to exhibit $\Sigma \in$ $\mathcal { M } _ { G , C }$ for which $\mathbf { H } ( \Sigma ) _ { E ^ { c } }$ <sub>,·</sub> has full column rank.

For each directed 2-cycle $( i , j )$ , let k be its witness from assumption 2 and define the local block $N = \{ i , j , k \}$ We specialize the drift matrix by retaining only the two edges of the 2-cycle and the single directed edge between k and its adjacent endpoint, and set all other allowed of-diagonal coeficients to zero. The retained of-diagonal coeficients are scaled by a suficiently small $\varepsilon > 0$ . With negative diagonal coeficients, the resulting drift matrix is Hurwitz.

Under this specialization, the stationary covariance is block diagonal across the sets N and the remaining singleton vertices. Each 2-cycle block contributes a nonsingular $3 \times 3$ minor of $\mathbf { H } ( \Sigma ) _ { E ^ { c } , \cdot }$ . For columns joining distinct blocks, at least one orientation of every vertex pair is a non-edge. Selecting one such non-edge row for every mixed column gives, as $\varepsilon  0 .$ , a diagonal minor with positive variance entries. Hence this mixed block is also nonsingular for suficiently small $\varepsilon > 0$

Combining the local and mixed minors gives a nonzero $( \mathbf { \Phi } _ { 2 } ^ { d } ) \times \mathbf { \Phi } ( \mathbf { \Phi } _ { 2 } ^ { d } )$ minor of $\mathbf { H } ( \Sigma ) _ { E ^ { c } , \cdot }$ , and Lemma 1 yields generic identifiability. □

## 4.2 Four-Node Non-Simple Identifiable Motifs

Theorem 1 provides a suficient condition for generic identifiability by requiring that every directed 2-cycle has a neighboring node adjacent to exactly one of its nodes. This condition is not necessary. We now investigate four-node graphs that violate this condition yet remain generically identifiable.

Identifiability in the presence of a direct 2-cycle is not governed by adjacency alone. A common parent, a common child, an asymmetric directed path, or an ad ditional 2-cycle may still fail to repair the rank deficiency created by a 2-cycle. What matters is how the trek structure of the graph constrains the covariance entries in the matrix $\mathbf { H } ( \pmb { \Sigma } ) _ { E ^ { c } , \cdot }$ . By Lemma 1, when $| V | = d = 4$ , it sufices to construct a nonzero $6 \times 6$ minor of $\mathbf { H } ( \pmb { \Sigma } ) _ { E ^ { c } , \cdot }$

## 4.2.1 Extensions of deficient three-node motifs

Figure 2 shows the deficient three-node motifs from which we construct four-node extensions. Adding a vertex is not suficient by itself; identifiability is restored only when the extension modifies the covariance structure so as to supply the missing independent constraint. The following proposition summarizes the identifiable one-vertex extensions.

Proposition 1 (One-vertex extensions of deficient motifs). Let $G _ { 0 }$ be one of the deficient three-node motifs in Figure 2, and let G be obtained by adding a fourth node v with exactly one adjacency to $G _ { 0 }$

The following extensions are generically identifiable: v is (i) a parent of a vertex of $G _ { 0 }$ , except in the commonchild setup (Figure 2a), where v may only be a parent of one vertex in a 2-cycle; (ii) a child of a vertex of

![](images/a697c034d33e4354c59c7f6e466d09328cc8da81c0f1a839d14b194ae3817f97.jpg)  
Figure 2: Three-node non-identifiable motifs.

$G _ { 0 } ;$ or (iii) part of a directed 2-cycle with a vertex of $G _ { 0 } ,$ except in the 2-cycles intersection setup (Figure 2d).

These extensions illustrate that identifiability is governed not by adjacency alone, but by whether the added structure supplies new independent covariance constraints. Identifiable configurations, failed extensions, and their rank-based explanations are given in Appendix.

## 4.3 Subgraph Identifiability

The previous section studied identifiability of the entire drift matrix. We now consider subgraph identifiability, asking when the drift coeficients of an induced subgraph can be recovered even if the full graph is not identifiable.

Theorem 2. Let $G ~ = ~ ( V , E )$ be a directed graph associated with the stationary linear SDE. Let $S \subseteq$ VSuppose that: $( i ) \mathcal { M } _ { G [ S ] , C _ { S S } }$ is globally (respectively, generically) identifiable; and (ii) no vertex outside S is a parent of a vertex in $S ,$ that is, $r \to s \not \in E$ for every $r \in V \setminus S , \ s \in S .$ Then $M _ { S S }$ is globally (respectively, generically) identifiable from $\Sigma _ { S S }$ , even if the complete drift matrix M is not identifiable.

Proof. Let $R = V \setminus S$ . The (S, S)-block of the Lyapunov equation 2 is $\mathbf { M } _ { S S } \boldsymbol { \Sigma } _ { S S } + \mathbf { M } _ { S R } \boldsymbol { \Sigma } _ { R S } + \boldsymbol { \Sigma } _ { S S } \mathbf { M } _ { S S } ^ { \top } +$ $\boldsymbol { \Sigma _ { S R } } \boldsymbol { \mathbf { M } } _ { S R } ^ { \top } + \boldsymbol { \mathbf { C } } _ { S S } = 0$ . By (ii) , ${ \bf M } _ { S R } = 0 ,$ , and this reduces to $\mathbf { M } _ { S S } \pmb { \Sigma } _ { S S } + \pmb { \Sigma } _ { S S } \mathbf { M } _ { S S } ^ { \top } + \mathbf { C } _ { S S } = 0$ . This is the Lyapunov equation associated with the induced subgraph G[S]. It involves only $\mathbf { M } _ { S S }$ , the observable covariance $\Sigma _ { S S }$ , and the known difusion block $\mathbf { C } _ { S S }$

If the induced model is globally identifiable, the equation uniquely determines $\mathbf { M } _ { S S }$ from $\Sigma _ { S S }$ For the generic case, let $\mathcal { A } _ { S }$ be the proper algebraic exceptional set on which the induced model may fail to be identifiable. Under the projection $\pi ( \mathbf { M } ) = \mathbf { M } _ { S S }$ the corresponding exceptional set in the full parameter space is $\ A _ { G } \ = \ \pi ^ { - 1 } ( \mathcal { A } _ { S } )$ . This set is algebraic and proper: any stable $\mathbf { M } _ { S S } \notin \mathcal { A } _ { S }$ can be extended to a stable full drift matrix of the form M = $\left( \begin{array} { c c } { \mathbf { M } _ { S S } } & { 0 } \\ { \mathbf { M } _ { R S } } & { \mathbf { M } _ { R R } } \end{array} \right)$ , by choosing ${ \bf M } _ { R R }$ stable. Thus, outside the proper exceptional set $\textstyle A _ { G } .$ , the block $\mathbf { M } _ { S S }$ lies outside $\mathcal { A } _ { S }$ and is uniquely identified from $\Sigma _ { S S }$ □

## 5 CAUSAL EFFECT IDENTIFICATION UNDER LATENT CONFOUNDING

Previous section studied identification of the complete drift matrix under full observability. We now consider the same class of potentially non-simple graphs under partial observability, where directed 2-cycles remain allowed. Since latent variables prevent full drift recovery from the observed covariance, we focus on identifying a target coeficient from a rank condition. Let $G = ( V , E )$ denote the graph associated with a stationary difusion process and let $O \subseteq V$ be the set of observed variables. We write $L : = V \backslash O$ for the latent variables. Herein, rather than recovering all model parameters, our objective is to identify causal efects from the observed stationary distribution.

Lemma 2. The retained covariance $\pmb { \Sigma } _ { O O }$ satisfies $M ^ { O } \Sigma _ { O O } + \Sigma _ { O O } ( M ^ { O } ) ^ { \top } + C _ { O O } = 0 ,$ , where $M ^ { O } : =$ $M _ { O O } + M _ { O L } \mathbf { T } _ { L \vert O }$ and $\mathbf { \Gamma } \Gamma _ { L | O } : = \Sigma _ { L O } \Sigma _ { O O } ^ { - 1 }$

By Lemma 2, the observed covariance matrix $\pmb { \Sigma } _ { O O }$ satisfies a Lyapunov equation associated with the efective observed drift matrix ${ \bf { M } } ^ { O }$ . When $( M ^ { O } ) _ { y x }$ is identifiable within the efective observed model, the relation below allows us to study when it coincides with the structural coeficient $m _ { y x }$ . For the target edge $x  y$

$$
( M ^ { O } ) _ { y x } = m _ { y x } + \sum _ { h \in L } m _ { y h } ( \Gamma _ { L | O } ) _ { h x } .
$$

Since $m _ { y h } = 0$ unless $h \in \operatorname { P a } ( y )$ , we obtain $( M ^ { O } ) _ { y x } =$ $m _ { y x }$ if and only $\begin{array} { r } { \mathrm { i f } \sum _ { h \in L \cap \operatorname { P a } ( y ) } m _ { y h } ( \Gamma _ { L | O } ) _ { h x } = 0 , } \end{array}$ Therefore, either $m _ { y h } = 0$ for all $h \in L , \mathrm { i . e . , P a } ( y ) \cap$ $L = \emptyset$ , or the summation vanishes, which is a nongeneric condition because it requires an exact cancellation among the free parameters.

Proposition 2 (Graphical suficient condition). Suppose that $( M ^ { O } ) _ { y x }$ is identifiable within the specified efective observed model. If, for every omitted parent $h \in L \cap \operatorname { P a } ( y ) , X _ { h } \bot X _ { x } \mid X _ { O \setminus \{ x \} }$ , then $( M ^ { O } ) _ { y x } =$ $m _ { y x }$ , and hence $m _ { y x }$ is identifiable from $( \pmb { \Sigma } _ { O O } , \pmb { C } _ { O O } )$

Proof. The condition implies $( \Gamma _ { L | O } ) _ { h x } = 0$ for every $h \in L \cap \operatorname { P a } ( y )$ . Therefore the bias term in the expression for $( M ^ { O } ) _ { y x }$ vanishes, giving $( M ^ { O } ) _ { y x } = m _ { y x } . \quad \square$

This condition is useful but limited. When $h \right. y \left.$ $x ,$ the trek-graph separation condition does not hold, since the path $h - y - x$ passes through the observed node $y \notin L \backslash h$ . Therefore, the graphical criterion does not guarantee $X _ { h } \perp X _ { x } \mid X _ { O \backslash x }$ This motivates the instrumental-variable approach below.

## 5.1 Instrumental Variables

Let $P = \mathrm { P a } ( y ) \cap O , U = \mathrm { P a } ( y ) \cap L$ . We represent every diagonal drift coeficient as a self-loop. Consequently, $y \in \operatorname { P a } ( y )$ and therefore $y \in P$ . The y-row of the drift is

$$
\sum _ { p \in P } m _ { y p } X _ { p } + \sum _ { u \in U } m _ { y u } X _ { u } .
$$

When $U \neq \emptyset$ , the observed parent set $P$ is generally not a valid adjustment set because marginalization introduces the bias term derived in the previous section. We therefore seek to recover the direct drift coeficients $m _ { y p }$ from observational covariances using instrumental variables. Let $Z = \{ z _ { 1 } , \dots , z _ { k } \} \subseteq O \setminus \mathrm { P a } ( y )$ be a set of observed instrumental variables. In particular, $P \cap Z = \emptyset$ , so no instrument has a direct edge into y.

Proposition 3 (Instrumental equations). For every $z \in Z$ , the $( y , z )$ -entry of the Lyapunov equation yields

$$
\sum _ { p \in P } m _ { y p } \Sigma _ { p z } + \sum _ { u \in U } m _ { y u } \Sigma _ { u z } + \sum _ { q \in \mathrm { P a } ( z ) } m _ { z q } \Sigma _ { y q } = - c _ { y z } ,
$$

since C is diagonal and $z \neq y ,$ so $c _ { y z } = 0$ . Here, q ranges over all parents $o f z .$

The last summation depends on the incoming coefi cients of z. Let $\begin{array} { r } { b _ { z } : = \sum _ { q \in \mathrm { P a } ( z ) } m _ { z q } \Sigma _ { y q } } \end{array}$

Lemma 3 (Root instruments). Suppose $z \in O$ has no incoming edges except its self-loop. Then $b _ { z } \ =$ $m _ { z z } \Sigma _ { y z }$ , where $\begin{array} { r } { m _ { z z } = - \frac { c _ { z z } } { 2 \Sigma _ { z z } } } \end{array}$ , is identifiable from the (z, z)-entry of the Lyapunov equation.

Throughout the remainder of this section, we assume that every instrument is a root node. We now impose an instrumental validity condition. The hidden parents of $y$ must be marginally independent with the instruments, that is, $\Sigma _ { u z } ~ = ~ 0$ for all $u \in U , z \in$ Z. Graphically, this condition holds generically when there is no trek between u and z. Conversely, if a trek between u and z exists, then $\Sigma _ { u z }$ is generically nonzero.

Lemma 4 (Linear identification system). Suppose $\Sigma _ { u z } ~ = ~ 0$ , for all $u \in U , z \in Z .$ , and suppose $b _ { z } \textit { } i s$ known for every $z \in Z$ . Then $\begin{array} { r } { \sum _ { p \in P } m _ { y p } \Sigma _ { p z } = - b _ { z } } \end{array}$

Collecting all instruments gives $\begin{array} { r } { \Sigma _ { P Z } ^ { \top } m _ { y P } = - b , \quad b = } \end{array}$ $( b _ { z _ { 1 } } , \cdots , b _ { z _ { k } } ) ^ { \top }$ The unknown vector is $\begin{array} { r l } { m _ { y P } } & { { } = } \end{array}$ $( m _ { y p } ) _ { p \in P } \in \mathbb { R } ^ { | P | }$

Proof. From the $( y , z )$ -entry of the Lyapunov equation, the instrumental validity assumption implies that $\Sigma _ { u z } = 0$ for every $u \in U , z \in Z ,$ , vanishing the hiddenparent terms. Moreover, because of the root instruments, the remaining known contribution is captured by b. Hence, $\begin{array} { r } { \sum _ { p \in P } m _ { y p } \Sigma _ { p z } + b _ { z } = 0 } \end{array}$ □

Since ran $\begin{array} { r } { \mathrm { k } ( \Sigma _ { P Z } ) \ \leq \ \operatorname* { m i n } \{ | P | , | Z | \} } \end{array}$ , full row rank requires $| Z | \ge | P |$ . Recall that P includes $y$ itself because the diagonal drift coeficient $m _ { y y }$ is treated as a self-loop coeficient.

Theorem 3 (Identification from instrumental variables). Assume that $\Sigma _ { u z } = 0$ for every $u \in U$ and $z \in Z , b _ { z }$ is known for every $z \in Z$ , rank $\left( \pmb { \Sigma } _ { P Z } \right) = \left| P \right|$ and $P \cap Z = \emptyset$ Then $m _ { y P }$ is uniquely identifiable from observational covariance. In particular, $m _ { y x }$ is identifiable for every $x \in P$

Proof. By Lemma 4, $m _ { y P }$ satisfies $\Sigma _ { P Z } ^ { \top } m _ { y P } = - b$ The matrix $\Sigma _ { P Z } \in \mathbb { R } ^ { | P | \times | Z | }$ . Since ran $\mathfrak { r } ( \pmb { \Sigma } _ { P Z } ) = | P |$ the matrix $\Sigma _ { P Z } ^ { \mathrm { ~ \tiny ~ | ~ } }$ has full column rank. Therefore $\Sigma _ { P Z } \Sigma _ { P Z } ^ { \top } m _ { y P } \stackrel { = } { = } - \Sigma _ { P Z } b$ , where $\pmb { \Sigma } _ { P Z } \pmb { \Sigma } _ { P Z } ^ { \top }$ is invertible. Thus, the linear system has at most one solution for $m _ { y P }$ . Since the true structural coeficient vector satisfies the system, the solution is unique. Thus every coeficient $m _ { y p } , p \in P $ , is identifiable. In particular, $m _ { y x }$ is identifiable. □

We now aim to give a graphical condition ensuring the rank assumption.

## 5.2 Graphical Conditions for the Instrument Rank Assumption

The identification result above requires ran $\operatorname { k } ( \pmb { \Sigma } _ { P Z } ) =$ $| P |$ . We now provide graphical conditions under which the rank assumption holds generically. The structure of the result depends on whether the instruments are directly connected to $P \backslash \{ y \}$ or whether directed paths contain intermediate vertices.

When there are no intermediate vertices, the problem reduces to a bipartite matching condition and admits an exact characterization through Hall’s theorem (Hall, 1987). With intermediate vertices, a system of vertex-disjoint directed paths provides a suficient condition for generic full rank. The latter condition is not necessary in general.

## 5.2.1 Direct Instruments

We first consider the case in which every directed path from an instrument to $P \backslash \{ y \}$ consists of a single edge.

Theorem 4 (Graphical rank characterization without intermediate vertices). Let $P ^ { \prime } = \{ p _ { 1 } , \cdot \cdot \cdot , p _ { r } \} = P \setminus$ $\{ y \}$ and let Z be a set ofroot instruments. Assume that no $z \in Z$ has a direct edge to y, and every directed path from Z to P<sup>′</sup> consists of a single ${ \mathcal { P } } d g e \ z \to p$ Define the bipartite graph $B = \left( Z , P ^ { \prime } , E _ { B } \right)$ , where $( z , p ) \in$ $E _ { B } \iff z \to p \in E$

Then, generically, ran $\operatorname { \ a k } ( \Sigma _ { P Z } ) ~ = ~ \left| P \right| ~ i f$ and only if there exist a parent $a \in P ^ { \prime }$ and two distinct instruments $z ^ { \prime } , z ^ { \prime \prime } \in Z$ such that $( i ) ~ z ^ { \prime }  ~ a$ and $z ^ { \prime \prime } \to a$ with $m _ { z ^ { \prime } z ^ { \prime } } \neq m _ { z ^ { \prime \prime } z ^ { \prime \prime } } ;$ and (ii) after deleting $a , z ^ { \prime } ,$ and $z ^ { \prime \prime }$ from $B ,$ the remaining bipartite graph contains a matching covering $P ^ { \prime } \setminus \{ a \}$

Proof sketch. For suficiency, we specialize the model parameters according to the prescribed matching and the two instruments adjacent to $a \ \in \ P ^ { \prime }$ , and show that the resulting covariance matrix contains a nonzero $| P | \times | P |$ minor. The matching guarantees independent contributions from all parents except $^ { a , }$ while the Lyapunov equation shows that the two instruments incident to a contribute an additional independent direction whenever $m _ { z ^ { \prime } z ^ { \prime } } \neq m _ { z ^ { \prime \prime } z ^ { \prime \prime } }$ . For necessity, generic full rank of $\Sigma _ { P Z }$ first implies that $\Sigma _ { P ^ { \prime } Z }$ has generic full rank, which is equivalent to the existence of a matching covering $P ^ { \prime }$ in the bipartite graph B. The remaining rank contribution must then come from the row indexed by $y ,$ and the Lyapunov equation shows that this is possible only if two instruments share a parent and have distinct self-drifts. □

## 5.2.2 Intermediate Vertices

When directed paths from Z to $P \setminus \{ y \}$ contain intermediate vertices, the support pattern of $\Sigma _ { P Z }$ is no longer represented by a bipartite graph alone. Nevertheless, a system of vertex-disjoint directed paths provides a suficient graphical condition for generic full rank.

Theorem 5 (Suficient graphical rank condition with intermediate nodes). Let $P ^ { \prime } = \{ p _ { 1 } , \cdot \cdot \cdot , p _ { r } \} = P \setminus \{ y \}$ and let Z be a set of root instruments. Assume that no $z \in Z$ has a direct edge to y.

Then, generically, rank $\ ( \pmb { \Sigma } _ { P Z } ) = | P |$ if there exist a parent $a \in P ^ { \prime }$ and two distinct instruments $z ^ { \prime } , z ^ { \prime \prime } \in$ $Z$ such that $( i ) \ \pi ^ { \prime } : \ z ^ { \prime } \  \ a$ and $\pi ^ { \prime \prime } : z ^ { \prime \prime }$ ⇝ a with $m _ { z ^ { \prime } z ^ { \prime } } \ne m _ { z ^ { \prime \prime } z ^ { \prime \prime } } ; \ ( i i ) \ t h e r e$ exist distinct instruments $w _ { 1 } , \cdot \cdot \cdot , w _ { r - 1 } \in Z \backslash \{ z ^ { \prime } , z ^ { \prime \prime } \}$ and pairwise vertexdisjoint direct paths $\rho _ { i } : w _ { i }  q _ { i }$ $; f o r i = 1 , \cdots , r - 1$ whose endpoints satisfy $\left\{ q _ { 1 } , \cdot \cdot \cdot q _ { r - 1 } \right\} = P ^ { \prime } \backslash \left\{ a \right\}$ ; and (iii) each $\rho _ { i }$ is vertex-disjoint from $\pi ^ { \prime } \cup \pi ^ { \prime \prime }$

Proof sketch. The proof follows the same strategy as Theorem 4. We specialize the model parameters so that only the prescribed paths are active. The vertexdisjointness assumptions isolate the covariance contributions of the paths $\rho _ { 1 } , \ldots , \rho _ { r - 1 } ,$ yielding independent covariance terms for the parents in $P ^ { \prime } \setminus \{ a \}$ , while ensuring that these paths do not interfere with the two paths terminating at a. The remaining $2 \times 2$ covariance block associated with $a , y ,$ and the instruments $z ^ { \prime } , z ^ { \prime \prime }$ is identical to that of Theorem 4 and is nonsingular whenever $m _ { z ^ { \prime } z ^ { \prime } } \neq m _ { z ^ { \prime \prime } z ^ { \prime \prime } }$ . Consequently, the selected $| P | \times | P |$ covariance minor is generically nonzero, implying rank $\mathbf { \nabla } \cdot ( \pmb { \Sigma } _ { P Z } ) = | P |$ □

Remark 1. The number of hidden parents of $y$ does not directly afect identifiability. Hidden parents may be arbitrarily many, provided that they are marginally uncorrelated with the instruments. Thus, the relevant obstruction is not the size of $U _ { ; }$ , but the existence of treks between U and $Z .$

Remark 2. The instrumental-variable results do not require the observed subgraph to be acyclic or simple. Reciprocal interactions may occur among the target, its observed parents, and intermediate observed vertices, provided that the instruments remain root vertices and the required trek-separation conditions between the instruments and latent parents are satisfied.

## 5.2.3 Bottleneck Investigation

Theorem 5 gives a suficient, but not necessary, graphical condition for generic full rank of $\Sigma _ { P Z }$ based on vertex-disjoint paths. We investigate whether recovery remains possible when the number of intermediate nodes is insuficient to satisfy this condition.

For each graph configuration, we generate 100 random stable drift matrices with diagonal difusion ma trices and compute the stationary covariance matrix directly from the Lyapunov equation. The generated graphs have a layered structure in which every directed path from $Z$ to $P ^ { \prime }$ passes through the set $Q$ of intermediate nodes, with no direct edges from $Z$ to $P ^ { \prime } .$ Hence, whenever $| Q | < | P ^ { \prime } | - 1$ , the $| P ^ { \prime } | - 1$ pairwise vertex-disjoint paths required by Theorem 5 cannot exist. We vary the bottleneck ratio $| Q | / | P ^ { \prime } |$ , recover $\hat { m } _ { y P ^ { \prime } }$ from $\Sigma _ { { P Z } } ,$ and report the mean relative error $\| \hat { m } _ { y P ^ { \prime } } - m _ { y P ^ { \prime } } \| _ { 2 } / \| m _ { y P ^ { \prime } } \| _ { 2 }$ . The graph generation procedure, parameter initializations, and implementation are provided in the Appendix.

Figure 3 shows that accurate recovery is possible even when $| Q | < | P ^ { \prime } | - 1$ , where the vertex-disjoint-path condition of Theorem 5 is guaranteed to fail. This demonstrates that the suficient graphical condition of Theorem 5 is not necessary for full-rank recovery. Moreover, larger parent sets require a higher bottleneck ratio before the estimation error decreases substantially, suggesting that recovering more coeficients requires proportionally more intermediate vertices.

## 6 RESTRICTIVENESS OF THEOREM 1

We empirically assess how restrictive the suficient graphical condition of Theorem 1 is relative to the kernel-rank criterion of Lemma 1; experimental details and the full results are reported in Appendix.

![](images/9bd04eae2a2dc70db95c0fcc940e31aaa8da6581c08bcaca0262bc8192806bc4.jpg)  
Figure 3: Recovery error of the target drift coeficients as a function of the bottleneck ratio $| Q | / | P ^ { \prime } |$ $Q$ is the intermediate nodes and $P ^ { \prime } = P \backslash \{ y \}$ Curves correspond to diferent numbers of parents. Latent parents $| U | = 5$

Among non-simple graphs with numerical full kernel rank, Theorem 1 certifies 94.4–97.8% of cases in the sparsest regime considered (edge probability 0.05). Coverage decreases with graph density, reaching 6.4% for $| V | = 1 2$ at edge probability 0.30. This decline is expected because denser graphs contain more overlapping 2-cycles, which increasingly violate the vertexdisjoint structure required by Theorem 1, even when the kernel matrix remains full rank. Thus, the theorem provides broad coverage in sparse regimes while leaving additional identifiable structures beyond its suficient conditions in denser graphs.

## 7 CONCLUSION AND FUTURE WORK

We studied identification from stationary covariance in non-simple linear SDEs through a common rankbased perspective. Directed 2-cycles are allowed in both the fully and partially observed settings. We established graphical conditions for generic identifiabil ity in the presence of directed 2-cycles, developed an instrumental-variable framework for identifying direct drift coeficients under latent confounding, and derived graphical conditions guaranteeing the required instrumental covariance rank. As this work focuses on the theoretical problem of identifiability, an important direction for future work is to investigate the statistical properties of the proposed framework under finite samples, including consistency, convergence rates, and robustness to covariance estimation error. Additional directions include extending the graphical identifiability results to broader classes of reciprocal structures and relaxing assumptions on the difusion matrix.

## References

<sup>˚</sup>Astr¨om, K. J. (2012). Introduction to stochastic control theory. Courier Corporation.

<sup>˚</sup>Astr¨om, K. J. and Murray, R. (2021). Feedback systems: an introduction for scientists and engineers. Princeton university press.

Boege, T., Drton, M., Hollering, B., Lumpp, S., Misra, P., and Schkoda, D. (2025). Conditional independence in stationary distributions of difusions. Stochastic Processes and their Applications, 184:104604.

Bongers, S. and Mooij, J. M. (2018). From random differential equations to structural causal models: The stochastic case. arXiv preprint arXiv:1803.08784, 3.

Cruz, K. G., Leow, Y. N., Le, N. M., Adam, E., Huda, R., and Sur, M. (2023). Cortical-subcortical interactions in goal-directed behavior. Physiological Reviews, 103(1):347–389. PMID: 35771984.

Dettling, P., Homs, R., Am´endola, C., Drton, M., and Hansen, N. R. (2023). Identifiability in continuous lyapunov models. SIAM Journal on Matrix Analysis and Applications, 44(4):1799–1821.

Ferrell Jr, J. E. (2002). Self-perpetuating states in signal transduction: positive feedback, double-negative feedback and bistability. Current opinion in cell biology, 14(2):140–148.

Fitch, K. (2019). Learning directed graphical models from gaussian data. arXiv preprint arXiv:1906.08050.

Foygel, R., Draisma, J., and Drton, M. (2012). Halftrek criterion for generic identifiability of linear structural equation models. The Annals of Statistics, pages 1682–1713.

Foygel Barber, R., Drton, M., Sturma, N., and Weihs, L. (2022). Half-trek criterion for identifiability of latent variable models. arXiv e-prints, pages arXiv– 2201.

Hall, P. (1987). On representatives of subsets. In Classic papers in Combinatorics, pages 58–62. Springer.

Hamilton, J. D. (2020). Time series analysis. Princeton university press.

Hansen, N. and Sokol, A. (2014). Causal interpretation of stochastic diferential equations.

Hansen, N. R. (2025). A trek rule for the lyapunov equation. Algebraic Statistics, 16(2):95–112.

Horn, R. A. and Johnson, C. R. (2012). Matrix analysis. Cambridge university press.

Kumor, D., Chen, B., and Bareinboim, E. (2019). Eficient identification in linear structural causal models with instrumental cutsets. Advances in Neural Information Processing Systems, 32.

Lorch, L., Krause, A., and Sch¨olkopf, B. (2024). Causal modeling with stationary difusions. In International Conference on Artificial Intelligence and Statistics, pages 1927–1935. PMLR.

Lorch, L., Zhang, J., Bunne, C., Krause, A., Sch¨olkopf, B., and Uhler, C. (2026). Latent causal difusions for single-cell perturbation modeling. arXiv preprint arXiv:2601.15341.

Marbach, D., Costello, J. C., K¨ufner, R., Vega, N. M., Prill, R. J., Camacho, D. M., Allison, K. R., Kellis, M., Collins, J. J., et al. (2012). Wisdom of crowds for robust gene network inference. Nature methods, 9(8):796–804.

Mogensen, S. W. (2023). Instrumental processes using integrated covariances. In Conference on Causal Learning and Reasoning, pages 620–641. PMLR.

Mogensen, S. W., Malinsky, D., and Hansen, N. R. (2018). Causal learning for partially observed stochastic dynamical systems. In UAI, pages 350– 360.

Øksendal, B. (2003). Stochastic diferential equations. In Stochastic diferential equations: an introduction with applications, pages 38–50. Springer.

Pearl, J. (2009). Causality. Cambridge university press.

Perkovi´c, E., Textor, J., Kalisch, M., and Maathuis, M. H. (2018). Complete graphical characterization and construction of adjustment sets in markov equivalence classes of ancestral graphs. Journal of Machine Learning Research, 18(220):1–62.

Richardson, T. and Spirtes, P. (2002). Ancestral graph markov models. The Annals of Statistics, 30(4):962– 1030.

Shpitser, I. and Pearl, J. (2006). Identification of joint interventional distributions in recursive semimarkovian causal models. In AAAI, pages 1219– 1226.

Shpitser, I., VanderWeele, T., and Robins, J. M. (2012). On the validity of covariate adjustment for estimating causal efects. arXiv preprint arXiv:1203.3515.

van Seeventer, G. and Salehkaleybar, S. (2026). Sign identifiability of causal efects in stationary stochastic dynamical systems. arXiv preprint arXiv:2603.08311.

Varando, G. and Hansen, N. R. (2020). Graphical continuous lyapunov models. In Conference on Uncertainty in Artificial Intelligence, pages 989–998. Pmlr.

# Appendix

## A PROOFS

## A.1 Proof of Theorem 1

Proof. By Lemma 1, it sufices to show that there exists $\mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf \Sigma \mathbf { \Sigma } \mathbf { \Sigma } \mathbf \Sigma \Sigma \mathbf { \Sigma } \mathbf \Sigma \Sigma \Sigma \mathbf { \Sigma } \mathbf \Sigma \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma$ such that rank $\begin{array} { r } { ( { \bf H } ( { \bf \underline { { \Sigma } } } ) _ { E ^ { c } , \cdot } ) = { \bf \Xi } _ { 2 } ^ { d } } \end{array}$ , where $| V | = d .$

Let the directed 2-cycles of G be $c _ { m } = \{ i _ { m } , j _ { m } \} , m = 1 , \ldots , s ,$ and let $k _ { m }$ denote their witnesses from the assumption of the theorem. Thus, after possibly exchanging $i _ { m }$ and $j _ { m } , k _ { m } \sim i _ { m }$ and $k _ { m } \nsim j _ { m }$ , each $k _ { m }$ belongs to no directed 2-cycle, and the vertices $k _ { 1 } , \ldots , k _ { s }$ are pairwise distinct.

Define $N _ { m } \ : = \ \{ i _ { m } , j _ { m } , k _ { m } \}$ and $R \ : = \ V \ \backslash \ \bigcup _ { m = 1 } ^ { s } N _ { m }$ . By the assumptions of the theorem, the sets $N _ { 1 } , \dots , N _ { s } , \{ r \} _ { r \in R }$ are pairwise disjoint and form a partition of V.

Parameter specialization. For every vertex $v \in V .$ , choose $m _ { v v } = - \alpha _ { v }$ , with $\alpha _ { v } > 0$

For each neighborhood $N _ { m } = \{ i _ { m } , j _ { m } , k _ { m } \}$ , retain all three edges $i _ { m }  j _ { m } , j _ { m }  i _ { m } .$ , and the unique directed edge between $i _ { m }$ and $k _ { m }$ . Choose all three corresponding drift coeficients strictly positive and nonzero.

Set all remaining of-diagonal coeficients equal to zero. In particular, this disconnects distinct neighborhoods from one another in the chosen parameter specialization, but both edges of every directed 2-cycle remain nonzero.

For a fixed neighborhood $N _ { m } = \{ i , j , k \}$ , write $m _ { i i } = - a , m _ { j j } = - b , m _ { k k } = - c ,$ with $a , b , c > 0 ,$ , and denote the two coeficients of the directed 2-cycle by $m _ { j i } = p > 0$ and $m _ { i j } = q > 0$ . Thus p corresponds to $i  j$ and $q$ to $j  i .$

Choose these coeficients such that $p q < a b .$ The $2 \times 2$ drift block on $\{ i , j \}$ is then Hurwitz. Since the witness edge is one-directional, the $3 \times 3$ block on $N _ { m }$ is block triangular after an appropriate ordering, with the eigenvalue $- c$ in addition to the two eigenvalues of the $2 \times 2$ cycle block. Hence every neighborhood block is Hurwitz. Therefore the full specialized matrix M is Hurwitz.

Since C is diagonal positive definite, the Lyapunov equation $\mathbf { M } \mathbf { \Sigma } \mathbf { \Sigma } + \mathbf { \Sigma } \mathbf { M } ^ { \top } + \mathbf { C } = 0$ has a unique positive definite solution Σ.

Moreover, M and C are block diagonal with respect to $N _ { 1 } , \dots , N _ { s } , \{ r \} _ { r \in R }$ . By uniqueness of the Lyapunov solution, Σ is block diagonal with respect to the same partition. Consequently, $\Sigma _ { u v } = 0$ whenever u and v belong to distinct blocks.

We choose the positive of-diagonal coeficients $p , q ,$ and the witness-edge coeficient in every neighborhood, suficiently small that each covariance block $\pmb { \Sigma } _ { N _ { m } N _ { m } }$ is strictly diagonally dominant:

$$
\Sigma _ { v v } > \sum _ { u \in N _ { m } \atop u \neq v } | \Sigma _ { u v } | , \qquad v \in N _ { m } .
$$

Such a choice is possible because the stationary covariance depends continuously on the drift coeficients and can be made arbitrarily close to a positive diagonal matrix by choosing all of-diagonal coeficients suficiently small. Importantly, all coeficients $p , q$ are chosen strictly positive; no directed 2-cycle is removed.

We will also use that $\Sigma _ { i k } > 0$ for every neighborhood. Indeed, the specialized drift matrix is Metzler, since all retained of-diagonal coeficients are positive. Hence $e ^ { t \mathbf { M } } \geq 0$ entrywise for $t \geq 0$ . Moreover, because there is a direct positive edge between i and $k ,$ the corresponding entry of $\mathbf { \phi } _ { e } ^ { \left( \mathbf { \phi } _ { e } \right) }$ is strictly positive for $t > 0$ . Using the

stationary covariance representation

$$
\Sigma = \int _ { 0 } ^ { \infty } e ^ { t \mathbf { M } } \mathbf { C } e ^ { t \mathbf { M } ^ { \top } } d t ,
$$

with C diagonal positive definite, it follows that $\Sigma _ { i k } > 0$

Local blocks. Fix one directed 2-cycle $i  j$ with witness $k ,$ where $k \sim i$ and $k \not \sim j$

Consider the three covariance columns $( i , j ) , ( i , k ) , ( j , k )$ . The displayed matrices use these oriented column labels for notational convenience. Changing the fixed ordering convention for an unordered covariance column only changes its sign and therefore does not afect nonsingularity.

Suppose first that $i  k .$ Then $k  i , k  j ,$ and $j \to k$ are non-edges. Selecting the rows $j \to k , k \to i$ , and $k  j \mathrm { \ g i v e s }$ , up to harmless row and column signs,

$$
{ \bf H } _ { m } = \left( \begin{array} { c c c } { { \Sigma _ { i k } } } & { { 0 } } & { { - \Sigma _ { k k } } } \\ { { 0 } } & { { \Sigma _ { i i } } } & { { \Sigma _ { i j } } } \\ { { 0 } } & { { \Sigma _ { i j } } } & { { \Sigma _ { j j } } } \end{array} \right) .
$$

Therefore

$$
\operatorname* { d e t } ( \mathbf { H } _ { m } ) = \Sigma _ { i k } \left( \Sigma _ { i i } \Sigma _ { j j } - \Sigma _ { i j } ^ { 2 } \right) .
$$

Since $\Sigma \succ 0 .$ the principal $2 \times 2$ minor satisfies $\Sigma _ { i i } \Sigma _ { j j } - \Sigma _ { i j } ^ { 2 } > 0$ . Moreover, $\Sigma _ { i k } > 0$ . Thus $\operatorname* { d e t } ( \mathbf { H } _ { m } ) \neq 0$

Now suppose that $k  i .$ . Then $i  k , k  j$ , and $j \to k$ are non-edges. Selecting these three rows gives

$$
{ \bf H } _ { m } = \left( \begin{array} { c c c } { - \Sigma _ { j k } } & { - \Sigma _ { k k } } & { 0 } \\ { 0 } & { \Sigma _ { i j } } & { \Sigma _ { j j } } \\ { \Sigma _ { i k } } & { 0 } & { - \Sigma _ { k k } } \end{array} \right) .
$$

Hence

$$
\operatorname* { d e t } ( \mathbf { H } _ { m } ) = \Sigma _ { k k } \left( \Sigma _ { i j } \Sigma _ { j k } - \Sigma _ { i k } \Sigma _ { j j } \right) .
$$

We now show directly from the Lyapunov equation that the expression in parentheses is nonzero.

Recall that $m _ { j i } = p > 0$ is the coeficient corresponding to the edge $i  j$ . Since $k  i$ and there is no edge between j and k, the $( j , k )$ )-entry of the Lyapunov equation gives $p \Sigma _ { i k } - ( b + c ) \Sigma _ { j k } = 0$ . Therefore $\begin{array} { r } { \Sigma _ { j k } = \frac { p } { b + c } \Sigma _ { i k } } \end{array}$ The $( j , j )$ )-entry of the Lyapunov equation gives $2 p \Sigma _ { i j } - 2 b \Sigma _ { j j } + C _ { j j } = 0$ , and hence $\begin{array} { r } { p \Sigma _ { i j } = b \Sigma _ { j j } - \frac { C _ { j j } } { 2 } } \end{array}$ Consequently,

$$
\begin{array} { c } { { \sum _ { i j } \sum _ { j k } - \sum _ { i k } \sum _ { j j } = \Sigma _ { i k } \left( \displaystyle \frac { p \Sigma _ { i j } } { b + c } - \Sigma _ { j j } \right) } } \\ { { = \Sigma _ { i k } \left( \displaystyle \frac { b \Sigma _ { j j } - C _ { j j } / 2 } { b + c } - \Sigma _ { j j } \right) } } \\ { { = - \displaystyle \frac { \sum _ { i k } } { b + c } \left( c \Sigma _ { j j } + \displaystyle \frac { C _ { j j } } { 2 } \right) . } } \end{array}
$$

Since $\Sigma _ { i k } > 0 , \Sigma _ { j j } > 0 , C _ { j j } > 0$ , and $b + c > 0$ , we obtain $\Sigma _ { i j } \Sigma _ { j k } - \Sigma _ { i k } \Sigma _ { j j } < 0$ . Therefore det $\left( \mathbf { H } _ { m } \right) \neq 0$

Thus, in both possible orientations of the witness edge, the three local columns admit a nonsingular $3 \times 3$ minor while both edges of the directed 2-cycle remain nonzero.

Mixed blocks. It remains to consider covariance columns whose vertices belong to distinct blocks among $N _ { 1 } , \dots , N _ { s } , \{ r \} _ { r \in R }$

Let A and B be two distinct such blocks. Consider all columns $( a , b ) , a \in A , b \in B$ . There are $| A | | B |$ such columns.

For every $a \ \in \ A$ and $b \in B ,$ , at least one of $a \ \to \ b$ and $b  a$ belongs to $E ^ { c }$ . Otherwise a and b would themselves form a directed 2-cycle. This would contradict either the vertex-disjointness of the directed 2-cycles, the assumption that no witness $k _ { m }$ belongs to a directed 2-cycle, or the definition of $R .$

For every mixed column $( a , b )$ , select one available non-edge row, either $a \to b$ or $b \to a .$ . Distinct mixed columns give distinct selected rows. Denote the resulting square submatrix by $\mathbf { H } _ { A B }$ . We show that $\mathbf { H } _ { A B }$ is strictly diagonally dominant.

Suppose the row selected for column $( a , b )$ is $a  b$ . Its entry in the corresponding column is, up to $\mathrm { s i g n } , \Sigma _ { b b }$ Within $\mathbf { H } _ { A B }$ , this row may also contain nonzero entries in columns $( a , c ) , c \in B , c \neq b .$ These entries are, up to $\mathrm { s i g n } , \Sigma _ { c b }$ . In particular, these terms need not vanish when b and c belong to the same neighborhood $N _ { m }$

Nevertheless, by the strict diagonal dominance of $\Sigma _ { B B }$

$$
\Sigma _ { b b } > \sum _ { \stackrel { c \in \cal B } { c \neq b } } | \Sigma _ { c b } | .
$$

Hence this row of $\mathbf { H } _ { A B }$ is strictly diagonally dominant.

If instead the selected row is $b \to a$ , its diagonal entry is, up to sign, $\Sigma _ { a a }$ , and its possible of-diagonal entries have the form $\textstyle \sum _ { c a , \ c } \in A , c \neq a$ . Therefore,

$$
\Sigma _ { a a } > \sum _ { \stackrel { c \in \cal { A } } { c \neq a } } | \Sigma _ { c a } | .
$$

This row is also strictly diagonally dominant.

Thus every row of $\mathbf { H } _ { A B }$ is strictly diagonally dominant. By the Levy–Desplanques theorem Horn and Johnson (2012), $\mathbf { H } _ { A B }$ is nonsingular.

This argument includes the case in which one block is a singleton $\{ r \} \subseteq R$ If both blocks are singletons, the corresponding block is simply $1 \times 1$ with nonzero entry equal, up to sign, to a positive variance.

Combining the blocks. For each $N _ { m } .$ , select the three local rows defining $\mathbf { H } _ { m }$ . For each unordered pair of distinct blocks $A , B ,$ select the rows defining $\mathbf { H } _ { A B }$

Together these rows correspond to all $\binom { d } { 2 }$ covariance columns and form a square $\textstyle { { \binom { d } { 2 } } \times { \binom { d } { 2 } } }$ submatrix of $\mathbf { H } ( \pmb { \Sigma } ) _ { E ^ { c } , \cdot }$ After suitable row and column permutations, this selected matrix is block diagonal.

Indeed, a row selected for a pair of blocks A, B can have a nonzero entry only in columns corresponding to the same pair of blocks. Any apparent entry involving a third block contains a covariance between distinct covariance blocks and therefore vanishes because Σ is block diagonal. Similarly, mixed rows vanish on local columns, and local rows associated with $N _ { m }$ vanish on columns associated with $N _ { \ell } , m \neq \ell .$

Thus the selected minor has the form

$$
\operatorname { d i a g } \left( \mathbf { H } _ { 1 } , \dots , \mathbf { H } _ { s } , \left\{ \mathbf { H } _ { A B } : A \neq B \right\} \right) .
$$

Every local block $\mathbf { H } _ { m }$ is nonsingular, and every mixed block $\mathbf { H } _ { A B }$ is nonsingular. Hence the entire selected minor is nonsingular.

Therefore, rank $\begin{array} { r } { ( \mathbf { H } ( \Sigma ) _ { E ^ { c } , \cdot } ) = { \binom { d } { 2 } } = \frac { d ( d - 1 ) } { 2 } } \end{array}$ . By Lemma 1, $\mathcal { M } _ { G , \mathbf { C } }$ is generically identifiable.

## A.2 Proof of Proposition 1

Proof. Let G be one of the four-node extensions listed in Proposition 1. Since $d = 4 .$ , the matrix $\mathbf { H } ( \pmb { \Sigma } ) _ { E ^ { c } }$ has ${ \binom { 4 } { 2 } } \ = \ 6$ columns. By Lemma 1, it is suficient to find one covariance matrix $\Sigma \in \mathcal { M } _ { G , C }$ such that rank $\left( \mathbf { \bar { H } } ( \Sigma ) _ { E ^ { c } , \cdot } \right) = 6$ . Equivalently, it is enough to exhibit a nonsingular $6 \times 6$ minor of $\mathbf { H } ( \pmb { \Sigma } ) _ { E ^ { c } } ,$

Some extensions satisfy the assumptions of Theorem 1 directly. In particular, if the added vertex is adjacent to exactly one endpoint of an isolated directed 2-cycle, generic identifiability follows immediately from Theorem 1. We therefore only consider the extensions that are not covered by that result.

For each remaining graph, we select six rows indexed by non-edges and consider the determinant of the corresponding $6 \times 6$ minor. Since every entry of $\mathbf { H } ( \pmb { \Sigma } )$ is linear in the entries of $\Sigma ,$ this determinant is a polynomial in

the covariance entries. Therefore, it is suficient to show that it is nonzero for one admissible covariance matrix. This proves that the determinant is not identically zero and hence that the selected rows are generically linearly independent.

Common-child extension. Label the common child by 1, the vertices of the directed 2-cycle by 2 and 3, and the added vertex by 4. Thus, the original motif contains $2  1 , 3  1 , 2  3$ , and $3  2 .$ Order the columns of the kernel matrix as (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4).

Consider first the extension $1  4$ . Using the non-edge rows $1  2 , 1  3 , 4  2 , 4  3 , 2  4 ,$ and $3  4 ,$ we obtain the minor

$$
K _ { \mathrm { c c } } ( \Sigma ) = \left( \begin{array} { c c c c c c c } { - \Sigma _ { 2 2 } } & { - \Sigma _ { 2 3 } } & { - \Sigma _ { 2 4 } } & { 0 } & { 0 } & { 0 } \\ { - \Sigma _ { 2 3 } } & { - \Sigma _ { 3 3 } } & { - \Sigma _ { 3 4 } } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { \Sigma _ { 1 2 } } & { 0 } & { \Sigma _ { 2 2 } } & { \Sigma _ { 2 3 } } \\ { 0 } & { 0 } & { \Sigma _ { 1 3 } } & { 0 } & { \Sigma _ { 2 3 } } & { \Sigma _ { 3 3 } } \\ { \Sigma _ { 1 4 } } & { 0 } & { 0 } & { - \Sigma _ { 3 4 } } & { - \Sigma _ { 4 4 } } & { 0 } \\ { 0 } & { \Sigma _ { 1 4 } } & { 0 } & { \Sigma _ { 2 4 } } & { 0 } & { - \Sigma _ { 4 4 } } \end{array} \right) .
$$

The directed paths $2  1  4$ and $3  1  4$ are treks. Therefore, $\Sigma _ { 2 4 }$ and $\Sigma _ { 3 4 }$ are not structurally forced to vanish.

To give an exact rank certificate, take $C = I _ { 4 }$ and

$$
M ^ { \star } = \left( \begin{array} { c c c c } { { - 4 } } & { { 1 } } & { { 1 } } & { { 0 } } \\ { { 0 } } & { { - 5 } } & { { 1 } } & { { 0 } } \\ { { 0 } } & { { 1 } } & { { - 6 } } & { { 0 } } \\ { { 1 } } & { { 0 } } & { { 0 } } & { { - 7 } } \end{array} \right) .
$$

This matrix has the required zero pattern and is Hurwitz by strict diagonal dominance. Let $\Sigma ^ { \star }$ be the unique solution of $M ^ { \star } \Sigma ^ { \star } { + } \Sigma ^ { \star } ( M ^ { \star } ) ^ { \top } { + } I _ { 4 } = 0$ . Exact evaluation gives det $K _ { \mathrm { c c } } ( \pmb { \Sigma } ^ { \star } )$ = 23289823567/15776105854472994800 $\neq$ 0. Hence the determinant of $K _ { \mathrm { c c } } ( \Sigma )$ is not identically zero. The six selected rows are therefore generically linearly independent, and the graph with $1  4$ is generically identifiable.

The optional reverse edge $4  1$ does not remove any of the selected non-edge rows. Moreover, because $\mathcal { M } _ { G , C }$ is defined by $\operatorname { s u p p } ( M ) \subseteq E$ , the same parameter point remains admissible for the graph containing $1  4$ and $4  1$ by setting the coeficient associated with the optional edge equal to zero. Thus the same minor also proves generic identifiability when both directions are present.

The reverse orientation 4 → 1 alone is excluded. Indeed, the paths $4 \right. 1 \left. 2$ and $4 \right. 1 \left. 3$ contain a collider at 1. Hence there is no trek from 4 to either 2 or $^ { 3 , }$ and the trek rule gives $\Sigma _ { 2 4 } = \Sigma _ { 3 4 } = 0$ . The column of $\mathbf { H } ( \pmb { \Sigma } ) _ { E ^ { c } } ,$ <sub>·</sub> indexed by (2, 3) is then identically zero: its only potentially nonzero entries occur in the rows $2  4$ and $3  4$ , where they are respectively $- \Sigma _ { 3 4 }$ and $\Sigma _ { 2 4 }$ . Consequently, rank $\mathbf { \Big ( H ( } \Sigma ) _ { E ^ { c } , \cdot } \Big ) < 6$ . This explains the exception in the common-child case.

Remaining extensions. The other extensions are proved by the same rank argument. For each labelled configuration G, the exact certificate is constructed computationally in the supplemental code. Specifically, the code constructs a rational Hurwitz drift matrix $M _ { G } ^ { \star }$ with the required graph support, sets $C = I _ { 4 }$ , solves the Lyapunov equation exactly over the rationals to obtain $\Sigma _ { G } ^ { \star } .$ , constructs ${ \bf H } ( \Sigma _ { G } ^ { \star } ) _ { E ^ { c } , \cdot }$ , and selects six non-edge rows forming a nonsingular $6 \times 6$ minor. Thus, for the selected row set $R _ { G }$ , with $K _ { G } ( \Sigma ) = \mathbf { H } ( \Sigma ) _ { R _ { G } , }$ , the supplemental code provides an exact certificate satisfying det $K _ { G } ( \Sigma _ { G } ^ { \star } ) \neq 0$

Therefore, in each case, the determinant is not identically zero and the selected six rows are generically linearly independent. The supplemental material contains the corresponding parameter specialization, selected non-edge rows, and exact nonzero determinant for each labelled graph.

Configurations obtained by relabeling vertices only permute the rows and columns of the selected minor and therefore preserve its rank. Similarly, when an indicated reverse edge is optional, adding that edge does not remove the rows of the original certificate, and the same parameter specialization remains admissible after assigning zero to the optional coeficient.

Thus, for every extension listed in Proposition 1, there exists $\pmb { \Sigma } \in \mathcal { M } _ { \mathbf { G } , \mathbf { C } }$ such that rank $\big ( \mathbf { H } ( \pmb { \Sigma } ) _ { E ^ { c } , \cdot } \big ) = 6$ . The conclusion follows from Lemma 1. □

## A.3 Proof of Lemma 2

Proof. Partition the drift matrix and stationary covariance according to the observed and hidden variables:

$$
\mathbf { M } = \left( \mathbf { M } _ { O O } \quad \mathbf { M } _ { O L } \right) , \qquad \pmb { \Sigma } = \left( \begin{array} { l l } { \pmb { \Sigma } _ { O O } } & { \pmb { \Sigma } _ { O L } } \\ { \pmb { \Sigma } _ { L O } } & { \mathbf { M } _ { L L } } \end{array} \right) .
$$

Since Σ is a stationary covariance matrix, it satisfies $\mathbf { M } \mathbf { \Sigma } \mathbf { \Sigma } + \mathbf { \Sigma } \mathbf { M } ^ { \top } + \mathbf { C } = 0$ . Taking the (O, O)-block gives

$$
\mathbf { M } _ { O O } \boldsymbol { \Sigma } _ { O O } + \mathbf { M } _ { O L } \boldsymbol { \Sigma } _ { L O } + \boldsymbol { \Sigma } _ { O O } \mathbf { M } _ { O O } ^ { \top } + \boldsymbol { \Sigma } _ { O L } \mathbf { M } _ { O L } ^ { \top } + \mathbf { C } _ { O O } = 0 .
$$

Because Σ is positive definite, $\pmb { \Sigma } _ { O O }$ is invertible. By the definition $\Gamma _ { L | O } = \Sigma _ { L O } \Sigma _ { O O } ^ { - 1 }$ , we have $\Sigma _ { L O } = \Gamma _ { L | O } \Sigma _ { O O }$ Moreover, using the symmetry of $\Sigma , \Sigma _ { O L } = \Sigma _ { L O } ^ { \top } = \Sigma _ { O O } \Gamma _ { L | O } ^ { \top }$

Therefore, $\mathbf { M } _ { O L } \pmb { \Sigma } _ { L O } = \mathbf { M } _ { O L } \mathbf { T } _ { L | O } \pmb { \Sigma } _ { O O }$ , and $\pmb { \Sigma } _ { O L } \mathbf { M } _ { O L } ^ { \top } = \pmb { \Sigma } _ { O O } \mathbf { \Gamma } _ { L | O } ^ { \top } \mathbf { M } _ { O L } ^ { \top }$

Substituting these expressions into the $( O , O )$ -block equation yields

$$
\big ( \mathbf { M } _ { O O } + \mathbf { M } _ { O L } \mathbf { r } _ { L | O } \big ) \boldsymbol { \Sigma } _ { O O } + \boldsymbol { \Sigma } _ { O O } \big ( \mathbf { M } _ { O O } + \mathbf { M } _ { O L } \mathbf { r } _ { L | O } \big ) ^ { \top } + \mathbf { C } _ { O O } = 0 .
$$

Defining $\mathbf { M } ^ { O } : = \mathbf { M } _ { O O } + \mathbf { M } _ { O L } \mathbf { r } _ { L | O }$ gives $\mathbf { M } ^ { O } \Sigma _ { O O } + \Sigma _ { O O } ( \mathbf { M } ^ { O } ) ^ { \top } + \mathbf { C } _ { O O } = 0$ , as claimed.

## A.4 Proof of Proposition 3

Proof. Since the stationary covariance matrix satisfies the continuous Lyapunov equation, its $( y , z )$ -entry is $( \boldsymbol { \mathbf { M } } \boldsymbol { \Sigma } ) _ { y z } + ( \boldsymbol { \Sigma } \boldsymbol { \mathbf { M } } ^ { \top } ) _ { y z } + c _ { y z } = 0$

Because C is diagonal and $z \neq y .$ , we have $c _ { y z } = 0$ . Moreover, $\begin{array} { r } { ( \mathbf { M } \pmb { \Sigma } ) _ { y z } = \sum _ { v \in V } m _ { y v } \Sigma _ { v z } . } \end{array}$ Since $m _ { y v } = 0$ whenever $v \not \in { \mathrm { P a } } ( y )$ $\begin{array} { r } { ( \mathbf { M } \Sigma ) _ { y z } = \sum _ { p \in P } m _ { y p } \Sigma _ { p z } + \sum _ { u \in U } m _ { y u } \Sigma _ { u z } } \end{array}$ , where $P = \mathrm { P a } ( y ) \bar { \cap } O$ and $U = \mathrm { P a } ( y ) \cap L$ Similarly, $\begin{array} { r } { ( \boldsymbol { \Sigma } \boldsymbol { \mathbf { M } } ^ { \top } ) _ { y z } = \sum _ { v \in V } \boldsymbol { \Sigma } _ { y v } ( \boldsymbol { \mathbf { M } } ^ { \top } ) _ { v z } = \sum _ { v \in V } \boldsymbol { \Sigma } _ { y v } m _ { z v } } \end{array}$ . Since $m _ { z v } = 0$ whenever v $\not \in \mathrm { P a } ( z ) , ( \mathbf { \Sigma M } ^ { \top } ) _ { y z } =$ $\begin{array} { r } { \sum _ { q \in \operatorname { P a } ( z ) } m _ { z q } \sum _ { y q } . } \end{array}$

Substituting these expressions into the $( y , z )$ -entry of the Lyapunov equation yields

$$
\sum _ { p \in P } m _ { y p } \Sigma _ { p z } + \sum _ { u \in U } m _ { y u } \Sigma _ { u z } + \sum _ { q \in \mathrm { P a } ( z ) } m _ { z q } \Sigma _ { y q } = 0 ,
$$

as claimed.

## A.5 Proof of Lemma 3

Proof. Since z has no parents other than itself, the parents set is $\mathrm { P a } ( z ) = \{ z \}$ . Hence $\begin{array} { r } { b _ { z } = \sum _ { q \in \mathrm { P a } ( z ) } m _ { z q } \Sigma _ { y q } = } \end{array}$ $m _ { z z } \Sigma _ { y z }$

It remains to show that $m _ { z z }$ is identifiable. Consider the (z, z)-entry of the Lyapunov equation $\begin{array} { r } { 2 \sum _ { v \in \mathrm { P a } ( z ) } m _ { z v } \Sigma _ { v z } + c _ { z z } = 0 } \end{array}$ . Since $P a ( z ) = \{ z \}$ , hence $2 m _ { z z } \Sigma _ { z z } + c _ { z z } = 0$

As Σ is a covariance matrix and z has nonzero stationary variance, we have: $\Sigma _ { z z } > 0$ . Thus $\begin{array} { r } { m _ { z z } = - \frac { c _ { z z } } { 2 \Sigma _ { z z } } } \end{array}$ Both $c _ { z z }$ and $\Sigma _ { z z }$ are observable or assumed known under the diagonal difusion parametrization, and hence $b _ { z } = m _ { z z } \Sigma _ { y z }$ and, consequently, $b _ { z }$ is identifiable. □

## A.6 Proof of Theorem 4

Proof. We first derive the covariance relation responsible for the additional rank. The $( z , y )$ -entry of the Lyapunov equation is $\begin{array} { r } { \sum _ { p \in P ^ { \prime } } m _ { y p } \Sigma _ { p z } + m _ { y y } \Sigma _ { y z } + m _ { z z } \Sigma _ { y z } = 0 } \end{array}$ . Hence

$$
\Sigma _ { y z } = - \frac { \displaystyle \sum _ { p \in P ^ { \prime } } m _ { y p } \Sigma _ { p z } } { m _ { y y } + m _ { z z } } .\tag{3}
$$

Suficiency Suppose there exist $a , z ^ { \prime } , z ^ { \prime \prime }$ , and a matching as in the statement. Let $P ^ { \prime } \backslash \{ a \} = \{ q _ { 1 } , \cdot \cdot \cdot , q _ { r - 1 } \}$ and let $w _ { i } \in Z \setminus \{ z ^ { \prime } , z ^ { \prime \prime } \}$ be the instrument matched to $q _ { i }$ . Consider the $( r + 1 ) \times ( r + 1 )$ submatrix

$$
\begin{array} { r } { \mathbf { K } = \pmb { \Sigma } _ { P Z } = \left( \begin{array} { l } { \sum _ { P ^ { \prime } , \{ z ^ { \prime } , z ^ { \prime \prime } , w _ { 1 } , \dots , w _ { r - 1 } \} } } \\ { \sum _ { y , \{ z ^ { \prime } , z ^ { \prime \prime } , w _ { 1 } , \dots , w _ { r - 1 } \} } } \end{array} \right) . } \end{array}
$$

To prove generic nonsingularity, it is enough to exhibit one parameter choice for which $\operatorname* { d e t } ( \mathbf { K } ) \neq 0$ . Indeed, after clearing the nonzero denominators in (3), det(K) is a rational function whose numerator is a polynomial in the model parameters. Showing that it is nonzero at one parameter point proves that this polynomial is not identically zero.

We construct a parameter specialization of the drift matrix. Retain only the of-diagonal coeficients corresponding to the edges $z ^ { \prime } \to a , z ^ { \prime \prime } \to a , w _ { i } \to q _ { i } .$ , for $i = 1 , \ldots , r - 1$ , together with the edges $p  y .$ , for $p \in P ,$ and set all remaining of-diagonal coeficients of M to zero. The parameters can be chosen with negative diagonal entries and suficiently small of-diagonal entries, so stability is preserved.

Under these values for parameters, the covariance submatrix on the rows $q _ { 1 } , \ldots , q _ { r - 1 }$ and columns $w _ { 1 } , \dots , w _ { r - 1 }$ is diagonal: $\Sigma _ { q _ { i } w _ { j } } = 0$ for $i \neq j$ , and $\Sigma _ { q _ { i } w _ { i } } \neq 0 .$ . Likewise, $\Sigma _ { a z ^ { \prime } } \neq 0 , \Sigma _ { a z ^ { \prime \prime } } \neq 0 .$ , while the remaining entries in the $z ^ { \prime } -$ and $z ^ { \prime \prime } .$ -columns vanish on the rows $P ^ { \prime } \setminus \{ a \}$

Consequently, after a permutation of rows and columns, the determinant of K factors as

$$
| \mathrm { d e t } ( \mathbf { K } ) | = ( \prod _ { i = 1 } ^ { r - 1 } | \Sigma _ { q _ { i } w _ { i } } | ) | \operatorname* { d e t } \binom { \Sigma _ { a z ^ { \prime } } } { \Sigma _ { y z ^ { \prime } } } \quad \Sigma _ { y z ^ { \prime \prime } } ) | .
$$

Under the chosen parametrization, in the equation (3), only the parent a contributes to the $z ^ { \prime } -$ and $z ^ { \prime \prime } .$ -columns, we obtain $\begin{array} { r } { \Sigma _ { y z ^ { \prime } } = - \frac { m _ { y a } \Sigma _ { a z ^ { \prime } } } { m _ { y y } + m _ { z ^ { \prime } z ^ { \prime } } } } \end{array}$ and $\begin{array} { r } { \Sigma _ { y z ^ { \prime \prime } } = - \frac { m _ { y a } \Sigma _ { a z ^ { \prime \prime } } } { m _ { y y } + m _ { z ^ { \prime \prime } z ^ { \prime \prime } } } } \end{array}$

Therefore,

$$
\operatorname* { d e t } { \biggl ( } \sum _ { a z ^ { \prime } } \quad \sum _ { a z ^ { \prime \prime } } { \biggr ) } = m _ { y a } \Sigma _ { a z ^ { \prime } } \Sigma _ { a z ^ { \prime \prime } } { \frac { m _ { z ^ { \prime \prime } z ^ { \prime \prime } } - m _ { z ^ { \prime } z ^ { \prime } } } { ( m _ { y y } + m _ { z ^ { \prime } z ^ { \prime } } ) ( m _ { y y } + m _ { z ^ { \prime \prime } z ^ { \prime \prime } } ) } } .
$$

This expression is nonzero because $m _ { y a } \neq 0 , \Sigma _ { a z ^ { \prime } } \neq 0 , \Sigma _ { a z ^ { \prime \prime } } \neq 0 , m _ { z ^ { \prime } z ^ { \prime } } \neq m _ { z ^ { \prime \prime } z ^ { \prime \prime } }$ . It follows that det $( { \bf K } ) \neq 0$ Hence, generically,

$$
\operatorname { r a n k } ( \Sigma _ { P Z } ) = \operatorname { r a n k } \left( \Sigma _ { P ^ { \prime } Z } \right) = r + 1 = | P | .
$$

Necessity Suppose, generically, rank $\left( \Sigma _ { P Z } \right) = \mathrm { r a n k } \left( \Sigma _ { P ^ { \prime } Z } \right) = r + 1 = | P |$ . Since adding one row can increase rank by at most one, necessarily rank $\begin{array} { r } { \left( \pmb { \Sigma } _ { P ^ { \prime } Z } \right) = r } \end{array}$

Under the assumption that every $Z { \mathrm { - t o } } { \mathrm { - } } P ^ { \prime }$ path is a single edge, the generic support pattern of $\Sigma _ { P ^ { \prime } Z }$ is precisely the bipartite graph B. Therefore, the generic rank theorem for bipartite matrices implies that B contains a matching covering $P ^ { \prime }$ . Moreover, an $( r + 1 ) \times ( r + 1 )$ nonsingular minor must use $r + 1$ distinct instrument columns. Choose such a set $\widetilde { Z } \subseteq Z , | \widetilde { Z } | = r + 1$ . Because $\Sigma _ { P ^ { \prime } , \widetilde { Z } }$ has rank r, its support graph contains a matching $\mathcal { H } _ { \mathcal { M } }$ covering $P ^ { \prime }$ . This matching uses exactly r instruments, leaving one instrument $z ^ { \prime \prime } \in \widetilde { Z }$ unmatched. The $z ^ { \prime \prime } .$ -column cannot vanish on all rows indexed by $P ^ { \prime } ,$ , because then (3) would also gives $\Sigma _ { y z ^ { \prime \prime } } = 0$ , and the corresponding column of the augmented matrix would be zero. Hence $z ^ { \prime \prime }$ is adjacent to at least one parent, say $a \in P ^ { \prime }$ . Let $z ^ { \prime }$ denote the instrument matched to a by $\mathcal { H } _ { \mathcal { M } }$ . Then $z ^ { \prime } \to a , z ^ { \prime \prime } \to a$ . After deleting $a , z ^ { \prime }$ , and $z ^ { \prime \prime } .$ the remaining edges of M constitute a matching covering $P ^ { \prime } \setminus \{ a \}$

Finally, the inequality $m _ { z ^ { \prime } z ^ { \prime } } \ne m _ { z ^ { \prime \prime } z ^ { \prime \prime } }$ does not need to be imposed as an additional condition in a generic rank statement. From $( 3 )$ , the rank augmentation associated with $z ^ { \prime }$ and $z ^ { \prime \prime }$ contains the factor $- \frac { 1 } { m _ { y y } + m _ { z ^ { \prime } z ^ { \prime } } } +$ $\frac { 1 } { m _ { y y } + m _ { z ^ { \prime \prime } z ^ { \prime \prime } } }$ , which vanishes precisely when $m _ { z ^ { \prime } z ^ { \prime } } = m _ { z ^ { \prime \prime } z ^ { \prime \prime } }$ . Since the diagonal drift coeficients are free parameters, this equality defines a proper algebraic subset of the parameter space and is excluded by genericity. Hence unequal self-drifts are not part of the graphical characterization. This completes the proof.

## A.7 Proof of Theorem 5

Proof. Let $| P | = | P ^ { \prime } | + 1 = r + 1$ . It is enough to show one admissible parameter point at which an $( r + 1 ) \times ( r + 1 )$ minor of $\Sigma _ { P Z }$ is nonzero. The determinant of every such minor is a rational function of the model parameters. If it is nonzero at one admissible parameter point, it is nonzero outside a proper algebraic subset of the parameter space.

Consider the columns indexed by $Z _ { 0 } ~ = ~ \{ w _ { 1 } , \ldots , w _ { r - 1 } , z ^ { \prime } , z ^ { \prime \prime } \}$ and order the rows as $q _ { 1 } , \ldots , q _ { r - 1 } , a , y$ . Let $\mathbf { K } : = \boldsymbol { \Sigma } _ { \left\{ q 1 , \ldots , q _ { r - 1 } , a , y \right\} , Z _ { 0 } } .$

We construct an instantiation of parameters. In the drift matrix, retain only the of-diagonal coeficients corresponding to the selected paths $\pi ^ { \prime } , \pi ^ { \prime \prime }$ , and $\rho _ { 1 } , \ldots , \rho _ { r - 1 }$ , together with the parent edges $p \to y , p \in P$ , and set all other of-diagonal coeficients to zero. Choose the path coeficients to be nonzero.

By the vertex-disjointness assumptions and the instrumental-validity condition, the internal vertices of the selected paths cannot belong to $P \cup \{ y \}$ . Consequently, the additional edges $p  y$ cannot create a directed connection between two distinct selected path systems.

Finally, choose the diagonal drift coeficients to be suficiently negative. More precisely, after fixing the ofdiagonal coeficients, choose $m _ { v v } < 0$ with suficiently large magnitude so that every Gershgorin disc of M lies strictly in the open left half-plane. The resulting drift matrix is therefore stable. We may, in addition, choose $m _ { z ^ { \prime } z ^ { \prime } } \neq m _ { z ^ { \prime \prime } z ^ { \prime \prime } }$

Under this instantiation, the covariance submatrix on the rows $q _ { 1 } , \ldots , q _ { r - 1 }$ and columns $w _ { 1 } , \dots , w _ { r - 1 }$ is diagonal: $\Sigma _ { q _ { i } w _ { j } } = 0$ for $i \neq j$ , and $\Sigma _ { q _ { i } w _ { i } } \neq 0$ whereas the directed path $\rho _ { i } : w _ { i }  q _ { i }$ . Consequently, after the chosen ordering, $\sum _ { \{ q _ { 1 } , \dots , q _ { r - 1 } \} , \{ w _ { 1 } , \dots , w _ { r - 1 } \} }$ is diagonal with nonzero diagonal entries. Hence, the determinant is nonzero.

Similarly, because every $\rho _ { i }$ is vertex-disjoint from $\pi ^ { \prime } \cup \pi ^ { \prime \prime }$ , there is no trek between $q _ { i }$ and either $z ^ { \prime } \ \mathrm { o r } \ z ^ { \prime \prime }$ under this instantiation. Therefore, $\Sigma _ { q _ { i } z ^ { \prime } } = \Sigma _ { q _ { i } z ^ { \prime \prime } } = 0$ , for $i = 1 , \ldots , r - 1$ . Thus K has the block lower-triangular form

$$
\mathbf { K } = \left( \begin{array} { c c } { \sum _ { \{ q _ { 1 } , \dots , q _ { r - 1 } \} , \{ w _ { 1 } , \dots , w _ { r - 1 } \} } } & { 0 } \\ { \mathbf { B } } & { \sum _ { \{ a , y \} , \{ z ^ { \prime } , z ^ { \prime \prime } \} } } \end{array} \right) ,
$$

where the block B need not vanish.

In particular, $\operatorname* { d e t } ( \mathbf { K } ) = \operatorname* { d e t } ( \Sigma _ { \{ q _ { 1 } , \dots , q _ { r - 1 } \} , \{ w _ { 1 } , \dots , w _ { r - 1 } \} } ) \operatorname* { d e t } ( \Sigma _ { \{ a , y \} , \{ z ^ { \prime } , z ^ { \prime \prime } \} } ) .$

It remains to show that $\operatorname* { d e t } ( \Sigma _ { \{ a , y \} , \{ z ^ { \prime } , z ^ { \prime \prime } \} } ) \neq 0$ . The directed paths $\pi ^ { \prime }$ and $\pi ^ { \prime \prime }$ imply, generically, $\Sigma _ { a z ^ { \prime } } \neq 0$ and $\Sigma _ { a z ^ { \prime \prime } } \neq 0$

Because $z ^ { \prime }$ is a root and has no direct edge to y, the $( y , z ^ { \prime } ) – \mathrm { e n t r y }$ of the Lyapunov equation is $\begin{array} { r } { \sum _ { p \in P ^ { \prime } } m _ { y p } \Sigma _ { p z ^ { \prime } } + } \end{array}$ $( m _ { y y } + m _ { z ^ { \prime } z ^ { \prime } } ) \Sigma _ { y z ^ { \prime } } = 0 .$ . Under the selected instantiation, $\Sigma _ { p z ^ { \prime } } = 0$ for every $p \in P ^ { \prime } \backslash \{ a \}$ . Hence $( m _ { y y } +$ $m _ { z ^ { \prime } z ^ { \prime } } ) \Sigma _ { y z ^ { \prime } } + m _ { y a } \Sigma _ { a z ^ { \prime } } = 0$ , and therefore $\begin{array} { r } { \Sigma _ { y z ^ { \prime } } = - \frac { m _ { y a } \Sigma _ { a z ^ { \prime } } } { m _ { y y } + m _ { z ^ { \prime } z ^ { \prime } } } } \end{array}$ and $\begin{array} { r } { \Sigma _ { y z ^ { \prime \prime } } = - \frac { m _ { y a } \Sigma _ { a z ^ { \prime \prime } } } { m _ { y y } + m _ { z ^ { \prime \prime } z ^ { \prime \prime } } } } \end{array}$

The denominators are nonzero because all diagonal drift coeficients were chosen strictly negative in this specialization. Therefore,

$$
\operatorname* { d e t } ( \Sigma _ { \{ a , y \} , \{ z ^ { \prime } , z ^ { \prime \prime } \} } ) = \operatorname* { d e t } \binom { \Sigma _ { a z ^ { \prime } } } { \Sigma _ { y z ^ { \prime } } } \Sigma _ { a z ^ { \prime \prime } } \Big ) = m _ { y a } \Sigma _ { a z ^ { \prime } } \Sigma _ { a z ^ { \prime \prime } } \frac { m _ { z ^ { \prime \prime } z ^ { \prime \prime } } - m _ { z ^ { \prime } z ^ { \prime } } } { ( m _ { y y } + m _ { z ^ { \prime } z ^ { \prime } } ) ( m _ { y y } + m _ { z ^ { \prime \prime } z ^ { \prime \prime } } ) } .
$$

This quantity is nonzero because $m _ { y a } \neq 0 , \Sigma _ { a z ^ { \prime } } \neq 0 , \Sigma _ { a z ^ { \prime \prime } } \neq 0$ , and $m _ { z ^ { \prime } z ^ { \prime } } \neq m _ { z ^ { \prime \prime } z ^ { \prime \prime } }$

It follows that det $( \mathbf { K } ) \neq 0$ . Hence $\Sigma _ { P Z }$ contains a nonzero $( r + 1 ) \times ( r + 1 )$ minor, and therefore, generically, rank $( \Sigma _ { P Z } ) = r + 1 = | P |$

## B FURTHER DETAILS ON THE FOUR-NODE EXTENSIONS

This section provides additional details on the four-node extensions associated with Proposition 1. In particular, we illustrate the identifiable configurations covered by the proposition and give examples of extensions that fail to restore generic identifiability.

The examples below emphasize that adding a fourth vertex is not suficient by itself to repair the rank deficiency of a deficient three-node motif. What matters is whether the added structure changes the trek and covariance structure in a way that introduces an additional independent constraint in $\mathbf { H } ( \pmb { \Sigma } ) _ { E ^ { c } , \ast }$

## B.1 Identifiable Four-Node Extensions

Figure 4 shows representative four-node configurations that are generically identifiable. The red arrows indicate admissible extensions involving the added vertex v. A solid red arrow is required, whereas a dashed red arrow denotes an optional edge. Blue connections indicate adjacencies whose orientation may be arbitrary; whenever several blue adjacencies are shown, at least one of the indicated connections must be present.

![](images/439f6881f8cd7dee98fe425a6e9d3fb139082799183f78b72d496127aae8ba1b.jpg)  
(a) Common child.

![](images/005b27778dd948c528155bc9708131c6a4173009cd90e93d92797ff7354a44a8.jpg)  
(b) Common parent.

![](images/61d762ad470001ff8a69e781107667740059128f27d78e76bb4ed7314cc7123f.jpg)

(c) 2-cycles intersection.

![](images/6d25af9299224668fcc9ba879057d0b32365f6899c7d93794292d694a48845f1.jpg)  
(d) 2-cycles intersection.

![](images/9d1000c23a239cd66b84fc561d9a5ded6bf6c32fa56f675406cfb07316f4b17f.jpg)  
(e) Coupling (f) Asymmetric extwo 2-cycles. tension.

Figure 4: Four-node motifs that are generically identifiable. Red arrows indicate admissible extensions involving the added vertex v. A solid red arrow is required, while a dashed red arrow denotes an optional choice. Blue lines indicate adjacency whose orientation is arbitrary. For a set of blue adjacencies, at least one of the indicated blue connections must be present, in either direction.

The extensions in Figures 4a–4d illustrate the configurations summarized in Proposition 1. These examples show that identifiability is not determined by adjacency alone. Rather, the added vertex must alter the covariance structure so that the missing rank contribution of the deficient three-node motif is recovered.

Figure 4e illustrates an additional rank-repair mechanism. Here, the graph consists of two vertex-disjoint directed 2-cycles coupled by directed edges between their vertices. Although the two directed 2-cycles are deficient when considered separately, their coupling as shown in Figure 4e renders the combined four-node graph generically identifiable.

Moreover, one of the connecting edges may be removed without destroying identifiability. Removing such an edge adds the corresponding non-edge row to $\mathbf { H } ( \pmb { \Sigma } ) _ { E ^ { c } , \ast }$ , while the remaining connection preserves the treks responsible for the covariance entries needed to obtain full column rank.

## B.2 Embedding the Motifs in Larger Graphs

The identifiable configurations above need not occur as isolated four-node graphs. The same block-rank argument can be used when several such motifs are embedded in a larger graph.

Suppose that the graph contains a collection of vertex-disjoint four-node motifs of the types described above, no vertex belongs to more than one such motif, and the subgraph induced by the remaining vertices is generically identifiable. Assume additionally that no pair of vertices belonging to two distinct blocks—either two diferent four-node motifs, or a four-node motif and the remaining subgraph—forms a directed 2-cycle.

Then the same block-rank construction used in the proof of Theorem 1 extends to the full graph. The motif-local blocks have full column rank by the results above, the remaining block has full column rank by assumption, and the columns involving vertices from distinct blocks can be handled by selecting suitable non-edge rows as in the mixed-block argument of Theorem 1. Hence, the complete graph is generically identifiable.

(a) Duplicate common (b) Duplicate common (c) Common-child failed child. parent. extension.  
![](images/c11b06c0cfbfc73d52dce928f2febc21faab471e42ec6835c764aed426792ec8.jpg)  
Figure 5: Four-node extensions that remain non-identifiable.

## B.3 Failed Extensions

The previous examples also show that introducing an additional vertex does not necessarily repair the rank deficiency. Figure 5 shows representative four-node extensions that remain non-identifiable.

The condition that the added structure creates a relevant trek connection with the vertices in the directed 2- cycle is essential. Consider, for example, the common-child motif and the extension shown in Figure 5c. In this configuration, there is no trek from the added vertex v to either vertex of the directed 2-cycle (i, j). By the trek rule, $\Sigma _ { v i } = \Sigma _ { v j } = 0$ . Consequently, the additional rows involving v do not introduce new covariance entries capable of repairing the missing rank contribution. The system becomes larger, but the original rank deficiency persists.

Extensions that merely duplicate the common-child or common-parent structures also fail to restore identifiability. As shown in Figures 5a and 5b, these symmetric attachments reproduce constraints already present in $\mathbf { H } ( \pmb { \Sigma } ) _ { E ^ { c } }$ and therefore do not supply an additional independent rank contribution.

In contrast, the asymmetric extension shown in Figure 4f breaks this redundancy. The asymmetry changes the trek structure and introduces an additional independent covariance constraint, thereby restoring generic identifiability.

Overall, these examples reinforce the interpretation of Proposition 1: the decisive feature of an extension is   
not the mere presence of an additional node or edge, but whether the resulting trek structure generates new   
covariance information that increases the rank of $\mathbf { H } ( \Sigma ) _ { E ^ { c } }$ ,·

## C TREK GRAPHS AND PROPOSITION 2

This section provides the graphical interpretation of Proposition 2 in the main paper. In particular, we explain how conditional independence in a stationary difusion can be read from the trek graph and why the condition $X _ { h } \perp X _ { x } \mid X _ { O \backslash \{ x \} }$ eliminates the marginalization term associated with a latent parent h of y.

Treks and the trek graph.

Definition 3 (Trek graph Boege et al. (2025)). Let $G = ( V , E )$ be the directed drift graph. Its trek graph $\widehat { G } = ( V , \widehat { E } )$ is the bidirected graph satisfying

$$
x  y \in { \widehat { E } } \quad \iff \quad \operatorname { A n } _ { G } ( x ) \cap \operatorname { A n } _ { G } ( y ) \neq \varnothing , \qquad x \neq y .
$$

Equivalently, $x  y \in { \widehat { E } }$ if and only if there exists a trek between x and $y$ in $G .$

The equivalence in Definition 3 follows directly from the definition of a trek. If a trek between x and y has top c, then $c \in \mathrm { A n } _ { G } ( x ) \cap \mathrm { A n } _ { G } ( y )$ . Conversely, any $c \in \mathrm { A n } _ { G } ( x ) \cap \mathrm { A n } _ { G } ( y )$ gives two directed paths $c \sim x$ and $c \sim y$ which together form a trek between x and y.

Definition 4 (Separation in the trek graph). Let $I , J , S \subseteq V$ be pairwise disjoint. We write $I \perp _ { \widehat { G } } J \mid S$ if every path in $\widehat { G }$ from a vertex in I to a vertex in J intersects S.

Since the linear Ornstein–Uhlenbeck model considered in this paper is a special case of the stationary difusions covered by Theorem 1.2 Boege et al. (2025), the separation condition in its first statement implies the corresponding conditional independence in our model.

Corollary 1 (Graphical condition for Proposition 2). Let $V = O \dot { \cup } L , l e t h \in L$ , and $l e t ~ x \in O . ~ I f \left\{ h \right\} \perp _ { \widehat { G } } \left\{ x \right\}$ $L \setminus \{ h \}$ , then $X _ { h } \perp X _ { x } \mid X _ { O \setminus \{ x \} }$

Proof. Apply Theorem 1.2 Boege et al. (2025) with $I = \{ h \} , J = \{ x \}$ , and $K = O \setminus \{ x \}$ . Because $V = O \dot { \cup } L$

$$
{ \begin{array} { r l } & { V \setminus ( I \cup J \cup K ) = V \setminus ( \{ h \} \cup \{ x \} \cup ( O \setminus \{ x \} ) ) } \\ & { \qquad = V \setminus ( \{ h \} \cup O ) } \\ & { \qquad = L \setminus \{ h \} . } \end{array} }
$$

The claim therefore follows from Theorem 1.2 Boege et al. (2025).

Lemma 5. Let $S = O \setminus \{ x \}$ . Then

$$
( \Gamma _ { L | O } ) _ { h x } = \frac { \Sigma _ { h x } - \Sigma _ { h S } \Sigma _ { S S } ^ { - 1 } \Sigma _ { S x } } { \Sigma _ { x x } - \Sigma _ { x S } \Sigma _ { S S } ^ { - 1 } \Sigma _ { S x } } .
$$

Consequently, $X _ { h } \perp X _ { x } \mid X _ { S } \implies ( \Gamma _ { L | O } ) _ { h x } = 0 .$

Proof. Since $\Gamma _ { L | O } = \Sigma _ { L O } \Sigma _ { O O } ^ { - 1 }$ , the expression follows from the block inverse formula applied to the partition $O = \{ x \} { \dot { \cup } } S$ . The numerator is the conditional covariance $\Sigma _ { h x | S } = \Sigma _ { h x } - \Sigma _ { h S } \Sigma _ { S S } ^ { - 1 } \Sigma _ { S x }$ , and the denominator is the conditional variance $\Sigma _ { x x | S } > 0$ . Since the stationary distribution is Gaussian, ${ \tilde { X _ { h } } } \perp X _ { x } \mid X _ { S } \Longleftrightarrow \Sigma _ { h x \mid S } = 0$ □

So, suppose $h  y$ and $x  y .$ where $h \in L$ and $x , y \in O$ . Then $\widehat { G }$ contains $h - y - x$ , because each directed edge is a trek. The required separator is $L \backslash \{ h \}$ , but the internal vertex y is observed. Therefore, $\{ h \} \mathbin { \not \sim } { \mathcal { L } } _ { \widehat { G } } \{ x \} \mathbin { \mid } L \backslash \{ h \}$ , so Theorem 1.2 Boege et al. (2025) does not guarantee $X _ { h } \perp X _ { x } \mid X _ { O \backslash \{ x \} }$ . This motivates the instrumental-variable approach.

## D COMPUTATIONAL DETAILS

All experiments were implemented in Python 3.13 using NumPy, SciPy, NetworkX, and Matplotlib. Computations were performed on a MacBook Air equipped with an Apple M4 processor and 16 GB of unified memory.

The complete source code used to generate all synthetic experiments, together with the parameter initializations and implementation details, is included in the supplementary material.

No trajectories were simulated in any experiment. All evaluations were performed at the population level. For every generated model, the stationary covariance matrix was obtained directly by solving the continuous Lyapunov equation

$$
\mathbf { M } \mathbf { \boldsymbol { \Sigma } } + \mathbf { \boldsymbol { \Sigma } } \mathbf { M } ^ { \top } + \mathbf { \boldsymbol { C } } = 0 .
$$

## D.1 Generic Identifiability

The numerical experiments investigate generic identifiability of individual entries of the drift matrix for Theorem 1. The same procedure was used for the graphical configurations displayed in Figure 2.

## D.1.1 Graph Generation

We considered directed graph structures containing one or more directed 2-cycles. Depending on the experiment, the graphs were either fixed to represent the local configurations shown in Figure 2 or generated according to the graphical assumptions under investigation. For each directed 2-cycle, additional adjacent vertices and edges were introduced according to the corresponding graph configuration. All remaining entries of the drift graph were set to zero unless explicitly included in the construction.

![](images/88fdff8cee740b07e26e101034979257bc952345fbf04fa185c8ad7c788406a5.jpg)  
(a) Dense configuration.

![](images/ed267fdc51de36ff3da4cdd947cb39e156bcd41da4847c15fab509ad960ae1b9.jpg)  
(b) Bottleneck configuration.  
Figure 6: IV setup

## D.1.2 Model Specification

For each experiment, we fixed a directed graph and constructed the corresponding symbolic drift matrix M. An of-diagonal entry $m _ { v u }$ was treated as a free parameter precisely when the graph contained the edge $u  v .$ The diagonal entries of M were also treated as free parameters. The difusion matrix C was restricted to be diagonal with strictly positive entries.

Where required by the graphical configuration, selected covariance entries were constrained to zero before solving the Lyapunov equations.

## D.1.3 Parameter Sampling

For each trial, positive parameters were sampled independently from $( 1 0 ^ { - 1 0 } , 1 0 ]$ , negative parameters from $[ - 1 0 , - 1 0 ^ { - 1 0 } )$ , and unconstrained of-diagonal drift parameters from [−10, 10]. A sampled drift matrix was retained only when all of its eigenvalues had strictly negative real parts. Up to 100 samples were drawn to obtain a Hurwitz drift matrix for each trial.

## D.1.4 Covariance Computation

For every accepted drift matrix, the stationary covariance entries were obtained by solving the linear system induced the continuous Lyapunov equation. Samples for which the system was inconsistent or the resulting covariance matrix was not positive definite were discarded.

## D.1.5 Numerical Identifiability Test

After fixing the generated stationary covariance matrix and the diagonal difusion parameters, identifiability was tested separately for each free entry of the drift matrix. We minimized and maximized the selected parameter subject to the Lyapunov equations, while allowing the remaining drift parameters to vary within [−10, 10]. Both optimization problems were solved using linear programming.

Let $\underline { m } _ { v u }$ and $\overline { { m } } _ { v u }$ denote the resulting minimum and maximum feasible values. The parameter $m _ { v u }$ was classified as numerically identifiable whenever

$$
| \overline { { { m } } } _ { v u } - \underline { { { m } } } _ { v u } | < 1 0 ^ { - 8 } .
$$

For each graphical configuration and each drift parameter, this procedure was repeated over 1000 random trials.   
We report the proportion of valid trials for which the parameter was classified as identifiable.

## D.2 Instrumental Variables

## D.2.1 Graph Configurations

We considered fixed graph configurations containing a target vertex $y ,$ observed parents $P ,$ latent parents U, and root instruments $Z .$ We considered configurations violating the bottleneck (reported in the main paper). The graph structure was fixed within each configuration, while the numerical model parameters were independently resampled.

Figure 6 illustrates two representative instrumental-variable configurations. The left panel shows a suficient configuration for the full-rank condition of Theorem 5, whereas the right panel illustrates a bottleneck configuration with bottleneck ratio $r = 1 / 3$

## D.2.2 Parameter Initialization

For every repetition, the diagonal entries of the drift matrix were sampled independently from $[ - 1 0 , - 1 0 ^ { - 8 } ]$ and the coeficient associated with each directed edge was sampled independently from [−10, 10]. All remaining of-diagonal entries were set to zero. Samples were repeatedly generated until the resulting drift matrix was Hurwitz. The difusion matrix was diagonal, with entries sampled independently from $[ 1 0 ^ { - 8 } , 1 0 ]$

The stationary covariance matrix was computed directly by solving the continuous Lyapunov equation. No trajectories or finite observational samples were generated.

## D.2.3 Recovery Procedure

Only the observed covariance matrix $\pmb { \Sigma } _ { O O }$ was supplied to the recovery procedure. Since the instruments are roots, their diagonal drift coeficients were obtained from $\begin{array} { r } { m _ { z z } = - \frac { c _ { z z } } { 2 \Sigma _ { z z } } } \end{array}$ , and the corresponding vector b was constructed from observed covariance entries. The target coeficients and the self-drift of $y$ were then estimated jointly by solving

$$
\begin{array} { r l } { [ \Sigma _ { P ^ { \prime } Z } ^ { \top }  } & { { } \Sigma _ { y Z } ^ { \top } ] [ \begin{array} { c } { \widehat { m } _ { y P ^ { \prime } } } \\ { \widehat { m } _ { y y } } \end{array} ] = - b } \end{array}
$$

using least squares.

## D.2.4 Evaluation Metric

Recovery accuracy was evaluated using the relative $\ell _ { 2 }$ error between the estimated and true vectors of target drift coeficients, $\frac { | | \hat { m } _ { y P ^ { \prime } } - m _ { y P ^ { \prime } } | | _ { 2 } } { | | m _ { y P ^ { \prime } } | | _ { 2 } }$ , where $m _ { y } P ^ { \prime }$ denotes the true vector of target drift coeficients and $\hat { m } _ { y P ^ { \prime } }$ its estimate. The reported results correspond to the mean relative error across 100 independent repetitions.

## D.3 Restrictiveness of Theorem 1

To assess how restrictive the suficient graphical condition of Theorem 1 is relative to the kernel-rank criterion of Lemma 1, we generated random directed graphs with $| V | \in \{ 6 , 8 , 1 0 , 1 2 \}$ . Each possible of-diagonal directed edge was included independently with probability $p \in \{ 0 . 0 5 , 0 . 1 0 , 0 . 1 5 , 0 . 2 0 , 0 . 2 5 , 0 . 3 0 \}$ . For each combination of $| V |$ and $p ,$ we retained 500 graphs containing at least one directed 2-cycle; graphs without directed 2-cycles were excluded. For each retained graph, we checked whether the graphical conditions of Theorem 1 were satisfied.

We then numerically evaluated the generic kernel-rank condition of Lemma 1. For each directed edge $u  v .$ the corresponding drift coeficient $m _ { v u }$ was sampled independently from a standard normal distribution. The diagonal of the resulting drift matrix was shifted uniformly so that its spectral abscissa was at most −1, thereby ensuring that the drift matrix was Hurwitz. The difusion matrix was diagonal, with its diagonal entries sampled independently from [0.5, 2]. The stationary covariance matrix Σ was then obtained by solving the continuous Lyapunov equation.

Given $\Sigma ,$ we constructed $\mathbf { H } ( \pmb { \Sigma } ) _ { E ^ { c } } ,$ <sub>·</sub>, whose rows correspond to directed non-edges $i \to j \not \in E$ and whose columns

![](images/86251ee087b8ce413e12ad9fa3fb4b0f13fc1ddafa8a7db89b37405715d1386b.jpg)  
Figure 7: Coverage of the suficient graphical condition in Theorem 1. Each cell reports the percentage of randomly generated non-simple graphs with numerical full kernel rank that are certified by Theorem 1, for diferent numbers of vertices |V| and edge probabilities. .

correspond to unordered vertex pairs $( k , l ) , k < l .$ Its entries were computed as

$$
\begin{array} { r } { H ( \Sigma ) _ { i \to j , ( k , l ) } = \left\{ \begin{array} { l l } { - \Sigma _ { l j } , } & { i = k , } \\ { \Sigma _ { k j } , } & { i = l , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \end{array}
$$

Graphs with fewer than $| V | ( | V | - 1 ) / 2$ directed non-edges were immediately classified as not having full kernel rank.

For the remaining graphs, the rank of $\mathbf { H } ( \pmb { \Sigma } ) _ { E ^ { c } } ,$ <sub>·</sub> was computed from its singular values. A singular value was treated as numerically nonzero when it exceeded

$$
\tau = 1 0 0 \mathrm { m a x } \{ \mathrm { r o w s } ( \mathbf { H } ) , \mathrm { c o l s } ( \mathbf { H } ) \} \epsilon _ { \mathrm { m a c h } } s _ { \mathrm { m a x } } ,
$$

where $\epsilon _ { \mathrm { m a c h } }$ is machine precision and $s _ { \mathrm { m a x } }$ is the largest singular value. Full kernel rank was declared when the resulting numerical rank equaled $| V | ( | V | - 1 ) / 2$ . To reduce sensitivity to exceptional parameter choices, up to three independent parameter draws were tested for each graph, and the graph was classified as numerically full rank if the required rank was attained for at least one draw.

Finally, for each $( | V | , p )$ combination, we report the coverage of Theorem 1 among numerically full-rank graphs, defined as the fraction of graphs satisfying the graphical condition of Theorem 1 among those attaining numerical full kernel rank. The random number generator was initialized with seed 42.

Figure 7 shows that Theorem 1 captures most numerically full-rank graphs in the sparse regime. At $p = 0 . 0 5$ coverage ranges from 94.4% to 97.8% across the considered graph sizes. As the graphs become denser, the suficient condition becomes increasingly conservative. For example, for $| V | = 1 2$ , coverage decreases from 95.9% at $p = 0 . 0 5$ to $6 . 4 \%$ at $p = 0 . 3 0$ . This behavior is consistent with the structural requirements of Theorem 1: denser graphs contain more overlapping directed 2-cycles, which fall outside its vertex-disjoint construction even when the kernel matrix has full column rank.