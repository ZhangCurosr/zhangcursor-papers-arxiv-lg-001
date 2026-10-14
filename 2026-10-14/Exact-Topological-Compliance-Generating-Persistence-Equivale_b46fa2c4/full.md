# Exact Topological Compliance: Generating Persistence-Equivalent Graphs

Mattie Ji<sup>∗</sup> University of Pennsylvania mji13@sas.upenn.edu

Indradyumna Roy<sup>∗</sup> Aalto University indradyumna.roy@aalto.fi

Vikas Garg YaiYai Ltd & Aalto University vgarg@csail.mit.edu

## Abstract

Persistent homology summarizes the topological evolution of a graph as persistence diagrams (PDs). Generating graphs whose topology matches a target descriptor is increasingly important, yet existing topology-aware generative models rely on soft regularization, producing graphs whose PDs may only be approximately similar. We initiate a principled investigation of the generation problem: produce graphs that realize a target PD exactly. We propose two approaches guided by the implicit local and global constraints a target PD imposes. The local approach builds degree-based PD-equivalent graphs iteratively, and we prove that every output realizes the target PD and that every degree-based PD-equivalent graph is reachable. However, as shown thereafter, local generation necessarily trades off between sample diversity and reaching infeasible states. We therefore recast generation as a global constraint satisfaction problem with a complete encoding solvable by modern constraint programming solvers. This is applicable to any vertex-based permutation equivariant filtration scheme and is minimal under the degree filtration. Experiments on four standard graph generation benchmarks confirm that both methods realize the target PD exactly. Together, our methods provide a principled treatment of strict topology-compliant graph generation. Code is available at https://github.com/Aalto-QuML/ExactCompliance.

## 1 Introduction

Topological structure is a primary driver of function in many real-world graphs. The presence of components, cycles, and higher-order voids govern stability and catalytic activity in molecules [59, 49, 55], binding affinity and virality [56, 57], information flow in brain networks [47], certain patterns in medical imaging [33], and structural risk in transportation, trade, and infrastructure networks [58, 28, 53]. Persistent homology [4, 14] is the standard tool for extracting this information. Given a graph G and a filtration scheme, persistent homology tracks the births and deaths of homological features along the induced subgraph sequence. The result is a multiset of pairs (b, d) called the persistence diagram (PD). PDs are stable under perturbation [8], computable with efficient algorithms [38], and have become the workhorse of topological data analysis (TDA) in machine learning, underpinning almost every topology-aware approach to graph and higher-order neural networks [19, 22, 2, 21, 24, 23, 62, 5, 44, 51, 39] and providing topological signals that standard message passing alone cannot recover (e.g., cycle detection).

In parallel, graph generation has recently emerged as a central tool across these same domains, with autoregressive [63], one-shot [46], and diffusion-based [36, 27, 52] models. Recent work conditions generation on domain-specific structural properties: spectral profile [32], molecular validity [10, 26], and structural constraints like planarity and acyclicity [30]. Conditioning on topology is a natural next step, but largely unexplored. A recent attempt [40], augments discrete diffusion with a 1-Wasserstein

Table 1: Overview of the core contributions in our paper.
<table><tr><td>Local Generation of PD-Equivalent Filtrations (Section 3) Local Generation for General/Vertex/Degree Filtrations Correctness and Diversity of Local Generations Perfect Generation on Acyclic Graphs (for General/Vertex) Impossibility Results for Local Generation</td><td>Algorithm 1, 2, 3 Theorem 3.1, 3.7, 3.9 Corollary 3.4, 3.8 Theorem 3.10</td></tr><tr><td>Global Generation of PD-Equivalent Graphs (Section 4) Global Generation for Any Canonical Vertex Filtration Scheme Φ Section 4.1 Soundness, Completeness, and Degree Minimality of Encoding</td><td>Theorem 4.2</td></tr><tr><td>Experiments (Section 5) Local vs. Global on Four Graph Generation Benchmarks Comparison Against PD-Conditioned Neural Diffusion Baselines</td><td>Table 2 Table 3, Figure 3</td></tr></table>

PD-matching loss. This soft regularizer nudges the learned distribution toward topological plausibility, but offers no guarantees that individually generated graphs attain the desired topological structure (e.g., the target number of cycles and components).

PD-conditioned graph generation from first principles. A PD is a finite summary of an underlying combinatorial structure and therefore collapses many graphs to the same image. The most widely studied filtration scheme in the graph learning literature is the vertex-degree function [2, 21, 40, 62, 19, 20]: it is permutation-equivariant and aligns naturally with the local message-passing inductive bias of GNNs. We adopt it as the primary filtration scheme throughout this paper and discuss extensions to other vertex-based, permutation-equivariant schemes (centrality [29], square clustering coefficients [25], triangle count [48], etc.) where they apply.

Even under this canonical choice, a PD is far from a complete descriptor. Figure 1 shows two non-isomorphic graphs with identical degree-based PDs; in general, graphs with differing structures and connectivity patterns, can share the same PD under a given filtration scheme. Each PD therefore identifies a class of compatible graphs, and conditioning on a PD amounts to sampling from this class. This raises the natural question: given an input PD andfiltration scheme Φ, can we generate strictly PD-compatible graphs under Φ?

![](images/8804160df36fdd82d798df945ce98bf735beab5fca2f1fd8c44ef7f46b49e8b3.jpg)  
Figure 1: Two non-isomorphic graphs with identical degree-based persistence. Vertex color encodes degree; edges inherit the color of their max-degree endpoint.

This motivates a precise criterion for any PD-conditioned generator. Given a filtration scheme Φ and a graph G, we let ${ \mathcal { P } } ( G _ { \bullet } ^ { \Phi } )$ ) denote the PDs associated to the induced subgraph sequence of G under Φ. We call a generator strictly PD-compliant if it satisfies

1. Soundness. Every sample G from the generator satisfies $\mathcal { P } ( G _ { \bullet } ^ { \Phi } ) = D$ exactly.

2. Completeness. Every PD-compatible graph can be created by the generator.

Soft-conditioned models violate soundness by construction. Aiming for strict PD-compliance is the natural starting point: it makes the conditioning problem precise, and exposes what an input PD does and does not pin down. Once strict compliance is attained, it can be relaxed in applications.

Why exact compliance? Exact generation produces families of graphs with identical persistence diagrams but different combinatorial properties. Such families provide controlled datasets and benchmarks for training and evaluation in topological learning, and one can test whether a model truly exploits the prescribed topology or leans on unrelated statistics. They also give clean training and evaluation data for graph generation and reasoning from topological descriptors, with no label noise from approximate PD matching.

Exact generation can also help find counterexamples to conjectures about what properties are determined by persistence, and approximate generation cannot serve this purpose at all. Exactness is also important when applications critically depend on the exact number of topological features, such as ring counts in molecular generation [43] and coverage holes in sensor networks [11].

Contributions. We initiate a principled investigation of strict PD-compliant graph generation; Table 1 summarizes the formal statements. Our contributions are threefold.

(1) Local iterative sampler (Section 3). Local decisions are a prerequisite for any globally correct generation and rely only on the neighborhood-level information standard GNN pipelines already provide, making them the natural starting point. We give a sampler that is sound and complete on general, vertex-based, and degree filtrations, but the sampler can run into infeasible states. We subsequently prove an impossibility result: any local approach must trade off between locally PD-compliant choices and reaching globally infeasible states.

(2) Global generation via constraint programming (Section 4). The impossibility motivates us to characterize the global constraints. We recast generation as a constraint satisfaction problem applicable to any vertex-based permutation-equivariant filtration scheme, and give concrete encodings for degree, triangle and clique counts, clustering, closeness, Katz, and PageRank. We prove our method is strictly PD-compliant, and is a minimal encoding for degree filtrations.

(3) Empirical validation (Section 5). On COMM-20, PLANAR, SBM, and ENZYMES, both methods realise the target degree-PD exactly, while neural diffusion baselines with PD-conditioning regularisers leave a structural approximation gap that does not close with more training.

Related Works. Persistence-based reconstructions have been studied for embedded graphs, geometric shapes, and metric-measure spaces [7, 6, 1, 3, 45, 35, 17, 37]. These approaches exploit extrinsic information such as coordinates, directions, distances, or sampling measures to recover or analyze a geometric realization. In contrast, we consider abstract graphs without prescribed embeddings and characterize the set of graphs realizing specified persistence diagrams under an intrinsic, permutationequivariant filtration. Our framework therefore complements geometric reconstruction methods by studying graphs sharing a common intrinsic PH signature. We only consider graph filtrations, as this is quite common for the usage of persistent homology in the GNN literature [19, 21, 22, 64, 48].

To compare the two, the persistent homology transform (PHT) [50] is injective [16, 9] on embedded graphs, but it does not resolve the graph isomorphism problem, as two different embeddings of the same abstract graph can have different PHTs. Our setting contains no coordinates, and we want to reconstruct abstract graphs with equivalent PDs, with the goal of benchmarking for graph isomorphisms. Another deviation from typical TDA considerations is that the GNN literature typically requires permutation equivariance for their filtrations. This imposes a strong rigidity on filtrations that typically do not show up for embedded graphs. For example, any permutation equivariant function on a cycle graph has to be constant, but certainly we can look at the height filtration on embedded cycles.

## 2 Background and Preliminaries

We assume a graph G with vertex set $V ( G )$ and edge set $E ( G )$ is finite, simple, and undirected.

Definition 2.1. Let G be a graph, a (general) filtration of G is a sequence of subgraphs of G as

$$
G _ { \bullet } : \varnothing = G _ { - 1 } \subseteq G _ { 0 } \subseteq \ldots \subseteq G _ { n } = G .
$$

Note here we do allow the $G _ { i } = G _ { i + 1 }$ for some i’s. We say a filtration of $G$ is vertex-based if each $G _ { i }$ is an induced subgraph of G on its vertex set $V ( G _ { i } ) \subseteq { \dot { V } } ( G )$ ). We say a vertex based filtration of G is degree-based if $\bar { V ( G _ { i } ) } = \{ v \in V ( G ) \mid \deg ( \bar { v } ) \leq i \}$ and n is the maximum degree in G. We write the degree-based filtration as $G _ { \bullet } ^ { \mathrm { d e g } }$

Given any filtration $G _ { \bullet }$ of $G .$ , there are lists of persistence pairs tracks the evolution of the births and deaths of its topological features on its (0-dimensional) vertex-features and (1-dimensional) cycle-features. The lists are constructed as follows.

1. For each $b \geq 0$ , we can view $G _ { b - 1 } \subseteq G _ { b }$ as a sequence of adding vertices first and then adding edges to $G _ { b } . \ \mathrm { \bf A n y }$ vertex v added is said to be born at time b and is thought of as a vertex representing the connected component that is {v}.

2. Whenever an edge is added between two components with vertex representatives v and w respectively in the stage $G _ { d - 1 } \subseteq G _ { d }$ . We kill off whichever vertex in v and w that is born later, assign its death time to $d ,$ and assign the other vertex as the representative of the new merged component. If v and w were born at the same time, we simply choose one of them to kill off. A vertex representing a component alive at $G _ { n } = G$ is assigned to die at time $+ \infty$

3. Whenever an edge is added from the same component to itself in the stage $G _ { b - 1 } \subseteq G _ { b }$ , this is an independent cycle, and we assign a tuple of the form $( b , \infty )$ to it.

Definition 2.2. Let $G _ { \bullet }$ be a filtration of G, its persistence diagram ${ \mathcal { P } } ( G _ { \bullet } )$ is a pair of two unlabeled multisets $\mathcal { P } _ { 0 } ( G _ { \bullet } )$ and $\mathcal { P } _ { 1 } ( G _ { \bullet } )$ . Here, $\mathcal { P } _ { 0 } ( \bar { G _ { \bullet } } )$ , the 0-dimensional persistence diagram, is the list of birth and death time $( b , d )$ for each vertex $v \in V ( G ) . \ \mathcal { P } _ { 1 } ( G _ { \bullet } )$ , the 1-dimensional persistence diagram, is the list of cycle birth times $( b , \infty )$ for each independent cycle in $G .$

Note the lists $\mathcal { P } _ { 0 } ( G _ { \bullet } ) , \mathcal { P } _ { 1 } ( G _ { \bullet } )$ are independent of the choices made in the construction, and the final output does not identify persistence pairs with simplices of the graph. Definition 2.2 is a special case of taking persistence diagrams on simplicial complexes (see [13]). Here we restricted to the case of filtrations on graphs where the only non-empty PDs are in degree 0 and 1.

Remark 2.3. Some sources [2, 22] alternatively define a filtration of $G$ to come from a function $f : V ( G ) \cup E ( G ) $ R such that for any edge $e = ( v , w ) , f ( e ) \geq \operatorname* { m a x } ( f ( v ) , f ( w ) )$ . The persistence pairs of any $\dot { f }$ is equivalent to that of Definition 2.2 by renormalizing f to land in $\mathbb { Z } _ { \geq 0 }$ . This is possible as $f$ can only take finitely many values, and the expressive power of persistence pairs only depend on the relative ordering of its values on the simplicies. Being vertex-based is equivalent to having $f ( e ) = \operatorname* { m a x } ( f ( v ) , f ( \bar { w } ) )$ . Note the degree function is vertex-based and already normalized.

In general the persistence pairs depend on the filtration. When such filtration is canonically defined (e.g., degree-based), it is a graph isomorphism invariant. Here, “canonically defined" refers to any permutation equivariant filtration scheme Φ (see Section 3 of [2]). For our purposes, we define:

Definition 2.4. A filtration scheme Φ assigns every graph G a filtration function $\Phi _ { G } : V ( G ) \cup$ $E ( G ) \to \mathbb { R }$ . Φ is permutation equivariant if for any graph isomorphism $\psi : G \to H , \phi _ { G } = \bar { \phi _ { H } } \circ \psi$ For any such Φ valued in $\mathbb { Z } _ { \geq 0 } .$ , we write $H _ { \bullet } ^ { \Phi }$ to denote the sequence of subgraphs associated to $\Phi$

Proposition 2.5 (Proposition 1 of [2]). A permutation equivariant filtration scheme gives the same lists of persistence diagrams on isomorphic graphs.

In this work, we are interested in generating graph filtrations whose persistence pairs match a given PH signature. For convenience, we will use the following two terminologies.

Definition 2.6. We say two filtrations $G _ { \bullet }$ and $H _ { \bullet }$ are PD-equivalent (persistence-diagram-equivalent) if $\mathcal { P } ( G _ { \bullet } ) = \mathcal { P } ( H _ { \bullet } )$ . We say $G$ and H are degree-PD-equivalent if $\mathcal { P } ( G _ { \bullet } ^ { \mathrm { d e g } } ) = \mathcal { P } ( H _ { \bullet } ^ { \mathrm { d e g } } )$ . The k-th truncation ${ \mathcal { P } } ^ { k } ( G _ { \bullet } )$ of a filtration $G _ { \bullet }$ is ${ \mathcal { P } } ( G _ { \bullet } ^ { k } : = G _ { - 1 } \subseteq G _ { 0 } \subseteq \ldots \subseteq G _ { k } )$

## 3 Local Generation

In this section, we describe a local generation process for producing PD-equivalent random filtrations. We then modify the process to adapt to vertex-based and degree-based filtrations. The methods here take a local approach to generation by iteratively generating a sequence of subgraphs $H _ { \bullet }$ such that their k-th truncations $( 1 ) { \overset { \vartriangle } { \mathcal { P } } } ^ { k } ( H _ { \bullet } ) = { \overset { \bullet } { \mathcal { P } } } ^ { k } ( G _ { \bullet } )$ agree for each $k \in \{ - 1 , 0 , . . . , \overline { { n } } \}$ and (2) the construction of $H _ { k }$ should only depend on $H _ { k - 1 }$ and PDs present in $G _ { \bullet } ^ { k }$ . The algorithms can generate all possible PD-equivalent filtrations if they terminate, but they can run into infeasible states that will be explained later in this section.

## 3.1 Producing PD-Equivalent General Filtrations

Here, we describe a local generation process to iteratively create a general filtration $H _ { \bullet }$ such that $\mathcal { P } ( H _ { \bullet } ) = \mathcal { P } ( G _ { \bullet } )$ . The exact details are outlined in Algorithm 1 (see Appendix E). Since we aim to make $H _ { \bullet }$ such that its truncations agrees with that of the hidden input $G _ { \bullet }$ , we describe a procedure to inductively build $H _ { k }$ from a given $H _ { k - 1 }$ whose previous truncations agree with that of $G _ { \bullet }$

1. (Vertex Addition): For each $( k , d ) \in \mathcal { P } _ { 0 } ( G _ { \bullet } )$ , add a new vertex to $H _ { k }$ with birth time $k$

2. (Component Merging): We traverse through the list of tuples in ${ \mathcal { P } } _ { 0 } ( G )$ that die at time k in random order. For each tuple (b, k), we pick a random component X with birth time b and a random component $Y \neq X$ with birth time $\leq b$ and add an edge between a random pair of nodes in $X \times { \bar { Y } }$ . We then update the component birth times appropriately.

3. (Cycle Addition): For each $( k , \infty ) \in \mathcal { P } _ { 1 } \overset { \cdot } { ( } G _ { \bullet } )$ , add an edge between a pair $( p , q ) \in X \times X$ for any component X such that $p , q$ does not already share an edge.

Note here it is important to traverse in random order during the component merging step. Otherwise, there are PD-equivalent filtrations to $G _ { \bullet }$ that Algorithm 1 could not make.

Theorem 3.1. Given $( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ the persistence pairs coming from a hidden filtration $G _ { \bullet } . \operatorname { I f }$ Algorithm 1 terminates, the filtration $H _ { \bullet }$ is PD-equivalent to $G _ { \bullet }$ . Furthermore, any filtration $H _ { \bullet } ^ { \check { \prime } }$ that is PD-equivalent to $G _ { \bullet }$ can be created by Algorithm 1.

The statement of Theorem 3.1 indicates that Algorithm 1 may not be able to terminate sometimes. It turns out Algorithm 1 may run out of cycles to add in the cycle addition stage.

![](images/84833186db783c653e37e4f9534eb58134125fc63f054d3873400e1cbac2ce6d.jpg)

Example 3.2. Let $G _ { \bullet }$ Figure 2: An example of how Algorithm 1 canbe the hidden filtration given run out of cycles in the cycle adding stage.in Figure 2. It is perfectly possible for Algorithm 1 to create $H _ { 0 } \subseteq H _ { 1 }$ , as $\bar { \mathcal { P } } ( \bar { \varnothing } \subseteq H _ { 0 } ) = \mathcal { P } ^ { 0 } ( \bar { G } _ { \bullet } )$ and $\mathcal { P } ( \emptyset \subseteq H _ { 0 } \subseteq H _ { 1 } ) = \mathcal { P } ^ { 1 } ( G _ { \bullet } )$ . However, when going from $H _ { 1 }$ to $H _ { 2 }$ , one cannot add a single edge to run out of cycles.

We remark that cycle addition would not be an obstruction if we allow multi-edges between nodes. For convenience, we derive a simple criterion for when cycle addition would succeed. For a graph B, we define its cycle capacity as $\textstyle \operatorname { c a p } ^ { \operatorname { c y c } } ( B ) : = \sum _ { A }$ $B  ^ { \frac { | V ( A ) | ( | V ( A ) | - 1 ) } { 2 } } - | E ( A ) |$ , component of

Proposition 3.3. Suppose $H _ { k - 1 }$ has been iteratively built. Algorithm 1 can always succeed in the vertex addition and component merging step. Furthermore, if the number of the tuples $( k , \infty )$ is bounded by $\mathrm { c a p } ^ { \mathrm { c y c } } ( H _ { k - 1 } )$ , then Algorithm 1 can succeed in the cycle addition step.

In particular, $\mathrm { c a p } ^ { \mathrm { c y c } } ( B )$ is always non-negative, so we have as a corollary that Algorithm 1 always achieves the desired generation on acyclic graphs.

Corollary 3.4. Algorithm 1 always terminates if ${ \mathcal { P } } _ { 1 } ( G _ { \bullet } )$ is empty $( \mathrm { i } . \mathrm { e } . , G$ is acyclic).

In practice, we could run Algorithm 1 repeatedly and sample from successful outputs (see Section 5.1).

Remark 3.5. Although we focused on vertex-based filtration, there are also edge-based filtrations. A filtration is edge-based if and only if all tuples in $\mathcal { P } _ { 0 }$ have birth time 0. Thus, Algorithm 1 (Appendix E) can generate all possible edge-based filtrations matching the same PD.

## 3.2 Producing PD-Equivalent Vertex-Based Filtrations

Given the PH signatures ${ \mathcal { P } } ( G _ { \bullet } )$ where $G _ { \bullet }$ is a hidden vertex-based filtration, we describe a local generation process to iteratively create a PD-equivalent vertex-based filtration $H _ { \bullet }$ . Algorithm 2 (see Appendix E for more details) is a modification of Algorithm 1 where:

1. (Vertex Addition): For each new vertex added, we record them in a list $N _ { k }$

2. (Component Merging): The pair of nodes $( x , y )$ selected has $\{ x , y \} \cap N _ { k } \neq \emptyset$

3. (Cycle Addition): The pair $( p , q )$ selected to make the cycle has $\{ p , q \} \cap N _ { k } \neq \emptyset$

Modifications 1-3 above would cause obstructions during component merging. To remedy this, we introduce an additional change to the component merging step, motivated by the following.

Proposition 3.6. In a vertex-based filtration, any vertex death at time k forces a trivial pair $( k , k )$

(Minimum Vertex Pivot): If there are no deaths, then ignore this step. Otherwise, let $b _ { \operatorname* { m i n } } : =$ inf <sub>component</sub> <sub>deaths</sub> <sub>of</sub> <sub>the</sub> $\operatorname { f o r m } ( b , k ) \ b .$ If the number of component whose birth time is $b _ { \mathrm { m i n } }$ is equal to the number of tuples of the form $( b _ { \operatorname* { m i n } } , k )$ , we draw an edge from a random $v \in N _ { k }$ to a random vertex in a random component whose birth time is $< b _ { \mathrm { m i n } }$ . Otherwise, we draw an edge from a random $v \in N _ { k }$ to a random vertex in a random component whose birth time $\mathrm { i s } \leq b _ { \mathrm { m i n } }$

There are still obstructions for cycle-addition, but most of preceding discussions in Section 3.1 apply. Theorem 3.7. Given $( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ from a hidden vertex-based filtration $G _ { \bullet }$ . If Algorithm 2 terminates, the filtration $H _ { \bullet }$ is vertex-based and PD-equivalent to $G _ { \bullet }$ . Furthermore, any vertex-based filtration

$H _ { \bullet } ^ { \prime }$ that is PD-equivalent to $G _ { \bullet }$ can be created by Algorithm 2. Furthermore, Algorithm 2 can always succeed in the stages for vertex addition, minimum vertex pivot, and component merging.

Corollary 3.8. Algorithm 2 always terminates if ${ \mathcal { P } } _ { 1 } ( G _ { \bullet } )$ is empty.

## 3.3 Producing PD-Equivalent Degree Filtrations

Given the PH signature $\mathcal { P } ( G _ { \bullet } ^ { \mathrm { d e g } } )$ , we now describe a modification from Algorithm 2 that generates a PD-equivalent degree-based filtration (see Algorithm 3 in Appendix E for exact details), with:

1. (Vertex Addition): For each new vertex v added, we assign a capacity number $\mathrm { c a p } ( v ) = k .$

2. (Minimum Vertex Pivot): We add the additional constraint that the edge must be drawn from $v \in N _ { k }$ to a vertex whose capacity is positive. We subtract 1 off the capacity of the two nodes.

3. (Component Merging): We ensure the pair of nodes $( x , y )$ selected has cap(x), $\mathrm { c a p } ( y ) > 0$ . We then subtract 1 off the capacity of the two nodes.

4. (Cycle Addition): We ensure the pair $( p , q )$ selected has $\mathrm { c a p } ( p ) , \mathrm { c a p } ( q ) > 0$ . We then subtract 1 off the capacity of the two nodes.

Theorem 3.9. Given $( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ the degree-based persistence coming from a hidden graph $G _ { \bullet } ^ { \mathrm { d e g } }$ . If Algorithm 3 terminates, the filtration $H _ { \bullet }$ is degree-based and has the same PDs as $G _ { \bullet } ^ { \mathrm { d e g } }$ . Furthermore, any degree-based filtration $( H ^ { \prime } ) _ { \bullet } ^ { \mathrm { d e g } }$ that is PD-equivalent to $G _ { \bullet } ^ { \mathrm { d e g } }$ can be created by Algorithm 3.

## 3.4 Fundamental Limitations to the Local Approach

One might wonder if the inability to overcome the cycle addition stage is because the local algorithms may not have been optimized enough. In our local approach, we seek to construct our graph filtration iteratively to match the k-th truncation of the given $\hat { \mathcal { P } } ( G _ { \bullet } )$ such that the construction up to $H _ { k }$ should only depend on ${ \mathcal { P } } ^ { k } ( G _ { \bullet } )$ . We show below that these obstructions are instead a fundamental limitation of any similar local approach to generation.

Theorem 3.10. There is no Algorithm A whose input is ${ \mathcal { P } } ( G _ { \bullet } )$ , ranging over all general (resp. vertex-based, degree-based) filtrations of all graphs, and output is a (resp. vertex-based, degree-based) filtration $H _ { \bullet }$ such that

1. Let $\mathcal { O } ( G _ { \bullet } )$ be the set of possible outputs Algorithm A can make from a fixed ${ \mathcal { P } } ( G _ { \bullet } )$ $\mathcal { O } ( G _ { \bullet } ) \dot { }$ is equal to the set of all general (resp. vertex-based, degree-based) H PDequivalent to $G _ { \bullet }$

2. (Local Approach) Let ${ \mathcal { O } } _ { k } ( G _ { \bullet } )$ denote the set of possible $H _ { - 1 } \subseteq \dots \subseteq H _ { k }$ that Algorithm A can make from a given $\mathcal { P } ( G _ { \bullet } ) . \mathcal { O } _ { k } ( G _ { \bullet } ) = \mathcal { O } ( G _ { - 1 } \subseteq . . . \subseteq G _ { k } )$

Conceptually, Theorem 3.10 is saying that trying to ensure perfect generation at each step of the filtration (i.e., proceeding locally) would miss the global structure of the PH signature. The idea is that conditions (1) and (2) forces every graph filtration PD-equivalent to a truncation of $G _ { \bullet }$ to admit an extension to a full PD-equivalent filtration. However, we can construct examples where no such extensions exist for general, vertex-based, and degree-based filtrations.

## 4 Global Generation via Constraint Programming Formulation

Here, we take a global approach to generating PD-equivalent degree filtrations. In fact, our model is applicable to all permutation equivariant filtration scheme Φ, which for simplicity of presentation we assume is valued in $\mathbb { Z } _ { \geq 0 }$ (this reduction is possible due to Remark 2.3).

In the local approach, one constructs a solution incrementally and makes a sequence of local decisions. This introduces a strong path dependence, which Theorem 3.10 shows are fundamentally limited. As persistent homology is inherently a global descriptor, we seek a way to fully solve the system as a single object and identify the entire feasible set of combinatorial constraints before any sampling occurs. This approach avoids the accumulation of local choices and enforces correctness on the global level. Solving this naturally leads to the usage of constraint programming (CP).

In this section, we recast strict PD-compliant generation as a constraint satisfaction problem over Boolean and integer variables. When $\Phi = \deg .$ , the encoding is minimal-sufficient: its satisfying assignments are in bijection with graphs H satisfying $\mathcal { P } ( H _ { \bullet } ^ { \mathrm { d e g } } ) = \left( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } \right)$ , and removing any single constraint family admits assignments that are not. Since our methodology applies to any permutation equivariant filtration Φ, we also discuss how to implement various filtration functions in the solver.

## 4.1 The Encoding Implementation

Let Φ be a vertex-based permutation equivariant filtration scheme (e.g., degree) valued in $\mathbb { Z } _ { \geq 0 }$ . Given an input $( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ coming from the PDs of Φ on a hidden graph G, we seek to generate a graph H whose associated PDs under Φ matches the input.

Pre-computed quantities. From $( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ we read off: the vertex set $V = [ n ]$ where $n = | \mathcal { P } _ { 0 } | ;$ ; the filtration value ${ { a } _ { i } } ,$ , equal to the birth time of $i \in V$ in ${ \mathcal P } _ { 0 } ;$ the maximum filtration stage $K = \mathrm { { \dot { m a x } } } _ { i }$ a<sub>i</sub>; the active vertex set at stage k, $V _ { k } = \{ i : a _ { i } \leq k \}$ ; the alive-component count $c c _ { k } = | \{ ( b , d ) \in \mathcal { P } _ { 0 }$ $b \leq k < d \} |$ ; the cumulative cycle count $\mathrm { c y } _ { k } = \left| \{ ( b , \infty ) \in \mathcal { P } _ { 1 } : b \leq k \} \right|$ ; and the per-birth survival count $\alpha ( k , \bar { b } ) = | \{ ( b , d ) \in \mathcal { P } _ { 0 } : d > k \} |$

Note: Since the underlying graph G is hidden, we do not know which vertex is assigned to birth/death pair in $\mathcal { P } _ { 0 }$ . However, this does not matter in the generation of the new graph, and we assign each i to the multiset of birth/death pairs in an arbitrary bijection.

Variables. We model the graph to generate as the lower triangular part of an adjacency matrix M with entries $M _ { i j }$ . To keep track of the persistence data, at each stage k, we also group the active vertices $V _ { k }$ partition into $c c _ { k }$ slots indexed by $[ c c _ { k } ] = \{ 0 , 1 , \ldots , c c _ { k } - 1 \}$ . Each slot is a model for a component of the graph at stage k and can be thought of as the induced subgraph on vertices in the slot. We use the following variables to be solved in constraint programming:

<table><tr><td>Variable</td><td>Domain</td><td>Semantics</td></tr><tr><td> $M _ { i j } , 1 \leq i < j \leq n$ </td><td>{0, 1}</td><td>edge  $\{ i , j \} \in E ( H )$ </td></tr><tr><td> $x _ { k c v } , c \in [ c c _ { k } ] , v \in V _ { k }$ </td><td>{0, 1}</td><td>vertex v in slot c at stage k</td></tr><tr><td> $\pi _ { k c i j } , c \in \mathsf { \bar { [ } } c c \dot { k } ] , \ i , j \in V _ { k } , \ i \neq j$ </td><td>{0, 1}</td><td>i is parent of j in spanning tree of slot c at stage k</td></tr><tr><td> $\ell _ { k c v } , c \in [ c c _ { k } ] , v \in V _ { k }$ </td><td> $\{ 0 , \ldots , n \}$ </td><td>depth of v from the root of slot c at stage k</td></tr><tr><td> $r _ { k c v } : = x _ { k c v } \wedge \bigwedge _ { u < v } \neg x _ { k c u }$ </td><td>{0, 1}</td><td>Is v the vertex with smallest index in slot c at stage k?</td></tr><tr><td> $e _ { k c b } : = \left( \exists v \in { \mathrm { s l o t } } _ { c } ^ { k } : a _ { v } = b \right)$   $\wedge \big ( \nexists v \in \mathrm { s l o t } _ { c } ^ { k } : a _ { v } < b \big )$ </td><td>{0, 1}</td><td>slot c at stage k was born at time b</td></tr></table>

Here the variables $\pi _ { \bullet } , \ell _ { \bullet }$ are introduced to optimize the constraint solver, see Remark 4.1.

Constraints. The encoding consists of six constraint families.

$\mathcal { C } _ { 1 }$ (Φ Alignment). Each vertex aligns to its prescribed value under Φ.

$$
\Phi _ { M } ( i ) = a _ { i } , \qquad \forall i \in [ n ]
$$

In the special case when Φ is degree, the constraint becomes:

$\mathcal { C } _ { 1 } ^ { \mathrm { d e g } }$ (degree). Each vertex realizes its prescribed degree.

$$
\sum _ { j \neq i } M _ { i j } = a _ { i } \qquad \forall i \in [ n ]
$$

$\mathcal { C } _ { 2 }$ (partition). At each stage, every active vertex belongs to exactly one slot.

$$
\sum _ { c \in [ c c _ { k } ] } x _ { k c v } = 1 \qquad \forall k , v \in V _ { k }
$$

$\mathcal { C } _ { 3 }$ (edge-slot consistency). The endpoints of every edge share a slot at every stage where both are active (i.e., there is no edge between two vertices at different slots).

$$
M _ { u v } \ \leq \ 1 - \vert \ x _ { k c u } - x _ { k c v } \vert \qquad \forall k , c \in [ c c _ { k } ] , u < v \in V _ { k }
$$

$\mathcal { C } _ { 4 }$ (slot connectivity). Each slot induces a connected subgraph at every stage. We witness this by forcing the existence of a spanning tree on the slot, encoded via parent pointers π rooted at the canonical root $r _ { k c v } ,$ , with depths ℓ ruling out cycles. The tree must use only graph edges within the slot $( \pi _ { k c i j } \leq M _ { i j } \land x _ { k c i } \land x _ { k c j } ) ;$ ; each non-root member has exactly one parent $\begin{array} { r } { ( \sum _ { u \neq v } \pi _ { k c u v } = } \end{array}$ $x _ { k c v } - r _ { k c v } ) ;$ and depths increase along parent edges $( \pi _ { k c i j } = 1 \Rightarrow \ell _ { k c j } = \ell _ { k c i } + 1 )$ . Note that the spanning tree here is to ensure connectivity and is not used for the final graph $( M _ { i j } ^ { \mathbf { \gamma } } \mathbf { s } )$ produced.

$\mathcal { C } _ { 5 }$ (Euler characteristics). Edge counts at each stage match the prescribed cycle-birth counts via $| E _ { k } | - | V _ { k } | + c c _ { k } = \mathrm { c y } _ { k }$

$$
\sum _ { i < j , ~ i , j \in V _ { k } } { M } _ { i j } ~ - ~ | V _ { k } | ~ + ~ c c _ { k } ~ = ~ \mathrm { c y } _ { k } \qquad \forall k .
$$

$\mathcal { C } _ { 6 }$ (birth-time survival). The number of slots whose oldest member was born at b equals $\alpha ( k , b )$ the number of $\mathcal { P } _ { 0 }$ pairs born at b still alive at stage k.

$$
\sum _ { c \in [ c c _ { k } ] } { \mathbf { 1 } } [ e _ { k c b } ] = \alpha ( k , b ) \qquad \forall k , b ,
$$

where $e _ { k c b }$ holds iff slot c contains a vertex born at b but no vertex born earlier. Note that when $\Phi$ is degree, we note that $e _ { k c b } = \left( \exists v \in \operatorname { s l o t } _ { c } ^ { k } : \deg ( v ) = b \right) \land \left( \nexists v \in \operatorname { s l o t } _ { c } ^ { k } : \deg ( v ) < b \right)$

Remark 4.1. We used spanning trees to enforce connectivity in $\mathcal { C } _ { 4 }$ as an optimization step for the solver. A naive approach to checking connectivity would be to ensure that any two vertices in a slot has a path between them, which is about quadratic in time. Checking a spanning tree is about linear.

```latex
Theorem 4.2 (Correctness of the encoding). Let $\mathcal { C } = \{ \mathcal { C } _ { 1 } , \ldots , \mathcal { C } _ { 6 } \} , \mathcal { F }$ the set of satisfying
assignments to C, and $\mathcal { H } ( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ the set of graphs H with $\mathcal { P } ( H _ { \bullet } ^ { \Phi } ) \overset { ^ { } } { = } ( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ . Define $\dot { H } ( \bar { \sigma } )$
from $\sigma \in { \mathcal { F } }$ by $E ( H ( \sigma ) \big ) = \{ \{ i , j \} : \sigma ( M _ { i j } ^ { - } ) = 1 \}$ . Then:
1. Soundness: $H ( \sigma ) \in \mathcal { H } ( \dot { \mathcal { P } _ { 0 } } , \dot { \mathcal { P } _ { 1 } } )$ for every $\sigma \in { \mathcal { F } } .$
2. Completeness: every $\dot { H } \in \mathcal { H } ( \dot { \mathcal { P } } _ { 0 } , \mathcal { P } _ { 1 } )$ admits at least one $\sigma \in { \mathcal { F } }$ with $H ( \sigma ) = H .$
3. Minimality for Degree: When $\Phi = \deg ,$ for each $i \in \{ 1 , \ldots , 6 \}$ , there exist a target
$( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ and an assignment σ that satisfies $\mathcal { C } \setminus \{ \mathcal { C } _ { i } \}$ but with $H ( \sigma ) \overset { \cdot } { \notin } \mathcal { H } ( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$
```

Here, soundness follows as all constraints imposed are formal properties satisfied by PDs, and completeness follows from carefully arguing these constraints completely determine the representatives. Note that the completeness of Theorem F.6 implies for any graph G the solver can generate, it can also reach every representative in the same graph isomorphism class.

The full proof of Theorem 4.2 (1-2) is in Appendix F. In Appendix G, we generated 6 assignments see (Figure 4) as counter-examples in the proof for the minimality of the encoding when $\Phi =$ deg. In Appendix I, we also provide a complexity and scaling analysis of the global encoder for degree.

## 4.2 Implementation of Various Filtration Schemes

To demonstrate that the framework is not limited to degree, we implement other filtration schemes. Since only $\mathcal { C } _ { 1 }$ (the alignment $\Phi _ { M } ( i ) = a _ { i } )$ depends on the filtration, we specify it for each below. Recall $M _ { i j }$ is the $( i , j )$ -th entry in the adjacency matrix and $\begin{array} { r } { d _ { i } = \sum _ { j \neq i } { M _ { i j } } } \end{array}$

(a) triangle and k-clique count. For each vertex $v ,$ its triangle count is the number of 3-cycles on G that includes the vertex v. We can encode this as

Triangle Count:

$$
\sum _ { j < l , \ j , l \neq i } M _ { i j } M _ { i l } M _ { j l } = a _ { i } \qquad \forall i \in [ n ] .
$$

Similar formulas also hold for k-clique count for $k \geq 3 .$

(b) clustering coefficient. The clustering coefficient of a vertex i is twice its triangle count $t _ { i }$ divided by $d _ { i } ( d _ { i } - 1 )$ when $d _ { i } \geq 2$ , and 0 otherwise. We can encode this as the equality

$$
2 \sum _ { j < l , \ j , l \neq i } M _ { i j } M _ { i l } M _ { j l } = a _ { i } \operatorname * { m a x } ( 1 , d _ { i } ( d _ { i } - 1 ) ) \qquad \forall i \in [ n ] .
$$

Remark 4.3. As $a _ { i }$ is rational but the solver is integer-based, we in practice scale by a positive integer D to give the equation $2 D t _ { i } = p _ { i }$ max $( 1 , d _ { i } \mathbf { \bar { ( } } d _ { i } - 1 ) )$ with $\mathrm { \Delta } p _ { i } = D a _ { i } \in \mathbb { Z }$ . This is exact because the persistence diagram depends only on the order of the filtration values, not their scale (Remark 2.3). We also apply this in practice to the subsequent filtrations.

(c) average neighbor degree. The average neighbor degree of a vertex is the average of degrees of its neighboring vertices. This can be encoded as $\begin{array} { r } { \big ( \sum _ { j \neq i } \breve { M } _ { i j } d _ { j } \big ) / \operatorname* { m a x } ( 1 , d _ { i } ) = a _ { i } } \end{array}$ , or more preferably,

Average Neighbor Degree: $\sum _ { j \neq i } M _ { i j } d _ { j } = a _ { i } \operatorname* { m a x } ( 1 , d _ { i } ) \qquad \forall i \in [ n ] .$

(d) closeness centrality. For a connected graph, the closeness centrality of a vertex i is $n - 1$ divided by the sum of shortest-path distances $\delta ( i , j )$ from every j to i. We can encode this as

Closeness Centrality

$$
n - 1 = a _ { i } \sum _ { j \neq i } \delta _ { i j } \qquad \forall i \in [ n ] .
$$

Here $\delta _ { i j }$ are integer variables forced to be $\delta ( i , j )$ by imposing the constraints: for each source i, (i) $\delta _ { i i } \stackrel { \smile } { = } 0 ; ( \mathrm { i i } ) \delta _ { i j } \geq 1$ for $j \neq i ; ( \operatorname { i i i } ) | \delta _ { i j } - \dot { \delta } _ { i k } | \leq 1$ when $\begin{array} { r } { \dot { M } _ { j k } = 1 ; } \end{array}$ and (iv) every $j \neq i$ has a neighbor $p _ { i j }$ with $\dot { \delta } _ { i p _ { i j } } = \delta _ { i j } - 1$ . This forces $\delta _ { i j } = \delta ( i , j )$

(e) random walk centralities. The same technique of pinning a value as the unique solution of a system covers both Katz centrality and PageRank. For a given α, we can encode them as

Katz centrality : $a _ { i } = 1 + \alpha \sum _ { j \neq i } M _ { i j } a _ { j } , \forall i \in [ n ]$ ; PageRank : $\begin{array} { r } { a _ { i } = \frac { 1 - \alpha } { n } + \alpha \displaystyle \sum _ { j \neq i } \frac { M _ { i j } } { d _ { j } } a _ { j } , \forall i \in [ n ] } \end{array}$

In Appendix J, we verify soundness and completeness of all filtrations above on small graphs.

## 5 Experiments

Datasets. We evaluate on four standard graph generation benchmarks: COMM-20 [63], PLANAR [32], SBM [32], and ENZYMES [34], spanning a range of sizes, densities, and topological regimes. We adopt the SPECTRE 64:16:20 train/validation/test split convention [32] uniformly across all four datasets. Statistics and splits are deferred to Appendix H.1, see also Remark H.1 for a technical detail on implementing Algorithm 3.

## 5.1 Comparing Local and Global Generation

In this first experiment we evaluate our two PD-exact generators against each other. For each test graph, we feed its degree-based PD $( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ as the target to both methods, and report the success rate, wall-clock cost, and (for the local sampler) attempts before first success.

Setup. The local degree-filtration generator (Algorithm 3) is allowed up to 20,000 rejection attempts per target before a run is marked as failed; the global CP solver (Sec. 4) is given a 1,800 s per-target time budget. Both run on a 120-CPU-core machine with all per-target generations parallelised across workers. We report success rate over the test split, mean wall-clock per target (charging the full budget on failed runs), and, for the local sampler, mean attempts to first success on successful runs. Results under random and vertex filtrations, together with median statistics, are deferred to Appendix H.4.

Table 2: Local degree-filtration sampler vs. the global CP solver. Wall-clock means charge the full attempt or time budget on failed runs.
<table><tr><td colspan="2"></td><td>COMM-20</td><td>PLANAR</td><td>SBM</td><td>ENZYMES</td></tr><tr><td rowspan="3">LOCAL (Alg. 3)</td><td>Success rate (%) ↑</td><td>100.0</td><td>100.0</td><td>15.0</td><td>99.1</td></tr><tr><td>Wall-clock per target (s) ↓</td><td>0.17</td><td>5.88</td><td>1465.0</td><td>6.94</td></tr><tr><td>Attempts to first success ↓</td><td>49.6</td><td>137.5</td><td>4583.5</td><td>431.5</td></tr><tr><td rowspan="2"> $\mathrm { G L O B A L } \left( \mathrm { C P } , \mathrm { S e c . ~ } 4 \right)$ </td><td>Success rate (%) ↑</td><td>100.0</td><td>100.0</td><td>82.5</td><td>100.0</td></tr><tr><td>Wall-clock per target (s) ↓</td><td>0.30</td><td>16.35</td><td>654.8</td><td>6.85</td></tr></table>

Discussion. Table 2 shows that on COMM-20, PLANAR, and ENZYMES both methods succeed essentially everywhere, with the local sampler 1.7–2.8× faster on the smaller datasets and on par on ENZYMES. However, in SBM, the local sampler succeeds on only 15% of targets and exhausts the budget on the rest, while CP returns a valid graph on 82.5% at less than half the wall-clock. Consistent with Theorem 3.10, local moves fail when the PD encodes a tightly constrained global structure, which CP’s global view recovers. The two methods are thus complementary. Appendix I.3 gives a practitioner’s guide to choosing a priori among them and approximate neural generators.

![](images/3a795cff040a7ae14fded96e6e529f55b1d247b2f2c334bf81e49c889ef9b84f.jpg)  
Figure 3: Qualitative example on an ENZYMES test graph. Node colour encodes degree. CP matches $( n , | E | , \beta _ { 0 } , \beta _ { 1 } )$ exactly, while neural variants add excess cycles. More in Appendix H.6.

## 5.2 Comparison with Neural Generative Baselines

We now compare against neural diffusion baselines that approximate PD-conditioning.

Setup. We compare two backbones, DIGRESS [52] and CONSTRUCT [30], both augmented with TAGG’s [40] topology-aware attention module (TAM) and 1-Wasserstein PDM loss, yielding DI-GRESS+TAGG and CONSTRUCT+TAGG. Both variants are trained per-dataset with PD conditioning supplied via TAM and regularised by PDM. At inference we feed each test graph’s own persistencelandscape vector $\mu _ { G _ { \mathrm { t e s t } } }$ rather than the dataset average $\mu _ { G ^ { \prime } }$ used by the original TAGG, granting the neural variants the strongest possible per-target signal and a paired one-to-one correspondence with the reference. CP receives the same target PD as direct input. Neural variants are evaluated over 5 random seeds; CP is deterministic. We measure topological fidelity by the mean paired 1-Wasserstein distance under the degree filtration, $\begin{array} { r } { \bar { W } _ { 1 } = \frac { 1 } { N } \sum _ { i } [ W _ { 1 } ( \mathcal { P } _ { 0 } ( G _ { \mathrm { r e f } } ^ { ( i ) } ) , \mathcal { P } _ { 0 } ( H ^ { ( i ) } ) ) + W _ { 1 } ( \mathcal { P } _ { 1 } ( G _ { \mathrm { r e f } } ^ { ( i ) } ) , \mathcal { P } _ { 1 } ( H ^ { ( i ) } ) ) ] } \end{array}$ ], with diagonal padding (the metric used as the PDM training loss). Details are deferred to Appendix H.5.

Discussion. Table 3 shows that CP achieves $\bar { W } _ { 1 } = 0$ on every dataset by construction, while the neural variants do not. (full table with std. dev. in Appendix H.5, Table 6) Figure 3 makes the same point per-graph: the neural variants merge two components into one and introduce excess cycles, while CP returns a different graph from the reference (many graphs share the same PD; see Appendix K) but matches every degree-PD invariant exactly. A second qualitative example, with a full per-graph breakdown of edge and cycle drift, is in Appendix H.6. In terms of runtime cost, neural variants are 3–38× faster than CP at inference on larger datasets but re-

Table 3: GLOBAL (CP) vs. TAGG-augmented neural baselines on degree-PD W<sup>¯</sup> <sub>1</sub> (0 = exact match).
<table><tr><td></td><td></td><td>COMM-20 PLANAR</td><td></td><td>SBM</td><td>ENZ.</td></tr><tr><td rowspan="3">CP</td><td> $\bar { W } _ { 1 } \downarrow$ </td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Time (s) ↓</td><td>0.30</td><td>16.35</td><td>654.8</td><td>6.85</td></tr><tr><td>Train (h) ↓</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td rowspan="3">CONSTR. +TAGG</td><td> $\bar { W } _ { 1 } \downarrow$ </td><td>9.35</td><td>43.23</td><td>434.48</td><td>17.35</td></tr><tr><td>Time (s) ↓</td><td>0.88</td><td>4.73</td><td>17.11</td><td>4.92</td></tr><tr><td>Train (h) ↓</td><td>9.4</td><td>24.3</td><td>59.8</td><td>61.6</td></tr><tr><td rowspan="3">DIGRESS +TAGG</td><td> $\bar { W } _ { 1 \downarrow }$ </td><td>4.52</td><td>38.77</td><td>464.99</td><td>17.06</td></tr><tr><td>Time (s) ↓</td><td>0.90</td><td>5.30</td><td>23.16</td><td>5.94</td></tr><tr><td>Train (h) ↓</td><td>8.9</td><td>32.5</td><td>47.6</td><td>49.2</td></tr></table>

quire 9–62 hours of per-dataset training; CP needs none. The two answer different questions: CP when exact PD-correctness is required and the constraint problem is tractable; neural variants when approximate correctness suffices and the training cost amortizes over many generations.

## 6 Conclusion

We initiated a principled study of strict PD-compliant graph generation: producing graphs whose persistence diagram under a given filtration matches a target exactly. Focusing on the degree filtration, we developed two complementary generators: a local rejection sampler with soundness and completeness guarantees, and a global constraint-programming formulation with a minimalsufficient encoding. Furthermore, we extended the global formulation to any vertex-based permutation equivariant filtration scheme. We showed that the local view, while cheap on easy instances, is fundamentally limited (Theorem 3.10) when the PD encodes tight global structure. Empirically, both methods produce graphs realizing the target PD exactly on standard benchmarks, while neural diffusion baselines with PD-conditioning regularizers leave a structural gap that more training does not close. We see this work as a first step: natural extensions include richer filtration schemes, controlled relaxations of strict compliance $( \mathrm { e . g }$ . bottleneck-ball targets), and using the constraint encoding as a building block inside learned generators.

## Acknowledgments and Disclosure of Funding

This project was conceived during discussions at NeurIPS 2025, with early ideas dating back to when MJ and IR were participating during 2025 Aalto Science Institute international summer research programme at Aalto University. MJ is partially supported by the National Science Foundation Graduate Research Fellowship (DGE-2236662). IR and VG acknowledge support from Saab-WASP (grant 411025). VG also acknowledges the Jane and Aatos Erkko foundation (grant 7001703) for their support. We are grateful to the anonymous program chair, area chairs and the reviewers for their constructive feedback and service. MJ would especially like to thank Thomas Brazelton who inspired MJ to think about the exact topological compliance for graphs. MJ would also like to thank Erin Chambers for helpful conversations as well as Yuqin Kewang, Guangyu Meng, and Shankha Mukherjee for support during rebuttal.

## References

[1] R. Anirudh, V. Venkataraman, K. N. Ramamurthy, and P. Turaga. A Riemannian Framework for Statistical Analysis of Topological Persistence Diagrams . In 2016 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 1023–1031, Los Alamitos, CA, USA, July 2016. IEEE Computer Society.

[2] R. Ballester and B. Rieck. On the expressivity of persistent homology in graph learning. In G. Wolf and S. Krishnaswamy, editors, Proceedings of the Third Learning on Graphs Conference, volume 269 of Proceedings ofMachine Learning Research, pages 42:1–42:31. PMLR, 26–29 Nov 2025.

[3] R. L. Belton, B. T. Fasy, R. Mertz, S. Micka, D. L. Millman, D. Salinas, A. Schenfisch, J. Schupbach, and L. Williams. Reconstructing embedded graphs from persistence diagrams. Computational Geometry, 90:101658, 2020.

[4] G. Carlsson. Topology and data. Bulletin ofthe American Mathematical Society, 46(2):255–308, 2009.

[5] M. Carriere, F. Chazal, Y. Ike, T. Lacombe, M. Royer, and Y. Umeda. Perslay: A neural network layer for persistence diagrams and new graph topological signatures. In S. Chiappa and R. Calandra, editors, Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings ofMachine Learning Research, pages 2786–2796. PMLR, 26–28 Aug 2020.

[6] F. Chazal, D. Cohen-Steiner, and Q. Mérigot. Geometric inference for probability measures. Foundations of Computational Mathematics, 11(6):733–751, Dec 2011.

[7] F. Chazal and S. Y. Oudot. Towards persistence-based reconstruction in Euclidean spaces. In Proceedings of the Twenty-Fourth Annual Symposium on Computational Geometry, SoCG ’08, pages 232–241. ACM, 2008.

[8] D. Cohen-Steiner, H. Edelsbrunner, and J. Harer. Stability of persistence diagrams. Discrete & Computational Geometry, 37(1):103–120, Dec 2006.

[9] J. Curry, S. Mukherjee, and K. Turner. How many directions determine a shape and other sufficiency results for two topological transforms. Transactions of the American Mathematical Society, Series B, 9:1006–1043, 2022.

[10] N. De Cao and T. Kipf. Molgan: An implicit generative model for small molecular graphs. arXiv preprint arXiv:1805.11973, 2018.

[11] V. De Silva and R. Ghrist. Coverage in sensor networks via persistent homology. Algebraic & Geometric Topology, 7(1):339–358, 2007.

[12] R. Dechter and J. Pearl. Tree clustering for constraint networks. Artificial Intelligence, 38(3):353–366, 1989.

[13] T. K. Dey and Y. Wang. Computational Topologyfor Data Analysis. Cambridge University Press, 2022.

[14] H. Edelsbrunner and D. Morozov. Persistent homology: Theory and practice. In Proceedings of the European Congress ofMathematics, 2012.

[15] I. P. Gent and B. Smith. Symmetry breaking during search in constraint programming. University of Leeds, School of Computer Studies Leeds, 1999.

[16] R. Ghrist, R. Levanger, and H. Mai. Persistent homology and euler integral transforms. Journal of Applied and Computational Topology, 2(1):55–60, Oct 2018.

[17] O. Hacquard, G. Blanchard, and C. Levrard. Statistical learning on measures: An application to persistence diagrams. Electronic Journal ofStatistics, 20(1):49–82, 2026.

[18] A. Hatcher. Algebraic Topology. New York : Cambridge University Press, 2002.

[19] C. Hofer, F. Graf, B. Rieck, M. Niethammer, and R. Kwitt. Graph Filtration Learning. In H. D. III and A. Singh, editors, Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings ofMachine Learning Research, pages 4314–4323. PMLR, 13–18 Jul 2020.

[20] C. Hofer, R. Kwitt, M. Niethammer, and A. Uhl. Deep learning with topological signatures. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, page 1633–1643, Red Hook, NY, USA, 2017. Curran Associates Inc.

[21] M. Horn, E. De Brouwer, M. Moor, Y. Moreau, B. Rieck, and K. Borgwardt. Topological graph neural networks. arXiv preprint arXiv:2102.07835, 2021.

[22] J. Immonen, A. Souza, and V. Garg. Going beyond persistent homology using persistent homology. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 63150–63173. Curran Associates, Inc., 2023.

[23] M. Ji, I. Roy, and V. K. Garg. Contraction and hourglass persistence for learning on graphs, simplices, and cells. In The Fourteenth International Conference on Learning Representations, 2026.

[24] M. Ji, A. H. Souza, and V. K. Garg. Graph persistence goes spectral. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[25] M. Ji, A. H. Souza, and V. K. Garg. On topological descriptors for graph products. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[26] W. Jin, R. Barzilay, and T. Jaakkola. Hierarchical generation of molecular graphs using structural motifs. In International conference on machine learning, pages 4839–4848. PMLR, 2020.

[27] J. Jo, S. Lee, and S. J. Hwang. Score-based generative modeling of graphs via the system of stochastic differential equations. In International conference on machine learning, pages 10362–10383. PMLR, 2022.

[28] B. Li, D. Ofori-Boateng, Y. R. Gel, and J. Zhang. A hybrid approach for transmission grid resilience assessment using reliability metrics and power system local network topology. Sustainable and Resilient Infrastructure, 6(1-2):26–41, 2021.

[29] B. Liang, S. Chen, L. Gui, H. Wang, Y. Yu, R. Xu, and K.-F. Wong. Centrality-guided pre-training for graph. In The Thirteenth International Conference on Learning Representations, 2025.

[30] M. Madeira, C. Vignac, D. Thanou, and P. Frossard. Generative modelling of structurally constrained graphs. Advances in Neural Information Processing Systems, 37:137218–137262, 2024.

[31] J. P. Marques-Silva and K. A. Sakallah. Grasp: A search algorithm for propositional satisfiability. IEEE Transactions on computers, 48(5):506–521, 1999.

[32] K. Martinkus, A. Loukas, N. Perraudin, and R. Wattenhofer. Spectre: Spectral conditioning helps to overcome the expressivity limits of one-shot graph generators. In International Conference on Machine Learning, pages 15159–15179. PMLR, 2022.

[33] G. Meng, P. Gu, P. Liang, J. P. Lalor, E. W. Chambers, and D. Z. Chen. TopoCL: Topological Contrastive Learning for Medical Imaging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 42681–42690, June 2026.

[34] C. Morris, N. M. Kriege, F. Bause, K. Kersting, P. Mutzel, and M. Neumann. Tudataset: A collection of benchmark datasets for learning with graphs. arXiv preprint arXiv:2007.08663, 2020.

[35] F. Mémoli and T. Needham. Distance distributions and inverse problems for metric measure spaces. Studies in Applied Mathematics, 149(4):943–1001, 2022.

[36] C. Niu, Y. Song, J. Song, S. Zhao, A. Grover, and S. Ermon. Permutation invariant graph generation via score-based generative modeling. In International conference on artificial intelligence and statistics, pages 4474–4484. PMLR, 2020.

[37] T. Ophelders and A. Schenfisch. Sweeping orders for simplicial complex reconstruction, 2025.

[38] N. Otter, M. A. Porter, U. Tillmann, P. Grindrod, and H. A. Harrington. A roadmap for the computation of persistent homology. EPJ Data Science, 6(1), Aug. 2017.

[39] T. Papamarkou, T. Birdal, M. M. Bronstein, G. E. Carlsson, J. Curry, Y. Gao, M. Hajij, R. Kwitt, P. Lio, P. Di Lorenzo, V. Maroulas, N. Miolane, F. Nasrin, K. Natesan Ramamurthy, B. Rieck, S. Scardapane, M. T. Schaub, P. Velickovi ˇ c, B. Wang, Y. Wang, G. Wei, and G. Zamzmi. Position: Topological deep learning is´ the new frontier for relational learning. In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 39529–39555. PMLR, 21–27 Jul 2024.

[40] J. Park, D. Lee, Y. Song, G. Wu, and W. H. Kim. Topology-aware graph diffusion model with persistent homology. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[41] J.-F. Puget. Symmetry breaking revisited. In International Conference on Principles and Practice of Constraint Programming, pages 446–461. Springer, 2002.

[42] J.-C. Régin. A filtering algorithm for constraints of difference in csps. In AAAI, volume 94, pages 362–367, 1994.

[43] T. J. Ritchie and S. J. Macdonald. The impact of aromatic ring count on compound developability–are too many aromatic rings a liability in drug design? Drug discovery today, 14(21-22):1011–1020, 2009.

[44] F. Russold and M. Kerber. Graphcode: Learning from multiparameter persistent homology using graph neural networks. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[45] A. Schenfisch, B. T. Fasy, S. Micka, D. L. Millman, and L. Williams. Efficient graph reconstruction and representation using augmented persistence diagrams. In Y. Bahoo and K. Georgiou, editors, Proceedings of the 34th Canadian Conference on Computational Geometry, CCCG 2022, Toronto Metropolitan University, Toronto, Ontario, Canada, August 25-27, 2022, pages 284–292, 2022.

[46] M. Simonovsky and N. Komodakis. Graphvae: Towards generation of small graphs using variational autoencoders. In International conference on artificial neural networks, pages 412–422. Springer, 2018.

[47] A. E. Sizemore, C. Giusti, A. Kahn, J. M. Vettel, R. F. Betzel, and D. S. Bassett. Cliques and cavities in the human connectome. Journal ofcomputational neuroscience, 44(1):115–145, 2018.

[48] J. Southern, J. Wayland, M. Bronstein, and B. Rieck. Curvature filtrations for graph generative model evaluation. In Proceedings ofthe 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc.

[49] J. Townsend, C. P. Micucci, J. H. Hymel, V. Maroulas, and K. D. Vogiatzis. Representation of molecular structures with persistent homology for machine learning applications in chemistry. Nature communications, 11(1):3230, 2020.

[50] K. Turner, S. Mukherjee, and D. M. Boyer. Persistent homology transform for modeling shapes and surfaces. Information and Inference: A Journal ofthe IMA, 2014.

[51] Y. Verma, A. H. Souza, and V. Garg. Topological neural networks go persistent, equivariant and continuous. In ICML 2024 AIfor Science Workshop, 2024.

[52] C. Vignac, I. Krawczuk, A. Siraudin, B. Wang, V. Cevher, and P. Frossard. Digress: Discrete denoising diffusion for graph generation. arXiv preprint arXiv:2209.14734, 2022.

[53] G. Villines. Topological signatures of structural risk: Persistent homology detects regime-dependent patterns in trade networks and catastrophe loss fields. Available at SSRN 6299678, 2026.

[54] A. Z. Wagner. Constructions in combinatorics via neural networks. arXiv preprint arXiv:2104.14516, 2021.

[55] B. Wang, S. Zheng, J. Wu, J. Li, and F. Pan. Inverse design of catalytic active sites via interpretable topology-based deep generative models. npj Computational Materials, 11(1):147, 2025.

[56] R. Wang, J. Chen, K. Gao, Y. Hozumi, C. Yin, and G.-W. Wei. Analysis of sars-cov-2 mutations in the united states suggests presence of four substrains and novel variants. Communications biology, 4(1):228, 2021.

[57] G.-W. Wei. Topological AI forecasting of future dominating viral variants. ArXiv, pages arXiv–2209, 2022.

[58] P. Woldemariam and N. Attoh-Okine. Topological data analysis for railway track geometry safety and maintenance. Journal ofComputing in Civil Engineering, 39(5):04025073, 2025.

[59] K. Xia, X. Feng, Y. Tong, and G. W. Wei. Persistent homology for the quantitative prediction of fullerene stability. Journal of computational chemistry, 36(6):408–422, 2015.

[60] K. Xu, W. Hu, J. Leskovec, and S. Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations (ICLR), 2019.

[61] L. Xu, F. Hutter, H. H. Hoos, and K. Leyton-Brown. Satzilla: portfolio-based algorithm selection for sat. Journal ofartificial intelligence research, 32:565–606, 2008.

[62] Z. Yan, T. Ma, L. Gao, Z. Tang, Y. Wang, and C. Chen. Neural approximation of graph topological features. In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, editors, Advances in Neural Information Processing Systems, 2022.

[63] J. You, R. Ying, X. Ren, W. Hamilton, and J. Leskovec. Graphrnn: Generating realistic graphs with deep auto-regressive models. In International conference on machine learning, pages 5708–5717. PMLR, 2018.

[64] Q. Zhao, Z. Ye, C. Chen, and Y. Wang. Persistence enhanced graph neural network. In S. Chiappa and R. Calandra, editors, Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pages 2896–2906. PMLR, 26–28 Aug 2020.

# Exact Topological Compliance: Generating Persistence-Equivalent Graphs (Appendix)

## A Limitations and Future Directions

There are interesting graph properties that are not preserved under PD-equivalent graphs. For example, planarity and information about clustering are not necessarily invariants under PD-equivalence. The generation methods discussed in this paper are not guaranteed to maintain them either. As a future direction, we could aim to generate graphs that preserves both topological properties and other non-topological properties. In Appendix K, we tested how diverse graphs in a single degree-based PD-equivalence class can be. Indeed, within one PD-equivalence class, clustering, triangle counts, spectra, and planarity all vary.

There are also a number of future directions that would be interesting to explore, for which our work did not focus on.

Higher-order PDs. We restricted to filtrations on graphs, where the PDs of dimension $\geq 2$ are empty. One may be interested to extend this to filtrations on simplicial complexes. As there are higher simplices, tracking their effects would require matrix-reduction procedures, which increases both computational cost and the chance of infeasible states. To handle higher-order PDs on simplicial complexes, we would need to incorporate additional constraints in Section 4. We anticipate it would be challenging to characterize simple combinatorial constraints for the higher-order PDs, and the CP solver would certainly take longer to solve.

Scalability. Exact global generation can be computationally expensive, and the CP formulation is not expected to scale uniformly to large or difficult instances. In Appendix I, we noted that the encoding size is polynomial; solving is NP-hard in general (it subsumes SAT), so no closed-form runtime bound applies. In regimes where both exact methods fail (harder SBM instances are the concrete example), neural sampling is the only tractable choice, at the cost of exact PD compliance.

General filtrations. The soundness and completeness of the encoding in Theorem 4.2 is independent of the given Φ. However, solve cost depends on the encoding complexity: degree is linear, whereas triangle and clustering add $O ( n ^ { 3 } )$ auxiliaries and are costlier (see Appendix J). One could imagine the costs would vary across general filtrations.

Neural interface. Exact persistence-diagram agreement does not imply that the generated graphs preserve other relevant properties. Coupling our methods with a neural generator to preserve properties topological summaries miss, such as planarity and spectral data, is a promising direction for future work, as is a downstream end-to-end demonstration. Since one motivation for exact compliance was to find counter-examples, one may ask if neural interface (along the themes of [54]) can help to find such examples.

## B Broader Impact Statement

This paper presents work whose goal is to advance the problem of strictly Persistence-Diagram compliant graph generations. As highlighted in the introduction, topological descriptors have a wide variety of usage in biology, infrastructure networks, and more. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## C Declaration of LLM Usage

We used large language models (LLMs) for help in the coding and visualization part of the paper. We also consulted LLMs in the writing of the paper, where we double checked the written statements. The proofs in Appendix D and Appendix F were written by human.

## D Proofs in Section 3

## D.1 Proofs in Section 3.1

Proof of Theorem 3.1. For the first statement, suppose the algorithm terminates at the end, then we have created a sequence of subgraphs at each stage as:

$$
H _ { \bullet } : H _ { - 1 } = \varnothing \subseteq H _ { 0 } \subseteq \ldots \subseteq H .
$$

Here we verify that $\mathcal { P } ( H _ { \bullet } ) = ( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ . Indeed, at each stage $k ,$ each vertex addition step adds in un-filled tuples of the form $( k , - )$ for each $( k , d ) \in \mathcal { P } _ { 0 } .$ . For each $( b , k ) \in \mathcal { P } _ { 0 }$ processed at the component merging step, the edge added in the component merging step goes from a component born at time b to one born at ${ \mathrm { t i m e } } \leq b _ { \mathrm { : } }$ , which has the effect of killing off the prior component. If they both happened to have the same birth time, we can without loss still kill off the first component. Thus, we see that the component merging step perfectly aligns with the $( b , k ) \mathrm { \bar { s } }$ processed in $\mathcal { P } _ { 0 }$ . The same discussion applies to the cycles $( k , \infty )$ processed in the cycle addition step.

Thus, we see that as long as the algorithm terminates, then by design the result filtration $H _ { \bullet }$ has persistence pairs strictly equal to $( \bar { \mathcal { P } _ { 0 } } , \mathcal { P } _ { 1 , }$ ).

For the second statement, by induction the proof reduces to the following case - suppose we have produced a sequence of subgraphs from the algorithm

$$
( G _ { k } ^ { \prime } ) _ { \bullet } : \varnothing = G _ { - 1 } ^ { \prime } \subseteq G _ { 0 } ^ { \prime } \subseteq \ldots \subseteq G _ { k } ^ { \prime } , \quad \mathcal { P } ^ { i } ( ( G _ { k } ^ { \prime } ) _ { \bullet } ) = \mathcal { P } ^ { i } ( G _ { \bullet } ) \mathrm { ~ f o r ~ } - 1 \leq i \leq k ,
$$

Suppose we have $G _ { k + 1 } ^ { \prime } \supseteq G _ { k } ^ { \prime }$ such that $\mathcal { P } ^ { k + 1 } ( ( G _ { k + 1 } ^ { \prime } ) _ { \bullet } ) = \mathcal { P } ^ { k + 1 } ( G _ { \bullet } )$ , we seek to show that $G _ { k } ^ { \prime }$ can be extended to $G _ { k + 1 } ^ { \prime }$ using the moves described in the algorithm. Note here the base case $k = - 1$ is vacuously true.

Indeed, let H be $G _ { k + 1 } ^ { \prime } - G _ { k } ^ { \prime }$ (which could have open edges missing a vertex) and suppose H has m edges.

Although in the step $G _ { k } ^ { \prime } \subseteq G _ { k + 1 } ^ { \prime }$ , the entire H is added at once, we can algorithmically think of this as adding individual vertices and edges one at a time. The theory of persistent homology ensures that the final persistence pairs produced at the end are well-defined independent of the order of simplices added, as long as whenever an edge $\boldsymbol { e } = ( u , v )$ is added, the vertices u and v are added first. Thus we can think of the step $G _ { k } ^ { \prime } \subseteq G _ { k + 1 } ^ { \prime }$ as

$$
G _ { k } ^ { \prime } \subseteq G _ { k , 0 } ^ { \prime } \subseteq G _ { k , 1 } ^ { \prime } \subseteq \ldots \subseteq G _ { k , m } ^ { \prime } = G _ { k + 1 } ^ { \prime }
$$

where $G _ { k } ^ { \prime } \subseteq G _ { k , 0 } ^ { \prime }$ adds all the vertices of H in, and for $i \geq 0 , G _ { k , i } ^ { \prime } \subseteq G _ { k , i + 1 } ^ { \prime }$ adds in exactly one edge from H. The step $G _ { k } ^ { \prime } \subseteq G _ { k , 0 } ^ { \prime }$ is exactly the first two steps of the algorithm.

Now from the step $G _ { k , 0 } ^ { \prime }  G _ { k , 1 } ^ { \prime }$ , an edge either kills a tuple in $\mathcal { P } _ { 0 }$ or makes a tuple in $\mathcal { P } _ { 1 }$ . If the edge $\boldsymbol { e } = ( v , w )$ adds a cycle, then $v ,$ w belongs to the same component at $G _ { k , 0 } ^ { \prime }$ . Observe that in general if two vertices $v ,$ w are in the same component of graph $A ,$ and $A \subseteq B ,$ , then $v ,$ w are in the same component of $B .$ . Thus, it is possible that an edge created in the cycle creation step of the algorithm links $v$ and $w .$ If the edge fills a tuple $( b , - ) \mathrm { t o } \left( \breve { b } , k + 1 \right)$ , then it is drawn from an undead component containing v whose birth time is at b to a component containing w that is born earlier. Now observe that in general if the component of $v \in A$ has birth time $b ,$ and $A \subseteq B$ , then $v \in B$ also has birth time $\leq \bar { b } \mathrm { ( i . e }$ . the birth time can only decrease). Thus, this can happen in the first tuple encountered in the componet merging step.

We can inductively align each step $G _ { k , i } ^ { \prime } \subset G _ { k , i + 1 } ^ { \prime }$ such that the edge e added, if contributing to the death time of a component, is the $j \mathrm { - t h }$ step in the for loop of the component merging step, and e is the j-th component-killing-edge added. Indeed, by the second observation in the last paragraph, it suffices to note that the cycle-creating-edges does not link different components. Thus, we conclude the order of component-killing-edges added in $G _ { k , 0 } ^ { \prime }  G _ { k , 1 } ^ { \prime }$ can be realized as the algorithmic steps occurring in the component merging step. As argued, above, the cycle-creating edges can all be deferred to the cycle creating step and are hence accounted as well. This concludes the proof.

Proof of Proposition 3.3. Suppose by induction we have iteratively built a graph $H _ { k - 1 }$ . If we just view $H _ { k - 1 }$ as the final graph, then by the first part of Theorem 3.1, we know that $\mathcal { P } ( \tilde { H } _ { - 1 } \subseteq H _ { 0 } \subseteq . . . H _ { k - 1 } ) = \mathcal { P } ^ { k } ( \tilde { G } _ { \bullet } ) \mathbf { \tilde { ( } \dagger ) }$

Now, clearly the vertex addition step can always be carried out, as we are just adding disjoint vertices to the graph. The component merging step can always be carried out. Indeed, by (†), we know the components in $H _ { k - 1 }$ and the components in $G _ { k - 1 }$ have the same multi-set of birth times. Thus, we can always find the desired edge between two components in the component merging step.

Finally, for the cycle addition step: We first note that, in general, for non-empty connected graphs A and B where A has more than 1 vertex, let C denote linking A and B by an edge. Observe that C always has room for cycles since we can draw an edge between any vertex in A not linked to B to vertices in $B .$

Now we observe that for a graph A with isolated vertices, write $A = A ^ { \prime } \sqcup A ^ { \prime \prime }$ where $A ^ { \prime }$ has no isolated vertices and $A ^ { \prime \prime }$ is a discrete graph (no edges). Then clearly $\mathrm { c a p } ^ { \mathrm { c y c } } ( A ) = \mathrm { c a p } ^ { \mathrm { c y c } } ( A ^ { \prime } )$ as isolated vertices cannot support cycles.

Now let $H _ { k } ^ { \prime } \supseteq H _ { k - 1 }$ be the new graph extending $H _ { k - 1 }$ that is created after the component merging step. Observe that all the edges e added in the component merging step fall into one of the following two scenarios (1) e is drawn between two isolated vertices or $_ { ( 2 ) e }$ is drawn between two connected graphs A and B where one of them has more than 1 vertex. Regardless, we see that the cycle capacity of the graph can only increase, and ca $\mathrm { { s ^ { c y c } } } ( H _ { k } ^ { \prime } ) \geq \mathrm { { c a p ^ { c y c } } } ( H _ { k } )$ . Finally, the cycle addition step can succeed if and only if $\mathrm { c a p } ^ { \mathrm { c y c } } ( H _ { k } ^ { \prime } )$ is $\geq$ the number of tuples of the form $( k , \infty )$ in $\mathcal { P } _ { 1 }$ . This holds since it holds for $\mathrm { c a p } ^ { \mathrm { c y c } } ( \bar { H } _ { k } )$ □

ProofofCorollary 3.4. By Proposition 3.3, we know that we can successfully build $H _ { k + 1 }$ from a given $H _ { k }$ if $\mathrm { c a p } ^ { \mathrm { c y c } } ( H _ { k } )$ is greater than or equal to the number of tuples of the form $( k + 1 , \infty )$ in $\mathcal { P } _ { 1 \cdot } \operatorname { I f } \mathcal { P } _ { 1 } ( G _ { \bullet } )$ is the empty list, this number is in particular zero. Since the cycle capacity of a graph is non-negative, it is above greater than or equal to 0. Thus, we can always successfully build $H _ { k + 1 }$ from a given $H _ { k }$ here. The statement then follows from induction. 口

## D.2 Proofs in Section 3.2

Proof of Proposition 3.6. It suffices to show that any non-trivial vertex death at a time must also be accompanied with trivial deaths. We suppose for contradiction that this statement is false.

Let $G _ { \bullet }$ be a vertex-based filtration of such counter-example at stage $G _ { k } \subseteq G _ { k + 1 }$ . Now suppose the non-trivial death occurs by an edge $\boldsymbol { e } = ( v , w )$ in the steps building $G _ { k }$ to $G _ { k + 1 }$ . Without loss, we can also take $v \in V ( G _ { k } )$ and $w \in V ( G _ { k + 1 } - G _ { k } )$ , because f is a vertex-based filtration. If w has not been linked to any edge yet, this can only constitute a trivial death as w is born later than v. Thus, it must be the case that w had an edge connected to it already. In the first time w is connected to some other vertex $v _ { 0 } ,$ either w dies because v is younger (giving a trivial pair) or w and v are born at the same time (still giving a trivial pair). Thus, we see we would get a trivial pair regardless. □

ProofofTheorem 3.7. By design, Algorithm 2 realizes a subset of possible steps taken in Algorithm 1. Thus Theorem 3.1 implies that the final $H _ { \bullet }$ , if created, has $\mathsf { \bar { P } } ( H _ { \bullet } ) = \mathsf { ( \bar { P } _ { 0 } , \mathcal { P } _ { 1 } ) }$ . Now we wish to ensure that $H _ { \bullet }$ is a vertex-based filtration. Indeed, in the component merging step and the cycle creation step, every edge that is added is conditioned have at least one of its vertices being supported in the newly created vertices at that stage. Thus, the filtration is vertex-based.

For any vertex-based filtration $H _ { \bullet } ^ { \prime }$ that is PD-equivalent to $G _ { \bullet }$ , we now argue how Algorithm 2 can create this. Indeed, if the minimum vertex pivot step is not imposed, then clearly Theorem 3.1 implies that the algorithm can recreate $H _ { \bullet } ^ { \prime } ,$ , as we had only imposed the minimum constraints to make them vertex-based. Thus, it suffices to show that this minimum vertex pivot can always be realized as the first step in $H _ { k } ^ { \prime } \subseteq H _ { k + 1 } ^ { \prime }$ for any vertex-based filtration (provided a vertex death has happened).

Indeed if vertex deaths occurred, then the component merging step has to come with the addition of new edges $\{ e _ { 1 } , . . . , e _ { r } \}$ . Now look at all components that the vertices of $e _ { i } \ ' _ { \mathbf { S } }$ belong in for all i and find the one with smallest birth time. Since the list of edges is non-empty, we can find a possibly non-unique component.

Now we claim we can perform the minimum vertex pivot there. Indeed, let B be the birth time of this component. By construction we know that $B \leq b _ { m i n } , \mathrm { a s } \ b _ { m i n }$ takes the minimum over a subset of the edges. If $B \ < \ b _ { m i n } ,$ then this can always be realize in the minimum vertex generation step of the algorithm. $\begin{array} { r } { \mathrm { ~ I f ~ } B { \bf \phi } = b _ { m i n } . } \end{array}$ we need to check that not all components born at $b _ { m i n }$ are set to die at $k + 1$ . Indeed, suppose for contradiction they are, then to kill off the last one, you need to make an edge from a component born earlier than $b _ { m i n } .$ , which implies $B < b _ { m i n } .$ Thus, we see the step for minimum vertex pivot is always possible and can go first.

Finally for the last sentence: clearly we can always do vertex addition, and the ability to do minimum vertex pivot follows from Proposition 3.6. The ability to succeed in component merging follows similarly as in Proposition 3.3 from induction and the fact that the newly created vertices from stage $k - 1$ to stage k are the same in the hidden filtration and the filtration we are creating. □

ProofofCorollary 3.8. This again follows similarly as to the proof of Corollary 3.8. Since we do not need to add cycles $\mathrm { i f } \mathcal { P } _ { 1 }$ is empty, Theorem 3.7 implies the algorithm always terminates. □

## D.3 Proofs in Section 3.3

Proof of Theorem 3.9. By design, Algorithm 3 realizes a subset of possible steps taken in Algorithm 2. Thus Theorem 3.7 implies that the final $H _ { \bullet }$ , if created, is vertex-based and has $\mathcal { P } ( \tilde { H } _ { \bullet } ) = ( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ ). This in particular implies H and G have the same number of vertices born at each stage, and the same number of edges added at each stage (†).

For the first statement, it suffices to show that the filtration is degree-based. Since this is already vertex based, it suffices to show that the degree of each vertex added at stage k is k. This is exactly what the capacity assigned to each vertex means. If the algorithm can terminate at the end, then the capacity of each node has to be 0 by (†) and pigeonhole’s principle. Thus, a node born at time k will obtain k many edges incident to it in the final graph. This gives the desired degree-based filtration.

For the second sentence: A degree-based filtration is in particular a vertex-based filtration. By Theorem 3.7, it can be realized by Algorithm 2. On the other hand, any realization of this in Algorithm 2 automatically passes the capacity check, as the original filtration is a degree-based filtration. Thus, it can be realized by Algorithm 3. □

## D.4 Proofs in Section 3.4

Proof of Theorem 3.10. We split the proof into the following cases.

For General Filtrations: Suppose for contradiction that such Algorithm A exists. Consider $G _ { \bullet } =$ $\left( G _ { - 1 } \subseteq G _ { 0 } \subseteq G _ { 1 } \subseteq G _ { 2 } \right)$ as given in Figure 2. By (1) and (2), we have that $( H _ { - 1 } \subseteq H _ { 0 } \subseteq H _ { 1 } ) \in$ $\operatorname { \mathcal { O } } _ { 1 } ( G _ { \bullet } )$ . By (1), we know that there must exist some $H _ { 2 }$ extending $H _ { 1 }$ such that ${ \mathcal { P } } ( H _ { - 1 } \subseteq H _ { 0 } \subseteq$ $H _ { 1 } \subseteq H _ { 2 } ) = { \mathcal { P } } ( G _ { \bullet } )$ . However, as indicated by the Example 3.2, no such example can exist.

For Vertex-Based Filtrations: Suppose for contradiction that such Algorithm A exists. Consider $G _ { \bullet }$ in the following diagram:

![](images/7df47bd1a118f59681ea8fbf26cc68fed8c62c34c317064f8f9587a1299a7937.jpg)

We have that $\mathcal { P } ( G _ { - 1 } \subseteq G _ { 0 } \subseteq G _ { 1 } ) = \mathcal { P } ( H _ { - 1 } \subseteq H _ { 0 } \subseteq H _ { 1 } )$ . Thus by (1) and (2), there should exist $H _ { 2 }$ extending $H _ { - 1 } \subseteq H _ { 0 } \subseteq H _ { 1 }$ such that $\mathcal { P } ( H _ { - 1 } \subseteq H _ { 0 } \subseteq H _ { 1 } \subseteq \dot { H } _ { 2 } ) = \mathcal { P } ( G _ { \bullet } )$ . We see from the persistence pair that this requires the addition of 4 edges (one can count the number of edges added based on how many vertex deaths and cycle births happened). We see from ${ \mathcal { P } } ( G _ { \bullet } )$ that 3 cycles are made and 1 trivial vertex death occurred in going from $G _ { 1 } \subseteq G _ { 2 }$ . However, by pigeonhole’s principle, there is no way to adjoin a new vertex v to $H _ { 1 }$ along with 4 edges from $H _ { 1 }$ to v such that this happens.

For Degree-Based Filtrations: Suppose for contradiction that such Algorithm A exists. Observe that (1) and (2) implies a sub-filtration of a degree-based filtration should be a degree-based filtration. However, this is not true. For example, consider the degree based filtration of either Graph 1 or Graph 2 in Figure 1 and truncate up to $t = 2$ . There are degree 1 vertices that are spawned at $t = 2 ,$ , which is a contradiction. □

## D.5 Additional Remark for Section 3

Remark D.1. Instead of asking Algorithm 1 to generate PD-equivalent graph filtrations, we could ask Algorithm 1 to defer the cycle addition to a later stage if it ran out of cycles to add. This would typically make a graph filtration that is no longer PD-equivalent, but we would expect it to be a closer approximation of the original graph filtration than many other models in the literature.

## E Pseudocode of the Local Generators

Here, we provide the full pseudocode for the three local generators of Section 3. Algorithm 1 is the general-filtration generator. In Algorithm 2, the changes with respect to Algorithm 1 are shown in violet. Later, in Algorithm 3, the further changes with respect to Algorithm 2 are shown in teal.

Algorithm 1: General filtration generator. PD-equivalent under any general filtration.   
Input: $( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ persistence pairs from a hidden filtration   
Output: $H , f _ { V } ,$ f specifying the filtration of the graph H, or FAIL   
Pre-process: $\begin{array} { r } { \tilde { B _ { 0 } } ( k ) \overset { \cdot } { = } \{ ( \dot { k } , d ) \in \mathcal { P } _ { 0 } \} , D _ { 0 } ( k ) = \{ ( \dot { b _ { , } } \dot { k } ) \in \mathcal { P } _ { 0 } \} , B _ { 1 } ( k ) = | \{ ( k , \infty ) \in \mathcal { P } _ { 1 } \} | ; } \end{array}$   
Initialize: $H  \emptyset ;$ ; union-find U; per-node arrays e (exists), c (comp-id), b (comp-birth); adjacency $A ;$   
for $k { = } 0 , \ldots , n$ (filtration steps)   
foreach $( k , d ) \in B _ { 0 } ( k )$ (Vertex Add.)   
add $v ; f _ { V } ( v )  k ; \mathbf { e } [ v ]  1 ; \mathbf { c } [ v ]  v ;$ b[v]←k;   
foreach $( b , k ) \in s h u f f e ( D _ { 0 } ( k ) )$ (Comp. Merge)   
$I _ { \mathrm { m } } [ i , j ]  \mathbf { e } [ i ] \land \mathbf { \bar { e } } [ j ] \land ( \mathbf { \dot { b } } [ i ] { = } b ) \land ( \mathbf { \dot { b } } [ j ] \leq b ) \land ( \mathbf { c } [ i ] \neq \mathbf { c } [ j ] ) \land \lnot A [ i , j ] ;$   
$( u , v ) \sim \mathrm { U n i f } ( I _ { \mathrm { m } } ) ; A [ u , v ]  1 ; f _ { E } ( u , v )  k ; \mathcal { U }$ .union $( u , v ) ;$   
$\mathbf { b } [ \dot { \mathcal { U } } . \mathrm { f i n d } ( u ) ] \longleftarrow \operatorname* { m i n } ( \mathbf { b } [ \dot { u } ] , \mathbf { b } [ v ] ) ;$   
for $j = 1 , \ldots , B _ { 1 } ( k )$ (Cycle Add.)   
$I _ { \mathrm { c } } [ i , j ] \longleftarrow { \bf e } [ i ] \wedge \mathrm { \dot { e } } [ j ] \wedge ( { \bf c } [ i ] = { \bf c } [ j ] ) \wedge \neg A [ i , j ] ;$   
if $\dot { I } _ { \mathrm { c } } = \mathbf { 0 }$ then return $\mathrm { { F A I L } ; }$   
$( u , v ) \sim \mathrm { U n i f } ( I _ { \mathrm { c } } ) ; A [ u , v ]  1 ; f _ { E } ( u , v )  k ;$   
return H, f<sub>V</sub>, f<sub>E</sub>;

Algorithm 2: Vertex-based filtration generator. Changes with respect to Alg. 1 are in violet.   
Input: $( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ persistence pairs from a hidden vertex-based filtration   
Output: H, $f _ { V } , f _ { E }$ specifying the vertex-based filtration of the graph H, or FAIL   
Pre-process: $\begin{array} { r } { \dot { B _ { 0 } ( k ) } \dot { = } \{ ( \dot { k } , d ) \in \mathcal { P } _ { 0 } \} , D _ { 0 } ( k ) = \{ ( b , k ) \in \mathcal { P } _ { 0 } \} , \check { B _ { 1 } } ( \dot { k } ) = | \{ ( k , \infty ) \in \mathcal { P } _ { 1 } \} | ; } \end{array}$   
Initialize: $H  \emptyset ;$ ; union-find U; per-node arrays e (exists), c (comp-id), b (comp-birth), n (newness);   
adjacency A;   
for $k { = } 0 , \ldots , n$ (filtration steps)   
$\mathbf { n }  \mathbf { 0 } ;$   
foreach $( k , d ) \in B _ { 0 } ( k )$ (Vertex Add.)   
add v; $f _ { V } ( v ) \longleftarrow k ; \mathbf { e } [ v ] \longleftarrow 1 ; \mathbf { c } [ v ] \longleftarrow v ; \mathbf { b } [ v ] \longleftarrow k ; \mathbf { n } [ v ] \longleftarrow 1 ;$   
if $D _ { 0 } ( k ) \neq \varnothing$ (Min. Vertex Pivot) then   
b<sub>min</sub> ←min $\{ b \colon ( b , k ) \in D _ { 0 } ( k ) \}$   
$\beta = b _ { \mathrm { m i n } } - 1$ if all components with birth $b _ { \mathrm { m i n } }$ are dying at $k ,$ else $\beta = b _ { \mathrm { m i n } } ;$   
$I _ { \mathrm { p } } [ i , j ]  ( \mathbf { b } [ i ] \leq \beta ) \land ^ { \cdot } \mathbf { \bar { n } } [ i ] \land \mathbf { n } [ j ] \land \lnot A [ i , j ] ;$   
$( u , v ) \sim \mathrm { U n i f } ( I _ { \mathrm { p } } ) ; A [ u , v ]  1 ; f _ { E } ( u , v )  k ; \mathcal { U }$ .union $( u , v ) ;$   
$\mathbf { b } [ \mathcal { U } . \mathrm { f i n d } ( u ) ]  \operatorname* { m i n } ( \mathbf { b } [ u ] , \mathbf { b } [ v ] ) ;$   
foreach $( b , k ) \in s h u f f e ( D _ { 0 } ( k ) )$ (Comp. Merge)   
$I _ { \mathrm { m } } [ i , j ] \gets \mathbf { e } [ i ] \wedge \mathbf { \tilde { e } } [ j ] \wedge ( \mathbf { \hat { b } } [ i ] = b ) \wedge \ \hat { ( \mathbf { b } [ j ] \leq b ) } \wedge ( \mathbf { c } [ i ] \neq \mathbf { c } [ j ] ) \wedge \neg A [ i , j ] \wedge ( \mathbf { n } [ i ] \vee \mathbf { n } [ j ] )$   
$( u , v ) \sim \mathrm { U n i f } ( I _ { \mathrm { m } } ) ; A [ u , v ]  1 ; f _ { E } ( u , v )  k ; \mathcal { U } . \mathrm { u n i o n } ( u , v ) ;$   
b[U.find $\begin{array} { r } { \mathbf { \Theta } . ( u ) ]  \operatorname* { m i n } ( \mathbf { b } [ u ] , \mathbf { b } [ v ] ) ; } \end{array}$   
for $j = 1 , \ldots , B _ { 1 } ( k ) \quad ( { \bf C y c l e \ A d d . } )$   
$I _ { \mathrm { c } } [ i , j ] \longleftarrow \mathbf { e } [ i ] \land \mathbf { \dot { e } } [ j ] \land ( \mathbf { c } [ i ] = \mathbf { c } [ j ] ) \land \lnot A [ i , j ] \land ( \mathbf { n } [ i ] \lor \mathbf { n } [ j ] )$   
if ${ \cal I } _ { \mathrm { c } } = \mathbf { 0 }$ then return FAIL;   
$( u , v ) \sim \mathrm { U n i f } ( I _ { \mathrm { c } } ) ; A [ u , v ]  1 ; f _ { E } ( u , v )  k ;$   
return $H , f _ { V } , f _ { E } ;$

Algorithm 3: Degree filtration generator. Changes with respect to Alg. 1 are in violet; further   
changes with respect to $\mathrm { A l g } . 2$ are in teal.   
Input: $( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ persistence pairs from a hidden degree-based filtration   
Output: $H , f _ { V } , { \bar { f } } _ { E }$ specifying the degree-based filtration of the graph $H ,$ or FAIL   
Pre-process: $\begin{array} { r } { \dot { B _ { 0 } ( k ) } \dot { = } \{ ( \dot { k } , d ) \in \mathcal { P } _ { 0 } \} , \ : D _ { 0 } ( k ) = \{ ( b , k ) \in \mathcal { P } _ { 0 } \} , \ : B _ { 1 } \ ' ( \dot { k } ) = | \{ ( k , \infty ) \in \mathcal { P } _ { 1 } \} | ; } \end{array}$   
Initialize: $H \gets \emptyset ;$ union-find $\mathcal { U } ;$ per-node arrays e (exists), c (comp-id), b (comp-birth), n (newness), κ   
(capacity); adjacency $A ;$   
for $\bar { k } = 0 , \ldots , \bar { n }$ (filtration steps)   
$\mathbf { n }  \mathbf { 0 } ;$   
foreach $( k , d ) \in B _ { 0 } ( k )$ (Vertex Add.)   
add $v ; f _ { V } ( v ) \gets k ; \mathbf { e } [ v ] \gets 1 ; \mathbf { c } [ v ] \gets v ; \mathbf { b } [ v ] \gets k ; \mathbf { n } [ v ] \gets 1 ; \kappa [ v ] \gets k ;$   
if $D _ { 0 } ( k ) \neq \varnothing$ (Min. Vertex Pivot) then   
$\begin{array} { r } { b _ { \operatorname* { m i n } }  \operatorname* { m i n } \{ b : ( b , k ) \in D _ { 0 } ( k ) \} ; } \end{array}$   
$\beta = b _ { \mathrm { m i n } } - 1$ if all components with birth $b _ { \mathrm { m i n } }$ are dying at $k ,$ else $\beta = b _ { \mathrm { m i n } } ;$   
$I _ { \mathrm { P } } [ i , j ] \gets ( \mathbf { b } [ i ] \le \beta ) \land \neg \mathbf { n } [ i ] \land \mathbf { n } [ j ] \land \neg A [ i , j ] \land ( \kappa [ i ] \lor ( j ] > 0 ) ;$   
if $\bar { I } _ { \mathrm { p } } { = } 0$ then return FAIL;   
$( u , \dot { v } ) \sim \mathrm { U n i f } ( I _ { \mathrm { p } } ) ; A [ u , v ]  1 ; f _ { E } ( u , v )  k ; \mathcal { U } . \mathrm { u n i o n } ( u , v ) ;$   
$\mathbf { b } [ \mathcal { U } . \mathrm { f n d } ( u ) ] {  } \operatorname* { m i n } ( \mathbf { b } [ \bar { u } ] , \mathbf { b } [ v ] ) ; \kappa [ u ] \ – = 1 ; \kappa [ v ] \mathrm { - = 1 ; }$   
foreach $( b , k ) \in s h u f f e ( D _ { 0 } ( k ) )$ (Comp. Merge)   
$I _ { \mathrm { m } } [ i , j ]  \mathrm { e } [ i ] \land \mathrm { e } [ j ] \land ( \mathbf { \dot { b } } [ i ] = b ) \land ( \mathbf { b } [ j ] \leq \dot { b } ) \land ( \mathbf { c } [ i ] \neq \mathbf { c } [ j ] ) \land \lnot A [ i , j ]$   
$\Lambda ( \mathbf { n } [ i ] \vee \mathbf { n } [ j ] ) \wedge ( \kappa [ i ] > 0 \wedge \kappa [ j ] > 0 ) ;$   
if $I _ { \mathrm { m } } = \mathbf { 0 }$ then return $\dot { \mathrm { F A I L } } ;$   
$( u , v ) \sim \mathrm { U n i f } ( I _ { \mathrm { m } } ) ; A [ u , v ]  1 ; f _ { E } ( u , v )  k ; \mathcal { U }$ .union $( u , v ) ;$   
$\mathbf { b } [ \dot { \mathcal { U } } . \mathrm { f n d } ( u ) ] { \longleftarrow } \operatorname* { m i n } ( \mathbf { b } [ \dot { u } ] , \mathbf { b } [ v ] ) ; \ x [ u ] \ - 1 ; \ \kappa [ v ] \ - = 1 ;$   
for $j = 1 , \ldots , B _ { 1 } ( k ) \quad ( { \bf C y c l e \ A d d . } )$   
$I _ { \mathrm { c } } [ i , j ] \longleftarrow \mathbf { e } [ i ] \land \mathbf { \bar { e } } [ j ] \land ( \mathbf { \bar { c } } [ i ] = \mathbf { c } [ j ] ) \land \lnot A [ i , j ]$   
$\Lambda ( \mathbf { n } [ i ] \dot { \vee } \mathbf { n } [ \dot { j } ] ) \wedge ( \dot { \kappa } [ i ] > 0 \dot { \wedge } \ddot { \kappa } [ j ] > 0 ) ;$   
if $I _ { \mathrm { c } } = \mathbf { 0 }$ then return $\mathrm { F A I L ; }$   
$( u , v ) \sim \mathrm { U n i f } ( I _ { \mathrm { c } } ) ; A [ u , v ] \gets 1 ; f _ { E } ( u , v ) \gets k ; \kappa [ u ] \gets 1 ; \kappa [ v ] \gets 1 ;$   
return $H , f _ { V } , f _ { E } ;$

## F Proof for Correctness of Encoding (Section 4)

In this section, we will prove the correctness of the encoder in Section 4 (Theorem 4.2). We first prove the soundness of the encoder, and we will break down the steps in a sequence of lemmas.

We first prove Lemma F.1 as a preliminary step.

Lemma F.1. Constraints $\mathcal { C } _ { 1 }$ to $\mathcal { C } _ { 6 }$ implies $\mathcal { C } _ { 7 }$ (ancestry): Each slot at stage $k { - } 1$ has exactly one successor at stage $k ,$ and the successor inherits all its vertices.

Proof of Lemma F.1. Suppose for contradiction that $\mathcal { C } _ { 7 }$ does not hold, then there exists two vertices v, w at slot c at stage $k - 1$ such that v and w belong in two different slots d and $d ^ { \prime }$ at stage k (Here we implicitly used $\mathcal { C } _ { 2 } )$

By $\mathcal { C } _ { 4 }$ , we know there is a sequence of edges connecting v to w and write $v  v _ { 1 }  . . .  v _ { n } $ w the path. Since v and w belong in two different component, there must exist a pair of adjacent vertices in the path here that belong in two different slots.

Thus, we without loss reduce to the case where $v < w$ share an edge, so $M _ { v w } = 1$ . Now by $\mathcal { C } _ { 3 }$ , we know that at $k + 1$ , we must have

$$
M _ { v w } = 1 \le 1 - \vert x _ { k + 1 , c v } - x _ { k + 1 , c w } \vert
$$

for all $c \in [ \mathrm { c c } _ { k } ]$ . However, since v and w belong in two different slots, there is a slot d such that $x _ { k + 1 , d v } = 1 { \mathrm { ~ a n d ~ } } x _ { k + 1 , d w } = 0$ , which implies

$$
M _ { v w } = 1 \leq 1 - | 1 - 0 | = 0 ,
$$

which is a contradiction.

Definition F.2. Let $\{ M _ { i j } \} _ { i < j }$ be the data specifying a graph H, whose vertex set is still $V = [ n ]$ We write $M ( k )$ to denote the collection of $M _ { i j } { \bf \bar { \Psi } _ { s } }$ such that $\Phi _ { M } ( i ) , \Phi _ { M } ( j ) \leq k$ . In other words, $M ( k )$ specifies the induced subgraph $H _ { k }$ of H of vertices whose value is $\leq k$

On the other hand, given the additional data $x _ { k c v }$ produced by the constraints. For each $k ,$ , we can define $H _ { k } ^ { \mathrm { c p } }$ as the subgraph of H whose vertices are the active vertices $V _ { k } \subseteq V$ , an edge $e = ( v , w ) \in \mathbb { \tilde { H } } _ { k } ^ { \mathrm { c p } }$ if v, w belongs to the same slot at stage k and $M _ { v w } = 1$

We also need the following lemma.

Lemma F.3. Following the notation in Definition F.2, the data of $M _ { i j }$ and $x _ { k c v }$ produced by constraints $\mathcal { C } _ { 1 } , . . . , \mathcal { C } _ { 6 }$ satisfies the following:

1. The active vertices at stage k are all vertices v with $\Phi _ { M } ( v ) = a _ { v } \leq k .$

2. Every slot at stage k has to have at least 1 vertex.

3. The number of slots at stage k is the number of components of $H _ { k }$

4. The vertices in each slot c at stage k correspond to the vertices in each component of $H _ { k }$

Proof. Throughout the proof, we implicitly use $\mathcal { C } _ { 2 }$ to ensure that every vertex belongs in exactly 1 slot.

For the first statement, we first note that $\Phi _ { M } ( v ) = a _ { v }$ is specified by $\mathcal { C } _ { 1 }$ , and the definition of $V _ { k }$ (the active vertex set at stage k) specifies that they are examining vertices v with $a _ { v } \leq k$ . In particular, we see that $V _ { k }$ and $H _ { k }$ shares the same vertex set.

For the second statement. The number of slots in $c c _ { k }$ is the number N of pairs $( b , d )$ there are such that $b \leq k <$ d from the input PDs. Now we have the equality

$$
N = \sum _ { b \leq k } \# \{ ( b , d ) \mid d > k \} = \sum _ { b \leq k } \sum _ { c \in [ c c _ { k } ] } \mathbf { 1 } [ e _ { k c b } ] ,
$$

where the second equality follows from $\mathcal { C } _ { 6 }$ . For a fixed c at stage k, clearly $e _ { k c b }$ can only be 1 for at most one choice of b. Thus, we see that

$$
\sum _ { b \leq k } \sum _ { c \in [ c c _ { k } ] } \mathbf { 1 } [ e _ { k c b } ] \leq | c c _ { k } | = N
$$

with equality if and only if for each $c , e _ { k c b } = 1$ for some b. The fact that $e _ { k c b } = 1$ ensures the existence of a vertex in slot c at stage k.

For the third statement. Suppose the number of components in $H _ { k }$ is less than $\left| c c _ { k } \right|$ . Since every slot has to have at least 1 vertex, there exists two vertices $v ,$ w in a component of $H _ { k }$ such that $v ,$ w belongs to two different slots. By choosing a path if necessary as before, we can without loss pretend that v, w are adjacent to each other. Since $v ,$ w are in different slots, there is a slot c such that $x _ { k c v } = 1$ and $x _ { k c w } = 0$ . Then we have that by $\mathcal { C } _ { 3 }$

$$
M _ { v w } = 1 \le 1 - \left| x _ { k c v } - x _ { k c w } \right| = 1 - \left| 1 - 0 \right| = 0 ,
$$

which is a contradiction.

Now suppose the number of components in $H _ { k }$ is greater than $\left| c c _ { k } \right|$ |. Since each component of $H _ { k }$ has at least 1 vertex, it must be the case that two vertices in different components are assigned to the same slot, which is impossible by $\mathcal { C } _ { 4 }$ . This proves the third statement.

For the fourth statement, by $\mathcal { C } _ { 4 }$ and (2) we know the vertices in each slot c at stage k is connected and non-empty, so each slot must belong to some component in $H _ { k }$ . By $\mathcal { C } _ { 3 }$ and similar argument as before, this assignment is an injection. ${ \bar { \mathbf { B } } } \mathbf { y } \left( 3 \right)$ , we know this map is a bijection. Thus, we know that the induced subgraph on vertices of $c \in \mathrm { c c } _ { k }$ is a subgraph of a component of $H _ { k }$ . The only ambiguity is if $H _ { k }$ contains a vertex $v _ { + }$ not in slot c. However, $\mathcal { C } _ { 2 }$ requires $v _ { + }$ to be in some slot, if it is in any slot other than $c ,$ the assignment is no longer an injection. This concludes the fourth statement.

Recall: In Definition 2.4, we write $H _ { \bullet } ^ { \Phi }$ to denote the sequence of subgraphs specified by the function $\Phi _ { H }$

Using Lemma F.1 and Lemma F.3, we conclude that:

Corollary F.4. $H _ { k } = H _ { k } ^ { \mathrm { c p } }$ for all k. Since the $H _ { k } { } ^ { \ ' } \mathrm { ~ }$ s form the subgraphs in the vertex based filtration function $\Phi _ { M } = \Phi _ { H }$ , this shows that

$$
\mathcal { P } ( \emptyset \subseteq H _ { 0 } ^ { \mathrm { c p } } \subseteq \ldots \subseteq H _ { n } ^ { c p } = H ) = \mathcal { P } ( H _ { \bullet } ^ { \Phi } ) .
$$

We now prove the soundness of the encoding in Theorem 4.2. (Note the notation change from H to $H ( \sigma )$ below)

Theorem F.5 (Soundness of Encoding). Given a fixed $( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ from a hidden graph $G .$ Let $\mathcal { C } = \{ \mathcal { C } _ { 1 } , . . . , \mathcal { C } _ { 6 } \}$ and $\mathcal { F }$ the set of assignment satisfying assignment to C. For every $\sigma \in { \mathcal { F } }$ , the graph $H ( \sigma )$ specified has $\mathcal { P } ( H ( \sigma ) _ { \bullet } ^ { \Phi } ) = ( \mathcal { P } _ { 0 } ^ { \mathbf { \bar { \Phi } } } , \mathcal { P } _ { 1 } )$

Proof. By Corollary F.4, we have that

$$
\begin{array} { r } { \mathcal { P } ( H ( \sigma ) _ { \bullet } ^ { \Phi } ) = \mathcal { P } ( H ( \sigma ) _ { \bullet } ^ { c p } : = \emptyset \subseteq H ( \sigma ) _ { 0 } ^ { \mathrm { c p } } \subseteq \dots \subseteq H ( \sigma ) _ { n } ^ { c p } = H ( \sigma ) ) . } \end{array}
$$

It suffices to prove the right hand side agrees with $( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$

For the ease of notation, we will write $\mathcal { P } ( \emptyset \subseteq H ( \sigma ) _ { 0 } ^ { \mathrm { c p } } \subseteq \dots \subseteq H ( \sigma ) _ { n } ^ { c p } = H ( \sigma ) ) = ( \mathcal { P } _ { 0 } ^ { c n } , \mathcal { P } _ { 1 } ^ { c n } ) .$

For 1-dimensional PDs: $\mathcal { C } _ { 5 }$ ensures that the number of cycles born at each stage k for $H ( \sigma ) _ { \bullet } ^ { c p }$ is the same as the number of cycles born at each stage k in $\mathcal { P } _ { 1 }$

Note that this also implies $H ( \sigma ) _ { k } ^ { c p }$ and $G _ { k } ^ { \Phi }$ has the same number of edges. As by construction they already have the same number of vertices and components.

For 0-dimensional PDs: It suffices to prove the following: For each fixed $( b , d )$ , the number $N ( b , d )$ of pairs of the form $( b , d )$ in $\mathcal { P } _ { 0 }$ and $\mathcal { P } _ { 0 } ^ { \bar { c } n }$ are the same.

For a fixed $b < d , N ( b , d )$ can be computed as

$N ( b , d ) = \#$ {components born at b and still alive at $d - 1 \} - \#$ {components born at b and still alive at $d \}$ This quantity is the same for both sides by $\mathcal { C } _ { 6 }$

For $b = d , N ( d , d )$ can be computed as

$$
N ( d , d ) = \# \{ { \mathrm { v e r t e x ~ d e a t h s ~ a t ~ s t a g e ~ } } d \} - \sum _ { b < d } N ( b , d ) .
$$

Since $H ( \sigma ) _ { k } ^ { c p }$ and $G _ { k } ^ { \Phi }$ has the same number of edges, vertices, components, and cycles for all $k ,$ we know that both sides have the same number $\check { N }$ of vertex deaths that occurred at stage d. We established above that the quantity $N ( b , d )$ is the same for both sides when $b < d .$ . Thus, it follows that the quantity $N ( d , d )$ is the same for both sides. This concludes the proof. □

Now we will prove completeness of the encoder.

Theorem F.6 (Completeness of Encoding). Let K be a graph such that $\mathcal { P } ( K _ { \bullet } ^ { \Phi } ) = ( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ , then there exists an assignment $\sigma \in { \mathcal { F } }$ such that $H ( \sigma ) = K$

Proof. Fix an ordering of the vertices of K in the list [n] and an ordering of the components of $K _ { k } ^ { \Phi }$ at each stage k. We give the following assignment to the variables:

1. $M _ { i j } = 1 \mathrm { i f } ( i , j ) \in E ( K )$ and i < j. M<sub>ij</sub> = 0 if neither $( i , j )$ or $( j , i )$ is in $E ( K )$

2. $x _ { k c v } = 1$ if vertex v is in component c at $K _ { k }$ and 0 otherwise. We can fill in all variables because $K _ { \bullet } ^ { \Phi }$ and $\mathcal { P } _ { 0 }$ have the same number of components at each stage.

3. We create $\pi _ { k c i j }$ and $\ell _ { k c v }$ by choosing any spanning tree of each component c at $K _ { k }$ coming out of the root node with minimum index.

4. $r _ { k c v } \mathrm { a n d } e _ { k c b }$ are dependent on the previous variables.

For the precomputed quantity, we have the choice of making an arbitrary bijection, here we associated each i with its value $\Phi _ { K } ( i )$ . We can do this because the assignment corresponds exactly birth times arising in $\mathcal { P } _ { 0 }$ by the hypothesis.

We now check that they satisfy the constraints $\mathcal { C } _ { 1 } , . . . , \mathcal { C } _ { 6 }$ one by one.

1. For $\mathcal { C } _ { 1 } \mathrm { { : } }$ : This is automatic from how we chose the bijection.

2. For $\mathcal { C } _ { 2 } \mathrm { { : } }$ A vertex v in $K _ { k }$ cannot belong to more than two components, so it follows that $x _ { k c v }$ is 1 for exactly one choice of c at a fixed $( k , v )$ . Thus it follows that $\begin{array} { r } { \sum _ { c \in [ c c _ { k } ] } x _ { k c v } = 1 } \end{array}$

3. For $\mathcal { C } _ { 3 } \mathrm { : }$ If u and v does not share an edge, then the inequality is always true. Thus, we consider when $u , v$ shares an edge. By construction here u, v has to be in the same component, so there is a $c \in [ c c _ { k } ]$ such that $x _ { k c u } = x _ { k c v } = 1$ . Otherwise, for any $c ^ { \prime } \neq c$ in [cc<sub>k</sub>], we have that $x _ { k c ^ { \prime } u } = x _ { k c ^ { \prime } v } = 0$ . In either cases we have that $1 \leq 1$

4. For $\mathcal { C } _ { 4 } \mathrm { : }$ Clearly the spanning tree made at each component satisfies the constraints given.

5. For $\mathcal { C } _ { 5 } \mathrm { : }$ This follows from a standard fact in algebraic topology (see for instance, Theorem 2.44 of [18]) that for a graph A,

$$
\# V ( A ) - \# E ( A ) = \# { \mathrm { c o m p o n e n t s ~ o f ~ } } A - \# \operatorname { i n d e p e n d e n t ~ c y c l e s ~ o f ~ } A .
$$

This is called the invariance of the Euler characteristics. In particular, $\mathcal { C } _ { 5 }$ is satisfied by rewriting the equation above to solve for the cycle counts.

6. For $\mathcal { C } _ { 6 } \mathrm { : }$ The vertex with minimum filtration value in a component is equal to the birth time of said component. Thus, the count holds for $K _ { \bullet } ^ { \Phi }$ as well.

Finally, we are able to prove Theorem 4.2. To recall, this is the statement that:

Theorem. Let $\mathcal { C } = \{ \mathcal { C } _ { 1 } , \ldots , \mathcal { C } _ { 6 } \}$ , F the set of satisfying assignments to C, and $\mathcal { H } ( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ the set of graphs H with $\mathcal { P } ( H _ { \bullet } ^ { \Phi } ) = ( \mathcal { P } _ { 0 } , \mathcal { \dot { P } } _ { 1 } )$ . Define $H ( \sigma )$ from $\sigma \in { \mathcal { F } }$ by $E ( H ( \sigma ) ) = \{ \{ i , j \} : \sigma ( M _ { i j } ) =$ 1}. Then:

1. Soundness: $H ( \sigma ) \in \mathcal { H } ( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ for every $\sigma \in { \mathcal { F } } .$

2. Completeness: every $H \in \mathcal { H } ( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ admits at least one $\sigma \in { \mathcal { F } }$ with $H ( \sigma ) = H$

3. Minimality for Degree: When $\Phi = \deg _ { \mathrm { { \ell } } }$ , for each $i \in \{ 1 , \ldots , 6 \}$ , there exist a target $( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$ and an assignment σ that satisfies $\mathcal { C } \setminus \{ \mathcal { C } _ { i } \}$ but with $H ( \sigma ) \not \in \mathcal { H } ( \mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } )$

ProofofTheorem 4.2. Soundness is proven in Theorem F.5, and completeness is proven in Theorem F.6. For minimality of degree, we removed each of the six constraints and generated an assignment that does not satisfy the removed constraints. The counter-examples are written in Appendix G and explained there. □

## G Per-Constraint Counterexamples for Degree-Based PD

For each necessary constraint $\mathcal { C } _ { i } ,$ we exhibit the a target $G$ we found in our search where, with $\mathcal { C } _ { i }$ disabled, the constraint solver returns a graph H with $\mathrm { P H } ( H ) \neq \mathrm { P H } ( G )$ . Throughout this section, we will always use degree-based filtration and consider degree-based persistence diagrams.

Counterexamples were found by enumerating up to 80 distinct CP solutions per (target, droppedconstraint) pair across all named graphs on $| V | \leq 6$ and many random samples for $| V | \le 9$ . Figure 4 displays the target graph and the counterexample produced.

In this section, we use $( b , d ) ^ { n }$ to indicate the tuple $( b , d )$ has multiplicity $n .$

Remark G.1. Before we proceed, we make a comment on how the $\mathcal { C } _ { 1 }$ constraint is verified. Recall that in the actual generation process, the underlying graph G is hidden from us, so we only have access to the persistence tuples. Therefore, $\mathcal { C } _ { 1 }$ is satisfied as long as one can match the output $\{ \Phi _ { H } ( v ) , v \in \mathbf { \bar { H } } \}$ to the multi-set of birth times that we could read off from $\mathcal { P } _ { 0 }$ . In particular, for degree, this means that the degree of the literal index i in H need not be the degree of the literal index i in G. Of course, after reordering if necessary, we can make this true.

## G.1 $\mathcal { C } _ { 1 }$ (degree)

Target. $G = \ o { 2 K _ { 2 } } { : } \ V = \{ 0 , 1 , 2 , 3 \} , E _ { G } = \{ ( 0 , 1 ) , ( 2 , 3 ) \}$ . Degrees $( 1 , 1 , 1 , 1 ) . ~ { \mathcal { P } } _ { 0 } ( G ) =$ $\{ ( 1 , \bar { 1 } ) ^ { 2 } , ( 1 , \infty ) ^ { 2 } \} , { \mathcal { P } } _ { 1 } ( G ) = \tilde { \emptyset } .$

Counterexample. $H \colon E _ { H } = \{ ( 0 , 1 ) , ( 0 , 3 ) \}$

Witness.

$$
M _ { 0 1 } = M _ { 0 3 } = 1 , \quad M _ { i j } = 0 { \mathrm { o t h e r w i s e } }
$$

$$
x _ { 1 , 0 , 0 } = x _ { 1 , 0 , 1 } = x _ { 1 , 0 , 3 } = 1 , \quad x _ { 1 , 1 , 2 } = 1
$$

$$
\pi _ { 1 , 0 , 0 , 1 } = \pi _ { 1 , 0 , 0 , 3 } = 1 , \quad \ell _ { 1 , 0 , 0 } = 0 , \ell _ { 1 , 0 , 1 } = \ell _ { 1 , 0 , 3 } = 1 .
$$

Verification.

$\mathcal { C } _ { \mathrm { 2 } } \mathrm { : }$ each vertex in one slot.

$\mathcal { C } _ { 3 } \mathrm { : }$ edges (0, 1), (0, 3) within slot 0.

$\mathcal { C } _ { 4 } \mathrm { : }$ slot 0 spans the star $K _ { 1 , 2 }$

• C<sub>5</sub>: 2 − 4 + 2 = 0 = cy<sub>1</sub>.

• C<sub>6</sub>: P<sub>c</sub> 1[e<sub>1,c,1</sub>] = 2 = α(1, 1).

Why $\mathcal { C } _ { 1 }$ fails. $\deg _ { H } ( 0 ) = 2 , \deg _ { H } ( 2 ) = 0 .$

PH Comparison. $\operatorname { P H } ( H ) = ( \{ ( 0 , \infty ) , ( 1 , 2 ) , ( 1 , \infty ) , ( 2 , 2 ) \} , \emptyset ) \neq \operatorname { P H } ( G )$

## G.2 $\mathcal { C } _ { 2 }$ (partition)

Target. $G = P _ { 6 } \colon E _ { G } = \{ ( 0 , 1 ) , ( 1 , 2 ) , ( 2 , 3 ) , ( 3 , 4 ) , ( 4 , 5 ) \}$ , degrees $( 1 , 2 , 2 , 2 , 2 , 1 ) . ~ { \mathcal { P } } _ { 0 } ( G ) =$ $\{ ( 1 , \bar { 2 } ) , ( 1 , \infty ) , ( 2 , 2 ) ^ { 4 } \} , \mathcal { P } _ { 1 } \tilde { ( G ) } = \emptyset$ . Decoded: $c c _ { 1 } = 2 , c c _ { 2 } = 1 , \alpha ( 1 , 1 ) = 2 , \alpha ( 2 , 1 ) = 1$ $\alpha ( 2 , 2 ) = 0$

Counterexample. $H = P _ { 3 } \sqcup K _ { 3 } \colon E _ { H } = \{ ( 0 , 5 ) , ( 1 , 5 ) , ( 2 , 3 ) , ( 2 , 4 ) , ( 3 , 4 ) \}$ . This is the disjoint union of the path graph 0-5-1 and a triangle on $\{ 2 , 3 , 4 \}$

Witness (relaxed partition; vertices {2, 3, 4} assigned to no slot at $k = 2 )$

$$
M _ { 0 5 } = M _ { 1 5 } = M _ { 2 3 } = M _ { 2 4 } = M _ { 3 4 } = 1
$$

$$
x _ { 1 , 0 , 0 } = 1 , x _ { 1 , 1 , 1 } = 1
$$

$$
x _ { 2 , 0 , 0 } = x _ { 2 , 0 , 1 } = x _ { 2 , 0 , 5 } = 1 , \quad x _ { 2 , 0 , 2 } = x _ { 2 , 0 , 3 } = x _ { 2 , 0 , 4 } = 0
$$

$$
\pi _ { 2 , 0 , 0 , 5 } = \pi _ { 2 , 0 , 5 , 1 } = 1 , \quad \ell _ { 2 , 0 , 0 } = 0 , \ \ell _ { 2 , 0 , 5 } = 1 , \ \ell _ { 2 , 0 , 1 } = 2 .
$$

Verification.

• $\mathcal { C } _ { 1 } \colon$ the multi-set of degrees agree.

$\mathcal { C } _ { 3 } \mathrm { : }$ edges (0, 5), (1, 5) in slot 0; (2, 3), (2, 4), (3, 4) outside any slot.

$\mathcal { C } _ { 4 } \mathrm { : }$ slot 0 spans path 0-5-1.

$\mathcal { C } _ { 5 } \colon | E _ { 2 } | - | V _ { 2 } | + c c _ { 2 } = 5 - 6 + 1 = 0 = \mathrm { c y _ { 2 } } .$

$\mathcal { C } _ { 6 } \mathrm { : }$ slot 0 elder-of-1 (contains 0, 1, no younger); not elder-of-2 (contains 5 but also younger). Sums match $\alpha ( 2 , 1 ) = 1 , \alpha ( 2 , 2 ) = 0 .$

The up-shot here is that since vertices 2, 3, 4 are not registered in the component, the pass for $\mathcal { C } _ { 5 }$ and $\mathcal { C } _ { 6 }$ were only conducted on the path graph $0 - 5 - 1$ . Thus, the assignment was satisfied.

Why $\mathcal { C } _ { 2 }$ fails. $\textstyle \sum _ { c } x _ { 2 , c , 2 } = 0 \neq 1$ , similarly for vertices 3 and 4.

PH Comparison. $\mathrm { P H } ( H ) = ( \{ ( 1 , 2 ) , ( 1 , \infty ) , ( 2 , 2 ) ^ { 3 } , ( 2 , \infty ) \} , \{ ( 2 , \infty ) \} ) \neq \mathrm { P H } ( G ) .$

## G.3 $\mathcal { C } _ { 3 }$ (edge-slot consistency)

Target. $G = 2 K _ { 3 } \colon E _ { G } = \{ ( 0 , 1 ) , ( 0 , 2 ) , ( 1 , 2 ) , ( 3 , 4 ) , ( 3 , 5 ) , ( 4 , 5 ) \}$ , all degrees 2. ${ \mathcal { P } } _ { 0 } ( G ) =$ $\{ ( 2 , 2 ) ^ { 4 } , ( 2 , \infty ) ^ { 2 } \} , \mathcal { P } _ { 1 } ( G ) = \{ ( 2 , \infty ) ^ { 2 } \}$

Counterexample. $H \cong C _ { 6 } \colon E _ { H } = \{ ( 0 , 1 ) , ( 0 , 2 ) , ( 1 , 4 ) , ( 2 , 3 ) , ( 3 , 5 ) , ( 4 , 5 ) \}$

Witness (partition respected, cross-slot edges allowed).

$$
{ \mathrm { s l o t } } 0 = \{ 0 , 1 , 2 \} : x _ { 2 , 0 , 0 } = x _ { 2 , 0 , 1 } = x _ { 2 , 0 , 2 } = 1
$$

$$
{ \mathrm { s l o t ~ } } 1 = \{ 3 , 4 , 5 \} : x _ { 2 , 1 , 3 } = x _ { 2 , 1 , 4 } = x _ { 2 , 1 , 5 } = 1
$$

$$
\pi _ { 2 , 0 , 0 , 1 } = \pi _ { 2 , 0 , 0 , 2 } = 1 , \quad \ell _ { 2 , 0 , 0 } = 0 , \ell _ { 2 , 0 , 1 } = \ell _ { 2 , 0 , 2 } = 1
$$

$$
\pi _ { 2 , 1 , 3 , 5 } = \pi _ { 2 , 1 , 5 , 4 } = 1 , \quad \ell _ { 2 , 1 , 3 } = 0 , \ell _ { 2 , 1 , 5 } = 1 , \ell _ { 2 , 1 , 4 } = 2 .
$$

## Verification.

• $\mathcal { C } _ { 1 } \colon$ the multi-set of degrees agree.

• $\mathcal { C } _ { \mathrm { 2 } } \mathrm { : }$ clear.

$\mathcal { C } _ { 4 } \mathrm { : }$ each slot uses 2 in-slot edges to span a path.

• C<sub>5</sub>: 6 − 6 + 2 = 2 = cy<sub>2</sub>.

$\mathcal { C } _ { 6 } \mathrm { : }$ each slot elder-of-2; sum = 2 = α(2, 2).

The up-shot here is that topologically the graph constructed has one component, but slot-wise there are still two slots going on. The calculations for $\mathcal { C } _ { 5 }$ and $\mathcal { C } _ { 6 }$ are conducted using slots rather than components, hence the assignment was satisfied.

Why $\mathcal { C } _ { 3 }$ fails. Edges (1, 4) and (2, 3) have endpoints in different slots.

PH Comparison. $\mathrm { P H } ( H = C _ { 6 } ) = ( \{ ( 2 , 2 ) ^ { 5 } , ( 2 , \infty ) \} , \{ ( 2 , \infty ) \} ) \neq \mathrm { P H } ( G ) .$

## G.4 $\mathcal { C } _ { 4 }$ (slot connectivity)

Target. $G = C _ { 6 } \colon E _ { G } = \{ ( 0 , 1 ) , ( 0 , 5 ) , ( 1 , 2 ) , ( 2 , 3 ) , ( 3 , 4 ) , ( 4 , 5 ) \} . \ \mathcal { P } _ { 0 } ( G ) = \{ ( 2 , 2 ) ^ { 5 } , ( 2 , \infty ) \}$ ${ \mathcal { P } } _ { 1 } ( G ) = \{ ( 2 , \infty ) \} .$

Counterexample. $H = 2 K _ { 3 } \colon E _ { H } = \{ ( 0 , 3 ) , ( 0 , 4 ) , ( 1 , 2 ) , ( 1 , 5 ) , ( 2 , 5 ) , ( 3 , 4 ) \}$

Witness (single slot covering disconnected graph).

$$
M \mathrm { e n c o d e s } H , x _ { 2 , 0 , v } = 1 \forall v .
$$

Spanning-tree variables π, ℓ are unconstrained (assign all zero).

Verification.

• $\mathcal { C } _ { 1 } \colon$ the multi-set of degrees agree.

$\mathcal { C } _ { \mathrm { 2 } } \mathrm { : }$ every vertex is in slot 0.

$\mathcal { C } _ { 3 } \mathrm { : }$ all edges within slot 0.

$$
\bullet \ { \mathcal { C } } _ { 6 } \colon 6 - 6 + 1 = 1 = { \mathrm { c y } } _ { 2 } .
$$

• C<sub>7</sub>: slot 0 elder-of-2; sum = 1 = α(2, 2).

Why $\mathcal { C } _ { 4 }$ fails. Slot 0’s induced subgraph in H is two disjoint triangles (so the subgraph is disconnected). No assignment of $\pi , \ell$ satisfies the spanning-tree depth constraint.

PH Comparison. $\operatorname { P H } ( H = 2 K _ { 3 } ) = ( \{ ( 2 , 2 ) ^ { 4 } , ( 2 , \infty ) ^ { 2 } \} , \{ ( 2 , \infty ) ^ { 2 } \} ) \neq \operatorname { P H } ( G )$

Before we move on to $\mathcal { C } _ { 5 }$ and $\mathcal { C } _ { 6 }$ .

## G.5 $\mathcal { C } _ { 5 }$ (Euler characteristics)

We searched all named graphs and 200+ Erdos–Rényi (ER) random graph samples on˝ $| V | \le 7$ without finding a graph-level counterexample. The smallest in our search appears at $| V | = 8$

Target. $G = \mathrm { E R } ( 8 , 0 . 4 )$ with seed 5:

$$
E _ { G } = \{ ( 0 , 7 ) , ( 1 , 6 ) , ( 2 , 3 ) , ( 2 , 6 ) , ( 2 , 7 ) , ( 3 , 4 ) , ( 3 , 7 ) , ( 4 , 6 ) , ( 5 , 6 ) , ( 5 , 7 ) \} .
$$

Degrees are $( 1 , 1 , 4 , 3 , 2 , 2 , 4 , 4 ) . \mathcal { P } _ { 0 } ( G ) \ = \ \{ ( 1 , 4 ) , ( 1 , \infty ) , ( 2 , 4 ) ^ { 2 } , ( 3 , 3 ) ^ { 2 } , ( 4 , 4 ) ^ { 2 } \} , \ \mathcal { P } _ { 1 } ( G ) \ = \ 0 .$ $\{ ( \bar { 4 } , \infty ) ^ { 3 } \}$

Counterexample. H:

$$
E _ { H } = \{ ( 0 , 6 ) , ( 1 , 6 ) , ( 2 , 4 ) , ( 2 , 5 ) , ( 3 , 6 ) , ( 3 , 7 ) , ( 4 , 5 ) , ( 4 , 7 ) , ( 5 , 7 ) , ( 6 , 7 ) \} .
$$

Witness. The graph H and its associated degree-based filtration specifies the $M _ { i j } { ' } \mathrm { s }$ and $x _ { k c v } \mathbf { \dot { s } } .$ . The spanning tree variables are done at each component. The following table records the progression of relevant data in G and H as the stages k progress.

<table><tr><td>k</td><td>|Vk|</td><td>|Ek(G)|</td><td> $| E _ { k } ( H ) |$ </td><td>CCk</td><td>cyk (G)</td><td>cyk (H)</td></tr><tr><td>1</td><td>2</td><td>0</td><td>0</td><td>2</td><td>0</td><td>0</td></tr><tr><td>2</td><td>4</td><td>0</td><td>0</td><td>4</td><td>0</td><td>0</td></tr><tr><td>3</td><td>6</td><td>2</td><td>3</td><td>4</td><td>0</td><td>1</td></tr><tr><td>4</td><td>8</td><td>10</td><td>10</td><td>1</td><td>3</td><td>3</td></tr></table>

Why $\mathcal { C } _ { 5 }$ Failed. Both graphs share the same per-level component structure and birth-time survival. A witness assigning each vertex to its component’s slot at every level satisfies $\mathcal { C } _ { 1 } – \mathcal { C } _ { 4 } , \mathcal { C } _ { 6 } . \mathrm { O n l y } \mathcal { C } _ { 5 } \mathrm { ^ { \circ } s }$ equation

$$
| E _ { k } | - | V _ { k } | + c c _ { k } = \mathrm { c y } _ { k }
$$

distinguishes the two graphs at $k = 3 \mathrm { : }$

PH Comparison. Here we have that $\mathcal { P } _ { 0 } ( H ) = \mathcal { P } _ { 0 } ( G )$ , but $\mathcal { P } _ { 1 } ( H ) = \{ ( 3 , \infty ) , ( 4 , \infty ) ^ { 2 } \} \neq \mathcal { P } _ { 1 } ( G ) =$ $\{ ( 4 , \infty ) ^ { 3 } \}$

## G.6 $\mathcal { C } _ { 6 }$ (birth-time survival)

Target $G = \mathrm { E R } ( 7 , 0 . 3 )$ with seed 28:

$$
E _ { G } = \left\{ ( 0 , 1 ) , ( 0 , 2 ) , ( 0 , 4 ) , ( 0 , 5 ) , ( 1 , 3 ) , ( 1 , 6 ) , ( 2 , 3 ) , ( 2 , 4 ) , ( 2 , 5 ) , ( 2 , 6 ) , ( 4 , 5 ) \right\} .
$$

Degrees are (4, 3, 5, 2, 3, 3, 2). $\mathcal { P } _ { 0 } ( G ) = \{ ( 2 , 3 ) , ( 2 , \infty ) , ( 3 , 3 ) ^ { 2 } , ( 3 , 4 ) , ( 4 , 4 ) , ( 5 , 5 ) \} , \mathcal { P } _ { 1 } ( G ) =$ $\{ ( \bar { 4 , \infty } ) , ( 5 , \dot { \infty } ) ^ { 4 } \}$

Counterexample. H:

$$
E _ { H } = \{ ( 0 , 2 ) , ( 0 , 6 ) , ( 1 , 5 ) , ( 1 , 6 ) , ( 2 , 3 ) , ( 2 , 6 ) , ( 3 , 4 ) , ( 3 , 5 ) , ( 4 , 5 ) , ( 4 , 6 ) , ( 5 , 6 ) \} .
$$

Witness. The graph H and its associated degree-based filtration specifies the $M _ { i j } { ' } \mathbf { s }$ and $x _ { k c v } \mathbf { \bar { s } } .$ The spanning tree variables are done at each component. At every level, $| V _ { k } | , | E _ { k } \rangle , c c _ { k }$ , and $\mathrm { c y } _ { k }$ are identical between G and H:

<table><tr><td>k</td><td>|Vk</td><td>|Ek|</td><td>CCk</td><td>cyk</td></tr><tr><td>2</td><td>2</td><td>0</td><td>2</td><td>0</td></tr><tr><td>3</td><td>5</td><td>3</td><td>2</td><td>0</td></tr><tr><td>4</td><td>6</td><td>6</td><td>1</td><td>1</td></tr><tr><td>5</td><td>7</td><td>11</td><td>1</td><td>5</td></tr></table>

Why $\mathcal { C } _ { 6 }$ Failed. The difference occurs in $\alpha ( k , b )$ at $k = 3$ . Recall $\alpha ( 3 , 2 )$ refers to the number of components born at 2 that is still alive at stage 3.

$$
\frac { \mid \alpha ( 3 , 2 ) } {  { H } \mid  { 1 } }
$$

PH Comparison. $\mathcal { P } _ { 0 } ( H ) = \{ ( 2 , 4 ) , ( 2 , \infty ) , ( 3 , 3 ) ^ { 3 } , ( 4 , 4 ) , ( 5 , 5 ) \} \neq \mathcal { P } _ { 0 } ( G )$ , and $\mathcal { P } _ { 1 } ( H ) = \mathcal { P } _ { 1 } ( G )$

## G.7 $\mathcal { C } _ { 7 }$ (ancestry)

For completeness, we also empirically tested adding vs. not adding the redundant constraint $\mathcal { C } _ { 7 }$ Across all 95 test cases (graphs up to $| \dot { V } | = 3 0 )$ , disabling $\mathcal { C } _ { 5 }$ yields no graph-level counterexample. Moreover, the solver runs 1.42× faster without $\mathcal { C } _ { 5 } \left( 2 3 . 0 \mathrm { s } \mathrm { v s } . \right.$ . 16.2 s aggregate).

![](images/84a5e04f8a864a6a89adbdee808caceef5af4280e7ba66780225660f48648452.jpg)  
Figure 4: Counterexamples in Appendix G. The left displays the target graph and the right displays a graph generated with constraint $\hat { \mathcal { C } } _ { i }$ removed that has a different degree PD.

## H Experimental Details

## H.1 Datasets

We evaluate on four standard graph generation benchmarks: COMM-20 (small dense two-community SBMs), PLANAR (fixed-size random planar graphs), SBM (large stochastic block model graphs), and ENZYMES (real-world enzyme protein contact maps from the TUDataset [34]). The first three are synthetic and follow the conventions of prior graph generation work [32, 52, 40, 30]; ENZYMES is a real-world dataset also previously used in prior work. We adopt the uniform 64/16/20% train/validation/test split convention from SPECTRE [32].

Aggregate statistics are listed in Table 4. The four datasets cover a range of regimes that stress different aspects of our methods. COMM-20 is the smallest and densest, with the lowest max degree; the easiest setting in which to demonstrate strict PD compliance. PLANAR has fixed $\lvert V \rvert = 6 4$ and every graph is guaranteed to be connected and planar by construction. SBM drives the difficulty: largest graphs (|V| up to 187), highest edge counts (|E| up to 1129), and highest max degree $( \bar { \Delta } = 1 \bar { 7 } . 1 )$ and is where constraint-programming scalability is most stressed. ENZYMES tests generalisation beyond synthetic benchmarks.

Table 4: Dataset statistics (all splits combined). Splits follow the uniform 64/16/20% convention.
<table><tr><td></td><td>COMM-20</td><td>PLANAR</td><td>SBM</td><td>ENZYMES</td></tr><tr><td>Total graphs</td><td>100</td><td>200</td><td>200</td><td>587</td></tr><tr><td>Splits (train / val / test)</td><td>64 / 16 / 20</td><td>128 / 32 / 40</td><td>128 / 32 / 40</td><td>376 / 94 / 117</td></tr><tr><td>Nodes (min / mean / max)</td><td>12 / 15.7 / 20</td><td>64 / 64 / 64</td><td>44 / 104.0 / 187</td><td>10 / 33.0 / 125</td></tr><tr><td>Edges (min / mean / max)</td><td>21 / 37.0 / 61</td><td>173 / 177.8 / 181</td><td>129 / 500.4 / 1129</td><td>16 / 63.3 / 149</td></tr><tr><td>Max. degree (mean)</td><td>6.5</td><td>9.0</td><td>17.1</td><td>6.1</td></tr></table>

## H.2 Baseline Methods

We compare against four neural baselines along two axes: the diffusion backbone (DIGRESS or CONSTRUCT) and whether the topology-aware components of TAGG (TAM module and PDM loss) are added. This yields DIGRESS, CONSTRUCT, DIGRESS+TAGG, and CONSTRUCT+TAGG.

## H.2.1 DiGress

DIGRESS [52] is a discrete denoising diffusion model and the foundation for both CONSTRUCT and TAGG. The forward process independently corrupts each node and edge category through transition matrices $Q _ { X } ^ { t } , Q _ { E } ^ { t }$ . The reverse process is a graph transformer $\phi _ { \theta }$ that maps a noisy graph $G ^ { t }$ to predicted distributions $( \hat { p } ^ { X } , \hat { p } ^ { E } )$ over clean-graph categories, trained with cross-entropy

$$
\mathcal { L } _ { \mathrm { C E } } ( G _ { 0 } , \hat { p } ^ { G } ) = \sum _ { i } \mathrm { C E } ( x _ { i } , \hat { p } _ { i } ^ { X } ) + \lambda \sum _ { i , j } \mathrm { C E } ( e _ { i j } , \hat { p } _ { i j } ^ { E } ) .
$$

We use the marginal-noise variant, which sets the limit distribution to the empirical node and edge marginals of the training set, and augment the denoising network with auxiliary cycle counts and Laplacian spectral features to mitigate the expressivity limits of message passing [60]. DIGRESS is unconstrained and serves as the vanilla diffusion baseline.

## H.2.2 ConStruct

CONSTRUCT [30] extends DIGRESS with hard constraints for edge-deletion-invariant properties (planarity, acyclicity, max degree, etc.). It introduces two modifications.

Edge-absorbing noise model. The forward edge transition sends each edge either to itself or to an absorbing no-edge state,

$$
Q _ { E } ^ { t } = \alpha _ { \mathrm { a b s } } ^ { t } I + \left( 1 - \alpha _ { \mathrm { a b s } } ^ { t } \right) \mathbf { 1 } _ { c } e _ { E } ^ { \prime } ,
$$

making the forward sequence a chain of subgraphs $G ^ { 0 } \supseteq G ^ { 1 } \supseteq \cdots \supseteq G ^ { T } = \emptyset$ and the reverse process pure edge insertion.

Connection to graph filtrations. The forward chain $G ^ { 0 } \supseteq G ^ { 1 } \supseteq \cdots \supseteq G ^ { T } = \emptyset$ (and its reverse) is structurally a general filtration of $G ^ { 0 }$ in the sense of Sec. 3.1, with the edge-insertion order set by the diffusion schedule rather than a canonical filter function. CONSTRUCT is not topologyaware and the schedule has no relation to persistent homology. Still, the subgraph-chain skeleton aligns more naturally with the filtration setting of persistent homology than DIGRESS’s independentedge corruption does, making CONSTRUCT a more filtration-compatible baseline even before any topological signal is added on top.

Projector. At each reverse step, given the candidate $\hat { G } ^ { t - 1 }$ and previous iterate $G ^ { t }$ , the projector iteratively inserts candidate edges $E ( \hat { G } ^ { t - 1 } ) \setminus E ( G ^ { t } )$ in random order, discarding any whose insertion violates the target property. Theorem 1 of [30] shows this recovers an optimal solution to $\mathrm { m i n } _ { G \in \mathcal { C } } \mathrm { G E D } ( \hat { G } ^ { t - 1 } , G )$ for $\mathcal { C } = \{ G : \mathcal { P } ( G ) , G \supseteq G ^ { t } \}$

We apply CONSTRUCT with planarity on PLANAR; for the other three datasets, which lack a uniform edge-deletion-invariant property, we run the same architecture with the projector disabled.

## H.2.3 TAGG: Topology-Aware Attention and PDM Loss

TAGG [40] uses the degree-based filtration as a conditioning signal and proposes two topology-aware components built on top of it, both of which we re-implement and apply to both CONSTRUCT and DIGRESS backbones.

PDM loss. The persistence diagrams $D _ { G _ { 0 } } , D _ { \hat { p } ^ { G _ { 0 } } }$ are computed under the degree-based filtration (matching our setting up to normalisation). The loss is the 1-Wasserstein distance with diagonal padding for cardinality mismatch and per-dimension matching to prevent $\mathcal { P } _ { 0 } / \mathcal { P } _ { 1 }$ confusion:

$$
\mathcal { L } _ { \mathrm { P D M } } ( G _ { 0 } , \hat { p } ^ { G _ { 0 } } ) = W _ { 1 } ( D _ { G _ { 0 } } , D _ { \hat { p } ^ { G _ { 0 } } } ) .
$$

The full objective is $\mathcal { L } = \mathcal { L } _ { \mathrm { C E } } + \alpha \mathcal { L } _ { \mathrm { P D M } }$

Topology-aware attention (TAM). A homological feature vector $\mu _ { G _ { 0 } } \in \mathbb { R } ^ { L \cdot S }$ , derived from the persistence landscape of $G _ { 0 }$ , is injected as a global bias in attention scores at every denoising layer:

$$
\begin{array} { r } { A _ { t } ^ { m } = \mathrm { s o f t m a x } \bigg ( \frac { Q _ { t } ^ { m } ( K _ { t } ^ { m } ) ^ { \top } } { \sqrt { d _ { k } } } \cdot Z _ { E _ { t } } ^ { m } + H _ { G _ { 0 } } \bigg ) , } \end{array}
$$

with $H _ { G _ { 0 } } = \mathrm { L i n } _ { \mu } ( \mu _ { G _ { 0 } } )$ . At inference, $G _ { 0 }$ is unknown, and the original TAGG substitutes the training-set average µ ′ as conditioning, making it a distribution-level method rather than a per-target one. To align it with our setting — where the target PD is given per sample — we instead feed the homological feature vector $\mu _ { G _ { \mathrm { t e s t } } }$ derived from each test graph as the conditioning signal at inference. This grants TAGG the strongest possible per-target topology signal, putting it on equal footing with our methods for the strict PD-compliance evaluation.

## H.2.4 Augmenting DiGress and ConStruct with TAGG

DIGRESS+TAGG adds both TAM and the PDM loss to DIGRESS, recovering the original TAGG configuration. CONSTRUCT+TAGG adds the same components on top of CONSTRUCT’s edgeabsorbing forward process and projector-augmented reverse process; the projector remains active and filters edges proposed by the TAGG-augmented denoiser. In both cases, inference uses the per-target $\mu _ { G _ { \mathrm { t e s t } } }$ derived from each test graph, as described above.

Hyperparameter selection. For each baseline we use the default hyperparameters proposed in the original paper for the datasets considered.

## H.3 Hardware

The constraint-programming solver and the local rejection sampler ran on a 120-CPU-core machine; no GPU was used. Neural diffusion baselines were trained on 4× NVIDIA A100 (40 GB) GPUs and run at inference on a single A100 or RTX 6000 Ada (48 GB).

## H.4 Local vs. Global: Full Comparison Across Filtrations

We expand Sec. 5.1 with the full set of local generators across the three filtration regimes considered in Sec. 3, together with median statistics and per-method commentary.

Methods compared. Recall from Sec. 3 that the local approach yields three rejection samplers, each tied to a different filtration scheme:

• RANDOM (Algorithm 1): general filtration. The vertex and edge filter values $f _ { V } , f _ { E }$ are not canonical and are sampled along with the graph.

• VERTEX (Algorithm 2): vertex-based filtration. Edges are forced to take time max $( f _ { V } ( u ) , f _ { V } ( v ) )$ , but $f _ { V }$ itself is not canonical and is sampled.

• DEGREE (Algorithm 3): degree-based filtration. Both vertex and edge filter values are uniquely determined by the graph, so the algorithm only needs to return the graph itself.

For RANDOM and VERTEX, the algorithm therefore returns the pair $( H , f )$ specifying both the graph and a filtration function realising the target PD; only DEGREE (and the global CP solver) work directly with the canonical degree filter $f _ { V } ( \boldsymbol { v } ) = \deg ( \boldsymbol { v } )$ . In this comparison the CP solver targets the degree filtration, so that it is compared with DEGREE on the same canonical filtration. Other filtrations are verified in Appendix J.

Remark H.1. In the implementation of Algorithm 2 and Algorithm $^ { 3 , }$ we took an alternative approach to the minimum vertex pivot step.

In any stage of vertex based filtrations, let $b _ { \operatorname* { m i n } } = \operatorname { i n f } _ { \mathbf { \mu } }$ component deaths of the form $( b , k { + } 1 )$ b. In the minimum vertex pivot step, we instead do the following procedure:

1. If there is only one component born at $b _ { \mathrm { m i n } } ,$ randomly pick a component born at $t < b _ { \mathrm { m i n } }$ and connect to a new vertex.

2. Otherwise, randomly pick a component born at $t \leq b _ { m i n }$ and connect to a new vertex.

If the algorithm terminates, this is still guaranteed to give a PD-equivalent output and could realize all possible PD-equivalent vertex-based filtrations. This choice involved less variables to check, but a caveat is that this could potentially fail the component merging step.

Setup. We follow the protocol of Sec. 5.1: each rejection sampler is allowed up to 20,000 proposals per target with 4 parallel evaluator threads, and CP is given a 1,800 s budget with 120 internal CP-SAT search workers. Wall-clocks are measured on a clean 120-CPU-core machine. We report success rate, mean and median per-target wall-clock (charging the full budget on failed runs), and mean and median attempts to first success on successful runs.

Results. Table 5 reports the full set of numbers.

Discussion. The three local samplers form a clean ordering from least to most constrained. RAN-DOM, working under a general filtration where $f _ { V } , f _ { E }$ are jointly sampled, accepts almost every proposal (mean attempts 1–3 across all four datasets) and finishes in seconds. VERTEX, with edges deterministically pinned to max $\left( f _ { V } ( u ) , f _ { V } ( v ) \right)$ , sees attempt counts grow to a few hundred on the larger benchmarks but still completes within 9 s on every dataset. DEGREE, where the filtration is fully canonical and the PD effectively determines the degree sequence, is where rejection becomes expensive: the per-attempt success probability collapses on large dense instances, and on SBM only 15% of test targets are solved within the 20,000-attempt budget. The CP solver targets the same degree filtration but, by enforcing the global constraints up front, raises the success rate to 82.5% on SBM and cuts the mean wall-clock by more than a factor of two. On COMM-20, PLANAR, and ENZYMES the two degree-filtration methods solve essentially all targets, with the local sampler 1.7–2.8× faster on the smaller datasets and on par on ENZYMES.

Tail behaviour. The mean–median gap on SBM and ENZYMES reflects a heavy right tail in attempt counts. For DEGREE/ENZYMES (where success is 99.1%), 50% of targets are solved in $\leq 1 5$ attempts, but the top 5% require ≥ 441 and the worst 17,331. The mean (431.5) is dominated by this tail and overstates the typical cost. This is the natural pattern of rejection sampling against PD constraints derived from heterogeneous graphs: most reference PDs are typical and easy to match by chance, but a few have rare structure that the proposal distribution rarely hits.

Practical recipe. The pair (success rate, mean attempts) characterises the difficulty of each (filtration, dataset) cell. Mean attempt $s \leq 1 0 0$ marks an easy cell where the local sampler is the right tool; $\geq 1 { , } 0 0 0$ a borderline cell; $\geq 1 0 { , } 0 0 0$ (or hitting max\_attempts) a hard cell where CP becomes necessary. A simple hybrid — run the local sampler with a modest cap (e.g. 1,000 attempts) and fall back to CP on rejection — achieves the best of both: free generation on the easy majority of targets, principled fallback on the hard tail. We use the local-only and CP-only configurations in the main paper for clarity of presentation.

Table 5: Full local/global comparison across the three filtration regimes. RANDOM and VERTEX target their respective filtrations; DEGREE (Alg. 3) and CP both target the canonical degree filtration. Wall-clock means include the full 20,000-attempt or 1,800 s budget on failed runs; attempt statistics are over successful runs only. CP performs a single solver call per target.
<table><tr><td colspan="2"></td><td>COMM-20</td><td>PLANAR</td><td>SBM</td><td>ENZYMES</td></tr><tr><td rowspan="5">RANDOM (Alg. 1)</td><td>Success (%) ↑</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Wall-clock mean (s) ↓</td><td>0.10</td><td>2.02</td><td>6.04</td><td>0.71</td></tr><tr><td>Wall-clock median (s) ↓</td><td>0.09</td><td>2.02</td><td>5.30</td><td>0.65</td></tr><tr><td>Attempts mean ↓</td><td>1.0</td><td>2.8</td><td>1.1</td><td>1.7</td></tr><tr><td>Attempts median ↓</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td rowspan="5">VERTEX (Alg. 2)</td><td>Success (%) ↑</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Wall-clock mean (s) ↓</td><td>0.11</td><td>2.82</td><td>8.70</td><td>0.86</td></tr><tr><td>Wall-clock median (s) ↓</td><td>0.10</td><td>2.82</td><td>6.31</td><td>0.76</td></tr><tr><td>Attempts mean ↓</td><td>1.1</td><td>258.6</td><td>476.6</td><td>81.0</td></tr><tr><td>Attempts median ↓</td><td>1</td><td>185.5</td><td>231.0</td><td>41.0</td></tr><tr><td rowspan="5">DEGREE (Alg. 3)</td><td>Success (%) ↑</td><td>100.0</td><td>100.0</td><td>15.0</td><td>99.1</td></tr><tr><td>Wall-clock mean (s) ↓</td><td>0.17</td><td>5.88</td><td>1465.0</td><td>6.94</td></tr><tr><td>Wall-clock median (s) ↓</td><td>0.13</td><td>4.62</td><td>~3000†</td><td>1.07</td></tr><tr><td>Attempts mean ↓</td><td>49.6</td><td>137.5</td><td>4583.5</td><td>431.5</td></tr><tr><td>Attempts median ↓</td><td>6</td><td>70.0</td><td>3552.5</td><td>15</td></tr><tr><td rowspan="3">CP (Sec. 4)</td><td>Success (%) ↑</td><td>100.0</td><td>100.0</td><td>82.5</td><td>100.0</td></tr><tr><td>Wall-clock mean (s) ↓</td><td>0.30</td><td>16.35</td><td>654.8</td><td>6.85</td></tr><tr><td>Wall-clock median (s) ↓</td><td>0.16</td><td>16.21</td><td>142.5</td><td>1.78</td></tr></table>

<sup>†</sup> DEGREE/SBM: only 6/40 targets succeed; the median is dominated by the failure budget. Successful runs alone have a median wall-clock of 162 s.

## H.5 Neural Baseline Comparison: Full Details

This appendix expands Sec. 5.2 with full descriptions of the training and inference protocols, the evaluation metric, the full results table with standard deviations (Table 6, the unabridged version of Table 3), and discussion of each (method, dataset) cell.

Inference protocol: paired vs. averaged conditioning. The original TAGG [40] substitutes the dataset-averaged landscape vector $\begin{array} { r } { \mu _ { G ^ { \prime } } = \frac { 1 } { | D _ { \mathrm { t r a i n } } | } \sum _ { G \in D _ { \mathrm { t r a i n } } } \mu _ { G } } \end{array}$ as the conditioning signal at every test sample, since the test graph $G _ { 0 }$ is unknown at inference. This makes TAGG a distribution-level method: it generates graphs whose topology resembles the training-set average, not a specific target. To align it with our setting — where the target PD is given per sample — we instead feed each test graph’s own $\mu _ { G _ { \mathrm { t e s t } } }$ at inference. Concretely, in the test loop we iterate the test data loader and pass each batch’s $\mu$ vector and n-nodes mask into the reverse-diffusion sampling routine, producing one generated graph per test reference. This change has two consequences: (i) it grants the TAGG variants the strongest possible per-target topology signal, putting them on equal footing with our CP solver for the strict PD-compliance evaluation; (ii) it yields a 1-to-1 correspondence between generated and reference graphs, enabling the paired $\bar { W } _ { 1 }$ metric. Within our paper, all three methods (CP, CONSTRUCT+TAGG, DIGRESS+TAGG) are evaluated under this paired protocol so they are pairwise comparable; absolute $\bar { W } _ { 1 }$ numbers are not directly comparable to the original TAGG paper’s reported numbers, which use the averaged $\mu _ { G ^ { \prime } }$ protocol.

Evaluation metric details. For each test reference $G _ { \mathrm { r e f } } ^ { ( i ) }$ paired with its generated graph $H ^ { ( i ) }$ , we compute persistence diagrams under the degree filtration using gudhi and pair them via the standard 1-Wasserstein distance with diagonal padding for unequal-cardinality multisets. The two homology dimensions are summed:

$$
\hat W _ { 1 } ^ { ( i ) } \ : = \ : W _ { 1 } \big ( \mathcal P _ { 0 } ( G _ { \mathrm { r e f } } ^ { ( i ) } ) , \mathcal P _ { 0 } ( H ^ { ( i ) } ) \big ) \ : + \ : W _ { 1 } \big ( \mathcal P _ { 1 } ( G _ { \mathrm { r e f } } ^ { ( i ) } ) , \mathcal P _ { 1 } ( H ^ { ( i ) } ) \big ) ,
$$

and we aggregate as $\begin{array} { r } { \bar { W } _ { 1 } = \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { i } \bar { W } _ { 1 } ^ { ( i ) } } \end{array}$ . The CP solver’s $\bar { W } _ { 1 } = 0 . 0 0 0$ entries serve as a sanity check: they confirm both that $\mathrm { C P ^ { \circ } s }$ output PD matches the target exactly and that our $W _ { 1 }$ measurement code returns zero on identical multisets (no float-precision dust). The numerical scale of $\bar { W } _ { 1 }$ varies across datasets because larger graphs have larger PDs — absolute magnitudes between datasets are not directly comparable, only relative orderings within a column.

Hardware and budgets. CP is run on a 120-core CPU slice with cp\_sat\_workers=120 and a 1,800 s per-target time limit. Neural inference runs on a single GPU (RTX 6000 Ada or A6000) per cell, single-process. Training is multi-GPU when available (typically 4 GPUs with DDP). The trained checkpoint per cell is selected as the lowest-validation-MMD-mean checkpoint among the top-3 lowest val/epoch\_NLL checkpoints, automated in the test branch of each train\_\*.py script.

Full results with standard deviations. Table 6 reports the unabridged version of Table 3, with neural variants summarised as mean ± std over 5 random seeds. CP is deterministic and reported as a single number. Standard deviations are uniformly small (1–7% of the mean), confirming that the residual gap between neural variants and CP is structural rather than due to seed noise.

Table 6: Full version of Table 3: GLOBAL (CP) vs. TAGG-augmented neural baselines, with standard deviations over 5 random seeds. CP is deterministic.
<table><tr><td colspan="3">CoMM-20</td><td>PLANAR</td><td>SBM</td><td>ENZYMES</td></tr><tr><td rowspan="3">GLOBAL (CP)</td><td> $\bar { W } _ { 1 } \downarrow$ </td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>Time/graph (s) ↓</td><td>0.30</td><td>16.35</td><td>654.8</td><td>6.85</td></tr><tr><td>Train (h) ↓</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td rowspan="3"></td><td> $\bar { W } _ { 1 } \downarrow$ </td><td> $9 . 3 5 \pm 0 . 9 7$ </td><td> $4 3 . 2 3 \pm 0 . 5 6$ </td><td> $4 3 4 . 4 8 \pm 1 7 . 5 4$ </td><td> $1 7 . 3 5 \pm 0 . 2 4$ </td></tr><tr><td>CoNSTRUCT+TAGG Time/graph (s) ↓</td><td> $0 . 8 8 \pm 0 . 0 1$ </td><td> $4 . 7 3 \pm 0 . 0 9$ </td><td> $1 7 . 1 1 \pm 0 . 0 9$ </td><td> $4 . 9 2 \pm 0 . 4 6$ </td></tr><tr><td>Train (h) ↓</td><td>9.4</td><td>24.3</td><td>59.8</td><td>61.6</td></tr><tr><td rowspan="3">DIGRESS+TAGG</td><td> $\bar { W } _ { 1 } \downarrow$ </td><td> $4 . 5 2 \pm 0 . 3 2$ </td><td> $3 8 . 7 7 \pm 1 . 0 7$ </td><td> $4 6 4 . 9 9 \pm 3 2 . 7 3$ </td><td> $1 7 . 0 6 \pm 0 . 4 4$ </td></tr><tr><td>Time/graph (s) ↓</td><td> $0 . 9 0 \pm 0 . 0 0$ </td><td> $5 . 3 0 \pm 0 . 2 8$ </td><td> $2 3 . 1 6 \pm 0 . 4 7$ </td><td> $5 . 9 4 \pm 0 . 0 5$ </td></tr><tr><td>Train (h) ↓</td><td>8.9</td><td>32.5</td><td>47.6</td><td>49.2</td></tr></table>

Table 7: Handling of failed CP runs on SBM. CP solves $3 3 / 4 0$ targets within the 1,800 s budget (the coverage reported in Table 2); its $\bar { W } _ { 1 }$ in Table 3 is over these solved targets. Restricting the neural baselines to the same 33 targets gives an equal comparison.
<table><tr><td>SBM,  $\bar { W } _ { 1 } \downarrow$ </td><td>All 40 targets</td><td>Same 33 targets (CP-solved)</td></tr><tr><td> $\mathrm { C o N S T R U C T + T A G G }$ </td><td>434.48</td><td>408.95</td></tr><tr><td> $\mathrm { D I G R E S S + T A G G }$ </td><td>464.99</td><td>454.49</td></tr><tr><td>CP (ours)</td><td>0.00 (33/40 solved)</td><td>0.00</td></tr></table>

## Per-dataset observations.

• COMM-20. The smallest and densest dataset. CP solves each target in 0.3 s — the constraint problem is small enough that solver overhead dominates — and is the only cell where CP is faster than the neural variants. DIGRESS+TAGG achieves the lowest neural $\bar { W } _ { 1 }$ (4.52), roughly halving CONSTRUCT+TAGG’s (9.35); the marginal-distribution transition seems to give a more PD-faithful denoising path on small graphs.

$\mathbf { P L A N A R } .$ . The fixed-size benchmark. CP scales to 16.35 s/graph, 3–3.5× slower than the neural variants, since the cycle structure under degree filtration (∼110 cycles, max degree 8–12) generates a moderately large slot/ancestry encoding. The two TAGG variants achieve similar $\bar { W } _ { 1 } \ ( { \sim } 4 0 )$ .

• SBM. The hardest cell. $7 / 4 0 \mathrm { C P }$ runs hit the 1,800 s timeout, contributing the worst-case 1,800 s to the mean (654.8 s). The neural variants run in 17–23 s/graph but post the largest $\bar { W } _ { 1 } \ : ( { \sim } 4 3 5 -$ 465), reflecting both the size of the PDs (max degree ∼21, $1 0 0 + H _ { 1 }$ points) and the difficulty of the topology distribution. $\mathrm { C P ^ { \prime } s } \bar { W } _ { 1 } = 0$ is computed over its 33 solved targets; Table 7 compares all methods on the same set of targets.

• ENZYMES. The most heterogeneous (real-world). CP and the neural variants are roughly tied on per-graph time (5–7 s); the neural variants achieve $\bar { W } _ { 1 } \approx 1 7$ , the lowest TAGG $\bar { W } _ { 1 }$ relative to PD scale.

## H.6 Qualitative Examples: Detailed Discussion

This appendix expands Figure 3 from Sec. 5.2 with a second test graph and a per-row breakdown of how each method’s output deviates from the reference. Both examples are drawn from the ENZYMES test split; node colour encodes degree, which under the degree filtration is also the filtration value of each vertex.

![](images/df6ba2eb75ad5af61bd4f10984aad9ded1c9d9eed53ddb808c5134240a22db43.jpg)  
Figure 5: Two qualitative examples on ENZYMES test graphs (enzymes\_87, top; enzymes\_93, bottom). Left to right: reference, GLOBAL (CP), CONSTRUCT+TAGG, DIGRESS+TAGG. Node colour encodes degree. CP matches $( n , | E | , \beta _ { 0 } , \beta _ { 1 } )$ exactly on both graphs; neural variants drift in edge count and cycle count.

Top row (enzymes\_87). The reference is a moderate-size enzyme contact graph with two connected components and a moderate cycle count. CP reproduces both invariants exactly: the same $\beta _ { 0 } .$ , the same $\beta _ { 1 }$ , and the same edge count, with vertices placed into two slots whose union recovers the target PD. The two neural variants both merge the two reference components into a single connected graph $( \beta _ { 0 } \colon 2  1 )$ and overshoot the target cycle count by $2 \mathrm { - } 3 \left( \beta _ { 1 } \right)$ . The merge is the more telling failure mode: it shows that even with the strongest per-target conditioning $( \mu _ { G _ { \mathrm { t e s t } } }$ rather than the training-set average), the diffusion denoiser cannot reliably preserve component count, because component structure is a global property that no local denoising step on its own can enforce. The excess cycles compound the issue — the PDM regulariser nudges the predicted PD toward the reference but does not commit to it.

Bottom row (enzymes\_93). A larger and denser reference, where the failure modes are amplified. Both neural variants generate 14–17 excess edges and more than double the target $\beta _ { 1 } .$ , while $\beta _ { 0 }$ stays roughly correct. This is the opposite failure pattern from the top row: rather than merging components, the denoiser over-fills the cycle structure within each component. The same structural cause is at work: the PDM loss is a soft penalty on the diagram, not a hard constraint, and when many edge configurations land at similar diffusion-loss values, the gradient-based optimiser settles in a region that is topologically off-target. CP, by contrast, returns a graph that is visibly different from the reference — node positions and edge layout do not match — but every degree-PD invariant is correct. This is the crucial separation our framework makes precise: a strict PD-compliant generator does not aim to recover the reference graph, only some graph in the PD-equivalence class.

## I Complexity and Scaling Analysis

Here we give a complexity and scaling analysis of our local sampler and global encoder when $\Phi = \deg$ . Since other filtrations only affects the constraint $\mathcal { C } _ { 1 } ,$ , the analysis outside of $\mathcal { C } _ { 1 }$ still applies to them. For the global encoding, we also ran empirical experiments to test its runtimes in practice.

Notation: We adopt the following notations in the analysis:

$n = | \mathcal { P } _ { 0 } |$ is the vertex count

• m is the edge count

$K = \operatorname* { m a x } _ { v } \deg ( v )$ is the max degree (this can be read off as the max filtration stage)

$\begin{array} { r } { c c _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { k } | \{ ( b , d ) \in \mathcal { P } _ { 0 } : b \le k < d \} | } \end{array}$ is the max concurrent alive component count.

## I.1 Local sampler

Each iteration of the sampling selects a vertex-birth or edge-birth event and updates a union-find structure, dominated by a preprocessing sort of the filtration events. This gives a single successful run the overall complexity of $\bar { O } ( ( n + m ) \log ( n + m ) )$ ) in time and $O ( n + m )$ in space. There are no bounds in general on how many runs one needs to try to get a successful outcome (Theorem 3.10), except for when we either have acyclic targets (Corollary 3.4) or stages where cycle-birth demand stays within the running graph’s cycle capacity (Proposition 3.3).

## I.2 Global CP encoding

The global encoder is summarized by the following table.

<table><tr><td>Constraint</td><td>Variables</td><td>Constraints</td></tr><tr><td> $\mathcal { C } ^ { \mathrm { d e g } }$ </td><td> $M _ { i j } \colon O ( n ^ { 2 } )$ </td><td> $O ( n )$ </td></tr><tr><td> $\mathcal { C } _ { 2 }$ </td><td> $x _ { k c v } \colon O ( K \cdot c c _ { \operatorname* { m a x } } \cdot n )$ </td><td> $O ( K n )$ </td></tr><tr><td> $\mathcal { C } _ { 3 }$ </td><td></td><td> $O \dot { ( } K \cdot \overset { \cdot } { c c } _ { \mathrm { m a x } } \cdot n ^ { 2 } )$ </td></tr><tr><td> $\mathcal { C } _ { 4 }$ </td><td> $\pi _ { k c i j } \colon O ( K \cdot c c _ { \operatorname * { m a x } } \cdot n ^ { 2 } ) ; \ell _ { k c v } , r _ { k c v } \colon O ( K \cdot c c _ { \operatorname * { m a x } } \cdot n )$ </td><td> $O ( K \cdot c c _ { \operatorname* { m a x } } \cdot n ^ { 2 } )$ </td></tr><tr><td> $\mathcal { C } _ { 5 }$ </td><td></td><td> $O ( K )$ </td></tr><tr><td> $\mathcal { C } _ { 6 }$ </td><td> $e _ { k c b } \colon O ( K \cdot c c _ { \operatorname* { m a x } } )$ </td><td> $O ( K \cdot c c _ { \operatorname* { m a x } } )$ </td></tr></table>

In total, the encoding uses $O ( K \cdot c c _ { \operatorname* { m a x } } \cdot n ^ { 2 } )$ variables and constraints. Memory is linear in this quantity (all measured instances fit within commodity RAM). Encoding size is polynomial; solving is NP-hard in general (it subsumes SAT), so no closed-form runtime bound applies. However, modern CP solvers are highly optimized in practice.

Furthermore, we also conducted empirical study of graph structural properties on CP runtime. We ran controlled scaling experiments on synthetic targets, varying one property at a time and running each through the CP solver with a 60s budget. Here we report on some empirical behaviors we observed.

1. Vertices n dominates model size $( O ( n ^ { 2 } )$ edge variables). We tested this on Erdos–Rényi graphs˝ at fixed $p = 0 . 3$ , while varying n (the number of vertices). We obtained on average 0.16s at $n = 1 0 , 0 . 8 7 \mathrm { s }$ at $n = 2 0$ , 1.42s at $n = 2 5$ . There is roughly $3 { - } 5 \times$ per doubling.

2. Density helps CP. We also observed that denser graphs merge components earlier and reduces the quantity $c c _ { \mathrm { m a x } }$ . We tested this on Erdos–Rényi graphs at fixed˝ $n = 2 5$ , while varying p. We obtained on average 6.83s at $p = 0 . 2 , 0 . 9 0 \mathrm { s }$ at $p = 0 . 5 ,$ , 0.60s at $p = 0 . 7$ . Thus, we empirically conclude that density affects cost through $c c _ { \mathrm { m a x } } .$

3. $c c _ { \mathrm { m a x } }$ is the dominant factor. We tested this on disjoint unions of k copies of $P _ { 3 }$ , isolating $c c _ { \mathrm { m a x } }$ while keeping K and cycle count fixed. We observed on average 0.03s at $c c _ { \operatorname* { m a x } } = 4 , 0 . 4 1 \mathrm { s } \mathrm { a t } 1 2$ 3.1s at 20, 12.8s at 24. The cost roughly doubles per +2 copies.

4. Cycles add a modest overhead. We tested this on disjoint unions of copies of $P _ { 3 }$ against disjoint unions of copies of $C _ { 3 }$ at matched $c c _ { \operatorname* { m a x } } . \mathrm { A t } c c _ { \operatorname* { m a x } } = 6 , 6 \times C _ { 3 }$ (6 cycles) was marginally slower than $6 \times P _ { 3 }$ (0 cycles), consistent with the extra cycle-birth constraints, though the effect was small relative to the dependence on $c c _ { \mathrm { m a x } }$ itself.

Remark I.1. The empirical behaviors above explain the behavior of the execution time on the SBM dataset (see Table 2). The SBM graphs have about 100 vertices on average, and 2-3 communities persisting across many filtration stages before merging. This drives $c c _ { \mathrm { m a x } }$ up over long ranges, hitting both the $\mathit { \Delta } _ { \overline { { n } } ^ { 2 } }$ size term in the encoding and the $c c _ { \mathrm { m a x } } – \mathrm { m }$ ultiplier in the slot/parent variable counts simultaneously. The observed SBM timeouts (30-min budget) all have $n \in [ 1 4 4 , 1 6 4 ]$ combined with high $c c _ { \mathrm { m a x } }$

## I.3 A practitioner’s guide to choosing the methods

Here we give some heuristics as to which method to choose (local sampler, global CP solver, and neural approximations) in practice.

1. Local sampler. This is preferred for sparse targets. Cycle counts grow slowly, so Proposition 3.3 (and Corollary 3.4 for acyclic targets) guarantees single-run termination.

2. Global CP solver. This is preferred for denser targets where Theorem 3.10’s infeasibility scenarios make local sampling unreliable, provided that the quantities n and $c c _ { \mathrm { m a x } }$ stay tractable. Empirically, we observe runtimes of sub-3s per graph at $n \leq 4 0 , c c _ { \mathrm { m a x } } \leq 5 ;$ and sub-30s at $n \leq 6 0 \mathrm { o r } c c _ { \mathrm { m a x } } \leq 1 0$

3. Neural approximation. In regimes where both exact methods struggle (e.g., harder SBM instances, as they are not sparse enough for local, and too large in $c c _ { \operatorname* { m a x } } \operatorname { f o r } \mathbf { C P } )$ , neural sampling is the only tractable choice, at the cost of exact PD compliance. As reported in the paper, one-off training cost is substantial, but amortized inference is fast. This pays off when many approximate graphs are needed quickly. Closing the gap in the third regime for our approach likely requires targeted algorithmic advances, or importing established techniques from the constraintprogramming literature: symmetry breaking [15, 41], conflict-driven clause learning [31], portfolio solvers [61], problem-specific decomposition [12, 42], none of which our current encoding exploits.

## J Experiments with Other Filtration Schemes

We verify the filtrations — degree, triangle, k-clique, clustering, average neighbor degree, closeness, Katz, and PageRank — for both soundness and completeness.

1. For soundness, from each generated graph, we recompute Φ with NetworkX, recompute its PDs, and check it matches the target. Over a suite of structurally diverse graphs (paths, cycles, stars, complete graphs, random trees, Erdos–Rényi, grids, barbells, wheels, Petersen) of˝ $\mathrm { s i z e } \leq 3 0$ every solution realizes the exact target, with no failures.

2. Completeness is checked by exhaustive enumeration up to isomorphism at $n \leq 6 \colon$ the CP solution set equals the full persistence-equivalence class.

Remark J.1. Overall, the solver cost depends on the encoding complexity. For degree, this is linear, whereas triangle and clustering add $O ( \stackrel { \bullet } { n } ^ { 3 } )$ auxiliaries and are costlier. On COMM-20, the success rate / median per-graph solve time are: degree 100% / 0.16 s, triangle 90% / 0.25 s, clustering 85% / 1.31 s. Scaling these richer filtrations further on remaining datasets and other benchmarks, is a promising direction for future work.

## K Diversity Within a Degree-Based PD-Equivalence Class

Here we test how diverse the graphs in a (degree-based) PD-equivalence class are. We first note that diversity is intrinsic to our methods, both of which are sound and complete over the PD-equivalence class. The local sampler produces distinct members through its random choices. The CP-SAT solver enumerates the entire class (all solutions of the encoding), and at scale random seeds return distinct exact members. Thus, in general, we would expect many diverse graphs to occur in a given input PD.

In the experiment, for each reference target we run the CP solver 20× with randomized search, and average pairwise structural distances over targets. The results are displayed below as:

<table><tr><td>Dataset</td><td># distinct</td><td>clustering</td><td>triangle</td><td>spectral</td><td>planar-rate</td><td>PD invariants (β0,  $\beta _ { 1 } ,$  deg, density)</td></tr><tr><td>COMM-20</td><td>20</td><td>0.064</td><td>0.190</td><td>0.059</td><td>0.087</td><td>0.000</td></tr><tr><td>PLANAR</td><td>18</td><td>0.020</td><td>0.212</td><td>0.020</td><td>0.000</td><td>0.000</td></tr><tr><td>SBM</td><td>14</td><td>0.026</td><td>0.099</td><td>0.021</td><td>0.000</td><td>0.000</td></tr><tr><td>ENZYMES</td><td>20</td><td>0.072</td><td>0.240</td><td>0.060</td><td>0.039</td><td>0.000</td></tr></table>

To elaborate on the columns of the table:

• # distinct is the median number of pairwise non-isomorphic graphs among the 20 solves.

• clustering is the mean pairwise absolute difference in average clustering coefficient (in [0, 1]).

• triangle is the mean pairwise relative difference in triangle count.

• spectral is the mean pairwise $\sqrt { n } -$ normalized $L _ { 2 }$ distance between sorted normalized-Laplacian spectra.

• planar-rate the fraction of class members that are planar.

• The last column (PD invariants) record the mean pairwise absolute difference of the PD-determined invariants $( \beta _ { 0 } , \beta _ { 1 }$ , average degree, density), which is exactly 0.

Note that the spectral column also serves as a surrogate for measuring graph edit distances. This is because exact GED is intractable at benchmark sizes (n up to ∼ 123), so the $\sqrt { n }$ -normalized spectral distance is the standard scalable stand-in.

The table confirms that within a degree-PD class, generations diversify in non-topological structure while their PD invariants stay exactly fixed.