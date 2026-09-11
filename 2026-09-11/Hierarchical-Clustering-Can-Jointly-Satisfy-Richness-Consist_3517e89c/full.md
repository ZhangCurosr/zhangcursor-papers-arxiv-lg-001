# Hierarchical Clustering Can Jointly Satisfy Richness, Consistency, and Scale Invariance

Daichi Kuroda   
School of Computer and Communication Sciences Ecole Polytechnique F´ed´erale de Lausanne (EPFL)<sup>´</sup> Lausanne, 1015, Switzerland   
Maximilien Dreveton   
LAMA, UMR-CNRS 8050,   
Universit´e Gustave Eifel   
5 Bd Descartes, 77454 Marne-la-Vall´ee, France Matthias Grossglauser   
School of Computer and Communication Sciences Ecole Polytechnique F´ed´erale de Lausanne (EPFL)<sup>´</sup> Lausanne, 1015, Switzerland

Patrick Thiran

School of Computer and Communication Sciences Ecole Polytechnique F´ed´erale de Lausanne (EPFL)<sup>´</sup> Lausanne, 1015, Switzerland

daichi.kuroda@epfl.ch

maximilien.dreveton@univ-eiffel.fr

matthias.grossglauser@epfl.ch

patrick.thiran@epfl.ch

## Abstract

Despite its ubiquity, clustering lacks a universally accepted definition of what is a cluster. Kleinberg’s Impossibility Theorem formalizes this dificulty by showing that no flat clustering method can simultaneously satisfy three natural axioms: scale invariance, richness, and consistency. In this paper, we ask whether this impossibility persists when the output is a hierarchy rather than a single partition. We show that, in contrast to the flat clustering setting, the hierarchical analog of these axioms are jointly satisfiable. In fact, there exist uncountably many hierarchical clustering methods satisfying these axioms, which we call admissible. We explicitly construct several admissible methods, including methods based on well-separated clusters and a non-binary version of single linkage. For certain pairs of admissible methods, the hierarchy produced by one always refines that produced by the other. This refinement relation defines a partial order on the class of admissible methods. This partially ordered set has no greatest element and contains uncountably many pairwise incompatible maximal elements, revealing substantial diversity among admissible methods. Nevertheless, this diversity is constrained: every admissible method contains a hierarchy of suficiently well-separated clusters, and every finite collection of admissible methods shares such a nontrivial common backbone.

Keywords: clustering; hierarchical clustering; axiomatic clustering; unsupervised learning; ultrametrics.

## 1 Introduction

Clustering is one of the most fundamental tasks in unsupervised learning. Given pairwise dissimilarities between data points, clustering seeks to uncover meaningful group structure without access to labels or ground truth. Yet this task is intrinsically underdetermined:

there is no universally accepted definition of a cluster. Axiomatic frameworks therefore provide a principled way to state and compare desirable properties of clustering methods.

A central result in this direction is Kleinberg’s Impossibility Theorem (Kleinberg, 2002), that shows that no flat clustering method mapping dissimilarities to partitions can simultaneously satisfy three natural axioms: scale invariance, richness, and consistency. This theorem has had a profound influence on the theory of clustering: It implies that some trade-of among the axioms, the problem formulation, and/or the clustering output, is unavoidable. A large body of subsequent work explored ways to circumvent this impossibility by weakening or modifying the axioms, or by modifying the input or output space (Ben-David and Ackerman, 2008; Zadeh and Ben-David, 2009; Strazzeri and S´anchez-Garc´ıa, 2022; Willson and Warnow, 2024). In contrast, in this paper, we ask a diferent question:

Does Kleinberg’s impossibility persist when the output is a hierarchy rather than a flat partition?

Out of the three Kleinberg axioms, consistency is arguably the most contentious for the flat clustering setting. It requires that if all within-cluster dissimilarities (with respect to the output partition) are decreased and all cross-cluster dissimilarities are increased, then the output clustering must remain unchanged. This formalizes the intuition that strengthening the evidence for a clustering should not overturn it. However, such transformations can also create an arbitrarily strong substructure within a cluster, thus revealing finer distinctions that a flat partition is unable to express. This tension lies at the heart of Kleinberg’s impossibility result. Figure 1 illustrates this phenomenon: Starting from a single cluster $C _ { 1 } \cup C _ { 2 }$ , a permissible strengthening can make each subcluster, $C _ { 1 }$ and $C _ { 2 }$ , much tighter than their union, thereby revealing two new clusters that a flat clustering method that respects the consistency axiom would be forced to ignore.

![](images/49840e23aecf0a8aeb030a8135ae32c4903070f94cedcf6992e051d048ee3d1a.jpg)  
Figure 1: Strengthening the cluster $C _ { 1 } \cup C _ { 2 }$ can create two well-separated subclusters, $C _ { 1 }$ and $C _ { 2 }$ , which a Kleinberg-consistent flat clustering method is nevertheless forced to ignore. Whereas hierarchical clustering methods can have both $C _ { 1 } \cup C _ { 2 }$ , and $C _ { 1 }$ and $C _ { 2 }$ as they are properly nested.

This observation suggests moving beyond flat clustering by considering hierarchical clustering methods. Indeed, a hierarchical clustering can preserve the desirable cluster $C _ { 1 } \cup C _ { 2 }$ while simultaneously incorporating the newly emerging subclusters $C _ { 1 }$ and $C _ { 2 } \colon$ strength ening the evidence for an existing cluster need not prevent the method from expressing finer structure within that cluster. Formally, a hierarchical clustering method is a map from dissimilarities to hierarchies, represented as laminar families of clusters. We formulate hierarchical analogs of scale invariance, richness, and consistency, and additionally impose permutation invariance as a basic symmetry requirement.

Our first main result is positive: Contrary to the flat setting, these axioms are jointly satisfiable in the hierarchical setting. In fact, there exist uncountably many admissible methods. We explicitly construct several admissible methods, including hierarchies based on well-separated clusters and a non-binary version of single linkage. In contrast, non-binary variants of other classical linkage methods (such as complete, average, Ward, centroid, and median linkage) fail to satisfy the axioms.

Beyond existence, we study the global structure of the family of admissible methods under the refinement order. This family forms a remarkably diverse partially ordered set: it has uncountable height, width, and cellularity, and contains uncountably many pairwise incompatible maximal elements. In particular, there is no greatest admissible method. Nevertheless, the axioms impose a nontrivial common structure. We prove a backbone property: every admissible method refines a hierarchy of suficiently well-separated clusters. Moreover, every finite collection of admissible methods shares such a common backbone. Thus, the axioms allow substantial diversity while still enforcing agreement on suficiently well-separated cluster structure.

We then consider a stronger requirement that is specific to hierarchical clustering. When the input dissimilarity is an ultrametric,<sup>1</sup> it already encodes a canonical hierarchical structure. It is therefore natural to require a hierarchical clustering method to recover this hierarchy exactly. We call this property exactness on ultrametrics. The resulting class of strongly admissible methods remains uncountable and retains the backbone and maximality phenomena described above. However, the additional requirement sharpens the order-theoretic structure: unlike the canonical admissible class, the strongly admissible class has a least element under refinement.

Finally, motivated by practical clustering pipelines, we study preprocessing transformations of the input dissimilarity. We derive general conditions under which composition with such a transformation preserves each axiom, and illustrate these principles with several common preprocessing operations.

Our work connects several strands of the clustering literature. It contributes to the eforts to understand and bypass Kleinberg’s impossibility theorem by modifying the axioms and/or the clustering problem formulation (Ben-David and Ackerman, 2008; Cohen-Addad et al., 2018; Willson and Warnow, 2024). It is also related to axiomatic and structura characterizations of hierarchical clustering methods (Carlsson and M´emoli, 2010; Ackerman et al., 2010; Ackerman and Ben-David, 2016), as well as to population-level axiomatizations of hierarchical clustering (Thomann et al., 2015; Arias-Castro and Coda, 2025).

In particular, hierarchical outputs have previously been shown to support positive axiomatic results: Carlsson and M´emoli (2010) characterize single linkage using an axiom system that difer substantially from Kleinberg’s framework, whereas Ackerman and Ben-David (2016) characterize linkage-based hierarchical methods using locality and a weaker consistency that only considers moving farther apart clusters that are already well-separated. Both works retain the numerical scales at which clusters merge, so their outputs are heightlabeled hierarchies (dendrograms), equivalently represented by ultrametrics. They therefore

study maps from input dissimilarities to output ultrametrics. Instead, we consider methods returning unweighted hierarchies, which retain only the nested cluster structure. Within this less structured output framework, our axioms more closely mirror Kleinberg’s original requirements while imposing fewer structural constraints. Accordingly, rather than characterizing a unique method or a prescribed algorithmic family, we study the much more diverse class of hierarchical methods satisfying these axioms. Section 6 provides a detailed comparison.

## 1.1 Definitions and Notation

Throughout the paper, X is a finite set of items with cardinality n. Moreover, as the labeling of the elements of X is irrelevant to an unsupervised task such as clustering, we implicitly assume ${ \mathcal { X } } = [ n ]$ , where $n = | \mathcal { X } |$ is finite and $[ n ] = \{ 1 , \dots , n \}$ . To avoid trivial cases, we always assume $n \geqslant 4 . ^ { 2 }$ A dissimilarity function on X is a map d: $\mathcal { X } \times \mathcal { X } \to \mathbb { R } _ { \geqslant 0 }$ such that $d ( x , x ) = 0$ and $d ( x , y ) = d ( y , x ) > 0$ for all distinct $x , y \in { \mathcal { X } }$ . We do not assume that d obeys the triangle inequality, hence d is not necessarily a distance. We denote by $\mathcal { D } ( \mathcal { X } )$ the set of all dissimilarity functions on X. We use the convention min $\varnothing = \infty$

A cluster C is a nonempty subset of $x ,$ and a partition of X is a set $\mathcal { C } = \{ C _ { 1 } , \ldots , C _ { k } \}$ of pairwise disjoint clusters whose union is X. Let ${ \bar { C } } : = { \mathcal { X } } \backslash C$ denote the complement of the cluster C with respect to X. We write $\mathcal { P } ( \mathcal { X } )$ for the set of all partitions of X.

## 1.2 Structure of the Paper

The remainder of the paper is organized as follows. In Section 2, we state the main axioms and establish the corresponding achievability results. In Section 3, we present several admissible hierarchical clustering methods. In Section 4, we study the structural properties of the set of admissible methods. In Section 5, we add exactness on ultrametrics to the axiom system and characterize preprocessing transformations that preserve the axioms. In Section $6 ,$ we discuss related work and, in Section 7, conclude the paper. Omitted proofs and technical lemmas are provided in the Appendix.

## 1.3 Use of Large Language Models (LLM)

Whereas the conceptualization of the project underlying this paper was carried out entirely by the authors, we also used GPT-5.6 Sol to improve clarity, wording, presentation, and to assist in identifying and correcting minor errors and typos in earlier drafts. However, the vast majority of the mathematical content and proofs were generated by us. The only mathematical results for which LLMs were used to develop proof techniques are Proposition 11, Lemmas 13 and 52, and Proposition $3 7 ( \mathrm { i v } )$ . We also used the model to assist with the numerical simulations in Appendix E. All text proposed by LLM was rigorously checked and edited by the authors, and we take full responsibility for this article.

## 2 From Impossibility to Achievability

In this section, we first recall Kleinberg’s impossibility theorem for flat clustering and then show that, in contrast, the hierarchical analogs of his axioms are jointly satisfiable.

## 2.1 Kleinberg’s Impossibility Theorem for Flat Clustering

We begin by recalling the axiomatic framework for flat clustering, introduced by Kleinberg (2002). A flat clustering method is a map

$$
f \colon { \mathcal { D } } ( { \mathcal { X } } )  { \mathcal { P } } ( { \mathcal { X } } )
$$

such that $f ( d )$ is a partition of X for every dissimilarity $d \in { \mathcal { D } } ( { \mathcal { X } } )$

We first introduce a transformation of dissimilarities, called strengthening<sup>3</sup> that reinforces a given clustering. Intuitively, a strengthening moves points within the same cluster closer to each other, and points in diferent clusters further apart.

Definition 1 (C-strengthening) Let $\mathcal { C } \in \mathcal { P } ( \mathcal { X } )$ and $d , d ^ { \prime } \in \mathcal { D } ( \mathcal { X } )$ be two dissimilarity functions. We say that $d ^ { \prime } \in \mathcal { D } ( \mathcal { X } )$ is a C-strengthening of d if for all clusters $C \in { \mathcal { C } }$ ，

• (Intra-cluster contraction) $d ^ { \prime } ( x , y ) \leqslant d ( x , y )$ for all $x , y \in C ;$

• (Inter-cluster expansion) $d ^ { \prime } ( x , y ) \geqslant d ( x , y )$ for all $x \in C , y \notin C$

We now formalize the properties that a clustering method should satisfy. These axioms capture natural requirements such as invariance to scaling, expressiveness, and stability under strengthening.

Definition 2 A flat clustering method f is:

• scale invariant if $f ( \beta d ) = f ( d )$ for every $d \in { \mathcal { D } } ( { \mathcal { X } } )$ and every $\beta > 0 .$

• rich if f is surjective, that is, for every $\mathcal { C } \in \mathcal { P } ( \mathcal { X } )$ , there exists $d \in { \mathcal { D } } ( { \mathcal { X } } )$ such that $f ( d ) = { \mathcal { C } } ,$ ;

• consistent $i f f ( d ) = f ( d ^ { \prime } )$ for every $d \in { \mathcal { D } } ( { \mathcal { X } } )$ and every $f ( d )$ -strengthening d<sup>1</sup> of d.

These axioms appear natural when considered individually, although the consistency axiom is sometimes viewed as contentious because it permits the creation of new clusters (see Section 1). Kleinberg established that these three requirements are not jointly satisfiable.

Theorem 3 (Kleinberg (2002)) There is no flat clustering method f that is simultaneously scale invariant, rich, and consistent.

## 2.2 Achievability of Hierarchical Clustering

We now turn to hierarchical clustering. Unlike flat clustering, which outputs a single partition of the dataset, hierarchical clustering outputs nested clusters. Formally, a hierarchy on X is a rooted tree whose leaves are the singleton sets $\{ x \}$ for $x \in \mathcal { X }$ and whose root is the full set X. We represent such a tree by its associated nested set of clusters.

Definition 4 (Hierarchy) Let $2 ^ { \mathcal { X } }$ denote the power set of X . A set $\Psi \subseteq 2 ^ { \mathcal { X } } \backslash \{ \emptyset \}$ is a hierarchy on X if it satisfies:

1. Laminarity: For all $C _ { 1 } , C _ { 2 } \in \Psi$ , we have either $C _ { 1 } \cap C _ { 2 } = \emptyset$ , or $C _ { 1 } \subseteq C _ { 2 }$ , or $C _ { 2 } \subseteq C _ { 1 }$

2. Root and leaves: $\boldsymbol { \mathcal { X } } \in \boldsymbol { \Psi }$ and $\{ x \} \in \Psi$ for every $x \in \mathcal { X }$

We denote by $\tau ( \mathcal { X } )$ the set of all hierarchies on $\mathcal { X }$

Example 1 Let $\mathcal { X } = \{ 1 , 2 , 3 , 4 \}$ . The set $\Psi = \left\{ \{ 1 , 2 , 3 , 4 \} , \{ 1 , 2 \} , \{ 3 , 4 \} , \{ 1 \} , \{ 2 \} , \{ 3 \} , \{ 4 \} \right\}$ is a hierarchy representing a binary tree with root $\{ 1 , 2 , 3 , 4 \}$ and two children t1, 2u and $\{ 3 , 4 \}$

A hierarchical clustering method is a map

$$
T \colon { \mathcal { D } } ( { \mathcal { X } } )  T ( { \mathcal { X } } )
$$

such that for every $d \in { \mathcal { D } } ( { \mathcal { X } } ) , T ( d )$ is a hierarchy on X. Kleinberg’s axioms of scale invariance, consistency, and richness admit natural analog in the hierarchical setting.

Definition 5 A hierarchical clustering method T on X is

1. scale invariant $i f T ( \beta d ) = T ( d )$ for every $\beta > 0$ and every $d \in { \mathcal { D } } ( { \mathcal { X } } )$ q;

2. partition rich if for every partition $\mathcal { C } \in \mathcal { P } ( \mathcal { X } )$ , there exists $d \in { \mathcal { D } } ( { \mathcal { X } } )$ such that ${ \mathcal { C } } \subseteq T ( d ) .$ ;

3. partition consistent if for every d P $\mathcal { D } ( \mathcal { X } )$ and every partition $\mathcal { C } \in \mathcal { P } ( \mathcal { X } )$ such that ${ \mathcal { C } } \subseteq T ( d )$ , we have $\mathcal { C } \subseteq T ( d ^ { \prime } )$ for every C-strengthening d<sup>1</sup> of $d ^ { 4 }$ ;

4. permutation invariant i $f T ( d _ { \phi } ) ~ = ~ \phi \cdot T ( d )$ for every permutation ϕ of X, where $d _ { \phi } ( x , y ) : = d ( \phi ^ { - 1 } ( x ) , \phi ^ { - 1 } ( y ) )$ and $\phi \cdot T ( d ) = \{ \{ \phi ( x ) \colon x \in C \} \colon C \in T ( d ) \}$

A hierarchical clustering method T is admissible if it satisfies all four axioms.

Definition 5 is a direct extension of Kleinberg’s original axioms to the hierarchical setting. The flat clustering method f is replaced by a hierarchical clustering method T, and the three axioms of Definition 2 are reformulated accordingly. We also explicitly include permutation invariance, requiring that the output depends only on the underlying dissimilarity structure and not on the labeling of the points. In Kleinberg’s original framework, this requirement is implicit, as clusterings are defined directly on sets rather than on a labeled representation.

We refer to the set of four axioms in Definition 5 as the canonical axiom system $\alpha _ { \mathrm { a d m } } .$ and we call admissible the hierarchical clustering methods that satisfy $\alpha _ { \mathrm { a d m } }$ . There are, however, other reasonable axiomatizations of hierarchical clustering. In particular, there is a well-known correspondence between ultrametrics and hierarchies, which is not addressed by the axioms of $\alpha _ { \mathrm { a d m } }$ because it has no direct counterpart in flat clustering. In Section 5.1, we study the consequences of such an additional requirement. Moreover, further alternative axioms are discussed in Appendix A, where we show that most of the results established for this canonical axiom system in fact remain valid under these alternative formulations. All the axioms and their alternatives are cataloged in Table 1.

The first main result of this paper is that the hierarchical clustering problem does admit (many) methods that satisfy $\alpha _ { \mathrm { a d m } }$ , in contrast with flat clustering.

Theorem 6 There exist uncountably many admissible methods.

The proof of Theorem 6 is constructive. In the next section, we give explicit examples of admissible hierarchical clustering methods.

## 3 Explicit Construction of Admissible Methods

In this section, we construct admissible methods from three complementary perspectives. We first examine the admissibility of some standard linkage methods, as they are the most widely used class of hierarchical clustering methods. We then introduce two families of methods defined through explicit separation conditions and finally study Bryant-Berry stable clusters.

These two parameterized families play a central role in the analysis of the class of admissible hierarchical clustering methods made in Section 4: the separation methods will be useful to order the admissible methods based on the cluster structures they produce, whereas the Bryant-Berry cluster construction will be useful for showing the existence of many admissible methods that are fundamentally diferent (more precisely, incompatible).

## 3.1 Linkage Methods

Linkage-based methods form a classical and widely used class of hierarchical clustering algorithms. They are typically defined as greedy agglomerative procedures: Starting from the singleton partition, clusters are iteratively merged according to a prescribed inter-cluster dissimilarity rule, such as single, complete, average, or Ward linkage. We refer to $\mathrm { A p \mathrm { - } }$ pendix C.2 for formal definitions of these classic linkage rules. Standard formulations of linkage algorithms merge exactly two clusters at each step, and hence always produce binary hierarchies.<sup>5</sup>

We first observe that this restriction to pairwise merges and hence to binary hierarchies is incompatible with permutation invariance, regardless of the specific linkage rule employed.

Lemma 7 Any hierarchical clustering method that is constrained to always produce a binary hierarchy is not permutation invariant.

Proof Given a dissimilarity d, we say that two elements $x \not = y$ have pairwise identical dissimilarity profiles if x and y are equidistant from every other element of X $( \mathrm { i } . \mathrm { e } . , \ \forall z \ \in$ ${ \mathcal { X } } \backslash \{ x , y \} : \quad d ( x , z ) = d ( y , z ) )$ . Let $d$ be a dissimilarity on $\mathcal { X } _ { : }$ , and suppose that $\mathcal { X }$ has three distinct elements $x _ { 1 }$ , x<sub>2</sub> and $x _ { 3 } .$ , such that any two have pairwise identical profiles. $\mathrm { A s }$ the elements of $S : = \{ x _ { 1 } , x _ { 2 } , x _ { 3 } \}$ have pairwise identical dissimilarity profiles, every permutation $\sigma$ of S that fixes ${ \mathcal { X } } \backslash S$ preserves $d .$ Permutation invariance therefore requires that $C \in T ( d ) \Longrightarrow \sigma ( C ) \in T ( d )$

Let $C \in T ( d )$ be non-singleton and suppose that $C \cap S \neq \emptyset$ . If $C \cap S = \{ x _ { i } \}$ , then $C = U \cup \{ x _ { i } \}$ for some nonempty $U ,$ , and exchanging $x _ { i }$ with another element of $S$ produces a cluster $U \cup \{ x _ { j } \} \in T ( d )$ incompatible with C. If $| C \cap S | = 2$ , exchanging one of these two elements with the third likewise produces a cluster incompatible with C. Both cases contradict laminarity. Hence every non-singleton cluster having a non-empty intersection with S contains all elements of $S ,$ and thus $T$ must be non-binary.

Motivated by Lemma $^ { 7 , }$ we consider variants of linkage methods whose output is not constrained to binary hierarchies. In these variants, whenever multiple pairs of clusters attain the same minimal inter-cluster dissimilarity, all clusters involved in this tie are merged together simultaneously. This produces hierarchies that are not necessarily binary, but that respect the symmetries of the input dissimilarity function. Among the classic linkage rules mentioned earlier, single linkage is the only one that produces admissible hierarchies, as established by the following proposition.

Proposition 8 (Admissibility of Linkage Methods) The non-binary variant of the single linkage method $T _ { \mathrm { S L } }$ is admissible. In contrast, non-binary variants of the complete, average, Ward, centroid, and median linkage violate at least one of the axioms in Definition ${ 5 . }$

Proof We prove the admissibility of $T _ { \mathrm { S L } }$ in Appendix C.1, and the non-admissibility of other linkage methods in Appendix C.2. ■

## 3.2 Separation Methods

We now introduce two classes of admissible hierarchical clustering methods, based on explicit separation conditions between within-cluster and cross-cluster dissimilarities. These methods declare a nonempty subset $C \subseteq { \mathcal { X } }$ to be a cluster if the internal dissimilarities within $C$ are suficiently small compared to the dissimilarities between elements of $C$ and of its complement $\bar { C } .$ . The resulting hierarchies are defined directly from $d ,$ without resorting to an agglomerative procedure. We quantify the separability of a cluster C by the ratios

$$
{ \frac { d ( x _ { 1 } , y ) } { d ( x _ { 2 } , z ) } } , \qquad x _ { 1 } , x _ { 2 } , y \in C , \ z \not \in C ,
$$

and we consider below two families of methods resulting from thresholding these ratios. The two families difer in the choice of the reference points $x _ { 1 } , x _ { 2 } \in C \colon$ : the first family enforces a worst-case comparison globally for any pair $x _ { 1 } , x _ { 2 } \in C$ , whereas the second family enforces this comparison locally by setting $x _ { 1 } = x _ { 2 }$

Definition 9 For any finite set $\mathcal { X }$ , define the global and local separabilities with respect to $d \in { \mathcal { D } } ( { \mathcal { X } } )$ and a proper subset $C \subsetneq { \mathcal { X } }$ with $| C | \geqslant 2$ , respectively by

$$
\varrho ( d , C ) : = \operatorname* { m a x } _ { x _ { 1 } , x _ { 2 } , y \in C , z \in \bar { C } } \frac { d ( x _ { 1 } , y ) } { d ( x _ { 2 } , z ) } a n d \tau ( d , C ) : = \operatorname* { m a x } _ { x , y \in C , z \in \bar { C } } \frac { d ( x , y ) } { d ( x , z ) } .
$$

We call separation margin sequence any sequence $\pmb { \eta } = ( \eta _ { m } ) _ { 1 \leqslant m \leqslant n - 2 }$ such that $0 < \eta _ { s } \leqslant 1$ for every s. For such sequence η, we define the set of $\pmb { \eta } - g l o b a l l y$ and η-locally separated clusters on $x ,$ respectively as

$$
T _ { \mathrm { g l o b } } ^ { \eta } ( d ) : = \left\{ C \subset \mathcal { X } : | C | \geqslant 2 , \varrho ( d , C ) < \eta _ { | C | - 1 } \right\} \cup \left\{ \{ x \} : x \in \mathcal { X } \right\} \cup \{ \mathcal { X } \} ;
$$

$$
T _ { \mathrm { l o c } } ^ { \eta } ( d ) : = \left\{ C \subseteq \mathcal { X } : | C | \geqslant 2 , \tau ( d , C ) < \eta _ { | C | - 1 } \right\} \cup \{ \{ x \} : x \in \mathcal { X } \} \cup \{ \mathcal { X } \} .
$$

The global (resp., local) separability $\varrho ( d , C ) \ ( \mathrm { r e s p . } , \ \tau ( d , C ) )$ measures the sensitivity of the global (resp., local) separability of the cluster C to the dissimilarity function d. The following proposition establishes that the maps $T _ { \mathrm { g l o b } } ^ { \eta } \colon d \mapsto T _ { \mathrm { g l o b } } ^ { \eta } ( d )$ and $T _ { \mathrm { l o c } } ^ { \eta } \colon d \mapsto T _ { \mathrm { l o c } } ^ { \eta } ( d )$ are admissible hierarchical clustering methods.

Proposition 10 For any separation margin sequence η, $T _ { \mathrm { g l o b } } ^ { \eta }$ and $T _ { \mathrm { l o c } } ^ { \eta }$ are admissible. Furthermore, $i f \eta \neq \eta ^ { \prime }$ , then $T _ { \mathrm { g l o b } } ^ { \eta } \neq T _ { \mathrm { g l o b } } ^ { \eta ^ { \prime } }$ and $T _ { \mathrm { l o c } } ^ { \eta } \neq T _ { \mathrm { l o c } } ^ { \eta ^ { \prime } }$

When $\pmb { \eta } = \mathbf { 1 } = \left( 1 , \cdots , 1 \right)$ , we simply write $T _ { \mathrm { g l o b } } , T _ { \mathrm { l o c } }$ instead of $T _ { \mathrm { g l o b } } ^ { \mathbf { 1 } } , T _ { \mathrm { l o c } } ^ { \mathbf { 1 } }$ . The clusters belonging to $T _ { \mathrm { g l o b } } ( d )$ and $T _ { \mathrm { l o c } } ( d )$ are the sets of points that are strictly closer to any point in the set than to any point outside the set. Among these methods, the most studied one is $T _ { \mathrm { l o c } } ( d )$ , which is called the Apresjan hierarchy (Apresjan, 1966). The clusters belonging to $T _ { \mathrm { l o c } } ( d )$ have been studied independently by diferent authors, and are referred to as Apresjan clusters, K-clumps, strong clusters, nice clusters, or valid clusters in the literature (Ackerman and Dasgupta, 2014; Diatta and Fichet, 1994; Balcan et al., 2008; Bryant and Berry, 2001).

The separation margins η control the sensitivity to separability $\varrho ( d , C )$ and $\tau ( d , C )$ The higher the values of η are, the more sensitive these methods become to close-to-one separabilities, but at the price of being too prone to noise. For example, consider $\mathcal { X } =$ $\{ 1 , 2 , 3 , 4 \}$ and $d \in { \mathcal { D } } ( { \mathcal { X } } )$ such that every pairwise dissimilarity is exactly equal to 1. For this d, because no nontrivial subset satisfies the global separation condition, $T _ { \mathrm { g l o b } }$ identifies only the root and leaves as clusters. However, if $d ( 1 , 2 )$ is slightly decreased to be equal to $1 - \varepsilon .$ , even for an extremely small $\varepsilon > 0$ such as $\varepsilon = 1 0 ^ { - 1 0 }$ , then t1, 2u belongs to the hierarchy returned by $T _ { \mathrm { g l o b } }$ . But, setting $\pmb { \eta } = \left( 1 - \boldsymbol { \varepsilon } ^ { \prime } \right) \cdot \mathbf { 1 }$ with $\varepsilon ^ { \prime } > \varepsilon .$ , the cluster t1, 2u does not belong to the hierarchy $T _ { \mathrm { g l o b } } ^ { \eta }$ , as $T _ { \mathrm { g l o b } } ^ { \eta }$ outputs only the root and leaves.

Example 2 Let $\mathcal { X } = \{ 1 , 2 , 3 , 4 , 5 \}$ and

$$
d = { \left( \begin{array} { l l l l l } { 0 } & { 1 } & { 2 } & { 4 } & { 3 } \\ { 1 } & { 0 } & { 4 } & { 5 } & { 6 } \\ { 2 } & { 4 } & { 0 } & { 5 } & { 6 } \\ { 4 } & { 5 } & { 5 } & { 0 } & { 7 } \\ { 3 } & { 6 } & { 6 } & { 7 } & { 0 } \end{array} \right) } .\tag{1}
$$

Then, as shown in Figure 2, the outputs of $T _ { \mathrm { S L } } , T _ { \mathrm { l o c } } , T _ { \mathrm { g l o b } }$ , and $T _ { \mathrm { g l o b } } ^ { \eta }$ with $\eta \ : = \ : \frac { 1 } { 2 } { \bf 1 }$ are $T _ { \mathrm { S L } } ( d ) = \{ \{ 1 , 2 , 3 , 5 \} , \{ 1 , 2 , 3 \} , \{ 1 , 2 \} \} \cup \{ \mathcal { X } \} \cup \{ \{ x \} : x \in \mathcal { X } \} ; T _ { \mathrm { l o c } } ( d ) ^ { > \infty } \{ \{ 1 , 2 , 3 \} , \{ 1 , 2 \} \} \cup \{ 1 , 2 \} \cup \{ 1 , 2 \} \cup \{ 1 , 2 \} .$ $\{ \mathcal { X } \} \cup \{ \{ x \} : x \in \mathcal { X } \} ; T _ { \mathrm { g l o b } } ( d ) = \{ \{ 1 , 2 \} \} \cup \{ \mathcal { X } \} \cup \{ \{ x \} : x \in \mathcal { X } \} ; T _ { \mathrm { g l o b } } ^ { \eta } ( d ) = \{ \{ \mathcal { X } \} \cup \{ \{ x \} : x \in \mathcal { X } \} : x \in \mathcal { X } \} .$ X u. g

## 3.3 Bryant-Berry Stable Clusters

Bryant and Berry (2001) introduced the notion of stable clusters, a combinatorial criterion that is strictly weaker than the separation conditions in Definition 9, yet still yields sets of clusters that form hierarchies. In the following, we recall their construction, by translating the similarity framework in Bryant and Berry (2001) to the dissimilarity-based framework used in this paper. We then demonstrate that the induced hierarchical clustering method is admissible.

![](images/b931adb6ba33e8546ceea006d1a991c2a5b33c8e351494f1d20b5f973511e3e9.jpg)  
(a) T<sub>SL</sub>pdq

![](images/c8f3d6e0937170b1724221ecf53731b2d1d57e3359e5355f2839e54cc3ac418c.jpg)  
(b) $T _ { \mathrm { l o c } } ( d )$

![](images/5db3ab374df4e7acaae5be52e0f6cc969a7010da5ad29a932f9268c25c905439.jpg)  
(c) $T _ { \mathrm { g l o b } } ( d )$

![](images/1bc049358269b3e39fa04ef3f1f6d715e6cae89fd8ff8a198afb1e81e21620ba.jpg)  
(d) $T _ { \mathrm { g l o b } } ^ { \eta } ( d )$  
Figure 2: Output of $T _ { \mathrm { S L } } , T _ { \mathrm { l o c } } , T _ { \mathrm { g l o b } }$ , and $T _ { \mathrm { g l o b } } ^ { \eta }$ with $\begin{array} { r } { \eta = \frac { 1 } { 2 } \mathbf { 1 } } \end{array}$ on the dissimilarity d given in Equation (1).

For $x , y , z \in \mathcal { X } .$ , the Bryant–Berry isolation weight of the pair $( x , y )$ with respect to $z$ is

$$
\rho _ { d } ( x y \mid z ) : = \operatorname* { m i n } \{ d ( x , z ) , d ( y , z ) \} - d ( x , y ) .
$$

Intuitively, $\rho _ { d } ( x y \mid z ) > 0$ means that the point z is farther from both x and y than they are from each other. For disjoint nonempty subsets U, $V , Z \subseteq { \mathcal { X } }$ , we define the average Bryant–Berry isolation weight by

$$
\overline { { \rho } } _ { d } ( U V \mid Z ) : = \frac { 1 } { | U | | V | | Z | } \sum _ { u \in U , v \in V , z \in Z } \rho _ { d } ( u v \mid z ) ,
$$

and the stable-cluster index of a proper subset $C \subseteq { \mathcal { X } }$ with $| C | \geqslant 2$ by

$$
\iota ^ { d } ( C ) : = \operatorname* { m i n } _ { \stackrel { U , V \neq \emptyset , ~ U \cap V = \emptyset } { U \cup V = C } } \overline { { \rho } } _ { d } ( U V \mid Z ) .
$$

A subset C is stable if $\iota ^ { d } ( C ) > 0 ,$ i.e., if for every bipartition $( U , V )$ of C and every nonempty set $Z$ external to $C ,$ the average isolation weight is strictly positive. The Bryant-Berry method is then<sup>6</sup>

$$
T _ { \mathrm { s t a b l e } } ( d ) : = \{ \mathcal { X } \} \cup \{ \{ x \} : x \in \mathcal { X } \} \cup \{ C \subseteq \mathcal { X } : | C | \geqslant 2 , \iota ^ { d } ( C ) > 0 \} .
$$

Bryant and Berry (2001) show that the set of stable clusters is laminar, so $T _ { \mathrm { s t a b l e } } ( d )$ is indeed a hierarchy. Computing $\iota ^ { d } ( C )$ requires minimizing over all bipartitions of C and all external witness sets, and Bryant and Berry (2001) show that deciding stability is NPhard in general. They further establish that the stable-cluster family is contained in the average-linkage hierarchy, thus providing a tractable outer approximation of their method.

We also consider variants of this method resulting from preprocessing the dissimilarity function with power transformations. This yields the following parameterized family of admissible methods, which turns out to be useful for the analysis in the next section. As shown in the next proposition, the Bryant–Berry method and its power-transformation variants satisfy the admissibility axioms.

Proposition 11 $T _ { \mathrm { s t a b l e } }$ is admissible. Furthermore, for every $p > 0$ , the composed method $T _ { \mathrm { s t a b l e } } ^ { ( p ) } : = T _ { \mathrm { s t a b l e } } \circ \mu _ { \mathrm { p o w e r } } ^ { p }$ is admissible, where $\mu _ { \mathrm { p o w e r } } ^ { p } ( d ) ( x , y ) = ( d ( x , y ) ) ^ { p }$ for every $x , y$

## 4 The Structure of the Set of Admissible Methods

This section studies the admissible class as a whole, by focusing on structural properties shared across all of its members rather than on any individual method. We establish three main insights, each developed in its own subsection. First, in Section 4.1, we highlight the diversity of the admissible class: there exist uncountably many pairwise incompatible admissible methods, and both the width and the height of the class (suitably defined) are uncountable. Second, we show in Section 4.2 that despite their diversity, all admissible methods agree on a nontrivial common core of conservative cluster structures. Finally, in Section 4.3, we study maximal admissible methods and show that there are uncountably many of them.

## 4.1 Comparing Admissible Methods: Refinement and Incompatibility

Section 3 introduced several admissible methods, including a number of parameterized variants. This naturally raises the question of how restrictive the admissibility axioms are, and to what extent they constrain the class of admissible methods.

## 4.1.1 Refinement: A Partial Order on Hierarchical Clustering Methods

Addressing the questions above requires comparing hierarchical clustering methods. We begin by formalizing the intuition that some methods always return more fine-grained cluster structures than others.

Definition 12 Let $T _ { 1 }$ and $T _ { 2 }$ be two hierarchical clustering methods. We say $T _ { 2 }$ refines $T _ { 1 }$ (or equivalently, that $T _ { 2 }$ is finer than $T _ { 1 } ) ,$ and denote it by $T _ { 1 } \subseteq T _ { 2 }$ , if

$$
T _ { 1 } ( d ) \subseteq T _ { 2 } ( d ) \quad f o r \ a l l \ d \in { \mathcal { D } } ( { \mathcal { X } } ) .
$$

The binary relation $\sqsubseteq$ is a partial order between hierarchical clustering methods. We denote by $\mathcal { H } ( \mathcal { X } )$ and $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ the sets of hierarchical clustering methods on X and hierarchical clustering methods on X that satisfy $\alpha _ { \mathrm { a d m } }$ , respectively (thus $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } ) \subsetneq \mathcal { H } ( \mathcal { X } ) \rangle$ . Then, $( { \mathcal { H } } ( { \mathcal { X } } ) , \subseteq )$ is a partially ordered set, and so is $( { \mathcal { H } } _ { \alpha _ { \mathrm { a d m } } } ( { \mathcal { X } } ) , \subseteq )$

Example 3 (Known and immediate refinement relations) Among the admissible methods introduced in Section ${ \mathcal { B } } ,$ we have the following refinement relationships:

(i) $T _ { \mathrm { l o c } } \subseteq T _ { \mathrm { S L } }$ and $T _ { \mathrm { l o c } } \equiv T _ { \mathrm { s t a b l e } }$

(ii) $T _ { \mathrm { g l o b } } ^ { \eta } \subseteq T _ { \mathrm { l o c } } ^ { \eta }$ for any η;

(iii) $T _ { \mathrm { g l o b } } ^ { \eta ^ { \prime } } \subseteq T _ { \mathrm { g l o b } } ^ { \eta }$ and $T _ { \mathrm { l o c } } ^ { \eta ^ { \prime } } \subseteq T _ { \mathrm { l o c } } ^ { \eta }$ for any $\eta ^ { \prime } \leqslant \eta$ (where inequalities are element-wise, $i . e .$ $\eta _ { s } ^ { \gamma } \leqslant \eta _ { s } \ f o r \ a l l \ s )$

Point (i) is established by Bryant and Berry (2001). $( T _ { \mathrm { l o c } } \subseteq T _ { \mathrm { S L } }$ has also been re-proven by Balcan et al. (2008) and Dreveton et al. $( 2 0 2 5 ) . )$ Point (ii) holds because $\tau ( d , C ) \leqslant$ $\varrho ( d , C )$ for every $C \subseteq { \mathcal { X } }$ and $d \in { \mathcal { D } } ( { \mathcal { X } } )$ . For point (iii), if $\eta ^ { \prime } \leqslant \eta ,$ then $\varrho ( d , C ) < \eta _ { | C | - 1 } ^ { \prime }$ implies $\varrho ( d , C ) < \eta _ { | C | - 1 }$ , and likewise $f o r \tau ( d , C )$ . Hence $T _ { \mathrm { g l o b } } ^ { \eta ^ { \prime } } \subseteq T _ { \mathrm { g l o b } } ^ { \eta }$ and $T _ { \mathrm { l o c } } ^ { \eta ^ { \prime } } \subseteq T _ { \mathrm { l o c } } ^ { \eta }$

## 4.1.2 Order-theoretic Terminology

Before analyzing $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ as a subset of the partially ordered set (poset) $( { \mathcal { H } } ( { \mathcal { X } } ) , \subseteq )$ , we first briefly recall the standard order-theoretic terminology.

In a poset $( P , \subseteq )$ , and a subset $S \subseteq P$ , an element $g \in S$ is a greatest element of S if $s \subseteq g$ for all $s \in S$ . An element $m \in S$ is a maximal element of S if there is no $s \in S$ such that $m \subsetneq s . \mathrm { ~ A ~ }$ poset may contain multiple maximal elements but at most one greatest element; if a greatest element exists, it is necessarily the unique maximal element. An element $a \in P$ is a lower bound of S if for every element $b \in S , a \subseteq b$ . Similarly, $a \in P$ is an upper bound of S if for every element $b \in S , b \subseteq a$ . The infimum (i.e., the greatest lower bound) of the subset S is the lower bound $a \in P$ of S such that $b \subseteq a$ for any lower bound b of S in P. The supremum $( \mathrm { i . e . }$ , the least upper bound or join) of the subset S is the upper bound $a \in P$ of S satisfying $a \subseteq b$ for any upper bound b of S in P.

Furthermore, two elements $x , y \in P$ are comparable if either $x \subseteq y$ or $y \subseteq x ;$ otherwise, they are incomparable. A chain is a subset $C \subseteq P$ in which every pair of distinct elements is comparable. The height of the poset P is the supremum of the cardinalities of all chains.

A nonempty subset $S \subseteq P$ is upward directed if, for every $x , y \in S$ , there exists $z \in S$ such that $x \subseteq z$ and $y \subseteq z$ . Equivalently, every finite nonempty subset of S has an upper bound that belongs to $S _ { ☉ }$ . Every chain is upward directed, but an upward-directed family need not be totally ordered.

We say that two elements $x , y \in P$ are upward compatible if they share a common upper bound in $P ;$ otherwise, they are said to be (upward) incompatible. When analyzing the structure of subsets within $P ,$ these concepts of comparability and compatibility give rise to diferent types of antichains. A standard (or weak) antichain is a subset $A \subseteq P$ in which no two distinct elements are comparable. The quantity used to capture the maximal size of these mutually incomparable sets is the width of the poset, defined as the supremum of the cardinalities of all standard antichains in P.

Besides incomparability, a stricter structural condition requires subsets to be mutually incompatible. A strong upward antichain is a subset $A \subseteq P$ where no two distinct elements are upward compatible (for any $x \neq y \in A$ , there is ${ \mathrm { n o ~ } } z \in P$ such that $x \subseteq z { \mathrm { ~ a n d ~ } } y \subseteq z$ Analogous to how width measures the maximum size of standard antichains, the cellularity of a poset is defined as the supremum of the cardinalities of all its strong upward antichains.

## 4.1.3 Diversity of Admissible Methods

Lemma 13 The family of admissible methods $\{  T _ { \mathrm { s t a b l e } } ^ { ( p ) } : p > 0 \} \subsetneq \mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ is pairwise incompatible and thus forms a strong upward antichain.

Lemma 13 reveals that $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ is genuinely diverse, as it shows that no single admissible method refines all the others: not only do admissible methods difer, but uncountably many pairs admit no common refinement (or upper bound in the order-theoretic terminology) even in the larger class $\mathcal { H } ( \mathcal { X } )$ . In particular, $( { \mathcal { H } } _ { \alpha _ { \mathrm { a d m } } } ( { \boldsymbol { \mathcal { X } } } ) , \subseteq )$ has no greatest element: there is no universal admissible method that simultaneously refines all others. We obtain the following theorem.

Theorem 14 The height, width, and cellularity of $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ are uncountable.

Proof Uncountable height follows immediately from the family of admissible methods $\left. T _ { \mathrm { l o c } } ^ { \eta } : \eta \in ( 0 , 1 ] , \eta = \eta \cdot \mathbf { 1 } \right.$ (, which is itself uncountable. Uncountable width and cellularity hold because the set $\left\{ T _ { \mathrm { s t a b l e } } ^ { ( p ) } : p > 0 \right\}$ is an uncountable set of admissible methods whose elements are pairwise incompatible by Lemma 13, and therefore mutually incomparable. Finally, we show that $T _ { \mathrm { S L } }$ and $T _ { \mathrm { s t a b l e } } ^ { ( p ) }$ with any $p > 0$ are incompatible in Appendix D.1.

## 4.2 Uniformity: A Well-separated Backbone

Section 4.1.3 established the diversity of the set of admissible methods $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ . This raises another natural question: do admissible methods nevertheless share some common cluster structure? As a motivating example, consider $T _ { \mathrm { S L } }$ and $T _ { \mathrm { s t a b l e } }$ . Both refine $T _ { \mathrm { l o c } } \mathrm { : }$ as noted in Section 4.1, $T _ { \mathrm { l o c } } \subseteq T _ { \mathrm { S L } }$ and $T _ { \mathrm { l o c } } \equiv T _ { \mathrm { s t a b l e } }$ . Hence, despite being incompatible, $T _ { \mathrm { S L } }$ and $T _ { \mathrm { s t a b l e } }$ share $T _ { \mathrm { l o c } }$ as common cluster structure. The following theorem shows that this is not a coincidence specific to $T _ { \mathrm { S I } }$ and $T _ { \mathrm { s t a b l e } }$ : every method in $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ refines a globally separated hierarchy $T _ { \mathrm { g l o b } } ^ { \eta }$ for some margin sequence η.

Theorem 15 (Backbone hierarchy) For any $T \in \mathcal H _ { \alpha _ { \mathrm { a d m } } } ( \mathcal X )$ , there exists a separation margin sequence η (that may depend on T and on $| \mathcal { X } | \big )$ such that $T _ { \mathrm { g l o b } } ^ { \eta } \subseteq T$

Although admissible methods may disagree on individual clusters, Theorem 15 reveals a structural common ground: every admissible method refines a globally η-separated hierarchy $T _ { \mathrm { g l o b } } ^ { \eta } .$

Whereas the separation margin sequence η is method-dependent, the following corollary shows that any finite collection of admissible methods admits a common backbone hierarchy, obtained by taking the pointwise minimum of their individual margin sequences.

Corollary 16 Let $T _ { 1 } , \cdots , T _ { m }$ be methods in $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ , with m finite. There exists a separation margin sequence η such that $T _ { \mathrm { g l o b } } ^ { \eta } \subseteq T _ { \ell }$ for all $\ell \in [ m ]$

Proof For each $\ell \in \ \lceil m \rceil$ , Theorem 15 provides a separation margin sequence $\eta ^ { ( \ell ) } =$ $( \eta _ { s } ^ { ( \ell ) } ) _ { 1 \leqslant s \leqslant n - 2 }$ such that $T _ { \mathrm { g l o b } } ^ { \eta ^ { ( \ell ) } } \subseteq T _ { \ell }$ . Define $\pmb { \eta } = \ ( \eta _ { s } ) _ { 1 \leqslant s \leqslant n - 2 }$ by $\begin{array} { r } { \eta _ { s } : = \operatorname* { m i n } _ { \ell \in [ m ] } \eta _ { s } ^ { ( \ell ) } } \end{array}$ , which satisfies $\eta _ { s } \in ( 0 , 1 ]$ for every s since the minimum is taken over a finite set of positive values. By construction, $\eta _ { s } \leqslant \eta _ { s } ^ { ( \ell ) }$ for all $\ell \in [ m ]$ and all $s \geqslant 1$ , so $T _ { \mathrm { g l o b } } ^ { \eta } \subseteq T _ { \mathrm { g l o b } } ^ { \eta ^ { ( \ell ) } } \subseteq T _ { \ell }$ ■

The preceding corollary shows that every finite family of admissible methods has an admissible common lower bound. This conclusion does not, however, extend to the entire admissible class. To make this precise, define the trivial method $T _ { \mathrm { t r i v } } \in { \mathcal { H } } ( \mathcal { X } )$ by

$$
T _ { \mathrm { t r i v } } ( d ) : = \{ \mathcal { X } \} \cup \{ \{ x \} : x \in \mathcal { X } \} \qquad \forall d \in \mathcal { D } ( \mathcal { X } ) .\tag{2}
$$

In other words, $T _ { \mathrm { t r i v } }$ is the method that always returns only the root and the singleton leaves. The following proposition shows that $T _ { \mathrm { t r i v } }$ is the infimum of $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ . Moreover, $T _ { \mathrm { t r i v } }$ does not satisfy partition richness and hence is not admissible. Hence Proposition 17 also implies that $( { \mathcal { H } } _ { \alpha _ { \mathrm { a d m } } } ( { \mathcal { X } } ) , \subseteq )$ has no least element.

Proposition 17 The infimum of $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ , computed in the poset $( { \mathcal { H } } ( { \mathcal { X } } ) , { \underline { { \subseteq } } } )$ , is $T _ { \mathrm { t r i v } }$

Proof As every hierarchy on X contains the root and all singleton clusters, $T _ { \mathrm { t r i v } } \subseteq T$ for every $T \in { \mathcal { H } } ( { \mathcal { X } } )$ . In particular, $T _ { \mathrm { t r i v } }$ is a lower bound of $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$

We now show that no nontrivial proper cluster belongs to every admissible method. Fix $d \in { \mathcal { D } } ( { \mathcal { X } } )$ and a nontrivial proper subset $C \subsetneq { \mathcal { X } }$ . Since $\varrho ( d , C ) > 0$ , we may choose a separation margin sequence η such that $0 < \eta _ { | C | - 1 } \leqslant \varrho ( d , C )$ . Then, by definition, $C \notin T _ { \mathrm { g l o b } } ^ { \eta } ( d )$ whereas $T _ { \mathrm { g l o b } } ^ { \eta }$ is admissible. Hence $C$ does not belong to the intersection of all admissible methods. Because this holds for every $d \in { \mathcal { D } } ( { \mathcal { X } } )$ and every nontrivial proper subset $C \subsetneq { \mathcal { X } }$ and because the root and singleton clusters belong to every hierarchy, for every d $\ b { l } \in \ b { \mathcal { D } } ( \ b { \mathcal { X } } )$ $\bigcap _ { T \in { \mathcal { H } } _ { \alpha _ { \mathrm { a d m } } } ( \chi ) } T ( d ) = T _ { \mathrm { t r i v } } ( d )$ . Therefore $T _ { \mathrm { t r i v } }$ is the infimum of $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ in $\mathcal { H } ( \mathcal { X } )$

Theorem 15 carries an additional, perhaps unexpected, implication: the partition richness axiom can be upgraded to a seemingly stronger one, hierarchical richness. Recall that partition richness only requires that for every partition ${ \mathcal { C } } ,$ there exists a dissimilarity d such that ${ \mathcal { C } } \subseteq T ( d )$ . Combined with the other axioms of $\alpha _ { \mathrm { a d m } }$ , it in fact implies that $T$ satisfies hierarchical richness, that is, for any $\Psi \in { \mathcal { T } } ( { \mathcal { X } } )$ , there exists $d \in { \mathcal { D } } ( { \mathcal { X } } )$ such that $\Psi \subseteq T ( d ) . ^ { 7 }$

## 4.3 Existence of Uncountably Many Maximal Admissible Methods

## 4.3.1 Intersection and Union of Methods

Corollary 16 shows that every finite collection of admissible methods has a lower bound in $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ , whereas by Proposition 17 an infinite collection of admissible methods may not admit a lower bound in $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ . There are, however, particular cases for which the infimum and the supremum, of an arbitrary, possibly uncountable, collection of admissible methods remain in $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ . We introduce intersections and unions of hierarchical clustering methods and determine when these operations remain hierarchical clustering methods and preserve admissibility.

Definition 18 For two hierarchical clustering methods $T _ { 1 } , T _ { 2 } \in { \mathcal { H } } ( { \mathcal { X } } )$ , define their intersection $T _ { 1 } \sqcap T _ { 2 }$ and union $T _ { 1 } \sqcup T _ { 2 }$ by

$$
( T _ { 1 } \sqcap T _ { 2 } ) ( d ) : = T _ { 1 } ( d ) \cap T _ { 2 } ( d ) \quad \ a n d \quad ( T _ { 1 } \sqcup T _ { 2 } ) ( d ) : = T _ { 1 } ( d ) \cup T _ { 2 } ( d ) \qquad \forall d \in \mathcal { D } ( \mathcal { X } ) .
$$

More generally, for any nonempty subset $L \subseteq { \mathcal { H } } ( { \mathcal { X } } )$ of hierarchical clustering methods, define $T _ { \sqcap L }$ and $T _ { \sqcup L }$ by

$$
T _ { \cap L } ( d ) \ : = \ \bigcap _ { T _ { i } \in L } T _ { i } ( d ) \quad a n d \quad T _ { \sqcup L } ( d ) \ : = \ \bigcup _ { T _ { i } \in L } T _ { i } ( d ) \qquad \forall d \in { \mathcal { D } } ( { \mathcal { X } } ) .
$$

For any dissimilarity function d and any nonempty $L \subseteq { \mathcal { H } } ( \chi ) , T _ { \cap L } ( d )$ is always a hierarchy, as the intersection of laminar families remains laminar and still contains $\mathcal { X }$ and all singletons. In contrast, $T _ { \scriptscriptstyle \perp L } ( d )$ need not be laminar, and therefore need not define a hierarchy in general. However, closure under union holds under an additional compatibility assumption, as captured by the following lemma.

Lemma 19 Let $L \subseteq { \mathcal { H } } ( { \mathcal { X } } )$ be a nonempty set of hierarchical clustering methods.

(i) $T _ { \Pi L } \in { \mathcal { H } } ( { \mathcal { X } } )$

(ii) $T _ { \sqcup L } \in { \mathcal { H } } ( { \mathcal { X } } )$ if and only if the methods in L are pairwise compatible, i.e., for all $T _ { i } , T _ { j } \in L , T _ { i } \sqcup T _ { j } \in \mathcal { H } ( \mathcal { X } )$

Proof (i) For any $d \in { \mathcal { D } } ( { \mathcal { X } } )$ , each $T ( d )$ is a laminar family containing X and all singletons. The intersection $\begin{array} { r } { T _ { \Pi L } ( d ) \ = \ \bigcap _ { T \in L } T ( d ) } \end{array}$ still contains X and all singletons, and remains laminar since laminarity is a pairwise condition preserved under intersection.

(ii) If the methods in L are pairwise compatible, then for any $d \in { \mathcal { D } } ( { \mathcal { X } } )$ and any $A , B \in T _ { \sqcup L } ( d )$ , there exist $T _ { i } , T _ { j } \in L$ with $A \in T _ { i } ( d )$ and $B \in T _ { j } ( d )$ . Pairwise compatibility implies that A, B are laminar. Together with X and all singletons being in $T _ { \sqcup L } ( d )$ , this proves $T _ { \sqcup L } \in { \mathcal { H } }$ . Conversely, if $T _ { \sqcup L } \in { \mathcal { H } }$ , then for any $T _ { i } , T _ { j } \in L , T _ { i } \sqcup T _ { j } \subseteq T _ { \sqcup L }$ inherits laminarity, so $T _ { i } \sqcup T _ { j } \in { \mathcal { H } } ( { \mathcal { X } } )$

Lemma 19 settles the structural question of when $T _ { \sqcap L }$ and $T _ { \scriptscriptstyle \perp L }$ define hierarchies. We now turn to the axiomatic question: under what conditions are these operations closed within the admissible class $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } ) ?$

Lemma 20 Let $L \subseteq \mathcal H _ { \alpha _ { \mathrm { a d m } } } ( \mathcal X )$ be nonempty.

(i) If L is finite, then $T _ { \Pi L } \in \mathcal H _ { \alpha _ { \mathrm { a d m } } } ( \mathcal X )$

(ii) If L is upward directed under Ď, then $T _ { \sqcup L } \in { \mathcal { H } } _ { \alpha _ { \mathrm { a d m } } } ( { \mathcal { X } } )$

Observe that finiteness of L is required to ensure $T _ { \Pi L } \in \mathcal H _ { \alpha _ { \mathrm { a d m } } } ( \mathcal X )$ . As a simple counterexample, consider $L = \{ T _ { \mathrm { g l o b } } ^ { \alpha \mathbf { 1 } } \colon \alpha \in ( 0 , 1 ] \}$ . Then L is an infinite subset of $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ , and $T _ { \Pi L } = T _ { \mathrm { t r i v } }$ is the trivial method defined in Equation (2), which is not admissible.

A partially ordered set admitting an infimum (resp. supremum) for every finite nonempty subset is called a meet-semilattice (resp. join-semilattice). Lemma 20 thus implies that $( { \mathcal { H } } _ { \alpha _ { \mathrm { a d m } } } ( { \boldsymbol { \mathcal { X } } } ) , \subseteq )$ is a meet-semilattice but not a join-semilattice.

## 4.3.2 Existence of Maximal Admissible Methods

A consequence of Lemma 13 is that $( { \mathcal { H } } _ { \alpha _ { \mathrm { a d m } } } ( { \boldsymbol { \mathcal { X } } } ) , \subseteq )$ has no greatest element and does not even admit a supremum in HpX q. Nevertheless, closure under unions of chains yields, via Zorn’s lemma, the existence of maximal elements.

Theorem 21 The set $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ has maximal elements, and every $T \in \mathcal H _ { \alpha _ { \mathrm { a d m } } } ( \mathcal X )$ is refined by at least one maximal element $T ^ { M } \in \mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ . Moreover, $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ contains an uncountable family of pairwise-incompatible maximal elements.

Proof We apply Zorn’s lemma (see, e.g., Shen and Vereshchagin (2002, Theorem 30)). Let $L \subseteq \mathcal H _ { \alpha _ { \mathrm { a d m } } } ( \mathcal X )$ be any chain (i.e., a subset totally ordered by Ď). We show that $T _ { \sqcup L } \in { \mathcal { H } } _ { \alpha _ { \mathrm { a d m } } } ( { \mathcal { X } } )$ , providing an upper bound for L in $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$

Because every chain is upward directed, Lemma 20(ii) yields that $T _ { \sqcup L } \in { \mathcal { H } } _ { \alpha _ { \mathrm { a d m } } } ( { \mathcal { X } } )$ . By Zorn’s lemma, $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ has maximal elements, and every $T \in \mathcal H _ { \alpha _ { \mathrm { a d m } } } ( \mathcal X )$ is refined by some maximal element.

For cardinality and structural incompatibility, consider the uncountable family $\{ T _ { \mathrm { s t a b l e } } ^ { ( p ) }$ $p > 0 \}$ from Lemma 13. Extend each $T _ { \mathrm { s t a b l e } } ^ { ( p ) }$ to a maximal method $M _ { p } . { \mathrm { ~ H ~ } } p \neq q .$ then there is no hierarchy-valued method that can refine both $M _ { p }$ and $M _ { q } .$ , because such a method would also refine both $T _ { \mathrm { s t a b l e } } ^ { ( p ) }$ and $T _ { \mathrm { s t a b l e } } ^ { ( q ) } .$ . Thus $\{ M _ { p } : p > 0 \}$ is an uncountable pairwise incompatible family of maximal elements.

Incompatibility arises when two methods select clusters that cannot coexist in a single hierarchy for some input. Conversely, any pairwise-compatible family of methods can be combined through their union, which remains hierarchy-valued.

Although $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ has no greatest element, Theorem 21 guarantees that every admissible method is refined by a maximal admissible method that cannot be further refined within $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$

Figure 3 summarizes the results of this section on the structure of $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$

![](images/2a1ec416ec36f7a5ebc2b048fe9263ee2fa33add03a7cc1e5b998d622c580f5d.jpg)  
Figure 3: Hasse diagrams of a subset of $\mathcal { \ H } _ { \alpha _ { \mathrm { a d m } } } ^ { \mathcal { H } ( \mathcal { X } ) } ( \mathcal { X } )$ under the refinement order $\sqsubseteq ,$ where transitive edges are omitted and an arrow from $T ^ { \prime }$ to T indicates that $T ^ { \prime } \subseteq T$ . The blue solid and red dashed boundaries delimit $\mathcal { H } _ { \alpha _ { \mathrm { a d m } } } ( \mathcal { X } )$ and $\mathcal { H } ( \mathcal { X } )$ , respectively. The superscript M denotes the maximal admissible element of each chain. In particular, the diagram includes the strong upward antichain $\{ T _ { \mathrm { s t a b l e } } ^ { ( p ) } \} _ { p \in [ n - 2 ] }$ , and chains $\{ T _ { \mathrm { g l o b } } ^ { \eta ^ { ( i ) } } \} _ { i \in [ n - 2 ] }$ , and $\{ T _ { \mathrm { l o c } } ^ { \eta ^ { ( i ) } } \} _ { i \in [ n - 2 ] }$ with $\mathbf { \eta } \mathbf { \eta } \mathbf { \Sigma } \mathbf { \eta } \mathbf { \Sigma } \mathbf { \eta } \mathbf { \Sigma } \mathbf { \eta } \mathbf { \Sigma } \mathbf { \eta } \mathbf { ( } i ) = \eta ^ { i } \mathbf { 1 } \mathbf { \Sigma }$ , for some $\eta \in ( 0 , 1 )$ .

## 5 Extensions of the Canonical Framework

The previous sections study hierarchical clustering under the canonical axiom system. We now examine the robustness of this framework in two directions that are specific to hierarchical clustering and its practical use. First, we add exactness on ultrametric inputs, a fidelity requirement with no direct analog in flat clustering. Second, we allow the input dissimilarity to be preprocessed before the hierarchy is constructed and ask which axioms are preserved under composition.

## 5.1 Exactness on Ultrametric Inputs

The admissibility axioms closely follow Kleinberg’s original setting. However, in the particular case where the input distance is an ultrametric, hierarchical clustering induces a known correspondence with dendrograms, which motivates an additional requirement tailored to hierarchical outputs. Recall that a distance d is an ultrametric if d satisfies the strong triangle inequality

$$
d ( x , y ) \ \leqslant \ \operatorname* { m a x } \{ d ( x , z ) , d ( y , z ) \} \quad \forall x , y , z \in \mathcal { X } ,
$$

and that a dendrogram $( \Psi , \theta )$ is a hierarchy $\Psi \in { \mathcal { T } }$ in which each node is assigned height through the dendrogram’s height function $\theta \colon \Psi \  \ \mathbb { R } _ { + } .$ , with leaves (singletons) having height 0. The function θ satisfies $\theta ( \{ x \} ) = 0$ for every $x \in \mathcal { X }$ and is strictly increasing from the leaves towards the root, that is, $\theta ( C ) < \theta ( C ^ { \prime } )$ for every $C \subsetneq C ^ { \prime }$

Ultrametric distances and dendrograms are in one-to-one correspondence (see $e . g .$ , Carlsson and M´emoli $\left( 2 0 1 0 \right) )$ : for every ultrametric $u ,$ , there exists a unique dendrogram $( \Psi _ { u } , \theta _ { u } )$ such that $u ( x , y ) = \theta _ { u } ( \mathrm { l c a } _ { \Psi _ { u } } ( x , y ) )$ for all $x , y \in { \mathcal { X } }$ , where lca $\Psi _ { u } \left( x , y \right)$ denotes the smallest cluster in $\Psi _ { u }$ containing both x and y. Moreover, the hierarchy $\Psi _ { u }$ can be constructed explicitly from the ultrametric u by

$$
\Psi _ { u } = \{ B _ { u } ( x , r ) : x \in \mathcal { X } , \ r \geqslant 0 \} \quad \mathrm { ~ w h e r e ~ } B _ { u } ( x , r ) : = \{ y \in \mathcal { X } : u ( x , y ) \leqslant r \} ,\tag{3}
$$

and the dendrogram’s height function $\theta _ { u }$ is given by $\theta _ { u } ( C ) = \operatorname* { m a x } _ { x , y \in C } u ( x , y )$

This correspondence motivates an additional fidelity requirement: when the input is already an ultrametric, the method should recover its canonical hierarchy (albeit we do not requires to output the associated height).

Definition 22 A hierarchical clustering method $T \in { \mathcal { H } } ( { \mathcal { X } } )$ is exact on ultrametrics $i f ,$ for every ultrametric $u ,$ we have $T ( u ) = \Psi _ { u }$

In particular, when restricted to ultrametric inputs, all methods that are exact on ultrametrics coincide.

We say that a hierarchical clustering method is strongly admissible if it is admissible and exact on ultrametrics. We denote this strengthened axiom system by $\alpha _ { \mathrm { a d m + } }$ , and write $\mathcal { H } _ { \alpha _ { \mathrm { a d m } + } } ( \mathcal { X } )$ for the class of strongly admissible hierarchical clustering methods on $\mathcal { X }$

We prove in the appendix that the non-binary variant of single linkage is exact on ultrametrics. Moreover, $T _ { \mathrm { g l o b } } ^ { \eta }$ and $T _ { \mathrm { l o c } } ^ { \eta }$ are exact on ultrametrics if and only if $\pmb { \eta } = \mathbf { 1 }$ . Finally, we also prove that for every $p > 0$ , the composed method $T _ { \mathrm { s t a b l e } } ^ { ( p ) } : = T _ { \mathrm { s t a b l e } } \circ \mu _ { \mathrm { p o w e r } } ^ { p }$ is exact on ultrametrics. Together, these results show that there exist uncountably many strongly admissible methods. Moreover, the results of Section 4 regarding the existence of uncountably many maximal elements and of a backbone hierarchy hold when we replace $\alpha _ { \mathrm { a d m } }$ by $\alpha _ { \mathrm { a d m + } }$ . There are, however, some important diferences. Indeed, for every strongly admissible method $T .$ , we have $T _ { \mathrm { g l o b } } \subseteq T$ Thus, whereas $( { \mathcal { H } } _ { \alpha _ { \mathrm { a d m } } } ( { \mathcal { X } } ) , \subseteq )$ has no least element (Proposition 17), $( { \mathcal { H } } _ { \alpha _ { \mathrm { a d m } + } } ( { \bar { \mathcal { X } } } ) , \subseteq )$ has one least element, $T _ { \mathrm { g l o b } }$ , which is thus the coarsest strongly admissible hierarchical method. In practice, $T _ { \mathrm { g l o b } }$ returns a hierarchy composed of many non-singleton clusters. To illustrate it, we provide in Appendix E numerical experiments to evaluate the size of the hierarchy returned by $T _ { \mathrm { g l o b } }$ on real data sets.

Theorem 23 The height, width, and cellularity of $\mathcal { H } _ { \alpha _ { \mathrm { a d m + } } } ( \mathcal { X } )$ are uncountable. In particular, $T _ { \mathrm { S L } } , T _ { \mathrm { g l o b } } , T _ { \mathrm { l o c } } $ and $T _ { \mathrm { s t a b l e } } ^ { ( p ) } \ f o r$ every $p > 0$ are strongly admissible. Moreover,

1. The backbone and maximality results of Section $\it 4$ remain valid;

2. $\mathcal { H } _ { \alpha _ { \mathrm { a d m } + } } ( \mathcal { X } )$ admits uncountably many pairwise incompatible maximal elements;

3. $T _ { \mathrm { g l o b } }$ is the least strongly admissible method under refinement.

## 5.2 Axiom-Preserving Preprocessing

Modern clustering pipelines rarely apply a hierarchical clustering method directly to the raw observations or their original pairwise dissimilarities. Instead, the data is typically preprocessed before the hierarchy is constructed. Common examples include dimensionality reduction, feature re-weighting, and the choice or transformation of a metric (for example using a kernel). Density-aware procedures also fit this framework: they first construct a nearest-neighbor-based dissimilarity and then apply a single-linkage-type method to the transformed data. These examples explain treating preprocessing as a part of the clustering pipeline and asking which axioms are preserved by it.

## 5.2.1 General Preservation Principles

We model preprocessing as a map $\mu \colon { \mathcal { D } } ( { \mathcal { X } } ) \to { \mathcal { D } } ( { \mathcal { X } } )$ , called a transformation. Given a hierarchical clustering method $T _ { i }$ , the composed method $T \circ \mu \colon d \mapsto T ( \mu ( d ) )$ first transforms the input dissimilarity and then applies $T$ . Because $\mu ( d )$ is a dissimilarity on the same domain as $d ,$ the composition $T \circ \mu$ is again a hierarchical clustering method.

Our goal is to study when the axiomatic properties of T are inherited by the composed method $T \circ \mu$ . Rather than studying this question separately for each method, we lift each axiom to transformations: a transformation preserves an axiom if composition with it preserves that axiom for every method satisfying it.

Definition 24 A transformation $\mu \colon { \mathcal { D } } ( { \mathcal { X } } ) \to { \mathcal { D } } ( { \mathcal { X } } )$ preserves an axiom α $i f ,$ for every method T that satisfies α, the composed method $T \circ \mu$ also satisfies α.

Lemma 25 Let $\mu \colon { \mathcal { D } } ( { \mathcal { X } } ) \to { \mathcal { D } } ( { \mathcal { X } } )$ . Each of the following conditions is suficient for $\mu$ to preserve the corresponding axiom:

• Scale invariance: for every d P $\mathcal { D } ( \mathcal { X } )$ and every $\beta > 0$ , there exists $\beta ^ { \prime } > 0$ such that $\mu ( \beta d ) = \beta ^ { \prime } \mu ( d )$ ;

• Partition richness: $\mu$ is surjective;

• Partition consistency: for every partition $\mathcal { C } \in \mathcal { P } ( \mathcal { X } )$ and every d, $d ^ { \prime } \in \mathcal { D } ( \mathcal { X } )$ , if d<sup>1</sup> is a C-strengthening of $d ,$ then $\mu ( d ^ { \prime } )$ is a C-strengthening of $\mu ( d )$ ;

• Permutation invariance: for every $d \in { \mathcal { D } } ( { \mathcal { X } } )$ and every permutation ϕ, $\mu ( d _ { \phi } ) = \mu ( d ) _ { \phi } ;$

• Exactness on ultrametrics: for every ultrametric $u , \mu ( u )$ is an ultrametric and $\Psi _ { \mu ( u ) } =$ $\Psi _ { u }$

Except for partition richness, each condition in Lemma 25 ensures that the input structure relevant to the corresponding axiom is preserved by the transformation: scaling remains scaling, strengthenings remain strengthenings, and permutations commute with the transformation. Similarly, ultrametrics are mapped to ultrametrics that induce the same hierarchy.

These observations are instances of a more general principle: the axioms above can be expressed by imposing a condition on the outputs associated with one or two dissimilarities satisfying a prescribed relation. In Appendix B, we formalize this class of relational properties and recover the corresponding parts of Lemma 25 as corollaries.

Corollary 26 Let $g \colon \mathbb { R } _ { \geq 0 }  \mathbb { R } _ { \geqslant 0 }$ satisfy $g ( 0 ) = 0$ and $g ( t ) > 0$ for every $t > 0$ . Define the pointwise transformation $\mu _ { g } \colon D ( \mathcal { X } )  \mathcal { D } ( \mathcal { X } )$ by

$$
\mu _ { g } ( d ) ( x , y ) : = g { \big ( } d ( x , y ) { \big ) } \qquad f o r \ a l l \ d \in { \mathcal { D } } ( { \mathcal { X } } ) \ a n d \ x , y \in { \mathcal { X } } .
$$

Then $\mu _ { g }$ preserves permutation invariance. Moreover,

$i f g$ is strictly increasing, then $\mu _ { g }$ preserves partition consistency and exactness on ultrametrics;

• if g is surjective onto $\mathbb { R } _ { \geqslant 0 }$ , then $\mu _ { g }$ preserves partition richness;

• if, for every $\beta > 0$ , there exists $c _ { \beta } > 0$ such that $g ( \beta t ) = c _ { \beta } g ( t )$ for every $t \geqslant 0$ , then $\mu _ { g }$ preserves scale invariance.

In particular, for every $p > 0 _ { \cdot }$ , the power transformation $\mu _ { \mathrm { p o w e r } } ^ { p } ( d ) ( x , y ) : = d ( x , y ) ^ { p }$ preserves scale invariance, partition richness, partition consistency, permutation invariance, and exactness on ultrametrics. Thus, $\mu _ { \mathrm { p o w e r } } ^ { p }$ preserves both admissibility and strong admissibility.

## 5.2.2 Examples and Applications

Minimax-path preprocessing and single linkage. Our first example gives an exact preprocessing representation of single linkage. Given a dissimilarity d $\in \mathcal { D } ( \mathcal { X } )$ , consider the weighted complete graph whose vertices are the points in X with edge-weight d. The bottleneck of a path $\gamma$ is the largest weight along the path, that ${ \mathrm { i s } } ,$ ma $\mathrm { x } _ { e \in \gamma } d ( e )$ . The minimum bottleneck between two points $x , y \in { \mathcal { X } }$ is

$$
B ^ { * } ( d ) ( x , y ) \ : = \ \left\{ \begin{array} { l l } { \underset { \gamma \in \Gamma _ { x , y } } { \operatorname* { m i n } } \ \underset { \{ u , v \} \in \gamma } { \operatorname* { m a x } } d ( u , v ) , } & { x \neq y , } \\ { 0 , } & { x = y , } \end{array} \right.\tag{4}
$$

where $\Gamma _ { x , y }$ denotes the set of paths from x to y.

Although the definition of $B ^ { * }$ given in (4) involves a minimum over all paths, this quantity can eficiently be computed from any minimum spanning tree M. Indeed, if $\gamma _ { x , y } ^ { M }$ denotes the unique path from x to y in M, then

$$
B ^ { * } ( d ) ( x , y ) \ = \ \operatorname* { m a x } _ { \{ u , v \} \in \gamma _ { x , y } ^ { M } } d ( u , v ) .
$$

The equivalence between this minimum-bottleneck construction and single linkage is standard; see, for example, Carlsson and M´emoli (2010).

The dissimilarity $B ^ { * } ( d )$ is an ultrametric, commonly called the subdominant ultrametric associated with $d ;$ its values are also known as minimax-path distances. Moreover, for every $r \geqslant 0$ , two points x and y satisfy $B ^ { * } ( d ) ( x , y ) \ \leqslant \ r$ if and only if they belong to the same connected component of the graph whose edges are the pairs $\{ u , v \}$ satisfying $d ( u , v ) \leqslant r$ . These connected components, as r varies, are exactly the clusters produced by single linkage. Hence, by the ultrametric-hierarchy correspondence detailed in Equation (3), we have $T _ { \mathrm { S L } } ( d ) ~ = ~ \Psi _ { B ^ { * } ( d ) }$ . Moreover, because $T _ { \mathrm { g l o b } }$ is exact on ultrametric inputs, we obtain $T _ { \mathrm { S L } } ( d ) = T _ { \mathrm { g l o b } } \big ( B ^ { * } ( d ) \big )$ , and hence

$$
T _ { \mathrm { S L } } ~ = ~ T _ { \mathrm { g l o b } } \circ B ^ { * } .\tag{5}
$$

Thus, single linkage can be viewed as first replacing the original dissimilarity by its minimaxpath ultrametric and then extracting the globally separated clusters. Observe also that the same reasoning applies to any method that is exact on ultrametric inputs; in particular, we may replace $T _ { \mathrm { g l o b } }$ in (5) with $T _ { \mathrm { l o c } }$ or $T _ { \mathrm { s t a b l e } }$

Finally, the transformation $B ^ { * } \colon d \in { \mathcal { D } } ( \mathcal { X } ) \mapsto B ^ { * } ( d )$ satisfies

$$
B ^ { * } ( \beta d ) = \beta B ^ { * } ( d ) , \qquad B ^ { * } ( d _ { \phi } ) = B ^ { * } ( d ) _ { \phi } , \quad \mathrm { a n d } \quad B ^ { * } ( u ) = u ,
$$

for every $\beta > 0$ , every permutation $\phi ,$ and every ultrametric $u .$ Hence, $B ^ { * }$ preserves scale invariance, permutation invariance, and exactness on ultrametrics. Together with the corresponding properties of $T _ { \mathrm { g l o b } }$ , the factorization $T _ { \mathrm { S L } } = T _ { \mathrm { g l o b } } \circ B ^ { * }$ yields these properties for single linkage.

PCA preprocessing. In this paragraph only, we restrict to Euclidean distances $d ,$ meaning that there exist points $( z _ { x } ) _ { x \in \mathcal { X } } \subset \mathbb { R } ^ { p }$ such that $d ( x , y ) ~ = ~ \| z _ { x } - z _ { y } \| _ { 2 }$ for all $x , y \in { \mathcal { X } }$ . Center these points, that is define $\tilde { z } _ { x } : = z _ { x } - \bar { z }$ where $\begin{array} { r } { \bar { z } = \frac { 1 } { | \mathcal { X } | } \sum _ { x \in \mathcal { X } } { z _ { x } } . } \end{array}$ , and let $P _ { r }$ be the orthogonal projection onto the subspace spanned by the first r principal components of the centered configuration $( \tilde { z } _ { x } ) _ { x \in \mathcal { X } }$ . Assuming that this principal subspace is uniquely defined, we define

$$
\mu _ { \mathrm { P C A } , r } ( d ) ( x , y ) : = \left. P _ { r } \widetilde { z } _ { x } - P _ { r } \widetilde { z } _ { y } \right. _ { 2 } .
$$

This definition does not depend on the chosen Euclidean realization, since centered realizations of the same dissimilarity difer only by an orthogonal transformation. Under the assumption that the projected points $P _ { r } \tilde { z } _ { x } , x \in \mathcal { X }$ are pairwise distinct,<sup>8</sup> $\mu _ { \mathrm { P C A } , r } ( d )$ is a dissimilarity.

Multiplying d by $\beta > 0$ amounts to multiplying the realizing configuration by $\beta _ { i }$ which leaves its principal subspace unchanged and multiplies all projected distances by $\beta .$ Hence $\mu _ { \mathrm { P C A } , r } ( \beta d ) = \beta \mu _ { \mathrm { P C A } , r } ( d )$ , and thus PCA preprocessing preserves scale invariance.

PCA preprocessing is also equivariant under relabelling. Indeed, permuting the points leaves the covariance operator, and hence the principal r-dimensional subspace, unchanged, while merely relabelling the projected points. Therefore, $\mu _ { \mathrm { P C A } , r } ( d _ { \phi } ) \ : = \ : \mu _ { \mathrm { P C A } , r } ( d ) _ { \phi }$ , and thus PCA preprocessing also preserves permutation invariance.

In general, however, PCA preprocessing need not preserve the remaining axioms. A partition strengthening may alter the principal subspace, so its image need not remain a strengthening of the projected dissimilarity. Moreover, for fixed $^ { r , }$ every output of $\mu _ { \mathrm { P C A } , r }$ is a Euclidean dissimilarity realizable in dimension at most $r .$ Because a general dissimilarity need not have this form, $\mu _ { \mathrm { P C A } , r }$ is not surjective onto $\mathcal { D } ( \mathcal { X } )$ . Finally, projecting an ultrametric realization need not produce an ultrametric inducing the same hierarchy.

HDBSCAN and mutual-reachability preprocessing. Fix $k \geqslant 1$ , and let $d _ { \mathrm { c o r e } } ( x )$ be the dissimilarity from x to its k-th nearest neighbor. The mutual-reachability transformation used in HDBSCAN is defined, for $x \neq y$ , by

$$
\begin{array} { r } { \mu _ { \mathrm { H D B S C A N } } ( d ) ( x , y ) : = \operatorname* { m a x } \big \{ d _ { \mathrm { c o r e } } ( x ) , d _ { \mathrm { c o r e } } ( y ) , d ( x , y ) \big \} , } \end{array}
$$

with zero diagonal. Since all nearest-neighbor dissimilarities scale together with $d ,$

$$
\mu _ { \mathrm { H D B S C A N } } ( \beta d ) = \beta \mu _ { \mathrm { H D B S C A N } } ( d ) \qquad \mathrm { f o r ~ e v e r y ~ } \beta > 0 .
$$

Moreover, the construction is equivariant under relabelling: $\mu _ { \mathrm { H D B S C A N } } ( d _ { \phi } ) = \mu _ { \mathrm { H D B S C A N } } ( d ) _ { \phi } .$ Hence mutual-reachability preprocessing preserves both scale invariance and permutation invariance of the downstream hierarchical method.

Similarity-valued preprocessing. The same preservation principle extends to pipelines expressed in terms of similarities. For example, the Gaussian kernel $\begin{array} { r } { \overline { { k } } _ { \sigma } ( x , y ) : = \exp \bigg \{ - \frac { ( d ( x , y ) ) ^ { 2 } } { 2 \sigma ^ { 2 } } \bigg \} } \end{array}$ is a strictly decreasing function of $d ( x , y )$ . It therefore converts a dissimilarity strengthening into the corresponding similarity strengthening: within-cluster similarities increase, whereas cross-cluster similarities decrease. The transformation also commutes with permutations. (Note that this example lies outside the dissimilarity-valued framework adopted above, so it should be understood using the similarity analog of the relevant axioms.)

## 6 Related Work

Prior to Kleinberg, Puzicha et al. (2000) developed an axiomatic framework for clustering objective functions, based on properties such as monotonicity, invariance, and robustness, and identified objectives satisfying these requirements. Kleinberg’s impossibility theorem (Kleinberg, 2002) subsequently motivated a broad line of work that modifies the axioms, the input, or the problem formulation in order to circumvent the impossibility result (Ben-David and Ackerman, 2008; Zadeh and Ben-David, 2009; Strazzeri and S´anchez-Garc´ıa, 2022; Willson and Warnow, 2024). Our work contributes to this line by asking whether the impossibility can be resolved by changing the form of the output, from a flat partition to a hierarchy, while retaining a dissimilarity-based input and direct hierarchical analogs of Kleinberg’s axioms.

More precisely, these works relax Kleinberg’s framework in diferent ways. Ben-David and Ackerman (2008) consider an analogous axiomatic setting for clustering-quality functions and establish the existence of functions satisfying their axioms. Zadeh and Ben-David (2009) take the desired number of clusters as an additional input and correspondingly restrict richness to partitions having that number of clusters. Cohen-Addad et al. (2018) introduce a cost function to estimate the number of clusters and weaken consistency by requiring it only when this estimate remains unchanged. Strazzeri and S´anchez-Garc´ıa (2022) restrict the transformations allowed by the consistency axiom and extend the setting to graph clustering. In the graph setting, Van Laarhoven and Marchiori (2014) adapt axioms for clustering-quality functions and introduce adaptive scale modularity, while Willson and Warnow (2024) translate Kleinberg’s axioms to unweighted graphs and identify clustering methods satisfying their axioms. A diferent structural perspective is developed by

Carlsson and M´emoli (2013), who study clustering schemes through functoriality, requiring compatibility of clustering outputs with suitable maps between input metric spaces.

A distinct line of work develops axiomatic characterizations specific to hierarchical clustering. Carlsson and M´emoli (2010) view hierarchical clustering as a map from finite metric spaces to proximity dendrograms, equivalently ultrametrics, and characterize single linkage using normalization, separation, and functoriality. Functoriality requires every distancenonincreasing map between input metric spaces to remain distance-nonincreasing between the corresponding output ultrametric spaces, including maps between datasets of diferent sizes. As distance-preserving relabelings are examples of such maps, functoriality is substantially stronger than permutation invariance. In fact, it has no direct counterpart in our framework, because it compares outputs across diferent ground sets and constrains their merge scales, whereas our outputs are unweighted hierarchies on a fixed set.

In a complementary direction, Ackerman et al. (2010) and Ackerman and Ben-David (2016) study characterizations of linkage-based methods. In particular, the latter show that locality together with outer consistency characterizes hierarchical linkage methods. Outer consistency increases dissimilarities between the clusters of the partition induced by cutting a dendrogram at a selected height, while keeping within-block dissimilarities fixed. In contrast, partition consistency is thus more faithful to Kleinberg’s consistency as it allows within-block contractions and applies to every partition represented in the hierarchy. As a result, several linkage methods other than single linkage satisfy outer consistency whereas failing partition consistency.

Related axiomatic frameworks have also been developed for asymmetric dissimilarities. In particular, Carlsson et al. (2014) introduce hierarchical quasi-clustering for asymmetric networks and obtain a uniqueness result under their axioms, while Carlsson et al. (2017) characterize broader families of admissible hierarchical methods for asymmetric networks.

These earlier frameworks use additional structural requirements to characterize particular methods or algorithmic families. Our axioms remain closer to Kleinberg’s original requirements and admit a much more diverse class of methods, whose structure under refinement is a key focus of our work.

A complementary perspective axiomatizes objective functions for evaluating hierarchical clusterings rather than the behavior of the clustering method itself. Dasgupta (2016) introduced a cost function for similarity-based hierarchical clustering, thereby formulating hierarchical clustering as an optimization problem. Building on this perspective, Cohen-Addad et al. (2019) give an axiomatic characterization of a broad class of admissible objective functions for both similarity- and dissimilarity-based hierarchical clustering, including Dasgupta’s objective. This difers from our approach: their axioms determine which objective functions appropriately score a hierarchy, whereas ours directly constrain the behavior of the map from dissimilarities to hierarchies.

Population-level approaches provide another, substantially diferent, axiomatic perspective on hierarchical clustering. Thomann et al. (2015) axiomatize hierarchical clustering of probability measures without assuming an underlying metric or dissimilarity. Their starting point is a user-specified clustering on a class of elementary measures, and they study how this clustering can be extended to more general distributions. Under suitable conditions, additivity and continuity requirements determine a unique such extension. More recently, Arias-Castro and Coda (2025) propose an axiomatic definition of hierarchical clustering based on the topology of density level sets. Rather than axiomatizing a clustering method and its behavior under transformations of the input, they seek to characterize which subsets of the support of a density should constitute population clusters. They first consider piecewise-constant densities and require clusters to have connected interior, not to split connected regions of constant density, and to be surrounded by regions of lower density. Among the cluster trees satisfying these requirements, they select the finest one, and subsequently extend the construction to more general densities, recovering Hartigan’s cluster tree under suitable conditions.

Our framework is diferent in both its input and its object of study. In the spirit of Kleinberg, we axiomatize maps from pairwise dissimilarities to hierarchies: The axioms constrain how the output hierarchy behaves when the input dissimilarity is rescaled, strengthened, or relabeled, as well as which hierarchical structures the method can realize. Population-level approaches such as Thomann et al. (2015) and Arias-Castro and Coda (2025) specify what a population hierarchy should be based on distributional or topological information, whereas ours impose structural requirements on hierarchical clustering methods that act directly on dissimilarities.

## 7 Conclusion

Kleinberg’s impossibility theorem shows that scale invariance, richness, and consistency cannot be jointly satisfied by a flat clustering method. In this work, we have shown that this incompatibility disappears when the output is allowed to be hierarchical. Natural hierarchical analogs of Kleinberg’s axioms are jointly satisfiable; in fact, there exist uncountably many hierarchical clustering methods satisfying them.

Moreover, our work reveals that the set of admissible hierarchical methods is extremely complex. Under the refinement order, the class of admissible methods is a poset that has uncountable height, width, and cellularity, has neither a greatest nor a least element, and contains uncountably many pairwise incompatible maximal elements. Nevertheless, admissible methods cannot difer arbitrarily. Every admissible method contains a hierarchy of suficiently well-separated clusters, and every finite collection of admissible methods therefore shares a nontrivial common backbone. Thus, the axioms simultaneously allow for substantial freedom in the additional clusters reported by a method while enforcing a common conservative hierarchical structure.

We also considered extensions that are specific to the hierarchical setting. Adding exactness on ultrametric inputs preserves the main structural picture while imposing a stronger connection between dissimilarities and their canonical hierarchies. In addition, we studied preprocessing transformations and identified conditions under which they preserve the axioms, allowing the framework to apply to clustering pipelines in which the input dissimilarity is transformed before the hierarchy is constructed.

This study raises several open questions. A first one is to determine which other hierarchical clustering methods are admissible, and, in particular, to identify explicit maximal admissible methods. Another one is to understand which additional axioms meaningfully reduce the large admissible class. Our results on ultrametric exactness provide one step in this direction, yielding, in particular, a least element among strongly admissible methods.

Finally, hierarchical clustering has an expressive advantage over flat clustering, as it does not require the number of clusters to be specified in advance and returns a richer, nested cluster structure that cannot be represented by a single partition. Nevertheless, some downstream tasks ultimately require a flat partition, thereby requiring a cut of the hierarchy, that is, a selection of a set of clusters from the hierarchy that forms a partition. Kleinberg’s theorem implies that no such cut-selection rule can induce a flat clustering method that simultaneously satisfies scale invariance, richness, and consistency. Characterizing the trade-ofs inherent in cut selection, as well as the benefits of hierarchy-aware downstream procedures that avoid this projection altogether, remains an interesting direction for future work.

## Appendix A. Alternative Axioms and Robustness

Whereas the main text focuses on the canonical axiom system $\alpha _ { \mathrm { a d m } }$ and its strengthening by exactness on ultrametrics, several closely related formulations are also natural in the hierarchical setting. This appendix collects the variants that are useful for comparison, describes their logical relations, and shows that the principal structural conclusions of Section 4 are robust to many of these choices.

We denote by α a requirement, or a property, that may or may not be satisfied by a given hierarchical clustering method. In particular, all axioms considered in this paper are requirements. For a requirement $\alpha$ , let

$$
\mathcal { H } _ { \alpha } ( \mathcal { X } ) : = \{ T \in \mathcal { H } ( \mathcal { X } ) : T \mathrm { ~ s a t i s f i e s ~ } \alpha \} .
$$

We say that a requirement $\alpha _ { 2 }$ is stronger than another requirement $\alpha _ { 1 }$ , denoted $\alpha _ { 2 } \vartriangle { \alpha } \Delta$ if $\mathcal { H } _ { \alpha _ { 2 } } ( \mathcal { X } ) \subseteq \mathcal { H } _ { \alpha _ { 1 } } ( \mathcal { X } )$ . If both $\alpha _ { 2 } \vartriangle { \alpha } \boldsymbol { \alpha } _ { 1 }$ and $\alpha _ { 1 } \simeq \alpha _ { 2 }$ hold, we say the two requirements are equivalent, and write $\alpha _ { 1 } \equiv \alpha _ { 2 }$ . Finally, the conjunction of $\alpha _ { 1 }$ and $\alpha _ { 2 }$ is denoted by $\alpha _ { 1 } \wedge \alpha _ { 2 }$

We retain the notation $\alpha _ { \mathrm { s c a } } , \alpha _ { \mathrm { p r } } , \alpha _ { \mathrm { p c } } , \alpha _ { \mathrm { p e r m } }$ for scale invariance, partition richness, partition consistency, and permutation invariance, respectively, and $\alpha _ { \mathrm { s u h r } }$ for exactness on ultrametrics. Thus

$$
\begin{array} { r } { \alpha _ { \mathrm { a d m } } : = \alpha _ { \mathrm { s c a } } \wedge \alpha _ { \mathrm { p r } } \wedge \alpha _ { \mathrm { p c } } \wedge \alpha _ { \mathrm { p e r m } } , \qquad \alpha _ { \mathrm { a d m } + } : = \alpha _ { \mathrm { a d m } } \wedge \alpha _ { \mathrm { s u h r } } \ \equiv \ \alpha _ { \mathrm { s c a } } \wedge \alpha _ { \mathrm { s u h r } } \wedge \alpha _ { \mathrm { p c } } \wedge \alpha _ { \mathrm { p e r m } } , } \end{array}
$$

where the last equivalence holds because exactness on ultrametrics implies partition richness.   
More generally, an axiom system is a conjunction of requirements.

Table 1 summarizes all the requirements/properties considered in this paper, grouped into four categories: admissibility, invariance, richness, and consistency.

## A.1 Alternative Axiom Formulations

## A.1.1 Order invariance

Scale invariance can be strengthened by requiring the output to depend only on the ordering of the pairwise dissimilarities.

Definition 27 (Order invariance $\alpha _ { \mathrm { o r d } } )$ A method $T \in { \mathcal { H } } ( { \mathcal { X } } )$ is order invariant $i f ,$ for every $d \in { \mathcal { D } } ( { \mathcal { X } } )$ and every strictly increasing $g : \mathbb { R } _ { \geq 0 }  \mathbb { R } _ { \geqslant 0 }$ satisfying $g ( 0 ) = 0$ , we have $T ( g \circ d ) = T ( d )$ , where $( g \circ d ) ( x , y ) : = g ( d ( x , y ) )$ for all elements x, $y \in \mathcal X$

<table><tr><td>Category</td><td>Symbol / name</td><td> $\gamma _ { \Pi }$ </td><td> $\gamma _ { \sqcup } ^ { \mathrm { u p } }$ </td><td> $\gamma _ { \sqcup }$ </td><td> $\gamma _ { \mathrm { r p } }$ </td></tr><tr><td rowspan="2">Admissibility</td><td> $\alpha _ { \mathrm { a d m } }$  (admissibility)</td><td></td><td></td><td></td><td>X X</td></tr><tr><td> $\alpha _ { \mathrm { a d m } + }$  (strong admissibility)</td><td></td><td></td><td>X</td><td></td></tr><tr><td rowspan="3">Invariance</td><td>(scale invariance)  $\alpha _ { \mathrm { s c a } }$ </td><td></td><td>√</td><td>√</td><td></td></tr><tr><td> $\alpha _ { \mathrm { o r d } }$  (order invariance)</td><td></td><td>√</td><td></td><td></td></tr><tr><td> $\alpha _ { \mathrm { p e r m } }$  (permutation invariance)</td><td>√</td><td>L</td><td>√</td><td>√</td></tr><tr><td rowspan="8">Richness variants</td><td> $\alpha _ { \mathrm { p r } }$  (partition richness)</td><td></td><td>X</td><td>√</td><td>√ X</td></tr><tr><td> $\alpha _ { \mathrm { c w r } }$ </td><td>(cluster-wise richness)</td><td>X √</td><td></td><td>X</td></tr><tr><td> $\alpha _ { \mathrm { h r } }$  (hierarchical richness)</td><td>X</td><td>√</td><td>√</td><td>X</td></tr><tr><td> $\alpha _ { \mathrm { s h r } }$ </td><td>(strict hierarchical richness)</td><td>X</td><td>X ×</td><td>X</td></tr><tr><td> $\alpha _ { \mathrm { u h r } }$ </td><td>(refinement on ultrametrics) √</td><td>V</td><td>√</td><td>√</td></tr><tr><td> $\alpha _ { \mathrm { s u h r } }$ </td><td>(exact on ultrametrics</td><td></td><td></td><td></td></tr><tr><td> $\alpha _ { \mathrm { b h } }$  (backbone-hierarchy)</td><td></td><td></td><td></td><td>X</td></tr><tr><td> $\alpha _ { \mathrm { b h } } ^ { 1 }$  (unit-backbone-hierarchy)</td><td>√</td><td>√</td><td>√</td><td>X</td></tr><tr><td rowspan="4">Consistency variants</td><td> $\alpha _ { \mathrm { p c } }$  (partition consistency)</td><td></td><td></td><td>√ X</td><td></td></tr><tr><td> $\alpha _ { \mathrm { c w c } }$  (cluster-wise consistency)</td><td>V</td><td>L</td><td></td><td></td></tr><tr><td>(hierarchical consistency)  $\alpha _ { \mathrm { h c } }$ </td><td>√</td><td>V</td><td>√</td><td>√</td></tr><tr><td> $\alpha _ { \mathrm { a b c } }$  (absence consistency)</td><td>√</td><td>√</td><td>X</td><td>√</td></tr></table>

Table 1: Summary of the (alternative) axioms and their structural properties. Gray shading indicates the canonical axioms; magenta shading indicates exactness on ultrametrics and strong admissibility. Unshaded rows correspond to requirements introduced in this appendix. A checkmark in the $\gamma _ { \Pi } , \gamma _ { \sqcup } ^ { \mathrm { u p } } , \mathrm { o r } \gamma _ { \sqcup }$ column means that the corresponding method class is closed, respectively, under finite nonempty intersections, unions of nonempty upward-directed families, or unions of pairwise compatible nonempty families; a cross indicates that no such closure property is asserted. The three closure properties $\gamma _ { \sqcap } , \gamma _ { \sqcup } ^ { \mathrm { u p } }$ , or $\gamma _ { \sqcup }$ are preserved under finite conjunctions of axioms; see Proposition 35. Similarly, a checkmark in the column $\gamma _ { \mathrm { r p } }$ means that the axiom is representable as a relational property or as a conjunction of relational properties (we refer to Section B for a precise definition and to Section B.2 for a proof of the checkmarks in the last column).

## A.1.2 Richness variants

Partition richness requires that for any partition, there is a dissimilarity function d that makes the partition part of the output hierarchy. In the hierarchical setting, one may instead require the realization of individual clusters or of entire hierarchies.

Definition 28 (Richness variants) A method $T \in { \mathcal { H } } ( { \mathcal { X } } )$ satisfies

• cluster-wise richness $\alpha _ { \mathrm { c w r } } \ i f ,$ for every nonempty $C \subseteq { \mathcal { X } }$ , there exists d $\in \mathcal { D } ( \mathcal { X } )$ such that $C \in T ( d )$ ;

• hierarchical richness $\alpha _ { \mathrm { h r } } \ \textit { i f } ,$ for every $\Psi \in { \mathcal { T } } ( { \mathcal { X } } )$ , there exists $d \in { \mathcal { D } } ( { \mathcal { X } } )$ such that $\Psi \subseteq T ( d )$ ;

• strict hierarchical richness $\alpha _ { \mathrm { s h r } }$ if, for every $\Psi \in { \mathcal { T } } ( { \mathcal { X } } )$ , there exists $d \in { \mathcal { D } } ( { \mathcal { X } } )$ such that $T ( d ) = \Psi ,$

• refinement on ultrametrics $\alpha _ { \mathrm { u h r } } \ i f , \ \Psi _ { u } \subseteq T ( u )$ for every ultrametric u.

Refinement on ultrametrics is an alternative to exactness on ultrametrics (Definition 22;   
denoted $\alpha _ { \mathrm { s u h r } } )$ , and parallel to $\alpha _ { \mathrm { s u h r } }$ , this can be regarded as a richness variant.

## A.1.3 Consistency variants

Partition consistency preserves all blocks of a partition under a strengthening of that partition. A natural alternative is to require the same property cluster by cluster. To compare the two formulations, we first extend the notion of strengthening.

Definition 29 (ψ-strengthening) Let $\psi \subseteq 2 ^ { \mathcal { X } } \backslash \{ \emptyset \}$ be nonempty. A dissimilarity $d ^ { \prime } \in$ $\mathcal { D } ( \mathcal { X } )$ is a ψ-strengthening of $d \in { \mathcal { D } } ( { \mathcal { X } } )$ if, for every $C \in \psi$

• (Intra-cluster contraction) $d ^ { \prime } ( x , y ) \leqslant d ( x , y )$ for all $x , y \in C ,$

• (Inter-cluster expansion) $d ^ { \prime } ( x , y ) \geqslant d ( x , y )$ for all $x \in C$ and y R C;

• (External pairs unconstrained) Pairs with both endpoints outside $\cup _ { C \in \psi } C$ are unconstrained.

Observe that in the case of a C-strengthening as defined in Definition $1 , { \mathcal { C } }$ is a partition, so there exists no pair $x , y \notin \bigcup _ { C \in { \mathcal { C } } } C$ as C covers X. However, because the set $\psi$ in Definition 29 does not necessarily cover $\mathcal { X }$ , we make the treatment of such external pairs explicit.

Definition 30 (Consistency variants) A method $T \in { \mathcal { H } } ( { \mathcal { X } } )$ satisfies

• cluster-wise consistency $\alpha _ { \mathrm { c w c } } \ i f ,$ whenever $C \in T ( d )$ , every tCu-strengthening d<sup>1</sup> of d satisfies $C \in T ( d ^ { \prime } )$ ;

• hierarchical consistency $\alpha _ { \mathrm { h c } } \ \mathrm { ~ } i f ,$ whenever $\psi \subseteq T ( d )$ , every ψ-strengthening d<sup>1</sup> of d satisfies $\psi \subseteq T ( d ^ { \prime } )$

For completeness, one can also state partition consistency in the reverse direction. Say that $d ^ { \prime }$ is a C-weakening of d when d is a C-strengthening of $d ^ { \prime } .$ , and call a method absence consistent $\left( \alpha _ { \mathrm { a b c } } \right)$ if

$$
\begin{array} { r l r l } { { \mathcal { C } } \oplus T ( d ) } & { { } \Longrightarrow } & { { \mathcal { C } } \oplus T ( d ^ { \prime } ) } \end{array}
$$

for every C-weakening d<sup>1</sup> of d. As shown below, this is simply the contrapositive formulation of partition consistency.

## A.2 Relations Among the Axioms

The elementary implications among the richness and invariance requirements are $\alpha _ { \mathrm { o r d } } \simeq \alpha _ { \mathrm { s c a } }$ $\alpha _ { \mathrm { s u h r } } \triangleright \alpha _ { \mathrm { s h r } } \triangleright \alpha _ { \mathrm { h r } } \triangleright \alpha _ { \mathrm { p r } } \triangleright \alpha _ { \mathrm { c w r } }$ , as well as $\alpha _ { \mathrm { s u h r } } \sim \alpha _ { \mathrm { u h r } } \vartriangle \simeq \alpha _ { \mathrm { h r } }$ . The following proposition shows that the consistency variants collapse more strongly than their definitions suggest.

Proposition 31 We have $\alpha _ { \mathrm { c w c } } \equiv \alpha _ { \mathrm { h c } } \triangleright \alpha _ { \mathrm { p c } } \equiv \alpha _ { \mathrm { a b c } } .$

Proof Hierarchical consistency implies cluster-wise consistency by taking $\psi = \{ C \}$ . Conversely, if $d ^ { \prime }$ is a ψ-strengthening of $d ,$ then it is a tCu-strengthening for every $C \in \psi$

Cluster-wise consistency therefore preserves every $C \in \psi$ , proving $\alpha _ { \mathrm { c w c } } \equiv \alpha _ { \mathrm { h c } }$ . Taking $\psi = { \mathcal { C } }$ for a partition ${ \mathcal { C } } \subseteq T ( d )$ gives $\alpha _ { \mathrm { h c } } \vartriangle { \alpha } _ { \mathrm { p c } }$

Finally, if $d ^ { \prime }$ is a C-weakening of $d ,$ then d is a C-strengthening of $d ^ { \prime }$ . Hence

$$
{ \mathcal { C } } \subseteq T ( d ^ { \prime } ) \Longrightarrow { \mathcal { C } } \subseteq T ( d )
$$

is precisely partition consistency applied to the pair $( d ^ { \prime } , d )$ , and its contrapositive is absence consistency. ■

We next record the relation between richness and the backbone theorem (Theorem 15). Introduce the auxiliary backbone property $\alpha _ { \mathrm { b h } }$

T satisfies $\begin{array} { r l r } { \alpha _ { \mathrm { b h } } } & { { } \Longleftrightarrow } & { T _ { \mathrm { g l o b } } ^ { \eta } \subseteq T } \end{array}$ for some separation margin sequence $\eta .$

Since $T _ { \mathrm { g l o b } } ^ { \eta }$ is hierarchically rich for all separation margin $\eta , \alpha _ { \mathrm { b h } } \ : \triangleright \alpha _ { \mathrm { h r } }$

Proposition 32 Let $\alpha _ { 1 } \vartriangle \alpha _ { \mathrm { 1 } } \vartriangle \alpha _ { \mathrm { k c a } } \wedge \alpha _ { \mathrm { p c } } )$ and α<sub>2</sub> $\vartriangleright ( \alpha _ { \mathrm { s c a } } \wedge \alpha _ { \mathrm { c w c } } )$ . Then,

$$
( \alpha _ { 1 } \wedge \alpha _ { \mathrm { p r } } ) \equiv ( \alpha _ { 1 } \wedge \alpha _ { \mathrm { h r } } ) \equiv ( \alpha _ { 1 } \wedge \alpha _ { \mathrm { b h } } ) , \quad a n d \quad ( \alpha _ { 2 } \wedge \alpha _ { \mathrm { c w r } } ) \equiv ( \alpha _ { 2 } \wedge \alpha _ { \mathrm { p r } } ) \equiv ( \alpha _ { 2 } \wedge \alpha _ { \mathrm { h r } } ) \equiv ( \alpha _ { 2 } \wedge \alpha _ { \mathrm { b h } } ) .
$$

Proof We prove the equivalences by the sandwiching argument i.e, by showing

$$
( \alpha _ { 1 } \wedge \alpha _ { \mathrm { p r } } ) \ltimes ( \alpha _ { 1 } \wedge \alpha _ { \mathrm { h r } } ) \ltimes ( \alpha _ { 1 } \wedge \alpha _ { \mathrm { b h } } ) \quad \mathrm { a n d } \quad ( \alpha _ { 1 } \wedge \alpha _ { \mathrm { p r } } ) \ltimes ( \alpha _ { 1 } \wedge \alpha _ { \mathrm { h r } } ) \ltimes ( \alpha _ { 1 } \wedge \alpha _ { \mathrm { b h } } ) .
$$

At the beginning of the subsection, we already observed $\alpha _ { \mathrm { b h } } \triangleright \alpha _ { \mathrm { h r } } \triangleright \alpha _ { \mathrm { p r } } \triangleright \alpha _ { \mathrm { c w r } }$ while Lemma 54 gives the other direction:

$$
\alpha _ { \mathrm { s c a } } \wedge \alpha _ { \mathrm { p r } } \wedge \alpha _ { \mathrm { p c } } \triangleright \alpha _ { \mathrm { b h } } .
$$

The same argument with cluster-wise richness and cluster-wise consistency gives

$$
\alpha _ { \mathrm { s c a } } \wedge \alpha _ { \mathrm { c w r } } \wedge \alpha _ { \mathrm { c w c } } \triangleright \alpha _ { \mathrm { b h } } .
$$

The two chains of equivalences stated in the proposition follow by sandwiching.

The following corollary follows by applying Proposition 32 to $\alpha _ { 1 } = \alpha _ { \mathrm { s c a } } \wedge \alpha _ { \mathrm { p c } } \wedge \alpha _ { \mathrm { p e r m } } .$

Corollary 33 The canonical axiom system admits the equivalent formulation

$$
\alpha _ { \mathrm { a d m } } \equiv \alpha _ { \mathrm { s c a } } \wedge \alpha _ { \mathrm { b h } } \wedge \alpha _ { \mathrm { p c } } \wedge \alpha _ { \mathrm { p e r m } } .
$$

Corollary 33 is useful because partition richness itself is not naturally preserved by intersections, whereas a common positive-margin backbone is. Hence this corollary, combined with Proposition 35, proves Lemma 20(i).

## A.3 Robustness of the Structural Results

We now ask to what extent the structural results of Section 4 persist under the alternative requirements introduced above. To state these results compactly, we introduce notation for several order-theoretic properties of the class of methods induced by an axiom system.

Definition 34 For an axiom system α, we write

$\alpha \in \gamma _ { \Pi } \iff T _ { \sqcap L } \in { \mathcal { H } } _ { \alpha } ( { \mathcal { X } } )$ for every finite nonempty $L \subseteq { \mathcal { H } } _ { \alpha } ( { \mathcal { X } } )$

$\alpha \in \gamma _ { \sqcup } ^ { \mathrm { u p } } \iff T _ { \sqcup L } \in { \mathcal { H } } _ { \alpha } ( \chi )$ for every nonempty upward-directed $L \subseteq { \mathcal { H } } _ { \alpha } ( { \mathcal { X } } )$

$\alpha \in \gamma _ { \sqcup } \iff T _ { \sqcup L } \in { \mathcal { H } } _ { \alpha } ( \chi )$ for every nonempty pairwise-compatible $L \subseteq { \mathcal { H } } _ { \alpha } ( { \mathcal { X } } )$

$\alpha \in \gamma _ { \mathrm { e m } } \iff e v e r y T \in \mathcal { H } _ { \alpha } ( \chi )$ is refined by some maximal element of $\mathcal { H } _ { \alpha } ( \mathcal { X } )$

Thus, $\gamma _ { \Pi } , \gamma _ { \sqcup } ^ { \mathrm { u p } } , \gamma _ { \sqcup }$ are the sets of requirements/properties that satisfy closure under finite intersections, upward-directed unions, and compatible unions, respectively, while $\gamma _ { \mathrm { e m } }$ is the set of requirements/properties that admit the maximal elements of the induced poset.

Since every upward-directed family is pairwise compatible, we have $\gamma _ { \sqcup } \subseteq \gamma _ { \sqcup } ^ { \mathrm { u p } }$ . Moreover, closure under upward-directed unions implies that every refinement chain has an upper bound in the same class; hence Zorn’s lemma gives $\gamma _ { \sqcup } ^ { \mathrm { u p } } \subseteq \gamma _ { \mathrm { e m } }$ . Therefore,

$$
\gamma _ { \sqcup } \subseteq \gamma _ { \sqcup } ^ { \mathrm { u p } } \subseteq \gamma _ { \mathrm { e m } } .\tag{6}
$$

In addition to refinement on ultrametrics $\alpha _ { \mathrm { u h r } }$ and to the backbone-hierarchy property $\alpha _ { \mathrm { b h } }$ introduced earlier, we will use the following stronger backbone property: T satisfies the unit-backbone property $\alpha _ { \mathrm { b h } } ^ { 1 } \mathrm { i f } T _ { \mathrm { g l o b } } \subseteq T$

Proposition 35 (Closure principles) The closure properties indicated by checkmarks in the $\gamma _ { \sqcap } , \gamma _ { \sqcup } ^ { \mathrm { u p } }$ , and $\gamma _ { \sqcup }$ columns of Table 1 hold. Moreover, these closure properties are preserved under finite conjunctions: if $\alpha = \textstyle \bigwedge _ { i = 1 } ^ { k } \alpha _ { i }$ , then any of the three closure properties $( \gamma _ { \Gamma } , \gamma _ { \Gamma } ^ { \mathrm { u p } }$ , and $\gamma _ { \sqcup } )$ shared by all $\alpha _ { i }$ is also satisfied by α.

Proof We first establish the closure properties of the individual axioms, following the categories in Table 1.

Throughout this proof, L denotes a nonempty family of hierarchical clustering methods. Whenever a union $T _ { \sqcup L }$ is considered, L is assumed to be either upward directed (when studying $\gamma _ { \sqcup } ^ { \mathrm { u p } } )$ or pairwise compatible (when studying $\gamma _ { \sqcup } )$ . In either case, $T _ { \scriptscriptstyle \perp L }$ is hierarchyvalued by Lemma 19, because every upward-directed family is pairwise compatible.

Invariance properties. Scale invariance, order invariance, and permutation invariance are preserved under both intersections and unions. For example, if every $T \in L$ is scale invariant, then

$$
T _ { \sqcap L } ( \beta d ) = \bigcap _ { T \in L } T ( \beta d ) = \bigcap _ { T \in L } T ( d ) = T _ { \sqcap L } ( d ) ,
$$

and the same calculation with unions gives $T _ { \sqcup L } ( \beta d ) = T _ { \sqcup L } ( d )$ . The arguments for order and permutation invariance are identical. Therefore, $\alpha _ { \mathrm { s c a } } , \alpha _ { \mathrm { o r d } } , \alpha _ { \mathrm { p e r m } } \in \gamma _ { \sqcap } \cap \gamma _ { \sqcup } ^ { \mathrm { u p } } \cap \gamma _ { \sqcup }$

Ultrametric properties. Refinement on ultrametrics is also preserved under intersections and compatible unions. Indeed, if every $T \in L$ satisfies $\Psi _ { u } \subseteq T ( u )$ , then

$$
\Psi _ { u } \subseteq \bigcap _ { T \in L } T ( u ) \subseteq \bigcup _ { T \in L } T ( u ) .
$$

Similarly, if every $T \in L$ is exact on ultrametrics, then $T _ { \Pi L } ( u ) = T _ { \sqcup L } ( u ) = \Psi _ { u }$ . Hence $\alpha _ { \mathrm { u h r } } , \alpha _ { \mathrm { s u h r } } \in \gamma _ { \mathsf {sqcap } } \cap \gamma _ { \mathsf { \sqcup } } ^ { \mathsf { u p } } \cap \gamma _ { \mathsf { \sqcup } }$

Richness properties. Cluster-wise, partition, and hierarchical richness are preserved by every nonempty compatible union. To see this, fix any $T _ { 0 } \in L$ . Any dissimilarity realizing the relevant richness requirement for $T _ { 0 }$ also realizes it for $T _ { \sqcup L } .$ , because $T _ { 0 } ( d ) \subseteq T _ { \sqcup L } ( d )$ Therefore $\alpha _ { \mathrm { c w r } } , \alpha _ { \mathrm { p r } } , \alpha _ { \mathrm { h r } } \in \gamma _ { \scriptscriptstyle \sqcup } \subseteq \gamma _ { \scriptscriptstyle \sqcup } ^ { \mathrm { u p } }$ . The same argument does not apply to strict hierarchical richness, because additional clusters contributed by other methods in $L$ may prevent the union from realizing a prescribed hierarchy exactly.

Consistency properties. We begin with cluster-wise consistency, so we let L be a finite nonempty family of cluster-wise consistent methods.

Suppose that $C \in T _ { \Pi L } ( d )$ . Hence $C \in T ( d )$ for every $T \in L$ , and thus, for every $\{ C \} -$ strengthening $d ^ { \prime }$ of $d ,$ we have $C \in T ( d ^ { \prime } )$ for every $T \in L$ . This means that $C \in T _ { \sqcap L } ( d ^ { \prime } )$ which proves $\alpha _ { \mathrm { c w c } } \in \gamma _ { \Pi }$

Now suppose that L is pairwise compatible and let $C \in T _ { \sqcup L } ( d )$ . Then $C \in T _ { 0 } ( d )$ for some $T _ { 0 } \in L$ . Cluster-wise consistency of $T _ { 0 }$ ensures $C \in T _ { 0 } ( d ^ { \prime } )$ for any tCu-strengthening $d ^ { \prime }$ of $d ,$ and hence $C \in T _ { 0 } ( d ^ { \prime } ) \subseteq T _ { \sqcup L } ( d ^ { \prime } )$ . Thus $\alpha _ { \mathrm { c w c } } \in \gamma _ { \Pi } \cap \gamma _ { \sqcup } ^ { \mathrm { u p } } \cap \gamma _ { \sqcup }$ . As $\alpha _ { \mathrm { h c } } \equiv \alpha _ { \mathrm { c w c } }$ by Proposition 31, the same conclusions hold for hierarchical consistency.

Partition consistency is likewise preserved under intersections. Indeed, let C be a partition such that ${ \mathcal { C } } \subseteq T _ { \sqcap L } ( d )$ . Then ${ \mathcal { C } } \subseteq T ( d )$ for every $T \in L ,$ , so every C-strengthening $d ^ { \prime }$ of d satisfies $\mathcal { C } \subseteq T ( d ^ { \prime } )$ for every $T \in L$ , and therefore $\mathcal { C } \subseteq T _ { \sqcap L } ( d ^ { \prime } )$ . This proves $\alpha _ { \mathrm { p c } } \in \gamma _ { \Gamma }$

To prove $\alpha _ { \mathrm { p c } } \in \gamma _ { \perp } ^ { \mathrm { u p } }$ , an additional argument is needed because the diferent clusters of a partition may initially come from diferent methods. Let L be upward directed and suppose ${ \mathcal { C } } = \{ C _ { 1 } , \ldots , C _ { m } \} \subseteq T _ { \sqcup L } ( d )$ . For each $j \in [ m ]$ , choose $T _ { j } \in L$ such that $C _ { j } \in T _ { j } ( d )$ . Since C is finite and L is upward directed, there exists $T ^ { \star } \in L$ refining all $T _ { 1 } , \ldots , T _ { m }$ . Hence ${ \mathcal { C } } \subseteq T ^ { \star } ( d )$ If $d ^ { \prime }$ is a C-strengthening of d, partition consistency of $T ^ { \star }$ yields ${ \mathcal { C } } \subseteq T ^ { \star } ( d ^ { \prime } ) \subseteq T _ { \sqcup L } ( d ^ { \prime } )$ . Thus $\alpha _ { \mathrm { p c } } \in \gamma _ { \perp } ^ { \mathrm { u p } }$ . Finally, $\alpha _ { \mathrm { a b c } } \equiv \alpha _ { \mathrm { p c } }$ , so absence consistency has the same closure properties.

Backbone property. Let $L = \{ T _ { 1 } , \dots , T _ { m } \}$ be finite and suppose that each $T _ { i }$ satisfies the backbone property with margin sequence $\pmb { \eta } ^ { ( i ) }$ , i.e., $T _ { \mathrm { g l o b } } ^ { \pmb { \eta } ^ { ( i ) } } \subseteq T _ { i }$ . Define the coordinate-wise minimum $\begin{array} { r } { \eta _ { s } : = \operatorname* { m i n } _ { i \in [ m ] } \eta _ { s } ^ { ( i ) } } \end{array}$ . Because L is finite, η is a separation margin sequence, and $T _ { \mathrm { g l o b } } ^ { \eta } \ \subseteq T _ { i }$ for every $i \in [ m ]$ . Therefore $T _ { \mathrm { g l o b } } ^ { \eta } \subseteq T _ { \sqcap L }$ , which proves $\alpha _ { \mathrm { b h } } \in \gamma _ { \Gamma }$ . For any nonempty union, the backbone of an arbitrary fixed member $T _ { 0 } \in L$ is contained in $T _ { \sqcup L } ;$ hence $\alpha _ { \mathrm { b h } } \in \gamma _ { \Gamma } \cap \gamma _ { \scriptscriptstyle \perp } ^ { \mathrm { u p } } \cap \gamma _ { \scriptscriptstyle \perp }$

This proves all positive closure claims for the individual axioms in the first three structural columns of Table 1.

Finally, we prove closure under conjunction. Let $\alpha = \textstyle \bigwedge _ { i = 1 } ^ { k } \alpha _ { i }$ . Then $\begin{array} { r } { \mathcal { H } _ { \alpha } ( \mathcal { X } ) = \bigcap _ { i = 1 } ^ { k } \mathcal { H } _ { \alpha _ { i } } ( \mathcal { X } ) } \end{array}$ Suppose that $\alpha _ { i } \in \gamma _ { \Pi }$ for every $i \in [ k ]$ , and let $L \subseteq { \mathcal { H } } _ { \alpha } ( { \mathcal { X } } )$ be finite and nonempty. Then $L \subseteq { \mathcal { H } } _ { \alpha _ { i } } ( { \mathcal { X } } )$ for every i. Moreover, because each $\alpha _ { i }$ satisfies $\gamma _ { \Pi }$ , we have $T _ { \sqcap L } \in \mathcal { H } _ { \alpha _ { i } } ( \mathcal { X } )$

for every $i \in [ k ]$ . Therefore, $\begin{array} { r } { T _ { \sqcap L } \in \bigcap _ { i = 1 } ^ { k } \mathcal { H } _ { \alpha _ { i } } ( \chi ) = \mathcal { H } _ { \alpha } ( \chi ) } \end{array}$ , proving that $\alpha \in \gamma _ { \Pi }$ . The arguments for $\gamma _ { \sqcup } ^ { \mathrm { u p } }$ and $\gamma _ { \mathrm { L } }$ are identical, replacing $T _ { \sqcap L }$ by $T _ { \scriptscriptstyle \perp L }$ ■

Corollary 36 (Closure of the admissible classes) The canonical admissible class satisfies $\alpha _ { \mathrm { a d m } } \in \gamma _ { \Pi } \cap \gamma _ { \sqcup } ^ { \mathrm { u p } }$ . Moreover, $\mathcal { H } _ { \alpha _ { \mathrm { a d m + } } } ( \mathcal { X } )$ is closed under arbitrary nonempty intersections and under nonempty upward-directed unions.

Proof By Corollary 33, $\alpha _ { \mathrm { a d m } } \equiv \alpha _ { \mathrm { s c a } } \wedge \alpha _ { \mathrm { b h } } \wedge \alpha _ { \mathrm { p c } } \wedge \alpha _ { \mathrm { p e r m } }$ . Each of these four requirements belongs to $\gamma _ { \Pi } .$ , so Proposition 35 gives $\alpha _ { \mathrm { a d m } } \in \gamma _ { \Gamma }$ . Directed-union closure follows directly from $\alpha _ { \mathrm { a d m } } = \alpha _ { \mathrm { s c a } } \wedge \alpha _ { \mathrm { p r } } \wedge \alpha _ { \mathrm { p c } } \wedge \alpha _ { \mathrm { p e r m } } ,$ , as each of these requirements belongs to $\gamma _ { \sqcup } ^ { \mathrm { u p } }$

For strong admissibility, recall that $\alpha _ { \mathrm { a d m } + } \equiv \alpha _ { \mathrm { s c a } } \wedge$ α<sub>suhr</sub> $\land \alpha _ { \mathrm { p c } } \land \alpha _ { \mathrm { p e r m } }$ . Scale invariance, exactness on ultrametrics, partition consistency, and permutation invariance are all preserved under arbitrary nonempty intersections. Exactness on ultrametrics also guarantees partition richness, so the resulting method remains strongly admissible. Directed-union closure follows from Proposition 35.

Proposition 35 gives a compact way to transfer the order-theoretic arguments of Section 4 to alternative axiom systems. The next result records the consequences that are most relevant for comparison with the canonical framework.

Proposition 37 (Robustness under alternative axioms) Let α be any conjunction of the axioms introduced in Section A.1. Then:

(i) Achievability. The set $\mathcal { H } _ { \alpha } ( \mathcal { X } )$ is nonempty. In particular, $T _ { \mathrm { g l o b } } , T _ { \mathrm { l o c } } , T _ { \mathrm { S L } } \in \mathcal { H } _ { \alpha } ( \chi )$

(ii) Maximal extensions. If strict hierarchical richness is not among the requirements defining α or if exactness on ultrametrics is required, then every $T \in { \mathcal { H } } _ { \alpha } ( { \mathcal { X } } )$ is refined by a maximal member of $\mathcal { H } _ { \alpha } ( \mathcal { X } )$

(iii) Finite intersections. If $\alpha \mapsto \alpha _ { \mathrm { b h } }$ and either strict hierarchical richness is not required or exactness on ultrametrics is required, then $\mathcal { H } _ { \alpha } ( \mathcal { X } )$ is closed under finite nonempty intersections.

(iv) Diversity versus order invariance. If order invariance is not among the requirements defining $\alpha ,$ then $\mathcal { H } _ { \alpha } ( \mathcal { X } )$ has uncountable height, width, and cellularity. If order invariance is required, then $\mathcal { H } _ { \alpha } ( \mathcal { X } )$ is finite.

(v) Least elements. Let $\alpha _ { \mathrm { b h } } ^ { 1 }$ denote the unit-backbone property $T _ { \mathrm { g l o b } } \subseteq T$ . We have

$$
\alpha _ { \mathrm { u h r } } \wedge \alpha _ { \mathrm { p c } } \triangleright \alpha _ { \mathrm { b h } } ^ { 1 } \quad a n d \quad \alpha _ { \mathrm { o r d } } \wedge \alpha _ { \mathrm { p c } } \wedge \alpha _ { \mathrm { p r } } \triangleright \alpha _ { \mathrm { b h } } ^ { 1 } .
$$

Moreover, if $\alpha \vartriangleright \alpha _ { \mathrm { b h } } ^ { 1 }$ , then $T _ { \mathrm { g l o b } }$ is the least element of the poset $( { \mathcal { H } } _ { \alpha } ( { \mathcal { X } } ) , { \underline { { \subseteq } } } )$

Proof For (i), we will show in later sections that the three methods $T _ { \mathrm { g l o b } } , T _ { \mathrm { l o c } } $ , and $T _ { \mathrm { S L } }$ are order invariant, cluster-wise consistent, exact on ultrametrics, and permutation invariant. Exactness on ultrametrics implies all three richness requirements as well as refinement on ultrametrics, and cluster-wise consistency implies partition consistency. Thus these methods satisfy every requirement introduced in this paper.

For (ii), with the exception of strict hierarchical richness, every other requirement is preserved by directed unions according to Proposition 35. If exactness on ultrametrics is required, strict hierarchical richness is redundant. Hence, α satisfies $\gamma _ { \sqcup } ^ { \mathrm { u p } }$ ; thus every nonempty chain has an upper bound in $\mathcal { H } _ { \alpha } ( \mathcal { X } )$ , and Zorn’s lemma gives a maximal refinement above every $T \in { \mathcal { H } } _ { \alpha } ( { \mathcal { X } } )$ (see the relationship (6)).

For (iii), all requirements other than the richness variants are preserved by finite intersections. The assumption $\alpha \vartriangle { \omega } \subset \alpha _ { \mathrm { b h } }$ ensures the existence of a common backbone to every finite intersection, and $\alpha _ { \mathrm { b h } } \triangleright \alpha _ { \mathrm { h r } } \triangleright \alpha _ { \mathrm { p r } } \triangleright \alpha _ { \mathrm { c w r } }$ restores every non-strict richness requirement. If strict hierarchical richness is required together with exactness on ultrametrics, exactness is preserved by intersections and implies strict hierarchical richness.

For (iv), suppose first that order invariance is not required. By Lemma 51 $T _ { \mathrm { s t a b l e } }$ satisfies all the requirements except possibly order invariance, and because power transformations preserve these requirements by Corollary 26, the methods $\{ T _ { \mathrm { s t a b l e } } ^ { ( p ) } : p > 0 \}$ also satisfy them. Lemma 13 shows that they are pairwise incompatible, yielding uncountable width and cellularity. To establish the uncountable height, for $a \in ( 0 , 1 ]$ , define a margin sequence by $\pmb { \eta } ^ { ( a ) } = a \cdot \mathbf { 1 }$ and set

$$
S _ { a } ( d ) : = T _ { \mathrm { g l o b } } ( d ) \cup T _ { \mathrm { l o c } } ^ { \eta ^ { ( a ) } } ( d ) .
$$

Both $T _ { \mathrm { g l o b } } ( d )$ and $T _ { \mathrm { l o c } } ^ { \eta ^ { ( a ) } } ( d )$ are subhierarchies of $T _ { \mathrm { l o c } } ( d )$ , so the union is a hierarchy. By Lemma 46 and Proposition 35, $S _ { a }$ is scale invariant, cluster-wise consistent, and permutation invariant. Moreover, for every ultrametric $u ,$

$$
\Psi _ { u } = T _ { \mathrm { g l o b } } ( u ) \subseteq S _ { a } ( u ) \subseteq T _ { \mathrm { l o c } } ( u ) = \Psi _ { u } ,
$$

so $S _ { a }$ is exact on ultrametrics and therefore satisfies every requirement except possibly order invariance.

The family $\{ S _ { a } : a \in ( 0 , 1 ] \}$ is an increasing chain. It is strict: for $0 < a < b \leqslant 1$ , choose $r \in [ a , b ) , L > 1 / r$ , and distinct $x _ { 1 } , x _ { 2 } , x _ { 3 } , z \in \mathcal { X }$ . With $\boldsymbol { C } = \{ x _ { 1 } , x _ { 2 } , x _ { 3 } \}$ , set

$$
d ( x _ { 1 } , x _ { 2 } ) = d ( x _ { 1 } , x _ { 3 } ) = r , \quad d ( x _ { 2 } , x _ { 3 } ) = r L , \quad d ( x _ { 1 } , z ) = 1 , \quad d ( x _ { 2 } , z ) = d ( x _ { 3 } , z ) = L ,
$$

and, for every remaining w $\notin \ : C \cup \ : \{ z \}$ , set $d ( x _ { 1 } , w ) = 1$ and $d ( x _ { 2 } , w ) = d ( x _ { 3 } , w ) = L$ Complete the remaining dissimilarities arbitrarily. Then $\tau ( d , C ) = r$ and $\varrho ( d , C ) = r L > 1$ 2 so $C \notin S _ { a } ( d )$ but $C \in S _ { b } ( d )$

Conversely, suppose order invariance is required. We say that two dissimilarities $d , d ^ { \prime } \in$ $\mathcal { D } ( \mathcal { X } )$ are order-equivalent, written $d \sim d ^ { \prime }$ , if they induce the same weak ordering of the $\binom { n } { 2 }$ unordered pairs, that is

$$
d ( x _ { 1 } , y _ { 1 } ) \ \leqslant \ d ( x _ { 2 } , y _ { 2 } ) \iff d ^ { \prime } ( x _ { 1 } , y _ { 1 } ) \ \leqslant \ d ^ { \prime } ( x _ { 2 } , y _ { 2 } ) \quad \ { \mathrm { ~ f o r ~ a l l ~ } } ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) \in { \mathcal { X } } .
$$

Each equivalence class corresponds to a weak ordering of the $\binom { n } { 2 }$ pairs, and there are only finitely many such weak orderings, so the quotient space $\mathcal { D } ( \mathcal { X } ) / \sim$ is finite. If $d \sim d ^ { \prime }$ , a strictly increasing interpolation of the finitely many distinct values taken by d and $d ^ { \prime }$ gives $d ^ { \prime } = g \circ d ;$ hence an order invariance hierarchical method is constant on each equivalence class. Because $\tau ( \mathcal { X } )$ is finite, $| \mathcal { H } _ { \alpha _ { \mathrm { o r d } } } ( \mathcal { X } ) | ~ \leqslant ~ | \mathcal { T } ( \mathcal { X } ) | ^ { | \mathcal { D } ( \mathcal { X } ) / { \sim } | } ~ < ~ \infty$ , and therefore every subclass $\mathcal { H } _ { \alpha } ( \mathcal { X } )$ satisfying order invariance is finite.

For (v), the first claim $\alpha _ { \mathrm { u h r } } \wedge \alpha _ { \mathrm { p c } }  \alpha _ { \mathrm { b h } } ^ { 1 }$ is established in Lemma 55, whereas the second claim is established in Lemma 56. Finally, if $\alpha \vartriangleright \alpha _ { \mathrm { b h } } ^ { 1 }$ , then every $T \in { \mathcal { H } } _ { \alpha } ( { \mathcal { X } } )$ refines $T _ { \mathrm { g l o b } } ,$ while $T _ { \mathrm { g l o b } } \in \mathcal { H } _ { \alpha } ( { \mathcal { X } } )$ by (i). Therefore $T _ { \mathrm { g l o b } }$ is the least element of $\mathcal { H } _ { \alpha } ( \mathcal { X } )$

## Appendix B. Relational Properties and Axiom-Preserving Transformations

Section 5.2 introduced transformations of dissimilarities and the notion of preservation of an axiom under preprocessing. In this appendix, we formalize a common mechanism behind several of the preservation results stated there. Indeed, many of the axioms considered in this paper compare the outputs of a method on two inputs that are related in a prescribed way. Scale invariance, for instance, compares $T ( d )$ and $T ( \beta d )$ , while partition consistency compares $T ( d )$ and $T ( d ^ { \prime } )$ when $d ^ { \prime }$ is a strengthening of d. Relational properties provide a common language for such requirements.

## B.1 Relational Properties

Definition 38 (Relational property) A relational property α is specified $b y$

• an input relation $R ^ { \alpha } \subseteq { \mathcal { D } } ( \mathcal { X } ) \times { \mathcal { D } } ( \mathcal { X } )$ q;

• an output relation $S ^ { \alpha } \subseteq { \mathcal { T } } ( \mathcal { X } ) \times { \mathcal { T } } ( \mathcal { X } )$

A hierarchical clustering method $T \in { \mathcal { H } } ( { \mathcal { X } } )$ satisfies the relational property α if

$$
( d , d ^ { \prime } ) \in R ^ { \alpha } \quad \Longrightarrow \quad \left( T ( d ) , T ( d ^ { \prime } ) \right) \in S ^ { \alpha } .
$$

More generally, we write $\alpha \in \gamma _ { \mathrm { r p } }$ if the requirement α can be expressed as a relational property or as a conjunction of relational properties. This is the meaning of the $\gamma _ { \mathrm { r p } }$ column in Table 1.

The usefulness of this formulation for preprocessing comes from the following simple principle: to preserve a relational property, it is suficient for the transformation to preserve its input relation.

Proposition 39 (Relational preservation principle) Let α be a relational property with input relation $R ^ { \alpha }$ , and let $\mu : { \mathcal { D } } ( { \mathcal { X } } ) \to { \mathcal { D } } ( { \mathcal { X } } )$ be a transformation. If

$$
( d , d ^ { \prime } ) \in R ^ { \alpha } \quad \Longrightarrow \quad \left( \mu ( d ) , \mu ( d ^ { \prime } ) \right) \in R ^ { \alpha } ,
$$

then $\mu$ preserves α.

Proof Let $T \in { \mathcal { H } } _ { \alpha } ( { \mathcal { X } } )$ . If $( d , d ^ { \prime } ) \in R ^ { \alpha }$ , the assumption gives $( \mu ( d ) , \mu ( d ^ { \prime } ) ) \in R ^ { \alpha }$ . Because $T$ satisfies $\alpha ,$ we have $\smash { \big ( T ( \mu ( d ) ) , T ( \mu ( d ^ { \prime } ) ) \big ) } \in S ^ { \alpha }$ . Thus $T \circ \mu$ satisfies $\alpha ,$ , and therefore $\mu$ preserves α.

We will also use the following elementary observation when an axiom is represented as a conjunction of relational properties.

Lemma 40 (Preservation under conjunction) Let $\left( \alpha _ { j } \right) _ { j \in J }$ be a family ofrequirements. $I f$ a transformation $\mu$ preserves $\alpha _ { j }$ for every $j \in J ,$ then it preserves the conjunction $\textstyle \bigwedge _ { j \in J } \alpha _ { j }$

Proof Let $T$ satisfy $\textstyle \bigwedge _ { j \in J } \alpha _ { j }$ . Then $T$ satisfies every $\alpha _ { j }$ . Because $\mu$ preserves each $\alpha _ { j }$ , the composition $T \circ \mu$ also satisfies every $\alpha _ { j }$ , and hence satisfies their conjunction. ■

Consequently, when $\alpha \in \gamma _ { \mathrm { r p } } .$ , preservation of α can be verified by checking that µ preserves the input relation of each relational property appearing in its representation. Notice that partition richness is handled diferently in Lemma 25: its preservation follows from surjectivity of the transformation rather than from the relational principle above.

## B.2 Relational Formulations of the Axioms

We now give relational representations of the axioms marked by a checkmark in the $\gamma _ { \mathrm { r p } }$ column of Table 1. These representations also make explicit which structure of the input must be preserved by a preprocessing transformation.

We denote the diagonal relation on hierarchies by

$$
\Delta _ { T } : = \{ ( \Psi , \Psi ) : \Psi \in \mathcal { T } ( \mathcal { X } ) \} .
$$

Scale and order invariance. Scale invariance is the relational property defined by

$$
\begin{array} { r } { R ^ { \alpha _ { \mathrm { s c a } } } = \{ ( d , \beta d ) : d \in { \mathcal D } ( { \mathcal X } ) , \ \beta > 0 \} , \qquad S ^ { \alpha _ { \mathrm { s c a } } } = \Delta \tau . } \end{array}\tag{7}
$$

Indeed, the corresponding implication is exactly $T ( \beta d ) = T ( d )$

Likewise, order invariance is relational. Let

$$
\mathcal { G } _ { \uparrow } : = \{ g : \mathbb { R } _ { \geqslant 0 } \to \mathbb { R } _ { \geqslant 0 } : g ( 0 ) = 0 , \ g \ \mathrm { s t r i c t l y ~ i n c r e a s i n g } \} .
$$

Then order invariance is obtained from

$$
R ^ { { \alpha } _ { \mathrm { o r d } } } = \{ ( d , g \circ d ) : d \in { \mathcal { D } } ( { \mathcal { X } } ) , \ g \in { \mathcal { G } } _ { \uparrow } \} , \qquad S ^ { { \alpha } _ { \mathrm { o r d } } } = \Delta _ { \mathcal { T } } .\tag{8}
$$

Permutation invariance. Here the required output relation depends on the permutation. For each permutation $\phi \in \Pi ( { \mathcal { X } } )$ , define the relational property $\alpha _ { \mathrm { p e r m } } ^ { \phi }$ by

$$
R ^ { \alpha _ { \mathrm { p e r m } } ^ { \phi } } = \{ ( d , d _ { \phi } ) : d \in { \mathcal { D } } ( { \mathcal { X } } ) \} , \qquad S ^ { \alpha _ { \mathrm { p e r m } } ^ { \phi } } = \{ ( \Psi , \phi \cdot \Psi ) : \Psi \in { \mathcal { T } } ( { \mathcal { X } } ) \} .\tag{9}
$$

Then $\begin{array} { r } { \alpha _ { \mathrm { p e r m } } \equiv \bigwedge _ { \phi \in \Pi ( \mathcal { X } ) } \alpha _ { \mathrm { p e r m } } ^ { \phi } } \end{array}$ . Indeed, satisfying every $\alpha _ { \mathrm { p e r m } } ^ { \phi }$ is precisely the requirement

$$
T ( d _ { \phi } ) = \phi \cdot T ( d ) \qquad \mathrm { f o r ~ e v e r y ~ } d \in { \mathcal { D } } ( { \mathcal { X } } ) , \ \phi \in \Pi ( { \mathcal { X } } ) .
$$

Consistency. Fix a partition $\mathcal { C } \in \mathcal { P } ( \mathcal { X } )$ . Define $\alpha _ { \mathrm { p c } } ^ { \mathcal { C } }$ by

$$
\begin{array} { r l } & { R ^ { \alpha _ { \mathrm { p c } } ^ { c } } = \left\{ ( d , d ^ { \prime } ) \in { \mathcal { D } } ( { \mathcal { X } } ) ^ { 2 } : d ^ { \prime } \mathrm { ~ i s ~ a ~ } { \mathcal { C } } \mathrm { - s t r e n g t h e n i n g ~ o f ~ } d \right\} , } \\ & { S ^ { \alpha _ { \mathrm { p c } } ^ { c } } = \left\{ ( \Psi , \Psi ^ { \prime } ) \in { \mathcal { T } } ( { \mathcal { X } } ) ^ { 2 } : { \mathcal { C } } \subseteq \Psi \Longrightarrow { \mathcal { C } } \subseteq \Psi ^ { \prime } \right\} . } \end{array}\tag{10}
$$

Partition consistency is therefore $\begin{array} { r } { \alpha _ { \mathrm { p c } } \equiv \bigwedge _ { \mathcal { C } \in \mathcal { P } ( \mathcal { X } ) } \alpha _ { \mathrm { p c } } ^ { \mathcal { C } } } \end{array}$

Cluster-wise consistency admits the analog representation. For every nonempty $C \subseteq { \mathcal { X } }$ let $\alpha _ { \mathrm { c w c } } ^ { C }$ be defined by

$$
\begin{array} { r l } & { R ^ { \alpha _ { \mathrm { c w c } } ^ { C } } = \left\{ ( d , d ^ { \prime } ) \in { \mathcal { D } } ( { \mathcal { X } } ) ^ { 2 } : d ^ { \prime } \mathrm { ~ i s ~ a ~ } \{ C \} \mathrm { - s t r e n g t h e n i n g ~ o f ~ } d \right\} , } \\ & { S ^ { \alpha _ { \mathrm { c w c } } ^ { C } } = \left\{ ( \Psi , \Psi ^ { \prime } ) \in { \mathcal { T } } ( { \mathcal { X } } ) ^ { 2 } : C \in \Psi \Longrightarrow C \in \Psi ^ { \prime } \right\} . } \end{array}\tag{11}
$$

Hence $\alpha _ { \mathrm { c w c } } \equiv \bigwedge _ { \mathcal { D } \ne C \subseteq \mathcal { X } } \alpha _ { \mathrm { c w c } } ^ { C }$ . By Proposition 31, $\alpha _ { \mathrm { h c } } \equiv \alpha _ { \mathrm { c w c } }$ and $\alpha _ { \mathrm { a b c } } \equiv \alpha _ { \mathrm { p c } } .$ , so hierarchical consistency and absence consistency are also representable as conjunctions of relational properties.

Ultrametric properties. For a hierarchy $\Psi \in { \mathcal { T } } ( { \mathcal { X } } )$ , let $\mathcal { U } _ { \Psi } : = \{ u \in \mathcal { U } ( \mathcal { X } ) : \Psi _ { u } = \Psi \}$ be the set of ultrametrics inducing $\Psi$ . To represent refinement on ultrametrics, define $\alpha _ { \mathrm { u h r } } ^ { \Psi }$ by

$$
\begin{array} { r } { R ^ { \alpha _ { \mathrm { u h r } } ^ { \Psi } } = \{ ( u , u ) : u \in \mathcal { U } _ { \Psi } \} , \qquad S ^ { \alpha _ { \mathrm { u h r } } ^ { \Psi } } = \{ ( \Psi ^ { \prime } , \Psi ^ { \prime } ) : \Psi ^ { \prime } \in \mathcal { T } ( \mathcal { X } ) , \ \Psi \subseteq \Psi ^ { \prime } \} . } \end{array}\tag{12}
$$

Then $\begin{array} { r } { \alpha _ { \mathrm { u h r } } \equiv \bigwedge _ { \Psi \in \mathcal { T } ( \mathcal { X } ) } \alpha _ { \mathrm { u h r } } ^ { \Psi } . } \end{array}$

Exactness on ultrametrics is obtained by strengthening the output relation. For each $\Psi \in { \mathcal { T } } ( { \mathcal { X } } )$ , define $\alpha _ { \mathrm { s u h r } } ^ { \Psi }$ by

$$
R ^ { \alpha _ { \mathrm { s u h r } } ^ { \Psi } } = \{ ( u , u ) : u \in \mathcal { U } _ { \Psi } \} , \qquad S ^ { \alpha _ { \mathrm { s u h r } } ^ { \Psi } } = \{ ( \Psi , \Psi ) \} .\tag{13}
$$

Thus $\begin{array} { r } { \alpha _ { \mathrm { s u h r } } \equiv \bigwedge _ { \Psi \in \mathcal { T } ( \mathcal { X } ) } \alpha _ { \mathrm { s u h r } } ^ { \Psi } . } \end{array}$

These representations establish the corresponding entries in the $\gamma _ { \mathrm { r p } }$ column of Table 1. In particular, as $\alpha _ { \mathrm { a d m + } } \equiv \alpha _ { \mathrm { s c a } } \wedge \alpha _ { \mathrm { s u h r } } \wedge \alpha _ { \mathrm { p c } } \wedge \alpha _ { \mathrm { p e r m } } , \alpha _ { \mathrm { a d m + } }$ also belongs to $\gamma _ { \mathrm { r p } } .$

## B.3 Applications to Axiom-Preserving Transformations

Proof [Proof of Lemma 25] For scale invariance, let $( d , \beta d ) \in R ^ { \alpha _ { \mathrm { s c a } } } . \mathrm { B y }$ assumption, there exists $\beta ^ { \prime } > 0$ such that $\mu ( \beta d ) = \beta ^ { \prime } \mu ( d )$ . Hence $\bigl ( \mu ( d ) , \mu ( \beta d ) \bigr ) = \bigl ( \mu ( d ) , \beta ^ { \prime } \mu ( d ) \bigr ) \in R ^ { \alpha _ { \mathrm { s c a } } }$ , and Proposition 39 ensures that µ preserves scale invariance.

Partition richness is the one property in the lemma that we verify directly, as it is not a relational property. Let $T$ be partition rich and let $\mathcal { C } \in \mathcal { P } ( \mathcal { X } )$ . There exists $d _ { 0 } \in \mathcal { D } ( \mathcal { X } )$ such that ${ \mathcal { C } } \subseteq T ( d _ { 0 } )$ . Because $\mu$ is surjective, choose $d \in { \mathcal { D } } ( { \mathcal { X } } )$ such that $\mu ( d ) = d _ { 0 }$ . Then ${ \mathcal { C } } \subseteq T ( \mu ( d ) )$ , so $T \circ \mu$ is partition rich.

For partition consistency, fix $\mathcal { C } \in \mathcal { P } ( \mathcal { X } )$ . The assumption states precisely that

$$
( d , d ^ { \prime } ) \in R ^ { \alpha _ { \mathrm { p c } } ^ { c } } \quad \Longrightarrow \quad \left( \mu ( d ) , \mu ( d ^ { \prime } ) \right) \in R ^ { \alpha _ { \mathrm { p c } } ^ { c } } ,
$$

and thus Proposition 39 shows that µ preserves $\alpha _ { \mathrm { p c } } ^ { \mathcal { C } }$ . Because C was arbitrary, Lemma 40 gives preservation of $\alpha _ { \mathrm { { p c } } }$

For permutation invariance, fix $\phi \in \Pi ( { \mathcal { X } } )$ . If $( d , d _ { \phi } ) \in R ^ { \alpha _ { \mathrm { p e r m } } ^ { \phi } }$ , then $\mu ( d _ { \phi } ) = \mu ( d ) _ { \phi } .$ and therefore $\displaystyle \bigl ( \mu ( d ) , \mu ( d _ { \phi } ) \bigr ) = \bigl ( \mu ( d ) , \mu ( d ) _ { \phi } \bigr ) \in R ^ { \alpha _ { \mathrm { p e r m } } ^ { \phi } }$ . Proposition 39 and the conjunction Lemma 40 give preservation of $\alpha _ { \mathrm { p e r m } }$

Finally, fix $\Psi \in { \mathcal { T } } ( { \mathcal { X } } )$ and $u \in \mathcal { U } _ { \Psi }$ . By assumption, $\mu ( u )$ is an ultrametric satisfying $\Psi _ { \mu ( u ) } = \Psi _ { u } = \Psi$ . Thus $\mu ( u ) \in \mathcal { U } _ { \Psi }$ , and hence $( u , u ) \in R ^ { \alpha _ { \mathrm { s u h r } } ^ { \Psi } } \implies \left( \mu ( u ) , \mu ( u ) \right) \in R ^ { \alpha _ { \mathrm { s u h r } } ^ { \Psi } }$ Again, Proposition 39 and Lemma 40 prove preservation of exactness on ultrametrics.

Proof [Proof of Corollary 26] The condition $g ( 0 ) = 0$ , together with $g ( t ) > 0$ for $t > 0$ ensures that $\mu _ { g } ( d ) \in { \mathcal { D } } ( { \mathcal { X } } )$ whenever $d \in { \mathcal { D } } ( { \mathcal { X } } )$

Suppose first that g is strictly increasing. Then g preserves all inequalities defining a C-strengthening. Hence, whenever $d ^ { \prime }$ is a C-strengthening of $d , \mu _ { g } ( d ^ { \prime } )$ is a C-strengthening of $\mu _ { g } ( d )$ . By Lemma 25, $\mu _ { g }$ therefore preserves partition consistency.

Moreover, if u is an ultrametric, then

$$
g ( u ( x , y ) ) \ \leqslant \ g ( \operatorname* { m a x } \{ u ( x , z ) , u ( y , z ) \} ) \ = \ \operatorname* { m a x } \{ g ( u ( x , z ) ) , g ( u ( y , z ) ) \} ,
$$

so $\mu _ { g } ( u )$ is an ultrametric. Since g is strictly increasing, it preserves the ordering of all pairwise dissimilarities, and therefore u and $\mu _ { g } ( u )$ induce the same hierarchy. Thus Lemma 25 also gives preservation of exactness on ultrametrics.

For every permutation $\phi \in \Pi ( { \mathcal { X } } )$

$$
\mu _ { g } ( d _ { \phi } ) ( x , y ) = g \big ( d ( \phi ^ { - 1 } ( x ) , \phi ^ { - 1 } ( y ) ) \big ) = \big ( \mu _ { g } ( d ) \big ) _ { \phi } ( x , y ) ,
$$

and hence $\mu _ { g } ( d _ { \phi } ) = \left( \mu _ { g } ( d ) \right) _ { \phi }$ . Therefore $\mu _ { g }$ preserves permutation invariance.

If g is surjective, let $d _ { 0 } \in { \mathcal { D } } ( { \mathcal { X } } )$ . For every unordered pair $\{ x , y \} \subseteq { \mathcal { X } }$ with x $\neq y .$ , choose $t _ { x y } > 0$ such that $g ( t _ { x y } ) = d _ { 0 } ( x , y )$ . Such a positive preimage exists because $d \mathrm { _ 0 } ( x , y ) > 0 ,$ g is surjective, and $g ( 0 ) = 0$ . Define $d ( x , x ) : = 0$ , and $d ( x , y ) = d ( y , x ) : = t _ { x y }$ for $x \neq y$ . Then $d \in { \mathcal { D } } ( { \mathcal { X } } )$ and $\mu _ { g } ( d ) = d _ { 0 }$ . Hence $\mu _ { g }$ is surjective and, by Lemma 25, preserves partition richness.

Finally, if for every $\beta > 0$ there exists $c _ { \beta } > 0$ such that $g ( \beta t ) = c _ { \beta } g ( t )$ for every $t \geqslant 0$ 2 then $\mu _ { g } ( \beta d ) = c _ { \beta } \mu _ { g } ( d )$ . Lemma 25 therefore gives preservation of scale invariance.

For $g ( t ) = t ^ { p }$ , with $p > 0$ , all the preceding conditions are satisfied, with $c _ { \beta } = \beta ^ { p }$ . Hence the power transformation preserves all five properties, and thus preserves both admissibility and strong admissibility.

## Appendix C. Proofs for Section 3 (Admissible Methods)

We now prove the requirements satisfied by the methods introduced in Section 3. Whenever convenient, we establish stronger requirements than those required for admissibility: e.g., we establish that $T _ { \mathrm { S L } } , T _ { \mathrm { g l o b } } , T _ { \mathrm { l o c } } \in \mathcal { H } _ { \alpha _ { \mathrm { o r d } } \wedge \alpha _ { \mathrm { c w c } } \wedge \alpha _ { \mathrm { a d m + } } } ( \mathcal { X } )$ , and $T _ { \mathrm { s t a b l e } } \in { \mathcal { H } } _ { \alpha _ { \mathrm { c w c } } \wedge \alpha _ { \mathrm { a d m } + } } ( \chi )$

## C.1 Single Linkage

## C.1.1 Single linkage and minimum spanning tree

Recall that the minimum bottleneck between two points $x , y \in { \mathcal { X } }$ is defined by

$$
B ^ { * } ( d ) ( x , y ) \ : = \ \left\{ \begin{array} { l l } { \underset { \gamma \in \Gamma _ { x , y } } { \operatorname* { m i n } } \ \underset { \{ u , v \} \in \gamma } { \operatorname* { m a x } } d ( u , v ) , } & { x \neq y , } \\ { 0 , } & { x = y , } \end{array} \right.\tag{14}
$$

where $\Gamma _ { x , y }$ denotes the set of paths from x to y. The value $B ^ { * } ( d ) ( x , y )$ can be computed by finding a minimum spanning tree (MST) of the complete graph on X with edge weights d.

Lemma 41 Let d P $\mathcal { D } ( \mathcal { X } )$ , and let M be a minimum spanning tree of the complete weighted graph on X with edge weights d. For every $x , y \in { \mathcal { X } }$ , let $\gamma _ { x y } ^ { d }$ denote the unique path from x to y in M. Then $\begin{array} { r } { B ^ { * } ( d ) ( x , y ) = \operatorname* { m a x } _ { ( x ^ { \prime } , y ^ { \prime } ) \in \gamma _ { x y } ^ { d } } d ( x ^ { \prime } , y ^ { \prime } ) } \end{array}$

Proof Let $e ^ { * }$ be an edge of maximal weight on $\gamma _ { x y } ^ { d } .$ and denote $t : = d ( e ^ { * } ) = \operatorname* { m a x } _ { e \in \gamma _ { x y } ^ { d } } d ( e )$ Because $\gamma _ { x y } ^ { d }$ is a path from x to $y ,$ we obtain from (14) $B ^ { * } ( d ) ( x , y ) \leqslant t$ . To prove the reverse inequality, observe that removing $e ^ { * }$ disconnects M into two components, separating x and $y .$ Any path from $x$ to $y$ must cross the resulting cut. If some crossing edge had weight strictly smaller than $t ,$ replacing $e ^ { * }$ by that edge would produce a spanning tree of smaller total weight, contradicting the minimality of $M$ . Hence every $x - y$ path contains an edge of weight at least $t ,$ so $B ^ { * } ( d ) ( x , y ) \geqslant t$

Lemma 42 For every $d \in { \mathcal { D } } ( { \mathcal { X } } ) , B ^ { * } ( d )$ is an ultrametric and $T _ { \mathrm { S L } } ( d ) = \Psi _ { B ^ { * } ( d ) }$ . Equivalently, for every r ě 0, the clusters $o f T _ { \mathrm { S L } } ( d )$ at level r are the connected components of the graph $G _ { r } ( d ) : = \left( \mathcal { X } , \{ \{ x , y \} : d ( x , y ) \leqslant r \} \right)$

Lemma 42 is the standard minimax-path characterization of single linkage (see, for example, Carlsson and M´emoli, 2010, Proposition 8), and thus we omit the proof.

## C.1.2 Properties of Single Linkage

We first record the transformation properties of the minimax map $B ^ { * }$ .

Lemma 43 Transformation $d \mapsto B ^ { * } ( d )$ preserves $\alpha _ { \mathrm { o r d } } , \alpha _ { \mathrm { s u h r } }$ , and $\alpha _ { \mathrm { p e r m } }$

Proof 1. Order invariance. Let $g : \mathbb { R } _ { \geq 0 }  \mathbb { R } _ { \geq 0 }$ be strictly increasing with $g ( 0 ) = 0$ . For every $d \in { \mathcal { D } } ( { \mathcal { X } } )$ and every $x , y \in { \mathcal { X } }$

$$
\begin{array} { r l } & { B ^ { * } ( g \circ d ) ( x , y ) : = \underset { \gamma \in \Gamma _ { x y } } { \mathrm { m i n } } \underset { \{ u , v \} \in \gamma } { \mathrm { m a x } } g ( d ( u , v ) ) = \underset { \gamma \in \Gamma _ { x y } } { \mathrm { m i n } } g \bigg ( \underset { \{ u , v \} \in \gamma } { \mathrm { m a x } } d ( u , v ) \bigg ) } \\ & { \quad \quad \quad = g \bigg ( \underset { \gamma \in \Gamma _ { x y } } { \mathrm { m i n } } \underset { \{ u , v \} \in \gamma } { \mathrm { m a x } } d ( u , v ) \bigg ) = g \big ( B ^ { * } ( d ) ( x , y ) \big ) . } \end{array}
$$

The second and third equalities use the strict monotonicity of $g .$ . Hence $B ^ { * } ( g \circ d ) = g \circ$ $B ^ { * } ( d )$ . Therefore the transformation $B ^ { * }$ preserves order equivalence of dissimilarities, and consequently preserves order invariance.

2. Exactness on ultrametric: Let u be an ultrametric. We prove $B ^ { * } ( u ) = u$ by sandwiching. The inequality $B ^ { * } ( u ) \leqslant u$ is immediate from the definition of $B ^ { * }$ , as the one-edge path from x to y gives $B ^ { * } ( u ) ( x , y ) \leqslant u ( x , y )$ . For the reverse inequality, fix $x , y \in { \mathcal { X } }$ and introduce an arbitrary x to y path $\boldsymbol { \gamma } = \left( \left( x _ { 0 } , x _ { 1 } \right) , \ldots , \left( x _ { m - 1 } , x _ { m } \right) \right)$ . Repeatedly applying the strong triangle inequality gives $u ( x , y ) \leqslant \mathrm { m a x } _ { i = 1 , \ldots , m } u ( x _ { i - 1 } , x _ { i } )$ . Because this holds for any x-to-y path $\begin{array} { r } { \gamma , u ( x , y ) \leqslant \operatorname* { m i n } _ { \gamma \in \Gamma _ { x , y } } \operatorname* { m a x } _ { ( u , v ) \in \gamma } u ( u , v ) = B ^ { * } ( u ) ( x , y ) } \end{array}$ . Hence $B ^ { * } ( u ) = u$

3. Permutation invariance: For a path $\gamma = ( ( u _ { 1 } , u _ { 2 } ) , ( u _ { 2 } , u _ { 3 } ) , \cdot \cdot \cdot , ( u _ { m - 1 } , u _ { m } ) )$ , define the path $\phi ^ { - 1 } \gamma = \bigl ( \bigl ( \phi ^ { - 1 } ( u _ { 1 } ) , \phi ^ { - 1 } ( u _ { 2 } ) \bigr ) , \bigl ( \phi ^ { - 1 } ( u _ { 2 } ) , \phi ^ { - 1 } ( u _ { 3 } ) \bigr ) , \cdot \cdot \cdot , ( \phi ^ { - 1 } ( u _ { m - 1 } ) , \phi ^ { - 1 } ( u _ { m } ) ) \bigr )$ . The map $\gamma \mapsto \phi ^ { - 1 } \gamma$ is a bijection from $\Gamma _ { x , y }$ onto $\Gamma _ { \phi ^ { - 1 } ( x ) , \phi ^ { - 1 } ( y ) }$ . Hence

$$
B ^ { * } ( d _ { \phi } ) ( x , y ) \ = \ B ^ { * } ( d ) ( \phi ^ { - 1 } ( x ) , \phi ^ { - 1 } ( y ) ) \ = \ B ^ { * } ( d ) _ { \phi } ( x , y ) .
$$

Lemma 44 Single linkage $T _ { \mathrm { S L } }$ satisfies α<sub>ord</sub>, α<sub>suhr</sub>, α<sub>perm</sub>, and $\alpha _ { \mathrm { c w c } }$

Proof Because $T _ { \mathrm { g l o b } }$ satisfies $\alpha _ { \mathrm { o r d } } , \alpha _ { \mathrm { s u h r } }$ , and $\alpha _ { \mathrm { p e r m } }$ , order invariance, exactness on ultrametrics, and permutation invariance follow immediately from Proposition 39, Lemma 42, and Lemma 43.

Cluster-wise consistency: Let $C \in T _ { \mathrm { S L } } ( d )$ , and let $d ^ { \prime }$ be a $\{ C \}$ -strengthening of $d .$ By Lemma 42, there exists $r \geqslant 0$ such that C is a connected component of $G _ { r } ( d )$

Because $d ^ { \prime }$ is a $\{ C \}$ -strengthening, every edge within C whose d-weight is at most r still has $d ^ { \prime } .$ -weight at most r. Thus C remains connected in $G _ { r } ( d ^ { \prime } )$ . Conversely, every edge between $C$ and C<sup>¯</sup> has d-weight greater than $^ { r , }$ and its weight can only increase under the strengthening. Hence no edge of $G _ { r } ( d ^ { \prime } )$ joins $C$ to $\bar { C } .$

Therefore C is also a connected component of $G _ { r } ( d ^ { \prime } )$ , and hence $C \in T _ { \mathrm { S L } } ( d ^ { \prime } )$ . Thus $T _ { \mathrm { S L } }$ is cluster-wise consistent.

## C.2 Other Linkage Methods

## C.2.1 Definitions of the Linkage Methods Considered

We briefly recall the non-binary variants of the standard linkage rules considered in Section 3.1. Recall that linkage methods proceed by iteratively merging clusters (starting with singletons). At a given step of the algorithm, let $\mathcal { C } _ { \mathrm { a c t i v e } }$ denote the current partition into active clusters, and let $D : \mathcal { C } _ { \mathrm { a c t i v e } } \times \mathcal { C } _ { \mathrm { a c t i v e } } \to \mathbb { R } _ { \geqslant 0 }$ denote the linkage dissimilarity between active clusters, as determined by the chosen linkage rule. The next merge occurs at the minimum inter-cluster dissimilarity

$$
r : = \operatorname* { m i n } _ { \stackrel { C _ { 1 } , C _ { 2 } \in \mathcal { C } _ { \mathrm { a c t i v e } } } { C _ { 1 } \neq C _ { 2 } } } D ( C _ { 1 } , C _ { 2 } ) .
$$

However, several pairs of active clusters may attain this minimum simultaneously. To merge all clusters involved in such ties simultaneously, form the graph whose vertices are the active clusters $C \in \mathcal { C } _ { \mathrm { a c t i v e } }$ , with an edge between $C _ { 1 }$ and $C _ { 2 }$ whenever $D ( C _ { 1 } , C _ { 2 } ) = r .$ Each nontrivial connected component of this graph is then merged into a single new cluster, obtained as the union of its vertices. In particular, if $C _ { 1 }$ is tied with $C _ { 2 }$ and $C _ { 2 }$ is tied with $C _ { 3 } .$ , then all three are merged together even if $D ( C _ { 1 } , C _ { 3 } ) > r$ . Each cluster created in this way is added to the output hierarchy, and the procedure is repeated with the resulting collection of active clusters.

The linkage dissimilarities considered here admit the Lance-Williams update (Murtagh and Contreras, 2017)

$$
D ( C _ { 1 1 } \cup C _ { 1 2 } , C _ { 2 } ) = \alpha _ { 1 } D ( C _ { 1 1 } , C _ { 2 } ) + \alpha _ { 2 } D ( C _ { 1 2 } , C _ { 2 } ) + \beta D ( C _ { 1 1 } , C _ { 1 2 } ) + \gamma \left| D ( C _ { 1 1 } , C _ { 2 } ) - D ( C _ { 1 2 } , C _ { 2 } ) \right| ,
$$

where the coeficients are given in Table 2. We take $D ( \{ x \} , \{ y \} ) = d ( x , y )$

Ward and Centroid linkages are typically defined for squared Euclidean dissimilarities, as the updates can become negative on arbitrary dissimilarities. We implicitly restrict to squared Euclidean dissimilarities when working with these linkages. Moreover, for linkage rules such as WPGMA and median linkage, the updated dissimilarity after a multiway merge may depend on the order in which clusters in a tied component are merged. To obtain a welldefined hierarchical clustering method, we therefore regard a fixed deterministic tie-breaking convention as part of the linkage rule. This choice will not afect the counterexamples below, since all comparisons determining the relevant merges are strict.

<table><tr><td>Method</td><td> $D ( C _ { 1 } , C _ { 2 } )$ </td><td> $( \alpha _ { 1 } , \alpha _ { 2 } , \beta , \gamma )$ </td></tr><tr><td>Single</td><td> $\operatorname* { m i n } _ { x \in C _ { 1 } , y \in C _ { 2 } } d ( x , y )$ </td><td> $\left( { \frac { 1 } { 2 } } , \ { \frac { 1 } { 2 } } , 0 , - { \frac { 1 } { 2 } } \right)$ </td></tr><tr><td>Complete</td><td> $\operatorname* { m a x } _ { x \in C _ { 1 } , y \in C _ { 2 } } d ( x , y )$ </td><td> $\left( { \begin{array} { l } { { \frac { 1 } { 2 } } , \ { \frac { 1 } { 2 } } , 0 , \ { \frac { 1 } { 2 } } } \end{array} } \right)$ </td></tr><tr><td>Unweighted average (UPGMA)</td><td> $\frac { 1 } { | C _ { 1 } | | C _ { 2 } | } \sum _ { x \in C _ { 1 } , y \in C _ { 2 } } d ( x , y )$ </td><td> $\begin{array} { r } { \left( \frac { | C _ { 1 1 } | } { | C _ { 1 } | } , \frac { | C _ { 1 2 } | } { | C _ { 1 } | } , 0 , 0 \right) } \end{array}$ </td></tr><tr><td>Weighted average (WPGMA)</td><td> $\frac { D ( C _ { 1 1 } , C _ { 2 } ) + D ( C _ { 1 2 } , C _ { 2 } ) } { 2 }$ </td><td> $\left( { \scriptstyle { \frac { 1 } { 2 } } } , { \scriptstyle { \frac { 1 } { 2 } } } , 0 , 0 \right)$ </td></tr><tr><td>Ward&#x27;s</td><td> $\frac { | C _ { 1 } | | C _ { 2 } | } { | C _ { 1 } | + | C _ { 2 } | } \ \| m _ { C _ { 1 } } - m _ { C _ { 2 } } \| ^ { 2 }$ </td><td> $\begin{array} { r } { \left( \frac { | C _ { 1 1 } | + | C _ { 2 } | } { | C _ { 1 } | + | C _ { 2 } | } , \frac { | C _ { 1 2 } | + | C _ { 2 } | } { | C _ { 1 } | + | C _ { 2 } | } , - \frac { | C _ { 2 } | } { | C _ { 1 } | + | C _ { 2 } | } , 0 \right) } \end{array}$ </td></tr><tr><td>Centroid (UPGMC)</td><td> $\| m c _ { 1 } - m c _ { 2 } \| ^ { 2 }$ </td><td> $\begin{array} { r } { \left( \frac { | C _ { 1 1 } | } { | C _ { 1 } | } , \frac { | C _ { 1 2 } | } { | C _ { 1 } | } , - \frac { | C _ { 1 1 } | | C _ { 1 2 } | } { | C _ { 1 } | ^ { 2 } } , 0 \right) } \end{array}$ </td></tr><tr><td>Weighted centroid (WPGMC / Median)</td><td> $\left\| \frac { m _ { C _ { 1 1 } } + m _ { C _ { 1 2 } } } { 2 } - m _ { C _ { 2 } } \right\| ^ { 2 }$ </td><td> $\left( { \frac { 1 } { 2 } } , \ { \frac { 1 } { 2 } } , \ - { \frac { 1 } { 4 } } , 0 \right)$ </td></tr></table>

Table 2: Classical hierarchical clustering linkages and their Lance–Williams coeficients. m denotes the centroid of cluster C.

## C.2.2 Non-admissibility of the Other Linkage Methods

We demonstrate the non-admissibility of complete, average, Ward, UPGMC, and WPGMC linkage methods by providing counterexamples that show they do not satisfy consistency.

Lemma 45 Complete, unweighted and weighted average, Ward, centroid, and median linkages do not satisfy partition consistency. Thus, none of them is admissible, regardless of the tie-breaking rule.

Proof Fix distinct points $1 , 2 , 3 , 4 \in \mathcal { X }$ , let $Y : = \mathcal { X } \backslash \{ 1 , 2 , 3 , 4 \} , A : = \{ 1 , 2 , 3 \}$ , and $\mathcal { C } : =$ $\{ A , \{ 4 \} \} \cup \{ \{ y \} : y \in Y \}$ (. Define $d , d ^ { \prime } \in \mathcal { D } ( \mathcal { X } )$ by

$$
d ( 1 , 2 ) = d ( 1 , 3 ) = 6 , \qquad d ( 1 , 4 ) = d ( 2 , 4 ) = 8 , \qquad d ( 2 , 3 ) = 4 , \qquad d ( 3 , 4 ) = q , \mathrm { ~ a n d }
$$

$$
d ^ { \prime } ( 1 , 2 ) = 3 , \qquad d ^ { \prime } ( i , j ) = d ( i , j ) \quad \mathrm { f o r ~ a l l ~ o t h e r ~ p a i r s . }
$$

Whenever at least one of $x , y$ belongs to Y, set $d ( x , y ) = d ^ { \prime } ( x , y ) = 1 0 0$ . Thus $d ^ { \prime }$ is a C-strengthening of d.

Choose q as in table below. For every value of q in the table, both d and $d ^ { \prime }$ are squared Euclidean dissimilarities. The relevant linkage values are also given in the table.

Under d, every rule first merges t2, 3u, since $d ( 2 , 3 ) = 4$ is the unique smallest initial dissimilarity. The table then gives $D ( \{ 2 , 3 \} , \{ 1 \} ) \ < \ \operatorname * { m i n } \big \{ D ( \{ 2 , 3 \} , \{ 4 \} ) , d ( 1 , 4 ) \big \}$ (, while all dissimilarities involving Y are larger. Hence the second merge forms A, so $A \in T ( d )$ and therefore ${ \mathcal { C } } \subseteq T ( d )$

Under $d ^ { \prime } ,$ the unique first merge is t1, 2u. The last two columns show that $q \ <$ min $\left\{ D ^ { \prime } ( \{ 1 , 2 \} , \{ 3 \} ) , D ^ { \prime } ( \{ 1 , 2 \} , \{ 4 \} ) \right\}$ (, so the second merge is $\{ 3 , 4 \}$ . Consequently, every subsequent cluster containing 3 also contains 4, and hence $A \notin T ( d ^ { \prime } )$

<table><tr><td>Linkage</td><td> $D ( \{ 2 , 3 \} , \{ 1 \} )$  q</td><td></td><td> $D ( \{ 2 , 3 \} , \{ 4 \} )$ </td><td> $D ^ { \prime } ( \{ 1 , 2 \} , \{ 3 \} )$ </td><td> $D ^ { \prime } ( \{ 1 , 2 \} , \{ 4 \} )$ </td></tr><tr><td>Complete</td><td>5</td><td>6</td><td>8</td><td>6</td><td>8</td></tr><tr><td>UPGMA/WPGMA</td><td></td><td>6</td><td>25</td><td>5</td><td>8</td></tr><tr><td>Centroid/Median</td><td>9-24-104110</td><td>5</td><td>101</td><td>17-4173</td><td></td></tr><tr><td>Ward</td><td></td><td>20 3</td><td>201 15</td><td></td><td>22-4293</td></tr></table>

Thus $A \in T ( d ) \backslash T ( d ^ { \prime } )$ , although $d ^ { \prime }$ is a C-strengthening of d and ${ \mathcal { C } } \subseteq T ( d )$ . Therefore each of the six linkage rules violates partition consistency. All comparisons are strict, so the conclusion is independent of the tie-breaking convention.

## C.3 Separation Methods: Proof of Proposition 10

Lemma 46 proves that $T _ { \mathrm { g l o b } } ^ { \eta }$ and $T _ { \mathrm { l o c } } ^ { \eta }$ are hierarchical clustering methods satisfying scale invariance, cluster-wise consistency, and permutation invariance, and Lemma 49 establishes strict hierarchical richness. Because cluster-wise consistency implies partition consistency and strict hierarchical richness implies partition richness, we conclude that $T _ { \mathrm { g l o b } } ^ { \eta }$ and $T _ { \mathrm { l o c } } ^ { \eta }$ are admissible for every separation margin sequence η. Lemma 47 shows that distinct margin sequences define distinct methods, completing the proof of Proposition 10. We also record the stronger properties of exactness on ultrametrics and order invariance for the unit-margin methods in Lemmas 48 and 50.

## C.3.1 Basic properties

Lemma 46 For every separation margin sequence $\eta , T _ { \mathrm { g l o b } } ^ { \eta }$ and $T _ { \mathrm { l o c } } ^ { \eta }$ are hierarchical clustering methods satisfying $\alpha _ { \mathrm { s c a } } , \alpha _ { \mathrm { c w c } }$ , and α<sub>perm</sub>.

Proof $T _ { \mathrm { l o c } }$ is a known hierarchical clustering method, and for any dissimilarity d, $T _ { \mathrm { l o c } } ( d )$ satisfies all the conditions in Definition 4 (Dreveton et al., 2025; Balcan et al., 2008). Let η be a separation margin sequence. By definition of the corresponding separation conditions, for every dissimilarity $d ,$ we have $T _ { \mathrm { g l o b } } ^ { \eta } ( d ) \subseteq T _ { \mathrm { l o c } } ^ { \eta } ( d ) \subseteq T _ { \mathrm { l o c } } .$ Because $T _ { \mathrm { l o c } } ( d )$ is laminar, each of its subfamilies is also laminar. Hence the outputs of $T _ { \mathrm { g l o b } } ^ { \eta }$ and $T _ { \mathrm { l o c } } ^ { \eta }$ are laminar as well. Moreover, by construction, each of these outputs contains the root X and all leaves $\{ x \}$ 2 $x \in \mathcal { X }$ . Therefore, $T _ { \mathrm { g l o b } } ^ { \eta }$ and $T _ { \mathrm { l o c } } ^ { \eta }$ are hierarchical clustering methods.

Next, we show that $T _ { \mathrm { l o c } } ^ { \eta }$ satisfies each property. The proofs for $T _ { \mathrm { g l o b } } ^ { \eta }$ are analogous and hence omitted.

$\begin{array} { r l } { 1 . } & { { } T _ { \mathrm { l o c } } ^ { \eta } } \end{array}$ satisfies $\alpha _ { \mathrm { s c a } }$ . Let $\beta > 0$ . For every $x , y , z \in { \mathcal { X } }$ we have $\begin{array} { r } { \frac { \beta \cdot d ( x , y ) } { \beta \cdot d ( x , z ) } = \frac { d ( x , y ) } { d ( x , z ) } } \end{array}$ . Hence $C \in T _ { \mathrm { l o c } } ^ { \eta } ( \beta d )$ if $C \in T _ { \mathrm { l o c } } ^ { \eta } ( d )$ , and therefore $T _ { \mathrm { l o c } } ^ { \eta } ( \beta d ) = T _ { \mathrm { l o c } } ^ { \eta } ( d )$

$\begin{array} { r l } { \mathcal { Q } . } & { { } T _ { \mathrm { l o c } } ^ { \eta } } \end{array}$ satisfies $\alpha _ { \mathrm { { c w c } } }$ . Let d be a dissimilarity function, and take $C \in T _ { \mathrm { l o c } } ^ { \eta } ( d )$ . Because the root and singleton clusters belong to $T _ { \mathrm { l o c } } ^ { \eta } ( d )$ for every dissimilarity, it sufices to consider a nontrivial proper cluster C. Consider a dissimilarity d<sup>1</sup> such that d<sup>1</sup> is a tCu-strengthening of d. Then, for every $x , y \in C , z \in \mathcal { X } \backslash C , d ^ { \prime } ( x , y ) \leqslant d ( x , y )$ and $d ^ { \prime } ( x , z ) \geqslant d ( x , z )$ hold. Thus

$$
\tau ( d ^ { \prime } , C ) \ = \ \operatorname* { m a x } _ { x , y \in C , z \in \bar { C } } \frac { d ^ { \prime } ( x , y ) } { d ^ { \prime } ( x , z ) } \ \leqslant \ \operatorname* { m a x } _ { x , y \in C , z \in \bar { C } } \frac { d ( x , y ) } { d ( x , z ) } \ = \ \tau ( d , C ) \ < \ \eta _ { | C | - 1 } ,
$$

establishing that $C \in T _ { \mathrm { l o c } } ^ { \eta } ( d ^ { \prime } )$

$\begin{array} { r l } { \mathcal { 3 } . } & { { } T _ { \mathrm { l o c } } ^ { \eta } } \end{array}$ satisfies $\alpha _ { \mathrm { p e r m } }$ . Let $\phi \in \Pi ( { \mathcal { X } } )$ be a permutation, and recall that $d _ { \phi } ( x , y ) : =$ $d ( \phi ^ { - 1 } ( x ) , \phi ^ { - 1 } ( y ) )$ (Definition 5). Hence, $d _ { \phi } ( \phi ( x ) , \phi ( y ) ) = d ( x , y )$ for all $x , y \in { \mathcal { X } }$ , and

$$
\tau ( d , C ) : = \operatorname* { m a x } _ { x , y \in C , z \in \bar { C } } \frac { d ( x , y ) } { d ( x , z ) } \ = \ \operatorname* { m a x } _ { x , y \in C , z \in \bar { C } } \frac { d _ { \phi } ( \phi ( x ) , \phi ( y ) ) } { d _ { \phi } ( \phi ( x ) , \phi ( z ) ) } \ = \ \operatorname* { m a x } _ { x , y \in \phi ( C ) , \atop z \in \phi ( C ) } \frac { d _ { \phi } ( x , y ) } { d _ { \phi } ( x , z ) } \ = \ \tau ( d _ { \phi } , \phi ( C ) ) ,
$$

where we used the fact that $\phi$ is a bijection and $\phi ( \bar { C } ) = \overline { { { \phi ( C ) } } }$ . Moreover, $| \phi ( C ) | = | C |$ Hence $\tau ( d , C ) < \eta _ { | C | - 1 } \mathrm { i f } \tau ( d _ { \phi } , \phi ( C ) ) < \eta _ { | \phi ( C ) | - 1 }$ . Therefore: $C \in T _ { \mathrm { l o c } } ^ { \eta } ( d ) \mathrm { i f f } \phi ( C ) \in T _ { \mathrm { l o c } } ^ { \eta } ( d _ { \phi } )$ Thus $T _ { \mathrm { l o c } } ^ { \eta } ( d _ { \phi } ) = \dot { \phi } \cdot \dot { T } _ { \mathrm { l o c } } ^ { \eta } ( d )$

Lemma 47 Let $\eta \neq \eta ^ { \prime }$ be two distinct separation margin sequences. Then $T _ { \mathrm { g l o b } } ^ { \eta } \neq T _ { \mathrm { g l o b } } ^ { \eta ^ { \prime } }$ and $T _ { \mathrm { l o c } } ^ { \eta } \neq T _ { \mathrm { l o c } } ^ { \eta ^ { \prime } }$

Proof Because $\eta \neq \eta ^ { \prime }$ , there exists $s \in [ n - 2 ]$ such that $\eta _ { s } \neq \eta _ { s } ^ { \prime }$ . Without loss of generality, assume $\eta _ { s } < \eta _ { s } ^ { \prime }$ . Choose $C \subsetneq { \mathcal { X } }$ with $| C | = s + 1$ , set $\begin{array} { r } { r : = \frac { \eta _ { s } + \eta _ { s } ^ { \prime } } { 2 } } \end{array}$ , and define

$d ( x , y ) = r$ for distinct x, $y \in C$ or $x , y \in { \bar { C } } ,$ and $d ( x , y ) = 1$ for other distinct $x , y$

Observe that $\varrho ( d , C ) ~ = ~ \tau ( d , C ) ~ = ~ r$ . Therefore, the set C belongs to $T _ { \mathrm { l o c } } ^ { \eta ^ { \prime } } ( d )$ and to $T _ { \mathrm { g l o b } } ^ { \eta ^ { \prime } } ( d )$ but belongs neither to $T _ { \mathrm { l o c } } ^ { \eta } ( d )$ nor to $T _ { \mathrm { g l o b } } ^ { \eta } ( d )$ , proving $T _ { \mathrm { l o c } } ^ { \eta } ( d ) ~ \neq ~ T _ { \mathrm { l o c } } ^ { \eta ^ { \prime } } ( d )$ and $T _ { \mathrm { g l o b } } ^ { \bar { \eta } } ( d ) \neq T _ { \mathrm { g l o b } } ^ { \eta ^ { \prime } } ( d )$ ■

## C.3.2 Specific properties for $\pmb { \eta } = \mathbf { 1 }$

Lemma 48 Let η be a separation margin sequence. These statements are equivalent: $\mathit { \Omega } ( i ) \eta = \mathbf { 1 } ; \left( i i \right) T _ { \mathrm { l o c } } ^ { \eta }$ satisfies $\alpha _ { \mathrm { s u h r } } ; ~ ( i i i ) ~ T _ { \mathrm { g l o b } } ^ { \eta }$ satisfies $\alpha _ { \mathrm { s u h r } }$

Proof We first show that (i) implies both (ii) and (iii). Assume that $\mathbf { \nabla } \eta \mathbf { \nabla } = \mathbf { 1 }$ , so that $T _ { \mathrm { l o c } } ^ { \eta } = T _ { \mathrm { l o c } }$ and $T _ { \mathrm { g l o b } } ^ { \eta } = T _ { \mathrm { g l o b } }$ . Fix an ultrametric $u \in \mathcal { U } ( \mathcal { X } )$ , and let $\Psi _ { u }$ denote its associated hierarchy. We prove that $\Psi _ { u } \subseteq T _ { \mathrm { g l o b } } ( u )$ and $T _ { \mathrm { l o c } } ( u ) \subseteq \Psi _ { u }$ . Combined with $T _ { \mathrm { g l o b } } ( u ) \subseteq$ $T _ { \mathrm { l o c } } ( u )$ , this establishes $\Psi _ { u } = T _ { \mathrm { g l o b } } ( u ) = T _ { \mathrm { l o c } } ( u )$

We first prove $\Psi _ { u } \subseteq T _ { \mathrm { g l o b } } ( u )$ . The root and singleton clusters belong to $T _ { \mathrm { g l o b } } ( u )$ by definition, so let $C \in \Psi _ { u }$ be a non-root and non-singleton cluster. By the ultrametric-ball representation of $\Psi _ { u }$ (see Equation (3)), there exists $r \geqslant 0$ such that $u ( x , y ) \leqslant r < u ( x , z )$ for all $x , y \in C , \ z \notin C$ . Hence, for all $x _ { 1 } , x _ { 2 } , y \in C$ and $z \not \in C$ , we have $\begin{array} { r } { \frac { u ( x _ { 1 } , y ) } { u ( x _ { 2 } , z ) } < 1 } \end{array}$ . Therefore $\varrho ( u , C ) < 1$ , and thus $C \in T _ { \mathrm { g l o b } } ( u )$ . This establishes $\Psi _ { u } \subseteq T _ { \mathrm { g l o b } } ( u )$

It remains to prove $T _ { \mathrm { l o c } } ( u ) \subseteq \Psi _ { u }$ . Again, the root and singleton clusters are immediate, so let $C \in T _ { \mathrm { l o c } } ( u )$ be a non-root and non-singleton cluster. Because $\tau ( u , C ) < 1$ , we have $u ( x , y ) < u ( x , z )$ for all $x , y \in C , \ z \notin C$ . Fix $x _ { 0 } \in C$ , and set $R _ { \mathrm { i n } } : = \operatorname* { m a x } _ { y \in C } u ( x _ { 0 } , y )$ and $\begin{array} { r } { R _ { \mathrm { o u t } } : = \operatorname* { m i n } _ { z \not \in C } u ( x _ { 0 } , z ) } \end{array}$ . Because $\mathcal { X }$ is finite, these extrema are attained, and the preceding strict inequalities imply $R _ { \mathrm { i n } } < R _ { \mathrm { o u t } }$ . Therefore, $B _ { u } ( x _ { 0 } , R _ { \mathrm { i n } } ) = C$ . By the ultrametric-ball representation of $\Psi _ { u } ,$ , this gives $C \in \Psi _ { u }$ . This establishes $T _ { \mathrm { l o c } } ( u ) \subseteq \Psi _ { u }$

Hence, $\Psi _ { u } = T _ { \mathrm { g l o b } } ( u ) = T _ { \mathrm { l o c } } ( u )$ for every ultrametric u, proving (i) implies (ii) and (iii). We now prove the converses by contraposition. Suppose that $\pmb { \eta } \neq \mathbf { 1 }$ . Then there exists $s \in [ n - 2 ]$ such that $\eta _ { s } < 1$ . Choose a subset $C \subsetneq { \mathcal { X } }$ with $| C | = s + 1$ and choose r such that $\eta _ { s } < r < 1$ . Define a dissimilarity u by

$$
u ( x , y ) = r { \mathrm { ~ f o r ~ d i s t i n c t ~ } } x , y \in C , \qquad { \mathrm { a n d } } \qquad u ( x , y ) = 1 { \mathrm { ~ f o r ~ o t h e r ~ d i s t i n c t . ~ } } y ,
$$

and observe that u is an ultrametric. Moreover, for every $x \in C , B _ { u } ( x , r ) = C ,$ so $C \in \Psi _ { u }$ However, $\tau ( u , C ) = \varrho ( u , C ) = r > \eta _ { s } = \eta _ { \vert C \vert - 1 }$ . Hence $C \notin T _ { \mathrm { l o c } } ^ { \eta } ( u )$ and $C \not \in T _ { \mathrm { g l o b } } ^ { \eta } ( u )$ . Thus neither $T _ { \mathrm { l o c } } ^ { \eta } ( u )$ nor $T _ { \mathrm { g l o b } } ^ { \eta } ( u )$ equals $\Psi _ { u } .$ . Therefore neither method is exact on ultrametrics. This proves (ii) implies (i) and (iii) implies (i).

Similarly, we obtain the strict hierarchical richness of $T _ { \mathrm { g l o b } } ^ { \eta }$ and $T _ { \mathrm { l o c } } ^ { \eta }$

Lemma 49 $T _ { \mathrm { g l o b } } ^ { \eta }$ and $T _ { \mathrm { l o c } } ^ { \eta }$ satisfy $\alpha _ { \mathrm { s h r } }$ for any separation margin sequence η.

Proof Fix an arbitrary hierarchy $\Psi \in { \mathcal { T } } ( { \mathcal { X } } )$ , and let $q \in \left( 0 , \operatorname* { m i n } _ { 1 \leqslant s \leqslant n - 2 } \eta _ { s } \right)$ . We will construct a dissimilarity d such that $T _ { \mathrm { g l o b } } ^ { \eta } ( d ) = T _ { \mathrm { l o c } } ^ { \eta } ( d ) = \Psi$ . View Ψ as a rooted tree, with root X and leaves the singletons txu for $x \in { \mathcal { X } } ;$ let depthpCq denote the depth of cluster $C \in \Psi$ , with $\mathrm { d e p t h } ( \mathcal { X } ) ~ = ~ 0$ Assign to every non-singleton $C \in \Psi$ the height $h ( C ) : = q ^ { \mathrm { d e p t h } ( C ) }$ , and assign height 0 to the singleton leaves. Define for every distinct $x , y ,$

$$
u _ { \Psi } ( x , y ) : = h \big ( \mathrm { l c a } _ { \Psi } ( x , y ) \big ) .
$$

Then u<sub>Ψ</sub> is an ultrametric whose associated hierarchy is exactly Ψ.

Let $C \in \Psi$ be non-singleton and proper cluster, and denote by $\operatorname { p a r } ( C )$ its parent in Ψ. For all $x _ { 1 } , x _ { 2 } , y ~ \in ~ C$ and $z \not \in C$ , we have $u _ { \Psi } ( x _ { 1 } , y ) \leqslant h ( C )$ and $u _ { \Psi } ( x _ { 2 } , z ) \geqslant h ( \mathrm { p a r } ( C ) )$ Using ${ \frac { h ( C ) } { h ( \operatorname* { p a r } ( C ) ) } } = q$ and the definition of $\varrho ( u \Psi , C )$ , we obtain $\varrho \leqslant q < \eta _ { | C | - 1 } . \mathrm { A s } \ \tau ( u _ { \Psi } , C ) \leqslant$ $\rho ( u _ { \Psi } , C )$ , it follows that every $C \in \Psi$ belongs to $T _ { \mathrm { g l o b } } ^ { \eta } ( u _ { \Psi } )$ and to $T _ { \mathrm { l o c } } ^ { \eta } ( u _ { \Psi } )$ , and thus

$$
\Psi \subseteq T _ { \mathrm { g l o b } } ^ { \eta } ( u _ { \Psi } ) \quad \mathrm { ~ a n d ~ } \quad \Psi \subseteq T _ { \mathrm { l o c } } ^ { \eta } ( u _ { \Psi } ) .
$$

Moreover, because $T _ { \mathrm { g l o b } }$ and $T _ { \mathrm { l o c } }$ are exact on ultrametrics (Lemma 48), we also have

$$
T _ { \mathrm { g l o b } } ^ { \eta } ( u _ { \Psi } ) \subseteq T _ { \mathrm { g l o b } } ( u _ { \Psi } ) = \Psi , \mathrm { a n d } T _ { \mathrm { l o c } } ^ { \eta } ( u _ { \Psi } ) \subseteq T _ { \mathrm { l o c } } ( u _ { \Psi } ) = \Psi ,
$$

Hence $T _ { \mathrm { g l o b } } ^ { \eta } ( u _ { \Psi } ) = T _ { \mathrm { l o c } } ^ { \eta } ( u _ { \Psi } ) = \Psi$ , and both methods satisfy strict hierarchical richness.

Lemma 50 Let η be a separation margin sequence. These statements are equivalent: $( i ) ~ \eta = { \bf 1 } ; ~ ( i i ) ~ T _ { \mathrm { l o c } } ^ { \eta } ~ s a t i s f i e s ~ \alpha _ { \mathrm { o r d } } ; ~ ( i i i ) ~ T _ { \mathrm { g l o b } } ^ { \eta } ~ s a t i s f i e s ~ \alpha _ { \mathrm { o r d } } .$

Proof We only prove the equivalence between (i) and (ii). The proof of the equivalence between (i) and (iii) is almost identical and thus is omitted.

To show that (i) implies (ii), let $\eta = 1$ , so that $T _ { \mathrm { l o c } } ^ { \eta } = T _ { \mathrm { l o c } }$ . Let $g : \mathbb { R } _ { \geqslant 0 }  \mathbb { R } _ { \geqslant 0 }$ be strictly increasing with $g ( 0 ) = 0$ . For every nonsingleton cluster $C \subsetneq { \mathcal { X } }$

$$
\tau ( d , C ) < 1 \iff d ( x , y ) < d ( x , z ) \quad { \mathrm { f o r ~ a l l ~ } } x , y \in C , \ z \not \in C .
$$

Because g is strictly increasing, this is equivalent to

$$
g ( d ( x , y ) ) < g ( d ( x , z ) ) \quad { \mathrm { f o r ~ a l l ~ } } x , y \in C , \ z \not \in C ,
$$

and hence to $\tau ( g \circ d , C ) < 1$ . Therefore $T _ { \mathrm { l o c } } ( g \circ d ) = T _ { \mathrm { l o c } } ( d )$

We now establish that (ii) implies (i). We prove it by contraposition. Consider a separation margin sequence $\eta \neq 1$ ; we will establish that $T _ { \mathrm { l o c } } ^ { \eta }$ is not order-invariant by constructing two dissimilarities $d , d ^ { \prime }$ such that $d ^ { \prime } = g \circ d$ for some strictly increasing function g satisfying $g ( 0 ) = 0$ , but for which $T _ { \mathrm { l o c } } ^ { \eta } ( d ) \neq T _ { \mathrm { l o c } } ^ { \eta } ( d ^ { \prime } )$

Because $\pm \mathbf { \delta 1 }$ , there exists $s \in [ n - 2 ]$ such that $\eta _ { s } < 1$ . Choose a subset $C \subseteq { \mathcal { X } }$ with $s = | C | - 1$ . Define d, $d ^ { \prime } \in \mathcal { D } ( \mathcal { X } )$ such that for distinct $x , y \in { \mathcal { X } }$ ，

$$
d ( x , y ) : = \left\{ \begin{array} { l l } { \eta _ { s } / 2 , } & { \mathrm { i f ~ } x , y \in C \mathrm { ~ o r ~ } x , y \in \bar { C } , } \\ { 1 , } & { \mathrm { i f ~ } x \in C , \ y \in \bar { C } , } \end{array} \right.
$$

and

$$
d ^ { \prime } ( x , y ) : = \left\{ \begin{array} { l l } { \eta _ { s } , } & { \mathrm { i f ~ } x , y \in C \mathrm { ~ o r ~ } x , y \in \bar { C } , } \\ { 1 , } & { \mathrm { i f ~ } x \in C , y \in \bar { C } . } \end{array} \right.
$$

Because $0 < \eta _ { s } / 2 < \eta _ { s } < 1$ there exists a strictly increasing $g : \mathbb { R } _ { \geqslant 0 }  \mathbb { R } _ { \geqslant 0 }$ satisfying $g ( 0 ) = 0 , g ( \eta _ { s } / 2 ) = \eta _ { s }$ , and $g ( 1 ) = 1$ . Hence $d ^ { \prime } = g \circ d$ C

Observe that for every $x , y \in C$ and $\begin{array} { r } { z \not \in C , \frac { d ( x , y ) } { d ( x , z ) } = \frac { \eta _ { s } / 2 } { 1 } < \eta _ { s } } \end{array}$ , hence $C \in T _ { \mathrm { l o c } } ^ { \eta } ( d )$ holds. However, fix any $z _ { 0 } \in \bar { C }$ . For every $x , y \in C$ with $\begin{array} { r } { x \neq y , \frac { d ^ { \prime } ( x , y ) } { d ^ { \prime } ( x , z _ { 0 } ) } = \frac { \eta _ { s } } { 1 } = \eta _ { s } \neq \eta _ { s } } \end{array}$ . Therefore $C \notin T _ { \mathrm { l o c } } ^ { \eta } ( d ^ { \prime } )$ . Hence $T _ { \mathrm { l o c } } ^ { \eta }$ is not order invariant.

## C.4 Bryant-Berry Stable Clusters: Proof of Proposition 11

We first establish the strong admissibility of the Bryant–Berry method. Moreover, Corollary 26 shows that $\mu _ { \mathrm { p o w e r } } ^ { p }$ preserves strong admissibility for every $p > 0 .$ . Hence $T _ { \mathrm { s t a b l e } } ^ { ( p ) } =$ $T _ { \mathrm { s t a b l e } } \circ \mu _ { \mathrm { p o w e r } } ^ { p }$ is strongly admissible for every $p > 0$ . In particular, $T _ { \mathrm { s t a b l e } }$ and all its power-transformed variants are admissible, proving Proposition 11.

Lemma 51 $T _ { \mathrm { s t a b l e } }$ is a hierarchical clustering method and satisfies $\alpha _ { \mathrm { s c a } } , \alpha _ { \mathrm { s u h r } } , \alpha _ { \mathrm { c w c } }$ and $\alpha _ { \mathrm { p e r m } }$ . In particular, $T _ { \mathrm { s t a b l e } }$ is strongly admissible.

Proof For every $d \in { \mathcal { D } } ( { \mathcal { X } } )$ , Bryant and Berry (Bryant and Berry, 2001) show that the stable clusters $\{ C \subseteq \mathcal { X } : \iota ^ { d } ( C ) > 0 \}$ form a laminar family. Because $T _ { \mathrm { s t a b l e } } ( d )$ additionally contains the root X and all singleton clusters, it is a hierarchy. Thus $T _ { \mathrm { s t a b l e } }$ is a hierarchical clustering method. We now verify the four axioms.

1. Scale invariance. For all $\beta > 0$ and $u , v , z \in \mathcal { X }$ , we have $\rho _ { \beta d } ( u v \mid z ) = \beta \rho _ { d } ( u v \mid z )$ . Hence

$$
\overline { { { \rho } } } _ { \beta d } ( U V \mid Z ) = \beta \overline { { { \rho } } } _ { d } ( U V \mid Z )
$$

for every admissible triplet $( U , V , Z )$ , and therefore $\iota ^ { \beta d } ( C ) = \beta \iota ^ { d } ( C )$ . Because $\beta > 0$ , we also have $\iota ^ { \beta d } ( C ) > 0 \mathrm { i f f } \iota ^ { d } ( C ) > 0$ . Thus $T _ { \mathrm { s t a b l e } } ( \beta d ) = T _ { \mathrm { s t a b l e } } ( d )$

2. Exactness on ultrametrics. Let u be an ultrametric, and let $\Psi _ { u }$ be its associated hierarchy. Because $T _ { \mathrm { l o c } } \equiv T _ { \mathrm { s t a b l e } }$ and $T _ { \mathrm { l o c } } ( u ) = \Psi _ { u }$ , we immediately obtain $\Psi _ { u } \subseteq T _ { \mathrm { s t a b l e } } ( u )$ . It remains to prove the reverse inclusion.

Let $C \in T _ { \mathrm { s t a b l e } } ( u )$ . The root and singleton cases are immediate, so assume that C is a non-singleton proper subset of X. Then $\iota ^ { u } ( C ) > 0$

We first show that every external point is equidistant from all points of C. To do so, fix $z \in \mathcal { X } \backslash C$ , and we will prove that the function $x \mapsto u ( x , z )$ is constant on C. By contradiction, suppose this is not the case. Let

$$
r : = \operatorname* { m i n } _ { x \in C } u ( x , z ) , \qquad U : = \{ x \in C : u ( x , z ) = r \} , \qquad V : = C \backslash U , \qquad Z : = \{ z \} .
$$

Then $U , V , Z$ are nonempty, and $U \cap V = \emptyset$ and $C = U \cup V$ . For every $x \in U$ and $y \in V$ we have $r = u ( x , z ) < u ( y , z )$ . Because u is an ultrametric, the two larger distances among $u ( x , y ) , u ( x , z ) , u ( y , z )$ are equal. Hence $u ( x , y ) = u ( y , z )$ . Therefore

$$
\rho _ { u } ( x y \mid z ) = \operatorname* { m i n } \{ u ( x , z ) , u ( y , z ) \} - u ( x , y ) = u ( x , z ) - u ( y , z ) < 0 .
$$

Averaging over $x \in U , y \in V .$ , and $Z \ = \ \{ z \}$ , we obtain $\overline { { { \rho } } } _ { u } ( U V \mid Z ) < 0$ , contradicting $\iota ^ { u } ( C ) > 0$ . Thus $u ( x , z )$ is constant on C. Denote its common value by $\lambda _ { z } : = u ( x , z ) , x \in C$

We next show that every internal dissimilarity is strictly smaller than this common external dissimilarity. Suppose, toward a contradiction, that there exist distinct $x , y \in C$ such that $u ( x , y ) \geqslant \lambda _ { z }$ . Because $u ( x , z ) = u ( y , z ) = \lambda _ { z }$ , the ultrametric inequality gives

$$
u ( x , y ) \leqslant \operatorname* { m a x } \{ u ( x , z ) , u ( y , z ) \} = \lambda _ { z } .
$$

Hence $u ( x , y ) = \lambda _ { z }$ . Define a relation „ on C by $a \sim b \iff u ( a , b ) < \lambda _ { z }$ . By the ultrametric inequality, „ is an equivalence relation on C. Because $u ( x , y ) = \lambda _ { z }$ , it has at least two equivalence classes. Let U be one equivalence class and set $V : = C \backslash U , Z : = \{ z \}$ . For every $a \in U$ and $b \in V$ , we have $u ( a , b ) \geqslant \lambda _ { z }$ . On the other hand, $u ( a , b ) \leqslant \operatorname* { m a x } \{ u ( a , z ) , u ( b , z ) \} =$ $\lambda _ { z } .$ . Thus $u ( a , b ) = \lambda _ { z }$ . Therefore $\rho _ { u } ( a b \mid z ) = \operatorname* { m i n } \{ u ( a , z ) , u ( b , z ) \} - u ( a , b ) = \lambda _ { z } - \lambda _ { z } = 0$ Hence $\overline { { { \rho } } } _ { u } ( U V \mid Z ) = 0$ , again contradicting $\iota ^ { u } ( C ) > 0$ . Therefore, for every $z \in \mathcal { X } \backslash C$ and all distinct $x , y \in C , u ( x , y ) < u ( x , z ) = u ( y , z )$

Now set $R : = \operatorname* { m a x } _ { x , y \in C } u ( x , y )$ . The preceding inequality implies that, for every $x \in C$ $B _ { u } ( x , R ) = C ,$ , where $B _ { u }$ is defined in Equation (3). Hence $C \in \Psi _ { u }$ by the ultrametric-ball representation of $\Psi _ { u }$ . This proves that $T _ { \mathrm { s t a b l e } } ( u ) \subseteq \Psi _ { u }$

3. Cluster-wise consistency: Let $C \in T _ { \mathrm { s t a b l e } } ( d )$ , and let d<sup>1</sup> be a tCu-strengthening of d.   
The root and singleton cases are immediate, so assume that C is nontrivial and proper.

Consider two nonempty disjoint sets $U , V \subseteq { \mathcal { X } }$ with $U \cup V \subseteq C$ , and a nonempty $Z \subseteq { \mathcal { X } } \backslash C$ . For every $a \in U , b \in V , z \in Z$ , we have $d ^ { \prime } ( a , b ) \leqslant d ( a , b ) , d ^ { \prime } ( a , z ) \geqslant d ( a , z )$ , and $d ^ { \prime } ( b , z ) \geqslant d ( b , z )$ . Thus,

$$
\begin{array} { r } { \rho _ { d ^ { \prime } } ( a b \mid z ) = \operatorname* { m i n } \{ d ^ { \prime } ( a , z ) , d ^ { \prime } ( b , z ) \} - d ^ { \prime } ( a , b ) \geqslant \operatorname* { m i n } \{ d ( a , z ) , d ( b , z ) \} - d ( a , b ) = \rho _ { d } ( a b \mid z ) . } \end{array}
$$

Averaging and then minimizing over all admissible triples gives $\iota ^ { d ^ { \prime } } ( C ) \geqslant \iota ^ { d } ( C ) > 0$ . Hence $C \in T _ { \mathrm { s t a b l e } } ( d ^ { \prime } )$ , proving cluster-wise consistency.

$\not \angle \cdot$ . Permutation invariance. Let ϕ be a permutation of X, and recall $d _ { \phi } ( x , y ) : = d ( \phi ^ { - 1 } ( x ) , \phi ^ { - 1 } ( y ) )$ For every $u , v , z \in \mathcal { X }$ , we have $\rho _ { d _ { \phi } } \big ( \phi ( u ) \phi ( v ) | \phi ( z ) \big ) \ = \ \rho _ { d } ( u v | z )$ . Moreover, the map $( U , V , Z ) \longmapsto \left( \phi ( U ) , \phi ( V ) , \phi ( Z ) \right)$ is a bijection between the triples admissible in the definition of $\iota ^ { d } ( C )$ and those admissible in the definition of $\iota ^ { d _ { \phi } } ( \phi ( C ) )$ q. Therefore $\iota ^ { d _ { \phi } } ( \phi ( C ) ) =$ $\iota ^ { d } ( C )$ for every $C \subseteq { \mathcal { X } }$ . Hence

$$
C \in T _ { \mathrm { s t a b l e } } ( d ) \iff \phi ( C ) \in T _ { \mathrm { s t a b l e } } ( d _ { \phi } ) ,
$$

proving $T _ { \mathrm { s t a b l e } } ( d _ { \phi } ) = \phi \cdot T _ { \mathrm { s t a b l e } } ( d )$

## Appendix D. Proofs for Section 4

## D.1 Incompatibility among Methods

Proof [Proof of Lemma 13] First, we consider $\mathcal { X } = [ 4 ]$ . We show that for any $0 < p < q$ the methods $T _ { \mathrm { s t a b l e } } ^ { ( p ) }$ and $T _ { \mathrm { s t a b l e } } ^ { ( q ) }$ admit no common refinement that outputs hierarchies. Let

$$
\mathcal { X } = \{ 1 , 2 , 3 , 4 \} , \qquad C _ { 1 } : = \{ 1 , 2 , 3 \} , \qquad C _ { 2 } : = \{ 2 , 3 , 4 \} .
$$

Then $C _ { 1 } \cap C _ { 2 } = \{ 2 , 3 \}$ is nonempty while neither $C _ { 1 } \subseteq C _ { 2 }$ nor $C _ { 2 } \subseteq C _ { 1 }$ , so $C _ { 1 }$ and $C _ { 2 }$ are not laminar-compatible; in particular, no hierarchy on $\mathcal { X }$ contains both. We construct $d \in { \mathcal { D } } ( { \mathcal { X } } )$ such that $C _ { 1 } \in T _ { \mathrm { s t a b l e } } ^ { ( p ) } ( d )$ and $C _ { 2 } \in T _ { \mathrm { s t a b l e } } ^ { ( q ) } ( d )$ , hence prove that $T _ { \mathrm { s t a b l e } } ^ { ( p ) }$ and $T _ { \mathrm { s t a b l e } } ^ { ( q ) }$ are mutually incompatible.

Construction of d. Let $r : = q / p > 1$ . Fix $\alpha > 0$ small enough that $\left( 1 + \alpha / 2 \right) ^ { r } < 2 ;$ such an α exists because $( 1 + \alpha / 2 ) ^ { r }  1$ as $\alpha \to 0 ^ { + }$ . Set $\begin{array} { r } { h : = 1 + \alpha , t _ { 0 } : = 1 + \frac { \alpha } { 2 } = \frac { 1 + h } { 2 } } \end{array}$ . Notice $\begin{array} { r } { t _ { 0 } ^ { r } = \left( \frac { 1 + h } { 2 } \right) ^ { r } < \frac { 1 + h ^ { r } } { 2 } } \end{array}$ , equivalently $1 + h ^ { r } > 2 t _ { 0 } ^ { r }$ holds. By continuity, both strict inequalities $t _ { 0 } ^ { r } < 2$ and $1 + h ^ { r } > 2 t _ { 0 } ^ { r }$ persist after a small perturbation of $t _ { 0 } { \mathrm { : } }$ there exists $\delta \in ( 0 , \alpha / 2 )$ such that $t : = t _ { 0 } + \delta$ satisfies $t ^ { r } < 2$ and $1 + h ^ { r } > 2 t ^ { r }$ . The constraint $\delta < \alpha / 2$ ensures $1 < t < h$ Finally, choose ε with $0 < \varepsilon < 1$ and $\varepsilon ^ { r } < 2 - t ^ { r }$ ; both constraints are compatible because $t ^ { r } < 2$

Pick any $M > h$ , and define $d \in { \mathcal { D } } ( { \mathcal { X } } )$ by prescribing the p-th powers of its values:

$$
\begin{array} { l l } { { d ( 1 , 2 ) ^ { p } = 1 , } } & { { d ( 1 , 3 ) ^ { p } = h , d ( 1 , 4 ) ^ { p } = M , } } \\ { { d ( 2 , 3 ) ^ { p } = \varepsilon , } } & { { d ( 2 , 4 ) ^ { p } = t , d ( 3 , 4 ) ^ { p } = t . } } \end{array}
$$

All six prescribed values are strictly positive, so d is a valid dissimilarity. By using the ordering $\varepsilon < 1 < t < h < M$ , we can check by direct computation that $C _ { 1 } \in T _ { \mathrm { s t a b l e } } ^ { ( p ) } ( d )$ whereas $C _ { 2 } \in T _ { \mathrm { s t a b l e } } ^ { ( q ) } ( d )$

Suppose, for contradiction, that T refines both $T _ { \mathrm { s t a b l e } } ^ { ( p ) }$ and $T _ { \mathrm { s t a b l e } } ^ { ( q ) } .$ . Then $T ( d )$ contains both $C _ { 1 }$ and $C _ { 2 }$ , contradicting laminarity.

For a domain $\mathcal { X } ^ { \prime }$ with $| \mathcal { X } ^ { \prime } | > 4$ , retain the above dissimilarities on $\{ 1 , 2 , 3 , 4 \}$ and set $d ( w , i ) = L$ for every new point w and every $i \in \{ 1 , 2 , 3 , 4 \}$ , where L is larger than every dissimilarity in the four-point construction; assign arbitrary positive symmetric dissimilarities among the new points. For any new external witness w, every isolation weight contributing to the stability of $C _ { 1 }$ under power $p$ or of $C _ { 2 }$ under power $q$ is strictly positive. An average over an external set containing both old and new witnesses is therefore a positive average of the already verified terms and these new positive terms. Hence $C _ { 1 } \in T _ { \mathrm { s t a b l e } } ^ { ( p ) } ( d )$ and $C _ { 2 } \in T _ { \mathrm { s t a b l e } } ^ { ( q ) } ( d )$ still hold on $\mathcal { X } ^ { \prime }$ , proving incompatibility for every $| { \mathcal { X } } ^ { \prime } | \geqslant 4$

Lemma 52 Let $p > 0$ . The methods $T _ { \mathrm { S L } }$ and $T _ { \mathrm { s t a b l e } } ^ { ( p ) }$ are incompatible.

Proof We first construct a dissimilarity function $d _ { 0 }$ on $\mathcal { X } = \{ 1 , 2 , 3 , 4 \}$ such that $T _ { \mathrm { S L } } ( d _ { 0 } )$ and $T _ { \mathrm { s t a b l e } } ^ { ( p ) } ( d _ { 0 } )$ are incompatible. Define $d _ { 0 } \in \mathcal { D } ( \mathcal { X } )$ such that

$$
d _ { 0 } ( 1 , 2 ) = 3 , \quad d _ { 0 } ( 1 , 3 ) = d _ { 0 } ( 1 , 4 ) = 1 0 , \quad d _ { 0 } ( 2 , 3 ) = 1 , \quad d _ { 0 } ( 2 , 4 ) = d _ { 0 } ( 3 , 4 ) = 4 .
$$

We can verify that $\{ 1 , 2 , 3 \} \in T _ { \mathrm { S L } } ( d _ { 0 } )$ while $\{ 2 , 3 , 4 \} \in T _ { \mathrm { s t a b l e } } ( d _ { 0 } )$ . The clusters t1, 2, 3u and t2, 3, 4u overlap, but neither contains the other. Thus $T _ { \mathrm { S L } } ( d _ { 0 } ) \cup T _ { \mathrm { s t a b l e } } ( d _ { 0 } )$ is not laminar.

Now fix $p > 0$ and define d entry-wise by $d ( x , y ) : = d _ { 0 } ( x , y ) ^ { 1 / p }$ . Then $\mu ^ { p } ( d ) = d _ { 0 }$ . Because single linkage is order invariant, $T _ { \mathrm { S L } } ( d ) ~ = ~ T _ { \mathrm { S L } } ( d _ { 0 } )$ , whereas, by definition, $T _ { \mathrm { s t a b l e } } ^ { ( p ) } ( d ) =$ $T _ { \mathrm { s t a b l e } } ( \mu ^ { p } ( d ) ) = T _ { \mathrm { s t a b l e } } ( d _ { 0 } )$ . Hence the same two non-laminar clusters belong to $T _ { \mathrm { S L } } ( d ) \cup$ $T _ { \mathrm { s t a b l e } } ^ { ( p ) } ( d )$

For $n > 4 ,$ extend $d _ { 0 }$ by setting $d _ { 0 } ( i , j ) = 2 0$ whenever at least one of $i , j$ belongs to $\{ 5 , \ldots , n \}$ . This does not afect the single-linkage cluster $\{ 1 , 2 , 3 \}$ . Moreover, for every new point z and every $u , v \in \{ 2 , 3 , 4 \} , \rho _ { d _ { 0 } } ( u v \mid z ) = 2 0 - d _ { 0 } ( u , v ) > 0$ . Thus $\iota ^ { d _ { 0 } } ( \{ 2 , 3 , 4 \} ) \times 0$ remains valid. Taking $d ( i , j ) = d _ { 0 } ( i , j ) ^ { 1 / p }$ completes the proof for every $n \geqslant 4 .$

## D.2 Well-separated Backbone Hierarchy

## D.2.1 Proof of Theorem 15

The canonical partition consistency axiom is formulated at the level of partitions: if a partition already appears in the hierarchy, then strengthening that partition preserves all of its blocks. To establish the existence of a well-separated backbone hierarchy, however, we need a cluster-wise consequence of this axiom: a cluster that is suficiently well separated from its complement must be selected by any admissible method.

We prove it in two steps. First, we show in Lemma 53 that partition consistency yields a cluster-wise consequence: if there exists a dissimilarity $d _ { 0 }$ such that $\{ C , { \bar { C } } \} \subseteq T ( d _ { 0 } )$ the cluster $C$ remains selected under any dissimilarity that does not increase dissimilarities within C and does not decrease dissimilarities from C to $\bar { C }$ . Second, partition richness provides a reference dissimilarity $d _ { 0 }$ realizing $\{ C , { \bar { C } } \}$ . Moreover, by scale invariance, we can rescale any dissimilarity d satisfying $\varrho ( d , C ) < \eta _ { C }$ , for a suficiently small constant $\eta _ { C } > 0$ by a factor $\beta > 0$ so that $\beta d ( x , y ) \leqslant d _ { 0 } ( x , y )$ for all $x , y \in C .$ , and $\beta d ( x , z ) \geqslant d _ { 0 } ( x , z )$ for all $x \in C , \ z \in { \bar { C } }$ . Lemma 53 then gives $C \in T ( \beta d )$ , and scale invariance yields $C \in T ( d )$ . This is the content of Lemma 54.

Lemma 53 Let T be a hierarchical method satisfying partition consistency, and let $C \subsetneq { \mathcal { X } }$ be a nonempty proper cluster. Let $d _ { 0 } , d \in { \mathcal { D } } ( { \mathcal { X } } )$ such that d is $a \left\{ C \right\}$ -strengthening of d<sub>0</sub>. If $\{ C , { \bar { C } } \} \subseteq T ( d _ { 0 } )$ then $C \in T ( d )$

Proof Define $d _ { 1 } \in \mathcal { D } ( \mathcal { X } )$ by setting

$$
d _ { 1 } ( x , y ) : = \operatorname* { m i n } \{ d ( x , y ) , d _ { 0 } ( x , y ) \} { \mathrm { ~ w h e n ~ } } x , y \in { \bar { C } } , \quad { \mathrm { ~ a n d ~ } } d _ { 1 } ( x , y ) : = d _ { 0 } ( x , y ) { \mathrm { ~ o t h e r w i s e } } .
$$

Then $d _ { 1 }$ is a $\{ C , { \bar { C } } \}$ -strengthening of $d _ { 0 }$ . Because $\{ C , { \bar { C } } \} \subseteq T ( d _ { 0 } )$ , partition consistency ensures that $\{ C , \bar { C } \} \subseteq T ( d _ { 1 } )$ . In particular, the partition ${ \mathcal { P } } _ { C } : = \{ C \} \cup \{ \{ x \} : x \in { \bar { C } } \}$ satisfies ${ \mathcal { P } } _ { C } \subseteq T ( d _ { 1 } )$ , because every hierarchy contains all singleton clusters.

We now prove that d is a P<sub>C</sub>-strengthening of $d _ { 1 }$ . Indeed, we have

$d ( x , y ) \in d _ { 0 } ( x , y ) = d _ { 1 } ( x , y )$ for every $x , y \in C ,$

$d ( x , y ) \geqslant d _ { 0 } ( x , y ) = d _ { 1 } ( x , y )$ for every $x \in C , y \in { \bar { C } } ,$

$d ( x , y ) \geqslant \operatorname* { m i n } \left\{ ( d ( x , y ) , d _ { 0 } ( x , y ) \right\} = d _ { 1 } ( x , y )$ for every $x , y \in { \bar { C } } .$

Thus, partition consistency ensures ${ \mathcal { P } } _ { C } \subseteq T ( d )$ , and hence $C \in T ( d )$

The previous lemma reduces the problem to finding a rescaling of a given dissimilarity d which is dominated by $d _ { 0 }$ inside $C ,$ but which dominates $d _ { 0 }$ across the cut $( C , { \bar { C } } )$ . Such a rescaling exists whenever the internal diameter of C under d is suficiently small compared to its distance from the complement.

Lemma 54 Let T satisfy scale invariance, partition richness, and partition consistency. For any cluster $C \subsetneq { \mathcal { X } }$ , with $| C | \geqslant 2$ , there exists $\eta _ { C } > 0$ such that, for every $d \in { \mathcal { D } } ( { \mathcal { X } } )$

$$
\varrho ( d , C ) < \eta _ { C } \quad \Longrightarrow \quad C \in { \cal T } ( d ) .
$$

If T further satisfies permutation invariance, η<sub>C</sub> only depends on the size |C| and $\eta _ { C } \in ( 0 , 1 ]$

Applying Lemma 54 to an admissible method $T _ { i }$ , define the separation margin sequence $\eta = ( \eta _ { s } ) _ { 1 \leqslant s \leqslant n - 2 }$ as provided by Lemma 54. Then every non-singleton proper cluster $C \in$ $T _ { \mathrm { g l o b } } ^ { \eta } ( d )$ satisfies $\varrho ( d , C ) < \eta _ { | C | - 1 } .$ , and therefore belongs to $T ( d )$ . Because both hierarchies contain the root and all singleton clusters, $T _ { \mathrm { g l o b } } ^ { \eta } ( d ) \subseteq T ( d )$ for every $d \in { \mathcal { D } } ( { \mathcal { X } } )$ , and hence $T _ { \mathrm { g l o b } } ^ { \eta } \subseteq T$ . This proves Theorem 15.

Proof [Proof of Lemma 54] By partition richness, choose $d _ { 0 } \in \mathcal { D } ( \mathcal { X } )$ such that $\{ C , { \bar { C } } \} \subseteq$   
$T ( d _ { 0 } )$ . Define $\begin{array} { r } { a : = \operatorname* { m i n } _ { x , y \in C } d _ { 0 } ( x , y ) } \end{array}$ and $b : = \operatorname* { m a x } _ { x \in C , \ z \in \bar { C } } d _ { 0 } ( x , z )$ . Both a and b are positive. x‰y   
Set $\begin{array} { r } { \eta _ { C } : = \frac { a } { b } } \end{array}$

Now let d be a dissimilarity satisfying $\varrho ( d , C ) < \eta _ { C }$ . Write $M : = \operatorname* { m a x } _ { x , y \in C } d ( x , y )$ and $\begin{array} { r } { m : = \operatorname* { m i n } _ { x \in C , \ z \in \bar { C } } d ( x , z ) } \end{array}$ . Then $\begin{array} { r } { \frac { M } { m } = \varrho ( d , C ) < \frac { a } { b } } \end{array}$ , and hence $\begin{array} { r } { \frac { b } { m } < \frac { a } { M } } \end{array}$ . Next, choose $\beta > 0$ such that $\begin{array} { r } { \frac { b } { m } < \beta < \frac { a } { M } } \end{array}$ . This choice ensures that,

$$
\beta d ( x , y ) \leqslant \beta M < a \leqslant d _ { 0 } ( x , y ) , \quad \mathrm { f o r ~ a l l ~ } x , y \in C , \mathrm { ~ a n d }
$$

$$
\beta d ( x , z ) \geqslant \beta m > b \geqslant d _ { 0 } ( x , z ) \quad { \mathrm { f o r ~ a l l ~ } } x \in C , z \in { \bar { C } } .
$$

Thus, by Lemma 53, applied to $\beta d ,$ we get $C \in T ( \beta d )$ . Finally, scale invariance gives $T ( \beta d ) = T ( d )$ , hence $C \in T ( d )$

Assume now that $T$ is permutation invariant. For every non-singleton proper cluster $C \subsetneq { \mathcal { X } }$ , define

$$
\Gamma _ { T } ( C ) : = \operatorname* { s u p } _ { \begin{array} { c } { d _ { 0 } \in \mathscr { D } ( \mathcal { X } ) } \\ { \{ C , \bar { C } \} \subseteq T ( d _ { 0 } ) } \end{array} } \frac { \operatorname* { m i n } _ { x , y \in C } d _ { 0 } ( x , y ) } { \operatorname* { m a x } _ { x \in C , \ z \in \bar { C } } d _ { 0 } ( x , z ) } .
$$

By the previous paragraph, for every $d \in { \mathcal { D } } ( { \mathcal { X } } )$ , we have: $\varrho ( d , C ) < \Gamma _ { T } ( C ) \Longrightarrow C \in T ( d )$

We now prove that $\Gamma _ { T } ( C )$ depends only on the cardinality of $C .$ . Let $C , C ^ { \prime } \subsetneq \mathcal { X }$ be two nontrivial subsets such that $| C | = | C ^ { \prime } |$ . Because $\mathcal { X }$ is finite, there exists a permutation $\phi \in \Pi ( { \mathcal { X } } )$ such that $\phi ( C ) = C ^ { \prime }$ . Because permutation invariance of $T$ implies $\Gamma _ { T } ( \phi ( C ) ) =$ $\Gamma _ { T } ( C )$ , we have $\Gamma _ { T } ( C ^ { \prime } ) = \Gamma _ { T } ( C )$ . Hence $\Gamma _ { T } ( C )$ is constant over all subsets of $\mathcal { X }$ having the same cardinality, and thus depends only on |C|. Thus, for $1 \leqslant s \leqslant | \mathcal { X } | - 2$ , we may define

$$
\eta _ { s } : = \Gamma _ { T } ( C )
$$

for any subset $C \subseteq { \mathcal { X } }$ with $s = | C | - 1$ . This is well-defined by the preceding argument, and, for every $C \subseteq { \mathcal { X } }$

$$
\varrho ( d , C ) < \eta _ { | C | - 1 } \quad \Longrightarrow \quad C \in T ( d ) .
$$

It remains to prove that, for $1 \leqslant s \leqslant | \mathcal { X } | - 2$ , one has $0 < \eta _ { s } \leqslant 1$ . Observe that $\eta _ { s } > 0$ follows from partition richness, as for each C there exists $d _ { 0 }$ such that $\{ C , { \bar { C } } \} \subseteq T ( d _ { 0 } )$ , and the corresponding ratio is strictly positive.

To establish $\eta _ { s } \leqslant 1$ , suppose by contradiction that $\eta _ { s } > 1$ for some $1 \leqslant s \leqslant | \mathcal { X } | - 2$ . Let d be the uniform dissimilarity on $x ,$ , namely $d ( x , y ) = 1$ for all $x \neq y$ . Then $\varrho ( d , C ) = 1 < \eta _ { s }$ for every subset $C \subsetneq { \mathcal { X } }$ of size $s + 1$ . Hence every such C belongs to $T ( d )$ . But there exist two subsets of $\mathcal { X }$ of size $s + 1$ that overlap without either containing the other. For instance, since $1 \leqslant s \leqslant n - 2$ , the sets $C _ { 1 } : = \{ 1 , \dots , s + 1 \}$ and $C _ { 2 } : = \{ 2 , \dots , s + 2 \}$ both have cardinality $s + 1$ , overlap nontrivially, and neither contains the other. This contradicts the laminarity of $T ( d )$ . Therefore $\eta _ { s } \leqslant 1$

## D.3 Additional Lemmas

Lemma 55 We have $\alpha _ { \mathrm { u h r } } \wedge \alpha _ { \mathrm { p c } } \triangleright \alpha _ { \mathrm { b h } } ^ { 1 }$

Proof Let $T \in { \mathcal { H } } _ { \alpha _ { \mathrm { u h r } } \wedge \alpha _ { \mathrm { p c } } } ( \chi )$ . We prove that $T _ { \mathrm { g l o b } } \subseteq T$ . Fix $d \in { \mathcal { D } } ( { \mathcal { X } } )$ and $C \in T _ { \mathrm { g l o b } } ( d )$ The cases $C = \mathcal { X }$ and $| C | = 1$ are immediate because every hierarchy contains the root and all singleton clusters. We therefore assume that $C \subsetneq { \mathcal { X } }$ and $| C | \geqslant 2$ . Define $\alpha : =$ $\operatorname* { m a x } _ { \boldsymbol { x } , \boldsymbol { y } \in C } d ( \boldsymbol { x } , \boldsymbol { y } )$ and $\beta : = \operatorname* { m i n } _ { z \in C } d ( x , z )$ . Because $C \in T _ { \mathrm { g l o b } } ( d )$ , we have $\alpha < \beta$ . Chooseα $<$ x‰y $\widetilde { \beta } < \beta$ and set $\gamma : =$ min $\left( \{ \widetilde { \beta } \} \cup \{ d ( z , w ) : z , w \in \bar { C } , \ z \neq w \} \right)$ . In particular, $0 < \gamma \leqslant \widetilde { \beta }$

Define $u \in { \mathcal { D } } ( { \mathcal { X } } )$ by ${ \dot { u ( x , x ) } } : = 0$ and, for distinct $x , y \in { \mathcal { X } }$

$$
u ( x , y ) : = \left\{ \begin{array} { l l } { \alpha , } & { x , y \in C , } \\ { \widetilde { \beta } , } & { x \in C , \ y \in \bar { C } \mathrm { ~ o r ~ } y \in C , \ x \in \bar { C } , } \\ { \gamma , } & { x , y \in \bar { C } . } \end{array} \right.
$$

Because $\alpha < { \widetilde { \beta } }$ and $\gamma \leqslant \widetilde { \beta }$ , u is an ultrametric: every triangle meeting both $C$ and $\bar { C }$ has two dissimilarities equal to $\beta ,$ while the third is either $\alpha \ \mathrm { o r } \ \gamma ,$ and triangles contained entirely in $C$ or in $\bar { C }$ are equilateral.

Recall the ultrametric-ball representation in Equation (3). For every $x \in C , B _ { u } ( x , \alpha ) =$ C because $u ( x , y ) \leqslant \alpha$ for $y \in C ,$ whereas $u ( x , z ) = \widetilde { \beta } > \alpha$ for $z \in \bar { C }$ . Hence $C \in \Psi _ { u } . \mathrm { ~ A s ~ } T$ satisfies refinement on ultrametrics, $C \in \Psi _ { u } \subseteq T ( u )$

Now consider the partition ${ \mathcal { P } } _ { C } : = \{ C \} \cup \left\{ \{ z \} : z \in { \bar { C } } \right\}$ (. Because every hierarchy contains all singleton clusters, we have $\mathcal { P } _ { C } \subseteq T ( u )$ . We claim that d is a P<sub>C</sub>-strengthening of u. Indeed, for distinct $x , y \in C , d ( x , y ) \leqslant \alpha = u ( x , y )$ . For $x \in C$ and $z \in \bar { C } , d ( x , z ) \geqslant \beta >$ $\widetilde { \beta } = u ( x , z )$ . Finally, for distinct $z , w \in \bar { C }$ , the points z and w belong to diferent singleton blocks of $\mathcal { P } _ { C }$ , and the definition of γ gives $d ( z , w ) \geqslant \gamma = u ( z , w )$

Partition consistency therefore yields ${ \mathcal { P } } _ { C } \subseteq T ( d )$ , and hence $C ~ \in ~ T ( d )$ Because $d \in { \mathcal { D } } ( { \mathcal { X } } )$ and $C \in T _ { \mathrm { g l o b } } ( d )$ were arbitrary, we conclude that $T _ { \mathrm { g l o b } } \subseteq T$ ■

Lemma 56 We have $\alpha _ { \mathrm { a d m } } \wedge \alpha _ { \mathrm { o r d } } \circ \alpha _ { \mathrm { b h } } ^ { 1 } , i . e .$ , for any $T \in { \mathcal { H } } _ { \alpha _ { \mathrm { a d m } } \wedge \alpha _ { \mathrm { o r d } } } , T _ { \mathrm { g l o b } } \subseteq T$ holds.

Proof Let T be admissible and order invariant. By Theorem 15, there exists a separation margin sequence $\eta = ( \eta _ { s } ) _ { 1 \leqslant s \leqslant n - 2 }$ such that $T _ { \mathrm { g l o b } } ^ { \eta } \subseteq T$ . Set $\eta _ { \mathrm { m i n } } : = \mathrm { m i n } _ { 1 \leqslant s \leqslant n - 2 } \eta _ { s } > 0$

Fix d $\mathbf { \chi } ^ { \prime } \in \mathcal { D } ( \mathcal { X } )$ . We construct a strictly increasing function $g : \mathbb { R } _ { \geq 0 }  \mathbb { R } _ { \geqslant 0 }$ , with $g ( 0 ) = 0$ such that $T _ { \mathrm { g l o b } } ( d ) \subseteq T _ { \mathrm { g l o b } } ^ { \eta } ( g \circ d )$

Let $0 < \delta _ { 1 } < \delta _ { 2 } < \cdots < \delta _ { m }$ be the distinct positive values taken by d. Choose $\begin{array} { r } { q > \frac { 1 } { \eta _ { \mathrm { m i n } } } } \end{array}$ 2 and prescribe $g ( 0 ) : = 0$ and $g ( \delta _ { i } ) : = q ^ { i }$ for all $i \in [ m ]$ . Because the sequences $( \delta _ { i } )$ and $( q ^ { i } )$ are strictly increasing, these prescribed values can be extended to a strictly increasing function $g : \mathbb { R } _ { \geq 0 }  \mathbb { R } _ { \geqslant 0 }$ with $g ( 0 ) = 0$ ; for instance, using piecewise-linear interpolation on $[ 0 , \delta _ { m } ]$ and extending linearly with positive slope beyond $\delta _ { m }$

Now let $C \in T _ { \mathrm { g l o b } } ( d )$ be a nontrivial proper cluster. By definition, $d ( x _ { 1 } , y ) < d ( x _ { 2 } , z )$ for every $x _ { 1 } , x _ { 2 } , y ~ \in ~ C$ and $z \not \in C$ whenever $d ( x _ { 1 } , y ) ~ > ~ 0$ . Hence, if $d ( x _ { 1 } , y ) ~ = ~ \delta _ { i }$ and $d ( x _ { 2 } , z ) = \delta _ { j }$ , then $i < j$ , and therefore

$$
\frac { g ( d ( x _ { 1 } , y ) ) } { g ( d ( x _ { 2 } , z ) ) } \ = \ q ^ { i - j } \ \leqslant \ \frac { 1 } { q } \ < \ \eta _ { \mathrm { m i n } } \ \leqslant \ \eta _ { | C | - 1 } .
$$

If $d ( x _ { 1 } , y ) = 0$ , the same inequality holds trivially because $g ( 0 ) = 0$

Therefore, $\varrho ( g \circ d , C ) < \eta _ { | C | - 1 }$ , and hence $C \in T _ { \mathrm { g l o b } } ^ { \eta } ( g \circ d )$ . The root and singleton clusters belong to both hierarchies by definition, so $T _ { \mathrm { g l o b } } ( d ) \subseteq T _ { \mathrm { g l o b } } ^ { \eta } ( g \circ d )$ . Using $T _ { \mathrm { g l o b } } ^ { \eta } \subseteq T$ we further obtain

$$
T _ { \mathrm { g l o b } } ( d ) \subseteq T _ { \mathrm { g l o b } } ^ { \eta } ( g \circ d ) \subseteq T ( g \circ d ) .
$$

Finally, order invariance of T gives $T ( g \circ d ) = T ( d )$ , and therefore $T _ { \mathrm { g l o b } } ( d ) \subseteq T ( d )$ . Because $d \in { \mathcal { D } } ( { \mathcal { X } } )$ was arbitrary, $T _ { \mathrm { g l o b } } \subseteq T$ ■

## Appendix E. Empirical Size of Backbone Hierarchy

We quantify the size of the backbone hierarchy across four standard scikit-learn datasets (Pedregosa et al., 2011): Iris, Wine, Breast Cancer, and Digits. We remove exact duplicate

feature vectors, standardize each feature, and use Euclidean dissimilarities. We count only nontrivial clusters $C$ satisfying $1 < | C | < n$ , and report the ratio of the number returned by $T _ { \mathrm { g l o b } }$ to the number returned by $T _ { \mathrm { S L } }$

Table 3: Size of $T _ { \mathrm { g l o b } }$ relative to the non-binary single-linkage hierarchy $T _ { \mathrm { S L } }$
<table><tr><td>Dataset</td><td>n</td><td> $T _ { \mathrm { g l o b } }$ </td><td> $T _ { \mathrm { S L } }$ </td><td>Ratio</td></tr><tr><td>Iris</td><td>149</td><td>42</td><td>145</td><td>0.290</td></tr><tr><td>Wine</td><td>178</td><td>43</td><td>176</td><td>0.244</td></tr><tr><td>Breast Cancer</td><td>569</td><td>107</td><td>567</td><td>0.189</td></tr><tr><td>Digits</td><td>1797</td><td>426</td><td>1795</td><td>0.237</td></tr></table>

Across these datasets, $T _ { \mathrm { g l o b } }$ contains approximately 19%–29% of the nontrivial clusters in $T _ { \mathrm { S L } }$ (23.0% in aggregate). Thus, the backbone is nontrivial on these examples.

## References

Margareta Ackerman and Shai Ben-David. A characterization of linkage-based hierarchical clustering. Journal of Machine Learning Research, 17(231):1–17, 2016.

Margareta Ackerman and Sanjoy Dasgupta. Incremental clustering: The case for extra clusters. Advances in Neural Information Processing Systems, 27, 2014.

Margareta Ackerman, Shai Ben-David, and David Loker. Characterization of linkage-based clustering. In COLT, volume 2010, pages 270–281, 2010.

Ju D Apresjan. An algorithm for constructing clusters from a distance matrix. Mashinnyi perevod: prikladnaja lingvistika, 9:3–18, 1966.

Ery Arias-Castro and Elizabeth Coda. An axiomatic definition of hierarchical clustering. Journal of Machine Learning Research, 26(10):1–26, 2025.

Maria-Florina Balcan, Avrim Blum, and Santosh Vempala. A discriminative framework for clustering via similarity functions. In Proceedings of the Fortieth annual ACM Symposium on Theory of Computing, pages 671–680, 2008.

Shai Ben-David and Margareta Ackerman. Measures of clustering quality: A working set of axioms for clustering. Advances in Neural Information Processing Systems, 21, 2008.

David Bryant and Vincent Berry. A structured family of clustering and tree construction methods. Advances in Applied Mathematics, 27(4):705–732, 2001.

Gunnar Carlsson and Facundo M´emoli. Classifying clustering schemes. Foundations of Computational Mathematics, 13(2):221–252, 2013.

Gunnar Carlsson, Facundo M´emoli, Alejandro Ribeiro, and Santiago Segarra. Hierarchical quasi-clustering methods for asymmetric networks. In International Conference on Machine Learning, pages 352–360. PMLR, 2014.

Gunnar Carlsson, Facundo M´emoli, Alejandro Ribeiro, and Santiago Segarra. Admissible hierarchical clustering methods and algorithms for asymmetric networks. IEEE Transactions on Signal and Information Processing over Networks, 3(4):711–727, 2017.

Gunnar E Carlsson and Facundo M´emoli. Characterization, stability and convergence of hierarchical clustering methods. Journal of Machine Learning Research, 11(Apr):1425– 1470, 2010.

Vincent Cohen-Addad, Varun Kanade, and Frederik Mallmann-Trenn. Clustering redemption–beyond the impossibility of Kleinberg’s axioms. Advances in Neural Information Processing Systems, 31, 2018.

Vincent Cohen-Addad, Varun Kanade, Frederik Mallmann-Trenn, and Claire Mathieu. Hierarchical clustering: Objective functions and algorithms. Journal of the ACM (JACM), 66(4):1–42, 2019.

Sanjoy Dasgupta. A cost function for similarity-based hierarchical clustering. In Proceedings of the Forty-Eighth Annual ACM Symposium on Theory of Computing, STOC ’16, page 118–127, New York, NY, USA, 2016. Association for Computing Machinery. ISBN 9781450341325.

Jean Diatta and Bernard Fichet. From Apresjan hierarchies and Bandelt-Dress weak hierarchies to quasi-hierarchies. In New approaches in classification and data analysis, pages 111–118. Springer, 1994.

Maximilien Dreveton, Matthias Grossglauser, Daichi Kuroda, and Patrick Thiran. Hierarchical linkage clustering beyond binary trees and ultrametrics, 2025. URL https: //arxiv.org/abs/2511.18056.

Jon Kleinberg. An impossibility theorem for clustering. Advances in Neural Information Processing Systems, 15, 2002.

Fionn Murtagh and Pedro Contreras. Algorithms for hierarchical clustering: an overview, II. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 7(6):e1219, 2017.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.

Jan Puzicha, Thomas Hofmann, and Joachim M Buhmann. A theory of proximity based clustering: Structure detection by optimization. Pattern Recognition, 33(4):617–634, 2000.

Alexander Shen and Nikolai Konstantinovich Vereshchagin. Basic set theory. American Mathematical Society, 2002.

Fabio Strazzeri and Rub´en J S´anchez-Garc´ıa. Possibility results for graph clustering: A novel consistency axiom. Pattern Recognition, 128:108687, 2022.

Philipp Thomann, Ingo Steinwart, and Nico Schmid. Towards an axiomatic approach to hierarchical clustering of measures. Journal of Machine Learning Research, 16(1):1949– 2002, 2015.

Twan Van Laarhoven and Elena Marchiori. Axioms for graph clustering quality functions. Journal of Machine Learning Research, 15(1):193–215, 2014.

James Willson and Tandy Warnow. Axioms for clustering simple unweighted graphs: No impossibility result. PLOS Complex Systems, 1(2):e0000011, 2024.

Reza Bosagh Zadeh and Shai Ben-David. A uniqueness theorem for clustering. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, pages 639–646, 2009.