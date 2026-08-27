# M-Fibration Theory with Applications to Neural Network Compression

Paolo Boldi

Dipartimento di Informatica, Universit\`a degli Studi di Milano, Milan, Italy paolo.boldi@unimi.it

## Abstract

The purpose of this paper is to provide a general, comprehensive, theoretical framework that allows one to deal with fibrations on graphs labelled on a commutative monoid. This is a genuine extension of the theory of graph fibrations (as introduced in [1]), that makes it possible to deal with weighted graphs, and also graphs labelled with other algebraic structures. The derived theory also lends itself naturally to consider approximate fibrations. As an example, we show how this framework can be applied to the compression of arbitrary neural networks (including CNNs), providing a strong theoretical underpinning to the recent results in [2].

Keywords: Network analysis, fibrations, neural networks

## 1 Introduction and Motivations

Graph fibrations were formally described in [1], derived from the much more general definition by Grothendieck [3]. The main motivation, at the time, was to have a formal tool to describe computations in distributed systems. Similar notions had already been used in very diferent settings, with diferent goals and under diferent names: equitable partitions [4, 5], the Weisfeiler–Leman test [6], colour refinement [7], groupoids [8].

While fibrations are still used in the context of distributed computing (see, e.g., [9]), in the last decades the fibration machinery was borrowed in other, quite diferent, playgrounds as means to describe symmetries in general networks: specifically, in the theory of dynamical systems [8, 10, 11] and in biology [12–14].

Although the theory of fibrations is elegant and simple, there are two important shortcomings that make its usage dificult in these more general contexts. The first is that it is challenging to adapt fibrations to weighted graphs, or more generally to graphs with labels that can be algebraically composed; the theory, as described in [1], allows for labels, but treats them in a “static” way: labels can exist on arcs, but they must be preserved exactly by morphisms. The second problem is that the presence of errors or noise is incompatible with the theory itself. One extra, or one missing arc, or even an arc with a weight that is slightly of, and everything breaks.

These problems can be overcome with ad hoc approaches, and in fact fibrations have been successfully applied also in these more general contexts [2, 13, 15]. Nonetheless, a well-grounded theoretical foundation behind their use is still missing.

The purpose of this paper is to provide a general, comprehensive, theoretical framework that allows one to deal with fibrations on graphs labelled on a commutative monoid. The idea is similar to that of weighted bisimulation of weighted automata [16], and akin to exact lumpability of Markov chains [17, 18], which is the special case in which labels are non-negative reals under addition; other instances of the same idea appear as equitable partitions of weighted graphs and their role in the spectral theory of graphs [4, 5] and, more recently, in the analysis of the expressive power of message-passing graph neural networks [19, 20], and as a tool for the lossless dimension reduction of linear programs [21]. What these approaches have in common is a partition-refinement algorithm [22]; what they lack is a common conceptual framework in which the objects being computed (bases and fibrations, rather than partitions) are first-class citizens, together with their universal properties. Providing such a framework is the purpose of this paper.

As an example of application, we show how the derived theory can be applied to the compression of arbitrary neural networks (including CNNs), providing a strong theoretical underpinning to the recent results in [2].

## 2 Notations and Preliminaries

Equivalence relations and partitions. An equivalence relation on a set X is a binary relation ∼ that is symmetric, reflexive and transitive. A partition of a set X is a collection Π of pairwise disjoint non-empty subsets of X whose union is X itself; the elements of Π are called the parts of the partition. Partitions and equivalence relations are in one-to-one correspondence: given an equivalence relation ∼ on X, the set of equivalence classes of ∼ is a partition of $X$ , denoted by $\Pi _ { \sim } .$ and given a partition Π of X, the relation ${ \sim } _ { \Pi }$ defined by $x \sim _ { \Pi } y$ if and only if x and y belong to the same part of Π is an equivalence relation on X.

An equivalence relation (or partition) ∼<sub>1</sub> is said to be finer than an equivalence relation ${ \sim } _ { 2 }$ (or, equivalently, ${ \sim } _ { 2 }$ is coarser than ${ \sim } _ { 1 } )$ if x $\sim _ { 1 \ y }$ implies $x \sim _ { 2 } y$ for all $x , y \in X$

Graphs and graph morphisms. A graph G is defined by $( N _ { G } , A _ { G } , s _ { G } , t _ { G } )$ , where $N _ { G }$ is a set of nodes, $A _ { G }$ is a set of arcs, and $t _ { G } , s _ { G } : A _ { G } \to N _ { G }$ provide the source and target node of each arc; in this paper, we only consider finite graphs, i.e., graphs with a finite set of nodes and arcs. In the following, all subscripts are omitted when they are clear from the context.

We write $G ( x , y )$ for the set of arcs from x to y, i.e., $G ( x , y ) = s ^ { - 1 } ( x ) \cap t ^ { - 1 } ( y )$ , and extend this notation to sets of nodes $X , Y \subseteq N _ { G }$ by writing $\begin{array} { r } { G ( X , Y ) = \bigcup _ { x \in X , y \in Y } G ( x , y ) } \end{array}$ . We also use − instead of $N _ { G }$ , so for instance $G ( - , y )$ is the set of arcs ending in y.

A simple graph is a graph with no parallel arcs, i.e., such that $| G ( x , y ) | \leq 1$ for all $x , y \in N _ { G }$

A graph morphism $f : G \to H$ is a pair of functions $f _ { N } : N _ { G } \to N _ { H }$ and $f _ { A } : A _ { G } \to A _ { H }$ such that the following diagrams commute:

$$
\begin{array} { c c } { { A _ { G } ~ { \xrightarrow { \ f _ { A } \ } } ~ A _ { H } } } & { { \qquad A _ { G } ~ { \xrightarrow { \ f _ { A } \ } } ~ A _ { H } } } \\ { { s _ { G } { \Biggl \downarrow } } } & { { \qquad { \ \underbrace { \Biggl \downarrow } s _ { H } } } } \\ { { N _ { G } ~ { \xrightarrow { \ f _ { N } \ } } ~ N _ { H } } } & { { \qquad N _ { G } ~ { \xrightarrow { \ f _ { N } \ } } ~ N _ { H } } } \end{array} \quad \quad \downarrow _ { H }
$$

A graph morphism $f : G \to H$ is surjective (also called epimorphic) (injective, bijective, resp.) if both $f _ { N }$ and $f _ { A }$ are surjective (injective, bijective, resp.) functions.

## 3 M-graphs and M-fibrations

A (standard) graph fibration [1] is a graph morphism $\varphi : G  H$ such that for every arc $\bar { a } \in A _ { H }$ and node $x \in \varphi ^ { - 1 } ( t ( \bar { a } ) )$ there exists a unique a $\in \varphi ^ { - 1 } ( \bar { a } ) \cap G ( - , x )$ . This condition is extremely rigid: while the whole theory of graph fibrations is very elegant and has many applications, it is not flexible enough to be applied to weighted or labelled neural networks, unless we just assume that morphisms simply preserve the labels/weights, which is usually something we don’t want. In order to overcome this limitation, we introduce the notion of M-fibrations: let us start with the definition of M-graph.

![](images/aab7f004142409909f7ef22bfc0229326bf2f0f23eb522785e4dba22528f83b7.jpg)  
Figure 1. Two graphs with labels on some monoid (for the moment, we do not specify the monoid operations). The underlying graph is the same, but the labelling is slightly diferent (in boldface, the labels that change).

Definition 3.1 (M-graph). Given a commutative monoid $( M , \oplus , 0 )$ , an M-graph is a graph G together with a function $\lambda _ { G } : A _ { G } \to M$ that assigns a label in M to each arc of G.

From now on, M will always denote a commutative monoid (sometimes enjoying further properties). In Figure 1, you can see two (simple) M-graphs: the monoid M is intentionally left unspecified. Note that here the underlying graph is the same, but the labelling is not.

Definition 3.2 (M-fibration). An M-fibration is a graph morphism $\varphi : G \to H$ between two M-graphs such that for every arc ${ \bar { a } } \in A _ { H }$ and node $x \in \varphi ^ { - 1 } ( t ( \bar { a } ) )$ we have

$$
\lambda ( \bar { a } ) = \bigoplus _ { a \in \varphi ^ { - 1 } ( \bar { a } ) \cap G ( - , x ) } \lambda ( a ) .
$$

It is easy to see that standard graph fibrations $\varphi : G  B$ are a special case of M-fibrations, where the monoid is $( \mathbb { N } , + , 0 )$ and all arcs in both G and B have label 1. In this case, the condition for being an M-fibration reduces to the condition for being a standard graph fibration (the set over which the sum is taken must have exactly one element).

Some basic properties of M-fibrations are the following:

Proposition 3.1. 1. The composition of two M-fibrations is an M-fibration.

2. A bijective M-fibration is an isomorphism of M-graphs (i.e., a bijective graph morphism that preserves the labels of arcs).

Proof. 1. Let $\varphi : G  H$ and $\psi : H \to K$ be two M-fibrations. Let $\bar { a } \in A _ { K }$ and $x \in ( \psi \circ \varphi ) ^ { - 1 } ( t ( \bar { a } ) )$ . Then we have:

$$
\bigoplus _ { \substack { a \in ( \psi \circ \varphi ) ^ { - 1 } ( a ) \cap C ( - , x ) } } \lambda ( a ) = \bigoplus _ { \substack { b \in \psi ^ { - 1 } ( a ) \cap H ( - , \varphi ( x ) ) a \in \varphi ^ { - 1 } ( b ) \cap C ( - , x ) } } \lambda ( a ) = \bigoplus _ { \substack { b \in \psi ^ { - 1 } ( a ) \cap H ( - , \varphi ( x ) ) } } \lambda ( b ) = \lambda ( a ) .
$$

2. Let $\varphi : G  H$ be a bijective M-fibration. Then for every arc $\bar { a } \in A _ { H }$ and node $x \in \varphi ^ { - 1 } ( t ( \bar { a } ) )$ we have

$$
\lambda ( \bar { a } ) = \bigoplus _ { a \in \varphi ^ { - 1 } ( \bar { a } ) \cap G ( - , x ) } \lambda ( a ) .
$$

But since $\varphi$ is bijective, there is exactly one node $x \in \varphi ^ { - 1 } ( t ( \bar { a } ) )$ and the set $\varphi ^ { - 1 } ( \bar { a } ) \cap G ( - , x )$ has exactly one element, so the last formula reduces to $\lambda ( \bar { a } ) = \lambda ( \varphi ^ { - 1 } ( \bar { a } ) )$ . □

## 4 Minimum base of an M-graph

The theory of graph fibrations [1] ensures that every graph has a minimum base, i.e., for every graph G there exists an epimorphic fibration $\varphi : G  B$ whose base B is fibration prime (it cannot be fibred non-trivially), and $B$ is unique up to isomorphism; moreover, all such minimal fibrations have the same node component (up to isomorphism of the base), although they may difer on arcs. In this section we prove that the same result holds for M-fibrations, in fact in a stronger form: since minimum bases turn out to be simple graphs, minimal M-fibrations are unique up to isomorphism.

Definition 4.1 (M-equitable partition). Given an M-graph G, an M-equitable partition of G is a partition Π of $N _ { G }$ such that for every two nodes $x , y \in N _ { G }$ such that $x \sim _ { \Pi } y$ and every part $C \in \Pi$ we have

$$
W ( C , x ) = W ( C , y )
$$

where $W ( A , z ) = \oplus _ { a \in G ( A , z ) } \lambda ( a )$ is the total weight of the arcs from A to z. When we have an equitable partition, given two parts $C , D \in \Pi$ , we write $W ( C , D )$ for the common value $o f W ( C , x )$ for all $x \in D$

The following proposition shows that M-fibrations and M-equitable partitions are two sides of the same coin: given an M-fibration, the equivalence relation on nodes defined by the fibers of the fibration is an equitable partition, and given an equitable partition, the canonical projection to the quotient graph is an M-fibration.

Proposition 4.1. 1. Let $\varphi : G \to H$ be an epimorphic M-fibration: then the partition $\Pi _ { \sim }$ associated with the equivalence relation ∼ defined by $x \sim y$ if and only $i f \varphi ( x ) = \varphi ( y )$ is an M-equitable partition of G.

2. Let Π be an M-equitable partition of G, and let $G / \Pi$ be the simple M-graph obtained as the quotient of G by Π (its nodes are the parts of Π, and there is an arc $( C , D )$ , labelled by $W ( C , D )$ , if and only if $G ( C , D ) \neq \emptyset )$ ; let $\pi : G \to G / \Pi$ be the canonical projection to the quotient graph then π is an epimorphic M-fibration.

Proof. 1. Let $x \in N _ { G }$ and let $C = \varphi ^ { - 1 } ( \bar { z } )$ for some $\bar { z } \in N _ { H }$ . Then, by definition,

$$
W ( C , x ) = \bigoplus _ { a \in G ( C , x ) } \lambda ( a ) .
$$

The image under $\varphi$ of each a appearing in the last formula is an arc of H starting from z¯ and ending in $\varphi ( x ) ;$ ; conversely, for every $\bar { a } \in H ( \bar { z } , \varphi ( x ) )$ , every $a \in \varphi ^ { - 1 } ( { \bar { a } } ) \cap G ( - , x )$ belongs to $G ( C , x )$ (its source is mapped to $\bar { z } )$ . Hence the sets $\varphi ^ { - 1 } ( \bar { a } ) \cap G ( - , x )$ , for $\bar { a } \in H ( \bar { z } , \varphi ( x ) ) ,$ ), partition $G ( C , x )$ , and we can rewrite the last formula factorizing by the image of a under $\varphi \colon$

$$
W ( C , x ) = \bigoplus _ { a \in G ( C , x ) } \lambda ( a ) = \bigoplus _ { \bar { a } \in H ( \bar { z } , \varphi ( x ) ) } \bigoplus _ { a \in \varphi ^ { - 1 } ( \bar { a } ) \cap G ( - , x ) } \lambda ( a ) = \bigoplus _ { \bar { a } \in H ( \bar { z } , \varphi ( x ) ) } \lambda ( \bar { a } ) ,
$$

where the last equality follows from the definition of M-fibration. The last expression does not depend on x but only on $\varphi ( x )$ , so if $x \sim y$ we have $W ( C , x ) = W ( C , y )$ , as desired. 2. The only non-trivial part is proving that π is an M-fibration. Let (C, D) be an arc of $G / \Pi$ and let $x \in \pi ^ { - 1 } ( D )$ . Then

$$
\bigoplus _ { a \in \pi ^ { - 1 } ( C , D ) \cap G ( - , x ) } \lambda ( a ) = \bigoplus _ { a \in G ( C , x ) } \lambda ( a ) = W ( C , x ) = W ( C , D ) = \lambda ( C , D ) .
$$

which is the condition required for $\pi$ to be an M-fibration.

The next lemma is crucial in proving the existence of a minimum base for M-fibrations.

Lemma 4.1. Let G be an M-graph. Then there exists a coarsest M-equitable partition of $G ,$ i.e., an M-equitable partition Π such that for every M-equitable partition $\Pi ^ { \prime }$ of G we have ${ \sim } _ { \Pi ^ { \prime } }$ is finer than ${ \sim } _ { \Pi }$

Proof. We build a sequence of partitions $\Pi _ { 0 } , \Pi _ { 1 } , . . .$ . of $N _ { G }$ as follows: $\Pi _ { 0 }$ is the trivial partition with a single part $N _ { G }$ , and for $i > 0$ we define $\Pi _ { i }$ as the partition of $N _ { G }$ whose parts are the equivalence classes of the relation $\sim _ { i }$ defined by

$$
x \sim _ { i } y \Leftrightarrow x \sim _ { \ O ^ { - 1 } } y { \mathrm { ~ a n d ~ } } W ( C , x ) = W ( C , y ) \quad \forall C \in \Pi _ { i - 1 } .
$$

This sequence of finer and finer partitions must eventually stabilize (because the set of nodes is finite), i.e., there exists k such that $\Pi _ { k } = \Pi _ { k + 1 }$ (and hence $\Pi _ { i } = \Pi _ { k }$ for all $i \geq k$ , since $\Pi _ { i }$ depends only on $\Pi _ { i - 1 } )$ . Note that $\Pi ^ { * } = \Pi _ { k }$ is M-equitable: indeed, $\Pi _ { k + 1 } = \Pi _ { k }$ says precisely that $W ( C , x ) = W ( C , y )$ for all $C \in \Pi _ { k }$ whenever $x \sim _ { k } y .$ . We claim that $\Pi ^ { * }$ is the coarsest M-equitable partition of $G .$ Suppose that $\Xi$ is an M-equitable partition of $G .$ We prove by induction on i that $\Xi$ is finer than $\Pi _ { i }$ for all $i \geq 0$ . The base case $i = 0$ is trivial. Suppose that $\Xi$ is finer than $\Pi _ { i }$ . Let $x , y$ be in the same part of $\Xi ;$ then they are also in the same part of $\Pi _ { i }$ . Let C be a part of $\Pi _ { i : }$ since Ξ is finer than $\Pi _ { i } ,$ there are parts $C _ { 1 } , \ldots , C _ { m }$ of Ξ such that $C = C _ { 1 } \cup \dots \cup C _ { m } \mathrm { ~ ( a ~ }$ disjoint union). Since $\Xi$ is an M-equitable partition and $x , y$ lie in the same part of $\Xi$ , we have $W ( C _ { j } , x ) = W ( C _ { j } , y )$ for every $j ,$ whence

$$
W ( C , x ) = \bigoplus _ { j = 1 } ^ { m } W ( C _ { j } , x ) = \bigoplus _ { j = 1 } ^ { m } W ( C _ { j } , y ) = W ( C , y ) ,
$$

so x, y are also in the same part of $\Pi _ { i + 1 }$ , and we conclude that $\Xi$ is finer than $\Pi _ { i + 1 }$ . By induction, $\Xi$ is finer than $\Pi _ { i }$ for all $i \geq 0$ , and in particular it is finer than $\Pi ^ { * } = \Pi _ { k }$ □

We are now ready to prove that every M-graph has a minimum base.

Definition 4.2 (M-fibration prime and minimal M-fibration). An M-graph G is M-fibration prime $i f$ every epimorphic M-fibration $\varphi : G \to H$ is an isomorphism. An M-fibration $\varphi : G \to H$ is minimal $i f$ it is epimorphic and H is M-fibration prime.

Note that every M-fibration prime graph is simple, i.e., it has no parallel arcs, because if G has two parallel arcs $a , b$ from $x$ to $y ,$ then G is non-trivially and epimorphically fibred onto the graph H obtained by merging the two arcs with a new arc labelled by $\lambda ( a ) \oplus \lambda ( b )$ (which may be 0: this is why labels are allowed to be 0).

As a consequence of the above results, we have the following theorem, which is the main result of this section (and the equivalent, in the context of M-fibrations, of the well-known result that every graph has a minimum base [1]):

Theorem 4.1. For every M-graph $G _ { i }$ , we have:

1. there exists a minimal M-fibration $\varphi : G  B .$ ;

2. B is unique up to isomorphism;

3. $\varphi$ is unique up to composition with an isomorphism of B.

In the following, we call B $t h e ^ { 1 }$ minimum base of G and $\varphi$ the minimum M-fibration $o f G$ , and denote B as $\hat { G }$ and $\varphi$ as $\mu _ { G } : G \to { \hat { G } }$

Proof. 1. Let Π be the coarsest M-equitable partition of $G ,$ , which exists by Lemma 4.1. Let $B = G / \Pi$ be the quotient graph and let $\pi : G  B$ be the canonical projection onto the quotient graph. By Proposition 4.1 (2), π is an epimorphic M-fibration. We claim that B is M-fibration prime. Suppose that $\psi : B \to H$ is an epimorphic M-fibration. Then, by Proposition 3.1 (1), $\psi \circ \pi : G \to H$ is an epimorphic M-fibration, and by Proposition 4.1 (1), it induces an M-equitable partition of G. The parts of this partition are unions of parts of Π (the fibres of $\psi \circ \pi$ are unions of fibres of $\pi )$ , so it is coarser than Π; but Π is the coarsest M-equitable partition of $G ,$ , so the two partitions coincide, and $\psi$ is injective (hence, being epimorphic, bijective) on nodes. It is then also a bijection on arcs: it is surjective by hypothesis, and if $\psi ( a ) = \psi ( a ^ { \prime } )$ then a and $a ^ { \prime }$ have the same source and the same target (by injectivity on nodes), so $a = a ^ { \prime }$ because $B$ is simple. Thus ψ is a bijective M-fibration, hence an isomorphism by Proposition 3.1 (2). So we conclude that $B$ is M-fibration prime.

2 Suppose $\varphi : G  B _ { 1 }$ is a minimal M-fibration, and let $\Pi _ { 1 }$ be the M-equitable partition of G induced by $\varphi .$ Since Π is the coarsest M-equitable partition of $G ,$ , we have that $\Pi _ { 1 }$ is finer than Π. Define $\chi : B _ { 1 } \to B$ by $\chi ( x ) = \pi ( y )$ for any $y \in \varphi ^ { - 1 } ( x )$ : this definition is well-posed because if $y _ { 1 } , y _ { 2 } \in \varphi ^ { - 1 } ( x )$ , then $y _ { 1 } \sim _ { \Pi _ { 1 } } y _ { 2 }$ , and since $\Pi _ { 1 }$ is finer than Π, we have $y _ { 1 } \sim _ { \Pi } y _ { 2 }$ , so $\pi ( y _ { 1 } ) = \pi ( y _ { 2 } )$ Both $B _ { 1 }$ and B are simple $( B _ { 1 }$ because it is M-fibration prime, B by construction), so χ extends to arcs as follows: if c is the arc of $B _ { 1 }$ from w to $z ,$ pick $a \in \varphi ^ { - 1 } ( c )$ (which exists because $\varphi$ is epimorphic); then a goes from a node of $\varphi ^ { - 1 } ( w ) \subseteq \pi ^ { - 1 } ( \chi ( w ) )$ to a node of $\varphi ^ { - 1 } ( z ) \subseteq \pi ^ { - 1 } ( \chi ( z ) )$ so $G ( \pi ^ { - 1 } ( \chi ( w ) ) , \pi ^ { - 1 } ( \chi ( z ) ) ) \neq \emptyset$ and B has an arc from $\chi ( w )$ to $\chi ( z )$ : we let $\chi ( c )$ be this arc. By construction χ is a graph morphism with $\pi = \chi \circ \varphi .$ , and it is epimorphic: on nodes because π is, and on arcs because every arc of B is of the form $\pi ( a ) = \chi ( \varphi ( a ) )$ for some $a \in A _ { G }$

We now show that $\chi$ is an M-fibration. Let c¯ be the arc of B from U to V (recall that the nodes of B are the parts of Π, so $\pi ^ { - 1 } ( U ) = U$ and $\pi ^ { - 1 } ( V ) = V$ as sets of nodes of G), and let $z \in \chi ^ { - 1 } ( V )$ ; fix $y \in \varphi ^ { - 1 } ( z )$ , so that $y \in V$ . Since $B _ { 1 }$ is simple, the arcs of $B _ { 1 }$ ending in z that are mapped by χ to c¯ are exactly the arcs $( w , z )$ of $B _ { 1 }$ with $\chi ( w ) = U$ . For every such arc, the set of arcs of G ending in y and mapped by $\varphi$ to $( w , z )$ is $G ( \varphi ^ { - 1 } ( w ) , y )$ , so the M-fibration condition for $\varphi$ reads $\lambda ( w , z ) = W ( \varphi ^ { - 1 } ( w ) , y )$ . On the other hand, if $w \in \chi ^ { - 1 } ( U )$ but $B _ { 1 }$ has no arc from w to $z ,$ then $G ( \varphi ^ { - 1 } ( w ) , y ) = \emptyset$ (an arc from $\varphi ^ { - 1 } ( w )$ to $y$ would be mapped by $\varphi$ to an arc from w to z), so $W ( \varphi ^ { - 1 } ( w ) , y ) = 0$ . Therefore

$$
\begin{array} { l } { { \displaystyle \bigoplus _ { \substack { c \in \chi ^ { - 1 } ( \bar { c } ) \cap B _ { 1 } ( - , z ) } } \lambda ( c ) = \bigoplus _ { w \in \chi ^ { - 1 } ( U ) \atop { w \Big ( \bigcup _ { w \in \chi ^ { - 1 } ( U ) } \varphi ^ { - 1 } ( w ) , \ y \Big ) } } W ( \varphi ^ { - 1 } ( w ) , y ) = } } \end{array}
$$

where we used that the sets $\varphi ^ { - 1 } ( w )$ , for $w \in \chi ^ { - 1 } ( U )$ , partition $\pi ^ { - 1 } ( U ) = U$ , and that $y \in V$ . This is precisely the condition for $\chi$ to be an M-fibration.

Hence $\chi : B _ { 1 } \to B$ is an epimorphic M-fibration; since $B _ { 1 }$ is M-fibration prime, χ is an isomorphism, which proves 2.

3. With the notation above, $\pi = \chi \circ \varphi$ gives $\varphi = \chi ^ { - 1 }$ ◦ π: every minimal M-fibration of G is obtained from the canonical projection π by composition with an isomorphism of its base, and any two minimal M-fibrations difer by an isomorphism between their bases. □

In Figure 2 we show the minimum base of one of the graphs in Figure 1, when the monoid is (R, +, 0) (left) or $( \mathbb { Z } _ { 3 1 } , + , 0 )$ (right). Note that the two minimum bases are diferent, because the two monoids are diferent. In both cases, as expected, the minimum base is a simple graph, because every non-simple M-graph is epimorphically fibred onto a simple M-graph obtained by merging parallel arcs.

![](images/b95e81b81f25af743e1609fa876b731c76958d13c2fcadeadd093790b128526e.jpg)  
Figure 2. The minimum base $\hat { G }$ of the graph G in Figure 1, when the monoid is (R, +, 0) (left) or $\left( \mathbb { Z } _ { 3 1 } , + , 0 \right) \left( \mathrm { r i g h t } \right)$ We have named each node of $\hat { G }$ with the corresponding part of the coarsest equitable partition of G. The only diference between the two minimum bases is the label of the arc from {0, 1} to {2}, which is 15 ⊕ 23: in Z this gives $1 5 + 2 3 = 3 8$ , while in $\mathbb { Z } _ { 3 1 }$ it gives $1 5 + 2 3 = 3 8 \equiv 7$ (mod 31).

![](images/0453666b5a3738eecb1a0fcbb69890aca50f02f5245b377b98d34779c4b813db.jpg)  
Figure 3. The minimum base $\hat { H }$ of the graph H in Figure 1, when the monoid is $( \mathbb { R } , + , 0 )$ . Again, we have named each node of H<sup>ˆ</sup> with the corresponding part of the coarsest equitable partition of H. The reason why H<sup>ˆ</sup> is diferent from G<sup>ˆ</sup> (see Figure 2, left) is that 0 and 1 cannot be merged (because the two opposite arcs have diferent labels (5 and 6, instead of 5 in the two directions), and furthermore the arcs coming from 3 and 4 also have diferent labels $( 8 + 2 = 1 0$ and 9, respectively)). Nodes 3 and 4 are merged because the arcs coming from 2 have the same label 30.

Conversely, the graph H in Figure 1, which difers from G only in two labels, has a diferent minimum base, shown in Figure 3: the minimum base is sensitive to every label, and a small change of the labels can change it substantially (this is the starting point of Section 6).

It is worth noting that the proof of Theorem 4.1 is constructive: it provides a simple algorithm to compute the minimum base of an M-graph, by iteratively refining the partition of nodes until it stabilizes. A naive implementation runs in time $O ( | A _ { G } | \cdot | N _ { G } | )$ ; using the “process the smaller half” technique of Paige and Tarjan [22], and assuming that equality in M is decidable in constant time, one obtains $O ( | A _ { G } | \log | N _ { G } | )$ , as done for $M = ( \mathbb { R } _ { \geq 0 } , + , 0 )$ (Markov-chain lumping) in [23]; for standard colour refinement, i.e., $M = ( \mathbb { N } , + , 0 )$ , this bound is known to be tight [24].

As we observed, the minimum base of an M-graph is always a simple graph, even if the original graph has parallel arcs. This may seem to contradict the fact that the minimum base of a standard graph fibration may have parallel arcs, but this is not the case: if a standard graph fibration has parallel arcs, then the minimum base is obtained by merging them into a single arc with label equa to the number of merged arcs, which is exactly what happens in the case of M-fibrations with $M = ( \mathbb { N } , + , 0 )$ . Another interesting property is that the minimum M-fibration is unique (up to isomorphism of the base), while the minimal fibrations of a standard graph may not be unique (even up to isomorphism of the base), because diferent minimal fibrations may difer on parallel arcs.

## 5 Properties of the minimum base of an M-graph and pullbacks

One further property of the minimum base of an M-graph is the following:

![](images/b58c1c3e933593d4456fb41ea5db6bae83afe8116d48e35bbefa0b64e099bfaa.jpg)

![](images/4fe7900624fe825e23f0d6cb0c11d69e77d4d802b1ded1fe9c6652a803585336.jpg)  
Figure 4. Top: the M-graphs of Example 5.1. Labels: $\lambda ( \alpha _ { i } ) = \lambda ( \beta _ { j } ) = 1 , \lambda ( \gamma ) = 2 , \lambda ( d _ { 1 1 } ) = \lambda ( d _ { 2 2 } ) = t ,$ $\lambda ( d _ { 1 2 } ) = \lambda ( d _ { 2 1 } ) = 1 - t ;$ moreover $\psi _ { G } ( d _ { i j } ) = \alpha _ { i }$ and $\psi _ { H } ( d _ { i j } ) = \beta _ { j }$ . Bottom: the cone $Q _ { t }$ over the cospan $G \right. B \left. H$

Theorem 5.1. For every M-graph G, $i f \varphi : G \to H$ is an epimorphic M-fibration, then there is an epimorphic M-fibration ψ $\colon H  { \hat { G } }$ such that $\mu _ { G } = \psi \circ \varphi ;$ furthermore H<sup>ˆ</sup> is isomorphic to $\hat { G }$

Proof. The construction of χ in the proof of Theorem 4.1 (2) never used the fact that $B _ { 1 }$ is prime, only that it is simple: hence, given an epimorphic M-fibration $\varphi \colon G \to H$ , apply it to $\varphi ^ { \prime } = \mu _ { H } \circ \varphi \colon G \to \hat { H }$ (an epimorphic M-fibration onto a simple graph, by Proposition 3.1) to get an epimorphic M-fibration $\chi \colon { \hat { H } }  { \hat { G } }$ with $\mu _ { G } = \chi \circ \mu _ { H } \circ \varphi ;$ then $\psi = \chi \circ \mu _ { H }$ is the required fibration. Moreover χ is an epimorphic M-fibration out of the M-fibration prime graph $\hat { H }$ , hence an isomorphism ${ \hat { H } } \cong { \hat { G } }$ □

Now all the properties proved about M-fibrations and minimum bases are analogous to (and in fact proper generalizations of) the properties of standard graph fibrations and minimum bases. But the proofs in [1] follow a diferent approach, based on the notion of universal total graphs (which arise from a right adjoint, and depend on the unique lifting of paths) and on the fact that fibrations are preserved by pullbacks. Unfortunately, this route is not available in the case of M-fibrations: unique path lifting fails (an arc labelled 2 may lift to two arcs labelled 1), and with it the universal property of universal total graphs (Example 5.2); moreover, the category of M-graphs and M-fibrations does not have pullbacks, in general (Example 5.1).

Example 5.1. Let $M = ( \mathbb { R } _ { \geq 0 } , + , 0 )$ and consider the M-graphs of Figure 4: B has nodes $x , x ^ { \prime }$ and one arc $\gamma \colon x  x ^ { \prime }$ labelled 2; G has nodes $a , a ^ { \prime }$ and two arcs $\alpha _ { 1 } , \alpha _ { 2 } \colon a  a ^ { \prime }$ labelled 1; H is isomorphic to G, but its nodes are called $b , b ^ { \prime }$ and its arcs $\beta _ { 1 } , \beta _ { 2 } \colon b  b ^ { \prime }$ , again labelled 1. The morphisms $\varphi _ { G } \colon G \to B ( a \mapsto x , a ^ { \prime } \mapsto x ^ { \prime } , \alpha _ { i } \mapsto \gamma )$ and $\varphi _ { H } \colon H \to B$ (analogously) are M-fibrations. A cone over this cospan is an M-graph Q with two M-fibrations $\psi _ { G } \colon Q \to G$ and $\psi _ { H } \colon Q \to H$ such that $\varphi _ { G } \circ \psi _ { G } = \varphi _ { H } \circ \psi _ { H } ;$ a morphism of cones u : $Q \to Q ^ { \prime }$ is itself an M-fibration with $\psi _ { G } ^ { \prime } \circ u = \psi _ { G }$ and $\psi _ { H } ^ { \prime } \circ u = \psi _ { H } ;$ a pullback of the cospan is a terminal cone. Commutation forces every node of $Q$ to lie over (a, b) or over $( a ^ { \prime } , b ^ { \prime } ) \ ( \mathrm { i . e . }$ , to be mapped by ψ<sub>G</sub>, ψ<sub>H</sub> to a, b or to $a ^ { \prime } , b ^ { \prime }$ , respectively), and every arc of $Q$ to go from a node over $( a , b )$ to a node over $( a ^ { \prime } , b ^ { \prime } )$ , being mapped to some $\alpha _ { i }$ by $\psi _ { G }$ and to some $\beta _ { j }$ by $\psi _ { H }$ . For a node z of $Q$

over $( a ^ { \prime } , b ^ { \prime } )$ let

$$
m _ { i j } ( z ) = \bigoplus \{ \lambda ( d ) : d \in Q ( - , z ) , \psi _ { G } ( d ) = \alpha _ { i } , \ \psi _ { H } ( d ) = \beta _ { j } \} ,
$$

the total in-weight of z along arcs lying over $( \alpha _ { i } , \beta _ { j } )$ . The M-fibration condition for $\psi _ { G }$ at z and $\alpha _ { i }$ reads $m _ { i 1 } ( z ) + m _ { i 2 } ( z ) = \lambda ( \alpha _ { i } ) = 1$ , and the one for $\psi _ { H }$ at z and $\beta _ { j }$ reads $m _ { 1 j } ( z ) + m _ { 2 j } ( z ) = 1$ hence, the matrix $m ( z ) = ( m _ { i j } ( z ) )$ is

$$
m ( z ) = { \binom { t } { 1 - t } } \qquad t \qquad { \mathrm { f o r ~ s o m e ~ } } t = t ( z ) \in [ 0 , 1 ] .
$$

Conversely, for every $t \in [ 0 , 1 ]$ the M-graph $Q _ { t }$ of Figure 4 (nodes s over $( a , b )$ and $z$ over $( a ^ { \prime } , b ^ { \prime } )$ , arcs $d _ { i j } \colon s  z$ with $\psi _ { G } ( d _ { i j } ) = \alpha _ { i } , \psi _ { H } ( d _ { i j } ) = \beta _ { j }$ , and labels $\lambda ( d _ { 1 1 } ) = \lambda ( d _ { 2 2 } ) = t .$ $\lambda ( d _ { 1 2 } ) = \lambda ( d _ { 2 1 } ) = 1 - t )$ is a cone with $t ( z ) = t \colon$ both fibration conditions hold at $z ,$ and there is nothing to check at s.

Now let $u \colon Q \to Q ^ { \prime }$ be a morphism of cones and $z \textrm { a }$ node of Q over $( a ^ { \prime } , b ^ { \prime } )$ . Every arc d $\in Q ( - , z )$ with $\psi _ { G } ( d ) = \alpha _ { i }$ and $\psi _ { H } ( d ) = \beta _ { j }$ is mapped by u to an arc $e \in Q ^ { \prime } ( - , u ( z ) )$ with $\psi _ { G } ^ { \prime } ( e ) = \alpha _ { i }$ and $\psi _ { H } ^ { \prime } ( e ) = \beta _ { j }$ , and the M-fibration condition for u says that $\lambda ( e )$ is the sum of the labels of the arcs of $Q ( - , z )$ mapped to $e ;$ summing over all such e we obtain $m _ { i j } ( u ( z ) ) = m _ { i j } ( z )$ , that ${ \mathrm { i s } } ,$ morphisms of cones preserve the matrix m. Hence, if a pullback P existed, the images in $P$ of the nodes z of the cones $Q _ { t } , t \in [ 0 , 1 ]$ , would be pairwise distinct (they have distinct matrices): as a consequence, $P$ would have uncountably many nodes. No finite (not even countable) pullback exists.

Note that in the standard case (all labels equal to 1) unique lifting makes the fibres of each arc singletons, so the matrix $m ( z )$ is $1 \times 1$ and forced to be (1): this is why the pullback of the underlying graphs is a pullback of standard fibrations (see [1, Theorem 6.1], which proves even more), and why the argument above needs labels diferent from 1; the obstruction is the non-uniqueness of the matrices $m ( z )$ , not their existence.

Example 5.2. The graphs B and G of Figure 4 also show that universal total graphs [1, Sect. 3] lose their universal property; note that here it sufices to take $M = ( \mathbb { N } , + , 0 )$ . Recall that the universal total graph of a graph at a node x is the in-tree of all paths ending in x, fibred onto the graph in the obvious way; its universal property [1, Thm. 3.1] states that for every fibration $\varphi \colon H  B$ and every node $y$ of H over x there is exactly one fibration from the universal total graph of B at x to $H$ sending the root to $y ;$ in particular, the universal total graphs of B at $x$ and of H at $y$ are isomorphic. For M-graphs the same construction (copying the labels) still yields an M-fibration, but the universal property fails: B and $G$ are in-trees, and each is its own universal total graph at $x ^ { \prime }$ and at $a ^ { \prime } { \mathrm { . } }$ , respectively; yet there is no M-fibration ψ : $B  G$ with $\psi ( x ^ { \prime } ) = a ^ { \prime }$ (the arc $\gamma$ would have to be mapped to $\alpha _ { 1 }$ or to $\alpha _ { 2 } ,$ and the fibration condition at $a ^ { \prime }$ for the other arc would read $0 = 1 )$ , and $B \not \cong G$ . What survives is the fact that the two in-trees are bisimilar as weighted transition systems, and indeed the whole theory of Section 4 can be seen as a theory of weighted bisimilarity of unfoldings; we shall not pursue this point of view here.

## 6 ε-approximate M-fibrations

We want to move on to study approximate versions of M-fibrations, which are relevant in applications where we tolerate that the fibration condition may hold only approximately. We first need to have a measure of distance between elements of the monoid:

Definition 6.1. A metric monoid $( M , \oplus , 0 , d )$ is given by a commutative monoid $( M , \oplus , 0 )$ and a metric d : $\cdot M \times M \to \mathbb { R } _ { \geq 0 }$ on M such that for all $x , y , z \in M$ we have d(x $\oplus z , y \oplus z ) = d ( x , y )$

By the triangular inequality, we have

$$
d ( x \oplus x ^ { \prime } , y \oplus y ^ { \prime } ) \leq d ( x \oplus x ^ { \prime } , y \oplus x ^ { \prime } ) + d ( y \oplus x ^ { \prime } , y \oplus y ^ { \prime } ) = d ( x , y ) + d ( x ^ { \prime } , y ^ { \prime } )
$$

and this extends to finite sums: this property is called sum-subadditivity.

Now, we can measure how much a morphism of M-graphs fails to be an M-fibration:

Definition 6.2 (Approximate M-fibration). Given a metric monoid $( M , \oplus , 0 , d )$ , an M-graph G, and a morphism of M-graphs $\varphi : G \to H$ , for every arc a¯ of H and every node $x \in \varphi ^ { - 1 } ( t ( \bar { a } ) )$ we define the local fibration error of $\varphi$ at $( { \bar { a } } , x )$ as

$$
\delta _ { \varphi } ( \bar { a } , x ) = d \Big ( \bigoplus _ { a \in \varphi ^ { - 1 } ( \bar { a } ) \cap { \cal G } ( - , x ) } \lambda ( a ) , \lambda ( \bar { a } ) \Big ) .
$$

The global fibration error $o f \varphi$ is then defined as

$$
\Delta ( \varphi ) = \operatorname * { m a x } _ { x \in N _ { G } } \sum _ { \bar { a } \in t ^ { - 1 } ( \varphi ( x ) ) } \delta _ { \varphi } ( \bar { a } , x ) .
$$

A morphism $\varphi$ is an ε-approximate M-fibration if $\Delta ( \varphi ) \leq \varepsilon .$

It is immediate to observe that $\Delta ( \varphi ) = 0$ if and only if $\varphi$ is an M-fibration (all local errors are zero). Moreover, global fibration error is subadditive with respect to composition of morphisms:

Proposition 6.1. $I f \varphi : G \to H$ and $\psi : H \to K$ are morphisms of M-graphs, then $\Delta ( \psi \circ \varphi ) \leq$ $\Delta ( \varphi ) + \Delta ( \psi )$

We can also define how far is a pair of graphs from being linked by an epimorphic M-fibration: Definition 6.3 (Fibration distance). We define the fibration distance between G and H as

$$
\mathrm { f d } ( G , H ) = \operatorname* { i n f } \{ \Delta ( \varphi ) : \varphi : G \to H \ i s \ e p i m o r p h i c \} .
$$

Moreover, we define the function $\beta _ { G } : \mathbb { R } _ { \geq 0 } $ N as

$$
\beta _ { G } ( \varepsilon ) = \operatorname* { m i n } \{ | N _ { H } | : \mathrm { f d } ( G , H ) \leq \varepsilon \} .
$$

The fd function is a quasi-metric in the sense of Lawvere: $\operatorname { f d } ( G , G ) = 0$ and $\operatorname { f d } ( G , B ) = 0$ if and only if $G$ can be epimorphically fibred over B. Furthermore, $\operatorname { f d } ( G , B ) \leq \operatorname { f d } ( G , H ) + \operatorname { f d } ( H , B )$ . But fd is not symmetric, hence it is not a metric.

The function $\beta _ { G }$ , in mundane words, says how much you can compress G with an approximate fibration if you allow for an error of at most ε. It is non-increasing (if you allow for more error, you can compress more), and $\beta _ { G } ( 0 ) = | N _ { \hat { G } } |$ (the number of nodes of the minimum base of G). Moreover, for suficiently large $\varepsilon , \beta _ { G } ( \varepsilon ) = 1$ (the trivial graph with one node and one loop labelled with 0 is an ε-approximate fibration of any graph, for suficiently large ε: you just take ε to be the maximum sum of incoming labels over all the nodes of G).

The function $\beta _ { G }$ expresses the trade-of between compression and error, and it is a natural object of study in applications.

![](images/8149dd4c4614fc0a355e0e9c651be5ff32e5eeeedf8786dda8f8457eebc1ae3e.jpg)  
Figure 5. An example showing that there is no coarsest ε-approximate M-equitable partition.

## 7 ε-approximate M-equitable partitions

Like in the exact case, we can draw a parallel between ε-approximate M-fibrations and ε-approximate M-equitable partitions. To define them, let us first observe that in a metric monoid $( M , \oplus , 0 , d )$ we can define a distance between vectors of elements of M of the same length: for $\mathbf { x } = ( x _ { 1 } , \ldots , x _ { k } ) , \mathbf { y } =$ $( y _ { 1 } , \dotsc , y _ { k } ) \in M ^ { k }$ , we define $\begin{array} { r } { d ( \mathbf { x } , \mathbf { y } ) = \sum _ { i = 1 } ^ { k } d ( x _ { i } , y _ { i } ) } \end{array}$ . This is a metric on $M ^ { k }$ and it is sumsubadditive with respect to the componentwise sum of vectors. We use the same notation d for this metric on vectors, and we write 0 for the vector of length k whose components are all 0.

Definition 7.1 (Approximate M-equitable partition). Given a metric monoid $( M , \oplus , 0 , d )$ , an M-graph G, and a partition Π of $N _ { G }$ with p parts, define the vector of local in-weights of a node x as

$$
\begin{array} { r } { \mathbf { w } _ { x } = \big ( W ( C , x ) \big ) _ { C \in \Pi } \in M ^ { p } . } \end{array}
$$

A vector $\mathbf { c } \in M ^ { p }$ is a center of radius r for the part D if $d ( \mathbf { w } _ { x } , \mathbf { c } ) \leq r$ for all $x \in D$ . The inequity of Π, υ(Π), is the minimum value r such that there exists a center of radius r for every part of Π. We say that Π is an ε-approximate M-equitable partition $i f v ( \Pi ) \leq \varepsilon$

Note that Π is a 0-approximate M-equitable partition if and only if there are centers of radius 0 for all parts. The existence of a center of radius 0 for a part D is equivalent to the existence of a vector c such that ${ \bf w } _ { x } = { \bf c }$ for all $x \in D$ , which precisely means that $\mathbf { w } _ { x } = \mathbf { w } _ { y }$ for all $x , y \in D$ , i.e., that Π is an exact M-equitable partition. Hence, the notion of ε-approximate M-equitable partition is indeed a generalization of the notion of M-equitable partition.

There is an obvious parallel between ε-approximate M-fibrations and ε-approximate M-equitable partitions (the analogue of Proposition 4.1); we state it without proof:

Proposition 7.1. 1. Let $\varphi : G \to H$ be an epimorphic ε-approximate M-fibration: then the partition Π<sub>∼</sub> associated with the equivalence relation ∼ defined by $x \sim y$ if and only if $\varphi ( x ) = \varphi ( y )$ is an ε-approximate M-equitable partition of G.

2. Let Π be an ε-approximate M-equitable partition of G, and let $G / \Pi$ be the simple M-graph obtained as the quotient of G by Π (its nodes are the parts of Π, and there is an arc $( C , D )$ 2 labelled by $W ( C , D )$ , if and only if $G ( C , D ) \neq \emptyset )$ ; let $\pi : G \to G / \Pi$ be the canonical projection to the quotient graph then π is an epimorphic ε-approximate M-fibration.

It is useful to observe that computing $\beta _ { G } ( \varepsilon )$ is equivalent to computing an ε-approximate M-equitable partition of G with the minimum number of classes. Observe that for $\varepsilon > 0$ , there is no coarsest ε-approximate M-equitable partition, as the following counterexample shows.

Example 7.1. Consider the graph G in Figure 5, with the monoid $( \mathbb { R } , + , 0 )$ and the Euclidean distance. Clearly $\beta _ { G } ( 0 ) = 4$ because G itself is fibration prime (all input weights are distinct). Let us find the inequity of all non-trivial partitions leaving u alone: $v ( \{ u | x y | z \} ) = 0 . 1 , v ( \{ u | x | y z \} ) =$ $0 . 1 , \upsilon ( \{ u | x z | y \} ) = 0 . 2 , \upsilon ( \{ u | x y z \} ) = 0 . 2 ;$ ; mixing u with other nodes produces an inequity of 4.5. Hence, for $0 . 1 \leq \varepsilon < 0 . 2$ , there are two ε-approximate M-equitable partitions of G producing non-isomorphic quotient M-graphs (the labels on the arcs will be diferent, and node y will be mapped to diferent nodes of the quotient graphs). The actual values of $\beta _ { G } ( \varepsilon )$ are:

$$
\beta _ { G } ( \varepsilon ) = \left\{ \begin{array} { l l } { 4 } & { \mathrm { i f ~ } \varepsilon \in [ 0 , 0 . 1 ) } \\ { 3 } & { \mathrm { i f ~ } \varepsilon \in [ 0 . 1 , 0 . 2 ) } \\ { 2 } & { \mathrm { i f ~ } \varepsilon \in [ 0 . 2 , 4 . 5 ) } \\ { 1 } & { \mathrm { i f ~ } \varepsilon \geq 4 . 5 . } \end{array} \right.
$$

Finding an ε-approximate M-equitable partition of G with the minimum number of classes is a dificult problem: it is an optimization problem, which can be formulated as a MILP, but we conjecture it to be NP-hard. We leave the study of this conjecture to future work.

We can overcome this dificulty by considering a greedy algorithm that iteratively refines the partition of nodes until it stabilizes, as in the exact case. The diference is that now we have to check whether two nodes are ε-approximately equivalent. This is what Algorithm 1 does: it computes an ε-approximate M-equitable partition of G, and returns the centers of the parts. Centers can be computed in diferent ways, depending on the monoid we are working with. One approach that works regardless of the monoid is to take any vector as center, or to take the Chebyshev center (the vector that minimizes the maximum distance from the vectors of the part); the latter solution makes the computation quadratic.

The algorithm is greedy, and it may not return a partition with the minimum number of classes, but it is guaranteed to return a partition with radius at most ε for each class. The algorithm is a generalization of the one in [23], which is the case $M = ( \mathbb { R } _ { \geq 0 } , + , 0 )$ and ctr is the mean.

The nodes and arcs of the ε-approximate minimum base are obtained by the ε-approximate equitable partition coming from Algorithm 1 (parts are the nodes, an arc is present from C to D if and only if $W ( C , D ) \ne 0 )$ , like the exact case. For the labels, though, we use the centers of the parts: the label of the arc $( C , D )$ is the C-component of the center of D.

The ε-approximate minimum base is not unique, because the centers of the parts are not unique. Once the partition has been computed, though, the number of nodes of the resulting base does not depend on the choice of the centers; the partition itself, however, may depend on the center rule adopted (see Example 7.2).

Example 7.2. Let us try to use Algorithm 1 to compute an ε-approximate M-equitable partition of the M-graph H of Figure 1, with $M = ( \mathbb { R } , + , 0 )$ and $\varepsilon = 1$ . We use the coordinatewise mean as center rule.

We start with the trivial partition $\Pi = \{ \{ 0 , 1 , 2 , 3 , 4 \} \}$ and we compute the local in-weights of the nodes: ${ \bf w } _ { 0 } = ( 1 6 ) , { \bf w } _ { 1 } = ( 1 4 ) , { \bf w } _ { 2 } = ( 3 8 ) , { \bf w } _ { 3 } = ( 3 0 ) , { \bf w } _ { 4 } = ( 3 0 )$ . The first seed is 0, and the candidate set is $\{ 0 , 1 \}$ ; the center is computed as $\mathbf { c } = ( 1 5 )$ (the average between (14) and (16)). No node is added to the candidate set. The other class that is created in the split is {3, 4} (they literally share the same vector). So we have a new partition $\Pi ^ { \prime } = \{ \{ 0 , 1 \} , \{ 2 \} , \{ 3 , 4 \} \}$ , which is diferent from the previous one, and we continue. The new vectors are $\mathbf { w } _ { 0 } = ( 6 , 0 , 1 0 )$ ${ \bf w } _ { 1 } = ( 5 , 0 , 9 ) , ~ { \bf w } _ { 2 } = ( 3 8 , 0 , 0 ) , ~ { \bf w } _ { 3 } = ( 0 , 3 0 , 0 ) , ~ { \bf w } _ { 4 } = ( 0 , 3 0 , 0 )$ . The first seed is 0, and the candidate set is {0, 1}; the center computed this time is ${ \bf { c } } = ( 5 . 5 , 0 , 9 . 5 )$ (the average between (6, 0, 10) and (5, 0, 9)). Proceeding this way, we will in fact not split any further, so the algorithm terminates with the ε-approximate M-equitable partition $\Pi = \{ \{ 0 , 1 \} , \{ 2 \} , \{ 3 , 4 \} \}$ , with centers $\mathbf { c } _ { 0 1 } = ( 5 . 5 , 0 , 9 . 5 ) , \mathbf { c } _ { 2 } = ( 3 8 , 0 , 0 )$ , and ${ \bf c } _ { 3 4 } = ( 0 , 3 0 , 0 )$ . The resulting base is shown in Figure 6: the nodes are the parts of the partition, and the arc from part C to part $D$ (present if H has an arc from C to D) is labelled by the C-th component of the center of $D .$ An exhaustive search shows that the algorithm (when using coordinatewise mean) produces a base with 4 nodes when $\varepsilon \in [ 0 , 1 )$ , 3 nodes when $\varepsilon \in [ 1 , 1 3 . 5 )$ , and 1 node when $\varepsilon \ge 1 3 . 5$ . The actual function $\beta _ { H } ( \varepsilon )$ is 4 in [0, 1), 3 in [1, 12) and 1 when $\varepsilon \ge 1 2$ . This comparison shows that the algorithm is not optimal in the interval [12, 13.5), but it is optimal in the other two intervals; using the exact $\ell ^ { 1 }$ Chebyshev center as center rule, instead, the algorithm attains $\beta _ { H } ( \varepsilon )$ for every $\varepsilon .$ Note that $\beta _ { H }$ never takes the value 2: the best partition with two parts, {0}, {1, 2, 3, 4}, has inequity 13, larger than the inequity 12 of the trivial partition — the smallest feasible ε is not monotone in the number of parts.

Algorithm 1 Certified ε-equitable partition (tolerant refinement, epsilon partition)   
Require: M-graph G over a metric monoid $( M , \oplus , 0 , d ) ; \varepsilon \ge 0 ;$ a center rule ctr mapping a finite   
family of vectors of $M ^ { p }$ to a vector of $M ^ { p } \ { \mathrm { ( e . g . } }$ , the coordinatewise mean, the $\ell ^ { 1 }$ Chebyshev   
center, or anyone of the vectors)   
1: Π $ \{ N _ { G } \}$ ▷ trivial partition   
2: repeat   
3: for every node x: $\mathbf { w } _ { x } \gets \left( W ( C , x ) \right) _ { C \in \Pi }$   
4: $\Pi ^ { \prime }  \emptyset$   
5: for each class $D \in \Pi$ do   
6: $R  D$   
7: while $R \neq \emptyset$ do   
8: $s \gets \mathrm { a }$ seed in R (the first time: any node; afterwards: the one farthest from the   
previous seed)   
9: $A  \{ x \in R : d ( \mathbf { w } _ { x } , \mathbf { w } _ { s } ) \leq 2 \varepsilon \}$ ▷ candidates   
10: $\mathbf { c }  \operatorname { c t r } ( \{ \mathbf { w } _ { x } : x \in A \} )$   
11: while m $\operatorname { a x } _ { x \in A } d ( \mathbf { w } _ { x } , \mathbf { c } ) > \varepsilon$ do   
12: remove from A the node x ̸= s farthest from c   
13: $\mathbf { c }  \operatorname { c t r } ( \{ \mathbf { w } _ { x } : x \in A \} )$   
14: end while   
15: $A  A \cup \{ x \in R : d ( \mathbf { w } _ { x } , \mathbf { c } ) \leq \varepsilon \}$ ▷ absorb   
16: add the group $A ,$ with center $\mathbf { c } ,$ to Π<sup>′</sup>   
17: $R \gets R \backslash A$   
18: end while   
19: end for   
20: if $\left| \Pi ^ { \prime } \right| = \left| \Pi \right|$ then break   
21: end if   
22: Π $ \Pi ^ { \prime }$   
23: until false   
24: return Π with its centers ▷ every class has radius $\leq \varepsilon \ { \mathrm { w . r . t . } }$ its own aggregated vectors

![](images/818d037210636ca4f9d2e2111a08ec0aaa590b8ee471d996ca1cf1f9793328b0.jpg)  
Figure 6. The ε-approximate base of the graph H in Figure 1 produced by Algorithm 1, when the monoid is $( \mathbb { R } , + , 0 )$ $\varepsilon = 1$ and we are using the coordinatewise mean as center rule (here it is in fact a smallest ε-approximate base: no ε-equitable partition with two parts exists). Observe that if we were using the monoid $( \mathbb { Z } , + , 0 )$ instead, we could not apply the coordinatewise mean as center rule (division is not defined in $\mathbb { Z } ) ;$ using the $\ell ^ { 1 }$ Chebyshev center with integer coordinates, instead, we would get $\mathbf { c } _ { 0 1 } = ( 5 , 0 , 1 0 )$ or (6, 0, 9) (both at distance 1 from $\mathbf { w } _ { 0 }$ and $\mathbf { w } _ { 1 } )$ , whereas $\mathbf { c } _ { 2 } = ( 3 8 , 0 , 0 )$ and ${ \bf c } _ { 3 4 } = ( 0 , 3 0 , 0 )$ would be unchanged.

## 8 Application: compression of neural networks

Fibrations embed a notion of symmetry in a graph: two nodes that can be fibred together, receive the same input from the same (or equivalent) sources. This notion of symmetry can be fruitfully applied to compress neural networks: two neurons that are equivalent in this sense can be merged into one, and the weights of the incoming arcs can be summed. This idea was introduced in [2]: here we show the mathematical foundations of this approach. The fact that we are using general monoids allows us to treat diferent types of neural networks, including convolutional neural networks.

Let us briefly describe how a neural network can be represented as an M-graph, using the well-known MNIST classification dataset as reference [25].

In a simple multilayer perceptron (MLP), the translation is quite simple: the nodes of the graph are the neurons, and the arcs are the synaptic connections between them. The weights on the arcs are the synaptic weights, so the natural monoid is $( \mathbb { R } , + , 0 )$ . In fact, it is convenient to add explicitly a bias node and to introduce a normalization factor that is diferent at each layer, so that the distances between the in-weights of the neurons are comparable across layers. Details are given in the caption of Figure 7.

In a convolutional neural network (CNN), it is convenient to represent the network as a graph where the nodes are the channels of the network, and the arcs are the convolutional kernels that connect them. The weights on the arcs are kernels of appropriate sizes. A detailed explanation is given in the caption of Figure 8.

## 9 Experimental results

For our experiments of neural network compression, we used three trained networks: two for the MNIST dataset (the LeNet-300-100 MLP and LeNet-5 CNN architectures) and one for the CIFAR-10 dataset [26]. The compression was done post-training, so it is not comparable to the method described in [2]. This time, we simply converted the trained networks into M-graphs, and then we applied our ε-approximate M-fibration algorithm (Algorithm 1) to compress them, and then finally re-converted the compressed graphs back into neural networks, and tried them on the test set to measure their performance. The results are summarized in Table 1, and represented in Figure 9.

For these networks, we observed a phase transition in the accuracy of the compressed networks:

(a) The MLP network: four fully-connected layers, every unit carries one number

![](images/26e349900f95d0d8d3ee91b8bc796b89797f766f4b25c16ecc3d89f4c31e717b.jpg)

![](images/61a64e6535ede17c2a302053e9db38a37ad39dad5a303b19a5dda80e079ea2f6.jpg)  
Figure 7. LeNet-300-100 [25], an MLP neural network for the MNIST classification problem. Above, the usua representation: here, layers are fully connected. When represented as an M-graph, weights live on the arcs (we added a bias node, to spread biases to the layers), and input and output layers are frozen (their units cannot be merged during the process). Instead of working on (R, +, 0), it is more convenient to use $( \mathbb { R } \oplus \mathbb { R } \oplus \mathbb { R } , + , 0 , d _ { 1 } \oplus d _ { 2 } \oplus d _ { 3 } )$ so that distances are measured diferently on each layer: explicitly, we can define $d _ { i } ( u , v ) = | u - v | / s _ { i }$ where $s _ { i }$ is the median distance between the total in-weights of the nodes of layer i: this way, we can normalize distances across layers when the actual weights vary significantly. Merging a class is the quotient construction: in-weights ← center, out-weights ← fibre sums.

![](images/76b755824aed71224ce59ada2e987934793a5b1e05c8e58c337da4ae2c3526d8.jpg)

![](images/966c1067d4a9b48fafc2fd1d3e9ac9e7eb34b1667d409d8eb69413504fc459ce.jpg)  
Figure 8. LeNet-5-style [25] CNN for the MNIST classification problem. Above, the usual representation. Here $\overset { \vartriangle } { \boldsymbol { M } } = ( \mathbb { R } ^ { 5 \times 5 } , + ) \oplus ( \mathbb { R } ^ { \mathrm { { i } 6 } } , + ) \stackrel { \vartriangle } { \boldsymbol { \oplus } } ( \mathbb { R } , + )$ , using $\ell ^ { 1 }$ -distance on each component (with per-layer scale, like in the case of the MLP). Pooling is not in the graph: it acts inside a channel (spatially), and the pixel-level graph fibres exactly onto this one.

![](images/b6b9685ccf3c485ab88ca3b6ecdc90ac7f0c2211500e12d02524efd07b43df8f.jpg)  
Figure 9. Fraction of surviving units (circles) and test accuracy (squares) of the compressed networks as a function of ε, for the three networks of Table 1; the dotted line is the accuracy of the uncompressed network, and the shaded band $0 . 3 5 \le \varepsilon \le 0 . 5$ contains the phase transition.

for small values of ε, the accuracy remains almost unchanged, and the size (in number of units and parameters) decreases, but when ε reaches a critical value (between 0.35 and 0.5 in all three cases), the accuracy drops dramatically.

The sharp transition is a property of the certified, worst-case criterion, not of the networks: heuristics that cluster units by tight (complete-linkage) groups instead of filling the defect budget of every node, and choosing as representative not a metric center but the mean or the least-squares solution, trade the guarantee for a smooth, gracefully degrading accuracy-versus-size curve. These ad hoc heuristics, although interesting in practice, fall beyond the scope of this paper, which is focused on the theoretical foundations of the method.

<table><tr><td>network</td><td>ε</td><td>units</td><td>parameters</td><td>accuracy</td></tr><tr><td rowspan="3">MLP</td><td>0</td><td>400 (100%)</td><td>267k (100%)</td><td>97.76%</td></tr><tr><td>0.4</td><td>305 5 (76%)</td><td>210k(79%)</td><td>97.53%</td></tr><tr><td>0.5</td><td>169 (42%)</td><td>121k (45%)</td><td>79.6%</td></tr><tr><td rowspan="3">CNN</td><td>0</td><td>176 (100%)</td><td>80.2k (100%)</td><td>99.09%</td></tr><tr><td>0.4</td><td>166 (94%)</td><td>74.7k (93%)</td><td>98.89%</td></tr><tr><td>0.5</td><td>18 (10%)</td><td>2.1k (2.6%)</td><td>14.7%</td></tr><tr><td rowspan="4">VGG16-BN</td><td>0</td><td>5248 (100%)</td><td>15.25M (100%)</td><td>94.16%</td></tr><tr><td>0.30</td><td>3610 (69%)</td><td>8.38M (55%)</td><td>91.53%</td></tr><tr><td>0.35</td><td>3233 (62%)</td><td>6.75M (44%)</td><td>90.18%</td></tr><tr><td>0.40</td><td>2493 (48%)</td><td>4.32M (28%)</td><td>10.0%</td></tr></table>

Table 1. Certified ε-approximate M-fibrations.

## 10 Reproducibility

For reproducibility, we provide a self-contained Python library that implements the algorithms described in this paper; furthermore, the library contains a script that is able to take any ONNX neural network and compress it using the ε-approximate M-fibration algorithm, testing it for accuracy and reporting the results. The library is available at https://github.com/boldip/mfib.

## 11 Conclusions

We introduced a general framework to study fibrations of weighted graphs, and we proved that the minimum base of a weighted graph is unique up to isomorphism. We also introduced the notion of ε-approximate fibration, and we showed that it can be used to compress neural networks.

## Acknowledgement

We thank Hern´an Makse, Ian Stewart and Sebastiano Vigna for many fruitful discussions on the topic of this paper.

## References

[1] P. Boldi and S. Vigna, “Fibrations of graphs,” Discrete Math., vol. 243, pp. 21–66, 2002.

[2] O. M. Velarde, L. C. Parra, P. Boldi, and H. A. Makse, “The role of fibration symmetries in geometric deep learning,” Proc. Natl. Acad. Sci. USA, vol. 123, no. 4, p. e2416552123, 2026.

[3] A. Grothendieck, “Technique de descente et th´eor\`emes d’existence en g´eom´etrie alg´ebrique, I. G´en´eralit´es. Descente par morphismes fid\`element plats,” in S´eminaire Bourbaki, Vol. 5, Exp. No. 190. Soci´et´e Math´ematique de France, 1959–1960, pp. 299–327.

[4] A. J. Schwenk, “Computing the characteristic polynomial of a graph,” in Graphs and Combinatorics, ser. Lecture Notes in Mathematics, R. A. Bari and F. Harary, Eds. Springer, 1974, vol. 406, pp. 153–172.

[5] C. Godsil and G. Royle, Algebraic Graph Theory, ser. Graduate Texts in Mathematics. Springer, 2001, vol. 207.

[6] B. Weisfeiler and A. Leman, “The reduction of a graph to canonical form and the algebra which appears therein,” Nauchno-Technicheskaya Informatsia, vol. 2, no. 9, pp. 12–16, 1968, english translation by G. Ryabov.

[7] D. G. Corneil and C. C. Gotlieb, “An eficient algorithm for graph isomorphism,” J. ACM, vol. 17, no. 1, pp. 51–64, 1970.

[8] M. Golubitsky and I. Stewart, Dynamics and Bifurcation in Networks: Theory and Applications of Coupled Diferential Equations. Philadelphia: SIAM, 2023.

[9] G. A. D. Luna and G. Viglietta, “Optimal computation in anonymous dynamic networks,” J. ACM, 2026, to appear; preliminary version in Proc. STOC 2023.

[10] M. Golubitsky, I. Stewart, and A. T¨or¨ok, “Patterns of synchrony in coupled cell networks with multiple arrows,” SIAM J. Appl. Dyn. Syst., vol. 4, no. 1, pp. 78–100, 2005.

[11] R. Joly, “Generic balanced synchrony patterns in network dynamics,” 2025, arXiv:2510.08187.

[12] F. Morone, I. Leifer, and H. A. Makse, “Fibration symmetries uncover the building blocks of biological networks,” Proc. Natl. Acad. Sci. USA, vol. 117, no. 15, pp. 8306–8314, 2020.

[13] I. Stewart, S. D. S. Reis, and H. A. Makse, “Dynamics and bifurcations in genetic circuits with fibration symmetries,” J. R. Soc. Interface, vol. 21, no. 217, p. 20240386, 2024.

[14] H. A. Makse, P. Boldi, F. Sorrentino, and I. Stewart, Symmetries of living systems: Symmetry fibrations and synchronization in biological networks. Cambridge University Press, 2027, in press; arXiv preprint arXiv:2502.18713.

[15] P. Boldi, I. Leifer, and H. A. Makse, “Quasifibrations of graphs to find symmetries in biological networks,” J. Stat. Mech., vol. 2022, no. 8, p. 083401, 2022.

[16] P. Buchholz, “Bisimulation relations for weighted automata,” Theoret. Comput. Sci., vol. 393, no. 1–3, pp. 109–123, 2008.

[17] J. G. Kemeny and J. L. Snell, Finite Markov Chains. Springer, 1976.

[18] P. Buchholz, “Exact and ordinary lumpability in finite Markov chains,” J. Appl. Probab., vol. 31, no. 1, pp. 59–75, 1994.

[19] C. Morris, M. Ritzert, M. Fey, W. L. Hamilton, J. E. Lenssen, G. Rattan, and M. Grohe, “Weisfeiler and Leman go neural: higher-order graph neural networks,” in Proc. 33rd AAAI Conference on Artificial Intelligence, 2019, pp. 4602–4609.

[20] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How powerful are graph neural networks?” in Proc. 7th International Conference on Learning Representations (ICLR), 2019.

[21] M. Grohe, K. Kersting, M. Mladenov, and E. Selman, “Dimension reduction via colour refinement,” in Algorithms – ESA 2014, ser. Lecture Notes in Computer Science, vol. 8737. Springer, 2014, pp. 505–516.

[22] R. Paige and R. E. Tarjan, “Three partition refinement algorithms,” SIAM J. Comput., vol. 16, no. 6, pp. 973–989, 1987.

[23] A. Valmari and G. Franceschinis, “Simple O(m log n) time Markov chain lumping,” in Tools and Algorithms for the Construction and Analysis of Systems (TACAS 2010), ser. Lecture Notes in Computer Science, vol. 6015. Springer, 2010, pp. 38–52.

[24] C. Berkholz, P. Bonsma, and M. Grohe, “Tight lower and upper bounds for the complexity of canonical colour refinement,” Theory Comput. Syst., vol. 60, no. 4, pp. 581–614, 2017.

[25] Y. LeCun, L. Bottou, Y. Bengio, and P. Hafner, “Gradient-based learning applied to document recognition,” Proc. IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.

[26] A. Krizhevsky, “Learning multiple layers of features from tiny images,” University of Toronto, Tech. Rep., 2009, technical Report; https://www.cs.toronto.edu/∼kriz/cifar.html.