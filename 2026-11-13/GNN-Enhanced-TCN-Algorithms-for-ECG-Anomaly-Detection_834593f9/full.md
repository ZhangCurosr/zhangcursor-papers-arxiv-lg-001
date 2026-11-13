# GNN-Enhanced TCN Algorithms for ECG Anomaly Detection

Angelica Simonetti<sup>1†</sup> and Ferdinando Zanchetta<sup>1\*†</sup>

\*Corresponding author(s). E-mail(s): ferdinando.zanchett2@unibo.it; Contributing authors: angelica.simonetti@gmail.com; <sup>†</sup>All authors contributed equally to this work.

## Abstract

Temporal convolutional networks (TCNs) are among the most efective algorithms to deal with time series data. To a time series can be also given the structure of directed graph, opening the doors to the usage of graph neural networks (GNNs) in this context. In this paper we develop a new pipeline involving novel geometric deep learning models that merge the capabilities of ordinary TCNs with the ones of GNNs to solve an anomaly detection problem on electrocardiogram signals.

## 1 Introduction

Over the last few years, temporal convolutional networks (TCNs) gained prominence as one of the most efective classes of deep learning (DL) algorithms (see [25], [1]) to deal with time series data ([15]), in many cases surpassing the efectiveness of more traditional DL algorithms like recurrent neural networks (RNNs, comprising Long Short-Term Memory networks aka LSTMs). On the other hand, when dealing with graph structured data, a particular class of geometric deep learning (see [6]) algorithms called graph neural networks (GNNs, see [6], [38]) has proven to be very efective at solving problems related with this type of data or at improving the capabilities of existing algorithms. In the context of time series, GNNs have been used jointly with time series specific architectures (e.g. RNNs or TCNs), for example, to study autoregressive models arising from graph time series or to forecast trafic speed or density (see the survey [38]). In the latter case, the graph structure involved is given by the positions of the sensors that record time series, see [39], or it is inferred by learning the weights of an adjacency matrix ([39]).

In this paper we explore a diferent and novel approach, that is to use the graph structure to encode time dependencies. We presented some preliminary findings based on this approach on the occasion of the conference GSI2025 ([41]). Our idea is to introduce a directed graph structure where the information associated to a given time step (a node) can be linked with the information coming from time steps in the past (this relation assumes the form of a directed edge). This allows us to couple the advantages of TCNs with the ones of graph convolutions. We prove that this approach enhances state-of-the-art TCNs architectures and we test it on the problem of identifying anomalies in electrocardiogram (ECG) signals. We shall now summarize the contents of the paper.

In the first part of the paper (Sections 2 and 3) we introduce temporal convolutions and graph convolutions from a mathematical viewpoint and we study how to see a time series as a directed graph. We explain how to see temporal convolutions as graph convolutions on directed digraphs thus proving that TCNs can be seen as GNNs for a certain choice of digraph structure (Proposition 3.7). As a consequence, we motivate in these sections why graph convolutions might be useful for the study of TCNs: by definition, as we shall explain, they exhibit a so called permutation equivariance that seems to be particularly relevant when dealing with time dependencies as the ones arising into the context of time series.

Motivated by our theoretical observations, we describe in Section 4 some DL models that we designed and used to deal with time series data. These are autoencoders employing both temporal convolutions (that can be seen as graph convolutions) and genuine graph convolutions.

In Section 5 we describe the experiments we carried out on real world datasets to test the efectiveness of our algorithms. In this paper, we considered problems dealing with ECG signals, that are signals recording the electrical activity of the heart. They can be used to monitor the health status of a human being and to diagnose various heart conditions. In addition, they are easier to acquire than other medical data as they can be recorded by non-invasive devices. Recordings of ECG signals may present many types of anomalies, going from biological ones (e.g. arrhythmias) to artifacts linked with recording issues (e.g corruption due to device malfunctioning, environmental noise, incorrect usage of the device, etc.). We have applied our algorithms to solve an anomaly detection problem on an established benchmark dataset, going beyond the quality detection problem we solved in [41]: the ECG5000 dataset. This dataset only contains genuine signals that can be normal or abnormal: we establish that GNN-enhanced autoencoders achieve state-of-art performance and allow the design of lightweight models (reducing the number of learnable parameters) that can possibly be deployed more easily than the ones already developed in the literature we consulted. This approach is diferent from other classical methods relying on hand crafted features in what it considers only raw signals as an input. We also study the performance of variational autoencoders in this context.

Appendix A contains the technical details about our software implementations for reproducibility and a summary of the main quantitative results obtained in [41] to complement our discussion.

Acknowledgements: Both authors wish to thank Andrea Malagoli, Andrea Arigliano and Giovanni Faglioni for sharing with us their expertise about ECG signals. We also thank Rita Fioresi and Pietro Demurtas for stimulating discussions.

## 2 Temporal Convolutional Networks

Convolutional neural networks (CNNs, see [26], [27], [24], [25]) are a type of deep learning architecture that has been introduced to process grid-like data, such as 2D images. It relies on specific layers called convolutional layers which apply learnable sliding filters to local regions of the inputs. CNNs have been also adapted to data organized in sequences (1D CNNs, see [22]) and, even more specifically, to time series $( \mathrm { T C N s } .$ , see [1] and the references therein). Let us start defining time series and establishing the notation we will use throughout the paper.

Definition 2.1. We denote by $\mathrm { T S } ( \mathbb { Z } , m )$ the set of functions $\mathbb { Z } \to \mathbb { R } ^ { m }$ . Let $\mathrm { T S } ( r , m )$ be the set of multivariate time series with m channels and of length $r _ { \mathrm { : } }$ , that is the set of functions from the discrete ordered set $[ r ] : = [ 0 , . . . , r - 1 ] \subseteq \mathbb { Z }$ to $\mathbb { R } ^ { m }$ . Given $\mathbf { x } \in \mathrm { T S } ( r , m )$ we will denote as $\mathbf { x } ( i ) .$ <sub>j</sub> (or simply as $\mathbf { x } _ { i j }$ when no confusion is possible), for $i = 0 , . . . , r - 1$ and $j = 1 , . . . , m$ , the jth coordinate of the vector $\mathbf { x } ( i ) \in \mathbb { R } ^ { m }$

As we anticipated, TCNs are built on 1D convolutions.

Definition 2.2. Given a filter $K \in \mathbb { R } ^ { f }$ , we can define a one-channel, one-dimensional (1D) convolution on $\mathrm { T S } ( \mathbb { Z } , 1 )$ as an operator

$$
\mathrm { c o n v 1 D : T S } ( \mathbb { Z } , 1 ) \to \mathrm { T S } ( \mathbb { Z } , 1 )
$$

$$
\operatorname { c o n v 1 D } ( \mathbf { x } ) ( j ) = \sum _ { i = 0 } ^ { f - 1 } K _ { i } \mathbf { x } ( \alpha ( j , i ) ) + b
$$

where $\alpha ( j , - ) : [ f ]  \mathbb { Z }$ are strictly monotonic index functions. Similarly, one obtains operators

$$
\mathrm { c o n v 1 D } : \mathrm { T S } ( r , 1 ) \to \mathrm { T S } ( l , 1 )
$$

$$
\operatorname { c o n v 1 D } ( \mathbf { x } ) ( j ) = \sum _ { i = 0 } ^ { f - 1 } K _ { i } \mathbf { x } ( \alpha ( j , i ) ) + b
$$

where $\alpha ( j , - ) : [ f ]  \mathbb { Z }$ are strictly monotonic index functions, $\mathbf { x } ( i ) : = 0 { \mathrm { ~ i f ~ } } i \notin [ r ]$ and $b \in \mathbb { R }$ is called bias. The numbers $K _ { i }$ , b are called the parameters or weights of the convolution.

The most commonly used index functions are of the form $\alpha ( j , i ) = ( n + d \cdot i ) + j$ for some integers $n , d .$ As a consequence, from now on we shall assume that the one dimensional convolutions we consider have this form, unless otherwise stated. $\operatorname { I f } ,$ $\alpha ( j , i ) \leq j$ for all $i , j$ then the convolution is said to be causal as it will look only ’backward’. $\operatorname { I f } | d | > 1$ , the convolution is called dilated. Dilated causal convolutions are the building blocks of TCNs, that are CNNs where only causal convolutions appear, as we shall describe below. We define multi-channel (i.e. handling multivariate time series) 1D convolutions, in two steps

Definition 2.3. We define operators

$$
\operatorname { c o n v 1 D } : \operatorname { T S } ( r , n ) \to \operatorname { T S } ( l , 1 ) , \qquad \operatorname { c o n v 1 D } ( \mathbf { x } ) ( i ) = \sum _ { j = 1 } ^ { n } \operatorname { c o n v 1 D } _ { j } ( \mathbf { x } ( - ) _ { j } ) ( i )
$$

where conv1 ${ \mathrm D } _ { j } : \mathrm { T S } ( r , 1 ) \to \mathrm { T S } ( l , 1 )$ are n diferent convolutions, and operators

$$
\mathrm { c o n v 1 D } : = \oplus \mathrm { c o n v 1 D } : \mathrm { T S } ( r , n ) \to \mathrm { T S } ( l , m ) \cong \oplus \mathrm { T S } ( l , 1 )
$$

Remark 2.4. Similar considerations apply to define convolutional operators $\mathrm { T S } ( \mathbb { Z } , n ) \to \mathrm { T S } ( \mathbb { Z } , m )$

We can now define temporal convolutions as follows.

Definition 2.5. We define temporal convolutions to be dilated 1D convolutions $\mathrm { T S } ( r , n ) \to \mathrm { T S } ( r , m )$ (or more generally $\mathrm { T S } ( \mathbb { Z } , n ) \to \mathrm { T S } ( \mathbb { Z } , m ) )$ , that is, dilated 1D convolutions transforming a multivariate discrete time series of a given length into a multivariate discrete time series of the same length. Convolutional neural networks are neural networks that employ a sequence of convolutional and and pooling layers alternated with activation functions, that is multivariate ridge functions. TCNs are defined as the subclass of CNNs that take as an input series $\mathbf { x } \in \mathrm { T S } ( r , n )$ and that employ as convolutional layers only temporal convolutions.

The usefulness of TCNs in the context of time series arises from the fact that causal convolutions by design are able to exploit temporal dependencies, while not sufering from some of the algorithmic problems of RNNs such as LSTMs: for example they appear to be faster to train and more scalable (see [1] for a discussion). It is often stated that CNNs are translation invariant. In our context, we can give to this statement a more precise mathematical meaning.

Proposition 2.6. The translation group $T _ { \mathbb { Z } } \cong ( \mathbb { Z } , + )$ acts on $\mathrm { T S } ( \mathbb { Z } , n )$ as follows:

$$
T _ { a } ( \pmb { x } ) = \pmb { x } \circ T _ { a } \qquad \pmb { x } \in \mathrm { T S } ( \mathbb { Z } , n ) , \quad a \in \mathbb { Z }
$$

where $T _ { \mathbb { Z } } \ni T _ { a } : \mathbb { Z } \to \mathbb { Z } , T _ { a } ( y ) = y + a$ . Consider a convolution conv1 $\mathrm { D } : \mathrm { T S } ( \mathbb { Z } , 1 ) $ $\mathrm { T S } ( \mathbb { Z } , 1 )$ as in 2.2. Then if for all $j \in \mathbb Z$ and for all $T _ { a } \in T _ { \mathbb { Z } }$ we have $T _ { a } ( \alpha ( j , - ) ) =$ $\alpha ( T _ { a } ( j ) , - )$ we have that conv1D is $T _ { \mathbb { Z } }$ equivariant. The same applies to the case of convolutions $\mathrm { T S } ( \mathbb { Z } , n ) \to \mathrm { T S } ( \mathbb { Z } , m )$

![](images/4556d6ae4b62fcde1c98f0cb5f2fcaffcc3b387241f7984c8146de69080a32c7.jpg)  
Fig. 1: One possible structure of a time-digraph, as described in Example 1. Here only adjacent connections and all the connections for node v are shown, and $d = 4$

Proof We only notice that, under the hypothesis of the statement,

$$
\begin{array} { l } { \displaystyle { T _ { a } ( \mathrm { c o n v 1 D } ( \mathbf x ) ) ( j ) = \mathrm { c o n v 1 D } ( \mathbf x ) ( T _ { a } ( j ) ) = \sum _ { i = 0 } ^ { f - 1 } K _ { i } \mathbf x ( \alpha ( T _ { a } ( j ) , i ) ) + b = } } \\ { \displaystyle \qquad = \sum _ { i = 1 } ^ { f } K _ { i } \mathbf x ( T _ { a } ( \alpha ( j , i ) ) ) + b = } \\ { \displaystyle \qquad = \sum _ { i = 1 } ^ { f } K _ { i } T _ { a } ( \mathbf x ) ( \alpha ( j , i ) ) + b = \mathrm { c o n v 1 D } ( T _ { a } ( \mathbf x ) ) ( j ) } \end{array}
$$

## Remark 2.7. Explicitly note that:

• Convolutions having $\alpha ( j , i ) = ( n { + } d { \cdot } i ) { + } j$ for some $n , d \in \mathbb { Z }$ satisfy the hypothesis of the previous proposition.

• The action of $T _ { \mathbb { Z } }$ does not restrict trivially to an action on $\mathrm { T S } ( l , n )$ , for a given natural number l. As a consequence, in this case, the mathematical notion of translation invariance for 1D (temporal) convolution of the form $\mathrm { T S } ( l , n ) \to$ $\mathrm { T S } ( l , m )$ , for example, must be adapted.

## 3 Time series as Directed Graphs

## 3.1 Generalities.

Definition 3.1. A directed graph (digraph) G is the datum $G = ( V _ { G } , E _ { G } , h _ { G } , t _ { G } )$ of two sets $V _ { G }$ (the set of vertices), $E _ { G }$ (the set of edges) and two functions $h _ { G } , t _ { G } :$ $E _ { G } \to V _ { G }$ associating to each edge e its head $h _ { G } ( e )$ and its tail $t _ { G } ( e )$ respectively. A morphism $\varphi : G \to H$ between two directed digraphs $G$ and H is the datum of two functions $\varphi _ { V } : V _ { G } \to V _ { H } , \varphi _ { E } : E _ { G } \to E _ { H }$ such that $h _ { H } \circ \varphi _ { E } = \varphi _ { V } \circ h _ { G }$ and $t _ { H } \circ \varphi _ { E } = \varphi _ { V } \circ t _ { G }$

From now on, for simplicity we will assume that our digraphs have at most one edge connecting two diferent nodes (for each direction) and at most one self loop for each node. In this case, given a digraph $G = ( V _ { G } , E _ { G } , h _ { G } , t _ { G } )$ and an ordering of the vertices $( v _ { i } ) _ { i \in [ | V _ { G } | ] } .$ , we can define the adjacency matrix $o f G$ as the matrix defined by $A _ { i j } = 1$ if there is there exists an edge having as tail $v _ { i }$ and head $v _ { j }$ or $A _ { i j } = 0$ otherwise. If the adjacency matrix of a graph is symmetric, we say that our graph is undirected. We can assign weights to the edges of a graph by letting the entries of the adjacency matrix to be arbitrary real numbers. When speaking about weighted digraphs, we shall always assume that there is an edge having as tail $v _ { i }$ and head $v _ { j }$ if and only if $A _ { i j } \neq 0$ . We speak in this case of weighted digraphs.

Definition 3.2. A digraph with features of dimension n is the datum $( G , F _ { G } =$ $( h _ { v } ) _ { v \in V _ { G } } )$ of a (weighted or unweighted) digraph G and vectors $h _ { v } \in \mathbb R ^ { n }$ of node features for each vertex $v \in V _ { G }$ . For a given digraph G, we shall denote as $\operatorname { F e a t } ( G , n )$ the set of all digraphs with features of dimension n having G as underlying digraph.

Real-world graph-structured datasets usually come in the form of one or more digraphs with features. Graph Neural Networks (GNNs, [38]) are models that are employed on graph-structured data using as building blocks the so called graph convolutions ([6], [11]): given a graph, they update each node feature vector combining the information contained in the feature vectors of adjacent nodes. In general, a graph convolution is a function gconv $: \operatorname { F e a t } ( G , n ) \to \operatorname { F e a t } ( G , m )$ . Recall that the symmetric group $S _ { n }$ of permutations of n elements acts by conjugacy on the set of real valued $n \times n$ matrices $\mathrm { M } _ { n , n } ( \mathbb { R } )$ . Indeed, if for a given $\pi \in S _ { n }$ we denote with $P _ { \pi }$ the matrix permutation matrix having as entries $( P _ { \pi } ) _ { i j } = \delta _ { i , \pi ( j ) }$ we have that, if $A \in \mathrm { M } _ { n , n } ( \mathbb { R } )$ $P _ { \pi } A$ and $A P _ { \pi } ^ { T }$ correspond to the matrix A where the rows or the columns respectively of A have been swapped according to π. Remembering that $P _ { \pi } ^ { - 1 } = P _ { \pi } ^ { T }$ for all $\pi \in S _ { n }$ acts then by conjugation on $\mathrm { M } _ { n , n } ( \mathbb { R } )$ by the rule $\pi \cdot A = P _ { \pi } A P _ { \pi } ^ { T }$ . The following is a classical result.

Theorem 3.3. There is a bijective correspondence between weighted digraphs having n vertices and the orbits of $\mathrm { M } _ { n , n } ( \mathbb { R } )$ under the action of the symmetric group $S _ { n }$ by conjugation.

Graph convolutions are the main building block of GNNs: they are operators that transform a graph with features into another. The vast majority of graph convolution operators $\operatorname { F e a t } ( G , n ) \to \operatorname { F e a t } ( G , m )$ that are used in practice have the form we now describe. Starting from a graph with features with the choice of an ordering on the vertices $( G , F _ { G } = ( h _ { v _ { i } } \in \mathbb { R } ^ { n } ) _ { i \in | V _ { G } | } ) \in \mathrm { F e a t } ( G , n )$ the most common graph convolutions produce a graph with features $( G , F _ { G } ^ { \prime } = ( h _ { v _ { i } } ^ { \prime } \in \mathbb { R } ^ { m } ) _ { i \in | V _ { G } | } ) \in \mathrm { F e a t } ( G , m ) \mathrm { b y }$ updating the node features of the considered digraph using a message passing mechanism that can be written in the following general form

$$
h _ { v _ { i } } ^ { \prime } = \psi ( h _ { v _ { i } } , \oplus _ { v _ { j } \in N ^ { \beta } ( v _ { i } ) } \varphi ( h _ { v _ { i } } , h _ { v _ { j } } , A _ { i j } , A _ { j i } )\tag{1}
$$

where $\beta \in \{ h , t , u \} , N ^ { h } ( v _ { i } ) = \{ v _ { j } \in | V _ { G } | | A _ { j i } \neq 0 \} , N ^ { t } ( v _ { i } ) = \{ v _ { j } \in | V _ { G } | | A _ { i j } \neq 0 \}$ ， $N ^ { u } ( v _ { i } ) = \bar { N } ^ { h } ( v _ { i } ) \bar { \cup } N ^ { t } ( v _ { i } )$ , ⊕ denotes a permutation invariant function and $\psi , \varphi$ denote diferentiable functions (weaker regularity assumptions can be made). Here by permutation invariant we mean that if we change the ordering chosen on the vertices (applying a permutation $\pi \in S _ { | V _ { G } | } )$ we end up with the same output in the sense that

$$
\begin{array} { r l } & { \oplus _ { v _ { \pi ( j ) } \in N ^ { \beta } ( v _ { \pi ( i ) } ) } \varphi ( h _ { v _ { \pi ( i ) } } , h _ { v _ { \pi ( j ) } } , ( P _ { \pi } A P _ { \pi } ^ { T } ) _ { i j } , ( P _ { \pi } A P _ { \pi } ^ { T } ) _ { j i } ) = } \\ & { \oplus _ { v _ { j } \in N ^ { \beta } ( v _ { i } ) } \varphi ( h _ { v _ { i } } , h _ { v _ { j } } , A _ { i j } , A _ { j i } ) } \end{array}
$$

In other words, these graph convolutions efectively define operators ${ \mathrm { F e a t } } ( G , n ) \to$ Feat $( G , m )$ : the choice of an ordering on the vertices is needed in practice for the actual computations and definitions of the message passing involved but does not influence the output. They are therefore permutation equivariant. Much of the success of the GNNs is currently being attributed to the fact that graph convolutions are permutation equivariant in the sense we outlined above. Many popular message passing mechanisms are a particular case of the following one:

$$
h _ { v _ { i } } ^ { \prime } = \sum _ { v _ { j } \in N ^ { \beta } ( v _ { i } ) } c _ { f ^ { \beta } ( i , j ) } A _ { f ^ { \beta } ( i , j ) } W h _ { v _ { j } } + l _ { i } A _ { i i } B h _ { v _ { i } } + e\tag{2}
$$

here $f ^ { \beta } ( i , j ) = ( i , j )$ if $v _ { j } \ \in \ N ^ { t } ( v _ { i } )$ and (j, i) if $v _ { j } \in N ^ { h } ( v _ { i } ) , \ : c _ { i j } , \ : l _ { i }$ are optional normalization coeficients, W, B are matrices of weights and e is a bias vector. For digraphs, the choice of $\beta$ should be thought of as whether a node embedding should be updated by looking at the information of the nodes that are sending to it a signal, by looking at the information of the nodes that are receiving from it a signal or both. These are three diferent semantics, all justifiable depending on the problem at hand. Two important graph convolutions arise as particular cases of the previous formula: Kipf and Welling’s graph convolution for undirected graphs (see [21]) and GraphSage convolution ([16]) as it appears in the popular package PyTorch Geometric ([30], [10]. We remark that in [21] and [16] the above convolutions are described only for undirected graphs, but the formulas also make sense for digraphs. In fact the standard implementations used by practitioners are already capable to handle digraphs and are being used also in that context. For example, some papers introducing attention mechanisms (e.g. GAT, see [13]) explicitly introduce this possibility. However, in the digraph case the underlying mathematical theory is somewhat less mature (see [7] and the reference therein, for the roots of the mathematics behind the directed graph Laplacian the reader is referred to [2]).

Definition 3.4. Consider a digraph morphism $\varphi : H  G$ . We define:

• A simple upsampling from Feat(G, n) to Feat(H, n) to be the pullback $\varphi ^ { * }$ $\operatorname { F e a t } ( G , n ) \to \operatorname { F e a t } ( H , n )$ associating to $( G , F _ { G } = ( h _ { v } ) _ { v \in V _ { G } } )$ the digraph with features $( H , \varphi ^ { * } F _ { G } : = ( h _ { \varphi _ { V } ( v ) } ) _ { v \in V _ { H } } )$

• If φ<sub>V</sub> is surjective, a simple downsampling from $\operatorname { F e a t } ( H , n )$ to Feat $( G , n )$ to be the operator φ : Feat $( H , n ) \to \operatorname { F e a t } ( G , n )$ taking a graph with features $( H , F _ { H } =$ $\{ h _ { v } \} _ { v \in V _ { H } } ) \mathrm { ~ t o ~ } ( G , \varphi _ { * } F _ { H } = \{ \varphi _ { * } h _ { w } \} _ { w \in V _ { G } } )$ where $\begin{array} { r } { \varphi _ { * } h _ { w } = \sum _ { v \in \varphi ^ { - 1 } ( w ) } h _ { v } . } \end{array}$

Finally, we give the following operative definition of a graph neural network.

Definition 3.5. Consider two weighted digraphs G and H. A graph neural network is defined to be an operator $\operatorname { F e a t } ( G , n ) \to \operatorname { F e a t } ( H , m )$ that is obtained concatenating together graph convolutions, simple graph upsampling and downsampling and activation functions.

Remark 3.6. Explicitly note that in the previous definition of GNN we allow for the convolutions and the (up)downsampling involved in a given GNN to arise as convolutions not involving necessarily the starting graph G and the ending graph H. That is, many diferent intermediate graphs can be used inside a GNN. In addition, we also underline that GNNs using popular mechanisms such as skip-connections or residual blocks, fit in our definition of GNNs.

## 3.2 Graph Neural Networks for time series as directed graphs

Data having the structure of a sequence can be given a structure of (weighted directed) graph with features where nodes are the words or units in the sequence, the node features are the information associated to them and the edges are the relations between these entities. In the case of time series, each node can represent a time step, features associated to it the information about that time step and directed edges the temporal relations: for example, to preserve time coherence the graph structure can be modeled in such a way that each node can receive information only from nodes in the past. We introduce two basic examples in order to illustrate how a time series can be given a structure of digraph with features.

Example 1: A multivariate time series $\mathbf { x } \in \mathrm { T S } ( n , m ) \cong \mathbb { R } ^ { n \times m }$ can be seen as an unweighted digraph with features as follows. To start with, we consider a set of $n \times m$ nodes $v _ { i j }$ , with $i = 1 , . . . , n$ and $j = 1 , . . . , m$ . Then, we create edges $v _ { i j }  v _ { l k }$ only if $l \geq i$ and the edge is not a self loop (i.e. edges receive information only from the present and the past). We assign the scalar $\mathbf { x } ( i - 1 ) _ { j }$ as feature for each node $v _ { i j }$ . This construction results in an unweighted digraph with features $( G _ { \mathbf { x } } , F _ { \mathbf { x } } = ( \mathbf { x } ( i ) _ { j } \in \mathbb { R } ) )$ . One can modify the topology of the graph just constructed. For example, one could create edges $v _ { i j }  v _ { l k }$ only if the edge is not a self-loop, $l \geq i$ and $l - i = 0 .$ 1 or if the edge is not a self-loop, $l \geq i ,$ , and $l - i$ is both divisible by a given positive integer d and is smaller than $k \cdot d$ for a given positive integer k. This construction results in the directed graph structure pioneered in [40]): see Figure 1.

Example 2: A multivariate time series $\mathbf { x } \in \mathrm { T S } ( n , m ) \cong \mathbb { R } ^ { n \times m }$ can be seen as a one dimensional time series with m channels. In this case, the time series can be turned into a digraph with features $( G _ { \mathbf { x } } , F _ { \mathbf { x } } = ( \mathbf { x } ( i - 1 ) \in \mathbb { R } ^ { m } ) )$ by considering a directed graph of n nodes $v _ { i } , i = 1 , . . . , n$ and edges $v _ { i }  v _ { l } \mathrm { ~ a r e }$ added if the edge is not a selfloop and $l \geq i , l - i = 1$ or and $l - i$ is both divisible by a given positive integer d and is smaller than $k \cdot d$ for a given positive integer k. We assign the vector $\mathbf { x } ( i - 1 ) \in \mathbb { R } ^ { m }$ as a vector of features for each node $v _ { i }$ . This completes the construction of the desired directed digraph with features.

These examples are of course just a starting point and one can take advantage of further domain knowledge to model the topology of the graph in a more specific way. For instance, already using the basic examples provided, one could use autocorrelation functions, usually employed to determine the order of ARMA models (see [37]), to choose the right value for parameters like k or d. At this point, the reader might wonder whether certain 1D convolution can be realized as graph convolutions given a certain graph structure. This is the content of the following proposition.

Proposition 3.7. Consider a convolution conv1D $\begin{array} { r l r } { \mathrm { ~ : ~ } } & { { } \mathrm { T S } ( r , 1 ) } & {  } & { { } \mathrm { T S } ( l , 1 ) } \end{array}$ conv $\begin{array} { r } { \begin{array} { r } { \mathrm { l D } ( \pmb { x } ) ( j ) = \sum _ { i = 0 } ^ { f - 1 } K _ { i } \pmb { x } ( \alpha ( j , i ) ) + b } \end{array} } \end{array}$ (as in Definition 2.2). Then there exist weighted digraphs G and H having $r , l$ nodes respectively, bijections ψ<sub>G</sub> : Feat $( \mathrm { G } , 1 ) \cong \mathrm { T S } ( r , 1 )$ , $\psi _ { H } : { \mathrm { F e a t } } ( \mathrm { H } , 1 ) \cong \mathrm { T S } ( r , 1 )$ and an operator g : Feat $( G , 1 ) \to \mathrm { F e a t } ( H , 1 )$ that is a composition of graph convolutions of the form (2) and simple upsampling or downsampling such that $\psi _ { H } ^ { - 1 }$ ◦ conv1 $\mathrm { D } \circ \psi _ { G } = g$

Proof Assume that $l \leq r$ and define G to be the digraph having r vertices $v _ { 1 } , \ldots ,$ v<sub>r</sub> and whose weighted adjacency matrix is given by $A _ { \alpha ( j , i ) + 1 , j } = K _ { i }$ if $1 \leq \alpha ( j , i ) \leq r , 0 \leq i \leq f - 1$ and zero otherwise. Let $\iota : H \subseteq G$ be the weighed subgraph of G consisting of the vertices $v _ { 1 } , . . . , v _ { l }$ . Define ψ<sub>G</sub> to be the map associating to $( G , h v _ { i } \in \mathbb { R } )$ the time series $\mathbf { x } \in \mathrm { T S } ( r , 1 )$ having $\mathbf { x } ( i ) \ : = \ : h _ { v _ { i + 1 } }$ and define $\psi _ { H }$ analogously. Consider the graph convolution gconv : $\operatorname { F e a t } ( G , 1 ) \to \operatorname { F e a t } ( G , 1 )$ arising from the message passing mechanism given by formula (2) with $\beta = h , W = B = 1 , l _ { i } = 0 , e = b$ and $c _ { i j } = 1$ . Then $g = \iota ^ { * }$ ◦ gconv. The case $l > r$ is similar. □

The previous lemma can be extended mutatis mutandis also to the case of multivariate time series and contains as a particular case the one of dilated convolutions. As a consequence, the process of learning the weights of a dilated convolution can be thought as the process of learning the weights of the adjacency matrix of a graph.

Remark 3.8. The previous proposition has an important conceptual consequence: ordinary TCNs can be seen as certain GNNs. As the latter are permutation equivariant in the sense that they depend only on the graph structure and not on the ordering of the nodes, this allows us to define networks that exhibit a type of equivariance that can be leveraged: this is a desirable feature that in the case of TCNs, as we explained before (see 2.7), proves to be more elusive.

In addition, GNNs applied to time series can carry a diferent meaning than ordinary TCNs as the design of the underlying digraph employed can be modeled to exhibit other important dependencies related with the problem at hand while at the same time preserving temporal coherence. It is worth noticing that we are not suggesting that GNNs should replace TCNs or that hybrid architectures should outperform in all applications. The point of view that we are advocating is that, being the TCNs compatible with GNNs, researchers should not be constrained to sticking to one of the two frameworks only. In this sense, our work should be seen as a work laying down the theory linking TCNs and GNNs and starting to explore this link in a specific problem rather than as a work proving a thesis. For example, we do not include in our work experiments about the possible gains in terms of explainability of the hybrid digraph based models, that will be subject of future work.

Remark 3.9. Simple Laplacian-based graph convolutions on undirected graphs can be seen as 1-step Euler discretizations of a heat equation (see for example [11], [6]). In general, graph convolutions can be thought as modeling a difusive process of the information contained in the nodes along the edges: in our context, directed graph convolutions applied to time series digraphs difuse the information through time by updating at each step node features using the information coming from a temporal neighbourhood.

## 4 Our Models

To study the efectiveness of employing graph convolutions in deep learning pipelines in order to analyze time series we propose a model that takes advantage of the timedigraph structure described in the previous section: an autoencoder reconstructing the time series, that can be used for clustering or inference purposes as we shall explain later. We will first focus on the core building block of the algorithm and then proceed to discuss the model in more detail. For a more synthetic and simplified description of part of these algorithms, see also our work [41].

![](images/ccb9b5745f4b9347b7a92388f603b6e529272c9cb88c6b4c27fb8d3bdd5fe3bc.jpg)  
Fig. 2: The encoder and decoder (a), and the skip-connections block (b)

## 4.1 Main Building Block

The main building block for all the models presented in this paper is a collection of layers that combines TCNs with GNNs. Inspired by what has already been done in the literature, we propose this main building block in two versions: an encoder and a decoder (they are sketched in Figure 2(a)). They can be used on their own or put together to construct an autoencoder model.

In the encoder version, the input is first given to a block of either n TCN or n GNN (we tested primarily Sage convolution-based layers as they appeared more efective after some preliminary tries) with skip connections, following the approach proposed in [35]. In the architectures we have tested, the TCN/GNN layers are simply a temporal convolution or a graph convolution (followed by dropout and an activation function), but more involved designs are possible. The efectiveness of skip-connections in the model developed in op.cit is due to the fact that stacking together the outputs of dilated convolutions with diferent dilations allows to consider short and long term time dependencies at the same time. Skip connections have been used also in the context of GNNs: for example, Sage convolutions [16] introduce them already in the message passing mechanism. This motivates the introduction of skip connections in GNNs handling digraphs with features coming from time series: in this context they do not only allow to bundle together information coming from "temporal neighbourhoods" of diferent radii as in the architecture developed in [35] but they also help reducing oversmoothing that traditionally curses GNNs architectures (see [5] for a discussion and the references therein). The skip-connections block is described in Figure 2(b) and works as follows. The input goes through the first TCN/GNN layer, followed by a 1-dimensional convolution and an activation function. We call the dimension of the output of this convolution skip dimension. Then the output is both stored and passed to the next TCN/GNN layer as input and so on. At the end all stored tensors are stacked together and passed to a series of m graph convolutions, each one followed by an activation function. We tested the convolutions: GCN (cfr. [21]), Sage (cfr. [16]), GAT (cfr. [13]).

The graph convolutions are defined to encode in the embedding of a given node, at each pass, the information provided by all the nodes in its neighbourhood. Now, looking at how a time-digraph is defined, one sees that in our set up this translates to building the embedding of a given node, that is a data point of the time series, using the information given by the data points that are close in time (short-term memory behavior) or at a certain distance d away in time (long-term memory behavior), where d is set in the construction of the graph.

Finally the intermediate output produced by the graph convolutions is given to an optional 1d convolution with kernel size 1 to adjust the number of channels and then to either an average pooling layer (i.e. a simple downsampling as in Definition 3.4) or a max pooling layer (that we can see as a nonlinear simple downsampling) that shrinks the temporal dimension of the graph. In other words, if we think about the time-graph as a time window of length T, the pooling layers outputs a time window, and therefore a time-graph, of length $T / s$ , thus realizing the characteristic bottleneck of an autoencoder as described for instance in, [35], [19]. We will refer to s as the shrinking factor.

The decoder changes the order of the blocks we just described, in a symmetric way. It starts with an upsample (i.e. a simple outsample as in Definition 3.4 or a nonlinear version of it) of the input time series which is then passed to the graph convolutions followed by the skip-connections block. It terminates with a 1d final convolution with kernel size 1 that reduces the number of channels, or hidden dimensions, thus giving as output a time series of the same dimensions as the initial input. In the case where the skip-connections block is built with GNN layers, this final convolution can be replaced by a final TCN layer. Both the encoder and the decoder can be modified to be bidirectional. In this variant each skip-connections block is paired with an additional skip connection block having the same structure but receiving as an input the original time series reversed or if the input is a digraph with features, the digraph with features obtained from the starting digraph and having as weighted adjacency matrix the transposed of the original one. The outputs of these two parallel blocks is then summed together and fed to the following block as before. With this modification, our model mimics bidirectional architectures sometimes found in RNN-based models ([4]).

## 4.2 Autoencoders and Variational Autoencoders

![](images/15a9aa403097ed8f3cf724b990cbea779ad1fb7d5098ab69dea11d544d399847.jpg)  
Fig. 3: The structure of the autoencoder.

We propose an autoencoder model, together with its variational counterpart. The autoencoder employs two main building blocks as in figure $2 ( \mathrm { a } )$ , first used as an encoder and then as a decoder and it produces as output the reconstruction of the given time series represented by the time graph. This is the architecture we employed in [41]. This model might include the following modification: after the encoding, the tensors in the hidden layer are passed both to the encoder and to a multi-layer perceptron classifier whose output is a label $\hat { y }$ (see Figure 3). The output $\hat { y }$ is confronted with the true label of the classification task using a suitable loss function ${ \mathcal { L } } _ { \mathrm { h d c l } }$ and concurs to the total loss computation, together with the reconstruction loss $\mathcal { L } _ { \mathrm { a e } } \mathrm { i }$

$$
{ \mathcal { L } } _ { \mathrm { t o t } } = { \mathcal { L } } _ { \mathrm { a e } } + { \mathcal { L } } _ { \mathrm { h d c l } }\tag{3}
$$

The variational autoencoder is built using exactly the same encoder and decoder, as usual. As for the loss, it is composed of the following contributions (see [20, 42]).

$$
\mathcal { L } _ { \mathrm { t o t } } = \mathcal { L } _ { \mathrm { a e } } + \mathcal { L } _ { \mathrm { m m d } } + \mathcal { L } _ { \mathrm { h d c l } }\tag{4}
$$

where the term ${ \mathcal { L } } _ { \mathrm { h d c l } }$ is optional, as for the case of the standard autoencoder, while $\mathcal { L } _ { \mathrm { m m d } }$ is the maximum mean discrepancy [14, 42]. As explained in $\mathbf { \nabla } _ { O p . }$ cit. divergence measures how similar two distributions are by comparing all of their moments, using

a kernel embedding trick. If $p ( x )$ and $q ( x )$ are the two distributions and $k ( x , x ^ { \prime } )$ is a positive definite kernel, then the MMD divergence is defined as follows:

$$
\mathcal { L } _ { \mathrm { m m d } } = \mathbb { E } _ { p ( x ) , p ( x ^ { \prime } ) } [ k ( x , x ^ { \prime } ) ] + \mathbb { E } _ { q ( x ) , q ( x ^ { \prime } ) } [ k ( x , x ^ { \prime } ) ] - 2 \mathbb { E } _ { p ( x ) , q ( x ^ { \prime } ) } [ k ( x , x ^ { \prime } ) ]\tag{5}
$$

Roughly, the more the two distributions are similar, the lower $\mathcal { L } _ { \mathrm { m m d } }$ will be, as the similarity of samples from each distribution should be close to the similarity of samples takes from both distributions, in other words $\mathcal { L } _ { \mathrm { m m d } } = 0$ if and only ${ \mathrm { i f ~ } } p = q$ . For our experiments we use the Gaussian kernel $k ( x , x ^ { \prime } ) = e ^ { - \frac { | | x - x ^ { \prime } | | ^ { 2 } } { 2 \sigma ^ { 2 } } }$

From now on we will refer to the AE models (respectively to the VAE models) as: TCNGraphAE (TCNGraphVAE) if the skip-connections block uses TCN layers, TGraphAE (TCNGraphVAE) if it is built with graph convolutions and TGraph-MixedAE (TCNGraphVAE) if the encoder uses graph convolutions and the decoder uses TCN layers. We will add the sufix hdcl if the total loss function of either the autoencoder or the variational autoencoder includes the term $\mathcal { L } _ { \mathrm { h d c l } }$ . If the skip connections blocks consist only of dilated convolutions and we do not have a final graph convolution to filter the signal, we obtain a TCN autoencoder with a structure similar to the one described in [35]. We call this model TCNAE and regard it as the state of the art model in this context, using it as a benchmark.

## 5 Experiments and results

We test our algorithms to solve anomaly detection problems concerning ECG datasets. To this aim we used the benchmark dataset called ECG5000, that is also considered by many other authors that developed and tested their algorithms on it. We also detail in Section 5.2 the results we obtained on an industry validation set we considered in [41]. As the pipeline proposed in this paper has been tested on the ECG5000 dataset only, further experiments would be needed to confirm its eficacy on diferent datasets. While the ECG5000 can be considered as one of the main benchmark datasets it has limits (see [33]).

## 5.1 Dataset ECG5000

The ECG5000 dataset is a benchmark dataset comprising of 5000 ECG signals that has been used by many authors (see [33] for a comprehensive introduction to the literature). Signals are labeled as normal or as anomalous depending on whether or not they represent normal rhythms. We did not preprocess these signals. As for the split, there is no consensus among the authors about which is the most appropriate split. Accordingly, we performed a train/valid/test split that we now explain. As a train set, we kept the first 500 signals of the database: this is the same training set that the authors considering only 500 samples as their training set consider. In the training set 58% of samples were normal, for a total of 292 normal samples. From the remaining samples we selected randomly 256 signals to form our valid set. The signals of the dataset not used in the train set and in the valid set have been used as a test set. The percentage of normal signals in the validation set was 57% and 58% in the test set (147 and 2480 normal samples respectively). Accordingly our split resulted in essentially equally balanced sub-datasets. We adopted a strategy diferent from the one we employed in the case of the signal quality recognition. We decided to follow the approach adopted in [31], for example, that is the one to use an autoencoder to embed the signals into a latent space where a clustering or a classification is performed. To help modeling the latent space in such a way that distinguishing the normal from the abnormal signals becomes easier, we decided to employ autoencoders feeding the hidden representations of the signals to an MLP for classification, as explained in Section 4. Each model we considered has been used as follows:

1) The model is trained on the training set: here, to the time series representing the signals the same structure of directed one used for the industry dataset has been employed for the graph convolutions involved in the model. To each time series x, a directed graph structure was given as described in Example 1. In particular, each time step x is a node and we have a directed edge $\mathbf x _ { k } \to \mathbf x _ { i }$ for each $k = i - 1 , i - d , i - 2 d , . . . , i - n \cdot \cdot \cdot$ with $d = 8$ . The number $n \cdot d ,$ called lookback window, was set to be $\frac { 1 } { 5 }$ of the total length of our sequences. All the graph convolutions appearing in the models were implemented in their directed version: the resulting convolutions then updated the node features of a node using only the information of nodes in the past. We chose $d = 8$ as the majority of the GNN skip-connection blocks we used had 7 layers. Combined with our lookback window size, this depth enables each point in the block’s output to aggregate information across the entire time series while maintaining a broader, more diluted receptive field. This lookback window choice was specifically designed to allow graph convolutions to cover the full sequence length twice. Furthermore, dividing the sequence into 5 sub-segments aligns naturally with standard ECG morphology, as a single heartbeat is broadly characterized by three major phases: the P-wave, QRS complex, and T-wave (corresponding to atrial depolarization, ventricular depolarization, and ventricular re-polarization, respectively). Between the first and the second the so called PR segment can be found and between the second and the third phase usually the ST segment is identified amounting to roughly 5 sub-intervals of an ECG waveform. The reader should think of this as a heuristic: we did not pinpoint the exact start and end points of these subdivisions. However, we believe that this choice can serve as a first default choice of this parameter. This choice is also aligned with the one we did in our work [41].

2) An SVC classifier is trained on the hidden representations of the signals in the latent space. Also, a K-means algorithm is applied to the representations in the latent space of the model to find two centroids that can be used for binary classification.

3) Classification is performed on the valid set using the SVC and the centroids obtained in the previous point. In addition, the MLP classifier trained in the training is also used on the valid set to evaluate its performance and a Kmeans algorithm is trained from scratch on the valid set to cluster the hidden representations into two clusters.

<table><tr><td colspan="2">Model</td><td>Precision</td><td>Recall</td><td> $\mathbf { F 1 ~ S c o r e }$ </td><td> $\mathbf { A c c u r a c y }$ </td><td>Parameters</td></tr><tr><td> $\overline { { { \mathrm { T G r a p h A E } } { \mathrm { h d c l } } } }$ </td><td></td><td> $\overline { { 0 . 9 7 4 \pm 0 . 0 0 5 } }$ </td><td> $\overline { { 0 . 9 7 3 \pm 0 . 0 0 4 } }$ </td><td> $\smash { 0 . 9 7 5 \pm 0 . 0 0 2 }$ </td><td> $\overline { { 0 . 9 7 9 \pm 0 . 0 0 1 } }$ </td><td>46k</td></tr><tr><td> $\mathrm { { T G r a p h M i x e d } \overline { { A } } E \_ h d c l }$ </td><td></td><td> $0 . 9 7 9 \pm 0 . 0 0 5$ </td><td> $0 . 9 6 8 \pm 0 . 0 0 8$ </td><td> $0 . 9 7 4 \pm 0 . 0 0 4$ </td><td> $0 . 9 7 9 \pm 0 . 0 0 1$ </td><td>158k</td></tr><tr><td> $\mathrm { T C N \dot { G } r a p h A E { 1 } \overline { { h } } d c { l } }$ </td><td></td><td> $0 . 9 8 5 \pm 0 . 0 0 4$ </td><td> $0 . 9 7 8 \pm 0 . 0 0 7$ </td><td> $0 . 9 8 1 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 5 \pm 0 . 0 0 1$ </td><td>287k</td></tr><tr><td> $\mathrm { T C N G r a p h A E 2 } _ { = } ^ { - } \mathrm { h d c l }$ </td><td></td><td> $0 . 9 9 1 \pm 0 . 0 0 1$ </td><td> $0 . 9 7 8 \pm 0 . 0 0 3$ </td><td> $0 . 9 8 4 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 7 \pm 0 . 0 0 1$ </td><td>159k</td></tr><tr><td> $\mathrm { T C N G r a \dot { p h A } E 3 } ^ { - } \mathrm { h d c l }$ </td><td></td><td> $0 . 9 8 3 \pm 0 . 0 0 7$ </td><td> $0 . 9 7 8 \pm 0 . 0 0 3$ </td><td> $0 . 9 8 1 \pm 0 . 0 0 2$ </td><td> $0 . 9 8 4 \pm 0 . 0 0 2$ </td><td>43k</td></tr><tr><td> $\underline { { \mathrm { T C N G r a p h A E 4 } } } _ { = } ^ { - } \mathrm { h d c l }$ </td><td></td><td> $0 . 9 8 4 \pm 0 . 0 0 3$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 4$ </td><td> $0 . 9 8 2 \pm 0 . 0 0 3$ </td><td> $0 . 9 8 5 \pm 0 . 0 0 3$ </td><td>82k</td></tr></table>

Table 1: Results of the standard autoencoders for label 0. Best scores are colored in purple.

<table><tr><td>Model</td><td> $\mathbf { P r e c i s i o n }$ </td><td>Recall</td><td> $\mathbf { F 1 ~ S c o r e }$ </td><td> $\mathbf { A c c u r a c y }$ </td><td>Parameters</td></tr><tr><td> $\overline { { \mathrm { T G r a p h A E } \_ \mathrm { h d c l } } }$ </td><td> $\overline { { 0 . 9 8 0 \pm 0 . 0 0 3 } }$ </td><td> $\overline { { 0 . 9 8 2 \pm 0 . 0 0 4 } }$ </td><td> $\overline { { 0 . 9 8 2 \pm 0 . 0 0 2 } }$ </td><td> $\overline { { 0 . 9 7 9 \pm 0 . 0 0 1 } }$ </td><td>46k</td></tr><tr><td> $\mathrm { { T G r a p h M i x e d } \overline { { A } } E \_ h d c l }$ </td><td> $0 . 9 7 9 \pm 0 . 0 0 2$ </td><td> $0 . 9 8 5 \pm 0 . 0 0 3$ </td><td> $0 . 9 8 2 \pm 0 . 0 0 1$ </td><td> $0 . 9 7 9 \pm 0 . 0 0 1$ </td><td>158k</td></tr><tr><td> $\mathrm { T C N G r a p h A E 1 \overline { { h } } d c l }$ </td><td> $0 . 9 8 4 \pm 0 . 0 0 5$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 3$ </td><td> $0 . 9 8 7 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 5 \pm 0 . 0 0 1$ </td><td>287k</td></tr><tr><td> $\mathrm { T C N G r a p h A E 2 } ^ { - } \mathrm { h d c l }$ </td><td> $0 . 9 8 5 \pm 0 . 0 0 2$ </td><td> $0 . 9 9 4 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 9 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 7 \pm 0 . 0 0 1$ </td><td>159k</td></tr><tr><td> $\mathrm { T C N G r a p h A E 3 \overline { { \Omega } } h d c l }$ </td><td> $0 . 9 8 5 \pm 0 . 0 0 2$ </td><td> $0 . 9 8 8 \pm 0 . 0 0 5$ </td><td> $0 . 9 8 7 \pm 0 . 0 0 2$ </td><td> $0 . 9 8 4 \pm 0 . 0 0 2$ </td><td>43k</td></tr><tr><td> $\mathrm { { T C N G r a \dot { p h A E 4 } } _ { = } ^ { - } h d c l }$ </td><td> $0 . 9 8 0 \pm 0 . 0 0 5$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 8 \pm 0 . 0 0 2$ </td><td> $0 . 9 8 5 \pm 0 . 0 0 3$ </td><td>82k</td></tr></table>

Table 2: Results of the standard autoencoders for label 1. Best scores are colored in purple.

We now clarify the relations between the components of our pipeline. The MLP classifier is trained jointly with the autoencoder: as the latent representations of the signals are exposed to class labels via the MLP, the autoencoder’s latent space is shaped accordingly. The SVC and K-means algorithms are subsequently trained independently on the latent representations extracted by the final trained model. This provides us with four distinct strategies to assign labels to a new batch of data:

a) The prediction from the trained MLP evaluated on the latent representation of the signal.

b) The prediction from the trained SVC evaluated on the latent representation of the signal.

c) The pseudo-label assigned by the K-means model (trained on the training set’s latent representations) evaluated on the latent representation of the signal.

d) The pseudo-label assigned by a new K-means model retrained directly on the latent representations of the new data batch.

Methods a) and b) operate in a fully supervised manner. In cases c) and d), an unsupervised algorithm, K-means, is used. However, because the latent embeddings were shaped by joint training with a supervised MLP, we classify these approaches as semi-supervised rather than strictly unsupervised. We performed a grid-search to test several models and diferent parameters repeating for each choice the 3 steps above and comparing the results obtained by the labelings a)-d). For each diferent model, we repeated this process 3 times using diferent random seeds. Finally, we selected, for each model type, the best performing models for evaluation on the test set. The evaluation on the test set has been conducted as in points 1)-3) above using the test set in place of the valid set: in this case we trained and evaluated our models for 10 times with 10 diferent random seeds, we discarded the best and the worst results for each score considered and then we computed the mean and standard deviation of the remaining scores.

Interestingly, the method that achieved an higher performance on both the valid and the test set was method c), similar to the performance of method d). As the former can be in principle used to label single out of sample signals we then reported in our paper the results obtained with this method. This behavior appeared to be consistent across essentially all good performing models. We report the results obtained with standard autoencoders in Table 1 and Table 2 while we report the results obtained using the variational autoencoders in Table 3 and in Table 4. Benchmark results are listed in Table 5: we have selected them by looking at results tabulated for models trained using train/valid/test splits similar to the ones we used in the paper [32] that we believe to be the most up to date on this particular problem.

<table><tr><td>Model</td><td>Precision</td><td>Recall</td><td> $\mathbf { F 1 ~ S c o r e }$ </td><td>Accuracy</td><td>Parameters</td></tr><tr><td> $\overline { { \mathrm { { T G r a p h V A E } } \mathrm { { h d c l } } } }$ </td><td> $\overline { { 0 . 9 7 9 \pm 0 . 0 0 4 } }$ </td><td> $\overline { { 0 . 9 7 6 \pm 0 . 0 0 3 } }$ </td><td> $\overline { { 0 . 9 7 6 \pm 0 . 0 0 4 } }$ </td><td> $\overline { { 0 . 9 8 0 \pm 0 . 0 0 3 } }$ </td><td>182k</td></tr><tr><td> $\mathrm { T G r a p h M i x e d V \overline { { A } } E \_ h d c l }$ </td><td> $0 . 9 7 7 \pm 0 . 0 0 3$ </td><td> $0 . 9 7 5 \pm 0 . 0 0 2$ </td><td> $0 . 9 7 6 \pm 0 . 0 0 2$ </td><td> $0 . 9 8 0 \pm 0 . 0 0 2$ </td><td>145k</td></tr><tr><td> $\mathrm { T C N \dot { G } r a p h V { A } E 1 \overline { { h } } d c l }$ </td><td> $0 . 9 9 2 \pm 0 . 0 0 2$ </td><td> $0 . 9 7 7 \pm 0 . 0 0 4$ </td><td> $0 . 9 8 4 \pm 0 . 0 0 2$ </td><td> $0 . 9 8 7 \pm 0 . 0 0 1$ </td><td>178k</td></tr><tr><td> $\mathrm { T C N G r a \dot { p h } V A E 2 ^ { - } h d c l }$ </td><td> $0 . 9 9 3 \pm 0 . 0 0 1$ </td><td> $0 . 9 7 2 \pm 0 . 0 0 3$ </td><td> $0 . 9 8 3 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 6 \pm 0 . 0 0 1$ </td><td>322k</td></tr></table>

Table 3: Results of the variational autoencoders for label 0. Best scores are colored in purple.

<table><tr><td>Model</td><td> $\mathbf { P r e c i s i o n }$ </td><td>Recall</td><td> $\mathbf { F 1 ~ S c o r e }$ </td><td>Accuracy</td><td>Parameters</td></tr><tr><td> $\overline { { \mathrm { T G r a p h V A E } \mathrm { h d c l } } }$ </td><td> $\overline { { 0 . 9 8 3 \pm 0 . 0 0 2 } }$ </td><td> $\overline { { 0 . 9 8 5 \pm 0 . 0 0 3 } }$ </td><td> $\overline { { 0 . 9 8 3 \pm 0 . 0 0 3 } }$ </td><td> $0 . 9 8 0 \pm 0 . 0 0 3$ </td><td>182k</td></tr><tr><td> $\mathrm { { T G r a p h i n i x e d V } \overline { { A } } E \quad h d c l }$ </td><td> $0 . 9 8 2 \pm 0 . 0 0 2$ </td><td> $0 . 9 8 4 \pm 0 . 0 0 2$ </td><td> $0 . 9 8 3 \pm 0 . 0 0 2$ </td><td> $0 . 9 8 0 \pm 0 . 0 0 2$ </td><td>145k</td></tr><tr><td> $\mathrm { T C N \dot { G } r a p h V { A } E 1 \overline { { h } } d c l }$ </td><td> $0 . 9 8 4 \pm 0 . 0 0 2$ </td><td> $0 . 9 9 4 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 9 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 7 \pm 0 . 0 0 1$ </td><td>178k</td></tr><tr><td> $\underline { { \mathrm { T C N G r a } \dot { \mathrm { p h V A E 2 } } } } _ { - } ^ { - } \mathrm { h d c l }$ </td><td> $0 . 9 8 0 \pm 0 . 0 0 2$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 8 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 6 \pm 0 . 0 0 1$ </td><td>322k</td></tr></table>

Table 4: Results of the variational autoencoders for label 1. Best scores are colored in purple.

![](images/380718747ad21db6756a1df5d0e80a997eaf0cade16a982a2e9694f6e90446e9.jpg)  
(a)

![](images/a0c632d51bd6c1339bb276bc7a26791b18765ed3acd1c04689ac39ee2f3e01b8.jpg)  
(b)  
Fig. 4: Representations of the hidden embeddings of the signals obtained by TGraphMixedVAE\_hdcl (a) and TCNGraphVAE1\_hdcl (b), and converted to 2D representations using TSNE. Label 0 indicates a normal signal, 1 an abnormal signal.

## 5.1.1 Discussion

From the experimental results it appears that in this case the best autoencoder archi tecture has an encoder and a decoder as detailed in Figure 2. Also, while Variational autoencoders proved on average to perform better when they had enough parameters, we noticed that the ordinary ones were able to achieve a good performance also in with model having a reduced number of parameters. We deem the number of parameters to be an important metric to evaluate the efectiveness of an algorithm: a small model can be deployed on more devices after appropriate quantization. From Table 5 it emerges that our proposed best model, a variational graph-based autoencoder, has a performance superior to the ones of the other variational autoencoders considered in our comparison, while at the same time having a smaller number of parameters. In addition, our proposed best small model still has high scores in the other metrics considered. This is noteworthy as this model has roughly half of the parameters of our proposed best model and roughly a tenth of the parameters of the model proposed in [32] that, to our knowledge, is the one achieving the best accuracy overall. In addition, while the model developed in [32] labels the signals using an autoencoder’s reconstruction error, ours uses the embedding into the latent space of an autoencoder to assign to each signal a label: as a result our algorithm, when deployed, has roughly half the size of our full trained model (as we need only the encoder of our autoencoder). In addition, the models we proposed appear to be stable under change of the seed used for the training: unfortunately, we do not know if the results reported by the other models we report in Table 5 have a similar behavior or if in op. cit. only one seed has been considered. Finally we report in table 6 the Wilcoxon signed-rank test across random seeds for selected pairs of models. It emerges that there is no statistically significant diference among models with a similar encoder and decoder structure while a significant one emerges when considering the comparisons between the models only involving graphs (TGraphAE/VAE) with the ones having TCNs both in the encoder and in the decoder (TCNGraphAE/VAE).

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Train/Val/Test</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>F1 Score</td><td rowspan=1 colspan=1>Params</td></tr><tr><td rowspan=1 colspan=1>Proposed best,TCNGraphVAE1</td><td rowspan=1 colspan=1>0.1/0.05/0.85</td><td rowspan=1 colspan=1>98.7%± 0.1</td><td rowspan=1 colspan=1>0.984± 0.002</td><td rowspan=1 colspan=1>0.994± 0.001</td><td rowspan=1 colspan=1>0.989± 0.001</td><td rowspan=1 colspan=1>178k</td></tr><tr><td rowspan=1 colspan=1>Proposed  bestsmall,TCNGraphAE4</td><td rowspan=1 colspan=1>0.1/0.05/0.85</td><td rowspan=1 colspan=1>98.5%± 0.3</td><td rowspan=1 colspan=1>0.980± 0.005</td><td rowspan=1 colspan=1>0.995± 0.001</td><td rowspan=1 colspan=1>0.988± 0.002</td><td rowspan=1 colspan=1>82k</td></tr><tr><td rowspan=1 colspan=1>Pereira et al. [31],VAE + SVM</td><td rowspan=1 colspan=1>0.08/0.02/0.9</td><td rowspan=1 colspan=1>98.43%</td><td rowspan=1 colspan=1>0.9847</td><td rowspan=1 colspan=1>0.9843</td><td rowspan=1 colspan=1>0.9844</td><td rowspan=1 colspan=1>273k</td></tr><tr><td rowspan=1 colspan=1>Matias et al. [28],VAE + local sim-ilarity score</td><td rowspan=1 colspan=1>0.05/0.05/0.9</td><td rowspan=1 colspan=1>97.11%</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>96.01</td><td rowspan=1 colspan=1>250k,(estimate)</td></tr><tr><td rowspan=1 colspan=1>ECG-NET [33],LSTM AE</td><td rowspan=1 colspan=1>0.05/0.005/0.9</td><td rowspan=1 colspan=1>98.56%</td><td rowspan=1 colspan=1>0.9794</td><td rowspan=1 colspan=1>0.9957</td><td rowspan=1 colspan=1>0.9875</td><td rowspan=1 colspan=1>1649k</td></tr><tr><td rowspan=1 colspan=1>AttentivECGRU[32]GRUAE+attention</td><td rowspan=1 colspan=1>0.05/0.005/0.9</td><td rowspan=1 colspan=1>98.96%</td><td rowspan=1 colspan=1>0.9912</td><td rowspan=1 colspan=1>0.9908</td><td rowspan=1 colspan=1>0.9909</td><td rowspan=1 colspan=1>743k</td></tr></table>

Table 5: Comparison of our results with other existing methods applied on the ECG5000 dataset. Best scores are colored in purple. The final two models reported in the table use only the normal samples in their train/valid sets. However, note that the ordering of the dataset has been preserved in all experiments: as a consequence their training and validation sets are subsets of our training set and our test set is a subset of theirs.

<table><tr><td>Model A</td><td>Model B</td><td>Wilcoxon p-value</td><td>Main difference</td></tr><tr><td>TCNGraphAE4 hdcl</td><td>TCNGraphAE3 hdcl</td><td>0.5566</td><td>Bidirectional and not bidi- rectional</td></tr><tr><td>TCNGraphAE1 hdcl</td><td>TCNGraphAE4 hdcl</td><td>0.9527</td><td>Lowest number of parame- ters vs greatest</td></tr><tr><td>TCNGraphVAE1 hdcl</td><td>TCNGraphAE4 hdcl</td><td>0.3750</td><td>Best variational vs best non- variational</td></tr><tr><td>TGraphMixedAE hdcl</td><td>TGraphAE hdcl</td><td>0.3726</td><td>Different decoder structure</td></tr><tr><td>TGraphMixedVAE_hdcl</td><td>TGraphVAE_hdcl</td><td>1.0000</td><td>Different decoder structure</td></tr><tr><td>TCNGraphVAE1 hdcl</td><td>TGraphVAE hdcl</td><td>0.0020</td><td>Different encoder and decoder structure</td></tr><tr><td>TCNGraphAE4 hdcl</td><td>TGraphAE_hdcl</td><td>0.0020</td><td>Different encoder and decoder structure</td></tr></table>

Table 6: Paired Wilcoxon signed-rank tests across ten training seeds. The final two comparisons are statistically significant.

## 5.2 Validation on industry data

In our preliminary work [41] we used unsupervised, non-variational variants of the models we employed in this paper to solve a quality detection problem. In that work, we applied a fully unsupervised method on an industry dataset of signals recorded using the ButterfLife Device [3] and that was provided to us by the company VST. The results we obtained confirmed that graph-based TCNs are competitive and can surpass state-of-art TCN-based architectures for these types of problems, while allowing for the development of models having a restricted number of parameters. We reproduce in appendix B the table of the results obtained in op. cit.

## 6 Conclusions

In this paper, we explained how to see TCNs as particular GNNs and we explained why from a theoretical point of view it might make more sense to consider time series as directed graphs for the design of deep learning architectures. This can be seen as one of the main goals of the paper: to introduce a theoretical framework justifying the combined usage of TCNs and GNNs. Our experiments confirmed that, if the cases we considered, graph convolutional layers added to established algorithms can help to reduce their number of parameters while not decreasing their performance and adding a slight stabilizing efect. As a third and final main contribution, we introduced a specific semi-supervised, parameter-eficient pipeline for the study of ECG signals. An important point is that only the encoder of the trained autoencoders used in the pipeline and the K-means centroids coordinates are needed after training to label signals in inference, thus reducing roughly by half the resources needed to use the trained autoencoder: this contrasts with other approaches such as the one of [33] that are based on the reconstruction error of an autoencoder: in these cases running the full autoencoder is necessary for inference and labeling purposes. We plan to continue the study of GNNs applied to time digraphs with features in the context of multivariate time series, constructing more complex time digraph structures and using more capable message passing mechanisms.

## 7 Funding Details

The research of the first author has been supported by the project PRIN - FRAIL and by CaLISTA CA 21109. The research of the second author has been supported by CaLISTA CA 21109, CaLIGOLA MSCA-2021-SE-01-101086123, MSCA-DN CaLiFor-NIA—101119552, PNRR MNESYS, the PNRR National Center for HPC, Big Data, and Quantum Computing CUP J33C22001170001, SimQuSec CUP J13C2200068000; INFN Sezione Bologna, Gast initiative and GNSAGA Indam.

## 8 Author Contribution

All authors contributed equally to this work.

## 9 Data availability statement

The description of the ECG5000 dataset can be found at https://www. timeseriesclassification.com/description.php?Dataset=ECG5000; we downloaded it from the TensorFlow repository http://storage.googleapis.com/download.tensorflow. org/data/ecg.csv. The dataset as it was downloaded is also available as a csv file in the supplementary materials of this paper. The source code is available upon request.

## A Models hyperparameters and implementation

We trained our model on workstations having RTX4090 and RTX5090 GPUs and we used the PyTorch library to write our models. All the remaining code was written in python and used standard numerical libraries such as Numpy, Pandas and Scikitlearn. To train our models, we used the Adam optimizer with a learning rate of 0.001 and the parameter drop\_last=True. To make our trainings reproducible we used built-in PyTorch methods to seed the random number generators called in the trainings we performed. Source code is available in the public repository link added upon acceptance, until then we prefer it to stay upon request. We detail here all the hyperparameters used in the training of our models. We list the most significant ones in two distinct tables: they correspond exactly to the models presented in the main body of the paper and whose results are presented in Table 4. We write here a short description and recollection of all the terms used in these tables:

• Num Channels: the number of skip connection layers and their output dimensions (list). The value after the comma is the bottleneck channel dimension post-encoder’s 1D graph convolution. The dimension used in the decoder are the same but in inverse order, except that in the case of the model flagged with the symbol ⋄.

• Skip dims: skip dimensions as per Section 4.

• GConv type: graph convolution details (if present) following the skip block: convolution type and hidden dimensions. For GAT, it includes the number of heads (e.g., GAT2H for 2 heads).

<table><tr><td>Model</td><td>Num Channels</td><td>Skip dims</td><td>GConv type</td><td>Kernel sizes</td><td>Dilations</td><td>Shrink</td></tr><tr><td>TGraphAE hdcl</td><td> $\left[ 1 2 8 , 6 4 , 3 2 , 1 6 , 8 , 4 , 2 \right]$ </td><td> $\overline { { [ 6 4 , 3 2 , 1 6 , 8 , 4 , 2 ] } }$ </td><td>Sage</td><td></td><td></td><td>35</td></tr><tr><td>TGraphMixedAE hdcl*</td><td>[64] * 7, 8</td><td>[16] * 7</td><td> $\mathrm { S a g e , \bar { [ 1 0 0 ] } }$ </td><td> $[ 3 ] * 7$ </td><td> $[ 1 , 2 , 4 , 8 , 1 6 , 3 2 , 6 4 ]$ </td><td>20</td></tr><tr><td>TCNGraphAE1_hdcl</td><td> $[ 1 2 8 , 1 1 \dot { 2 } , 9 \dot { 6 } , 8 0 , 6 4 , 4 8 , 8 ]$ </td><td> $[ 6 4 , 4 \dot { 8 } , 3 \dot { 2 } , 1 6 , 8 , 4 ]$ </td><td> $\bar { \mathrm { G A T 1 H } } \bar { }$ </td><td> $[ 8 ] * 6$ </td><td> $\cdot _ { \ [ 1 , 2 , 4 , 8 , 1 6 , 3 2 ] }$ </td><td>35</td></tr><tr><td>TCNGraphAE2 hdcl†</td><td> $[ 6 4 , 4 8 , 3 2 , 1 6 , 8 , 4 , 4 ]$ </td><td> $[ 3 2 , 1 6 , 8 , 4 , 2 , 2 ]$ </td><td> $\mathrm { S a g e , ~ [ 1 6 , 1 6 ] }$ </td><td> $[ 8 ] * 6$ </td><td> $[ 1 , 2 , 1 , 4 , 1 , 8 ]$ </td><td>20</td></tr><tr><td>TCNGraphAE3_hdcl</td><td> $\left[ 6 4 , 4 8 , 3 2 , 1 6 , 8 , 4 , 4 \right]$ </td><td>[32, 16, 8, 4, 2, 2]</td><td>Sage, [16, 16]</td><td> $[ 8 ] * 6$ </td><td> $[ 1 , 2 , 1 , 4 , 1 , 8 ]$ </td><td>10</td></tr><tr><td>TCNGraphAE4 hdcl†</td><td> $[ 6 4 , 4 8 , 3 2 , 1 6 , 8 , 4 , 4 ]$ </td><td> $[ 3 2 , 1 6 , 8 , 4 , 2 , 2 ]$ </td><td> $\underline { { \mathrm { S a g e , ~ } [ 1 6 , 1 6 ] } }$ </td><td> $[ 8 ] * 6$ </td><td> $\underline { { [ 1 , 2 , 1 , 4 , 1 , 8 ] } }$ </td><td>10</td></tr></table>

Table 7: Hyperparameters of the Autoencoder algorithms used for the dataset ECG5000

<table><tr><td>Model</td><td>Num Channels</td><td>Skip dims</td><td>GConv type</td><td>Kernel sizes</td><td>Dilations</td><td>Shrink</td></tr><tr><td>TGraphVAE hdcl</td><td> $\overline { { [ 6 4 ] * 6 , 2 } }$ </td><td>[16] * 6</td><td> $\overline { { \mathrm { S a g e , ~ } [ 1 0 0 ] } }$ </td><td></td><td></td><td>35</td></tr><tr><td>TGraphMixedVAE hdcl*</td><td> $[ \dot { 1 } 2 \dot { 8 } ] * 7 , 8$ </td><td>[32] * 7</td><td> $\Breve { \mathrm { G A T 1 H } } ^ { \prime }$ </td><td> $[ 3 ] * 7$ </td><td> $[ 1 , 2 , 4 , 8 , 1 6 , 3 2 , 6 4 ]$ </td><td>35</td></tr><tr><td>TCNGraphVAE1_hdcl</td><td> $[ 1 2 8 , 1 1 2 , 9 6 , 8 0 , 6 4 , 4 8 , 8 ]$ </td><td>[16] * 6</td><td> $\mathrm { S a g e , ~ [ 1 0 0 ] }$ </td><td> $[ 1 6 , 1 2 , 8 , 4 , 2 , 2 ]$ </td><td> $[ 1 , 2 , 2 , 2 , 2 , 2 ]$ </td><td>20</td></tr><tr><td>TCNGraphVAE2 hdcl†</td><td> $[ 1 2 8 , 1 1 2 , 9 6 , 8 0 , 6 4 , 4 8 , 8 ]$ </td><td> ${ \bigl [ } 1 6 { \bigr ] } * 6$ </td><td> $\mathrm { S a g e , } \ [ 1 0 0 ]$ </td><td> $[ 1 6 , 1 2 , 8 , 4 , 2 , 2 ]$ </td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td> $[ 1 , 2 , 2 , 2 , 2 , 2 ]$ </td><td>10</td></tr></table>

Table 8: Hyperparameters of the Variational Autoencoder algorithms used for the dataset ECG5000

• Pool type, Downsample, Upsample: standard layer notation is used (e.g., PyTorch Geometric).

• Shrink: the reduction factor of the average pooling applied by the downsample at the end of the encoder.

• Kernel sizes: a list of the kernel sizes of each 1D convolution employed.

• Dilations a list of the dilation factor of each 1D convolution employed.

Tables 7 and 8 list all model hyperparameters, found using a grid search, for the dataset ECG5000. We applied SiLU activation, batch normalization, and dropout. The models TGraphAE\_hdcl and TGraphVAE\_hdcl feature a final 1D dilated convolution (dilation 1) after their decoder. The models TGraphMixedAE\_hdcl and TGraphMixedVAE\_hdcl are listed with an asterisk to signal the fact that in this case the hyperparameters num Channels and skip dims are not the same for the encoder and the decoder. The ones reported in the table are those of the encoders, that are made of graph convolutions. The decoders had twice the number of num Channels and skip dims of the correspondent encoders. Finally, the models marked with a dagger † are bidirectional (see Section 4).

## B Results on unsupervised quality detection

In this section we reproduce the table of the results we obtained in [41].

The names of the models are coherent with both the ones introduced in the main body of this paper and the ones found in [41]. From the table, one can notice that graph-based algorithms slightly outperform the baseline TCN in terms of performance and stability. We refer to our work [41] for a full description of the experiments and the methodology employed and for the details about approaches A and B that difer mainly as they use Kmeans [23] or DBscan [9] to find clusters and assign labels in an unsupervised way in the latent spaces of the autoencoders considered.

<table><tr><td rowspan="2">Model</td><td colspan="2"> $\mathrm { P o s i t i v e \ c l a s s = \ l a b e l \ 1 }$ </td><td colspan="2">Positive class = label 0</td><td rowspan="2">Accuracy</td><td rowspan="2">Params</td></tr><tr><td>Precision</td><td>Recall</td><td>Precision</td><td>Recall</td></tr><tr><td colspan="7">Kmeans, approach A</td></tr><tr><td> $\overline { { \mathrm { T G r a p h M i x e d A E } } }$ </td><td> $\overline { { 0 . 9 7 3 \pm 0 . 0 0 6 } }$ </td><td> $\overline { { 0 . 9 7 4 \pm 0 . 0 1 1 } }$ </td><td> $\overline { { 0 . 8 5 4 \pm 0 . 0 5 1 } }$ </td><td> $\overline { { 0 . 8 4 7 \pm 0 . 0 3 5 } }$ </td><td> $\overline { { 0 . 9 5 5 \pm 0 . 0 0 6 } }$ </td><td>145k</td></tr><tr><td>TGraphAE</td><td> $0 . 9 6 0 \pm 0 . 0 0 7$ </td><td> $0 . 9 9 3 \pm 0 . 0 0 6$ </td><td> $0 . 9 5 2 \pm 0 . 0 4 0$ </td><td> $0 . 7 6 5 \pm 0 . 0 4 4$ </td><td> $0 . 9 5 9 \pm 0 . 0 0 3$ </td><td>44k</td></tr><tr><td>TCNGraphAE1</td><td> $0 . 9 6 7 \pm 0 . 0 0 3$ </td><td> $0 . 9 9 8 \pm 0 . 0 0 2$ </td><td> $0 . 9 8 5 \pm 0 . 0 1 4$ </td><td> $0 . 8 0 6 \pm 0 . 0 1 6$ </td><td> $0 . 9 6 9 \pm 0 . 0 0 2$ </td><td>33k</td></tr><tr><td>TCNGraphAE2</td><td> $0 . 9 6 5 \pm 0 . 0 0 2$ </td><td> $0 . 9 9 7 \pm 0 . 0 0 1$ </td><td> $0 . 9 7 6 \pm 0 . 0 0 7$ </td><td> $0 . 7 9 6 \pm 0 . 0 1 2$ </td><td> $0 . 9 6 6 \pm 0 . 0 0 2$ </td><td>258k</td></tr><tr><td>TCNAE1</td><td> $0 . 9 6 6 \pm 0 . 0 1 2$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 5$ </td><td> $0 . 9 6 6 \pm 0 . 0 3 2$ </td><td> $0 . 7 9 8 \pm 0 . 0 7 4$ </td><td> $0 . 9 6 4 \pm 0 . 0 0 9$ </td><td>19k</td></tr><tr><td>TCNAE2</td><td> $0 . 9 4 9 \pm 0 . 0 0 5$ </td><td> $0 . 9 9 9 \pm 0 . 0 0 1$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 6$ </td><td> $0 . 6 9 2 \pm 0 . 0 3 1$ </td><td> $0 . 9 5 4 \pm 0 . 0 0 4$ </td><td>208k</td></tr><tr><td colspan="7">Dbscan, approach B</td></tr><tr><td>TGraphMixedAE</td><td> $0 . 9 8 4 \pm 0 . 0 0 5$ </td><td> $\overline { { 0 . 9 4 4 \pm 0 . 0 1 2 } }$ </td><td> $\overline { { 0 . 7 4 5 \pm 0 . 0 3 7 } }$ </td><td> $\overline { { 0 . 9 0 9 \pm 0 . 0 2 8 } }$ </td><td> $\overline { { 0 . 9 3 9 \pm 0 . 0 0 7 } }$ </td><td>145k</td></tr><tr><td>TĠraphAE</td><td> $0 . 9 6 8 \pm 0 . 0 0 6$ </td><td>0.989 ± 0.006</td><td>0.933 ± 0.034</td><td> $0 . 8 1 3 \pm 0 . 0 3 8$ </td><td> $0 . 9 6 2 \pm 0 . 0 0 2$ </td><td>44k</td></tr><tr><td>TCNGraphAE1</td><td> $0 . 9 7 1 \pm 0 . 0 0 4$ </td><td> $0 . 9 9 1 \pm 0 . 0 0 5$ </td><td> $0 . 9 4 0 \pm 0 . 0 2 8$ </td><td> $0 . 8 2 9 \pm 0 . 0 2 2$ </td><td> $0 . 9 6 6 \pm 0 . 0 0 3$ </td><td>33k</td></tr><tr><td>TCNGraphAE2 TCNÁE1</td><td> $0 . 9 7 9 \pm 0 . 0 0 7$ </td><td> $0 . 9 8 5 \pm 0 . 0 0 6$ </td><td> $0 . 9 1 3 \pm 0 . 0 3 1$ </td><td> $0 . 8 7 7 \pm 0 . 0 4 3$ </td><td> $0 . 9 6 7 \pm 0 . 0 0 1$ </td><td>258k</td></tr><tr><td>TCNAE2</td><td> $0 . 9 7 1 \pm 0 . 0 0 7$ </td><td> $0 . 9 8 5 \pm 0 . 0 1 1$ </td><td> $0 . 9 1 3 \pm 0 . 0 5 7$ </td><td> $0 . 8 3 3 \pm 0 . 0 4 3$ </td><td> $0 . 9 6 2 \pm 0 . 0 0 7$ </td><td>19k</td></tr><tr><td></td><td> $0 . 9 7 3 \pm 0 . 0 0 6$ </td><td> $0 . 9 8 8 \pm 0 . 0 0 5$ </td><td> $0 . 9 2 5 \pm 0 . 0 2 7$ </td><td> $0 . 8 4 6 \pm 0 . 0 3 9$ </td><td> $0 . 9 6 6 \pm 0 . 0 0 2$ </td><td>208k</td></tr></table>

Table 9: Best scores in purple and second best in blue.

## References

[1] Bai, S., Kolter, J.Z., Koltun, V.: An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. ArXiv abs/1803.01271, 2018.

[2] Bauer, F.: Normalized graph Laplacians for directed graphs. Linear Algebra and its Applications 436(11), 4193-4222, 2012.

[3] Salton, F., Kette, S., Confalonieri, P., Fonda S., Lerda, S., Hughes, M., Confalonieri, M., Ruaro, B.: Clinical Evaluation of the ButterfLife Device for Simultaneous Multiparameter Telemonitoring in Hospital and Home Settings. Diagnostics 12, 3115, 2022.

[4] Schuster, M., Paliwal, K.: Bidirectional Recurrent Neural Networks. IEEE Transactions on Signal Processing 45(11), 1997.

[5] Bodnar, C., Di Giovanni, F., Chamberlain, B.P., Lio’, P., Bronstein, M.M.: Neural Sheaf Difusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs. ArXiv abs/2202.04579, 2022.

[6] Bronstein, M.M., Bruna, J., LeCun, Y., Szlam, A., Vandergheynst, P.: Geometric Deep Learning: Going beyond Euclidean data. IEEE Signal Process. Mag. 34(4), 18–42, 2017.

[7] Cao, D., Wang, Y., Duan, J., Zhang, C., Zhu, X., Huang, C., Tong, Y., Xu, B., Bai, J., Tong, J., Zhang, Q.: Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting. Advances in Neural Information Processing Systems (NeurIPS) 33, 17766-17778, 2020.

[8] Cortes, C., Vapnik, V.: Support-vector networks. Machine Learning 20(3), 1995.

[9] Ester, M., Kriegel, H.P., Sander, J., Xu, X.: A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. Proc. 2nd Int. Conf.

on Knowledge Discovery and Data Mining (KDD), 1996.

[10] Fey, M., Lenssen, J.E.: Fast Graph Representation Learning with PyTorch Geometric. ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.

[11] Fioresi, R., Zanchetta, F.: Deep Learning and Geometric Deep Learning: an introduction for mathematicians and physicists. Int. J. Geom. Methods Mod. Phys. 20, 2023.

[12] Gers, F.A., Schmidhuber, J., Cummins, F.: Learning to Forget: Continual Prediction with LSTM. Neural Computation 12, 2451–2471, 2000.

[13] Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio’, P., Bengio, Y.: Graph Attention Networks. ArXiv abs/1710.10903, 2017.

[14] Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B., Smola, A.: A Kernel Two-Sample Test. J. Mach. Learn. Res. 13, 723–773, 2012.

[15] Hamilton, J.D.: Time Series Analysis. 2nd ed. Princeton University Press, 1994.

[16] Hamilton, W.L., Ying, Z., Leskovec, J.: Inductive Representation Learning on Large Graphs. NIPS, 2017.

[17] Wang, X., Ji, H., Shi, C., Wang, B., Cui, P., Yu, P.S., Ye, Y.: Heterogeneous Graph Attention Network. WWW Conference, 2019.

[18] Hochreiter, S., Schmidhuber, J.: Long Short-Term Memory. Neural Computation 9, 1735–1780, 1997.

[19] Khandual, A., Dutta, Lenkaa, R., Nayak, S. R..: MED-NET: a novel approach to ECG anomaly detection using LSTM auto-encoders. Int. J. Comput. Appl. Technol. 65, 343, 2021.

[20] Kingma, D.P., Welling, M.: Auto-Encoding Variational Bayes. ArXiv abs/1312.6114, 2013.

[21] Kipf, T., Welling, M.: Semi-Supervised Classification with Graph Convolutional Networks. ArXiv abs/1609.02907, 2016.

[22] Kiranyaz, S., Avcı, O., Abdeljaber, O., Ince, T., Gabbouj, M., Inman, D. J.: 1D Convolutional Neural Networks and Applications: A Survey. ArXiv abs/1905.03554, 2019.

[23] Lloyd, S.P.: Least squares quantization in PCM. IEEE Trans. Inf. Theory 28, 1982.

[24] Krizhevsky, A., Sutskever, I., Hinton, G.E.: Imagenet classification with deep convolutional neural networks. NIPS, 1097–1105, 2012.

[25] LeCun, Y., Bengio, Y., Hinton, G.: Deep Learning. Nature 521, 436–444, 2015.

[26] LeCun, Y., Bottou, L., Bengio, Y., Hafner, P.: Gradient-based learning applied to document recognition. Proc. IEEE 86, 2278–2324, 1998.

[27] LeCun, Y., Cortes, C., Burges, C.J.: MNIST handwritten digit database. ATT Labs. Available online.

[28] Matias, P., Folgado, D., Gamboa, H., Carreiro, A.: Robust anomaly detection in time series through variational AutoEncoders and a local similarity score. Biosignals, 2021.

[29] Mehrab, A., Lapenna M., Zanchetta F., Simonetti A., Faglioni F., Malagoli A., Fioresi R.: Kolmogorov–Arnold and LSTM Convolutional Network Models for Quality Recognition of PPG Signals. Entropy 27(4), 1–11, 2025.

[30] Paszke, A., Gross, S., Massa, F., et al.: PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems (NeurIPS) 32, 2019.

[31] Pereira, J., Silveira, M.: Unsupervised representation learning and anomaly detection in ECG sequences. Int. J. Data Min. Bioinform., 2019.

[32] Roy, M., Halder, A., Majumder, S., Biswas, U.: AttentivECGRU: GRU-based autoencoder with attention mechanism. Applied Soft Computing 167, 2024.

[33] Roy, M., Majumder, S., Halder, A., Biswas, U.: ECG-NET: A deep LSTM autoencoder for detecting anomalous ECG. Eng. Appl. Artif. Intell. 124, 2023.

[34] Tealab, A.: Time series forecasting using artificial neural networks methodologies: A systematic review. Future Computing and Informatics Journal, 2018.

[35] Thill, M., Konen, W., Wang, H., Bäck, T.: Temporal convolutional autoencoder for unsupervised anomaly detection in time series. Appl. Soft Comput. 112, 107751, 2021.

[36] Tong, Z., Liang, Y., Sun, C., Rosenblum, D.S., Lim, A.: Directed Graph Convolutional Network. ArXiv 2004.13970, 2020.

[37] Tsay, R.S.: Analysis of Financial Time Series. Technometrics 48, 316–316, 2005.

[38] Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., Yu, P.S.: A Comprehensive Survey on Graph Neural Networks. IEEE Trans. Neural Netw. Learn. Syst. 32, 4–24, 2021.

[39] Wu, Z., Pan, S., Long, G., Jiang, J., Zhang, C.: Graph WaveNet for Deep Spatial Temporal Graph Modeling. ArXiv, 2019.

[40] Zanchetta, F., Simonetti, A., Faglioni, G., Malagoli, A., Fioresi, R.: A Geometric Deep Learning approach to blood pressure regression. Proc. Machine Learning Research, 2022.

[41] Simonetti, A., Zanchetta, F.: Graph Neural Networks and Time Series as Directed Graphs for Quality Recognition. LNCS, Springer, 2025.

[42] Zhao, S., Song, J., Ermon, S.: InfoVAE: Balancing Learning and Inference in Variational Autoencoders. AAAI, 2019.