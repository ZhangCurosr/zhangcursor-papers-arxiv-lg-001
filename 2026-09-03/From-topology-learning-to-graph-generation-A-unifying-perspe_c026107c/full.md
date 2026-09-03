# From topology learning to graph generation: A unifying perspective

Xiaowen Dong\*, Hoi-To Wai\*, Siheng Chen, Laura Toni, and Dorina Thanou\*

## Abstract

Learning graph structures from data is a fundamental problem that spans a wide range of signal processing and machine learning tasks. While significant effort has been made to tackle the problem, existing research has largely evolved along two parallel directions. The first seeks to infer the topology of an individual graph from observations supported on it, whereas the second seeks to learn a generative distribution from observed graph instances, enabling the sampling of new graphs. This review presents a unified framework that connects these formulations by viewing them as inverse problems of a common generation process for graph data. We review the major methodologies within this framework, highlight their relationships, strengths, and limitations, and identify opportunities for integrating ideas across paradigms. By bridging graph topology learning and graph generation, this review provides a broader cross-disciplinary perspective on the field and outlines promising directions for future research.

## I. INTRODUCTION

Learning the underlying structure from data plays a fundamental role in subsequent processing and analysis tasks. In many domains, this structure is naturally represented as a graph, which provides a flexible mathematical framework for modeling pairwise or higher-order relationships among entities. While graphs are often treated as given, in many practical settings they are only partially observed or entirely unknown, making structure learning a fundamental challenge. Broadly, graph structure learning can be viewed from two complementary perspectives. The first seeks to learn the topology of an individual graph by estimating its nodes and edges from data. The second aims to learn a graph generative mode that captures the distribution over a collection of graphs, enabling the characterization and synthesis of graph ensembles, as well as sampling of new graph instances. Throughout this article, we use the term “graph structure learning” to refer to learning either the topology of individual graphs directly or the distributions governing collections of graphs indirectly.

![](images/995633d84b06ddb894baf8a9700ca8c69e7e4b33310695bb5fe471f4652d9759.jpg)  
Fig. 1: An illustration of the two problem instances of graph structure learning in social networks.

The two perspectives differ not only in the final objective, but also in the nature of the observed data. In the first formulation, which we term “learning graph topology”, observations typically consist of node observations supported on an unknown graph, whereas in the second, which we term “learning graph generation”, the observations are graph samples drawn from an underlying graph distribution. This distinction is illustrated through the following social network example (Fig. 1). In the top row, we observe preferences or behaviors of a group of individuals in the form of graph signals, and aim to learn the graph topology that characterizes the social relations among them. The social network topology can be used to identify key individuals in the network, or facilitate downstream tasks such as user-level predictions. In the bottom row, we observe social interactions among groups of individuals in the form of graph samples, and aim to learn a probability distribution that governs the generation of such interactions. The learned distribution can then be used to predict future interactions or generate an entirely new social network topology.

To provide a unified perspective, this review introduces a two-stage generation process of graph-related data in Fig. 2 and defines the learning problems as the inverse problems of each stage. We move from left to right in the figure, where we use Θ to denote the parameter space for the graph generation process, G to denote the space of graph topologies (represented by an adjacency matrix $\mathbf { W } \in \mathbb { R } ^ { N \times N }$ for example), and X to denote the space of node observations, i.e., graph signals (represented by a vector $\mathbf { x } \in \mathbb { R } ^ { N } )$ . The first stage of the generation process is defined by a mapping $h : \Theta \to \mathcal { G }$ from the parameter space to the graph topology space, which describes how a graph is generated from its underlying parameters. For $\theta \in \Theta , h ( \theta )$ typically refers to a probability distribution supported on ${ \mathcal { G } } .$ . The second stage is defined by a mapping $f : \mathcal G  \mathcal { X }$ from the graph topology space to the graph signal space, which describes how node observations are generated from a given graph. Together, these two mappings describe a common generation process for graph-structured data. Within this framework, the two main graph structure learning problems naturally arise as inverse problems of the corresponding generation stages. In particular, learning graph topology corresponds to inferring G from observed x via the inverse of $f ,$ while learning graph generation corresponds to inferring θ from observed G via the inverse of h.

![](images/f7e75ac4f4ca0f6081ea910891426ed84ff166c4931305c3cfa40a5419ddea32.jpg)  
Fig. 2: An illustration of the two families of approaches to graph structure learning.

Existing approaches for graph structure learning can be broadly categorized into two families: 1) modelbased learning and 2) data-driven learning<sup>1</sup>. For graph topology learning, the two approaches differ by whether the f-map is assumed to be known beforehand (denoted by f<sub>X</sub> in Fig. 2) or not (denoted by $\widehat { f } _ { \mathbf { X } } )$ . In the model-based setting, $f _ { \mathbf { X } }$ is typically derived from statistical or physical principles. This leads to a rigorous mathematical formulation that often enables strong identifiability guarantees, but its applicability is inherently limited by the validity of the underlying modeling assumptions. In contrast, data-driven approaches make minimal assumptions about the data-generating process and instead learn an implicit $\widehat { f } _ { \mathbf { X } }$ directly from data. They are more adaptive as they naturally allow learning to be guided by a downstream task. For example, the growing literature on graph structure learning and graph rewiring, as termed by the graph machine learning (GML) community, aims to refine an existing graph topology to better serve downstream tasks such as node or graph classification. A similar trade-off exists for learning graph generation. Herein, model-based and data-driven learning are then distinguished by whether the h-map is assumed to be known beforehand (denoted by $h _ { \mathbf { G } }$ in Fig. 2) or not (denoted by $\widehat { h } _ { \mathbf { G } } )$ . The former typically relies on chosen random graph models, while the latter facilitates learning via latent node representations, which are then used for graph generation. As a consequence, it often enables the learning of a probability distribution, instead of merely a point estimate. Similar to graph topology learning, this increased flexibility comes at the cost of reduced interpretability and weaker theoretical guarantees compared to explicitly specified generative models.

Several recent articles have contributed important perspectives on graph structure learning. [1] and [2] provided early overviews of model-based learning with a focus on tools from graph signal processing (GSP), addressing primarily the problem of learning graph topology from node-level observations. [3] and [4] surveyed graph structure learning in the GML literature, where the emphasis is on task-driven rewiring of graphs, while [5] and [6] focused on deep generative models for graph generation. This review aims to bridge the various research communities through three main contributions. First, we provide a unified framework (Fig. 2) that places graph topology learning and graph generation, together with their model-based and data-driven formulations, under a common perspective, highlighting both their complementary strengths and their potential synergies. Second, the proposed framework naturally suggests new problem formulations that span traditional boundaries. For example, rather than treating graph generation and graph signal generation as separate stages, one might think about combining the $h -$ map for G and f-map for x, thus linking the graph-parameter space Θ directly to the graph-signal space $\mathcal { X } .$ Third, this unifying perspective may also lead to the transfer of methodologies across traditionally separate communities. For example, GSP concepts such as graph filtering and signal smoothness may be combined with or incorporated into the design of deep graph generative models, while advances in modern generative learning may in turn inspire new approaches to graph topology inference.

This review article is structured as follows. Section II presents the overall framework, formulates the problem instances, and discusses the two families of approaches. Subsequently, Sections III and IV focus on learning graph topology and learning graph generation, respectively, and Section V discusses their extensions to downstream tasks. Section VI bridges these problem instances, focusing in particular on new problem formulations and the transfer of methodologies across different communities. Finally, Section VII concludes with future perspectives on the problem of graph structure learning and beyond.

## II. FRAMEWORK

In this section, we present the unifying framework behind our review. We first introduce the main problem instances, together with the relevant mathematical notation (Section II-A), and then describe the families of approaches in detail (Section II-B). To fix notation, in the rest of this article, each graph, denoted by $G ,$ consists of a node set V of N nodes and an edge set $\mathcal { E }$ with edge weights represented by an adjacency matrix $\mathbf { W } \in \mathbb { R } ^ { N \times N }$ . The node observations are represented by a graph signal x : V → R.

## A. Problem instances

We first define the fundamental instances of the graph structure learning problem that are discussed in this review.

Problem 1 (Learning graph topology). Given a set of M graph signals $\{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { M } \in \mathbb { R } ^ { N }$ , the goal is to learn a graph G of N nodes whose topology is represented by an adjacency matrix $\mathbf { W } \in \mathbb { R } ^ { N \times N }$ according to a known mapping $f _ { \mathbf { X } } : { \mathcal { G } }  { \mathcal { X } }$ (or an unknown $\widehat { f } _ { \mathbf { X } } )$ from G to x.

Problem 2 (Learning graph generation). Given a set of M graphs $\{ G _ { i } \} _ { i = 1 } ^ { M }$ whose topologies are represented by adjacency matrices $\{ \mathbf { W } _ { i } \} _ { i = 1 } ^ { M } \in \mathbb { R } ^ { N \times N }$ (and each potentially with graph signals $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { D } \in$ $\mathbb { R } ^ { N } )$ , the goal is to learn a generative model, according to a known mapping h<sub>G</sub> $\mathsf { \Omega } _ { : } : \Theta \to \mathcal { G }$ (or an unknown $\widehat { h } _ { \mathbf { G } } )$ that represents a probability distribution, such that a new graph instance $G _ { i + 1 }$ with an adjacency matrix $\mathbf { W } _ { i + 1 }$ can be generated.

Although Problem 1 and Problem 2 have traditionally been studied separately by different communities, they correspond to different stages of the proposed generation process of graph-related data, a connection we will revisit in Section VI. For both problems, it is often desirable to impose topological constraints on the learned or generated graph, such as its node degree distribution. We do not treat this as a separate problem instance, but we discuss constrained graph structure learning whenever relevant. Meanwhile, the above-mentioned formulations do not explicitly consider a downstream task. When such a task is at hand, these formulations may be extended as follows:

Problem 3 (Task-driven graph structure learning). Given a set of M graph signals $\{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { M } \in \mathbb { R } ^ { N }$ (or M graphs $\{ G _ { i } \} _ { i = 1 } ^ { M } )$ and additional side information pertaining to a downstream task, the goal is to learn a graph G whose topology is represented by an adjacency matrix $\mathbf { W } \in \mathbb { R } ^ { N \times N }$ (or the parameters of a graph generative model $\theta \in \Theta )$ . In particular, the selection criterion is influenced by the downstream task which can be defined by labels associated with the given data, or an optimization objective specified by the task.

## B. Families of approaches

Next, the existing literature can be broadly categorized into the following two families of approaches:

Approach 1 (Model-based learning). Algorithms belonging to this family assume that $f _ { \mathbf { X } }$ or $h _ { \mathbf { G } }$ is already known. Early examples involving a known $f _ { \mathbf { X } }$ include 1) probabilistic graphical models (PGMs) [7], where one aims to learn the conditional independence structure, represented in the form of a graph, from observations made on random variables which are nodes in the graph, and 2) dynamical processes on graphs, such as network diffusion or disease propagation [8], [9]. More recently, there is a surge of interest from the GSP community in this problem [1], [2], where one wishes to learn the topology of the graph that supports the observed signals according to certain prior knowledge that is captured by a known model $f _ { \mathbf { X } }$ (an instance of Problem 1). For example, the smoothness assumption on x can be interpreted as $\mathbf { x }$ following a multivariate Gaussian distribution (hence a known $f _ { \mathbf { X } }$ as a stochastic map) with the inverse covariance being the corresponding graph Laplacian matrix, and the goal is to learn $G$ from $\mathbf { x } .$ Both PGM- and GSP-based approaches emphasize the importance of modeling: the likelihood of the observed data is evaluated given a candidate graph, and learning often amounts to performing a maximum-likelihood (MLE) or maximum-a-posteriori (MAP) estimation.

A complementary line of model-based learning focuses on a known $h _ { \mathbf { G } }$ for the entire graph $G ,$ , where each data point is a graph instance. For example, consider a stochastic block model (SBM) (hence a known $h _ { \mathbf { G } }$ as a stochastic map) with parameters $p$ and $q$ for the intra- and inter-community edge probabilities, respectively. The goal is to learn $p$ and $q$ from a collection of observed graphs $\{ G _ { i } \} _ { i = 1 } ^ { M }$ (an instance of Problem 2), often via an MLE approach [10], [11]; once learned, $p$ and $q$ can then be used to generate new graph instances.

Approach 2 (Data-driven learning). In practice, the mapping $f _ { \mathbf { X } }$ or $h _ { \mathbf { G } }$ may not be readily available, and may therefore need to be learned from data. We first consider an unknown $\widehat { f } _ { \mathbf { X } }$ for x. In this setting, one may rely on a prototype model with unknown parameters that determine its characteristics, or equip the prototype model with learnable components, thereby offering more flexibility. In this case, one may observe pairs of node-level observations and graph topologies, from which a mapping from G to x can be learned and later used to infer a graph from new node signals (an instance of Problem 1). As a concrete example, we can think of x being generated from a fixed but unknown graph filter (e.g., polynomial filter or diffusion kernel) applied to random noise, or from an unknown diffusion process on G. The goal is to learn the filter or diffusion process $\widehat { f } _ { \mathbf { X } }$ from observed pairs of x and G, which serve as training samples. The learned mapping can then be used to infer G given new observations x.

Similarly, we may consider an unknown mapping $\widehat { h } _ { \mathbf { G } }$ for G. In this case, the training data may consist of graph-level observations, such as graph instances collected over time. These graphs may be viewed as realizations of an unknown probabilistic model $\widehat { h } _ { \mathbf { G } }$ . The goal is to learn $\widehat { h } _ { \mathbf { G } }$ from a collection of observed $\{ G _ { i } \} _ { i = 1 } ^ { M }$ (an instance of Problem 2); once learned, $\widehat { h } _ { \mathbf { G } }$ can then be used to generate new graph instances. Representative works include deep graph generative models [5], [6].

Problems 1 and 2 correspond to the second and first columns of Fig. 2, respectively, while Approaches 1 and 2 correspond to the first and second rows, respectively. Problem 3 can be considered as an extension of both Problems 1 and 2 by incorporating downstream tasks. The following sections discuss each of these problems in greater detail.

## III. LEARNING GRAPH TOPOLOGY

## A. Learning graph topology with a known mapping f<sub>X</sub>

We first consider Problem 1 where a set of M graph signals $\{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { M } \in \mathbb { R } ^ { N }$ is observed according to a known mapping $f _ { \mathbf { X } } : { \mathcal { G } }  { \mathcal { X } }$ from G to x. Given this mapping, the goal is to learn from x a graph G of N nodes represented by an adjacency matrix $\mathbf { W } \in \mathbb { R } ^ { N \times N }$ (or an equivalent representation). We discuss several representative examples of this problem and its extensions.

Example 1 (Smooth signals on graphs). Consider a signal x that is smooth on the graph G, where smoothness is intuitively understood as neighboring nodes in the graph having similar signal values or distributions. From a probabilistic viewpoint, this is often encoded by a known mapping f<sub>X</sub> which represents a distribution of x that is characterized by a representation of G.

Smoothness is one of the most natural assumptions on the characteristics of the graph signal. Early instances of Example 1 include those characterized by the PGMs [7]. In PGMs, observations are made on a collection of random variables in the form of a matrix $\mathbf { X } \in \mathbb { R } ^ { N \times M }$ , where the rows correspond to the random variables, the existence of conditional independence among which is captured by the presence of edges in a graph representation (hence an implicit smoothness assumption). In the case of Gaussian Markov random fields (GMRFs), this leads to a joint distribution encoded by $f _ { \mathbf { X } } \mathbf { : }$

$$
f _ { \mathbf { X } } : p ( \mathbf { x } | \Theta ) = \frac { | \Theta | ^ { 1 / 2 } } { ( 2 \pi ) ^ { N / 2 } } \mathrm { e x p } \big ( - \frac { 1 } { 2 } \mathbf { x } ^ { T } \Theta \mathbf { x } \big ) ,\tag{1}
$$

where $\Theta \in ^ { N \times N }$ is the inverse covariance or precision matrix. In this case, learning the graph G amounts to learning $\mathbf { \Theta } ^ { \Theta , }$ , often via an MLE approach. For example, the graphical lasso method proposes to solve the following problem [12]:

$$
\operatorname* { m a x } _ { \Theta } \mathrm { ~ l o g d e t } ( \Theta ) - \mathrm { t r } ( \hat { \Sigma } \Theta ) - \rho | | \Theta | | _ { 1 } ,\tag{2}
$$

where $\begin{array} { r } { \hat { \mathbf { \Sigma } } = \frac { 1 } { M - 1 } \mathbf { X } \mathbf { X } ^ { T } } \end{array}$ is the empirical covariance, logdet(·) and $\operatorname { t r } ( \cdot )$ represent the log determinant and trace operators, respectively, and $\rho$ is a regularization parameter.

More recently, the field of GSP has made advances in solving Example 1 by making an explicit assumption on signal smoothness. Consider the graph Laplacian matrix and its eigendecomposition, i.e., $\mathbf { L } = \mathbf { U } \mathbf { A } \mathbf { U } ^ { T }$ . Under a factor analysis model of a graph signal x (columns of X above), i.e., $\mathbf { x } = \mathbf { U } \mathbf { h } + \mathbf { n }$ where h $\sim \mathcal { N } ( \mathbf { 0 } , \mathbf { \Lambda } \mathbf { \Lambda } ^ { \dag } )$ represents a coefficient vector under the eigenvector basis U and $\mathbf { n } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { N } ^ { 2 } \mathbf { I } )$ is additive Gaussian noise, the distribution of x follows [13]:

$$
f _ { \mathbf { X } } : p ( \mathbf { x } | \mathbf { L } ) \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { L } ^ { \dag } + \sigma _ { N } ^ { 2 } \mathbf { I } ) ,\tag{3}
$$

where $\dagger$ represents the Moore-Penrose pseudo-inverse. Based on this, [13] proposes to solve for L (equivalently the graph adjacency matrix) via an optimization problem whose objective includes a trace term of the so-called Laplacian quadratic form, i.e., $\operatorname { t r } ( \mathbf { X } ^ { T } \mathbf { L } \mathbf { X } )$ , which explicitly penalizes variation of signal values across connected nodes, hence promoting smoothness on the learned graph. Other approaches based on a similar objective include [14], [15], [16]. Going beyond the Gaussian assumption, recent work has also considered learning from signals corrupted with exponential family noise [17].

We note that there is a clear link between the PGM and GSP approaches to Example 1; indeed, if we restrict the precision matrix Θ to be a symmetric and positive definite matrix with non-positive off diagonal entries, of which the graph Laplacian L is an example, the ways in which signal smoothness is being promoted in the two approaches are equivalent (i.e., through the minimization of the Laplacian quadratic form). More recent work has extended this notion of smoothness by taking into account additional node attributes that, together with the graph topology, influence the observed graph signals. In [18], this leads to the following distribution of x:

$$
f _ { \mathbf { X } } : \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { L } ^ { \dagger } \otimes \mathbf { K } + \sigma _ { N } ^ { 2 } \mathbf { I } ) ,\tag{4}
$$

where K is a chosen kernel matrix defined over pairs of node attributes, and $\otimes$ represents the Kronecker product. This in turn leads to an optimization problem over L with a regularization term of $\mathbf { x } ^ { T } ( \mathbf { L } \otimes \mathbf { K } ^ { \dagger } ) \mathbf { x }$

Interestingly, if one associates the kernel matrix with another graph Laplacian $\mathbf { L } _ { \mathbf { K } } = \mathbf { K } ^ { \dagger }$ and defines the Kronecker product ${ \bf L } _ { \otimes } = { \bf L } \otimes { \bf L } _ { \bf K }$ as a Laplacian-like operator, then this term becomes ${ \bf x } ^ { T } { \bf L } _ { \otimes } { \bf x } ,$ which can be interpreted as a modified Laplacian quadratic form that promotes signal smoothness on the learned graph (represented by L) and in the chosen kernel space (represented by K) simultaneously. On the other hand, a few approaches have made the smoothness assumption not on the graph signal itself, but a constituent of it. For example, [19] posits that the signal is a combination of a smooth component and a remainder explained by non-graph-related regressors; [20] and [21] both consider signals that can be expressed as a linear combination of building block patterns, which are then assumed to be smooth on the learned graph.

The additional constraint implicitly imposed in the kernel space of node attributes makes the approach in [18] robust against randomly missing entries in the node observations, a scenario also considered in [22] and [23]. In these approaches, a mask matrix that indicates the missing entries is assumed known, and an optimization problem with a similar smoothness-based objective as above is solved. A similar imperfect setting is considered in [24] and [25], where observations are only made on a subset of nodes in the graph and a subgraph among those is learned. In [24], this is achieved by decomposing the Laplacian quadratic form into three terms that correspond to the observed nodes, unobserved nodes, and interactions between them, which are learned in a joint optimization problem.

In parallel, an increasingly rich literature considers different forms of learning output. In many cases, the smoothness measure is kept in the objective of the optimization problem, with additional terms constraining the output form. One output form of particular interest is the simultaneous learning of multiple graphs and, as a typical example, time-varying graphs. The early works of [26] and [27] generalize the classical graphical lasso and GSP-based topology learning, respectively, to the time-varying setting, where the learned graphs at consecutive timestamps are constrained to be similar. While [27] enforces this via the Frobenius norm of differences in the weighted adjacency matrices, [28] further constrains the number of edges modified across consecutive timestamps to be small via an $\ell _ { 1 } { \mathrm { - n o r m } }$ based regularization. Various extensions have since been proposed to deal with online streaming data [29], partial node observations [30], [31], and the presence of hidden nodes [32]. Going beyond time-varying graphs, [33] and [34] generalize the graphical lasso and GSP approaches, respectively, in learning multiple graphs, while [35] proposes to learn a multi-view graph together with a consensus topology, given node observations from each view. Similarly, [36] associates each view with a class label in a classification task, therefore learning class-specific adjacency matrices.

As a final remark, we can see above that algorithmically the smoothness criterion is often incorporated into an optimization problem, which can then be solved by for example the block-coordinate descent (BCD) or alternating direction method of multipliers (ADMM) algorithms. In practice, it may be challenging to tune the hyperparameters in these formulations, such as $\rho$ in Eq. (2). To address this, [37] proposes to unroll an algorithm that solves a reformulated version of Eq. (2), where $\rho$ is modeled as a problem-dependent neural network. Similarly, [38] proposes to unroll an algorithm designed to solve a GSP-based formulation, and [39] further considers a Bayesian extension that allows for the estimation of uncertainty of learned edges. All these can thus be considered as a form of algorithm learning, although they are still based on prototypes of formulations that explicitly promote signal smoothness.

Example 2 (Dynamical processes on graphs). Consider a signal that is the outcome of a physical dynamical process on the graph, such as graph-based diffusion. This process is encoded by a known mapping f<sub>X</sub> which maps an initial state of the signal and a graph G to observed signals x.

Such a process naturally encodes the temporal dynamics behind the observed signals on the graph and, as a consequence, the observations often come with a temporal dimension. Early instances of graph diffusion can be found in epidemiology, where a disease is assumed to have been transmitted over a contact network among a group of individuals, having originated from a random individual in the network. The transmission probability from node i to node $j ,$ given that i is already infected, is governed by a mapping $f _ { \mathbf { X } }$ , which is often a function of the graph topology. For example, in [8], this probability is defined directly using the edge weight $w _ { i j }$

$$
f _ { \mathbf { X } } : p \big ( x _ { j } \big | x _ { i } , w _ { i j } \big ) = h \big ( x _ { j } - x _ { i } \big ) w _ { i j } ,\tag{5}
$$

where $x _ { i }$ and $x _ { j }$ are the infection times of nodes i and $j ,$ respectively, with $x _ { i } ~ < ~ x _ { j }$ , and $h ( t )$ is a transmission time model that is assumed known. Given only observations on the node states (such as $x _ { i }$ for all nodes), the goal is to learn the topology of the unknown contact network G. In the case of [8], learning $G$ amounts to learning the edge weights $w _ { i j }$ as the conditional probabilities of disease transmission, which can be done via an MLE approach. A similar approach based on a binary graph topology and a uniform transmission probability $w _ { i j }$ can be found in [9].

Going beyond infectious disease models, the vector autoregressive model (VARM) and its variants have been used to study temporal dynamics on graphs, which introduce both temporal and spatial dependencies in the data. For example, in [40], the signal at time step $t , \mathbf { x } [ t ]$ , is represented as a linear combination of transformed observations in the past K time steps and a random noise process $\mathbf { n } [ t ]$

$$
f _ { \mathbf { X } } : \mathbf { x } [ t ] = \sum _ { k = 1 } ^ { K } p _ { k } ( \mathbf { W } ) \mathbf { x } [ t - k ] + \mathbf { n } [ t ] ,\tag{6}
$$

where $p _ { k } ( \mathbf { W } )$ is a polynomial in the adjacency matrix W. Given temporal observations x[t], [40] proposes to solve an optimization problem over W by minimizing the approximation error according to Eq. (6), together with a sparsity constraint on both W and the polynomial coefficients. Alternatively, the same signal model can be posed as a Gaussian process with the choice of the kernel function encoding the spatiotemporal dependencies [41]. A VARM can also be interpreted as a state-based model (similar to a Kalman filter). For example, [42] posits the following signal model:

$$
f _ { \mathbf { X } } : \mathbf { x } [ t ] = \mathbf { U } \mathbf { h } [ t ] + \mathbf { n } [ t ] , \ \mathbf { h } [ t ] = \mathbf { A } \mathbf { h } [ t - 1 ] + \mathbf { v } [ t ] ,\tag{7}
$$

where U is chosen to be the eigenvector matrix of L (hence it can be interpreted as a graph Fourier basis as in [13]), and h is a latent state variable governed by the state transition matrix A. It is further assumed that ${ \bf A } = { \bf U } ^ { T } { \bf R } { \bf U }$ , where R is a diagonal matrix capturing correlation in time, and $\mathbf { n } [ t ] \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { N } ^ { 2 } \mathbf { I } )$ and $\mathbf { v } [ t ] \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { \boldsymbol { \Lambda } } ^ { \dagger } )$ , where Λ is the eigenvalue matrix of L. Under these assumptions, the temporal difference signal admits the following distribution:

$$
p ( \mathbf { d } | \mathbf { L } ) \sim \mathcal { N } \Big ( \mathbf { 0 } , \big ( \mathbf { L } ^ { \dag } + \sigma _ { N } ^ { 2 } ( \mathbf { I } + \mathbf { R } \mathbf { R } ^ { T } ) \big ) \Big ) .\tag{8}
$$

Therefore, it can be seen that the difference signal d is smooth on the graph, and the model in Eq. (8) encodes a notion of spatiotemporal smoothness governed by A, which introduces a space-time coupling. Based on this, [42] formulates an optimization problem to learn L and R jointly. A similar model, based on the graphical lasso formulation and also considering partial node observations, can be found in [43]. More generally, the dynamical process can incorporate additional influence from instantaneous effects, by combining a VARM with a structural equation model (SEM) into a structural vector autoregressive model (SVARM) [44], or by extending the state-based interpretation above into a state-space model [45].

Apart from those based on the graph Laplacian L, approaches discussed above can generally be used to learn a graph with directed edges (i.e., an asymmetric adjacency matrix W). This ability to learn directed graphs is an important difference from the approaches to Example 1. Of particular interest are directed acyclic graphs (DAGs), which come with an interpretation of causal relations between the variables that are represented by the nodes in the graph. The reader is referred to [46] for a historical overview of DAG learning. The literature mostly adopts the linear SEM (while the discussion here focuses on learning DAGs, the SEM has also been used in learning generic directed graphs [47]):

$$
f _ { \mathbf { X } } : \mathbf { x } = \mathbf { W } ^ { T } \mathbf { x } + \mathbf { n } ,\tag{9}
$$

where W represents the causal dependencies between the nodes (i.e., the $i j \cdot$ -th entry of W, $w _ { i j }$ , is nonzero if there is a causal link from i to j), and n is observational noise. Under this model, existing approaches mainly differ in how the acyclicity constraint on W is being enforced. While traditional approaches mostly rely on a combinatorial constraint, NOTEARS [48] proposes to enforce acyclicity via the constraint $h ( \mathbf { W } ) = \mathrm { t r } ( e ^ { \mathbf { W } \circ \mathbf { W } } ) - n = 0$ , where ◦ represents the Hadamard product and $e ^ { ( \cdot ) }$ the matrix exponential, thus turning a discrete optimization into a continuous equality-constrained program which is then solved via the augmented Lagrangian method. Various extensions have been proposed since then: [49] and [50] extend [48] to deal with time-series data, [51] analyzes the role of sparsity in W, [52] further considers sparsity in n which is interpreted as having a few root causes of the observations, and [53] proposes a joint estimation of W and the unknown noise level in n.

More recently, the literature has studied an interesting example of network processes often encountered in social and economic sciences, i.e., games played on networks [54]. These are strategic decision-making processes where each individual in the network takes an action to maximize their own payoff, which in turn depends on the actions of their neighbors. For example, in a network game with linear-quadratic payoffs [55], an individual i chooses their action $x _ { i }$ to maximize their payoff $u _ { i } \mathrm { : }$

$$
u _ { i } = b _ { i } x _ { i } - \frac { 1 } { 2 } x _ { i } ^ { 2 } + \beta x _ { i } \sum _ { j \sim i } w _ { i j } x _ { j } ,\tag{10}
$$

where $b _ { i }$ is the marginal benefit i receives by taking the action $x _ { i }$ . It is shown in [55] that given certain assumptions, this game admits the following unique and stable Nash equilibrium:

$$
f _ { \mathbf { X } } : \mathbf { x } = \left( \mathbf { I } - \beta \mathbf { W } \right) ^ { - 1 } \mathbf { b } ,\tag{11}
$$

where $\beta$ is a parameter that determines the nature of the strategic interaction. [56] exploits this relationship to design an optimization problem that learns an unknown W and parameter $\beta$ from observed equilibrium actions x. Another example similar to network games is the consensus process studied in [57].

As a remark, many of the dynamical processes above can be summarized in the following generic form proposed in [58]:

$$
f _ { \mathbf { X } } : \dot { x } _ { i } ( t ) = g _ { i } \big ( x _ { i } ( t ) \big ) + \displaystyle \sum _ { j \sim i } w _ { i j } h \big ( x _ { i } ( t ) , x _ { j } ( t ) ; \pmb { \theta } _ { i } \big ) ,\tag{12}
$$

where $x _ { i } ( t )$ and ${ \dot { x } } _ { i } ( t )$ are the time-series observation at node i and its time derivative at time t, respectively, $g _ { i }$ is a self-update function on i, and h (with its parameters $\pmb \theta _ { i } )$ governs the potentially nonlinear dynamical process on the graph and its impact on i. As examples, [58] specializes this to gene dynamics and opinion dynamics based on two different choices of h. To learn the edge weights $w _ { i j }$ , it assumes the existence of stationary points of the process above, and designs an inverse problem that optimizes over $w _ { i j }$ and the parameters $\theta _ { i }$ given perturbed stationary points.

Example 3 (Filtering processes on graphs). Consider a signal that is the outcome of a filtering process on the graph. The filter characteristic is encoded by a known mapping $f _ { \mathbf { X } }$ which maps an initial graph signal $\mathbf { x } _ { \mathrm { 0 } }$ and a graph G to observed signals x.

The smoothness-based model and some of the dynamical processes discussed above, e.g., Eq. (6), can be unified under the language of GSP via a filtering process. In this case, the observed signals x can be written as:

$$
f _ { \mathbf { X } } : \mathbf { x } = g ( \mathbf { S } ) \mathbf { x } _ { 0 } = \mathbf { U } g ( \pmb { \Lambda } ) \mathbf { U } ^ { T } \mathbf { x } _ { 0 } ,\tag{13}
$$

where $\mathbf { S } = \mathbf { U } \pmb { \Lambda } \mathbf { U } ^ { T }$ is a (symmetric) graph shift operator representing the topology of graph $G ,$ and $g ( \cdot )$ defines the characteristics of the filter via the interpretation of ${ \bf U } ^ { T } { \bf x } _ { 0 }$ being a Fourier-like transform on the graph. This signal model is more general and flexible than the ones above: indeed, if S is chosen to be the graph Laplacian L and $g ( \cdot )$ a low-pass filter [59], this corresponds to the smoothness-based model in Eq. (3); alternatively, $g ( \cdot )$ may act as a frequency-selective filter that only allows specific graph frequencies to pass, resulting in a band-limited (but not necessarily smooth) signal [60]. On the other hand, the VARM can also be interpreted as graph filtering with S being the adjacency matrix.

Broadly speaking, two main approaches have been deployed in the context of graph topology learning. The first posits that the graph filter corresponds to a diffusion-like process on the graph, such as that in [40] discussed above. Other notable processes studied in the literature include heat diffusion in [61] and the consensus dynamics in [57]. For example, in [61], $g ( \cdot )$ represents the heat kernel based on the choice of S being the graph Laplacian L:

$$
f _ { \mathbf { X } } : \mathbf { x } = g ( \mathbf { L } ) \mathbf { x } _ { 0 } = e ^ { - \tau \mathbf { L } } \mathbf { x } _ { 0 } ,\tag{14}
$$

where $\tau$ is a parameter that controls the speed of diffusion. [61] further assumes that the signal is a linear combination of the outcomes of a small number of heat diffusion processes (hence similar in spirit to [52]), which may have originated from different nodes with different parameters τ. A dictionary-learning algorithm is then applied to solve for L given x.

In the second approach, $g ( \cdot )$ represents a generic degree-K polynomial graph filter:

$$
f _ { \mathbf { X } } : \mathbf { x } = g ( \mathbf { S } ) \mathbf { x } _ { 0 } = \sum _ { k = 0 } ^ { K } g _ { k } \mathbf { S } ^ { k } \mathbf { x } _ { 0 } ,\tag{15}
$$

where $\mathbf { S }$ can be chosen as either the adjacency matrix [62] or graph Laplacian [63], [57]. Under the assumption of a symmetric S and that the initial signal $\mathbf { x } _ { \mathrm { 0 } }$ is i.i.d. Gaussian, this has the interpretation of a stationary process on the graph [64]. In this case, the covariance matrix $\mathbf { C } = \mathbb { E } ( \mathbf { x } \mathbf { x } ^ { T } )$ commutes with S; hence, the two share the same set of eigenvectors. Once the eigenvectors of S, which are termed spectral templates in [62], are obtained from C, the eigenvalues are found via an optimization problem, often together with a sparsity constraint on S.

Similar to Example 1 and Example 2, the standard approaches above have been extended to a variety of settings. In terms of the input information, [24] considers the partial node observation setting with the presence of hidden nodes. Regarding the form of output, [65] and [66] consider the task of learning product graphs. For example, in [66], the graph signal admits the following distribution:

$$
f _ { \mathbf { X } } : \mathbf { x } = [ \sum _ { p = 0 } ^ { K _ { P } } \sum _ { q = 0 } ^ { K _ { Q } } g _ { p q } ( \mathbf { W } _ { P } ) ^ { p } \otimes ( \mathbf { W } _ { Q } ) ^ { q } ] \mathbf { x } _ { 0 } + \mathbf { n } ,\tag{16}
$$

where $\mathbf { W } _ { P }$ and ${ \bf W } _ { Q }$ are the adjacency matrices for the graphs $G _ { P }$ and $G _ { Q }$ that form the product graph, $\{ g _ { p q } \}$ are the filter coefficients, and n is the noise. The approach based on spectral templates can then be applied to learn the graph topologies of $G _ { P }$ and $G _ { Q }$ separately, before they are combined to reconstruct G. Another example is [67], which learns bipartite graphs using the same approach. Finally, there are also efforts that aim to deal with the time-varying setting [68] and multi-graph setting [69], [70].

## B. Learning graph topology with an unknown mapping $\widehat { f } _ { \mathbf { X } }$

We now consider the data-driven counterpart of Section III-A in solving Problem 1. In this case, the mapping from the graph G to the observed signals x is either completely unknown, or follows a prototype mapping but with unknown parameters that determine its precise characteristics. The goal is to learn from x the graph G by learning the unknown mapping $\widehat { f } _ { \mathbf { X } }$ , via training examples and a criterion on the quality of the learned mapping. There are two key considerations in this case: 1) how the relation between G and x is relaxed from a known to an unknown mapping, and 2) how this relaxation impacts the corresponding learning strategy. In what follows, we discuss how the various approaches differ in these two aspects, by focusing on the extension of two examples in Section III-A.

Example 4 (Learning unknown filtering processes). Consider a signal that is the outcome of a filtering process on the graph. The filter characteristic is encoded by an unknown mapping $\widehat { f } _ { \mathbf { X } }$ which maps an initial graph signal $\mathbf { x } _ { \mathrm { 0 } }$ and a graph G to observed signals x.

We begin with Example 4 as it provides a smoother transition in terms of the methodologies covered in this section. In the graph filtering formulation, the signal model generally follows that in Eq. (15), where $\left\{ g _ { k } \right\}$ are the filter coefficients. Note that most approaches discussed after Eq. (15) assume a stationary graph process, in which case the eigenvectors of the graph shift operator S can be obtained directly from that of the covariance of the observed signals, and the learning of S can be achieved without the knowledge of $\left\{ g _ { k } \right\}$ . When this assumption does not hold for generic graph signals, however, learning $\left\{ g _ { k } \right\}$ , i.e., what constitutes the unknown mapping $\widehat { f } _ { \mathbf { X } }$ , often becomes necessary, either explicitly or implicitly via learning $g ( \mathbf { S } )$ directly.

We first discuss an interesting case that bridges the graphical lasso and GSP formulations. Recall the GMRF in Eq. (1), where the support of the precision matrix Θ indicates the presence of conditional dependence between the random variables represented by the nodes in the graph. From this perspective, learning Θ can be interpreted as learning a graph, and in the case of Θ being the graph Laplacian it leads to the smoothness model in Eq. (3). In [71], this specific case is relaxed such that $\begin{array} { r } { \mathbf { \Theta } \mathbf { \Theta } \Theta = g ( \mathbf { S } ) = \sum _ { k = 0 } ^ { K } g _ { k } \mathbf { S } ^ { k } } \end{array}$ where $g ( \cdot )$ is a generic polynomial. In other words, x still follows the same distribution as in Eq. (1), but its interpretation now depends on the characteristics of $g ( \cdot )$ : if it is no longer a monotonically increasing function of the eigenvalue of $\mathbf { S } ,$ the model will not necessarily promote signal smoothness but capture a more flexible behavior. Given that Θ and S commute, [71] proposes to solve the following joint optimization problem:

$$
\operatorname* { m a x } _ { \boldsymbol { \Theta } , \boldsymbol { \mathrm { S } } } ~ \log \operatorname* { d e t } ( \boldsymbol { \Theta } ) - \operatorname { t r } ( \hat { \boldsymbol { \Sigma } } \boldsymbol { \Theta } ) - \rho | | \boldsymbol { \Theta } | | _ { 0 } , ~ \mathrm { s . t . } ~ \boldsymbol { \Theta } \boldsymbol { \mathrm { S } } = \boldsymbol { \mathrm { S } } \boldsymbol { \Theta } ,\tag{17}
$$

where $\left\{ g _ { k } \right\}$ are implicitly learned via Θ. A similar approach is proposed in [72], where $\Theta = h _ { \theta } ( \mathbf { L } ) ^ { \dagger }$ follows a number of specific parametric forms, and hence the associated parameter θ is explicitly learned together with the graph Laplacian L.

Moving on to the GSP formulation of Eq. (15), where we write $\mathbf { G } = g ( \mathbf { S } )$ in short, [73] proposes a two-step approach to joint graph topology learning and filter identification. First, given that $\textbf { x } =$ $\mathbf { G x } _ { 0 }$ , we have $\pmb { \Sigma } _ { \mathbf { x } } = \mathbf { G } \pmb { \Sigma } _ { \mathbf { x } _ { 0 } } \mathbf { G } ^ { T }$ , where $\pmb { \Sigma } _ { \mathbf { x } _ { 0 } }$ and $\pmb { \Sigma } _ { \mathbf { x } }$ are the true input and output covariance matrices, respectively. Assuming the former is known, G can be estimated given the empirical output covariance $\begin{array} { r } { \hat { \mathbf { \Sigma } } _ { \mathbf { x } } = \frac { 1 } { m - 1 } \mathbf { X } \mathbf { X } ^ { T } } \end{array}$ by solving a manifold-constrained least-squares problem. Second, once the estimated $\hat { \mathbf { G } }$ is obtained, S can be learned by finding a sparse matrix that commutes with $\hat { \mathbf { G } }$ . This approach based on the constraint on commutativity also helps avoid the need for eigendecomposition and the assumption of a symmetric S in [62] and [63]. In practice, if the true input covariance is not known, one could instead resort to the knowledge of input-output pairs of an unknown filtering system. For example, [74] proposes to solve the following joint optimization problem:

$$
\operatorname* { m i n } _ { \mathbf { S } , \mathbf { g } } \ \vert \vert \mathbf { X } - \sum _ { k = 0 } ^ { K } g _ { k } \mathbf { S } ^ { k } \mathbf { X } _ { 0 } \vert \vert \ \mathrm { ~ s . t . ~ } \ \mathrm { s u p p } ( \mathbf { S } ) \subseteq \mathcal { S } ,\tag{18}
$$

where $\mathbf { X } _ { 0 }$ and X are the input and output signals in matrix format, respectively, g is a vector of filter coefficients $\left\{ g _ { k } \right\}$ , and $s$ is the set with the support of S that is assumed known. A similar approach can be found in [75] which, instead of learning the filter coefficients explicitly, proposes to learn the filter matrix $\mathbf { G } = g ( \mathbf { S } )$ with a commutativity constraint as in [73].

Most approaches discussed in this section and Section III-A follow a deterministic setting, where only a point estimate of the graph topology is obtained. To address this, [76] proposes a Bayesian approach and considers the following MAP estimation:

$$
\operatorname* { m a x } _ { \mathbf { W } , \theta } \ p ( \mathbf { W } , \theta | \mathbf { X } _ { 0 } , \mathbf { X } ) { \mathrm { o r ~ e q u i v a l e n t l y } } \ \operatorname* { m a x } _ { \mathbf { W } , \theta } \ p ( \mathbf { X } | \mathbf { W } , \theta , \mathbf { X } _ { 0 } ) p ( \mathbf { W } )\tag{19}
$$

based on the assumptions on the input $\mathbf { X } _ { 0 }$ and filter coefficients $\pmb \theta .$ In Eq. (19), the likelihood is governed by the unknown graph filtering process, while the prior represents any knowledge on the graph topology. Given that expressions for both may not be tractable in general, [76] proposes drawing a sample from the posterior distribution to approximate the MAP estimate. The algorithm proceeds iteratively: given an estimate of θ, W is updated via the application of annealed Langevin dynamics; θ is then updated via gradient descent, and the process repeats. Based on the filtering process in Eq. (15), the likelihood is computed using input-output signal pairs together with estimates of W and θ (the latter of which is equivalent to $\{ g _ { k } \} )$ . The prior is approximated via a graph neural network (GNN), which is trained using a set of ground truth adjacency matrices.

Example 5 (Learning unknown dynamics). Consider a signal that is the outcome of a dynamical process on the graph. This process is encoded by an unknown mapping $\widehat { f } _ { \mathbf { X } }$ which maps an initial state of the signal and a graph G to observed signals x.

We now move on to unknown dynamical processes on graphs. Although many of these processes can be interpreted as graph-based filtering, the methodologies developed in the literature cover a broader range of techniques and increasingly favor a neural network based design, as we shall see in this section.

As discussed in Section III-A, the classical SEM in Eq. (9) has been widely used to capture (causal) relations between nodes in a DAG. However, it is an inherently linear model and therefore might be restrictive when nonlinear relations are present. The literature has taken a number of approaches to address this. In [77], the following nonlinear form is considered:

$$
{ \widehat { f } } _ { \mathbf { X } } : x _ { j } = f _ { j } ( \mathbf { x } ) + n _ { j } ,\tag{20}
$$

where $f _ { j }$ is a generic nonlinear function that governs node $j ^ { \circ } \mathbf { s }$ observation $x _ { j }$ , and $n _ { j }$ is the noise. [77] chooses $f _ { j }$ to be a multilayer perceptron (MLP) with L hidden layers: $\mathbf { M L P } ( \mathbf { x } ; \mathbf { A } _ { j } ^ { ( 1 ) } , \cdot \cdot \cdot , \mathbf { A } _ { j } ^ { ( L ) } ) =$ $\sigma \Big ( \mathbf { A } _ { j } ^ { ( L ) } \sigma \big ( \cdot \cdot \cdot \sigma ( \mathbf { A } _ { j } ^ { ( 1 ) } \mathbf { x } ) \big ) \Big )$ , where $\mathbf { A } _ { j } ^ { ( \ell ) } \in \mathbb { R } ^ { m _ { \ell } \times m _ { \ell - 1 } } , m _ { 0 } = N$ , and $\sigma$ is a nonlinear activation function. Under this setting, one can see that $x _ { j }$ is independent of $x _ { i }$ if the i-th column of ${ \bf A } _ { j } ^ { ( 1 ) }$ is all zero. This motivates the definition of the edge weights $w _ { i j } = | | i \ t h . . . \mathrm { c o l u m n } ( \mathbf { A } _ { j } ^ { ( 1 ) } ) | | _ { 2 }$ in a graph of functional dependence. Sparsity and the same DAG constraint as in [48] can then be imposed on its adjacency matrix

W. Based on this, [77] proposes to solve an optimization problem with respect to ${ \bf A } _ { j } ^ { ( \ell ) }$ by minimizing a reconstruction loss, hence recovering a DAG underlying the data observations.

An alternative way of incorporating nonlinear relations into the SEM is proposed in [78]. Eq. (9) can be rewritten in matrix form as $\mathbf { X } = ( \mathbf { I } - \mathbf { W } ^ { T } ) ^ { - 1 } \mathbf { N }$ , motivating the following nonlinear extension of the SEM:

$$
\widehat { f } _ { \mathbf { X } } : \mathbf { X } = f _ { 2 } \big [ ( \mathbf { I } - \mathbf { W } ^ { T } ) ^ { - 1 } f _ { 1 } ( \mathbf { N } ) \big ] ,\tag{21}
$$

where $f _ { 1 }$ and $f _ { 2 }$ are generic functions. In [78], $f _ { 1 }$ is chosen to be an identity mapping and $f _ { 2 }$ a 2-layer MLP: ML $\begin{array} { r } { \mathrm { P } \big ( ( \mathbf { I } - \mathbf { W } ^ { T } ) ^ { - 1 } \mathbf { N } , \mathbf { A } ^ { ( 1 ) } , \mathbf { A } ^ { ( 2 ) } \big ) = \sigma \big ( \big [ ( \mathbf { I } - \mathbf { W } ^ { T } ) ^ { - 1 } \mathbf { N } \big ] \mathbf { A } ^ { ( 1 ) } \big ) \mathbf { A } _ { j } ^ { ( 2 ) } } \end{array}$ , hence enabling the model to capture nonlinear relations between the observations X and the adjacency matrix W. To learn W, [78] proposes an approach similar to that of variational auto-encoders (VAEs) [79], where an encoder (another MLP followed by the linear transformation of $\mathbf { I } - \mathbf { W } ^ { T } )$ encodes the input X into the latent variable N, which is then passed through the decoder in Eq. (21). Unlike an implicit graph of functional dependence as in [77], [78] solves for an explicit DAG with adjacency matrix W, by minimizing the evidence lower bound (ELBO) subject to the DAG constraint as in [48].

The encoder-decoder architecture in [78] is a powerful framework for generative modeling, which we will discuss in more depth in Section IV. Here we discuss another popular application of this framework in learning a data-driven graph topology from node observations. The main modeling assumption is that the temporal evolution of node observations, e.g., trajectories of interacting objects represented by x[t] for $t = 1 , \cdots , T$ and X in matrix form, is governed by a latent interaction graph. The goal is to recover this latent graph for explaining past and predicting future trajectories. A representative framework is [80], in which an encoder takes X as input and returns a factorized distribution of $w _ { i j }$ that represents the edge types between node pairs i and j. A decoder then predicts X given the latent graph W:

$$
\widehat { f } _ { \mathbf { X } } : p _ { \theta } ( \mathbf { X } | \mathbf { W } ) = \prod _ { t = 1 } ^ { T } p _ { \theta } ( \mathbf { x } [ t + 1 ] | \mathbf { x } [ t ] , \cdots , \mathbf { x } [ 1 ] , \mathbf { W } ) .\tag{22}
$$

Under a Markovian assumption, $p _ { \pmb { \theta } } ( \mathbf { x } [ t + 1 ] \vert \mathbf { x } [ t ] , \cdots , \mathbf { x } [ 1 ] , \mathbf { W } ) = p _ { \pmb { \theta } } ( \mathbf { x } [ t + 1 ] \vert \mathbf { x } [ t ] , \mathbf { W } )$ and [80] proposes to model it via a GNN. Taking a similar variational approach as in [78], the parameters associated with the encoder and decoder are trained by maximizing the ELBO. Building upon [80], [81] further tackles the challenges of amortized inference, i.e., learning potentially different DAGs for different time-series data at the same time, under the assumption of a shared underlying dynamical process. These examples demonstrate the significant flexibility offered by the encoder-decoder framework, which does not need to rely on a prototype dynamical or filtering process such as the SEM.

In the examples discussed above, training is often facilitated via a loss function measuring how the learned graph helps reconstruct the observed data. Alternatively, in scenarios where ground truth graphs are available, they can be used together with the node observations to learn a mapping $\widehat { f } _ { \mathbf { X } }$ directly. For example, [82] considers a graph signal x as the outcome of an intervention performed on a certain node in a causal graph. Datasets collected from such interventions are combined with the ground truth causal graph to train a transformer-like architecture; once trained, the model can be used to reconstruct causal graphs from new interventional datasets. In a similar attempt, [83] considers node observations as equilibrium actions of a network game, whose utility function is, unlike in [56], not known a priori. This leads to a more general characterization of equilibrium actions satisfying:

$$
\widehat { f } _ { \mathbf { X } } : \mathbf { x } = \mathcal { F } ( \mathbf { W } ) \mathcal { H } ( \mathbf { b } ) ,\tag{23}
$$

where b is the marginal benefit as in Eq. (11). Based on the relationship in Eq. (23), [83] uses pairs of equilibrium actions x and networks W to train a transformer-like architecture, which implicitly captures the underlying utility function of the game. Once trained, the model can be used to reconstruct the network given new equilibrium actions. These two examples again demonstrate the flexibility of learning unknown network dynamics given the information at hand.

A high-level illustration of the methodologies for learning graph topology can be found in Fig. 3.

## C. Incorporating domain knowledge in topology learning

To conclude this section, we discuss the role of domain knowledge. As in many optimization or learning problems, prior or domain knowledge can provide useful inductive biases that guide the search toward desirable solutions, and the problem of graph topology learning is no exception. Various approaches have attempted to incorporate topological priors or constraints into the learning process. One of the most typical constraints is sparsity, which posits that the learned graph topology contains a relatively small number of significant edges. For example, this is promoted via the $\ell _ { 1 } { \mathrm { - n o r m } }$ of the precision matrix in Eq. (2) of the graphical lasso formulation, a combination of a logarithmic term on the node degrees and Frobenius norm of the adjacency matrix in [14], and explicitly specifying the number of desired edges in [16] and [84]. Another topological constraint that has attracted increasing interest is the product-graph structure. For example, [85], [86] consider a graph that can be factorized as the Cartesian product of two graphs, i.e., $G = G _ { P } \oplus G _ { Q }$ , which implies $\mathbf { L } = \mathbf { L } _ { P } \oplus \mathbf { L } _ { Q } = ( \mathbf { U } _ { P } \mathbf { A } _ { P } \mathbf { U } _ { P } ^ { T } ) \oplus ( \mathbf { U } _ { Q } \mathbf { A } _ { Q } \mathbf { U } _ { Q } ^ { T } )$ , where ⊕ represents the Kronecker sum. From this, the same factor analysis model as in [13] is adopted, where U is replaced with $\mathbf { U } _ { P } \otimes \mathbf { U } _ { Q }$ . This leads to the following distribution of x:

$$
f _ { \mathbf { X } } : p ( \mathbf { x } | \mathbf { L } ) \sim \mathcal { N } \big ( \mathbf { 0 } , ( \mathbf { L } _ { P } \oplus \mathbf { L } _ { Q } ) ^ { \dag } + \sigma _ { N } ^ { 2 } \mathbf { I } \big ) .\tag{24}
$$

Based on this, [85] proposes to jointly solve for $\mathbf { L } _ { P }$ and $\mathbf { L } _ { Q }$ promoting smoothness on $G _ { P }$ and $G _ { Q }$ respectively, with additional sparsity and rank constraints. In a similar effort, [87] further extends the setting above to learning the Kronecker product of two graphs. Extending beyond sparse and product graphs, [88] proposes to learn a k-component graph by enforcing the first k eigenvalues of the learned graph Laplacian to be zero, [89] enforces the learned graph to have a similar density of motifs compared to a given reference graph, and [48] learns a DAG by introducing a novel acyclicity constraint.

![](images/1ea4701ca0f4c6584d8dc5c621f30cd408361503058c22d7a91641a748d15e15.jpg)  
Fig. 3: An illustration of methodologies for learning graph topology.

The consideration of topological constraints above is motivated by both their prevalence in realworld networks, and the ease with which they can be explicitly encoded in closed forms amenable to optimization. With the mathematical ease, however, comes the restrictive set of constraints that may be considered. Important characterizations in network science, such as the distribution of node degrees or local clustering coefficients, are more difficult to encode and hence have been less studied. To address this, recent literature has started to investigate implicit ways to incorporate topological constraints. For example, in [38], the learned graph is encouraged to be similar to a template graph in a latent space constructed by a VAE, hence implicitly capturing topological properties of the template graph. Similarly, in [76], using a collection of ground truth graphs, a GNN is trained to output a score function that approximates the gradient of the log prior on the vectorized adjacency matrix, hence implicitly capturing any domain knowledge encoded by the ground truth adjacency matrices. Because these approaches directly make use of ground truth graphs, they can be viewed as data-driven ways of imposing topological constraints. The contrast between the explicit and implicit approaches hence mirrors that between the model-based and data-driven approaches discussed in Section III-A and Section III-B.

## IV. LEARNING GRAPH GENERATION

## A. Learning graph generative models with a known mapping h<sub>G</sub>

Having discussed the problem of learning graph topology from observed signals, we now turn to Problem 2, which focuses on learning how graphs themselves are generated. Specifically, we consider a set of M observed graphs $\{ G _ { i } \} _ { i = 1 } ^ { M }$ , each endowed with the adjacency matrix $\mathbf { W } _ { i } \in \mathbb { R } ^ { N \times N }$ . We assume a known graph generation mapping $h _ { \mathbf { G } } : \Theta \to \mathcal { G }$ such that each $G _ { i }$ can be treated as the realization of a distribution specified by $h _ { \mathbf { G } } ( \theta )$ , where $\theta \in \Theta$ is the parameter for the distribution. The goal is to learn $\theta$ from the observed $\{ G _ { i } \} _ { i = 1 } ^ { M }$ , which can then be used to generate new graph instances. From a statistical learning perspective, solving Problem 2 is equivalent to solving a parameter estimation problem. In the following, we discuss examples of the known graph generation mapping $h _ { \mathbf { G } }$ and the corresponding parameter estimation approaches, as well as their extensions.

Example 6 (Erdos–R˝ enyi (ER) model).´ This is one of the simplest models that can be used to describe ‘unstructured’ graph topology. The parameter space is given by the set of positive real numbers $\Theta \equiv \mathbb { R } _ { + + }$ denoted by $\theta = p .$ . The n-node graph $G = ( \nu , \mathcal { E } ) \sim h _ { \mathbf { G } } ( p )$ has its node set fixed at $\mathcal { V } = \{ 1 , \ldots , n \}$ and its edge set generated such that each edge is included in $\mathcal { E }$ with probability $\mathbb { P } ( ( i , j ) \in \mathcal { E } ) = p ,$ independently of other edges [90].

The study of graph generative model learning for Example 6 is instrumental. Since each edge is a Bernoulli random variable with the parameter $p ,$ the latter parameter can be straightforwardly estimated via the MLE principle, i.e.,

$$
\widehat { p } = \frac { \sum _ { i = 1 } ^ { m } | \mathcal { E } ( G _ { i } ) | } { m \times { \binom { n } { 2 } } } ,\tag{25}
$$

where $| \mathcal { E } ( G _ { i } ) |$ denotes the cardinality of the edge set of the ith observed graph $G _ { i }$ . Note that the above is a consistent estimator such that ${ \widehat { p } } \to p$ in mean-square sense as $m  \infty$

A closely related setup to the ER model is the $\beta \mathrm { . }$ -model [91] motivated by random graphs with heterogeneous degrees. To represent the different degrees of nodes, the parameter space is extended to $\Theta \equiv \mathbb { R } _ { + + } ^ { n }$ , denoted as $\theta = ( \beta _ { 1 } , \ldots , \beta _ { n } )$ . The n-node graph $G = ( \mathcal { V } , \mathcal { E } ) \sim h _ { \mathbf { G } } ( \theta )$ is generated such that $\begin{array} { r } { \mathbb { P } ( ( i , j ) \in \mathcal { E } ) = \frac { e ^ { \beta _ { i } + \beta _ { j } } } { 1 + e ^ { \beta _ { i } + \beta _ { j } } } } \end{array}$ , independently of other potential edges. As seen, $\beta _ { i }$ increases the probability that node i is connected to a neighboring node and thus the expected degree. Estimating the parameters via the MLE principle can be achieved by solving for $\{ \hat { \beta } _ { i } \} _ { i = 1 } ^ { n }$ in the following set of equations:

$$
\frac { 1 } { m } \sum _ { s = 1 } ^ { m } d _ { i } ( G _ { s } ) = \sum _ { j \neq i } \frac { e ^ { \hat { \beta } _ { i } + \hat { \beta } _ { j } } } { 1 + e ^ { \hat { \beta } _ { i } + \hat { \beta } _ { j } } } , \ i = 1 , \dots , n ,\tag{26}
$$

where $d _ { i } ( G _ { s } )$ denotes the degree of node i in the sth observed graph $G _ { s }$ . Note that (26) can be solved using an efficient fixed-point iteration algorithm, see [91], and the resultant estimator is consistent. A similar setup to the β-model can also be found in the Chung-Lu model [92] which follows a different parameterization for $\mathbb { P } ( ( i , j ) \in \mathcal { E } )$ . Lastly, a generalized graph generative model that covers ER and $\beta$ models as special cases is the exponential random graph model (ERGM) [93]. The parameter space is given by $\Theta \equiv \mathbb { R } ^ { d }$ . The n-node graph $G = ( \mathcal { V } , \mathcal { E } ) \sim h _ { \mathbf { G } } ( \theta )$ is generated such that

$$
\mathbb { P } ( G ; \boldsymbol { \theta } ) = \frac { \exp ( \boldsymbol { \theta } ^ { \top } { \mathbf s } ( G ) ) } { Z ( \boldsymbol { \theta } ) } ,\tag{27}
$$

where $\mathbf { s } ( G ) \in \mathbb { R } ^ { d }$ is a vector of sufficient statistics that captures the topological properties of $G , \mathrm { e . g . }$ ., the number of edges, triangles, or other subgraph counts, and $\begin{array} { r } { Z ( \theta ) = \sum _ { G \in \mathcal { G } } \exp ( \theta ^ { \top } \mathbf { s } ( G ) ) } \end{array}$ is the partition function. The MLE problem can be formulated as

$$
\operatorname* { m a x } _ { \theta \in \Theta } \sum _ { i = 1 } ^ { m } \log p ( G _ { i } ; \theta ) = \sum _ { i = 1 } ^ { m } \theta ^ { \top } \mathbf { s } ( G _ { i } ) - m \log Z ( \theta ) .\tag{28}
$$

The challenge in solving the above MLE problem is that the partition function $Z ( \theta )$ is intractable to compute, for which several approximation methods have been proposed [94], [95]. Moreover, [95] shows that the MLE problem is consistent under certain conditions on the sufficient statistics $\mathbf { s } ( G )$

Beyond learning the graph generation model directly from observed graphs, the literature has also considered an end-to-end learning problem provided that the mapping $f _ { \mathbf { X } }$ is known and the only available data are $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { M }$ . The goal is to invert the composite function $f _ { \mathbf { X } } \circ h _ { \mathbf { G } }$ by sidestepping the inversion of $f _ { \mathbf { X } }$ , i.e., graph topology learning from graph signals in Section III-A. Under the assumptions that $f _ { \mathbf { X } }$ is related to a low-pass graph filter and that $h _ { \mathbf { G } }$ generates graphs with heterogeneous degrees, [96] studied how to rank nodes according to the magnitude of their eigenvector centrality in an end-to-end learning fashion. Building on this setting, [97] proposed to apply factor analysis to relax the stationary graph signal assumption, while [98] extended the approach to inferring centrality from multiple graphs.

Example 7 (Preferential attachment (PA) model). This is a model for describing graphs with powerlaw degree distributions [99]. We consider the affine PA model in [100]. The parameter space is given by $\Theta \equiv \mathbb { Z } _ { + + } \times ( - 1 , \infty )$ with $\theta = ( m , \delta ) \in \Theta$ . Here, m is the number of edges to attach from a new node to existing nodes, and δ is the initial attractiveness of each node. The n-node graph $G = ( \mathcal { V } , \mathcal { E } ) \sim h _ { \mathbf { G } } ( \theta )$ has its node set fixed at $\mathcal { V } = \{ v _ { 1 } , \ldots , v _ { n } \}$ and its edge set generated sequentially such that each new node i connects to m existing nodes with probability proportional to their degree plus δ, independently of other edges. In detail, let $\mathrm { P A } _ { t , i - 1 } ( \delta )$ be the graph generated after adding the (i − 1)-th edge for the tth node, and let $\mathrm { P A } _ { 0 , 0 } ( \delta ) = \mathcal { G } _ { 0 }$ be an initial graph. We include the ith edge to be added between the t-th node and j-th existing node $( j < t )$ with the probability,

$$
\mathbb { P } ( ( v _ { t } , v _ { j } ) \in \mathcal { E } ( \mathrm { P A } _ { t , i } ( \delta ) ) | \mathrm { P A } _ { t , i - 1 } ( \delta ) ) = \frac { d _ { j } ( \mathrm { P A } _ { t , i - 1 } ( \delta ) ) + \delta } { \sum _ { k = 1 } ^ { t - 1 } ( d _ { k } ( \mathrm { P A } _ { t , i - 1 } ( \delta ) ) + \delta ) } , \forall \ j < t ,\tag{29}
$$

where $d _ { j } ( \mathcal { G } )$ denotes the degree of node $j$ in graph G.

For Example 7, since exactly m edges are added for every new node, the parameter m can be directly computed from counting the number of edges in the observed graph. To estimate δ, [100] has derived and studied an estimator based on the MLE principle. In particular,

$$
\widehat { \delta } = \arg \operatorname* { m a x } _ { \delta > - 1 } \left\{ \sum _ { k = 1 } ^ { \infty } \log ( k + \delta ) \frac { N _ { > k } ( n ) - N _ { > k } ^ { 0 } ( n ) } { n } - \frac { 1 } { 2 + \delta / m } \sum _ { k = 0 } ^ { \infty } \log ( k + 1 + \delta ) \frac { N _ { > k } ^ { 0 } ( n ) } { n } \right\} ,\tag{30}
$$

where $N _ { > k } ( n )$ and $N _ { > k } ^ { 0 } ( n )$ are the numbers of nodes with degree larger than k in the observed and initial graph, respectively. The estimator $\widehat { \delta }$ is shown to be consistent and asymptotically normal as $n \to \infty$

Example 8 (Stochastic Block Model (SBM)). This is a popular model for describing graphs with block structure. The parameter space is given by $\Theta \equiv \mathbb { Z } _ { + + } \times \mathbb { R } _ { + } ^ { K } \times \mathbb { R } _ { + } ^ { K \times K }$ with $\theta = ( K , p , P ) \in \Theta$ . Here, K is the number of blocks, $\pmb { p } \in \mathbb { R } _ { + } ^ { K }$ is the block assignment probability vector with $\pmb { p } ^ { \top } \mathbf { 1 } = 1$ , and $\pmb { P } \in \mathbb { R } _ { + } ^ { K \times K }$ is the matrix of edge probabilities between blocks. With this, the n-node graph $G = ( \nu , \mathcal { E } ) \sim h _ { \mathbf { G } } ( \theta )$ has its node set fixed at $\mathcal { V } = \{ 1 , \ldots , n \}$ . A block assignment vector $z \in \{ 1 , \ldots , K \} ^ { n }$ is then drawn according to $\mathbb { P } ( z _ { i } = k ) = p _ { k }$ for each $i \in \{ 1 , \ldots , n \}$ , and the edge set is generated such that each edge is included in E with probability $\mathbb { P } ( ( i , j ) \in \mathcal { E } ) = P _ { z _ { i } , z _ { j } }$ , independently of other edges [90]. Note that as a slight variation, the block assignment vector z can also be treated as a parameter to be learned, which is often the case in community detection problems.

For Example 8, we consider two sub-cases. First, when the block assignment vector z is known, the parameter P can also be estimated via MLE [101], i.e., by estimating each entry $P _ { k , l }$ by the proportion of edges between nodes in blocks k and l similar to (25). Secondly, when the block assignment vector z is unknown (yet K is known), [102] proposed an expectation-maximization (EM) method to approximate the corresponding MLE problem. Herein, the MLE problem is given by

$$
\operatorname* { m a x } _ { \theta \in \Theta } \sum _ { i = 1 } ^ { m } \log p ( \mathbf { W } _ { i } ; \theta )\tag{31}
$$

with

$$
p ( \mathbf { W } _ { i } ; \theta ) = \sum _ { z _ { i } \in \{ 1 , \ldots , K \} ^ { n } } p ( \mathbf { W } _ { i } , z _ { i } ; \theta ) ,\tag{32}
$$

$$
p ( \mathbf { W } _ { i } , z _ { i } ; \theta ) = \prod _ { j = 1 } ^ { n } p _ { [ z _ { i } ] _ { j } } \prod _ { j < k } P _ { [ z _ { i } ] _ { j } , [ z _ { i } ] _ { k } } ^ { [ \mathbf { W } _ { i } ] _ { j k } } ( 1 - P _ { [ z _ { i } ] _ { j } , [ z _ { i } ] _ { k } } ) ^ { 1 - [ \mathbf { W } _ { i } ] _ { j k } } .\tag{33}
$$

The summation in (31) is intractable as it involves $K ^ { n }$ terms. To remedy, [102] proposed to approximate the MLE problem by an EM method treating z as the latent variable. In particular, the E-step computes a set of sufficient statistics consisting of the conditional expectation of p and P given the current estimate of $\theta$ and the observed $\mathbf { W } _ { i } .$ , and the M-step updates the estimates for θ using the sufficient statistics. Also see [103] for a similar variational algorithm and [104] which proved the consistency of such methods. A popular alternative to EM or variational algorithms is to apply spectral clustering on each observed graph to identify the block assignment vector [105], and then estimate the parameters similarly to Eq. (25). In particular, [106] showed that such an estimator is also consistent under certain conditions.

An end-to-end learning approach has also been considered for SBM-like models. This leads to the blind community detection problem, where the goal is to learn the block assignment vector z from observed graph signals $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { M }$ without observing the graph topology G (hence the term “blind”). This problem can be treated as a special case of Problem 3, where the downstream task is to predict the block assignment vector z given x. Surprisingly, the blind community detection problem can be handled by applying simple spectral clustering on the covariance matrix of graph signals [107], [108], with [108] also providing theoretical guarantees for this problem under certain conditions.

## B. Learning graph generative models with an unknown $\widehat { h } _ { \mathbf { G } }$

We now consider the data-driven counterpart of Section IV-A in solving Problem 2. Given a collection of observed graphs $\{ G _ { i } \} _ { i = 1 } ^ { M }$ , with adjacency matrices $\{ \mathbf { W } _ { i } \} _ { i = 1 } ^ { M } \in \mathbb { R } ^ { N \times N }$ , the goal is to learn an unknown graph generation mapping $\widehat { h } _ { \mathbf { G } } : \Theta \to \mathcal { G }$ , which represents a probability distribution over graphs. Once learned, $\widehat { h } _ { \mathbf { G } }$ can be used to sample new graph instances, complete partially observed graphs, or learn latent representations useful for downstream prediction tasks.

In contrast to model-based graph generative models such as the ER/PA models and SBM discussed in Section IV-A, where the form of $h _ { G }$ is specified a priori, data-driven graph generative models learn $\widehat { h } _ { \mathbf { G } }$ directly from graph observations. This provides greater flexibility in modeling complex graph distributions, but typically comes at the cost of weaker identifiability and fewer statistical recovery guarantees. Many data-driven graph generative models introduce a latent variable $z \in { \mathcal { Z } }$ , from which a graph is generated according to

$$
z \sim p ( z ) , \quad G \sim p _ { \theta } ( G \mid z ) ,\tag{34}
$$

where $p _ { \theta } ( G | z )$ is parameterized by a neural network. The latent representation may encode global graph level structure, node-level embeddings, or both. Learning then amounts to estimating the parameters θ of the generative process from the observed graph collection $\{ G _ { i } \} _ { i = 1 } ^ { M }$

Example 9 (Graph variational auto-encoders (Graph VAEs)). Graph VAEs learn a latent representation of each observed graph and decode it into a graph topology. Given an observed graph G, an encoder $q _ { \phi } ( z | G )$ maps G to a latent variable z, sampled from a simple prior distribution, typically a standard Gaussian. A decoder $p _ { \theta } ( G | z )$ then reconstructs or generates a graph from z.

The model is typically trained by maximizing the ELBO, i.e.,

$$
\mathcal { L } ( \theta , \phi ) = \mathbb { E } _ { q _ { \phi } ( z | G ) } [ \log p _ { \theta } ( G | z ) ] - D _ { \mathrm { K L } } ( q _ { \phi } ( z | G )  p ( z ) ) .\tag{35}
$$

The objective consists of two complementary terms. The first term, $\mathbb { E } _ { q _ { \phi } ( z | G ) }$ [log $p _ { \theta } ( G | z ) ]$ , is the reconstruction term. It encourages the decoder to generate graph instances that closely resemble the observed graph $G$ when sampled from the latent representation z. Maximizing this term improves the fidelity of the generated graphs. The second term, $D _ { \mathrm { K L } } \left( q _ { \phi } ( z | G ) \parallel p ( z ) \right)$ , is the Kullback–Leibler (KL) divergence between the learned latent distribution and a chosen prior distribution, typically a standard Gaussian [109]. Minimizing this term encourages the latent representations inferred from observed graphs to remain close to the prior distribution, thereby preventing the latent space from becoming overly specialized to the training data. As a result, nearby points in the latent space tend to generate similar graphs, yielding a smooth and structured latent representation that supports interpolation, sampling, and generation of previously unseen graph instances. Representative examples of this family include [110], [111], [112].

An important distinction from the representation-learning viewpoint discussed in Section III-A is the role of the latent space. In graph topology learning, latent variables are typically introduced to model the generation of graph signals conditioned on an underlying graph structure. By contrast, in graph VAEs, the latent variable z models the variability of the graph topology itself. The latent space therefore captures a distribution over graphs rather than a distribution over signals observed on a graph.

The latent-variable formulation adopted by graph VAEs provides a flexible mechanism for learning graph distributions. However, it does not explicitly model the sequential dependencies that may arise between graph components during graph generation. More generally, many classical graph generative models assume that edges are generated independently, either globally as in ER models or conditionally on latent variables as in SBM. While such assumptions lead to tractable likelihood functions and efficient learning procedures, they can be restrictive when modeling real-world graphs, which often exhibit higherorder dependencies such as clustering, motifs, and long-range dependencies.

Example 10 (Autoregressive (AR) graph generative models). Autoregressive graph generative models construct a graph through a sequence of conditional generation steps. Rather than generating the entire graph at once, they factorize the graph distribution as

$$
p _ { \theta } ( G ) = \prod _ { t = 1 } ^ { T } p _ { \theta } ( a _ { t } \mid a _ { < t } ) ,\tag{36}
$$

where $a _ { t }$ denotes a generation action, such as adding a node, creating an edge, assigning an attribute, or terminating the generation process.

This formulation enables the model to capture complex dependencies between graph components while naturally enforcing structural constraints during generation. Representative examples include GraphRNN [113] and GraphAF [114]. While autoregressive models often produce high-quality samples when generating small to medium-size graphs, they remain sensitive to node ordering and their sequential nature can lead to computationally expensive training and generation, particularly for large graphs. More recently, hierarchical autoregressive formulations have been proposed to alleviate these limitations by generating graphs through multiscale expansion rather than individual node additions. For example, [115] introduces an algorithm based on iterative local expansion, which first generates a coarse graph representation and then progressively expands selected nodes into local subgraphs using a diffusion model. Instead of autoregressively predicting every node or edge, the method autoregressively refines graph resolution, combining hierarchical graph growth with local diffusion-based generation. This substantially improves scalability while preserving the expressiveness of diffusion models, as discussed next, enabling the generation of graphs with several thousand nodes and better extrapolation to unseen graph sizes. Along a different line of research, AutoGraph [116] demonstrates that autoregressive graph generation can be formulated as a sequence modeling problem by introducing a reversible graph-to-sequence transformation. This enables scalable graph generation with standard decoder-only Transformers, effectively bridging graph generation and autoregressive language modeling.

Example 11 (Graph generative adversarial networks). Generative adversarial networks (GANs) formulate graph generation as an adversarial learning problem between two neural networks: a generator that synthesizes graph instances from latent variables and a discriminator that aims to distinguish generated graphs from real ones. Unlike VAEs, GANs learn an implicit graph distribution without explicitly maximizing the data likelihood. Instead, the generator is trained to produce realistic graph structures that fool the discriminator, leading to a minimax optimization problem.

More specifically, given a latent variable $z \sim p ( z )$ , the generator $G _ { \theta }$ produces a graph ${ \widehat { G } } = G _ { \theta } ( z )$ , while the discriminator $D _ { \phi }$ predicts whether a graph originates from the training data or from the generator. The two networks are trained by solving the adversarial objective

$$
\operatorname* { m i n } _ { \theta } \operatorname* { m a x } _ { \phi } \mathbb { E } _ { G \sim p _ { \mathrm { d a t a } } } \left[ \log D _ { \phi } ( G ) \right] + \mathbb { E } _ { z \sim p ( z ) } \left[ \log \left( 1 - D _ { \phi } ( G _ { \theta } ( z ) ) \right) \right] .\tag{37}
$$

Early graph GANs include MolGAN [117], which combines adversarial learning with reinforcement learning (RL) to generate small molecular graphs while optimizing desired chemical properties. More recently, SPECTRE [118] demonstrated that conditioning one-shot graph generators on spectral graph representations substantially improves their expressive power, highlighting the benefits of incorporating global structural information into implicit graph generative models.

Compared with autoregressive and variational approaches, GANs generate graphs in a single forward pass without requiring explicit likelihood estimation. However, adversarial training is often unstable, suffering from issues such as mode collapse and optimization difficulties, particularly in discrete graph domains.

Example 12 (Diffusion-based graph generative models). Graph diffusion models generate graphs through an iterative denoising process [119]. Like graph VAEs, they introduce latent representations to model the variability of graph structures. However, rather than learning a direct mapping from a latent variable $z \ t o$ a graph $G ,$ diffusion models define a sequence of latent graph representations $G _ { 0 } , G _ { 1 } , \dots , G _ { T }$ , where $G _ { 0 }$ denotes an observed graph and $G _ { T }$ corresponds to a highly corrupted or noisy graph representation. Thus, unlike classical generative models, which learn a direct mapping from noise to data, diffusion models generate samples through a sequence of refinement steps, gradually evolving coarse structures into fine-grained details.

The generative process is defined through a forward diffusion process $q ( G _ { t } | G _ { t - 1 } )$ , which progressively corrupts the graph structure, and a learned reverse process $p _ { \theta } ( G _ { t - 1 } | G _ { t } )$ , which reconstructs the graph by iteratively removing noise. A key feature of diffusion models is that the forward diffusion process $q ( G _ { t } | G _ { t - 1 } )$ is specified a priori and is not learned from data. This process moves different graphs into the same simple noise space, which is easy to sample from and serves as the starting point for generation. Along the way, it also creates a smooth sequence of intermediate noisy graphs connecting data to noise. Learning is therefore restricted to the reverse process $p _ { \theta } ( G _ { t - 1 } | G _ { t } )$ , which aims to reconstruct the graph by iteratively removing noise. Graph generation is then performed by sampling from a simple prior distribution at time $T$ and applying the learned reverse dynamics until a graph instance is obtained. We note that this reverse process does not recover one particular original graph. Instead, it starts from random noise and gradually turns it into a graph instance from the same underlying probability distribution.

Depending on the application, the diffusion process may act directly on the adjacency matrix, on node and edge attributes, or on continuous latent graph representations. A diverse family of graph diffusion models has consequently emerged. Early approaches include score-based approaches [120], [121], [122], discrete diffusion [123], [124], and geometric diffusion models for molecular conformation generation including DGSM [125] and GeoDiff [126]. They have also been employed as intermediate steps in specific generative schemes, such as hierarchical generation through iterative local expansion [115]. More recent developments have extended diffusion to continuous-time formulations [127], [128], mixture diffusion models [129], latent diffusion models [130], [131], and structure-aware diffusion processes incorporating explicit graph priors such as SBMs [132]. Diffusion has also been successfully combined with AR generation through approaches such as PARD [133] and Arrow-Diff [134], illustrating the increasing convergence between sequential and diffusion-based graph generation paradigms. For an overview of graph diffusion models we point the reader to [135].

In contrast to graph VAEs, where a single latent variable captures the graph distribution, diffusion models represent the generation process through a sequence of latent graph states. This iterative formulation often provides greater flexibility in modeling complex graph distributions and enables the generation of high-quality graph samples. Compared with autoregressive approaches, diffusion models generate graphs through iterative refinement rather than sequential graph construction, allowing them to capture global graph structure more effectively. Despite their empirical success, diffusion models typically require a large number of denoising steps during generation, resulting in substantial computational costs. This observation has motivated the development of flow-based and flow-matching approaches, which seek to directly learn the transport dynamics between simple and target graph distributions.

Example 13 (Normalizing flows and flow-matching graph generative models). Normalizing flows and flow-matching models constitute a family of likelihood-based generative approaches that learn a transformation between a simple reference distribution and the target graph distribution. Unlike autoregressive or diffusion models, which generate graphs sequentially or through iterative denoising, these methods explicitly model the probability transport from a tractable latent distribution to the graph space. Recent developments have further extended this paradigm from invertible transformations to continuous transport dynamics, establishing flow matching as an efficient alternative to diffusion-based graph generation.

Given an invertible transformation $f _ { \theta } ,$ graph generation is performed by sampling a latent variable $z \sim p ( z )$ and applying the inverse transformation $G = f _ { \theta } ^ { - 1 } ( z )$ , while the exact likelihood is computed using the change-of-variables formula:

$$
\log p ( G ) = \log p ( z ) + \log \left| \operatorname* { d e t } \left( { \frac { \partial f _ { \theta } ( G ) } { \partial G } } \right) \right| .\tag{38}
$$

Early graph normalizing flows include GraphNVP [136], which introduced invertible coupling layers for molecular graph generation, and GraphAF [114], which combines autoregressive graph generation with flow-based transformations to improve scalability. Since graph generation often involves discrete node and edge attributes, subsequent work extended normalizing flows to discrete graph domains. GraphDF [137] proposed discrete normalizing flows for molecular graph generation, while [138] generalized flow-based modeling to categorical variables through continuous transformations.

More recently, flow matching has emerged as a powerful alternative to classical normalizing flows. Instead of explicitly learning an invertible transformation and optimizing Jacobian determinants, flow matching directly learns the time-dependent transport dynamics between a simple reference distribution and the target graph distribution. Specifically, a neural vector field

$$
\frac { d G _ { t } } { d t } = v _ { \theta } ( G _ { t } , t )\tag{39}
$$

is learned to transport samples continuously from the source to the target distribution. Following the general flow-matching framework of Lipman et al. [139], and the generalization to discrete state space [140], DeFoG was among the first methods to adapt flow matching to graph generation, demonstrating competitive sample quality with substantially fewer sampling steps than diffusion models [141]. In the meantime, variational flow-matching objectives have improved the stability and expressiveness of graph generation [142]. More recent developments include constructing probability paths with smooth velocities that respect the graph geometry [143], interpreting conditional diffusion on graphs as a stochastic control problem with multiple reward signals [144], and principled transport maps for categorical variables extending flow-based generative modeling to discrete domains [145].

Unlike diffusion models, which rely on a fixed forward corruption process and a sequence of reverse denoising steps, flow-matching models directly learn the continuous transport dynamics between the ref erence and target distributions. This formulation can lead to substantially faster sampling while preserving the ability to model complex graph distributions.

While the previous Examples 9–13 focus on learning the empirical distribution of observed graphs, a different family of algorithms grounded in Bayesian generative approaches instead aims to learn sampling policies that approximate posterior distributions over graph structures conditioned on observed data. This alternative formulation is illustrated in Example 14.

Example 14 (Bayesian graph structure generation via generative flow networks). Bayesian graph structure learning aims to characterize the posterior distribution over graph structures given observed data. Given observations D, the objective is to sample graph structures according to

$$
p ( G \mid D ) \propto p ( D \mid G ) p ( G ) ,\tag{40}
$$

where $p ( D | G )$ denotes the likelihood and $p ( G )$ a prior over graph structures. Unlike the previous graph generative models, which aim to learn the empirical distribution of observed graphs, Bayesian graph structure learning seeks to characterize the posterior distribution over graph structures induced by the observed data.

![](images/eea839b88903c8121e57d3538ebdb3b203143c4a5a6eb7f8d0d3bc64619854ce.jpg)  
Fig. 4: An illustration of methodologies for learning graph generation. The illustration of deep graph generative models is adapted from that in [6].

A recent line of work addresses this problem using generative flow networks (GFlowNets), with representative examples including [146], [147]. GFlowNets learn a stochastic policy that sequentially constructs a graph through a sequence of actions, such as adding an edge or assigning a parent node. Rather than maximizing the likelihood of observed graph instances, they optimize a flow consistency objective such that the probability of generating a graph is proportional to a prescribed reward. In Bayesian structure learning, this reward is naturally chosen as the unnormalized posterior, $R ( G ) = p ( D | G ) p ( G )$ 2 leading to a learned sampling policy satisfying $P _ { \theta } ( G ) \propto R ( G )$ . Consequently, the learned policy directly approximates the posterior distribution over graph structures, enabling efficient sampling of multiple plausible graphs rather than producing a single point estimate.

Compared with classical Bayesian inference methods based on Markov chain Monte Carlo (MCMC), GFlowNets amortize the sampling process by learning a reusable policy, often resulting in substantially faster posterior exploration while maintaining sample diversity. Moreover, unlike graph VAEs, autoregressive models, diffusion models, and flow-matching approaches, which model the distribution of observed graph instances, GFlowNets are specifically designed to approximate posterior distributions over graph structures. This capability naturally supports Bayesian model averaging, uncertainty quantification, and robust structure discovery.

A high-level illustration of the methodologies for learning graph generation can be found in Fig. 4.

## C. Incorporating domain knowledge in graph generation

Similar to topology learning, graph generation may also benefit from additional domain knowledge. While a known random graph model might have already specified the topological properties of the generated graph, a graph generative model trained from data may be constrained to learn or sample from a distribution whose support is restricted to admissible graphs. For example, PRODIGY introduces a plug-and-play projection mechanism that can be combined with pretrained continuous or discrete graph diffusion models. At each sampling step, the current graph representation is projected toward a userspecified constrained space, allowing hard and potentially non-differentiable requirements to be imposed without retraining the underlying generative model [148]. ConStruct instead modifies the discrete diffusion process itself through an edge-absorbing noise model and a constraint-aware projector, ensuring that intermediate and final graphs remain within structural families satisfying properties such as planarity or acyclicity [149]. From this perspective, hard constraints play analogous roles in topology learning and graph generation: in the former they restrict the feasible graph estimate, while in the latter they restrict the support of the learned graph distribution.

By contrast, there also exist attempts in which topological properties are absorbed from observed graph samples into the parameters or latent representation of a generative model, without defining an explicit feasibility set or topology-specific penalty. For example, NetGAN [150] learns the distribution of random walks on observed networks, thereby reproducing degree, community, and motif statistics without specifying these quantities explicitly. Similarly, GraphRNN [113] and GRAN [151] generate graphs through conditional distributions over sequential graph-construction decisions, while SPECTRE [118] learns a distribution over spectral representations that encode global graph structure. Unlike projectionbased approaches such as PRODIGY and ConStruct, these methods encourage topological validity only in distribution and do not guarantee that every generated sample satisfies the properties present in the training graphs.

## V. TASK-DRIVEN GRAPH STRUCTURE LEARNING

The formulations presented in Sections III and IV treat graph topology and graph generation as objectives in their own right. In many practical applications, however, the graph is not the final goal but rather an intermediate representation that supports a downstream task, such as node classification, link prediction, recommendation, or molecular property prediction. In these settings, the graph should be learned jointly with the downstream objective, leading to a task-driven formulation of graph structure learning, as defined in Problem 3.

Unlike the previous two formulations, task-driven graph structure learning does not simply invert either the f-map or h-map. Instead, the downstream objective introduces an additional supervision signal that jointly influences the learning process. From the perspective of our framework, the downstream task acts as an additional constraint on the inverse problem, biasing the learned graph topology or graph generative model toward representations that are useful for prediction rather than solely for explaining the observations. The additional supervision also changes the structure of the optimization problem: since the graph influences the downstream model, which in turn provides the supervision for learning the graph, the two components must be optimized jointly. In many cases, this mutual dependency naturally gives rise to a nested optimization problem, where the graph structure is updated according to the downstream objective, while the downstream model is simultaneously learned on the current graph.

![](images/357d0571424401b7bdc82c29b07c50d93ca1d308586345ec84e3064e1893763b.jpg)  
Fig. 5: An illustration of task-driven graph topology learning.

Following the general idea outlined above, recent bilevel optimization (BLO) algorithms have played an important role in task-driven learning, particularly for the case of graph topology learning. These advances are critical as the learning formulations involve implicit functions of the graph topology. In general, a BLO problem is described as:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { \mathbf { W } , \theta } } & { { } J _ { U L } ( \mathbf { W } , \theta ) } \\ { \mathrm { s . t . } } & { { } \theta \in \arg \displaystyle \operatorname* { m i n } _ { \hat { \theta } } J _ { L L } ( \mathbf { W } , \hat { \theta } ) , } \end{array}\tag{41}
$$

where $J _ { U L } , J _ { L L }$ are the upper-level (UL) and lower-level (LL) objective functions, respectively. A possible setup is that $J _ { U L } ( \cdot )$ models the performance of downstream tasks under the graph topology W and the auxiliary variable θ. The latter is learned by minimizing $J _ { L L } ( \cdot )$ that may incorporate the typical graph signal data. Although (41) can be non-convex, recent works have developed efficient algorithms such as hypergradient-based methods [152] and penalty-based methods [153]. These works have made it possible to handle the BLO that arises in task-driven graph topology learning. In what follows, we review two examples of recent efforts in developing downstream task-induced topology learning techniques. A highlevel illustration of the methodologies in this case can be found in Fig. 5.

Example 15 (Tasks through supervised graph neural networks). In this example, the target downstream task is defined through the mapping $g _ { \mathbf { Y } } : \mathcal { X } \times \mathcal { G }  \mathcal { Y }$ where $\mathcal { V }$ is a finite set describing the class labels on (a subset of) nodes. As $y \in \mathcal { V }$ represents the prediction target, the outputs of $g \mathbf { \mathbf { Y } }$ will be absorbed as a regularizer in the learning of the graph topology, i.e., affecting $J _ { U L } ( \cdot )$ in (41). Meanwhile, the exact form of $g \mathbf { Y }$ is typically unknown and is modeled as a GNN to be learned by minimizing a training loss in $J _ { L L } ( \cdot )$

To simultaneously learn $g \mathbf { \mathbf { Y } }$ and the graph topology that supports it, [154] proposed a bilevel optimization framework for learning graph structures that optimize the performance of GNNs on downstream tasks. The upper-level objective $J _ { U L } ( \cdot )$ measures the performance of the GNN on a validation set, while the lower-level objective $J _ { L L } ( \cdot )$ corresponds to the training loss of the GNN on a training set. Concretely,

$$
\begin{array} { r } { J _ { U L } ( \mathbf { W } , \pmb { \theta } ) = \mathcal { L } ( \mathrm { G N N } _ { \pmb { \theta } } ( \mathbf { W } , \mathbf { x } _ { \mathrm { v a l } } ) , \mathbf { y } _ { \mathrm { v a l } } ) , } \end{array}\tag{42}
$$

$$
\begin{array} { r } { J _ { L L } ( \mathbf { W } , \theta ) = \mathcal { L } ( \mathrm { G N N } _ { \theta } ( \mathbf { W } , \mathbf { x } _ { \mathrm { t r a i n } } ) , \mathbf { y } _ { \mathrm { t r a i n } } ) , } \end{array}
$$

where $\mathrm { G N N } _ { \pmb { \theta } } ( \mathbf { W } , \mathbf { x } )$ denotes the output of a GNN defined on the graph W and input features x with parameters $\theta ,$ and $\mathcal { L } ( \cdot )$ is a standard training loss function (e.g., cross-entropy loss). The BLO problem is then tackled using a hypergradient-based method. With a similar goal of leveraging downstream tasks to inform graph topology learning, [155] proposes Pro-GNN based on a multi-objective formulation. This formulation can be interpreted as a penalized BLO with $J _ { L L } ( \cdot )$ defined similarly to that in (42) and $J _ { U L } ( \cdot )$ chosen as in Section III-A. The resulting BLO can be seen as a variant of [154]; also see [3], [4], [156] for an overview. The recent work in [157] further expanded the task-driven graph topology learning framework to incorporate learnable homophilic and heterophilic graphs, thereby increasing the expressivity of the learned structures. Finally, we note that the GNNs are used in (42) as an example and they may be replaced by any graph machine learning model.

Example 16 (Tasks through latent states). In this example, the target downstream tasks are directly related to the functions performed by a graph system, $e . g .$ , graph signal denoising, sampling set selection, and utility maximization in an economic network. We can capture the downstream task by modeling the output of $g \mathbf { Y } : \mathcal { G }  \mathcal { Y }$ as a latent state. A typical setup for $\mathcal { V }$ is the continuous space, e.g., $\mathcal { Y } = \mathbb { R } ^ { n \times d }$ The output of $g \mathbf { \mathbf { Y } }$ is incorporated as a regularizer in the learning of the graph topology in $J _ { U L }$ of (41). Meanwhile, the form of $g \mathbf { \mathbf { Y } }$ is typically assumed to be known and depends on the specific task.

Recent work [158] studies graph topology learning in socioeconomic and biological systems by leveraging downstream task optimization. Specifically, [158] models the downstream task through an equilibrium-seeking problem (e.g., a network game), which constitutes the LL problem in the BLO framework, and postulates that the graph topology results from a formation process that maximizes the utility evaluated at equilibrium.

The upper-level objective $J _ { U L } ( \cdot )$ measures simultaneously the fit of the learned graph topology to the observed graph signals and incorporates a regularization function depending on the equilibrium of the underlying network game or dynamical system $\pmb { \theta } ^ { \star } ( \mathbf { W } )$ . The lower-level objective $J _ { L L } ( \cdot )$ characterizes its equilibrium condition. Concretely,

$$
\operatorname* { m i n } _ { \mathbf { W } , \theta } \ J _ { U L } ( \mathbf { W } , \pmb { \theta } ) = \mathcal { L } ( \mathbf { W } , \mathbf { X } ) + \lambda \ \mathcal { R } ( \pmb { \theta } ) ,\tag{43}
$$

$$
\mathrm { s . t . } \theta \in \arg \operatorname* { m i n } _ { \hat { \theta } } J _ { L L } ( \mathbf { W } , \hat { \pmb { \theta } } ) = \mathcal { F } ( \mathbf { W } , \hat { \pmb { \theta } } ) ,
$$

where $\mathcal { L } ( \cdot )$ is a loss function measuring the fit of W to the observed graph signal data X [cf. Section III], $\mathcal { R }$ is a regularization function that depends on the equilibrium of the underlying network game or dynamical system, and $\mathcal { F }$ characterizes its equilibrium condition. In a similar spirit, [159] considers graph signal clustering as the downstream task and jointly learns the graph topology, [160] learns a graph representation to be used as a feature for classification, and [161] proposes to learn the graph topology for sampling set selection. By exploiting the structure of the task, these works also show that a single-level formulation may suffice to approximately incorporate the downstream task as a direct regularizer in the objective of graph topology learning.

Unlike Example 15, where the downstream task is specified through a supervised GNN that can be directly learned, Example 16 assumes that the downstream task is specified through a latent state arising from known dynamics. Works addressing unknown dynamics remain relatively scarce. The recent work in [162] considers using observed trajectory data to learn the underlying dynamics that provide graph priors. Research along this direction is still in its infancy, and further work is needed to fully understand the interplay between graph topology learning and downstream tasks. The discussion above also provides an interesting interpretation in which the task at hand effectively serves as the domain knowledge that may benefit learning. [158] presents such an example, in which a viable social network topology is assumed to be one for which the associated equilibrium actions of a game played on that network would maximize the total utility of all individuals in the network (the task, which is reflected by the ${ \mathcal { R } } ( \theta )$ term in Eq. (43)). It is demonstrated that such an implicit bias can actually be interpreted using explicit topological properties, such as heterogeneity in node degrees, thereby bridging the gap between implicit and explicit graph priors.

Departing from graph topology learning, recent works have also considered learning graph generation models for downstream tasks. The underlying idea is similar, but the primary object to be learned becomes the graph generation model, e.g., a classical random graph model. For instance, for node classification tasks, [163] proposes learning the posterior distribution of the parameters of an SBM from a potentially noisy graph and using it to sample new graph instances in an ensemble GNN model, while [164] and [165] propose learning a non-uniform ER model in which each edge is independently associated with a Bernoulli distribution. Graph diffusion models can also be adapted to downstream tasks. For example, [166] proposes a score-based diffusion model to perform graph classification rather than merely generating graph instances, [167] shows how to augment graph topology with a pre-trained diffusion model to serve various downstream tasks, and [168] considers augmenting a small graph with additional nodes generated by a latent diffusion model for node classification. Beyond the standard formulation of minimizing classification error, a related work [169] considers an RL-based formulation to learn a policy network for graph generation, where the policy is directly influenced by a task-related reward such as synthetic accessibility or drug-likeness.

## VI. BRIDGING TOPOLOGY LEARNING WITH GRAPH GENERATION

Although topology learning and graph generation have been discussed separately, they share several fundamental characteristics. Recognizing these commonalities not only provides a higher-level synthesis of the two problems but may also inspire new formulations and methodologies. In this section, we summarize the commonalities and discuss some recent developments that bridge the two.

## A. A unified generating process for graph data

Although topology learning and graph generation have traditionally evolved as distinct research directions, the framework illustrated in Fig. 2 and presented in this review reveals that they are in fact complementary views of the same data-generating process. Indeed, topology learning focuses on recovering the latent graph structure from node observations by modeling the observation process $f _ { \mathbf { X } }$ . Under specific assumptions, such as smoothness, dynamical processes, or graph filtering, learning the graph simultaneously identifies the mechanism governing the observed signals. Consequently, the learned graph is often not only a structural object but also defines a generative model that explains the node observations, enabling, for example, the synthesis or prediction of graph signals according to the known or learned observation process. Conversely, graph generation assumes that graph instances are directly observed and aims to learn the graph-generating mechanism $h _ { \mathbf { G } }$ . Rather than estimating a single graph, these approaches learn a probability distribution over graphs, enabling the synthesis of previously unseen graphs with similar statistical and structural properties. Viewed together, these two families address complementary components of a single generative hierarchy, illustrated in Fig. 2, where latent generative parameters first determine graph structure, which subsequently gives rise to graph-structured observations through statistical, physical, or filtering processes. Existing methods typically recover only one stage of this hierarchy: topology learning focuses on the mapping between $\mathcal { G }$ and $x ,$ while graph generation focuses on that between $\Theta$ and ${ \mathcal { G } } .$ Consequently, neither fully models the complete mechanism responsible for the observed graph data.

Recent work has begun to bridge this gap by combining elements from the two traditionally separate stages discussed above. For example, [170] formulates topology learning as a graph deconvolution problem, where the observed graph is modeled as the output of a latent graph passed through a graph convolutional operator. Therefore, this can be interpreted as learning a mapping $h _ { \mathbf { G } }$ for $G ,$ which, however, is inspired by a mapping $f _ { \mathbf { X } }$ (i.e., convolutional filtering) commonly used to model node observations x. Another example is the work in [171], which introduces an explicit probabilistic data-generating process for heterogeneous graphs and proposes to jointly learn graph topology together with the parameters governing graph formation, thus explicitly combining the mappings $h _ { \mathbf { G } }$ and $f _ { \mathbf { X } }$ . A complementary line of work extends graph generation beyond the topology alone. Graph-aware generative models such as [172], [173] incorporate graph operators into the generation of graph-supported signals, thereby modeling not only the structural object $G$ but also observations generated conditionally on its geometry. These approaches illustrate two directions of methodological transfer, leveraging the single generative hierarchy in Fig. 2: models developed for topology learning can inform graph generation, while generative modeling can extend topology learning from point estimates to probabilistic models of structures and their associated observations.

## B. Bridging communities in graph structure learning

The development of graph structure learning has been driven by several research communities that have historically evolved largely independently, including PGMs, GSP, GML, and network science. Although these communities often study similar graph-structured data, they have been motivated by different scientific questions, leading to the formulation of graph structure learning as an inverse problem. As a consequence, many methods that appear fundamentally different can instead be understood as learning different components of the same generating process for graph data introduced in Section VI-A, under different priors.

At a high level, GSP views the graph as a latent physical or statistical structure that explains observed signals, i.e., a graph provides the domain over which the observations live. The primary objective is therefore to recover graph topologies that best support observed data under assumptions such as smoothness, diffusion, or dynamical evolution, as explicitly discussed in Section III-A. In particular, most of the approaches rely on a notion of smoothness of the graph signal, either in the global sense (i.e., smoothness over all graph edges in Example 1) or in a more local sense (e.g., in Example 2 and diffusionlike graph filtering in Example 3). The learned graph is expected to reflect meaningful interactions between variables and provide a data structure for downstream signal processing tasks including denoising, interpolation, forecasting, and system identification. As a matter of fact, a signal representation model is often employed and from this perspective graph topology learning can be understood as a form of representation learning for better processing of graph signals. For example, in the factor analysis model in Eq. (3), the eigenvector basis can be viewed as a latent space (or an encoder) which is implicitly learned via the learning of L, and the assumption on the signal x is mediated through that on its latent representation h. Similarly, in the filter-based model in Eq. (13), the latent space becomes the eigenvector basis of S [62], [63], or a dictionary that is a function of S [61].

Similar to GSP, PGMs aim to recover interaction structures from observations, but define these interactions through conditional dependence relationships rather than graph signal models. Consequently, PGMs and GSP often arrive at closely related optimization problems despite originating from different modeling assumptions. There is also an analogy between classical PGM and dynamical-process-based approaches, where the graph captures the conditional dependence (bi-directional in PGMs) or conditional probability (directional in diffusion) between a node pair. Both can then be interpreted using a unified GSP language.

By contrast, GML primarily views the graph as a latent variable for representation learning. Rather than explicitly recovering the underlying interaction network, the objective is to learn representations that optimize generative or downstream predictive tasks. While GML has traditionally focused on discriminative problems such as node classification, graph classification, and link prediction, recent advances have increasingly focused on the specific problem of graph generation, where the objective is to learn expressive distributions over graph-structured data for applications including molecule design, simulation, and data augmentation. Most of the state-of-the-art methods in this category follow an encoder-decoder paradigm, sharing a similar philosophy to GSP approaches for topology learning. However, the latent space (encoder) can often be learned in a data-driven manner (note that [38] already leverages this idea to implicitly incorporate graph priors), or predefined based on a noise model (see Example 12).

A third perspective arises from network science and statistical network modeling, where the primary objective is to understand the principles governing network organization. Instead of learning a single graph instance, these approaches seek probabilistic mechanisms capable of explaining graph populations through properties such as community organization, degree distributions, motif statistics, or temporal evolution. Classical random graph models, SBMs, and PA models exemplify this viewpoint by treating graphs as realizations of underlying probabilistic mechanisms (i.e., p(G|Θ)) that explain why observed networks exhibit particular structural characteristics. These generative models are usually selected according to the context of the target complex systems under study. For example, the ER model (Example 6) is used to model random networks, the SBM (Example 8) is used to model community structures in social networks, and the PA model (Example 7) is used to model scale-free networks in biological systems. Going beyond standard random graph models, recent studies have considered task-induced graph models [158], which are used to motivate the prior knowledge of the graph topology in the context of graph structure learning.

TABLE I: A unified view of major communities in graph structure learning.
<table><tr><td rowspan=1 colspan=1>Perspective</td><td rowspan=1 colspan=1>Observations</td><td rowspan=1 colspan=1>Object of learning</td><td rowspan=1 colspan=1>Representative tasks</td><td rowspan=1 colspan=1>Representative models / methods</td></tr><tr><td rowspan=1 colspan=1>ProbabilisticGraphical Models(III-A)</td><td rowspan=1 colspan=1>Data samples X</td><td rowspan=1 colspan=1>Conditional depen-dence graph G</td><td rowspan=1 colspan=1>Structure learning, covariance esti-mation, causal discovery</td><td rowspan=1 colspan=1>Gaussian  graphical  models,Bayesian networks</td></tr><tr><td rowspan=1 colspan=1>Graph    SignalProcessing (III-A,IⅢI-B, V)</td><td rowspan=1 colspan=1>Graph signals X</td><td rowspan=1 colspan=1>Signal domain G</td><td rowspan=1 colspan=1>Topology learning, denoising, in-terpolation, forecasting</td><td rowspan=1 colspan=1>Smoothness priors, graph filtering,dictionary learning</td></tr><tr><td rowspan=1 colspan=1>Network Science(IV-A)</td><td rowspan=1 colspan=1>Graph samples G</td><td rowspan=1 colspan=1>Graph-generatingmechanisms p(G)</td><td rowspan=1 colspan=1>Community detection, motif dis-covery, network evolution</td><td rowspan=1 colspan=1>ER, ERGM, PA, SBM</td></tr><tr><td rowspan=1 colspan=1>Graph Generation(IV-B, V)</td><td rowspan=1 colspan=1>Graph samples G</td><td rowspan=1 colspan=1>Generative modelp(G)</td><td rowspan=1 colspan=1>Graph synthesis, simulation, uncer-tainty quantification</td><td rowspan=1 colspan=1>VAEs, GANs, normalizing flows,diffusion models, ARs</td></tr><tr><td rowspan=1 colspan=1>Graph  MachineLearning  (III-B,IV-B, V)</td><td rowspan=1 colspan=1>Graphs G, node at-tributes X</td><td rowspan=1 colspan=1>Predictive functionp(y|G, X)</td><td rowspan=1 colspan=1>Node/graph classification, regres-sion, link prediction</td><td rowspan=1 colspan=1>Graph neural networks, graphtransformers</td></tr></table>

Viewed through the lens of the unified generating process for graph data, these communities differ primarily in the level of abstraction at which learning is performed. GSP and PGMs seek interaction structures that explain observed data, GML learns representations and graph structures that optimize downstream predictive or generative objectives, while network science seeks the latent mechanisms governing graph formation itself. Thus, these perspectives address complementary learning problems defined on different components of the same generating process. See Table I for an overview. Recognizing these connections provides opportunities to transfer ideas across communities by, for example, combining physically grounded graph priors from GSP with the flexibility of GML, or enriching deep graph generative models with the mechanistic principles developed in network science. Similarly, the divide between model-based and data-driven approaches is often less clear than that idealized in Fig. 2, and recent developments often leverage the strengths of both for the task at hand, with typical examples including learned priors [38], [76], or constraint-aware generators [148], [149].

## VII. FUTURE PERSPECTIVES

The unified perspective presented throughout this review naturally gives rise to a new generation of formulations of graph structure learning that jointly model multiple components of the generating process for graph data. Rather than treating graph topology learning and graph generation as separate problems, these approaches seek unified models capable of explaining how relational systems emerge, evolve, and generate observations.

## A. From graphs with pairwise relations to higher-order structures and richer geometries

Throughout this review, we have focused on learning the structure of pairwise graphs. However, many real-world systems exhibit interactions that cannot be adequately captured by pairwise relationships alone, suggesting that graphs may not always constitute the most appropriate relational representation. This motivates extending graph structure learning beyond pairwise graphs to higher-order structures such as hypergraphs [174], [175], [176], [177], simplicial complexes [178], [179], [180], and cell complexes [181], as well as richer geometries including cellular sheaves [182], [183], [184], vector bundles [185], and product manifolds [186].

More fundamentally, future systems for graph structure learning may treat the choice of relational representation itself as part of the learning problem. Rather than assuming a priori that observations arise from a graph, learning algorithms could infer whether pairwise graphs, hypergraphs, simplicial and cell complexes, or richer geometries provide the most appropriate representation of the underlying interactions and their associated feature transformations. From this perspective, graph topology learning naturally generalizes from recovering an unknown graph to discovering a general topological or geometric domain, while graph generation extends from learning generative distributions over graphs to learning generative distributions over these richer representations. More broadly, this evolution suggests that the unified perspective developed throughout this review may itself be viewed as a special case of a broader paradigm: learning relational representations directly from data, where the underlying domain geometry is no longer prescribed but inferred.

## B. From learning a graph structure to learning a generating process

The unified perspective developed throughout this review suggests a broader objective for graph structure learning. Rather than treating topology learning, graph generation, and downstream tasks as separate problems, they could be viewed as complementary components in a common relational datagenerating process as discussed in Section VI-A. The mechanisms identified through such a holistic approach would shed light on how relational structures are formed, evolve, and generate observations.

Under this perspective, the graph becomes an intermediate latent representation rather than the final object of learning.

Recent developments have begun to move in this direction. Probabilistic formulations of graph topology learning model the observed graph through latent graph-generating processes, thereby explicitly reasoning about how graph structures arise [171]. Likewise, emerging approaches in causal structure learning employ structured search and RL to infer the mechanisms governing directed dependencies rather than merely estimating connectivity patterns [187]. Although these first methods remain specialized, they illustrate a broader shift from estimating relational structures toward explaining the processes that generate the observations.

## C. Toward a universal learning paradigm via foundation models

As discussed so far, existing graph topology learning and graph generation methods are typically designed for specific input data and modeling assumptions, graph domains, or downstream tasks. As graph-structured data continue to grow in scale and diversity, an important future direction is to develop generalist models capable of learning transferable representations and structural priors across heterogeneous graph learning problems. Recent advances in foundation models open new opportunities for integrating probabilistic learning with semantic and domain knowledge. For example, large language models have recently been leveraged to guide causal graph discovery by reducing the combinatorial search space and incorporating prior knowledge into the learning process [188]. More broadly, foundation models may provide a common interface for combining graph signal processing, graph generative models, causal inference, and domain knowledge within a unified probabilistic learning framework, enabling graph learning systems that are more data-efficient and transferable across diverse application domains.

## D. Promising applications

Although we have intentionally focused on the technical aspects of graph topology learning and graph generation in this review, they naturally enable numerous applications in biological, chemical, social, and urban networks, to name a few. Looking five to ten years ahead, we believe the most consequential applications of these tools are likely to emerge in systems where relational structure is latent, time-varying, and directly designable. On the one hand, graph topology learning may evolve from the offline recovery of a single static graph toward uncertainty-aware, multiscale inference of changing interactions from streaming observations, providing a structural foundation for self-updating physical infrastructure, industrial processes, autonomous collectives, social networks, and biomedical mechanisms. Existing work on graph-based physical simulation already illustrates the potential to learn interaction structure jointly with system dynamics [189], while large-scale graph-based forecasting demonstrates the broader opportunity for relational models in complex dynamical systems [190]. On the other hand, deep graph generation may move beyond reproducing observed graph statistics toward goal-oriented structural design, generating proteins, molecular complexes, materials, and network configurations with specified functional and physical properties [191], [192]. Together, the most transformative systems may combine the two paradigms in a closed loop: inferring the current relational structure from observations, generating promising candidate structures for intervention, evaluating them through simulations or experiments, and updating both the inferred topology and the generative model from feedback.

## REFERENCES

[1] X. Dong, D. Thanou, M. Rabbat, and P. Frossard, “Learning graphs from data: A signal representation perspective,” IEEE Signal Processing Magazine, vol. 36, no. 3, pp. 44–63, 2019.

[2] G. Mateos, S. Segarra, A. G. Marques, and A. Ribeiro, “Connecting the dots: Identifying network structure via graph signal processing,” IEEE Signal Processing Magazine, vol. 36, no. 3, pp. 16–43, 2019.

[3] Y. Zhu, W. Xu, J. Zhang, Y. Du, J. Zhang, Q. Liu, C. Yang, and S. Wu, “A survey on graph structure learning: Progress and opportunities,” arXiv preprint arXiv:2103.03036v2, 2022.

[4] H. Attali, D. Buscaldi, and N. Pernelle, “Rewiring techniques to mitigate oversquashing and oversmoothing in GNNs: A survey,” arXiv preprint arXiv:2411.17429, 2024.

[5] X. Guo and L. Zhao, “A systematic survey on deep generative models for graph generation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 5, p. 5370–5390, 2023.

[6] Y. Zhu, Y. Du, Y. Wang, Y. Xu, J. Zhang, Q. Liu, and S. Wu, “A survey on deep graph generation: Methods and applications,” in Proceedings of the Learning on Graphs Conference, 2022, pp. 47:1–47:21.

[7] D. Koller and N. Friedman, Probabilistic graphical models: Principles and techniques. MIT Press, 2009.

[8] S. A. Myers and J. Leskovec, “On the convexity of latent social network inference,” in Proceedings of Advances in Neural Information Processing Systems, 2010, pp. 1741–1749.

[9] M. Gomez-Rodriguez, J. Leskovec, and A. Krause, “Inferring networks of diffusion and influence,” in Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2010, pp. 1019–1028.

[10] C. Wiuf, M. Brameier, O. Hagberg, and M. P. H. Stumpf, “A likelihood approach to analysis of network data,” Proceedings of the National Academy of Sciences, vol. 103, no. 20, pp. 7566–7570, 2006.

[11] T. Tabouy, P. Barbillon, and J. Chiquet, “Variational inference for stochastic block models from sampled data,” Journal of the American Statistical Association, vol. 115, no. 529, pp. 455–466, 2020.

[12] J. Friedman, T. Hastie, and R. Tibshirani, “Sparse inverse covariance estimation with the graphical Lasso,” Biostatistics, vol. 9, no. 3, pp. 432–441, 2008.

[13] X. Dong, D. Thanou, P. Frossard, and P. Vandergheynst, “Learning Laplacian matrix in smooth graph signal representations,” IEEE Transactions on Signal Processing, vol. 64, no. 23, pp. 6160–6173, 2016.

[14] V. Kalofolias, “How to learn a graph from smooth signals,” in Proceedings of the International Conference on Artificial Intelligence and Statistics, 2016, pp. 920–929.

[15] H. E. Egilmez, E. Pavez, and A. Ortega, “Graph learning from data under structural and Laplacian constraints,” IEEE Journal of Selected Topics in Signal Processing, vol. 11, no. 6, pp. 825–841, 2017.

[16] S. P. Chepuri, S. Liu, G. Leus, and A. O. Hero, “Learning sparse graphs under smoothness prior,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2017, pp. 6508–6512.

[17] C. Shi and G. Mishne, “Graph Laplacian learning with exponential family noise,” IEEE Transactions on Signal and Information Processing over Networks, vol. 11, pp. 641–654, 2025.

[18] X. Pu, S. L. Chau, X. Dong, and D. Sejdinovic, “Kernel-based graph learning from smooth signals: A functional viewpoint,” IEEE Transactions on Signal and Information Processing over Networks, vol. 7, pp. 192–207, 2021.

[19] J. Guo, S. Moses, and Z. Wang, “Graph learning from signals with smoothness superimposed by regressors,” IEEE Signal Processing Letters, vol. 30, pp. 942–946, 2023.

[20] K. Watanabe, K. Maeda, T. Ogawa, and M. Haseyama, “Learning graph Laplacian from intrinsic patterns via Gaussian process,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2023.

[21] B. Ma, M. McNeil, and P. Bogdanov, “GIST: Graph inference for structured time series,” in Proceedings of the SIAM International Conference on Data Mining, 2023, pp. 433–441.

[22] P. Berger, G. Hannak, and G. Matz, “Efficient graph learning from noisy and incomplete data,” IEEE Transactions on Signal and Information Processing over Networks, vol. 6, pp. 105–119, 2020.

[23] A. Karaaslanli and S. Aviyente, “Graph learning from noisy and incomplete signals on graphs,” in Proceedings of the IEEE Statistical Signal Processing Workshop, 2021.

[24] A. Buciulea, S. Rey, and A. G. Marques, “Learning graphs from smooth and graph-stationary signals with hidden variables,” IEEE Transactions on Signal and Information Processing over Networks, vol. 8, pp. 273–287, 2022.

[25] H.-S. Nguyen and H.-T. Wai, “Learning graph from smooth signals under partial observation: A robustness analysis,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2026, pp. 61–65.

[26] D. Hallac, Y. Park, S. Boyd, and J. Leskovec, “Network inference via the time-varying graphical lasso,” in Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2017, pp. 205–213.

[27] V. Kalofolias, A. Loukas, D. Thanou, and P. Frossard, “Learning time varying graphs,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2017, pp. 2826–2830.

[28] K. Yamada, Y. Tanaka, and A. Ortega, “Time-varying graph learning based on sparseness of temporal variation,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2019, pp. 5411–5415.

[29] A. Natali, E. Isufi, M. Coutino, and G. Leus, “Learning time-varying graphs from online data,” IEEE Open Journal of Signal Processing, vol. 3, pp. 212–228, 2022.

[30] X. Yang, M. Sheng, Y. Yuan, and T. Q. S. Quek, “Network topology inference from heterogeneous incomplete graph signals,” IEEE Transactions on Signal Processing, vol. 69, pp. 314–327, 2020.

[31] S. Bagheri, G. Cheung, T. Eadie, and A. Ortega, “Joint signal interpolation/time-varying graph estimation via smoothness and low-rank priors,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2024, pp. 9646–9650.

[32] R. Ye, X. Jiang, H. Feng, J. Wang, R. Qiu, and X. Hou, “Time-varying graph learning from smooth and stationary graph signals with hidden nodes,” EURASIP Journal on Advances in Signal Processing, vol. 2024, no. 33, 2024.

[33] P. Danaher, P. Wang, and D. M. Witten, “The joint graphical lasso for inverse covariance estimation across multiple classes,” Journal of the Royal Statistical Society Series B: Statistical Methodology, vol. 76, no. 2, p. 373–397, 2014.

[34] M. Navarro, Y. Wang, A. G. Marques, C. Uhler, and S. Segarra, “Joint inference of multiple graphs from matrix polynomials,” Journal of Machine Learning Research, vol. 23, no. 76, pp. 1–35, 2022.

[35] A. Karaaslanli and S. Aviyente, “Multiview graph learning with consensus graph,” IEEE Transactions on Signal and Information Processing over Networks, vol. 7, pp. 161–176, 2025.

[36] S. S. Saboksayra, G. Mateos, and M. Cetin, “Online discriminative graph learning from multi-class smooth signals,” Signal Processing, vol. 186, no. 108101, 2021.

[37] H. Shrivastava, X. Chen, B. Chen, G. Lan, S. Aluru, H. Liu, and L. Song, “GLAD: Learning sparse graph recovery,” in Proceedings of the International Conference on Learning Representations, 2020.

[38] X. Pu, T. Cao, X. Zhang, X. Dong, and S. Chen, “Learning to learn graph topologies,” in Proceedings of Advances in Neural Information Processing Systems, 2021, pp. 4249–4262.

[39] M. Wasserman and G. Mateos, “Graph structure learning with interpretable bayesian neural networks,” arXiv preprint arXiv:2406.14786, 2024.

[40] J. Mei and J. M. F. Moura, “Signal processing on graphs: Causal modeling of unstructured data,” IEEE Transactions on Signal Processing, vol. 65, no. 8, pp. 2077–2092, 2017.

[41] C. Cui, P. Banelli, and P. M. Djuric, “Topology inference of directed graphs by Gaussian processes with sparsity´ constraints,” IEEE Transactions on Signal Processing, vol. 72, pp. 2147–2159, 2024.

[42] Y. Liu, L. Yang, K. You, W. Guo, and W. Wang, “Graph learning based on spatiotemporal smoothness for time-varying graph signal,” IEEE Access, vol. 7, pp. 62 372–62 386, 2019.

[43] A. Javaheri, A. Amini, F. Marvasti, and D. P. Palomar, “Learning spatiotemporal graphical models from incomplete observations,” IEEE Transactions on Signal Processing, vol. 72, pp. 1361–1374, 2024.

[44] Y. Shen, G. B. Giannakis, and B. Baingana, “Nonlinear structural vector autoregressive models with application to directed brain networks,” IEEE Transactions on Signal Processing, vol. 67, no. 20, pp. 5325–5339, 2019.

[45] M. Coutino, E. Isufi, T. Maehara, and G. Leus, “State-space network topology identification from partial observations,” IEEE Transactions on Signal and Information Processing over Networks, vol. 6, pp. 211–225, 2020.

[46] G. Mateos, S. Rey, H. Ajorlou, and M. Tepper, “Concomitant DAG learning: On the roles of noise adaptivity, sparsity, and non-negativity,” arXiv preprint arXiv:2605.23537, 2026.

[47] E. Ceci, Y. Shen, G. B. Giannakis, and S. Barbarossa, “Graph-based learning under perturbations via total least-squares,” IEEE Transactions on Signal Processing, vol. 68, pp. 2870–2882, 2020.

[48] X. Zheng, B. Aragam, P. K. Ravikumar, and E. P. Xing, “DAGs with NO TEARS: Continuous optimization for structure learning,” in Proceedings of Advances in Neural Information Processing Systems, 2018, pp. 9492–9503.

[49] R. Pamfil, N. Sriwattanaworachai, S. Desai, P. Pilgerstorfer, P. Beaumont, K. Georgatzis, and B. Aragam, “DYNOTEARS: Structure learning from time-series data,” in Proceedings of the International Conference on Artificial Intelligence and Statistics, 2020, pp. 1595–1605.

[50] G. D’Acunto, P. D. Lorenzo, and S. Barbarossa, “Multiscale causal structure learning,” Transactions on Machine Learning Research, 2023.

[51] I. Ng, A. E. Ghassami, and K. Zhang, “On the role of sparsity and DAG constraints for learning linear DAGS,” in Proceedings of Advances in Neural Information Processing Systems, 2020, pp. 17 943–17 954.

[52] P. Misiakos, C. Wendler, and M. Puschel, “Learning DAGs from data with few root causes,” in ¨ Proceedings of Advances in Neural Information Processing Systems, 2023, pp. 16 865–16 888.

[53] S. S. Saboksayr, G. Mateos, and M. Tepper, “Colide: Concomitant linear DAG estimation,” in Proceedings of the International Conference on Learning Representations, 2024.

[54] M. O. Jackson and Y. Zenou, “Games on networks,” in Handbook of Game Theory, vol. 4, Peyton Young and Shmuel Zamir (Eds.), 2014, pp. 95–163.

[55] C. Ballester, A. Calvo-Armengol, and Y. Zenou, “Who’s who in networks. Wanted: The key player,” ´ Econometrica, vol. 74, no. 5, pp. 1403–1417, 2006.

[56] Y. Leng, X. Dong, J. Wu, and A. Pentland, “Learning quadratic games on networks,” in Proceedings of the International Conference on Machine Learning, 2020, pp. 5820–5830.

[57] Y. Zhu, M. T. Schaub, A. Jadbabaie, and S. Segarra, “Network inference from consensus dynamics with unknown parameters,” IEEE Transactions on Signal and Information Processing over Networks, vol. 6, pp. 300–315, 2020.

[58] H.-T. Wai, A. Scaglione, B. Barzel, and A. Leshem, “Joint network topology and dynamics recovery from perturbed stationary points,” IEEE Transactions on Signal Processing, vol. 67, no. 17, pp. 4582–4596, 2019.

[59] R. Ramakrishna, H.-T. Wai, and A. Scaglione, “A user guide to low-pass graph signal processing and its applications: Tools and applications,” IEEE Signal Processing Magazine, vol. 37, no. 6, pp. 74–85, 2020.

[60] S. Batreddy, A. Siripuram, and J. Zhang, “Graph learning under spectral sparsity constraints,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2021, pp. 5405–5409.

[61] D. Thanou, X. Dong, D. Kressner, and P. Frossard, “Learning heat diffusion graphs,” IEEE Transactions on Signal and Information Processing over Networks, vol. 3, no. 3, pp. 484–499, 2017.

[62] S. Segarra, A. G. Marques, G. Mateos, and A. Ribeiro, “Network topology inference from spectral templates,” IEEE Transactions on Signal and Information Processing over Networks, vol. 3, no. 3, pp. 467–483, 2017.

[63] B. Pasdeloup, V. Gripon, G. Mercier, D. Pastor, and M. G. Rabbat, “Characterization and inference of graph diffusion processes from observations of stationary signals,” IEEE Transactions on Signal and Information Processing over Networks, vol. 4, no. 3, pp. 481–496, 2018.

[64] A. G. Marques, S. Segarra, G. Leus, and A. Ribeiro, “Stationary graph processes and spectral estimation,” IEEE Transactions on Signal Processing, vol. 65, no. 22, pp. 5911–5926, 2017.

[65] A. Einizade and S. H. Sardouie, “Learning product graphs from spectral templates,” IEEE Transactions on Signal and Information Processing over Networks, vol. 9, pp. 357–372, 2023.

[66] C. Zhang, Y. He, and H.-T. Wai, “Product graph learning from multi-attribute graph signals with inter-layer coupling,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2023.

[67] S. Batreddy, A. Siripuram, and J. Zhang, “Learning bipartite graphs from spectral templates,” Signal Processing, vol. 227, no. 109732, 2025.

[68] R. Shafipour and G. Mateos, “Online topology inference from streaming stationary graph signals with partial connectivity information,” Algorithms, vol. 13, no. 9, p. 228, 2020.

[69] M. Navarro, S. Rey, A. Buciulea, A. G. Marques, and S. Segarra, “Joint network topology inference in the presence of hidden nodes,” IEEE Transactions on Signal Processing, vol. 72, pp. 2710–2725, 2024.

[70] C. Zhang and H.-T. Wai, “Learning multiplex graph with inter-layer coupling,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2024, pp. 12 916–12 920.

[71] A. Buciulea, J. Ying, A. G. Marques, and D. P. Palomar, “Polynomial graphical lasso: Learning edges from Gaussian graph stationary signals,” IEEE Transactions on Signal Processing, vol. 73, pp. 1153–1167, 2025.

[72] H. E. Egilmez, E. Pavez, and A. Ortega, “Graph learning from filtered signals: Graph system and diffusion kernel identification,” IEEE Transactions on Signal and Information Processing over Networks, vol. 5, no. 2, pp. 360–374, 2019.

[73] R. Shafipour, S. Segarra, A. G. Marques, and G. Mateos, “Directed network topology inference via graph filter identification,” in Proceedings of the IEEE Data Science Workshop, 2018, pp. 210–214.

[74] A. Natali, M. Coutino, and G. Leus, “Topology-aware joint graph filter and edge weight identification for network processes,” in Proceedings of the IEEE International Workshop on Machine Learning for Signal Processing, 2020.

[75] S. Rey, V. M. Tenorio, and A. G. Marques, “Robust graph filter identification and graph denoising from signal ´ observations,” IEEE Transactions on Signal Processing, vol. 71, pp. 3651–3666, 2023.

[76] M. Sevilla and S. Segarra, “Bayesian topology inference on partially known networks from input-output pairs,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2024, pp. 9721–9725.

[77] X. Zheng, C. Dan, B. Aragam, P. Ravikumar, and E. P. Xing, “Learning sparse nonparametric DAGs,” in Proceedings of the International Conference on Artificial Intelligence and Statistics, 2020, pp. 3414–3425.

[78] Y. Yu, J. Chen, T. Gao, and M. Yu, “DAG-GNN: DAG structure learning with graph neural networks,” in Proceedings of the International Conference on Machine Learning, 2019, pp. 7154–7163.

[79] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” in Proceedings of the International Conference on Learning Representations, 2014.

[80] T. Kipf, E. Fetaya, K.-C. Wang, M. Welling, and R. Zemel, “Neural relational inference for interacting systems,” in Proceedings of the International Conference on Machine Learning, 2018, pp. 2688–2697.

[81] S. Lowe, D. Madras, R. Zemel, and M. Welling, “Amortized causal discovery: Learning to infer causal graphs from ¨ time-series data,” in Proceedings of the Conference on Causal Learning and Reasoning, 2022, pp. 509–525.

[82] N. R. Ke, S. Chiappa, J. Wang, A. Goyal, J. Bornschein, M. Rey, T. Weber, M. Botvinic, M. Mozer, and D. J. Rezende, “Learning to induce causal structure,” in Proceedings of the International Conference on Learning Representations, 2023.

[83] E. Rossi, F. Monti, Y. Leng, M. Bronstein, and X. Dong, “Learning to infer structures of network games,” in Proceedings of the International Conference on Machine Learning, 2022, pp. 18 809–18 827.

[84] J. Ying, X. Han, R. Zhou, X. Wang, and H. C. So, “Network topology inference with sparsity and Laplacian constraints,” in Proceedings of the IEEE International Conference on Information, Communication and Networks, 2023, pp. 283–288.

[85] S. K. Kadambari and S. P. Chepuri, “Product graph learning from multi-domain data with sparsity and rank constraints,” IEEE Transactions on Signal Processing, vol. 69, pp. 5665–5680, 2021.

[86] C. Shi and G. Mishne, “Learning Cartesian product graphs with Laplacian constraints,” in Proceedings of the International Conference on Artificial Intelligence and Statistics, 2024, pp. 2521–2529.

[87] M. A. Lodhi and W. U. Bajwa, “Learning product graphs underlying smooth graph signals,” arXiv preprint arXiv:2002.11277, 2020.

[88] S. Kumar, J. Ying, J. V. de M. Cardoso, and D. P. Palomar, “Structured graph learning via Laplacian spectral constraints,” in Proceedings of Advances in Neural Information Processing Systems, 2019.

[89] S. Rey, T. M. Roddenberry, S. Segarra, and A. G. Marques, “Enhanced graph-learning schemes driven by similar distributions of motifs,” IEEE Transactions on Signal Processing, vol. 71, pp. 3014–3027, 2023.

[90] E. Kolaczyk, Statistical analysis of network data with R. Springer, 2014.

[91] S. Chatterjee, P. Diaconis, and A. Sly, “Random graphs with a given degree sequence,” The Annals of Applied Probability, vol. 21, no. 4, p. 1400–1435, 2011.

[92] F. Chung and L. Lu, “Connected components in random graphs with given expected degree sequences,” Annals of combinatorics, vol. 6, no. 2, pp. 125–145, 2002.

[93] D. Lusher, J. Koskinen, and G. Robins, Exponential random graph models for social networks: Theory, methods, and applications. Cambridge University Press, 2013.

[94] S. Chatterjee and P. Diaconis, “Estimating and understanding exponential random graph models,” The Annals of Statistics, vol. 41, no. 5, pp. 2428–2461, 2013.

[95] M. Schweinberger and J. Stewart, “Concentration and consistency results for canonical and curved exponential-family models of random graphs,” The Annals of Statistics, vol. 48, no. 1, pp. 374–396, 2020.

[96] T. M. Roddenberry and S. Segarra, “Blind inference of eigenvector centrality rankings,” IEEE Transactions on Signal Processing, vol. 69, pp. 3935–3946, 2021.

[97] Y. He and H.-T. Wai, “Detecting central nodes from low-rank excited graph signals via structured factor analysis,” IEEE Transactions on Signal Processing, vol. 70, pp. 2416–2430, 2022.

[98] ——, “Online inference for mixture model of streaming graph signals with sparse excitation,” IEEE Transactions on Signal Processing, vol. 70, pp. 6419–6433, 2023.

[99] A.-L. Barabasi and R. Albert, “Emergence of scaling in random networks,” ´ Science, vol. 286, no. 5439, pp. 509–512, 1999.

[100] F. Gao and A. van der Vaart, “On the asymptotic normality of estimating the affine preferential attachment network models with random initial degrees,” Stochastic Processes and their Applications, vol. 127, no. 11, pp. 3754–3775, 2017.

[101] P. W. Holland, K. B. Laskey, and S. Leinhardt, “Stochastic blockmodels: First steps,” Social Networks, vol. 5, no. 2, pp. 109–137, 1983.

[102] T. A. Snijders and K. Nowicki, “Estimation and prediction for stochastic blockmodels for graphs with latent block structure,” Journal of Classification, vol. 14, no. 1, pp. 75–100, 1997.

[103] J.-J. Daudin, F. Picard, and S. Robin, “A mixture model for random graphs,” Statistics and Computing, vol. 18, no. 2, pp. 173–183, 2008.

[104] D. L. Sussman, M. Tang, D. E. Fishkind, and C. E. Priebe, “A consistent adjacency spectral embedding for stochastic blockmodel graphs,” Journal of the American Statistical Association, vol. 107, no. 499, pp. 1119–1128, 2012.

[105] K. Rohe, S. Chatterjee, and B. Yu, “Spectral clustering and the high-dimensional stochastic blockmodel,” The Annals of Statistics, vol. 39, no. 4, p. 1878–1915, 2011.

[106] L. Su, W. Wang, and Y. Zhang, “Strong consistency of spectral clustering for stochastic block models,” IEEE Transactions on Information Theory, vol. 66, no. 1, pp. 324–338, 2019.

[107] H.-T. Wai, S. Segarra, A. E. Ozdaglar, A. Scaglione, and A. Jadbabaie, “Blind community detection from low-rank excitations of a graph filter,” IEEE Transactions on Signal Processing, vol. 68, pp. 436–451, 2019.

[108] T. M. Roddenberry, M. T. Schaub, H.-T. Wai, and S. Segarra, “Exact blind community detection from signals on multiple graphs,” IEEE Transactions on Signal Processing, vol. 68, pp. 5016–5030, 2020.

[109] T. N. Kipf and M. Welling, “Variational graph auto-encoders,” in NeurIPS Workshop on Bayesian Deep Learning, 2016.

[110] M. Simonovsky and N. Komodakis, “GraphVAE: Towards generation of small graphs using variational autoencoders,” arXiv preprint arXiv:1802.03480, 2018.

[111] W. Jin, R. Barzilay, and T. Jaakkola, “Junction tree variational autoencoder for molecular graph generation,” in Proceeding of the International Conference on Machine Learning, 2018, pp. 2323–2332.

[112] C. Vignac and P. Frossard, “Top-N: Equivariant set and graph generation without exchangeability,” in Proceedings of the International Conference on Learning Representations, 2022.

[113] J. You, R. Ying, X. Ren, W. L. Hamilton, and J. Leskovec, “GraphRNN: Generating realistic graphs with deep autoregressive models,” in Proceedings of the International Conference on Machine Learning, 2018, pp. 5708–5717.

[114] C. Shi, M. Xu, Z. Zhu, W. Zhang, M. Zhang, and J. Tang, “GraphAF: A flow-based autoregressive model for molecular graph generation,” in Proceedings of the International Conference on Learning Representations, 2020.

[115] A. Bergmeister, K. Martinkus, N. Perraudin, and R. Wattenhofer, “Efficient and scalable graph generation through iterative local expansion,” in Proceedings of the International Conference on Learning Representations, 2024.

[116] D. Chen, M. Krimmel, and K. Borgwardt, “Flatten graphs as sequences: Transformers are scalable graph generators,” in Proceedings of Advances in Neural Information Processing Systems, 2025, pp. 69 071–69 109.

[117] N. De Cao and T. Kipf, “MolGAN: An implicit generative model for small molecular graphs,” in ICML Workshop on Theoretical Foundations and Applications of Deep Generative Models, 2018.

[118] K. Martinkus, A. Loukas, N. Perraudin, and R. Wattenhofer, “SPECTRE: Spectral conditioning helps to overcome the expressivity limits of one-shot graph generators,” in Proceedings of the International Conference on Machine Learning, 2022, pp. 15 159–15 179.

[119] C.-H. Lai, Y. Song, D. Kim, Y. Mitsufuji, and S. Ermon, “The principles of diffusion models,” arXiv preprint arXiv:2510.21890, 2025.

[120] C. Niu, Y. Song, J. Song, S. Zhao, A. Grover, and S. Ermon, “Permutation invariant graph generation via score-based generative modeling,” in Proceedings of the International Conference on Artificial Intelligence and Statistics, 2020, pp. 4474–4484.

[121] J. Jo, S. Lee, and S. J. Hwang, “Score-based generative modeling of graphs via the system of stochastic differential equations,” in Proceedings of the International Conference on Machine Learning, 2022, pp. 10 362–10 383.

[122] Q. Yan, Z. Liang, Y. Song, R. Liao, and L. Wang, “Swingnn: Rethinking permutation invariance in diffusion models for graph generation,” in Transactions on Machine Learning Research, 2024.

[123] C. Vignac, I. Krawczuk, A. Siraudin, B. Wang, V. Cevher, and P. Frossard, “DiGress: discrete denoising diffusion for graph generation,” in Proceedings of the International Conference on Learning Representations, 2023.

[124] K. K. Haefeli, K. Martinkus, N. Perraudin, and R. Wattenhofer, “Diffusion models for graphs benefit from discrete state spaces,” Proceedings of Learning on Graphs Conference, 2022.

[125] S. Luo, C. Shi, M. Xu, and J. Tang, “Predicting molecular conformation via dynamic graph score matching,” in Proceedings of Advances in Neural Information Processing Systems, 2021, pp. 19 784–19 795.

[126] M. Xu, L. Yu, Y. Song, C. Shi, S. Ermon, and J. Tang, “GeoDiff: A geometric diffusion model for molecular conformation generation,” in Proceedings of the International Conference on Learning Representations, 2022.

[127] Z. Xu, R. Qiu, Y. Chen, H. Chen, X. Fan, M. Pan, Z. Zeng, M. Das, and H. Tong, “Discrete-state continuous-time diffusion for graph generation,” in Proceedings of Advances in Neural Information Processing Systems, 2024, pp. 79 704–79 740.

[128] A. Siraudin, F. D. Malliaros, and C. Morris, “Cometh: A continuous-time discrete-state graph diffusion model,” arXiv preprint arXiv:2406.06449, 2024.

[129] J. Jo, D. Kim, and S. J. Hwang, “Graph generation with diffusion mixture,” in Proceedings of the International Conference on Machine Learning, 2024, pp. 22 371–22 405.

[130] L. Yang, Z. Huang, Z. Zhang, Z. Liu, S. Hong, W. Zhang, W. Yang, B. Cui, and L. Zhang, “Graphusion: Latent diffusion for graph generation,” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 11, pp. 6358–6371, 2024.

[131] A. Siraudin and C. Morris, “Principled latent diffusion for graphs via Laplacian autoencoders,” arXiv preprint arXiv:2601.13780, 2026.

[132] J. Su and S. Wu, “SBGD: Improving graph diffusion generative model via stochastic block diffusion,” in Proceedings of the International Conference on Machine Learning, 2025, pp. 57 112–57 130.

[133] L. Zhao, X. Ding, and L. Akoglu, “Pard: Permutation-invariant autoregressive diffusion for graph generation,” in Proceedings of Advances in Neural Information Processing Systems, 2024, pp. 7156–7184.

[134] T. Bernecker, G. Rehawi, F. P. Casale, J. Knauer-Arloth, and A. Marsico, “Random walk diffusion for efficient large-scale graph generation,” Transactions on Machine Learning Research, 2025.

[135] Y. Shou, W. Ai, T. Meng, and K. Li, “Graph diffusion models: A comprehensive survey of methods and applications,” Computer Science Review, vol. 59, p. 100854, 2026.

[136] M. Kaushalya, I. Katushiko, N. Kosuke, and A. Motoki, “GraphNVP: An invertible flow model for generating molecular graphs,” arXiv preprint arXiv:1905.11600, 2019.

[137] Y. Luo, K. Yan, and S. Ji, “Graphdf: A discrete flow model for molecular graph generation,” in Proceedings of the International Conference on Machine Learning, 2021, pp. 7192–7203.

[138] P. Lippe and E. Gavves, “Categorical normalizing flows via continuous transformations,” in Proceedings of the International Conference on Learning Representations, 2021.

[139] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in Proceedings of the International Conference on Learning Representations, 2023.

[140] A. Campbell, J. Yim, R. Barzilay, T. Rainforth, and T. Jaakkola, “Generative flows on discrete state-spaces: enabling multimodal flows with applications to protein co-design,” in Proceedings of the International Conference on Machine Learning, 2024, pp. 5453–5512.

[141] Y. Qin, M. Madeira, D. Thanou, and P. Frossard, “DeFoG: Discrete flow matching for graph generation,” in Proceedings of the International Conference on Machine Learning, 2025, pp. 50 269–50 326.

[142] F. Eijkelboom, G. Bartosh, C. A. Naesseth, M. Welling, and J.-W. van de Meent, “Variational flow matching for graph generation,” in Proceedings of Advances in Neural Information Processing Systems, 2024, pp. 11 735–11 764.

[143] K. Jiang, J. Cui, X. Dong, and L. Toni, “Bures-wasserstein flow matching for graph generation,” in Proceedings of the International Conference on Learning Representations, 2026.

[144] V. M. Tenorio, N. Zilberstein, S. Segarra, and A. G. Marques, “Graph guided diffusion: Unified guidance for conditional graph generation,” arXiv preprint arXiv:2505.19685, 2026.

[145] D. Roos, O. Davis, F. Eijkelboom, M. M. Bronstein, M. Welling, <sup>˙</sup>I. <sup>˙</sup>I. Ceylan, L. Ambrogioni, and J.-W. van de Meent, “Categorical flow maps,” in Proceedings of the International Conference on Machine Learning, 2026.

[146] T. Deleu, A. Gois, C. C. Emezue, M. Rankawat, S. Lacoste-Julien, S. Bauer, and Y. Bengio, “Bayesian structure learning´ with generative flow networks,” in Proceedings of the Conference on Uncertainty in Artificial Intelligence, 2022, pp. 518–528.

[147] T. Deleu, M. Nishikawa-Toomey, J. Subramanian, N. Malkin, L. Charlin, and Y. Bengio, “Joint bayesian inference of graphical structure and parameters with a single generative flow network,” Proceedings of Advances in Neural Information Processing Systems, pp. 31 204–31 231, 2023.

[148] K. Sharma, S. Kumar, and R. Trivedi, “Plug-and-play controllable graph generation with diffusion models,” in Proceedings of the International Conference on Machine Learning, 2024, pp. 44 545–44 564.

[149] M. Madeira, C. Vignac, D. Thanou, and P. Frossard, “Generative modelling of structurally constrained graphs,” in Proceedings of Advances in Neural Information Processing Systems, 2024, pp. 137 218–137 262.

[150] A. Bojchevski, O. Shchur, D. Zugner, and S. G¨ unnemann, “NetGAN: Generating graphs via random walks,” in¨ Proceeding of the International Conference on Machine Learning, 2018, pp. 610–619.

[151] R. Liao, Y. Li, Y. Song, S. Wang, C. Nash, W. L. Hamilton, D. Duvenaud, R. Urtasun, and R. S. Zemel, “Efficient graph generation with graph recurrent attention networks,” in Proceedings of Advances in Neural Information Processing Systems, 2019, pp. 4255–4265.

[152] S. Ghadimi and M. Wang, “Approximation methods for bilevel programming,” arXiv preprint arXiv:1802.02246, 2018.

[153] J. Kwon, D. Kwon, S. Wright, and R. D. Nowak, “A fully first-order method for stochastic bilevel optimization,” in Proceedings of the International Conference on Machine Learning, 2023, pp. 18 083–18 113.

[154] L. Franceschi, M. Niepert, M. Pontil, and X. He, “Learning discrete structures for graph neural networks,” in Proceedings of the International Conference on Machine Learning, 2019, pp. 1972–1982.

[155] W. Jin, Y. Ma, X. Liu, X. Tang, S. Wang, and J. Tang, “Graph structure learning for robust graph neural networks,” in Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2020, pp. 66–74.

[156] Y. Chen and L. Wu, “Graph neural networks: Graph structure learning,” in Graph neural networks: Foundations, frontiers, and applications. Springer, 2022, pp. 297–321.

[157] A. Raghuvanshi, G. Mateos, and S. P. Chepuri, “Task-driven heterophilic graph structure learning,” arXiv preprint arXiv:2512.23406, 2025.

[158] C. Zhang, S. Liu, H.-T. Wai, and A. M.-C. So, “Learning graph topology with functional priors via bilevel optimization,” arXiv preprint arXiv:2606.13885, 2026.

[159] A. Karaaslanli and S. Aviyente, “Simultaneous graph signal clustering and graph learning,” in Proceedings of th International Conference on Machine Learning, 2022, pp. 10 762–10 772.

[160] S. Batreddy, A. Siripuram, and J. Zhang, “Robust graph learning for classification,” Signal Processing, vol. 211, p. 109120, 2023.

[161] S. N. Sridhara, E. Pavez, and A. Ortega, “Towards joint graph learning and sampling set selection from data,” in Proceedings of the Asilomar Conference on Signals, Systems, and Computers, 2024, pp. 1168–1172.

[162] Q. Shao, H. Guo, J. Chen, D. Chen, and W. Yu, “From uniform to learned graph priors: Diffusion for structure discovery,” arXiv preprint arXiv:2606.11831, 2026.

[163] Y. Zhang, S. Pal, M. Coates, and D. Ustebay, “Bayesian graph convolutional neural networks for semi-supervised classification,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2019, pp. 5829–5836.

[164] P. Elinas, E. V. Bonilla, and L. Tiao, “Variational inference for graph convolutional networks in the absence of graph data and adversarial settings,” in Proceedings ofAdvances in Neural Information Processing Systems, 2020, pp. 18 648–18 660.

[165] A. Kazi, L. Cosmo, S.-A. Ahmadi, N. Navab, and M. M. Bronstein, “Differentiable graph module (DGM) for graph convolutional networks,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 2, pp. 1606–1617, 2022.

[166] J. J. C. Xian, S. Mahdavi, R. Liao, and O. Schulte, “From graph diffusion to graph classification,” in ICML Workshop on Structured Probabilistic Inference and Generative Modeling, 2024.

[167] W. Tang, H. Mao, D. Dervovic, I. Brugere, S. Mishra, Y. Xie, and J. Tang, “Cross-domain graph data scaling: A showcas with diffusion models,” Proceedings of Advances in Neural Information Processing Systems, pp. 123 619–123 648, 2025.

[168] Y. Wang, C. Liu, and Y. Yang, “Diffusion on graph: Augmentation of graph structure for node classification,” Transactions on Machine Learning Research, 2025.

[169] J. You, B. Liu, Z. Ying, V. Pande, and J. Leskovec, “Graph convolutional policy network for goal-directed molecular graph generation,” Proceedings of Advances in Neural Information Processing Systems, 2018.

[170] M. Wasserman, S. Sihag, G. Mateos, and A. Ribeiro, “Learning graph structure from convolutional mixtures,” Transactions on Machine Learning Research, 2023.

[171] K. Jiang, B. Tang, X. Dong, and L. Toni, “Heterogeneous graph structure learning through the lens of data-generating processes,” in Proceedings of the International Conference on Artificial Intelligence and Statistics, 2025, pp. 928–936.

[172] S. Rozada, V. K.B., A. Cavallo, A. Marques, H. Jamali-Rad, and E. Isufi, “Graph-aware diffusion for signal generation,” in Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing, 2026, pp. 461–465.

[173] Y. B. Uslu, S. Hadou, S. Rozada, S. S. Bidokhti, and A. Ribeiro, “Graph signal generative diffusion models,” in Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing, 2026, pp. 626–630.

[174] B. Tang, S. Chen, and X. Dong, “Learning hypergraphs from signals with dual smoothness prior,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2023.

[175] B. Tang, K. Jiang, L. Toni, S. Chen, and X. Dong, “A Markov random field model for hypergraph-based machine learning,” arXiv preprint arXiv:2308.14172v4, 2025.

[176] B. T. Brown, H. Zhang, D. L. Lau, and G. R. Arce, “Scalable hypergraph structure learning with diverse smoothness priors,” IEEE Transactions on Signal and Information Processing over Networks, vol. 11, pp. 1072–1086, 2025.

[177] I. Duta and P. Lio, “SPHINX: Structural prediction using hypergraph inference network,” in Proceedings of the International Conference on Machine Learning, 2025, pp. 14 884–14 901.

[178] A. Buciulea, E. Isufi, G. Leus, and A. G. Marques, “Learning graphs and simplicial complexes from data,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2024, pp. 9861–9865.

[179] ——, “Learning the topology of a simplicial complex using simplicial signals: A greedy approach,” in Proceedings of the IEEE Sensor Array and Multichannel Signal Processing Workshop, 2024.

[180] S. Gurugubelli and S. P. Chepuri, “Simplicial complex learning from edge flows via sparse clique sampling,” in Proceedings of the European Signal Processing Conference, 2024, pp. 2332–2336.

[181] C. Battiloro, I. Spinelli, L. Telyatnikov, M. M. Bronstein, S. Scardapane, and P. D. Lorenzo, “From latent graph to latent topology inference: Differentiable cell complex module,” in Proceedings of the International Conference on Learning Representations, 2024.

[182] C. Bodnar, F. D. Giovanni, B. P. Chamberlain, P. Lio, and M. M. Bronstein, “Neural sheaf diffusion: A topological perspective on heterophily and oversmoothing in GNNs,” in Proceedings of Advances in Neural Information Processing Systems, 2022, pp. 18 527–18 541.

[183] L. D. Nino, S. Barbarossa, and P. D. Lorenzo, “Learning sheaf Laplacian optimizing restriction maps,” in Proceedings of the Asilomar Conference on Signals, Systems, and Computers, 2024, pp. 59–63.

[184] L. D. Nino, G. D’Acunto, S. Barbarossa, and P. D. Lorenzo, “Learning the structure of connection graphs,” in Proceeding of IEEE International Conference on Acoustics, Speech and Signal Processing, 2026, pp. 76–80.

[185] J. Bamberger, F. Barbero, X. Dong, and M. M. Bronstein, “Bundle neural networks for message diffusion on graphs,” in Proceedings of the International Conference on Learning Representations, 2025.

[186] H. S. de Ocariz Borde, ´ A. Arroyo, I. Morales, I. Posner, and X. Dong, “Neural latent geometry search: Product manifold <sup>´</sup> inference via gromov-hausdorff-informed bayesian optimization,” in Proceedings of Advances in Neural Information Processing Systems, 2023, pp. 38 370–38 403.

[187] D. Yang, G. Yu, J. Wang, Z. Wu, and M. Guo, “Reinforcement causal structure learning on order graph,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2023, pp. 10 737–10 744.

[188] T. Jiralerspong, X. Chen, Y. More, V. Shah, and Y. Bengio, “Efficient causal graph discovery using large language models,” in ICLR Workshop on How Far Are We From AGI, 2024.

[189] A. Sanchez-Gonzalez, J. Godwin, T. Pfaff, R. Ying, J. Leskovec, and P. Battaglia, “Learning to simulate complex physics with graph networks,” in Proceedings of the International Conference on Machine Learning, 2020, pp. 8459–8468.

[190] R. Lam, A. Sanchez-Gonzalez, M. Willson, P. Wirnsberger, M. Fortunato, F. Alet, S. Ravuri, T. Ewalds, Z. Eaton-Rosen, W. Hu, A. Merose, S. Hoyer, G. Holland, O. Vinyals, J. Stott, A. Pritzel, S. Mohamed, and P. Battaglia, “Learning skillful medium-range global weather forecasting,” Science, vol. 382, no. 6677, pp. 1416–1421, 2023.

[191] J. L. Watson, D. Juergens, N. R. Bennett, B. L. Trippe, J. Yim, H. E. Eisenach, W. Ahern, A. J. Borst, R. J. Ragotte, L. F. Milles, B. I. M. Wicky, N. Hanikel, S. J. Pellock, A. Courbet, W. Sheffler, J. Wang, P. Venkatesh, I. Sappington, S. Vazquez Torres, A. Lauko, V. De Bortoli, E. Mathieu, S. Ovchinnikov, R. Barzilay, T. S. Jaakkola, F. DiMaio, M. Baek,´ and D. Baker, “De novo design of protein structure and function with RFdiffusion,” Nature, vol. 620, pp. 1089–1100, 2023.

[192] C. Zeni, R. Pinsler, D. Zugner, A. Fowler, M. Horton, X. Fu, Z. Wang, A. Shysheya, J. Crabb ¨ e, S. Ueda, R. Sordillo, ´ L. Sun, J. Smith, B. Nguyen, H. Schulz, S. Lewis, C.-W. Huang, Z. Lu, Y. Zhou, H. Yang, H. Hao, J. Li, C. Yang, W. Li,

R. Tomioka, and T. Xie, “A generative model for inorganic materials design,” Nature, vol. 639, pp. 624–632, 2025.