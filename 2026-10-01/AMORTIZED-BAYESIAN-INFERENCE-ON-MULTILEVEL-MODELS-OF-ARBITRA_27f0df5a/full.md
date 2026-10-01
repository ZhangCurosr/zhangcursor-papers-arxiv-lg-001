# AMORTIZED BAYESIAN INFERENCE ON MULTILEVEL MODELS OF ARBITRARY STRUCTURE

## A PREPRINT

Daniel Habermann Department of Statistics TU Dortmund University, Germany daniel.habermann@tu-dortmund.de

Stefan T. Radev Department of Cognitive Science Rensselaer Polytechnic Institute, USA

Andreas Bulling Institute for Visualisation and Interactive Systems University of Stuttgart, Germany

Paul-Christian Bürkner Department of Statistics TU Dortmund University, Germany paul.buerkner@gmail.com

September 30, 2026

## ABSTRACT

We develop a general method for amortized Bayesian inference on multilevel models of arbitrary structure. Given a generative model specified as a directed acyclic graph, our method automatically derives valid factorizations of the joint posterior and matching neural network architectures. The key steps, graph expansion and graph inversion, yield an inverse graph that determines how inference networks are stacked and conditioned, producing factorizations that amortize over the number of groups and the number of observations within each group. Unlike approaches that simplify the dependency structure to speed up learning or inference, our method preserves all conditional independence and exchangeability assumptions of the generative model. Across three case studies, it closely matches gold-standard samplers on models with more than 6,500 parameters while reducing inference to a near-instant forward pass once trained.

Keywords amortized Bayesian inference · simulation-based inference · neural posterior estimation · multilevel models · graph inversion

## 1 Introduction

Amortized Bayesian inference (ABI) approximates the posterior distribution of a Bayesian model by training neural networks on simulated pairs of parameters and datasets [1]. Because it only requires simulation and no likelihood evaluation, ABI enables Bayesian inference for models that are otherwise computationally intractable. In contrast to classical approaches, such as approximate Bayesian computation [ABC; 2], it scales much better to high-dimensional data [3–5] and is also amortized: the computational cost is only incurred once during training. This enables Bayesian inference for real-time applications [6] as well as inference across millions of independent datasets [7]. Consequently, ABI has become instrumental in a growing range of problems in the quantitative sciences [8–13].

Multilevel models, however, are particularly challenging to amortize. The dimension of their posterior depends on the number of groups and can vary from one dataset to the next, which is at odds with the fixed-width inputs and outputs assumed by standard neural approximators.

Recent neural network architectures relax the fixed-dimension requirement. Transformer-based approximators represent parameters and data as tokens and use masked attention to encode assumed dependencies, allowing arbitrary conditioning patterns [14]. Meta-amortized approaches generalize further, amortizing across families of structurally similar models with varying parameterizations, design matrices, and sample sizes [15]. Neither addresses the dependency structure of multilevel models. Attention masks must be pre-specified for each model and do not guaranty conditiona independencies, and meta-amortized architectures have so far been restricted to exchangeable observations, with exchangeable parameters left as an open problem.

Prior work [10, 16–19] has made progress on amortized inference for two-level models, deriving the network architecture by hand for that model class in each case. Deriving such an architecture is implicitly a choice of a particular factorization of the posterior: because each network has fixed input and output dimensions, a posterior whose dimension grows with the number of groups cannot be approximated in one piece but only as a product of factors that are learned separately and reused across groups. The choice of posterior factorization decides whether the approximator amortizes over the number of groups at all. Two questions therefore arise: Does such a factorization exist for a given generative model, and if so, which factors are amortizable?

The defining assumption of a multilevel model is that parameters at a given level are exchangeable: they are drawn from a common distribution, whose hyperparameters are themselves estimated from the data. This assumption allows partial pooling of information across groups, rather than fitting each group in isolation or ignoring the grouping structure entirely [20]. It also enables amortization: once we condition on the hyperparameters, a single neural network with shared weights can infer each group independently from its group-specific data alone [18, 19]. This requires a factorization of the joint posterior that makes the corresponding conditional independencies explicit, and not every factorization does. Even for a minimal two-level model, five valid factorizations exist (see Appendix A), of which only two enable amortization over the number of groups.

For multilevel models of arbitrary structure and complexity, we show that a suitable factorization of the posterior can be determined from the model’s generative graph alone. It represents the model as an annotated directed acyclic graph (DAG) from which data can be simulated. To derive this factorization, we first apply graph expansion, which splits each exchangeable node into two instances and thereby makes the exchangeability of groups explicit in the graph structure. Graph inversion then enumerates the valid inverse factorizations of the expanded graph, which allows us to identify those that amortize over the number of groups. Even when no such factorization exists, we recover amortization by selecting a factorization that leaves the fewest group-level parameters conditioned on other groups. Then, we infer the remaining parameters autoregressively.

In contrast to prior work, in which the network architecture is derived by hand for a particular model class, our method determines the complete approximator automatically: which posterior factorization to use, which inference and summary networks are required, and how they are stacked and conditioned. Even though our focus lies on multilevel models, this extends amortized Bayesian inference to any model that can be expressed as a DAG, including crossed designs, for which no amortized method has previously been available.

We demonstrate our method in three case studies of increasing structural complexity: a two-level model of the eight schools data, a three-level model of inter-rater agreement on skin lesion segmentations with crossed image and annotator effects, and a four-level model of the UK Breeding Bird Survey with more than 6,500 parameters. In all three, the derived approximators closely match the posteriors obtained by Stan.

An implementation of our method is available in the open-source BayesFlow library [21] as the GraphicalApproximator class.

## 2 Methods

The core of our method is a four-step transformation of the generative graph into a complete neural network architecture for ABI. First, graph expansion makes the exchangeability of groups explicit in the graph structure (Section 2.3). Second, graph inversion enumerates the valid factorizations of the posterior implied by the expanded graph (Sections 2.2 and 2.4). Third, amortization analysis compares the factorizations and selects the one that makes the most group-level parameters independently amortizable (Section 2.5). Fourth, architecture derivation turns the selected inverse graph into a concrete set of approximators (Sections 2.6 and 2.7): which nodes can share an inference network, how the networks are conditioned on one another, and how the observations are summarized for each of them.

## 2.1 Generative graphs and ancestral sampling

Almost all Bayesian models are generative, which means that they not only enable inference of parameter values θ conditioned on some observed data y, but also allow for generating simulated data conditioned on θ. This property is used extensively when following a principled Bayesian workflow [22], for example, in the form of posterior predictive checks [23] by comparing actually observed data to data simulated under the assumptions of a model, or when performing prior elicitation [24] by investigating the relationship between different prior specifications and hypothetical outcomes.

Let $G = ( V , E )$ denote a DAG with a set of nodes V and edges $E ,$ in which each node $v \in V$ represents a set of random quantities, and each edge encodes a conditional dependency. The nodes partition into latent nodes, whose values form the parameters $\theta ,$ and observed nodes, whose values form the data y. We refer to a latent node that is not a root as an interior node. Because $G$ is a DAG, the joint distribution factorizes as

$$
p ( \theta , y ) = \prod _ { v \in V } p ( v \mid \operatorname { P a } _ { G } ( v ) ) ,\tag{1}
$$

where $\operatorname { P a } _ { G } ( v )$ denotes the parents of $v$ in $G ,$ and nodes with $\mathrm { P a } _ { G } ( v ) = \varnothing$ are the root nodes, which are specified by their marginal prior distributions. Simulating new data can then be achieved by an ancestral sampling scheme: nodes are visited in topological order of $G ,$ starting from the roots and proceeding until the leaf nodes are reached.

For illustration purposes, we will use a simple two-level model as a running example. We apply the same method to more complex three- and four-level models in Section 3. The example model underlies the eight schools study [25], a meta-analysis of coaching effects across eight schools, which we analyze in full in Section 3.1. In this model, the population mean $\mu$ and population standard deviation τ inform group-level means $\lambda _ { j } \sim \mathrm { N o r m a l } ( \mu , \tau )$ , where $j$ is the group index. The group-level parameters $\lambda _ { j }$ in turn inform the observed data $y _ { j i } .$ , where the index i refers to individual observations in group $j .$ . The observations additionally depend on a shared observation-level standard deviation $\omega .$ . This model can be represented by the DAG shown in Figure 1 (left). The nodes $\lambda _ { j }$ and $y _ { j i }$ are marked with a dashed box, denoting that the groups are exchangeable. The factorization of the joint model for this example is given as:

$$
p ( \mu , \tau , \omega , \{ \lambda _ { j } \} , \{ y _ { j i } \} ) = p ( \mu ) p ( \tau ) p ( \omega ) \prod _ { j = 1 } ^ { J } p ( \lambda _ { j } \mid \mu , \tau ) p ( y _ { j } \mid \lambda _ { j } , \omega ) ,\tag{2}
$$

where $y _ { j }$ denotes all observations of group $j ,$ and the set notation $\{ \lambda _ { j } \}$ and $\{ y _ { j i } \}$ highlights that both the number of groups and the number of observations within groups can vary across datasets.

Two further annotations are required to simulate from the graph. First, each node v carries its conditional distribution $p ( v \mid \operatorname { P a } _ { G } ( v ) )$ . Second, each node carries a distribution $p ( N _ { v } )$ over the number of samples $N _ { v }$ it draws for each combination of its parents’ values. Because $N _ { v }$ is drawn anew for every such combination, groups within a dataset may differ in size. Thus, the networks encounter a range of group and dataset sizes during training and can be applied to any size at inference. An interior node that can draw more than one sample per combination of its parents’ values introduces a group index, which is carried by that node and by all of its descendants. We call such a node a grouping factor. A grouping factor is nested in another if it is a descendant of it, and two grouping factors are crossed if neither is nested in the other. For the two-level model, $N _ { \lambda }$ is the number of groups and $N _ { y }$ is the number of observations within a group. The node $\lambda _ { j }$ carries the group index j it contributes itself, while $y _ { j i }$ carries both $j$ and its own observation index i. For root nodes, which have no parents, $\bar { N _ { v } }$ is instead fixed to a common value across all roots, equal to the number of simulated datasets. Appendix B shows a full implementation of the two-level model.

## 2.2 Graph inversion

We describe graph inversion before graph expansion since inversion motivates why expansion is needed. Section 2.1 shows that simulating data from G follows a forward direction: nodes are visited in topological order, from marginal prior distributions along conditional dependencies to observed data. When performing parameter inference, the reverse is required: To infer the posterior $p ( \theta \mid y )$ , a corresponding graph must start from the observed data $y$ and reason back to the latent parameters θ. Following the nomenclature in the literature [26, 27], we call this process graph inversion, and the resulting posterior factorization an inversefactorization.

Computing such an inverse factorization is a non-trivial task, as it is not sufficient to simply reverse all edges. To illustrate this, consider $\mu$ and $\tau$ both pointing to $\{ \lambda _ { j } \} \colon \mu \to \{ \lambda _ { j } \}  \tau$ . Two parents with a common child form a collider: conditioning on the child makes them dependent, whereas conditioning on a common parent makes its children independent. Reversing the edges yields $\mu \left. \{ \lambda _ { j } \} \right. \tau$ , which incorrectly asserts independence of $\mu$ and $\tau$ given $\{ \lambda _ { j } \}$ . In multilevel models, for example, larger population variability in $\tau$ also implies larger uncertainty about the population mean in $\mu .$

To correctly capture this dependency, the inverse graph must additionally feature either an edge from $\mu$ to $\tau$ or from $\tau \ \mathrm { t o } \ \mu .$ . The former corresponds to first inferring $p ( \mu \mid \{ \lambda _ { j } \} )$ ) and then $\grave { p ( \tau \mid \mu , \{ \lambda _ { j } \} ) }$ ); the latter corresponds to first inferring $p ( \tau \mid \{ \lambda _ { j } \} )$ and then $p ( \boldsymbol { \mu } \mid \boldsymbol { \tau } , \{ \lambda _ { j } \} )$ . In practice, however, it is often computationally preferable to infer $\mu$ and τ jointly using a single inference network, learning $p ( \mu , \tau \mid \{ \lambda _ { j } \} )$ directly. This example also demonstrates that, for a given generative model, multiple inverse factorizations may exist.

Assuming µ and τ are inferred jointly, this model still has five valid inverse factorizations (Appendix A). Three of these factorizations retain $\{ \lambda _ { j } \}$ as a single factor and therefore require all groups to be inferred jointly, breaking amortization. The remaining two factorize the posterior over $\{ \lambda _ { j } \}$ into $\begin{array} { r } { \prod _ { j = 1 } ^ { J } p ( \lambda _ { j } \mid y _ { j } , \mu , \tau , \omega ) } \end{array}$ , allowing each group-level parameter to be inferred independently.

## 2.2.1 Algorithm

The example above shows that the choice of inverse factorization has direct consequences for amortization. Algorithms for computing an inverse factorization from the structure of a generative graph have been proposed by Stuhlmüller et al. [26] and Webb et al. [27]. Our method is agnostic to the chosen inversion algorithm. Here, we build on the algorithm of Stuhlmüller et al., which computes an inverse factorization by ordering nodes and adding them to the inverse graph, with dependencies determined by the original graph structure. Because each node ordering yields one inverse factorization, varying the ordering enumerates the inverse factorizations of a model, from which we then select the most suitable one according to Section 2.4.

Stuhlmüller et al. proposed this algorithm for Bayesian networks of fixed structure, in which the learned inverses amortize across different values observed at the same nodes and serve as proposals within a Monte Carlo sampler. They do not explicitly consider multilevel models. Applied to a multilevel model, their algorithm fixes the number of groups, as each $\lambda _ { j }$ becomes a separate node with its own inverse conditional, valid only for that particular J.

We therefore extend the algorithm in two ways: First, we expand exchangeable nodes before inversion (Section 2.3), which allows verifying that all groups admit the same inverse conditional. Second, we use the inverse graph not to sample from directly, but to determine the architecture of a set of neural approximators that together enable full amortized posterior inference (Section 2.4).

Given the generative graph G, the inverse graph H is constructed as follows: observed nodes are added first, and then latent nodes are processed one by one in a fixed order. When adding a node v to H, its parents are set to a minimal subset S of nodes already in H such that, given S, knowing the values of any additional node provides no further information about v. Formally, S d-separates v from H \ S in G.

The optimal factorization depends on both the graph structure and the number of groups at each level. The cost of graph inversion is independent of the data. Because graph expansion splits each exchangeable node into exactly two instances, the graph being inverted has a fixed size determined by the model specification. Since graph inversion is computationally negligible (on the order of milliseconds compared to minutes or hours for network training), all factorizations of even complex multilevel models can be enumerated, and the most suitable one selected without meaningful overhead (see Section 2.4).

## 2.3 Exchangeable nodes and graph expansion

Before applying a graph inversion algorithm, any graph containing exchangeable nodes must be expanded to make that exchangeability explicit in its structure. In a generative graph, an exchangeable node stands for an entire population of groups, and their exchangeability is represented only implicitly, through repeated sampling from that node. This suffices for simulation, but the inversion algorithm operates on the graph structure alone: it sees a single node and has no way to know how many groups it represents. D-separation would then force the group-level parameters to be conditioned jointly on the data of all groups, breaking amortization.

To resolve this, each interior node that can draw more than one sample per combination of its parents’ values is split into two instances. The inversion algorithm can then verify via d-separation that each instance conditions only on its own data and the global parameters, indicating that all J groups can be treated identically. By symmetry, every pair of instances has the same d-separation relations, so splitting into two instances is sufficient to reveal whether any instance depends on another.

Concretely, the graph expansion procedure is given in Algorithm 1. Figure 1 (second panel) shows the expanded version of the generative graph on the left.

## 2.4 Graph inversion with expanded graphs

The graph inversion algorithm detailed in Section 2.2 can now be directly applied to the expanded graph without modifications. Because each exchangeable node now appears as two independent instances, d-separation can determine whether a group-level node depends on data or parameters from outside its own group.

Algorithm 1 Graph expansion.   
Require: generative graph $G ,$ sample sizes $p ( N _ { v } )$   
Ensure: expanded graph $G ^ { \prime }$   
$\operatorname { D e } _ { G ^ { \prime } } ( v ) ;$ : descendants of v in $G ^ { \prime } ,$ excluding v itself   
1: $G ^ { \prime }  \dot { G }$   
2: Q ← nodes of $G ,$ topologically ordered   
3: while $Q \neq \emptyset$ do   
4: v ← pop first element of Q   
5: if v interior and $\mathrm { P r } ( N _ { v } > 1 ) > 0$ then   
6: D ← subgraph of G<sup>′</sup> induced by v and $\mathrm { D e } _ { G ^ { \prime } } ( v )$   
7: $E _ { \mathrm { i n } } \left. \{ \hat { u } \right. \hat { w } \in G ^ { \prime } : u \notin D , \overline { { w } } \in D \}$   
8: remove $\dot { D }$ and $E _ { \mathrm { i n } }$ from $\dot { G } ^ { \prime }$   
9: remove the nodes of $D$ from $Q$   
10: for $k \in \{ 1 , 2 \}$ do   
11: $D _ { k } $ copy of D with nodes $\{ w _ { k } : w \in D \}$   
12: add $D _ { k }$ to $\mathrm { \bar { \it G } } ^ { \prime }$   
13: for each $u  w \in E _ { \mathrm { i n } }$ do   
14: add edge u → w<sub>k</sub>   
15: append $\mathrm { D e } _ { G ^ { \prime } } ( v _ { k } )$ in topological order to $Q$

![](images/5e135afe38e5eca7028242f690a93b22feca5d48461be2d934795fb2b28111a1.jpg)  
Figure 1: Graph processing pipeline. Graph description: Directed acyclic graph of a two-level model with population mean $\mu ,$ population standard deviation $\tau ,$ group-level means $\{ \lambda _ { j } \}$ , observations $y _ { j i } .$ , and observation-level standard deviation ω. The dashed box denotes that the group-level means $\lambda _ { j }$ and observations $y _ { j i }$ are sampled independently. Graph expansion: The interior node $\lambda _ { j }$ is split into two conditionally independent instances $\lambda _ { 1 }$ and $\lambda _ { 2 } ,$ each receiving the same parent nodes $\mu$ and $\tau .$ . This makes the exchangeability of groups explicit in the graph structure, allowing the inversion to determine that each instance conditions only on its own data and the global parameters. Graph inversion: Inversion of the expanded two-level model using outer-nodes-first ordering $( \mu , \tau , \omega , \lambda _ { 1 } , \lambda _ { 2 } )$ . In the inverse graph, each $\lambda _ { j }$ conditions only on its own group data $y _ { j } = \{ y _ { j i } \}$ and the global parameters $\mu , \tau , \omega ,$ so the per-group inference becomes amortizable: a single inference network can be used for all $\dot { \boldsymbol { J } }$ groups. Network architecture: Architecture derived from the inverse graph. Each inference network has its own summary network. The local summary network encodes the observation of each group into a group-level summary, from which the local inference network infers $\lambda _ { j }$ The global summary network pools all observations into a single representation of the dataset, from which the global inference network infers $\mu , \tau ,$ , and ω. The dashed box denotes that the local summary and inference networks are applied independently for each group.

We call a latent node independently amortizable when none of its instances condition on another instance of the same node in the inverse graph. All instances of such a node share the same inverse conditional, so a single inference network with shared weights can infer every group independently, applied once per group. Global parameters, the root nodes of the generative graph, can be merged into a single node and inferred jointly by one inference network.

For the two-level model, inversion with the outer-nodes-first ordering yields the graph in Figure 1 (third panel). Each $\lambda _ { j }$ conditions on its own data $y _ { j }$ and on the global parameters $\mu , \tau ,$ , and $\omega ,$ and is therefore independently amortizable. Merging $( \mu , \tau , \omega )$ reduces the number of required inference networks from four to two: one for the global parameters and one shared across all $J$ groups.

In contrast, the inner-nodes-first ordering $\lambda _ { 1 } , \lambda _ { 2 } , \mu , \tau , \omega$ yields an inverse graph in which $\lambda _ { 2 }$ conditions on $\lambda _ { 1 }$ . The group-level parameters are then not independently amortizable, and no single inference network can infer each $\lambda _ { j }$ on its own.

## 2.5 Non-independent amortization

No ordering might yield an inverse factorization in which every grouping factor node is independently amortizable. The typical case is a crossed design, in which observations depend simultaneously on two or more non-nested grouping factors.

Let v and w be exchangeable nodes of two such factors, both parents of an observation node $y .$ After graph expansion, v and w appear as instances $v _ { 1 } , v _ { 2 }$ and $w _ { 1 } , w _ { 2 }$ . We write $y _ { 1 1 }$ for the observations that depend on $v _ { 1 }$ and $w _ { 1 } .$ , and $y _ { 2 1 }$ for those that depend on $v _ { 2 }$ and $w _ { 1 }$ . When constructing the inverse graph, observed nodes enter first since their values are known without inference and are therefore available to condition every node that follows. Conditioning on the observations opens the colliders at $y , { \bf s o } v _ { 1 }$ and $v _ { 2 }$ remain dependent through the path $v _ { 1 } \right. y _ { 1 1 } \left. w _ { 1 } \right. y _ { 2 1 } \left. v _ { 2 }$ This path is blocked only by conditioning on $w _ { 1 }$ , that is, only if w enters the inverse graph before v. $\boldsymbol { \mathrm { B y } }$ symmetry, the same holds for $w _ { 1 }$ and $w _ { 2 } \operatorname { i f } v _ { 1 }$ was added first. Thus, the instances of whichever factor enters first remain dependent and must be conditioned on one another, whereas the instances of the second factor are d-separated given the first and can be inferred independently.

For crossed factors, a factorization of the posterior can therefore either make one factor independently amortizable or the other, but never both. Inferring the instances of the remaining factor jointly would fix the output dimension, preventing amortization over the number of groups. Thus, we instead infer them autoregressively with a single shared set of weights.

Let $v _ { 1 } , \ldots , v _ { K }$ denote the instances of a latent node that is not independently amortizable, and let $c _ { k }$ collect the conditions that the inverse graph assigns to $v _ { k }$ apart from the other instances of the same node, with $\boldsymbol { c } = \left( c _ { 1 } , \dots , c _ { K } \right)$ For any ordering of the instances, the chain rule of conditional probability gives

$$
p ( v _ { 1 } , \dots , v _ { K } \mid c ) = \prod _ { k = 1 } ^ { K } p ( v _ { k } \mid v _ { 1 : k - 1 } , c _ { k } ) ,\tag{3}
$$

where $v _ { 1 : k - 1 } = \left( v _ { 1 } , \ldots , v _ { k - 1 } \right)$ and $v _ { 1 : 0 } = \emptyset$ . Each factor in Equation (3) is a density over a single instance $v _ { k } ,$ whose dimension does not depend on $K$ . Thus, a single inference network can represent all of them. In contrast, their conditions are not of fixed size: the data and group-level parameters in $c _ { k }$ vary across datasets, and the number of preceding instances $v _ { 1 : k - 1 }$ grows with $k .$

To rectify this, we compress the conditions of each step into a fixed-size summary. Because the groups are linked through the crossed factor, the observations of every group are informative about $v _ { k } . \ \mathbf { A }$ permutation-invariant set network therefore summarizes the observations of all groups, encoded as described in Section 2.6, together with the preceding $v _ { 1 : k - 1 }$ . Its output then has the same size for every step and every number of groups, so a single inference network can approximate each factor of Equation (3).

Using this architecture, sequential inference does not force sequential evaluation during training. Because the true values of $v _ { 1 : k - 1 }$ are provided by the simulator, all K outputs are computed in a single pass. Training is nonetheless more expensive than for an independently amortizable node, since the summary network summarizes a set of K instances at each of the $K$ steps, which scales quadratically with K. Only sampling is sequential, and drawing from the approximate posterior therefore requires K passes through the inference network.

The choice of posterior factorization can now be made based on computational efficiency. Both the cost of the summary network during training and the number of passes during sampling grow with the number of instances of the sequentially inferred node. Because all factorizations can be enumerated at negligible cost (Section 2.2), we select the one that leaves the fewest instances to be inferred sequentially. This information is available in advance from the sample size distributions $p ( N _ { v } )$ provided as part of the model specification. In a crossed design, this amortizes over the grouping factor with the most groups and infers the other sequentially.

## 2.6 Summary networks

Often in amortized Bayesian inference, so-called summary networks are trained jointly with the inference networks [9]. Their purpose is three-fold: First, by learning to compress raw data into informative summary statistics, they provide the inference networks with fixed-width conditions. Second, they are used to present the inference networks with a representation that is aligned with the structure of the data. For example, set-based summary networks like DeepSets [28] or SetTransformers [29] can be used to treat observations as exchangeable, or architectures such as Time Series Transformers [30] can encode time series. Third, using a summary network increases computational efficiency if a lower-dimensional representation of the data suffices to inform posterior inference [2].

However, in a multilevel model, such summary networks are not applicable without modifications because the observations do not form a single exchangeable set. Observations are exchangeable within a group, and the groups themselves are exchangeable. Consequently, a summary of the observations must reflect this structure.

Which encodings respect such a structure has been characterized for two basic cases: Hartford et al. [31] show that a linear layer that is equivariant to permutations of the groups of crossed grouping factors must combine each observation with its means over each grouping factor and over all observations. For nested grouping factors, Wang et al. [32] show that every linear equivariant layer must pool observations at each level of the hierarchy and then broadcast the result back to the individual observations. The encodings of both crossed and nested grouping factors are thus built by pooling over groups of observations, and the model structure defines over which groups the pooling occurs.

We build this encoding for arbitrary generative graphs. As with the conditions of the inference networks, the generative graph fully defines which grouping factors are nested or crossed, and thus, over which groups the encoding has to be pooled.

The observations can be grouped by any combination of the group indices they carry, with one restriction: a nested grouping factor is only exchangeable within the grouping factors it is nested in. This is because the index of the inner grouping factor refers to a different entity for each index of the outer grouping factor. Concretely, for the breeding bird survey in Section 3.3, in which squares are nested within regions, the square with index i in region j is not identical to the square with index i in region k.

Each inference network receives its own summary network because the parameters it conditions on and the level to which the observations must be pooled differ between networks. Consider an inference network for a node v that is conditioned on observations and a set of parameters. We summarize these conditions in three steps. First, each observation $y _ { o }$ is concatenated with the parameter values from which it was simulated, restricted to the conditions of v. We denote the resulting extended observations as $e _ { o }$ . This ensures that every condition enters the summary together with the observations to which it relates. Second, each extended observation $e _ { o }$ is further extended by the log sizes of the groups to which it belongs and is then encoded by a stack of exchangeable layers,

$$
e _ { o } \gets \sigma \Big ( W e _ { o } + b + \sum _ { u } W _ { u } \bar { e } _ { u , o } \Big ) ,\tag{4}
$$

where the sum runs over all groupings u, and $\bar { e } _ { u , o }$ is the mean of the current encoding over the observations that fall into the same group as o under grouping u. That is, each layer updates an observation with the averages over every group to which it belongs. For the breeding bird survey, the count of a square in a given year is thus combined with the averages over the entire dataset, its region, its square, its year, and that region in that year.

Third, the encoded observations are pooled to the level of v: for every instance $v _ { k } .$ , all encoded observations that descend from $v _ { k }$ are averaged, and their log number is appended. This is necessary because averaging discards the number of observations. Finally, we also directly append the conditioned parameters that are ancestors of v. They are already contained in the summary, but appending them passes them to the inference network unchanged and allows conditioning the inference network for groups without observations. The result now has a fixed size, regardless of the number of groups and observations, and serves as the conditions of the inference network.

## 2.7 Network architecture

The network architecture follows directly from the selected inverse graph H, corresponding to one factorization of the joint posterior. The observation nodes are the roots of H: their values are known, so they require no network. For every other node, the nodes pointing to its instances in H are its conditions.

H also determines how these nodes are grouped into networks. They are sorted into stages based on when their conditions become available: the first stage holds the nodes that condition only on observations, and each subsequent stage holds the nodes whose conditions are inferred at earlier stages. Each node belongs to exactly one stage, and each stage is assigned one inference network, conditioned on the union of its nodes’ parents.

Summary and inference networks are trained jointly on simulated pairs $( \theta , y )$ , with the parameter conditions taken from the simulator. Writing $\phi _ { n }$ for the weights of network $n ,$ collected in $\phi ,$ , and $\psi _ { n }$ for those of its summary network collected in $\psi ,$ the networks are trained by minimizing the negative log density of the simulated parameters,

$$
\mathcal { L } ( \phi , \psi ) = - \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \sum _ { n } \sum _ { i } \log q _ { \phi _ { n } } \big ( \theta _ { n , i } ^ { ( m ) } \ | \ h _ { \psi _ { n } } ( c _ { n , i } ^ { ( m ) } ) \big ) ,\tag{5}
$$

where $h _ { \psi _ { n } } ( c _ { n , i } ^ { ( m ) } )$ is the fixed-size summary of the conditions of instance $i ,$ m indexes the M simulated datasets (which may vary in their number of groups and observations), the second sum runs over the inference networks, and the third runs over the instances at which each is applied. The loss $\mathcal { L } ( \phi , \psi )$ can be optimized by standard backpropagation, and for a sufficiently large simulation budget, minimizing $\mathcal { L } ( \phi , \psi )$ minimizes the expected forward Kullback-Leibler divergence between the approximate and the true posterior [33].

During training, the inference networks n can be evaluated in any order since all parameters and observations are known. During sampling, however, the inference networks need to be evaluated in the order in which their conditions become available.

## 3 Case studies

We evaluate our method on three applications based on real-world datasets with increasingly complex multilevel structures:

1. A two-level model of the “eight schools” data of Rubin [25], simple enough that its posterior factorizations can still be derived by hand.

2. A three-level beta regression model of inter-rater agreement on the ISIC Archive Multi-Annotator Dermoscopic Skin Lesion Segmentation Dataset [34], with crossed image and annotator effects.

3. A four-level negative binomial model of yearly bird counts collected in the UK Breeding Bird Survey [35], in which nested region and square effects are crossed with a year effect.

For each case study, we apply a series of model checks. Computational validity, which can be compromised, for example, by incomplete network training, is assessed using simulation-based calibration [36, 37]. Inferential accuracy is assessed by posterior predictive checks [23] and comparisons to Stan [38] as a gold-standard sampler. Finally, the partial pooling of the group-level parameters $\lambda _ { j }$ toward their hierarchical mean is assessed by a pooling factor [39].

## 3.1 Eight schools

As a first example, we analyze the eight-schools data of Rubin [25], the running example of Section 2. Eight schools independently conducted randomized experiments to estimate the effect of a coaching program on SAT scores. For each school $j ,$ the study reports an estimated treatment effect $y _ { j }$ together with its standard error $\sigma _ { j }$ , which is treated as known. Because the school-specific effects $\lambda _ { j }$ are assumed to be drawn from a common population, they are modeled hierarchically:

$$
\begin{array} { r } { \mu \sim \mathrm { N o r m a l } ( 0 , 5 ) , \tau \sim \mathrm { N o r m a l } ^ { + } ( 0 , 2 0 ) } \\ { \lambda _ { j } \sim \mathrm { N o r m a l } ( \mu , \tau ) , y _ { j } \sim \mathrm { N o r m a l } ( \lambda _ { j } , \sigma _ { j } ) , } \end{array}\tag{6}
$$

where $\mu$ is the population mean effect and $\tau$ is the between-school standard deviation. Since the standard errors are part of the observed data rather than quantities to be inferred, we simulate them alongside the treatment effects as $\sigma _ { j } \sim \mathrm { N o r m a l } ^ { + } ( 1 5 , 5 )$ , which covers the range of standard errors reported in the original study.

The generative graph of Equation $( 6 )$ contains a root node holding $( \mu , \tau )$ , an exchangeable node holding $\lambda _ { j } ,$ and a data node holding $( y _ { j } , \sigma _ { j } )$ . Because $\mu$ and τ are both roots, they carry no group index and are merged into a single node inferred by one network. Of the two inverse factorizations of this graph, our algorithm selects $\begin{array} { r } { p ( \mu , \tau \mid y ) \prod _ { j = 1 } ^ { J } p ( \lambda _ { j } \mid } \end{array}$ $\mu , \tau , y _ { j } )$ , which amortizes over the number of schools.

## 3.1.1 Network architecture

The approximator derived from the inverse graph consists of a single summary network, which pools the per-school observations, and two inference networks: one for the merged root node $( \mu , \tau )$ and one for the school effects $\lambda _ { j }$ . We use a SetTransformer [29] with a summary dimension of 16 as the summary network, and affine coupling flows [40] with a depth of 12 as the inference networks.

## 3.1.2 Model training

We generate training data by ancestral sampling from Equation $( 6 ) ,$ , using a simulation budget of M = 100 000 datasets. The number of schools is drawn uniformly from [1, 100] for each simulated dataset, inference on the original data is then performed at $J = 8$ . The parameter τ and the observed standard errors $\sigma _ { j }$ are log-transformed before training. The networks are trained jointly for 100 epochs with a batch size of 4096 using the Adam optimizer [41] with a cosine decay schedule from an initial learning rate of $1 0 ^ { - 3 }$

## 3.1.3 Results

Figure 2 (top row) compares marginal posterior intervals for $\mu , \tau$ , and the school-specific effects $\lambda _ { j }$ on the original data of Rubin [25] against a reference implementation in Stan [38]. The estimates agree closely, showing that the architecture derived from the inverse graph recovers the correct posterior. Network training takes about a minute on a standard desktop computer, whereas drawing 4000 independent posterior samples takes about 0.04 s, approximately as long as Stan requires to reach an effective sample size of 4000 for this model.

## 3.2 Skin Lesions

To demonstrate our method on a three-level model with a crossed design, we analyze the ISIC Archive Multi-Annotator Dermoscopic Skin Lesion Segmentation Dataset [34]. The dataset consists of dermoscopic images of skin lesions, each of which was segmented by several expert annotators who separated the region of the lesion from the surrounding skin. Because each image was annotated by multiple annotators and each annotator segmented multiple images, image and annotator effects are crossed rather than nested.

We model how strongly two annotators agree on a given image. For each image with at least two annotators, we compute the Dice similarity coefficient between the masks of every pair of annotators, yielding one response in (0, 1) per image and annotator pair. Because the response is a proportion, we model it using a beta regression with crossed image and annotator effects:

$$
\begin{array} { r l r } & { \alpha \sim \mathrm { N o r m a l } ( 1 , 1 ) , } & { \beta _ { \mathrm { T 3 } } , \beta _ { \mathrm { m i x } } \sim \mathrm { N o r m a l } ( 0 , 0 . 5 ) } \\ & { \gamma \sim \mathrm { L o g N o r m a l } ( \log 1 5 , 0 . 5 ) , \quad \mu _ { I } , \mu _ { A } \sim \mathrm { N o r m a l } ( 0 , 0 . 2 5 ) } \\ & { \sigma _ { I } \sim \mathrm { N o r m a l } ^ { + } ( 0 , 0 . 5 ) , } & { \sigma _ { A } \sim \mathrm { N o r m a l } ^ { + } ( 0 , 0 . 3 ) } \\ & { u _ { j } \sim \mathrm { N o r m a l } ( \mu _ { I } , \sigma _ { I } ) , } & { v _ { k } \sim \mathrm { N o r m a l } ( \mu _ { A } , \sigma _ { A } ) } \\ & { \theta _ { j k l } = \log \mathrm { i t } ^ { - 1 } ( \alpha + u _ { j } + v _ { k } + v _ { l } + \beta _ { \mathrm { T 3 } } c _ { k l } + \beta _ { \mathrm { m i x } } m _ { k l } ) } \\ & { y _ { j k l } \sim \mathrm { B e t a } ( \theta _ { j k l } \gamma , ~ ( 1 - \theta _ { j k l } ) \gamma ) } \end{array}\tag{7}
$$

where $y _ { j k l }$ is the observed agreement between annotators k and l on image $j ,$ and $\theta _ { j k l }$ is its expected value. The beta likelihood is parameterized by $\theta _ { j k l }$ and the precision γ, so a larger γ concentrates responses more tightly around $\theta _ { j k l }$ The intercept α is the population-level agreement on the logit scale. The image effect $u _ { j }$ and the annotator effect $v _ { k }$ are drawn around means $\mu _ { I }$ and $\mu _ { A }$ with standard deviations $\sigma _ { I }$ and $\sigma _ { A }$ . Because each response compares two annotators, the agreement between annotators k and l on image j depends on both $v _ { k }$ and $v _ { l }$

The dataset records which tool the annotators used to produce a segmentation: a manually traced polygon or a fully automated algorithm. Two covariates carry that information. The first, $c _ { k l } \in \{ 0 , 1 , 2 \}$ , counts how many of the two masks were produced automatically, so $\beta _ { \mathrm { T 3 } }$ is the change in agreement contributed by each automated mask. The second, $m _ { k l }$ , indicates that the two annotators used different tools, so $\beta _ { \mathrm { m i x } }$ captures the agreement lost because the boundaries were drawn by different procedures rather than because either was automated.

The generative graph of Equation (7) contains two root nodes, one holding the hyperparameters $\eta = \left( \mu _ { I } , \sigma _ { I } , \mu _ { A } , \sigma _ { A } \right)$ and the other holding the shared parameters $\xi = \left( \alpha , \gamma , \beta _ { \mathrm { T 3 } } , \beta _ { \mathrm { m i x } } \right)$ , two exchangeable nodes holding the image effects $u _ { j }$ and the annotator effects $v _ { k } .$ , respectively, and a data node holding $y _ { j k l } , c _ { k l } , m _ { k l }$ . Because both exchangeable nodes are parents of the same observation node, no factorization makes both of them independently amortizable (Section 2.5).

Eight Schools

![](images/9563d12eb09876c9c32f05a3c3fb4cab0172ba56f8480b2f31db29e0e590cd5b.jpg)  
Skin Lesions

![](images/77959a34fca159bde21efa34f078eb4409f572cc15631d08d05530092c985733.jpg)

![](images/7af7325f377525ae5c115d4f1865cf9b19cf30fc8bd20a8c91bc1e4d22612666.jpg)

![](images/04c2774ce067490141402aa2e70de66d8b46b2d583248726685eedb2082f193d.jpg)

![](images/26a01d031432a3f27adde1229342a07951b45721ea586cd7a9cdfe7e8861a09a.jpg)

UK Breeding Bird Survey  
![](images/4f0ac5c8bf1866165253849d75ca9a6831ad4a6f088a36ae0cf34c93adcb0591.jpg)

![](images/04f24a1d442c696a98f49ec933a371ba5b1400d95f3c079bc03ffc7fcdf10be9.jpg)

![](images/b055f2898fb673db9fd1b2ab5584237e27951b8e7a7adf6eb2a30e80c3224ce0.jpg)

![](images/277d5af5d6d5c414113cd28839a8f4303bc11382447974efe58cfe8ed2bb3158.jpg)  
Figure 2: Comparison of our method against Stan on the three case studies. Left: marginal posterior intervals of the global parameters, showing the posterior mean (point) and 90% credible intervals (line) under our approximator (blue) and Stan (red). Right: The posterior mean (grey) and posterior standard deviation (gold) of each group-level parameter under our method, plotted against the corresponding Stan estimate. The dashed line marks exact agreement and N is the number of groups.

Our algorithm selects the factorization that leaves the factor with the fewest instances to be inferred autoregressively,

$$
\begin{array} { r l r } {  { p ( \eta , \xi , \{ v _ { k } \} , \{ u _ { j } \} \mid y , c , m ) = p ( \eta , \xi \mid y , c , m ) } } \\ & { } & { \times \prod _ { k = 1 } ^ { K } p ( v _ { k } \mid v _ { 1 : k - 1 } , \eta , \xi , y , c , m ) \times \prod _ { j = 1 } ^ { J } p ( u _ { j } \mid \eta , \xi , \{ v _ { k } \} , y _ { j } , c , m ) , } \end{array}\tag{8}
$$

which amortizes over the $J = 2 3 9 4$ images and infers the $K = 1 5$ annotator effects autoregressively.

## 3.2.1 Network architecture

Three affine coupling flows [40] with a depth of 10 are used as inference networks: one for the merged root nodes η and ξ, and one for each of the two sets of group-level effects. Their conditions are reduced by five summary networks with a summary dimension of 128: a two-step chain (Section 2.6) over the observation tensor, a per-image summary of the same tensor, a pooling of the annotator effects, and an encoder for the preceding draws $v _ { 1 : k - 1 }$

## 3.2.2 Model training

We generate training data by ancestral sampling from Equation (7), using a simulation budget of $M = 2 0 0 0 0 0$ datasets. We vary the number of annotators K uniformly in $K \in [ 1 , 2 5 ]$ and the number of images J uniformly in $J \in [ 1 , 2 5 0 0 ]$ In the observed data, not every image is annotated by every annotator. We replicate this sparsity pattern by randomly masking the observed data with a binary indicator of whether the annotator segmented that image. Both γ and the two group-level standard deviations are log-transformed before training. The networks are trained jointly for 200 epochs with a batch size of 64 using the Adam optimizer [41] with a cosine decay schedule from an initial learning rate of $1 0 ^ { - 3 }$

## 3.2.3 Results

Figure 2 (center row) compares the marginal posterior intervals of the global parameters with those obtained by Stan [38], together with the posterior means and standard deviations of all image and annotator effects. The estimates obtained by our model agree well with the gold-standard results. Network training takes about one and a half hours on a desktop GPU, whereas drawing 4000 independent posterior samples takes about 2 s, compared to about 13 min for Stan to reach an effective sample size of 4000. This is a type of model that can already benefit from amortized inference; for example, if many model refits are required for cross validation. With these timings, the training cost is recovered after about seven model refits.

## 3.3 UK Breeding Bird Survey

As an example of a four-level model, we analyze the UK breeding bird survey [35]. For this data, volunteers walk two transects across a 1 km grid square twice per breeding season and record every bird they encounter, so the survey yields two counts per square per year. Squares are nested within survey regions, while the year of the survey applies to every square. Region and square effects are therefore crossed with a year effect.

We analyze the counts of the European robin. In accordance with the survey guidelines, we take the larger of the two visit counts for each square and year, and we treat a square that was surveyed without recording a robin as having a count of zero. We exclude the 628 of 7 072 surveyed squares in which no robin was recorded in any year. The resulting dataset comprises $N = 8 9 8 6 5$ observations on 6 444 squares in 82 regions over the 31 years from 1994 to 2024. That is, 45% of all square-year combinations are observed: regions contain between 4 and 246 squares, and a square was surveyed in between 1 and 31 years. A small number of squares cover twice the standard area, which we account for by an offset $o _ { r s t } = \log 2$ on the log scale. Because the counts are overdispersed, we model them with a three-level negative binomial regression,

$$
\begin{array} { r l } & { \quad \sigma _ { r } , \sigma _ { s } , \sigma _ { t } , \kappa \sim \mathrm { N o r m a l } ^ { + } ( 0 , 1 ) } \\ & { \quad \mu \sim \mathrm { N o r m a l } ( 0 , 5 ) , \quad \quad \quad \beta _ { t } \sim \mathrm { N o r m a l } ( 0 , \sigma _ { t } ) } \\ & { \alpha _ { r } \sim \mathrm { N o r m a l } ( \mu , \sigma _ { r } ) , \quad \quad \alpha _ { r s } \sim \mathrm { N o r m a l } ( \alpha _ { r } , \sigma _ { s } ) } \\ & { y _ { r s t } \sim \mathrm { N e g B i n o m i a l } \big ( \exp ( \alpha _ { r s } + \beta _ { t } + o _ { r s t } ) , \kappa \big ) , } \end{array}\tag{9}
$$

where $\mathrm { N e g B i n o m i a l } ( m , \kappa )$ denotes the negative binomial distribution with mean m and variance $m + \kappa ^ { 2 } m ^ { 2 }$ . Here, $\mu$ is the mean log-intensity, $\alpha _ { r }$ is the effect of region $r , \alpha _ { r s }$ is the effect of square s within region $r ,$ and $\beta _ { t }$ is the effect of year t. The year effects are centered at zero, so that $\mu$ carries the overall level and $\beta _ { t }$ describes the deviation of each year from it.

The generative graph of Equation (9) contains a root node holding $( \mu , \sigma _ { r } , \sigma _ { s } , \sigma _ { t } , \kappa )$ , three exchangeable nodes holding the region, square, and year effects, respectively, and a data node holding $\left( y _ { r s t } , o _ { r s t } \right)$ . The square node is a descendant of the region node, whereas the year node shares only the root node with them, and both reach the same observations. Writing $\xi = ( \mu , \sigma _ { r } , \sigma _ { s } , \sigma _ { t } , \kappa )$ for the global parameters, our algorithm selects the factorization

$$
\begin{array} { l } { { \displaystyle p ( \xi , \{ \beta _ { t } \} , \{ \alpha _ { r } \} , \{ \alpha _ { r s } \} \mid y ) = p ( \xi \mid y ) \prod _ { t = 1 } ^ { T } p ( \beta _ { t } \mid \beta _ { 1 : t - 1 } , \xi , y ) } } \\ { { \displaystyle ~ \times \prod _ { r = 1 } ^ { R } p ( \alpha _ { r } \mid \xi , \{ \beta _ { t } \} , y ) \prod _ { r , s } p ( \alpha _ { r s } \mid \xi , \alpha _ { r } , \{ \beta _ { t } \} , y ) . } } \end{array}\tag{10}
$$

The factorization amortizes over the $R = 8 2$ regions and the 6 444 squares, leaving only the $T = 3 1$ year effects to be estimated autoregressively by the mechanism of Section 2.5. Sampling therefore requires T passes through the sequential inference network, in addition to one pass for the global, region, and square effects, respectively.

## 3.3.1 Network architecture

The approximator derived from Equation (10) consists of four affine coupling flows [40]: one for the global parameters ξ with a depth of 8 and hidden layers of width 256, and one each for the year, region, and square effects with a depth of 6 and hidden layers of width 128. Their conditions are reduced by eight DeepSets [28].

## 3.3.2 Model training

We generate training data by ancestral sampling from Equation (9), using a simulation budget of $M = 1 0 0 0 0 0 0$ datasets. Each dataset has between 1 and 100 regions with up to 250 squares each, observed over $T = 3 1$ years. Counts are passed to the networks as $\log ( 1 + y _ { r s t } )$ together with the offset. The networks are trained jointly for 100 epochs with a batch size of 2 and gradients accumulated over 4 batches, using the Adam optimizer [41] with a cosine decay schedule from an initial learning rate of $1 0 ^ { - 3 } ~ \mathrm { t o } ~ 1 0 ^ { - 5 }$ and a global gradient norm clip of 1.

Counts are passed to the networks as $\log ( 1 + y _ { r s t } )$ together with the offset. The networks are trained jointly for 200 epochs with a batch size of 512 using the Adam optimizer [41] with a cosine decay schedule from an initial learning rate of $1 0 ^ { - 3 }$

## 3.3.3 Results

Figure 2 (bottom row) compares marginal posterior estimates of the neural approximator with those obtained by Stan. Generally, the posterior estimates agree well with the gold-standard sampler results. One exception is $\sigma _ { r }$ and $\sigma _ { s } ,$ which are slightly too low. Network training takes about 12 hours on a desktop GPU, whereas sampling an effective sample size of 4000 draws takes about 4 s, compared with about 35 min for Stan.

## 4 Discussion

We have developed a method that derives neural architectures for amortized Bayesian inference on arbitrary multilevel models automatically from the model specification. Graph expansion and graph inversion reduce the choice of posterior factorization to a question of d-separation, and the resulting inverse graph determines the full architecture. Every conditional independence and exchangeability assumption of the generative model is retained. In crossed designs, where no factorization makes every grouping factor independently amortizable, autoregressive inference keeps the approximator amortized over the number of groups.

Across three case studies, the derived approximators closely matched Stan on models with more than 6,500 parameters, while reducing inference to a near-instant forward pass. Although these models have tractable likelihoods to allow the comparison with Stan, the method does not require one and thus also applies to purely simulation-based multilevel models.

However, amortization over the number of groups typically holds only within the range seen during training, so models with many groups require training on many groups. Combining our method with compositional score matching [42] could relax this requirement. Further, as any method trained on simulations, the approximators may become inaccurate under model misspecification [43]. Methods such as self-consistency losses [44] improve robustness in such out-of-simulation settings and, since they modify the training objective rather than the architecture, can in principle be combined with out approach.

Together, these results establish graph-based derivation as a principled and general route to amortized inference of multilevel models of arbitrary structure.

## Acknowledgments

This work was supported by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) Projects 508399956, 528702768, and 569534706. Paul-Christian Bürkner acknowledges support from DFG Collaborative Research Center 391 (Spatio-Temporal Statistics for the Transition of Energy and Transport) – 520388526. Stefan T. Radev was funded by the National Science Foundation under Grant No. 2448380.

## References

[1] Andrew Zammit-Mangion, Matthew Sainsbury-Dale, and Raphaël Huser. Neural methods for amortized inference. Annual Review of Statistics and Its Application, 12(1):311–335, March 2025. ISSN 2326-831X. doi: 10.1146/ annurev-statistics-112723-034123.

[2] Jean-Michel Marin, Pierre Pudlo, Christian P Robert, and Robin J Ryder. Approximate Bayesian computational methods. Statistics and computing, 22(6):1167–1180, 2012.

[3] Lars Dingeldein, Pilar Cossio, and Roberto Covino. Simulation-based inference of single-molecule experiments. Current Opinion in Structural Biology, 91:102988, 2025. ISSN 0959-440X. doi: 10.1016/j.sbi.2025.102988.

[4] Lingyi Zhou, Stefan T Radev, William H Oliver, Aura Obreja, Zehao Jin, and Tobias Buck. Bridging simulations and observations: New insights into galaxy formation simulations via out-of-distribution detection and Bayesian model comparison: Evaluating galaxy formation simulations under limited computing budgets and sparse dataset sizes. Astronomy & Astrophysics, 701:A44, 2025.

[5] Rafael Orozco, Ali Siahkoohi, Mathias Louboutin, and Felix J Herrmann. ASPIRE: iterative amortized posterior inference for Bayesian inverse problems. Inverse Problems, 41(4):045001, 2025.

[6] Maximilian Dax, Stephen R. Green, Jonathan Gair, Nihar Gupte, Michael Pürrer, Vivien Raymond, Jonas Wildberger, Jakob H. Macke, Alessandra Buonanno, and Bernhard Schölkopf. Real-time inference for binary neutron star mergers using machine learning. Nature, 639(8053):49–53, March 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-08593-z.

[7] Mischa von Krause, Stefan T. Radev, and Andreas Voss. Mental speed is high until age 60 as revealed by analysis of over a million participants. Nature Human Behaviour, 6(5):700–708, February 2022. ISSN 2397-3374. doi: 10.1038/s41562-021-01282-7.

[8] Pedro J. Gonçalves, Jan-Matthis Lueckmann, Michael Deistler, Marcel Nonnenmacher, Kaan Öcal, Giacomo Bassetto, Chaitanya Chintaluri, William F. Podlaski, Sara A. Haddad, Tim P. Vogels, David S. Greenberg, and Jakob H. Macke. Training deep neural density estimators to identify mechanistic models of neural dynamics. eLife, 9, September 2020. ISSN 2050-084X. doi: 10.7554/elife.56261.

[9] Stefan T. Radev, Frederik Graw, Simiao Chen, Nico T. Mutters, Vanessa M. Eichel, Till Bärnighausen, and Ullrich Köthe. OutbreakFlow: Model-based Bayesian inference of disease outbreak dynamics with invertible neural networks and its application to the COVID-19 pandemics in Germany. PLOS Computational Biology, 17(10): e1009472, October 2021. ISSN 1553-7358. doi: 10.1371/journal.pcbi.1009472.

[10] Jonas Arruda, Yannik Schälte, Clemens Peiter, Olga Teplytska, Ulrich Jaehde, and Jan Hasenauer. An amortized approach to non-linear mixed-effects modeling based on neural posterior estimation. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 1865–1901. PMLR, 21–27 Jul 2024.

[11] Johann Brehmer, Gilles Louppe, Juan Pavez, and Kyle Cranmer. Mining gold from implicit models to improve likelihood-free inference. Proceedings ofthe National Academy ofSciences, 117(10):5242–5249, February 2020. ISSN 1091-6490. doi: 10.1073/pnas.1915980117.

[12] Antoine Wehenkel, Laura Manduchi, Jens Behrmann, Luca Pegolotti, Andrew C. Miller, Guillermo Sapiro, Ozan Sener, Marco Cuturi, and Jörn-Henrik Jacobsen. Simulation-based inference for cardiovascular models. arXiv preprint arXiv:2307.13918, July 2023.

[13] Yi Zhang and Lars Mikelsons. Solving stochastic inverse problems with stochastic BayesFlow. In 2023 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM), pages 966–972. IEEE, June 2023. doi: 10.1109/aim46323.2023.10196190.

[14] Manuel Gloeckler, Michael Deistler, Christian Weilbach, Frank Wood, and Jakob H. Macke. All-in-one simulation based inference. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

[15] Jerry M. Huang, Lukas Schumacher, Niek Stevenson, and Stefan T. Radev. CogFormer: Learn all your models once. arXiv preprint arXiv:2603.20520, March 2026.

[16] Pedro Rodrigues, Thomas Moreau, Gilles Louppe, and Alexandre Gramfort. HNPE: Leveraging global parameters for neural posterior estimation. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 13432–13443. Curran Associates, Inc., 2021.

[17] Lukas Heinrich, Siddharth Mishra-Sharma, Chris Pollard, and Philipp Windischhofer. Hierarchical neural simulation-based inference over event ensembles. Transactions on Machine Learning Research, 2024. ISSN 2835-8856.

[18] Daniel Habermann, Marvin Schmitt, Lars Kühmichel, Andreas Bulling, Stefan T. Radev, and Paul-Christian Bürkner. Amortized Bayesian multilevel models. Bayesian Analysis, pages 1 – 30, January 2025. ISSN 1936-0975. doi: 10.1214/25-ba1570.

[19] Šimon Kucharský and Paul-Christian Bürkner. Amortized Bayesian mixture models. Statistics and Computing, 36(4), June 2026. ISSN 1573-1375. doi: 10.1007/s11222-026-10911-y.

[20] Andrew Gelman. Multilevel (hierarchical) modeling: What it can and cannot do. Technometrics, 48(3):432–435, August 2006. ISSN 1537-2723. doi: 10.1198/004017005000000661.

[21] Lars Kühmichel, Jerry M Huang, Valentin Pratz, Jonas Arruda, Hans Olischläger, Daniel Habermann, Simon Kucharsky, Lasse Elsemüller, Aayush Mishra, Niels Bracher, et al. Bayesflow 2: Multi-backend amortized bayesian inference in python. arXiv preprint arXiv:2602.07098, 2026.

[22] Andrew Gelman, Aki Vehtari, Daniel Simpson, Charles C. Margossian, Bob Carpenter, Yuling Yao, Lauren Kennedy, Jonah Gabry, Paul-Christian Bürkner, and Martin Modrák. Bayesian workflow. arXiv preprint arXiv:2011.01808, November 2020.

[23] Jonah Gabry, Daniel Simpson, Aki Vehtari, Michael Betancourt, and Andrew Gelman. Visualization in bayesian workflow. Journal ofthe Royal Statistical Society Series A: Statistics in Society, 182(2):389–402, January 2019. ISSN 1467-985X. doi: 10.1111/rssa.12378.

[24] Petrus Mikkola, Osvaldo A. Martin, Suyog Chandramouli, Marcelo Hartmann, Oriol Abril Pla, Owen Thomas, Henri Pesonen, Jukka Corander, Aki Vehtari, Samuel Kaski, Paul-Christian Bürkner, and Arto Klami. Prior knowledge elicitation: The past, present, and future. Bayesian Analysis, 19(4), December 2024. ISSN 1936-0975. doi: 10.1214/23-ba1381.

[25] Donald B. Rubin. Estimation in parallel randomized experiments. Journal ofEducational Statistics, 6(4):377–401, 1981. ISSN 0362-9791. doi: 10.2307/1164617.

[26] Andreas Stuhlmüller, Jacob Taylor, and Noah Goodman. Learning stochastic inverses. In C.J. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Weinberger, editors, Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc., 2013.

[27] Stefan Webb, Adam Golinski, Rob Zinkov, N. Siddharth, Tom Rainforth, Yee Whye Teh, and Frank Wood. Faithful inversion of generative models for effective amortized inference. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018.

[28] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabás Póczos, Ruslan Salakhutdinov, and Alexander Smola. Deep sets. In Proceedings ofthe 31st International Conference on Neural Information Processing Systems, NIPS’17, pages 3394–3404, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964.

[29] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. Set Transformer: A framework for attention-based permutation-invariant neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pages 3744–3753. PMLR, 09–15 Jun 2019.

[30] Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, and Liang Sun. Transformers in time series: A survey. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI-2023, pages 6778–6786. International Joint Conferences on Artificial Intelligence Organization, August 2023. doi: 10.24963/ijcai.2023/759.

[31] Jason Hartford, Devon Graham, Kevin Leyton-Brown, and Siamak Ravanbakhsh. Deep models of interactions across sets. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings ofMachine Learning Research, pages 1909–1918. PMLR, 10–15 Jul 2018.

[32] Renhao Wang, Marjan Albooyeh, and Siamak Ravanbakhsh. Equivariant networks for hierarchical structures. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 13806–13817. Curran Associates, Inc., 2020.

[33] Stefan T. Radev, Marvin Schmitt, Lukas Schumacher, Lasse Elsemüller, Valentin Pratz, Yannik Schälte, Ullrich Köthe, and Paul-Christian Bürkner. BayesFlow: Amortized Bayesian workflows with neural networks. Journal of Open Source Software, 8(89):5702, September 2023. ISSN 2475-9066. doi: 10.21105/joss.05702.

[34] Kumar Abhishek, Jeremy Kawahara, and Ghassan Hamarneh. IMA++: ISIC archive multi-annotator dermoscopic skin lesion segmentation dataset. arXiv preprint arXiv:2512.21472, 2025.

[35] Dario Massimino, Stephen R. Baillie, Dawn E. Balmer, Richard I. Bashford, Richard D. Gregory, Sarah J. Harris, James J. N. Heywood, Leah A. Kelly, David G. Noble, James W. Pearce-Higgins, Michael J. Raven, Kate Risely, Paul Woodcock, Simon R. Wotton, and Simon Gillings. The Breeding Bird Survey of the United Kingdom. Global Ecology and Biogeography, 34(1), December 2024. ISSN 1466-8238. doi: 10.1111/geb.13943.

[36] Sean Talts, Michael Betancourt, Daniel Simpson, Aki Vehtari, and Andrew Gelman. Validating Bayesian inference algorithms with simulation-based calibration. arXiv preprint arXiv:1804.06788, 2018.

[37] Martin Modrák, Angie H. Moon, Shinyoung Kim, Paul Bürkner, Niko Huurre, Kateˇrina Faltejsková, Andrew Gelman, and Aki Vehtari. Simulation-based calibration checking for Bayesian computation: The choice of test quantities shapes sensitivity. Bayesian Analysis, 20(2), June 2025. ISSN 1936-0975. doi: 10.1214/23-ba1404.

[38] Stan Development Team. Stan modeling language users guide and reference manual, 2.39.0, 2026.

[39] Andrew Gelman and Iain Pardoe. Bayesian measures of explained variance and pooling in multilevel (hierarchical) models. Technometrics, 48(2):241–251, 2006.

[40] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real NVP. In International Conference on Learning Representations, 2017.

[41] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

[42] Jonas Arruda, Vikas Pandey, Catherine Sherry, Margarida Barroso, Xavier Intes, Jan Hasenauer, and Stefan Radev. Compositional amortized inference for large-scale hierarchical Bayesian models. In International Conference on Learning Representations, volume 2026, pages 51672–51696, 2026.

[43] Marvin Schmitt, Paul-Christian Bürkner, Ullrich Köthe, and Stefan T Radev. Detecting model misspecification in amortized Bayesian inference with neural networks. In DAGM German Conference on Pattern Recognition, pages 541–557. Springer, 2023.

[44] Aayush Mishra, Daniel Habermann, Marvin Schmitt, Stefan Radev, and Paul-Christian Bürkner. Robust amortized Bayesian inference with self-consistency losses on unlabeled data. In International Conference on Learning Representations, volume 2026, pages 9374–9405, 2026.

## A Enumerating inverse factorizations

(1)

$$
p ( \mu , \tau , \omega , \{ \lambda _ { j } \} \mid \{ y _ { i j } \} ) = p ( \mu , \tau \mid \{ y _ { i j } \} ) p ( \{ \lambda _ { j } \} \mid \{ y _ { i j } \} , \mu , \tau ) p ( \omega \mid \{ y _ { i j } \} , \{ \lambda _ { j } \} )\tag{2}
$$

$$
p ( \mu , \tau , \omega , \{ \lambda _ { j } \} \mid \{ y _ { i j } \} ) = p ( \mu , \tau \mid \{ y _ { i j } \} ) \prod _ { j = 1 } ^ { j } p ( \lambda _ { j } \mid y _ { j } , \mu , \tau , \omega ) p ( \omega \mid \mu , \tau , \{ y _ { i j } \} )\tag{3}
$$

$$
p ( \mu , \tau , \omega , \{ \lambda _ { j } \} \mid \{ y _ { i j } \} ) = p ( \mu , \tau \mid \{ \lambda _ { j } \} ) p ( \{ \lambda _ { j } \} \mid \{ y _ { i j } \} , \omega ) p ( \omega \mid \{ y _ { i j } \} )\tag{4}
$$

$$
p ( \mu , \tau , \omega , \{ \lambda _ { j } \} \mid \{ y _ { i j } \} ) = p ( \mu , \tau \mid \{ y _ { i j } \} , \omega ) \prod _ { j = 1 } ^ { J } p ( \lambda _ { j } \mid y _ { j } , \mu , \tau , \omega ) p ( \omega \mid \{ y _ { i j } \} )\tag{5}
$$

$$
p ( \mu , \tau , \omega , \{ \lambda _ { j } \} \mid \{ y _ { i j } \} ) = p ( \mu , \tau \mid \{ \lambda _ { j } \} ) p ( \{ \lambda _ { j } \} \mid \{ y _ { i j } \} ) p ( \omega \mid \{ y _ { i j } \} , \{ \lambda _ { j } \} )
$$

Factorizations (1), (3), and (5) infer $\{ \lambda _ { j } \}$ jointly, breaking amortization over the number of groups. In contrast, factorizations (2) and (4) estimate each group-level effect independently instead.

## B Full implementation of a two-level hierarchical model

To make this point concrete, the table in Figure 3 shows an example annotation for the two-level model on the left-hand side. The sampling functions f and sample size functions g are provided by the user. To allow for arbitrary nesting, the generated samples are internally stored in long format along with their group index. For each internal node, creating new draws corresponds to creating a full join of all parent nodes using the index columns as keys. For each row in this dataframe, we then generate a sample size s by running the respective sample size function, use the row values as input for the sampling function and execute this function s times with constant inputs. The generated draws are assigned a new index and appended to a new dataframe, along with the indices of the inputs.

![](images/db04a1aa681c1e7f75fb030c7df8ccfb91343caa42d861c58925888451f46735.jpg)

<table><tr><td>Node</td><td>Sampling function f</td><td>Sample size function g</td><td>Output shape</td></tr><tr><td>μ</td><td> $f _ { \mu } \sim \mathcal { N } ( 0 , 1 )$ </td><td> $g _ { \mu } = 1 0 0$ </td><td> $\mu : ( 1 0 0 , 1 )$ </td></tr><tr><td>T</td><td> $f _ { \tau } \sim \mathrm { E x p } ( 1 )$ </td><td> $g _ { \tau } = 1 0 0$ </td><td> $\tau : ( 1 0 0 , 1 )$ </td></tr><tr><td>ω</td><td> $f _ { \omega } \sim \mathrm { E x p } ( 1 )$ </td><td> $g _ { \omega } = 1 0 0$ </td><td> $\omega : ( 1 0 0 , 1 )$ </td></tr><tr><td>λ</td><td> $f _ { \lambda } \sim { \mathcal { N } } ( \mu , \tau )$ </td><td> $g _ { \lambda } \sim \mathcal { U } ( 1 , 2 0 )$ </td><td> $\lambda : ( 1 0 0 , g _ { \lambda } , 1 )$ </td></tr><tr><td>y</td><td> $f _ { y } \sim \mathcal { N } ( \lambda _ { j } , \omega )$ </td><td> $g _ { y } \sim \mathcal { U } ( 1 , 1 0 )$ </td><td> $y : ( 1 0 0 , g _ { \lambda } , g _ { y } , 1 )$ </td></tr></table>

Figure 3: Annotated two-level model graph (left) with its function specification (right). Each node is annotated with a sampling function f and a sample size function g. The root nodes $\mu , \tau$ and ω have fixed sample sizes of 100. $\lambda _ { j }$ draws a sample size uniformly from [1, 20] and observations draw a sample size uniformly from [1, 10]. The right columns the output shape for each node, where the trailing dimension is the data dimension (always 1 for scalar parameters).