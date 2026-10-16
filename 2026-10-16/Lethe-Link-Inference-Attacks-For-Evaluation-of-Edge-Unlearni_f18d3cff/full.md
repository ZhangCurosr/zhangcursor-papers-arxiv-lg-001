# Lethe: Link Inference Attacks For Evaluation of Edge Unlearning Methods

Andrea D’Angelo University of Aarhus Aarhus, Denmark andrea@cs.au.dk

Francesco Gullo University of L’Aquila L’Aquila, Italy francesco.gullo@univaq.it

Davide Mottin University of Aarhus Aarhus, Denmark davide@cs.au.dk

Sayan Ranu Indian Institute of Technology Delhi Delhi, India sayanranu@iitd.ac.in

Giovanni Stilo Luiss Guido Carli University Rome, Italy gstilo@luiss.it

## Abstract

Graph Unlearning (GU) aims to remove the influence of specific training data from a Graph Neural Network without retraining anew. While most methods address removal of nodes and features, Edge Unlearning (EU), aiming at forgetting edges, has recently attracted dedicated attention as a standalone problem due to the hardness of completely removing the impact of an edge in the model. Despite this, no dedicated benchmark for EU exists besides isolated evaluations on feature-rich datasets such as Cora and Citeseer where GNNs exploit node features rather than graph structure. As such, we show two systematic limitations in existing evaluations: dedicated EU methods can be slower than retraining anew, and accuracy metrics are uninformative due to the negligible impact of edge removal. To address this gap, we introduce Lethe<sup>1</sup>, the first comprehensive benchmark for Edge Un learning, featuring 15 methods and 8 datasets. Our benchmark emphasizes large graphs, introduces evaluation tasks of varying difficulty, and proposes empirical measures based on link inference attacks rather than proxy accuracy metrics. We provide a comprehensive empirical analysis, demonstrate the shortcomings of current evaluations, and release Lethe as a reproducible, extensible benchmark.

## 1 Introduction

Graph Unlearning (GU) methods aim to remove the influence of a subset of training data from a Graph Neural Network (GNN) without retraining it from scratch. This problem has gained increasing popularity due to recent regulatory requirements such as the European General Data Protection Regulation (GDPR), which enforces the right to beforgotten, allowing users to request the removal of their data from trained models [28].

GU has attracted significant attention [3] due to the widespread use of graph-structured data in applications involving sensitive information, such as social networks [24] and medical records [1]. In such scenarios, the scale of data makes retraining a GNN from scratch prohibitively expensive in terms of time and computational resources [8]. GU methods have emerged as the prime alternative, enabling the removal of specific nodes or edges without full retraining.

In this paper, we focus on Edge Unlearning (EU) as a fundamental subproblem of GU that targets the removal of information associated with specific edges in the graph. In many applications, sensitive information is inherently relational and encoded in edges rather than individual nodes. For instance, in a social network, a user may request the removal of a connection (e.g., a past interaction). In this case, the model holder must eliminate the influence of that specific edge while preserving the remaining graph structure [30, 11]. While most GU methods address both node and edge unlearning, the latter has recently attracted dedicated attention due to the inherent hardness of ensuring that the removed edge cannot be inferred through other mechanisms, with several methods proposed exclusively for this setting [26, 30].

![](images/928fbb313a1368fad0860de04dd2c9173c95204cec128f68bf381e75e3c7c676.jpg)

![](images/220f182b76739c13099f325948f45fd25f66611ef41ed1930f963ad18deca4d6.jpg)  
Figure 1: Main benchmark findings. (a) Inefficiency: Some methods take longer than retraining the model from scratch (Gold Model) or exhibit negligible runtime advantage; (b) Uninformative measures: No significant drop in accuracy after removing up to 50% of edges.

Despite its importance, there is currently no dedicated benchmark for EU. Existing EU methods construct the experimental setups without standardized protocols, often relying on widely used datasets that are neither consistently adopted across studies nor rigorously evaluated for this task. The most recent benchmark by Jain et al. [19] focuses on node unlearning; this work offers a complementary view by providing the set of necessary tools for evaluating GU on both node and edge unlearning. While [13] includes both node and edge unlearning, its evaluation relies primarily on UMIA and accuracy metrics, limiting the breadth of its evaluation protocol and motivating the shortcomings discussed below.

Critically, we find that the evaluations in the literature suffer from inherent limitations. Unlike node or feature unlearning, EU is fundamentally structure-driven: removing an edge alters the graph topology, leaving all nodes and node features intact. As a result, the effectiveness of an unlearning method depends on whether the model actually relies on structural (i.e., edges, subgraphs) signals. Existing works predominantly evaluate on small, feature-rich datasets like Cora [22], Citeseer [16] and Pubmed [32], where GNNs can effectively leverage node features rather than structural information, rendering edge removal largely inconsequential. This leads to two major problems, which we visualize in Figure 1: (a) The runtime of methods on these datasets is comparable (and sometimes higher) to that of the Gold Model obtained retraining the model anew, defeating the purpose of using unlearning methods. The only marginal gains are typically below one second and thus not practically meaningful. (b) On bigger datasets, the impact of edge removal on accuracy is negligible, even retraining from scratch on a graph with up to half of the edges removed (which is unrealistic) results in stable accuracy on most datasets $( \Delta \bar { \leq } 5 \% )$ . This is consistent with the literature on GCNs [20], notably dropping edges for boosting performance (like in the case of DropEdge [25]), and shows that accuracy reported by existing GU methods on such datasets is not informative for evaluating unlearning effectiveness. These observations suggest that the settings used to evaluate EU methods are not fully aligned with the EU task, leading to potentially fallacious conclusions about method’s effectiveness and efficiency.

To fill this gap, we present Lethe, the first dedicated benchmark for EU, spanning 15 methods, 8 datasets, and 5 model architectures. Lethe solves both problems by (i) only employing datasets big enough for runtime to matter, (ii) employing three link inference attacks that are more informative than accuracy alone on most datasets and (iii) proposing different forget set selection strategies based on an edge centrality measure called WalkCentrality. Lethe is openly available<sup>2</sup>, reproducible, extensible and focuses on structural properties of graphs rather than feature-rich representations.

Lethe sheds light on the efficacy of existing MU and GU methods, surpassing standard, accuracybased evaluations on small graphs. Specifically, we show how link inference attacks highlight several criticalities across methods, and how selecting the forget set based on a centrality measure can lead to systematic failures in defending from such attacks.

## 2 Related Work

Edge Unlearning. Graph Unlearning methods aim to approximate the effect of retraining a GNN from scratch after removing a subset of training data [3]. Existing approaches for GU that are also defined and evaluated on EU either prove their effectiveness empirically, like in the case of GNNDelete [4], or provide theoretical certification of the closeness of their solution to the Gold Model, like in the case of CGU [5], IDEA [11], and ScaleGUN [33].

While many GU methods handle edge unlearning as one supported request type, a dedicated line of work targets it as a standalone problem. GNNDelete [4] formalizes two structural objectives: deleted edge consistency, requiring that predictions for removed edges resemble those for non-existent edges, and neighborhood influence, ensuring predictions in the local vicinity of a removed edge remain stable. CEU [30] provides certified guarantees for edge unlearning via quasi-Newton parameter updates with Gaussian noise injection to mask gradient residuals. Unlink to Unlearn [26] avoids parameter modification entirely, instead masking the forgotten edge during message passing at inference time.

All of these methods, both for GU and EU, employ small, feature-rich datasets like Cora [22] and Citeseer [16] in their experimental evaluation, and argue that the results generalize to large-scale graphs. ScaleGUN is the only one observing how these small datasets are not enough for a fair evaluation. We bring this observation to its logical conclusion: the community needs a benchmark for evaluating EU methods in fair condition on these datasets. Moreover, several general-purpose Machine Unlearning algorithms are applicable to GNNs. These methods include (i). fine-tuning the GNN on the retain graph or, conversely, applying Gradient Ascent on the forget set of the graph [17], (ii). modifying the weights of the GNN directly, like in the case of SalUn [14] and UNSIR [27], (iii). distillation-based methods with a teacher model, e.g. Bad Teaching [6] and Scrub [21]. Ours is the first work to benchmark them for edge removal and compare them to dedicated GU methods.

Benchmarking of Graph Unlearning. Despite increasing attention on the topic, GU (and EU consequently) has received limited surveying and benchmarking efforts. [19] proposes the most recent GU benchmark concluding that most GU methods are not ready for practical deployment yet, due to poor scaling on large graphs. However, their work focuses only on the task of node unlearning while EU is essential for a complete evaluation of graph unlearning methods, as it addresses cases that node unlearning cannot model. Our study complements the existing benchmark [19] by defining the protocol for the edge setting, thereby covering an important class of scenarios that node unlearning alone cannot capture, in which graph size and edge density are even more critical. We aim to provide the community with all the necessary tools to fully evaluate their GU methods, both on node and edge unlearning. Similarly, OpenGU [13] provides a general benchmark for both tasks. However, while OpenGU focuses on the membership inference attack (MIA) from general machine unlearning, our analysis suggests that MIA is often the least informative in a graph context. We address this by implementing three dedicated link inference attacks specifically designed for the graph unlearning setting, and two strategies for centrality-guided forget set selection. Consequently, we provide all methods and attacks within a standardized, reproducible, and extensible framework.

Link Inference Attacks. Unlearning efficacy in the edge setting is most directly assessed via link inference: adversarial probing of a model’s outputs to determine whether a specific edge was present during training. Prior work has shown that GNNs are susceptible to such attacks through shadow model approaches [18], influence function-based methods [29], and general inference frameworks [34]. In EU evaluation, link inference attacks serve as the adversarial counterpart to accuracy-based utility metrics, providing a direct test of whether edge information has been forgotten, or if it can be adversarially reconstructed. However, existing works adopt incompatible threat models and attack implementations, making cross-method comparison difficult. Lethe aims to standardize this evaluation component to guide future research in the field.

Critique of current benchmarks. Many prior works in the literature of GNNs highlight the increasing necessity of evolving the evaluation techniques in the field. Palowitch et al. [23] argue that

![](images/3d56007a3c135cf2cf34a3d1867ae80794fe102cd1eb30fed86f0f9709d6f29c.jpg)  
Figure 2: Runtime (sec.) of GU methods on Cora, Citeseer and Pubmed across 5 different GNN architectures. Note that CGU and ScaleGUN only work on linear models (i.e., SGC). The red line represents the Gold Model: anything above implies that it takes more time than retraining the model from scratch.

GNNs are usually evaluated on the same handful of small datasets with low structural information. Coupette et al. [7] stress the importance of good graph benchmarks, with a focus on structural signals, something that the most popular datasets are still missing. Bechler-Speicher et al. [2], in their position paper, argue that graph learning is at risk of losing relevance because of how poorly designed graph learning benchmarks are, the over-reliance on the same few datasets, and the neglect of structurally interesting graphs. This need naturally carries over to the task of EU, where forgetting means being unable to reconstruct the edge from any unremoved graph signal. As opposed to previous benchmarks focusing exclusively on node unlearning, our focus is to collect a set of datasets designed for profiling EU, with the aim of complementing GU with measures based on link inference attacks and dataset hardness measures to help the community quantify progress in EU in a systematic manner.

## 3 Limitations of Current Evaluations

In this section, we provide empirical evidence of two main limitations in the evaluations of EU in the literature, namely the usage of graphs that are either too small and/or over-reliant on node features. Due to the lack of benchmarks for EU, most works define their own experimental settings, usually relying on popular datasets that are often used in the context of graph learning. Indeed, most edge unlearning methods evaluate on standard benchmarks like Cora, Citeseer, and Pubmed [3], relying on the test accuracy of the unlearned model as a proxy for efficacy [30, 5, 11, 4]. However, these evaluation settings are not well aligned with the EU goals as they falter in one of two distinct ways:

(i) Small-scale datasets: negligible efficiency gains. Datasets such as Cora, Citeseer and Pubmed are extremely small, making retraining from scratch very cheap. Even in the original papers, reported training times for datasets such as Cora and Citeseer typically range from 3 to 5 seconds [30].

In our empirical evaluation (Figure 2) most methods, both general MU and GU methods, run in time comparable to that of retraining on Cora, Citeseer, and Pubmed, with simpler methods like Fine-tuning being slightly faster. However, such gains are not only negligible in practice, but also unlikely to generalize. Indeed, differences of this magnitude can be attributed to statistical noise or machine- and implementation-specific factors, such as hardware variability or framework overhead (e.g., PyTorch initialization latency), rather than meaningful algorithmic efficiency. The graph density of these datasets also plays a role in the runtime overhead. CEU [30] introduces the notion of infected nodes in their implementation, as the nodes affected by the removal of a forgotten edge within k-hops, where k is the number of GNN layers. However, removing even 1% of edges in these high-density graphs infects almost all nodes within two hops, meaning that MU methods are effectively operating on almost the entire graph, leaving little room to amortise the cost of unlearning against the cost of retraining. We show this in detail in Section A.2.

Table 1: Gold Model test accuracy (%) across architectures and datasets at 5%, 20%, and 50% forget-set size. Original shows the accuracy of the unmodified model (upper bound). ∆ is the accuracy change relative to Original.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Architecture</td><td rowspan="2">Original</td><td colspan="2">Gold Model (5%)</td><td colspan="2">Gold Model (20%)</td><td colspan="2">Gold Model (50%)</td></tr><tr><td>Acc</td><td>Δ</td><td>Acc</td><td>Δ</td><td>Acc</td><td>∆</td></tr><tr><td rowspan="5">Computers</td><td>GCN</td><td>79.7</td><td>81.1</td><td>+1.3</td><td>81.6</td><td>+1.9</td><td>75.2</td><td>-4.5</td></tr><tr><td>GIN</td><td>42.7</td><td>43.8</td><td>+1.1</td><td>25.0</td><td>-17.6</td><td>36.6</td><td>-6.1</td></tr><tr><td>GAT</td><td>81.8</td><td>81.4</td><td>-0.4</td><td>80.9</td><td>-0.9</td><td>79.1</td><td>-2.7</td></tr><tr><td>GraphSAGE</td><td>84.0</td><td>83.9</td><td>-0.1</td><td>83.4</td><td>-0.5</td><td>81.1</td><td>-2.9</td></tr><tr><td>SGČCGU</td><td>75.6</td><td>74.4</td><td>-1.2</td><td>74.6</td><td>-1.1</td><td>73.5</td><td>-2.2</td></tr><tr><td rowspan="5">Photos</td><td>GCN</td><td>91.0</td><td>89.1</td><td>-2.0</td><td>90.3</td><td>-0.8</td><td>88.1</td><td>-2.9</td></tr><tr><td>GIN</td><td>66.3</td><td>74.5</td><td>+8.2</td><td>70.7</td><td>+4.4</td><td>65.2</td><td>-1.0</td></tr><tr><td>GAT</td><td>91.6</td><td>90.9</td><td>-0.7</td><td>90.4</td><td>-1.2</td><td>88.9</td><td>-2.7</td></tr><tr><td>GraphSAGE</td><td>93.5</td><td>92.9</td><td>-0.7</td><td>92.2</td><td>-1.4</td><td>90.3</td><td>-3.2</td></tr><tr><td>SGČCGU</td><td>87.3</td><td>87.4</td><td>+0.1</td><td>86.3</td><td>-1.0</td><td>85.1</td><td>-2.2</td></tr><tr><td rowspan="5">DBLP</td><td>GCN</td><td>82.5</td><td>82.6</td><td>+0.1</td><td>83.1</td><td>+0.6</td><td>82.3</td><td>-0.2</td></tr><tr><td>GIN</td><td>83.7</td><td>83.2</td><td>-0.5</td><td>83.9</td><td>+0.2</td><td>82.4</td><td>-1.3</td></tr><tr><td>GAT</td><td>83.4</td><td>82.3</td><td>-1.2</td><td>82.3</td><td>-1.2</td><td>81.9</td><td>-1.5</td></tr><tr><td>GraphSAGE</td><td>84.3</td><td>84.4</td><td>+0.1</td><td>84.3</td><td>+0.0</td><td>83.9</td><td>-0.4</td></tr><tr><td>SGČCGU</td><td>71.0</td><td>70.6</td><td>-0.4</td><td>70.3</td><td>-0.6</td><td>69.7</td><td>-1.2</td></tr></table>

Table 2: Datasets included in Lethe and their key properties. Avg. deg: average node degree. Density: edge density. $\Delta _ { \mathrm { 2 0 \% } } \mathrm { : }$ accuracy change Gold−Identity (pp) at $k { = } 2 0 \%$ forget set.
<table><tr><td>Dataset</td><td>Domain</td><td>|V|</td><td>|E|</td><td> $\mathbf { A v g . d e g }$ </td><td>Density</td><td>Features</td><td>Classes</td><td> $\Delta _ { 2 0 \% }$ </td></tr><tr><td>Cora</td><td>Citation</td><td>2.7K</td><td>10.6K</td><td>7.8</td><td>0.29%</td><td>1433</td><td>7</td><td>-3.1</td></tr><tr><td>Citeseer</td><td>Citation</td><td>3.3K</td><td>9.2K</td><td>5.5</td><td>0.17%</td><td>3703</td><td>6</td><td>-1.5</td></tr><tr><td>Pubmed</td><td>Citation</td><td>19.7K</td><td>44.3K</td><td>4.5</td><td>0.02%</td><td>500</td><td>3</td><td>+0.3</td></tr><tr><td>Computers</td><td>Co-purchase</td><td>13.8K</td><td>492K</td><td>35.8</td><td>0.52%</td><td>767</td><td>10</td><td>+0.5</td></tr><tr><td>Photos</td><td>Co-purchase</td><td>7.6K</td><td>238K</td><td>31.1</td><td>0.81%</td><td>745</td><td>8</td><td>-2.2</td></tr><tr><td>DBLP</td><td>Citation</td><td>17.7K</td><td>106K</td><td>6.0</td><td>0.07%</td><td>1639</td><td>4</td><td>+0.1</td></tr><tr><td>Flickr</td><td>Social</td><td>89K</td><td>900K</td><td>10.1</td><td>0.02%</td><td>500</td><td>7</td><td>-2.3</td></tr><tr><td>arxiv</td><td>Citation</td><td>169K</td><td>1.2M</td><td>7.1</td><td>0.01%</td><td>128</td><td>40</td><td>-0.6</td></tr></table>

(ii) Feature-dominated datasets: negligible impact on accuracy. A second, orthogonal issue arises in datasets characterized by highly informative node features, typically in the form of bag-ofwords representations or word embeddings. In such settings, node features alone are often sufficient to achieve strong predictive performance, reducing the relative importance of graph structure.

Table 1 shows the Gold Model test accuracy across all architectures and datasets when the forget set is 5% and 20% of the training set. For most architectures the change in accuracy is within 3%, except for GIN [31] which does not normalize across neighbor embeddings and is thus more affected by edge removals. In general, removing edges has only a limited effect on model predictions, leading to negligible changes in accuracy. This makes it difficult to meaningfully evaluate the effectiveness of EU methods through this proxy metric, as the task itself becomes insensitive to structural modifications, so relying on accuracy only can be misleading.

## 4 Lethe

Our benchmarking methodology comprises two sections. Section 4.1 elaborates on the observations in Section 3 and introduces Lethe, our benchmark that fills the literature gap (refer to Section A.1 for a detailed description of the link inference attacks included). Section 4.2 describes the two forget set selection strategies included in Lethe.

## 4.1 Design of Lethe

The findings of Section 3 highlight a set of concrete requirements that a benchmark for Edge Unlearning must satisfy. We state the three following desiderata and use them to construct Lethe.

$D _ { 1 } { : }$ Scale and graph density. The practical relevance of MU depends on retraining being expensive. On small graphs, as shown in Figure 2, retraining from scratch typically takes only a few seconds, making any unlearning overhead negligible in absolute terms. Lethe prioritises graphs of sufficient size (>5K nodes, >100K edges) in which the Gold Model constitutes a genuine computational cost. For this reason, while Lethe supports Cora, Citeseer, and Pubmed, we omit them from the discussion of results. Table 2 summarises the datasets included in Lethe and their alignment with $D _ { 1 }$

$D _ { 2 } { \mathrm { : } }$ : Structural informativeness. On feature-rich datasets, node embeddings alone carry sufficient discriminative signal, and the graph structure acts as a weak regularizer. Edge removal then has negligible effect on the model, making accuracy metrics uninformative. Crucially, EU is inherently a privacy problem. The objective is not only to maintain performance on downstream tasks, but to ensure that the model no longer encodes information about the forgotten edges. Accuracy alone cannot capture this notion of forgetting, as it does not quantify whether structural information can still be inferred from the model. To address this gap, we adapt three established link inference attacks to measure whether a method is effectively unlearning: UMIA [21], Link Stealing (LS) [18], and LinkTeller (LT) [29]. We assume access to the unlearned model and retain graph, and the adversary must classify a candidate edge $( u , v )$ as either a forgotten edge (positive) or a non-existing edge (negative). Details and implementation of these attacks are available in Appendix A.1. We report the ROC-AUC of these scores over positive (forgotten) and negative (non-existing) pairs.

$D _ { 3 } { \mathrm { : } }$ : Task difficulty diversity. Not all edges are equally hard to unlearn. Edges that appear in sparser zones and are not structurally redundant have a strong footprint in the learned representations and are less straightforward to forget. Conversely, highly-central edges participate in many messagepassing paths and leave a weaker trace across node representations. A rigorous benchmark must evaluate methods across this spectrum. To this end, we define the metric of WalkCentrality (Section 4.2) as a yardstick to implement two forget set selections strategies in Lethe.

Following $D _ { 1 } - D _ { 3 }$ , Lethe includes 8 datasets and 15 methods, assessed using three complementary link inference attacks, each evaluated under two different settings (low and high centrality, explained in Section 4.2).

## 4.2 Settings of Edge Unlearning: WalkCentrality

Current works simply sample a random percentage (usually either 5% or 20%) of edges to build the forget set. However, not all edges are equally important, given the message passing systems of GNNs. Intuitively, some edges will participate in more message passing paths than others. To model this property in Lethe, we introduce two different difficulty settings for edge unlearning, leveraging the model-agnostic measure of WalkCentrality.

We ask: given an edge e, how redundant is it in the message passing mechanism? This gives us an intuition of how much “information” the edge will carry during the message passing, and how much its propagation is smoothed from redundant paths.

Since GCNs propagate features using the symmetrically normalized adjacency matrix $\hat { A } \ =$ $\tilde { D } ^ { - 1 / 2 } \tilde { A } \tilde { D } ^ { - 1 / 2 }$ , where ${ \tilde { A } } = A + I$ adds self-loops, we use $\hat { A }$ in place of the raw adjacency matrix. This captures both degree normalization and self-loops, giving a more faithful measure of how much a given edge influences the learned representations. Any k-step walk passing through edge $( i  j )$ at step t decomposes uniquely into a t-step walk from some source to $i ,$ the edge $( i  j )$ , and a (k − t − 1)-step walk from $j$ to some destination. Summing the weights of these walks over all t, excluding the factor $\hat { A } _ { i j }$ of the edge itself (so that the score measures the propagation surrounding the edge), gives the walk-based centrality for edge (i, j) across a k-layer GCN:

$$
\operatorname { W a l k C e n t r a l i t y } ( i , j ) = \sum _ { t = 0 } ^ { k - 1 } \left( \mathbf { 1 } ^ { \top } \hat { A } ^ { t } \right) _ { i } \cdot \left( \hat { A } ^ { k - t - 1 } \mathbf { 1 } \right) _ { j }
$$

This is efficient to compute: precomputing $\{ \hat { A } ^ { 0 } \mathbf { 1 } , \dotsc , \hat { A } ^ { k - 1 } \mathbf { 1 } \}$ and $\{ \mathbf { 1 } ^ { \top } \hat { A } ^ { 0 } , \dotsc , \mathbf { 1 } ^ { \top } \hat { A } ^ { k - 1 } \}$ reduces the centrality of every edge to a simple lookup and sum, requiring $O ( k \cdot | E | )$ total for sparse graphs. WalkCentrality allows us to score and rank all edges in a graph. By selecting 5% of edges with the lowest/highest WalkCentrality, we configure the low/high centrality settings of Lethe.

Observation. Ignoring learnable weights for simplicity, a k-layer GNN with normalized adjacency message passing computes: $H ^ { ( k ) } = \hat { A } ^ { k } X$ , where A<sup>ˆ</sup> is the normalized adjacency matrix and the entry $\hat { A } _ { u v } ^ { k }$ is the aggregate contribution of all length-k walks from u to v. For an edge $( i , j )$ , the term $( \mathbf { 1 } ^ { \top } \hat { A } ^ { t } ) ,$ <sub>i</sub> is how much signal reaches i after t steps of propagation, and $( \hat { A } ^ { k - t - 1 } \mathbf { 1 } ) _ { j }$ is how much signal spreads out of in the steps that remain. Summing their product over every way of splitting the steps between the two endpoints gives $\begin{array} { r } { \sum _ { t = 0 } ^ { k - 1 } ( \mathbf { 1 } ^ { \top } \hat { A } ^ { t } ) _ { i } ( \hat { A } ^ { k - i - 1 } \mathbf { 1 } ) _ { j } } \end{array}$ , which is exactly the WalkCentrality score used in Lethe.

Because $( \hat { A } ^ { t } \mathbf { 1 } )$ grows with the connectivity of i and of its t-hop neighbourhood, a high score means the edge sits between endpoints that are already carrying a large volume of propagated signal and therefore that its own share of that signal is small. This Observation shows how WalkCentrality is directly derived from a mechanistic interpretation of the GNN’s message passing system.

To measure this directly rather than only through link inference attacks, we ran experiments on three datasets (Photos, Computers and DBLP, with a GCN, 5% forget set), showing that WalkCentrality is strongly negatively correlated with training loss, and that low-centrality removals are localized and have a large impact. Results are reported in Appendix B.

## 5 Experimental Evaluation

We conduct the experiments on a Linux-based high-performance computing cluster equipped with Intel Xeon Gold 6140M CPUs (2.30 GHz, 108 cores), 880 GB of RAM, and an NVIDIA A100 GPU (80 GB memory). We tune the hyperparameters on the learning rate of general MU methods to adapt them to GNNs. For graph-specific methods, we use the settings described in the original papers. A broad overview of the methods that we use is present in Appendix D. All the architectures are trained for 100 epochs with the Adam optimizer with early stopping using a patience of 10 epochs and a minimum improvement threshold of 0.01. Original and Gold Models are cached between methods and settings to provide a stable reference.

Specifically, the Gold Model and attack-based metrics evaluate two complementary but distinct properties of an unlearning method (refer to the definition of EU in Appendix A). The Gold Model represents the outcome of retraining from scratch after removing the forgotten edges and therefore serves as the reference for retraining fidelity. Attack-based metrics, instead, ask: after unlearning, can an adversary still recover information about the edges that were requested to beforgotten?

Therefore, the Gold Model provides the reference for functional behavior after forgetting, whereas attack-based metrics measure privacy leakage (the ideal value being 0.5, so the attack is basically random guessing). Importantly, this is also the case for the most recent node unlearning benchmark of [19], that we aim to complement: they also consider an attack AUC of 0.5 as ideal, regardless of the Gold Model. We adapt to this standard, which is consistent with the unlearning literature.

The methods we benchmark in this experimental evaluation are either general-purpose MU methods and GU methods. As general-purpose MU methods, we evaluate Fine-tuning [17], Negative Gradient [17], Advanced Negative Gradient [17], UNSIR [27], BadTeaching [6], SCRUB [21], Fisher Forgetting [15], Selective Synaptic Dampening [15], SalUn [14]. For GU, we evaluate IDEA [11], CGU [5], CEU [30], ScaleGUN [33], UnlinkToUnlearn [26] and GNNDelete [4]. The application of general MU methods to Edge Unlearning is a unique contribution of this work.

In Appendix C, we report the results of all the ablation studies we perform, and their consequences on the Research Questions we answer in this section. Specifically, we ablate: (i). The attacker’s accessible information for link inference attacks in Appendix C.1, (ii). The forget set selection strategy, in Appendix C.2, and (iii). The downstream task, in Appendix C.3.

## 5.1 Results

Here we outline the findings of the experiments using Lethe. We report the results obtained on GCNs, which is the most popular GNN architecture, with a forget set of 5%. We aim to answer three research questions targeting fundamental open issues in EU evaluation:

• RQ1: Are link inference attacks more informative than accuracy? Which one is the most effective?

Table 3: Accuracy and link inference attacks across all methods. Upper block: GCN architecture. Lower block: SGC architecture. Acc: test accuracy (% ; ↑ better). UMIA: E-UMIA AUC (≈0.5 = indistinguishable from random). LT: LinkTeller AUC. LS: Link Stealing AUC. Shaded rows are reference baselines. <sup>§</sup> GNNDelete collapses model performance, as found in [19].
<table><tr><td></td><td colspan="4">Photos</td><td colspan="4">Computers</td><td colspan="4">DBLP</td><td colspan="4">Flickr</td><td colspan="4">ogbn-arxiv</td></tr><tr><td>Method</td><td>Acc</td><td>UMIA</td><td>LT</td><td>LS</td><td>Acc</td><td>UMIA</td><td>LT</td><td>LS</td><td>Acc</td><td>UMIA</td><td>LT</td><td>LS</td><td>Acc</td><td>UMIA</td><td>LT</td><td>LS</td><td>Acc</td><td>UMIA</td><td>LT</td><td>LS</td></tr><tr><td>Original</td><td>90.8</td><td>0.526</td><td>0.480</td><td>0.639</td><td>79.9</td><td>0.500</td><td>0.564</td><td>0.525</td><td>82.5</td><td>0.505</td><td>0.392</td><td>0.733</td><td>49.0</td><td>0.510</td><td>0.391</td><td>0.746</td><td>53.1</td><td>0.505</td><td>0.636</td><td>0.833</td></tr><tr><td>GoldModel</td><td>90.3</td><td>0.524</td><td>0.490</td><td>0.635</td><td>81.6</td><td>0.505</td><td>0.569</td><td>0.587</td><td>83.1</td><td>0.501</td><td>0.366</td><td>0.729</td><td>48.3</td><td>0.511</td><td>0.369</td><td>0.765</td><td>53.3</td><td>0.508</td><td>0.628</td><td>0.833</td></tr><tr><td>Finetuning</td><td>88.8</td><td>0.507</td><td>0.493</td><td>0.714</td><td>79.1</td><td>0.491</td><td>0.557</td><td>0.554</td><td>82.6</td><td>0.497</td><td>0.381</td><td>0.733</td><td>38.5</td><td>0.499</td><td>0.364</td><td>0.750</td><td>52.3</td><td>0.505</td><td>0.619</td><td>0.829</td></tr><tr><td>NegGrad</td><td>80.6</td><td>0.515</td><td>0.475</td><td>0.597</td><td>76.3</td><td>0.512</td><td>0.558</td><td>0.661</td><td>82.5</td><td>0.503</td><td>0.391</td><td>0.732</td><td>45.1</td><td>0.519</td><td>0.375</td><td>0.741</td><td>52.7</td><td>0.504</td><td>0.628</td><td>0.829</td></tr><tr><td>AdvNegGrad</td><td>89.3</td><td>0.525</td><td>0.492</td><td>0.716</td><td>69.1</td><td>0.512</td><td>0.550</td><td>0.807</td><td>82.6</td><td>0.495</td><td>0.376</td><td>0.731</td><td>44.3</td><td>0.517</td><td>0.366</td><td>0.741</td><td>51.4</td><td>0.503</td><td>0.621</td><td>0.816</td></tr><tr><td>UNSIR</td><td>87.2</td><td>0.512</td><td>0.489</td><td>0.867</td><td>67.5</td><td>0.495</td><td>0.563</td><td>0.838</td><td>82.7</td><td>0.505</td><td>0.373</td><td>0.733</td><td>42.8</td><td>0.521</td><td>0.382</td><td>0.749</td><td>50.8</td><td>0.504</td><td>0.626</td><td>0.804</td></tr><tr><td>BadTeaching</td><td>90.8</td><td>0.515</td><td>0.486</td><td>0.734</td><td>82.0</td><td>0.507</td><td>0.558</td><td>0.530</td><td>82.9</td><td>0.497</td><td>0.376</td><td>0.733</td><td>48.1</td><td>0.504</td><td>0.344</td><td>0.844</td><td>52.6</td><td>0.504</td><td>0.628</td><td>0.815</td></tr><tr><td>Scrub</td><td>85.4</td><td>0.497</td><td>0.484</td><td>0.605</td><td>47.9</td><td>0.511</td><td>0.532</td><td>0.839</td><td>82.8</td><td>0.487</td><td>0.379</td><td>0.730</td><td>42.4</td><td>0.522</td><td>0.388</td><td>0.726</td><td>43.6</td><td>0.501</td><td>0.625</td><td>0.786</td></tr><tr><td>FisherForg.</td><td>90.6</td><td>0.502</td><td>0.492</td><td>0.623</td><td>80.4</td><td>0.504</td><td>0.544</td><td>0.521</td><td>81.7</td><td>0.495</td><td>0.382</td><td>0.730</td><td>48.8</td><td>0.513</td><td>0.368</td><td>0.742</td><td>52.8</td><td>0.501</td><td>0.627</td><td>0.832</td></tr><tr><td>SSD</td><td>90.8</td><td>0.497</td><td>0.487</td><td>0.670</td><td>80.3</td><td>0.510</td><td>0.549</td><td>0.535</td><td>82.9</td><td>0.506</td><td>0.388</td><td>0.734</td><td>48.9</td><td>0.510</td><td>0.381</td><td>0.743</td><td>52.5</td><td>0.505</td><td>0.626</td><td>0.826</td></tr><tr><td>SalUn</td><td>89.7</td><td>0.525</td><td>0.467</td><td>0.727</td><td>83.8</td><td>0.514</td><td>0.541</td><td>0.543</td><td>83.1</td><td>0.495</td><td>0.381</td><td>0.732</td><td>48.2</td><td>0.506</td><td>0.356</td><td>0.855</td><td>51.7</td><td>0.504</td><td>0.617</td><td>0.808</td></tr><tr><td>IDEA</td><td>90.8</td><td>0.515</td><td>0.467</td><td>0.641</td><td>79.9</td><td>0.508</td><td>0.551</td><td>0.525</td><td>82.4</td><td>0.599</td><td>0.469</td><td>0.705</td><td>48.9</td><td>0.511</td><td>0.374</td><td>0.746</td><td>53.3</td><td>0.504</td><td>0.626</td><td>0.833</td></tr><tr><td>UtU</td><td>91.0</td><td>0.526</td><td>0.480</td><td>0.639</td><td>79.7</td><td>0.500</td><td>0.564</td><td>0.525</td><td>82.5</td><td>0.505</td><td>0.392</td><td>0.733</td><td>49.4</td><td>0.510</td><td>0.391</td><td>0.746</td><td>53.3</td><td>0.505</td><td>0.636</td><td>0.833</td></tr><tr><td>GNNDelete§</td><td>34.6</td><td>0.520</td><td>0.450</td><td>0.530</td><td>37.9</td><td>0.516</td><td>0.497</td><td>0.826</td><td>75.6</td><td>0.633</td><td>0.410</td><td>0.865</td><td>42.2</td><td>0.496</td><td>0.445</td><td>0.625</td><td>22.6</td><td>0.508</td><td></td><td>0.606 0.682</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>SGC architecture</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Original (SGC)</td><td>87.3</td><td>0.483</td><td>0.419</td><td>0.538</td><td>75.8</td><td>0.517</td><td>0.541</td><td>0.542</td><td>71.0</td><td>0.604</td><td>0.494</td><td>0.600 47.8</td><td></td><td>0.511</td><td>0.082</td><td>0.902</td><td>30.6</td><td>0.522</td><td>0.479</td><td>0.533</td></tr><tr><td>GoldModel (SGC)</td><td>85.7</td><td>0.485</td><td>0.427</td><td>0.542</td><td>73.0</td><td>0.517</td><td>0.539</td><td>0.532</td><td>70.5</td><td>0.601</td><td>0.488</td><td>0.552</td><td>44.9</td><td>0.502</td><td>0.059</td><td>0.906</td><td>26.6</td><td>0.519</td><td>0.520</td><td>0.534</td></tr><tr><td>CGU</td><td>92.4</td><td>0.504</td><td>0.436</td><td>0.621</td><td>86.3</td><td>0.508</td><td>0.544</td><td>0.512</td><td>75.6</td><td>0.594</td><td>0.486</td><td>0.613</td><td>5.3</td><td>0.495</td><td>0.081</td><td>0.656</td><td>37.4</td><td>0.536</td><td></td><td>0.505</td></tr><tr><td>ScaleGUN</td><td>88.0</td><td>0.487</td><td>0.436</td><td>0.604</td><td>73.7</td><td>0.509</td><td>0.530</td><td>0.514</td><td>78.7</td><td>0.596</td><td>0.478</td><td>0.510</td><td>48.8</td><td>0.513</td><td>0.065 0.059</td><td>0.638 0.902</td><td>36.3</td><td>0.529</td><td>0.545 0.534</td><td>0.506</td></tr><tr><td>CEU</td><td>87.3</td><td>0.496</td><td>0.431</td><td>0.538</td><td>75.8</td><td>0.531</td><td>0.533</td><td>0.542</td><td>71.0</td><td>0.596</td><td>0.481</td><td>0.600</td><td>47.8 0.507</td></table>

• RQ2: Are the task centrality settings informative? Do WalkCentrality-based settings reveal qualitatively different behaviours?

• RQ3: Are certified EU methods applicable in practice and resilient to link inference attacks?

## 5.1.1 New insights from Link Inference Attacks

Table 3 compares the results among Link Inference Attacks and Accuracy. Across all benchmark datasets, the spread in test accuracy among non-collapsing (i.e., excluding GNNDelete) unlearning methods is at most 11 percentage points (pp) on Photos GCN and under 1.5 pp on DBLP GCN. The vast majority of methods cluster within ±2 pp of the Original model, making accuracy unable to distinguish any ranking. E-UMIA, usually used in traditional MU, is equally blind: across all methods and all datasets it produces $\mathrm { A U C } \approx 0 . 4 8 \mathrm { \dot { – } } 0 . 5 3$ with no method-discriminating signal, consistent with the finding of [19] in the node setting. The sole exception is GNNDelete on DBLP $( \mathrm { U M A } = 0 . 6 3 3 )$ which reflects how model-collapse changes the loss distribution rather than actual unlearning. We find that GNNDelete often collapses the model in the EU setting consistently with past findings [19]. We define model collapse in terms of accuracy, where the drop is so large that the model is rendered effectively unusable to continue supporting that downstream task.

LS is a more informative attack. Table 3 shows LS AUC across benchmark datasets (GCN, high centrality setting). On three of the five datasets, LS produces a spread of at least 20 pp across methods, with a consistent ranking structure. To confirm the qualitative observations above, we test whether the three attacks differ in their discriminative power across methods. For each dataset, we compare the variance of each metric across all non-collapsing methods using one-sided F-tests $( H _ { 0 } { \mathrm { : } }$ equal variance; $H _ { 1 } \colon \sigma ^ { 2 } ( \mathrm { L S } ) > \sigma ^ { 2 } ( \cdot ) )$ . On AmazonPhotos, AmazonComputers, Flickr, and ogbn-arxiv, the standard deviation of LS across methods is 6–18× larger than that of E-UMIA (7.2×, $1 7 . 9 \times , 6 . 0 \times , 7 . 9 \times$ respectively) and 3–12× larger than LT; all eight F-tests reject equal variance at $p < 1 0 ^ { - 3 }$ . For accuracy, one-sample t-tests on ∆Acc (deviation from Identity) yield $p = 0 . 1 7$ on AmazonComputers, $p = 0 . 3 8$ on DBLP, and $p = 0 . 0 7$ on ogbn-arxiv, confirming that methods produce no significant accuracy change on these datasets. On AmazonPhotos and Flickr the test is significant $( p < 0 . 0 5 )$ , but the mean deviation is −2.2 pp and −2.8 pp, driven by methods that sacrifice accuracy without any corresponding reduction in LS, confirming that accuracy variation, where present, does not reflect unlearning effectiveness.

Answer to RQ1: Link inference attacks provide substantially more information than accuracy, which is essentially uninformative across all benchmark datasets. E-UMIA is uniformly blind. Link Stealing is the most effective attack in this setting (high centrality): it produces a consistent, dataset-agnostic ranking with a spread of20–30pp that accuracy cannot reproduce. On GCN, only a fraction of the methods reduce LS below the Gold Model, and only at a non-trivial accuracy cost; most methods either match Identity or are strictly worse.

## 5.1.2 Impact of centrality-based sampling

All EU works in the literature sample forget set edges randomly. Lethe, instead, uses two settings: sampling the lowest and highest centrality edges defined in Section 4.2). We report per-method values in Appendix F.

On low-centrality edges, every method achieves LT $\begin{array} { l l l } { \Delta \mathrm { U C } } & { \approx } & { 0 . 7 5 \mathrm { - } 0 . 8 5 . } \end{array}$ except on Flickr. On highcentrality edges, LT collapses to a broader range for every method, landing near the random-chance boundary, but moving between 0.38–0.64 across datasets. The two settings are shown as connected dot pairs in Figure 3(a): every dumbbell spans roughly 0.35 AUC units. Figure 3(b) shows test accuracy for the same methods and settings on the same x-axis scale. For ev-

![](images/80bd845b3d9232ad88c4d9d6ed1f007dd89404ce083a7664a780b2efa743157c.jpg)  
Figure 3: (a) LinkTeller AUC and (b) Test accuracy mean and std across all methods on each dataset when unlearning low centrality (orange) and high centrality (blue) edges.

ery dataset, the low and high centrality dots on utility nearly overlap. This behavior is explained by how structural information propagates in GNNs. Low-centrality edges participate in relatively few message-passing paths and thus have a highly localized effect on node representations. When such an edge is removed, the change it induces is not compensated by alternative paths, making the absence of the edge clearly detectable by link inference attacks. High-centrality edges lie on many messagepassing paths and contribute to a large number of node representations. Due to the smoothing effect of GNN aggregation, their influence is diffused across the graph and partially redundant with other paths. When these edges are removed, their signal is absorbed by the surrounding structure.

To confirm that the low-high centrality LT gap is consistent, we apply a Wilcoxon signed-rank test [10] to the paired LT observations obtained by crossing all non-collapsing unlearning methods. The test statistic equals $W = 9 5 9 1$ , the theoretical maximum $( W _ { \mathrm { m a x } } = \dot { N } ( N \mathrm { \bar { + } } 1 ) / 2 )$ , because every one of the low-centrality LT values exceeds the corresponding high-centrality value $( p < 1 0 ^ { - 2 3 } )$ . The same test applied to test accuracy yields $W = 3 5 6 8 ( p = 0 . 1 4 )$ , confirming that the centrality setting has no detectable impact on predictive utility metrics such as accuracy.

Answer to RQ2: WalkCentrality-based settings, specificallyforget sets composed oflow/high centrality edges, highlight two main criticalities in current EU evaluation: (i). Low-centrality forget sets remain highly exposed to LinkTeller for every method on four offive datasets. Because these edges have a localized and non-redundant contribution to node representations, their removal leaves a detectable footprint that is not mitigated by existing unlearning strategies. (ii). Accuracy alone is not able to detect this discrepancy, as it remains largely unchanged in both settings. This confirms that utility-based metrics fail to capture the structural footprint of edges, and can therefore lead to misleading conclusions.

## 5.1.3 Applicability of certified methods in practice

CEU, CGU, and ScaleGUN all require a linear architecture such as SGC. They provide formal guarantees of closeness to the Gold Model, but as a trade-off, they are not applicable to non-linear GNNs, such as GCNs. This is an important limitation, as SGC consistently underperforms GCN in our benchmark: by 3.6 pp on AmazonPhotos, 4.2 pp on AmazonComputers, and 1.2 pp on Flickr (refer to Table 3). The certified guarantee therefore applies to a model that is weaker than any expressive architecture, mirroring the same restriction reported for certified node unlearning by [19]. Moreover, these methods are often tested with very small forget sets, in the order of tens of edges. When stress-tested with up to 5% of edges as the forget set, they are not scalable in terms of runtime, as shown in Figure 4. A runtime higher than the one of the Gold Model defeats the purpose of MU itself, as it is always the ideal solution.

Certified methods do not show consistent reduction on adversary success on attacks. As shown in Table 3, On Photos, both CGU and ScaleGUN worsen LS relative to Identity by 8.3 pp (CGU) and 6.6 pp respectively; CGU also worsens it on DBLP (+1.3 pp). The improvements they achieve on Flickr (−24.6 pp and −26.4 pp respectively) come at a severe accuracy cost: CGU collapses to 5.3% test accuracy on Flickr, disqualifying it as a viable solution on that dataset. All three certified methods increase LT relative to Identity on ogbn-arxiv (CEU +3.3 pp, CGU +6.6 pp, ScaleGUN +5.5 pp), meaning the unlearned model is more susceptible to influencebased link inference than the original model that was never modified. This disconnect arises because the certified guarantee bounds closeness to the Gold Model in parameter space, not in the attack-relevant geometry captured by LT and LS

![](images/b2e5105fc83f37f9c33a4c6fa97ae578ff21b5a95ddd52ae97efbeddc7698a27.jpg)  
Figure 4: Runtime of certified EU methods relative to the Gold Model (retraining from scratch) at 5% forget-set size across all benchmark datasets. All bars above the dashed line indicate methods slower than retraining.

Answer to RQ3: Certified EU methods impose a double restriction: they require SGC, a linear architecture that doesn’t match the performance ofnon-linear ones, and are only computationally viable when removing a handful of edges. At 5% forget set size, their runtime makes them impractical for all but the smallest datasets, and their formal guarantee does not translate into consistent reduced adversary success on link inference attacks.

## 6 Conclusion

In this work, we present Lethe, a specialized benchmark for Edge Unlearning. Lethe complements existing benchmarks by: (i) Proposing the WalkCentrality metric and using it to implement different forget set selection strategies, (ii). Implementing different link inference attacks to test the reliability of existing GU methods, and (iii). Implementing several general-purpose MU methods and extending them to the graph setting. In particular, we show how the evaluation of EU methods is systematically limited by dataset selection and uninformative metrics for feature-rich datasets. We show that accuracy and UMIA attacks are not informative for the task, and stress the importance of dedicated link inference attacks. We provide a comprehensive benchmark and a modular implementation that can be easily extended with other baselines, datasets, or measures. So far, no previous effort has had such a breadth of inference attacks, but we encourage the community to expand the evaluation.

As link inference attacks will get better over time, we need a benchmark that is both up-to-date and easily extensible. With Lethe, we tested state-of-the-art Link Inference attacks, but we could not possibly cover all existing attacks, or all downstream tasks for GNNs. At the same time, our aim is to give the community a benchmark that will adapt to future work. For this reason, as shown in Appendix E, Lethe is easily extendible to other datasets, models, tasks and even alternative negativeset constructions (e.g., held-out edges). Particular care was put into making the code modular and extensible towards this goal.

## Acknowledgments and Disclosure of Funding

This work was supported by Danish Data Science Academy, which is funded by the Novo Nordisk Foundation (NNF21SA0069429).

## References

[1] David Ahmedt-Aristizabal, Mohammad Ali Armin, Simon Denman, Clinton Fookes, and Lars Petersson. Graph-based deep learning for medical diagnosis and analysis: Past, present and future. Sensors, 21(14):4758, 2021.

[2] Maya Bechler-Speicher, Ben Finkelshtein, Fabrizio Frasca, Luis Müller, Jan Tönshoff, Antoine Siraudin, Viktor Zaverkin, Michael M. Bronstein, Mathias Niepert, Bryan Perozzi, Mikhail Galkin, and Christopher Morris. Position: Graph learning will lose relevance due to poor benchmarks. In Forty-second International Conference on Machine Learning Position Paper Track, 2025.

[3] Min Chen, Zhikun Zhang, Tianhao Wang, Michael Backes, Mathias Humbert, and Yang Zhang. Graph unlearning. In Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security, CCS ’22, page 499–513, New York, NY, USA, 2022. Association for Computing Machinery.

[4] Jiali Cheng, George Dasoulas, Huan He, Chirag Agarwal, and Marinka Zitnik. Gnndelete: A general strategy for unlearning in graph neural networks, 2023.

[5] Eli Chien, Chao Pan, and Olgica Milenkovic. Certified graph unlearning, 2022.

[6] Vikram S Chundawat, Ayush K Tarun, Murari Mandal, and Mohan Kankanhalli. Can bad teaching induce forgetting? unlearning in deep networks using an incompetent teacher. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 7210–7217, 2023.

[7] Corinna Coupette, Jeremy Wayland, Emily Simons, and Bastian Rieck. No metric to rule them all: Toward principled evaluations of graph-learning datasets. In Forty-second International Conference on Machine Learning, 2025.

[8] Kate Crawford. Atlas of AI: Power, Politics, and the Planetary Costs of Artificial Intelligence. Yale University Press, 2022.

[9] Andrea D’Angelo, Claudio Savelli, Gabriele Tagliente, Flavio Giobergia, Elena Baralis, and Giovanni Stilo. Erasure: A modular and extensible framework for machine unlearning. In Proceedings ofthe 34th ACM International Conference on Information and Knowledge Management, CIKM ’25, page 6346–6350, New York, NY, USA, 2025. Association for Computing Machinery.

[10] Janez Demšar. Statistical comparisons of classifiers over multiple data sets. Journal ofMachine Learning Research, 7(1):1–30, 2006.

[11] Yushun Dong, Binchi Zhang, Zhenyu Lei, Na Zou, and Jundong Li. Idea: A flexible framework of certified unlearning for graph neural networks. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’24, page 621–630, New York, NY, USA, 2024. Association for Computing Machinery.

[12] Cynthia Dwork. Differential privacy. In Michele Bugliesi, Bart Preneel, Vladimiro Sassone, and Ingo Wegener, editors, Automata, Languages and Programming, pages 1–12, Berlin, Heidelberg, 2006. Springer Berlin Heidelberg.

[13] Bowen Fan, Yuming Ai, Xunkai Li, Zhilin Guo, Zhu Lei, Guang Zeng, Rong-Hua Li, and Guoren Wang. OpenGU: A comprehensive benchmark for graph unlearning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2026.

[14] Chongyu Fan, Jiancheng Liu, Yihua Zhang, Eric Wong, Dennis Wei, and Sijia Liu. Salun: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation. arXiv preprint arXiv:2310.12508, 2023.

[15] Jack Foster, Stefan Schoepf, and Alexandra Brintrup. Fast machine unlearning without retraining through selective synaptic dampening. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 12043–12051, 2024.

[16] C.Lee Giles, Kurt D. Bollacker, and Steve Lawrence. Citeseer: An automatic citation indexing<sup>˜</sup> system. In I. Witten, R. Akscyn, and F.Shipman III, editors,<sup>˜</sup> Proceedings of the Third ACM Conference on Digital Libraries (DL ’98), pages 89–98, New York, 1998. ACM Press. https: //clgiles.ist.psu.edu/papers/DL-1998-citeseer.pdf.

[17] Aditya Golatkar, Alessandro Achille, and Stefano Soatto. Eternal sunshine of the spotless net: Selective forgetting in deep networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9304–9312, 2020.

[18] Xinlei He, Jinyuan Jia, Michael Backes, Neil Zhenqiang Gong, and Yang Zhang. Stealing links from graph neural networks. In 30th USENIX Security Symposium (USENIX Security 21), pages 2669–2686. USENIX Association, August 2021.

[19] Samyak Jain, Ronak Kalvani, sainyam galhotra, and Sayan Ranu. Is graph unlearning ready for practice? a benchmark on efficiency, utility, and forgetting. In The Fourteenth International Conference on Learning Representations, 2026.

[20] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks, 2017.

[21] Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, and Eleni Triantafillou. Towards unbounded machine unlearning. Advances in neural information processing systems, 36, 2024.

[22] Andrew Kachites McCallum, Kamal Nigam, Jason Rennie, and Kristie Seymore. Automating the construction of internet portals with machine learning. In Proceedings of the 17th International Conference on Machine Learning (ICML), pages 327–334, 2000.

[23] John Palowitch, Anton Tsitsulin, Brandon Mayer, and Bryan Perozzi. Graphworld: Fake graphs bring real insights for gnns. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’22, page 3691–3701, New York, NY, USA, 2022. Association for Computing Machinery.

[24] Jiezhong Qiu, Jian Tang, Hao Ma, Yuxiao Dong, Kuansan Wang, and Jie Tang. Deepinf: Social influence prediction with deep learning. In Proceedings ofthe 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’18, page 2110–2119, New York, NY, USA, 2018. Association for Computing Machinery.

[25] Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. Dropedge: Towards deep graph convolutional networks on node classification. In International Conference on Learning Representations, 2020.

[26] Jiajun Tan, Fei Sun, Ruichen Qiu, Du Su, and Huawei Shen. Unlink to unlearn: Simplifying edge unlearning in gnns. In Companion Proceedings ofthe ACM Web Conference 2024, WWW ’24, page 489–492, New York, NY, USA, 2024. Association for Computing Machinery.

[27] Ayush K Tarun, Vikram S Chundawat, Murari Mandal, and Mohan Kankanhalli. Fast yet effective machine unlearning. IEEE Transactions on Neural Networks and Learning Systems, 2023.

[28] Sandra Wachter, Brent Mittelstadt, and Luciano Floridi. Why a Right to Explanation of Automated Decision-Making Does Not Exist in the General Data Protection Regulation. International Data Privacy Law, 7(2):76–99, 2017.

[29] Fan Wu, Yunhui Long, Ce Zhang, and Bo Li. Linkteller: Recovering private edges from graph neural networks via influence analysis, 2021.

[30] Kun Wu, Jie Shen, Yue Ning, Ting Wang, and Wendy Hui Wang. Certified edge unlearning for graph neural networks. In Proceedings ofthe 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’23, page 2606–2617, New York, NY, USA, 2023. Association for Computing Machinery.

[31] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2019.

[32] Zhilin Yang, William Cohen, and Ruslan Salakhudinov. Revisiting semi-supervised learning with graph embeddings. In Maria Florina Balcan and Kilian Q. Weinberger, editors, Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 40–48, New York, New York, USA, 20–22 Jun 2016. PMLR.

[33] Lu Yi and Zhewei Wei. Scalable and certifiable graph unlearning: Overcoming the approximation error barrier. Poster presented at International Conference on Learning Representations (ICLR) 2025, 2025.

[34] Zhikun Zhang, Min Chen, Michael Backes, Yun Shen, and Yang Zhang. Inference attacks against graph neural networks, 2021.

## LLM Usage Disclosure

During the preparation of this work, the authors used LLMs to correct typos and grammatical mistakes, and to implement basic methods (the code of the methods was implemented manually). After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## Broader Impact

Lethe is designed as a defensive evaluation tool: its purpose is to help researchers identify and correct weaknesses in EU methods before those methods are deployed in privacy-sensitive settings. The link inference attacks it implements are, however, inherently dual-use. We release the full benchmark and attack implementations publicly to support reproducible research, as their implementation is already publicly available. We encourage practitioners to treat a low attack AUC as an empirical signal rather than a privacy certificate: it indicates that a specific adversary, operating under the defined threat model, cannot recover the forgotten edges. All datasets used in this benchmark are publicly available and contain no personally identifiable information.

## Released artifact and data access

The released artifact consists of the full benchmark codebase, configuration files, model and unlearning implementations, link-inference attack implementations. We refer the reader to the repository. The benchmark datasets themselves are not redistributed here, as they are all publicly available. The data was only preprocessed to make it compliant with the framework architecture.

## A Task and Attacks

We focus on the setting of edge unlearning for node classification on graphs, and formalize the notions of graph learning, unlearning, and certified unlearning.

Graph learning. Given a parameter space Θ, we refer to a Graph Neural Network (GNN) with parameters $\theta \in \Theta$ as $M _ { \theta } . \mathrm { \ A }$ graph G is a triple $G = ( N , E , X )$ ) with node set $N _ { \ast }$ , edge set $E \subseteq N \times N$ and node features $X ; T = \bar { \{ ( i , y _ { i } ) : i \in \bar { \mathcal { N } _ { T } } \} }$ is a training set defined over G, where $\mathcal { N } _ { T } \subseteq N$

Let $A ( G , T )$ be a randomized training algorithm that takes as input a graph G and a training set $T _ { \ast }$ and outputs the optimal model parameters $\theta ^ { \ast } \in \Theta$ by minimizing the empirical risk over the labeled nodes in $T \colon \theta ^ { * } = { \arg \operatorname* { m i n } _ { \theta } } { \mathcal { L } } ( \theta , G , T )$ . For ease of notation, we will refer to $M _ { \theta ^ { * } }$ ∗ as $M ^ { * }$

Edge Unlearning. At any time after model training, the model holder can receive an unlearning request that specifies a set of edges to be forgotten, called the edge forget set $E _ { f } \subseteq E$ . The retain graph $G ^ { \prime }$ is the result of the removal of $E _ { f }$ from G, i.e., $G ^ { \prime } = ( \tilde { N , } E \setminus \tilde { E } _ { f } , X )$

It follows that the ideal solution would be to retrain $M ^ { * }$ from scratch on $G ^ { \prime }$ . We refer to this solution as the gold model $M _ { G } = A ( G ^ { \prime } , T )$ , parametrized by $\theta _ { G }$ using the same training algorithm as $M ^ { * }$

Definition 1 (Gold model). Given a randomized training algorithm A and a retain graph $G ^ { \prime }$ , the gold model $M _ { G } = A ( G ^ { \prime } , T )$ is the model obtained by running A from scratch on $\dot { G ^ { \prime } }$ , without any exposure to edges in theforget set.

The aim of the model holder at this stage is to avoid full retraining, i.e., $M _ { G }$ , and instead use an unlearning method to approximate it as closely as possible:

Definition 2 (Edge Unlearning (EU)). Given a trained model $M ^ { * } = A ( G , T )$ , the retain graph $G ^ { \prime }$ and the forget set $E _ { f } ,$ , the goal is to find an updated model $M _ { \theta _ { u } }$ that closely approximates $M _ { G }$ and leaks no information on the edges in $E _ { f }$

In the following, we refer to the unlearned model $M _ { \theta _ { \imath } }$ as simply $M _ { u }$ . Clearly, the gold model $M _ { G }$ i the reference output model for almost all metrics in evaluating unlearning.

Certified Unlearning. Certified methods formalize the notion of closeness between $M _ { u }$ and $M _ { G }$ by drawing on the framework of differential privacy [12]. Specifically, an EU method is said to be $( \epsilon , \delta )$ -certified if, for every forget set $E _ { f } \subseteq E$ and every subset $\Theta ^ { \prime } \subseteq { \bar { \Theta } }$ , we have:

(i) $P [ \theta _ { u } \in \Theta ^ { \prime } ] \leq e ^ { \epsilon } P [ \theta _ { G } \in \Theta ^ { \prime } ] + \delta$

$$
\begin{array} { r l } { ( i i ) } & { { } P [ \theta _ { G } \in \Theta ^ { \prime } ] \le e ^ { \epsilon } P [ \theta _ { u } \in \Theta ^ { \prime } ] + \delta } \end{array}
$$

Together, conditions (i) and (ii) impose a bidirectional bound on the distributional discrepancy between $\theta _ { u }$ and $\theta _ { G }$ . Intuitively, ϵ controls how tightly the two distributions must agree, while δ allows for a small relaxation. This definition is adapted from [11, 5, 30].

## A.1 Link Inference Attacks

While accuracy-based metrics ensure that $M _ { u }$ preserves predictive performance, they are largely insensitive to edge removals if node features are informative (Section 3). To directly assess whether the forgotten edge information has been removed from the model, Lethe adopts three established link inference attacks, whereby an adversary with black-box access to $M _ { u }$ queries the model to determine whether a specific edge $e \in E _ { f }$ was present during training. The goal is to standardize evaluation of the resilience of unlearning methods to adversaries attempting to recover forgotten information.

Threat model. We consider a membership inference adversary that has access to the unlearned model $M _ { u }$ and to the retain graph $G ^ { \prime } = ( N , \dot { E ^ { } } \backslash E _ { f } , X )$ , but not to the original graph G or to $E _ { f } .$ . The adversary must classify a candidate edge $( u , v )$ as either a forgotten edge (positive) or a non-existing edge (negative). Crucially, all attacks query $M _ { u }$ using $G ^ { \prime }$ as the message-passing graph: providing G would trivially inflate the scores for edges in $E _ { f }$ through direct propagation, regardless of what the model has actually learned. A well-unlearned model should be indistinguishable from the gold model $M _ { G } = A ( G ^ { \prime } , T )$ , making forgotten edges look like non-existing ones; we report AUC (or accuracy) relative to the 0.5 random-chance baseline.

Positive and negative sets. For each attack, the positive set consists of the forgotten edges $E _ { f }$ deduplicated to undirected pairs. The negative set is drawn uniformly at random from all non-existing pairs $( u , v ) \notin E$ , balanced to match $| \bar { E } _ { f } |$ . This unbiased sampling avoids introducing structural confounders, such as selecting non-edges $( u , v )$ where u and v have many shared neighbours that would systematically inflate their scores relative to forgotten edges whose direct connection has been removed.

Edge Membership Inference Attack (UMIA). We adapt the population membership inference attack of [21] to the edge setting. Because edge removal affects nodes indirectly through the messagepassing neighbourhood rather than directly, we operate at the node level on the set of infected nodes (see Section 3). These are the nodes whose k-hop neighbourhood overlaps with at least one endpoint of a forgotten edge, and whose learned representations may carry residual information about $E _ { f }$

The attack proceeds as follows. (1) Run $M _ { u }$ on $G ^ { \prime }$ and collect the per-node cross-entropy loss for every node. (2) Assign label 1 to infected nodes in the forget set $\dot { \sqrt { f } }$ and label 0 to test nodes $\mathcal { N } _ { T }$ (3) Train a logistic regression classifier on the scalar losses using 3-fold stratified cross-validation (4) Report the mean accuracy. An accuracy of 0.5 indicates perfect unlearning: the loss distribution over infected nodes is indistinguishable from the test distribution.

Table 4: Fraction of nodes (%) affected by 2-hop neighbourhood expansion when removing a random subset of edges. Values averaged over 5 random forget-set samples. Bold entries exceed 80%.
<table><tr><td>Dataset</td><td>|V|</td><td>|E|</td><td>1%</td><td>5%</td><td>10%</td><td>20%</td><td>50%</td></tr><tr><td>Cora</td><td>2,708</td><td>5,278</td><td> $5 8 . 0 { \pm } 2 . 3 $ </td><td> ${ \bf 8 5 . 0 \pm 0 . 5 }$ </td><td> ${ \bf 9 0 . 7 \pm 0 . 6 }$ </td><td> ${ \bf 9 4 . 2 \pm 0 . 3 }$ </td><td>97.6±0.4</td></tr><tr><td>Citeseer</td><td>3,327</td><td>4,552</td><td> $2 6 . 3 { \pm } 1 . 3 $ </td><td> $5 3 . 1 { \pm } 0 . 9 $ </td><td> $6 5 . 1 { \pm } 0 . 4 $ </td><td> $7 5 . 9 2 0 . 3 $ </td><td> $\mathbf { 8 8 . 5 \pm 0 . 2 }$ </td></tr><tr><td>Pubmed</td><td>19,717</td><td>44,324</td><td> $7 6 . 2 \pm 1 . 4$ </td><td> ${ \bf 9 4 . 5 } \pm { \bf 0 . 3 }$ </td><td> ${ \bf9 7 . 8 } { \pm } { \bf 0 . 2 }$ </td><td> ${ \bf 9 9 . 3 \pm 0 . 1 }$ </td><td> ${ \approx } 1 0 0 \pm 0 . 0$ </td></tr></table>

LinkTeller (LT). LinkTeller [29] infers edge existence by measuring the influence of node v’s features on node u’s output embedding. If the model retains memory of a forgotten edge $( u , v )$ perturbing $v { \mathrm { s } }$ input features will produce a larger change in $u \mathrm { { s } }$ output than for a truly non-existing pair, because the learned parameters still encode the dependency introduced during training on $G .$

Formally, the influence score is the directional derivative of $u \mathrm { { s } }$ embedding with respect to $v { \mathrm { s } }$ features, in the direction of $v { \mathrm { s } }$ own feature values, approximated via a first-order finite difference:

$$
s ( u , v ) = \left. \frac { M _ { u } ( X + \varepsilon \cdot \mathbf { x } _ { v } \mathbf { e } _ { v } ^ { \top } , G ^ { \prime } ) [ u ] \ - \ M _ { u } ( X , G ^ { \prime } ) [ u ] } { \varepsilon } \right. _ { 2 } ,\tag{1}
$$

where $\mathbf { e } _ { v }$ is the standard basis vector for node v, so that $\mathbf { x } _ { v } \mathbf { e } _ { v } ^ { \top }$ perturbs only row v of the feature matrix by a fraction ε of its own value. Both model calls use $\bar { G ^ { \prime } }$ as the edge index. We compute $s ( u , v )$ for all positive pairs $( u , v ) \in E _ { f }$ and all negative pairs, and report the ROC-AUC of the resulting scores. An AUC of 0.5 indicates that the model assigns equivalent influence scores to forgotten and non-existing edges.

Link Stealing Attack (LS). Link stealing attacks [18] exploit the observation that a GNN trained on a graph tends to produce more similar output distributions for pairs of nodes that are connected than those that are not. If unlearning is effective, the posteriors of forgotten edge endpoints should become as dissimilar as those of non-existing pairs.

Given the unlearned model $M _ { \underline { { u } } }$ evaluated on $G ^ { \prime }$ , we compute the softmax class posteriors $p _ { v } =$ softmax $\left( M _ { u } ( X , G ^ { \prime } ) [ v ] \right) \in \mathbb { R } ^ { \tilde { C } }$ for every node $v ,$ where $\bar { \boldsymbol { C } }$ is the number of classes. For each candidate pair $( u , v )$ , the attack score is the cosine distance between posteriors:

$$
s ( u , v ) = 1 - \frac { p _ { u } \cdot p _ { v } } { \| p _ { u } \| _ { 2 } \| p _ { v } \| _ { 2 } } .\tag{2}
$$

We report the ROC-AUC of these scores over positive (forgotten) and negative (non-existing) pairs, symmetrised to be $\ge 0 . 5 .$ . An AUC of 0.5 indicates that the model’s posteriors no longer reveal membership of edges in $E _ { f }$

## A.2 Infected nodes study

As shown in Table 4, removing even 1% of edges on these datasets infects 58–76% of all nodes within two hops, meaning that MU methods are effectively operating on almost the entire graph, leaving little room to amortise the cost of unlearning against the cost of retraining. When the forget set grows to 5% of edges, the infected set of nodes within two hops represent more than 85% of nodes.

## B WalkCentrality experiments

In this section we show two empirical experiments to support the definition of WalkCentrality.

(i) Redundancy absorbs removal. We compute the correlation between WalkCentrality and the increase in training loss when the corresponding edges are removed, that is, how much the model’s fit depended on them. The correlation scores are shown in Table 5.

The strong negative correlation confirms that the higher an edge’s WalkCentrality, the more completely the rest of the graph absorbs its removal.

Table 5: Spearman correlation between WalkCentrality and the increase in training loss after edge removal.
<table><tr><td>Dataset</td><td> $\rho$  (WalkCentrality, loss increase)</td></tr><tr><td>Photos</td><td>-0.989</td></tr><tr><td>Computers</td><td>-0.993</td></tr><tr><td>DBLP</td><td>-0.989</td></tr></table>

Table 6: Endpoint displacement after edge removal for low- and high-WalkCentrality edges.
<table><tr><td>Dataset</td><td>Low WalkCentrality</td><td>High WalkCentrality</td><td>Ratio</td></tr><tr><td>Photos</td><td>0.309</td><td>0.060</td><td>5.2×</td></tr><tr><td>Computers</td><td>0.274</td><td>0.042</td><td>6.7×</td></tr><tr><td>DBLP</td><td>0.289</td><td>0.056</td><td>5.3×</td></tr></table>

(ii) Low-centrality removals are localized and large. For a removed edge $\boldsymbol { e } = ( u , v )$ , let

$$
d _ { i } = \lVert h _ { i } ( G ) - h _ { i } ( G \setminus e ) \rVert _ { 2 }
$$

be how far node $i \ ' s$ hidden representation moves (64-dimensional post-ReLU layer). We report

$$
m ( e ) = \sqrt { d _ { u } ^ { 2 } + d _ { v } ^ { 2 } } ,
$$

the displacement of the removed edge’s own two endpoints, and give the median of $m ( e )$ over the edges of each forget set. Results are shown in Table 6.

The two results form a single chain: high-WalkCentrality edges are redundant (i), so their removal displaces their endpoints only slightly (ii). LinkTeller measures exactly this endpoint influence, which is why it reaches $0 . 7 3 \mathrm { - } 0 . 8 5$ in the low-WalkCentrality setting and falls to 0.38–0.64 in the high-WalkCentrality one (Figure 3(a)).

## C Ablation Studies

## C.1 Ablation on the attacker’s accessible information

In this ablation study, we ablate the adversary’s accessible information across four levels for LinkStealing (LS): we ablate the attacker’s access to the unlearned model + 0%, 25%, 50%, 100% of the retained graph (the 100% setting is the one in the paper) on GCN.

We tested this ablation on both Photos and Flickr, the datasets with the highest LS spread in the paper, across all unlearning methods.

Table 7 shows the results for this ablation. Notice how, uniformly for each method, the attacker’s success increases as they gain access to more information. However, different percentages of the retain graph behave closely.

Table 8 shows that this spread is consistent on both datasets, with one-sided F-test $\sigma ^ { 2 } ( \mathrm { L S } ) \mathrm { > }$ $\sigma ^ { 2 } ( \mathrm { E - U M A } )$ in brackets.

At every level with any structural access it separates methods significantly, where accuracy (spread within ±7 pp of the original model on flickr and ±12 on Photos) and E-UMIA do not. So, across all settings, LS is significantly more informative than both E-UMIA and Accuracy.

Because of this, we can conclude that LS discriminates better than accuracy and E-UMIA even when the attacker has less information, showing that the answer to RQ1 is robust.

The conclusion is therefore stable to this ablation, although attack effectiveness decreases substantially when structural information is unavailable (as expected).

Table 7: LinkStealing ablation on attacker’s access.
<table><tr><td>Method</td><td>None</td><td>25%</td><td>50%</td></tr><tr><td>Identity</td><td>0.518</td><td>0.641</td><td>0.639</td></tr><tr><td>Gold Model</td><td>0.547</td><td>0.673</td><td>0.672</td></tr><tr><td>Finetuning</td><td>0.522</td><td>0.636</td><td>0.634</td></tr><tr><td>NegGrad</td><td>0.564</td><td>0.681</td><td>0.678</td></tr><tr><td>Adv.NegGrad</td><td>0.554</td><td>0.685</td><td>0.682</td></tr><tr><td>UNSIR</td><td>0.776</td><td>0.861</td><td>0.863</td></tr><tr><td>BadTeaching</td><td>0.581</td><td>0.699</td><td>0.696</td></tr><tr><td>Scrub</td><td>0.662</td><td>0.757</td><td>0.758</td></tr><tr><td>Fisher</td><td>0.521</td><td></td><td></td></tr><tr><td></td><td></td><td>0.653</td><td>0.651</td></tr><tr><td>SSD</td><td>0.513</td><td>0.626</td><td>0.624</td></tr><tr><td>SalUn</td><td>0.609</td><td>0.730</td><td>0.726</td></tr><tr><td>spread</td><td>26 pp</td><td>23 pp</td><td>24 pp</td></tr></table>

Table 8: F-tests on LinkStealing attacker’s access ablation
<table><tr><td>Access to  $G ^ { \prime }$ </td><td>Photos</td><td>Flickr</td></tr><tr><td>0%</td><td> $2 6 \mathrm { p p } ( \boldsymbol { p = } 3 \times 1 0 ^ { - 9 } )$ </td><td> $2 { \tt p p } ( p { = } 0 . 4 )$ </td></tr><tr><td>25%</td><td> $2 3 \mathrm { p p } (  { p = } 2 \times 1 0 ^ { - 8 } )$ </td><td> $6 \mathsf { p p } ( p { = } 3 \times 1 0 ^ { - 3 } )$ </td></tr><tr><td>50%</td><td> $2 4 \mathrm { p p } (  { p = } 2 \times 1 0 ^ { - 8 } )$ </td><td>11 pp  $( p { = } 1 \times 1 0 ^ { - 5 } )$ </td></tr><tr><td>100%</td><td>24  $\mathsf { p p } ( p { = } 1 \times 1 0 ^ { - 8 } )$ </td><td>14 pp  $( p { = } 2 \times 1 0 ^ { - 6 } )$ </td></tr></table>

## C.2 Other forget set selection strategies

To show that WalkCentrality forget sets act as difficulty bounds, we introduce two new, more realistic selection strategies, and compare them.

• Community-based forgetting: We simply pick a class of users whose edges we will be removing. For Photos, that is a product category. For DBLP, it is a research community. For arxiv, it is a subject area. Both endpoints need to be in the same class.

• Time-based forgetting: We use arxiv and date each edge as the max among the publication year of the endpoints (a citation can’t exist until the later paper is published). We order the edges, and we remove all the older ones to reach 5%.

All experiments are run with a GCN on a 5% forget set and across all unlearning methods.

For scenario 1, we show that the accuracies and E-UMIA are still uninformative with respect to our settings. This is reported in Table 9.

Table 10 shows the Link Teller value on a subset of unlearning methods. It is easy to see that, for all methods (except GNNdelete, who is also collapsing the model), the community Link Teller is between the low and high centrality settings. Basically, our settings are stress-testing the methods; realistic forget sets, like from a community, are likely to land between our values, because they contain some low centrality edges along with high centrality ones.

For scenario 2, on arxiv, the same claims hold. Accuracy and E-UMIA are uninformative, as shown in Table 11.

The realistic, time-based selection of forget set lands in the middle of the two centrality settings in terms of Link Teller, as shown in 12, although the gap is much narrower with respect to other datasets.

Accuracy and UMIA are equally uninformative when theforget set selection is not random, but   
selected through different means, like Community or Time. In these cases, the Link Teller values   
fall between the low and high WalkCentrality settings, which correctly act as "difficulty" bounds   
for link inference retrieval.

Table 9: Metrics on the community-built forget set on DBLP.
<table><tr><td>Metric</td><td>Low Centrality</td><td>High Centrality</td><td>Community</td></tr><tr><td>Accuracy (test)</td><td>0.8802</td><td>0.8873</td><td>0.8699</td></tr><tr><td>Accuracy (forget)</td><td>0.8901</td><td>0.8964</td><td>0.8548</td></tr><tr><td>E-UMIA</td><td>0.5162</td><td>0.5128</td><td>0.4981</td></tr></table>

Table 10: Link Teller values for the community-built forget set on DBLP.
<table><tr><td>Method</td><td>Low Centrality</td><td>High Centrality</td><td>Community</td></tr><tr><td>Original</td><td>0.7480</td><td>0.4535</td><td>0.6786</td></tr><tr><td>GoldModel</td><td>0.7486</td><td>0.4589</td><td>0.6773</td></tr><tr><td>Finetuning</td><td>0.7481</td><td>0.4717</td><td>0.6756</td></tr><tr><td>NegGrad</td><td>0.7788</td><td>0.4487</td><td>0.6701</td></tr><tr><td>AdvNegGrad</td><td>0.7491</td><td>0.4600</td><td>0.6697</td></tr><tr><td>UNSIR</td><td>0.7572</td><td>0.4626</td><td>0.6715</td></tr><tr><td>BadTeaching</td><td>0.7492</td><td>0.4571</td><td>0.6769</td></tr><tr><td>Scrub</td><td>0.7473</td><td>0.4573</td><td>0.6768</td></tr><tr><td>FisherForg.</td><td>0.7424</td><td>0.4545</td><td>0.6764</td></tr><tr><td>SSD</td><td>0.7436</td><td>0.4658</td><td>0.6788</td></tr><tr><td>SalUn</td><td>0.7545</td><td>0.4481</td><td>0.6659</td></tr><tr><td>GNNDelete</td><td>0.6630</td><td>0.4329</td><td>0.3982</td></tr></table>

## C.3 Results on Link Prediction

All of the graph-based Unlearning methods that we include in our benchmark were designed and tested for node classification. Hence, Lethe is also benchmarked on node classification. In this Section, we ablate the downstream task of link prediction to Lethe and run it on Photos and DBLP with GCN, under both WalkCentrality forget-set settings (high and low centrality).

To be as faithful as possible to the main text experiments, the encoder is the same GCN; a decoder scores a node pair (u, v) from the two endpoint embeddings, $s ( u , v ) = \mathrm { M L P } ( [ \mathbf { z } _ { u } \| \mathbf { z } _ { v } ] )$

We would like to clarify that, even in this setting, WalkCentrality is unchanged. It is defined on ${ \hat { A } } ,$ a property of the graph, independent of the task or the model. LinkTeller also remains unchanged, and Link Stealing needs only one change: the pairwise distance is taken on the node embeddings instead of on softmax-normalised posteriors. The attack logic is otherwise identical.

The resulting link prediction models are strong, so the evaluation is meaningful: the original model reaches an AUC of 0.960 on Photos, 0.856 on DBLP on the test set.

We then measure both link inference attacks under low- and high-centrality forget sets, exactly as in Section 5. We also report the same statistical tests. Results are shown in Table 13, and the mean across all methods is reported in Table 14.

Applying the same one-sided Wilcoxon signed-rank test used in the paper, paired over methods, gives $p = \bar { 3 } \times \bar { 1 } 0 ^ { - 5 }$ for Link Stealing and $p = \bar { 1 } \times 1 0 ^ { - 3 }$ for LinkTeller when the comparisons are pooled. This is exactly coherent with the Node classification results we show in the paper. However, as shown in the table, the magnitude of the gap is smaller relative to node classification.

Both WalkCentrality and the Lethe pipeline effortlessly generalize to link prediction and similar downstream tasks. The RQ2 conclusion replicates on link prediction: low-centralityforget sets are more exposed to link inference than high-centrality ones, for every method, on every dataset, under both Link Stealing and Link Teller: all paired comparisons agree in direction.

Finally, the collapse reported in the paper reproduce on the new task: GNNDelete collapses under link prediction as well, losing between 12 and 52 AUC points relative to the original model depending on dataset and setting, consistent with our node classification experiments and with [19].

Table 11: Performance across different node groups and temporal windows.
<table><tr><td>Metric</td><td>Low Centrality</td><td>High Centrality</td><td>Temporal Window</td></tr><tr><td>Accuracy (test)</td><td>0.5168</td><td>0.5151</td><td>0.5138</td></tr><tr><td>Accuracy (forget)</td><td>0.5276</td><td>0.5262</td><td>0.5220</td></tr><tr><td>E-UMIA</td><td>0.5051</td><td>0.5040</td><td>0.5020</td></tr></table>

Table 12: Performance across different node groups and temporal windows for each method.
<table><tr><td>Method</td><td>Low Centrality</td><td>High Centrality</td><td>Temporal Window</td></tr><tr><td>Original</td><td>0.7906</td><td>0.6357</td><td>0.7179</td></tr><tr><td>GoldModel</td><td>0.7707</td><td>0.6284</td><td>0.7185</td></tr><tr><td>Finetuning</td><td>0.7677</td><td>0.6191</td><td>0.7183</td></tr><tr><td>NegGrad</td><td>0.7665</td><td>0.6275</td><td>0.7150</td></tr><tr><td>AdvNegGrad</td><td>0.7622</td><td>0.6211</td><td>0.7205</td></tr><tr><td>UNSIR</td><td>0.7681</td><td>0.6264</td><td>0.7154</td></tr><tr><td>BadTeaching</td><td>0.7677</td><td>0.6279</td><td>0.7141</td></tr><tr><td>Scrub</td><td>0.7659</td><td>0.6246</td><td>0.7192</td></tr><tr><td>FisherForg.</td><td>0.7681</td><td>0.6275</td><td>0.7134</td></tr><tr><td>SSD</td><td>0.7664</td><td>0.6256</td><td>0.7185</td></tr></table>

## D Methods

Original and Gold Model. The Original model (M<sup>∗</sup>) is the model before unlearning and serves as a reference point for evaluating utility and forgetting. The Gold model $( M _ { G } )$ is obtained by retraining from scratch after removing the forget set and therefore represents the exact retraining baseline.

Fine-Tuning. Fine-Tuning adapts the pretrained model using the available training data without explicitly optimizing against the forget set. It provides a simple baseline for evaluating whether standard model adaptation can achieve unlearning.

NegGrad and AdvancedNegGrad. NegGrad performs gradient ascent on the forget set to increase the loss on forgotten examples. AdvancedNegGrad (NegGrad+) combines this forgetting objective with gradient descent on retained data, aiming to preserve model utility while degrading performance on the forget set.

UNSIR. UNSIR uses an impair–repair procedure in which the model is first impaired using noisebased perturbations and subsequently repaired using retained data. The procedure aims to remove information associated with the forget set while recovering performance on retained examples.

Bad Teaching and SCRUB. Bad Teaching uses knowledge distillation to train a student model while suppressing information associated with the data to be forgotten. SCRUB similarly uses a teacher–student distillation framework, combining forgetting and retention objectives to remove information while maintaining utility.

Fisher Forgetting and SSD. Fisher Forgetting uses the Fisher information matrix to identify parameters that are important for the forget set and performs weight scrubbing. SSD (Selective Synaptic Dampening) similarly uses Fisher-information-based parameter importance, selectively dampening parameters associated with the data to be forgotten.

SalUn. SalUn uses parameter saliency to identify model weights associated with the forget set and applies a saliency-guided unlearning procedure involving random-label training.

GNNDelete. GNNDelete is a graph-native unlearning method that inserts a trainable, layer-wise deletion operator into the GNN, which transforms the node representations in the vicinity of the deleted nodes or edges while the original model weights remain frozen. It is optimized to jointly satisfy two properties: consistency on deleted edges and preserved influence on neighboring nodes.

Table 13: Link prediction performance on Photos and DBLP across low- and high-centrality forget sets. LP = Link Prediction. LT = Link Teller. Low = Low WalkCentrality set. High = High WalkCentrality set.
<table><tr><td rowspan="3">Method</td><td colspan="4">Photos</td><td colspan="4">DBLP</td></tr><tr><td colspan="2">Low</td><td colspan="2">High</td><td colspan="2">Low</td><td colspan="2">High</td></tr><tr><td>LP AUC</td><td>LT</td><td>LP AUC</td><td>LT</td><td>LP AUC</td><td>LT</td><td>LP AUC</td><td>LT</td></tr><tr><td>Original</td><td>0.960</td><td>0.947</td><td>0.956</td><td>0.929</td><td>0.856</td><td>0.757</td><td>0.849</td><td>0.719</td></tr><tr><td>Gold Model</td><td>0.955</td><td>0.949</td><td>0.813</td><td>0.932</td><td>0.859</td><td>0.741</td><td>0.851</td><td>0.674</td></tr><tr><td>Finetuning</td><td>0.944</td><td>0.959</td><td>0.929</td><td>0.922</td><td>0.855</td><td>0.736</td><td>0.847</td><td>0.724</td></tr><tr><td>NegGrad</td><td>0.947</td><td>0.957</td><td>0.937</td><td>0.922</td><td>0.855</td><td>0.736</td><td>0.848</td><td>0.724</td></tr><tr><td>Adv. NegGrad</td><td>0.945</td><td>0.958</td><td>0.941</td><td>0.924</td><td>0.855</td><td>0.736</td><td>0.848</td><td>0.724</td></tr><tr><td>GNNDelete</td><td>0.747</td><td>0.940</td><td>0.748</td><td>0.900</td><td>0.731</td><td>0.761</td><td>0.589</td><td>0.698</td></tr></table>

Table 14: Mean LS and LT values for Link prediction.
<table><tr><td>Attack</td><td>Forget Set</td><td>Photos</td><td>DBLP</td></tr><tr><td>Link Stealing</td><td>Low Centrality</td><td>0.950</td><td>0.860</td></tr><tr><td>Link Stealing</td><td>High Centrality</td><td>0.888</td><td>0.812</td></tr><tr><td>LinkTeller</td><td>Low Centrality</td><td>0.954</td><td>0.741</td></tr><tr><td>LinkTeller</td><td>High Centrality</td><td>0.926</td><td>0.713</td></tr></table>

CGU and CEU. CGU and CEU are graph-native methods based on influence-function approximations. They estimate the effect of removing training instances on the model and use this information to update the GNN without full retraining. Both provide (ε, δ)-style certified guarantees under their respective assumptions.

IDEA. IDEA is a graph-native influence-function-based unlearning method that approximates the effect of individual training instances on the model. It provides an instance-level certified guarantee while avoiding complete retraining.

ScaleGUN. ScaleGUN combines influence-function-based parameter updates with approximate propagation mechanisms designed for scalability on graphs. It provides (ε, δ)-style certified guarantees while reducing the computational cost of graph unlearning.

UnlinkToUnlearn. UnlinkToUnlearn performs graph-native unlearning by removing forget-set edges from the input graph and simply re-running inference on this retain graph with the original, unmodified model weights. It requires no parameter update and no explicit retain set, relying instead on the absence of the forgotten edges to block their influence during message passing.

## E Running Experiments with Lethe

Lethe follows a fully configuration-driven design similarly to ERASURE [9], but completely rebuilding the framework from scratch to support graphs. All experiments are launched through a single entry point:

```batch
python main.py <config>.jsonc
```

Each configuration file (JSONC format) fully specifies an experiment through four top-level sections.

Data. The data block defines the dataset and partitioning strategy. The graph is split into train, validation, test, forget, and retain partitions. The forget set is constructed by a centralityaware splitter whose mode field selects either low-centrality or high-centrality edges, and whose percentage controls the forget-set size (e.g. 5%). A fixed split\_seed ensures reproducibility across runs.

Table 15: Overview of the unlearning methods considered in this work. “Graph-native” denotes methods that were originally proposed for and evaluated on graph neural networks (GNNs).
<table><tr><td>Method</td><td>Family</td><td>Graph-native</td><td>Certified</td><td>Needs retain set</td><td>Needs forget set</td></tr><tr><td>Original (M*)</td><td>Reference</td><td></td><td></td><td>x</td><td>x</td></tr><tr><td>Retrain / Gold (MG)</td><td>Reference</td><td></td><td>Exact</td><td>√</td><td>x</td></tr><tr><td>Fine-Tuning</td><td>Training</td><td>x</td><td>x</td><td>√</td><td>x</td></tr><tr><td>NegGrad</td><td>Gradient ascent</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>AdvancedNegGrad (NegGrad+)</td><td>Gradient ascent + descent</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>UNSIR</td><td>Noise / impair-repair</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>Bad Teaching</td><td>Knowledge distillation</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>SCRUB</td><td>Knowledge distillation</td><td>×</td><td>x</td><td>√</td><td>√</td></tr><tr><td>Fisher Forgetting</td><td>Weight scrubbing (Fisher)</td><td>x</td><td>x</td><td>√</td><td>x</td></tr><tr><td>SSD</td><td>Weight dampening (Fisher)</td><td>x</td><td>x</td><td>√(train)</td><td>7</td></tr><tr><td>SalUn</td><td>Saliency mask + random labels</td><td>×√</td><td>x</td><td>√</td><td></td></tr><tr><td>GNNDelete</td><td>Structural / deletion operator</td><td></td><td>x</td><td>x</td><td></td></tr><tr><td>CGU</td><td>Influence function</td><td>S</td><td>3 (ε, δ)</td><td>√</td><td>V</td></tr><tr><td>CEU IDEA</td><td>Influence function</td><td></td><td>3 (ε, δ)</td><td>√</td><td></td></tr><tr><td>ScaleGUN</td><td>Influence function</td><td></td><td>√(instance-level)</td><td>√ V</td><td></td></tr><tr><td></td><td>Influence function + approx. propagation</td><td></td><td>3 (ε, δ)</td><td></td><td></td></tr><tr><td>UnlinkToUnlearn</td><td>Propagation on retain graph</td><td></td><td>x</td><td>x</td><td></td></tr></table>

Predictor. The predictor block specifies the GNN architecture (e.g., GCN, GIN, GAT, GRAPH-SAGE, SGC) together with training hyperparameters such as learning rate, number of epochs, and loss function. A unique alias identifies the model for caching: the original model M<sup>∗</sup> and the Gold Model M<sub>G</sub> are trained once and reused across all unlearning methods in the same configuration.

Unlearners. The unlearners field is a list of unlearning methods to evaluate sequentially. Each entry specifies a method class and its hyperparameters. The Identity baseline (original model, no unlearning) and the Gold Model (full retraining on the retain set) are always included as reference points. Adding a new method requires only appending a new entry to this list.

Evaluator. The evaluator block enumerates the metrics to compute for each method: runtime, node classification accuracy (test, forget, and train splits), macro-F1 score, E-UMIA, LinkTeller, and Link Stealing. Results are serialised to a JSON file for downstream analysis and aggregation.

Configurations. All configurations used in our benchmark are provided under configs/benchmark/lethe/, with one file per (dataset, architecture, centrality-setting) triple. Extending Lethe to a new dataset or method requires writing a new configuration file or appending to an existing one; no changes to the framework code are needed.

## E.1 Extending Lethe

Lethe is designed to be modular: every component is resolved at runtime from the class and parameters fields of the configuration file, so adding a new dataset, architecture, unlearning method, or attack requires no changes to the framework core.

New dataset. Any dataset available in PyTorch Geometric can be included by pointing the data block at the corresponding class:

"datasource": {1 "class": "torch\_geometric.datasets.MyDataset",   
2 "parameters": {"root": "resources/data"}}

Custom datasets can be integrated by wrapping them in a TorchGeometricDataSource-compatible class that exposes the standard PyG Data interface.

New GNN architecture. A new architecture is a standard torch.nn.Module placed under lethe/model/graphs/. It only needs to implement a forward(x, edge\_index) signature. Once defined, it is referenced in the predictor block of any configuration file without further registration.

New unlearning method. A new unlearning method subclasses GraphUnlearner and implements the unlearning logic. The base class provides access to the trained model, the forget and retain partitions, and the infected-node computation. The new class is then added to the unlearners list of any configuration file:

Table 16: RunTime comparison on GCN architecture.
<table><tr><td>Method</td><td>DBLP</td><td>Photos</td><td>Computers</td><td>Flickr</td><td>ogbn-arxiv</td></tr><tr><td>Gold Model – GCN</td><td>9.1</td><td>12.4</td><td>17.1</td><td>14.0</td><td>45.6</td></tr><tr><td>Finetuning</td><td>0.8</td><td>7.9</td><td>8.9</td><td>2.2</td><td>9.5</td></tr><tr><td>NegGrad</td><td>0.8</td><td>7.2</td><td>8.9</td><td>3.8</td><td>9.6</td></tr><tr><td>AdvNegGrad</td><td>0.9</td><td>6.9</td><td>9.0</td><td>3.7</td><td>9.9</td></tr><tr><td>SCRUB</td><td>0.9</td><td>6.5</td><td>9.2</td><td>1.8</td><td>10.1</td></tr><tr><td>BadTeaching</td><td>0.9</td><td>7.1</td><td>9.1</td><td>3.8</td><td>12.3</td></tr><tr><td>SSD</td><td>0.9</td><td>6.8</td><td>9.0</td><td>1.6</td><td>9.9</td></tr><tr><td>Fisher</td><td>1.1</td><td>9.2</td><td>10.5</td><td>21.1</td><td>13.4</td></tr><tr><td>UNSIR</td><td>2.5</td><td>12.3</td><td>17.6</td><td>78.3</td><td>77.3</td></tr><tr><td>SalUn</td><td>10.1</td><td>12.7</td><td>31.2</td><td>6.2</td><td>49.9</td></tr><tr><td>GNNDelete</td><td>1.2</td><td>5.1</td><td>4.8</td><td>17.0</td><td>20.9</td></tr><tr><td>IDEA</td><td>5.6</td><td>6.6</td><td>5.8</td><td>24.0</td><td>42.0</td></tr></table>

{"class" 1 : "path.to.MyUnlearner", "parameters": {...}}

New link inference attack. A new attack subclasses GraphMeasure and implements a process(evaluation) method that computes a scalar score per candidate edge pair and records results via evaluation.add\_value. It is added to the evaluator.measures list exactly like the built-in attacks (UMIA, LinkTeller, Link Stealing), with no additional wiring required.

## F Complementary Results

Figure 5 expands Figure 3 of the main paper to all five benchmark datasets, showing the individual LinkTeller AUC and test accuracy for every unlearning method under the low- and high-centrality forget set settings.

The low-high centrality gap in LinkTeller AUC is consistent across all datasets and all methods: low-centrality forget sets (orange diamonds) yield higher AUC values, while high-centrality forget sets (blue circles) push most methods toward the random baseline at 0.5. This is a strong structural result: no method is able to protect low-centrality edges from LinkTeller inference, regardless of the specific unlearning strategy applied. The right panels confirm that test accuracy is nearly identical between the two settings on every dataset, reinforcing that accuracy cannot distinguish this failure mode.

## F.1 Complementary Architecture Trends

To complement the GCN and SGC-centric benchmark view in the main paper, Figure 6 show results of methods for GIN, GAT, and GRAPHSAGE.

First, the low-/high-centrality bifurcation in LinkTeller remains architecture-agnostic: LinkTeller is extremely high, while the high-centrality setting collapses to a much lower range for all three architectures. Second, utility sensitivity is architecture-dependent: GRAPHSAGE and GAT retain higher hard-setting accuracy overall, while GIN exhibits a visibly broader spread, reflecting stronger dependence on graph structure.

These results show that the main qualitative conclusions of Lethe persist beyond GCN to other GNN architectures.

Finally, Tables 16,17,18 show, respectively, the running times of all the methods on GCN, GIN, and SGC architectures. The tables show that, other than the certified methods, two non-certified methods fail to scale: UNSIR and SalUn.

![](images/409b63c1cc7df0ccde4d6b30149beebea95b7a17ff79c865634d5e921f460906.jpg)

![](images/e804bd2dbf3ee30999d3fafb6aa4a3eccec19de571f527e30069a0eb5658d1fb.jpg)

![](images/26912233f0b2fde8db6e54e82086ed298673287afce3b64a7b8d70ff13dd8e1c.jpg)

![](images/6da29f6e9d2e1705c5d2c4c69427b5f96ddb3c1ccf57211b8597e389db798d16.jpg)

![](images/1cba561b6223916553fab74bfebe785cb5670186b701b1a01e251a5c86121bcd.jpg)  
Figure 5: Per-method LinkTeller AUC (left) and test accuracy (right) for all five datasets under GCN, for low-centrality (orange diamonds) and high-centrality (blue circles) forget sets. The dashed line marks the random-chance AUC baseline of 0.5. GNNDelete is included for completeness but excluded from statistical analysis.

![](images/61f3a947391747575d23e0285421903e76dca2cc7deb5c495bc77e08b9e401e5.jpg)  
Figure 6: Aggregated complementary results for GIN, GAT, and GRAPHSAGE.

Table 17: RunTime comparison on GIN architecture.
<table><tr><td>Method</td><td>DBLP</td><td>Photos</td><td>Computers</td><td>Flickr</td><td>ogbn-arxiv</td></tr><tr><td>Gold Model – GIN</td><td>7.2</td><td>13.9</td><td>23.6</td><td>31.1</td><td>100</td></tr><tr><td>Finetuning</td><td>1.1</td><td>6.5</td><td>8.0</td><td>37.6</td><td>11.7</td></tr><tr><td>NegGrad</td><td>1.1</td><td>7.0</td><td>8.2</td><td>39.3</td><td>20.4</td></tr><tr><td>AdvNegGrad</td><td>1.5</td><td>6.6</td><td>9.0</td><td>39.0</td><td>23.9</td></tr><tr><td>SCRUB</td><td>2.3</td><td>6.6</td><td>9.7</td><td>46.6</td><td>20.6</td></tr><tr><td>BadTeaching</td><td>2.0</td><td>7.8</td><td>9.5</td><td>48.3</td><td>23.6</td></tr><tr><td>SSD</td><td>1.6</td><td>6.7</td><td>8.7</td><td>39.6</td><td>24.5</td></tr><tr><td>Fisher</td><td>1.1</td><td>9.6</td><td>11.0</td><td>72.7</td><td>325</td></tr><tr><td>UNSIR</td><td>8.9</td><td>11.7</td><td>23.3</td><td>139</td><td>433</td></tr><tr><td>SalUn</td><td>10.9</td><td>12.7</td><td>30.3</td><td>80.7</td><td>106</td></tr><tr><td>IDEA</td><td>5.6</td><td>6.0</td><td>24.0</td><td>25.2</td><td>42.0</td></tr></table>

Table 18: Runtime comparison on SGC architecture.
<table><tr><td>Method</td><td>DBLP</td><td>Photos</td><td>Computers</td><td>Flickr</td><td>ogbn-arxiv</td></tr><tr><td>Gold Model – SGC</td><td>18.4</td><td>19.7</td><td>35.5</td><td>56.3</td><td>38.5</td></tr><tr><td>Finetuning</td><td>2.1</td><td>3.9</td><td>11.8</td><td>17.9</td><td>11.5</td></tr><tr><td>NegGrad</td><td>2.2</td><td>4.4</td><td>9.6</td><td>19.0</td><td>11.8</td></tr><tr><td>AdvNegGrad</td><td>2.3</td><td>4.3</td><td>9.9</td><td>17.3</td><td>12.1</td></tr><tr><td>SCRUB</td><td>2.3</td><td>4.2</td><td>9.2</td><td>18.7</td><td>12.5</td></tr><tr><td>BadTeaching</td><td>2.5</td><td>4.5</td><td>9.8</td><td>18.6</td><td>15.0</td></tr><tr><td>SSD</td><td>2.2</td><td>4.3</td><td>10.0</td><td>18.1</td><td>12.2</td></tr><tr><td>Fisher</td><td>2.4</td><td>5.2</td><td>10.6</td><td>19.0</td><td>17.0</td></tr><tr><td>UNSIR</td><td>5.6</td><td>8.1</td><td>18.1</td><td>90.2</td><td>83.0</td></tr><tr><td>SalUn</td><td>4.0</td><td>7.1</td><td>16.4</td><td>34.3</td><td>51.0</td></tr><tr><td>IDEA</td><td>4.2</td><td>5.3</td><td>11.8</td><td>22.5</td><td>49.9</td></tr><tr><td>CEU</td><td>30.8</td><td>22.9</td><td>69.6</td><td>78.0</td><td>64.8</td></tr><tr><td>CGU</td><td>35.5</td><td>316.0</td><td>247.0</td><td>1576.0</td><td>59.3</td></tr><tr><td>ScaleGUN</td><td>51.8</td><td>3096.0</td><td>162.0</td><td>3008.0</td><td>71.5</td></tr></table>

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The main claims are made clear in the abstract and introduction and are followed through in later sections. All claims match experimental results.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We make our positioning with respect to other benchmarks clear in the paper, and we discuss limitations in the Conclusions.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [N/A]

Justification: This paper is a benchmark paper with empirical evaluations. No theoretical results are presented. However, we back empirical claims with large-scale evaluation and statistical testing.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The code, along with instructions to run it, is made publicly available and the link is given. We discuss our settings, both hardware and Software, in Section 5.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The code repository is made available and instructions are available in the repository.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: We detail all the necessary details in Section 5, including data splits, hyperparameters, and optimizer.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report statistical tests for all results in Section 5.1.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: All experiments were run on the same hardware that we describe in detail.   
Runtime is an explicit part of experimentation.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: We read the Ethics Guidelines and we made sure to comply in every respect.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The paper discusses positive societal impacts in supporting privacy regulations such as GDPR (Sections 1 and 5). The paper also discusses negative societal impacts and security considerations by studying link inference attacks, adversarial reconstruction of forgotten edges, which is the main motivation of the benchmark.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: We do not release new data nor models.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We cite and reference all the authors of the original methods used in the benchmark. Their methods were faithfully implemented.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: We release the benchmark code, which is open and well-documented.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: There was no crowdsourcing involved in this benchmark study.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: There was no crowdsourcing involved in this benchmark study.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: A dedicated section in the appendix specifies the usage of LLMs in the preparation of this work.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.