# A Lightweight Plastic-Memory Framework for Graph Few-Shot Class-Incremental Learning

Zihan Mei, Zhili Qin, Tongze Zhang, Hongyuan Liu, Junming Shao\*, Qinli Yang

University of Electronic Science and Technology of China

{zihanmei, zhangtongze, hongyuanliu, junmshao, qlyang}@uestc.edu.cn, qinzhili@outlook.com

## Abstract

Graph Incremental Learning has garnered increasing attention as dynamic graph data continues to emerge across diverse fields. Conventional approaches primarily address catastrophic forgetting by preserving node-related knowledge through replay or distillation techniques; however, they often incur high computational costs and inefficiency. This issue is further exacerbated in real-world scenarios where labeled data for new classes is scarce. In this paper, we propose a novel lightweight plastic-memory framework specifically designed for few-shot incremental learning on graphs. The core idea of our framework is the construction of a plastic-memory module that evolves over time, continuously updating and expanding its memory to accommodate new classes while retaining previously learned knowledge. In contrast to existing techniques, our memory module is both lightweight and effective, featuring an innovative evolving micro-clustering structure that dynamically updates representations of class prototypes, sub-prototypes, and their interaction weights. Building on this memory module, we introduce a memory-driven meta-learning framework that enhances adaptability to new tasks in its inner loop while maintaining stability for earlier tasks in the outer loop. Extensive experiments on four benchmark datasets demonstrate the framework’s superior performance in balancing stability for old knowledge and adaptability to new knowledge.

## Introduction

Graph Incremental Learning has emerged as a critical area of research due to the increasing prevalence of dynamic graph data in real-world applications, such as social networks, recommendation systems, and biological networks (Xia et al. 2021; Yuan and Zhao 2024). These domains often involve evolving structures and relationships, requiring models that can adapt to new information without forgetting prior knowledge. A key challenge is the scarcity of labeled data for new classes, making effective training difficult. Few-shot learning, which generalizes from limited labeled examples, becomes crucial in graph-based tasks where extensive annotations are costly or impractical. As a result, integrating few-shot learning with graph incremental learning—Graph Few-Shot Class-Incremental Learning (GFS-CIL)—has emerged as an essential research direction. GF-SCIL requires models to learn distinct node features with scarce labeled samples while retaining old knowledge and quickly adapting to new information. This scenario involves addressing the stability-plasticity dilemma, balancing the prevention of catastrophic forgetting with the efficient integration of new knowledge.

Despite significant progress in graph incremental learning, existing methods face critical limitations in addressing GFSCIL, necessitating more flexible and lightweight frameworks. Traditional methods mitigate forgetting by explicitly storing old-class samples or fixing feature space topology, which leads to high memory consumption (Snell, Swersky, and Zemel 2017; Zhou et al. 2022; Zhou and Cao 2021). Additionally, graph construction and updates are highly sensitive to few-shot data distributions, with limited labeled samples making the topology prone to noise (Kim et al. 2019; Tian et al. 2024; Zhou et al. 2022), and traditional knowledge distillation methods exacerbating forgetting due to extreme class imbalance (Dong et al. 2021; Tao et al. 2020). Furthermore, most frameworks lack adaptability to dynamic incremental scenarios, relying on fixed network structures or complex multi-stage training strategies that are ill-suited for evolving class streams (Kim et al. 2019; Tian et al. 2024), while meta-learning-based dynamic networks face challenges in balancing forgetting and computational costs (Chi et al. 2022). Finally, graph models often fail to balance stability and plasticity, with parameter updates leading to overfitting (Dong et al. 2021; Kim et al. 2019) or weight stagnation due to insufficient updates for few-shot tasks and excessive regularization (Kirkpatrick et al. 2017).

GFSCIL confronts three interconnected challenges. Catastrophic forgetting, inherent to incremental learning, is amplified in graphs due to their relational complexity, where small structural shifts can disrupt learned dependencies. Simultaneously, the plasticity-stability trade-off demands careful equilibrium: adapting to new classes without overwriting prior knowledge. Finally, label scarcity—a hallmark of few-shot learning—imposes severe constraints on feature generalizability, requiring models to infer robust node representations from minimal annotated examples. Interestingly, these issues resonate with longstanding problems in data stream clustering, where algorithms must dynamically update clusters under evolving data distributions with minimal supervision, which means both fields face similar challenges(Silva et al. 2013; Zubaroglu and Atalay 2021).˘

![](images/b320ec4c21f5db956968f3aa287381d931dc7816a9713ac884a322645360e52a.jpg)  
Figure 1: Hierarchical Memory Structure. Traditional methods rely on node-level data to retain knowledge, leading to inefficiency. Our method introduces a hierarchical memory structure with three stages: (1) Retention & Refinement: Learned nodes are clustered into micro-clusters to consolidate class representations. (2) Disentangled Micro-Memory: New nodes are grouped into semantic-specific micro-clusters, enabling memory evolution. (3) Lightweight Memory: Only statistical information is retained, achieving a compact and efficient memory design.

Inspired by these parallels, we introduce Lightweight Plastic-Memory with Micro-Clustering framework, termed LPMC, to tackle these challenges. At the heart of our framework lies a plastic-memory module that evolves dynamically over time, continuously updating and expanding its memory to incorporate new classes while preserving previously acquired knowledge, the concept of our method is shown in Figure 1. Unlike existing methods, our memory module employs an innovative evolving micro-clustering structure, which enables the dynamic representation of class prototypes, sub-prototypes, and their interaction weights in real time. Specifically, microclusters are formed by grouping nodes around multiple cluster centers that represent subprototypes, while these cluster centers are further organized around a class center, representing the class prototype. This hierarchical structure ensures efficient and adaptive memory management, making our framework both lightweight and effective for dynamic graph environments.

Building on this memory module, we draw inspiration from the Model-Agnostic Meta-Learning (MAML) framework (Finn, Abbeel, and Levine 2017) to propose a memorydriven meta-learning framework. Specifically, our approach integrates meta-learning in the inner loop to enhance adaptability to new tasks, while employing Graph Pseudo Incremental Learning in the outer loop to preserve stability for earlier tasks. This dual-loop design optimizes the model’s ability to adapt to new tasks while maintaining a balance between stability, plasticity, and training efficiency.

LPMC’s lightweight design is based on replacing raw data with prototypes, global structural updates with local adjustments, and retraining with meta-optimization. This is achieved through three key mechanisms. First, the Hierarchical Micro-Clustering Representation reduces memory overhead by maintaining a small set of dynamically adjusted prototypes and sub-prototypes, rather than complete samples or graph structures. Second, the Dynamic Evolution of Micro-Clusters enables the seamless integration of new knowledge by locally adjusting sub-prototypes and class prototypes during incremental phases, eliminating the need for global retraining or full graph reconstruction. Finally, the Dual-Loop Optimization mechanism enhances adaptability and stability: the inner loop uses meta-learning for rapid task adaptation, while the outer loop employs pseudo-incremental learning to stabilize old tasks through lightweight memory replay, avoiding redundant computations. Together, these mechanisms ensure an efficient and flexible framework for incremental learning.

In summary, our contributions can be outlined as follows:

• We propose a lightweight plastic-memory framework featuring evolving micro-clustering that dynamically organizes class prototypes and sub-prototypes through hierarchical clustering and maintains inter-class discrimination through adaptive interaction weights.

• We design a memory-driven dual-loop framework where the inner loop implements task-specific fast adaptation via gradient meta-updates in meta-learning, while the outer loop employs graph pseudo incremental learning to consolidate structural knowledge.

• Our proposed LPMC achieves new state-of-the-art performance and faster runtime on four major benchmark datasets under various GFSCIL scenarios.

## Related Work

In recent years, several methods specifically designed to address the challenges of GFSCIL have been proposed. (Tan et al. 2022) introduces a hierarchical attention framework to balance forgetting and accuracy, but struggles with class imbalances and limited generalization. Another method leverages memory-enhanced knowledge distillation(Li et al. 2024), showing improved performance but facing challenges with multiple training rounds and ineffective prototype updating when labeled data is scarce.

Although existing research specifically targeting GFS-CIL is still limited, valuable insights into some of its key challenges have been explored in the fields of fewshot learning and incremental learning. In graph few-shot learning, approaches can be broadly categorized into three groups: meta-learning-based, pre-training-based, and mixed methods (Yu et al. 2024). Meta-learning-based methods have been particularly influential, enabling models to adapt quickly to new tasks with limited data. These methods enhance the model’s ability to capture graph structural information through node-level, edge-level, and subgraph-level adaptations, while also improving rapid adaptation capabilities through graph-level and task-level optimizations. Notable examples include: Meta-GNN (Zhou et al. 2019), which integrates meta-learning with graph neural networks (GNNs) to create a generalizable framework independent of specific GNN architectures. G-Meta (Huang and Zitnik 2020), which represents nodes using local subgraphs and employs subgraph-based meta-learning. TENT (Wang et al. 2022), which adapts to new data distributions by minimizing the entropy of test-time predictions. TEG (Kim et al. 2023), which focuses on learning task-specific node embeddings. GPN (Ding et al. 2020), which learns class prototype representations for rapid adaptation in GFSL scenarios. In contrast, pre-training-based methods leverage large-scale pretrained models to achieve faster convergence and adaptation to specific tasks. For instance, GPPT (Sun et al. 2022) accelerates adaptation by transforming downstream tasks into a format similar to the pre-training task using graph prompt functions. Studies have demonstrated that combining pretrained knowledge with parameter fine-tuning yields strong performance on benchmark datasets (Yu et al. 2024).

Incremental learning is generally divided into instance, domain, and class incremental learning (CIL) (Luo et al. 2020), with CIL focusing on learning new categories over time while retaining knowledge of previous ones (Belouadah, Popescu, and Kanellos 2021; Masana et al. 2022; Zhou et al. 2024; Mittal, Galesso, and Brox 2021). Existing CIL methods can be broadly classified into three categories: model expansion, fixed representation, and finetuning (Belouadah, Popescu, and Kanellos 2021). Model expansion increases capacity to accommodate new knowledge; fixed representation preserves the backbone while updating the classifier; and fine-tuning modifies only the final layers. These methods include dynamic networks, which expand the model structure to adapt to data stream changes; data and parameter regularization, such as Topology-aware Weight Preserving (TWP) (Liu, Yang, and Wang 2021) and Elastic Weight Consolidation (EWC) (Kirkpatrick et al. 2017), which resist forgetting by regularizing parameters or data representations; knowledge distillation methods like Learning without Forgetting (LwF) (Li and Hoiem 2017), which minimize the discrepancy between old and new model outputs to retain previous knowledge; data replay techniques, including Gradient Episodic Memory (GEM) (Lopez-Paz and Ranzato 2017) and Experience Replay GNN (ER-GNN) (Zhou and Cao 2021), which store previous instances and adjust learning to prevent forgetting; and model correction methods, which reduce bias in the predictions of incremental learners. This categorization illustrates the diverse strategies aimed at addressing catastrophic forgetting in CIL, each focusing on different aspects of model adaptation. Similar challenges also exist in the field of datastream clustering, where existing methods excel in real-time adaptation to evolving data distributions (Zubaroglu and Atalay 2021;˘ Silva et al. 2013). For instance, methods like Chameleon (Xu et al. 2017) and DenStream (Cao et al. 2006) employ adaptive mechanisms to handle concept drift and irregular cluster shapes, while others, such as StreamSW (Reddy and Bindu 2019), SNCStream+ (Barddal et al. 2016), and MC-NN (Zhao et al. 2008), balance historical and recent data or enhance noise resilience. While robust and scalable, they often assume fully observable or static data, limiting their applicability to graph-structured few-shot learning.

## Problem Statement

Let $\mathcal { G } ~ = ~ ( \nu , \mathcal { E } , \mathbf { X } )$ represent a graph, where V denotes the set of nodes, E denotes the set of edges, and $\textbf { X } \in$ $\mathbb { R } ^ { | \nu | \times d }$ represents the node feature matrix. Alternatively, the graph can be expressed as $\begin{array} { r c l } { \mathcal { G } } & { = } & { \{ { \bf A } , { \bf X } \} } \end{array}$ , where A is the adjacency matrix capturing the connections between nodes. In the context of class incremental learning, we consider a progressive sequence of learning sessions $\boldsymbol { S } = \{ S _ { 0 } , S _ { 1 } , \ldots , S _ { T } \}$ with corresponding datasets $\{ \mathcal { D } ^ { 0 } , \mathcal { D } ^ { 1 } , . . . \dot { , } \tilde { \mathcal { D } } ^ { T } \}$ , where $\bar { \mathcal { D } } ^ { i } \ = \ \{ \bf A  _ { C ^ { i } } , { \bf \bar { X } } _ { C ^ { i } } \}$ . Here, $C ^ { i }$ represents the label space for session $i ,$ and the label spaces are disjoint across sessions, i.e., $C ^ { i } \cap C ^ { j } = \emptyset \mathrm { f o r } i \neq j$

The Few-shot Class-incremental Learning (FSCIL) scenario is defined as follows: For an N-way K-shot incremental node classification task, the first session $S _ { 0 }$ uses $\mathcal { D } ^ { 0 }$ as the base dataset, providing sufficient data for conventional semi-supervised or supervised node classification training. Subsequent sessions $\bar { S ^ { i } } \left( i \geq 1 \right)$ involve datasets $\mathcal { D } ^ { i } ( i \geq 1 )$ which contain few-shot datasets with N novel classes, each represented by K labeled nodes. The objective is to design a model capable of maintaining strong classification performance across both base and novel classes while adapting to the evolving label space through successive learning sessions.

## Methodology

## Pre-training Framework

In the pre-training phase, we adopt SimGRACE (Xia et al. 2022), a self-supervised contrastive learning method that leverages graph perturbation to learn structural and nodelevel representations. SimGRACE supports general GNN backbones such as GAT (Velickovi ˇ c et al. 2017), GCN (Kipf ´ and Welling 2016), and GraphSAGE (Zhang et al. 2019). We use a 2-layer GAT as the feature extractor, which applies multi-head self-attention to aggregate neighborhood features. The propagation rules are defined as follows:

$$
h _ { i } ^ { \prime } = \sigma \left( \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \sum _ { v _ { j } \in \mathcal { N } ( v _ { i } ) } \alpha _ { i j } ^ { k } W ^ { k } h _ { j } \right)\tag{1}
$$

$h _ { i } ^ { \prime }$ the updated node representation, $\sigma$ is the activation function, $\bar { \alpha _ { i j } ^ { k } }$ represents the attention coefficient between nodes $v _ { i }$ and $v _ { j }$ for head $k , W ^ { k }$ is the weight matrix, and $K$ is the number of attention heads, where each head considers the neighbors $v _ { j }$ of node $v _ { i }$ in the graph.

The primary goal of applying contrative learning is to strengthen the alignment between augmented views of the same graph while reducing the similarity between different graphs. This process is guided by a contrastive loss function, defined as:

$$
\mathcal { L } _ { \mathrm { c o n t r a s t i v e } } = - \sum _ { i = 1 } ^ { N } \log \frac { \exp ( \sin ( \mathbf { z } _ { i } , \mathbf { z } _ { i } ^ { + } ) / t ) } { \sum _ { j = 1 } ^ { N } \exp ( \sin ( \mathbf { z } _ { i } , \mathbf { z } _ { j } ) / t ) }\tag{2}
$$

Here, $\mathbf { Z } _ { i }$ represents the original graph-level embedding, and ${ \bf z } _ { i } ^ { + }$ is its Gaussian-perturbed counterpart, serving as the positive sample. The embeddings $\mathbf { Z } _ { j }$ correspond to other nodes in the batch, acting as negative samples. By constructing positive and negative sample pairs, the model’s robustness and generalization capabilities are significantly improved. The temperature parameter t adjusts the model’s sensitivity to differences between positive and negative pairs, while the similarity metric captures the closeness between embeddings, enabling the model to learn critical features during training.

![](images/2feecde260ab91097a09b72e255ced16a56db417f4526fa5911a1d7e50591dce.jpg)  
Figure 2: Overview of the LPMC framework for GFSCIL. (a)Pseudo Class Incremental Learning with Meta-Learning: Tasks sample base and N-way pseudo novel classes. Base classes remain fixed during inner-loop meta-training, while pseudo nove classes are integrated into the base set after each session.(b)Graph neural network: Comprises a GNN encoder and a prototypical network classifier with multi-sub-prototypes. (c)Inside Hierarchical Memory Structure, Memory Cluster Module: Constructs Mico-Clustering layer through DBSCAN and distance metric, Prototype Layer is constructed through MC layer. Memory Distillation Module: Interact with the GNN encoder to reduce knowledge forgetting by reducing the variation of class prototypes. (d)Knowledge Transfer: the total loss $\mathcal { L } _ { \mathrm { t o t a l } } = \alpha \mathcal { L } _ { \mathrm { c l s } } + \beta \mathcal { L } _ { \mathrm { d i s t i l } }$ (with learnable coefficients $\alpha , \beta )$ is back-propagated to the GNN encoder.

## Plasitic-Memory Construction and Memory-Driven Training Framework

As discussed, GFSCIL emphasizes incremental learning and mitigating forgetting. While pre-training offers basic decision-making ability, it falls short of addressing GFS-CIL’s core challenges. To this end, we propose the LPMC framework (Figure 2), which integrates a plastic-memory module and a memory-driven training scheme to improve model plasticity and stability.

Plastic-Memory Construction with Micro-Clustering The memory module M consists of a micro-cluster layer and a prototype layer:

$$
\mathcal { M } = ( \mathcal { M } . m c , \mathcal { M } . p r o t o t y p e s )\tag{3}
$$

The micro-cluster layer M.mc ensures efficient and stable prototype updates by summarizing new and old data using statistical representations rather than storing raw node features. It models each micro-cluster as a local embedding distribution within a class, with at least one cluster per class. The prototype layer M.prototypes captures class-level feature representations. Implementation details follow.

To track embedding distribution shifts, let $\{ \mathcal { C } _ { k } ^ { ( t ) } \} _ { k = 1 } ^ { K }$ be the micro-cluster set associated with one class at time $t ,$ which has K clusters, $K \in \mathbb { Z } ^ { + }$ . For a single micro-cluster,

$$
\mathcal { C } _ { k } \triangleq \left( \mathbf { c } _ { k } , \mathbf { S } _ { 1 } ^ { k } , \mathbf { S } _ { 2 } ^ { k } , n _ { k } , r _ { k } \right) \in \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } \times \mathbb { N } \times \mathbb { R } ^ { + }\tag{4}
$$

These attributes represent the centroid of the microcluster, the linear sum and square sum of node embeddings, the member count, and the radius of the cluster.

These attributes are the centroid $\begin{array} { r } { \mathbf { c } _ { k } = \frac { 1 } { n _ { k } } \mathbf { S } _ { 1 } ^ { k } } \end{array}$ , where $\mathbf { S } _ { 1 } ^ { k } =$ $\scriptstyle \sum _ { i = 1 } ^ { n _ { k } } .$ xˆ ,the squared sum $\begin{array} { r } { \mathbf { S } _ { 2 } ^ { k } = \sum _ { i = 1 } ^ { n _ { k } } \hat { \mathbf { x } } _ { i } ^ { \odot 2 } } \end{array}$ , and the adaptive radius

$$
r _ { k } = \lambda \cdot \frac { 1 } { d } \sum _ { j = 1 } ^ { d } \sqrt { \operatorname* { m a x } \left( \frac { ( \mathbf { S } _ { 2 } ^ { k } ) _ { j } } { n _ { k } } - \left( \frac { ( \mathbf { S } _ { 1 } ^ { k } ) _ { j } } { n _ { k } } \right) ^ { 2 } , \epsilon \right) }\tag{5}
$$

where d is the embedding dimension, $\epsilon > 0$ is the preset minimum radius to ensure numerical stability, and λ is a learnable scalar controlling the overall radius scale.

In the methodology for updating micro-clusters when new data arrives, the labeled data is determined whether belongs to an existing micro-cluster by measuring its Euclidean distance to the center of the micro-cluster relative to the radius of the micro-cluster.

For incoming sample embeddings $\mathcal { X } ^ { ( t ) } = \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { n }$ at time $t ,$ we assign them to existing micro-clusters using an adaptive radius criterion. The detailed update rules for assignment and statistical maintenance are summarized in Table 1.

Table 1: Micro-cluster update rules at time t
<table><tr><td>Step</td><td>Update Rule</td></tr><tr><td>Assignment</td><td> $\overline { { \mathcal { A } _ { k } ^ { ( t ) } } } = \left\{ \mathbf { x } _ { i } \in \mathcal { X } ^ { ( t ) } \mid \| \mathbf { x } _ { i } - \mathbf { c } _ { k } ^ { ( t - 1 ) } \| _ { 2 } \leq \rho r _ { k } ^ { ( t - 1 ) } \right\}$ </td></tr><tr><td>Count update</td><td> $n _ { k } ^ { ( t ) } = \dot { n _ { k } ^ { ( t - 1 ) } } + | \mathcal { A } _ { k } ^ { ( t ) } |$ </td></tr><tr><td>Linear sum</td><td> $\begin{array} { r } { \mathbf { S } _ { 1 } ^ { ( k , t ) } = \mathbf { \tilde { S } } _ { 1 } ^ { ( k , t - 1 ) } + \sum _ { \mathbf { x } \in \mathcal { A } _ { k } ^ { ( t ) } } \mathbf { x } } \end{array}$ </td></tr><tr><td>Square sum</td><td> $\begin{array} { r } { \mathbf { S } _ { 2 } ^ { ( k , t ) } = \mathbf { S } _ { 2 } ^ { ( k , t - 1 ) } + \sum _ { \mathbf { x } \in \mathcal { A } _ { k } ^ { ( t ) } } \mathbf { x } ^ { 2 } } \end{array}$ </td></tr></table>

The remaining samples constitute the residual set $\chi _ { \mathrm { r e s } } ^ { ( t ) } =$ $\textstyle \mathcal { X } ^ { ( t ) } \backslash \bigcup _ { k = 1 } ^ { K } \mathcal { A } _ { k } ^ { ( t ) }$ . We apply DBSCAN (Schubert et al. 2017) to these residuals to generate new micro-clusters:

$$
\mathcal { C } _ { \mathrm { n e w } } ^ { ( t ) } = \left\{ \mathcal { N } _ { \varepsilon } ^ { ( t ) } ( \mathbf { x } ) \mid | \mathcal { N } _ { \varepsilon } ^ { ( t ) } ( \mathbf { x } ) | \geq n _ { \mathrm { m i n } } , \mathbf { x } \in \mathcal { X } _ { \mathrm { r e s } } ^ { ( t ) } \right\} ,\tag{6}
$$

where $\mathcal { N } _ { \varepsilon } ( \mathbf { x } ) = \{ \mathbf { x } ^ { \prime } \in \mathcal { X } _ { \mathrm { r e s } } ^ { ( t ) } \mid \| \mathbf { x } ^ { \prime } - \mathbf { x } \| _ { 2 } \leq \varepsilon \}$ . Let $\mathbf { p } _ { k }$ denote the centroid of micro-cluster $\mathcal { C } _ { k }$ and $n _ { k }$ its cardinality. With $N _ { c }$ the current total samples of class $c ,$ the relative density is $\delta _ { k } = n _ { k } / N _ { c }$ , and the class prototype is computed as

$$
\mathbf { P } _ { c } = \sum _ { k } \delta _ { k } \cdot \mathbf { p } _ { k } .\tag{7}
$$

Memory Augmented class incremental learning We propose a dual-loop meta-learning framework comprising an outer loop with graph-based pseudo-class incremental learning (GPIL) and an inner loop for meta-training via new class simulation.

Outer Loop: GPIL with Prototype Distillation. The outer step implements Graph Pseudo Incremental Learning (GPIL). In GPIL, as incremental sessions proceed, the number of base classes grows while new classes shrink. To counter forgetting, we adopt a Memory Distillation Module with the following loss:

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = 1 - \frac { 1 } { C } \sum _ { i = 1 } ^ { C } \frac { \hat { \bf p } _ { \mathrm { p r e v } } ^ { i } \cdot \hat { \bf p } _ { \mathrm { c u r r } } ^ { i } } { T _ { \mathrm { d i s t i l } } }\tag{8}
$$

where $C$ is the number of selected base classes, $\hat { \mathbf { p } } _ { \mathrm { p r e v } } ^ { i } , \hat { \mathbf { p } } _ { \mathrm { c u r r } } ^ { i }$ are the normalized prototypes for the i-th class in the previous and current models, respectively, and $T _ { \mathrm { d i s t i l } }$ is the temperature parameter.

Inner Step: Prototypical Network with Meta-Learning. After each pseudo-incremental step, samples from the remaining new class labels are used for metaupdates. This reinforces base class knowledge and improves generalization. Model parameters are updated as

$$
\theta = \theta - \eta \nabla _ { \theta } \mathcal { L } _ { \mathrm { c l } } ( \theta ) - \gamma \eta \nabla _ { \theta } \mathcal { L } _ { \mathrm { d i s t i l } } ( \theta ) ,\tag{9}
$$

where η is the outer-loop learning rate and $\nabla _ { \theta } \mathcal { L } _ { \mathrm { c l } } , \nabla _ { \theta } \mathcal { L } _ { \mathrm { d i s t i l } }$ are the gradients of the classification and distillation losses, respectively.

Given a query embedding $f _ { \theta } ( \mathbf { x } _ { i } ) \in \mathbb { R } ^ { d }$ and the k-th subprototype $\mathbf { p } _ { k } ^ { ( c ) } \in \mathbb { R } ^ { d }$ of class $c ,$ their distance is defined as

$$
d \big ( f _ { \boldsymbol { \theta } } ( \mathbf { x } _ { i } ) , \mathbf { p } _ { k } ^ { ( c ) } \big ) = 1 - \frac { f _ { \boldsymbol { \theta } } ( \mathbf { x } _ { i } ) \cdot \mathbf { p } _ { k } ^ { ( c ) } } { \| f _ { \boldsymbol { \theta } } ( \mathbf { x } _ { i } ) \| \| \mathbf { p } _ { k } ^ { ( c ) } \| T _ { \mathrm { c l } } } ,\tag{10}
$$

where $T _ { \mathrm { c l } }$ is a temperature parameter. Each class c has $K _ { c }$ sub-prototypes with associated density weights $\{ \rho _ { k } ^ { ( c ) } \} _ { k = 1 } ^ { K _ { c } }$ For a batch of N query samples $\{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ with groundtruth labels $y _ { i }$ , the classification loss is

$$
\mathcal { L } _ { \mathrm { c l } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp \Bigl ( - \sum _ { k = 1 } ^ { K _ { y _ { i } } } \rho _ { k } ^ { ( y _ { i } ) } d \bigl ( f _ { \theta } ( \mathbf { x } _ { i } ) , \mathbf { p } _ { k } ^ { ( y _ { i } ) } \bigr ) \Bigr ) } { \sum _ { c = 1 } ^ { C } \exp \Bigl ( - \sum _ { k = 1 } ^ { K _ { c } } \rho _ { k } ^ { ( c ) } d \bigl ( f _ { \theta } ( \mathbf { x } _ { i } ) , \mathbf { p } _ { k } ^ { ( c ) } \bigr ) \Bigr ) } ,
$$

where $C$ is the total number of classes seen during training.

Table 2: Statistics of evaluation datasets.
<table><tr><td>Dataset</td><td># Nodes</td><td># Edges</td><td># Features</td><td>Class Split</td></tr><tr><td>Amazon Clothing</td><td>24,919</td><td>91,680</td><td>9,034</td><td>17/30/27</td></tr><tr><td>CoraFull</td><td>19,793</td><td>126,842</td><td>8,710</td><td>30/20/20</td></tr><tr><td>CS</td><td>18,333</td><td>163,788</td><td>6805</td><td>0/5/10</td></tr><tr><td>Computers</td><td>13,752</td><td>491,722</td><td>767</td><td>0/5/5</td></tr></table>

## Experiments

## Experimental Setup

Datasets. Our evaluation utilizes four widely-used realworld datasets: Amazon Clothing, CoraFull, CoauthorCS, and Computers. Table 2 provides the statistics and partitions of the datasets. Class split refers to the division of dataset categories based on our training framework into base classes, novel train classes, and novel test classes . Both base classes and novel train classes are accessible during the training phase, while novel test classes are only introduced during the testing phase.

Baselines. In evaluating our methodology, we benchmark against nine significant models to comprehensively demonstrate the effectiveness of our approach. These include three state-of-the-art methods specifically designed for GFSCIL: HAG-Meta (Tan et al. 2022), Geometer (Lu et al. 2022), and Mecoin (Li et al. 2024). Additionally, we compare against six foundational learning frameworks tailored for graph class-incremental learning scenarios: Elastic Weight Consolidation (EWC) (Kirkpatrick et al. 2017), Learning without Forgetting (LwF) (Li and Hoiem 2017), Topology-aware Weight Preserving (TWP) (Liu, Yang, and Wang 2021), Gradient Episodic Memory (GEM) (Lopez-Paz and Ranzato 2017), Memory Aware Synapses (MAS) (Aljundi et al. 2018), and Experience Replay GNN (ER-GNN) (Zhou and Cao 2021). These comparisons aim to highlight our model’s advancements in mitigating knowledge forgetting, improving accuracy, and enhancing generalization in the GFCIL setting.

Table 3: Main experiment results on the Amazon clothing, CoraFull, CS and Computers datasets under different N-way K-shot settings. Detailed results are provided in supplementary materials.
<table><tr><td></td><td colspan="5">Amazon Clothing dataset (3-way 5-shot)</td><td colspan="5">CoraFull dataset (2-way 5-shot)</td></tr><tr><td rowspan="2">Method</td><td colspan="2">Acc. in sessions (%) ↑</td><td></td><td rowspan="2">PD↓</td><td rowspan="2">Average</td><td colspan="3">Acc. in sessions (%) ↑</td><td rowspan="2">PD↓</td><td rowspan="2">Average ACC↑</td></tr><tr><td>0</td><td>5</td><td>9</td><td>ACC↑</td><td>0 5</td><td>10</td></tr><tr><td>ERGNN</td><td>62.40</td><td>29.15</td><td>29.48</td><td>32.92</td><td>34.87</td><td>73.43</td><td>21.84</td><td>11.30</td><td>62.13</td><td>29.55</td></tr><tr><td>GEM</td><td>77.54</td><td>27.88</td><td>28.82</td><td>48.72</td><td>37.11</td><td>69.06</td><td>14.78</td><td>7.52</td><td>61.54</td><td>22.65</td></tr><tr><td>MAS</td><td>68.70</td><td>27.74</td><td>28.91</td><td>39.79</td><td>35.17</td><td>69.06</td><td>47.04</td><td>46.39</td><td>22.67</td><td>49.99</td></tr><tr><td>LWF</td><td>51.97</td><td>18.00</td><td>28.75</td><td>23.22</td><td>27.80</td><td>73.60</td><td>13.91</td><td>7.46</td><td>66.14</td><td>24.02</td></tr><tr><td>EWC</td><td>78.26</td><td>29.93</td><td>31.92</td><td>46.34</td><td>39.84</td><td>69.06</td><td>18.59</td><td>12.55</td><td>56.51</td><td>27.14</td></tr><tr><td>TWP</td><td>65.83</td><td>25.85</td><td>26.86</td><td>38.97</td><td>32.48</td><td>69.06</td><td>23.29</td><td>13.77</td><td>55.29</td><td>27.84</td></tr><tr><td>Geometer</td><td>76.28</td><td>30.31</td><td>19.91</td><td>56.37</td><td>36.82</td><td>72.23</td><td>32.79</td><td>16.32</td><td>55.91</td><td>35.52</td></tr><tr><td>HAG-Meta</td><td>84.15</td><td>61.42</td><td>50.79</td><td>33.36</td><td>63.79</td><td>87.62</td><td>63.38</td><td>51.47</td><td>36.15</td><td>66.57</td></tr><tr><td>Mecoin</td><td>77.78</td><td>64.60</td><td>56.18</td><td>21.60</td><td>66.60</td><td>75.53</td><td>64.97</td><td>60.10</td><td>15.43</td><td>66.22</td></tr><tr><td>Ours</td><td>81.20</td><td>70.92</td><td>69.09</td><td>12.11</td><td>73.86</td><td>72.12</td><td>67.59</td><td>65.00</td><td>7.12</td><td>66.79</td></tr><tr><td rowspan="2">Method</td><td colspan="4">CS dataset (1-way 5-shot)</td><td></td><td colspan="4">Computers dataset (1-way 5-shot)</td><td></td></tr><tr><td colspan="4">Acc. in sessions (%) ↑</td><td>Average</td><td colspan="4">Acc. in sessions (%) ↑</td><td>Average</td></tr><tr><td>ERGNN</td><td>0</td><td>5</td><td>10</td><td>PD↓ 69.88</td><td>ACC↑</td><td>0</td><td>2</td><td>5</td><td>PD↓</td><td>ACC↑</td></tr><tr><td></td><td>100.00</td><td>36.93</td><td>30.12</td><td></td><td>38.41</td><td>100.00</td><td>33.33</td><td>16.67</td><td>83.33</td><td>40.83</td></tr><tr><td>GEM</td><td>100.00</td><td>17.03</td><td>18.09</td><td>81.91</td><td>30.30</td><td>100.00</td><td>33.33</td><td>16.67</td><td>83.33</td><td>40.83</td></tr><tr><td>MAS</td><td>100.00</td><td>59.54</td><td>63.92</td><td>36.08</td><td>60.68</td><td>100.00</td><td>33.57</td><td>21.64</td><td>78.36</td><td>44.75</td></tr><tr><td>LWF</td><td>100.00</td><td>36.40</td><td>32.51</td><td>67.49</td><td>38.30</td><td>100.00</td><td>33.33</td><td>16.84</td><td>83.16</td><td>40.86</td></tr><tr><td>EWC</td><td>100.00</td><td>36.62</td><td>38.63</td><td>61.37</td><td>38.74</td><td>100.00</td><td>33.33</td><td>16.67</td><td>83.33</td><td>40.83</td></tr><tr><td>TWP</td><td>100.00</td><td>47.14</td><td>52.02</td><td>47.98</td><td>44.44</td><td>100.00</td><td>33.33</td><td>16.67</td><td>83.33</td><td>40.83</td></tr><tr><td>Geometer</td><td>60.60</td><td>28.86</td><td>29.63</td><td>30.97</td><td>28.11</td><td>59.40</td><td>23.57</td><td>13.20</td><td>46.20</td><td>27.19</td></tr><tr><td>HAG-Meta</td><td>20.00</td><td>10.00</td><td>6.66</td><td>13.34</td><td>11.24</td><td>20.00</td><td>14.28</td><td>10.00</td><td>10.00</td><td>14.09</td></tr><tr><td>Mecoin</td><td>97.83</td><td>77.88</td><td>62.21</td><td>35.62</td><td>77.50</td><td>91.44</td><td>54.94</td><td>67.66</td><td>23.78</td><td>74.64</td></tr><tr><td>Ours</td><td>98.01</td><td>81.95</td><td>73.33</td><td>24.68</td><td>84.06</td><td>93.60</td><td>91.43</td><td>81.00</td><td>12.60</td><td>88.53</td></tr></table>

Table 4: Training Epochs and Running Time of Our Method and SOTA (Mecoin)
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Epochs</td><td rowspan=1 colspan=1>PD</td><td rowspan=1 colspan=1>Average Acc</td><td rowspan=1 colspan=1>Time</td></tr><tr><td rowspan=1 colspan=1>Amazonclothing</td><td rowspan=1 colspan=1>MecoinOurs</td><td rowspan=1 colspan=1>200010</td><td rowspan=1 colspan=1>21.6011.01</td><td rowspan=1 colspan=1>66.6070.02</td><td rowspan=1 colspan=1>2108s78.3s</td></tr><tr><td rowspan=1 colspan=1>CoraFull</td><td rowspan=1 colspan=1>MecoinOurs</td><td rowspan=1 colspan=1>200010</td><td rowspan=1 colspan=1>15.4310.57</td><td rowspan=1 colspan=1>66.2265.48</td><td rowspan=1 colspan=1>OOM105.8s</td></tr><tr><td rowspan=1 colspan=1>computers</td><td rowspan=1 colspan=1>MecoinOurs</td><td rowspan=1 colspan=1>200010</td><td rowspan=1 colspan=1>23.7813.59</td><td rowspan=1 colspan=1>74.6485.73</td><td rowspan=1 colspan=1>1152s7.48s</td></tr><tr><td rowspan=1 colspan=1>CS</td><td rowspan=1 colspan=1>MecoinOurs</td><td rowspan=1 colspan=1>200010</td><td rowspan=1 colspan=1>35.6221.60</td><td rowspan=1 colspan=1>77.5083.22</td><td rowspan=1 colspan=1>1336s12.7s</td></tr></table>

## Main Results

The comparative results for few-shot node classification across various datasets and settings are summarized in the Table 3. From these results, we draw several key observations:

Superior Performance of LPMC: The LPMC framework consistently achieves state-of-the-art performance across all four datasets, demonstrating its effectiveness in mitigating knowledge forgetting and maintaining high accuracy in GFSCIL tasks. Specifically, LPMC successfully balances Performance Drop (PD) and average accuracy compared to other baselines.

Consistency Across Diverse Settings: LPMC demonstrates consistently superior performance across all four datasets, each with distinct N-way K-shot configurations. Whether handling multi-class tasks like Amazon Clothing (3-way) and CoraFull (2-way) or single-class tasks like CS and Computers (1-way), LPMC excels in both knowledge retention and task adaptation. This consistency underscores the robustness of LPMC’s design, ensuring reliable performance across diverse graph-based learning scenarios. Additional experimental results under different settings provided in supplementary materials also demonstrate LPMC’s performance in more resource-constrained scenarios.

Comparison with Other Baselines: While some existing models, such as Mecoin and HAG-Meta, achieve higher accuracy in initial sessions on certain datasets, their high forgetting rates significantly degrade their long-term performance. In contrast, LPMC maintains low PD values and the highest average accuracy across all sessions, outperforming these models in subsequent tasks. On the CoraFull dataset, Mecoin starts with comparable accuracy but suffers from higher PD and lower average accuracy than LPMC.

Efficiency and Practical Implications: Beyond superior accuracy and consistency, LPMC demonstrates remarkable efficiency compared to baseline models. We mainly compare our method with Mecoin as it is the SOTA efficient method specially designed for GFSCIL. As explicitly quantified in Table 4, our framework requires fewer training rounds and running time to achieve lower forgetting rates and higher average accuracy, making it highly suitable for practical applications with limited training resources. Even with minimal training, LPMC outperforms baselines well before reaching peak performance. Theoretical analysis of time complexity is provided in supplementary materials.

Table 5: Performance degradation (PD) and Average Accuracy under different ablation settings.
<table><tr><td>Pretrain</td><td>micro-cluster</td><td>meta train</td><td>Amazon_clothing</td><td>CoraFull</td><td>computers</td><td>CS</td></tr><tr><td>√</td><td>√</td><td>√</td><td>12.11/73.86</td><td>7.12/66.79</td><td>12.60/88.53</td><td>24.68/84.06</td></tr><tr><td>w/o</td><td></td><td></td><td>(+3.68/-2.02)</td><td>(+0.77/-1.33)</td><td>(+11.05/-12.01)</td><td>(+1.12/-0.94)</td></tr><tr><td></td><td>w/o</td><td></td><td>(+5.51/-1.41)</td><td>(+3.96/-0.34)</td><td>(-0.92/-2.73)</td><td>(+2.99/-6.94)</td></tr><tr><td></td><td></td><td>w/o</td><td>(+0.39/-10.01)</td><td>(+0.55/-10.35)</td><td>(+7.38/-6.22)</td><td>(+5.96/-4.76)</td></tr></table>

Table 6: Analysis of Backbone
<table><tr><td colspan="10">Amazon Clothing dataset (3-way 5-shot)</td><td colspan="3">Average</td></tr><tr><td rowspan="2">Method</td><td></td><td colspan="9">Acc. in each session (%) ↑</td><td rowspan="2"></td><td rowspan="2">PD↓</td></tr><tr><td>0</td><td></td><td>1</td><td>2</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td>GAT</td><td></td><td>81.20 80.94 78.93 77.97 75.81 70.92 68.24 68.17 67.30 69.09 12.11</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>ACC↑ 73.86</td></tr><tr><td>GCN</td><td></td><td></td><td></td><td></td><td>82.2381.7079.8276.7875.6571.2366.7668.8767.1665.7116.52</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>73.59</td></tr><tr><td>GraphSAGE 82.4880.9481.43 75.9376.45 73.85 71.32 70.42 67.97 66.4915.99</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>74.73</td></tr></table>

CoraFull dataset (2-way 5-shot)
<table><tr><td rowspan="2">Backbone</td><td colspan="10">Acc. in each session (%) ↑</td><td rowspan="2"></td><td rowspan="2">PD↓</td><td rowspan="2">Average ACC↑</td></tr><tr><td>0</td><td>1</td><td></td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>GAT</td><td></td><td></td><td></td><td></td><td></td><td>72.12 70.20 70.96 67.59 66.96 67.59 63.50 62.58 65.63 62.58 65.00 7.12</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>66.79</td></tr><tr><td>GCN</td><td></td><td></td><td></td><td></td><td></td><td>72.29 68.20 72.50 68.89 67.86 64.66 66.67 64.35 63.59 64.24 64.79 7.50</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>67.09</td></tr><tr><td>GraphSAGE 72.79 70.00 70.00 71.48 67.86 68.28 64.00 67.42 65.17 62.88 64.71 8.08</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>67.69</td></tr></table>

## Ablation Study

We conducted ablation studies on four datasets to evaluate the impact of pre-training, meta-training, and the microcluster structure on performance and forgetting. As shown in Table 5, the micro-clustering mechanism in LPMC yields well-separated classes and compact intra-class distributions, indicating more discriminative prototype learning. On the simpler Computers dataset, strong performance is achieved even without micro-clustering, suggesting that basic prototypes suffice. However, LPMC shows clear advantages for more complex datasets, highlighting its strength in handling intricate graph structures. Visualizations of ablation study results are provided in supplementaty materials.

## Backbone Analysis

In our study, we evaluated the impact of three different backbones—GCN, GAT, and GraphSAGE —on the performance of our model across two datasets: Amazon Clothing (3-way 5-shot) and CoraFull (2-way 5-shot).The experimental results shown in Table 6 reveal that the performance of the training framework is not significantly influenced by the choice of backbone, as all three architectures yield similar results in terms of forgetting rate and accuracy. These findings highlight the robustness and generalizability of our training framework, which performs effectively across different backbone architectures.

## Parameters Analysis

We assessed model robustness with respect to two key hyper-parameters—(1) the number of inner-update steps in meta-training and (2) DBSCAN’s minPts—on the Amazon Clothing benchmark under the 3-way 5-shot setting.

The inner-step count chiefly governs both initial accuracy and the final average. Adding steps, especially from very small values, yields clear gains; once the budget exceeds

![](images/51a01b45e2855acad224aa634d6c862a4e444b0b76ff3fbf01ef9f6166cc9000.jpg)  
Figure 3: Impact of Inner step and MinPts on Accuracy.

10–20 steps, however, forgetting rises slightly—likely because the outer loop is not trained long enough. Meanwhile, MinPts exerts almost no influence on either forgetting or average accuracy.

## Conclusion

In this paper, we tackle the challenges of few-shot class incremental learning on dynamic graphs by proposing a lightweight plastic-memory framework. Our novel plasticmemory module dynamically integrates new class knowledge while preserving prior knowledge, addressing the inefficiencies and high computational costs of existing methods. The micro-clustering structure enhances class node feature characterization, ensuring both stability and adaptability in evolving graph data. Additionally, our memory-driven meta-learning framework with a dual-loop architecture improves task adaptation while maintaining performance on previously learned tasks. Extensive experiments on benchmark datasets demonstrate the framework’s ability to balance stability and adaptability, with generalization error analysis confirming its robustness across different feature extractors.This work advances graph incremental learning in resource-constrained and data-scarce environments, with future research focusing on domain extension and efficiency optimization for large-scale dynamic graphs.

## References

Aljundi, R.; Babiloni, F.; Elhoseiny, M.; Rohrbach, M.; and Tuytelaars, T. 2018. Memory aware synapses: Learning what (not) to forget. In Proceedings ofthe European conference on computer vision (ECCV), 139–154.

Barddal, J. P.; Gomes, H. M.; Enembreck, F.; and Barthes,\` J.-P. 2016. SNCStream+: Extending a high quality true anytime data stream clustering algorithm. Information Systems, 62: 60–73.

Belouadah, E.; Popescu, A.; and Kanellos, I. 2021. A comprehensive study of class incremental learning algorithms for visual tasks. Neural Networks, 135: 38–54.

Cao, F.; Estert, M.; Qian, W.; and Zhou, A. 2006. Densitybased clustering over an evolving data stream with noise. In Proceedings of the 2006 SIAM international conference on data mining, 328–339. SIAM.

Chi, Z.; Gu, L.; Liu, H.; Wang, Y.; Yu, Y.; and Tang, J. 2022. Metafscil: A meta-learning approach for few-shot class incremental learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 14166– 14175.

Ding, K.; Wang, J.; Li, J.; Shu, K.; Liu, C.; and Liu, H. 2020. Graph prototypical networks for few-shot learning on attributed networks. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, 295–304.

Dong, S.; Hong, X.; Tao, X.; Chang, X.; Wei, X.; and Gong, Y. 2021. Few-shot class-incremental learning via relation knowledge distillation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, 1255–1263.

Finn, C.; Abbeel, P.; and Levine, S. 2017. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning, 1126–1135. PMLR.

Huang, K.; and Zitnik, M. 2020. Graph meta learning via local subgraphs. Advances in neural information processing systems, 33: 5862–5874.

Kim, J.; Kim, T.; Kim, S.; and Yoo, C. D. 2019. Edgelabeling graph neural network for few-shot learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 11–20.

Kim, S.; Lee, J.; Lee, N.; Kim, W.; Choi, S.; and Park, C. 2023. Task-equivariant graph few-shot learning. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 1120–1131.

Kipf, T. N.; and Welling, M. 2016. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.

Kirkpatrick, J.; Pascanu, R.; Rabinowitz, N.; Veness, J.; Desjardins, G.; Rusu, A. A.; Milan, K.; Quan, J.; Ramalho, T.; Grabska-Barwinska, A.; et al. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy ofsciences, 114(13): 3521–3526.

Li, D.; Zhang, A.; Gao, J.; and Qi, B. 2024. An Efficient Memory Module for Graph Few-Shot Class-Incremental Learning. arXiv preprint arXiv:2411.06659.

Li, Z.; and Hoiem, D. 2017. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12): 2935–2947.

Liu, H.; Yang, Y.; and Wang, X. 2021. Overcoming catastrophic forgetting in graph neural networks. In Proceedings ofthe AAAI conference on artificial intelligence, 8653– 8661.

Lopez-Paz, D.; and Ranzato, M. 2017. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30.

Lu, B.; Gan, X.; Yang, L.; Zhang, W.; Fu, L.; and Wang, X. 2022. Geometer: Graph few-shot class-incremental learning via prototype representation. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining, 1152–1161.

Luo, Y.; Yin, L.; Bai, W.; and Mao, K. 2020. An appraisal of incremental learning methods. Entropy, 22(11): 1190.

Masana, M.; Liu, X.; Twardowski, B.; Menta, M.; Bagdanov, A. D.; and Van De Weijer, J. 2022. Class-incremental learning: survey and performance evaluation on image classification. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5): 5513–5533.

Mittal, S.; Galesso, S.; and Brox, T. 2021. Essentials for class incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3513–3522.

Reddy, K. S. S.; and Bindu, C. S. 2019. StreamSW: A density-based approach for clustering data streams over sliding windows. Measurement, 144: 14–19.

Schubert, E.; Sander, J.; Ester, M.; Kriegel, H. P.; and Xu, X. 2017. DBSCAN revisited, revisited: why and how you should (still) use DBSCAN. ACM Transactions on Database Systems (TODS), 42(3): 1–21.

Silva, J. A.; Faria, E. R.; Barros, R. C.; Hruschka, E. R.; Carvalho, A. C. d.; and Gama, J. 2013. Data stream clustering: A survey. ACM Computing Surveys (CSUR), 46(1): 1–31.

Snell, J.; Swersky, K.; and Zemel, R. 2017. Prototypical networks for few-shot learning. Advances in neural information processing systems, 30.

Sun, M.; Zhou, K.; He, X.; Wang, Y.; and Wang, X. 2022. Gppt: Graph pre-training and prompt tuning to generalize graph neural networks. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 1717–1727.

Tan, Z.; Ding, K.; Guo, R.; and Liu, H. 2022. Graph few-shot class-incremental learning. In Proceedings of the fifteenth ACM international conference on web search and data mining, 987–996.

Tao, X.; Hong, X.; Chang, X.; Dong, S.; Wei, X.; and Gong, Y. 2020. Few-shot class-incremental learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12183–12192.

Tian, S.; Li, L.; Li, W.; Ran, H.; Ning, X.; and Tiwari, P. 2024. A survey on few-shot class-incremental learning. Neural Networks, 169: 307–324.

Velickoviˇ c, P.; Cucurull, G.; Casanova, A.; Romero, A.; Lio,´ P.; and Bengio, Y. 2017. Graph attention networks. arXiv preprint arXiv:1710.10903.

Wang, S.; Ding, K.; Zhang, C.; Chen, C.; and Li, J. 2022. Task-adaptive few-shot node classification. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 1910–1919.

Xia, F.; Sun, K.; Yu, S.; Aziz, A.; Wan, L.; Pan, S.; and Liu, H. 2021. Graph Learning: A Survey. IEEE Transactions on Artificial Intelligence, 2(2): 109–127.

Xia, J.; Wu, L.; Chen, J.; Hu, B.; and Li, S. Z. 2022. Simgrace: A simple framework for graph contrastive learning without data augmentation. In Proceedings of the ACM Web Conference 2022, 1070–1079.

Xu, J.; Li, F.; Chen, K.; Zhou, F.; Choi, J.; and Shin, J. 2017. Dynamic chameleon authentication tree for verifiable data streaming in 5G networks. IEEE Access, 5: 26448–26459.

Yu, X.; Fang, Y.; Liu, Z.; Wu, Y.; Wen, Z.; Bo, J.; Zhang, X.; and Hoi, S. C. 2024. Few-shot learning on graphs: from meta-learning to pre-training and prompting. arXiv preprint arXiv:2402.01440.

Yuan, B.; and Zhao, D. 2024. A survey on continual semantic segmentation: Theory, challenge, method and application. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Zhang, S.; Tong, H.; Xu, J.; and Maciejewski, R. 2019. Graph convolutional networks: a comprehensive review. Computational Social Networks, 6(1): 1–23.

Zhao, J.-j.; Huang, X.-h.; Qiong, S.; and Yan, M. 2008. Realtime feature selection in traffic classification. The Journal of China Universities of Posts and Telecommunications, 15: 68–72.

Zhou, D.-W.; Wang, Q.-W.; Qi, Z.-H.; Ye, H.-J.; Zhan, D.- C.; and Liu, Z. 2024. Class-incremental learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Zhou, D.-W.; Ye, H.-J.; Ma, L.; Xie, D.; Pu, S.; and Zhan, D.-C. 2022. Few-shot class-incremental learning by sampling multi-phase tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(11): 12816–12831.

Zhou, F.; and Cao, C. 2021. Overcoming catastrophic forgetting in graph neural networks with experience replay. In Proceedings of the AAAI Conference on Artificial Intelligence, 4714–4722.

Zhou, F.; Cao, C.; Zhang, K.; Trajcevski, G.; Zhong, T.; and Geng, J. 2019. Meta-gnn: On few-shot node classification in graph meta-learning. In Proceedings ofthe 28th ACM International Conference on Information and Knowledge Management, 2357–2360.

Zubaroglu, A.; and Atalay, V. 2021. Data stream clustering:˘ a review. Artificial Intelligence Review, 54(2): 1201–1236.