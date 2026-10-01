# About the Influence of Workflow Topology on Task Intensity Prediction through Graph Learning

Max Otto<sup>1</sup>, Haci Ismail Aslan<sup>1</sup>, Joel Witzke<sup>1</sup>, Jonathan Bader<sup>1</sup>, and Odej Kao<sup>1</sup>

<sup>1</sup> Dept. Electrical Engineering and Computer Science, Technical University of Berlin, Berlin, Germany

Abstract—Efficient resource provisioning for large-scale workflows on cloud infrastructures is a critical performance engineering challenge. These workflows are often structured as directed acyclic graphs (DAGs), where under-provisioning can cause critical bottlenecks and over-provisioning leads to unnecessary costs. Accurate, task-level prediction of resource intensity (e.g., CPU load and memory usage) is essential for mitigating these issues. While task-level features are commonly used for prediction, the performance impact of the workflow’s overall topological structure is often overlooked or assumed. The central question of our work is: To what extent does what part of the DAG topology influence task-level resource intensity, and what is the most effective way to model this influence?

This paper presents a comprehensive benchmark to systematically quantify the impact of graph topology on task intensity prediction. We evaluate and compare a spectrum of modeling approaches. Our findings demonstrate that topology is a critical feature for accurate prediction. Models incorporating important topological information, even through simple handcrafted features, significantly outperform baseline models. We show that graph-native models provide the highest accuracy, achieving low mean absolute errors for both CPU and memory predictions, and can still be combined with simple topological features that they do not learn for better performance.

Index Terms—resource utilization prediction, workflow performance modeling, graph deep learning, transfer learning, cloud computing

## I. INTRODUCTION

Cloud computing has become the foundation for a wide range of digital services, offering elastic and on-demand access to computational resources [7]. It supports applications in domains such as web search, social networking, e-business, and industrial automation [7], [36]. To meet diverse and dynamic demands, modern cloud platforms employ automated resource provisioning and scheduling strategies.

A common abstraction for representing computational processes in the cloud is the workflow, typically modeled as a directed acyclic graph (DAG). In such graphs, nodes represent individual tasks, while directed edges encode dependencies between tasks [13], [21], as depicted in Figure 1. These dependencies define execution order and often reflect underlying data or control flows in real-world applications.

Each task in a workflow consumes computing resources such as CPU or memory. Accurately estimating these requirements is crucial, especially in shared or pay-per-use environments. Over-provisioning leads to underutilized infrastructure and increased cost [24], while under-provisioning risks service degradation, bottlenecks, or even task failures [4], [11], [23]. Furthermore, resource demands can exhibit variability and burstiness over time [3], making it insufficient to rely solely on historical averages. For effective resource scheduling, systems must account not only for expected usage but also for peaks and anomalies in task intensity.

![](images/50ab8fd9889250207547e223537953eabb88c55f4beb8bfd627f26526a5aa01a.jpg)  
Fig. 1: Example workflow DAG with CPU/memory intensities.

In recent years, predictive models have been developed to anticipate task-level resource needs prior to execution. Among these, graph-based deep learning methods, particularly Graph Neural Networks (GNNs), have shown promise in capturing both task features and the topological structure of workflows [36], [16]. These models exploit the connectivity and hierarchical relationships in workflows to improve predictions of resource usage. However, many existing approaches either target specific model architectures or lack reusability across different scenarios. Furthermore, the relationship between workflow topology and task intensity remains underexplored.

This work systematically studies how the structure of a workflow graph relates to the intensity of its tasks. We approach this in multiple steps. First, we train an MLP to predict task intensities on measured task-level features, with and without additional handcrafted topological features, and compare the accuracies.

Furthermore, we apply various graph learning models for the same task resource predictions, with and without the same topological features as in the MLP experiment, to compare the graph learning models to the MLP and each other, as well as explore the influence of topological features on the models accuracies.

Finally, we test the real-world applicability of such workflow structure-based approaches. Using the penultimate layer embeddings of a graph learning model that we train on small workflow graphs, in and out of concatenation with topological features, linked with a regressor model, we predict task intensities of larger workflows to explore how such embeddings generalize as well as regression errors.

## II. RELATED WORK

Prediction of resource consumption is far from a new problem and has already been extensively studied from multiple angles in existing literature.

A fine-grained approach lies in the time series forecasting of resource consumption. Clustering and adaptive networkbased fuzzy inference systems (ANFIS) [6], or an ensemble model using multiple predictor sets with dynamically adjustable members [8], yielded promising results for CPU load time series prediction. Deep learning based on diffusion convolutional recurrent neural networks (DCRNN) was also successfully used for CPU load predictions between 5 and 60 minutes into the future [2].

Within the context of scientific workflows that require highperformance computing, the runtime, disk space, and memory consumption of tasks were estimated using their parameter and input data size [12]. To account for estimation errors the authors propose an online estimation process based on the MAPE-K loop (Monitoring, Analysis, Planning, Execution, and Knowledge). Another approach to predict the task intensity (here, a combination of CPU and memory consumption) has been conducted with reinforcement learning in the form of gradient bandits and evaluated on real-world workflows [4]. Memory predictions of tasks also enable adjustable resource scheduling models for scientific workflows, which can either focus on high throughput or minimization of resource usage [32]. The models are based on the slow-peaks model and thus assume resource exhaustion towards the end of a task’s execution time.

Previous work also investigated the use of graph features and GNNs. One implements a scheduling framework that combines graph neural networks and deep reinforcement learning (DRL) using a reinforcement learning agent that is trained through the proximal policy optimization (PPO) algorithm to optimize makespan and energy consumption in dynamic cloud workflow environments [9]. A different approach using various graph neural network models on the offline cloud workflow database Alibaba Cluster-Trace-V2018 concluded that homogeneous graphs are a better base for GNNs, than heterogeneous graphs [16]. Furthermore, on the dataset, graph attention networks (GAT) are slightly inferior to graph convolutional networks (GCN), and GNNs predict task performance more accurately for tasks located later in the workflow.

The same dataset was also used for task intensity prediction with a transformer model that encodes the DAG information in a position embedding block [36]. Experiments showed that the transformer model received better results with graph structure information than without. Also, two graph topology-based models are among the best three evaluated machine learning models of the paper.

The last two approaches, which include graph learning for task resource estimation, indicate a connection between task resource intensity and topological graph or task information, which is explored, in detail and in connection to particular structural components, throughout this paper, within the scope of static offline cloud computing workflows and graph-based deep learning.

## III. METHODS

## A. About the Applied Deep Learning Models

The following sections provide information about the different deep learning models that are applied throughout this paper. The majority of these models leverage topological information in order to infer task intensities through different approaches. Each model exhibits distinct strengths.

1) Multi Layer Perceptron (MLP): The multi-layer perceptron [27] [25] is a widely used and very basic neural network, that does not apply any topological techniques by itself and is used as a baseline network for this paper.

2) Graph Convolutional Networks (GCN): Graph convolutional networks [19] are neural networks of the GNN family, based on convolutional neural networks. They are explicitly designed for applications where the data is encoded in the form of graphs and are based on message passing.

3) Graph Attention Networks (GATs): Graph attention networks [33] are graph neural networks that bring an attention mechanism into the neighborhood message-passing.

4) Graph Isomorphism Networks (GINs): Graph isomorphism networks [35] are GNNs that achieve the same discriminative power as the and the Weisfeiler-Lehman graph isomorphism test [34].

5) GraphSAGE: GraphSAGE (SAmple and aggreGatE) [18] is a GNN that aggregates neighbor features in a khop fashion. It starts by aggregating immediate neighbors and iteratively increases the search depth k to receive information from more distant neighborhoods. This mechanism allows GraphSAGE to capture both local and higher-order graph structures efficiently.

6) DAGTransformer: The DAGTransformer, introduced in [36], is a transformer encoder model that leverages task positions within the graph as positional encoding. Moreover, it uses the neighbors of tasks as an attention mask, enabling a GNN-like aggregation mechanism.

7) H2GCN: H2GCN [39] is a graph neural network, that was designed to tackle the challenges that are raised by heterophile graphs (graphs in which neighboring nodes often have very dissimilar labels), by employing special aggregation and embedding strategies.

8) Ensemble Models: Ensembles combine multiple weaker classifier models into a more robust and more efficient one [22]. For its simplicity and good performance, this paper utilizes supervised soft voting [22] ensembles of combined graph learning networks.

9) XGBoost: XGBoost [10] is a scalable system that handles nonlinearity by using a boosted tree structure, instead of activation functions as neural networks do. We utilize XGBoost for downstream regression experiments based on penultimate layers of pre-trained classification networks.

## B. Utilized Topological Features

To evaluate the impact of network topology in our experiments, we need a set of corresponding topological features. Many such features can be derived directly from the graph’s structure. The following sections describe these features in detail.

## 1) Node-Based Features:

a) Node Path Length.: In a graph, the path length between two nodes is the minimum number of edges that connect them. For this paper, we construct a feature vector for each node by calculating its path length to every other node in the graph. For example, a feature we call node 1 path length refers to the path length from the target node (i.e., the one being predicted) to ‘task node 1’ of the workflow.

b) DAG In & Out.: Input and output tasks of the current task are specified in the form of a flattened sparse matrix called DAG in & out [36]. In simpler terms, this is the normalized adjacency matrix for the given task. For a workflow with 7 tasks, the DAG in & out feature vector for each task has 15 dimensions. These dimensions are composed of 7 values for incoming edges, 7 for outgoing edges, and one final value for the task’s node ID.

c) Node Degree.: Node degree describes the number of edges of a node [15], treating each neighboring node equally.

d) Node Centrality.: Node centrality emphasizes the importance of nodes in a graph. Many centralities exist, and in this paper, three of them are utilized. These are eigenvector centrality, betweenness centrality, and closeness centrality:

Eigenvector centrality [29] is a score that is based on the premise, that a node in a graph is important if the node’s neighbors are important.

By the intuition behind betweenness centrality [5], a node is important, if it lies on many shortest paths between other nodes in the graph.

Closeness centrality [26] measures the lengths of paths between the node v and other nodes: If a node is part of many shortest path lengths to other nodes in a graph, it is well connected and therefore important.

e) Clustering Coefficient.: The local clustering coefficient measures how connected a node’s neighbors are within the graph [20], [30].

f) Node2Vec.: Node2Vec [17] is an unsupervised method of learning node embeddings in such a way, that nodes from the same neighborhood receive similar embeddings, based on random walks.

g) Color Refinement and Weisfeiler-Lehman (WL) Graph Kernel.: The color refinement (or naive vertex refinement) algorithm is a graph isomorphism test, that iteratively refines node colorings based on structural similarities.

During each iteration, the color of a node is updated based on its current color and the colors of its neighbors [14] and compressed into a new color. We use the final colors from the color refinement algorithm directly as node features.

h) Graphlets.: Subgraph patterns within a larger graph are called graphlets [1]. They represent the local structure of a graph by considering the connectivity patterns of a fixed number of nodes and edges.

This paper counts the occurrence of the task node in graphlets with up to three task nodes (edges, paths, triangles). The number of occurrences in each graphlet type is then passed as a feature embedding vector.

2) Katz Index: The Katz index [37] is a method used to measure the influence of a node within a graph, taking into account both direct and indirect connections. Each row of the Katz matrix captures the influence of a specific node on all other nodes in the graph with respect to direct and indirect connections. These influences are aggregated, resulting in a vector, that stores the total influence of each node across the entire graph. This vector can be used as a node feature vector.

Moreover, by aggregating the entries of the final influence vector, a graph-level Katz index can be acquired.

## C. Evaluation Metrics

Evaluating a deep learning model requires selecting appropriate metrics that reflect its performance on a given task. This paper mainly uses accuracy [31]. It is a common metric for classification problems, measuring the proportion of correctly predicted samples.

To evaluate regression models, we utilize Mean Absolute Error (MAE) [28], which describes the average size of errors within the set of predictions, as well as the coefficient of determination (or R<sup>2</sup>, R 2) [38], which is a value between 0 and 1 that explains the variation in the model’s predictions. A model that always predicts the average has an R-squared value of 0, and a perfectly fitting model has an R-squared value of 1.

## D. Alibaba Cluster-Trace-V2018 Dataset

Our experiments are performed on the Alibaba Cluster-Trace-V2018 based dataset from Yu et al.[36]. It consists of cloud workflows that include 7 tasks, and the CPU and memory usages of the 7th tasks are predicted. These workflows have been extracted from two .csv files, batch task and batch instance, which can be found within the Cluster-Trace-V2018 Github.

For each task, the authors extract 34 measured task-level features, mostly by aggregating instance-level features, such as CPU and memory usages or execution times. They create class labels by clustering mean and maximum of the task’s according usages.

The dataset is split into multiple splits, differing by training set size, validation set size, and test set size.

For example, when applying the split6 2 2 , Yu et al. utilize 60% of the data for training, and 20% for validation and testing, respectively.

## IV. DETERMINING TOPOLOGICAL FEATURE IMPORTANCES WITH MLP

This section details two experiments designed to evaluate the importance of different topological features when predicting task resource usage. To achieve this, we train a Multi-Layer

Perceptron (MLP) on workflows from the Alibaba dataset (see Section III-D) and compare the model’s performance using different topological feature combinations.

a) First MLP Experiment.: Within the scope of the first experiment, the MLP only learns from topological features of the predicted task: the only inputs that the MLP learns from are the different structural features that are compared and combined, including a baseline feature of node lengths to 7th tasks (with a variance of 0). Thus, the input vector for the target task (e.g., the 7th task in a workflow) contains a set of topological features of only the predicted 7th task.

b) Results of First MLP Experiment.: The results of the first experiment are depicted in Table I, demonstrating MLP classification accuracies of the mentioned 7th tasks with varying topological input features over various splits<sup>1</sup> of the mentioned dataset. For CPU intensity prediction, large increases in accuracy are achieved by introducing Weisfeiler-Lehman-based features, as described in III-B1. Katz indices and the centrality features (concatenated into ‘3 centralities’), as well as node path lengths, which are all explained in Section III-B, also enhance the prediction accuracies of the MLP significantly. It is noteworthy, that path lengths to tasks further away are superior to closer tasks. This is likely due to more variance in the 7th tasks’ path lengths to further tasks.

Node degrees, clustering coefficients and graphlets perform very similarly, likely because these three features share many characteristics.

The findings of the first experiment for the MLP in memory intensity prediction also emphasize the importance of topological information in predicting the memory intensity. The MLP’s base line accuracy for memory prediction from 7th tasks node path lengths starts at 90.98%.

While only a few structure-based features improve the memory accuracy over 90.98% alone, when introducing all workflow structure features to the MLP together, they interact in a way that increases accuracy by almost 4%. The highest accuracy is gained when combining all topological features during CPU prediction as well, stressing the importance of this feature interaction.

The ineffectiveness of Node2Vec based features (Section III-B1f) might be explained by the small graph size of 7 task workflows. With the small graph size, there could not be enough variance in task neighborhoods to form expressive embeddings.

In summary, the first MLP experiment shows that adding topological information of the predicted tasks leads to higher accuracy than the baseline features, indicating that the tasks structural characteristics alone provide useful information for predicting task intensity, when using a very basic neural network.

c) Second MLP Experiment.: Within the scope of the second experiment, the MLP learns task intensities from the entire workflow instead of only the predicted task, including the following features:

• the measured task level intensity features of all earlier tasks, explained in Section III-D

• topological features of all earlier tasks and the predicted task

This approach allows for the exploration of how each topological feature, in combination with measured features and other tasks’ topological features, influences accuracy, as opposed to only leveraging the predicted task’s topological features, as in the first experiment, creating a link from MLP to graph learning.

d) Results of Second MLP Experiment.: The results of the second experiment are depicted in Table II, which is constructed similarly to the table from the first MLP experiment. It shows that the MLP benefits from the inclusion of previous task’s topological features during training without measured features, in comparison to the accuracies of the MLP in experiment 1. However, the difference here is not immense, so the topology of the predicted task’s node is likely the most important. In combination with measured features, all implemented topological features add to the performance of the MLP that is trained on measured features, displaying the effectiveness of the combination of topological and measured features of all tasks.

Similarly to the first experiment, for memory and CPU intensity prediction, the MLP can take great advantage of Weisfeiler-Lehman and centrality-based features, but also topological features that were not as promising in experiment 1. The most dominant topological feature in experiment 2 is the DAG in&out sparse matrix, which contains the normalized adjacency list. Since it holds the entire workflow structure now, this is expected.

Moreover, all other employed structural features increase the MLP accuracy significantly into a similar range of accuracy. Additionally, training the MLP using a combination of all topological features also increases accuracy drastically, only surpassed by the sparse matrix.

The results of the second MLP experiment demonstrate that incorporating the topological features of all tasks within the workflow is highly beneficial for task-intensity prediction using a basic deep learning model, substantially improving the accuracy of the MLP even when only measured features are used.

## V. GRAPH DEEP LEARNING ON WORKFLOWS

The experiments in Section IV established that workflow topology is a useful feature for predicting task intensities. Building on these findings, this section transitions to the evaluation of the same dataset with graph-based learning models, which are specifically designed for such data.

The models that we decided to compare to each other, and to the MLP of the previous section, include GIN, GAT, GraphSAGE, as well as an ensemble of these three. Moreover, we benchmark the performance of H2GCN, GCN and DAGTransformer. All models are explained in Section III-A.

<table><tr><td>Feature Method</td><td>CPU Accuracy  $( 1 ^ { \mathrm { s t } } ~ \mathrm { S p l i t } )$ </td><td>CPU Accuracy (2ⁿd Split)</td><td>CPU Accuracy  $( 3 ^ { \mathrm { r d } } ~ \mathrm { S p l i t } )$ </td><td>Memory Accuracy</td></tr><tr><td>DAG In &amp; Out</td><td>51.87%</td><td>52.18%</td><td>51.99%</td><td>91.65%</td></tr><tr><td>Node Degrees</td><td>48.27%</td><td>48.24%</td><td>48.15%</td><td>90.98%</td></tr><tr><td>Clustering Coefficient</td><td>48.14%</td><td>48.25%</td><td>48.18%</td><td>90.98%</td></tr><tr><td>Graphlets</td><td>48.27%</td><td>48.10%</td><td>48.39%</td><td>90.98%</td></tr><tr><td>Node 1 Path Length</td><td>70.74%</td><td>70.50%</td><td>70.00%</td><td>90.98%</td></tr><tr><td>Node 2 Path Length</td><td>65.55%</td><td>63.30%</td><td>64.58%</td><td>90.98%</td></tr><tr><td>Node 3 Path Length</td><td>54.93%</td><td>54.03%</td><td>55.24%</td><td>90.98%</td></tr><tr><td>Node 4 Path Length</td><td>47.16%</td><td>46.41%</td><td>47.30%</td><td>90.98%</td></tr><tr><td>Node 5 Path Length</td><td>45.53%</td><td>45.39%</td><td>44.37%</td><td>90.98%</td></tr><tr><td>Node 6 Path Length</td><td>44.55%</td><td>45.47%</td><td>44.99%</td><td>90.98%</td></tr><tr><td>Node 7 Path Length</td><td>45.34%</td><td>45.46%</td><td>44.67%</td><td>90.98%</td></tr><tr><td>All Path Lengths</td><td>74.56%</td><td>74.26%</td><td>73.88%</td><td>93.06%</td></tr><tr><td>Weisfeiler-Lehman</td><td>69.70%</td><td>70.15%</td><td>68.24%</td><td>90.98%</td></tr><tr><td>Katz Nodelevel</td><td>62.05%</td><td>59.19%</td><td>62.33%</td><td>90.98%</td></tr><tr><td>Katz Graphlevel</td><td>65.55%</td><td>65.91%</td><td>65.76%</td><td>90.98%</td></tr><tr><td>Eigenvector Centrality</td><td>63.61%</td><td>64.74%</td><td>73.88%</td><td>90.98%</td></tr><tr><td>Betweenness Centrality</td><td>47.75%</td><td>47.83%</td><td>48.36%</td><td>91.69%</td></tr><tr><td>Closeness Centrality</td><td>66.62%</td><td>67.05%</td><td>66.61%</td><td>90.98%</td></tr><tr><td>3 Centralities Node2Vec embedding</td><td>73.97%</td><td>74.12%</td><td>73.44% 44.90%</td><td>92.89%</td></tr><tr><td>All Topology Features</td><td>45.15% 75.93%</td><td>44.79%</td><td>75.07%</td><td>90.98%</td></tr><tr><td></td><td></td><td>75.76%</td><td></td><td>94.42%</td></tr></table>

TABLE I: Impact of topological features on CPU and memory intensity prediction (experiment 1).

<table><tr><td>Feature Method</td><td>CPU Accuracy (1st Split)</td><td>CPU Accuracy  $( 2 ^ { \mathrm { n d } }$  Split)</td><td>CPU Accuracy  $( 3 ^ { \mathrm { r d } }$  Split)</td><td>Memory Accuracy</td></tr><tr><td>Only Measured Features</td><td>86.60%</td><td>86.82%</td><td>87.00%</td><td>96.21%</td></tr><tr><td>Only All Topology Features</td><td>77.63%</td><td>77.20%</td><td>77.55%</td><td>95.04%</td></tr><tr><td>DAG In &amp; Out</td><td>90.16%</td><td>89.43%</td><td>90.22%</td><td>98.51%</td></tr><tr><td>Clustering Coefficient</td><td>88.59%</td><td>87.53%</td><td>87.83%</td><td>97.80%</td></tr><tr><td>Node Degrees</td><td>88.33%</td><td>87.40%</td><td>88.86%</td><td>97.72%</td></tr><tr><td>Graphlets</td><td>87.98%</td><td>87.66%</td><td>89.25%</td><td>97.76%</td></tr><tr><td>All Path Lengths</td><td>88.91%</td><td>87.65%</td><td>88.86%</td><td>97.73%</td></tr><tr><td>3 Centralities</td><td>89.04%</td><td>88.28%</td><td>89.36%</td><td>97.87%</td></tr><tr><td>Weisfeiler-Lehman</td><td>88.38%</td><td>87.84%</td><td>89.01%</td><td>97.31%</td></tr><tr><td>Katz Nodelevel</td><td>89.30%</td><td>87.29%</td><td>88.50%</td><td>97.89%</td></tr><tr><td>All Topology Features</td><td>89.15%</td><td>89.00%</td><td>89.71%</td><td>98.21%</td></tr></table>

TABLE II: Impact of topological features when topology information is considered for all nodes (experiment 2).

<table><tr><td>GNN</td><td>CPU Acc. (1st Split)</td><td> $\mathrm { C P U ~ A c c . ~ ( 2 ^ { n d } ~ S p l i t ) }$ </td><td> $\mathrm { C P U ~ A c c . } ~ ( 3 ^ { \mathrm { r d } } ~ \mathrm { S p l i t ) }$ </td></tr><tr><td>GCN</td><td> $8 8 . 3 8 \pm 0 . 4 \%$ </td><td> $8 8 . 1 8 \pm 0 . 2 \%$ </td><td> $8 8 . 9 1 \pm 0 . 1 \%$ </td></tr><tr><td>GAT</td><td> $8 9 . 8 1 \pm 0 . 2 \%$ </td><td> $8 9 . 6 1 \pm 0 . 2 \%$ </td><td> $9 0 . 4 8 \pm 0 . 3 \%$ </td></tr><tr><td>GIN</td><td> $8 9 . 8 7 \pm 0 . 1 \%$ </td><td> $8 9 . 6 3 \pm 0 . 2 \%$ </td><td> $9 0 . 6 4 \pm 0 . 3 \%$ </td></tr><tr><td>GraphSAGE</td><td> $8 7 . 8 0 \pm 0 . 2 \%$ </td><td> $8 7 . 3 5 \pm 0 . 2 \%$ </td><td> $8 8 . 1 8 \pm 0 . 2 \%$ </td></tr><tr><td>H2GCN</td><td> $8 7 . 8 4 \pm 0 . 1 \%$ </td><td> $8 7 . 1 1 \pm 0 . 1 \%$ </td><td> $8 8 . 2 0 \pm 0 . 1 \%$ </td></tr><tr><td>DAGTransformer (stated in paper)</td><td> $9 1 . 2 5 \pm 0 . 0 4 \%$ </td><td> $9 1 . 1 1 \pm 0 . 0 5 \%$ </td><td> $9 2 . 1 5 \pm 0 . 1 3 \%$ </td></tr><tr><td>DAGTransformer (reproduced)</td><td> $9 0 . 5 4 \pm 0 . 1 \%$ </td><td> $8 9 . 9 6 \pm 0 . 0 5 \%$ </td><td> $9 0 . 6 1 \pm 0 . 3 \%$ </td></tr><tr><td>GNN Ensemble</td><td> $9 1 . 1 2 \pm 0 . 1 \%$ </td><td> $9 0 . 5 2 \pm 0 . 2 \%$ </td><td> $9 1 . 5 9 \pm 0 . 2 \%$ </td></tr></table>

TABLE III: Graph learning models CPU average prediction accuracies without topological features.

All models are trained on the 7th task of each workflow, with and without all of the additional structural features for all tasks, that were employed in section IV, to explore a possible impact of additional topological information on the graph learning results. The accuracies of these models on the Alibaba Cluster-Trace-V2018 dataset can be found in Tables III and IV for CPU usage prediction and Table V for memory usage prediction. In the tables, acc. refers to accuracy.

a) Discussion of the Graph Learning Results.: We find that the GNN accuracies lay in a similar area to the MLP accuracies of experiment 2 in Section IV. While GCN, Graph-SAGE, and H2GCN perform slightly worse in comparison to the MLP with all nodes’ topological information, GAT, GIN,

DAGTransformer, and the GNN ensemble outperform it, even without the addition of features that describe the workflow structures.

The strong performance of the GIN model implies that workflow isomorphism is an important factor in predicting CPU and memory intensities. This finding is consistent with the accuracy improvements observed in Section IV, where introducing Weisfeiler-Lehman labels to the MLP also enhanced its performance. This significant influence of isomorphism may be attributed to the dataset’s composition, as it consists exclusively of 7-task workflows. Such uniformity likely leads to a high number of isomorphic graphs, causing certain structural patterns to repeat frequently within the data.

<table><tr><td>GNN</td><td> $\mathrm { C P U ~ A c c . ~ ( 1 ^ { s t } ~ S p l i t ) }$ </td><td> $\mathrm { C P U ~ A c c . ~ ( 2 ^ { n d } ~ S p l i t ) }$ </td><td> $\mathrm { C P U ~ A c c . } ~ ( 3 ^ { \mathrm { r d } } ~ \mathrm { S p l i t ) }$ </td></tr><tr><td>GCN</td><td> $9 0 . 3 4 \pm 0 . 3 \%$ </td><td> $8 9 . 6 7 \pm 0 . 1 \%$ </td><td> $9 0 . 3 5 \pm 0 . 4 \%$ </td></tr><tr><td>GAT</td><td> $9 0 . 5 4 \pm 0 . 1 \%$ </td><td> $8 9 . 8 9 \pm 0 . 1 \%$ </td><td> $9 0 . 8 9 \pm 0 . 2 \%$ </td></tr><tr><td>GIN</td><td> $9 0 . 5 6 \pm 0 . 1 \%$ </td><td> $9 0 . 1 4 \pm 0 . 2 \%$ </td><td> $9 0 . 7 4 \pm 0 . 0 5 \%$ </td></tr><tr><td>GraphSAGE</td><td> $8 8 . 9 7 \pm 0 . 4 \%$ </td><td> $8 8 . 0 8 \pm 0 . 3 \%$ </td><td> $8 9 . 1 0 \pm 0 . 1 \%$ </td></tr><tr><td>H2GCN</td><td> $8 9 . 3 1 \pm 0 . 1 \%$ </td><td> $8 8 . 7 3 \pm 0 . 3 \%$ </td><td> $8 9 . 8 1 \pm 0 . 2 \%$ </td></tr><tr><td>DAGTransformer (stated in paper)</td><td></td><td></td><td></td></tr><tr><td>DAGTransformer (reproduced)</td><td> $9 0 . 7 5 \pm 0 . 1 \%$ </td><td> $9 0 . 3 7 \pm 0 . 2 \%$ </td><td> $9 0 . 8 9 \pm 0 . 2 \%$ </td></tr><tr><td>GNN Ensemble</td><td> $9 1 . 2 1 \pm 0 . 1 \%$ </td><td> $9 0 . 2 4 \pm 0 . 1 \%$ </td><td> $9 1 . 4 5 \pm 0 . 3 \%$ </td></tr></table>

TABLE IV: Graph learning models CPU average prediction accuracies with topological features.

<table><tr><td>GNN</td><td>Memory Accuracy</td><td>Memory Accuracy with Topological Features</td></tr><tr><td>GCN</td><td> $9 7 . 6 7 \pm 0 . 0 3 \%$ </td><td> $9 8 . 5 2 \pm 0 . 0 3 \%$ </td></tr><tr><td>GAT</td><td> $9 8 . 3 4 \pm 0 . 1 \%$ </td><td> $9 8 . 6 8 \pm 0 . 1 \%$ </td></tr><tr><td>GIN</td><td> $9 8 . 4 7 \pm 0 . 2 \%$ </td><td> $9 8 . 5 9 \pm 0 . 1 \%$ </td></tr><tr><td>GraphSAGE</td><td> $9 7 . 3 3 \pm 0 . 1 \%$ </td><td> $9 7 . 7 8 \pm 0 . 4 \%$ </td></tr><tr><td>H2GCN</td><td> $9 0 . 9 8 \pm 0 . 0 1 \%$ </td><td> $9 0 . 9 8 \pm 0 . 0 0 4 \%$ </td></tr><tr><td>DAGTransformer (stated)</td><td> $9 8 . 5 6 \%$ </td><td></td></tr><tr><td>DAGTransformer (reproduced)</td><td> $9 8 . 3 5 \pm 0 . 1 \%$ </td><td> $9 8 . 5 9 \pm 0 . 1 \%$ </td></tr><tr><td>GNN Ensemble</td><td> $9 8 . 3 2 \pm 0 . 2 \%$ </td><td> $9 8 . 3 6 \pm 0 . 3 \%$ </td></tr></table>

TABLE V: Graph learning models memory average prediction.

The efficiency of GAT can possibly be traced back to its attention mechanism, which enables it to recognize more complex patterns than other GNN networks. It allows the GAT network to learn weights for a task’s neighbors. These neighbor weights can be learned by patterns such as the neighbor’s influence on other nodes, which ultimately defines node centrality as described in Section III-B1. Since leveraging task centralities as well as the Katz index helped the MLP network in Section IV to enhance its accuracy when predicting task intensities a lot, these could be the key patterns for the GAT performances.

Although GraphSAGE alone does not perform as well as other models, it enhances the performance of the GNN ensembles; thus, its special aggregation strategy might add to the features of the other mentioned GNNs.

From the H2GCN accuracies, it can be deducted that workflow heterophily is not very problematic on the dataset, since other GNNs were able to outperform the H2GCN, which was explicitly created to tackle graphs with strong heterophilic properties that negatively affect other GNNs.

Lastly, while we were not able to exactly reproduce the claimed accuracies, the DAGTransformer performed extremely well, which is expected, as the dataset was also used in the original DAGTransformer paper. It should be mentioned that the DAGTransformer also attends to different neighboring nodes in different ways, so it likely learns similar patterns as the GAT.

Furthermore, by comparing the model performances in Tables III, IV, and V, we find that the accuracies of almost all introduced geometric deep learning models are slightly improved when additional graph structural features are included, compared to when these features are not present, allowing GCN and partially H2GCN to exceed the MLP accuracies. This leads to the conclusion that even geometric neural networks do not have the ability to capture some important graph structure of the cloud workflows, which can be leveraged for better intensity predictions.

The only exception to this behavior is the GNN ensemble, which covers a wide area of graph structure by evaluating multiple graph-based networks. From this behavior, we can derive that it already gathers a large part of the information that the additional topology-based features introduce to other models by itself.

## VI. TRANSFERABILITY OF EMBEDDINGS: REGRESSING THE ACTUAL CPU & MEMORY UTILIZATION

Previously, throughout this paper, all employed deep learning and graph learning models were trained and evaluated on 7-task large workflow graphs, and using classification accuracy. While classification is helpful for benchmarking models in this case, it might not represent use cases of such graph learning models in the real world.

For real applications, it is most likely that predictions would need to be made over various workflow graph sizes, and exact predictions might be more useful than classification. Therefore, a regression model would be more applicable.

Moreover, it would not be computationally efficient to train a large graph learning model for each workflow size. While there are likely many possible pipelines to approach this problem, a good alternative that fits our structure could be inferring the penultimate layer of a pre-trained model, which contains aggregated structure information (if it can generalize), on new graph sizes, and train a small regressor on that layer to transfer the model information for each workflow size.

## A. The Regression Experiment

To simulate such an approach, we introduce a dataset of 20th and 18th tasks, in order to test the graph learning models generalization capabilities. It is extracted from the Alibaba

Cluster-Trace-V2018 dataset similarly as explained in section III-D, except for workflow size. Instead of clustered class labels, we use measured data of the predicted tasks as ground truth.

By inferring with pre-trained models and training on the resulting penultimate layers, we explore CPU and memory regression, with and without using the topological features.

Since it is the best performing model that profits from the topological features that we benchmarked for classification, we leverage the pre-trained DAGTransformer embeddings to train an XGBoost regressor model, that predicts CPU and memory usage of the 18th and 20th tasks.

First, the XGBoost model is fed by penultimate layer embeddings of the standard DAGTransformer. Afterwards, all previously introduced topological features are computed for the predicted tasks and aggregated with the penultimate layer embeddings to explore a possible enhancement of the XGBoost’s predictions.

## B. Results of the XGBoost Model on Penultimate Layers

Tables VI and VII show mean absolute error and $R ^ { 2 }$ scores of these XGBoost models: avg is short for average, max for maximum, and Topo references additional topological features of the predicted task. These results, as well as the regression plots in Figures 2a to 2b, show that the penultimate layers can be useful for building regression models that predict task-level CPU and memory utilization, even for larger workflows, while weaker for most max CPU predictions.

<table><tr><td>task</td><td>avg CPU</td><td>max CPU</td><td>avg mem</td><td>max mem</td></tr><tr><td>20 Mean</td><td>8.42</td><td>22.76</td><td>0.085</td><td>0.10</td></tr><tr><td>20 Mean Topo</td><td>7.77</td><td>22.51</td><td>0.074</td><td>0.09</td></tr><tr><td>20 Max</td><td>9.79</td><td>34.60</td><td>0.11</td><td>0.12</td></tr><tr><td>20 Max Topo</td><td>9.53</td><td>33.11</td><td>0.10</td><td>0.10</td></tr><tr><td>18 Mean</td><td>11.51</td><td>30.24</td><td>0.06</td><td>0.08</td></tr><tr><td>18 Mean Topo</td><td>10.67</td><td>28.42</td><td>0.06</td><td>0.08</td></tr><tr><td>18Max</td><td>13.40</td><td>52.44</td><td>0.08</td><td>0.09</td></tr><tr><td>18 Max Topo</td><td>12.75</td><td>52.80</td><td>0.08</td><td>0.09</td></tr></table>

TABLE VI: MAE values for average CPU & memory utilization.

<table><tr><td>task</td><td>avg CPU</td><td>max CPU</td><td>avg mem</td><td>max mem</td></tr><tr><td>20 Mean</td><td>0.85</td><td>0.41</td><td>0.95</td><td>0.94</td></tr><tr><td>20 Mean Topo</td><td>0.87</td><td>0.51</td><td>0.96</td><td>0.95</td></tr><tr><td>20 Max</td><td>0.83</td><td>0.45</td><td>0.93</td><td>0.92</td></tr><tr><td>20 Max Topo</td><td>0.84</td><td>0.50</td><td>0.94</td><td>0.94</td></tr><tr><td>18 Mean</td><td>0.79</td><td>0.68</td><td>0.41</td><td>0.61</td></tr><tr><td>18 Mean Topo</td><td>0.81</td><td>0.75</td><td>0.46</td><td>0.60</td></tr><tr><td>18Max</td><td>0.77</td><td>0.66</td><td>0.53</td><td>0.59</td></tr><tr><td>18 Max Topo</td><td>0.80</td><td>0.64</td><td>0.39</td><td>0.67</td></tr></table>

TABLE VII: $\mathbb { R } ^ { 2 }$ scores for average CPU & memory utilization.

The Tables VI and VII show, that mean absolute errors and $R ^ { 2 }$ of regression predictions of 20th and 18th tasks reach MAEs down to 8.42 for CPU prediction and 0.085 for memory prediction. The regression plots in Figure 2 show that wrong predictions in task intensities are often those with very high values, which could be traced back to a low amount of such training data within the utilized dataset, but generally, the predictions are close to the diagonal.

![](images/85b687ec2f3b5853e1bc22e45dddd9e8172734da9cd4c3e0016117f1c1a9ae02.jpg)  
(a) Regression of measured instance CPU average aggregated by mean for 20th tasks with topology.

![](images/d6236bb552b64dab8878c9d3e493d1a29ddba67fec1094ee83f7034040a8acaf.jpg)  
(b) Regression of measured instance memory maximum aggregated by maximum for 20th tasks with topology.  
Fig. 2: Regression plots when model embeddings are used to feed XGBoost.

Moreover, as shown in Tables VI and VII, training the regression model with a concatenation of the penultimate layer and additional topological information of the predicted task node as input features increases its performance, even though the DAGTransformer already collects topological information. This reinforces that even graph-learning-based embeddings/models cannot capture the entire topological image of the predicted task/workflow.

## VII. CONCLUSION

The results of this paper show that learnable patterns connect many different aspects of workflow topology and task intensity.

Thus, by introducing different aspects of task and workflow topology as features, even simple models can efficiently explore a variety of patterns, and by combining these features, a high accuracy can be achieved, as shown by the experiments in Section IV.

Moreover, we explored a variety of graph-learning-based models in Section V, which mostly outperformed the MLP.

While graph learning is very effective at predicting CPU and memory usage of future tasks, even these models do not always capture all aspects of topology that are useful. Thus, topology-based features can also slightly improve the performance of graph learning models.

Next, we explored the generalization of Graph learning and topological features by combining a DAGTransformer backbone with an XGBoost regressor. While trained on 7- task graphs, the model effectively generalizes to 18 and 20-task workflows, but additional topological input features furthermore enhanced the MAE’s of the pipeline.

In essence, the results of this paper show that it can be advantageous to add many different aspects of workflow graph structure to a model’s input features when predicting task CPU and memory needs, even if the model already collects some graph information.

## ACKNOWLEDGEMENTS

This paper was supported by the Swarmchestrate project of the European Union’s Horizon 2023 Research and Innovation programme under grant agreement no. 101135012 and the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) as FONDA (Project 414984028, SFB 1404).

## REFERENCES

[1] N. K. Ahmed, J. Neville, R. A. Rossi, and N. Duffield, “Efficient graphlet counting for large networks,” in 2015 IEEE international conference on data mining. IEEE, 2015, pp. 1–10.

[2] M. S. Al-Asaly, M. A. Bencherif, A. Alsanad, and M. M. Hassan, “A deep learning-based resource usage prediction model for resource provisioning in an autonomic cloud computing environment,” Neural Computing and Applications, vol. 34, no. 13, pp. 10 211–10 228, 2022.

[3] A. Ali-Eldin, O. Seleznjev, S. Sjostedt-de Luna, J. Tordsson, and¨ E. Elmroth, “Measuring cloud workload burstiness,” in 2014 IEEE/ACM 7th International Conference on Utility and Cloud Computing. IEEE, 2014, pp. 566–572.

[4] J. Bader, N. Zunker, S. Becker, and O. Kao, “Leveraging reinforcement learning for task resource allocation in scientific workflows,” in 2022 IEEE International Conference on Big Data (Big Data). IEEE, 2022, pp. 3714–3719.

[5] M. Barthelemy, “Betweenness centrality in large complex networks,” The European physical journal B, vol. 38, no. 2, pp. 163–168, 2004.

[6] K. B. Bey, F. Benhammadi, A. Mokhtari, and Z. Guessoum, “Cpu load prediction model for distributed computing,” in 2009 Eighth International Symposium on Parallel and Distributed Computing, 2009, pp. 39–45.

[7] G. Boss, P. Malladi, D. Quan, L. Legregni, and H. Hall, “Cloud computing,” IBM white paper, vol. 321, pp. 224–231, 2007.

[8] J. Cao, J. Fu, M. Li, and J. Chen, “Cpu load prediction for cloud environment based on a dynamic ensemble model,” Software: Practice and Experience, vol. 44, no. 7, pp. 793–804, 2014.

[9] S. Chandrasiri and D. Meedeniya, “Energy-efficient dynamic workflow scheduling in cloud environments using deep learning,” Sensors, vol. 25, no. 5, p. 1428, 2025.

[10] T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” in Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, 2016, pp. 785–794.

[11] P. Cong, G. Xu, T. Wei, and K. Li, “A survey of profit optimization techniques for cloud providers,” ACM Computing Surveys (CSUR), vol. 53, no. 2, pp. 1–35, 2020.

[12] R. F. da Silva, G. Juve, E. Deelman, T. Glatard, F. Desprez, D. Thain, B. Tovar, and M. Livny, “Toward fine-grained online task characteristics estimation in scientific workflows,” in Proceedings of the 8th Workshop on Workflows in Support of Large-Scale Science, ser. WORKS ’13. New York, NY, USA: Association for Computing Machinery, 2013, p. 58–67. [Online]. Available: https://doi.org/10.1145/2534248.2534254

[13] R. Diestel, Graph theory. Springer (print edition); Reinhard Diestel (eBooks), 2024.

[14] B. L. Douglas, “The weisfeiler-lehman method and graph isomorphism testing,” arXiv preprint arXiv:1101.5211, 2011.

[15] P. GA, S. M, M. CN, S. TG, K. S, A. J, S. R, and B. PG, “Using graph theory to analyze biological networks,” BioData Min., 2011.

[16] M. Gao, Y. Li, and J. Yu, “Workload prediction of cloud workflow based on graph neural network,” in Web Information Systems and Applications: 18th International Conference, WISA 2021, Kaifeng, China, September 24–26, 2021, Proceedings 18. Springer, 2021, pp. 169–189.

[17] A. Grover and J. Leskovec, “node2vec: Scalable feature learning for networks,” in Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining, 2016, pp. 855– 864.

[18] W. Hamilton, Z. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” Advances in neural information processing systems, vol. 30, 2017.

[19] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” arXiv preprint arXiv:1609 .02907, 2016.

[20] Y. Li, Y. Shang, and Y. Yang, “Clustering coefficients of large networks,” Information Sciences, vol. 382, pp. 350–358, 2017.

[21] X. Ma, H. Xu, H. Gao, and M. Bian, “Real-time multiple-workflow scheduling in cloud environments,” IEEE Transactions on Network and Service Management, vol. 18, no. 4, pp. 4002–4018, 2021.

[22] A. Manconi, G. Armano, M. Gnocchi, and L. Milanesi, “A soft-voting ensemble classifier for detecting patients affected by covid-19,” Applied Sciences, vol. 12, no. 15, p. 7554, 2022.

[23] M. Masdari and M. Zangakani, “Efficient task and workflow scheduling in inter-cloud environments: challenges and opportunities,” The Journal of Supercomputing, vol. 76, no. 1, pp. 499–535, 2020.

[24] T. Mehmood, S. Latif, and S. Malik, “Prediction of cloud computing resource utilization,” in 2018 15th International Conference on Smart Cities: Improving Quality of Life Using ICT & IoT (HONET-ICT). IEEE, 2018, pp. 38–42.

[25] L. Noriega, “Multilayer perceptron tutorial,” School of Computing. Staffordshire University, vol. 4, no. 5, p. 444, 2005.

[26] K. Okamoto, W. Chen, and X.-Y. Li, “Ranking of closeness centrality for large-scale social networks,” in International workshop on frontiers in algorithmics. Springer, 2008, pp. 186–195.

[27] M.-C. Popescu, V. E. Balas, L. Perescu-Popescu, and N. Mastorakis, “Multilayer perceptron and neural networks,” WSEAS Transactions on Circuits and Systems, vol. 8, no. 7, pp. 579–588, 2009.

[28] J. Qi, J. Du, S. M. Siniscalchi, X. Ma, and C.-H. Lee, “On mean absolute error for deep neural network based vector-to-vector regression,” IEEE Signal Processing Letters, vol. 27, pp. 1485–1489, 2020.

[29] B. Ruhnau, “Eigenvector-centrality—a node-centrality?” Social networks, vol. 22, no. 4, pp. 357–365, 2000.

[30] S. N. Soffer and A. Vazquez, “Network clustering coefficient without degree-correlation biases,” Physical Review E—Statistical, Nonlinear, and Soft Matter Physics, vol. 71, no. 5, p. 057101, 2005.

[31] M. Sokolova, N. Japkowicz, and S. Szpakowicz, “Beyond accuracy, f-score and roc: a family of discriminant measures for performance evaluation,” in Australasian joint conference on artificial intelligence. Springer, 2006, pp. 1015–1021.

[32] B. Tovar, R. F. da Silva, G. Juve, E. Deelman, W. Allcock, D. Thain, and M. Livny, “A job sizing strategy for high-throughput scientific workflows,” IEEE Transactions on Parallel and Distributed Systems, vol. 29, no. 2, pp. 240–253, 2018.

[33] P. Velickoviˇ c, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Ben-´ gio, “Graph attention networks,” arXiv preprint arXiv:1710 .10903, 2017.

[34] B. Weisfeiler and A. Leman, “The reduction of a graph to canonical form and the algebra which appears therein,” nti, Series, vol. 2, no. 9, pp. 12–16, 1968.

[35] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How powerful are graph neural networks?” arXiv preprint arXiv:1810 .00826, 2018.

[36] J. Yu, M. Gao, Y. Li, Z. Zhang, W. H. Ip, and K. L. Yung, “Workflow performance prediction based on graph structure aware deep attention neural network,” Journal of Industrial Information Integration, vol. 27, p. 100337, 2022.

[37] J. Zhan, S. Gurung, and S. P. K. Parsa, “Identification of top-k nodes in large networks using katz centrality,” Journal of Big Data, vol. 4, no. 1, p. 16, 2017.

[38] D. Zhang, “A coefficient of determination for generalized linear models,” The American Statistician, vol. 71, no. 4, pp. 310–316, 2017.

[39] J. Zhu, Y. Yan, L. Zhao, M. Heimann, L. Akoglu, and D. Koutra, “Beyond homophily in graph neural networks: Current limitations and effective designs,” Advances in neural information processing systems, vol. 33, pp. 7793–7804, 2020.