# Fermat ProbCover: Coverage-Based Active Learning on Fermat Distance Graphs

Gokul Bhusal<sup>1</sup>, Delanna Do<sup>2</sup>, Kevin Miller<sup>3</sup>, Ekaterina Merkurjev<sup>4</sup> and Stephan Wojtowytsch<sup>2</sup> <sup>1</sup> Department of Mathematics,

University of California, Los Angeles, CA 90095, USA.

<sup>2</sup>Department of Mathematics,

University of Pittsburgh, Pittsburgh, PA 15213, USA.

<sup>3</sup> Department of Mathematics,

Brigham Young University, UT 84602, USA.

<sup>4</sup> Department of Mathematics and CMSE,

Michigan State University, MI 48824, USA.

Abstract—Active learning aims to reduce annotation cost by selecting the most informative examples for labeling. In the low-budget regime, where uncertainty-based methods often fail, coverage-based active learning methods such as ProbCover have been shown to be effective. However, these methods typically rely on Euclidean distance, which may not adequately capture the underlying geometry of data. In this article, we propose PC-Fermat, a coverage-based active learning method that replaces Euclidean coverage neighborhoods with neighborhoods defined by a Fermat distance, i.e., a data-adaptive distance that reflects the underlying distribution of the data points. To facilitate computation, we construct a weighted Euclidean neighborhood graph and use a truncated shortest-path algorithm to compute the Fermat coverage graph. We also establish sufficient conditions under which this construction exactly preserves the Fermat coverage computed on the complete graph. Finally, we evaluate PC-Fermat on eight classification datasets across varying labeling budgets, with particular emphasis on the very low-budget regime. Our experimental results demonstrate that PC-Fermat outperforms several existing active learning methods on the majority of the datasets.

Impact Statement—Sufficient amounts of labeled data are essential for achieving strong performance in machine learning tasks. However, obtaining high-quality labeled data is often expensive, time-consuming, and subject to practical or ethical constraints. These challenges are particularly evident in domains such as healthcare, biology, and remote sensing, where expert annotation is costly and data collection may involve privacy and safety considerations. Active learning has emerged as a promising approach for addressing limited labeling budgets by selecting the most informative samples for annotation. By reducing the number of samples that require labeling while preserving data diversity and representativeness, active learning can substantially improve annotation efficiency. The proposed algorithm advances this goal, and our experimental results demonstrate that it consistently improves the performance of supervised and semisupervised learning algorithms while requiring fewer labeled samples. These improvements can help lower annotation costs, facilitate the use of machine learning methods in data-scarce settings, and broaden the applicability in realworld applications.

Index Terms—Active learning, Fermat distance, Diversity sampling.

## I. INTRODUCTION

driven by the availability of large amounts of labeled data. However, obtaining sufficient amounts of labeled data for real-world applications is often expensive, time-consuming, and subject to ethical and practical considerations [1, 2]. Active learning [3] seeks to address this challenge by reducing the amount of labeled data required for effective model training. Specifically, given a fixed labeling budget of b examples, active learning methods aim to select a subset of b examples from an unlabeled dataset for annotation, with the goal of maximizing the predictive performance of a model trained on the resulting labeled data.

We distinguish one-shot coreset-selection methods from adaptive batch-selection methods in active learning. One-shot, coreset-selection methods [4] identify the set of examples for labeling up front, prior to any data labeling or model training. In contrast, adaptive batch-selection methods repeat a three-step cycle at each iteration: (i) train a model on the current labeled set, (ii) adaptively select unlabeled examples for labeling, and (iii) update the labeled set. In either setting, one can think of the active learning methods as defining a criterion (or acquisition function) that quantifies the utility of including a currently unlabeled example in the set to be labeled. Coreset selection methods are useful for batch-selection methods, at least as an initial step and for maintaining data diversity.

Acquisition functions for active learning can generally be classified into uncertainty-based [3, 5–8] and representation-based [9, 10] criteria. Within the adaptive, batch-mode setting of active learning, uncertainty-based methods prioritize samples for which the current model is least confident, allowing the classifier to refine its decision boundaries and to maximize the added value of new annotation. Representation-based criteria, on the other hand, seek to select samples from diverse regions of the data distribution, ensuring that all regions of the input space are adequately covered or “represented” in the labeled set.

One of the key challenges in the practical application of active learning is to balance uncertainty-based and representation-based query strategies, a problem that is often viewed as an exploration-exploitation trade-off. Representation-based methods encourage exploration by selecting samples that provide broad coverage of the underlying data distribution, whereas uncertainty-based methods promote exploitation by querying samples near the model’s decision boundary to refine its predictions. This balance is particularly important in the low-budget regime, where many uncertainty-based active learning strategies may fail to outperform random sampling [9, 10]. In particular, models trained on only a small number of labeled examples can produce unreliable uncertainty estimates, limiting the effectiveness of uncertainty-based selection. In contrast, coreset-selection methods prioritize representative sample selection and can achieve better performance in low-budget settings. These observations highlight the importance of prioritizing exploration over exploitation during the early stages of active learning, analogous to early-stage reinforcement learning.

## A. Our Approach

We focus on one-shot coreset-selection methods for representation-based active learning to support effective learning under limited labeling budgets. Among such methods, ProbCover [9] provides a metric-based approach to coreset selection that seeks to maximize dataset coverage by selecting samples that effectively cover the underlying data distribution. In more technical terms, ProbCover produces a coverage graph of ϵ-nearest neighbor distances to tabulate the extent to which a selected coreset “covers” the dataset. Importantly, Prob-Cover does not require a particular distance metric; rather, it simply relies on a metric structure over the representation space. This observation motivates our approach: instead of modifying the ProbCover selection mechanism, we investigate how its performance can be improved by equipping it with a data-adaptive metric that better captures the underlying geometry of highdimensional data.

The effectiveness of such an acquisition function depends critically on accurately measuring relationships between samples in the underlying data geometry. The quality of coreset selection depends on the distance metric used to characterize the data geometry. In highdimensional spaces, conventional Euclidean distances may fail to capture the intrinsic low-dimensional structure of the data distribution, potentially leading to less representative sample selection.

To address this issue, we integrate data-adaptive, shortest-path metrics to more accurately capture the data geometry within a ProbCover approach for coreset selection. The idea is to construct a weighted graph associated with the dataset X and then compute shortestpath distances on the graph to capture the underlying structure of X . One particular class of densityweighted path metrics is the Fermat distance, where the edge weights are calculated using a power of the Euclidean distance. Fermat distance has been successfully applied to a range of problems, including semisupervised methods and clustering methods [11–15] and high-dimensional signal processing [16, 17].

## B. Contributions

The contributions of the paper are the following:

1) We propose PC-Fermat, a variant of ProbCover that replaces Euclidean coverage neighborhoods with Fermat distance-based neighborhoods. This provides a data-adaptive approach to constructing coverage graphs that accounts for the underlying geometry of the data.

2) We present an efficient procedure for constructing Fermat coverage graphs using a weighted Euclidean neighborhood graph and truncated shortest-path computations. We establish conditions under which this construction exactly preserves the Fermat neighborhoods defined on the complete graph.

3) We evaluate PC-Fermat across eight classification datasets and under varying labeling budgets. We compare its performance against ProbCover and several active learning methods.

The remainder of the article is organized as follows. In Section II, we present background information on Fermat distance and graph construction techniques. In Section III, we derive our proposed method. The results of the experiments on benchmark datasets and the discussion of the results are presented in Section IV. We gather some concluding remarks in Section V.

## II. BACKGROUND

In this section, we introduce the p-Fermat distance, which forms the basis of our proposed method. We then describe the construction of the Fermat coverage graph and present an efficient procedure for computing its neighborhoods. The notation used throughout the paper is summarized in Table I.

## A. Fermat Distance

Discrete Fermat distances can be viewed as classical shortest-path distances in a graph in which edge lengths are weighted according to a penalty parameter $p \in [ 1 , \infty )$

<table><tr><td>Variable</td><td>Description</td></tr><tr><td>X</td><td>dataset, vertex set (both data and covering graphs)</td></tr><tr><td> $\mathcal { L }$ </td><td>labeled set</td></tr><tr><td>U</td><td>unlabeled set</td></tr><tr><td>Q</td><td>query set</td></tr><tr><td> $n , d$ </td><td>Number of data points and feature dimension</td></tr><tr><td> $C$ </td><td>Number of classes</td></tr><tr><td>b p</td><td>Labeling budget</td></tr><tr><td> $\dot { \ell } _ { p } ( x , y , \mathcal { X } )$ </td><td>Power applied to edge lengths p-weighted graph Fermat distance between x and y</td></tr><tr><td>€</td><td>Euclidean ball radius for connecting geometric graph</td></tr><tr><td>r</td><td>Fermat ball radius for covering graph</td></tr><tr><td></td><td></td></tr></table>

TABLE I: Notation Table

![](images/6f9cd44a1814bf57d76614a172b284901f72caa8bdd9e510a31cd98050c10ef1.jpg)  
Fig. 1: Fermat paths between two points at opposite ends of a moonshaped dataset for different p values.

Definition 1 (Fermat Distance). For $p \in [ 1 , \infty ) , x , y \in$ $\mathbb { R } ^ { d } .$ , and some finite set $\boldsymbol { \mathcal { X } } \subset \mathbb { R } ^ { D }$ , the (discrete) unrooted $p \textmd { - }$ weighted Fermat distance from x to y is:

$$
\ell _ { p } ( x , y , \mathcal { X } ) = \operatorname* { m i n } _ { \pi = \{ x _ { i } \} _ { i = 1 } ^ { T } } \bigg ( \sum _ { i = 1 } ^ { T - 1 } \| x _ { i } - x _ { i + 1 } \| ^ { p } \bigg ) ,\tag{1}
$$

where $\pi \subset \mathcal { X }$ is a path of dataset points with $x _ { 1 } = x$ and $x _ { T } = y$ and $\| \cdot \|$ is the Euclidean norm.

The quantity $\ell _ { p } ( \cdot , \cdot , \mathcal { X } )$ can be interpreted as the shortestpath distance on the complete graph induced by ${ \mathcal { X } } ,$ where each edge is assigned the weight $\| { x } _ { i } - { x } _ { j } \| ^ { p }$ . When $p \ = \ 1 .$ , we recover the Euclidean distance. As $p$ increases, the largest edge lengths along a path contribute heavily to the distance defined in (1). This results in paths through higher-density regions with shorter edge lengths being promoted. When $p  \infty ,$

$$
\ell _ { \infty } ( x , y ) = \operatorname* { m i n } _ { \pi = \{ x _ { i } \} _ { i = 1 } ^ { T } } \operatorname* { m a x } _ { i = 1 , \cdots T - 1 } \| x _ { i } - x _ { i + 1 } \| ,
$$

which is called the longest-leg path distance [12]. Outside the extreme values of $p = 1$ and $p = \infty , \ell _ { p }$ balances taking a path through regions of high density and taking a “short”, Euclidean distance path.

To build intuition for the Fermat distance, Figure 1 shows Fermat paths between two points at opposite ends of a moon-shaped cluster for $p = 1 , 2 , 5$ . For $p = 1$ , the path is the straight segment between the endpoints, and the Fermat distance reduces to the Euclidean distance due to the triangle inequality. For larger p, long steps are heavily penalized, so the path instead takes many short steps through dense regions and follows the shape of the data distribution.

Toy Example. It is important to note here how this change to a data-adaptive metric facilitates our approach. By changing the notion of distance between data points $x , y \in { \mathcal { X } } ,$ , we capture a fundamentally different geometric structure represented in the dataset. Consider Figure 2, which shows a two-moon dataset consisting of $n = 6 0 0$ points in $\mathbb { R } ^ { 2 }$ sampled from two interleaving halfcircles, one per class, with Gaussian noise of standard deviation 0.09. We select a point at the top of the lower moon, which lies inside the arc of the upper moon, and compare its Euclidean ball with a Fermat ball.

The Fermat metric here considers points in a particular moon to be closer to each other than points in different moons that are close in Euclidean distance. In other words, the Fermat ball stretches along the dense region of the lower moon since crossing the sparse gap is expensive under the Fermat distance. This illustrates how the Fermat ball adapts to the shape of the data distribution and captures more interesting clustering geometries than plain Euclidean distances can. We exploit this different geometry to improve the ProbCover approach for coreset selection in active learning.

## B. Graph Construction

Consider a dataset $\mathcal { X } = \{ x _ { 1 } , x _ { 2 } , \dotsc , x _ { n } \} \subset \mathbb { R } ^ { d } .$ , where each point is represented either by its original features or by a learned embedding. We first construct a weighted ϵ-neighborhood graph $G _ { \epsilon } = ( \boldsymbol { \mathcal { X } } , \mathcal { E } _ { \epsilon } )$ , where

$$
\mathcal { E } _ { \epsilon } = \{ ( x _ { i } , x _ { j } ) : \| x _ { i } - x _ { j } \| _ { 2 } ^ { p } \leq \epsilon ^ { p } , ~ i \neq j \} .
$$

The parameter $\epsilon > 0$ controls the sparsity and connectivity of the graph. A small ϵ yields a sparse graph that captures local relationships but may result in disconnected components, whereas a large ϵ captures broader neighborhood structure at the risk of introducing spurious connections.

In ProbCover [9], unweighted $G _ { \epsilon }$ serves directly as the coverage graph. In contrast, our proposed method uses weighted $G _ { \epsilon }$ to construct a coverage graph based on Fermat distances. Specifically, we define the Fermat coverage graph $G _ { F } = ( \lambda , \mathbf { \hat { \mathcal { E } } } _ { F } )$ , where

$$
\xi _ { F } = \{ ( x _ { i } , x _ { j } ) : \ell _ { p } ( x _ { i } , x _ { j } ) \leq r , \ i \neq j \} .
$$

Here, $\ell _ { p }$ denotes the Fermat distance defined in Section 1 and $r > 0$ denotes the Fermat coverage radius, measured on the p-powered distance scale induced by the edge weights $\| \hat { \boldsymbol { x } } _ { i } - \boldsymbol { x } _ { j } \| _ { 2 } ^ { p }$ . The ϵ-neighborhood graph $G _ { \epsilon }$ and the Fermat coverage graph $G _ { F }$ are closely related. In particular, when $r \geq \epsilon ^ { p } ,$ , every edge of $G _ { \epsilon }$ is also an edge of $G _ { F , }$ , and the two graphs have the same connected components (Lemma 1 in Appendix). We describe an efficient procedure for constructing $G _ { F }$ in the following subsection.

## C. Efficient Construction of the Fermat Coverage Graph

The Fermat distance can be computed using Dijkstra’s algorithm, but applying it to a complete, weighted Euclidean distance graph over X is computationally expensive. Constructing the Fermat coverage graph $G _ { F } ,$ however, only requires identifying points within a prescribed

![](images/2665f3b4148dbdebf4fa5fd496cdb48d9f8db424d883524ffba9c34c06080fe0.jpg)  
Fig. 2: (a) The two-moon dataset with the chosen center (star) at the tip of the lower moon. (b) Euclidean ball with $\epsilon = 0 . 3 ;$ the dashed circle marks its radius. (c) Fermat ball with $p = 3$ and $r = 0 . 1 \mathrm { 5 } ^ { p }$ . The Euclidean ball crosses the gap into the upper moon, while the Fermat ball, despite containing more points, follows the dense region of the lower moon and contains points that are mostly of the same class.

Fermat distance threshold based on the parameter r. Specifically, for each $x \in \mathcal { X } ,$ , we consider the Fermat ball

$$
B _ { F } ( x ; r ) = \{ x ^ { \prime } \in \mathcal { X } : \ell _ { p } ( x , x ^ { \prime } ) \leq r \} ,
$$

where $r > 0$ denotes the Fermat coverage radius, measured on the p-power distance scale.

We exploit this observation in two steps. First, rather than using a (dense) complete, weighted Euclidean distance graph, we perform shortest-path computations on the (sparser) weighted ϵ-neighborhood graph $G _ { \epsilon }$ Provided that $\begin{array} { r l } { r } & { { } \leq \quad \epsilon ^ { p } } \end{array}$ , this restriction preserves the Fermat balls exactly, as established in Lemma 2 in the Appendix. Even if $r \geq \epsilon ^ { p } ,$ , we maintain that distances in the geometric neighborhood graph more closely align with distances intrinsic to the dataset than those in the fully connected graph, assuming that ϵ is so large that the data sample resolves the data manifold at length scale ϵ.

Second, we run Dijkstra’s algorithm from each vertex with a distance cutoff of r. Each search terminates once the smallest tentative distance exceeds this threshold. The vertices reached within the cutoff form exactly the Fermat ball $B _ { F } ( x ; r )$ .

These Fermat balls determine the edge set $\mathcal { E } _ { F }$ , allowing us to construct the Fermat coverage graph without computing all pairwise Fermat distances.

## III. PROPOSED METHOD

Let $\mathcal { X } = \{ x _ { 1 } , x _ { 2 } , \cdot \cdot \cdot x _ { n } \} \in \mathbb { R } ^ { d }$ be a dataset, represented either by raw features or by learned embeddings. The labeled set ${ \mathcal { L } } \subseteq { \mathcal { X } }$ contains the points whose labels $y _ { i } =$ $y ( x _ { i } ) \in \{ 1 , 2 , \cdot \cdot \cdot , C \}$ are known. The unlabeled set is $\lambda =$ $\mathcal { X } \backslash \mathcal { L }$ . Given a budget $b ,$ the goal is to select a query set $\mathcal { Q } \subseteq \mathcal { U }$ (with $| \mathscr { Q } | = b )$ whose points will be sent for annotation.

Our method builds on the ProbCover algorithm [9], replacing its Euclidean distance with the Fermat distance. We summarize the procedure below in three parts.

a) Graph construction: We first construct the Fermat coverage graph $G _ { F }$ following the procedure described in Section II-B and Section $\mathrm { I I - C } .$ Specifically, we compute the Fermat ball $B _ { F } ( x ; r )$ for each $x \in \mathcal { X }$ using truncated

Dijkstra’s algorithm on the weighted ϵ-neighborhood graph $G _ { \epsilon }$ . The coverage graph $G _ { F }$ is then formed by connecting each point x to all other points in $B _ { F } ( x ; r )$ b) Sampling: We initialize the coverage graph by removing all incoming edges to the vertices covered by the labeled set, i.e. to every $x ^ { \prime } \in B _ { F } ( x ; r )$ for $x \in { \mathcal { L } } ,$ so that already covered points no longer contribute to the selection. We then perform b iterations. In each iteration, we select the vertex $x _ { \mathrm { m a x } }$ with the highest out-degree for annotation, remove all incoming edges to $x _ { \mathrm { m a x } }$ and its neighboring vertices, and add $x _ { \mathrm { m a x } }$ to the query set Q. The procedure terminates once the query budget is exhausted, i.e., when $| \mathcal { Q } | = b$

c) Prediction: After the query budget is exhausted, we augment the labeled set with the queried samples and train a 1-nearest neighbor (1-NN) classifier on the updated labeled set. We evaluate classification performance using a 1-NN classifier based on Euclidean distance to ensure consistency across all comparison methods. Additionally, Euclidean 1-NN classification is computationally less expensive than a Fermat distance-based 1- NN classifier. The use of a simple classifier is predicated on the assumption of a data-sparse setting, which does not allow for the training of advanced models like neural networks.

An overview of the PC-Fermat framework is illustrated in Figure 3.

## IV. EMPIRICAL RESULTS

In this section, we report a set of empirical results, comparing our method with other active learning strategies. First, we detail the datasets used in the experiments.

## A. Datasets

• The COIL-20 dataset [18] contains 1440 normalized images of 20 objects (i.e., classes). The objects were placed on a motorized turntable. With a fixed camera, the turntable was rotated through 360 degrees to capture different poses every 5 degrees. There are a total of 72 images per object. The dataset consists of 1,440 data points and has 1,024 attributes.

![](images/ecb753a5519e1efd6ee6f5cc79e1bf45fda8a883472177cdd3b8ff06a4fd3698.jpg)  
Fig. 3: Flowchart of PC-Fermat framework. Given an input dataset, we construct a weighted Euclidean ϵ-neighborhood graph and obtain the Fermat coverage graph using shortest-path computations. A greedy coverage-based selection procedure then identifies b representative samples Finally, predictions are made using a 1-NN classifier, and classification accuracy is evaluated across different labeling budgets.

```latex
Algorithm 1 ProbCover on Fermat Distance Graph
Input: Unlabeled set $\overline { { \boldsymbol { u } , } }$ labeled set ${ \overline { { \mathcal { L } } } } ,$ budget $b ,$ Eu
clidean neighborhood radius $\epsilon ,$ Fermat coverage
radius $r ,$ exponent $p$
Output: Query set Q
X ← learned embeddings or raw features of $\mathcal { U } \cup \mathcal { L }$
// Graph construction
Construct the ϵ-neighborhood weighted graph $\begin{array} { r l } { G _ { \epsilon } } & { { } = } \end{array}$
(X, E ), where $\mathcal { E } _ { \epsilon } \overset { \smile } { = } \{ ( x , x ^ { \prime } ) : \| x - x ^ { \prime } \| _ { 2 } ^ { p } \leq \epsilon ^ { p } , ~ x \neq x ^ { \prime } \} ,$
with edge weights $\| x - x ^ { \prime } \| _ { 2 } ^ { p }$
for all $x \in \mathcal { X }$ do
$B _ { F } ( x ; r ) \ \gets$ vertices reached by Dijkstra on the
weighted $G _ { \epsilon }$ from x with distance cutoff r
end
Construct the Fermat coverage graph G<sub>F</sub> = (X , E<sub>F</sub>),
where $\mathcal { E } _ { F } = \{ ( x , x ^ { \prime } ) : x ^ { \prime } \in \check { B } _ { F } ( x ; r ) , \ \overset { \cdot } { x ^ { \prime } } \neq x \}$
// Greedy coverage-based sample selection
for all $c \in { \mathcal { L } }$ do
Remove incoming edges to vertices covered by $c { : }$
$\mathcal { E } _ { F }  \mathcal { E } _ { F } \setminus \{ ( x ^ { \prime } , x \overset {  } { ) } \in \overset {  } { \mathcal { E } _ { F } } : x \in \mathcal { B } _ { F } ( c ; r ) \}$
end
Q ← ∅
for i ← 1 to b do
c ← arg max outdeg<sub>G</sub>(u)
u∈U
Q ← Q ∪ {c}
Remove incoming edges to vertices covered by c:
$\mathcal { E } _ { F }  \mathcal { E } _ { F } \setminus \{ ( x ^ { \prime } , x ) \in \bar { \mathcal { E } _ { F } } : ( c , x ) \in \mathcal { E } _ { F } \}$
end
return Q
```

• The Opt-Digits dataset [19] is a multiclass dataset of 5,620 handwritten digits. Each handwritten digit is recorded as an $8 \times 8$ matrix. Thus, each element has 64 attributes. There are 10 classes.

• The Landsat dataset [20] contains 6,435 elements, each with multi-spectral values of pixels in 3x3 neighborhoods in a satellite image. For this dataset, there are six classes, and the classification of an element is derived from the ground truth label of the central pixel. Each of the data elements contains the pixel values in the four spectral bands of each pixel in the 3×3 neighborhood. Therefore, each feature vector consists of 36 values. The six classes are red soil, cotton crop, gray soil, damp gray soil, soil with vegetation stubble, and very damp gray soil.

• The USPS dataset [21] contains 9,298 grayscale images of handwritten digits, each with a resolution of 16 × 16 pixels. The images were obtained by automatically scanning envelopes by the United States Postal Service in Buffalo, New York. The grayscale images are centered and normalized and show various font styles. There are 10 classes; each element is an image of a handwritten digit “0” through “9”.

• The Jasper Ridge dataset [22] is widely used for hyperspectral image classification. The original dataset consists of a 512 × 614 pixel image with 224 spectral channels. For our experiments, we consider a 100 × 100 subimage spanning 198 channels. Due to dense water vapor and atmospheric effects, the channels 1–3, 108–112, 154–166 and 220–224 are removed.

• The Pendigits dataset [23] is a collection of penbased digit samples. It has a total of 10,992 samples with 16 features and 10 target classes (digits 0-9). The database is created by collecting 250 samples from 44 writers.

• The MNIST dataset [24] is a handwritten digit classification dataset consisting of 70,000 grayscale images of handwritten digits from 0 to 9. Each image has a resolution of $2 8 \times 2 8$ pixels and is flattened into a 784-dimensional feature vector. We use a variational autoencoder (VAE) [25] to learn a lower-dimensional latent representation of the images, which is then used as the embedding for our experiments. Specifically, each 784-dimensional input is mapped to a 20-dimensional latent representation.

TABLE II: Datasets used in the experiments.
<table><tr><td>Dataset</td><td>Samples</td><td>Features</td><td>Type</td><td>Classes</td></tr><tr><td>Coil-20 Opt-Digits</td><td>1,440</td><td>1,024</td><td>Raw</td><td>20</td></tr><tr><td rowspan="4">Landsat USPS Jasper</td><td>5,620</td><td>64</td><td>Raw</td><td>10</td></tr><tr><td>6,435</td><td>36</td><td>Raw</td><td>6</td></tr><tr><td>9,298</td><td>256 198</td><td>Raw</td><td>10</td></tr><tr><td>10,000</td><td></td><td>Raw</td><td>4</td></tr><tr><td>Pendigits</td><td>10,992</td><td>16</td><td>Raw</td><td>10</td></tr><tr><td>MNIST</td><td>70,000</td><td>20</td><td>Embedd.</td><td>10</td></tr><tr><td>Fashion-MNIST</td><td>70,000</td><td>30</td><td>Embedd.</td><td>10</td></tr></table>

• The Fashion-MNIST dataset [26] consists of 70, 000 images with 10 classes. Each image has a 784- dimensional feature vector. Similar to MNIST, we use a variational autoencoder (VAE) [25] to learn a lower-dimensional latent representation of the images, which is then used as the embedding for our experiments. Specifically, each 784-dimensional input is mapped to a 30-dimensional latent representation.

The details of the datasets are outlined in Table II.

## B. Comparison to Other Methods

In this section, we present empirical results comparing our proposed method, ProbCover with a Fermat distance graph (hereafter referred to as PC-Fermat), with several existing active learning methods in the very low-budget regime. In particular, we compare PC-Fermat against Random Sampling, Margin Sampling, KMeans++ [27], ProbCover [9] (hereafter referred to as PC-Euclidean), and TypiClust [10]. The results of the experiments are shown in Figure 4. For a fair comparison, we use a 1-Nearest Neighbor (NN) classifier as the underlying classifier model for all active learning strategies. The 1- NN classifier uses Euclidean distance for all methods.

Next, we give a brief overview of our comparison methods. In particular, Random selects unlabeled samples uniformly at random. Margin is an uncertainty-based method that queries samples with the smallest prediction margin between the top two classes. Moreover, KMeans++ [27] is a seeding algorithm that starts by first selecting one point at random, then iteratively choosing subsequent centers with probabilities proportional to their squared distance from the nearest existing center. This iterative process leads to more diverse and effective seeding than random initialization. ProbCover [9] (PC-Euclidean) is an active learning method for low-budget regimes and it seeks to maximize probability coverage. Lastly, TypiClust [10] clusters the feature space for diversity and queries the typical examples in each cluster that favor the selection of representative examples.

For the implementation of ProbCover (PC-Euclidean) and TypiClust, we utilize the code obtained from the authors’ repository. For KMeans++, we use the implementation available in the scikit-learn Python library [28]. Our Fermat ProbCover (PC-Fermat) implementation is based on the original ProbCover codebase with the necessary modifications to support the Fermat distance computation.

## C. Hyperparameter Selection

In this section, we describe the hyperparameter selection procedure used in our experiments. One of the important hyperparameters of our method is the radius of the (Euclidean or Fermat) ball for ProbCover, denoted by r. The parameter r directly controls the trade-off between coverage and local purity. A larger value of r provides higher coverage but results in less pure neighborhoods. In addition, most of the datasets considered in our experiments do not have pre-computed embeddings. Therefore, we carefully tune r for each dataset and report the selected values in the supplementary materials.

In the case of PC-Fermat with Fermat power p, we couple $\boldsymbol { r } ~ = ~ \epsilon ^ { p }$ in light of Lemma 2 and thus seek a single parameter ϵ for which the majority of data points are within a few large connected components to ensure that the geometry of the data clusters is well-resolved. We illustrate a typical and a non-typical situation in the growth of large connected components in Figure 5 as ϵ increases using the COIL-20 and Landsat datasets, respectively.

## D. Performance and Discussion

For each dataset, all methods except Margin sampling are initialized with an empty labeled set. Given a labeling budget b, we query b data points for their labels. For Margin sampling, we initialize the labeled set randomly with one label per class. Our proposed method PC-Fermat uses Fermat distance for graph construction, while PC-Euclidean uses Euclidean distance for graph construction. Figure 4 reports the classification accuracy of all evaluated methods across the datasets.

Overall, our proposed PC-Fermat achieves the most competitive performance across most datasets, outperforming the random sampling baseline and generally improving upon the competing methods. We observe in Figure 5 that the ϵ-neighborhood graph for COIL-20 has multiple connected components of large comparable sizes for a wide range of values of ϵ, suggesting a set of several well-separated clusters. In this specific setting, Margin outperforms our PC-Fermat and all other representation-based methods, suggesting an early transition from an exploration-dominated to an exploitationdominated regime. In the more typical case of the Landsat dataset (Figure 5, right), data cannot be separated as neatly into Euclidean clusters.

![](images/a47fe46de6966108e0bfee6a9cd745fd0b67aaaa398f232be250f660be90a3a5.jpg)

![](images/30f9c806998f4b128c10110e17edf7b8cb1a0c5be41541f694e73900bf848a3b.jpg)  
(b)

(a)  
![](images/b7a2f3fde31b7ee4505a958639da222ceb20aaacd355b539cbc07d7016648bde.jpg)

![](images/ae40712b032973e7f8ecd9f2d831c68ba621aaaf19f4f13d0670e5538367a200.jpg)

(c)  
![](images/e0cc595f116e5464dd67e8349beac76bb124b9932c4fd569714432f5850a5a6e.jpg)

(d)  
![](images/701833a1a96496a92c9eee533ed69173c059b86485270d88d49dc7206166158d.jpg)

(e)  
![](images/8e4c0e6b7f0514d5ee97038c210008fc2dbe3b803c069626f04b431c1f3a961f.jpg)  
(g)

(f)  
![](images/a077092b6d0c4eb46b84f97896e8af835afd9f355ecb59044a11f40ac4635e8f.jpg)  
(h)  
Fig. 4: Accuracy vs. # of labeled points for each dataset: (a) COIL-20 (b) Opt-Digits (c) Landsat (d) USPS (e) Pendigits (f) Jasper (g) Fashion-MNIST (h) MNIST.

![](images/209955aa0aad625e177af9d658ff19648592885dbd86d18620b7b50a38e95468.jpg)  
(a)

![](images/4d45db343ad9f83b466297bb9c91bd54c517287b13fce65214ad6aa548bc1ce5.jpg)  
(b)  
Fig. 5: Evolution of largest connected components of the ϵ-neighborhood graph with increasing ϵ. Each colored band shows the size of one component; the black line shows their total size, and the red dashed line marks the total number of points. COIL-20 (left) contains several well-separated clusters, while Landsat (right) is dominated by a single connected component.

## V. CONCLUSION

In this paper, we introduce Fermat ProbCover (PC-Fermat), a coverage-based active learning method that integrates the Fermat distance, a density-aware metric, into the ProbCover framework. The proposed method is well suited to the low-budget regime, where only a few labeled examples are available, a common scenario for many applications since manual data labeling is frequently both time-consuming and cost-prohibitive. Experimental results on eight classification datasets under varying labeling budgets show that our method outperforms the Euclidean variant of ProbCover and other active learning methods on most of the datasets.

In terms of computational cost, random sampling and kmeans++ are significantly more lightweight algorithms than ProbCover, and PC-Fermat increases the computation cost further by requiring intrinsic distances to be computed, at least up to the threshold value of $\boldsymbol { r } ~ = ~ \boldsymbol { \epsilon } ^ { p } ,$ , at which point Dijkstra’s algorithm terminates early. However, in the examples considered above, it is the most consistently competitive algorithm and is recommended when the cost of data annotation vastly exceeds that of finding a representative subset; this is often the case in practical applications.

## VI. ACKNOWLEDGMENTS

The work of EM was partly supported by grant NSF DMS-2512644. The work of DD and SW was partly supported by grant NSF DMS-2424801.

## REFERENCES

[1] C. Chai, K. Jin, N. Tang, J. Fan, L. Qiao, Y. Wang, Y. Luo, Y. Yuan, and G. Wang, “Mitigating data scarcity in supervised machine learning through reinforcement learning guided data generation,” in IEEE International Conference on Data Engineering, 2024, pp. 3613–3626.

[2] G. Bhusal, E. Merkurjev, and G.-W. Wei, “Persistent Laplacian-enhanced algorithm for scarcely la-

beled data classification,” Machine Learning, vol. 113, no. 10, pp. 7267–7292, 2024.

[3] B. Settles, Active Learning. Morgan & Claypool Publishers LLC, 2012, vol. 6, no. 1. [Online]. Available: https://doi.org/10.2200/s00429ed1v01y 201207aim018

[4] H. Hardiman-Mostow, J. Mauro, A. Weihs, and A. L. Bertozzi, “Topology-aware active learning on graphs,” arXiv preprint arXiv:2510.25892, 2025.

[5] A. L. Bertozzi, X. Luo, A. M. Stuart, and K. C. Zygalakis, “Uncertainty quantification in graph-based classification of high dimensional data,” SIAM/ASA Journal on Uncertainty Quantification, vol. 6, no. 2, pp. 568–595, 2018.

[6] Y.-L. Qiao, C. X. Shi, C. Wang, H. Li, M. Haberland, X. Luo, A. M. Stuart, and A. L. Bertozzi, “Uncertainty quantification for semi-supervised multi-class classification in image processing and ego-motion analysis of body-worn videos,” Image Processing: Algorithms and Systems, vol. 2019, no. 11, pp. 264–1, 2019.

[7] K. Miller and J. Calder, “Poisson reweighted Laplacian uncertainty sampling for graph-based active learning,” SIAM Journal on Mathematics of Data Science, vol. 5, no. 4, pp. 1160–1190, 2023.

[8] G. Bhusal, K. Miller, and E. Merkurjev, “MALADY: Multiclass active learning with auction dynamics on graphs,” IEEE Transactions on Artificial Intelligence, vol. 7, pp. 385 – 398, 2025.

[9] O. Yehuda, A. Dekel, G. Hacohen, and D. Weinshall, “Active learning through a covering lens,” Advances in Neural Information Processing Systems, vol. 35, pp. 22 354–22 367, 2022.

[10] G. Hacohen, A. Dekel, and D. Weinshall, “Active learning on a budget: Opposite strategies suit high and low budgets,” Proceedings of the 39th International Conference on Machine Learning, vol. 162, pp. 8458–8483, 2022.

[11] A. Little, D. McKenzie, and J. M. Murphy, “Balancing geometry and density: Path distances on highdimensional data,” SIAM Journal on Mathematics of

Data Science, vol. 4, no. 1, pp. 72–99, 2022.

[12] A. Little, M. Maggioni, and J. M. Murphy, “Pathbased spectral clustering: Guarantees, robustness to outliers, and fast algorithms,” Journal of Machine Learning Research, vol. 21, no. 6, pp. 1–66, 2020.

[13] D. Mckenzie and S. Damelin, “Power weighted shortest paths for clustering Euclidean data,” Foundations of Data Science, vol. 1, no. 3, pp. 307–327, 2019.

[14] P. Vincent and Y. Bengio, “Density-sensitive metrics and kernels,” in Snowbird Learning Workshop, 2003.

[15] A. S. Bijral, N. Ratliff, and N. Srebro, “Semisupervised learning with density based distances,” Proceedings of the Twenty-Seventh Conference on Uncertainty in Artificial Intelligence, pp. 43–50, 2011.

[16] S. Zhang and J. M. Murphy, “Hyperspectral image clustering with spatially-regularized ultrametrics,” Remote Sensing, vol. 13, no. 5, p. 955, 2021.

[17] A. Manousidaki, A. Little, and Y. Xie, “Clustering and visualization of single-cell rna-seq data using path metrics,” PLOS Computational Biology, vol. 20, no. 5, p. e1012014, 2024.

[18] COIL-20, “COIL-20 Data $\mathrm { S e t } , { ^ { \prime \prime } }$ Technical report CUCS-005-96, 1996.

[19] OptDigits, “Optical Recognition of Handwritten Digits,” UCI Machine Learning Repository, 1998, DOI: https://doi.org/10.24432/C50P49.

[20] “Landsat data set,” https://archive.ics.uci.edu/da taset/146/statlog+landsat+satellite, 2024.

[21] “USPS data set,” https://www.csie.ntu.edu.tw/ <sup>∼</sup>cjlin/libsvmtools/datasets/multiclass.html#usps, 2024.

[22] “Jasper data set,” https://lesun.weebly.com/hyp erspectral-data-set.html, 2024.

[23] E. Alpaydin and F. Alimoglu, “Pen-Based Recognition of Handwritten Digits,” UCI Machine Learning Repository, 1996, DOI: https://doi.org/10.24432/C5MG6K.

[24] Y. LeCun, C. Cortes, and C. Burges, “MNIST handwritten digit database,” AT&T Labs [Online], vol. 2, 2010.

[25] D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” International Conference on Learning Representations, 2014.

[26] H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms,” arXiv preprint arXiv:1708.07747, 2017.

[27] D. Arthur, S. Vassilvitskii et $a l . ,$ “k-means++: The advantages of careful seeding,” in Soda, vol. 7, 2007, pp. 1027–1035.

[28] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

## APPENDIX

In this section, we establish the two properties of the Fermat coverage graph. The first result characterizes its relationship with the underlying Euclidean neighborhood graph in terms of edge inclusion and connectivity. The second result provides conditions under which computing Fermat distances on the Euclidean neighborhood graph preserves the Fermat distance computed from the complete graph.

Lemma 1. If we construct an ϵ-neighborhood graph for the geometry graph and then build the coverage graph based on pairwise p-Fermat distances in the geometry graph, then $i f$ $r \geq \epsilon ^ { p }$

• every edge of the geometry graph is also an edge of the coverage graph and

• both graphs have the same connected components.

Proof: Let $G _ { \epsilon } = ( \boldsymbol { \mathcal { X } } , \mathcal { E } _ { \epsilon } )$ be the ϵ-neighborhood graph for the geometry graph and let $G _ { F } \doteq \left( \mathcal { X } , \mathcal { E } _ { F } \right)$ be the coverage graph based on p-Fermat distances on $G _ { F } .$ Consider any edge $( x , y ) \in \mathcal { E } _ { \epsilon }$ . By definition of $\epsilon -$ neighborhood graph, the distance between the vertices x and y satisfies

$$
d ( x , y ) \leq \epsilon .
$$

For $p \in [ 1 , \infty )$ , and vertices $x , y \in { \mathcal { X } } ,$ let $\ell _ { p } ^ { \epsilon } ( x , y )$ denote the p-Fermat distance between the vertices x and y over some path. If $r \geq \epsilon ^ { p }$ we have the following relation

$$
\ell _ { p } ^ { \epsilon } ( x , y ) \leq d ( x , y ) ^ { p } \leq \epsilon ^ { p } \leq r .
$$

This gives $\ell _ { p } ^ { \epsilon } ( x , y ) \leq r$ . So by the definition of a neighborhood graph, this implies that $( x , y ) \in ~ \mathcal { E } _ { F }$ . Hence, every edge of the geometry graph is also an edge of the coverage graph.

Because $\mathcal { E } _ { \epsilon } \subseteq \mathcal { E } _ { F } .$ , any path in $G _ { \epsilon }$ is also a path in $G _ { F }$ . This means each connected component of the geometry graph is contained in a connected component of the coverage graph. It remains to show that the coverage graph does not merge two distinct connected components of the geometry graph.

To this end, consider an edge $( x , y ) \in \mathcal { E } _ { F }$ . By definition of the coverage graph, there exists a path in $G _ { \epsilon }$ connecting x and y with $\ell _ { p } ^ { \epsilon } \leq r$ . In particular, such a path certifies that x and y lie in the same connected component of $G _ { \epsilon }$ . Hence, there is no edge in the coverage graph that connects vertices from two different components of the geometry graph. Therefore, both graphs have the same connected components.

Lemma 2. Let $\epsilon > 0 , p > 1$ , and $0 ~ < ~ r ~ \le ~ \epsilon ^ { p }$ . Denote by $\ell _ { p }$ and $\ell _ { p } ^ { \epsilon }$ the Fermat distances computed on the complete weighted graph and the weighted ϵ-neighborhood graph $G _ { \epsilon } ,$ respectively. Then, for any $x , y \in { \mathcal { X } }$

$$
\ell _ { p } ( x , y ) \leq r \quad \Longleftrightarrow \quad \ell _ { p } ^ { \epsilon } ( x , y ) \leq r .
$$

Furthermore, whenever either inequality holds,

$$
\ell _ { p } ( x , y ) = \ell _ { p } ^ { \epsilon } ( x , y ) .
$$

Proof:

Since $G _ { \epsilon }$ is a subgraph of the complete graph, we have

$$
\ell _ { p } ( x , y ) \leq \ell _ { p } ^ { \epsilon } ( x , y ) .
$$

Thus, if $\ell _ { p } ^ { \epsilon } ( x , y ) \leq r ,$ then $\ell _ { p } ( x , y ) \leq r .$

Conversely, suppose $\ell _ { p } ( x , y ) \leq r ,$ and let $\pi ^ { * }$ be a shortest path between x and y in the complete weighted graph. Then

$$
\sum _ { \left( x _ { u } , x _ { v } \right) \in \pi ^ { * } } \| x _ { u } - x _ { v } \| _ { 2 } ^ { p } \leq r .
$$

Since all edge weights are nonnegative, every edge along $\pi ^ { * }$ satisfies

$$
\| x _ { u } - x _ { v } \| _ { 2 } ^ { p } \leq r \leq \epsilon ^ { p } .
$$

Since $p > 1 .$ , it follows that $\| x _ { u } - x _ { v } \| _ { 2 } \leq \epsilon .$ . Thus, every edge of $\pi ^ { * }$ belongs to $G _ { \epsilon } ,$ and hence

Combining the two inequalities yields

$$
\ell _ { p } ^ { \epsilon } ( x , x ) \leq \ell _ { p } ( x , x ) .
$$

which completes the proof.

$$
\ell _ { p } ^ { \epsilon } ( x , y ) = \ell _ { p } ( x , y ) \leq r ,
$$