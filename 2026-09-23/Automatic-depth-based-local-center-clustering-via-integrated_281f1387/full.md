# Automatic depth-based local center clustering via β-integrated local depth and adaptive grouping

Siyi Wang, Alexandre Leblanc, Paul D. McNicholas

## Abstract

Clustering is an unsupervised learning technique that partitions unlabeled data into groups. Most existing methods require user-specified parameters, such as the number of clusters or neighborhood size. Conversely, we propose automatic depth-based local center clustering (A-DLCC), a fully data-driven method that eliminates numerical parameter tuning. A-DLCC uses the β-integrated local depth to identify stable exemplars, points consistently central across multiple locality levels, termed local centers, which are ranked by their representativeness. Each local center induces a group of similar points, with group-level similarity measured by a proposed nonparametric metric called group-level local similarity. To guide merging, we incorporate the bottleneck path idea from graph theory, which forms the basis of our adaptive merging criterion. Based on this criterion, we design a single agglomeration rule in which a group is either absorbed by a neighbor it reaches better than itself or bonded to a neighbor that both sides find more reachable than their own background, every merge being additionally required to be carried by a contact stronger than a configuration-model null expects. The rule automatically estimates the number of clusters and decides when to stop merging. Experiments on synthetic and real data show that A-DLCC produces interpretable clustering results without parameter tuning.

Keywords: statistical depth; local depth; automatic clustering; unsupervised learning; exemplar

## 1. Introduction

Clustering is a central task in unsupervised learning, with applications in representation learning [1], pattern recognition [2], image analysis [3] and other natural sciences [4]. The primary objective of clustering is to produce meaningful and interpretable partitions of data without annotated class labels. Clustering is therefore a standard tool for data analysis, especially when little or no prior knowledge about the data is available.

Many clustering algorithms have been proposed, ranging from classic methods such as Kmeans [5], hierarchical clustering [6], density-based approaches [7, 8] and Gaussian mixture models (GMM) [9], to more recent approaches based on graph theory [10] and manifold learning [11]. As real-world data become larger and more heterogeneous [12], many algorithms require tuning multiple parameters or careful model selection to achieve satisfactory performance.

In practice, model and parameter selection are often performed by running the clustering algorithm repeatedly with different settings, then choosing the best result according to some criterion, such as silhouette width [13] for Kmeans or the Bayesian information criterion (BIC) [14] for model-based clustering. However, methods that can estimate the number of clusters automatically and deliver high-quality results without the need for parameter selection remain relatively limited.

## 1.1. Literature review

Many approaches have been proposed for parameter-free or adaptive clustering. Density-based clustering algorithms are often considered adaptive, as they do not require a pre-specified number of clusters and can detect clusters of arbitrary shape and size. However, they typically depend on other parameters, such as neighborhood size or density thresholds. Mean-shift [15] is a classic example, which iteratively shifts data points toward regions of higher density, effectively discovering clusters without requiring K as input. However, mean-shift relies on a kernel bandwidth parameter, which critically affects its performance. Recent developments have focused on improving efficiency and sensitivity to parameter choices, for example by using k-nearest neighbor (kNN) density estimates instead of fixed bandwidths, as in NN-robust mean-shift (NN-RMS) [16] and NN-blurring mean-shift [17]. Nevertheless, mean-shift-based methods still require users to specify parameters that control density estimation.

FINCH [18] is a representative example of hierarchical clustering that aims to produce clustering results automatically. At each iteration, it merges all mutually closest groups (or those connected by a path), generating a hierarchy of clusterings at multiple levels of granularity without requiring any parameter input. However, the final selection from the resulting hierarchy still depends on user decisions or post hoc evaluation.

For methods that rely on interpretable parameters, such as the number of clusters K, rule-of-thumb heuristics are commonly employed. The eigengap heuristic in spectral clustering [19] is a well-known example, but its reliability diminishes for non-Gaussian cluster structures, as demonstrated by John et al. [20]. To address this, they proposed the spectrum method, introducing the multimodality gap for selecting K in non-Gaussian clusters, and further enhancing the process by employing a density-aware kernel and GMM on the eigenvector matrix instead of Kmeans. Another direction in automatic spectral clustering focuses on adaptively selecting the affinity matrix, which can also be regarded as a key parameter. For example, Fan et al. [21] proposed using the relative eigengap or Bayesian optimization to automatically select both the affinity matrix and its hyperparameters from candidate options. However, their approach still requires the number of clusters to be specified in advance, and thus cannot simultaneously determine both the optimal K and the affinity matrix.

A theoretically principled approach to automatic clustering is to define an objective function and seek the partition that optimizes it. Most methods in this category are extensions of Kmeans, accepting split or merge operations that most improve the objective in each iteration. Early examples include X-means [22], which optimizes the BIC but still requires specifying a range of cluster numbers in advance. More recent developments address this limitation. For instance, unsupervised Kmeans (U-Kmeans) [23] maximizes an entropy-based objective via an expectation-maximization-like algorithm, with the estimated number of clusters stabilizing over iterations. K<sup>∗</sup>means [24] formulates the objective using the minimum description length principle. This approach, especially when combined with UMAP [25] for dimensional reduction, has performed well in estimating K even for large datasets. However, these criteria are generally most effective for clusters with convex structure, and there is still a lack of widely applicable and computationally efficient objective function that generalizes to more complex cluster shapes.

## 1.2. Motivation and contribution

Despite various efforts toward automatic clustering, most existing methods still require parameter tuning, user intervention, or rely on strong assumptions about cluster structure—often reflecting the limits of the clustering algorithms themselves. Depth-based local center clustering (DLCC) [26] is a recently proposed method that uses statistical depth to address a broad range of clustering challenges, including non-convex shapes, unbalanced sizes, overlapping clusters, and high-dimensional data. However, DLCC itself requires the selection of a neighborhood size and similarity threshold to determine and group local centers (i.e., points that are locally central within a subset of the data), which limits its usability in practice without prior knowledge. This motivates the development of a fully automatic version of DLCC that removes assumptions on cluster size and number. The new method keeps the flexibility of DLCC and removes the need for tuning parameters.

Several obstacles arise when attempting to remove parameters from DLCC. The neighborhood size parameter determines definitions of both neighborhoods and local centers, and thus influences the similarity between neighborhoods as well. The threshold parameter governs how local centers are grouped. Eliminating parameter tuning thus requires to solve the following main challenges: first, how to identify exemplars (local centers) without relying on a fixed parameter for neighborhood size; second, how to define a nonparametric measure of similarity between these exemplars, focusing on the similarity between their associated point groups rather than between individual points; and third, how to design an algorithm that can automatically determine the number of clusters or provide an appropriate stopping rule for merging. To address these challenges, we propose automatic depth-based local center clustering (A-DLCC). Our approach consists of the following main contributions.

• Building on β-integrated local depth (β-ILD; [27]), we introduce a parameter-free method for defining exemplars—points that consistently occupy central positions within local neighborhoods of varying size. We also establish a representativeness ranking, where more representative points are those most often identified as locally central, particularly in larger neighborhoods. This ranking may have broader applications, such as centroid initialization in other clustering methods.

• Inspired by partitioned local depth (PaLD; [28]), we propose a nonparametric measure for quantifying similarity between disjoint groups of points, termed group-level local similarity (GLS). This measure inherits desirable properties from PaLD, such as naturally assigning zero similarity to completely separated groups and invariance under similarity transformations (i.e., rotation, dilation, and shift).

• We define intra-group reachability, inter-group reachability, and the relative reachability ratio, and design an adaptive merging criterion based on these measures to guide group aggregation. The criterion is applied through one agglomeration rule with two directional modes of merging (absorption and bonding), a null-model test of the contact between the groups on the depth graph, and a final reconsideration of small groups. The same rule covers both the connected-shape and the well-separated scenarios.

These contributions may also be useful for exemplar selection, group similarity measurement, and the design of merging criteria in other settings.

The remainder of the paper is organized as follows. Section 2 reviews essential concepts, including spatial depth, β-ILD, and DLCC. Section 3 presents our approach for defining local centers and point representativeness using β-ILD. Section 4 details the A-DLCC algorithm, including group-level similarity, adaptive merging, and other modifications from the original DLCC for a fully automatic framework. Section 5 demonstrates that A-DLCC achieves satisfactory and interpretable results on a wide variety of both synthetic and real datasets. Finally, Section 6 concludes the paper and outlines potential future directions.

## 2. Preliminaries

We review the concepts from data depth used in A-DLCC.

## 2.1. Local depth and integrated local depth

A depth function provides a center-outward ordering of data points and generalizes univariate order statistics such as the median, quantiles, and ranks to higher dimensions [29]. In clustering, early studies mainly used data depth as an auxiliary tool—computing depth within each cluster to refine partitions. More recently, local depth concepts have been used to define exemplars directly. For instance, Francisci et al. [30] adopt τ - local depth [31] for mean-shift-like clustering, and DLCC [26] uses $\beta -$ local depth (β-LD) [32] to construct the similarity matrix and define local centers.

Figure 1 illustrates the construction of sample β-LD. Given a point $\mathbf { x } _ { \mathrm { 0 } }$ and a dataset X with n observations, we first create the reflected set $\mathbf { X } _ { R \mathbf { x } _ { 0 } } = \{ \mathbf { X } \cup 2 \mathbf { x } _ { 0 } - \mathbf { x } _ { j } : \mathbf { x } _ { j } \in \mathbf { X } \}$ , ensuring central symmetry at $\mathbf { x } _ { \mathrm { 0 } }$ The $\beta \mathrm { . }$ -neighborhood of $\mathbf { x } _ { \mathrm { 0 } }$ is defined as the $\lceil n \beta \rceil$ points in X with the highest depth values with respect to (w.r.t.) ${ \bf X } _ { R { \bf x } _ { \mathrm { C } } }$ (highlighted in yellow in Figure 1). The sample β-LD of $\mathbf { x } _ { \mathrm { 0 } }$ is defined as the depth of $\mathbf { x } _ { \mathrm { 0 } }$ relative to its $\beta \mathrm { . }$ neighborhood:

$$
\mathrm { L D } ^ { \beta } ( \mathbf { x } _ { 0 } \mid \mathbf { X } ) = D ( \mathbf { x } _ { 0 } \mid \mathbf { N } _ { \mathbf { x } _ { 0 } } ^ { \beta } ) ,\tag{1}
$$

where D denotes the chosen depth function and $\mathbf { N } _ { \mathbf { x } _ { 0 } } ^ { \beta }$ is the $\beta \mathrm { . }$ -neighborhood of $\mathbf { x } _ { \mathrm { 0 } }$

![](images/2645fd3eb8ba0e52a52fa26eff3061675c0c4c1bbf202f6514d57bf8b22c1d45.jpg)

![](images/bce8108a24e15187cbc49ef08c239faf28636ac2ae3c5fd1dd170d59bd60ed3d.jpg)  
Figure 1. β-local halfspace depth construction for a sample of size $n = 3 0$ , with $\beta = 0 . 3 .$ (a) Construction of the $\beta \mathrm { . }$ -local depth region for $\mathbf { x } _ { \mathrm { 0 } }$ . (b) Depth-defined neighbors for $\mathbf { x } _ { \mathrm { 0 } }$ . The local depth of $\mathbf { x } _ { \mathrm { 0 } }$ is computed w.r.t. these neighbors.

β-ILD extends the idea of β-LD by integrating local depth values over a range of locality levels $\beta \in$ $( 0 , 1 ]$ . This integration leads to the resulting depth capturing both local and global structures in the data without fixing a particular $\beta .$

Definition 2.1 (Sample β-integrated local depth). Given a dataset $\mathbf { X } = \{ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n } \}$ , the sample $\beta \ – I L D$ for a point x is defined as

$$
\mathrm { I L D } ^ { \beta _ { b } } ( \mathbf { x } \mid \mathbf { X } ) = \sum _ { i = 1 } ^ { b } \mathrm { L D } ^ { \beta _ { i } } ( \mathbf { x } \mid \mathbf { X } ) \int _ { \beta _ { i - 1 } } ^ { \beta _ { i } } w ( \beta ) d \beta ,\tag{2}
$$

where $\beta _ { i + 1 } = \beta _ { i } + 1 / n ,$ , and $w ( \beta )$ is a weightingfunction satisfying $\begin{array} { r } { \int _ { \beta _ { 0 } } ^ { \beta _ { b } } w ( \beta ) d \beta = 1 } \end{array}$

In practice, $\beta _ { 0 }$ is set to a small positive value so that local depth is well-defined for any sample size larger than $n \beta _ { 0 }$ (this paper sets $\begin{array} { r } { \beta _ { 0 } = \operatorname* { m i n } ( \frac { 2 d - 1 } { n } , \frac { 9 } { n } ) } \end{array}$ , where d is the dimension of the data, to avoid instability of local depth values in very small neighborhoods), and $\beta _ { b } = 1$ . The weighting function is taken to be the uniform distribution between $\beta _ { 0 }$ and $\beta _ { b }$

Although many notions of data depth have been proposed, most are computationally intensive in high dimensions (e.g., halfspace depth [33] and projection depth [34] require $O ( n ^ { d } )$ , simplicial depth [35] requires $O ( n ^ { d + 1 } )$ ; see [36] for a comprehensive overview). For this reason, DLCC adopts spatial depth [37], which can be computed in $O ( n d )$ time per point. Since A-DLCC relies on β-ILD, which requires repeated local depth computations across multiple locality levels, it also uses spatial depth for computational efficiency. The spatial depth for a point z with respect to a sample X is

$$
D _ { \mathrm { S D } } ( \mathbf { z } \mid \mathbf { X } ) = 1 - \left\| \sum _ { i = 1 } ^ { n } { \frac { \mathbf { z } - \mathbf { x } _ { i } } { n \| \mathbf { z } - \mathbf { x } _ { i } \| } } \right\| .\tag{3}
$$

## 2.2. DLCC algorithm

With the basic concepts of depth and local depth in place, we briefly review the DLCC framework, which consists of the following steps:

Step 1 Construct the depth-based similarity matrix $\mathbb { S } ,$ where each entry is defined as $\mathbb { S } _ { i , j } = D ( \mathbf { x } _ { j } \mid \mathbf { X } _ { R \mathbf { x } _ { i } } )$ In both DLCC and A-DLCC, symmetry is enforced by averaging $\mathbb { S } _ { i , j }$ and $\mathbb { S } _ { j , i }$ . Given a neighborhood size parameter (or equivalently, $\beta )$ , define each point’s neighborhood accordingly.

Step 2 For each neighborhood identify the deepest point, and filter these candidates by additional criteria. The filtered local centers are then grouped using either the “min” or “max” strategies, both of which allow a cluster to have multiple exemplars.

Step 3 The resulting local center groups determine the number of clusters. Remaining points are assigned to clusters based on their similarity scores to each group and whether they are included in the neighborhoods of the group’s local centers. This produces the temporary clusters and ambiguous points that remain unlabeled, because of a lack of similarity to any local center.

Step 4 Finally, classification techniques such as kNN, random forest, or the maximal depth classifier are applied to assign the unlabeled points to an existing cluster.

Figure 2 shows the flowchart of the A-DLCC algorithm. While the overall workflow follows that of the original DLCC, certain steps indicated by corresponding Section or Algorithm labels (i.e., defining and grouping local centers and constructing temporary clusters) have been modified to eliminate the need for predefined parameters. The Python implementation is available at GitHub<sup>1</sup>.

## 2.3. Summary of notations

To improve clarity, we summarize in Table 1 the main notations used in subsequent sections.

![](images/d882803922dc46e6f9dfddc35a1fe51e408afaf8cc5f7d5c0e581c5ddbe93f90.jpg)  
Figure 2. Flowchart for the A-DLCC algorithm

Table 1. Summary of notation.
<table><tr><td>Notation</td><td>Description</td></tr><tr><td> $\mathbf { X }$ </td><td>Dataset,  $\left\{ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n } \right\}$ </td></tr><tr><td> $n , d$ </td><td>Number of observations and di-  $\mathbf { X }$ </td></tr><tr><td> $\beta , b$ </td><td>mensions of Locality level and number of lo- cality levels</td></tr><tr><td> $\mathrm { L D } ^ { \beta } ( \mathbf { x } ) , \mathrm { I L D } ^ { \beta } ( \mathbf { x } )$   $\mathbf { N } _ { \mathbf { x } } ^ { \beta }$ </td><td>β-LD and  $\beta { \mathrm { - I I L D } }$  at x</td></tr><tr><td> $f$ </td><td>β-neighborhood of point x Frequency</td></tr><tr><td> $\mathbf { c } , \mathbf { C }$   $\mathcal { E } , \mathcal { E } ^ { \prime }$ </td><td>local center and set of local centers Sets of locally deep points;  $\mathcal { E } ^ { \prime } =$ </td></tr><tr><td> $\mathbb { S } , \mathrm { D S } ( \cdot , \cdot )$ </td><td> $\mathcal { E } \setminus \mathbf { C }$  Depth-based similarity matrix, and similarity between two points,</td></tr><tr><td> $\mathbf { G } _ { t }$ </td><td>i.e.,  $\mathbb { S } _ { i , j } = \mathrm { D S } ( \mathbf { x } _ { i } , \mathbf { x } _ { j } )$  Group of points assigned to local</td></tr><tr><td> $\mathrm { G L S } ( i , j )$ </td><td>center  $\mathbf { c } _ { t }$  Initial group-level similarity be-</td></tr><tr><td> $\operatorname { r s } _ { \mathbf { G } } ( \cdot , \cdot )$ </td><td>tween  $\mathbf { G } _ { i }$  and  $\mathbf { G } _ { j }$  Reachable similarity within group  $\mathbf { G }$ </td></tr></table>

<table><tr><td>Notation</td><td>Description</td></tr><tr><td> $\mathbb { G }$ </td><td>Group similarity matrix</td></tr><tr><td> $\mu _ { \mathbf G } ( \mathbf { x } )$ </td><td>Intra-group similarity of  $\mathbf { x }$  in  $\mathbf { G }$ </td></tr><tr><td> $\mu _ { \mathbf { G } _ { i }  \mathbf { G } _ { j } } ( \mathbf { x } )$ </td><td>Between-group similarity from  $\mathbf { x } \in \mathbf { G } _ { i }$  to group  $\mathbf { G } _ { j }$ </td></tr><tr><td> $\rho \mathbf { G } _ { i } {  } \mathbf { G } _ { j }$ </td><td>Relative reachability ratio from  $\mathbf { G } _ { i }$  to  $\mathbf { G } _ { j }$ </td></tr><tr><td> $\tilde { \mathbb { S } } , \tilde { \mathbb { G } }$ </td><td>Reachable versions of S and  $\mathbb { G }$ </td></tr><tr><td> $\bar { \phi } _ { i } , \phi _ { i }$ </td><td>Background level of  $\mathbf { G } _ { i }$  and its clipped version (base threshold)</td></tr><tr><td> $\omega _ { i | j }$ </td><td>Disruption suffered by  $\mathbf { G } _ { i }$  when merged with  $\mathbf { G } _ { j }$ </td></tr><tr><td> $Q _ { \mathrm { t h } } ( i \mid j )$ </td><td>Adaptive threshold applied by  $\mathbf { G } _ { i }$  t to the</td></tr><tr><td> $\Delta Q _ { i j }$ </td><td>merge with  $\mathbf { G } _ { j }$  Modularity gain of joining  $\mathbf { G } _ { i }$  and  $\mathbf { G } _ { j }$  on the depth graph</td></tr><tr><td> $\mathbf { g } _ { k }$   $\mathbf { G _ { g } } , \mathbf { b } ( \mathbf { g } )$ </td><td> $\mathbf { A }$  group of local centers Points of a group of local centers, and the weakest similarity its connectivity relies</td></tr><tr><td> $\mathcal { T } _ { k }$ </td><td>on Temporary cluster corresponding to gk</td></tr></table>

## 3. β-integrated local depth-based point representativeness

We propose a nonparametric method to identify exemplars within a dataset, together with a representativeness ranking. Recall the β-ILD definition in Section 2. For a dataset $\mathbf { X } = \{ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n } \}$ , define a sequence of locality levels $\beta _ { 1 } < \cdots < \beta _ { b }$ with $\beta _ { b } = 1$ and $\beta _ { i + 1 } = \beta _ { i } + 1 / n .$ . At each level $\beta _ { j }$ , we compute the local depth $\mathrm { L D } ^ { \beta _ { j } } \left( \mathbf { x } _ { i } \right)$ for every data point $\mathbf { x } _ { i } .$ . Collect these values into an $n \times b$ matrix L, with entry $( i , j )$ given by $\mathrm { L D } ^ { \beta _ { j } } \left( \mathbf { x } _ { i } \right)$ .

Similarly, for β-ILD with a uniform weighting function across locality levels, a matrix storing ILD values can be constructed, denoted IL. Specifically, for each row i in $\mathbb { L } ,$ the ILD values are computed as

$$
\mathbb { I L } _ { i , j } = \frac { 1 } { j } \sum _ { u = 1 } ^ { j } \mathbb { L } _ { i , u } , \quad j = 1 , \ldots , b .
$$

Definition 3.1 (Locally deep point). For each data point $\mathbf { x } _ { i }$ and locality level $\beta _ { j } ,$ consider the neighborhood $\mathbf { N } _ { \mathbf { x } _ { i } } ^ { \beta _ { j } }$ . The locally deep pointfor the subset defined by $\mathbf { N } _ { \mathbf { x } _ { i } } ^ { \beta _ { j } }$ is the point $\mathbf { z } \in \mathbf { N } _ { \mathbf { x } _ { i } } ^ { \beta _ { j } }$ that maximizes $\mathrm { I L D } ^ { \beta _ { j } } ( { \bf z } )$

It is important to note that for any point $\mathbf { z } \in \mathbf { N } _ { \mathbf { x } _ { i } } ^ { \beta _ { j } }$ , the ILD value is computed based on the neighborhood $\mathbf { N } _ { \mathbf { z } } ^ { \beta _ { j } }$ , rather than the reference neighborhood $\mathbf { N } _ { \mathbf { x } _ { i } } ^ { \beta _ { j } }$ . Specifically, when determining the locally deep point of $\mathbf { N } _ { \mathbf { x } _ { i } } ^ { \beta }$ , we compare the values in the j-th column of the matrix IL.

![](images/6b6cb53872a49c5d889f7f6c26022ae04f9a3017604782f0a567b48a564dec0d.jpg)

![](images/e5d5b138524181682754ac0378496a1b681d013ff53d22c5fe6da7f5ce1cd336.jpg)  
Figure 3. ILD values for two chosen points (Point 1000 and Point 2000) in the Yale-B dataset (Section 5.2) across varying locality levels. Colors indicate the locally deepest point in each neighborhood.

Applying the above definition to each of the nb subsets (one for each $\mathbf { x } _ { i }$ and each $\beta _ { j } )$ yields a collection of candidate local centers. Naturally, the frequency of each candidate local center can be defined.

Definition 3.2 (Frequency of locally deep point). The frequency f of a locally deep point z is defined as the number ofsubsets across all locality levels and points in which z emerges as the ILD-based local center.

Remark 3.1. We denote by E the ordered set of all locally deep points sorted by decreasing frequency, and by ${ \mathcal { E } } ^ { \prime } \subseteq { \mathcal { E } }$ those not selected as local centers in Section 3.2. The set $\mathcal { E } ^ { \prime }$ is referred to in Section 4.3.

Compared to ${ \beta \mathrm { - L D } } ,$ the β-ILD provides a smoothed summary of local depth values, reducing fluctuations and thereby avoiding the identification of an excessive number of locally deep points. As higher locality levels correspond to larger neighborhoods that naturally include more points, any given point is more likely to appear in the neighborhoods of others. If a point is consistently identified as locally deep at these large locality levels, its frequency will be accordingly high. Conversely, if a point is only locally deep at small locality levels, its frequency will be low. Figure 3 illustrates this behavior. Point 70, which corresponds to a global depth median for the complete sample is consistently chosen as the locally deep point of most neighborhoods for both points, when the locality level is large enough. The frequency order thus reflects a natural ranking of representativeness among the defined locally deep points.

## 3.1. Possible Application

Representative points have been widely used in both supervised and unsupervised learning. Here, we explore a simple application of locally deep points by incorporating them into a Kmedoids clustering procedure, with details shown in Algorithm 1.

Table 2. ARI values of Kmedoids variants across five datasets. The data set Olive is from the pgmm package [38]. Other datasets are discussed in Section 5.2. Depth and Depth-2 correspond to Algorithm 1 with only\_exemplars set to TRUE and FALSE, respectively; PAM refers to the standard method implemented in the cluster package [39].
<table><tr><td>Dataset</td><td>Depth</td><td>Depth-2</td><td>PAM</td></tr><tr><td>BC  $( n = 5 6 9 , d = 3 0 , K = 2 )$ </td><td>0.7488</td><td>0.7488</td><td>0.6079</td></tr><tr><td>Seed  $( n = 2 1 0 , d = 7 , K = 3 )$ </td><td>0.7103</td><td>0.7103</td><td>0.7103</td></tr><tr><td>Wine  $( n = 1 7 8 , d = 1 3 , K = 3 )$ </td><td>0.7137</td><td>0.7137</td><td>0.7411</td></tr><tr><td>Olive  $( n = 5 7 2 , d = 8 , K = 3 )$ </td><td>0.7451</td><td>0.7451</td><td>0.7247</td></tr><tr><td>Seg  $( n = 2 0 8 6 , d = 1 8 , K = 7 )$ </td><td>0.5203</td><td>0.5178</td><td>0.4550</td></tr></table>

Algorithm 1 Depth-based Kmedoids clustering with exemplar initialization   
Input: Data matrix X, number of clusters $K ,$ ordered exemplar set E, depth-based neighborhood information, logical parameter   
only\_exemplars, max iterations.   
Output: Cluster assignments C, final medoids ${ \mathcal { M } } .$   
1: Set locality level $\overset { \vartriangle } { \beta } = 1 / ( 1 . 5 K )$   
2: Initialize center list $\mathcal { M } = [ ]$ with first center as top exemplar: $\mathcal { M } _ { 1 } = \mathcal { E } _ { 1 }$   
3: while length $( \mathcal { M } ) < K$ do   
4: Append to $\mathcal { M }$ the first remaining exemplar in E not in $\textstyle \bigcup _ { \mathbf { x } \in \mathcal { M } } \mathbf { N } _ { \mathbf { x } } ^ { \beta } .$   
5: end while   
6: Assign each point to the closest medoid in $\mathcal { M } ,$ and obtain initial cluster assignment $\mathcal { C } _ { \mathrm { p r e v } }$   
7: repeat   
8: for each cluster $k = 1$ to K do   
9: Extract indices $\mathcal { T } _ { k }$ of cluster $k .$   
10: if only\_exemplars then   
11: Compute depth of exemplars $\varepsilon$ w.r.t. cluster k.   
12: Set new medoid $\mathcal { M } _ { k } = \mathrm { \bar { a r g } } \operatorname* { m a x } _ { e \in \mathcal { E } } D _ { k } ( e )$   
13: else   
14: Compute depth of cluster points w.r.t. cluster $k .$   
15: Set $\bar { \mathcal { M } } _ { k } = \mathrm { \bar { a r g } } \operatorname* { m a x } _ { i \in \mathcal { T } _ { k } } \bar { D } _ { k } ( i )$   
16: end if   
17: end for   
18: Reassign each point to the closest medoid in $\mathcal { M } ,$ and obtain new assignment $\mathcal { C } _ { \mathrm { n e w } } .$   
19: Set $\mathcal { C } _ { \mathrm { p r e v } }  \mathcal { C } _ { \mathrm { n e w } }$   
20: until convergence or iteration limit reached

Compared to the standard Kmedoids algorithm, the proposed method introduces two modifications. First, the initialization phase selects the most representative locally deep points—ranked by their frequency—while enforcing diversity by ensuring that no newly chosen exemplar lies within the empirical β-neighborhood $\mathbf { N } _ { \mathbf { x } } ^ { \beta }$ of previously selected ones. This strategy encourages both representativeness and separation among initial medoids, and can be extended to other clustering methods requiring initialization. Second, during each iteration, medoid updates are guided by data depth: depending on a user-specified setting, the medoid for each cluster is selected either from all cluster members or restricted to locally deep exemplars. Table 2 presents a comparison of clustering performance across methods. Notably, restricting medoid candidates to only the locally deep points yields comparable results to using all points, which supports the representativeness of these locally deep points.

## 3.2. Local centers for A-DLCC

Now, our target is to define local centers from those locally deep points, analogue to the filtering procedures in DLCC. We expect a local center should be deep in its own neighborhood, and with comparatively high frequency. In order to define local centers, we first introduce the following definition.

Definition 3.3 (Self Centrality Level). Let x be a locally deep point and let $\mathrm { I L D } ^ { \beta } ( \mathbf { x } )$ denote its ILD value at locality level $\beta \in ( 0 , 1 ]$ . Define the self centrality level set of x as

$$
\mathcal { B } ( \mathbf { x } ) : = \left\{ \beta \in ( 0 , 1 ] : \mathrm { I L D } ^ { \beta } ( \mathbf { x } ) = \operatorname* { m a x } _ { \mathbf { z } \in \mathbf { N } _ { \mathbf { x } } ^ { \beta } } \mathrm { I L D } ^ { \beta } ( \mathbf { z } ) \right\} ,
$$

$i . e . ,$ , the set oflocality levels at which x attains the maximal ILD value within its own $\beta .$ -neighborhood $\mathbf { N } _ { \mathbf { x } } ^ { \beta _ { b } }$ $H B ( { \bf x } )$ is non-empty, the self centrality level of x is defined as

$$
B ^ { * } ( \mathbf { x } ) : = \arg \operatorname* { m a x } _ { \beta \in B ( \mathbf { x } ) } \mathrm { I L D } ^ { \beta } ( \mathbf { x } ) .
$$

$B ( \mathbf { x } )$ is the set of levels $\beta$ for which the point x has the largest ILD among all points in its neighborhood

at level $\beta .$ Then $B ^ { * } ( \mathbf { x } )$ is the locality level corresponding to the largest depth value achieved across all levels in $B ( \mathbf { x } )$

Definition 3.4 (Local Center). Let x be a locally deep point with a non-empty selfcentrality level set $B ( \mathbf { x } )$ and corresponding self centrality level $B ^ { * } ( \mathbf { x } )$ . Let $\bar { \bf N } _ { \bf x } ^ { \bar { B } ^ { * } ( { \bf x } ) }$ denote the neighborhood of x under locality level $B ^ { * } ( \mathbf { x } )$ . We say that x is a local center ifit satisfies thefollowing stability condition:

$$
\frac { 1 } { \left| \left\{ { \bf z } : { \bf x } \in { \bf N } _ { { \bf z } } ^ { B ^ { * } ( { \bf x } ) } \right\} \right| } \sum _ { \{ { \bf z } : { \bf x } \in { \bf N } _ { { \bf z } } ^ { B ^ { * } ( { \bf x } ) } \} } \mathbb { I } \left\{ { \bf x } = \arg \operatorname* { m a x } _ { { \bf u } \in { \bf N } _ { { \bf z } } ^ { B ^ { * } ( { \bf x } ) } } \mathrm { I L D } ^ { B ^ { * } ( { \bf x } ) } ( { \bf u } ) \right\} > 0 . 5 ,
$$

where $\mathbf { N } _ { \mathbf { z } } ^ { B ^ { * } ( \mathbf { x } ) }$ is the neighborhood of z under the same locality level $B ^ { * } , \mid \cdot \mid$ represents cardinality and $\mathbb { I } \{ \cdot \}$ is the indicatorfunction.

In summary, a local center is a locally deep point that has a well-defined self centrality level $B ^ { * } ( \mathbf { x } )$ and is most frequently identified as the locally deepest point within the neighborhoods it belongs to at this level.

## 4. Automatic-depth based local center clustering

With defined local centers, the next target for A-DLCC is to group them. The A-DLCC algorithm has three parts: defining group similarities without parameters, determining the number of clusters, and constructing temporary clusters.

## 4.1. Group-level local similarity

For clarity of latter discussions, we use $\mathrm { D S } ( \cdot , \cdot )$ to denote the depth-based similarity, where $\mathrm { D S } ( \mathbf { x } _ { i } , \mathbf { x } _ { j } ) =$ $\mathbb { S } _ { i , j }$ corresponds to the similarity directed from $\mathbf { x } _ { j }$ to $\mathbf { x } _ { i } , \mathrm { i . e . , } D ( \mathbf { x } _ { j } \mid \mathbf { X } _ { R \mathbf { x } _ { i } } )$ . While the depth-based similarity matrix may be asymmetric, in A-DLCC we follow DLCC and use the symmetrized version obtained by averaging $\mathbb { S } _ { i , j }$ and $\mathbb { S } _ { j , i }$ , and, for notational simplicity, we continue to denote it by S. This symmetrized similarity is then used to define the assignment of data points to local centers.

Assume the local centers are denoted by $\mathbf { C } = \{ \mathbf { c } _ { 1 } , \mathbf { c } _ { 2 } , \ldots , \mathbf { c } _ { T } \}$ . Note that the indices of c differ from those of the original dataset. Each data point is assigned to the group associated with its most similar local center. Specifically, for each $t = 1 , \dots , T$ , we define

$$
\mathbf { G } _ { t } = \left\{ \mathbf { x } _ { u } \in \mathbf { X } \bigg | \mathbf { c } _ { t } = \arg \operatorname* { m a x } _ { \mathbf { c } _ { i } \in \mathbf { C } } \mathrm { D S } ( \mathbf { x } _ { u } , \mathbf { c } _ { i } ) \right\} ,\tag{4}
$$

the group of points in the data set whose most similar center is $\mathbf { c } _ { t } .$ As a result, the groups $\left\{ \mathbf { G } _ { t } \right\}$ form a partition of X into $T$ groups, and overlap-based measures cannot be used to quantify the similarity between different groups. Instead, inspired by the concept of PaLD [28], we design a similarity measure for disjoint groups. Specifically, for each pair of groups $( \mathbf { G } _ { i } , \mathbf { G } _ { j } )$ , we define the group-level local similarity $\mathrm { G L S } ( i , j )$ as follows. For any randomly chosen $\mathbf { x } _ { u } \in \mathbf { G } _ { i }$ and $\mathbf { x } _ { v } \in \mathbf { G } _ { j }$ , define the local focus region as

$$
\begin{array} { r } { U _ { \mathbf { x } _ { u } , \mathbf { x } _ { v } } = \left\{ \mathbf { z } \in \mathbf { X } \mid \mathrm { D S } ( \mathbf { x } _ { u } , \mathbf { z } ) \geq \mathrm { D S } ( \mathbf { x } _ { u } , \mathbf { x } _ { v } ) \ \mathrm { o r } \ \mathrm { D S } ( \mathbf { x } _ { v } , \mathbf { z } ) \geq \mathrm { D S } ( \mathbf { x } _ { v } , \mathbf { x } _ { u } ) \right\} . } \end{array}
$$

Define the directional similarity from $\mathbf { G } _ { j }$ to $\mathbf { G } _ { i }$ as

$$
W _ { \mathbf { G } _ { j }  \mathbf { G } _ { i } } = \frac { 1 } { | \mathbf { G } _ { i } | | \mathbf { G } _ { j } | } \sum _ { \mathbf { x } _ { u } \in \mathbf { G } _ { i } } \sum _ { \mathbf { x } _ { v } \in \mathbf { G } _ { j } } H \bigl ( \mathbf { x } _ { u } , \mathbf { x } _ { v } \mid \mathbf { G } _ { j } \cap U _ { \mathbf { x } _ { u } , \mathbf { x } _ { v } } \bigr ) ,
$$

where for any set $\mathbf { G }$

$$
H ( \mathbf { x } _ { u } , \mathbf { x } _ { v } \mid \mathbf { G } ) = { \frac { 1 } { | \mathbf { G } | } } \sum _ { \mathbf { z } \in \mathbf { G } } \left[ \mathbb { I } { \big ( } \mathrm { D S } ( \mathbf { z } , \mathbf { x } _ { u } ) > \mathrm { D S } ( \mathbf { z } , \mathbf { x } _ { v } ) { \big ) } + { \frac { 1 } { 2 } } \mathbb { I } { \big ( } \mathrm { D S } ( \mathbf { z } , \mathbf { x } _ { u } ) = \mathrm { D S } ( \mathbf { z } , \mathbf { x } _ { v } ) { \big ) } \right] .
$$

Then, the group-level local similarity between $\mathbf { G } _ { i }$ and $\mathbf { G } _ { j }$ is the symmetric average

$$
\mathrm { G L S } ( i , j ) = \frac { 1 } { 2 } ( W _ { { \bf G } _ { i }  { \bf G } _ { j } } + W _ { { \bf G } _ { j }  { \bf G } _ { i } } ) .
$$

Based on this, we define $\mathbb { G } = \{ \mathrm { G L S } ( i , j ) \} _ { i , j \in \{ 1 , \dots , T \} }$ as the group-level local similarity matrix.

We note that similarity $W _ { \mathbf { G } _ { j }  \mathbf { G } }$ admits the following probabilistic interpretation. Randomly sample a pair of points $V _ { i } \in { \bf G } _ { i } , V _ { j } \in { \bf G } _ { j }$ . Then, uniformly sample a point $Z \in U _ { V _ { i } , V _ { j } } \cap { \bf G } _ { j }$ . The value $W _ { \mathbf { G } _ { j }  \mathbf { G } }$ i represents the probability that such a point $Z$ is more similar to $V _ { i } \in \mathbf { G } ,$ <sub>i</sub> than to $V _ { j } \in { \bf G } _ { j } , \mathrm { i . e . }$ .,

$$
W _ { \mathbf { G } _ { j }  \mathbf { G } _ { i } } = \mathbb { P } ( \mathrm { D S } ( Z , V _ { i } ) > \mathrm { D S } ( Z , V _ { j } ) ) .
$$

Ties, if any, are broken uniformly at random. A large value of $W _ { \mathbf { G } _ { j }  \mathbf { G } _ { i } }$ implies that $\mathbf { G } _ { j }$ is not wellseparated from $\mathbf { G } _ { i }$ . The GLS matrix symmetrizes this relationship by averaging $W _ { \mathbf { G } _ { j }  \mathbf { G } }$ and $W _ { \mathbf { G } _ { i }  \mathbf { G } _ { j } }$ giving a balanced estimate of the mutual similarity between the two groups.

Proposition 4.1 (Zero similarity under strong separation). Let $\mathbf { G } _ { i } , \mathbf { G } _ { j } \subseteq \mathbf { X }$ be two disjoint groups. Suppose that, for all $\mathbf { x } _ { u } \in \mathbf { G } _ { j }$ , we have

$$
\operatorname* { m i n } _ { \mathbf { x } _ { v } \in \mathbf { G } _ { j } } \mathrm { D S } ( \mathbf { x } _ { u } , \mathbf { x } _ { v } ) > \operatorname* { m a x } _ { \mathbf { z } \in \mathbf { G } _ { i } } \mathrm { D S } ( \mathbf { x } _ { u } , \mathbf { z } ) .
$$

Then the directional group similarity from $\mathbf { G } _ { j }$ to $\mathbf { G } _ { i }$ satisfies $W _ { \mathbf { G } _ { j }  \mathbf { G } _ { i } } = 0$

In practice, only similarities between nearby groups are typically of interest. To accelerate processing, one can omit GLS evaluations for group pairs that satisfy the following centroid-based separation check

$$
\operatorname* { m i n } _ { \mathbf { x } _ { u } , \mathbf { x } _ { v } \in \mathbf { G } _ { j } } \mathrm { D S } ( \mathbf { x } _ { u } , \mathbf { x } _ { v } ) > \operatorname* { m a x } _ { \mathbf { z } \in \mathbf { G } _ { i } } \mathrm { D S } ( \mathbf { c } _ { j } , \mathbf { z } ) , \qquad \operatorname* { m i n } _ { \mathbf { x } _ { u } , \mathbf { x } _ { v } \in \mathbf { G } _ { i } } \mathrm { D S } ( \mathbf { x } _ { u } , \mathbf { x } _ { v } ) > \operatorname* { m a x } _ { \mathbf { z } \in \mathbf { G } _ { j } } \mathrm { D S } ( \mathbf { c } _ { i } , \mathbf { z } ) .
$$

Although this condition does not guarantee that $\mathrm { G L S } ( i , j ) = 0$ , the similarity is generally small enough that omitting the computation should have negligible impact on the subsequent adaptive grouping processes.

Proposition 4.2 (Invariance under similarity transformations). For any pair of groups $\mathbf { G } _ { i }$ and $\mathbf { G } _ { j } ,$ , if a similarity transformation $T \ ( i . e .$ , any combination of rotation, dilation, and translation) is applied to $\mathbf { G } _ { i } \cup \mathbf { G } _ { j } ,$ , then $\mathrm { G L S } ( i , j )$ remains unchanged for any input similarity measure DS that preserves the ordinal relationships among points under the transformation, i.e., $i f \operatorname { D S } ( \mathbf { x } _ { u } , \mathbf { x } _ { v } ) > \operatorname { D S } ( \mathbf { x } _ { u } , \mathbf { x } _ { q } )$ , then $\mathrm { D S } ( T ( \mathbf { x } _ { u } ) , T ( \mathbf { x } _ { v } ) ) > \mathrm { D S } ( T ( \mathbf { x } _ { u } ) , T ( \mathbf { x } _ { q } ) )$

In particular, the invariance of $\mathbb { S }$ under similarity transformations ensures that Proposition 4.2 holds directly in our setting, that is, G is also invariant under similarity transformation.

## 4.2. Adaptive grouping strategy

The group similarity alone does not determine whether two groups should merge. We introduce an adaptive grouping rule that decides, for every pair of adjacent groups, whether one is part of the other (absorption) or whether the two are two parts of one structure (bonding). The rule builds on the bottleneck path [40,

41] concept in graph theory. The idea is to measure the strength of connection between two nodes as the strongest path between them, where each path’s strength is defined by its weakest edge. We adopt this idea to define reachable similarity using symmetric similarity matrices.

Definition 4.1 (Reachable similarity). Let N be the number of nodes, and let $S$ be a symmetric similarity matrix with entries $S _ { i , j }$ in [0, 1]. The reachable similarity $\operatorname { r s } ( i , j )$ between any two nodes i and j is defined as the maximum, over all paths connecting i to j, of the minimum similarity along that path. Formally, for any path $\pmb { p } = ( p _ { 0 } , p _ { 1 } , \dots , p _ { m } )$ with $p _ { 0 } = i$ and $p _ { m } = j $ , its bottleneck similarity is min<sub>0≤ℓ<m</sub> $S _ { p _ { \ell } , p _ { \ell + 1 } } .$ Then, the reachable similarity between nodes i and j is given by

$$
\operatorname { r s } ( i , j ) = \operatorname* { m a x } _ { p : i \to j } \ \operatorname* { m i n } _ { 0 \leq \ell < m } S _ { p _ { \ell } , p _ { \ell + 1 } } .
$$

In our framework, $S$ may denote either the pointwise similarity matrix $\mathbb { S }$ or the groupwise similarity matrix G. We denote their corresponding reachable similarity matrices as $\tilde { \mathbb { S } }$ and $\tilde { \mathbb { G } }$ , respectively.

Definition 4.2 (Intra- and between-group reachability). For any groups $\mathbf { G } _ { i }$ and $\mathbf { G } _ { j }$ , and for any $\mathbf { x } \in \mathbf { G } _ { i }$ define

$$
I n t r a - g r o u p s i m i l a r i t y : \quad \mu _ { \mathbf { G } _ { i } } ( \mathbf { x } ) = \frac { 1 } { | \mathbf { G } _ { i } | - 1 } \sum _ { \mathbf { z } \in \mathbf { G } _ { i } \backslash \mathbf { x } } \mathrm { r s } _ { \mathbf { G } _ { i } } ( \mathbf { x } , \mathbf { z } ) ,\tag{5}
$$

Between-group similarity:

$$
\mu _ { \mathbf { G } _ { i }  \mathbf { G } _ { j } } ( \mathbf { x } ) = \frac { 1 } { | \mathbf { G } _ { j } | } \sum _ { \mathbf { z } \in \mathbf { G } _ { i } } \mathrm { r s } _ { \mathbf { G } _ { i } \cup \mathbf { G } _ { j } } ( \mathbf { x } , \mathbf { z } ) ,\tag{6}
$$

where $\operatorname { r s } _ { \mathbf { G } } ( \cdot , \cdot )$ denotes the reachable similarity computed within the set $\mathbf { G } ,$ and $| \mathbf { G } _ { i } |$ and $| \mathbf { G } _ { j } |$ are the sizes of $\mathbf { G } _ { i }$ and $\mathbf { G } _ { j } ,$ respectively.

Definition 4.3 (Relative reachability ratio). Given two groups $\mathbf { G } _ { i }$ and $\mathbf { G } _ { j }$ , the relative reachability ratio from $\mathbf { G } _ { i } t o \mathbf { G } _ { j }$ is defined as

$$
\rho _ { \mathbf { G } _ { i }  \mathbf { G } _ { j } } = \frac { 1 } { | \mathbf { G } _ { i } | } \sum _ { \mathbf { x } \in \mathbf { G } _ { i } } \frac { \mu _ { \mathbf { G } _ { i }  \mathbf { G } _ { j } } ( \mathbf { x } ) } { \mu _ { \mathbf { G } _ { i } } ( \mathbf { x } ) } ,\tag{7}
$$

and the quantity min $\imath \{ \rho _ { \mathbf { G } _ { i }  \mathbf { G } _ { j } } , \rho _ { \mathbf { G } _ { j }  \mathbf { G } _ { i } } \}$ is referred to as the minimal reachability ratio of the merge between $\mathbf { G } _ { i }$ and $\mathbf { G } _ { j }$

We now introduce the acceptance rule for determining whether two groups should be merged. Different clustering scenarios may require different minimal reachability ratios. For connected shapes, stricter thresholds help avoid incorrect merges, while in well-separated settings, lower thresholds are acceptable. We therefore assign a base threshold to each group by evaluating the relative reachability ratio $\rho _ { \mathbf { G } _ { i } \to \mathbf { X } \backslash \mathbf { G } }$ ,which we call the background level of $\mathbf { G } _ { i }$ and denote by $\bar { \phi } _ { i } = \rho _ { \mathbf { G } _ { i } \to \mathbf { X } \backslash \mathbf { G } }$ . The base threshold is its clipped version

$$
\phi _ { i } = \operatorname* { m i n } \left( \operatorname* { m a x } ( \bar { \phi } _ { i } , 0 . 9 ) , 0 . 9 9 \right) .\tag{8}
$$

The bounding interval [0.9, 0.99] ensures that the threshold remains within a reasonable range, avoiding values that are too permissive or overly strict. A background level $\bar { \phi } _ { i } \geq 1$ means that the points of $\mathbf { G } _ { i }$ reach the rest of the data at least as well as they reach each other. This case is treated separately in Section 4.2.1.

This base threshold prevents all groups from collapsing into one, but pairwise merges usually need a stricter criterion. We define a merge-specific acceptance threshold that penalizes merges with large disruption. Consider two groups $\mathbf { G } _ { i }$ and $\mathbf { G } _ { j }$ , and let $\mu _ { \mathbf { G } _ { i j } } ( \mathbf { x } ) = \mu _ { \mathbf { G } _ { i } \cup \mathbf { G } _ { j } } ( \mathbf { x } )$ denote the average reachable similarity of x to all other points in $\mathbf { G } _ { i } \cup \mathbf { G } _ { j } \ ( \mathrm { i . e . }$ , its intra-group reachability after merging). We define the disruption $\omega _ { i \mid j }$ suffered by $\mathbf { G } _ { i }$ in the merge as the proportion of its points whose intra-group reachability decreases after merging,

$$
\omega _ { i | j } = \frac { | \{ \mathbf { x } \in \mathbf { G } _ { i } : \mu _ { \mathbf { G } _ { i } } ( \mathbf { x } ) > \mu _ { \mathbf { G } _ { i j } } ( \mathbf { x } ) \} | } { | \mathbf { G } _ { i } | } ,\tag{9}
$$

and $\omega _ { j \mid i }$ symmetrically. Keeping the two sides separate, rather than averaging them, lets each side judge the merge from its own point of view; a small group swallowed by a large one and a large group touched by a small one are disrupted very differently. The adaptive threshold that $\mathbf { G } _ { i }$ applies to the merge with $\mathbf { G } _ { j }$ is

$$
Q _ { \mathrm { t h } } ( i \mid j ) = \phi _ { i } + \omega _ { i \mid j } ^ { 2 } \left( \operatorname* { m a x } ( q , \phi _ { i } ) - \phi _ { i } \right) , \qquad q = 0 . 9 7 5 .\tag{10}
$$

Here, the effect of ω is dampened by squaring, to avoid overreacting to mild degradations that commonly occur when merging distinct groups. The threshold interpolates between the background level of $\mathbf { G } _ { i }$ and a fixed upper level $q .$ In all experiments q is fixed as 0.975 without tuning.

## 4.2.1. Isolated local centers and cells without a boundary

Two checks precede the merging rule.

The first check covers small and simple data sets in which each cluster produces a single local center. If the local centers are far from each other relative to the size of their groups, no merge should even be considered. Given locality level $\beta = \operatorname* { m a x } _ { t } | \mathbf { G } _ { t } | / 2 n$ , we check the neighbors of each local center: if no other local center appears in the neighborhood $\mathbf { N } _ { \mathbf { c } _ { t } } ^ { \beta }$ of any local center $\mathbf { c } _ { t } .$ , all local centers are treated as singleton groups and the merging step is skipped altogether.

The second check identifies groups that have no boundary of their own. When $\bar { \phi } _ { i } = \rho _ { \mathbf { G } _ { i }  \mathbf { X } \backslash \mathbf { G } _ { i } } \geq 1$ the points of $\mathbf { G } _ { i }$ reach the rest of the data at least as well as they reach each other, so $\mathbf { G } _ { i }$ is only a candidate. A candidate is set aside if its mean intra-group reachability is below that of every non-candidate linked to it by a direct GLS edge $( \mathbb { G } _ { i , j } = \tilde { \mathbb { G } } _ { i , j } \neq 0$ and $\bar { \phi } _ { j } < 1 )$ ; if it has no such neighbor, it is set aside only when its size is smaller than the median group size. Set-aside groups skip the merging rule, and their local centers are demoted to ordinary points. We call the remaining groups units. After the units have been grouped, every observation is assigned to the retained local center it is most similar $^ { \mathrm { t o , } }$ so the points originally in a set-aside group join a unit.

## 4.2.2. Absorption and bonding

Let $\mathbf { g } _ { 1 } , \ldots , \mathbf { g } _ { K }$ be the current groups of local centers and for a group g, let $\begin{array} { r } { \mathbf { G } _ { \mathbf { g } } = \bigcup _ { t : \mathbf { c } _ { t } \in \mathbf { g } } \mathbf { G } _ { t } } \end{array}$ be its points. The matrices $\mathbb { G }$ and $\tilde { \mathbb { G } }$ are not updated as these groups are merged. An entry $\mathbb { G } _ { i , j }$ remains the group-level similarity between the initial groups of local centers $\mathbf { c } _ { i }$ and $\mathbf { c } _ { j }$ . Two groups are adjacent when some pair of their local centers has positive group-level local similarity. For every adjacent pair $\left( \mathbf { g } _ { a } , \mathbf { g } _ { b } \right)$ we compute the two relative reachability ratios $\rho _ { a  b } = \rho _ { { \bf G } _ { { \bf g } _ { a } }  { \bf G } _ { { \bf g } _ { b } } }$ and $\rho _ { b  a }$ and the two disruptions $\omega _ { a \mid b }$ and $\omega _ { b | a }$ , and admit the pair to exactly one of two kinds of merge. The ratios themselves decide which one. If at least one of $\rho _ { a \to b }$ and $\rho _ { b \to a } \mathrm { i s } \geq 1 $ , that side reaches the other at least as well as it reaches itself and has no boundary of its own relative to the other; the two groups are not peers, and the pair is examined for absorption only. If both ratios are below 1, each group is self-contained relative to the other; the two are peers, and the pair is examined for bonding only.

1. Absorption. Group $\mathbf { g } _ { s }$ is absorbed by group $\mathbf { g } _ { w }$ when

$$
\begin{array} { r } { \rho _ { s \to w } \geq 1 \qquad \mathrm { a n d } \qquad \rho _ { w \to s } > \operatorname * { m i n } \bigl ( \bar { \phi } _ { w } ^ { - s } , 0 . 9 9 \bigr ) , } \end{array}\tag{11}
$$

where

$$
\bar { \phi } _ { w } ^ { - s } = \frac { 1 } { | \mathbf { G } _ { \mathbf { g } _ { w } } | } \sum _ { \mathbf { x } \in \mathbf { G } _ { \mathbf { g } _ { w } } } \frac { 1 } { \mu _ { \mathbf { G } _ { \mathbf { g } _ { w } } } ( \mathbf { x } ) } \cdot \frac { 1 } { | \mathcal { B } | } \sum _ { \mathbf { z } \in \mathcal { B } } \mathrm { r s } _ { \mathbf { x } } ( \mathbf { x } , \mathbf { z } ) , \qquad \mathcal { B } = \mathbf { X } \setminus ( \mathbf { G } _ { \mathbf { g } _ { w } } \cup \mathbf { G } _ { \mathbf { g } _ { s } } ) ,\tag{12}
$$

is the background level of $\mathbf { g } _ { w }$ after $\mathbf { g } _ { s }$ has been left out of the reference set $B .$ If ${ \bf { g } } _ { s }$ is large and nearby it lifts $\bar { \phi } _ { w }$ , so $\bar { \phi } _ { w } ^ { - s }$ averages only over $B .$ . The reachable similarities are the same rs used for $\bar { \phi } _ { w } \colon$ they come from the full data, and $\bar { \phi } _ { w } ^ { - s }$ differs from $\bar { \phi } _ { w }$ only in which points enter the outer average. $\rho _ { s \to w } \geq 1$ means $\mathbf { g } _ { s }$ reaches $\mathbf { g } _ { w }$ at least as well as it reaches itself, hence has no boundary of its own relative to $\mathbf { g } _ { w } . \rho _ { w \to s } >$ min $( \bar { \phi } _ { w } ^ { - s } , 0 . 9 9 )$ means $\mathbf { g } _ { w }$ reaches $\mathbf { g } _ { s }$ more than it reaches the rest of the data with g<sub>s</sub> left out; the 0.99 is the same cap already used for $\phi _ { i } . \mathrm { A }$ group that could be absorbed by several neighbors goes with the one maximizing $\rho _ { s \to w }$

2. Bonding. Two peers, i.e., groups with $\rho _ { a  b } < 1$ and $\rho _ { b \to a } < 1$ , bond when each side finds the other more reachable than its own threshold,

$$
\rho _ { a \to b } > Q _ { \mathrm { t h } } ( a \mid b ) \qquad \mathrm { a n d } \qquad \rho _ { b \to a } > Q _ { \mathrm { t h } } ( b \mid a ) ,\tag{13}
$$

with $Q _ { \mathrm { t h } }$ as in (10). Once either group has internal links, bonding is subject to one further condition, that it creates no new weakest link. Let $\mathbf { b } ( \mathbf { g } )$ be the weakest edge of the maximum spanning tree of $\mathbb { G }$ restricted to the local centers of g, i.e., the weakest group-level similarity on which the internal connectivity of g already relies $( \mathrm { b } ( \mathbf { g } ) = \infty$ for a singleton). When at least one of the two bottlenecks is finite, the bond is accepted only if their contact is not weaker than what either of them already tolerates,

$$
\operatorname* { m a x } _ { \mathbf { c } _ { i } \in \mathbf { g } _ { a } , \mathbf { c } _ { j } \in \mathbf { g } _ { b } } \mathbb { G } _ { i , j } \ \geq \ q \operatorname* { m i n } \{ \mathrm { b } ( \mathbf { g } _ { a } ) , \mathrm { b } ( \mathbf { g } _ { b } ) \} ,\tag{14}
$$

with the same slack $q$ as in (10). When both groups are singletons, both bottlenecks are infinite and (14) is not imposed. Condition (14) is what keeps elongated structures from being chained through a dip in similarity while letting fragments of one structure, whose mutual contacts are as strong as their internal ones, be reunited.

## 4.2.3. Community-level contact

Conditions (11) and (13) compare two groups with each other and with their own background, but they do not account for the size of the groups relative to the whole data set. The same reachability ratios are more likely to reflect genuine structure when the two groups are small, whereas for two large groups the same ratios can arise simply because their size makes substantial contact statistically expected; merging in the latter case carries a much higher risk. To correct for this size effect we compare the observed contact between two groups with the contact expected from their size alone, using a standard null model of graph community structure.

Let D be the depth graph on $\mathbf { X } \colon$ every observation $\mathbf { x } _ { u }$ is linked to its $r _ { u }$ most similar observations, where $r _ { u } = | \mathbf { G } _ { t } |$ for the group $\mathbf { G } _ { t }$ containing $\mathbf { x } _ { u }$ , the edge weights are the depth-based similarities stored in $\mathbb { S } ,$ and the graph is symmetrized by taking, for each pair, the larger of the two directed weights. For two groups $\mathbf { G } _ { i }$ and $\mathbf { G } _ { j }$ let $e _ { i j }$ be the total weight of the edges between them, $k _ { i }$ and $k _ { j }$ the total weights incident to them, and m the total weight of D. The modularity gain of joining the two groups [42] is

$$
\Delta Q _ { i j } = \frac { e _ { i j } } { m } - \frac { k _ { i } k _ { j } } { 2 m ^ { 2 } } .\tag{15}
$$

$\Delta Q _ { i j } > 0$ means that the two groups share more weight than the configuration model, which places edges at random given the weight of every observation, would predict for two groups of their size. We call such a pair a community-level contact. A merge of ${ \bf g } _ { a }$ and $\mathbf { g } _ { b }$ , whether absorption or bond, is accepted only if at least one pair of groups $\mathbf { G } _ { i } \subseteq \mathbf { G } _ { \mathbf { g } _ { a } } , \mathbf { G } _ { j } \subseteq \mathbf { G } _ { \mathbf { g } _ { b } }$ is a community-level contact. The test is carried out at the level of the initial groups $\mathbf { G } _ { t }$ and not at the level of ${ \bf g } _ { a }$ and $\mathbf { g } _ { b }$ as wholes, because the modularity of a partition has a resolution limit [43]: the gain of joining two large communities is negative even when they are genuinely connected, whereas the gain between two adjacent initial groups is not affected by the size of the groups they have grown into. One kind of absorption is exempt from the test. When $\omega _ { s | w } = 0$ , no point of $\mathbf { g } _ { s }$ loses intra-group reachability in the union, so g is not a community adjacent to $\mathbf { g } _ { w }$ but a part of its reachability basin; the question the null model asks does not arise.

## 4.2.4. Iteration and reconsideration of small groups

The rule is applied to all adjacent pairs of the current groups at once; the new groups are the connected components of the accepted absorptions and bonds. The pass is repeated on the new groups, with $\mathbf { G _ { g } } , \bar { \phi } ,$ ω and $\mathrm { b } ( \cdot )$ recomputed for the merged groups, until no merge is accepted. Nothing changes between the first pass over singletons and the later passes over groups except that condition (14) only becomes active once a group has internal links.

Finally, small groups are reconsidered. For each group g<sub>k</sub> let $\begin{array} { r } { \mathcal { T } _ { k } = \bigcup _ { t : \mathbf { c } _ { t } \in \mathbf { g } _ { k } } \mathbf { G } _ { t } } \end{array}$ be its temporary cluster, where the $\mathbf { G } _ { t }$ are recomputed over the retained local centers only, so that the observations of groups that were set aside in Section 4.2.1 are included. Every $\mathcal { T } _ { i }$ with fewer points than half the median cluster size is paired with the cluster $\tau _ { j }$ it is most similar to, where the similarity between two groups of local centers is max $\tilde { \mathbb { G } } _ { u , v } \mathbb { G } _ { u , v }$ over their local centers, balancing direct and reachable similarity. Denoting the union by ${ \mathcal { T } } _ { i j } = { \mathcal { T } } _ { i } \cup { \mathcal { T } } _ { j }$ , the merge is accepted if the union is not more connected to the outside than the looser of the two parts already was,

$$
\begin{array} { r } { \rho _ { \mathcal { T } _ { i j } \to \mathbf { X } \backslash \mathcal { T } _ { i j } } \ \le \ \operatorname* { m a x } \{ \rho _ { \mathcal { T } _ { i } \to \mathbf { X } \backslash \mathcal { T } _ { i } } , \ \rho _ { \mathcal { T } _ { j } \to \mathbf { X } \backslash \mathcal { T } _ { j } } \} , } \end{array}\tag{16}
$$

provided the union is not larger than the largest current cluster and the pair contains a community-level contact. A small group that passes (16) is a fragment whose points are better described as part of a neighbor than as a mode of their own; a small group that fails it is kept, however small. The step is repeated until no small group moves. The complete procedure is summarized in Algorithm 2.

The rule has no strategy switch and no data-set-specific branch. Absorption is what the “min” strategy of DLCC does when it attaches a non-representative local center to a representative one, and bonding is what the “max” strategy does when it links fragments of a connected shape; the two are decided pair by pair from the same quantities, so a data set can contain both situations at once. Figure 4 in Section 4.3 illustrates the resulting groups on two toy examples, a data set of interlocked shapes and one of eleven normal components, in which the two kinds of merge dominate respectively.

Algorithm 2 Adaptive grouping of local centers   
Input: Local centers $\mathbf { C } = \{ \mathbf { c } _ { 1 } , \hdots , \mathbf { c } _ { T } \}$ , point groups $\mathbf { G } _ { 1 } , \hdots , \mathbf { G } _ { T } ,$ similarity matrix S, group-level similarity matrix G and its   
reachable version $\tilde { \mathbb { G } } ,$ slack $q = 0 . 9 7 5 .$   
Output: Groups of local centers g $1 , \ldots , \mathbf { g } _ { K }$ and temporary clusters $\mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { K }$   
1: If no local center lies in the β-neighborhood of another one with β = max $| \mathbf { G } _ { t } | / 2 n ,$ , return the singletons.   
2: Compute $\bar { \phi } _ { t }$ for all groups; set aside the groups with $\bar { \phi } _ { t } \geq \mathrm { ~ . ~ }$ 1 that meet the criteria of Section 4.2.1; the remaining groups are the   
units.   
3: Build the depth graph D and compute $\Delta Q _ { i j } \ \mathrm { b y } \left( 1 5 \right)$ for all adjacent pairs of units.   
4: Initialize $\mathbf { g } _ { 1 } , \ldots ,$ g<sub>K</sub> as the singletons of the units.   
5: repeat   
6: For every adjacent pair $( \mathbf { g } _ { a } , \mathbf { g } _ { b } )$ compute $\rho _ { a \to b } , \rho _ { b \to a }$ and $\omega _ { a | b } , \omega _ { b | a }$ on $\mathbf { G } _ { \mathbf { g } _ { a } }$ and $\mathbf { G } _ { \mathbf { g } _ { b } }$   
7: Among the pairs with ma $\mathfrak { c } ( \rho _ { a  b } , \rho _ { b  a } ) \geq 1$ 1, mark as absorptions those satisfying (11), each absorbed group keeping only   
the neighbor it reaches best; the other pairs of this kind are left unmerged.   
8: Among the pairs with max $\ : ( \rho _ { a  b } , \rho _ { b  a } ) \ : < \ : 1 \ :$ , mark as bonds those satisfying (13) and, when either group has internal   
links, (14).   
Discard marked pairs without a community-level contact, unless the pair is an absorption with $\omega _ { s | w } = 0 .$   
10: Replace the groups by the connected components of the marked pairs; recompute ϕ<sup>¯</sup> and b(·) for the new groups.   
11: until no pair is marked   
12: Assign every observation to its most similar retained local center; form $\begin{array} { r } { \mathcal { T } _ { k } = \bigcup _ { t : \mathbf { c } _ { t } \in \mathbf { g } _ { k } } \mathbf { G } _ { t } } \end{array}$   
13: while some $\begin{array} { r } { | \mathcal { T } _ { i } | < \frac { 1 } { 2 } } \end{array}$ median $( | T _ { 1 } | , \dots , | T _ { K } | )$ has not been examined do   
14: Pair $\tau _ { i }$ with the cluster $\tau _ { j }$ maximizing ma $\begin{array} { r } { \mathrm { { c } } _ { \mathbf { c } _ { u } \in \mathbf { g } _ { i } , \mathbf { c } _ { v } \in \mathbf { g } _ { j } } \tilde { \mathbb { G } } _ { u , v } \mathbb { G } _ { u , v } . } \end{array}$   
15: Merge g<sub>i</sub> into g<sub>j</sub> if (16) holds, $| \mathcal { T } _ { i j } | \overset { - } { \leq }$ ma $\mathrm { x } _ { k } \mid \mathcal { T } _ { k } \mid$ and the pair contains a community-level contact; recompute the cluster   
sizes.   
16: end while

## 4.3. Temporary clusters

In the DLCC framework, the next step is to construct temporary clusters by partitioning the data into two parts: observations that can be confidently clustered, and those left unlabeled. The former are grouped into what we term temporary clusters. Unlike the original DLCC, which relies on unique neighbors based on a fixed locality parameter, here we define retained points, a parameter-free alternative, to construct temporary clusters.

Definition 4.4 (Retained Points). Let $\mathbf { g } _ { 1 } , \ldots , \mathbf { g } _ { K }$ denote K groups of local centers, and let $\mathbf { G } _ { t }$ denote the group ofpoints assigned to local center $\mathbf { c } _ { t }$ as defined in (4),for $t = 1 , \ldots , T .$ . Define the initial temporary cluster as $\begin{array} { r } { \mathcal { T } _ { k } = \bigcup _ { t : \mathbf { c } _ { t } \in \mathbf { g } _ { k } } \mathbf { G } _ { t } . \ A } \end{array}$ point $\mathbf { x } _ { p } \in \mathbf { G } _ { t } \subset \mathcal { T } _ { k }$ is called a retained point of T<sub>k</sub> if

$$
\operatorname* { m a x } _ { \mathbf { x } _ { q } \in \mathcal { T } _ { k } \backslash \mathbf { G } _ { t } } \mathrm { D S } ( \mathbf { x } _ { p } , \mathbf { x } _ { q } ) > \operatorname* { m a x } _ { \mathbf { x } _ { h } \in \bigcup _ { j \neq k } \mathcal { T } _ { j } } \mathrm { D S } ( \mathbf { x } _ { p } , \mathbf { x } _ { h } ) ,
$$

meaning that the maximum similarity between $\mathbf { x } _ { p }$ and any point in $\mathcal { T } _ { k } \setminus \mathbf { G } _ { t }$ exceeds its maximum similarity with points in other temporary clusters.

Definition 4.4 assumes that g<sub>k</sub> contains multiple local centers, which may not always be the case. We handle singleton groups $\mathbf { g } _ { k } = \{ \mathbf { c } _ { u } \}$ as follows. Recall that $\mathcal { E } ^ { \prime } = \mathcal { E } \setminus \mathbf { C }$ , as defined in Remark 3.1, consists of the locally deep points not selected as local centers. Consider points $\mathbf { z } \in { \mathcal { E } } ^ { \prime }$ that are more similar to $\mathbf { c } _ { u }$ than to any other local center, i.e., $\begin{array} { r } { \mathrm { D S } ( \mathbf { z } , \mathbf { c } _ { u } ) > \mathrm { m a x } _ { \mathbf { c } _ { t } \in \mathbf { C } \backslash \mathbf { c } _ { u } } \mathrm { D S } ( \mathbf { z } , \mathbf { c } _ { t } ) } \end{array}$ . For each such point, we compute its depth with respect to $\mathbf { G } _ { u }$ and weight it by a frequency-based coefficient

$$
D _ { f } ( \mathbf { z } ) = \frac { | \mathcal { E } ^ { \prime } | - \sum _ { \mathbf { x } \in \mathcal { E } ^ { \prime } } \mathbb { I } \left[ f _ { \mathbf { x } } > f _ { \mathbf { z } } \right] } { | \mathcal { E } ^ { \prime } | } D ( \mathbf { z } \mid \mathbf { G } _ { u } ) ,\tag{17}
$$

where $f _ { \mathbf { x } }$ is the frequency of $\mathbf { x } .$ The frequency-based weight favors candidates that are more frequently identified as the locally deep point across varying neighborhoods, while the depth value quantifies points centrality within $\mathbf { G } _ { u } .$ A high depth value helps reduce the risk of including a locally deep point that is ambiguous or potentially misassigned to this group. The point z maximizing $D _ { f }$ then serves as a local center and is added to the singleton group $\mathbf { g } _ { k }$ to ensure that Definition 4.4 is well-defined.

The remaining steps follow the temporary cluster update procedure of the “min strategy” from the original DLCC algorithm [26] with minor adaptations to accommodate the nonparametric setting, using the same score function. The score for assigning a point $\mathbf { x } _ { i }$ to the kth temporary cluster is defined as

$$
\mathrm { s c o r e } _ { i \vert k } = \frac { \mathrm { m a x } _ { \mathbf { c } _ { j } \in \mathbf { g } _ { k } } \mathrm { D S } ( \mathbf { c } _ { j } , \mathbf { x } _ { i } ) - \mathrm { m a x } _ { \mathbf { c } _ { t } \in \mathbf { C } \backslash \mathbf { g } _ { k } } \mathrm { D S } ( \mathbf { c } _ { t } , \mathbf { x } _ { i } ) } { \mathrm { m a x } \{ \mathrm { m a x } _ { \mathbf { c } _ { j } \in \mathbf { g } _ { k } } \mathrm { D S } ( \mathbf { c } _ { j } , \mathbf { x } _ { i } ) , \mathrm { m a x } _ { \mathbf { c } _ { t } \in \mathbf { C } \backslash \mathbf { g } _ { k } } \mathrm { D S } ( \mathbf { c } _ { t } , \mathbf { x } _ { i } ) \} } .\tag{18}
$$

The full procedure is summarized in Algorithm 3, where, instead of relying on a predefined neighborhood size parameter, the output temporary cluster $\mathcal { T } _ { k }$ is required to retain at least half of its initial size.

![](images/034259c4dffe19a88cd6c310c2cc7206f1706857b01b9c2aad8e4a6338aea739.jpg)  
(a)

![](images/7d75f7d0c4a40045edbee0692138ce91d02bbcda0ca031d3fb26d9e536dd6a51.jpg)  
(b)  
Figure 4. Temporary cluster results for two toy examples. (a) Bainba, where the groups are formed by bonding fragments of the interlocked shapes; (b) Normal, eleven normal components, where the groups are formed by absorption of the less representative local centers. Colors indicate retained points in each temporary cluster, while black points are unlabelled and left to be classified.

Figure 4 illustrates the temporary clusters in the two toy examples. Most points are correctly clustered, while ambiguous points—typically those near cluster boundaries—remain unlabelled. The remaining points

```latex
Algorithm 3 Generate and update temporary clusters
Input: Groups of local centers $\left\{ \mathbf { g } _ { 1 } , \ldots , \mathbf { g } _ { K } \right\}$ , groups of points $\mathbf { G } _ { 1 } , \hdots , \mathbf { G } _ { T } ,$ , similarity matrix $\mathbb { S } ,$ locally deep points $\mathcal { E } ^ { \prime } .$
Output: Temporary clusters $\mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { K }$
1: For any singleton $\mathbf { g } _ { k } = \{ \mathbf { c } _ { u } \}$ , add a locally deep point z with the highest $D _ { f } ( \mathbf { z } \mid \mathbf { G } _ { s } )$ value; update $T ,$ and current groups of
points $\mathbf { \bar { G } } _ { 1 } , \dots , \mathbf { \bar { G } } _ { T } .$
2: Initialize ${ \dot { \mathcal { T } } } _ { k } \gets \bigcup _ { t : { \mathbf { c } } _ { t } \in { \mathbf { g } } _ { k } } { \mathbf { G } } _ { t }$ for $k = 1 , \ldots , K$
3: Update each $\mathcal { T } _ { k }$ using Definition 4.4, let remaining points be $\hat { \mathcal { T } } _ { k } .$
4: Generate score pools $\scriptstyle { S _ { k } }$ and $\hat { S } _ { k }$ for points in each $\bar { \tau _ { k } }$ and $\hat { \mathcal { T } } _ { k }$ by (18).
5: Set α $ \{ \underset { \mathrm { 0 } } { \mathrm { m e a n } } ( \bigcup _ { k = 1 } ^ { K } \hat { S } _ { k } ) , \ : \ : \ : \mathrm { i f } \ : \bigcup _ { k = 1 } ^ { K } \hat { S } _ { k } \neq \emptyset $
0,
6: for k = 1 to K do
7: $\mathsf { s i z e } \gets | S _ { k } | + | \hat { S } _ { k } |$
8: if $| \hat { S } _ { k } | > | S _ { k } |$ then
9: $\mathrm { S e t } \hat { \alpha } _ { 1 } \gets 0 . 5 \alpha$
10: else
11: Se $\hat { \alpha } _ { 1 }  \alpha$
12: end if
13: for each $i \in \mathcal { T } _ { k }$ with score $< \hat { \alpha } 1$ do
14: ${ \mathcal { T } } _ { k } \gets { \mathcal { T } } _ { k } \setminus \{ i \} , \quad { \hat { \mathcal { T } } } _ { k } \gets { \hat { \mathcal { T } } } _ { k } \cup \{ i \}$ , and update $\scriptstyle { S _ { k } }$ and $\hat { S } _ { k }$ accordingly.
15: end for
16: Let SS ← sorted scores in $S _ { k }$ (descending)
17: $\begin{array} { r } { i d x \gets \arg \operatorname* { m a x } _ { j = \mathrm { f l o o r } ( | S S | / 2 ) \mathrm { t o } | S S | - 1 } \tilde { ( S S [ j ] - S S [ j + 1 ] ) } } \end{array}$
18: $\begin{array} { r } { \hat { \alpha } _ { 2 } \gets \mathrm { m i n } ( S S [ i d x ] , 1 - \frac { { s \mathrm { i } z \mathrm { e } / 2 - | S S | } } { | \hat { S } _ { k } | } } \end{array}$ quantile of $\hat { S } _ { k } )$
19: for each $u \in \hat { \mathcal { T } } _ { k }$ with score > αˆ do
20: $\hat { \mathcal { T } } _ { k } \gets \hat { \mathcal { T } } _ { k } \setminus \{ u \} , ~ \mathcal { T } _ { k } \gets \mathcal { T } _ { k } \cup \{ u \}$
21: end for
22: end for
```

are then classified using classical methods.

## 5. Applications

We first compare A-DLCC with DLCC on several synthetic datasets, then with other methods considered “adaptive” or “automatic” on real datasets.

## 5.1. Synthetic data

We evaluate A-DLCC and DLCC on four synthetic datasets: Starbeam (from [26]), Blend and Bainba (generated using the mlbench R package [44]), and Agg (from [45]). As shown in Figure 5, A-DLCC achieves similar performance to DLCC, but without parameter tuning. There is one notable exception. The Agg dataset contains two pairs of shapes connected by a “bridge”. DLCC separates both pairs, whereas A-DLCC separates only the pair joined by the thinner bridge (lower left) and treats the pair joined by the wider bridge (right) as a single cluster. This outcome is primarily because the merging criterion in A-DLCC is based on reachable similarity, and when there is a path connecting points in the two shapes with a density comparable to that inside the shapes, the reachable similarities between points in the two shapes are comparable to those within each shape; the disruption ω is then small on both sides and the bond (13) is accepted. Additionally, without a fixed locality parameter, a local center can be defined within the bridge if, at a certain locality level, its neighborhood contains a balanced mix of points from both shapes. For the spiral structures in Blend, DLCC and A-DLCC also show minor differences; however, neither method fully separates the two spirals (see the discussion in [26] for further details).

![](images/5a27793922bc543bc0e560a5a36b581bc838c9d179b5e2e384b899b1f7716689.jpg)  
(a) DLCC (Starbeam)

![](images/52253d93bf100702b0b66a6fffad6aa24c4ba95869e4352f8d1b865a9274a57a.jpg)  
(b) A-DLCC (Starbeam)

![](images/763765343c7398a2e925d7353bf51edccf30810ff821b7d16b61fb36a4e9be30.jpg)  
(c) DLCC (Blend)

![](images/7df3e7c2aaf87af7488eaac10165b5a64b30a556b8375fb928e1a644bee32da3.jpg)  
(d) A-DLCC (Blend)

![](images/46ccb787a8e7375450489bc315d1de512c090361140507bc3b7dcc268bc3bced.jpg)  
(e) DLCC (Agg)

![](images/bffb0c2ded1bc44c073c478822671c5772d13c8d0e370733987c5c096050b464.jpg)  
(f) A-DLCC (Agg)

![](images/424f906149610707eeb11cad6e8e323e90d5ca92e794a365d82bea526186baad.jpg)  
(g) DLCC (Bainba)

![](images/00baa968c3b9b5fc50301d8bc1d53ea8ca6687839aa4b520babe383c892e135f.jpg)  
(h) A-DLCC (Bainba)  
Figure 5. Comparison of clustering results by DLCC and A-DLCC on different synthetic datasets.

## 5.2. Real data

We now proceed to experiments on real datasets. Specifically, we consider the eight datasets summarized in Table 3. Iris, Seed, Wine, Pa, BC, and Seg are widely used UCI benchmark datasets for clustering evaluation. Yale-B and Optidigits are included in the R package PPCI [46]; the former is a face dataset containing

2000 images of 10 individuals, which is a subset from [47] compressed to $3 0 \times 2 0$ pixels, and the latter is a handwritten digit dataset.

Table 3. Summary of the datasets
<table><tr><td>Dataset</td><td>Instances</td><td>Attributes</td><td>Clusters</td></tr><tr><td>Iris</td><td>150</td><td>4</td><td>3</td></tr><tr><td>Seed</td><td>210</td><td>7</td><td>3</td></tr><tr><td>Wine</td><td>178</td><td>13</td><td>3</td></tr><tr><td>Parkinsons (Pa)</td><td>195</td><td>22</td><td>2</td></tr><tr><td>Breast Cancer (BC)</td><td>569</td><td>30</td><td>2</td></tr><tr><td>Segmentation (Seg)</td><td>2086</td><td>18</td><td>7</td></tr><tr><td>Yale-B</td><td>2000</td><td>600</td><td>10</td></tr><tr><td>Optidigits</td><td>5620</td><td>64</td><td>10</td></tr></table>

## 5.2.1. External Metrics

To assess clustering performance, we adopt three widely used external evaluation metrics: adjusted Rand index (ARI) [48], normalized mutual information (NMI) [49], and purity [50].

The ARI measures pairwise similarity between clustering results and ground truth, with an expected value of 0 under random class assignment and a value of 1 under perfect agreement. As it penalizes both over-segmentation and under-segmentation, ARI is considered a balanced and widely adopted metric in clustering research. NMI quantifies the shared information between the predicted and true clusterings, normalized to [0, 1]. NMI remains robust even when the number of clusters is smaller than the ground truth, but may not penalize over-segmentation as strongly as ARI. Purity measures the fraction of points in each cluster that belong to the best-matching true class, averaged over all clusters. Its value also ranges from 0 to 1. It is particularly informative when the number of clusters exceeds the true number of classes, as it reflects optimal matching to ground truth labels. However, purity tends to favor over-segmentation.

## 5.2.2. Comparison algorithms and experimental setup

For method comparison, we first include DLCC to compare against A-DLCC, since A-DLCC is designed as an automatic, parameter-free alternative to DLCC. Note that DLCC itself is not an automatic clustering method, and we use it as a reference method with optimally tuned parameters (assuming the true number of clusters is known) to assess whether A-DLCC can achieve comparable or even superior performance without parameter tuning. Beyond this, we include representative algorithms from several major categories: (1) mean-shift type, (2) spectral clustering with automatic estimation of the number of clusters, (3) hierarchical clustering, and (4) Kmeans-type algorithms with automatic splitting and merging. Specifically, the selected methods are NN-RMS [16] (mean-shift), Spectrum [20] (spectral), FINCH [18] (hierarchical), and U-Kmeans [23] (Kmeans type).

All these methods are capable of estimating the number of clusters, denoted by ${ \hat { K } } ,$ , and are compared against the ground truth K. However, most of them still require additional parameters to be specified for optimal performance. For NN-RMS, which requires a neighborhood parameter k, we follow the recommendation in [16] and test k values from 6 to 50, reporting the best result. For Spectrum, we evaluate both the eigengap and multimodality gap methods for determining the number of clusters, and present the better one. For FINCH, which provides multiple clustering results at different levels, we select the result whose number of clusters is closest to the true K. For U-Kmeans, although the algorithm is designed to be parameter-free, its objective function includes a tuning parameter γ. The original paper suggests setting $\gamma = \exp ( - \hat { K } / 2 5 0 )$ , whereas the published demo code uses $\gamma = \exp ( - \hat { K } / 4 5 0 )$ . We test both settings and report the better outcome. For methods involving randomness, we report the best result among 10 independent runs. To avoid any ambiguity, all “best” or “better” results in this section refer to those achieving the highest ARI, which may not coincide with the best values for NMI or purity.

For A-DLCC, it is important to note that the construction of temporary clusters does not require any parameter tuning. However, the final step of the DLCC framework involves classifying the remaining observations, which may require parameters depending on the classification method used. Following the original DLCC paper, we employ two simple yet effective classification methods: depth-based kNN for the Seed, BC, and Optidigits datasets, and random forests for all remaining datasets. To fairly demonstrate the effectiveness of the A-DLCC framework, we do not tune the parameters of these classifiers to optimize performance; for kNN, the number of neighbors is fixed at 7, and for random forests, the number of trees is fixed at 100. Since a random forest depends on its random seed, the classification step of A-DLCC is repeated with 100 seeds on every data set that uses random forests, and the mean over the 100 runs is reported.

## 5.2.3. Evaluation

Table 4 compares the clustering performance of the A-DLCC method with DLCC, as well as several other automatic clustering algorithms, across a range of real datasets [51, 46]. The values in parentheses indicate the percentages of points clustered and corresponding metric scores before the classification step. As shown, the temporary clusters generally achieve strong performance, justifying the last classification step. As expected, A-DLCC achieves performance comparable to DLCC, which relies on parameter selection. Notably, in some datasets, A-DLCC slightly outperforms the parameterized DLCC and even achieves purity of 1.000 in the Yale-B data. A-DLCC matches or exceeds the other automatic methods in both estimated cluster number and clustering accuracy, except on Seg.

Figures 6, 7 and 8 visualize the A-DLCC clustering results alongside the ground truth labels using tdistributed stochastic neighbor (tSNE) embeddings [52], a nonlinear dimensionality reduction technique that projects high-dimensional data into a low-dimensional space for visualization. Figure 6 shows datasets with clusters that are nearly visually separated and balanced in size, where the local centers identified by A-DLCC generally coincide with the geometric centers of the clusters. Figure 7 presents the Pa and BC datasets, both exhibiting clear cluster size imbalance and some degree of overlap. Here, the unlabelled points in the temporary clustering panels are mostly found near overlapping regions and cluster boundaries. Figure 8 highlights that for complex datasets such as Yale-B, Optidigits, and Seg, A-DLCC identifies local centers corresponding to visually separated point groups, even when these groups are relatively small. In contrast, the ground truth labels often group such small, well-separated point groups together with larger point collections. This behavior reflects the fact that A-DLCC does not assume cluster size balance when defining local centers. Seg is the exception, where several local centers fall where two or three classes overlap. The central groups are then joined because each reaches the next about as well as itself. DLCC uses a comparatively large neighborhood size and so does not define a local center in that overlap. Across all datasets, the A-DLCC clusters are visually coherent under tSNE.

Table 4. Comparison of clustering results for real datasets. For each dataset, the best value for each metric is shown in bold. “–” in K<sup>ˆ</sup> indicates the true K is provided; <sup>∗</sup> indicates K is selected from the method’s output. For A-DLCC, values in parentheses indicate the temporary clustering coverage (in the K<sup>ˆ</sup> row) and the corresponding temporary clustering ARI, Purity, and NMI (in each metric row). For the data sets classified with random forests (Iris, Wine, Pa, Seg, Yale-B), the final A-DLCC values are means over 100 random seeds.
<table><tr><td>Dataset</td><td>Metric</td><td>A-DLCC</td><td>DLCC</td><td>Spectrum</td><td>NN-RMS</td><td>FINCH</td><td>U-Kmeans</td></tr><tr><td rowspan="4">Iris</td><td>K</td><td>3 (79.3%)</td><td>1</td><td>3</td><td>3</td><td>3*</td><td>3</td></tr><tr><td>ARI</td><td>0.846 (0.936)</td><td>0.878</td><td>0.851</td><td>0.746</td><td>0.886</td><td>0.746</td></tr><tr><td>Purity</td><td>0.945 (0.975)</td><td>0.957</td><td>0.945</td><td>0.900</td><td>0.960</td><td>0.900</td></tr><tr><td>NMI</td><td>0.830 (0.905)</td><td>0.858</td><td>0.832</td><td>0.798</td><td>0.871</td><td>0.798</td></tr><tr><td rowspan="4">Seed</td><td>K</td><td>3 (77.1%)</td><td></td><td>3</td><td>3</td><td>3*</td><td>3</td></tr><tr><td>ARI</td><td>0.762 (0.948)</td><td>0.775</td><td>0.596</td><td>0.766</td><td>0.701</td><td>0.732</td></tr><tr><td>Purity</td><td>0.914 (0.981)</td><td>0.919</td><td>0.838</td><td>0.914</td><td>0.886</td><td>0.905</td></tr><tr><td>NMI</td><td>0.714 (0.916)</td><td>0.731</td><td>0.631</td><td>0.729</td><td>0.678</td><td>0.723</td></tr><tr><td rowspan="4">Wine</td><td>K</td><td>3 (77.0%)</td><td></td><td>3</td><td>3</td><td>3*</td><td>3</td></tr><tr><td>ARI</td><td>0.911 (0.978)</td><td>0.930</td><td>0.917</td><td>0.816</td><td>0.686</td><td>0.895</td></tr><tr><td>Purity</td><td>0.970 (0.993)</td><td>0.978</td><td>0.972</td><td>0.938</td><td>0.888</td><td>0.966</td></tr><tr><td>NMI</td><td>0.889 (0.968)</td><td>0.911</td><td>0.883</td><td>0.809</td><td>0.714</td><td>0.865</td></tr><tr><td rowspan="4">Pa</td><td>K</td><td>2 (70.3%)</td><td></td><td>4</td><td>2</td><td>6*</td><td>2</td></tr><tr><td>ARI</td><td>0.397 (0.394)</td><td>0.422</td><td>0.106</td><td>0.289</td><td>0.036</td><td>-0.098</td></tr><tr><td>Purity</td><td>0.839 (0.839)</td><td>0.856</td><td>0.862</td><td>0.928</td><td>0.754</td><td>0.754</td></tr><tr><td>NMI</td><td>0.242 (0.245)</td><td>0.320</td><td>0.292</td><td>0.289</td><td>0.032</td><td>0.098</td></tr><tr><td rowspan="4">BC</td><td>K</td><td>2 (78.7%)</td><td>一</td><td>2</td><td>3</td><td>3*</td><td>2</td></tr><tr><td>ARI</td><td>0.786 (0.964)</td><td>0.748</td><td>0.701</td><td>0.673</td><td>0.647</td><td>0.659</td></tr><tr><td>Purity</td><td>0.944 (0.991)</td><td>0.933</td><td>0.919</td><td>0.931</td><td>0.933</td><td>0.901</td></tr><tr><td>NMI</td><td>0.726 (0.931)</td><td>0.655</td><td>0.579</td><td>0.582</td><td>0.559</td><td>0.546</td></tr><tr><td rowspan="4">Seg</td><td>K</td><td>12 (84.3%)</td><td></td><td>7</td><td>12</td><td>12*</td><td>4</td></tr><tr><td>ARI</td><td>0.254 (0.293)</td><td>0.589</td><td>0.361</td><td>0.510</td><td>0.479</td><td>0.368</td></tr><tr><td>Purity</td><td>0.549 (0.597)</td><td>0.766</td><td>0.569</td><td>0.716</td><td>0.657</td><td>0.531</td></tr><tr><td>NMI</td><td>0.545 (0.588)</td><td>0.673</td><td>0.562</td><td>0.624</td><td>0.619</td><td>0.581</td></tr><tr><td rowspan="4">Yale-B</td><td>K</td><td>11 (74.8%)</td><td></td><td>9</td><td>14</td><td>12*</td><td>7</td></tr><tr><td>ARI</td><td>0.975 (0.974)</td><td>0.990</td><td>0.757</td><td>0.847</td><td>0.798</td><td>0.437</td></tr><tr><td>Purity</td><td>1.000 (1.000)</td><td>0.995</td><td>0.790</td><td>0.957</td><td>0.845</td><td>0.570</td></tr><tr><td>NMI</td><td>0.986 (0.986)</td><td>0.990</td><td>0.900</td><td>0.895</td><td>0.895</td><td>0.612</td></tr><tr><td rowspan="4">Optidigits</td><td>K</td><td>13 (74.4%)</td><td></td><td>9</td><td>14</td><td>11*</td><td>3</td></tr><tr><td>ARI</td><td>0.834 (0.901)</td><td>0.814</td><td>0.653</td><td>0.780</td><td>0.000</td><td>0.231</td></tr><tr><td>Purity</td><td>0.937 (0.970)</td><td>0.906</td><td>0.741</td><td>0.907</td><td>0.122</td><td>0.291</td></tr><tr><td>NMI</td><td>0.851 (0.912)</td><td>0.839</td><td>0.776</td><td>0.841</td><td>0.080</td><td>0.451</td></tr></table>

## 5.3. A case study on Anuran Calls

Anuran Calls [53] is a widely used, highly unbalanced dataset in the clustering literature. It contains 7195 audio syllables from 10 frog species, each described by 22 variables. The cluster sizes range from as few as 68 to as many as 3478 observations, making it a particularly challenging scenario for unsupervised algorithms.

Table 5 reports the performance of A-DLCC and other adaptive clustering baselines mentioned earlier.

![](images/190dde7954320d8b4f489d9363f6c6947e77db2edb0d12522a94d0144e231d21.jpg)  
Figure 6. t-SNE visualizations for the Iris (top), Wine (middle), and Seed (bottom) datasets. For each dataset, the three panels represent the true labels, the temporary clusters identified by A-DLCC with local centers marked by black triangles, and the final A-DLCC clustering results, respectively.

A-DLCC significantly outperforms all competing methods in terms of ARI and NMI, and attains the highest purity as well. It achieves an ARI above 0.9 despite estimating a higher number of clusters $( \hat { K } = 1 6 )$ than the ground truth. Although Spectrum yields the value of $\hat { K }$ that is the closest to the true number, it fails to detect small classes. For example, the smallest species $( n \ = \ 6 8 )$ is not separated out. Methods like NN-RMS and FINCH achieve high purity but much lower ARI, since they over-split the largest class (the species with 3478 observations) into multiple clusters.

Figure 9 visualizes the A-DLCC result and compares it with the ground truth. The two largest species are recovered almost perfectly: of the 3555 observations in the largest cluster, 3468 belong to the largest species (total 3478 observations), and 988 of the 1121 observations of the second largest species fall in one cluster. The local centers of the largest species, which occupy a long, dense region of the embedding, are bonded into one group because each of them reaches its neighbors about as well as it reaches itself and the neighbors are community-level contacts on the depth graph; the same rule keeps the small species on the periphery apart, because their background level is low and their contacts with the large groups are weaker than the null model expects. The higher number of clusters comes from the medium-sized species:

![](images/9a7ededd9d5ac96218c744ffc8e8130b8c520af0815ea2ad5b9dafa6ac924257.jpg)  
Figure 7. t-SNE visualizations for the Pa (top) and BC (bottom) datasets. Panel meanings are as in Figure 6.

the species with 672 and 542 observations are each split into two to four clusters that correspond to visually separated point groups in the tSNE embedding, and the three smallest species (114, 68 and 148 observations) are partly gathered in one cluster. Since the large classes are pure and the splits concern comparatively few observations, the ARI remains high. A-DLCC recovers both the dominant and the smaller structures, even with a larger estimated number of clusters.

The tendency of A-DLCC to estimate a larger number of clusters in high-dimensional, complex datasets is consistent with our previous discussion. Without strong assumptions about data structure or cluster size, A-DLCC naturally identifies small groups of points that are not similar to any major clusters as separate clusters. This is a consequence of the fully automatic design.

Table 5. Comparison of clustering results on Anuran Calls; the final A-DLCC values are means over 100 random-forest seeds.
<table><tr><td>Dataset</td><td>Metric</td><td>A-DLCC</td><td>Spectrum</td><td>NN-RMS</td><td>FINCH</td><td>U-Kmeans</td></tr><tr><td rowspan="4">Anuran Calls</td><td>K</td><td>16 (92.9%)</td><td>7</td><td>19</td><td>13*</td><td>2</td></tr><tr><td>ARI</td><td>0.918 (0.952)</td><td>0.590</td><td>0.394</td><td>0.232</td><td>0.551</td></tr><tr><td>Purity</td><td>0.927 (0.963)</td><td>0.647</td><td>0.894</td><td>0.776</td><td>0.636</td></tr><tr><td>NMI</td><td>0.806 (0.872)</td><td>0.550</td><td>0.647</td><td>0.552</td><td>0.597</td></tr></table>

## 6. Conclusions and future directions

In this paper, we proposed the A-DLCC algorithm, which does not require tuning parameters and can automatically adapt to a wide range of clustering scenarios. A-DLCC outperforms the other automatic clustering methods tested on real high-dimensional data, including cases with balanced and unbalanced clusters.

![](images/92a4f162c25f4558d7bfed7b6888652bc03dd701b364ecd3f439429da35c85aa.jpg)

Figure 8. t-SNE visualizations for the Seg (top), Yale-B (middle), and Optidigits (bottom) datasets. Panel meanings are as in Figure 6.  
![](images/b29ecd2d191caf18279775e7b0c88e5f1b0beaae9775a976a9d6823a3c31801e.jpg)  
Figure 9. t-SNE visualization of Anuran Calls: ground truth, temporary clusters with local centers (triangles), and the A-DLCC result.

Nevertheless, for some complex high-dimensional datasets, A-DLCC tends to estimate a larger number of clusters than the ground truth, particularly when the underlying structure is ambiguous or contains small, weakly connected groups. The same challenge applies to nearly all unsupervised clustering methods. Except for simple or synthetic datasets with clear cluster structure and labels, it is almost impossible to recover the “true” number of clusters without prior knowledge in many real-world cases. As noted by Jeon et al. [54], ground truth labels themselves may correspond to ambiguous or even arbitrary groupings, rather than a well-defined clustering. Such labels may not be reflected by the structures that are visible in the data.

For adaptive clustering methods, what matters most is that the results are meaningful and interpretable without prior knowledge of the number of clusters. Under A-DLCC, for example, ambiguous small clusters may indicate outliers within larger groups or reveal overlapping regions involving several distinct classes, and may point to targets for manual inspection or domain-specific analysis. Several limitations and open problems remain.

Computational scalability is a concern. While the original DLCC had $O ( n ^ { 3 } )$ complexity, A-DLCC can be even more demanding. For example, as the nonparametric design considers all neighborhood sizes, computing β-ILD requires $O ( n ^ { 2 } )$ operations, and the worst-case complexity of constructing G is $O ( n ^ { 3 } )$ . Developing faster estimation methods for local depth remains an important challenge for all methodologies that rely on it.

Beyond computational issues, theoretical understanding of the merging process also deserves further investigation. The grouping rule of A-DLCC is built from quantities with a clear meaning (the relative reachability ratio, the background level, the disruption and the modularity gain), and it applies the same rule to every pair of groups without a strategy choice; nevertheless, the way the threshold (10) combines these quantities, and the value of the slack $q ,$ remain heuristic rather than being derived from a model. Ideally, one would prefer a more principled approach where merging decisions are guided by a well-defined objective function. However, to date, no universally accepted objective function exists that performs well across both convex and non-convex cluster shapes and in varying dimensionalities. Developing such an objective function or internal criterion remains an open challenge, and could support more principled model selection across different clustering algorithms.

## References

[1] A. H. Liu, H.-J. Chang, M. Auli, W.-N. Hsu, and J. Glass, “Dinosr: Self-distillation and online clustering for self-supervised speech representation learning,” Advances in Neural Information Processing Systems, vol. 36, pp. 58346–58362, 2023.

[2] E. Diday, G. Govaert, Y. Lechevallier, and J. Sidi, “Clustering in pattern recognition,” in Digital Image Processing: Proceedings of the NATO Advanced Study Institute held at Bonas, France, June 23-July 4, 1980, pp. 19–58, Springer, 1981.

[3] H. Mittal, A. C. Pandey, M. Saraswat, S. Kumar, R. Pal, and G. Modwel, “A comprehensive survey of image segmentation: clustering methods, performance parameters, and benchmark datasets,” Multimedia Tools and Applications, pp. 1–26, 2022.

[4] O. Kisi, S. Heddam, K. S. Parmar, A. Petroselli, C. Külls, and M. Zounemat-Kermani, “Integration of gaussian process regression and k means clustering for enhanced short term rainfall runoff modeling,” Scientific Reports, vol. 15, no. 1, p. 7444, 2025.

[5] S. Lloyd, “Least squares quantization in pcm,” IEEE transactions on information theory, vol. 28, no. 2, pp. 129–137, 1982.

[6] S. C. Johnson, “Hierarchical clustering schemes,” Psychometrika, vol. 32, no. 3, pp. 241–254, 1967.

[7] M. Ester, H.-P. Kriegel, J. Sander, X. Xu, et al., “A density-based algorithm for discovering clusters in large spatial databases with noise,” in kdd, vol. 96, pp. 226–231, 1996.

[8] A. Rodriguez and A. Laio, “Clustering by fast search and find of density peaks,” science, vol. 344, no. 6191, pp. 1492–1496, 2014.

[9] C. Fraley and A. E. Raftery, “Model-based clustering, discriminant analysis, and density estimation,” Journal of the American statistical Association, vol. 97, no. 458, pp. 611–631, 2002.

[10] A. Ng, M. Jordan, and Y. Weiss, “On spectral clustering: Analysis and an algorithm,” Advances in neural information processing systems, vol. 14, 2001.

[11] R. Souvenir and R. Pless, “Manifold clustering,” in Tenth IEEE International Conference on Computer Vision (ICCV’05) Volume 1, vol. 1, pp. 648–653, IEEE, 2005.

[12] A. K. Jain, “Data clustering: 50 years beyond k-means,” Pattern recognition letters, vol. 31, no. 8, pp. 651–666, 2010.

[13] P. J. Rousseeuw, “Silhouettes: a graphical aid to the interpretation and validation of cluster analysis,” Journal of Computational and Applied Mathematics, vol. 20, pp. 53–65, 1987.

[14] G. Schwarz, “Estimating the dimension of a model,” The Annals of Statistics, vol. 6, no. 2, pp. 461– 464, 1978.

[15] K. Fukunaga and L. Hostetler, “The estimation of the gradient of a density function, with applications in pattern recognition,” IEEE Transactions on information theory, vol. 21, no. 1, pp. 32–40, 1975.

[16] C. Cariou, S. Le Moan, and K. Chehdi, “A novel mean-shift algorithm for data clustering,” IEEE Access, vol. 10, pp. 14575–14585, 2022.

[17] G. Beck, T. Duong, M. Lebbah, H. Azzag, and C. Cérin, “A distributed approximate nearest neighbors algorithm for efficient large scale mean shift clustering,” Journal of Parallel and Distributed Computing, vol. 134, pp. 128–139, 2019.

[18] S. Sarfraz, V. Sharma, and R. Stiefelhagen, “Efficient parameter-free clustering using first neighbor relations,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8934–8943, 2019.

[19] U. Von Luxburg, “A tutorial on spectral clustering,” Statistics and computing, vol. 17, pp. 395–416, 2007.

[20] C. R. John, D. Watson, M. R. Barnes, C. Pitzalis, and M. J. Lewis, “Spectrum: fast density-aware spectral clustering for single and multi-omic data,” Bioinformatics, vol. 36, no. 4, pp. 1159–1166, 2020.

[21] J. Fan, Y. Tu, Z. Zhang, M. Zhao, and H. Zhang, “A simple approach to automated spectral clustering,” Advances in Neural Information Processing Systems, vol. 35, pp. 9907–9921, 2022.

[22] P. Dan, “Extending k-means with efficient estimation of the number of clusters,” in Proceedings of the 17th International Conf. on Machine Learning, ICML-2000, pp. 727–734, 2000.

[23] K. P. Sinaga and M.-S. Yang, “Unsupervised k-means clustering algorithm,” IEEE access, vol. 8, pp. 80716–80727, 2020.

[24] L. Mahon and M. Lapata, “K\*-means: A parameter-free clustering algorithm,” arXiv preprint arXiv:2505.11904, 2025.

[25] L. McInnes, J. Healy, and J. Melville, “Umap: Uniform manifold approximation and projection for dimension reduction,” arXiv preprint arXiv:1802.03426, 2018.

[26] S. Wang, A. Leblanc, and P. D. McNicholas, “Depth-based local center clustering: A framework for handling different clustering scenarios,” arXiv preprint arXiv:2505.09516, 2025.

[27] S. Wang, A. Leblanc, and P. D. McNicholas, “β-integrated local depth and corresponding partitioned local depth representation,” arXiv preprint arXiv:2506.14108, 2025.

[28] K. S. Berenhaut, K. E. Moore, and R. L. Melvin, “A social perspective on perceived distances reveals deep community structure,” Proceedings of the National Academy of Sciences, vol. 119, no. 4, p. e2003634119, 2022.

[29] P. Mozharovskyi, Data depth: computation, applications, and beyond. PhD thesis, Institut Polytechnique de Paris, 2022.

[30] G. Francisci, C. Agostinelli, A. Nieto-Reyes, and A. N. Vidyashankar, “Analytical and statistical properties of local depth functions motivated by clustering applications,” Electronic Journal of Statistics, vol. 17, no. 1, pp. 688–722, 2023.

[31] C. Agostinelli and M. Romanazzi, “Local depth,” Journal of Statistical Planning and Inference, vol. 141, no. 2, pp. 817–830, 2011.

[32] D. Paindaveine and G. Van Bever, “From depth to local depth: a focus on centrality,” Journal of the American Statistical Association, vol. 108, no. 503, pp. 1105–1119, 2013.

[33] T. J. W., “Mathematics and the picturing of data,” Proceedings of the International Congress of Mathematicians, Vancouver, 1975, vol. 2, pp. 523–531, 1975.

[34] Y. Zuo and R. Serfling, “General notions of statistical depth function,” Annals of statistics, pp. 461– 482, 2000.

[35] R. Y. Liu, “On a notion of data depth based on random simplices,” The Annals of Statistics, pp. 405– 414, 1990.

[36] K. Mosler and P. Mozharovskyi, “Choosing among notions of multivariate depth statistics,” Statistical Science, vol. 37, no. 3, pp. 348–368, 2022.

[37] R. Serfling, “A depth function and a scale curve based on spatial quantiles,” in Statistical Data Analysis Based on the L 1-Norm and Related Methods, pp. 25–38, Springer, 2002.

[38] P. D. McNicholas, A. ElSherbiny, A. F. McDaid, and T. B. Murphy, pgmm: Parsimonious Gaussian Mixture Models, 2021. R package version 1.2.5.

[39] M. Maechler, P. Rousseeuw, A. Struyf, M. Hubert, and K. Hornik, cluster: Cluster Analysis Basics and Extensions, 2023. R package version 2.1.4.

[40] H. N. Gabow and R. E. Tarjan, “Algorithms for two bottleneck optimization problems,” Journal of Algorithms, vol. 9, no. 3, pp. 411–417, 1988.

[41] P. Chebotarev, “The graph bottleneck identity,” Advances in Applied Mathematics, vol. 47, no. 3, pp. 403–413, 2011.

[42] M. E. J. Newman and M. Girvan, “Finding and evaluating community structure in networks,” Physical Review E, vol. 69, no. 2, p. 026113, 2004.

[43] S. Fortunato and M. Barthélemy, “Resolution limit in community detection,” Proceedings of the National Academy of Sciences, vol. 104, no. 1, pp. 36–41, 2007.

[44] F. Leisch, E. Dimitriadou, K. Hornik, et al., “mlbench: Machine learning benchmark problems,” R package version 2.1-5, 2024. https://CRAN.R-project.org/package=mlbench.

[45] A. Gionis, H. Mannila, and P. Tsaparas, “Clustering aggregation,” ACM Transactions on Knowledge Discovery from Data, vol. 1, no. 1, pp. 1–30, 2007.

[46] D. P. Hofmeyr and N. G. Pavlidis, “PPCI: an R package for cluster identification using projection pursuit,” The R Journal, 2019.

[47] A. S. Georghiades, P. N. Belhumeur, and D. J. Kriegman, “From few to many: Illumination cone models for face recognition under variable lighting and pose,” IEEE transactions on pattern analysis and machine intelligence, vol. 23, no. 6, pp. 643–660, 2002.

[48] L. Hubert and P. Arabie, “Comparing partitions,” Journal of classification, vol. 2, pp. 193–218, 1985.

[49] A. Strehl and J. Ghosh, “Cluster ensembles—a knowledge reuse framework for combining multiple partitions,” Journal of machine learning research, vol. 3, no. Dec, pp. 583–617, 2002.

[50] Y. Zhao and G. Karypis, “Criterion functions for document clustering: Experiments and analysis,” Tech. Rep. 01-40, Department of Computer Science, University of Minnesota, 2001.

[51] D. Dua and C. Graff, “Uci machine learning repository,” University of California, Irvine, School of Information and Computer Sciences, 2019.

[52] L. Van der Maaten and G. Hinton, “Visualizing data using t-sne.,” Journal of machine learning research, vol. 9, no. 11, 2008.

[53] J. J. Diaz, J. G. Colonna, R. B. Soares, C. M. Figueiredo, and E. F. Nakamura, “Compressive sensing for efficiently collecting wildlife sounds with wireless sensor networks,” in 2012 21st International Conference on Computer Communications and Networks (ICCCN), pp. 1–7, IEEE, 2012.

[54] H. Jeon, M. Aupetit, D. Shin, A. Cho, S. Park, and J. Seo, “Measuring the validity of clustering validation datasets,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.