# Multi-View Fair Clustering Guided by Cross-View Sensitive Information Discrepancy

Mudi Jiang<sup>a</sup>, Jiahui Zhou<sup>a</sup>, Xinying Liu<sup>a</sup>, Zengyou He<sup>a</sup> and Zhikui Chen<sup>a,∗</sup>

<sup>a</sup>School ofSoftware, Dalian University ofTechnology, Dalian, Liaoning, China

A R T I C L E I N F O

Keywords: Fair clustering Multi-view clustering Unsupervised learning Group fairness

## A BS T R AC T

Multi-view clustering (MVC) aims to uncover latent cluster structures by exploiting complementary information from multiple views. Despite substantial progress in clustering performance, fairness remains an important concern when MVC is applied to socially sensitive scenarios. Recent fair multi-view clustering methods have introduced fairness constraints into representation learning or clustering assignments. However, these methods generally treat diferent views under a largely uniform fairness mechanism, without explicitly distinguishing their varying levels of sensitive dependence during cross-view learning. In practice, diferent views may encode substantially diferent levels of sensitive information. Ignoring such cross-view discrepancy can allow highly sensitive-dependent views to influence less sensitive-dependent ones during crossview learning, potentially degrading both clustering performance and fairness. To address this issue, we propose a novel multi-view fair clustering framework guided by cross-view sensitive information discrepancy. Specifically, we estimate the sensitive dependence of each view and develop a bias-ranked asymmetric alignment mechanism that encourages views with higher sensitive dependence to learn from those with lower sensitive dependence, while cross-view discrepancies are further exploited to adaptively regulate the alignment process. Moreover, fairness regularization is imposed on the consensus soft assignments to further promote group fairness. Extensive experiments on benchmark datasets demonstrate that the proposed method achieves a favorable balance between clustering quality and group fairness. Furthermore, robustness analysis under heterogeneous sensitive dependence demonstrates that the proposed method is more robust to cross-view sensitive-information heterogeneity than existing multiview fair clustering approaches, maintaining more stable fairness while preserving competitive clustering performance.

## 1. Introduction

Multi-view data [26] describe the same set of instances from multiple perspectives, providing complementary information that cannot be fully captured by a single view. Such data are particularly valuable for clustering analy-<sup>.</sup> sis [28], where latent structures must be uncovered without label supervision. Multi-view clustering [10] therefore seeks to integrate complementary information across views to obtain more informative and reliable clustering structures. Existing studies have investigated a wide range of techniques, including subspace learning, matrix factorization, graphbased modeling, and deep representation learning, and have achieved substantial advances in clustering performance. Nevertheless, conventional multi-view clustering methods are primarily designed to optimize clustering efectiveness, with limited consideration of whether the resulting partitions treat diferent demographic groups equitably. This limitation is particularly critical in socially sensitive applications such as financial services, healthcare, and personne management, where clustering outcomes may influence downstream decisions and potentially reinforce disparities associated with sensitive attributes.

Fairness has consequently received increasing attention in clustering research [6]. Depending on the level at which equitable treatment is defined, existing fairness notions are generally studied from individual and group perspectives. Individual fairness [15] emphasizes local consistency, requiring instances with similar characteristics to receive comparable clustering outcomes. In contrast, group fairness [9] focuses on the distribution of clustering outcomes across populations defined by sensitive attributes, with the aim of preventing specific demographic groups from being systematically favored or disadvantaged. This notion is closely related to the principle of disparate impact [31], which concerns disparities in outcomes across protected groups even in the absence of explicitly discriminatory rules. Since many practical clustering applications are primarily concerned with population-level disparities, this work focuses on group fairness in the multi-view setting.

In recent years, fairness has received increasing attention in multi-view clustering [44, 42, 20, 36, 43, 13]. Existing methods mainly extend conventional multi-view clustering frameworks by introducing fairness regularization into representation learning or cluster assignments. However, they generally do not explicitly distinguish the varying levels of sensitive dependence across views, although diferent views may encode considerably diferent amounts of sensitive information. When such heterogeneous views are fused or encouraged to interact without accounting for these diferences, information associated with sensitive attributes from views with higher sensitive dependence may influence those with lower sensitive dependence, potentially weakening low-dependence representations and compromising both clustering quality and fairness.

To address the above issue, we propose a novel multi-view fair clustering framework that explicitly models cross-view discrepancies in sensitive information and leverages them to guide the training process. Specifically, the framework first employs view-specific autoencoders to learn latent representations for individual views, based on which the dependence between each view and the sensitive attribute is quantified. These estimates are then used to guide an asymmetric alignment mechanism, in which views with higher sensitive dependence are encouraged to learn from those with lower sensitive dependence. Meanwhile, pairwise and global cross-view discrepancies are further exploited to determine the relative importance and adaptive strength of the alignment. In this manner, cross-view interaction is explicitly regulated according to the heterogeneous levels of sensitive information encoded by diferent views, thereby mitigating the propagation of sensitive information while preserving useful complementary structures. Finally, fairness regularization is imposed on the consensus soft assignments aggregated across views to further promote group fairness at the clustering-assignment level. By jointly optimizing representation learning, discrepancy-guided asymmetric alignment, and assignment-level fairness, the proposed framework seeks to achieve a favorable trade-of between clustering quality and fairness.

To evaluate the proposed framework, we conduct extensive experiments on five benchmark multi-view datasets with fairness constraints. The results show that our method achieves a favorable balance between clustering quality and group fairness, with competitive performance compared with state-of-the-art multi-view clustering and fairness-aware clustering methods. Moreover, the robustness analysis under heterogeneous view bias demonstrates that the proposed framework can better cope with cross-view sensitive-information heterogeneity, maintaining more stable fairness while preserving competitive clustering performance under increasing sensitive dependence in the target view.

The main contributions of this paper are summarized as follows:

• We first investigate the cross-view sensitive information discrepancy problem in multi-view fair clustering, explicitly considering the heterogeneous dependence of diferent views on sensitive attributes.

• We develop a joint optimization framework that integrates discrepancy-guided asymmetric view alignment, multi-view clustering consistency learning, and fairness regularization on consensus assignments, jointly promoting discriminative representations, consistent clustering structures, and group-fair assignments.

• Extensive experiments on five benchmark datasets with fairness constraints demonstrate that the proposed method achieves a favorable balance between clustering quality and group fairness. Moreover, the robustness analysis under heterogeneous view bias further shows that our method is better able to handle cross-view sensitive-information heterogeneity than existing fair multi-view clustering methods, maintaining more stable fairness while preserving competitive clustering performance under increasing sensitive dependence in the target view.

The rest of this paper is organized as follows. Section 2 summarizes the related studies. Section 3 presents the proposed method. Section 4 provides the experimental results. Finally, Section 5 concludes this work.

## 2. Related work

## 2.1. Fair Clustering

## 2.1.1. Individualfairness

Individual fairness considers fairness from the perspective of local consistency among data instances. Its underlying principle is that samples exhibiting similar characteristics in the original feature space should receive comparable clustering outcomes, regardless of their sensitive-group memberships. Rather than assessing fairness solely at the population level, individual fairness imposes constraints on the relative treatment of neighboring or structurally similar instances. Jung et al. [15] introduced an early formulation based on distance-aware fairness constraints that relate clustering decisions to the local structure of the data. Building on this foundation, subsequent studies developed bicriteria approximation algorithms that jointly control clustering quality and fairness violations [27]. Later work further improved the approximation guarantees for diferent clustering objectives, including �-means and �- median [4, 34]. Meanwhile, several studies have focused on reducing the computational complexity of individually fair clustering, thereby improving its applicability to large-scale datasets [8, 2].

## 2.1.2. Group fairness

Group fairness, in contrast, focuses on disparities among populations defined by sensitive attributes, such as race or gender. Its primary objective is to prevent the resulting clusters from exhibiting systematically imbalanced or discriminatory distributions across sensitive groups. Existing approaches generally incorporate fairness at diferent stages of the clustering pipeline

Pre-processing methods modify or augment the input data prior to clustering so that the transformed data better satisfy fairness requirements. Representative strategies include fairlet-based decomposition, which partitions the dataset into small balanced subsets [9, 1], and fair coreset construction, which compresses the original dataset while preserving both clustering structure and fairness properties [32, 12]. Other studies introduce carefully designed antidote samples to steer the subsequent clustering process toward fairer solutions [7].

In-processing approaches incorporate fairness directly into the clustering objective or optimization procedure. This category includes fairness-constrained extensions of classical clustering algorithms, such as spectral clustering and �- median clustering [17, 33], as well as deep clustering frameworks that jointly optimize clustering quality and fairness through adversarial learning or multi-objective optimization [41, 19]. In contrast, post-processing methods leave the original clustering procedure unchanged and instead refine the obtained solution, for example by reassigning samples or adjusting cluster centers to improve fairness [16, 14].

Overall, existing fair clustering research has developed a broad spectrum of mechanisms for promoting fairness at either the individual or group level. However, most existing methods are primarily designed for single-view data, and their applicability to multi-view clustering remains relatively limited.

## 2.2. Multi-View Clustering

## 2.2.1. Traditional multi-view clustering

Multi-view clustering aims to uncover a common clustering structure by jointly exploiting the complementary and shared information embedded in multiple views. Existing studies have addressed this problem from various perspectives. Graph-based methods typically construct view-specific similarity graphs to characterize neighborhood or structural relationships within each view and subsequently integrate them into a unified representation for clustering [22, 24]. Another representative line of research is subspace learning, which projects heterogeneous views into a shared low-dimensional space to enhance common information while suppressing view-specific noise [21, 38]. Matrix factorization-based methods pursue a similar objective by decomposing multi-view observations into compact latent factors, from which the underlying clustering structure can be inferred [35, 37]. To capture more complex relationships among samples, kernel-based methods employ nonlinear mappings and integrate multiple kernel representations to model cross-view dependencies [25, 39]. More recently, deep multi-view clustering has attracted increasing attention. By leveraging neural networks to learn nonlinear high-level representations, these methods enable representation learning and clustering to be optimized in a unified manner, ofering greater flexibility in modeling heterogeneous multi-view data [23, 11].

Despite their efectiveness in exploiting cross-view complementarity, most conventional multi-view clustering methods primarily focus on improving clustering consistency and representation quality, while fairness is rarely considered as an explicit learning objective. Consequently, the learned representations and clustering assignments may still reflect information associated with sensitive attributes, potentially leading to unequal clustering outcomes across diferent demographic groups. This limitation has motivated increasing research interest in incorporating fairness considerations into the multi-view clustering process.

## 2.2.2. Fairness-related multi-view clustering

Recent studies have extended multi-view clustering by explicitly incorporating fairness into representation learning, cluster assignment, or multi-view fusion. Fair-MVC [44] introduces group-level fairness constraints by encouraging balanced proportions of protected groups across clusters. DFMVC [42] further integrates contrastive representation learning with fairness-aware assignment regularization, guiding subgroup distributions toward a predefined target while preserving cross-view consistency. Xu et al. [36] enhance fair representation learning by incorporating Kolmogorov–Arnold networks into the multi-view clustering framework, with the aim of capturing more complex feature dependencies while improving both robustness and fairness.

Other studies promote fairness through clustering structures or optimization mechanisms. FMSC [20] develops a spectral clustering formulation that encourages group fairness by regulating the connectivity patterns of protectedgroup subgraphs within clusters. AFMVC [13] adopts an adversarial learning strategy to suppress sensitive information in view-specific latent representations, employing a discriminator with gradient reversal to reduce the predictability of sensitive attributes while preserving clustering quality. More recently, FLFMVC [43] introduces a fairness-aware late-fusion framework that jointly considers robustness, demographic balance, and fair cluster assignment through fairness-aware graph filtering, group-normalized regularization, and fairness-guided discretization.

Despite these advances, existing multi-view fair clustering methods mainly introduce fairness constraints into conventional multi-view clustering frameworks, while paying limited attention to the heterogeneity of sensitive information across views. In practice, diferent views may exhibit markedly diferent degrees of dependence on sensitive attributes, and conventional fusion or consistency learning may therefore facilitate the propagation of sensitive information across views, potentially compromising both clustering quality and fairness. Motivated by this limitation, we explicitly model cross-view discrepancies in sensitive information to guide asymmetric view alignment, while further imposing fairness regularization on the consensus assignments.

## 3. Method

![](images/96b0d5f61980c85bfa62d985e78938725f49bf5956debe79048d440ff1af2af0.jpg)  
Figure 1: Overview of the proposed multi-view fair clustering framework.

In this section, we introduce the overall architecture of the proposed multi-view fair clustering framework. As illustrated in Fig. 1, the framework comprises three main components: Multi-view Representation Learning, Bias-Ranked Asymmetric View Alignment, and Fairness Regularization on Consensus Assignments. The first component learns informative view-specific representations to capture complementary clustering structures across multiple views. The second explicitly accounts for heterogeneous sensitive dependence across views and performs asymmetric contrastive alignment, whereby views with higher sensitive dependence are guided by those with lower sensitive dependence. The third further enhances fairness at the assignment level by imposing a demographic parity constraint on the consensus soft assignments aggregated across views. Finally, the learned view-specific representations are aggregated to derive the final clustering assignments.

## 3.1. Notations and Problem Definition

Let $\mathcal { X } = \{ X ^ { \left( v \right) } \} _ { v = 1 } ^ { V }$ denote a multi-view dataset comprising � instances observed from � distinct views, where $X ^ { ( v ) } \in \mathbb { R } ^ { N \times d _ { v } }$ is the feature matrix of the �-th view and $d _ { v }$ denotes its feature dimensionality. Correspondingly, $X _ { i } ^ { ( v ) }$ represents the feature vector of the �-th instance in view �. Each instance is further associated with a sensitive attribute $S _ { i } ~ \in ~ S$ , where  denotes the set of sensitive groups. Given a predefined number of clusters �, the objective of multi-view fair clustering is to exploit the complementary information across multiple views to uncover meaningful cluster structures while mitigating disparities among diferent sensitive groups.

## 3.2. Multi-View Representation Learning and Clustering

To capture complementary information across multiple views while preserving their view-specific characteristics, we employ an independent autoencoder for each view. Specifically, the encoder $E ^ { ( v ) }$ maps $X ^ { ( v ) }$ into a latent representation:

$$
Z ^ { ( v ) } = E ^ { ( v ) } ( X ^ { ( v ) } ; \theta _ { E } ^ { ( v ) } ) ,\tag{1}
$$

where $Z ^ { ( v ) } \in \mathbb { R } ^ { N \times d _ { \tilde { z } } }$ � denotes the latent representation of view �, and $\theta _ { E } ^ { ( v ) }$ denotes the parameters of the corresponding encoder.

The resulting latent representation is then fed into a view-specific decoder ${ \cal { D } } ^ { ( v ) }$ to reconstruct the original input:

$$
X _ { R } ^ { ( v ) } = D ^ { ( v ) } ( Z ^ { ( v ) } ; \theta _ { D } ^ { ( v ) } ) ,\tag{2}
$$

where $X _ { R } ^ { ( v ) }$ denotes the reconstructed feature matrix and $\theta _ { D } ^ { ( v ) }$ denotes the parameters of the corresponding decoder. To retain the intrinsic information contained in each view, we minimize the reconstruction discrepancy between the original and reconstructed features:

$$
\mathcal { L } _ { R } = \sum _ { v = 1 } ^ { V } \left\| X ^ { ( v ) } - X _ { R } ^ { ( v ) } \right\| _ { F } ^ { 2 } .\tag{3}
$$

Before joint optimization, the view-specific autoencoders are initialized through reconstruction pretraining, which provides a stable representation space for subsequent clustering and fairness learning.

Based on the learned latent representations, we further construct a shared clustering structure across views. Specifically, the latent representations from all views are concatenated to form a unified multi-view representation:

$$
H = [ Z ^ { ( 1 ) } ; Z ^ { ( 2 ) } ; \cdots ; Z ^ { ( V ) } ] .\tag{4}
$$

�-means is then applied to obtain global pseudo cluster labels, which provide a common clustering target for all views. Let $Y \in \{ 0 , 1 \} ^ { N \times K }$ denote the corresponding one-hot pseudo-label matrix, where $Y _ { i k } = 1$ indicates that instance � is assigned to cluster �. During training, the pseudo labels are periodically updated according to the current multi-view representation, allowing the shared clustering structure to progressively adapt to the learned feature space.

For each view �, we introduce � learnable cluster centroids $\{ \mu _ { k } ^ { ( v ) } \} _ { k = 1 } ^ { K }$ and construct a view-specific soft assignment matrix $Q ^ { ( v ) } \in \mathbb { R } ^ { N \times K }$ using a Student’s �-distribution kernel:

$$
Q _ { i k } ^ { ( v ) } = \frac { ( 1 + \| Z _ { i } ^ { ( v ) } - \mu _ { k } ^ { ( v ) } \| _ { 2 } ^ { 2 } / \alpha ) ^ { - ( \alpha + 1 ) / 2 } } { \displaystyle \sum _ { k ^ { \prime } = 1 } ^ { K } ( 1 + \| Z _ { i } ^ { ( v ) } - \mu _ { k ^ { \prime } } ^ { ( v ) } \| _ { 2 } ^ { 2 } / \alpha ) ^ { - ( \alpha + 1 ) / 2 } } .\tag{5}
$$

Here, $Q _ { i k } ^ { ( v ) }$ represents the soft probability of assigning instance � to cluster � in view �, and � denotes the degreeof-freedom parameter of the Student’s �-distribution, which is set to 1 in our experiments. The view-specific cluster

centroids are initialized by applying �-means to the corresponding latent representations and are subsequently optimized jointly with the network parameters.

To encourage each view to preserve a clustering structure consistent with the shared multi-view partition, we minimize the KL divergence between the global pseudo target � and each view-specific soft assignment $Q ^ { ( v ) }$ :

$$
\mathcal { L } _ { C } = \sum _ { v = 1 } ^ { V } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } Y _ { i k } \log ( Y _ { i k } / Q _ { i k } ^ { ( v ) } ) .\tag{6}
$$

By minimizing $\scriptstyle { \mathcal { L } } _ { C }$ , the fused representation � captures the shared clustering structure by integrating information from all views, while $Q ^ { ( v ) }$ characterizes the soft clustering behavior of each individual view. The learned latent representations $Z ^ { ( v ) }$ are subsequently used to estimate cross-view sensitive information discrepancy and guide asymmetric view alignment, while the soft assignments $Q ^ { ( v ) }$ are aggregated to construct the consensus assignment for fairness regularization.

## 3.3. Asymmetric View Alignment Guided by Sensitive Information Discrepancy

Diferent views may exhibit varying degrees of dependence on the sensitive attribute. Therefore, directly enforcing cross-view consistency without accounting for such diferences may allow sensitive information from views with higher sensitive dependence to influence those with lower sensitive dependence. To address this issue, we estimate the sensitive dependence of each view and leverage the resulting cross-view discrepancy to guide asymmetric representation alignment.

Given the latent representation $Z ^ { ( v ) }$ and the sensitive attribute �, we quantify the sensitive information contained in view � using the normalized Hilbert–Schmidt Independence Criterion (NHSIC):

$$
B ^ { ( v ) } = \mathrm { N H S I C } ( Z ^ { ( v ) } , S ) ,\tag{7}
$$

where a larger $\boldsymbol { B } ^ { ( v ) }$ indicates stronger statistical dependence between the representation of view � and the sensitive attribute. The resulting sensitive-dependence score is used to characterize the relative amount of sensitive information across views.

To perform cross-view alignment without directly constraining the clustering representation space, each latent representation is further mapped into a view-specific projection space:

$$
U ^ { ( v ) } = P ^ { ( v ) } ( Z ^ { ( v ) } ) ,\tag{8}
$$

where $P ^ { ( v ) }$ denotes the projection head associated with view �. For any pair of views $( h , l )$ satisfying

$$
B ^ { ( h ) } > B ^ { ( l ) } ,\tag{9}
$$

view ℎ is regarded as having relatively higher sensitive dependence, while view � exhibits lower sensitive dependence. Accordingly, we align the former toward the latter rather than treating the two views equally during cross-view alignment. To explicitly preserve this directional relationship, the representation of the lower-dependence view is treated as a fixed reference:

$$
\bar { U } ^ { ( l ) } = \mathrm { s g } ( U ^ { ( l ) } ) ,\tag{10}
$$

where sg(⋅) denotes the stop-gradient operation. The directed contrastive alignment between views ℎ and � is then defined as

$$
\mathcal { L } _ { h \to l } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { e ^ { ( \sin ( U _ { i } ^ { ( h ) } , \bar { U } _ { i } ^ { ( l ) } ) / \tau ) } } { \displaystyle \sum _ { j = 1 } ^ { N } e ^ { ( \sin ( U _ { i } ^ { ( h ) } , \bar { U } _ { j } ^ { ( l ) } ) / \tau ) } } ,\tag{11}
$$

where sim(⋅, ⋅) denotes cosine similarity and � is the temperature parameter. Representations of the same instance across the two views constitute positive pairs, whereas those of diferent instances are treated as negative pairs. By detaching the lower-dependence branch, only the higher-dependence branch is updated toward the reference representation.

Moreover, diferent view pairs may exhibit diferent levels of sensitive information discrepancy. For each directed pair $( h , l )$ , we therefore define a discrepancy-aware weight as

$$
w _ { h l } = \frac { B ^ { ( h ) } - B ^ { ( l ) } } { \displaystyle \sum _ { ( p , q ) \in \mathcal { A } } \left( B ^ { ( p ) } - B ^ { ( q ) } \right) } ,\tag{12}
$$

where

$$
\mathcal { A } = \left\{ ( h , l ) \vert B ^ { ( h ) } > B ^ { ( l ) } \right\}\tag{13}
$$

denotes the set of all directed high-to-low view pairs. Accordingly, view pairs with larger sensitive information discrepancies receive greater emphasis during alignment. The asymmetric alignment objective before adaptive calibration is formulated as

$$
\mathcal { L } _ { A } ^ { \mathrm { r a w } } = \sum _ { ( h , l ) \in \mathcal { A } } w _ { h l } \mathcal { L } _ { h  l } .\tag{14}
$$

To further characterize the overall heterogeneity of sensitive information across views, we define

$$
g = \operatorname* { m i n } \left( \frac { B _ { \mathrm { m a x } } - B _ { \mathrm { m i n } } } { \bar { B } + \epsilon } , 1 \right) ,\tag{15}
$$

where

$$
\bar { B } = \frac { 1 } { V } \sum _ { v = 1 } ^ { V } B ^ { ( v ) } ,\tag{16}
$$

with $B _ { \mathrm { m a x } } = \operatorname* { m a x } _ { v } B ^ { ( v ) }$ and $B _ { \mathrm { m i n } } = \mathrm { m i n } _ { v } B ^ { ( v ) }$ . Here, � is a small positive constant for numerical stability. A larger � indicates greater heterogeneity in sensitive information across views.

During optimization, � is further used to adaptively calibrate the efective contribution of the alignment objective relative to the clustering objective. A larger cross-view sensitive-information discrepancy leads to stronger alignment, whereas the alignment efect is reduced when diferent views exhibit similar levels of sensitive dependence. The resulting calibrated alignment objective is denoted as $\mathcal { L } _ { A }$ . In this way, cross-view sensitive information discrepancy determines the alignment direction, the relative importance of diferent view pairs, and the efective strength of representation alignment.

## 3.4. Fairness Regularization

Although the preceding asymmetric alignment mitigates sensitive information at the representation level, such representation-level regulation does not necessarily guarantee fairness in the final clustering assignments. In particular, residual sensitive dependence may still be reflected in the aggregated cluster distribution across diferent groups. To explicitly promote group fairness, we introduce an additional fairness regularization term based on the aggregated soft assignments across multiple views.

Given the view-specific soft assignment matrices $\{ Q ^ { ( v ) } \} _ { v = 1 } ^ { V }$ , we construct the consensus assignment as

$$
Q _ { \mathrm { c o n s } } = \frac { 1 } { V } \sum _ { v = 1 } ^ { V } Q ^ { ( v ) } .\tag{17}
$$

Here, $Q _ { \mathrm { c o n s } } \in \mathbb { R } ^ { N \times K }$ represents the aggregated clustering assignment shared across all views.

Let  denote the set of sensitive groups and

$$
I _ { g } = \left\{ i \mid S _ { i } = g \right\}\tag{18}
$$

denote the set of instances belonging to group �, with $N _ { g } = | I _ { g } |$ . For each sensitive group � and cluster �, the average assignment probability is defined as

$$
\bar { Q } _ { g k } = \frac { 1 } { N _ { g } } \sum _ { i \in I _ { g } } Q _ { \mathrm { c o n s } , i k } ,\tag{19}
$$

while the overall average assignment probability for cluster � is

$$
\bar { Q } _ { k } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } Q _ { \mathrm { c o n s } , i k } .\tag{20}
$$

Based on these soft assignments, we encourage each sensitive group to exhibit a cluster distribution close to the overall population distribution. Accordingly, the fairness regularization is formulated as

$$
\mathcal { L } _ { F } = \frac { 1 } { | S | K } \sum _ { g \in S } \sum _ { k = 1 } ^ { K } \left( \bar { Q } _ { g k } - \bar { Q } _ { k } \right) ^ { 2 } .\tag{21}
$$

By minimizing ${ \mathcal { L } } _ { F } ,$ , the expected assignment probability to each cluster is encouraged to be consistent across diferent sensitive groups, providing a diferentiable relaxation of demographic parity at the clustering-assignment level. Importantly, this fairness constraint is imposed on the consensus assignments rather than directly on individual latent representations, allowing the view-specific representations to retain information useful for clustering while jointly promoting fairness in the final partition.

## 3.5. Overall Objective Function

The proposed framework jointly considers representation preservation, clustering consistency, discrepancy-guided cross-view alignment, and assignment-level fairness. Accordingly, the overall objective is formulated as

$$
\mathcal { L } = \mathcal { L } _ { R } + \lambda _ { C } \mathcal { L } _ { C } + \mathcal { L } _ { A } + \lambda _ { F } \mathcal { L } _ { F } ,\tag{22}
$$

where $\mathcal { L } _ { R }$ preserves the intrinsic information of each view through reconstruction, $\scriptstyle { \mathcal { L } } _ { C }$ encourages view-specific assignments to follow the shared clustering structure, $\mathcal { L } _ { A }$ regulates cross-view representation alignment according to sensitive information discrepancy, and $\mathcal { L } _ { F }$ promotes demographic parity on the consensus assignments. The hyperparameters $\lambda _ { C }$ and $\lambda _ { F }$ control the contributions of the clustering and fairness objectives, respectively, while the strength of $\mathcal { L } _ { A }$ is adaptively determined by the cross-view sensitive information discrepancy as described above. The overall training procedure is shown in Algorithm 1.

Algorithm 1 Training Procedure of the Proposed Framework   
Input: Multi-view data $\boldsymbol { \mathcal { X } } = \{ \boldsymbol { X } ^ { ( v ) } \} _ { v = 1 } ^ { V } ;$ sensitive attribute �; number of clusters �; trade-of parameters $\lambda _ { C }$ and $\lambda _ { F } ;$   
temperature parameter �; number of training epochs �.   
Output: Final clustering assignments.   
1: Pre-train the view-specific autoencoders by minimizing $\mathcal { L } _ { R } .$   
2: Obtain the initial latent representations $\{ \bar { Z ^ { ( v ) } } \} _ { v = 1 } ^ { V }$ and initialize the view-specific cluster centroids.   
3: for � = 1 to � do   
4: Encode each view to obtain $Z ^ { ( v ) }$ and construct � using Eq. 4.   
5: Periodically apply �-means to � to update the global pseudo labels �.   
6: Compute the view-specific soft assignments $Q ^ { ( \stackrel { \smile } { v } ) }$ and the clustering loss $\scriptstyle { \mathcal { L } } _ { C }$ using Eq. 6.   
7: Estimate the sensitive dependence $\bar { B } ^ { ( v ) }$ of each view.   
8: Construct directed view pairs according to $\boldsymbol { B } ^ { ( v ) }$ and compute the asymmetric alignment loss $\mathcal { L } _ { A }$ using Eqs. 14   
and 15.   
9: Aggregate $\{ Q ^ { ( v ) } \} _ { v = 1 } ^ { V }$ into $Q _ { \mathrm { { c o n s } } }$ and compute the fairness loss $\mathcal { L } _ { F }$ using Eq. 21.   
10: Compute the overall objective according to Eq. 22 and update the network parameters.   
11: end for   
12: Apply �-means to � to obtain the final clustering assignments.

## 4. Experiments

## 4.1. Experimental Setup

## 4.1.1. Datasets

We evaluate the proposed method on five benchmark datasets commonly used in multi-view fair clustering, including Credit, Bank, Law, Mfeat, and COIL. The detailed statistics of these datasets are provided in Table 1.

For the single-view datasets Credit, Bank, and Law, we follow [44] to construct two complementary views by applying nonlinear transformations, such as Sigmoid and ReLU, to the original feature representations. For the multiview datasets Mfeat and COIL, we adopt the experimental setting in [40] and generate binary sensitive attributes by independently assigning each sample to one of two sensitive groups according to a Bernoulli distribution with parameter 0.5. To ensure a consistent and computationally feasible comparison across all competing methods, we further randomly sample 5,000 instances from Credit and 10,000 instances from Law, whose original sample sizes are 29,537 and 18,692, respectively. This subsampling strategy is necessary because several baseline methods, including FMSC and MCPL, fail to produce clustering results on datasets of this scale within feasible computational resources.

## Table 1

The summary statistics on datasets used in the performance evaluation.
<table><tr><td>Dataset</td><td>#Samples</td><td>#Clusters</td><td>#Features</td><td>Sensitive Attribute</td></tr><tr><td>Credit</td><td>5000</td><td>5</td><td>22/22</td><td>Gender</td></tr><tr><td>Bank</td><td>2907</td><td>2</td><td>12/12</td><td>Marital</td></tr><tr><td>Law</td><td>10000</td><td>2</td><td>10/10</td><td>Gender</td></tr><tr><td>Mfeat</td><td>2000</td><td>10</td><td>216/76/64/6/240/47</td><td>Synthetic Binary</td></tr><tr><td>COIL</td><td>1440</td><td>20</td><td>1021/3304/6750</td><td>Synthetic Binary</td></tr></table>

## 4.1.2. Evaluation metrics

We evaluate clustering performance using two widely adopted metrics, Clustering Accuracy (ACC) and Normalized Mutual Information (NMI). Both metrics measure the consistency between the obtained clustering assignments and the corresponding ground-truth labels. Larger values of ACC and NMI indicate better clustering performance.

To evaluate group fairness, we consider Balance (BAL) [44] and Demographic Parity Diference (DPD)<sup>1</sup> as grouplevel fairness measures. BAL quantifies the degree to which diferent sensitive groups are proportionally represented within each cluster, whereas DPD measures the disparity in clustering outcomes across sensitive groups. A higher BAL value indicates a more balanced group distribution, while a lower DPD value corresponds to a smaller demographic disparity and, consequently, better fairness.

## 4.1.3. Comparison methods

To comprehensively evaluate the proposed method, we compare it with representative approaches from three categories: single-view fair clustering, state-of-the-art multi-view clustering, and multi-view fair clustering. Since existing multi-view fair clustering methods remain relatively limited, we additionally include the first two categories to provide broader comparisons in terms of clustering performance, fairness, and their trade-of.

For single-view fair clustering methods, features from diferent views are concatenated into a unified representation before clustering, allowing these methods to be directly applied in the multi-view setting. BFKM [30] extends the conventional �-means objective by incorporating fairness and balance constraints as penalty terms and solves the resulting optimization problem via coordinate descent. VFC [45] adopts a variational formulation in which group disparity is characterized through a KL-divergence-based penalty, thereby enabling a flexible balance between clustering utility and demographic parity. FFC [29] employs a multi-stage optimization strategy consisting of fairnessaware initialization, relaxed constrained optimization, and local refinement. FairDen [18] incorporates group-level fairness constraints into a density-based spectral clustering framework constructed from density-connectivity distances, enabling the identification of arbitrarily shaped clusters while preserving group fairness.

State-of-the-art multi-view clustering methods are designed to exploit complementary and consistent information across multiple views without explicitly incorporating fairness constraints. These methods are included to evaluate the clustering capability of the proposed approach and to examine the fairness characteristics of conventional multi-view clustering methods. MCPL [3] leverages pseudo-label information and latent graph structures to learn discriminative representations, followed by a label fusion mechanism for obtaining the final clustering assignments. CGL [22] jointly exploits spectral embedding and low-rank tensor learning to construct a consensus graph that captures shared structural information across views. 3MC [5] utilizes deep representations extracted from multiple encoder layers and develops a multi-layer, multi-level contrastive learning framework that jointly performs inter-view feature contrastive learning and inter-layer semantic label contrastive learning.

Table 2  
Comparison of various clustering methods in terms of two accuracy metrics and two fairness metrics. For each metric across the datasets, the best score is shown in bold, and the second-best is underlined. A dash “–” indicates unavailable results: FairMVC is restricted to two-view datasets (Credit, Bank, Law), while FFC fails to produce results on Mfeat and COIL due to excessive memory usage on high-dimensional data.
<table><tr><td>Method</td><td>Metric</td><td>Credit</td><td>Bank 0.635</td><td>Law 0.550</td><td>Mfeat 0.797</td><td>COIL 0.689</td><td>Mean Value 0.608</td></tr><tr><td>BFKM</td><td>ACC NMI BAL DPD ↓</td><td>0.370 0.188 0.348 0.028</td><td>0.056 0.289 0.066</td><td>0.081 0.430 0.003</td><td>0.754 0.432 0.009</td><td>0.784 0.371 0.009</td><td>0.373 0.374 0.023</td></tr><tr><td>VFC</td><td>ACC NMI BAL DPD ↓</td><td>0.370 0.186 0.341 0.028</td><td>0.633 0.056 0.289 0.097</td><td>0.542 0.063 0.428 0.011</td><td>0.925 0.856 0.434 0.014</td><td>0.750 0.821 0.347 0.011</td><td>0.644 0.396 0.368 0.032</td></tr><tr><td>FFC</td><td>ACC NMI BAL DPD ↓</td><td>0.368 0.187 0.340 0.031</td><td>0.633 0.056 0.286 0.081</td><td>0.549 0.081 0.428 0.001</td><td>1</td><td>一</td><td></td></tr><tr><td>FairDen</td><td>ACC NMI BAL DPD ↓</td><td>0.420 0.046 0.343 0.036</td><td>0.638 0.062 0.271 0.097</td><td>0.802 0.011 0.400 0.063</td><td>0.554 0.595 0.240 0.011</td><td>0.823 0.905 0.318 0.010</td><td>0.647 0.324 0.314 0.043</td></tr><tr><td>MCPL</td><td>ACC NMI BAL DPD ↓</td><td>0.330 0.104 0.356 0.021</td><td>0.672 0.068 0.288 0.091</td><td>0.839 0.050 0.417 0.064</td><td>0.831 0.814 0.404 0.010</td><td>0.699 0.831 0.347 0.010</td><td>0.674 0.373 0.362 0.039</td></tr><tr><td>CGL</td><td>ACC NMI BAL DPD ↓</td><td>0.368 0.123 0.355 0.028</td><td>0.641 0.061 0.285 0.094</td><td>0.888 0.078 0.362 0.064</td><td>0.997 0.994 0.450 0.010</td><td>0.599 0.818 0.347 0.010</td><td>0.699 0.415 0.360 0.041</td></tr><tr><td>3MC</td><td>ACC NMI BAL DPD ↓</td><td>0.354 0.174 0.344 0.032</td><td>0.635 0.059 0.285 0.094</td><td>0.563 0.048 0.420 0.066</td><td>0.649 0.680 0.438 0.010</td><td>0.501 0.659 0.339 0.011</td><td>0.540 0.324 0.365 0.043</td></tr><tr><td>FairMVC</td><td>ACC NMI BAL DPD ↓</td><td>0.402 0.117 0.341 0.029</td><td>0.623 0.033 0.288 0.092</td><td>0.588 0.077 0.424 0.096</td><td>一 一 一</td><td>一 一 一</td><td></td></tr><tr><td>FMSC</td><td>ACC NMI BAL DPD ↓</td><td>0.387 0.126 0.357 0.031</td><td>0.652 0.094 0.305 0.110</td><td>0.862 0.047 0.432 0.074</td><td>0.293 0.211 0.412 0.017</td><td>0.806 0.895 0.347 0.013</td><td>0.600 0.275 0.371 0.049</td></tr><tr><td>AFMVC</td><td>ACC NMI BAL DPD ↓</td><td>0.465 0.210 0.352 0.028</td><td>0.749 0.094 0.302 0.081</td><td>0.836 0.047 0.420 0.054</td><td>0.864 0.825 0.440 0.009</td><td>0.750 0.840 0.365 0.010</td><td>0.733 0.403 0.376 0.036</td></tr><tr><td>FLFMVC</td><td>ACC NMI BAL DPD ↓</td><td>0.353 0.162 0.384 0.008</td><td>0.589 0.065 0.316 0.001</td><td>0.582 0.046 0.433 0.000</td><td>0.878 0.841 0.445 0.009</td><td>0.453 0.594 0.329 0.011</td><td>0.571 0.342 0.381 0.006</td></tr><tr><td>Ours</td><td>ACC NMI BAL DPD ↓</td><td>0.423 0.217 0.379 0.017</td><td>0.680 0.071 0.286 0.080</td><td>0.842 0.049 0.418 0.054</td><td>0.968 0.931 0.446 0.009</td><td>0.709 0.838 0.367 0.009</td><td>0.724 0.421 0.379 0.034</td></tr></table>

Multi-view fair clustering methods explicitly incorporate fairness considerations into multi-view representation learning or clustering optimization. FairMVC [44] promotes demographic parity by aligning the distribution of sensitive groups within individual clusters with their global distribution, while simultaneously enhancing multi-view representations through contrastive learning. FMSC [20] develops a one-stage spectral clustering framework equipped with a graph-based fairness regularizer, thereby integrating fair representation learning and clustering into a unified optimization procedure without requiring post-processing. AFMVC [13] adopts an adversarial learning framework with a gradient reversal mechanism to reduce the dependence between learned clustering representations and sensitive attributes, thereby mitigating sensitive-information leakage and improving group fairness. FLFMVC [43] introduces a multi-level fairness mechanism that jointly refines base partitions, promotes balanced demographic distributions, and guides fair cluster assignments.

For all comparison methods, the hyperparameters are set according to the default configurations provided in their publicly available implementations. For our method, $\lambda _ { C }$ and $\lambda _ { F }$ are both set to 0.1. All experiments are conducted on a PC equipped with an Intel Core i7-10700F CPU at 2.90 GHz, 16 GB of RAM, and an NVIDIA GeForce RTX 1660 GPU with 6 GB of memory. For methods involving stochastic procedures, we independently repeat each experiment ten times and report the average results.

## 4.2. Experimental Results

Table 2 reports the quantitative comparison among all competing methods. Based on these results, we highlight the following observations.

• Overall performance: Overall, our method achieves a favorable trade-of between clustering utility and group fairness. As reported in Table 2, Ours ranks first in terms of mean NMI and second in terms of both mean ACC and BAL. These results demonstrate that the proposed method preserves strong clustering capability while maintaining a high level of fairness, yielding consistently balanced performance across diferent evaluation criteria.

• Compared with single-view fair clustering methods: Among the single-view fair clustering baselines, including BFKM, VFC, FFC, and FairDen, our method generally delivers superior overall performance, particularly in terms of clustering quality. This advantage highlights the efectiveness of leveraging complementary information across multiple views while simultaneously preserving competitive group fairness.

• Compared with conventional multi-view clustering methods: Relative to conventional multi-view clustering approaches such as MCPL, CGL, and 3MC, our method achieves a more desirable trade-of between clustering quality and fairness. Although some conventional multi-view methods attain strong clustering results on individual datasets, their advantages are less consistent when fairness is jointly considered. In contrast, our method maintains competitive clustering accuracy while exhibiting more stable fairness performance across datasets.

• Compared with multi-view fair clustering methods: Among the fair multi-view clustering baselines, AFMVC achieves the highest mean ACC, whereas FLFMVC attains the strongest overall fairness performance. In comparison, our method ranks first in mean NMI and second in both mean ACC and BAL, indicating a favorable balance between clustering performance and fairness. Relative to AFMVC, our method improves group fairness at the cost of only a marginal decrease in ACC. Compared with FLFMVC, our method retains substantially stronger clustering performance while still achieving competitive fairness.

## 4.3. Ablation Study

In this subsection, we assess the contribution of each loss component by removing them individually and analyzing the corresponding changes in clustering performance and fairness. The detailed results are presented in Table 3, from which several observations can be drawn. First, removing either $\mathcal { L } _ { R }$ or $\scriptstyle { \mathcal { L } } _ { C }$ generally leads to decreased ACC and NMI, confirming the importance of reconstruction learning and clustering supervision in preserving informative representations and maintaining clustering quality. Second, excluding either the asymmetric alignment loss $\mathcal { L } _ { A }$ or the fairness regularization term $\mathcal { L } _ { F }$ typically results in lower BAL and higher DPD, indicating that these two components enhance fairness from complementary perspectives. Specifically, $\mathcal { L } _ { A }$ mitigates sensitive information at the representation level through asymmetric cross-view alignment, whereas $\mathcal { L } _ { F }$ directly regularizes the consensus assignments to promote group-level fairness. Notably, removing $\mathcal { L } _ { F }$ improves clustering performance on several datasets; however, such gains are accompanied by a clear deterioration in fairness, highlighting the inherent tradeof between clustering utility and fairness. Overall, the complete model achieves a more favorable balance by jointly incorporating all four objectives.

Ablation study on diferent loss combinations. Each loss configuration is treated as an individual method. The best result for each metric is shown in bold.
<table><tr><td>Loss</td><td>Metric</td><td>Credit</td><td>Bank</td><td>Law</td><td>Mfeat</td><td>COIL</td></tr><tr><td rowspan="4"> ${ \mathsf { w } } / { \mathsf { o } } ~ { \mathcal { L } } _ { R }$ </td><td>ACC</td><td>0.414</td><td>0.659</td><td>0.838</td><td>0.923</td><td>0.695</td></tr><tr><td>NMI</td><td>0.198</td><td>0.066</td><td>0.047</td><td>0.882</td><td>0.835</td></tr><tr><td>BAL</td><td>0.368</td><td>0.283</td><td>0.406</td><td>0.439</td><td>0.354</td></tr><tr><td>DPD ↓</td><td>0.019</td><td>0.091</td><td>0.055</td><td>0.009</td><td>0.010</td></tr><tr><td rowspan="4"> $w / \circ \mathcal { L } _ { C }$ </td><td>ACC</td><td>0.417</td><td>0.657</td><td>0.824</td><td>0.914</td><td>0.675</td></tr><tr><td>NMI</td><td>0.193</td><td>0.064</td><td>0.045</td><td>0.864</td><td>0.819</td></tr><tr><td>BAL</td><td>0.381</td><td>0.282</td><td>0.418</td><td>0.448</td><td>0.363</td></tr><tr><td>DPD ↓</td><td>0.017</td><td>0.100</td><td>0.053</td><td>0.008</td><td>0.009</td></tr><tr><td rowspan="4"> ${ \mathsf { w } } / { \mathsf { o } } ~ { \mathcal { L } } _ { A }$ </td><td>ACC</td><td>0.422</td><td>0.663</td><td>0.836</td><td>0.968</td><td>0.756</td></tr><tr><td>NMI</td><td>0.217</td><td>0.070</td><td>0.047</td><td>0.930</td><td>0.851</td></tr><tr><td>BAL</td><td>0.358</td><td>0.283</td><td>0.415</td><td>0.439</td><td>0.326</td></tr><tr><td>DPD ↓</td><td>0.025</td><td>0.094</td><td>0.056</td><td>0.010</td><td>0.010</td></tr><tr><td rowspan="4"> ${ \mathsf { w } } / { \mathsf { o } } ~ { \mathcal { L } } _ { F }$ </td><td>ACC</td><td>0.454</td><td>0.676</td><td>0.845</td><td>0.946</td><td>0.761</td></tr><tr><td>NMI</td><td>0.229</td><td>0.070</td><td>0.051</td><td>0.909</td><td>0.839</td></tr><tr><td>BAL</td><td>0.354</td><td>0.282</td><td>0.412</td><td>0.438</td><td>0.325</td></tr><tr><td>DPD ↓</td><td>0.027</td><td>0.097</td><td>0.055</td><td>0.010</td><td>0.010</td></tr><tr><td rowspan="4">Full Model</td><td>ACC</td><td>0.423</td><td>0.680</td><td>0.842</td><td>0.968</td><td>0.709</td></tr><tr><td>NMI</td><td>0.217</td><td>0.071</td><td>0.049</td><td>0.931</td><td>0.838</td></tr><tr><td>BAL</td><td>0.379</td><td>0.286</td><td>0.418</td><td>0.446</td><td>0.367</td></tr><tr><td>DPD ↓</td><td>0.017</td><td>0.080</td><td>0.054</td><td>0.009</td><td>0.009</td></tr></table>

## 4.4. Parameter Sensitivity Analysis

![](images/1223cbd49c3f803bbbcba30351d3a868a303d952bdb06e30c5ac14e5b5e3818c.jpg)  
(a) 6<sub>F</sub>

![](images/aabafbdece48f1c39183bddcafa4b9cc785229217619673dfe35c05b69ad0257.jpg)  
(b) 6<sub>C</sub>  
Figure 2: Parameter sensitivity analysis of $\lambda _ { F }$ and $\lambda _ { C }$ in terms of ACC and BAL.

Our framework jointly optimizes reconstruction, clustering, asymmetric alignment, and fairness objectives, where $\lambda _ { C }$ and $\lambda _ { F }$ control the contributions of the clustering and fairness regularization terms, respectively. To evaluate the sensitivity of these two hyperparameters, we vary one parameter at a time while fixing the other to its default value of 0.1. Specifically, both $\lambda _ { C }$ and $\lambda _ { F }$ are selected from {0.01, 0.05, 0.1, 0.2, 0.5}, and the corresponding ACC and BAL results are reported in Fig. 2.

As shown in Fig. 2, the two hyperparameters exhibit diferent efects on clustering performance and fairness. When $\lambda _ { F }$ increases from a relatively small value, BAL improves noticeably, while the fairness performance becomes relatively stable once $\lambda _ { F }$ reaches a moderate level. Further increasing $\lambda _ { F }$ brings only marginal fairness gains while tending to reduce clustering accuracy, indicating that an excessively strong fairness constraint may compromise clustering utility. For $\lambda _ { C } .$ , a larger value generally favors clustering accuracy, whereas overemphasizing the clustering objective leads to a noticeable degradation in BAL. These observations reveal a clear trade-of between clustering quality and fairness. Overall, setting $\lambda _ { C } = \lambda _ { F } = 0 . 1$ provides a favorable balance between the two objectives and is therefore adopted as the default configuration in our experiments.

## 4.5. Robustness Analysis under Heterogeneous Sensitive Dependence

To examine the efect of heterogeneous sensitive dependence across views, we construct controlled variants of COIL and Law with progressively increasing sensitive dependence in a selected target view. For each dataset, we select an informative target view according to its single-view clustering performance and inject sensitive-attributerelated perturbations only into this view, while keeping all remaining views, cluster labels, and sensitive attributes unchanged.

After feature-wise standardization, let $\mathbf { x } _ { i }$ denote the representation of the �-th sample in the target view. We construct a sensitive perturbation $\delta _ { i }$ by centering the sensitive signal within each ground-truth cluster and constraining the perturbation direction to be orthogonal to the between-class discriminative subspace. The perturbed representation is defined as

$$
\mathbf { x } _ { i } ^ { \rho } = \mathcal { M } \left( \mathbf { x } _ { i } + \sqrt { \rho } \delta _ { i } \right) ,
$$

where $\rho$ controls the strength of the injected sensitive information and (⋅) preserves the overall feature scale.

The perturbation is designed to increase the sensitive dependence of the target view while minimizing interference with its original discriminative structure. Based on the resulting sensitive-dependence levels, we use $\rho \in$ {0.001, 0.005, 0.01} for COIL and $\rho \in \{ 0 . 0 0 1 , 0 . 0 1 , 0 . 0 5 \}$ for Law, corresponding to the Low, Medium, and High settings.

## Table 4

Sensitive-dependence statistics under diferent levels of cross-view heterogeneity.
<table><tr><td rowspan="2">Setting</td><td colspan="2">COIL</td><td colspan="2">Law</td></tr><tr><td>Target NHSIC</td><td>BiasGap</td><td>Target NHSIC</td><td>BiasGap</td></tr><tr><td rowspan="2">Original Low</td><td>0.0087</td><td>0.0056</td><td>0.0090</td><td>0.0003</td></tr><tr><td>0.0205</td><td>0.0174</td><td>0.0150</td><td>0.0057</td></tr><tr><td>Medium</td><td>0.0675</td><td>0.0644</td><td>0.0422</td><td>0.0328</td></tr><tr><td>High</td><td>0.1257</td><td>0.1225</td><td>0.1378</td><td>0.1284</td></tr></table>

To verify the induced sensitive-information heterogeneity, we report two statistics for each setting in Table 4. Target NHSIC denotes the dependence between the selected target view and the sensitive attribute, while BiasGap measures the diference between the maximum and minimum NHSIC values across all views. Accordingly, Target NHSIC reflects the sensitive dependence of the perturbed target view, whereas BiasGap characterizes the resulting cross-view sensitive-information discrepancy.

As shown in Table 4, both Target NHSIC and BiasGap increase monotonically from the Original to the High setting on COIL and Law. The increasing Target NHSIC confirms that the sensitive dependence of the selected view is progressively strengthened, while the corresponding increase in BiasGap demonstrates that the discrepancy in sensitive dependence across views becomes increasingly pronounced. These results validate the constructed variants as controlled settings with progressively stronger cross-view sensitive-information heterogeneity.

Based on these controlled variants, we further compare the clustering performance and fairness of diferent methods as cross-view sensitive-information heterogeneity increases. The results are shown in Fig. 3. On COIL, our method maintains an almost unchanged BAL score as the sensitive dependence of the target view increases, whereas the competing methods exhibit more noticeable fairness degradation, especially under the High setting. Meanwhile, its ACC remains competitive across all perturbation levels. A similar but more pronounced pattern is observed on Law. The ACC and BAL of our method remain nearly unchanged from the Original to the High setting, whereas the competing methods become increasingly afected by the injected sensitive information. In particular, AFMVC exhibits a clear decline in both clustering utility and fairness under the High-heterogeneity setting, while FMSC shows a substantial decrease in ACC as the perturbation strength increases.

![](images/c970f311813e2e9ac2bd49f260ad0bb5df108a5ece19fbe13a2ab5240d178048.jpg)  
(a) COIL

![](images/97c5a0cf412dad549aa5cae300b2529c4591436033cc6c5c23abef0b895574d5.jpg)  
(b) COIL

![](images/53c5766b4e1e2c9c2a54925090be988422694ef1d0891c8f14ab5a5598ef6414.jpg)  
(c) Law

![](images/603e447ccf37c66ea873f9699cb3bd99d0a7ec93d67c9bf4bb273621424c13d3.jpg)  
(d) Law  
Figure 3: Comparison of multi-view fair clustering methods under increasing cross-view sensitive-information heterogeneity.

Overall, these results indicate that explicitly modeling view-specific sensitive dependence and regulating crossview interactions improves robustness to heterogeneous sensitive information. The proposed method therefore maintains a more stable balance between clustering quality and fairness as the sensitive-information discrepancy across views increases.

## 5. Conclusion

In this paper, we propose a novel multi-view fair clustering framework guided by cross-view sensitive information discrepancy. The proposed method explicitly estimates the sensitive dependence of diferent views and employs a bias-ranked asymmetric alignment mechanism, encouraging views with higher sensitive dependence to learn from those with lower sensitive dependence while adaptively regulating the alignment process according to cross-view discrepancy. In addition, fairness regularization is imposed on the consensus soft assignments to further promote group fairness at the clustering-assignment level. By jointly considering cross-view sensitive-information heterogeneity, representation alignment, and assignment-level fairness, the proposed framework efectively balances clustering accuracy and group fairness. Experimental results on five benchmark datasets demonstrate competitive clustering performance together with strong group fairness. Moreover, the robustness analysis under heterogeneous sensitive dependence further confirms that our method can better cope with cross-view sensitive-information heterogeneity than existing multi-view fair clustering approaches, particularly in maintaining stable fairness as the sensitive dependence of the target view increases.

In future work, we plan to investigate more scalable strategies for estimating and regulating cross-view sensitive information, and further extend the framework to more complex scenarios, such as continuous sensitive attributes, incomplete multi-view data, and streaming multi-view settings.

## Acknowledgments

This work has been supported by the National Natural Science Foundation of China under Grant Nos. 62476038 and 62472064.

## References

[1] Ahmadian, S., Epasto, A., Knittel, M., Kumar, R., Mahdian, M., Moseley, B., Pham, P., Vassilvitskii, S., Wang, Y., 2020. Fair hierarchical clustering, in: Advances in Neural Information Processing Systems, pp. 21050–21060.

[2] Bateni, M., Cohen-Addad, V., Epasto, A., Lattanzi, S., 2024. A scalable algorithm for individually fair k-means clustering, in: Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, PMLR. pp. 3151–3159.

[3] Cai, R., Chen, H., Mi, Y., Luo, C., Horng, S.J., Li, T., 2024. Multi-view clustering via pseudo-label guide learning and latent graph structure recovery. Pattern Recognition 151, 110420.

[4] Chakrabarty, D., Negahbani, M., 2021. Better algorithms for individually fair k-clustering, in: Proceedings of the 35th International Conference on Neural Information Processing Systems.

[5] Chen, Z., Wu, X.J., Xu, T., Li, H., Kittler, J., 2025. Multi-layer multi-level comprehensive learning for deep multi-view clustering. Information Fusion 116, 102785.

[6] Chhabra, A., Masalkovaite, K., Mohapatra, P., 2021. An overview of fairness in clustering. IEEE Access 9, 130698–130720.˙

[7] Chhabra, A., Singla, A., Mohapatra, P., 2022. Fair clustering using antidote data, in: Proceedings of The Algorithmic Fairness through the Lens of Causality and Robustness, PMLR. pp. 19–39.

[8] Chhaya, R., Dasgupta, A., Choudhari, J., Shit, S., 2022. On coresets for fair regression and individually fair clustering, in: Proceedings of The 25th International Conference on Artificial Intelligence and Statistics, PMLR. pp. 9603–9625.

[9] Chierichetti, F., Kumar, R., Lattanzi, S., Vassilvitskii, S., 2017. Fair clustering through fairlets, in: Advances in Neural Information Processing Systems, Curran Associates, Inc.. pp. 5029–5037.

[10] Fang, U., Li, M., Li, J., Gao, L., Jia, T., Zhang, Y., 2023. A comprehensive survey on multi-view clustering. IEEE Transactions on Knowledge and Data Engineering 35, 12350–12368.

[11] Gao, Q., Lian, H., Wang, Q., Sun, G., 2020. Cross-modal subspace clustering via deep canonical correlation analysis, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 3938–3945.

[12] Huang, L., Jiang, S., Vishnoi, N., 2019. Coresets for clustering with fairness constraints, in: Advances in Neural Information Processing Systems, pp. 7587–7598.

[13] Jiang, M., Zhou, J., Hu, L., Liu, X., He, Z., Chen, Z., 2026. Adversarial fair multi-view clustering. Pattern Recognition , 113708.

[14] Jones, M., Nguyen, H., Nguyen, T., 2020. Fair k-centers via maximum matching, in: International conference on machine learning, PMLR. pp. 4940–4949.

[15] Jung, C., Kannan, S., Lutz, N., 2020. Service in Your Neighborhood: Fairness in Center Location, in: 1st Symposium on Foundations of Responsible Computing, pp. 5:1–5:15.

[16] Kleindessner, M., Awasthi, P., Morgenstern, J., 2019a. Fair k-center clustering for data summarization, in: International Conference on Machine Learning, PMLR. pp. 3448–3457.

[17] Kleindessner, M., Samadi, S., Awasthi, P., Morgenstern, J., 2019b. Guarantees for spectral clustering with fairness constraints, in: International Conference on Machine Learning, PMLR. pp. 3458–3467.

[18] Krieger, L., Beer, A., Matthews, P., Thiesson, A.M., Assent, I., 2025. Fairden: Fair density-based clustering, in: Proceedings of the 13th International Conference on Learning Representations.

[19] Li, P., Zhao, H., Liu, H., 2020. Deep fair clustering for visual learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9070–9079.

[20] Li, R., Hu, H., Du, L., Chen, J., Jiang, B., Zhou, P., 2024. One-stage fair multi-view spectral clustering, in: Proceedings of the 32nd ACM International Conference on Multimedia, pp. 1407–1416.

[21] Li, R., Zhang, C., Hu, Q., Zhu, P., Wang, Z., 2019a. Flexible multi-view representation learning for subspace clustering., in: Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, pp. 2916–2922.

[22] Li, Z., Tang, C., Liu, X., Zheng, X., Zhang, W., Zhu, E., 2021. Consensus graph learning for multi-view clustering. IEEE Transactions on Multimedia 24, 2461–2472.

[23] Li, Z., Wang, Q., Tao, Z., Gao, Q., Yang, Z., 2019b. Deep adversarial multi-view clustering network, in: Proceedings of the 28th International Joint Conference on Artificial Intelligence, p. 2952–2958.

[24] Liang, Y., Huang, D., Wang, C.D., 2019. Consistency meets inconsistency: A unified graph learning framework for multi-view clustering, in: Proceedings of the 2019 IEEE International Conference on Data Mining (ICDM), IEEE. pp. 1204–1209.

[25] Liu, J., Liu, X., Yang, Y., Liao, Q., Xia, Y., 2023. Contrastive multi-view kernel learning. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 9552–9566.

[26] Liu, X., Liang, K., Wang, J., Liu, S., Wang, X., Wang, H., 2026. Two decades of multi-view clustering: Taxonomy, application, and challenge. IEEE Transactions on Pattern Analysis and Machine Intelligence 48, 3744–3764.

[27] Mahabadi, S., Vakilian, A., 2020. Individual fairness for k-clustering, in: Proceedings of the 37th International Conference on Machine Learning, PMLR. pp. 6586–6596.

[28] Oyewole, G.J., Thopil, G.A., 2023. Data clustering: application and trends. Artificial Intelligence Review 56, 6439–6475.

[29] Pan, R., Zhong, C., 2023. Fairness first clustering: A multi-stage approach for mitigating bias. Electronics 12, 2969.

[30] Pan, R., Zhong, C., Qian, J., 2023. Balanced fair k-means clustering. IEEE Transactions on Industrial Informatics 20, 5914–5923.

[31] Rutherglen, G., 1987. Disparate impact under title vii: an objective theory of discrimination. Virginia Law Review 73, 1297.

[32] Schmidt, M., Schwiegelshohn, C., Sohler, C., 2020. Fair coresets and streaming algorithms for fair k-means, in: Approximation and Online Algorithms, pp. 232–251.

[33] Thejaswi, S., Ordozgoiti, B., Gionis, A., 2021. Diversity-aware k-median: Clustering with fair center representation, in: Machine Learning and Knowledge Discovery in Databases. Research Track, pp. 765–780.

[34] Vakilian, A., Yalciner, M., 2022. Improved approximation algorithms for individually fair clustering, in: Proceedings of The 25th International Conference on Artificial Intelligence and Statistics, PMLR. pp. 8758–8779.

[35] Wang, Y., Wu, L., Lin, X., Gao, J., 2018. Multiview spectral clustering via structured low-rank matrix factorization. IEEE Transactions on Neural Networks and Learning Systems 29, 4833–4843.

[36] Xu, H., Wang, Q., Wang, B., Gao, Q., 2025. Deep fair multi-view clustering with attention kan, in: Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pp. 5061–5070.

[37] Yang, Z., Liang, N., Yan, W., Li, Z., Xie, S., 2020. Uniform distribution non-negative matrix factorization for multiview clustering. IEEE Transactions on Cybernetics 51, 3249–3262.

[38] Yang, Z., Xu, Q., Zhang, W., Cao, X., Huang, Q., 2019. Split multiplicative multi-view subspace clustering. IEEE Transactions on Image Processing 28, 5147–5160.

[39] Yuan, C., Zhu, Y., Zhong, Z., Zheng, W., Zhu, X., 2022. Robust self-tuning multi-view clustering. World Wide Web 25, 489–512.

[40] Zafar, M.B., Valera, I., Rodriguez, M.G., Gummadi, K.P., 2017. Fairness Constraints: Mechanisms for Fair Classification, in: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, PMLR. pp. 962–970.

[41] Zhang, H., Davidson, I., 2021. Deep fair discriminative clustering. arXiv preprint arXiv:2105.14146 .

[42] Zhao, B., Wang, Q., Tao, Z., Feng, W., Gao, Q., 2024. Dfmvc: Deep fair multi-view clustering, in: Proceedings of the 32nd ACM International Conference on Multimedia, pp. 8090–8099.

[43] Zhao, T., Chen, Y., Du, L., 2026. Fairness-aware late-fusion multi-view clustering with balance and robustness, in: Proceedings of the 14th International Conference on Computational Visual Media (CVM).

[44] Zheng, L., Zhu, Y., He, J., 2023. Fairness-aware multi-view clustering, in: Proceedings of the 2023 SIAM International Conference on Data Mining, SIAM. pp. 856–864.

[45] Ziko, I.M., Yuan, J., Granger, E., Ayed, I.B., 2021. Variational fair clustering, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 11202–11209.