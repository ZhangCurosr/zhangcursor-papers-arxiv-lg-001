# GBTSVR: Twin Support Vector Regression via Granular Ball Computing

A. Quadir<sup>a</sup>, M. Tanveer<sup>a,∗</sup>

<sup>a</sup>Department of Mathematics, Indian Institute of Technology Indore, Simrol, Indore, 453552, Madhya Pradesh, India

## Abstract

Regression models used in real-world applications are often challenged by noise, outliers, data heterogeneity, and asymmetric error characteristics, which severely limit the efectiveness of conventional point-wise learning models. While twin support vector regression (TSVR) partially addresses these issues through the use of nonparallel regression bounds, it still relies on individual samples as basic learning units. As a result, TSVR remains highly sensitive to noisy and contaminated observations and fails to leverage the underlying structural and regional information inherent in complex datasets, leading to suboptimal robustness and generalization performance. To address these limitations, we introduce granular ball twin support vector regression (GBTSVR) and its quantile extension, granular ball twin support vector quantile regression (GB-TSVQR). The proposed models integrate granular ball computing with TSVR, replacing individual data points with compact, structure-preserving granular balls as fundamental learning units. This region-level representation efectively suppresses noise and outliers while preserving intrinsic data geometry, thereby enhancing robustness, stability, and generalization. GBTSVR constructs nonparallel lower- and upper-bound regression functions over granular balls, significantly reducing sensitivity to sample-level perturbations. Building upon this framework, GB-TSVQR further

incorporates quantile regression principles to simultaneously capture heterogeneous trends and asymmetric error distributions across diferent quantile levels. Extensive experiments on synthetic datasets, time-series benchmarks, and outlier-contaminated data demonstrate that the proposed GBTSVR and GB-TSVQR consistently outperform state-of-the-art regression models. The results confirm that learning at an appropriate granularity level, combined with twin and quantile regression principles, provides a powerful and scalable solution for complex regression tasks.

Keywords: Granular ball computing, Twin support vector regression, Quantile regression, Dual coordinate decent algorithm.

## 1. Introduction

Machine learning, as an integral branch of artificial intelligence, has undergone rapid development and has become a transformative force across numerous scientific and engineering disciplines due to its capability to uncover hidden structures and patterns from complex datasets [1]. Within this broad area, supervised learning plays a pivotal role, where models are trained using labeled data to establish relationships between input features and corresponding outputs, enabling accurate prediction and inference [2, 3]. Regression, in particular, constitutes a core task in supervised learning, focusing on learning functional mappings between explanatory variables and continuous response values [4].

Among the various regression techniques, support vector regression (SVR) [5] has gained considerable attention owing to its strong theoretical foundation and robust predictive performance. SVR is derived from the support vector machine (SVM) framework [6], which is widely recognized for its success in classification problems. SVM-based models have demonstrated efectiveness in a wide range of applications, including medical diagnosis such as Alzheimer’s diagnosis [7], EEG signal classification [8] among many others. Grounded in statistical learning theory, SVMs are designed to achieve good generalization performance when applied to unseen data [9]. By extending the SVM formulation to accommodate continuous outputs, SVR becomes a powerful tool for regression and function approximation tasks [10]. Consequently, SVR has been successfully applied in diverse real-world scenarios, including brain age estimation [11, 12], long-term stock index forecasting [13], prediction of gas difusion coeficients [14], bank profitability estimation [15], and financial market analysis [16].

However, most conventional SVR-based approaches construct the decision function using two parallel hyperplanes, which may fail to adequately capture the inherent asymmetry present in real-world data distributions. This limitation can lead to the loss of valuable asymmetric information associated with diferent data points. To overcome this drawback, twin support vector regression (TSVR) [17] is introduced, which learns two nonparallel regression functions simultaneously. These functions are designed to estimate the upper and lower ε-insensitive bounds of the unknown regression function, thereby efectively modeling asymmetric data characteristics and providing a more flexible and informative representation of the underlying regression structure. Shao et al. [18] enhanced the TSVR framework by incorporating a regularization term, leading to the development of the ε-twin support vector regression (ε-TSVR). This formulation results in stable, positive definite quadratic programming problems in the dual space and significantly improves regression performance. To further enhance generalization ability, Xu et al. [19] proposed an asymmetric ν-twin support vector regression with pinball loss (Asy-ν-TSVR), which explicitly accounts for data asymmetry. Later, Gupta and Gupta [20] introduced the asymmetric Lagrangian ν-twin support vector regression with pinball loss (URALTSVR), enabling finer control over fitting errors within an asymmetric tube. Although the TSVR framework, through its two nonparallel hyperplanes, efectively captures asymmetric characteristics in data, it may still overlook latent heterogeneous information that is not directly observable across all data points. To simultaneously model both unobserved heterogeneity and asymmetric behavior, the concept of quantile regression (QR) is integrated into the ε-TSVR framework, giving rise to the twin support vector quantile regression (TSVQR) model [21]. This integration allows the regression model to provide a more comprehensive and flexible representation of the conditional distribution of the response variable. Despite significant progress in SVR and its variants, existing regression frameworks continue to face persistent challenges when applied to real-world data. In particular, most regression models operate at the level of individual data points, making them inherently sensitive to noise, outliers, and sample redundancy. This point-wise learning paradigm not only degrades predictive stability under data contamination but also leads to high computational overhead, especially for large-scale and noisy datasets. Moreover, classical TSVR-based models, while capable of capturing asymmetric error structures through dual nonparallel regressors, still rely on raw samples and therefore remain vulnerable to heterogeneity and local irregularities in the data distribution [22].

Human cognition typically follows a global-to-local perception strategy, forming a coarse overall understanding before refining details [23]. In visual perception, the brain first captures global structure and topology, then focuses on local variations. In contrast, many machine learning models learn at the level of individual samples or features, which can lead to issues with scalability, computational cost, and robustness in large, noisy, or complex regression datasets.

Granular computing addresses this by organizing data into higher-level information granules, enabling learning at appropriate abstraction levels. Granular ball (GB) learning, introduced by Xia et al. [24], partitions data into hyperspherical regions of varying radii, where each granular ball represents a local data distribution and preserves key structural characteristics while reducing noise [25]. Prior work shows that selecting suitable granularity improves generalization and scalability, aligning with hierarchical human information processing [26].

Granular computing provides an alternative perspective by learning at suitable abstraction levels rather than processing individual samples [27]. It groups data into meaningful granules that retain intrinsic structure while reducing irrelevant variation. Among these methods, granular ball computing represents data using compact, highpurity hyperspherical regions, efectively aggregating similar samples to reduce redundancy and noise while capturing underlying distributions [7]. Although granular ball–based approaches have shown strong classification performance [28, 29], their application to regression, particularly within the TSVR framework, remains limited, motivating this work [30].

To address the limitations of point-wise regression learning, we propose granular ball twin support vector regression (GBTSVR), a novel regression framework that integrates granular ball computing with TSVR. In GBTSVR, granular balls are treated as the fundamental learning units instead of individual samples. Each granular ball encapsulates local data geometry, enabling the regression model to learn from structured regions rather than isolated points. This design significantly reduces sensitivity to noise and outliers while preserving essential regression trends. Building upon this framework, we further introduce granular ball twin support vector quantile regression (GB-TSVQR), which incorporates quantile regression into the granular ball–based TSVR architecture. By jointly modeling lower and upper nonparallel quantile regressors at diferent quantile levels, GB-TSVQR captures both heterogeneous and asymmetric information in the data distribution. The use of granular balls ensures that quantile estimation is robust to extreme deviations and local distributional irregularities, while the twin-regressor formulation maintains computational eficiency through smaller, well-structured optimization problems.

The main contributions of this paper are summarized as follows:

• We propose granular ball twin support vector regression (GBTSVR), a novel regression framework that integrates granular ball computing with twin support vector regression. By using granular balls as region-level learning units instead of individual samples, the proposed model efectively mitigates noise, reduces redundancy, and enhances robustness in regression tasks.

• We further introduce granular ball twin support vector quantile regression (GB-TSVQR), which extends GBTSVR by incorporating quantile regression. This model jointly captures heterogeneous and asymmetric distributional characteristics across diferent quantile levels, providing a more comprehensive representation of the underlying data structure.

• The proposed GB-TSVQR formulates the learning task as two reduced-size quadratic programming problems (QPP), each associated with a single set of constraints defined over granular balls rather. Furthermore, a dual coordinate descent algorithm is employed to solve each QPP eficiently, resulting in significantly accelerated training while maintaining high regression accuracy.

• Comprehensive experiments on synthetic datasets, time-series datasets, and outlierinjected regression benchmarks demonstrate that GBTSVR and GB-TSVQR consistently outperform state-of-the-art regression models in terms of accuracy, robustness, and stability across diverse noise settings.

Related work on granular-ball computing and the mathematical formulation of TSVR are presented in Section S.I of the Supplementary Material.

## 2. Granular-Ball–Based Regression Framework

Let a regression dataset be denoted by $\mathcal { D } = \{ ( x _ { i } , t _ { i } ) \} _ { i = 1 } ^ { n } , x _ { i } \in \mathbb { R } ^ { l } , t _ { i } \in \mathbb { R } \}$ , where $x _ { i }$ represents the input vector and $y _ { i }$ is the corresponding response. Directly learning regression functions from point-wise samples is often vulnerable to noise, outliers, and redundant observations. To address these limitations, the proposed framework first transforms the original dataset into a compact set of structured granules and subsequently performs regression at the granule level.

![](images/1a02f142259a4958ed30671b58e110ea1728e549aba33e49b7aeec8403490bfc.jpg)  
(a) The original dataset

![](images/8f90e798e98224a6f8d41212426b1c4bead46b17c90079fcb28f4d37a74386fa.jpg)  
(b) Generated granular ball in first iteration

![](images/d64191f15650cbd9fdfcd04e623b81b4dee610f0d9b1a2897b971c65dd60c7a3.jpg)  
(c) Generated granular ball in second iteration

![](images/3715d392d69e540a72d4366042d1b31a4aa100f02473de66429c8d7eddea411e.jpg)  
(d) Generated granular ball in middle it eration

![](images/6a95767fbe8c44d125a0768562b0a73083d993207838da9e880a3e2682db424d.jpg)  
(e) Results after stop splitting

![](images/3aad83c29d30b471ac709946248aaa2876afaac257ee9c5f14a8c0a05dd78929.jpg)  
(f) Extracted granular balls

Figure 1: This figure demonstrates the construction of granular balls by segmenting the sinc dataset corrupted with Gaussian noise of zero mean and a standard deviation of 0 2.  
![](images/f5a1982f4591db27f30353d9d476eb694363e74d69ead83365d3cc6b21b3b030.jpg)  
Figure 2: Overview of the GBTSVR Framework.

To enable the use of purity concepts originally defined for classification, the continuous response variable is first discretized into a finite set of categories. Specifically, the target variable $t \in \mathbb { R }$ is partitioned into k non-overlapping intervals, where each interval groups samples exhibiting similar response characteristics. All samples falling within the same interval are assigned a common discrete label from $\{ 1 , 2 , \ldots , k \}$ . This discretization step allows the notion of purity to be extended to regression settings and serves as the foundation for defining a regression-oriented granularity quality measure.

A granular ball $G B ( c , r )$ summarizes a subset of data points through its center $c \in \mathbb { R } ^ { l }$ and radius $r \in \mathbb { R } ^ { + }$ , and is defined as $G B ( c , r ) = \{ x \in \mathbb { R } ^ { l } | d ( x , c ) \leq r \}$ , where $d ( x , c ) = \sqrt { \textstyle \sum _ { i = 1 } ^ { l } ( x _ { i } - c _ { i } ) ^ { 2 } }$ denotes the Euclidean distance. For a granular ball containing p samples $\{ x _ { j } \} _ { j = 1 } ^ { p }$ , the center is computed as the centroid of the enclosed points $\begin{array} { r } { c } { { \small c \ = \ \frac { 1 } { p } \sum _ { j = 1 } ^ { p } x _ { j } } } \end{array}$ . The radius quantifies the dispersion of samples within the ball and is defined using the average distance to improve robustness against outliers and is defined as $\begin{array} { r } { r = { \frac { 1 } { p } } \sum _ { j = 1 } ^ { p } \| x _ { j } - c \| } \end{array}$ . Each granular ball thus captures both the central tendency and the local variability of a group of neighboring samples, providing a compact and noise-tolerant data representation.

To ensure that each granular ball corresponds to a coherent local structure, a purity criterion is imposed. For a granular ball $G B _ { i }$ purity is defined as the proportion of samples sharing the dominant label within the ball. A predefined threshold $\tau \in ( 0 , 1 ]$ enforces homogeneity among the enclosed samples. Let $\{ G B _ { i } \} _ { i = 1 } ^ { p }$ denote the set of granular balls obtained after partitioning the dataset. The generation of granular balls is formulated as the following optimization problem:

$$
\begin{array} { l l } { \operatorname* { m i n } _ { \mathbf { \theta } \atop \mathbf { \Theta } _ { \mathbf { \Theta } } } \varrho _ { 1 } \cdot \frac { n } { \sum _ { i = 1 } ^ { p } | G B _ { i } | } + \varrho _ { 2 } \cdot p } \\ { \quad } \\ { \mathrm { s . t . ~ p u r i t y } ( G B _ { j } ) \geq \tau , ~ j = 1 , 2 , \ldots , p , } \end{array}\tag{1}
$$

where n is the total number of samples, $\left| G B _ { i } \right|$ dnotes the cardinality of the $i ^ { t h }$ granular ball, p is the number of granular balls, and $\varrho _ { 1 } , \varrho _ { 2 }$ are weighting coeficients that balance compactness and granularity. The construction process follows an iterative coarse-to-fine strategy. Initially, all samples are grouped into a single granular ball. If its purity is below the threshold τ, the ball is split into smaller granules. This procedure continues until all granular balls satisfy the purity constraint, yielding a set of compact and informative granules that preserve the intrinsic structure of the data.

At the outset, the entire training dataset is treated as a single granular ball, as shown in Figure 1a. This initial granule is then iteratively partitioned based on the proposed regression-oriented quality criterion, leading to progressively more homogeneous granular balls, as illustrated in Figures 1b–1d. The splitting process continues until all granular balls satisfy the predefined regression purity threshold, at which point the algorithm converges, as depicted in Figure 1e. The final set of granular balls, shown in Figure 1f, demonstrates the capability of granular ball computing to efectively capture the intrinsic structure and distributional characteristics of the regression data.

## 3. The Proposed Granular Ball Twin Support Vector Regression

In this section, we present the mathematical formulation of the proposed granular ball twin support vector regression (GBTSVR) model. The core idea of GBTSVR is to integrate granular ball computing with twin support vector regression by constructing regression functions using granular balls as learning units instead of individual data samples. Each granular ball summarizes a local region of the input space through its center, radius, and associated statistical information, enabling the model to capture the intrinsic data structure while suppressing the influence of noise and outliers. By learning two nonparallel regression functions from these granular representations, GBTSVR achieves improved robustness, enhanced generalization, and reduced computational complexity. The mathematical formulation of GBTSVR is given as follows:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { w _ { 1 } , b _ { 1 } } \frac { 1 } { 2 } \| \pmb { \mathcal { Y } } + \pmb { \mathcal { R } } - e \epsilon _ { 1 } - ( C w _ { 1 } + e b _ { 1 } ) \| ^ { 2 } + \mathcal { D } _ { 1 } e ^ { T } \eta _ { 1 } } \\ { \mathrm { ~ s . t . ~ } \pmb { \mathcal { Y } } + \mathcal { R } - ( C w _ { 1 } + e b _ { 1 } ) \geq e \epsilon _ { 1 } - \eta _ { 1 } , } \\ { \displaystyle \eta _ { 1 } \geq 0 , } \end{array}\tag{2}
$$

and

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { w _ { 2 } , b _ { 2 } } \frac { 1 } { 2 } \| \boldsymbol { \mathcal { Y } } + \mathcal { R } + e \epsilon _ { 2 } - ( C w _ { 2 } + e b _ { 2 } ) \| ^ { 2 } + \mathcal { D } _ { 2 } e ^ { T } \eta _ { 2 } } \\ { \mathrm { ~ s . t . ~ } ( C w _ { 2 } + e b _ { 2 } ) - \boldsymbol { \mathcal { Y } } - \mathcal { R } \geq e \epsilon _ { 2 } - \eta _ { 2 } , } \\ { \displaystyle \eta _ { 2 } \geq 0 , } \end{array}\tag{3}
$$

where $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 2 }$ are positive regularization constants, $\epsilon _ { 1 }$ and $\epsilon _ { 2 }$ are nonnegative parameters, and $\eta _ { 1 } , \eta _ { 2 }$ denotes the corresponding slack vectors.

The GBTSVR approach learns two afine functions, $f _ { 1 } ( c ) = w _ { 1 } ^ { T } c + b _ { 1 }$ and $f _ { 2 } ( c ) =$ $w _ { 2 } ^ { T } c + b _ { 2 }$ , which respectively define the lower and upper ϵ-insensitive regression boundaries. Given the training dataset $( C , \ y )$ , the function $f _ { 1 } ( c )$ models the downbound regressor, while $f _ { 2 } ( c )$ characterizes the up-bound regressor. The final regression output is obtained by taking the average of these two functions. An intuitive geometric interpretation of this mechanism is illustrated in Fig. 2.

The leading term in the objective functions given in (2) and (3) represents the aggregate of squared deviations between the center of the GB and the shifted regression functions $\pmb { y } + \pmb { \mathcal { R } } = w _ { 1 } ^ { T } c + b _ { 1 } + \epsilon _ { 1 } \mathrm { o r } y + \pmb { \mathcal { R } } = w _ { 2 } ^ { T } c + b _ { 2 } - \epsilon _ { 2 }$ . In the proposed GBTSVR framework, these distances are computed with respect to the centers of the granular balls rather than individual data points. Consequently, minimizing this term encourages the regression functions $f _ { 1 } ( c )$ and $f _ { 2 } ( c )$ to accurately approximate the $\epsilon _ { 1 } -$ insensitive lower bound and $\epsilon _ { 2 } .$ -insensitive upper bound regressors at the granular-ball level, thereby enhancing robustness to noise and outliers. The imposed constraints enforce that the estimated regression functions $f _ { 1 } ( c )$ and $f _ { 2 } ( c )$ maintain a minimum separation of $\epsilon _ { 1 }$ and $\epsilon _ { 2 } .$ , respectively, from the tangential hyperplane associated with each granular ball. Specifically, the responses corresponding to the granular-ball centers are constrained to lie at least $\epsilon _ { 1 }$ above $( f _ { 1 } ( c ) ~ ,$ ) and at least $\epsilon _ { 2 }$ below $f _ { 2 } ( c )$ . To accommodate violations of these margin requirements, slack variable vectors $\eta _ { 1 }$ and $\eta _ { 2 }$ are incorporated to quantify deviations when the prescribed distances are not satisfied. The second term in the objective function penalizes the cumulative slack, thereby controlling the fitting error and balancing regression accuracy with model regularization.

The Lagrangian corresponding to the problem (2) is given by

$$
\begin{array} { r } { \mathcal { L } = \displaystyle \frac { 1 } { 2 } \| y + \mathcal { R } - e \epsilon _ { 1 } - ( C w _ { 1 } + e b _ { 1 } ) \| ^ { 2 } + \mathcal { D } _ { 1 } e ^ { T } \eta _ { 1 } - \alpha ^ { T } ( \mathcal { Y } + \mathcal { R } - ( C w _ { 1 } + e b _ { 1 } ) - e \epsilon _ { 1 } + \eta _ { 1 } ) } \\ { - \beta ^ { T } \eta _ { 1 } , \eqno { ( 4 ) } } \end{array}
$$

where $\alpha \in \mathbb { R } ^ { k \times 1 }$ and $\beta \in \mathbb { R } ^ { k \times 1 }$ are the vectors of Lagrangian multipliers. Using the Karush-Kuhn-Tucker (K.K.T.) conditions, we have

$$
\frac { \partial \mathcal { L } } { \partial w _ { 1 } } = - C ^ { T } ( \pmb { \mathcal { Y } } + \mathcal { R } - e \epsilon _ { 1 } - ( C w _ { 1 } + e b _ { 1 } ) ) + C ^ { T } \alpha ,\tag{5}
$$

$$
\frac { \partial \mathcal { L } } { \partial b _ { 1 } } = - e ^ { T } ( \boldsymbol { y } + \mathcal { R } - e \epsilon _ { 1 } - ( C w _ { 1 } + e b _ { 1 } ) ) + e ^ { T } \alpha ,\tag{6}
$$

$$
\frac { \partial \mathcal { L } } { \partial \eta _ { 1 } } = \mathcal { D } _ { 1 } e - \alpha - \beta .\tag{7}
$$

By jointly considering Eqs. (5) and (6), we obtain

$$
- { \binom { C ^ { T } } { e ^ { T } } } \left( ( y + \mathcal { R } - e \epsilon _ { 1 } ) - ( C ~ e ) { \binom { w _ { 1 } } { b _ { 1 } } } \right) + { \binom { C ^ { T } } { e ^ { T } } } \alpha = 0 .\tag{8}
$$

Define, $\mathcal { G } = ( C ~ e ) , \mathcal { F } = \boldsymbol { y } + \mathcal { R } - e \epsilon _ { 1 }$ , and $u _ { 1 } = ( w _ { 1 } b _ { 1 } ) ^ { T }$ , then (8) leads to

$$
\begin{array} { r } { - \boldsymbol { \mathcal { G } } ^ { T } \boldsymbol { \mathcal { F } } + \boldsymbol { \mathcal { G } } ^ { T } \boldsymbol { \mathcal { G } } \boldsymbol { u } _ { 1 } + \boldsymbol { \mathcal { G } } ^ { T } \boldsymbol { \alpha } = 0 , } \\ { \quad \quad } \\ { \mathrm { i . e . , } u _ { 1 } = ( \boldsymbol { \mathcal { G } } ^ { T } \boldsymbol { \mathcal { G } } ) ^ { - 1 } \boldsymbol { \mathcal { G } } ^ { T } ( \boldsymbol { \mathcal { F } } - \boldsymbol { \alpha } ) . } \end{array}\tag{9}
$$

The direct inversion of $\mathcal { G } ^ { T } \mathcal { G }$ is computationally demanding and may lead to numerical instability. To alleviate this issue, a regularization term δI, where I is an identity matrix of compatible size, is introduced into (9), thereby stabilizing the computation

and facilitating an eficient solution. Thus,

$$
u _ { 1 } = ( { \mathcal { G } } ^ { T } { \mathcal { G } } + \delta I ) ^ { - 1 } { \mathcal { G } } ^ { T } ( { \mathcal { F } } - \alpha ) .\tag{10}
$$

By applying the K.K.T. optimality conditions together with Eq. (10), the dual formulation corresponding to problem (2) can be derived as follows:

$$
\begin{array} { r l } & { \operatorname* { m a x } - \displaystyle \frac { 1 } { 2 } { \alpha } ^ { T } \mathscr { G } ( \mathscr { G } ^ { T } \mathscr { G } ) ^ { - 1 } \mathscr { G } ^ { T } \alpha + \mathscr { F } ^ { T } \mathscr { G } ( \mathscr { G } ^ { T } \mathscr { G } ) ^ { - 1 } \mathscr { G } ^ { T } \alpha - \mathscr { F } ^ { T } \alpha } \\ & { } \\ & { \mathrm { ~ s . t ~ } 0 \leq \alpha \leq { \mathscr { D } _ { 1 } e } . } \end{array}\tag{11}
$$

Similarly, the Wolfe dual associated with problem (3) is derived as follows:

$$
\begin{array} { r l } & { \operatorname* { m a x } - \cfrac { 1 } { 2 } \gamma ^ { T } \mathcal { G } ( \mathcal { G } ^ { T } \mathcal { G } ) ^ { - 1 } \mathcal { G } ^ { T } \gamma - \mathcal { H } ^ { T } \mathcal { G } ( \mathcal { G } ^ { T } \mathcal { G } ) ^ { - 1 } \mathcal { G } ^ { T } \gamma + \mathcal { H } ^ { T } \gamma } \\ & { \quad \mathrm { s . t ~ } 0 \leq \gamma \leq \mathcal { D } _ { 2 } e , } \end{array}\tag{12}
$$

where $\mathcal { H } = \boldsymbol { y } + \mathcal { R } + e \boldsymbol { \epsilon } _ { 2 }$ and

$$
u _ { 2 } = { \binom { w _ { 2 } } { b _ { 2 } } } = ( { \mathcal { G } } ^ { T } { \mathcal { G } } + \delta I ) ^ { - 1 } { \mathcal { G } } ^ { T } ( { \mathcal { H } } + \gamma ) .\tag{13}
$$

After computing the vectors $u _ { 1 }$ and $u _ { 2 }$ from (9) and (13), respectively, the corresponding lower and upper bounding regression functions are determined. The final regression estimate is then formed as follows:

$$
f ( c ) = \frac { 1 } { 2 } \big ( f _ { 1 } ( c ) + f _ { 2 } ( c ) \big ) = \frac { 1 } { 2 } ( \boldsymbol { w } _ { 1 } + \boldsymbol { w } _ { 2 } ) ^ { T } c + \frac { 1 } { 2 } ( b _ { 1 } + b _ { 2 } ) .\tag{14}
$$

## 4. The Proposed Granular Ball Twin Support Vector Quantile Regression

Conventional TSVR models often struggle to efectively capture heterogeneous and asymmetric data characteristics. The proposed Granular Ball Twin Support Vector

Quantile Regression (GB-TSVQR) model addresses these challenges by integrating granular ball computing with TSVR. This integration enables the use of high-level data abstractions, improving robustness against noise and outliers while preserving the data’s intrinsic structure. Furthermore, unlike traditional regression approaches, GB-TSVQR avoids large matrix inversions, resulting in a more computationally eficient, scalable training process suitable for complex, large-scale datasets.

The minimization problems of GB-TSVQR are as follows:

$$
\begin{array} { r l r } {  { \operatorname* { m i n } _ { w _ { 1 } , b _ { 1 } } \frac { 1 } { 2 } ( \| w _ { 1 } \| ^ { 2 } + b _ { 1 } ^ { 2 } ) + \mathcal { D } _ { 1 } \tau e ^ { T } ( \mathcal { Y } + \mathcal { R } - C w _ { 1 } - b _ { 1 } e ) } } \\ & { } & \\ & { } & { \quad + \mathcal { D } _ { 1 } e ^ { T } \eta _ { 1 } } \\ & { } & \\ & { } & { \quad \mathrm { s . t . } \ \mathcal { Y } + \mathcal { R } - ( C w _ { 1 } + e b _ { 1 } ) \geq e \epsilon _ { 1 } - \eta _ { 1 } , } \\ & { } & \\ & { } & { \quad \eta _ { 1 } \geq 0 , } \end{array}\tag{15}
$$

and

$$
\begin{array} { r l } & { \displaystyle \underset { w _ { 2 } , b _ { 2 } } { \operatorname* { m i n } } \frac { 1 } { 2 } ( \| w _ { 2 } \| ^ { 2 } + b _ { 2 } ^ { 2 } ) + \mathcal { D } _ { 2 } ( 1 - \tau ) e ^ { T } ( C w _ { 2 } + b _ { 2 } e - \pmb { \mathscr { Y } } - \pmb { \mathscr { R } } ) } \\ & { \quad \quad \quad \quad + \mathcal { D } _ { 2 } e ^ { T } \eta _ { 2 } } \\ & { \quad \quad \quad \quad \quad \quad \mathrm { s . t . } ( C w _ { 2 } + e b _ { 2 } ) - \pmb { \mathscr { Y } } - \mathscr { R } \geq e \epsilon _ { 2 } - \eta _ { 2 } , } \\ & { \quad \quad \quad \quad \eta _ { 2 } \geq 0 , } \end{array}\tag{16}
$$

where $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 2 }$ are positive regularization constants, $\eta _ { 1 }$ and $\eta _ { 2 }$ denote the slack variable vectors, $\epsilon _ { 1 }$ and $\epsilon _ { 2 }$ are strictly positive parameters, and $\tau \in ( 0 , 1 )$ represents the quantile parameter.

From (15) and (16), the proposed GB-TSVQR mitigates the impact of training errors through the quantile parameter τ. As τ varies from 0 to 1, the regression function f(c) shifts upward across training samples, capturing distributional characteristics at diferent quantile levels and enabling modeling of heterogeneous data via quantilespecific structures. GB-TSVQR is formulated as a pair of $\mathrm { Q P P s , }$ each with a single constraint set to construct either the lower or upper bounding regression function. For a fixed τ and dataset, $f _ { 1 } ( c )$ learns the $\epsilon _ { 1 }$ -insensitive lower-bound regressor, while $f _ { 2 } ( c )$ estimates the $\epsilon _ { 2 } .$ -insensitive upper-bound regressor. As these functions are nonparallel, the resulting regression band is asymmetric, allowing more accurate representation of quantile-dependent characteristics.

The Lagrangian corresponding to the problem (15) is given by:

$$
\begin{array} { l } { \displaystyle \mathcal { L } = \frac { 1 } { 2 } ( \| w _ { 1 } \| ^ { 2 } + b _ { 1 } ^ { 2 } ) + \mathcal { D } _ { 1 } e ^ { T } \eta _ { 1 } + \mathcal { D } _ { 1 } \tau e ^ { T } ( \pmb { \mathcal { Y } } + \mathcal { R } - C w _ { 1 } - b _ { 1 } e ) } \\ { \displaystyle - \alpha ^ { T } ( \pmb { \mathcal { Y } } + \mathcal { R } - ( C w _ { 1 } + e b _ { 1 } ) - \epsilon _ { 1 } + \eta _ { 1 } ) - \beta ^ { T } \eta _ { 1 } , } \end{array}\tag{17}
$$

where $\alpha \in \mathbb { R } ^ { k \times 1 }$ and $\beta \in \mathbb { R } ^ { k \times 1 }$ denote the Lagrange multiplier vectors. By applying the K.K.T. optimality conditions, we obtain

$$
\frac { \partial \mathcal { L } } { \partial w _ { 1 } } = w _ { 1 } - \mathcal { D } _ { 1 } \tau C ^ { T } e + C ^ { T } \alpha ,\tag{18}
$$

$$
\frac { \partial \mathcal { L } } { \partial b _ { 1 } } = b _ { 1 } - \mathcal { D } _ { 1 } \tau e ^ { T } e + e ^ { T } \alpha ,\tag{19}
$$

$$
\frac { \partial \mathcal { L } } { \partial \eta _ { 1 } } = \mathcal { D } _ { 1 } e - \alpha - \beta .\tag{20}
$$

Define $\mathcal { G } = ( C \it { \Delta } e )$ and $u _ { 1 } = ( w _ { 1 } b _ { 1 } ) ^ { T }$ , then (18) and (19) leads to

$$
u _ { 1 } = \mathcal { D } _ { 1 } \tau \boldsymbol { G } ^ { T } \boldsymbol { e } - \boldsymbol { G } ^ { T } \alpha .\tag{21}
$$

Using the above KKT conditions, we obtain the following dual representations of optimization problem (15),

$$
\begin{array} { l } { \operatorname* { m a x } - \displaystyle \frac { 1 } { 2 } \alpha ^ { T } ( \boldsymbol { \mathcal { G } } \boldsymbol { \mathcal { G } } ^ { T } ) \alpha + \mathcal { D } _ { 1 } \tau e ^ { T } ( \boldsymbol { \mathcal { G } } \boldsymbol { \mathcal { G } } ^ { T } ) \alpha - ( \boldsymbol { \mathcal { Y } } + \mathcal { R } - \epsilon _ { 1 } e ) ^ { T } \alpha } \\ { \mathrm { ~ s . t ~ } 0 \leq \alpha \leq \mathcal { D } _ { 1 } e . } \end{array}\tag{22}
$$

Similarly, the Wolfe dual associated with problem (16) is derived as follows:

$$
\begin{array} { r l } & { \operatorname* { m a x } - \displaystyle \frac { 1 } { 2 } \gamma ^ { T } ( \boldsymbol { \mathcal { G } } \boldsymbol { \mathcal { G } } ^ { T } ) \gamma + \mathcal { D } _ { 2 } ( 1 - \tau ) e ^ { T } ( \boldsymbol { \mathcal { G } } \boldsymbol { \mathcal { G } } ^ { T } ) \gamma + ( \boldsymbol { \mathcal { Y } } + \boldsymbol { \mathcal { R } } + \epsilon _ { 2 } e ) ^ { T } \gamma } \\ & { \quad \mathrm { s . t } \ 0 \leq \gamma \leq \mathcal { D } _ { 2 } e , } \end{array}\tag{23}
$$

where

$$
\boldsymbol { u } _ { 2 } = \binom { w _ { 2 } } { b _ { 2 } } = - \mathcal { D } _ { 2 } ( 1 - \tau ) \boldsymbol { \mathcal { G } } ^ { T } \boldsymbol { e } + \boldsymbol { \mathcal { G } } ^ { T } \boldsymbol { \gamma } .\tag{24}
$$

To solve optimization problems (22) and (23), we employ the dual coordinate descent method (DCDM) [31, 32]. Let $\widetilde { \mathcal { G } } = \mathcal { G } \mathcal { G } ^ { T } , p _ { 1 } = \mathcal { D } _ { 1 } \tau \mathcal { G } \mathcal { G } ^ { T } e - \boldsymbol { y } - \boldsymbol { \mathcal { R } } + \epsilon _ { 1 } e$ , and $p _ { 2 } = \mathcal { D } _ { 2 } ( 1 - \tau ) \mathcal { G } \mathcal { G } ^ { T } e + \boldsymbol { y } + \mathcal { R } + \epsilon _ { 2 } e$ . With these definitions, the optimization problems in (22) and (23) can be reformulated as follows:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { \alpha } f ( \alpha ) = \frac { 1 } { 2 } \alpha ^ { T } \widetilde { \mathcal { G } } \alpha - p _ { 1 } ^ { T } \alpha } \\ { \mathrm { s . t } 0 \leq \alpha \leq \mathcal { D } _ { 1 } e , } \end{array}\tag{25}
$$

and

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { \boldsymbol { \gamma } } f ( \boldsymbol { \gamma } ) = \frac { 1 } { 2 } \boldsymbol { \gamma } ^ { T } \widetilde { \mathcal { G } } \boldsymbol { \gamma } - p _ { 2 } ^ { T } \boldsymbol { \gamma } } \\ { \mathrm { s . t } \ \boldsymbol { 0 } \leq \boldsymbol { \gamma } \leq \mathcal { D } _ { 2 } \boldsymbol { e } . } \end{array}\tag{26}
$$

The DCDM procedure begins with randomly initialized vectors α and $\gamma ^ { 0 } .$ . At each iteration, a single coordinate is chosen and updated by solving the corresponding onedimensional optimization problem, while all remaining variables are held fixed.

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { \kappa } f ( \alpha _ { j } + \kappa I _ { j } ) } \\ { \mathrm { s . t . ~ } 0 \leq \alpha _ { j } + \kappa \leq \mathcal { D } _ { 1 } , ~ j = 1 , 2 , \ldots , k , } \end{array}\tag{27}
$$

and

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { \kappa ^ { * } } g ( \gamma _ { j } + \kappa ^ { * } I _ { j } ) } \\ { \mathrm { s . t . ~ } 0 \leq \gamma _ { j } + \kappa ^ { * } \leq \mathcal { D } _ { 2 } , \displaystyle j = 1 , 2 , \ldots , k . } \end{array}\tag{28}
$$

where $I _ { j }$ denotes the vector with 1 in the $j ^ { t h }$ coordinate and 0 elsewhere. The objective functions of (27) and (28) are simplified as:

$$
f ( \alpha _ { i } + \kappa I _ { j } ) = \frac { 1 } { 2 } \widetilde { \mathcal { G } } _ { j j } \kappa ^ { 2 } + \nabla _ { j } f ( \alpha _ { j } ) \kappa + f ( \alpha _ { j } ) , j = 1 , 2 , \ldots , k ,\tag{29}
$$

and

$$
g ( \gamma _ { i } + \kappa ^ { * } I _ { j } ) = \frac { 1 } { 2 } \widetilde { \mathcal { G } } _ { j j } { \kappa ^ { * } } ^ { 2 } + \nabla _ { j } g ( \gamma _ { j } ) { \kappa ^ { * } } + f ( \kappa _ { j } ) , j = 1 , 2 , \ldots , k ,\tag{30}
$$

where $\nabla _ { j } f$ and $\nabla _ { j } g$ denote the $j ^ { t h }$ components of the gradient vectors $\nabla f$ and $\nabla g$ respectively. By incorporating the box constraints $0 \leq \alpha _ { j } + \kappa \leq \mathcal { D } _ { 1 }$ and $0 \leq \gamma _ { j } + \kappa ^ { * } \leq$ $\mathcal { D } _ { 2 } ,$ the solutions that minimize problems (27) and (28) yield the following updates under bounded constraints:

$$
\alpha ^ { \mathrm { n e w } } = \operatorname* { m i n } \Bigg ( \operatorname* { m a x } \Bigg ( \alpha _ { j } - \frac { \nabla _ { j } f ( \alpha _ { j } ) } { \widetilde { g } _ { j j } } , 0 \Bigg ) , \mathcal { D } _ { 1 } \Bigg ) ,\tag{31}
$$

and

$$
\gamma ^ { \mathrm { n e w } } = \operatorname* { m i n } \left( \operatorname* { m a x } \left( \gamma _ { j } - \frac { \nabla _ { j } g ( \gamma _ { j } ) } { \widetilde { g } _ { j j } } , 0 \right) , \mathcal { D } _ { 2 } \right) .\tag{32}
$$

After obtaining the optimal dual variables α and γ using Algorithm 1, the corresponding lower and upper bounding regression functions can be expressed as:

$$
\begin{array} { r } { f _ { 1 } ( \boldsymbol { c } ) = \mathcal { D } _ { 1 } \tau e ^ { T } \mathcal { G } \boldsymbol { c } - \boldsymbol { \alpha } ^ { T } \mathcal { G } \boldsymbol { c } , } \end{array}\tag{33}
$$

Algorithm 1 Dual Coordinate Descent Method (DCDM)   
1: Input: Input dataset $( C , { \mathcal { R } } , { \mathcal { Y } } ) .$ , regularization parameters $\mathcal { D } _ { 1 } , \mathcal { D } _ { 2 }$ , quantile param  
eter $\tau ,$ and tube parameters $\epsilon _ { 1 } , \epsilon _ { 2 }$   
2: Output: Optimal dual variables α and $\gamma .$   
3: Initialize $\alpha = 0$ and $\gamma = 0 .$   
4: Calculate   
$\widetilde { \boldsymbol { G } } = \boldsymbol { G } \boldsymbol { G } ^ { T } ,$   
$p _ { 1 } = \mathcal { D } _ { 1 } \tau \mathcal { G } \mathcal { G } ^ { T } e - \mathcal { Y } - \mathcal { R } + \epsilon _ { 1 } e ,$   
and   
$p _ { 2 } = \mathcal { D } _ { 2 } ( 1 - \tau ) \mathcal { G } \mathcal { G } ^ { T } e + \boldsymbol { y } + \mathcal { R } + \epsilon _ { 2 } \boldsymbol { e } .$   
5: if α has not converged then   
6: for $j = 1 , 2 , \dotsc , n$ do   
7: $\nabla _ { j } f ( \alpha ) \gets ( \bar { G } \alpha ) _ { \underline { { j } } } - p _ { 1 j }$   
8: $\begin{array} { r l } & { \alpha _ { j } \gets \operatorname* { m i n } \Bigl \{ \operatorname* { m a x } \Bigl [ \alpha _ { j } - \frac { \top _ { j } f ( \alpha ) } { \widetilde { \mathcal { G } } _ { j j } } , 0 \Bigr ] , \mathcal { D } _ { 1 } \Bigr \} } \\ & { \quad \ldots \ : } \end{array}$   
9: end for   
10: end if   
11: if γ has not converged then   
12: for $j = 1 , 2 , . . . . , n$ do   
13: $\nabla _ { j } g ( \gamma )  ( \mathcal { \bar { G } } \gamma ) _ { \underline { { j } } } - p _ { 2 j }$   
14: $\begin{array} { r } { \gamma _ { j }  \operatorname* { m i n } \biggr \{ \operatorname* { m a x } \biggr [ \gamma _ { j } - \frac { \mathsf { \tilde { \nabla } } _ { j } g ( \gamma ) } { \tilde { g } _ { j j } } , 0 \biggr ] , \mathcal { D } _ { 2 } \biggr \} } \end{array}$   
15: end for   
16: end if

and

$$
\begin{array} { r } { f _ { 2 } ( c ) = - \mathcal { D } _ { 2 } ( 1 - \tau ) e ^ { T } \mathcal { G } c + \gamma \mathcal { G } c . } \end{array}\tag{34}
$$

The final regression function is then defined as the average of these two bounds,

$$
\begin{array} { l } { \displaystyle f ( c ) = \frac { 1 } { 2 } \big ( f _ { 1 } ( c ) + f _ { 2 } ( c ) \big ) } \\ { \displaystyle \quad = \frac { 1 } { 2 } ( \gamma - \alpha ) ^ { T } \mathcal { G } c + \frac { 1 } { 2 } \left[ \mathcal { D } _ { 1 } \tau - \mathcal { D } _ { 2 } ( 1 - \tau ) \right] e ^ { T } \mathcal { G } c . } \end{array}\tag{35}
$$

The first component of $f ( c )$ , governed by the diference between γ and $\alpha ,$ captures the asymmetric characteristics associated with diferent quantile locations of the training samples. The second component, modulated by the quantile parameter $\tau ,$ models the

heterogeneity present across quantile levels.

## 5. Experimental Results

This section presents a comprehensive empirical evaluation of the proposed GBTSVR and GB-TSVQR models across diverse datasets, including UCI benchmarks, synthetic data, and time-series datasets. The performance of GBTSVR and GB-TSVQR is systematically compared with several representative state-of-the-art regression models, such as support vector regression (SVR) [5], twin SVR (TSVQR) [21], twin support vector quantile regression (TSVQR) [21], $L _ { \mathrm { 1 } } \mathrm { - n o r m }$ twin support vector quantile regression $\left( L _ { 1 } { - } \mathrm { T S V Q R } \right)$ [33], and twin proximal support vector regression with Gauss-Laplace mixed noise (GL-TPSVR) [34]. The experimental setup is presented in Section S.II, and experiments on time-series datasets are described in Section S.III of the Supplementary Material.

## 5.1. Experimental Evaluation on UCI Datasets

To evaluate the efectiveness of the proposed GBTSVR and GB-TSVQR models, experiments are conducted on benchmark datasets from the UCI Machine Learning Repository [35]. The datasets are grouped by size: small-scale (103–500 samples), medium-scale (501–1,000), and large-scale (1,001–20,640). Table 1 reports the performance of GBTSVR, GB-TSVQR, and baseline models (SVR, TSVR, TSVQR, $L _ { \mathrm { 1 } } – \mathrm { T S V Q R }$ , and GL-TPSVR) using RMSE, MAE, Pos Error, and $\mathrm { N e g }$ Error, with RMSE as the primary metric (lower is better). GB-TSVQR shows consistently strong results, achieving the lowest RMSE on 13 of 18 datasets, second place on 3, third on 1, and fourth on 1, demonstrating stable and competitive performance across diverse data characteristics. GBTSVR attains top performance on 2 datasets and ranks second on 12, highlighting its strong competitiveness. The average RMSE of GBTSVR and GB-TSVQR are 0<sub>.</sub>01757 and 0<sub>.</sub>04343, respectively, compared to 0<sub>.</sub>21323 (SVR), 0 45104 (TSVR), 0 87892 (TSVQR), 0 193 $\left( L _ { 1 } { \mathrm { - } } \mathrm { T S V Q R } \right)$ , and 0 74136 (GL-TPSVR), showing clear performance gains. For a more reliable assessment, models are ranked per dataset. Table 1 reports mean ranks based on RMSE, where lower values indicate better performance. GBTSVR and GB-TSVQR achieve the best average ranks of 2<sub>.</sub>17 and 1<sub>.</sub>44, outperforming SVR (3<sub>.</sub>44), TSVR (4<sub>.</sub>89), TSVQR (6<sub>.</sub>39), L<sub>1</sub>-TSVQR (3<sub>.</sub>67), and GL-TPSVR (6<sub>.</sub>00), confirming their superior and consistent performance across datasets.

Table 1: Performance comparison in terms of RMSE, MAE, Pos Error, and Neg Error on benchmark UCI datasets for the proposed GBTSVR and GB-TSVQR models with baseline models.
<table><tr><td>Dataset 2014_2015 CSM dataset</td><td>Metric</td><td>SVR [5]</td><td>TSVR [17]</td><td>TSVQR [21]</td><td>L1-TSVQR [33]</td><td>GL-TPSVR [34]</td><td>GBTSVR</td><td>GB-TSVQR 0.00065</td></tr><tr><td>(185 × 11)</td><td>RMSE MAE Pos Error</td><td>0.37972 0.22234 0.43571</td><td>1.33951 0.77661 0.96693</td><td>1.23384 0.79133 1.20578</td><td>0.40545 0.39927 0.9502</td><td>0.44613 0.35916 0.66887</td><td>0.00755 0.00366 0.00601</td><td>0.00057 0.00058</td></tr><tr><td>Abalone (4177× 7)</td><td>Neg Error RMSE MAE</td><td>0.17018 0.0062 0.00448 0.00327</td><td>0.45941 0.38768 0.2541 0.26615</td><td>0.50247 0.07667 0.07666</td><td>0.01384 0.00656 0.00652 3.00031</td><td>0.3288 0.45811 0.42865 0.8016</td><td>0.00213 0.00265 0.00216</td><td>0.00006 0.00073 0.00062 0.00069</td></tr><tr><td>Ailerons (13750×40)</td><td>Pos Error Neg Error RMSE MAE Pos Error</td><td>0.00632 0.02209 0.01923 0.02077</td><td>0.00008 0.33117 0.05774 0.0367</td><td>1.20578 6.07666 0.66636 0.52217 0.58268</td><td>0.00294 0.25623 0.12289 0.25647</td><td>0.01387 0.37509 0.37486 0.38727</td><td>0.00178 0.00252 0.04457 0.04232 0.02988</td><td>0.00026 0.00488 0.00411 0.00347</td></tr><tr><td>Airfoil_Self_Noise (1503 × 5)</td><td>Neg Error RMSE MAE Pos Error Neg Error</td><td>0.00786 0.28274 0.20323 0.3071 0.13614</td><td>0.24246 0.00089 0.00062 0.22328 0.76481</td><td>0.25812 0.25334 0.20893 0.32777</td><td>0.00828 0.01395 0.00831 0.66023</td><td>0.37486 0.38897 0.31075 0.73755</td><td>0.04323 0.0068 0.00636 0.00424</td><td>0.00432 0.00372 0.00345 0.00415</td></tr><tr><td>Auto-price (159×15)</td><td>RMSE MAE Pos Error Neg Error</td><td>0.24931 0.19605 0.23621 0.07557</td><td>0.38768 0.07398 0.12724 0.32189</td><td>0.17763 0.80669 0.50819 0.57795</td><td>0.00334 0.28821 0.18854 0.69921</td><td>0.00057 3.28642 3.27921 3.27921</td><td>0.00658 0.02016 0.01866 0.01919</td><td>0.00335 0.00621 0.00463 0.00492</td></tr><tr><td>California_Housing (20640× 8)</td><td>RMSE MAE Pos Error Neg Error</td><td>0.03665 0.02896 0.03217 0.01771</td><td>0.09586 0.06542 0.11739 0.03737</td><td>0.32036 1.08419 0.6768 0.8649 0.3006</td><td>0.02409 0.06521 0.05708 0.76576 0.01636</td><td>0.07104 0.5106 0.50848 0.57085 0.50848</td><td>0.00644 0.00327 0.00191 0.00086</td><td>0.00416 0.00216 0.00148</td></tr><tr><td>cpu_pref (209×9)</td><td>RMSE MAE Pos Error Neg Error RMSE</td><td>0.04586 0.03447 0.0291 0.04264</td><td>0.27086 0.08627 0.09582 0.03893</td><td>1.40836 0.52211 0.68427 0.21991</td><td>0.06605 0.05858 0.85349 0.00574</td><td>0.40356 0.4009 0.94783 0.4009</td><td>0.00201 0.00694 0.00429 0.00576 0.00156</td><td>0.50848 0.00326 0.00264 0.02488 0.02303</td></tr><tr><td>Delta_Ailerons (7129× 5)</td><td>MAE Pos Error Neg Error RMSE</td><td>0.04249 0.03477 0.03072 0.03713</td><td>0.56647 0.3872 0.37268 0.40281</td><td>0.79094 0.62824 0.59352 0.65489</td><td>0.04912 0.03218 0.83757 0.00208</td><td>1.21404 1.21235 0.85387 0.21235</td><td>0.00804 0.00662 0.00608 0.00696</td><td>0.00108 0.00088 0.00959 0.00443</td></tr><tr><td>Elevators (16599 × 18)</td><td>MAE Pos Error Neg Error RMSE</td><td>0.03372 0.0287 0.03375 0.02553</td><td>0.13731 0.10325 0.91917 0.0711</td><td>0.66501 0.54301 0.6381 0.49129</td><td>0.03541 0.02593 0.74026 0.01173</td><td>0.80182 0.80105 0.80105 0.1463</td><td>0.01107 0.00805 0.00936 0.00386</td><td>0.00033 0.00029 0.0003 0.00017</td></tr><tr><td>Forest_Fires (517×12)</td><td>MAE Pos Error Neg Error RMSE</td><td>0.0651 0.05662 0.05158 0.0597 0.54586</td><td>0.43301 0.20324 0.68209 0.13282</td><td>0.43246 0.30098 0.33517 0.27568</td><td>0.04367 0.02473 0.63219 0.12326</td><td>1.04341 1.04249 1.04249 0.00442</td><td>0.07838 0.07397 0.0828 0.07337</td><td>0.00382 0.00271 0.00371 0.00252</td></tr><tr><td>(522 × 19)</td><td>MAE Pos Error Neg Error RMSE MAE</td><td>0.40149 0.48379 0.37045 0.67367</td><td>0.9994 0.53642 0.82147 0.38698 0.45633</td><td>1.14417 0.73335 0.78185 0.63256 1.11197</td><td>0.59813 0.09423 0.85473 0.01151</td><td>0.8 0.72305 0.70685 0.72451</td><td>0.00333 0.00208 0.0026 0.00147</td><td>0.00257 0.00208 0.70683 0.72455</td></tr><tr><td>(536×8)</td><td>Pos Error Neg Error RMSE MAE</td><td>0.5302 0.53896 0.52346 0.06821</td><td>0.35412 0.37481 0.32918 0.33652</td><td>0.88999 0.64741 1.05004 0.78118</td><td>0.6076 0.09137 0.956 0.00774 0.06562</td><td>0.68112 0.53737 0.5555 0.52306 0.20268</td><td>0.00143 0.00107 0.00108 0.00105 0.06314</td><td>0.67283 0.52719 0.4517 0.57572 0.02465</td></tr><tr><td></td><td>Pos Error Neg Error RMSE MAE</td><td>0.03258 0.88743 0.66495 0.04578 0.03507</td><td>0.25849 0.25296 0.26427 0.25929</td><td>0.76648 0.76648 1.42405</td><td>0.06023 0.84386 0.00478 0.11576</td><td>0.18235 0.34912 0.14317 0.40139</td><td>0.06299 0.6299 0.03515</td><td>0.01683 0.20009 0.19349 0.03959</td></tr><tr><td>Parkinsons_Telemonitoring</td><td>Pos Error Neg Error RMSE MAE</td><td>0.03398 0.03685 0.13258 0.1195</td><td>0.04277 0.00007 0.0289 0.38589 0.28757</td><td>0.57988 0.94067 0.2519 1.38416 0.89471</td><td>0.11086 0.8536 0.00155 0.79682 0.33874</td><td>0.39876 0.19543 0.39876 0.7042 0.69278</td><td>0.00596 0.00619 0.00126 0.00611 0.00494</td><td>0.03358 0.03561 0.02577 0.00334 0.00227</td></tr><tr><td>qsar_aquatic toxicity (546×8)</td><td>Pos Error Neg Error RMSE MAE</td><td>0.11572 0.12403 1.17137</td><td>0.29069 0.28483 0.70742</td><td>1.08453 0.63711 0.94532</td><td>0.77838 0.6384 0.03365</td><td>0.82559 0.69278 0.83423</td><td>0.00532 0.00443 0.00348</td><td>0.00203 0.00242 0.0007 0.00048</td></tr><tr><td>slump_test</td><td>Pos Error Neg Error RMSE</td><td>1.05181 1.77232 0.8905 0.03548</td><td>0.50963 0.495 0.52322 0.15373</td><td>0.76585 0.87165 0.63717 0.65404</td><td>0.01587 0.92651 0.00558 0.0038</td><td>0.59304 1.30703 0.33931 0.62284</td><td>0.00221 0.00204 0.00257 0.01146</td><td>0.00037 0.0009 0.00774</td></tr><tr><td>Stocks_Domain</td><td>MAE Pos Error Neg Error RMSE</td><td>0.02898 0.01786 0.03601 0.00139</td><td>0.11546 0.12202 0.10931 0.86978</td><td>0.46946 0.48837 0.45173 0.95779</td><td>0.00193 0.76379 0.00698 0.02281</td><td>0.62208 0.96907 0.62208 0.16985</td><td>0.00876 0.00565 0.01132 0.00265</td><td>0.00109 0.03158 0.06677 0.00344</td></tr><tr><td>(4177×7)</td><td>MAE Pos Error Neg Error</td><td>0.00133 0.00125 0.0022</td><td>0.71006 0.6632 0.45319</td><td>0.63238 1.31818</td><td>0.01859 0.96242</td><td>0.16984 0.77989</td><td>0.00216 0.00178</td><td>0.00261 0.00087</td></tr><tr><td>Average RMSE Average Rank</td><td></td><td>0.21323 3.44</td><td>0.45104 4.89</td><td>0.3772 0.87892 6.39</td><td>0.00619 0.193 3.67</td><td>0.16984 0.74136 6</td><td>0.00252 0.01757 2.17</td><td>0.00304 0.04343 1.44</td></tr></table>

The proposed models is denoted by <sup>†</sup>.

To further substantiate the performance of the proposed GBTSVR and GB-TSVQR models, we carried out a rigorous statistical evaluation using the Friedman test, followed by the Nemenyi post hoc analysis. The Friedman test [36] is employed to examine whether the observed diferences among competing models are statistically significant. Under this test, all models are ranked separately on each dataset, and the null hypothesis assumes no significant performance diference among the models, meaning their average ranks are equal. The resulting Friedman statistic follows a chi-squared distribution, denoted as $\chi _ { F } ^ { 2 }$ , with degrees of freedom equal to $p - 1$ where $p$ is the total number of compared models. This statistic is computed as follows $\begin{array} { r } { \chi _ { F } ^ { 2 } = { \frac { 1 2 N } { p ( p + 1 ) } } \left\lceil \sum _ { j } \Re _ { j } ^ { 2 } - { \frac { p ( p + 1 ) ^ { 2 } } { 4 } } \right\rceil } \end{array}$ where N represents the total number of datasets, while $\Re _ { j }$ denotes the average rank of the $j ^ { t h }$ model among the $p$ competing models. The $F _ { F }$ statistic is calculated using the formula: $\begin{array} { r } { F _ { F } = \frac { ( N - 1 ) \chi _ { F } ^ { 2 } } { N ( p - 1 ) - \chi _ { F } ^ { 2 } } } \end{array}$ , where the F-distribution has (p − 1) and $( N - 1 ) \times ( p - 1 )$ degrees of freedom. For $p \ = \ 7$ models evaluated on N = 18 datasets, the Friedman test yielded $\chi _ { F } ^ { 2 } = 8 0 . 3 4$ and a corresponding $F _ { F } = 4 9 . 3 8$ . The $F _ { F }$ statistic follows an F-distribution with (6 102) degrees of freedom. From the F-distribution table, the critical value at a 5% significance level is 2 1887. Since the observed $F _ { F }$ value exceeds this threshold, the null hypothesis is rejected, indicating statistically significant diferences among the models. Next, we perform the Nemenyi post hoc test [37] to conduct pairwise comparisons among the models. A significant diference in performance between any two models is identified when the diference in their average ranks exceeds a predefined threshold called the critical diference (C.D.). If the diference in mean ranks between two models is greater than the C.D., the model with the higher rank is considered statistically better than the one with the lower rank. The C.D. is calculated using the following formula: $\begin{array} { r } { \mathrm { C . D . } = q _ { \alpha } \sqrt { \frac { p ( p + 1 ) } { 6 N } } } \end{array}$ , where $q _ { \alpha }$ represents the two-tailed critical value for the Nemenyi test, obtained from the corresponding distribution table. Referring to the table, the calculated C.D. at a 5% significance level is 2 12, with $q _ { \alpha } = 2 . 9 4 9$ . The diferences in average ranks between the GBTSVR and GB-TSVQR models with the baseline SVR, TSVR, TSVQR, $L _ { \mathrm { 1 } } { \mathrm { - } } \mathrm { T S V Q R }$ , and GL-TPSVR models are (1 27 2), (2 72 3 45), (4 22 4 95), (1 50 2 23), and (3 83 4 56), respectively. Based on the Nemenyi post hoc test, the proposed GBTSVR model demonstrates statistically significant improvements over most baseline models, except SVR and $L _ { \mathrm { 1 } } { \mathrm { - } } \mathrm { T S V Q R }$ . Similarly, GB-TSVQR outperforms all baselines except SVR. Both GBTSVR and GB-TSVQR achieve lower average ranks than SVR, with GBTSVR also ranking below $L _ { \mathrm { 1 } } – \mathrm { T S V Q R }$ , indicating superior generalization. Along with consistently low RMSE and strong performance across statistical analyses, these results confirm that GBTSVR and GB-TSVQR outperform existing baselines in both accuracy and generalization.

## 5.2. Experimental Evaluation on Synthetic Datasets

Table 2: Overview of the synthetic datasets used in the experiments.
<table><tr><td>Dataset</td><td>Generating Function</td><td>Input Domain</td></tr><tr><td>Function 1</td><td> $y _ { i } = s i n ( x _ { i } ) + \epsilon _ { i }$ </td><td> $x _ { i } \in [ 0 , 2 \pi ]$ </td></tr><tr><td>Function 2</td><td> $\begin{array} { r } { y _ { i } = \frac { s i n ( 3 x _ { i } ) } { 3 x _ { i } } + \epsilon _ { i } } \end{array}$ </td><td> $x _ { i } \in [ - 4 , 4 ]$ </td></tr><tr><td>Function 3</td><td> $y _ { i } = s i n ( x _ { i } ) c o s ( x _ { i } ^ { 2 } ) + \epsilon _ { i }$ </td><td> $x _ { i } \in [ 0 , 2 \pi ]$ </td></tr><tr><td>Function 4</td><td> $y _ { i } = x _ { i } c o s ( x _ { i } ) + \epsilon _ { i }$ </td><td> $x _ { i } \in [ - 4 , 4 ]$ </td></tr><tr><td>Function 5</td><td> $y _ { i } = ( 2 x _ { i } ^ { 2 } - x _ { i } + 1 ) \exp ( - \frac { x _ { i } ^ { . } } { 2 } ) + \epsilon _ { i }$ </td><td> $x _ { i } \in [ - 4 , 4 ]$ </td></tr></table>

The noise term $\epsilon _ { i }$ follows diferent distributions depending on the dataset:  
$- \ N ( 0 , 0 . 2 ^ { 2 } )$ denotes a Gaussian random variable with zero mean and standard deviation 0 2. $- \mathcal { U } [ - 0 . 2 , 0 . 2 ]$ represents a uniformly distributed random variable over the interval [-0.2, 0.2]. $- \mathcal { T } ( 1 0 )$ indicates a Student’s t-distributed random variable with 10 degrees of freedom.

To conduct a more comprehensive assessment of the proposed GBTSVR and GB-TSVQR models against the competing baseline models, this subsection considers a set of synthetic datasets corrupted by diferent noise patterns. The detailed characteristics of these datasets are summarized in Table 2. Specifically, five distinct synthetic functions are generated, and each function is contaminated with three types of noise, Gaussian, uniform, and Student’s t-distributed noise, yielding a total of 15 synthetic datasets. Each dataset consists of 500 data samples, enabling a systematic evaluation of model robustness under varying noise conditions. Table 3 reports the RMSE results obtained by the proposed GBTSVR and GB-TSVQR models in comparison with the baseline approaches on the synthetic datasets. Across the synthetic datasets, the proposed GBTSVR and GB-TSVQR models consistently demonstrate strong performance. They achieve superior results in most cases and remain highly competitive on the remaining datasets, indicating reliable efectiveness on each dataset. These outcomes highlight the robustness and stable generalization capability of the proposed models under diverse noise conditions.

Table 3: RMSE performance on synthetic datasets comparing the proposed GBTSVR and GB-TSVQR models with baseline models.
<table><tr><td>Dataset</td><td>Metric</td><td>SVR [5]</td><td>TSVR [17]</td><td>TSVQR [21]</td><td>L1-TSVQR [33]</td><td>GL-TPSVR [34]</td><td>GBTSVR{</td><td>GB-TSVQR†</td></tr><tr><td colspan="9">Type 1: Gaussian noise with zero mean and a standard deviation of 0.2.</td></tr><tr><td>Function 1</td><td>RMSE</td><td>0.98218</td><td>0.65454</td><td>0.65441</td><td>0.98456</td><td>0.70711</td><td>0.00081</td><td>0.00067</td></tr><tr><td>Function 2</td><td>RMSE</td><td>1.34294</td><td>0.66864</td><td>0.66667</td><td>0.95784</td><td>0.75896</td><td>0.00095</td><td>0.00042</td></tr><tr><td>Function 3</td><td>RMSE</td><td>1.12861</td><td>0.78956</td><td>0.62874</td><td>0.85742</td><td>0.84528</td><td>0.00085</td><td>0.00058</td></tr><tr><td>Function 4</td><td>RMSE</td><td>0.95022</td><td>0.74268</td><td>0.45929</td><td>0.91428</td><td>0.71587</td><td>0.00074</td><td>0.00062</td></tr><tr><td>Function 5</td><td>RMSE</td><td>0.96827</td><td>0.79533</td><td>0.59533</td><td>0.87458</td><td>0.70711</td><td>0.00082</td><td>0.00063</td></tr><tr><td colspan="9">Type 2: Uniformly distributed noise within the range [-0.2, , 0.2].</td></tr><tr><td>Function 1</td><td>RMSE</td><td>0.971238</td><td>0.63034</td><td>0.63029</td><td>0.98456</td><td>0.70711</td><td>0.00104</td><td>0.00062</td></tr><tr><td>Function 2</td><td>RMSE</td><td>1.34184</td><td>0.67669</td><td>0.62668</td><td>0.96482</td><td>0.72894</td><td>0.00092</td><td>0.00045</td></tr><tr><td>Function 3</td><td>RMSE</td><td>1.27608</td><td>0.75482</td><td>0.60874</td><td>0.82598</td><td>0.82874</td><td>0.00082</td><td>0.00061</td></tr><tr><td>Function 4</td><td>RMSE</td><td>0.99742</td><td>0.74744</td><td>0.44749</td><td>0.90748</td><td>0.70711</td><td>0.00072</td><td>0.00062</td></tr><tr><td>Function 5</td><td>RMSE</td><td>0.92873</td><td>0.78927</td><td>0.61278</td><td>0.85678</td><td>0.79684</td><td>0.00081</td><td>0.00065</td></tr><tr><td colspan="9">Type 3: Noise modeled by a Student&#x27;s t-distribution with 10 degrees of freedom.</td></tr><tr><td>Function 1</td><td>RMSE</td><td>0.99722</td><td>0.63318</td><td>0.63296</td><td>0.97854</td><td>0.70711</td><td>0.00085</td><td>0.00065</td></tr><tr><td>Function 2</td><td>RMSE</td><td>1.37822</td><td>0.66392</td><td>0.66387</td><td>0.89827</td><td>0.70824</td><td>0.00089</td><td>0.00046</td></tr><tr><td>Function 3</td><td>RMSE</td><td>1.21161</td><td>0.72847</td><td>0.65274</td><td>0.80472</td><td>0.79862</td><td>0.00082</td><td>0.0006</td></tr><tr><td>Function 4</td><td>RMSE</td><td>0.99864</td><td>0.75625</td><td>0.45721</td><td>0.95287</td><td>0.75789</td><td>0.00075</td><td>0.00065</td></tr><tr><td>Function 5</td><td>RMSE</td><td>0.92874</td><td>0.74982</td><td>0.62874</td><td>0.84692</td><td>0.78462</td><td>0.00081</td><td>0.00063</td></tr></table>

The proposed models is denoted by <sup>†</sup>.

Table 4: RMSE comparison on synthetic datasets with injected outliers for the proposed GBTSVR and GB-TSVQR models and baseline models.
<table><tr><td>Dataset</td><td>Metric</td><td>SVR [5]</td><td>TSVR [17]</td><td>TSVQR [21]</td><td>L1-TSVQR [33]</td><td>GL-TPSVR [34]</td><td>GBTSVR</td><td>GB-TSVQR{</td></tr><tr><td colspan="9">Type 1: Gaussian noise with zero mean and a standard deviation of 0.2.</td></tr><tr><td>Function 1</td><td>RMSE</td><td>1.00601</td><td>0.82001</td><td>0.82043</td><td>0.96129</td><td>0.93508</td><td>0.00076</td><td>0.00018</td></tr><tr><td>Function 2</td><td>RMSE</td><td>0.30138</td><td>1.49669</td><td>1.49712</td><td>0.92229</td><td>0.93209</td><td>0.94228</td><td>0.0204</td></tr><tr><td>Function 3</td><td>RMSE</td><td>0.87983</td><td>1.46122</td><td>0.31359</td><td>0.9801</td><td>0.93056</td><td>0.00097</td><td>0.00018</td></tr><tr><td>Function 4</td><td>RMSE</td><td>1.71677</td><td>1.1704</td><td>1.16405</td><td>0.97147</td><td>0.99251</td><td>0.94252</td><td>0.09715</td></tr><tr><td>Function 5</td><td>RMSE</td><td>0.65021</td><td>0.92874</td><td>0.92931</td><td>0.91363</td><td>0.94905</td><td>0.92991</td><td>0.09109</td></tr><tr><td colspan="9">Type 2: Uniformly distributed noise within the range [–0.2, , 0.2].</td></tr><tr><td>Function 1</td><td>RMSE</td><td>1.06672</td><td>0.81115</td><td>0.81094</td><td>0.92037</td><td>0.98172</td><td>0.00112</td><td>0.00092</td></tr><tr><td>Function 2</td><td>RMSE</td><td>0.4897</td><td>0.63984</td><td>0.6394</td><td>0.936</td><td>0.96028</td><td>0.94239</td><td>0.03178</td></tr><tr><td>Function 3</td><td>RMSE</td><td>1.89222</td><td>0.97275</td><td>0.32858</td><td>0.95603</td><td>0.8899</td><td>0.00129</td><td>0.0054</td></tr><tr><td>Function 4</td><td>RMSE</td><td>1.41344</td><td>0.64551</td><td>0.646</td><td>0.86553</td><td>0.89218</td><td>0.94258</td><td>0.06429</td></tr><tr><td>Function 5</td><td>RMSE</td><td>0.46209</td><td>1.22392</td><td>1.22275</td><td>0.98602</td><td>0.96304</td><td>0.93616</td><td>0.04854</td></tr><tr><td colspan="9">Type 3: Noise modeled by a Student&#x27;s t-distribution with 10 degrees of freedom.</td></tr><tr><td>Function 1</td><td>RMSE</td><td>1.06288</td><td>0.98732</td><td>0.9885</td><td>0.97617</td><td>0.94138</td><td>0.00085</td><td>0.00066</td></tr><tr><td>Function 2</td><td>RMSE</td><td>0.6821</td><td>1.17977</td><td>1.17943</td><td>0.97227</td><td>0.91816</td><td>0.93914</td><td>0.06809</td></tr><tr><td>Function 3</td><td>RMSE</td><td>1.26534</td><td>1.26867</td><td>0.28526</td><td>0.90042</td><td>0.95026</td><td>0.00118</td><td>0.00359</td></tr><tr><td>Function 4</td><td>RMSE</td><td>1.99009</td><td>0.90583</td><td>0.90957</td><td>0.8756</td><td>0.94032</td><td>0.94255</td><td>0.07723</td></tr><tr><td>Function 5</td><td>RMSE</td><td>0.45302</td><td>0.79612</td><td>0.798</td><td>0.95525</td><td>0.86489</td><td>0.93937</td><td>0.05649</td></tr></table>

The proposed models is denoted by <sup>†</sup>.

To comprehensively evaluate the robustness of the proposed GBTSVR and GB-TSVQR models under outliers, we injected anomalous samples into the synthetic datasets. Specifically, 10% of samples were randomly selected, and one randomly chosen feature of each was perturbed by multiplying it with an outlier factor of 10. This controlled corruption enables systematic assessment of model stability under severe data contamination. The RMSE results on these outlier-contaminated datasets are reported in Table 4. Both GBTSVR and GB-TSVQR consistently show strong performance across all noise types and generating functions. Notably, GB-TSVQR achieves the lowest RMSE in most cases, while GBTSVR remains highly competitive, substantially outperforming baseline models. This consistent performance across diferent noise distributions demonstrates the enhanced robustness, stability, and generalization capability of the proposed granular-ball-based approaches in handling extreme deviations and outlier-prone data.

## 6. Conclusion

This paper introduced GBTSVR and GB-TSVQR, two robust regression frameworks that integrate granular ball computing with TSVR to address key challenges encountered in real-world regression tasks, including noise, outliers, data heterogeneity, and asymmetric error behavior. By replacing point-wise learning with granular ball–level representations, the proposed models efectively exploit the latent structural organization of data, leading to enhanced robustness, stability, and computational eficiency. Extensive empirical evaluations are conducted to validate the efectiveness of the proposed approaches. The experiments spanned 18 benchmark UCI datasets, multiple synthetic datasets under diverse noise settings, synthetic datasets with explicitly injected outliers, and real-world time-series datasets. To ensure a fair and rigorous assessment, the results were further subjected to comprehensive statistical significance tests. Across these diverse experimental settings, GBTSVR and GB-TSVQR consistently demonstrated superior or highly competitive performance compared to state-ofthe-art regression models in terms of RMSE, robustness to contamination, and training eficiency. The experimental findings confirm that learning over granular balls significantly reduces sensitivity to noisy and extreme samples while preserving essential data characteristics. Moreover, the twin nonparallel regression formulation with eficient dual coordinate descent enables scalable learning without compromising generalization. The results demonstrate the practical value of granular ball–based regression, establishing GBTSVR and GB-TSVQR as efective solutions for complex scenarios. Future work will extend the framework to multivariate and high-dimensional regression, online and streaming settings, and deep hybrid architectures integrating granular representations with neural networks to improve adaptability and scalability.

## CRediT authorship contribution statement

A. Quadir: Conceptualization, Methodology, Software, Data curation, Writing - original draft. M. Tanveer: Conceptualization, Resources, Supervision, Project administration, Funding acquisition, Writing - review & editing.

## References

[1] S. Goldenberg, G. Nir, and S. Salcudean. A new era: Artificial intelligence and machine learning in prostate cancer. Nature Reviews Urology, 16(7):391–403, 2019.

[2] T. Hastie, R. Tibshirani, and J. Friedman. Overview of supervised learning. In The elements of statistical learning: Data mining, inference, and prediction, pages 9–41. Springer, 2008.

[3] G. Liu, Z. Lai, H. Kong, and J. Xu. Projection clustering support vector machine for semi-supervised learning. Pattern Recognition, 182:114739, 2026.

[4] X. Zhang, T. Wang, X. Wang, F. Fan, Y. Cheung, and I. Bose. Causality-informed neural networks for regularized learning in regression problems. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 56(3):1895–1910, 2026. doi: 10.1109/TSMC.2025.3646993.

[5] D. Basak, S. Pal, and D. Patranabis. Support vector regression. Neural Information Processing-Letters and Reviews, 11(10):203–224, 2007.

[6] C. Cortes and V. Vapnik. Support-vector networks. Machine Learning, 20:273– 297, 1995.

[7] A. Quadir and M. Tanveer. Granular ball twin support vector machine with pinball loss function. IEEE Transactions on Computational Social Systems, 12 (5):3891–3900, 2025. doi: 10.1109/TCSS.2024.3411395.

[8] B. Richhariya and M. Tanveer. EEG signal classification using universum support vector machine. Expert Systems with Applications, 106:169–182, 2018.

[9] A. Quadir, M. Ganaie, and M. Tanveer. Intuitionistic fuzzy generalized eigenvalue proximal support vector machine. Neurocomputing, 608:128258, 2024.

[10] Y. Wang, J. Wu, Z. Hu, and G. McLachlan. A new algorithm for support vector regression with automatic selection of hyperparameters. Pattern Recognition, 133:108989, 2023.

[11] M. Tanveer, M. Ganaie, I. Beheshti, T. Goel, N. Ahmad, K. Lai, K. Huang, Y. Zhang, J. Ser, and C. Lin. Deep learning for brain age estimation: A systematic review. Information Fusion, 96:130–143, 2023.

[12] A. Quadir and M. Tanveer. TRKM: Twin restricted kernel machines for classification and regression. Neural Networks, 197:108449, 2026. ISSN 0893-6080. doi: https://doi.org/10.1016/j.neunet.2025.108449.

[13] E. Elattar, J. Goulermas, and Q. Wu. Electric load forecasting based on locally weighted support vector regression. IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), 40:438–447, 2010. doi: 10. 1109/TSMCC.2010.2040176.

[14] C. Onyekwena, Q. Xue, Q. Li, Y. Wan, S. Feng, H. Umeobi, H. Liu, and B. Chen. Support vector machine regression to predict gas difusion coeficient of biocharamended soil. Applied Soft Computing, 127:109345, 2022.

[15] A. Zahariev, P. Angelov, and S. Zarkova. Estimation of bank profitability using vector error correction model and support vector regression. Economic Alternatives, 28(2):157–170, 2022.

[16] H. Zhang, Q. Wu, and F. Li. Application of online multitask learning based on least squares support vector regression in the financial market. Applied Soft Computing, 121:108754, 2022.

[17] X. Peng. TSVR: an eficient twin support vector machine for regression. Neural Networks, 23(3):365–372, 2010.

[18] Y. Shao, C. Zhang, Z. Yang, L. Jing, and N. Deng. An ε-twin support vector machine for regression. Neural Computing and Applications, 23(1):175–185, 2013.

[19] Y. Xu, X. Li, X. Pan, and Z. Yang. Asymmetric ν-twin support vector regression. Neural Computing and Applications, 30(12):3799–3814, 2018.

[20] D. Gupta and U. Gupta. On robust asymmetric lagrangian ν-twin support vector [22] M. Tanveer, T. Rajani, R. Rastogi, Y. Shao, and M. Ganaie. Comprehensive review on twin support vector machines. Annals of Operations Research, 339 (3):1223–1268, 2024.

regression using pinball loss function. Applied Soft Computing, 102:107099, 2021.

[21] Y. Ye, Z. Xu, J. Zhang, W. Chen, and Y. Shao. Twin support vector quantile regression. Expert Systems with Applications, 237:121239, 2024.

[23] X. Liu, S. Li, R. An, Z. Wang, Y. Xian, and X. Bai. Topological feature learning for cross-view multi-target track-to-track association in UAV-Based observation. Pattern Recognition, 181:114650, 2026.

[24] S. Xia, Y. Liu, X. Ding, G. Wang, H. Yu, and Y. Luo. Granular ball computing classifiers for eficient, scalable and robust learning. Information Sciences, 483: 136–152, 2019.

[25] S. Xia, D. Peng, D. Meng, C. Zhang, G. Wang, E. Giem, W. Wei, and Z. Chen. Ball kk-means: Fast adaptive clustering with no bounds. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(1):87–99, 2022. doi: 10.1109/ TPAMI.2020.3008694.

[26] S. Xia, S. Zheng, G. Wang, X. Gao, and B. Wang. Granular ball sampling for noisy label classification or imbalanced classification. IEEE Transactions on Neural Networks and Learning Systems, 34(4):2144–2155, 2021.

[27] S. Xia, X. Dai, G. Wang, X. Gao, and E. Giem. An eficient and adaptive granular-ball generation method in classification problem. IEEE Transactions on Neural Networks and Learning Systems, 35(4):5319–5331, 2022.

[28] A. Quadir, M. Sajid, and M. Tanveer. Granular ball twin support vector machine.

IEEE Transactions on Neural Networks and Learning Systems, 36(7):12444– 12453, 2025. doi: 10.1109/TNNLS.2024.3476391.

[29] M. Tanveer, R. Sharma, A. Quadir, and M. Sajid. Enhancing robustness and efficiency of least square twin SVM via granular computing. Pattern Recognition, page 112021, 2025.

[30] A. Quadir, M. Sajid, M. Tanveer, and P. Suganthan. Enhanced feature based granular ball twin support vector machine. arXiv preprint arXiv:2410.05786, 2024.

[31] C. Hsieh, K. Chang, C. Lin, S. Keerthi, and S. Sundararajan. A dual coordinate descent method for large-scale linear SVM. In Proceedings of the 25th international conference on Machine learning, pages 408–415, 2008.

[32] B. Gu, X. Geng, W. Shi, Y. Shan, Y. Huang, Z. Wang, and G. Zheng. Solving large-scale support vector ordinal regression with asynchronous parallel coordinate descent algorithms. Pattern Recognition, 109:107592, 2021.

[33] Y. Ye, C. Wang, J. Tian, and W. Chen. L1-norm twin support vector quantile regression. Applied Soft Computing, 169:112580, 2025.

[34] C. Liu and Q. Qian. Twin proximal support vector regression with gauss–laplace mixed noise. Pattern Recognition, 169:111860, 2026.

[35] D. Dua and C. Graf. UCI machine learning repository. Available: http://archive.ics.uci.edu/ml, 2017.

[36] J. Demšar. Statistical comparisons of classifiers over multiple data sets. The Journal ofMachine Learning Research, 7:1–30, 2006.

[37] M. Friedman. The use of ranks to avoid the assumption of normality implicit in the analysis of variance. Journal of the American Statistical Association, 32 (200):675–701, 1937.

[38] L. Chen. Topological structure in visual perception. Science, 218(4573):699– 700, 1982.

[39] G. Wang. DGCC: Data-driven granular cognitive computing. Granular Computing, 2:343–355, 2017.

[40] D. Cheng, Y. Li, S. Xia, G. Wang, J. Huang, and S. Zhang. A fast granular-ballbased density peaks clustering algorithm for large-scale data. IEEE Transactions on Neural Networks and Learning Systems, 35:17202–17215, 2024.

[41] Q. Zhang, C. Wu, S. Xia, F. Zhao, M. Gao, Y. Cheng, and G. Wang. Incremental learning based on granular ball rough sets for classification in dynamic mixedtype decision system. IEEE Transactions on Knowledge and Data Engineering, 35(9):9319–9332, 2023.

[42] J. Yang, Z. Liu, S. Xia, G. Wang, Q. Zhang, S. Li, and T. Xu. 3WC-GBNRS++: A novel three-way classifier with granular-ball neighborhood rough sets based on uncertainty. IEEE Transactions on Fuzzy Systems, 32(8):4376–4387, 2024.

## Supplementary Material

## S.1. Related Work

In this section, we examine prior work related to granular-ball computing and the mathematical formulation of TSVR.

## S.1.1. Granular-ball Computing

Rooted in classical theories of granular computing, recent studies have increasingly drawn inspiration from human cognitive perception, particularly the global-tolocal or coarse-to-fine information processing mechanism observed in human cognition [38]. Guided by this insight, Wang [39] introduced the paradigm of multigranularity cognitive computing, which advocates analyzing data by first capturing high-level structural patterns and subsequently refining them into finer representations. This hierarchical strategy provides an intuitive and eficient framework for understanding complex data distributions, closely mirroring human cognitive behavior.

Building on this theoretical foundation, Xia et al. [24] developed granular ball computing (GBC) as a practical and interpretable computational model. In contrast to traditional learning methods that operate directly on individual samples, GBC groups data into a collection of compact granules, known as granular balls. Each granular ball characterizes a localized data region through essential statistical and geometric descriptors, including its center, radius, density, and purity. By replacing point-wise representations with these region-level abstractions, GBC substantially reduces computational overhead while enhancing robustness to noise, outliers, and redundant information.

Owing to these advantages, GBC has been successfully applied across a broad spectrum of learning tasks. In clustering, granular-ball-based approaches enable eficient identification of intrinsic cluster structures while maintaining local density characteristics [40]. For classification tasks, granular ball classifiers exploit hierarchi cal data representations to achieve stronger generalization and improved resilience to noisy or mislabeled samples [28, 7]. Granular ball sampling methods further capitalize on these representations to select informative and representative subsets, making them particularly efective for large-scale learning problems [26]. Beyond these areas, GBC has also been integrated into rough set theory to enhance decision boundary approximation [41], as well as into three-way decision frameworks, where the uncertainty-aware nature of granulated data leads to more reliable decision-making [42].

## S.1.2. Twin Support Vector Regression

The twin support vector regression (TSVR) model formulates the regression task by solving two smaller and independent quadratic programming problems. The first optimization problem learns the parameters of the lower bounding regressor and is given by

$$
\begin{array} { l } { \displaystyle \mathop { \operatorname* { m i n } } _ { w _ { 1 } , b _ { 1 } } \frac { 1 } { 2 } \| Y - e \epsilon _ { 1 } - ( A w _ { 1 } + e b _ { 1 } ) \| ^ { 2 } + \mathcal { D } _ { 1 } e ^ { T } \eta _ { 1 } } \\ { \mathrm { s . t . } Y - e \epsilon _ { 1 } - ( A w _ { 1 } + e b _ { 1 } ) \geq - \eta _ { 1 } , } \\ { \eta _ { 1 } \geq 0 , } \end{array}\tag{36}
$$

whereas the second optimization problem determines the parameters of the upper bounding regressor and is expressed as

$$
\begin{array} { l } { \displaystyle \mathop { \operatorname* { m i n } } _ { w _ { 2 } , b _ { 2 } } \frac { 1 } { 2 } \| Y + e \epsilon _ { 2 } - ( A w _ { 2 } + e b _ { 2 } ) \| ^ { 2 } + \mathcal { D } _ { 2 } e ^ { T } \eta _ { 2 } } \\ { \mathrm { s . t . ~ } ( A w _ { 1 } + e b _ { 1 } ) - Y - e \epsilon _ { 2 } \geq - \eta _ { 2 } , } \\ { \displaystyle \eta _ { 2 } \geq 0 , } \end{array}\tag{37}
$$

where $\mathcal { D } _ { 1 } , ~ \mathcal { D } _ { 2 }$ are regularization parameters, $\epsilon _ { 1 } , \epsilon _ { 2 } > 0$ define the insensitive zones, $\eta _ { 1 }$ and $\eta _ { 2 }$ denote the slack variable vectors.

## S.2. Experimental setup

The experiments were conducted on a workstation powered by an Intel(R) Xeon(R) Gold 6226R processor operating at 2.90 GHz, paired with 256 GB of RAM. The system runs Windows 11, and all implementations were carried out in Python 3.11. Each dataset was randomly split into training and testing sets using a 70/30 proportion. To determine the optimal hyperparameters, we employed five-fold cross-validation together with grid search across the range $\mathcal { D } _ { 1 } = \mathcal { D } _ { 2 } = \{ 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , \dotsc , 1 0 ^ { 5 } \}$ . The parameters $\epsilon _ { 1 }$ and $\epsilon _ { 2 }$ are tuned by selecting values from the set {0<sub>.</sub>001<sub>,</sub> 0<sub>.</sub>005<sub>,</sub> 0<sub>.</sub>01<sub>,</sub> 0<sub>.</sub>05<sub>,</sub> 0<sub>.</sub>1<sub>,</sub> 0<sub>.</sub>15<sub>,</sub> 0<sub>.</sub>2<sub>,</sub> 0<sub>.</sub>25}. To assess the performance of the proposed GBTSVR and GB-TSVQR models, we employ four metrics, including mean absolute error (MAE), root mean squared error (RMSE), negative error (Neg Error), and positive error (Pos Error). The following are the specific definitions of these metrics:

$$
{ \mathrm { R M S E } } = { \sqrt { { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } ( f ( x _ { i } ) - y _ { i } ) ^ { 2 } } } ,\tag{38}
$$

$$
\mathrm { { M A E } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | ( f ( x _ { i } ) - y _ { i } ) | ,\tag{39}
$$

$$
\operatorname { P o s } { \mathrm { E r r o r } } = { \frac { 1 } { n } } \sum _ { i = 1 , f ( x _ { i } ) \leq y _ { i } } ^ { n } | ( f ( x _ { i } ) - y _ { i } ) | ,\tag{40}
$$

$$
\mathrm { N e g ~ E r r o r } = \frac { 1 } { n } \sum _ { i = 1 , f ( x _ { i } ) > y _ { i } } ^ { n } | ( f ( x _ { i } ) - y _ { i } ) | .\tag{41}
$$

## S.3. Experiments on Time-series Datasets

To demonstrate the efectiveness of the proposed GBTSVR and GB-TSVQR models in real-world regression scenarios, we conduct extensive experiments on benchmark time-series datasets. The comparative results of the proposed models against several state-of-the-art baseline methods are reported in Table S.1, using RMSE, MAE,

Table S.1: Performance comparison of the proposed GBTSVR and GB-TSVQR models and baseline models on time-series datasets in terms of RMSE, MAE, Poss Error, and Neg Error.
<table><tr><td>Dataset</td><td>Metric</td><td>SVR [5]</td><td>TSVR [17]</td><td>TSVQR [21]</td><td>L1-TSVQR [33]</td><td>GL-TPSVR [34]</td><td>GBTSVR</td><td>GB-TSVQR</td></tr><tr><td rowspan="4">NNGC1_dataset_D1_V1_003 (430,5)</td><td>RMSE</td><td>0.9938</td><td>0.01041</td><td>0.81541</td><td>0.00875</td><td>0.98186</td><td>0.00064</td><td>0.00073</td></tr><tr><td>MAE</td><td>0.84253</td><td>0.00826</td><td>0.68345</td><td>0.00786</td><td>0.83049</td><td>0.00053</td><td>0.00072</td></tr><tr><td>Pos Error</td><td>0.76537</td><td>0.00662</td><td>0.65806</td><td>0.00736</td><td>0.86224</td><td>0.00059</td><td></td></tr><tr><td>Neg Error</td><td>0.90556</td><td>0.01014</td><td>0.70846</td><td>0.00742</td><td>0.802</td><td>0.00048</td><td>0.00072</td></tr><tr><td rowspan="4">NNGC1_dataset_D1_V1_004 (545,5)</td><td>RMSE</td><td>0.65551</td><td>0.00277</td><td>0.46681</td><td>0.0084</td><td>0.65176</td><td>0.00033</td><td>0.00085</td></tr><tr><td>MAE</td><td>0.54944</td><td>0.00241</td><td>0.35603</td><td>0.0075</td><td>0.5423</td><td>0.00026</td><td>0.00077</td></tr><tr><td>Pos Error</td><td>0.38007</td><td>0.00191</td><td>0.38925</td><td>0.0073</td><td>0.37873</td><td>0.00031</td><td>0.00022</td></tr><tr><td>Neg Error</td><td>0.60589</td><td>0.00264</td><td>0.29995</td><td>0.0065</td><td>0.60606</td><td>0.0002</td><td>0.00008</td></tr><tr><td rowspan="4">NNGC1_dataset_D1_V1_005 (430,5)</td><td>RMSE</td><td>0.04651</td><td>0.00061</td><td>0.2303</td><td>0.00354</td><td>0.41303</td><td>0.00892</td><td>0.00011</td></tr><tr><td>MAE</td><td>0.03901</td><td>0.00051</td><td>0.2275</td><td>0.00141</td><td>0.41152</td><td>0.00726</td><td>0.00009</td></tr><tr><td>Pos Error</td><td>0.04316</td><td>0.00053</td><td></td><td>0.00041</td><td>0.41152</td><td>0.00805</td><td></td></tr><tr><td>Neg Error</td><td>0.03287</td><td>0.0005</td><td>0.2275</td><td>0.00082</td><td></td><td>0.00466</td><td>0.00001</td></tr><tr><td rowspan="4">NNGC1_dataset_D1_V1_006 (610,5)</td><td>RMSE</td><td>0.25026</td><td>0.00862</td><td>0.33471</td><td>0.00203</td><td>0.47458</td><td>0.00194</td><td>0.00047</td></tr><tr><td>MAE</td><td>0.20081</td><td>0.00515</td><td>0.28032</td><td>0.00147</td><td>0.41319</td><td>0.00157</td><td>0.00045</td></tr><tr><td>Pos Error</td><td>0.19707</td><td>0.0035</td><td>0.15724</td><td>0.00203</td><td>0.42924</td><td>0.00149</td><td>0.00045</td></tr><tr><td>Neg Error</td><td>0.20309</td><td>0.00783</td><td>0.3064</td><td>0.00776</td><td>0.11591</td><td>0.00171</td><td>0.0001</td></tr><tr><td rowspan="4">NNGC1_dataset_D1_V1_007 (610,5)</td><td>RMSE</td><td>0.10424</td><td>0.00311</td><td>0.19239</td><td>0.00885</td><td>0.4195</td><td>0.00405</td><td>0.00054</td></tr><tr><td>MAE</td><td>0.08292</td><td>0.00196</td><td>0.16827</td><td>0.00418</td><td>0.40649</td><td>0.00325</td><td>0.00052</td></tr><tr><td>Pos Error</td><td>0.09092</td><td>0.00122</td><td>0.07523</td><td>0.00102</td><td>0.40649</td><td>0.00348</td><td>0.00052</td></tr><tr><td>Neg Error</td><td>0.07657</td><td>0.00316</td><td>0.17842</td><td>0.00238</td><td></td><td>0.00287</td><td></td></tr><tr><td rowspan="4">NNGC1_dataset_D1_V1_008 (540,5)</td><td>RMSE</td><td>0.10357</td><td>0.00411</td><td>0.2791</td><td>0.00106</td><td>0.42315</td><td>0.00368</td><td>0.00011</td></tr><tr><td>MAE</td><td>0.07853</td><td>0.00305</td><td>0.25947</td><td>0.00087</td><td>0.41017</td><td>0.00293</td><td>0.00008</td></tr><tr><td>Pos Error</td><td>0.07673</td><td>0.00288</td><td>0.03178</td><td>0.00109</td><td>0.41272</td><td>0.00303</td><td>0.00007</td></tr><tr><td>Neg Error</td><td>0.08046</td><td>0.00361</td><td>0.26088</td><td>0.00604</td><td>0.00067</td><td>0.0027</td><td>0.00095</td></tr><tr><td rowspan="4">NNGC1_dataset_D1_V1_009 (540,5)</td><td>RMSE</td><td>0.55713</td><td>0.03569</td><td>0.56568</td><td>0.00306</td><td>0.6612</td><td>0.00253</td><td>0.00038</td></tr><tr><td>MAE</td><td>0.43811</td><td>0.01896</td><td>0.4437</td><td>0.00101</td><td>0.53367</td><td>0.0019</td><td>0.00031</td></tr><tr><td>Pos Error</td><td>0.44102</td><td>0.01095</td><td>0.43426</td><td>0.00306</td><td>0.58708</td><td>0.00143</td><td>0.0001</td></tr><tr><td>Neg Error</td><td>0.43566</td><td>0.03257</td><td>0.45002</td><td>0.00829</td><td>0.36526</td><td>0.00228</td><td>0.00034</td></tr><tr><td rowspan="4">NNGC1_dataset_D1_V1_010 (585,5)</td><td>RMSE</td><td>0.12526</td><td>0.0058</td><td>0.36128</td><td>0.00919</td><td>0.4219</td><td>0.00441</td><td>0.00045</td></tr><tr><td>MAE</td><td>0.10006</td><td>0.00319</td><td>0.33974</td><td>0.00418</td><td>0.40322</td><td>0.00345</td><td>0.00042</td></tr><tr><td>Pos Error</td><td>0.09756</td><td>0.00173</td><td>0.01862</td><td>0.0012</td><td>0.40322</td><td>0.0033</td><td>0.00004</td></tr><tr><td>Neg Error</td><td>0.10261</td><td>0.0052</td><td>0.34158</td><td>0.00274</td><td></td><td>0.0037</td><td>0.0003</td></tr><tr><td rowspan="4">NNGC1_dataset_E1_V1_008 (740,5)</td><td>RMSE</td><td>0.24381</td><td>0.00426</td><td>0.39578</td><td>0.00239</td><td>0.46686</td><td>0.00133</td><td>0.0006</td></tr><tr><td>MAE</td><td>0.20354</td><td>0.00304</td><td>0.31672</td><td>0.00168</td><td>0.41277</td><td>0.00115</td><td>0.00058</td></tr><tr><td>Pos Error</td><td>0.16215</td><td>0.0031</td><td>0.1089</td><td>0.00247</td><td>0.4531</td><td>0.00126</td><td>0.00006</td></tr><tr><td>Neg Error</td><td>0.27285</td><td>0.00293</td><td>0.32652</td><td>0.00619</td><td>0.0638</td><td>0.00093</td><td></td></tr><tr><td rowspan="4">NNGC1_dataset_E1_V1_009 (740,5)</td><td>RMSE</td><td>0.23091</td><td>0.00517</td><td>0.40271</td><td>0.00227</td><td>0.46194</td><td>0.00147</td><td>0.00046</td></tr><tr><td>MAE</td><td>0.19294</td><td>0.00391</td><td>0.33089</td><td>0.00163</td><td>0.41035</td><td>0.00126</td><td>0.00044</td></tr><tr><td>Pos Error</td><td>0.151</td><td>0.00381</td><td>0.10054</td><td>0.00237</td><td>0.44534</td><td>0.00141</td><td>0.00001</td></tr><tr><td>Neg Error</td><td>0.26455</td><td>0.00421</td><td>0.3395</td><td>0.007</td><td>0.05693</td><td>0.00097</td><td>0.00045</td></tr><tr><td rowspan="4">NNGC1_dataset_E1_V1_010 (650,5)</td><td>RMSE</td><td>0.44161</td><td>0.00815</td><td>0.48839</td><td>0.0033</td><td>0.60468</td><td>0.0012</td><td>0.00011</td></tr><tr><td>MAE</td><td>0.35476</td><td>0.00544</td><td>0.38271</td><td>0.00131</td><td>0.49549</td><td>0.00093</td><td>0.00009</td></tr><tr><td>Pos Error</td><td>0.35456</td><td>0.0055</td><td>0.39629</td><td>0.00031</td><td>0.5912</td><td>0.00093</td><td>0.00023</td></tr><tr><td>Neg Error</td><td>0.35498</td><td>0.00537</td><td>0.37188</td><td>0.00877</td><td>0.20837</td><td>0.00092</td><td>0.00092</td></tr><tr><td colspan="2"</table>

The proposed models is denoted by <sup>†</sup>.

Pos Error, and Neg Error as evaluation metrics. From the results, it is evident that the proposed granular-ball-based models consistently deliver superior predictive performance across the considered time-series datasets. In particular, GB-TSVQR achieves the best RMSE performance on the majority of datasets, closely followed by GBTSVR, significantly outperforming classical SVR variants and other twin-based regression models. The average RMSE values further validate this observation. GB-TSVQR attains the lowest average RMSE among all competing models, while GBTSVR also achieves a notably lower average RMSE compared to the baseline models. This indicates that incorporating granular-ball computing efectively enhances the robustness and generalization capability of TSVR frameworks, particularly for complex and noisy time-series data. The consistent dominance of GBTSVR and GB-TSVQR across multiple metrics highlights their strong modeling capacity and robustness. These results confirm that the proposed GBTSVR and GB-TSVQR models are well suited for time-series regression tasks and provide a powerful alternative to existing SVRbased and twin regression models.