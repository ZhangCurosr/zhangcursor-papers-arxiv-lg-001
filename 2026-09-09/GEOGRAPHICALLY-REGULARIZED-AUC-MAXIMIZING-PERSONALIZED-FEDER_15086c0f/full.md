# GEOGRAPHICALLY REGULARIZED AUC-MAXIMIZING PERSONALIZED FEDERATED LEARNING

## A PREPRINT

Mayu Hiraishi   
Department of Medicine   
Wakayama Medical University   
Wakayama, Japan   
m-hira@wakayama-med.ac.jp

Kensuke Tanioka Department of Biomedical Sciences and Informatics Doshisha University Kyoto, Japan ktanioka@mail.doshisha.ac.jp

Toshio Shimokawa Department of Medicine Wakayama Medical University Wakayama, Japan shimokaw@wakayama-med.ac.jp

September 9, 2026

## ABSTRACT

Accurate diagnostic and risk-prediction models are important for supporting clinical decision-making during infectious disease outbreaks. However, privacy and governance requirements may restrict patient-level data sharing across healthcare institutions, and data distributions often vary. Moreover, AUC is widely used to evaluate discriminative performance, motivating its direct optimization in model development. We propose geographically regularized AUC-maximizing personalized federated learning (GrAUC-PFL), which directly optimizes a smooth pairwise AUC surrogate to learn personalized models while keeping patient-level data local and accounting for institutional heterogeneity. Graph-based regularization encourages geographically neighboring institutions to have similar coefficient vectors while retaining a personalized models. Simulations and a realdata application suggest improved discriminative performance, particularly when geographically neighboring institutions have similar data-generating characteristics.

Keywords AUC maximization · geographic proximity · graph regularization · medical diagnosis · personalized federated learning

## 1 Introduction

During large-scale infectious disease epidemics, such as the COVID-19 pandemic, accurate diagnostic and risk prediction models are essential for identifying patients at high risk of severe disease, supporting timely clinical decision making, and allocating limited healthcare resources efficiently. Developing reliable diagnostic models often benefits data collected across multiple healthcare institutions. However, in recent years, privacy concerns and regulatory requirements have often restricted patient-level data sharing, resulting in difficulties in aggregating data across multiple institutions [Zhang and Singh, 2026]. For example, the General Data Protection Regulation in the European Union [European Parliament and Council of the European Union, 2016] imposes strict requirements on patient-level data sharing across institutions, making centralized data aggregation challenging in practice.

In such situations, Federated learning (FL) [McMahan et al., 2017] enables collaborative model training without directly sharing patient-level data, which have been applied to various medical applications [e.g. Dayan et al., 2021, Agbley et al., 2021]. However, clinical data often exhibit heterogeneity across institutions or regions owing to differences in patient populations, clinical practice, and evaluation criteria, which may not be adequately captured by a single global model used in conventional FL. Personalized federated learning (PFL) [Fallah et al., 2020, Jiang et al., 2019] addresses this limitation by training personalized models, whilst utilizing information from other institutions.

In addition, healthcare institutions located in close geographical proximity are observed to share similar patient demographics and diagnostic patterns. In fact, hospitals serve geographically defined catchment areas, which may affect COVID-19 hospitalization forecasts [Meakin and Funk, 2024], while demographic structure and population movement can lead to similar epidemiological dynamics in neiboring regions [Rader et al., 2020]. These observations suggest that geographic proximity can provide relevant information for one another in infectious disease settings. Therefore, beyond relationships inferred from the model parameters or observed data, geographic proximity can provide an additional information on similarities among institutions. This concept has been exploited in spatial statistical methods [e.g. Brunsdon et al., 1996, Gelfand et al., 2003]. Therefore, in the PFL, incorporating geographical proximity is expected to improve the estimation accuracy of local models by utilising information from other participating organisations.

Furthermore, class imbalances are likely to arise in outcomes. Consequently, in such situations, direct AUC maximiza tion [Yan et al., 2003, Yang and Ying, 2022] can provide improved discriminative performance by explicitly optimizing scores based on the ranking of positive and negative cases [Natole et al., 2018]. Therefore, we employ AUC direct maximization.

Motivated by these challenges, we propose geographically regularized AUC-maximizing personalized federated learning (GrAUC-PFL), which directly maximizing a smooth pairwise AUC surrogate within a PFL framework. GrAUC-PFL uses graph-based regularization that encourages geographically neighboring institutions to have similar model parameters. This allows each institution to retain personalized diagnostic models, while selectively borrowing information from its neighbors. GrAUC-PFL integrates direct AUC maximization with externally defined geographical structure within a unified PFL framework. Unlike similarity-based PFL approaches such as Huang et al. [2021], Chen et al. [2022], which infer client relationships from local model updates or parameter similarities, GrAUC-PFL uses geographical proximity as a prior structural information sharing independently to be guided by external knowledge rather than potentially noisy local model estimates. This may be particularly beneficial when local data are limited.

Because the proposed optimization problem combines a non-decomposable AUC objective function with graph-based regularization, we develop an efficient optimization algorithm based on the Alternating Direction Method of Multipliers (ADMM) [Boyd et al., 2011], extending the optimization framework of Liu et al. [2025]. Thus, the proposed framework provides a computationally efficient approach for estimating personalized diagnostic models from geographically distributed medical data.

The remainder of this article is organised as follows. Section 2 explain the related works of GrAUC-PFL. Section 3 introduces GrAUC-PFL, the objective function, and the algorithm. A numerical simulation is reported in Section 4 and a real data application is presented in Section 5. We discuss the results of numerical simulation and real-data application in Section 6 and conclude the article with Section 7.

## 2 Related works

## 2.1 Personalized federated learning (PFL)

In this section, we briefly review conventional federated learning (FL) and then introduce PFL. FL is a distributed learning framework in which multiple institutions collaboratively train a single global model without sharing raw data In standard FL, each institution sends model updates (e.g., parameters or gradients) to the server, enabling the global model to be updated without the need to provide data to external institutions. In the case of client-server structure, the server distributes the global model to each institution, and each institution updates the model using local data. The locally updated model parameters are transmitted to the central server, where they are aggregated (e.g. averaged), and then shared with each institution once again as the global model. A shared global model is estimated over multiple communication rounds. The image on the left in Figure 1 describes an example of the structure of FL.

However, it may not be possible for the single global model to account adequately for the heterogeneity of institutions. For example, if the true models differ across institutions, the resulting global model may fail to achieve high AUC at individual institutions. Furthermore, patient demographics, sample sizes and prevalence rates often vary between institutions. When information is integrated, institutions with larger sample sizes may dominate the optimization, resulting in poorer performance at smaller institutions. To overcome this issue, the PFL framework has been proposed. In PFL, updating the local model using local data is the same procedure as in standard FL. When the local model or parameter are aggregated at the central server, each institution retains its own personalized model while leveraging information from other institutions, which is described at the right side of Figure 1. In PFL, when estimating personalized models, information from other institutions without sharing raw data stored at each institution. Various approaches have been proposed to leverage information from other institutions.

Fallah et al. [2020] extends FedAvg [McMahan et al., 2017] by incorporating gradient correction for each institution, thus taking personalization into account. Another approach adds the regularization term penalizing the deviation between the personalized and global models [Li et al., 2020]. In terms of group structure, clustered federated learning Sattler et al. [2019] uses the gradient direction for clustering in the framework of multi-task learning, which shares information within the same clusters. In FedAMP [Huang et al., 2021], adaptively exchanges information among clients according to the similarity of their personalized models. Chen et al. [2022] shares information among neighboring clients based on graph-structured relationships, while updating the global model and local models. These PFL methods rely on a shared global model or infer client relationships from model updates or parameter similarities. In contrast, instead of estimating relationships between clients based on model updates, we incorporate geographical proximity as prior information through graph-based regularization, thereby promoting the sharing of information between geographically close institutions whilst maintaining personalized models.

![](images/2c644101ee599ea907683e81093c4cf1a22491c7ffb3d7e2a9a2d0d5dfe5fa68.jpg)  
Figure 1: Images of federated learning and personalized federated learning.

## 2.2 AUC maximization

## Definition of AUC

The ROC curve depicts the relationship between the true positive rate (TPR) and false positive rate (FPR) as the threshold for the scoring function varies. For the scoring function $s ( x )$ , the TPR and FPR when using threshold $\nu \in \mathbb { R }$ are defined as $T P R ( \nu ) \bar { = } P ( s ( X ) \geq \nu | Y = 1 )$ and $\begin{array} { r } { F \bar { P } R ( \nu ) = \dot { P r } ( s ( X ) \geq \nu | Y = 0 ) } \end{array}$ ), respectively. The ROC curve is then given as the set of $( F P R ( \nu ) , T P R ( \nu ) )$ obtained as ν varies.

Let the target variable $y \in \{ 0 , 1 \}$ be a binary variable representing the observed true label, where a positive instance is denoted by $y = 1$ and a negative instance by $y = 0$ . Let the covariates corresponding to the positive and negative examples be denoted by $\pmb { x } ^ { + } \in \mathbb { R } ^ { p }$ and $\pmb { x } ^ { - } \in \mathbb { R } ^ { p } .$ , respectively, and assume these are independent random variables following probability distributions $P ^ { + }$ and $P ^ { - }$ , respectively. In this case, the AUC is defined as the probability that a positive subject achieves a higher score than a negative subject. Let $f ( { \pmb x } )$ be the scoring function and $I ( \cdot )$ be the indicator function. Then AUC as the probability is defined as follows:

$$
\operatorname { A U C } ( f ) = P { \big ( } ( f ( x ^ { + } ) > f ( x ^ { - } ) { \big ) } = \mathbb { E } _ { \mathbf { x } ^ { + } \sim P ^ { + } , \mathbf { x } ^ { - } \sim P ^ { - } } \left[ I { \big ( } f ( \mathbf { x } ^ { + } ) > f ( \mathbf { x } ^ { - } ) { \big ) } \right] .\tag{1}
$$

In the case of actual data, where the covariates for positive cases are denoted by $\pmb { x } _ { i } ^ { + } ( i = 1 , 2 , \ldots , n )$ and those for negative cases by $\pmb { x } _ { j } ^ { - } \ ( j = 1 , 2 , . . . , m )$ , the AUC is expressed as the following:

$$
\mathrm { A U C } = \frac { 1 } { n m } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { m } I ( f ( \pmb { x } _ { i } ^ { + } ) > f ( \pmb { x } _ { j } ^ { - } ) )\tag{2}
$$

where $f ( \pmb { x } ) : \mathbb { R } ^ { p } $ R is the function to predict the parameters, and we assume a linear scoring function $f ( \pmb { x } ) = \beta ^ { \top } \pmb { x }$ where β signifies the parameter to calculate the score. $I ( \cdot )$ is the indicator function returning 1 if $f ( \pmb { x } _ { i } ^ { + } ) > f ( \pmb { x } _ { i } ^ { - } )$ Eq. (2) corresponds to the standardized Mann-Whitney U-statistic [Mann and Whitney, 1947], which represents the probability that a randomly selected positive sample will be assigned a higher score than a randomly selected negative sample.

Under the scoring model, AUC can be expressed as a pairwise comparison of score differences:

$$
\begin{array} { r l } { \displaystyle \mathrm { A U C } = \frac { 1 } { n m } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { m } I \big ( \beta ^ { \top } \pmb { x } _ { i } ^ { + } > \beta ^ { \top } \pmb { x } _ { j } ^ { - } \big ) } & { } \\ { = \frac { 1 } { n m } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { m } I \left( \beta ^ { \top } ( \pmb { x } _ { i } ^ { + } - \pmb { x } _ { j } ^ { - } ) > 0 \right) } & { } \\ { = \frac { 1 } { n m } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { m } I \left( f ( \pmb { x } _ { i } ^ { + } ) - f ( \pmb { x } _ { j } ^ { - } ) > 0 \right) } & { } \end{array}\tag{3}
$$

Eq. (3) shows that AUC maximization can be interpreted as a ranking problem based on pairwise differences.

## Surrogate function of AUC maximization

As shown in Eq. (3), AUC depends on the discontinuous indicator function, which is not differentiable and not suitable for gradient-based optimization. Therefore, it is common to approximate it using a surrogate function. Various surrogate loss functions have been proposed [Yuan et al., 2021, Tian et al., 2011], and in this study, we employ the logistic loss [Sulam et al., 2017]. This provides a smooth approximation while preserving the ranking structure and enable efficient optimization.

For a pair $( i , j )$ , the logistic-based surrogate can be described as:

$$
\ell \big ( \beta ; x _ { i } ^ { + } , x _ { j } ^ { - } \big ) = \log \big ( 1 + \exp [ { - ( \boldsymbol \beta ^ { \top } \boldsymbol x _ { i } ^ { + } - \boldsymbol \beta ^ { \top } \boldsymbol x _ { j } ^ { - } ) - q } ] \big )
$$

where $q \ ( q \geq 0 )$ is the margin constant value. Here, $\beta$ signifies the parameter to calculate the score. The surrogate loss corresponding to AUC is defined as follows:

$$
\frac { 1 } { n m } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { m } \log \big ( 1 + \exp [ - ( { \boldsymbol { \beta } } ^ { \top } \mathbf { x } _ { i } ^ { + } - { \boldsymbol { \beta } } ^ { \top } \mathbf { x } _ { j } ^ { - } ) - q ] \big ) .\tag{4}
$$

This surrogate function preserves the ranking structure of each pair while providing a differentiable approximation of the AUC objective function. This enables efficient optimization using gradient-based methods.

## 3 Geographically regularized AUC-maximizing personalized federated learning (GrAUC-PFL)

## 3.1 Framework of GrAUC-PFL

We first introduce the framework of GrAUC-PFL. Let $z = 1 , 2 , \dots , Z$ denote institutions, each of which is treated as a client with the FL framework and has its own local dataset consisting of outcomes and covariates for the disease, along with geographical information such as latitude and longitude. Because raw data cannot be shared externally, each institution aims to estimate its own model parameter $\beta _ { z }$ based solely on the relevant local data.

The objective of this study is to utilize information from multiple institutions and directly maximize AUC, as a measure of discriminative performance, through a unified PFL framework with geographical proximity. To account for this spatial structure, we assume that proximate institutions tend to share similar risk factors, and incorporate this similarity into the model through a regularization structure that encourages similar parameters among nearby institutions. Specifically, we adopt an $L _ { \mathrm { 1 } } \mathrm { - } \mathrm { t y p e }$ penalty term based on differences between the coefficient of selected pairs of institutions, as in spatially clustered coefficients (SCC) [Li and Sang, 2019]. The pairs of institutions are determined by the edges of a graph constructed based on their geographical locations. If the difference between a pairs of coefficients is non-zero, the edge represents a boundary; otherwise, the institutions are regarded as belonging to the same cluster. To construct the graph, a minimum spanning tree (MST) is employed to select the edge as in the SCC. In this method, $Z - 1$ edges connecting all institutions are constructed at minimum. This significantly reduces computational cost compared to a complete graph, which has $Z ( Z - 1 ) / 2$ edges. This formulation is also related to fused lasso regularization [Tibshirani et al., 2004], allowing existing algorithms to be readily applied. Moreover, when the number of subject in a institution is small, estimation of the coefficients may be unstable. By incorporating information from neighboring institutions, the estimation accuracy can be improve compared to independent estimation.

Next, we describe the communication scheme between the server and local participating institutions for parameter updates. We adopt the the PFL framework proposed by Liu et al. [2025]. In this framework, the server and each institution have distinct roles. Each institution maintains its own local model and updates its model parameters using only local data. Meanwhile, the server coordinates the communication among institutions and enforces the graph-based constraints that capture the relationships between institutions. During each iteration, only model-related information is exchanged between the server and the institutions, while patient-level data remain local. By iteratively exchanging this information, each institution can estimate a personalized model that reflects both its local data and the information shared across related institution. The proposed framework is illustrated in Figure 2.

![](images/e9820c6da27d4effeca8da7d161894fc674e5a04c0491c16643a8f301d559661.jpg)  
Figure 2: Image of the framework of Personalized Federated Learning considering the geographical proximity.

## 3.2 Optimization problem of GrAUC-PFL

In this section, we define the optimization problem of GrAUC-PFL. For institutions $z = 1 , \ldots , Z ,$ , each institution has its own local dataset $\{ ( \pmb { x } _ { h z } , y _ { h z } ) \} _ { h = 1 } ^ { N _ { z } }$ , where $\pmb { x } _ { h z } \in \mathbb { R } ^ { p }$ is a covariate vector and $y _ { h z } \in \{ 0 , 1 \}$ is a binary outcome. For the covariates at institution $z , \{ x _ { i z } ^ { + } \} _ { i = 1 } ^ { n _ { z } }$ and $\{ \pmb { x } _ { j z } ^ { - } \} _ { j = 1 } ^ { m _ { z } }$ represent the positive and negative samples, respectively. Here, $N _ { z } = n _ { z } + m _ { z }$ . Also, we consider the geographical proximity between institutions, defined through a distance measure between them, and incorporate this information into the model to encourage similarity between nearby coefficient vectors. In this study, we use the distance computed from latitude and longitude, and those in the institution z denote $\pmb { s } _ { z } = ( \mathrm { l o n } _ { z } , \mathrm { l a t } _ { z } ) \in \mathbf { \bar { \mathbb { R } } } ^ { 2 }$ , where $\mathrm { l o n } _ { z }$ denote longitude and $\mathrm { l a t } _ { z }$ latitude.

We consider the linear scoring function:

$$
f _ { z } ( \pmb { x } _ { z } ) = \beta ( \pmb { s } _ { z } ) ^ { \top } \pmb { x } _ { z } .
$$

To simplify the notation, we shall denote $\beta ( s _ { z } )$ by $\beta _ { z }$ . The AUC at institution z can be interpreted as the probability that a positive sample is ranked higher than a negative one.

$$
\begin{array} { r } { \mathrm { A U C } _ { z } = \frac { 1 } { n _ { z } m _ { z } } \displaystyle \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } I \big ( f _ { z } ( \pmb { x } _ { i z } ^ { + } ) > f _ { z } ( \pmb { x } _ { j z } ^ { - } ) \big ) } \\ { = \frac { 1 } { n _ { z } m _ { z } } \displaystyle \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } I \big ( \beta _ { z } ^ { \top } ( \pmb { x } _ { i z } ^ { + } - \pmb { x } _ { j z } ^ { - } ) > 0 \big ) } \end{array}
$$

However, because the indicator function is non-differentiable, it is difficult to optimize this objective directly. Therefore, we replace this with a pairwise logistic loss as a surrogate function at institution z.

$$
\ell _ { z } ( \beta _ { z } ) = \frac { 1 } { n _ { z } m _ { z } } \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } \log \left( 1 + \exp [ - ( \beta _ { z } ^ { \top } x _ { i z } ^ { + } - \beta _ { z } ^ { \top } x _ { j z } ^ { - } ) - q ] \right)
$$

where $q \geq 0$ is the margin parameter. Using this surrogate loss, the optimization problem of GrAUC-PFL is formulated as follows:

$$
\mathcal { L } ( \boldsymbol { \beta } ) = \frac { 1 } { Z } \sum _ { z = 1 } ^ { Z } \frac { 1 } { n _ { z } m _ { z } } \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } \log \left( 1 + \exp [ - ( \beta _ { z } ^ { \top } { \boldsymbol x } _ { i z } ^ { + } - \beta _ { z } ^ { \top } { \boldsymbol x } _ { j z } ^ { - } ) - \boldsymbol q ] \right)\tag{5}
$$

where $\lambda ( > 0 )$ is a tuning parameter and $\| \cdot \| _ { 1 }$ denotes $L _ { 1 }$ norm. The first term of Eq. (5) is to optimize the surrogate function of AUC maximization. The parameter $\beta _ { z }$ is updated on each local institution and consolidated on the server, and updated as $\beta .$ . The second term is a weighted fused penalty imposed on the differences between institution-specific coefficient vectors. $\{ u , v \} \in E$ represents an edge connecting institutions u and $v ,$ where E is the edge set constructed from MST. This penalty encourages geographically close institutions to have similar coefficient vectors, and may reduce the difference between certain pairs to exactly zero. Consequently, the institutions connected through the graph can share identical parameter vectors, yielding a clustering structure among institutions in terms of their regression coefficients. $w _ { u v }$ is a distance-based weight reflecting the geographical proximity between institutions u and v using a Gaussian kernel. The distance between the institutions u and v is defined as $m _ { u v } ^ { \dagger } = d ( s _ { u } , s _ { v } )$ , where $d ( \cdot , \cdot )$ is the distance metric. For this metric calculation, we use the Haversine formula. Based on this distance, we define the weight as

$$
w _ { u v } = \exp \left( - \frac { m _ { u v } ^ { \dagger 2 } } { 2 \phi ^ { 2 } } \right) .\tag{6}
$$

ϕ is a bandwidth parameter determined by $w _ { d i s t } \cdot \phi _ { b a s e }$ , where $\phi _ { b a s e } = \mathrm { m e d i a n } \{ m _ { u v } ^ { \dagger } : \{ u , v \} \in E \}$ . The weight $w _ { u v }$ decreases as the distance $m _ { u v } ^ { \dagger }$ increases, assigning larger weights to geographically closer institutions, consistent with the principle of spatial weighting that observations at closer distances receive greater influence[e.g. Brunsdon et al., 1996]. The bandwidth ϕ determines how quickly the weight decays with distance: smaller $\phi$ lead to more rapidly decreasing weights, while larger values result in more uniform weighting across institutions. Here, $\phi _ { b a s e }$ is set to the median of the pairwise distances over the edges in the MST. This is motivated by the median heuristic commonly used in the bandwidth of the Gaussian kernel and is widely adopted in kernel methods [e.g. Garreau et al., 2018, Gretton et al., 2012]. This provides a robust estimate of the typical distance scale, as MST captures relationship between institutions in local neighborhood. The MST-based median provides a scale that is consistent with the graph structure used in the regularization. On the other hand, since the appropriate scale depends on the data distribution, the scaling parameter $w _ { d i s t } ( \geq 0 )$ is selected via cross-validation.

## 3.3 Reformulation of GrAUC-PFL

In the framework of PFL, parameter $\beta _ { z }$ needs to be updated at each institution by using its own local data. To update these parameters by distributed learning efficiently, we adpot the ADMM. However, the second term couples institution specific parameters, preventing the update from being decomposed across institutions and making the problem difficult to solve. To address this issue, it needs to reformulate the pairwise differences between coefficients of connected institutions into an alternative representation. First, the pairwise differences between the coefficients are expressed as

$$
\left[ \begin{array} { c } { \beta _ { u _ { 1 } } - \beta _ { v _ { 1 } } } \\ { \beta _ { u _ { 2 } } - \beta _ { v _ { 2 } } } \\ { \vdots } \\ { \beta _ { u _ { | E | } } - \beta _ { v _ { | E | } } } \end{array} \right] \in \mathbb { R } ^ { | E | p } , \mathrm { ~ w h e r e ~ } ( u _ { o } , v _ { o } ) \in E \mathrm { ~ } ( o = 1 , 2 , \ldots , | E | ) .\tag{7}
$$

As the difference structure of the coefficients is defined based on MST, we begin by using the edges of the graph to represent the relationship between institutions. From the graph $( V ^ { \dagger } , E )$ , where $V ^ { \dagger }$ denotes a set of $Z$ locations, and $E$ denotes the set of edges. After computing pairwise distances based on the coordinate data, an adjacency matrix is constructed to represent geographical proximity, and a graph is defined accordingly. We use the MST to determine the edge set E. MST connects all institutions while minimizing the total pairwise distance and results in a sparse structure with $| E | = Z - 1 { \mathrm { e d g e s } }$ . Based on E, the incidence matrix $D = ( \bar { d _ { o z } } ) \in \mathbb { R } ^ { | E | \times Z }$ is defined as follows:

$$
\left\{ \begin{array} { l } { d _ { o z } = + 1 \left( z = u _ { o } \right) } \\ { d _ { o z } = - 1 \left( z = v _ { o } \right) } \\ { d _ { o z } = 0 \mathrm { ~ ( o t h e r w i s e ) } } \end{array} \right. \quad , \quad \left( o = 1 , 2 , \ldots , | E | \right)\tag{8}
$$

Next, since $\beta _ { z }$ is a institution-specific vector, all coefficient vectors are concatenate into a single vector:

$$
\beta = v e c ( \beta _ { 1 } , \beta _ { 2 } , \ldots , \beta _ { Z } ) \in \mathbb { R } ^ { Z p } .
$$

where $v e c ( \cdot )$ represents a vec operator. This formulation allows the difference between each pair of coefficients to be expressed using linear operators. Here, the incidence matrix D defines a scalar difference operator, while $\beta$ is vector-valued. Therefore, $\dot { D }$ must be extended to act on vector-valued parameters. To achieve this, the Kronecker product is applied with $I _ { p }$ and denoted by Ω:

$$
D \otimes I _ { p } \in \mathbb { R } ^ { | E | p \times Z p } = \Omega .
$$

where ⊗ is the Kronecker product and $I _ { p }$ is the identity matrix.

Using Ω, the pairwise differences between institution-specific coefficient vectors can be expressed as

$$
( D \otimes I _ { p } ) \beta = \Omega \beta
$$

where the resulting vector is obtained by summing the differences along all edges. $\Omega \beta$ is equivalent to Eq. (7). Therefore, the objective function in Eq. (5) can be rewritten as

$$
\operatorname* { m i n } _ { \{ \beta \} } \frac { 1 } { Z } \sum _ { z = 1 } ^ { Z } \frac { 1 } { n _ { z } m _ { z } } \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } \log \left( 1 + \exp [ - ( \beta _ { z } ^ { \top } x _ { i z } ^ { + } - \beta _ { z } ^ { \top } x _ { j z } ^ { - } ) - q ] \right) + \lambda \| W \Omega \beta \| _ { 1 }\tag{9}
$$

where $\pmb { W } = \mathrm { d i a g } ( \pmb { w } ) \otimes \pmb { I _ { p } } \in \mathbb { R } ^ { | \pmb { E } | \pmb { p } \times | \pmb { E } | p }$ is a block-diagonal weight matrix. From here, to simplify the notation, we shall represent the subscripts of $w _ { u v }$ using a single index and express it as a vector as $\pmb { w } = ( w _ { 1 } , w _ { 2 } , \dots , w _ { | E | } )$ Furthermore, to apply Eq. (9) to the framework of PFL and utilize the ADMM algorithm, the objective function in Eq. (9) can be rewritten with the constraint as

$$
\operatorname* { m i n } _ { \{ \beta , \delta \} } \frac { 1 } { Z } \sum _ { z = 1 } ^ { Z } \frac { 1 } { n _ { z } m _ { z } } \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } \log \left( 1 + \exp [ - ( \beta _ { z } ^ { \top } x _ { i z } ^ { + } - \beta _ { z } ^ { \top } x _ { j z } ^ { - } ) - q ] \right) + \lambda \| W \delta \| _ { 1 } , \quad \mathrm { s . t . } \ \Omega \beta = \delta .\tag{10}
$$

δ denotes the vector whose elements are the difference between the coefficients u and $v$ for each variable. This reformulation enables the use of the ADMM because it separates the coupled terms and facilitates tractable optimization. Specifically, the introduction of the auxiliary variable allows the optimization problem to be decomposed into subproblems. In the framework of the PFL, the updates for $\beta$ must be performed locally because they require data from each institution, whereas the global variables, including $\delta ,$ are updated on the server side. Therefore, the communication overhead between the server and each institution associated with the updates can be reduced. Eq. (10) can be solved based on the corresponding Lagrangian function, which can be defined as follows:

$$
\begin{array} { c } { \displaystyle \mathcal { L } _ { \rho } ( \beta , \delta , \gamma ) = \frac { 1 } { Z } \sum _ { z = 1 } ^ { Z } \frac { 1 } { n _ { z } m _ { z } } \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } \log \left( 1 + \exp [ - ( \beta _ { z } ^ { \top } { \pmb x } _ { i z } ^ { + } - \beta _ { z } ^ { \top } { \pmb x } _ { j z } ^ { - } ) - q ] \right) } \\ { \displaystyle + \lambda \sum _ { o = 1 } ^ { | E | } w _ { o } \| \delta _ { o } \| _ { 1 } + \gamma ^ { \top } ( \Omega \beta - \delta ) + \frac { \rho } { 2 } \| \Omega \beta - \delta \| _ { 2 } ^ { 2 } } \end{array}\tag{11}
$$

where $\begin{array} { r } { \pmb { \delta } = ( \pmb { \delta } _ { 1 } , \pmb { \delta } _ { 2 } , \dots , \pmb { \delta } _ { | E | } ) ^ { \top } = ( \pmb { \delta } _ { \xi } ) , ( \pmb { \xi } = 1 , 2 , \dots , | E | p ) } \end{array}$ and $\pmb { \delta } _ { o } = ( \delta _ { o 1 } , \delta _ { o 2 } , \ldots , \delta _ { o p } ) ^ { \top } \in \mathbb { R } ^ { p }$ is different vector corresponding to $o , \gamma \in \mathbb { R } ^ { | E | p }$ is the Lagrangian multiplier and $\rho ( \rho > 0 )$ is the tuning parameter. $\| \cdot \| _ { 2 } ^ { 2 }$ is the Euclidean norm.

## 3.4 Update β

In this subsection, we explain how to derive the updated formula of $\beta . \ \mathrm { A s } \ \beta$ depends on the local data of each institution, within the PFL framework, the parameters must be updated at each institution. In the PFL framework proposed in Liu et al. [2025], the updated formula of $\beta$ can be decomposed into a server-side term that can be evaluated centrally and a client-specific term that depends on local data. This procedure enables parameter estimation that accounts for institutional heterogeneity while reducing computational costs and preserving data privacy. The updated formula of $\beta$ is as follows:

$$
\boldsymbol { \beta } ^ { ( t + 1 ) } = \boldsymbol { \beta } ^ { ( t , g ) } - \boldsymbol { r } ^ { - 1 } \tau \nabla f ( \boldsymbol { \beta } ^ { ( t ) } ) ,\tag{12}
$$

$$
\mathrm { w h e r e } \beta ^ { ( t , g ) } = r ^ { - 1 } H \beta ^ { ( t ) } - r ^ { - 1 } \tau \big [ - \rho \Omega ^ { \top } \pmb \delta ^ { ( t ) } + \Omega ^ { \top } \gamma ^ { ( t ) } \big ] .\tag{13}
$$

$\beta ^ { ( t , g ) }$ is the global component which depends on the regularization term and ADMM variables. $\beta ^ { ( t ) }$ is $\beta$ at tth step. $\nabla f ( \beta )$ is the gradient of $f ( \beta )$ , which represents the first term of Eq. (11). $\tau \left( \tau > 0 \right)$ is a step-size parameter that controls the curvature of the quadratic approximation. H is a positive definite matrix, which can be defined as follows:

$$
\mathbf { } H = r I - \rho \tau \boldsymbol { \Omega } ^ { \top } \boldsymbol { \Omega } .\tag{14}
$$

$r \left( r > 0 \right)$ is a scaling parameter for H, which is set as

$$
r > \rho \tau \varsigma _ { \mathrm { m a x } } ( \Omega ^ { \top } \Omega ) + \operatorname* { m a x } \left( \frac { \tau \mu } { 2 } , 1 \right) .\tag{15}
$$

First, We explain the Lipschitz continuity of the gradient of $f ( \beta )$ to derive this majorizing function.

Lemma 1. The gradient of $f ( \beta )$ is Lipschitz continuous with $\begin{array} { r c l } { \mu } & { = } & { \operatorname* { m a x } _ { z \in \{ 1 , 2 , \ldots , Z \} } ( \mu _ { z } ) } \end{array}$ , where $\begin{array} { r l } { \mu _ { z } } & { { } = } \end{array}$ $\begin{array} { r } { \varsigma _ { \mathrm { m a x } } \big ( { \frac { 1 } { 4 n _ { z } m _ { z } } } \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } d _ { i j z } d _ { i j z } ^ { \top } \big ) } \end{array}$ , in which $d _ { i j z } = \pmb { x } _ { i z } ^ { + } - \pmb { x } _ { j z } ^ { - } , ( i = 1 , \dots , n _ { z } ; j = 1 , \dots , m _ { z } )$ . Here, $\varsigma _ { \mathrm { m a x } } ( O )$ describes the largest eigenvalue of matrix O.

The proof of Lemma 1 is shown in Appendix A. Using the Lipschitz continuity of $\nabla f ( \beta )$ , which denotes the gradient of $f ( \beta )$ , established in Lemma 1, the update of $\beta$ is obtained by approximating the first term of Eq. (11) with a quadratic function, while retaining the linear and quadratic penalty terms in their original form. When H is a positive definite matrix, the updated formula of $\beta$ can be derived based on the following function:

$$
\tilde { \mathcal { L } } _ { \rho } ( \beta ; \beta ^ { ( t ) } , \delta ^ { ( t ) } , \gamma ^ { ( t ) } ) = f ( \beta ^ { ( t ) } ) + \nabla f ( \beta ^ { ( t ) } ) ^ { \top } ( \beta - \beta ^ { ( t ) } ) + \frac { 1 } { 2 \tau } ( \beta - \beta ^ { ( t ) } ) ^ { \top } H ( \beta - \beta ^ { ( t ) } ) + \gamma ^ { \top } \Omega \beta + \frac { \rho } { 2 } \| \Omega \beta - \delta \| _ { 2 } ^ { 2 } .\tag{16}
$$

where $\beta ^ { ( t ) }$ is $\beta$ at tth step. The following Theorem 2 shows that the results of the updated β yields a sufficient decrease in the augmented Lagrangian function of Eq. (11). Here, Theorem 2 states that Eq. (16) is the majorizing function [Hunter and Lange, 2004] of Eq. (11) by choosing r appropriately. Theorem 2 can be derived in the same manner as Lemma 1 in Liu et al. [2025].

Theorem 2. For an update of ${ \bf \dot { \rho } } _ { \beta , }$ the difference in the augmented Lagrangianfunction satisfies thefollowing condition when $r > \rho \tau \varsigma _ { \mathrm { m a x } } ( \Omega ^ { \top } \Omega ) +$ max $\textstyle { \bigl ( } { \frac { \tau \mu } { 2 } } , { \bar { 1 } } { \bigr ) }$ :

$$
\mathcal { L } _ { \rho } ( \beta ^ { ( t + 1 ) } , \delta ^ { ( t ) } , \gamma ^ { ( t ) } ) - \mathcal { L } _ { \rho } ( \beta ^ { ( t ) } , \delta ^ { ( t ) } , \gamma ^ { ( t ) } ) \leq - \bigg [ \frac { \mathrm { { c m i n } } ( H ) } { \tau } + \frac { \rho \cdot \varsigma _ { \operatorname* { m i n } } ( \Omega ^ { \top } \Omega ) } { 2 } - \frac { \mu } { 2 } \bigg ] \| \beta ^ { ( t + 1 ) } - \beta ^ { ( t ) } \| _ { 2 } ^ { 2 } .\tag{17}
$$

where $\varsigma _ { m a x } ( O )$ and $\varsigma _ { m i n } ( \pmb { O } )$ )denote the largest and smallest eigenvalue of matrix O, respectively. This property follows from the uniform boundedness ofthe Hessian ofthe logistic loss.

The proof of Theorem 2 is in Appendix B. Using Eq. (16), the updated formula of $\beta$ is defined as

$$
\beta ^ { ( t + 1 ) } = r ^ { - 1 } \pmb { H } \beta ^ { ( t ) } - r ^ { - 1 } \tau \big [ \nabla f ( \beta ^ { ( t ) } ) - \rho \pmb { \Omega } ^ { \top } \pmb { \delta } ^ { ( t ) } + \pmb { \Omega } ^ { \top } \gamma ^ { ( t ) } \big ] .\tag{18}
$$

When H is just a positive definite, the updated formula of $\beta$ includes the inverse of $Z p \times Z p$ matrix. Compared with direct updates with the inverse matrix, whose computational cost is $O \{ ( Z p ) ^ { 3 } \}$ , the update of Eq. (18), avoiding matrix inversion requires matrix-vector multiplications. Therefore, the computational cost is reduced to ${ \hat { O } } \{ ( Z p ) ^ { 2 }  \bar { \} }$ . Here, in Eq. (18), only $\nabla f ( \beta ^ { ( t ) } )$ depends on the data at each institution, while the remaining components are handled on the server. Therefore, to reduce the computational load, Liu et al. [2025] decomposed Eq. (18), as Eq. (12) and Eq. $( 1 3 ) . \beta ^ { ( t , g ) }$ is the global component which depends on the regularization term and ADMM variables. This is computed at the central server. By contrast, the gradient of the loss function $\nabla f ( \beta )$ is the local component. $\nabla f ( \beta )$ is can be decomposed into $\nabla f _ { z } ( \beta _ { z } )$ , which can be computed independently at each institution. As a result, updating $\beta ^ { ( t , g ) }$ is performed on the server first, then $\beta ^ { ( t + 1 ) }$ is computed at each institution as $\beta _ { z } ^ { ( t + 1 ) }$

## Update on server

To update $\beta ,$ , Eq. (13) is updated first on the server. After the global component is updated, the server distributes $\beta _ { z } ^ { ( t , g ) }$ to each institution to update $\beta _ { z }$ .

## Update at local institutions

Each local institution downloads $\beta _ { z } ^ { ( t , g ) }$ and updates $\beta _ { z }$ . Let $\begin{array} { r } { f _ { z } = \frac { 1 } { n _ { z } m _ { z } } \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } \log \left( 1 + \exp [ - ( \beta _ { z } ^ { \top } d _ { i j z } ) - q ] \right) } \end{array}$ then $\beta _ { z }$ can be updated by the following:

$$
\beta _ { z } ^ { ( t + 1 ) } = \beta _ { z } ^ { ( t , g ) } - r ^ { - 1 } \tau \nabla f _ { z } ( \beta _ { z } ^ { ( t ) } )\tag{19}
$$

where

$$
\nabla f _ { z } ( \beta _ { z } ) = \frac { 1 } { n _ { z } m _ { z } } \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } \big ( - d _ { i j z } \sigma ( - ( \beta _ { z } ^ { \top } d _ { i j z } + q ) ) \big ) .
$$

Here, $\sigma ( a ) = 1 / ( 1 + \exp ^ { - a } )$ . Updated $\beta _ { z }$ is sent back to the server and merged as $\beta ^ { ( t + 1 ) }$ by updating Eq. (13).

Remark 1. Unless $\pmb { H } \neq r \pmb { I } - \rho \tau \pmb { \Omega } ^ { \top } \pmb { \Omega } ,$ even ifH is a positive semidefinite matrix, then the updatedformula $f o r \beta$ in Eq. (16) involves the inverse matrix. Therefore, maintaining the inverse matrix on the central server can become a significant burden, particularly when the number ofvariables or the number ofparticipating institutions is large. From the right-hand side of Eq. (16), the terms related to $\beta$ are

$$
\nabla f ( { \beta ^ { ( t ) } } ) ^ { \top } ( \beta - \beta ^ { ( t ) } ) + \frac { 1 } { 2 \tau } ( \beta - \beta ^ { ( t ) } ) ^ { \top } { \pmb { H } } ( \beta - \beta ^ { ( t ) } ) + \gamma ^ { \top } \Omega \beta + \frac { \rho } { 2 } \| \Omega \beta - \delta \| _ { 2 } ^ { 2 } + \mathrm { c o n s t } ,\tag{20}
$$

where const is a constant value. Differentiate Eq. (20) by β and set it as 0;

$$
\begin{array} { r l } & { \nabla f ( \beta ^ { ( t ) } ) + \frac { 1 } { \tau } H ( \beta - \beta ^ { ( t ) } ) + \boldsymbol { \Omega } ^ { \top } \boldsymbol { \gamma } + \rho \boldsymbol { \Omega } ^ { \top } ( \boldsymbol { \Omega } \beta - \delta ) = \mathbf { 0 } } \\ { \quad \iff } & { \nabla f ( \beta ^ { ( t ) } ) + \tau ^ { - 1 } H \beta - \tau ^ { - 1 } H \beta ^ { ( t ) } + \boldsymbol { \Omega } ^ { \top } \boldsymbol { \gamma } + \rho \boldsymbol { \Omega } ^ { \top } \boldsymbol { \Omega } \beta - \rho \boldsymbol { \Omega } ^ { \top } \delta = \mathbf { 0 } } \\ { \quad \iff } & { \left( \tau ^ { - 1 } H + \rho \boldsymbol { \Omega } ^ { \top } \boldsymbol { \Omega } \right) \beta = \tau ^ { - 1 } H \beta ^ { ( t ) } - \nabla f ( \beta ^ { ( t ) } ) - \boldsymbol { \Omega } ^ { \top } \boldsymbol { \gamma } + \rho \boldsymbol { \Omega } ^ { \top } \delta } \end{array}
$$

From $E q . \ ( 2 0 )$ , we can obtain the updatedformula ofβ as

$$
\boldsymbol { \beta } ^ { ( t + 1 ) } = ( \tau ^ { - 1 } \boldsymbol { H } + \rho \boldsymbol { \Omega } ^ { \top } \boldsymbol { \Omega } ) ^ { - 1 } \big [ \tau ^ { - 1 } \boldsymbol { H } \boldsymbol { \beta } ^ { ( t ) } - \nabla f ( \boldsymbol { \beta } ^ { ( t ) } ) + \rho \boldsymbol { \Omega } ^ { \top } \boldsymbol { \delta } ^ { ( t ) } - \boldsymbol { \Omega } ^ { \top } \boldsymbol { \gamma } ^ { ( t ) } \big ] .\tag{21}
$$

Therefore, in Liu et al. [2025], H is defined as Eq. (14), which allows calculation without requiring the inverse matrix. To obtain this, we substitute Eq. (14) into thefirst term ofEq. (21):

$$
\boldsymbol { \tau } ^ { - 1 } \boldsymbol { H } + \rho \boldsymbol { \Omega } ^ { \top } \boldsymbol { \Omega } = \boldsymbol { \tau } ^ { - 1 } ( r \boldsymbol { I } - \rho \boldsymbol { \tau } \boldsymbol { \Omega } ^ { \top } \boldsymbol { \Omega } ) + \rho \boldsymbol { \Omega } ^ { \top } \boldsymbol { \Omega } = r \boldsymbol { \tau } ^ { - 1 } \boldsymbol { I } ,
$$

then,

$$
( \boldsymbol { \tau } ^ { - 1 } \boldsymbol { H } + \rho \boldsymbol { \Omega } ^ { \top } \boldsymbol { \Omega } ) ^ { - 1 } = r \boldsymbol { \tau } ^ { - 1 } \boldsymbol { I } .
$$

Therefore, with H as in $E q . \ ( l 4 ) $ , the updatedformula ofβ, the inverse matrix is reduced to a scalar multiple ofthe identity matrix, thereby avoiding the need to compute the inverse matrix directly.

## 3.5 Update δ

Next, δ is updated on the server after $\beta$ has been updated. The updated formula of $\delta = \left( \delta _ { \xi } \right)$ is as follows:

$$
\delta _ { \xi } ^ { ( t + 1 ) } = S _ { \psi } \big ( \nu _ { \xi } ^ { ( t ) } \big ) , \quad ( \xi = 1 , 2 , \dots , | E | p )\tag{22}
$$

where $S _ { \psi } ( \cdot )$ denotes soft thresholding operator. $\begin{array} { r } { \psi = \frac { \lambda \tilde { w } _ { \xi } } { \rho } , \pmb { \nu } ^ { ( t ) } = ( \nu _ { \xi } ^ { ( t ) } ) } \end{array}$ is defined as $\Omega \beta ^ { ( t ) } + \rho ^ { - 1 } \gamma . \ \tilde { w } _ { \xi }$ denotes the edge weight $w _ { u v }$ corresponding to $\delta _ { \xi }$ . Now we explain the procedure for deriving Eq. (22). The terms associated with δ in Eq. (11) are

$$
\lambda \sum _ { o = 1 } ^ { | E | } w _ { o } \| \pmb { \delta } _ { o } \| _ { 1 } + \gamma ^ { \top } ( \pmb { \Omega } \beta - \pmb { \delta } ) + \frac { \rho } { 2 } \| \pmb { \Omega } \beta - \pmb { \delta } \| _ { 2 } ^ { 2 } .\tag{23}
$$

Transform Eq. (23):

$$
\rho ^ { - 1 } \gamma ^ { \top } ( \Omega \beta - \delta ) + \frac 1 2 \| \Omega \beta - \delta \| _ { 2 } ^ { 2 } + \frac { \lambda } { \rho } \sum _ { o = 1 } ^ { | E | } w _ { o } \| \delta _ { o } \| _ { 1 } = \frac 1 2 \| \delta - ( \Omega \beta + \rho ^ { - 1 } \gamma ) \| _ { 2 } ^ { 2 } + \frac { \lambda } { \rho } \sum _ { o = 1 } ^ { | E | } w _ { o } \| \delta _ { o } \| _ { 1 } .\tag{24}
$$

Then, we divide the right-hand side of Eq. (24) into differentiable term and other term.

$$
\left\{ \begin{array} { l l } { \mathcal { F } ( \pmb { \delta } ) = \frac { 1 } { 2 } \| \pmb { \delta } - ( \pmb { \Omega } \pmb { \beta } + \rho ^ { - 1 } \pmb { \gamma } ) \| _ { 2 } ^ { 2 } } \\ { \mathcal { G } ( \pmb { \delta } ) = \frac { \lambda } { \rho } \sum _ { o = 1 } ^ { | E | } w _ { o } \| \pmb { \delta } _ { o } \| _ { 1 } } \end{array} \right.
$$

Using this, the proximity operator of G(δ) is defined as follows:

$$
\mathrm { p r o x } _ { \psi _ { o } \parallel \cdot \parallel _ { 1 } } ( \pmb { \nu } _ { o } ) = \underset { \delta _ { o } } { \mathrm { a r g m i n } } ( \psi _ { o } \| \pmb { \delta } _ { o } \| _ { 1 } + \frac { 1 } { 2 } \| \delta _ { o } - \pmb { \nu } _ { o } \| _ { 2 } ^ { 2 } ) .\tag{25}
$$

Eq. (25) is separable with respect to each component. Therefore, the problem reduces to solving, for each component $\xi \colon$

$$
\operatorname * { a r g m i n } _ { \delta _ { \xi } } \psi \vert \delta _ { \xi } \vert + \frac { 1 } { 2 } ( \delta _ { \xi } - v _ { \xi } ) ^ { 2 } .
$$

Therefore, the minimizer is obtained as Eq. ( 22).

## 3.6 Update γ

The update of $\gamma$ is performed by the deviation from the constraint, using the gradient descent method as

$$
\small \gamma ^ { ( t + 1 ) } = \gamma ^ { ( t ) } + \rho ( \Omega \beta ^ { ( t + 1 ) } - \delta ^ { ( t + 1 ) } ) .\tag{26}
$$

The detail of the Algorithm of GrAUC-PFL is shown in Algorithm 1.

Algorithm 1 Geographically regularized AUC-maximizing personalized federated learning (GrAUC-PFL)   
Require: $X _ { z } ( z = 1 , 2 , \cdots , Z ) , y _ { z } , \Omega , H , r , w _ { u v } , \rho , \tau , q$   
Ensure: $\beta _ { z } ~ ( z = 1 , 2 , \ldots , Z ) , \delta , \gamma$   
1: Set t ← 1   
2: Set initial values $\beta _ { z } ^ { ( 0 ) } , \delta ^ { ( 0 ) }$ and $\gamma ^ { ( 0 ) }$   
3: while $\mathcal { L } _ { \rho } ^ { ( t ) } - \mathcal { L } _ { \rho } ^ { ( t + 1 ) } \geq \epsilon$ do   
4: Update B:   
Server:   
5: Update $\pmb { \beta } ^ { ( t , g ) }  \boldsymbol { r } ^ { - 1 } \pmb { H } \pmb { \beta } ^ { ( t ) } - \boldsymbol { r } ^ { - 1 } \tau \big [ - \rho \pmb { \Omega } ^ { \top } \pmb { \delta } ^ { ( t ) } + \pmb { \Omega } ^ { \top } \pmb { \gamma } ^ { ( t ) } \big ]$   
Local institution:   
Download $\beta _ { z } ^ { ( t ) }$ from Server   
6: Update $\beta _ { z } ^ { ( t + 1 ) } \gets \beta _ { z } ^ { ( t , g ) } - r ^ { - 1 } \tau \nabla f _ { z } ( \beta _ { z } ^ { ( t ) } )$ , where $\begin{array} { r } { \nabla f _ { z } ( \beta _ { z } ) = \frac { 1 } { n _ { z } m _ { z } } \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } \big ( - { d _ { i j z } } \sigma ( - ( \beta _ { z } ^ { \top } d _ { i j z } + q ) ) \big ) } \end{array}$   
Upload $\beta _ { z } ^ { ( t + 1 ) }$ to Server   
Server:   
Integrate $\beta _ { z } ^ { ( t + 1 ) }$ to form $\beta ^ { ( t + 1 ) }$   
7: Update $\tilde { \delta } ^ { ( t + 1 ) }$ based on Eq. (22)   
8: Update $\gamma ^ { ( t + 1 ) }  \gamma ^ { ( t ) } + \overset { ^ { \prime } } { \rho } ( \Omega \beta ^ { ( t + 1 ) } - \delta ^ { ( t + 1 ) } )$   
9: end while

## 3.7 Selection of tuning parameters

We select the tuning parameters $\lambda , \rho ,$ and $w _ { d i s t }$ for the weight of the regularization term $w _ { u v } ,$ using stratified K-fold cross-validation. Stratified sampling is employed so that the proportion of positive and negative cases is approximately maintained in each fold. Patients are randomly assigned to folds with approximately equal fold sizes. At each step, GrAUC-PFL is trained jointly across all institutions using the corresponding training subsets. The candidates of the tuning parameters are sent to the server and used in server-side calculations. Once the AUC has been calculated at each institution, the results are aggregated at the central server to calculate the overall mean. The combination of parameters that maximized the mean validation AUC across institutions is selected. The combination of tuning parameters that achieved the highest mean validation AUC across the K-folds is selected. As for the calculations of the mean AUC values for CV, the following methods are available:

$$
\mathrm { A U C } _ { \mathrm { C V } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { z } m _ { z } } \sum _ { z = 1 } ^ { Z } \sum _ { ( i , j ) \in C _ { k } } I \Bigl ( \hat { \beta } _ { z k } ^ { \top } ( \pmb { x } _ { i z } ^ { + } - \pmb { x } _ { j z } ^ { - } ) > 0 \Bigr )
$$

where $C _ { k }$ denotes the Cartesian product of the set of positive samples and that of negative samples belonging to kth fold. $\hat { \beta } _ { z k }$ represents estimated $\hat { \beta } _ { z }$ corresponding to kth fold.

## 4 Numerical simulation

## 4.1 Simulation design

We conduct four numerical simulations to evaluate the performance of GrAUC-PFL. In this section, we describe the simulation design based on Chen and Huang [2025]. The settings for each of the four simulations are as follows: Simulation 1 examines the case where a relationship exists between the structure of the mean vector and the geographical group structure. Simulation 2 involves a scenario where groups exist within the mean structure that are related to geographical coordinates, but where the sample size varies by institutions. In simulation 3, the institutions have a mean structure in group structure, but this is not related to their geographical coordinates. Simulation 4 examines all institutions sharing a single mean structure, and no correspondence exists between the geographical and mean structures.

In all simulations, positive and negative labels are denoted by y, and data that differ for each class are generated. Then, we generate the coordinates based on the proximity of institutions.

The common settings is explained first. Let the number of institutions be $Z = 2 0 \ ( z = 1 , 2 , \dots , 2 0 )$ . The outcome is defined as $y _ { h z } = \bar { \{ 0 , 1 \} } \bar { ( h = 1 , . . . , N _ { z } ) }$ , where $y _ { h z } = 1$ represents a positive sample and $y _ { h z } = 0$ represents a negative one. For each institution, $n _ { z }$ and $m _ { z }$ denote the number of positive and negative samples, respectively, such that $N _ { z } = n _ { z } + m _ { z }$ . The prevalence rate was set to be the same across all institutions, with π $( = n _ { z } / N _ { z } ) \in$ $\{ 0 . 3 , 0 . 2 5 , 0 . 2 , 0 . 1 5 , 0 . 1 , 0 . 0 5 \}$

Next, set the dimension of the explanatory variables at institution z as $\begin{array} { r } { \pmb { X _ { z } } \in \mathbb { R } ^ { N _ { z } \times p } \mathrm { t o } p = 1 8 . } \end{array}$ , designating 3 of these variables as active variables and the remaining 15 as noise. Let $\pmb { x } _ { h z } \in \mathbb { R } ^ { p }$ denote the vector of explanatory variables for individual h at institution z. For negative samples $( y _ { h z } = 0 )$ , we assume that $\pmb { x } _ { j z } ^ { - } \sim N ( \mathbf { 0 } , \pmb { I } _ { p } )$ , and for positive samples $( y _ { h z } = 1 )$ ,

$$
\pmb { x } _ { i z } ^ { + } \sim N ( \pmb { \mu } _ { z } ^ { * } , \pmb { I } _ { p } )\tag{27}
$$

where $\pmb { \mu } _ { z } ^ { \ast } = ( \mu _ { z } ^ { \ast } , \mu _ { z } ^ { \ast } , \mu _ { z } ^ { \ast } , 0 , \ldots , 0 ) ^ { \top } \in \mathbb { R } ^ { p }$ . The first three variables are informative variables, and the remaining are noise. The settings for $\mu _ { z } ^ { * }$ are explained in each simulation setting.

Now, we explain the setting of GrAUC-PFL. The weight for each edge $( u , v )$ is calculated based on Eq. (6). We use R package geosphere [Hijmans, 2024] to compute pairwise distance from the coordinate data, and the MST is obtained using $\mathrm { i } \operatorname { g r a p h }$ package [Csárdi et al., 2026]. The constant value $q = 0 . 0 1$ , and each simulation was repeated 100 times. For the initial values of the parameters, the coefficient vector $\beta$ was initialized by randomly sampling from a standard normal distribution for each institution $z ; \beta _ { z } ^ { ( 0 ) } \sim N ( 0 , I _ { p } )$ . The auxiliary variable δ is initialized as $\pmb { \delta } ^ { ( 0 ) } = \Omega \beta ^ { ( 0 ) }$ to ensure that the consistency condition is satisfied at initializationzw. The dual variable γ is initialized as the zero vector.

For the compared methods, we adopted two approaches. The first one is a local approach that maximizes the AUC surrogate independently at each institution, which is named ’individual’. The second method is an approach that directly maximizes the AUC within the framework of Federated Averaging (FedAvg) [McMahan et al., 2017], which estimates a common coefficient vector using conventional FL.

The evaluation metric was the AUC for each institution by using test data and calculating the mean value across institutions. For the AUC calculation for each institution, we used roc function of the pROC package in R [Robin et al., 2011].

## Simulation 1

Simulation 1 examines cases where similarities exist in the structure of institutions based on geographical proximity. The number of samples at each institution z in the training data is set to $N _ { z , t r a i n } \in \{ 5 0 , 3 0 0 \}$ , and the number of samples in the test data is set to $N _ { z , t e s t } = 1 0 0$ . The 20 institutions are divided into two clusters $( z = 1$ to 10 and $z = 1 1 6 2 0 )$ consisting of geographically adjacent institutions. Assuming the center coordinates of each cluster are $\pmb { c } _ { 1 } = ( 0 . 1 , 0 . 1 ) ^ { \top }$ and $\bar { { \mathbf { c } } _ { 2 } } = \bar { ( 0 . 9 , 0 . 3 ) } ^ { \top }$ respectively, the coordinates of the individual institutions are generated as follows:

$$
( \mathrm { l o n } _ { z } , \mathrm { l a t } _ { z } ) = { \pmb { c } } _ { g ( z ) } + { \pmb { \epsilon } } _ { z } , \quad { \pmb { \epsilon } } _ { z } \sim N ( 0 , 0 . 0 3 ^ { 2 } I _ { 2 } ) .
$$

where $g ( z ) = 1$ for $z \leq 1 0$ and $g ( z ) = 2 { \mathrm { ~ f o r ~ } } z \geq 1 1$ . Here, $g ( z )$ is a function allocating clustering number for institution z.

Next, for the mean of the covariates for the positive samples, we set $\mu _ { z } ^ { * }$ as $\pmb { \mu } _ { l g ( z ) } ^ { * } = ( \mu _ { l g ( z ) } ^ { * } , \mu _ { l g ( z ) } ^ { * } , \mu _ { l g ( z ) } ^ { * } , 0 , \ldots , 0 ) ^ { \top } \in$ $\mathbb { R } ^ { p }$ depending on l and $g ( z )$ . The signal is controlled by the mean shift $\mu _ { l g ( z ) } ^ { * }$ in the first three variables, while the remaining variables act as noise. $\mu _ { l g ( z ) } ^ { * }$ is set for each group $g ( z ) \in \{ 1 , 2 \}$ , and three different levels $l = \{ 1 , 2 , 3 \}$ The pattern for $\mu _ { l g ( z ) } ^ { * }$ is shown in Table 1. Based on the above settings, training and test datasets are independently generated from the same data-generating process with the same prevalence π using Eq. (27).

Table 1: The pattern of $\mu _ { l g ( z ) } ^ { * }$ . l represents the levels of the mean structure and $g ( z )$ is the group of institutions with geographical similarity.
<table><tr><td></td><td>group g(z)</td></tr><tr><td>level l</td><td>1 2</td></tr><tr><td>1</td><td>0.9 -0.9</td></tr><tr><td>2</td><td>0.7 -0.7</td></tr><tr><td>3</td><td>0.5 -0.5</td></tr></table>

The tuning parameters λ and $\rho ,$ and $w _ { d i s t }$ are determined using 5-fold cross-validation applied to the entire PFL model.

## Simulation 2

Simulation 2 examines the situations where the number of patients in training data varies by institutions. The mean structure and the geographical coordinates are identical to those in simulation 1. To reflect heterogeneous sample sizes across institutions, the training sample size for each institution $N _ { z , t r a i n }$ is randomly assigned from {30, 50, 100, 200, 300} with equal probability and is fixed throughout each simulation replicate. The test sample size is set as $N _ { z , t e s t } = 2 0 0$ . As the minimum sample size of the training data is small, a 3-fold cross-validation is performed to generate positive/negative pairs for each fold. Furthermore, the sample sizes for positive and negative cases are calculated principally based on prevalence; however, for institutions with small sample sizes, the numbers are adjusted to ensure that at least four cases are allocated to each category. Other settings are same as those in simulation 1.

## Simulation 3

As with simulation 1, the data in simulation 3 possesses a group structure based on the mean values of the covariates for positive samples; however, geographical coordinates are assumed independent of this mean-value structure.

Now we explain the settings. For each institution $( z = 1 , 2 , . . . , 2 0 )$ , its longitude and latitude are independently generated as

$$
\mathrm { l o n } _ { z } \sim \mathrm { U n i f } ( 1 3 0 , 1 4 5 ) , \quad \mathrm { l a t } _ { z } \sim \mathrm { U n i f } ( 3 1 , 4 5 ) .\tag{28}
$$

Thus, institutional locations are distributed randomly over a predefined rectangular geographical region, without imposing geographic group structure. The other settings including the mean structure of the covariates are identical to those in simulation 1.

## Simulation 4

We examine cases where the mean structure is not related to geographical coordinates, for which institutions are generated in the same manner as Eq. (28) in simulation 3. To introduce continuous heterogeneity across institutions without imposing a group structure, institution-specific mean parameters are generated for each institution. Specifically, the mean vector for the positive class in the institution z is then defined as

$$
\pmb { \mu } _ { z } ^ { * } = ( \mu _ { z } ^ { * } , \mu _ { z } ^ { * } , \mu _ { z } ^ { * } , 0 , \ldots , 0 ) ^ { \top } \in \mathbb { R } ^ { p } , \mu _ { z } ^ { * } \sim \operatorname { U n i f } ( 0 . 5 , 0 . 9 ) .
$$

Using this, the covariates for positive samples are generated based on Eq. (27). All other simulation settings, including the number of samples for traning and test data, the generation of predictor variables and evaluation procedure were identical to those used in Simulation 1.

## 4.2 Simulation results

![](images/2ffe72735ba0ad5da92a712543fadd4fb85b1f9d791ce8747d2a2ddd19154656.jpg)  
Figure 3: The results of simulation 1 in $N _ { z , t r a i n } = 5 0$ . The vertical axis shows the mean AUC as assessed by the institution, whilst the horizontal axis represents the method. For each plot, ’meanabs’ mentions the mean structure of $\mu _ { l g ( z ) } ^ { * }$ at Table 1 and ’prev’ denotes the prevalence rate. The methods are as follows: ’GrAUC-PFL’ refers to the proposed method, ’FedAvg’ refers to the Federated Average, and ’individual”’ refers to the method in which models are built for each institution and their AUC are evaluated separately.

![](images/b66cd8d6cb278c13cfccce516ab949424bae64512049dceae75cad03d39a3ccd.jpg)  
Figure 4: The results of simulation 1 in $N _ { z , t r a i n } = 3 0 0 .$

The results of the numerical simulation 1 are shown in Figure 3 for $N _ { z , t r a i n } = 5 0$ and Figure 4 for $N _ { z , t r a i n } = 3 0 0$ GrAUC-PFL achieved the highest mean AUC across all patterns for both $N _ { z , t r a i n } = 5 0$ and 300. Its performance improved as the difference in $\mu _ { l g ( z ) } ^ { * }$ increased, whereas FedAvg showed no notable change. Prevalence had relatively little effect on performance. For $\tilde { N } _ { z , t r a i n } = 3 0 0$ , compared with the results for $N _ { z , t r a i n } = 5 0$ , the AUC of the individual method was close to that of GrAUC-PFL, especially when the prevalence value was large. In simulation 2,

GrAUC-PFL achieved better results in almost all patterns, shown in Figure 7 in Appendices C. Similar to the results of simulation 1, the higher the signal and the higher prevalence, the greater the estimation accuracy. By contrast, FedAvg consistently exhibited the lowest performance. In simulation 3, shown in Figure 8 for $N _ { z , t r a i n } = 5 0$ and Figure 9 for $N _ { z , t r a i n } = 3 0 0$ in Appendices C, the results of GrAUC-PFL were slightly better than those of the individual AUC-maximization method in $\mu _ { l g ( z ) } ^ { * }$ were higher. In the other scenarios, the individual method yielded good results, although the difference in values compared to GrAUC-PFL was not particularly large. In simulation 4, whose results are described in Figure 10 in Appendices C, the results of GrAUC-PFL were almost the same as those of FedAvg when the prevalence was 0.3 and 0.25 for $N _ { z , t r a i n } = 5 0$ . Although FedAvg was superior to GrAUC-PFL for the remaining patterns, the difference between them was slight.

## 5 Real data application

To evaluate the practical usefulness of GrAUC-PFL, we applied it to a real-world dataset. In this section, we first describe the characteristic of the dataset and preprocessing procedures, and then present the predictive performance compared to other methods and estimated parameter patterns obtained by the proposed.

## 5.1 Data description and Experimental Setting

To evaluate the practical usefulness of GrAUC-PFL, we apply it to a real-world dataset. First, we explain the dataset for the application. The real-data application is conducted using the publicly available COVID-19 Case Surveillance Public Use Data with Geography, which contains nationwide surveillance records collected in the United States provided by the Centers for Disease Control and Prevention (CDC) [Centers for Disease Control and Prevention (CDC), 2026]. This dataset is collected from the beginning of 2020 and is accessed on 2 February 2026. This study uses publicly available de-identified data; therefore, institutional review board approval and informed consent are not required. Owing to computational constraints, a subset of 100, 000 records from the original dataset is retrieved for the analysis. Records containing missing, unknown, or unusable variables values are excluded, resulting in 11, 628 complete cases for analysis, and the outcome is hospitalization. Six covariates are selected after excluding records with missing or unknown values. The details of these variables are listed in Table 2.

These data include state and county names and the corresponding geographic codes. Counties are treated as individual units, and geographical proximity is considered. To calculate pairwise geographic distances, the geographic coordinates for each county are derived from county boundary shapefiles obtained from the U.S. Census Bureau’s Topologically Integrated Geographic Encoding and Referencing (TIGER)/Line shapefiles using the R package tigris [Walker, 2025]. To calculate the pairwise distance, we use the R package geosphere and the igraph to construct a graph of the MST. The analysis data comprise 18 counties, arranged as shown in Figure 5. There are some neighboring counties on the west and east coasts. Table 3 lists the patient data by county. Prevalence was calculated as the proportion of hospitalization in each county. The number of hospitalizations in some counties is small and the prevalence in many counties is low.

Next, we explain the procedure used. For each county, we construct a training dataset by randomly sampling hospitalized and non-hospitalized patients separately, and use the remaining observations as the test dataset. First, we partition the data within each county into training and test datasets using outcome-stratified sampling. Approximately 20% of samples from each outcome class are allocated to the training dataset, whereas the remaining samples are assigned to the test dataset. For counties with relatively small numbers of inpatients, the allocation is adjusted to ensure that both outcome classes are represented in the training and test datasets. The tuning parameter $\lambda , \rho ,$ and $w _ { d i s t }$ are determined via stratified 3-fold cross-validation using the training dataset.

The evaluation index is the mean AUC of each county. To calculate the AUC for each county, the R package pROC is used. The compared methods are FedAvg and a localized approach that individually maximize alternative measures of AUC in each county, similar to the numerical simulation.

Table 2: Data summary by county.
<table><tr><td>Variable</td><td>Categories</td></tr><tr><td>age_group</td><td>0-17, 18–49, 50–64, ≥65 yrs</td></tr><tr><td>sex race</td><td>Female, Male</td></tr><tr><td></td><td>American Indian, Alaska Native, Asian, Black, Multiple/Other, Native Hawaiian, Other Pacific Islander, White</td></tr><tr><td>ethnicity</td><td>Hispanic, Non-Hispanic</td></tr><tr><td>current_status</td><td>Laboratory-confirmed case, Probable case</td></tr><tr><td>symptom_status</td><td>Asymptomatic, Symptomatic</td></tr></table>

![](images/c31d3362047858f35df0ca2c6d570eeda450f484b56b4b5068bf874ae8b4798c.jpg)  
Figure 5: Geographic distribution across the contiguous United States, where Alaska, Hawaii, and U.S. territories are excluded from the map visualization.

Table 3: Data distribution by county.
<table><tr><td>State</td><td>County</td><td>Total Nz</td><td>Hospitalization</td><td>Prevalence</td><td>State</td><td>County</td><td>Total Nz</td><td>Hospitalization</td><td>Prevalence</td></tr><tr><td>UT</td><td>Salt Lake</td><td>1,935</td><td>33</td><td>0.017</td><td>NY</td><td>Kings</td><td>212</td><td>189</td><td>0.892</td></tr><tr><td>FL</td><td>Palm Beack</td><td>888</td><td>32</td><td>0.036</td><td>NY</td><td>Queen</td><td>131</td><td>111</td><td>0.847</td></tr><tr><td>FL</td><td>Broward</td><td>1,282</td><td>37</td><td>0.029</td><td>TX</td><td>Dallas</td><td>195</td><td>20</td><td>0.103</td></tr><tr><td>TX</td><td>Bexar</td><td>849</td><td>37</td><td>0.044</td><td>NY</td><td>Nassau</td><td>166</td><td>4</td><td>0.024</td></tr><tr><td>AZ</td><td>Maricopa</td><td>1,107</td><td>256</td><td>0.231</td><td>NY</td><td>Suffolk</td><td>1,212</td><td>39</td><td>0.032</td></tr><tr><td>CA</td><td>Riverside</td><td>522</td><td>14</td><td>0.027</td><td>CA</td><td>Santa Clara</td><td>274</td><td>9</td><td>0.033</td></tr><tr><td>MI</td><td>Wayne</td><td>603</td><td>42</td><td>0.070</td><td>CA</td><td>Alameda</td><td>524</td><td>79</td><td>0.151</td></tr><tr><td>CA</td><td>Los Angels</td><td>1,006</td><td>22</td><td>0.022</td><td>NY</td><td>New York</td><td>34</td><td>27</td><td>0.794</td></tr><tr><td>CA</td><td>San Diego</td><td>578</td><td>57</td><td>0.099</td><td>CA</td><td>Orange</td><td>110</td><td>10</td><td>0.091</td></tr></table>

## 5.2 Results

The results of the mean AUC of each county were shown in Table 4. GrAUC-PFL was superior to the compared methods. Table 5 describes the AUC by county. GrAUC-PFL showed better results in half of the counties. For Los Angeles, Santa Clara, and Riverside, the AUC of the individual AUC maximisation method was lower, whereas the GrAUC-PFL and FedAvg methods were superior for those counties. However, the results of FedAvg in Dallas, Orange, Nassau and New York were rather better than those of the other methods. Next, Figure 6 shows the heatmap of $\hat { \beta }$ of

GrAUC-PFL. The dendrogram confirmed that the data had been successfully clustered into the following groups: East Coast (Nassau, Suffolk, New York, Kings, and Queens), West Coast (Riverside, Maricopa, San Diego, Los Angeles, and Orange), and Florida (Broward and Palm Beach), along with adjacent regions within the same state.

Table 4: Results of mean AUC applying the real-world data.
<table><tr><td></td><td>GrAUC-PFL</td><td>FedAvg</td><td>individual</td></tr><tr><td>AUC</td><td>0.651</td><td>0.628</td><td>0.633</td></tr></table>

Table 5: Results of AUC by each county.
<table><tr><td>County</td><td>GrAUC-PFL</td><td>FedAvg</td><td>individual</td><td>County</td><td>GrAUC-PFL</td><td>FedAvg</td><td>individual</td></tr><tr><td>Salt Lake</td><td>0.659</td><td>0.409</td><td>0.689</td><td>Kings</td><td>0.792</td><td>0.781</td><td>0.865</td></tr><tr><td>Palm Beach</td><td>0.469</td><td>0.493</td><td>0.452</td><td>Queens</td><td>0.644</td><td>0.439</td><td>0.651</td></tr><tr><td>Broward</td><td>0.573</td><td>0.434</td><td>0.573</td><td>Dallas</td><td>0.752</td><td>0.837</td><td>0.776</td></tr><tr><td>Bexar</td><td>0.703</td><td>0.544</td><td>0.694</td><td>Nassau</td><td>0.269</td><td>0.438</td><td>0.269</td></tr><tr><td>Maricopa</td><td>0.719</td><td>0.711</td><td>0.725</td><td>Suffolk</td><td>0.742</td><td>0.742</td><td>0.742</td></tr><tr><td>Riverside</td><td>0.759</td><td>0.758</td><td>0.441</td><td>Santa Clara</td><td>0.594</td><td>0.520</td><td>0.397</td></tr><tr><td>Wayne</td><td>0.814</td><td>0.777</td><td>0.815</td><td>Alameda</td><td>0.773</td><td>0.721</td><td>0.775</td></tr><tr><td>Los Angels</td><td>0.581</td><td>0.488</td><td>0.485</td><td>New York</td><td>0.786</td><td>0.950</td><td>0.595</td></tr><tr><td>San Diego</td><td>0.627</td><td>0.388</td><td>0.626</td><td>Orange</td><td>0.455</td><td>0.870</td><td>0.818</td></tr></table>

![](images/dcd2734a4424e4b03c2d85a71e024f97ea066f04b33b1fcc62caef661bddaa9e.jpg)  
Figure 6: Heatmap of $\hat { \beta }$ of GrAUC-PFL.

## 6 Discussion

In numerical simulations, we examined the performance in four setting scenarios compared to other AUC maximization methods. In simulations 1 and 2, where geographically neighboring institutions shared similar underlying data distributions, GrAUC-PFL demonstrated superior performance compared to the other methods. Simulation 2 further demonstrated that this advantage was maintained, even when training sample sizes varied across institutions, reflecting common features of multicenter studies. In contrast, the individual AUC maximization yielded lower AUC values, suggesting that relying solely on local data may be insufficient when sample size are limited. FedAvg also showed limited performance even when $N _ { z , \mathrm { t r a i n } } = 3 0 0$ , indicating that a single global model may fail to capture institutiona heterogeneity. These findings suggest that GrAUC-PFL can benefit institutions with limited local data by utilizing information from geographically similar institutions while retaining institution-specific model. Simulations 3 and 4 further examined scenarios in which geographical proximity did not reflect similarities in the data distribution. Under these conditions, GrAUC-PFL performed comparable to individual method, without significant performance degradation. These findings suggest that the benefit of spatial regularization depends on whether geographical proximity reflects data similarity, while its misspecification did not result in substantial performance loss in our simulations.

The real-world data application to COVID-19 data further demonstrated potential benefit of the GrAUC-PFL, which achieved the highest mean AUC across counties. Across the counties, GrAUC-PFL outperformed the compared methods in Los Angeles, Santa Clara, and Riverside, where disease prevalence was relatively low. These results suggest that borrowing information from geographically related counties can complement limited local information. In counties where individual models performed best, GrAUC-PFL generally achieved comparable AUC values, suggesting that geographic information sharing did not compromise local predictive performance. However, FedAvg outperformed GrAUC-PFL in some counties such as Nassau and Orange. The coefficient heatmap showed that GrAUC-PFL estimated similar coefficient patterns among geographically neighboring counties, consistent with the intended effect of geographical regularization. This suggests that, in these counties, the overall trends common to all counties may have provided more useful information than local adaptations based on geographical proximity.

## 7 Conclusion

In this study, we proposed GrAUC-PFL, a personalized federated learning framework that maximizes AUC directly, while incorporating geographic similarity among institutions. GrAUC-PFL enables building personalized models, while leveraging information from geographically related institutions without disclosing data externally. Numerical simulations and its application to real-world data demonstrated its effectiveness in improving discriminative performance.

For limitation of GrAUC-PFL is that its assumption that geographically proximate institutions tend to share similar model parameters, which may not always hold in practice. Future work could extend the framework to alternative measure of similarity, such as similarities in local models or data distributions [e.g. Wang et al., 2023, Huang et al., 2021]. In addition, although we adopted a logistic-based surrogate loss for AUC maximization, investigating alternative surrogate losses and their effects on predictive performance and computational efficiency remains an important direction for future research.

## References

S. Zhang and M.M. Singh. Privacy and security in health big data: A NIST-guided systematic review of technologies, challenges, and future directions. Information, 17(2), 2026. ISSN 2078-2489. doi:10.3390/info17020148.

European Parliament and Council of the European Union. Regulation (eu) 2016/679 of the european parliament and of the council of 27 april 2016 on the protection of natural persons with regard to the processing of personal data and on the free movement of such data (general data protection regulation), 2016.

B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Proceedings ofthe 20th International Conference on Artificial Intelligence and Statistics, volume 54 of Proceedings ofMachine Learning Research, pages 1273–1282. PMLR, 2017.

I. Dayan, H. R. Roth, A. Zhong, et al. Federated learning for predicting clinical outcomes in patients with covid-19. Nature Medicine, 27(10):1735–1743, 2021. doi:10.1038/s41591-021-01506-3.

B. L. Y. Agbley, J. Li, A. U. Haq, E. K. Bankas, S. Ahmad, I. O. Agyemang, D. Kulevome, W. D. Ndiaye, B. Cobbinah, and S. Latipova. Multimodal melanoma detection with federated learning. In 2021 18th International Computer Conference on Wavelet Active Media Technology and Information Processing (ICCWAMTIP), pages 238–244, 2021. doi:10.1109/ICCWAMTIP53232.2021.9674116.

A. Fallah, A. Mokhtari, and A. Ozdaglar. Personalized federated learning: A meta-learning approach. In NeurIPS 2020 (Advances in Neural Information Processing Systems), pages 3557–3568, 2020.

Y. Jiang, J. Konecný, K. Rush, and S. Kannan. Improving federated learning personalization via model agnostic metaˇ learning. arXiv preprint arXiv:1909.12488, 2019.

S. Meakin and S. Funk. Quantifying the impact of hospital catchment area definitions on hospital admissions forecasts: Covid-19 in england, september 2020–april 2021. BMC Medicine, 22(1):163, 2024. doi:10.1186/s12916-024-03369- 0.

B. Rader, S.V. Scarpino, A. Nande, et al. Crowding and the shape of covid-19 epidemics. Nature Medicine, 26: 1829–1834, 2020. doi:10.1038/s41591-020-1104-0.

C. Brunsdon, A. S. Fotheringham, and M.E. Charlton. Geographically weighted regression: A method for exploring spatial non-stationarity. Geographical Analysis, 28(4):281–298, 1996. doi:10.1111/j.1538-4632.1996.tb00936.x.

A. E. Gelfand, H. J. Kim, C. F. Sirmans, and S. Banerjee. Spatial modeling with spatially varying coefficient processes. Journal ofthe American Statistical Association, 98(462), 2003. doi:10.1198/016214503000170.

L. Yan, R. Dodier, M.C. Mozer, and R. Wolniewicz. Optimizing classifier performance via an approximation to the wilcoxon-mann-whitney statistic. In Proceedings ofthe Twentieth International Conference on International Conference on Machine Learning, ICML’03, pages 848–855. AAAI Press, 2003. ISBN 1577351894.

T. Yang and Y. Ying. AUC maximization in the era of big data and AI: A survey. ACM Computing Surveys, 55(8):1–37, 2022.

Michael Natole, Jr., Yiming Ying, and Siwei Lyu. Stochastic proximal algorithms for AUC maximization. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 3710–3719. PMLR, 2018.

Y. Huang, L. Chu, Z. Zhou, L. Wang, J. Liu, J. Pei, and Y. Zhang. Personalized cross-silo federated learning on non-iid data. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 35, page 7865–7873, 2021. doi:10.1609/aaai.v35i9.16960.

F. Chen, G. Long, Z. Wu, T. Zhou, and J. Jiang. Personalized federated learning with a graph. In Proceedings ofthe Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, pages 2575–2582. International Joint Conferences on Artificial Intelligence Organization, 7 2022. doi:10.24963/ijcai.2022/357.

S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends® in Machine learning, 3(1):1–122, 2011.

W. Liu, X Mao, X. Zhang, and X. Zhang. Robust personalized federated learning with sparse penalization. Journal of the American Statistical Association, 120(549):266–277, 2025. doi:10.1080/01621459.2024.2321652

T. Li, S. Hu, A. Beirami, and V. Smith. Ditto: Fair and robust federated learning through personalization. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139, pages 6357–6368. PMLR, 2020.

F. Sattler, K.-R. Müller, and W. Samek. Clustered federated learning: Model-agnostic distributed multitask optimization under privacy constraints. IEEE Transactions on Neural Networks and Learning Systems, 32:3710–3722, 2019.

H. B. Mann and D. R. Whitney. On a test of whether one of two random variables is stochastically larger than the other. The annals of mathematical statistics, 18(1):50–60, 1947.

Z. Yuan, Y. Yan, M. Sonka, and T. Yang. Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 3020–3029, 2021. doi:10.1109/ICCV48922.2021.00303.

Yingjie Tian, Yong Shi, Xiaojun Chen, and Wenjing Chen. AUC maximizing support vector machines with feature selection. volume 4, pages 1691–1698, 2011. doi:10.1016/j.procs.2011.04.183. Proceedings of the International Conference on Computational Science, ICCS 2011.

J. Sulam, R. Ben-Ari, and P. Kisilev. Maximizing AUC with deep learning for classification of imbalanced mammogram datasets. In Proceedings of the Eurographics Workshop on Visual Computing for Biology and Medicine, page 131–135. Eurographics Association, 2017. doi:10.2312/vcbm.20171246.

F. Li and H. Sang. Spatial homogeneity pursuit of regression coefficients for large datasets. Journal of the American Statistical Association, 114(527):1050–1062, 2019. doi:10.1080/01621459.2018.1529595.

R. Tibshirani, M. Saunders, S. Rosset, J. Zhu, and K. Knight. Sparsity and smoothness via the fused lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology, 67(1):91–108, 2004. doi:10.1111/j.1467- 9868.2005.00490.x.

D. Garreau, W. Jitkrittum, and M. Kanagawa. Large sample analysis of the median heuristic. arXiv preprint arXiv:1707.07269, 2018.

A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola. A kernel two-sample test. Journal of Machine Learning Research, 13:723–773, 2012.

D. R. Hunter and K. Lange. A tutorial on mm algorithms. The American Statistician, 58(1):30–37, 2004.

Y. Chen and Y. Huang. Maximizing area under the receiver operating characteristic curve for biomarker combination. Statistica Sinica, 2025. doi:10.5705/ss.202024.0195. Accepted paper.

R.J. Hijmans. geosphere: Spherical Trigonometry, 2024. URL https://CRAN.R-project.org/package= geosphere. R package version 1.5-20.

G. Csárdi, T. Nepusz, V. Traag, S. Horvát, F. Zanini, D. Noom, and K. Müller. igraph: Network Analysis and Visualization in R, 2026. URL https://CRAN.R-project.org/package=igraph. R package version 2.1.4.

X. Robin, N. Turck, A. Hainard, N. Tiberti, F. Lisacek, J.-C. Sanchez, and M. Müller. pROC: an open-source package for r and s+ to analyze and compare roc curves. BMC Bioinformatics, 12:77, 2011. doi:10.1186/1471-2105-12-77.

Centers for Disease Control and Prevention (CDC). Covid-19 case surveillance public use data with geography. https: //data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data-with-Ge/ n8mc-b4w4, 2026. Accessed: 2026-02-02.

K. Walker. tigris: Load Census TIGER/Line Shapefiles, 2025. URL https://CRAN.R-project.org/package= tigris. R package version 2.2.1.

Y. Wang, Y. Tong, Z. Zhou, R. Zhang, S. J. Pan, L. Fan, and Q. Yang. Distribution-regularized federated learning on non-IID data. In Proceedings of 2023 IEEE 39th International Conference on Data Engineering (ICDE), pages 2113–2125, 2023. doi:10.1109/ICDE55515.2023.00164.

## Appendices

## A Proof of Lemma 1

Proof. The first term of the logistic surrogate function at institution z is

$$
f _ { z } ( \beta _ { z } ) = \frac { 1 } { n _ { z } m _ { z } } \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } \log \left( 1 + \exp [ - ( \beta _ { z } ^ { \top } x _ { i z } ^ { + } - \beta _ { z } ^ { \top } x _ { j z } ^ { - } ) - q ] \right)\tag{29}
$$

Here, we denote ${ \bf { d } } _ { i j z } = { \bf { x } } _ { i z } ^ { + } - { \bf { x } } _ { j z } ^ { - }$ . Eq. (29) at a pair can be expressed as

$$
\ell _ { i j z } ( \pmb { \beta } _ { z } ) = \log \left( 1 + \exp [ - ( \pmb { \beta } _ { z } ^ { \top } \pmb { d } _ { i j z } ) - q ] \right)\tag{30}
$$

The gradient and Hessian matrix of $\operatorname { E q } .$ . (30) are given as follows:

$$
\begin{array} { r l } & { \quad \nabla \ell _ { i j z } ( \beta _ { z } ) = - \sigma ( - \beta _ { z } ^ { \top } d _ { i j z } - q ) d _ { i j z } , } \\ & { \quad \nabla ^ { 2 } \ell _ { i j z } ( \beta _ { z } ) = \sigma ( - \beta _ { z } ^ { \top } d _ { i j z } - q ) \{ 1 - \sigma ( - \beta _ { z } ^ { \top } d _ { i j z } - q ) \} d _ { i j z } d _ { i j z } ^ { \top } . } \end{array}
$$

where $\sigma ( \cdot )$ is a sigmoid function. Let $u _ { i j z } = - \beta _ { z } ^ { \top } d _ { i j z } - q .$ . Since $\sigma ( u _ { i j z } ) \in ( 0 , 1 )$ , it follows that

$$
\sigma ( u _ { i j z } ) \{ 1 - \sigma ( u _ { i j z } ) \} \leq \frac { 1 } { 4 } ,
$$

Then, since $d _ { i j z } d _ { i j z } ^ { \top }$ is positive semidefinite, for each pair,

$$
\sigma ( u _ { i j z } ) \{ 1 - \sigma ( u _ { i j z } ) \} d _ { i j z } \pm \frac { \top } { i } d _ { i j z } d _ { i j z } ^ { \top } .
$$

where $\preceq$ denotes the positive semidefinite ordering. Averaging over all pairs yields

$$
\begin{array} { l } { { \nabla ^ { 2 } f _ { z } ( \beta _ { z } ) = \displaystyle \frac { 1 } { n _ { z } m _ { z } } \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } \sigma ( u _ { i j z } ) \{ 1 - \sigma ( u _ { i j z } ) \} d _ { i j z } d _ { i j z } ^ { \top } } } \\ { { \displaystyle \qquad \preceq \frac { 1 } { n _ { z } m _ { z } } \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } \frac { 1 } { 4 } d _ { i j z } d _ { i j z } ^ { \top } } } \\ { { \displaystyle \qquad = \frac { 1 } { 4 n _ { z } m _ { z } } \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } d _ { i j z } d _ { i j z } ^ { \top } } } \end{array}\tag{31}
$$

Here, we set

$$
\mu _ { z } = \varsigma _ { \mathrm { m a x } } \bigg ( \frac { 1 } { 4 n _ { z } m _ { z } } \sum _ { i = 1 } ^ { n _ { z } } \sum _ { j = 1 } ^ { m _ { z } } d _ { i j z } d _ { i j z } ^ { \top } \bigg ) ,
$$

where $\varsigma _ { m a x } ( O )$ denotes the largest eigenvalue of $o .$ Hence, the Hessian is uniformly bounded. Let $Q _ { z } \ =$ $\begin{array} { r } { \frac { 1 } { 4 n _ { z } m _ { z } } \sum _ { i = 1 } ^ { n _ { z } } \dot { \sum _ { j = 1 } ^ { m _ { z } } } { d _ { i j z } d _ { i j z } ^ { \top } } } \end{array}$ , for any $\pmb { g } \in \mathbb { R } ^ { p }$ , it holds

$$
\begin{array} { r } { \pmb { g } ^ { \top } \nabla ^ { 2 } f _ { z } ( \beta _ { z } ) \pmb { g } \leq \pmb { g } ^ { \top } \pmb { Q } _ { z } \pmb { g } \leq \varsigma _ { \mathrm { m a x } } ( \pmb { Q } _ { z } ) \| \pmb { g } \| _ { 2 } ^ { 2 } , } \end{array}
$$

where $\| \cdot \| _ { 2 }$ is Euclidean norm. Then,

$$
\begin{array} { r l } & { \pmb { g } ^ { \top } \nabla ^ { 2 } f _ { z } ( \beta _ { z } ) \pmb { g } \le \varsigma _ { \operatorname* { m a x } } ( \pmb { Q } _ { z } ) \lVert \pmb { g } \rVert _ { 2 } } \\ & { \qquad = \pmb { g } ^ { \top } \left( \varsigma _ { \operatorname* { m a x } } ( \pmb { Q } _ { z } ) \pmb { I } \right) \pmb { g } } \\ & { \qquad \nabla ^ { 2 } f _ { z } ( \beta _ { z } ) \preceq \varsigma _ { \operatorname* { m a x } } ( \pmb { Q } _ { z } ) \pmb { I } } \end{array}
$$

From Eq. (31), $\varsigma _ { \mathrm { m a x } } ( Q _ { z } ) I - \nabla ^ { 2 } f _ { z } ( \beta _ { z } )$ is positive semidefinite. Therefore, for all eigenvalue of $\nabla ^ { 2 } f _ { z } ( \beta _ { z } )$ , it holds

$$
\varsigma _ { k } ( \nabla ^ { 2 } f _ { z } ( \beta _ { z } ) ) \leq \varsigma _ { \mathrm { m a x } } ( Q _ { z } )
$$

For the largest eigenvalue, it also holds:

$$
\begin{array} { r } { \varsigma _ { \operatorname* { m a x } } \bigl ( \nabla ^ { 2 } f _ { z } ( \beta _ { z } ) \bigr ) \leq \varsigma _ { \operatorname* { m a x } } ( Q _ { z } ) . } \end{array}\tag{32}
$$

Here, $\nabla ^ { 2 } f _ { z } ( \beta _ { z } )$ is an symmetric positive-semidefinite matrix,

$$
\begin{array} { r } { \varsigma _ { \operatorname* { m a x } } ( \nabla ^ { 2 } f _ { z } ( \beta _ { z } ) ) = \| \nabla ^ { 2 } f _ { z } ( \beta _ { z } ) \| _ { 2 } . } \end{array}
$$

Then, Eq. (32) can be expressed as

$$
\| \nabla ^ { 2 } f _ { z } ( \beta _ { z } ) \| _ { 2 } \leq \varsigma _ { \mathrm { m a x } } ( Q _ { z } ) = \mu _ { z } .\tag{33}
$$

Next, for any $\beta _ { z } , \beta _ { z } ^ { \prime } \in \mathbb { R } ^ { p }$ , by the mean-value theorem,

$$
\nabla f _ { z } ( \beta _ { z } ) - \nabla f _ { z } ( \beta _ { z } ^ { \prime } ) = \int _ { 0 } ^ { 1 } \nabla ^ { 2 } f _ { z } \big ( \beta _ { z } ^ { \prime } + c ( \beta _ { z } - \beta _ { z } ^ { \prime } ) \big ) ( \beta _ { z } - \beta _ { z } ^ { \prime } ) d c
$$

Take the norm

$$
\begin{array} { r l } & { \quad \| \nabla f _ { z } ( \beta _ { z } ) - \nabla f _ { z } ( \beta _ { z } ^ { \prime } ) \| _ { 2 } \leq \displaystyle \int _ { 0 } ^ { 1 } \| \nabla ^ { 2 } f _ { z } ( \beta _ { z } ^ { \prime } + c ( \beta _ { z } - \beta _ { z } ^ { \prime } ) ) \| _ { 2 } \| \beta _ { z } - \beta _ { z } ^ { \prime } \| _ { 2 } d c } \\ & { \Longrightarrow \| \nabla f _ { z } ( \beta _ { z } ) - \nabla f _ { z } ( \beta _ { z } ^ { \prime } ) \| _ { 2 } \leq \displaystyle \left( \int _ { 0 } ^ { 1 } \| \nabla ^ { 2 } f _ { z } ( \beta _ { z } ^ { \prime } + c ( \beta _ { z } - \beta _ { z } ^ { \prime } ) ) \| _ { 2 } d c \right) \| \beta _ { z } - \beta _ { z } ^ { \prime } \| _ { 2 } d c } \end{array}\tag{34}
$$

From Eq. (33),

$$
\int _ { 0 } ^ { 1 } \| \nabla ^ { 2 } f _ { z } ( \beta _ { z } ^ { \prime } + c ( \beta _ { z } - \beta _ { z } ^ { \prime } ) ) \| _ { 2 } d c \leq \int _ { 0 } ^ { 1 } \mu _ { z } d c = \mu _ { z } ,
$$

Then, Eq. (34) can be expressed as

$$
\| \nabla f _ { z } ( \beta _ { z } ) - \nabla f _ { z } ( \beta _ { z } ^ { \prime } ) \| _ { 2 } \leq \mu _ { z } \| \beta _ { z } - \beta _ { z } ^ { \prime } \| _ { 2 }\tag{35}
$$

Therefore, the gradient at institution z is $\mu _ { z }$ Lipschitz continuous. For $\beta ,$

$$
f ( \beta ) = \frac { 1 } { Z } \sum _ { z = 1 } ^ { Z } f _ { z } ( \beta _ { z } ) .
$$

Then,

$$
\nabla f ( { \boldsymbol { \beta } } ) = \frac { 1 } { Z } \nabla f _ { z } ( { \boldsymbol { \beta } } _ { z } ) , ~ ( z = 1 , 2 , \cdot \cdot \cdot , Z )
$$

For $\beta , \beta ^ { \prime }$

$$
\| \nabla f ( \beta ) - \nabla f ( \beta ^ { \prime } ) \| _ { 2 } ^ { 2 } = \frac { 1 } { Z ^ { 2 } } \sum _ { z = 1 } ^ { Z } \| \nabla f ( \beta _ { z } ) - \nabla f ( \beta _ { z } ^ { \prime } ) \| _ { 2 } ^ { 2 }
$$

Therefore, with Eq. (35), we have,

$$
\begin{array} { r l } { \displaystyle \| \nabla f ( \boldsymbol { \beta } ) - \nabla f ( \boldsymbol { \beta } ^ { \prime } ) \| _ { 2 } ^ { 2 } = \frac { 1 } { Z ^ { 2 } } \sum _ { z = 1 } ^ { Z } \| \nabla f ( \boldsymbol { \beta } _ { z } ) - \nabla f ( \boldsymbol { \beta } _ { z } ^ { \prime } ) \| _ { 2 } ^ { 2 } } & { } \\ { \displaystyle \le \frac { 1 } { Z ^ { 2 } } \sum _ { z = 1 } ^ { Z } \mu _ { z } ^ { 2 } \| \beta _ { z } - \beta _ { z } ^ { \prime } \| _ { 2 } ^ { 2 } } & { } \\ { \displaystyle \le \frac { \mu _ { \mathrm { m a x } } ^ { 2 } } { Z ^ { 2 } } \sum _ { z = 1 } ^ { Z } \| \beta _ { z } - \beta _ { z } ^ { \prime } \| _ { 2 } ^ { 2 } } & { } \\ { \displaystyle = \frac { \mu _ { \mathrm { m a x } } ^ { 2 } } { Z ^ { 2 } } \| \beta - \beta ^ { \prime } \| _ { 2 } ^ { 2 } . } \end{array}
$$

where $\mu _ { \mathrm { m a x } } = \mathrm { m a x } _ { 1 \leq z \leq Z } \mu _ { z }$ . From above,

$$
\| \nabla f ( \beta ) - \nabla f ( \beta ^ { \prime } ) \| _ { 2 } \leq \frac { \mu _ { \operatorname* { m a x } } } { Z } \| \beta - \beta ^ { \prime } \| _ { 2 } \leq \mu _ { \operatorname* { m a x } } \| \beta - \beta ^ { \prime } \| _ { 2 } .
$$

Thus, we have $L = \operatorname* { m a x } _ { z } ( \mu _ { z } )$

## B Proof of Theorem 2

Proof. We prove that the following Eq. (17) is the majorizing function of GrAUC-PFL to update $\beta ,$ , and demonstrate that Theorem 2 holds when $\begin{array} { r } { r > \bar { \rho \tau } \varsigma _ { \mathrm { m a x } } ( \Omega ^ { \top } \Omega ) + \mathrm { m a x } ( \frac { \tau \mu } { 2 } , \bar { 1 } ) } \end{array}$ , which is

$$
\mathcal { L } _ { \rho } ( \boldsymbol { \beta } ^ { ( t + 1 ) } , \boldsymbol { \delta } ^ { ( t ) } , \boldsymbol { \gamma } ^ { ( t ) } ) - \mathcal { L } _ { \rho } ( \boldsymbol { \beta } ^ { ( t ) } , \boldsymbol { \delta } ^ { ( t ) } , \boldsymbol { \gamma } ^ { ( t ) } ) \leq - \biggl [ \frac { \zeta _ { \operatorname* { m i n } } ( H ) } { \tau } + \frac { \rho \cdot \zeta _ { \operatorname* { m i n } } ( \Omega ^ { \top } \Omega ) } { 2 } - \frac { \mu } { 2 } \biggr ] \| \boldsymbol { \beta } ^ { ( t + 1 ) } - \boldsymbol { \beta } ^ { ( t ) } \| _ { 2 } ^ { 2 } .
$$

First of all, differentiate Eq. (16) with respect to $\beta$ and set the result equal to zero.

$$
\nabla \tilde { \mathcal { L } } _ { \rho } ( \beta ; \beta ^ { ( t ) } , \delta ^ { ( t ) } , \gamma ^ { ( t ) } ) = \nabla f ( \beta ^ { ( t ) } ) + \frac { 1 } { \tau } \pmb { H } ( \beta - \beta ^ { ( t ) } ) + \Omega ^ { \top } \gamma ^ { ( t ) } + \rho \Omega ^ { \top } ( \Omega \beta - \delta ^ { ( t ) } ) = \mathbf { 0 } .\tag{36}
$$

Substitute $\beta ^ { ( t + 1 ) }$ into $\beta$ in Eq. (36)

$$
\nabla f ( { \boldsymbol { \beta } } ^ { ( t ) } ) - \frac { 1 } { \tau } { \boldsymbol { H } } ( { \boldsymbol { \beta } } ^ { ( t ) } - { \boldsymbol { \beta } } ^ { ( t + 1 ) } ) + { \boldsymbol { \Omega } } ^ { \top } { \boldsymbol { \gamma } } ^ { ( t ) } + \rho { \boldsymbol { \Omega } } ^ { \top } ( { \boldsymbol { \Omega } } { \boldsymbol { \beta } } ^ { ( t + 1 ) } - { \boldsymbol { \delta } } ^ { ( t ) } ) = \mathbf { 0 } .\tag{37}
$$

Apply $( \beta ^ { ( t ) } - \beta ^ { ( t + 1 ) } ) ^ { \top }$ to Eq. (37) from the left:

$$
0 = ( \beta ^ { ( t ) } - \beta ^ { ( t + 1 ) } ) ^ { \top } \left[ \nabla f ( \beta ^ { ( t ) } ) - \frac { 1 } { \tau } { \cal H } ( \beta ^ { ( t ) } - \beta ^ { ( t + 1 ) } ) + \Omega ^ { \top } \gamma ^ { ( t ) } + \rho \Omega ^ { \top } ( \Omega \beta ^ { ( t + 1 ) } - \delta ^ { ( t ) } ) \right] .\tag{38}
$$

From Lemma 1, if the smooth function $f : \mathbb { R } ^ { p } \mapsto$ R has Lipschitz continuous gradient, the following inequality holds:

$$
f ( \beta ^ { ( t + 1 ) } ) \leq f ( \beta ^ { ( t ) } ) + \nabla f ( \beta ^ { ( t ) } ) ^ { \top } ( \beta ^ { ( t + 1 ) } - \beta ^ { ( t ) } ) + \frac { \mu } { 2 } \| \beta ^ { ( t + 1 ) } - \beta ^ { ( t ) } \| _ { 2 } ^ { 2 }
$$

$$
f ( { \boldsymbol { \beta } } ^ { ( t ) } ) - f ( { \boldsymbol { \beta } } ^ { ( t + 1 ) } ) \geq - \nabla f ( { \boldsymbol { \beta } } ^ { ( t ) } ) ^ { \top } ( { \boldsymbol { \beta } } ^ { ( t + 1 ) } - { \boldsymbol { \beta } } ^ { ( t ) } ) - \frac { \mu } { 2 } \| { \boldsymbol { \beta } } ^ { ( t + 1 ) } - { \boldsymbol { \beta } } ^ { ( t ) } \| _ { 2 } ^ { 2 }
$$

$$
\nabla f ( { \boldsymbol { \beta } } ^ { ( t ) } ) ^ { \top } ( { \boldsymbol { \beta } } ^ { ( t ) } - { \boldsymbol { \beta } } ^ { ( t + 1 ) } ) \leq f ( { \boldsymbol { \beta } } ^ { ( t ) } ) - f ( { \boldsymbol { \beta } } ^ { ( t + 1 ) } ) + \frac { \mu } { 2 } \| { \boldsymbol { \beta } } ^ { ( t + 1 ) } - { \boldsymbol { \beta } } ^ { ( t ) } \| _ { 2 } ^ { 2 }\tag{39}
$$

With Eq. (39), Eq. (38) can be derived the following inequality:

$$
\begin{array} { r l } & { \mathbf { 0 } = ( \beta ^ { ( t ) } - \beta ^ { ( t + 1 ) } ) ^ { \top } \Big [ \nabla f ( \beta ^ { ( t ) } ) - \frac { 1 } { \tau } H ( \beta ^ { ( t ) } - \beta ^ { ( t + 1 ) } ) + \Omega ^ { \top } \gamma ^ { ( t ) } + \rho \Omega ^ { \top } ( \Omega \beta ^ { ( t + 1 ) } - \delta ^ { ( t ) } ) \Big ] } \\ & { \quad \le f ( \beta ^ { ( t ) } ) - f ( \beta ^ { ( t + 1 ) } ) + \frac { \mu } { 2 } \| \beta ^ { ( t ) } - \beta ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \frac { 1 } { \tau } \| \beta ^ { ( t ) } - \beta ^ { ( t + 1 ) } \| _ { H } ^ { 2 } } \\ & { \qquad + ( \beta ^ { ( t ) } - \beta ^ { ( t + 1 ) } ) ^ { \top } \Omega ^ { \top } \gamma ^ { ( t ) } + \rho ( \beta ^ { ( t ) } - \beta ^ { ( t + 1 ) } ) ^ { \top } \Omega ^ { \top } ( \Omega \beta ^ { ( t + 1 ) } - \delta ^ { ( t ) } ) } \\ & { \quad = f ( \beta ^ { ( t ) } ) - f ( \beta ^ { ( t + 1 ) } ) + \frac { \mu } { 2 } \| \beta ^ { ( t ) } - \beta ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \frac { 1 } { \tau } \| \beta ^ { ( t ) } - \beta ^ { ( t + 1 ) } \| _ { H } ^ { 2 } } \\ & { \qquad + \gamma ^ { ( t ) \top } ( \Omega \beta ^ { ( t ) } - \Omega \beta ^ { ( t + 1 ) } ) ^ { \top } + \rho ( \Omega \beta ^ { ( t ) } - \Omega \beta ^ { ( t + 1 ) } ) ^ { \top } ( \Omega \beta ^ { ( t + 1 ) } - \delta ^ { ( t ) } ) } \end{array}\tag{40}
$$

where $\| \pmb { a } \| _ { H } = \pmb { a } ^ { \top } \pmb { a }$ . Next, as for the fifth term of Eq. (40),

$$
\begin{array} { r l } & { ~ \gamma ^ { ( t ) } ( \Omega \beta ^ { ( t ) } - \Omega \beta ^ { ( t + 1 ) } ) + \gamma ^ { ( t ) \top } \delta ^ { ( t ) } - \gamma ^ { ( t ) \top } \delta ^ { ( t ) } } \\ & { = \gamma ^ { ( t ) } ( \Omega \beta ^ { ( t ) } - \delta ^ { ( t ) } ) - \gamma ^ { ( t ) } ( \Omega \beta ^ { ( t + 1 ) } - \delta ^ { ( t ) } ) } \end{array}\tag{41}
$$

Regarding the sixth term of Eq. (40), we use the following vector identity in the same manner in Liu et al. [2025]:

$$
( \pmb { a } - \pmb { b } ) ^ { \top } ( \pmb { b } - \pmb { c } ) = \frac { 1 } { 2 } ( \| \pmb { a } - \pmb { c } \| _ { 2 } ^ { 2 } - \| \pmb { a } - \pmb { b } \| _ { 2 } ^ { 2 } - \| \pmb { b } - \pmb { c } \| _ { 2 } ^ { 2 } ) .
$$

Then, the sixth term of Eq. (40) can be rewritten as

$$
\begin{array} { l } { \displaystyle \rho ( \Omega \beta ^ { ( t ) } - \Omega \beta ^ { ( t + 1 ) } ) ^ { \top } ( \Omega \beta ^ { ( t + 1 ) } - \delta ^ { ( t ) } ) } \\ { \displaystyle = \frac { \rho } { 2 } \big ( \| \Omega \beta ^ { ( t ) } - \delta ^ { ( t ) } \| _ { 2 } ^ { 2 } - \| \Omega \beta ^ { ( t ) } - \Omega \beta ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \Omega \beta ^ { ( t + 1 ) } - \Omega \delta ^ { ( t ) } \| _ { 2 } ^ { 2 } \big ) . } \end{array}\tag{42}
$$

With Eq. (41) and Eq. (42), Eq. (40) can be expressed as follows:

$$
\begin{array} { l } { { \displaystyle { \bf 0 } \leq f ( \boldsymbol { \beta } ^ { ( t ) } ) - f ( \boldsymbol { \beta } ^ { ( t + 1 ) } ) + \frac { \mu } { 2 } \| \boldsymbol { \beta } ^ { ( t ) } - \boldsymbol { \beta } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \frac { 1 } { \tau } \| \boldsymbol { \beta } ^ { ( t ) } - \boldsymbol { \beta } ^ { ( t + 1 ) } \| _ { H } ^ { 2 } } } \\ { { \displaystyle \quad \quad + \gamma ^ { ( t ) } ( \Omega \boldsymbol { \beta } ^ { ( t ) } - \delta ^ { ( t ) } ) - \gamma ^ { ( t ) } ( \Omega \boldsymbol { \beta } ^ { ( t + 1 ) } - \delta ^ { ( t ) } ) } } \\ { { \displaystyle \quad \quad + \frac { \rho } { 2 } \| \Omega \boldsymbol { \beta } ^ { ( t ) } - \delta ^ { ( t ) } \| _ { 2 } ^ { 2 } - \frac { \rho } { 2 } \| \Omega \boldsymbol { \beta } ^ { ( t ) } - \Omega \boldsymbol { \beta } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \frac { \rho } { 2 } \| \Omega \boldsymbol { \beta } ^ { ( t + 1 ) } - \delta ^ { ( t ) } \| _ { 2 } ^ { 2 } } } \end{array}\tag{43}
$$

Here, the first, fifth, and seventh term of the right-hand side in Eq. (43) express the Lagrangian function $\mathcal { L } _ { \rho } ( \beta ^ { ( t ) } , \delta ^ { ( t ) } , \gamma ^ { ( t ) } )$ , and the second, sixth, and ninth term in Eq. (43) is the Lagrangian function $\bar { \mathcal { L } _ { \rho } ( \beta ^ { ( t + 1 ) } , \delta ^ { ( t ) } , \gamma ^ { ( t ) } ) }$ respectively. Therefore,

$$
\mathbf { 0 } \leq \mathcal { L } _ { \rho } ( \beta ^ { ( t ) } , \delta ^ { ( t ) } , \gamma ^ { ( t ) } ) - \mathcal { L } _ { \rho } ( \beta ^ { ( t + 1 ) } , \delta ^ { ( t ) } , \gamma ^ { ( t ) } ) + \frac { \mu } { 2 } \| \beta ^ { ( t ) } - \beta ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \frac { 1 } { \tau } \| \beta ^ { ( t ) } - \beta ^ { ( t + 1 ) } \| _ { H } ^ { 2 } - \frac { \rho } { 2 } \| \Omega \beta ^ { ( t ) } - \Omega \beta ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 }
$$

$$
\mathcal { L } _ { \rho } ( \beta ^ { ( t + 1 ) } , \delta ^ { ( t ) } , \gamma ^ { ( t ) } ) - \mathcal { L } _ { \rho } ( \beta ^ { ( t ) } , \delta ^ { ( t ) } , \gamma ^ { ( t ) } ) \leq \frac { \mu } { 2 } \| \beta ^ { ( t + 1 ) } - \beta ^ { ( t ) } \| _ { 2 } ^ { 2 } - \frac { 1 } { \tau } \| \beta ^ { ( t + 1 ) } - \beta ^ { ( t ) } \| _ { H } ^ { 2 } - \frac { \rho } { 2 } \| \beta ^ { ( t + 1 ) } - \beta ^ { ( t ) } \| _ { \Omega ^ { \gamma } \Omega }\tag{44}
$$

To show that the expression (44) is monotonically decreasing, the right-hand side should be simplified to the form $- c \| \beta ^ { ( t + 1 ) } - \beta ^ { ( t ) } \| _ { 2 } ^ { 2 }$ . By the Rayleigh quotient theorem for a symmetric matrix M,

$$
\varsigma _ { \mathrm { m i n } } ( M ) \leq \frac { { { x ^ { * } } ^ { \top } } M { { x ^ { * } } } } { \| { { x ^ { * } } } \| _ { 2 } ^ { 2 } } \leq \varsigma _ { \mathrm { m a x } } ( M ) , \quad \forall { { x ^ { * } } } \neq { \mathbf { 0 } }
$$

$$
\operatorname { S m i n } ( M ) \| x ^ { * } \| _ { 2 } ^ { 2 } \leq x ^ { * } ^ { \top } M x ^ { * } \leq \varsigma _ { \operatorname* { m a x } } ( M ) \| x ^ { * } \| _ { 2 } ^ { 2 }\tag{45}
$$

Applying Eq. (45) to the second and third terms of the right-hand side in Eq. (44),respectively. When $\Delta : = \beta ^ { ( t + 1 ) } - \beta ^ { ( t ) }$

$$
\begin{array} { r } { \| \boldsymbol { \Delta } \| _ { H } ^ { 2 } \geq \varsigma _ { \mathrm { m i n } } ( H ) \| \boldsymbol { \Delta } \| _ { 2 } ^ { 2 } , \quad \| \boldsymbol { \Delta } \| _ { \Omega ^ { \top } \Omega } ^ { 2 } \geq \varsigma _ { \mathrm { m i n } } ( \Omega ^ { \top } \Omega ) \| \boldsymbol { \Delta } \| _ { 2 } ^ { 2 } . } \end{array}
$$

Substituting the corresponding terms from Eq. (44) gives

$$
- \frac 1 \tau \| \Delta \| _ { H } ^ { 2 } \leq - \frac 1 \tau \varsigma _ { \mathrm { m i n } } ( { H } ) \| \Delta \| _ { 2 } ^ { 2 } , \quad \mathrm { a n d }\tag{46}
$$

$$
- \frac { \rho } { 2 } \| \Delta \| _ { \Omega ^ { \top } \Omega } ^ { 2 } \leq - \frac { \rho } { 2 } \varsigma _ { \mathrm { m i n } } ( \Omega ^ { \top } \Omega ) \| \Delta \| _ { 2 } ^ { 2 } .\tag{47}
$$

Therefore, the right-hand side of Eq. (44) can be rewritten with Eq. (46) and Eq. (47) as

$$
\mathcal { L } _ { \rho } ( \beta ^ { ( t + 1 ) } , \delta ^ { ( t ) } , \gamma ^ { ( t ) } ) - \mathcal { L } _ { \rho } ( \beta ^ { ( t ) } , \delta ^ { ( t ) } , \gamma ^ { ( t ) } ) \leq - \bigg [ \frac { \varsigma _ { \operatorname* { m i n } } ( H ) } { \tau } + \frac { \rho \varsigma _ { \operatorname* { m i n } } ( \Omega ^ { \top } \Omega ) } { 2 } - \frac { \mu } { 2 } \bigg ] \| \beta ^ { ( t + 1 ) } - \beta ^ { ( t ) } \| _ { 2 } ^ { 2 } .
$$

Eq. (17) ensures that the augmented Lagrangian function decreases at each step of updating $\beta ,$ since $\lVert \beta ^ { ( t + 1 ) } - \beta ^ { ( t ) } \rVert _ { 2 } ^ { 2 }$ is strictly positive under the above condition. This can be achieved when

$$
\frac { \varsigma _ { \mathrm { m i n } } ( H ) } { \tau } + \frac { \rho \cdot \varsigma _ { \mathrm { m i n } } ( \Omega ^ { \top } \Omega ) } { 2 } - \frac { \mu } { 2 } > 0 .\tag{48}
$$

As mentioned, H, defined in Eq. (14), is required to be a positive definite so that the constructed function defined in Eq. (17) is a valid upper-bounding function. To investigate the conditions under which H is positive definite, we consider the eigen decomposition of $\check { \Omega } ^ { \top } \Omega = Q \Lambda Q ^ { \top }$

$$
\begin{array} { r l } & { H = r I - \rho \tau \boldsymbol { \Omega } ^ { \top } \boldsymbol { \Omega } } \\ & { \quad = r I - \rho \tau \boldsymbol { Q } \boldsymbol { \Lambda } \boldsymbol { Q } ^ { \top } } \\ & { \quad = r \boldsymbol { Q } \boldsymbol { Q } ^ { \top } - \rho \tau \boldsymbol { Q } \boldsymbol { \Lambda } \boldsymbol { Q } ^ { \top } } \\ & { \quad = \boldsymbol { Q } ( r I - \rho \tau \boldsymbol { \Lambda } ) \boldsymbol { Q } ^ { \top } } \end{array}
$$

Here, $Q  { \mathrm { ~ \textrm ~ { ~ ~ } ~ } }  { \mathrm { i s } }  { \mathrm { ~ \textrm ~ { ~ ~ } ~ } }  { \mathrm { a n } }$ orthogonal matrix consisting of the eigenvectors of Ω and $\begin{array} { r l } { \pmb { \Lambda } } & { { } = } \end{array}$ $\mathrm { d i a g } ( _ { \mathsf { S } 1 } ( \pmb { \Omega } ^ { \top } \pmb { \Omega } ) , _ { \mathsf { S } 2 } ( \pmb { \Omega } ^ { \top } \pmb { \Omega } ) , \dots , _ { \mathsf { S } Z p } ( \pmb { \Omega } ^ { \top } \pmb { \Omega } ) )$ is a diagonal matrix consisting of the corresponding eigenvalues. For this, r ${ \boldsymbol { \mathit { l } } } \ - \ \rho \tau { \boldsymbol { \Lambda } }$ is a diagonal matrix, and its diagonal entries are given by $r I - \rho \tau { \bf A } =$ $\mathrm { d i a g } \big ( r \mathrm { ~ - ~ } \rho \tau _ { \Sigma _ { 1 } } ( \Omega ^ { \top } \Omega ) , \varsigma _ { 2 } ( \dot { \Omega } ^ { \top } \Omega ) , \dots , r \mathrm { ~ - ~ } \rho \tau _ { \Sigma _ { Z p } } ( \Omega ^ { \top } \Omega ) \big )$ $\Omega ^ { \top } \Omega$ is positive semidefinite, all its eigenvalues are nonnegative. Therefore, the eigenvalues of H are given by

$$
\varsigma _ { k ^ { * } } ( H ) = r - \rho \tau \varsigma _ { k ^ { * } } ( \Omega ^ { \top } \Omega ) .\tag{49}
$$

The condition that H is positive definite is equivalent to the condition that $\varsigma _ { k ^ { * } } ( H )$ is greater than 0 for some $k ^ { * }$ Therefore, H becomes a positive definite matrix by setting $r > \rho \tau \varsigma _ { k ^ { * } } ( \Omega ^ { \top } \Omega )$ . In this case, r represents the minimum value required for H to be a positive definite matrix. This condition ensures the convexity of the surrogate function. From $\mathrm { E q . ~ } ( 4 9 ) , r - \rho \tau \varsigma _ { k ^ { * } } ( \Omega ^ { \top } \Omega )$ is decreasing in $\varsigma _ { k ^ { * } } ( \Omega ^ { \top } \Omega )$ . Therefore, $\varsigma _ { \mathrm { m i n } } ( H ) = r - \rho \tau \varsigma _ { \mathrm { m a x } } ( \Omega ^ { \top } \Omega )$ . Substituting $r - \rho \tau \varsigma _ { \mathrm { m a x } } ( \Omega ^ { \top } \Omega )$ into Eq. (48):

$$
\frac { 1 } { \tau } \big ( r - \rho \tau \varsigma _ { \mathrm { m a x } } ( \Omega ^ { \top } \Omega ) \big ) + \frac { \rho } { 2 } \varsigma _ { \mathrm { m i n } } ( \Omega ^ { \top } \Omega ) - \frac { \mu } { 2 } > 0
$$

$$
\frac { r } { \tau } > \rho \varsigma _ { \mathrm { m a x } } ( \Omega ^ { \top } \Omega ) - \frac { \rho } { 2 } \varsigma _ { \mathrm { m i n } } ( \Omega ^ { \top } \Omega ) + \frac { \mu } { 2 }
$$

$$
r > \rho \tau _ { \mathrm { { S m a x } } } ( \Omega ^ { \top } \Omega ) - \frac { \rho \tau } { 2 } \varsigma _ { \mathrm { { m i n } } } ( \Omega ^ { \top } \Omega ) + \frac { \tau \mu } { 2 }
$$

Here, since $\varsigma _ { \mathrm { m i n } } ( \Omega ^ { \top } \Omega )$ , we have $- { \frac { \rho } { 2 } } \varsigma _ { \mathrm { m i n } } ( \Omega ^ { \top } \Omega ) ~ \leq ~ 0$ . Therefore, a sufficient condition is given by $r \_ { \mathrm { ~ ~ } } >$ $\begin{array} { r } { \rho \tau \varsigma _ { \mathrm { m a x } } ( \Omega ^ { \top } \Omega ) + \frac { \tau \mu } { 2 } } \end{array}$ . Furthermore, to ensure the positive definite of H, Liu et al. [2025] sets r as

$$
r > \rho \tau \varsigma _ { \mathrm { m a x } } ( \Omega ^ { \top } \Omega ) + \operatorname* { m a x } \left( \frac { \tau \mu } { 2 } , 1 \right) .
$$

## C Results of numerical simulations

![](images/a758f988da0ea9e6ac41e39142299dc5258db37f697b93e238a2825f1b0935b8.jpg)  
Figure 7: The results of simulation 2 in that each institution owns different sample size from $N _ { z , t r a i n } ~ =$ {30, 50, 100, 200, 300}.

![](images/3a2e43fb36e553ed1d48bc98e3fd7d14e5e481b24160bd834e3dfe065a1e1372.jpg)  
Figure 8: The results of simulation 3 in $N _ { z , t r a i n } = 5 0 $

![](images/465158a8e1cfcb8dce3c0c2ada131d942a2edb5b52cb91292d30660bb43ca88f.jpg)  
Figure 9: The results of simulation 3 in $N _ { z , t r a i n } = 3 0 0 .$

![](images/db1a06a2764e2a8d7a58e2f72e019bf3642434dff7e571910e5e509c72d1e9d2.jpg)  
Figure 10: The results of simulation 4. The plots in the left column are results at $N _ { z , t r a i n } = 5 0$ at each institution and those in the right column are at $N _ { z , t r a i n } = 3 0 0$