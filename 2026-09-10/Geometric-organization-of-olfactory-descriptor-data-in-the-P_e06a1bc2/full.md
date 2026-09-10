# Geometric organization of olfactory descriptor data in the Poincaré disk

Aniss Aiman Medbouhi Department of Robotics, Perception and Learning School of Electrical Engineering and Computer Science KTH Royal Institute of Technology, Sweden

Farzaneh Taleb Department of Robotics, Perception and Learning School of Electrical Engineering and Computer Science KTH Royal Institute of Technology, Sweden

Giovanni Luca Marchetti Department of Mathematics School of Engineering Sciences KTH Royal Institute of Technology, Sweden

Danica Kragic Department of Robotics, Perception and Learning School of Electrical Engineering and Computer Science KTH Royal Institute of Technology, Sweden

ORCID iDs

Aniss Aiman Medbouhi: 000-0002-6649-3325

Farzaneh Taleb: 0000-0003-4482-1460

Giovanni Luca Marchetti 0009-0004-8248-229X

Danica Kragic: 0000-0003-2965-2953

## Correspondence to be sent to

Address: Lindstedtsvägen 24, 114 28 Stockholm, Sweden. Email: medbouhi@kth.se

## Abstract

Odor quality is commonly represented using high dimensional descriptor profiles, yet their low dimensional organization remains unclear. We investigated whether a two-dimensional hyperbolic embedding can provide an interpretable representation of this structure. We applied hyperbolic metric multidimensional scaling to two complementary datasets: 480 Sagar rating profiles from three participants rating 160 odorants on 15 continuous descriptors, and 4983 GoodScents–Leffingwell molecules annotated with 138 binary descriptors. The embeddings substantially preserved pairwise descriptor distances, supporting subsequent analyses of radial and angular organization. In Sagar, rating profile entropy was strongly and negatively associated with hyperbolic radius (Pearson r = −0.77 ± 0.05), with diffuse profiles closer to the center and concentrated profiles closer to the boundary. This radial organization emerged primarily at the level of the full descriptor profile, rather than any individual descriptor, and remained robust across alternative descriptor representations, participant specific analyses, and averaged ratings. Sweet, musky, fruity, and pleasantness showed the strongest directional trends (mean $R ^ { 2 } = 0 . 4 0 { \bf t o } 0 . 5 1 )$ . In GoodScents–Leffingwell, active label entropy, reflecting descriptor multiplicity, increased with radius $( r = 0 . 8 7 \pm 0 . 0 3 )$ , whereas orthogonalized descriptor entropy, reflecting spread across orthogonal modes, decreased with radius $( r = - 0 . 8 8 \pm 0 . 0 1 )$ ). Related binary descriptors occupied coherent localized high-density regions. These findings reveal complementary radial and angular organization in the hyperbolic representation of olfactory descriptor data. They support hyperbolic mapping as an interpretable descriptive framework in which radius summarizes global profile properties, while the angular component captures continuous descriptor gradients and categorical organization.

Keywords: olfactory perception, hyperbolic geometry, dimensionality reduction.

## 1 Introduction

Human odor perception is challenging to organize within a simple coordinate system. While substantial progress has been made in characterizing visual and auditory perception through mathematical models and structured representations (Sucholutsky et al., 2023; Brohan et al., 2023; Du et al., 2022; Ganis et al., 2004; Friederici, 2012), olfaction lacks a comparable theoretical framework. Unlike vision, where color perception has been systematically mapped through the Commission Internationale de l’Éclairage (1931) color spaces, or audition, where auditory signals can be systematically represented in the frequency domain (Evans, 1977), no comparably established and widely accepted mapping exists for the olfactory perceptual space. Odor quality does not vary along a single dominant physical continuum, and similar odor percepts can arise from chemically diverse molecules. Perceptual descriptions also depend on the odorants presented and their concentration, the response task, the available vocabulary, and individual and cultural differences in perceptual and verbal strategies (Kaeppler & Mueller, 2013; Doty, 2025). Consequently, in the present study, we treat an olfactory perceptual space as a representation of similarities and differences among odor percepts or descriptor profiles, rather than as a direct and universal mapping from molecular structure to subjective experience.

A long tradition of olfactory research has used similarity judgments, sorting tasks, descriptor ratings, factor analysis, and multidimensional scaling to characterize these relationships. Such studies have identified broad hedonic and semantic trends, but they have not produced a single agreed dimensional organization of odor quality (Schiffman, 1974; Madany Mamlouk & Martinetz, 2004; Koulakov et al., 2011; Magnasco et al., 2015; Meister, 2015). The resulting maps depend on the experimental setup, stimulus set, the descriptors, the participants, and the analysis method (Kaeppler & Mueller, 2013). A recent computational work has constructed learned odor representations from molecular and perceptual data, namely Principal Odor Map (Lee et al., 2023), while recent taxonomy based approaches have explicitly examined hierarchical relations among odor descriptors (Sajan et al., 2026). These developments provide powerful representations for prediction, but the geometric structure of descriptor based odor spaces remains poorly understood.

Most low dimensional representations of odor perception have been formulated in Euclidean space. Hyperbolic geometry offers an alternative representation in which the amount of available space increases exponentially with distance from the origin. This property makes hyperbolic spaces effective for representing data with branching or hierarchical organization (Sarkar, 2012; Nickel & Kiela, 2017; Klimovskaia et al., 2020; Zhou & Sharpee, 2021; Zhang et al., 2022). In olfaction, Zhou et al. (2018) reported that the statistics of natural odor mixtures and human perceptual descriptions were compatible with hyperbolic geometry. However, it remains unclear whether an interpretable two-dimensional hyperbolic representation can preserve relationships among odor descriptor profiles, retain established perceptual dimensions such as pleasantness (Crocker & Henderson, 1927; Khan et al., 2007; Koulakov et al., 2011; Snitz et al., 2013; Licon et al., 2018) as interpretable directions, and reveal how global profile properties are organized along its radial coordinate. It is also unknown whether comparable geometric organization appears across continuous participant ratings and large binary descriptor databases. Here, we investigate these questions by employing a hyperbolic version of metric multidimensional scaling (MDS) (Torgerson, 1952; Kruskal, 1964; Walter, 2004; Sala et al., 2018; Keller-Ressel & Nargang, 2020). This method represents each observation as a point in the two-dimensional Poincaré disk by minimizing differences between pairwise distances in the original descriptor space and the corresponding pairwise hyperbolic distances in the embedding. This choice is not intended as an estimate of the intrinsic dimensionality of olfactory perception, nor as evidence that neural olfactory representations are themselves two-dimensional or hyperbolic. Rather, the two-dimensional representation facilitates both visualization and interpretation by allowing center to boundary variation to be distinguished from angular organization.

We analyze two complementary datasets. The Sagar dataset (Sagar et al., 2023) contains continuous perceptual ratings for monomolecular odorants from three participants, allowing us to examine graded descriptor profiles, organization within individual participants, and participant averaged ratings. The GoodScents–Leffingwell dataset (Barsainyan et al., 2023) contains binary expert annotations for several thousand molecules and provides a larger scale representation of how odor descriptors are jointly assigned across molecules. Using these datasets, we first verify pairwise distance preservation and then address three main questions: (i) Are global properties of descriptor profiles associated with hyperbolic radius?; (ii) Do continuous descriptors show directional organization and binary descriptors occupy localized coherent regions?; (iii) Are the observed relationships robust across random initializations, alternative descriptor representations, and where available, individual and averaged ratings? Statistical significance is assessed using restricted permutation tests adapted to the structure of each dataset.

We previously presented preliminary analyses of the Sagar dataset in conference abstracts (Taleb et al., 2025; Medbouhi et al., 2025). These preliminary contributions used a hyperbolic contrastive learning objective optimized with Riemannian stochastic gradient descent and identified an initial association between descriptor profile entropy and hyperbolic radius. The present study substantially extends this datasetspecific observation into a broader and statistically validated framework for interpreting hyperbolic olfactory representations. We replace the contrastive objective with hyperbolic metric multidimensional scaling and Riemannian Adam optimization, which enables application to the substantially larger and binary GoodScents–Leffingwell dataset. We further introduce an explicit radial–angular decomposition, evaluate pairwise distance preservation, develop complementary hyperbolic visualizations of continuous descriptor directions and binary descriptor regions, and assess the resulting organization through restricted permutation tests and extensive robustness analyses.

The results reveal complementary radial and angular organization. In the Sagar data, rating profile entropy is strongly and negatively associated with radius, such that diffuse descriptor profiles are located closer to the center and more concentrated profiles closer to the boundary. This relationship is stronger than the radial association of any individual descriptor and persists across alternative descriptor representations, participant specific analyses, and averaged ratings. Several continuous descriptors, including sweet, musky, fruity, and pleasantness, show consistent directional trends. In the GoodScents–Leffingwell data, radius is also associated with entropy related properties of the descriptor profiles, although the direction and interpretation of these relationships depend on how entropy is defined. Related binary odor descriptors additionally occupy coherent regions of the disk. Together, these findings support two-dimensional hyperbolic mapping as a descriptive framework for separating global descriptor profile organization from descriptor specific gradients and categorica odor quality structure.

## 2 Materials and methods

## 2.1 Study overview

We first construct descriptor vectors for each observation in the two datasets. We then learn a two-dimensional hyperbolic embedding that preserves pairwise distances between these vectors. The learned representation is evaluated in two stages. First, we quantify how well hyperbolic distances preserve the input descriptor geometry. Second, we test whether interpretable variables are organized along the radial and angular components of the Poincaré disk. The radial analyses examine entropy and continuous descriptor ratings. The angular analyses use tangent space directional trends for continuous ratings and hyperbolic density regions for binary annotations.

## 2.2 Data sources

Sagar. We used the publicly available Sagar dataset (Sagar et al., 2023), obtained from the Pyrfume repository (Hamel et al., 2024). The original study collected perceptual ratings for 160 unique monomolecular odor stimuli per subject, across S = 3 subjects. In our union dataset, each subject contributes 160 subject–odorant observations, giving N = 480 observations in total. Because the odor sets are not identical across subjects, these 480 observations correspond to 195 unique Compound Identifiers (CIDs): 125 CIDs are observed for all three subjects, 35 CIDs are observed only for subject 1, and 35 CIDs are observed for subjects 2 and 3.

Each subject rated odors using 18 perceptual descriptors. 15 descriptors are common to all subjects: intensity, pleasantness, fishy, burnt, sour, decayed, musky, fruity, sweaty, cool, floral, sweet, warm, bakery, and spicy. The remaining descriptors are subject-specific and are therefore excluded from our analysis to ensure a common descriptor space across subjects. All ratings were normalized to the range [−1,1], yielding descriptor vectors in [−1,1]<sup>15</sup>. The diversity of odorants and perceptual attributes makes this dataset well suited for investigating the human odor perception.

GoodScents–Leffingwell. We also used the GoodScents– Leffingwell (GSLF) dataset distributed with OpenPOM (Barsainyan et al., 2023), which combines odor annotations from the GoodScent (n.d.) company and Leffingwell & Associates (2001) databases following the curation procedure of Lee et al. (2022). The resulting dataset contains 4983 molecules annotated with 138 expert-defined odor descriptors, such as fruity, jasmin, and leathery. Unlike the Sagar dataset, these annotations are binary labels rather than continuous ratings: each descriptor is either present or absent for a given molecule.

Thus, each molecule is represented by a multi-label binary vector $b _ { i } \in \{ 0 , 1 \} ^ { 1 3 8 }$

Each molecule appears once in the dataset, so there is no subject-level or repeated-measures structure. This makes GSLF complementary to Sagar: Sagar provides continuous subject-specific perceptual ratings for a smaller set of odorants, whereas GSLF provides a larger-scale binary descriptor representation over thousands of molecules. We use GSLF to test whether radial entropy organization and descriptor-region structure also appear in a large expert annotated odor dataset.

## 2.3 Hyperbolic metric multidimensional scaling

Poincaré model. Hyperbolic geometry is a non Euclidean geometry with constant negative curvature. We employ the Poincaré disk model to embed perceptual descriptor data: the goal is to learn a two-dimensional hyperbolic representation that preserves pairwise input distances. The Poincaré disk $\mathbb { P } ^ { 2 }$ represents the two-dimensional hyperbolic space inside the open Euclidean unit disk,

$$
\mathbb { P } ^ { 2 } = \{ z \in \mathbb { R } ^ { 2 } : \| z \| < 1 \} ,
$$

where $\| \cdot \|$ denotes the Euclidean norm. Although the disk is drawn in Euclidean coordinates, distances are measured using the hyperbolic metric. For points $z _ { 1 } , z _ { 2 } \in \mathbb { P } ^ { 2 }$ , the hyperbolic distance is

$$
d _ { \mathbb { P } } ( z _ { 1 } , z _ { 2 } ) = \operatorname { a r c o s h } \left( 1 + { \frac { 2 \lVert z _ { 1 } - z _ { 2 } \rVert ^ { 2 } } { ( 1 - \lVert z _ { 1 } \rVert ^ { 2 } ) ( 1 - \lVert z _ { 2 } \rVert ^ { 2 } ) } } \right) .
$$

Distances therefore expand near the boundary of the disk. This allows many mutually separated observations to be represented at large radii, which is useful when the data contain branching or hierarchical structure.

Radial and angular organization. The two-dimensional representation also provides a useful descriptive decomposition. The hyperbolic radius $d _ { \mathbb { P } } ( z , 0 )$ measures center to boundary position, whereas angular position describes direction around the disk. Neither coordinate has an intrinsic perceptual meaning. We establish empirically their interpretation by testing associations with entropy and descriptor values. In addition, the absolute angular orientation can rotate or reflect across model initializations, so only relative directional and regional organization is interpreted.

Objective function. Given perceptual descriptor vectors $\{ x _ { i } \} _ { i = 1 } ^ { N }$ , we compute input distances $D _ { i j } ^ { \mathrm { i n } } = \| x _ { i } - x _ { j } \|$ . We learn points $\{ z _ { i } \} _ { i = 1 } ^ { N } \subset \mathbb { P } ^ { 2 }$ , called the embeddings, by minimizing the following loss:

$$
\mathcal { L } = \frac { 1 } { N ^ { 2 } } \sum _ { i , j = 1 } ^ { N } \left( D _ { i j } ^ { \mathrm { i n } } - D _ { i j } ^ { \mathrm { e m b } } \right) ^ { 2 } ,
$$

where $D _ { i j } ^ { \mathrm { e m b } } = d _ { \mathbb { P } } ( z _ { i } , z _ { j } )$ are the hyperbolic embedding distances. This is the metric multidimensional scaling principle expressed with hyperbolic distances (Torgerson, 1952; Kruskal, 1964; Sala et al., 2018; Keller-Ressel & Nargang, 2020).

Optimization. The embedding coordinates are initialized in the Poincaré disk via a hyperbolic analog of Gaussian distribution (Nagano et al., 2019), and optimized by minimizing the loss L with Riemannian Adam (Becigneul & Ganea, 2019). Full geometric expressions, optimization updates, numerical stability procedures and implementation details are provided in the Supplementary material A.

## 2.4 Evaluation of embedding quality and geometric organization

We evaluate the learned embeddings in two steps. First, we use distance preservation as a quality control measure to verify that the hyperbolic metric MDS optimization has produced embeddings that preserve the input descriptor geometry. Second, we analyze the structure of the learned embeddings by asking how entropy, continuous descriptor ratings, and binary descriptor labels are organized in the Poincaré disk. These analyses are divided into radial organization, and angular organization comprising directional trends for continuous descriptors and high density regions for binary descriptors.

Embedding quality. As a quality control step, we evaluate whether the learned hyperbolic embeddings preserve the pairwise geometry of the input descriptor space. This check is important because the subsequent radial and directional analyses are meaningful only if the embedding retains the main distance structure of the original data. For each training configuration, we compute the fixed input distance matrix $D ^ { i n }$ . For each random seed $m ,$ we compute the corresponding hyperbolic embedding distance matrix $D ^ { e m b , m }$ . We then vectorize the upper triangular entries of both matrices, excluding the diagonal, and compute Pearson and Spearman correlations between the two distance vectors. Pearson correlation measures linear agreement between input and embedding distances, whereas Spearman correlation measures preservation of the distance ranking. We report mean and standard deviation over random seeds. We use these correlations as embedding quality metrics, not as hypothesis tests, and therefore do not report analytic p-values for them.

## 2.4.1 Entropy

Entropy is a classical quantity in thermodynamics, statistical physics, and information theory, where it is used to quantify disorder, uncertainty, or the spread of a probability distribution. In order to evaluate and analyze the structure of the inferred hyperbolic embedding, we propose to employ entropy as a scalar summary of descriptor organization for each observation. The two datasets contain different types of descriptor values, so entropy has to be defined through a dataset specific probability vector. In the Sagar dataset, descriptors are continuous ratings, and entropy summarizes the spread of graded descriptor strengths. In the GSLF dataset, descriptors are binary annotations, and entropy summarizes the multiplicity of active odor labels. In robustness analyses, we also compute entropy after transforming descriptor vectors into an orthogonalized representation, which tests whether radial entropy organization depends on the orthogonal modes of variation.

For each observation i, we construct a probability vector

$$
p _ { i } = ( p _ { i , 1 } , \ldots , p _ { i , n } ) ,
$$

defined over the n descriptors, and compute

$$
H _ { i } = - \sum _ { k = 1 } ^ { n } p _ { i , k } { \log } p _ { i , k } .
$$

This common entropy formula is used for both datasets, but the construction of $p _ { i }$ differs according to the data type. This distinction is important because the resulting entropy measures answer related but not identical questions.

Rating profile entropy. For continuous descriptor ratings, as in the Sagar dataset, each observation is represented by a vector $x _ { i } \in \mathbb { R } ^ { n }$ , where $x _ { i , k }$ is the rating of descriptor k for observation i. Since these ratings can be positive or negative after normalization, we convert them into a probability distribution using a softmax:

$$
p _ { i , k } = \frac { \exp ( x _ { i , k } ) } { \sum _ { \ell = 1 } ^ { n } \exp ( x _ { i , \ell } ) } .
$$

We refer to the resulting $H _ { i }$ quantity as rating profile entropy. It is high when the continuous descriptor profile is diffuse across many descriptors, and low when the profile is concentrated on one or a few descriptors. Therefore, in the Sagar dataset, the radial analysis asks whether odors with diffuse or ambiguous continuous rating profiles are placed differently in the hyperbolic embedding from odors with more concentrated descriptor profiles.

Active label entropy. For binary descriptor annotations, as in the GSLF dataset, each molecule is represented by a vector $b _ { i } \in \{ 0 , 1 \} ^ { n }$ , where $b _ { i , k } = 1$ if descriptor k is assigned to molecule i, and $b _ { i , k } = 0$ otherwise. Applying the same softmax construction directly to the binary vector would give positive probability to inactive labels, because $\mathtt { a x p } ( 0 ) > 0$ . This would make absent descriptors contribute to the entropy, which is not the intended interpretation of a binary descriptor annotation. We therefore define the probability distribution only over the active descriptor set. Let

$$
A _ { i } = \{ k : b _ { i , k } = 1 \}
$$

be the set of active descriptors for molecule $i ,$ and let

$$
K _ { i } = \left| A _ { i } \right|
$$

be the number of active descriptors. For molecules with at least one active descriptor, we define

$$
p _ { i , k } = { \left\{ \begin{array} { l l } { 1 / K _ { i } , } & { { \mathsf { i f } } \ k \in A _ { i } , } \\ { 0 , } & { { \mathsf { o t h e r w i s e } } . } \end{array} \right. }
$$

With this definition, the entropy reduces to

$$
H _ { i } = - \sum _ { k = 1 } ^ { n } p _ { i , k } \log p _ { i , k } = \log K _ { i } .
$$

Molecules with no active descriptor have undefined entropy and are excluded from entropy based analyses. Thus, in the GSLF dataset, entropy measures the breadth or multiplicity of the expert descriptor profile. We refer to this quantity as active label entropy. It is high when many descriptors are assigned to a molecule, and low when only one or a few descriptors are assigned.

Importantly, active label entropy is not equivalent to the rating profile entropy used for continuous descriptor ratings. A molecule can have many active binary labels and therefore high active label entropy, while a hypothetical continuous rating profile over the same descriptors could still be concentrated on one or a few dominant qualities and therefore have low rating profile entropy. Thus, active label entropy measures descriptor multiplicity, whereas rating profile entropy measures the spread of graded descriptor strengths. The sign of the radius entropy relationship should therefore be interpreted relative to the entropy definition used in each dataset.

Orthogonalized descriptor entropy. For robustness analysis, we express the descriptor matrix in an orthogonal coordinate system, so that the resulting dimensions no longer correspond to individual descriptor ratings or labels but to orthogonal modes of variation in the descriptor data. We then construct $p _ { i }$ by applying a softmax, and refer to the resulting quantity as orthogonalized descriptor entropy. This entropy asks whether the transformed descriptor profile of an observa tion is balanced across several orthogonal modes of variation, or whether it is dominated by one or a few modes. A high orthogonalized descriptor entropy indicates a more diffuse profile across modes, whereas a low value indicates a more concentrated one. This entropy is interpreted as a robustness measure rather than as a direct measure of descriptors profile or multiplicity.

Overall, these entropy measures ask related but distinct questions. Rating profile entropy tests whether diffuse graded descriptor profiles are organized along the hyperbolic radius. Active label entropy tests whether molecules associated with many odor qualities are organized radially. Orthogonalized descriptor entropy tests whether radial entropy organization persists after replacing the original descriptor basis by orthogonal modes of variation.

## 2.4.2 Geometric organization.

Radial organization. We evaluate whether scalar quantities associated with observations are organized along the radial coordinate of the learned hyperbolic embedding. For each random seed $m = 1 , \ldots , M$ , let $z _ { i } ^ { ( m ) } \in \mathbb { P } ^ { 2 }$ denote the learned embedding of observation i. Its hyperbolic radius is defined as

$$
r _ { i } ^ { ( m ) } = d _ { \mathbb { P } } ( z _ { i } ^ { ( m ) } , 0 ) .
$$

Let y be a scalar variable associated with observation i. Depending on the analysis, $y _ { i }$ can be an entropy value, or a continuous descriptor rating. For each seed m, we compute the signed Pearson correlation

$$
c _ { m } ( y ) = \mathrm { c o r r } ( r ^ { ( m ) } , y ) ,
$$

where $r ^ { ( m ) }$ is the vector of radii and y is the vector of scalar values for the corresponding observations. We also compute the Spearman rank correlation as a non-parametric robustness measure. We report mean and standard deviation over random seeds.

In the Sagar union dataset, each observation i corresponds to a subject–odorant pair $( o , s )$ , with subject $s \in \{ 1 , 2 , 3 \}$ . In this case, the generic index i can be identified with $( o , s )$ . For entropy, the scalar value is $y _ { ( o , s ) } = H _ { ( o , s ) }$ . For descriptor k, the scalar value is $y _ { ( o , s ) } = x _ { ( o , s ) , k }$

In the GSLF dataset, each observation corresponds to one annotated molecule. For radial analysis, the scalar variable is an entropy value $y _ { i } = H _ { i }$

For the radial permutation test, the observed statistic associated with y is the mean Pearson correlation over seeds:

$$
T _ { \mathrm { r a d , o b s } } ( y ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } c _ { m } ( y ) .
$$

The sign of the radial correlation indicates whether the scalar quantity tends to increase or decrease toward the boundary.

Tangent space representation. To characterize directional variation in continuous descriptor ratings, we use the tangent space of the Poincaré disk at the origin. The tangent space $T _ { 0 } \mathbb { P } ^ { 2 }$ is a two-dimensional Euclidean vector space that provides a linear coordinate system centered at the origin of the disk. This allows standard linear regression to be applied to the embedded observations while retaining their radial and directional organization.

For an embedded point $z \in \mathbb { P } ^ { 2 }$ , its tangent space coordinate is obtained using the logarithmic map at the origin:

$$
\log _ { 0 } ( z ) = { \left\{ \begin{array} { l l } { \operatorname { a r t a n h } \left( \left\| z \right\| \right) { \frac { z } { \left\| z \right\| } } , } & { z \neq 0 , } \\ { 0 , } & { z = 0 . } \end{array} \right. }
$$

This transformation preserves the direction of the point from the origin while expressing its position in a linear coordinate system.

For visualization, a tangent vector $u \in T _ { 0 } \mathbb { P } ^ { 2 }$ can be mapped back to the Poincaré disk using the exponential map at the origin:

$$
\exp _ { 0 } ( u ) = { \left\{ \begin{array} { l l } { \operatorname { t a n h } ( \| u \| ) { \frac { u } { \| u \| } } , } & { u \neq 0 , } \\ { 0 , } & { u = 0 . } \end{array} \right. }
$$

The exponential and logarithmic maps are inverses of one another at the origin. General expressions and additional geometric operations used during optimization are provided in the Supplementary material A.

Angular organization of continuous descriptors: directional trends. For continuous descriptor ratings, we analyze directional trends in the learned Poincaré disk. This analysis is applied to the Sagar dataset, where descriptor values are graded ratings. While the radial analysis captures center to boundary variation, the directional analysis captures whether a descriptor changes primarily along a dominant angular direction in the embedding. We therefore fit, for each descriptor, a best-fitting plane over the tangent-space coordinates of the embedded points, and consider the direction in which this plane increases most steeply as a global summary of the descriptor’s directional trend.

More formally, for each seed $m ,$ we first map the embedded points to the tangent space at the origin:

$$
u _ { ( o , s ) } ^ { ( m ) } = \log _ { 0 } \left( z _ { ( o , s ) } ^ { ( m ) } \right) .
$$

The tangent space $T _ { 0 } \mathbb { P } ^ { 2 }$ is identified with $\mathbb { R } ^ { 2 }$ . For each descriptor $k ,$ we fit a linear regression in this tangent space:

$$
\begin{array} { r } { x _ { ( o , s ) , k } \approx \mathsf { \mathsf { \mathsf { B } } } _ { k , m } ^ { T } u _ { ( o , s ) } ^ { ( m ) } + b _ { k , m } . } \end{array}
$$

Here, $x _ { o , s , k }$ is the value of descriptor k for odorant o and subject s. The parameters $\beta _ { k , m } \in \mathbb { R } ^ { 2 }$ and $b _ { k , m } \in \mathbb { R }$ are obtained by minimizing the squared error:

$$
\sum _ { ( o , s ) \in I } \left( x _ { ( o , s ) , k } - \beta _ { k , m } ^ { T } u _ { ( o , s ) } ^ { ( m ) } - b _ { k , m } \right) ^ { 2 } .
$$

The vector $\beta _ { k , m }$ gives the direction in the tangent space along which the fitted descriptor k increases most strongly under this global linear approximation. For visualization, we normalize this direction as

$$
\nu _ { k , m } = \frac { \beta _ { k , m } } { \| \beta _ { k , m } \| } ,
$$

and map the corresponding vector back to the Poincaré disk using the exponential map:

$$
\begin{array} { r } { \gamma _ { k , m } ( t ) = \exp _ { 0 } \left( t \nu _ { k , m } \right) . } \end{array}
$$

$\gamma _ { k , m } ( t )$ is represented as an arrow indicating the direction of steepest increase in the rating of descriptor k, and t only controls the displayed arrow length. Since the absolute orientation of the embedding can rotate or reflect across random seeds, these arrows are used only for visualization of representative embeddings. Their displayed lengths are arbitrary and do not represent the strength of the directional trend, which is quantified separately using the coefficient of determination $R ^ { 2 }$

For descriptor k and seed $m ,$ , let $\hat { x } _ { o , s , k } ^ { ( m ) }$ be the value predicted by the fitted linear model. We define

$$
R _ { k , m } ^ { 2 } = 1 - \frac { \sum _ { \left( o , s \right) \in I } { \left( x _ { \left( o , s \right) , k } - \hat { x } _ { \left( o , s \right) , k } ^ { \left( m \right) } \right) ^ { 2 } } } { \sum _ { \left( o , s \right) \in I } { \left( x _ { \left( o , s \right) , k } - \bar { x } _ { k } \right) ^ { 2 } } } .
$$

Here, $\bar { x } _ { k }$ is the mean value of descriptor k over all subject– odorant observations. A high value of the coefficient of determination $R _ { k , m } ^ { 2 }$ indicates that descriptor k is well summarized

by a global directional trend in the tangent-space representation. We report the mean and standard deviation of $R _ { k , m } ^ { 2 }$ over random seeds.

For the angular permutation test, the observed statistic for descriptor k is the mean directional $R ^ { 2 }$ over seeds:

$$
T _ { k , \mathrm { a n g , o b s } } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } R _ { k , m } ^ { 2 } .
$$

This directional analysis should be interpreted as a global first-order summary. It may not capture nonlinear, radial, or multi-cluster organization of a descriptor.

Permutation tests. For statistical assessment, we use permutation tests that keep the learned embedding fixed and repeatedly break the association between embedding positions and the scalar values being tested. The permutation scheme depends on the dataset structure and on the analysis setting.

For the Sagar dataset, the $N = 4 8 0$ subject–odorant observations are not fully independent, because some odorants are observed across multiple subjects. Standard analytic correlation p-values may therefore underestimate uncertainty. We instead employ restricted permutation tests, following the principle that permutations should preserve the dependence struc ture of the data (Winkler et al., 2014, 2015). The resulting null distribution describes how large the radial or directional statistic could be if the tested values were not systematically aligned with the embedding, while preserving selected aspects of the subject and odorant structure.

We propose two restricted permutation schemes for the Sagar union dataset, which test different null hypotheses. First, in the within-subject (WS) permutation, values are shuffled separately for each subject. Given a subject $s ,$ let

$$
I _ { s } = \{ o : ( o , s ) \in I \}
$$

be the set of odorants observed for subject s. For each permutation $b = 1 , \ldots , B$ , we draw a random permutation $\sigma _ { s } ^ { ( \dot { b } ) }$ of $I _ { s }$ and define

$$
y _ { ( o , s ) } ^ { ( b , \mathrm { W S } ) } = y _ { \left( \sigma _ { s } ^ { ( b ) } ( o ) , s \right) } .
$$

This permutation preserves each subject’s distribution of values, but breaks the match between values and embedding positions within that subject. The corresponding null hypothesis is that, within each subject, the scalar values are exchangeable across odorants and are not specifically aligned with the embedding. Thus, the within-subject permutation tests whether the observed effect holds within subjects rather than being driven only by subject-specific rating biases or offsets.

Second, in the odorant-block (OB) permutation, we shuffle whole odorant profiles. For each odorant $^ { O , }$ we define its observed subject pattern as

$$
P ( o ) = \{ s : ( o , s ) \in I \} .
$$

For each subject pattern $P ,$ let

$$
O _ { P } = \{ o : P ( o ) = P \}
$$

be the set of odorants with the same observed subject pattern. In the Sagar union dataset, the observed patterns are {1,2,3}, {1}, and {2,3}. For each pattern $P ,$ , we draw a random permutation $\sigma _ { P } ^ { ( b ) }$ of $O _ { P }$ and define

$$
y _ { ( o , s ) } ^ { ( b , \mathrm { O B } ) } = y _ { \left( \sigma _ { P ( o ) } ^ { ( b ) } ( o ) , s \right) } .
$$

This permutation preserves the subject profile of each odorant and the missingness structure of the repeated-measures data, but randomizes which odorant profile is attached to which embedding location. Intuitively, it is like swapping whole odorant ratings between compatible odorants: a rating observed for subjects {1,2,3} can only be swapped with another rating observed for subjects {1,2, 3}, and similarly for the other subject patterns. The corresponding null hypothesis is that odorantlevel profiles are exchangeable among odorants with the same observed subject pattern and are not systematically aligned with the embedding.

For the GSLF dataset, each observation corresponds to one molecule and each molecule appears once. Therefore, there is no subject-level or repeated-measures structure to preserve. We use a molecule-level permutation test, denoted $p _ { \mathrm { m o l } }$ . For each permutation $b = 1 , \ldots , B ,$ we draw a random permutation $\sigma ^ { ( b ) }$ of the molecule indices and define

$$
y _ { i } ^ { ( b , \mathrm { m o l } ) } = y _ { \sigma ^ { ( b ) } ( i ) } .
$$

The learned embedding is kept fixed, and the scalar values are shuffled across molecules. This tests whether the observed radial association is stronger than expected if the scalar values were exchangeable across molecules.

For radial analyses, the same procedure is applied to any scalar variable $y .$ In Sagar, y can be entropy or a continuous descriptor rating. In GSL $. \mathsf { F } , y$ is an entropy variable derived from the binary descriptor label. For each permutation scheme $Q ,$ where $Q$ can be WS, OB, or mol depending on the analysis, we recompute the mean Pearson radius–value correlation over seeds and denote the resulting permuted statistic by $T _ { \mathrm { r a d } , b } ^ { Q } ( y )$ The radial permutation p-value is two-sided because the correlation is signed:

$$
p _ { \mathrm { r a d } , Q } ( y ) = \frac { 1 + N _ { \mathrm { r a d } , Q } ( y ) } { B + 1 } ,
$$

where $N _ { \mathrm { r a d } , Q } ( y )$ is the number of permutations such that

$$
| T _ { \mathrm { r a d } , b } ^ { Q } ( y ) | \geq | T _ { \mathrm { r a d } , \mathrm { o b s } } ( y ) | .
$$

A small radial p-value therefore indicates that the observed radial association is stronger than expected after breaking the association between scalar values and embedding radius under the corresponding permutation null model.

For the angular descriptor analysis in Sagar, the WS and OB permutation schemes are applied to the descriptor values $x _ { ( o , s ) , k }$ . For each descriptor k and each scheme $Q ,$ , we refit the tangent-space linear regression after permutation and denote the resulting mean permuted $R ^ { 2 }$ over seeds by $T _ { k , \mathrm { a n g } , b } ^ { Q } .$ . Since larger values of $R ^ { 2 }$ indicate stronger directional organization, the angular permutation p-value is one-sided:

$$
p _ { k , \mathrm { a n g } , \mathcal { Q } } = \frac { 1 + N _ { k , \mathrm { a n g } , \mathcal { Q } } } { B + 1 } ,
$$

where $N _ { k , \mathrm { a n g } , Q }$ is the number of permutations such that

$$
T _ { k , \mathrm { a n g } , b } ^ { Q } \geq T _ { k , \mathrm { a n g } , \mathrm { o b s } } .
$$

A small angular p-value indicates that the descriptor is better explained by a global tangent-space direction than expected after the descriptor values are permuted under the corresponding restricted null model.

In the Sagar union results, we report mainly $p _ { \mathrm { { W S } } }$ and $p _ { \mathrm { O B } }$ In the GSLF radial entropy results, we report $p _ { \mathrm { m o l } }$ . The +1 correction avoids zero p-values when using a finite number of random permutations (Phipson & Smyth, 2010).

Angular organization of binary descriptors: high-density regions. For a dataset with binary descriptor annotations, such as the GoodScents–Leffingwell dataset, descriptor values do not represent graded perceptual intensities, but only the presence or absence of a given odor label. In this setting, the tangent-space regression used above for continuous descriptors cannot be applied: a binary variable can be spatially concentrated, but it does not define a perceptual gradient as for a continuous rating. We therefore analyze binary descriptors through their spatial concentration in the embedding. Intuitively, if a binary odor label such as fruity or floral is meaningfully organized in the learned representation, then the molecules annotated with this label should occupy a coherent region of the disk rather than being uniformly scattered across the embedding. With this analysis, we evaluate whether different odor descriptors or descriptor families occupy coherent and distinguishable regions of the Poincaré disk. In particular, we ask whether the angular component of the learned hyperbolic representation reflects categorical structure in olfactory perception, with different angular sectors corresponding to different perceptual qualities.

This visualization is inspired by the descriptor-region visualizations used in the Principal Odor Map work (Lee et al., 2023). However, since our embedding space is hyperbolic, we adapt the density estimation and contour construction to the geometry of the Poincaré disk. Given a learned embedding $\{ z _ { i } \} _ { i = 1 } ^ { N }$ , with $z _ { i } \in { \mathbb { P } } ^ { 2 }$ , and a binary descriptor $k ,$ let

$$
b _ { i , k } \in \{ 0 , 1 \}
$$

denote whether molecule i is annotated with descriptor k. We define the set of molecules annotated with this descriptor as

$$
I _ { k } = \{ i : b _ { i , k } = 1 \} .
$$

The goal is to estimate where, in the Poincaré disk, the molecules annotated with descriptor k are concentrated. We define a descriptor density by placing a smooth kernel around

each positive example and averaging these kernels. For $z \in \mathbb { P } ^ { 2 }$ this gives

$$
{ \widehat { \mathsf { p } } } _ { k } ( z ) = { \frac { 1 } { \left| I _ { k } \right| } } \sum _ { i \in I _ { k } } \exp \left( - { \frac { d _ { \mathbb { P } } ( z , z _ { i } ) ^ { 2 } } { 2 h ^ { 2 } } } \right) ,
$$

where $h > 0$ is a bandwidth hyperparameter controlling the smoothness of the density estimate. A larger value of h gives smoother and broader descriptor regions, whereas a smaller value gives sharper and more localized regions.

We then display the region where this descriptor density is highest. For a density threshold $\lambda ,$ we define the corresponding upper-level region as

$$
\Omega _ { k } ( \lambda ) = \left\{ z \in \mathbb { P } ^ { 2 } : \widehat { \mathsf { p } } _ { k } ( z ) \geq \lambda \right\} .
$$

Intuitively, $\Omega _ { k } ( \lambda )$ contains the points of the Poincaré disk where descriptor k has density at least λ. Increasing λ keeps only the densest core of the descriptor, while decreasing λ gives a larger region.

The parameter we choose is not λ directly, but a target mass level

$$
\tau \in ( 0 , 1 ) .
$$

For example, $\tau = 0 . 8$ means that we want to show the densest region containing approximately 80% of the estimated descriptor mass. The threshold $\lambda _ { k } ( \tau )$ is then calculated as the density level whose upper-level region contains approximately a fraction τ of the total descriptor mass.

Since the embedding lies in the Poincaré disk, this mass is computed using the hyperbolic area element $d A _ { \mathbb { P } }$ . Thus, $\lambda _ { k } ( \tau )$ is defined by

$$
\frac { \int _ { \Omega _ { k } ( \lambda _ { k } ( \tau ) ) } \widehat { \pmb { \rho } } _ { k } ( z ) d A _ { \mathbb { P } } ( z ) } { \int _ { \mathbb { P } ^ { 2 } } \widehat { \pmb { \rho } } _ { k } ( z ) d A _ { \mathbb { P } } ( z ) } \approx \tau .
$$

The displayed descriptor region is therefore

$$
\Omega _ { k } ( \tau ) : = \Omega _ { k } ( \lambda _ { k } ( \tau ) ) .
$$

In other words, $\Omega _ { k } ( \tau )$ is the high-density region of the Poincaré disk containing approximately a fraction τ of the descriptorspecific hyperbolic KDE mass.

This construction allows us to visualize whether binary odor descriptors form coherent high-density regions in the learned hyperbolic representation, and whether different olfactory categories are associated with distinct angular sectors of the Poincaré disk.

## 3 Results

## 3.1 Sagar

The Sagar results are presented in three stages. We first report the main radial and directional organization in the union embedding, then evaluate robustness to alternative descriptor representations, and finally examine subject-specific and subject-averaged patterns.

<table><tr><td>Variable</td><td>Distance Pearson</td><td>Distance Spearman</td><td>Radial Pearson</td><td>Radial Spearman</td><td> $p _ { \mathrm { { W S } } }$ </td><td>POB</td></tr><tr><td>Rating profile entropy Intensity descriptor rating</td><td> $0 . 8 2 \pm 0 . 0 2$ </td><td> $0 . 8 1 \pm 0 . 0 2$ </td><td> $- 0 . 7 7 \pm 0 . 0 5$   $0 . 4 2 \pm 0 . 0 5$ </td><td> $- 0 . 7 2 \pm 0 . 0 5$   $0 . 3 8 \pm 0 . 0 5$ </td><td> $< 0 . 0 0 1$   $< 0 . 0 0 1$ </td><td> $< 0 . 0 0 1$   $< 0 . 0 0 1$ </td></tr></table>

Table 1: Main radial organization results for the Sagar union embedding. Values are reported as mean and standard deviation over 10 random seeds. Restricted permutation p-values were computed with 1000 random permutations.

![](images/f784de3e026830a4a2f12b6f0ed065e929b3408e95c539c763365f4a00bc16f5.jpg)

(a) Embedding colored by rating profile entropy  
![](images/9de045f69742c8181f8bf721a590a7184e6d1b931378c86fce98128d0a08af2e.jpg)  
(c) Entropy–radius correlation plot, Pearson correlation is -0.82 and Spearman is -0.78.

![](images/e4c35335def0214d3970a4d53a65b76e75b9c788548ffb5e878dc1cf1efb4f1d.jpg)

(b) Embedding colored by intensity rating  
![](images/96ab5a76edd2f0fae175b527aa7e9108e85fec48854455a489c335038f50f9b1.jpg)  
(d) Intensity–radius correlation plot, Pearson correlation is 0.44 and Spearman is 0.41.

Figure 1: Representative Sagar union embedding for random seed $m = 5$ Points are colored by rating profile entropy and by intensity descriptor rating. Correlation plots show the corresponding radius associations.

Alt text: Four panels. Rating profile entropy is higher near the center of the Poincaré disk and lower near the boundary, producing a strong negative correlation with radius. Intensity shows the opposite but weaker radial tendency.

## 3.1.1 Main results

Radial organization. For the main configuration, trained on the original descriptor vectors, the learned hyperbolic embeddings preserved the input perceptual geometry well (Table 1). Distance Pearson and Spearman correlations compare pairwise distances in the input descriptor space with pairwise hyperbolic distances in the learned embedding. These embeddinglevel metrics are shared by the entropy and intensity analyses, and reached $0 . 8 2 \pm 0 . 0 2$ and $0 . 8 1 \pm 0 . 0 2$ , respectively.

Rating profile entropy computed from the original descriptors was strongly organized along the radial coordinate of the Poincaré disk. The radius–entropy correlation was strongly negative for both Pearson correlation, $- 0 . 7 7 \pm 0 . 0 5$ , and Spearman correlation, $- 0 . 7 2 { \pm } 0 . 0 5$ . This indicates that high-entropy observations tend to lie closer to the center, whereas lowentropy observations tend to lie closer to the boundary. This radial organization was supported by both restricted permutation tests, with $p _ { \mathrm { W S } } < 0 . 0 0 1$ and $p _ { \mathrm { O B } } < 0 . 0 0 1$ . Figure 1 illustrates these relationships for a representative random seed.

Among individual descriptors, intensity was the only descriptor showing a radial association (Table 1), with moderate radius–intensity correlations of $0 . 4 2 \pm 0 . 0 5$ for Pearson and $0 . 3 8 \pm 0 . 0 5$ for Spearman. The positive correlation indicates that higher-intensity observations tend to lie closer to the boundary. All other descriptors had radial correlations below 0.2 in absolute value and are therefore not presented in the main table; the full descriptor-level results are reported in the Supplementary material (Table S1). Importantly, the absolute radius–entropy correlation was substantially larger than the radius–intensity correlation, suggesting that the radial organization of the embedding is better explained by the rating profile entropy over the full descriptor vector than by any single perceptual descriptor.

Angular organization of continuous descriptors. In addition to the radial organization of entropy and intensity, several descriptors showed angular organization in the learned hyper bolic embedding (Table 2). Directional $R ^ { 2 }$ quantifies how well each descriptor is explained by a global linear direction in the tangent-space representation of the Poincaré disk. Only descriptors with mean directional $R ^ { 2 } > 0 . 2$ are shown in Table 2; the full descriptor-level table is reported in the Supplementary

material (Table S2).

The strongest directional trends were observed for sweet, musky, fruity, and pleasantness, with mean directional $R ^ { 2 }$ values ranging from 0.40 to 0.51. As visualized for sweet and musky in Figure 2 (additional figures for all other descriptors are available in the Supplementary material in Figure S1), this indicates that these four descriptors vary primarily along dominant directions in the Poincaré disk rather than along the radial coordinate alone. All reported angular trends in Table 2 were supported by both restricted permutation tests, with $p _ { \mathrm { W S } } < 0 . 0 0 1$ and $p _ { \mathrm { O B } } < 0 . 0 0 1$

![](images/d60e1932e0f1e1a9da01ccfa92a5937d82026d3c4bd0320018e64351fda8e497.jpg)

(a) Sagar embedding colored by sweet rating; directional angle at 158.40 degrees and $R ^ { 2 } = 0 . 5 8$  
![](images/4b1f730f5c144b3984afbb06268ea4fc7a920db1de5051d4575992d67e790d0d.jpg)  
(b) Sagar embedding colored by musky rating; directional angle at 279.97 degrees and $R ^ { 2 } = 0 . 4 3$

Figure 2: Representative Sagar union embedding for random seed $m = 5 ,$ , colored by sweet and musky ratings. Arrows indicate the fitted tangent space directions of steepest descriptor increase. The displayed angles and $R ^ { 2 }$ coefficient of determination values are seed specific.

Alt text: Two Poincaré disk embeddings. Sweet ratings increase toward the upper left along a diagonal direction, whereas musky ratings increase downward along an approximately vertical direction. Both descriptors show clear directional organization.

<table><tr><td>Descriptor</td><td>Directional  $R ^ { 2 }$ </td><td> $p _ { \mathrm { { W S } } }$ </td><td>POB</td></tr><tr><td>Sweet</td><td> $0 . 5 1 \pm 0 . 1 2$ </td><td> $< 0 . 0 0 1$ </td><td>&lt; 0.001</td></tr><tr><td>Musky</td><td> $0 . 5 1 \pm 0 . 0 6$ </td><td> $< 0 . 0 0 1$ </td><td>&lt; 0.001</td></tr><tr><td>Fruity</td><td> $0 . 4 6 \pm 0 . 0 9$ </td><td> $< 0 . 0 0 1$ </td><td>&lt; 0.001</td></tr><tr><td>Pleasantness</td><td> $0 . 4 0 \pm 0 . 0 9$ </td><td> $< 0 . 0 0 1$ </td><td>&lt; 0.001</td></tr><tr><td>Decayed</td><td> $0 . 3 4 \pm 0 . 0 6$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Warm</td><td> $0 . 2 6 \pm 0 . 1 2$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Floral</td><td> $0 . 2 5 \pm 0 . 1 1$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Bakery</td><td> $0 . 2 5 \pm 0 . 1 3$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Fishy</td><td> $0 . 2 3 \pm 0 . 0 4$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr></table>

Table 2: Angular organization results for the Sagar union embedding, sorted by decreasing mean directional $R ^ { 2 }$ . Values are reported as mean and standard deviation over 10 random seeds. Restricted permutation p-values were computed with 1000 random permutations.

## 3.1.2 Ablation and robustness analyses of the radial entropy organization

To test whether the radius–entropy relationship depends on the particular descriptor representation, we performed an ablation and robustness analysis (Table 3). This analysis addresses three possible concerns. First, since intensity was the only individual descriptor with a non-negligible radial association, the entropy effect could potentially be driven by intensity. Second, the effect could depend on redundant or correlated descriptors in the original descriptor space. Third, the effect could depend on the original descriptor coordinate system.

We tested these possibilities in two complementary ways. In the first three rows of Table 3, the embedding is kept fixed and trained on the original descriptors, while the entropy variable is recomputed using alternative descriptor representations: descriptors excluding intensity, pruned descriptors, and orthogonalized descriptors. This isolates the effect of changing the entropy definition without changing the learned embedding geometry. The pruned descriptor representation was constructed to reduce descriptor redundancy: descriptors were removed greedily until no remaining pair of descriptors had absolute correlation higher than 0.3. The removed descriptors were pleasantness, decayed, musky, fruity, sweet, and bakery. In the last two rows, the embedding itself is retrained using either descriptors excluding intensity or pruned descriptors, and entropy is computed in the corresponding reduced descriptor space. This provides a stronger test of whether the radius– entropy organization persists when the removed descriptors are excluded from the geometry used to learn the embedding.

The radius–entropy relationship remained strong and negative across all configurations. When intensity was removed only from the entropy computation, the radial Pearson correlation remained close to the main result, with $- 0 . 7 5 \pm 0 . 0 6$ and the radial Spearman correlation was $- 0 . 7 0 { \pm } 0 . 0 6$ . When intensity was removed both from the training input and from the entropy computation, the correlations remained similarly strong, with radial Pearson correlation $- 0 . 7 4 \pm 0 . 0 5$ and radial Spearman correlation $- 0 . 6 9 \pm 0 . 0 6$ . This indicates that the entropy-radius relationship is not merely an intensity effect.

<table><tr><td>Training input</td><td>Entropy type</td><td>Dist. Pearson</td><td>Dist. Spearman</td><td>Radial Pearson</td><td>Radial Spearman</td><td>pws</td><td>POB</td></tr><tr><td rowspan="2">Original descriptors</td><td>Descriptors excl. intensity</td><td></td><td></td><td> $- 0 . 7 5 \pm 0 . 0 6$   $- 0 . 6 8 \pm 0 . 0 4$ </td><td> $- 0 . 7 0 \pm 0 . 0 6$   $- 0 . 6 5 \pm 0 . 0 5$ </td><td> $< 0 . 0 0 1$   $< 0 . 0 0 1$ </td><td> $< 0 . 0 0 1$   $< 0 . 0 0 1$ </td></tr><tr><td>Pruned descriptors Orthogonalized descriptors</td><td> $0 . 8 2 \pm 0 . 0 2$ </td><td> $0 . 8 1 \pm 0 . 0 2$ </td><td> $- 0 . 6 9 \pm 0 . 0 7$ </td><td> $- 0 . 7 9 \pm 0 . 0 9$ </td><td> $< 0 . 0 0 1$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td rowspan="2">Descriptors excl. intensity</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Descriptors excl. intensity</td><td> $0 . 8 3 \pm 0 . 0 2$ </td><td> $0 . 8 1 \pm 0 . 0 2$ </td><td> $- 0 . 7 4 \pm 0 . 0 5$ </td><td> $- 0 . 6 9 \pm 0 . 0 6$ </td><td> $< 0 . 0 0 1$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td>Pruned descriptors</td><td>Pruned descriptors</td><td> $0 . 8 2 \pm 0 . 0 1$ </td><td> $0 . 8 1 \pm 0 . 0 1$ </td><td> $- 0 . 7 2 \pm 0 . 0 5$ </td><td> $- 0 . 6 6 \pm 0 . 0 8$ </td><td> $< 0 . 0 0 1$ </td><td> $< 0 . 0 0 1$ </td></tr></table>

Table 3: Ablation and robustness analysis of the radius–entropy relationship in the Sagar union embedding. Values are reported as mean and standard deviation over 10 random seeds. For the first three rows, the same original-input embedding is used; therefore, the distance-preservation metrics are shared. Restricted permutation p-values were computed with 1000 random permutations.

The effect also persisted when entropy was computed on pruned descriptors, and when the embedding was retrained using the pruned descriptor space. In both cases, the radial correlations remained negative and substantial, with Pearson correlations between −0.68 and −0.72. Finally, entropy computed from orthogonalized descriptors also showed a strong radial relationship, with radial Pearson correlation $- 0 . 6 9 { \pm } 0 . 0 7$ and radial Spearman correlation $- 0 . 7 9 \pm 0 . 0 9$ . Thus, the radial entropy organization is not specific to the original descriptor basis.

Across all ablation and robustness configurations, the distance-preservation metrics remained high, with distance Pearson and Spearman correlations around 0.81–0.83. All radius–entropy associations were supported by both restricted permutation tests, with $p _ { \mathrm { W S } } < 0 . 0 0 1$ and $p _ { \mathrm { O B } } < 0 . 0 0 1$ . Overall, these results support the interpretation that hyperbolic radius primarily reflects the entropy of the perceptual descriptor profile, rather than being driven by a single descriptor, by descriptor redundancy, or by the original coordinate system.

## 3.1.3 Subject-level and averaged-rating robustness analyses

We next tested whether the radial entropy organization and angular descriptor organization were stable across subjects, and whether they were also present at the level of subject aver aged ratings. The above studied main Sagar union embedding contains subject–odorant observations from all three subjects. Therefore, an apparent radial entropy effect could in principle be influenced by subject-level pooling effects, for example if one subject systematically produced higher-entropy ratings and was also placed closer to the center of the embedding.

We considered two complementary settings. First, we analyzed each subject separately within the union embedding. In this case, no new embedding was trained: for each random seed, we used the embedding trained on the full union dataset and then restricted the analysis to the 160 observations belonging to a single subject. This tests whether the entropy-radius relationship is visible within each subject’s observations in the shared union geometry. Second, we trained new embeddings on averaged descriptor ratings. For this averaged-rating analysis, descriptor vectors were averaged across subjects for each CID, retaining the 125 CIDs observed in all three subjects. This gives one consensus descriptor vector per odorant. We then trained hyperbolic embeddings on these averaged descriptor vectors and computed entropy from the averaged descriptor profile. This tests whether the radial and angular organization is also present at the level of averaged-ratings odor perception, after removing subject-specific rating variability.

For these robustness analyses, we used the same permutation logic as in the main evaluation, but adapted the exchangeability unit to the setting considered. For subject-level analyses within the union embedding, we used a within-subject permutation, denoted $p _ { \mathrm { W } \mathrm { S } }$ , by shuffling the tested variable across odorants within the selected subject. For the averaged-rating embedding, each point corresponds to one CID, so we used a CID-level permutation, denoted $p _ { \mathrm { C I D } }$ , by shuffling the tested variable across CIDs. Here, the tested variable is entropy for the radial analysis and the descriptor value for the angular analysis. Radial entropy p-values test whether the radius–entropy correlation is unusually strong in either direction, whereas angular p-values test whether the directional $R ^ { 2 }$ is unusually large.

Radial entropy organization. Table 4 reports the subjectlevel and averaged-rating radial profile entropy results. When the union embedding was analyzed separately within each subject, the entropy-radius relationship remained strongly negative: the radial Pearson correlations were $- 0 . 7 2 \pm 0 . 0 7$ for subject $, - 0 . 8 6 \pm 0 . 0 3$ for subject 2, and $- 0 . 6 1 \pm 0 . 0 6$ for subject 3. This shows that the radial entropy organization is not merely a consequence of pooling subjects in the union dataset, but is already present within each subject’s observations.

<table><tr><td>Analysis</td><td> $N _ { \mathrm { C I D } }$ </td><td>Radial Pearson</td><td>Radial Spearman</td><td>p</td></tr><tr><td>Subject 1</td><td>160</td><td> $- 0 . 7 2 \pm 0 . 0 7$ </td><td> $- 0 . 6 9 \pm 0 . 1 0$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td>Subject 2</td><td>160</td><td> $- 0 . 8 6 \pm 0 . 0 3$ </td><td> $- 0 . 8 7 \pm 0 . 0 3$ </td><td>&lt; 0.001</td></tr><tr><td>Subject 3</td><td>160</td><td> $- 0 . 6 1 \pm 0 . 0 6$ </td><td> $- 0 . 6 3 \pm 0 . 0 9$ </td><td>&lt; 0.001</td></tr><tr><td>Average</td><td>125</td><td> $- 0 . 8 2 \pm 0 . 0 8$ </td><td> $- 0 . 7 8 \pm 0 . 1 2$ </td><td> $< 0 . 0 0 1$ </td></tr></table>

Table 4: Subject-level and averaged-rating robustness analysis of the radial entropy organization. Values are reported as mean and standard deviation over 10 random seeds. The p-value column reports $p _ { \mathrm { { W S } } }$ for subject-level rows and p<sub>CID</sub> for the averaged-rating row, both computed with 1000 random permutations.

The strength of this organization nevertheless varied across individuals. Subject 2 showed the strongest radius–entropy association, whereas subjects 1 and 3 showed weaker but

![](images/7c55643ac08ccc20723f6b4c9198fe7eac4f10b3a7e03ec68cb1a2312c311432.jpg)

![](images/b50cf1471468a2bc46122eb8caaea5b9cb8d4889c381cf94b6e81a1fd2b229cd.jpg)

![](images/11e6847b10629a6b66ce5e274b9606f31e0fc1ce4ed01596a055ab922041f322.jpg)

![](images/6a1130a4b858f6d192504dc0ccefe1472279f9d6659ec0bdb2981532748bd6d9.jpg)  
(a) Subject 1. Pearson correlation (b) Subject 2. Pearson correlation (c) Subject 3. Pearson correlation (d) Averaged ratings. Pearson is - is -0.78 and Spearman is -0.78. is -0.90 and Spearman is -0.90. is -0.65 and Spearman is -0.66. 0.77 and Spearman is -0.72.

Figure 3: Poincaré embeddings colored by rating profile entropy for the Sagar dataset (random seed m = 5): individual subjects and averaged ratings.

Alt text: Four embeddings for the three individual subjects and averaged ratings. In every case, higher rating profile entropy occurs predominantly toward the center and lower entropy toward the boundary, although the spatial distributions differ.

still substantial associations. This variability may reflect differences in rating behavior, perceptual strategy, or the structure of each subject’s odor perceptual space. Importantly, the averaged-rating embedding also showed a strong negative radius–entropy correlation, −0.82 ± 0.08, indicating that the effect is preserved at the level of averaged-ratings odor perception despite individual variability. This radial–entropy relationship, across individual subjects and at the averaged level, is illustrated in Figure 3 which shows the different embeddings for one random seed.

Angular organization of continuous descriptors. Figure 4 summarizes the corresponding angular descriptor organization. The first three columns show the union embedding restricted to one subject at a time, whereas the last column shows embeddings trained directly on descriptor ratings averaged across subjects. Directional $R ^ { \dot { 2 } }$ quantifies how well each descriptor is explained by a global tangent-space direction in the Poincaré disk.

Several descriptor directions were stable across each of the three available subjects and in the averaged-rating embedding. Pleasantness, sweet, musky, fruity, floral, burnt, and bakery all had mean directional $R ^ { 2 } > 0 . 2$ in all four configurations: subject 1, subject 2, subject 3, and the averaged-rating embedding. These seven descriptors also had permutation p-values below 0.001 in all four configurations. The strongest and most consistent angular directions were associated with pleasantness, sweet, musky, and fruity, in agreement with the union-level angular analysis reported in Table 2. This supports the interpretation that angular position captures meaningful perceptual descriptor gradients, while radial position is primarily associated with the entropy of the descriptor profile.

The angular organization also revealed subject-level variability. Subject 2 generally showed stronger directional organization than subjects 1 and 3, especially for pleasantness, sweet, fruity, and musky. Subject 3 showed strong angular organization for fishy, decayed, musky, fruity, sweaty, sweet, and warm, whereas subject 1 showed weaker but still detectable directional trends. Thus, while similar descriptor families tend to define angular directions across subjects, the strength of these directions differs between individuals. This suggests that angular coordinates capture perceptual descriptor gradients that are partly shared across subjects, but also modulated by individual rating patterns.

![](images/ebb6b0fc442ba5710c0c4b1fa047d9afb4a69dcb5a3826cdfa645c7bf0e3fafb.jpg)  
Figure 4: Subject-level and averaged-rating angular descriptor organization. Each entry of this heatmap shows the mean directional $R ^ { 2 }$ over 10 random seeds.  
Alt text: Heatmap comparing directional $R ^ { 2 }$ across descriptors for three subjects and averaged ratings.

## 3.2 GoodScents–Leffingwell (GSLF)

The GSLF dataset provides a complementary test of the proposed geometric organization in a larger expert annotated odor dataset. Unlike Sagar, GSLF descriptors are binary multi-label annotations rather than continuous subject ratings. Therefore, entropy has a different interpretation. Active label entropy measures the breadth of the binary descriptor profile, whereas orthogonalized descriptor entropy measures whether the transformed descriptor profile is diffuse across several orthogonal modes of variation or dominated by one or a few modes.

<table><tr><td>Entropy type</td><td>Dist. Pearson</td><td>Dist. Spearman</td><td>Radial Pearson</td><td>Radial Spearman</td><td>Pmol</td></tr><tr><td>Active label entropy</td><td></td><td></td><td> $0 . 8 7 \pm 0 . 0 3$ </td><td> $0 . 8 8 \pm 0 . 0 3$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td>Pruned active label entropy</td><td> $0 . 6 9 \pm 0 . 0 1$ </td><td> $0 . 6 8 \pm 0 . 0 1$ </td><td> $0 . 7 8 \pm 0 . 0 3$ </td><td> $0 . 7 8 \pm 0 . 0 3$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td>Orthogonalized descriptor entropy</td><td></td><td></td><td> $- 0 . 8 8 \pm 0 . 0 1$ </td><td> $- 0 . 9 2 \pm 0 . 0 2$ </td><td> $< 0 . 0 0 1$ </td></tr></table>

Table 5: Radial entropy organization results for the GSLF dataset. Distance Pearson and Spearman correlations are shared embedding quality metrics because computed from the same learned embeddings. Values are reported as mean and standard deviation over 10 random seeds. Molecule level permutation p-values, denoted $p _ { \mathrm { m o l } } .$ , were computed with 1000 random permutations.

Radial entropy organization. Table 5 reports the radial entropy results for the GSLF embedding. As an embedding quality check, the learned hyperbolic representations preserved the input binary descriptor geometry with distance Pearson correlation $0 . 6 9 \pm 0 . 0 1$ and distance Spearman correlation $0 . 6 8 \pm 0 . 0 1$ . These values indicate that the embedding retains a substantial part of the pairwise descriptor structure, although the distance preservation is lower than in the Sagar dataset. This could be explained by the sparse binary nature of the GSLF descriptor matrix and its higher dimensionality (138 descriptors for GSLF compared to 15 for Sagar).

Active label entropy showed a strong positive association with hyperbolic radius, with radial Pearson correlation $0 . 8 7 \pm 0 . 0 3$ and radial Spearman correlation $0 . 8 8 \pm 0 . 0 3$ . This indicates that molecules annotated with broader descriptor profiles tend to lie closer to the boundary of the Poincaré disk. Importantly, this result should not be interpreted in the same way as the negative radius–entropy correlation observed in Sagar. In Sagar, rating profile entropy measures the spread of continuous descriptor strengths, whereas in GSLF active label entropy measures the number of active binary labels. Thus, the positive correlation in GSLF suggests that molecules associated with many odor qualities occupy more peripheral regions of the embedding.

The radial organization remained strong after reducing descriptor redundancy. When active label entropy was computed from the pruned descriptor representation, where remaining labels have absolute correlation inferior or equal to 0.3, the radial Pearson and Spearman correlations remained high with both at $0 . 7 8 \pm 0 . 0 3$ . This indicates that the association between active label entropy and radius is not solely driven by correlated or redundant descriptors.

Finally, orthogonalized descriptor entropy showed a strong negative association with hyperbolic radius, with radial Pearson correlation $- 0 . 8 8 \pm 0 . 0 1$ and radial Spearman correlation $- 0 . 9 2 \pm 0 . 0 2$ . This result has a different interpretation from active label entropy. After orthogonalization, the dimensions no longer correspond to individual odor labels, but to orthogonal modes of variation in the descriptor data. Orthogonalized descriptor entropy therefore tests whether the transformed score profile is balanced across several modes or dominated by one or a few modes. This negative correlation is consistent with the Sagar results when entropy is computed from continuous or orthogonalized descriptor profiles. It indicates that, in the orthogonalized descriptor representation, molecules closer to the center have more diffuse profiles across orthogonal modes, whereas molecules closer to the boundary have profiles domi nated by one or a few modes.

All three radial entropy associations were significant under molecule-level permutation testing, with $p _ { \mathrm { m o l } } < 0 . 0 0 1$ . Overall, these results show that the radial coordinate of the GSLF embedding captures entropy-related structure, but the interpretation depends on the entropy definition. Active label entropy reflects descriptor multiplicity, whereas orthogonalized descriptor entropy reflects spread across orthogonal modes of variation. Figure 5 illustrates these relationships for a representative random seed.

Angular organization of binary descriptors. We next examined whether binary odor descriptors occupy coherent regions of the GSLF embedding. Because GSLF labels are binary annotations, they do not define graded descriptor directions in the same sense as the continuous ratings in Sagar. We therefore used the hyperbolic KDE visualization described in the Method section to identify high density regions associated with descriptor families.

Following Lee et al. (2023), Figure 6 shows representative high density regions for three broad descriptor families: floral, meaty, and ethereal. Molecules annotated with related descriptors tend to occupy nearby regions of the Poincaré disk. Floral descriptors such as floral, muguet, lavender, and jasmin form a coherent region, whereas meaty descriptors such as meaty, savory, beefy, and roasted occupy a distinct region. Ethereal descriptors such as ethereal, cognac, fermented, and alcoholic form a third region. These three descriptor families are spatially separated and occupy different angular sectors of the disk, suggesting that angular position captures categorical structure among binary odor labels.

Figure 7 provides a finer grained visualization of the fruity descriptor family. The broader fruity region contains or overlaps with several fruit related descriptor regions, including melon, banana, apple, pear, pineapple, grapefruit, black currant, grape, raspberry, berry, strawberry, apricot, plum, peach, cherry, orange. By contrast, bergamot, lemon, and coconut appear farther from the main fruity region. Thus, the embedding captures both broad odor families and finer categorical distinctions within a family: related fruit descriptors tend to occupy a common sector of the disk, while individual descriptors remain locally distinguishable.

These visualizations suggest a complementary organiza-

![](images/073dde8df3159fd04eba2c52679f920264c6b839e58126afcce31950e75fa37f.jpg)

(a) Embedding colored by active label entropy  
![](images/b04db75a97dab922db5a617586ebb9af8e9df18e3c3e0c7c30752fa0a39ba9a5.jpg)

(b) Radius correlation plot for active label entropy, Pearson is 0.86 and Spearman is 0.88.  
![](images/49b8850f2c2b53a23e62233566bd326f9c501f68e9776831878ff6e5b8048a2b.jpg)

(c) Embedding colored by orthogonalized descriptor entropy  
![](images/3b7acf5ad9cb0b00a32a9ec2f6c2c2a3291a9121ae4179cb4d71b380dc45b167.jpg)  
(d) Radius correlation plot for orthogonalized descriptor entropy, Pearson is -0.87 and Spearman is -0.91.  
Figure 5: Representative GSLF embedding and radius associations for active label entropy and orthogonalized descriptor entropy, at random seed $m = 1$

Alt text: Four panels showing radial entropy patterns in the GSLF embedding. Active label entropy increases strongly from the center toward the boundary, whereas orthogonalized descriptor entropy decreases strongly with radius.

![](images/a07781f5f74fcb78b1c262fc5d980a4d72476eca2b0d2d98cdcd625b9dd6f3b0.jpg)  
Figure 6: High density regions for representative GSLF descriptor families in the Poincaré disk. Gray points show all molecules in the embedding. Filled regions and contours show hyperbolic KDE upper level sets for broad descriptor families and related subdescriptors, using a KDE mass level of $\tau = 5 \%$ Alt text: Poincaré disk showing spatially separated descriptor families. Floral descriptors occupy the upper region, meaty descriptors the lower region, and ethereal descriptors the lower right region. Related subdescriptors form overlapping localized contours within each family.

![](images/faf968876d0b0f2d8060a159b396a15d948ba07ae61dbac3f92bc379fade23dd.jpg)  
Figure 7: High density regions for the fruity descriptor family in the GSLF embedding. Gray points show all molecules. The filled region corresponds to the broader fruity descriptor and is shown with KDE mass level $\tau = 2 0 \%$ . Contours correspond to fruit related subdescriptors and are shown with KDE mass level $\tau = 1 \%$

Alt text: Poincaré disk showing a broad fruity descriptor region surrounded by localized fruit subdescriptor contours. Many fruit descriptors cluster within or near the main fruity region, while bergamot, lemon, and coconut form more separated regions.

tion of the GSLF embedding. The radial coordinate is associated with entropy related properties of descriptor profiles, whereas angular position appears to reflect categorical olfactory structure, separating broad odor families and organizing finer subcategories within them.

## 4 Discussion

## 4.1 Summary and interpretation

The present work shows that olfactory descriptor data exhibit complementary radial and angular organization when represented in the two-dimensional Poincaré disk. In the Sagar dataset, hyperbolic radius was most strongly associated with rating profile entropy, whereas individual descriptors were better characterized by directional trends. The radial association persisted across the robustness analyses, supporting the interpretation that radius reflects a global property of the descriptor profile rather than any single perceptual descriptor. More diffuse profiles were located closer to the center, whereas more concentrated profiles were located closer to the boundary.

The directional analysis provided a complementary description of odor quality. Several descriptors, including sweet, musky, fruity, pleasantness, and decayed, were well summarized by dominant tangent-space directions. In particular, the strong directional trend of pleasantness is consistent with previous work identifying pleasantness as an important organizing axis of olfactory perception (Crocker & Henderson, 1927; Khan et al., 2007; Koulakov et al., 2011; Snitz et al., 2013; Licon et al., 2018). The hyperbolic representation therefore retained an established perceptual dimension while revealing a distinct radial organization related to the descriptor profile as a whole. Similar directional patterns were observed in the subject-specific and subject-averaged analyses, although their strength varied among the three subjects.

The GSLF results extended this geometric decomposition to a larger dataset of binary expert annotations. In this setting, active label entropy increased with radius, indicating that molecules assigned a larger number of odor descriptors tended to occupy more peripheral regions of the disk. This positive association does not contradict the negative radius–entropy relationship observed in Sagar, because active label entropy measures descriptor multiplicity, whereas rating profile entropy measures how diffusely continuous rating strength is distributed across descriptors. A complementary pattern emerged within GSLF itself: orthogonalized descriptor entropy decreased with radius, showing that molecules closer to the center had more diffuse profiles across orthogonal modes of variation, whereas those closer to the boundary were dominated by fewer modes, in line with Sagar. Thus, the direction of the radial association depends on the property summarized by the entropy measure, while radius consistently captures global structure in the descriptor profile. In addition, related binary descriptors occupied coherent high-density regions, with broad odor families and finer subcategories appearing in distinct angular sectors of the disk.

Together, these findings support hyperbolic mapping as an interpretable descriptive framework in which radius summarizes global properties of descriptor profiles, while the angular component captures descriptor-specific gradients through directional trends in continuous ratings and categorical organization through localized high-density regions for binary descriptors.

## 4.2 Limitations and future work

The present study has several limitations that also point toward useful directions for future work. First, the present analysis is based on descriptor data and does not directly test neural mechanisms of olfactory coding. Although the results are compatible with the hypothesis that olfactory perception has non-Euclidean structure, linking these geometric features to neural representations will require analyses of brain data, such as fMRI or EEG, collected from a sufficiently large and diverse participant sample.

A further limitation is the small number of subjects in the continuous rating dataset. The subject specific analyses provide initial evidence that the observed radial and directional organization is not restricted to a single rating profile. However, larger and more diverse samples will be needed to characterize the consistency of these patterns and their variability across the broader population. Such studies could also benefit from richer descriptor vocabularies provided by trained assessors or odor experts. Indeed, hyperbolic spaces can be seen as continuous analogs of trees (Krioukov et al., 2010) and thus, hyperbolic geometry may be most informative when descriptors are organized into an explicit multilevel taxonomy with substantial branching and depth, rather than as a flat list of broad odor qualities. Data of this kind would make it possible to test more directly whether hyperbolic embeddings capture hierarchical relations among odor categories, for example via hyperbolic tree geometric inference methods such as Medbouhi et al. (2026).

The spatial analysis of binary descriptors should also be interpreted cautiously. The hyperbolic KDE regions used for binary descriptors are qualitative visualizations. They provide descriptive evidence that odor labels occupy coherent regions of the disk, but they do not constitute formal statistical tests of category separation. Future work could complement these visualizations with quantitative measures of spatial concentration, overlap, and separation between descriptor families.

Beyond the perceptual descriptor space itself, the present study does not incorporate molecular structure. Integrating molecular features with perceptual descriptors in a joint hyperbolic framework could help determine how chemical similarity relates to the radial entropy organization and angular odor category structure observed here. Such a model could also clarify which aspects of the learned geometry arise from perceptual judgments and which are already present in the molecular organization of the odorants.

Finally, the present work does not include behavioral confidence ratings, emotion measures, cognitive style measures, personality measures, or clinical assessments of olfactory function. The differences observed in the subject specific analyses should therefore be interpreted as differences in rating patterns, rather than as evidence for particular cognitive, personality, or sensory mechanisms. Previous work nevertheless suggests that olfactory perception and confidence in sensory judgments may be influenced by personality traits (Shepherd et al., 2017; Seo et al., 2013), including neuroticism (Croy et al., 2011), agreeableness and openness to experience (Tyagi et al., 2024), as well as by emotion and cognitive biases (Chen & Dalton, 2005). The strong radial association observed here indicates that entropy is a global organizing property of olfactory descriptor profiles in the learned perceptual representation. However, because entropy was computed from the ratings rather than directly judged by the participants, its psychological interpretation remains tentative. Studies with larger samples could combine descriptor ratings with direct judgments of perceptual clarity, complexity, ambiguity, and confidence to investigate whether unusually high entropy reflects uncertainty or less differentiated judgments, and whether unusually low entropy reflects confidence, overconfidence, or a restricted response strategy. Separately, psychophysical and clinical measures of olfactory function could be employed to examine whether embeddings confined to a limited region of the Poincaré disk are associated with reduced perceptual differentiation or olfactory impairment, rather than with differences in vocabulary, scale use, or rating strategy.

## Conflicts of interest

No relevant conflict of interest declared.

## Funding

This work has been supported by the Swedish Research Council, Knut and Alice Wallenberg Foundation, and the European Research Council (ERC-2023-SyG 10118977 D2Smell).

## Acknowledgements

The authors wish to thank Pawel Andrzej Herman for his valuable feedback. The authors acknowledge the use of artificial intelligence (AI) tools to assist with language editing and code development. All AI assisted code was reviewed, tested, and validated by the authors. The authors retained full responsibility for the scientific design, analyses, interpretations, and conclusions.

## Data availability

The Sagar dataset is publicly available through the Pyrfume repository (Hamel et al., 2024), and the GoodScents– Leffingwell dataset is available through OpenPOM (Barsainyan et al., 2023). Our code used to generate the embeddings, perform the geometric and statistical analyses, and reproduce the figures is available at https://github.com/ anissmedbouhi/HyperSmell.

## References

Barsainyan, A. A., Kumar, R., Saha, P., & Schmuker, M. (2023). Openpom. Retrieved from https://github.com/ ARY2260/openpom

Becigneul, G., & Ganea, O.-E. (2019). Riemannian adaptive optimization methods. In International conference on learning representations. Retrieved from https://openreview .net/forum?id=r1eiqi09K7

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., Choromanski, K., . . . others (2023). Rt-2: Vision-languageaction models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818.

Chen, D., & Dalton, P. (2005). The effect of emotion and personality on olfactory perception. Chemical Senses, 30(4), 345–351. Retrieved from https://academic.oup.com/ chemse/article-abstract/30/4/345/270246 doi: 10 .1093/chemse/bji029

Crocker, E. C., & Henderson, L. (1927). Analysis and classification ofodors: an effort to develop a workable method. Robbins Perfumer Company.

Croy, I., Springborn, M., Lötsch, J., & Johnston, A. N. B. (2011). Agreeable smellers and sensitive neurotics–correlations among personality traits and sensory thresholds. PLoS ONE, 6(3), e18701. Retrieved from https://journals.plos.org/plosone/ article?id=10.1371/journal.pone.0018701 doi: 10.1371/journal.pone.0018701

de l’Éclairage, C. I. (1931). Commission internationale de l’eclairage proceedings. Cambridge University Press Cambridge.

Doty, R. L. (2025, 01). Odors as cognitive constructs: history of odor classification and attempts to map odor percepts to physical and chemical parameters. Chemical Senses, 50, bjaf022. Retrieved from https://doi.org/10.1093/ chemse/bjaf022 doi: 10.1093/chemse/bjaf022

Du, Y., Liu, Z., Li, J., & Zhao, W. X. (2022). A survey of vision-language pre-trained models. arXiv preprint arXiv:2202.10936.

Evans, E. (1977). Frequency selectivity at high signal levels of single units in cochlear nerve and nucleus. Psychophysics and physiology of hearing, 185–192.

Friederici, A. D. (2012). The cortical language circuit: from auditory perception to sentence comprehension. Trends in cognitive sciences, 16(5), 262–268.

Ganea, O., Bécigneul, G., & Hofmann, T. (2018). Hyperbolic neural networks. Advances in neural information processing systems, 31.

Ganis, G., Thompson, W. L., & Kosslyn, S. M. (2004). Brain areas underlying visual mental imagery and visual perception: an fmri study. Cognitive Brain Research, 20(2), 226–241.

GoodScent. (n.d.). The good scents company. Retrieved from http://www.thegoodscentscompany.com/

Hamel, E. A., Castro, J. B., Gould, T. J., Pellegrino, R., Liang, Z., Coleman, L. A., . . . others (2024). Pyrfume: A window to the world’s olfactory data. Scientific data, 11(1), 1220.

Kaeppler, K., & Mueller, F. (2013). Odor classification: a review of factors influencing perception-based odor arrangements. Chemical senses, 38(3), 189–209.

Keller-Ressel, M., & Nargang, S. (2020). Hydra: A method for strain-minimizing hyperbolic embedding of network- and distance-based data. Journal of Complex Networks, 8(1), cnaa002. doi: 10.1093/comnet/cnaa002

Khan, R. M., Luk, C.-H., Flinker, A., Aggarwal, A., Lapid, H., Haddad, R., & Sobel, N. (2007). Predicting odor pleasantness from odorant structure: Pleasantness as a reflection of the physical world. The Journal of Neuroscience, 27, 10015 - 10023. Retrieved from https:// api.semanticscholar.org/CorpusID:13710261

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In International conference on learning representations (iclr). Retrieved from https://arxiv.org/ abs/1412.6980

Klimovskaia, A., Lopez-Paz, D., Bottou, L., & Nickel, M. (2020). Poincaré maps for analyzing complex hierarchies in singlecell data. Nature Communications. doi: 10.1038/s41467 -020-16822-4

Kochurov, M., Karimov, R., & Kozlukov, S. (2020). Geoopt: Riemannian optimization in pytorch.

Koulakov, A., Kolterman, B. E., Enikolopov, A., & Rinberg, D. (2011). In search of the structure of human olfactory space. Frontiers in systems neuroscience, 5, 9271.

Krioukov, D., Papadopoulos, F., Kitsak, M., Vahdat, A., & Boguñá, M. (2010, Sep). Hyperbolic geometry of complex networks. Phys. Rev. E, 82, 036106. Retrieved from https://link.aps.org/doi/10.1103/PhysRevE .82.036106 doi: 10.1103/PhysRevE.82.036106

Kruskal, J. B. (1964). Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. Psychometrika, 29(1), 1–27. doi: 10.1007/BF02289565

Lee, B. K., Mayhew, E. J., Sanchez-Lengeling, B., Wei, J. N., Qian, W. W., Little, K., . . . others (2022). A principal odor map unifies diverse tasks in human olfactory perception. BioRxiv, 2022–09.

Lee, B. K., Mayhew, E. J., Sanchez-Lengeling, B., Wei, J. N., Qian, W. W., Little, K. A., . . . Wiltschko, A. B. (2023). A principal odor map unifies diverse tasks in olfactory perception. Science, 381(6661), 999-1006. Retrieved from https://www.science.org/doi/abs/10 .1126/science.ade4401 doi: 10.1126/science.ade4401

Leffingwell, & Associates. (2001). Database of perfumery materials and performance. Retrieved from http://www .leffingwell.com/

Licon, C. C., Manesse, C., Dantec, M., Fournel, A., & Bensafi, M. (2018). Pleasantness and trigeminal sensations as salient dimensions in organizing the semantic and physiological spaces of odors. Scientific Reports, 8, 8444. doi: 10.1038/ s41598-018-26510-5

Madany Mamlouk, A., & Martinetz, T. (2004). On the dimensions of the olfactory perception space. Neurocomputing, 58-60, 1019-1025. Retrieved from https://www.sciencedirect.com/science/

article/pii/S0925231204001663 (Computational Neuroscience: Trends in Research 2004) doi: https://doi.org/10.1016/j.neucom.2004.01.161

Magnasco, M. O., Keller, A., & Vosshall, L. B. (2015). On the dimensionality of olfactory space. bioRxiv. Retrieved from https://www.biorxiv.org/content/ early/2015/07/06/022103 doi: 10.1101/022103

Medbouhi, A. A., García-Castellanos, A., Marchetti, G. L., Pelt, D., Bekkers, E. J., & Kragic, D. (2026). Randomized hypersteiner: A stochastic delaunay triangulation heuristic for the hyperbolic steiner minimal tree. Retrieved from https://arxiv.org/abs/2510.09328

Medbouhi, A. A., Taleb, F., Marchetti, G. L., & Kragic, D. (2025). Modeling the hierarchy of the human olfactory perceptual space via hyperbolic embeddings. In 8th annual conference on cognitive computational neuroscience. Amsterdam, The Netherlands. Retrieved from https://2025 .ccneuro.org/poster/?id=FijOA6b5lY (Extended abstract, Poster A143)

Meister, M. (2015, jul). On the dimensionality of odor space. eLife, 4, e07865. Retrieved from https://doi.org/10 .7554/eLife.07865 doi: 10.7554/eLife.07865

Nagano, Y., Yamaguchi, S., Fujita, Y., & Koyama, M. (2019). A wrapped normal distribution on hyperbolic space for gradientbased learning. In International conference on machine learning.

Nickel, M., & Kiela, D. (2017). Poincaré embeddings for learning hierarchical representations. Advances in neural information processing systems, 30.

Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn. Statistical Applications in Genetics and Molecular Biology, 9(1), Article 39. doi: 10.2202/1544-6115.1585

Sagar, V., Shanahan, L. K., Zelano, C. M., Gottfried, J. A., & Kahnt, T. (2023). High-precision mapping reveals the structure of odor coding in the human brain. Nature neuroscience, 1–8.

Sajan, A., Sluis, S., Haydarlou, R., Abeln, S., Lisena, P., Troncy, R., . . . Mouhib, H. (2026, 07). Hierarchies of smell: Structuring the molecular odor space using semantic taxonomies and machine learning. ChemicalSenses, bjag020. Retrieved from https://doi.org/10.1093/chemse/bjag020 doi: 10.1093/chemse/bjag020

Sala, F., De Sa, C., Gu, A., & Re, C. (2018, 10–15 Jul). Representation tradeoffs for hyperbolic embeddings. In J. Dy & A. Krause (Eds.), Proceedings of the 35th international conference on machine learning (Vol. 80, pp. 4460–4469). PMLR.

Sarkar, R. (2012). Low distortion delaunay embedding of trees in hyperbolic plane. In M. van Kreveld & B. Speckmann (Eds.), Graph drawing (pp. 355–366). Berlin, Heidelberg: Springer Berlin Heidelberg.

Schiffman, S. S. (1974). Contributions to the physicochemica dimensions of odor: a psychophysical approach. Annals of the New York Academy of Sciences, 237(1), 164–183.

Seo, H. S., Lee, S., & Cho, S. (2013). Relationships between personality traits and attitudes toward the sense of smell. Frontiers in Psychology, 4, 901. Retrieved from https://www.frontiersin.org/ articles/10.3389/fpsyg.2013.00901/full doi: 10 .3389/fpsyg.2013.00901

Shepherd, D., Hautus, M. J., & Urale, P. W. B. (2017). Personality and perceptions of common odors. Chemosensory Perception, 10(1), 1–12. Retrieved from https://link.springer.com/article/10.1007/ s12078-016-9220-4 doi: 10.1007/s12078-016-9220-4

Snitz, K., Yablonka, A., Weiss, T., Frumin, I., Khan, R. M., & Sobel, N. (2013). Predicting odor perceptual similarity from odor structure. PLoS computational biology, 9(9), e1003184.

Sucholutsky, I., Muttenthaler, L., Weller, A., Peng, A., Bobu, A., Kim, B., . . . others (2023). Getting aligned on representational alignment. arXiv preprint arXiv:2310.13018.

Taleb, F., Medbouhi, A. A., Marchetti, G. L., & Kragic, D. (2025). Towards discovering the hierarchy of the olfactory perceptual space via hyperbolic embeddings. Science Communications Worldwide. Retrieved from https://www.world-wide.org/cosyne-25/towards -discovering-hierarchy-olfactory-037f6ccb doi: 10.57736/0d78-6155

Torgerson, W. S. (1952). Multidimensional scaling: I. theory and method. Psychometrika, 17(4), 401–419. doi: 10.1007/ BF02288916

Tyagi, P., Bansal, S., Sharma, A., & Tiwary, U. S. (2024). Differences in olfactory functioning: The role of personality and gender. Journal of Sensory Studies, 39(1), e13097. Retrieved from https://onlinelibrary.wiley.com/doi/ abs/10.1111/joss.12907 doi: 10.1111/joss.12907

Ungar, A. (2009). A gyrovector space approach to hyperbolic geometry (Vol. 1). doi: 10.2200/ S00175ED1V01Y200901MAS004

Walter, J. A. (2004). H-mds: a new approach for interactive visualization with multidimensional scaling in the hyperbolic space. Information Systems, 29(4), 273- 292. Retrieved from https://www.sciencedirect.com/ science/article/pii/S0306437903001005 (Knowledge Discovery and Data Mining (KDD 2002)) doi: https:// doi.org/10.1016/j.is.2003.10.002

Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014). Permutation inference for the general linear model. NeuroImage, 92, 381–397. doi: 10.1016/ j.neuroimage.2014.01.060

Winkler, A. M., Webster, M. A., Vidaurre, D., Nichols, T. E., & Smith, S. M. (2015). Multi-level block permutation. NeuroImage, 123, 253–268. doi: 10.1016/j.neuroimage.2015.05 .092

Zhang, H., Rich, P., Lee, A., & Sharpee, T. (2022, 12). Hippocampal spatial representations exhibit a hyperbolic geometry that expands with experience. Nature Neuroscience, 26, 1-9. doi: 10.1038/s41593-022-01212-4

Zhou, Y., & Sharpee, T. O. (2021). Hyperbolic geometry of gene expression. iScience, 24(3), 102225.

Zhou, Y., Smith, B. H., & Sharpee, T. O. (2018). Hyperbolicgeometry of the olfactory space. Science Advances, 4(8),eaaq1458. doi: 10.1126/sciadv.aaq1458

# Supplementary material

## A Hyperbolic geometry and optimization details

This Supplementary material provides the geometric expressions and optimization details underlying the hyperbolic metric MDS model. The main text contains the concepts needed to understand the radial and directional analyses, whereas the general Poincaré ball expressions and Riemannian optimization updates are provided here for reproducibility.

## A.1 Geometry of the Poincaré ball

We start to describe the model where we embed our data, and explicit the metric tensor and the derived hyperbolic distance Formally, the n-dimensional hyperbolic space is the unique simply-connected Riemannian manifold with constant curvature equal to −1. The hyperbolic space admits several models; in this work, we focus on the Poincaré ball model. The latter is the Riemannian manifold given by the Euclidean ball

$$
\mathbb { P } ^ { n } = \left\{ z \in \mathbb { R } ^ { n } \mid \left\| z \right\| < 1 \right\} ,
$$

where ∥ · ∥ denotes the Euclidean norm, equipped with a Riemannian metric consisting of the Euclidean inner product scaled by a factor that reflects the curvature of the space.

At a point $z \in \mathbb { P } ^ { n }$ , we define the conformal factor as

$$
\lambda _ { z } = \frac { 2 } { 1 - \| z \| ^ { 2 } } .
$$

For tangent vectors $\nu , w \in T _ { z } \mathbb { P } ^ { n }$ identified with $\mathbb { R } ^ { n }$ , the Riemannian metric is then

$$
g _ { z } ( \nu , w ) = \lambda _ { z } ^ { 2 } \langle \nu , w \rangle ,
$$

where $\langle \cdot , \cdot \rangle$ denotes the ordinary Euclidean inner product. The corresponding Riemannian norm of a tangent vector is

$$
\| \nu \| _ { z } = \sqrt { g _ { z } ( \nu , \nu ) } = \lambda _ { z } \| \nu \| .
$$

As for any Riemannian manifold, $\mathbb { P } ^ { n }$ can be seen as a metric space when equipped with the geodesic distance, i.e., the length of the shortest path between two points of the manifold. Formally, the geodesic distance between two points $x , y \in \mathbb { P } ^ { n }$ is defined as:

$$
d _ { \mathbb { P } } ( x , y ) = \operatorname* { i n f } _ { \gamma } \int _ { [ 0 , 1 ] } \sqrt { g _ { \gamma ( t ) } ( \gamma ^ { \prime } ( t ) , \gamma ^ { \prime } ( t ) ) } d t ,
$$

where $\gamma \colon [ 0 , 1 ] \to \mathbb { P } ^ { n }$ is a smooth curve with $\gamma ( 0 ) = x , \gamma ( 1 ) = y$ , and $\gamma$ denotes the first derivative of $\gamma .$ Explicitly, the distance can be computed via the simple expression:

$$
d _ { \mathbb { P } } ( x , y ) = \operatorname { a r c o s h } \left( 1 + { \frac { 2 \lVert x - y \rVert ^ { 2 } } { ( 1 - \lVert x \rVert ^ { 2 } ) \left( 1 - \lVert y \rVert ^ { 2 } \right) } } \right) .
$$

where arcosh(r) = ln $\left( r + { \sqrt { r ^ { 2 } - 1 } } \right)$ , for $r \geq 1$

In particular, the hyperbolic radius of a point z is

$$
\begin{array} { r } { d _ { \mathbb { P } } ( 0 , z ) = 2 \operatorname { a r t a n h } ( \| z \| ) . } \end{array}
$$

Consequently, equal Euclidean displacements correspond to increasingly large hyperbolic distances as points approach the boundary of the disk.

## A.2 Möbius addition

The exponential map, logarithmic map, and parallel transport can be written using Möbius addition. For $x , y \in \mathbb { P } ^ { n }$ , Möbius addition is defined as (Ungar, 2009):

$$
x \oplus y = { \frac { \left( 1 + 2 \langle x , y \rangle + \| y \| ^ { 2 } \right) x + \left( 1 - \| x \| ^ { 2 } \right) y } { 1 + 2 \langle x , y \rangle + \| x \| ^ { 2 } \| y \| ^ { 2 } } } .
$$

The additive inverse of x under this operation is its Euclidean negative, $- x .$

## A.3 Tangent spaces and geometric maps

The tangent space $T _ { z } \mathbb { P } ^ { n }$ at any point $z \in \mathbb { P } ^ { n }$ can be identified with $\mathbb { R } ^ { n }$ as a vector space, although its inner product depends $\mathsf { o n } z$ through the Riemannian metric.

The exponential map sends a tangent vector $\nu \in T _ { z } \mathbb { P } ^ { n }$ to a point on the manifold by following the geodesic starting at z in the direction v. As derived by Ganea et al. (2018), for $\nu \neq 0 ,$ , it is given by

$$
\exp _ { z } ( \nu ) = z \oplus \left[ \operatorname { t a n h } \left( \frac { \lambda _ { z } \lVert \nu \rVert } { 2 } \right) \frac { \nu } { \lVert \nu \rVert } \right] ,
$$

and

$$
\exp _ { z } ( 0 ) = z .
$$

Conversely, the logarithmic map sends a point $y \in \mathbb { P } ^ { n }$ to the tangent vector at z that points along the geodesic from $z \tan y .$ . Let

$$
d = ( - z ) \oplus y .
$$

For $y \neq z ,$

$$
\log _ { z } ( y ) = \frac { 2 } { \lambda _ { z } } \mathrm { a r t a n h } ( \rVert d \rVert ) \frac { d } { \rVert d \rVert } ,
$$

and

$$
\log _ { z } ( z ) = 0 .
$$

The Riemannian norm of this tangent vector equals the hyperbolic distance:

$$
\begin{array} { r } { \left\| \mathbf { l o g } _ { z } ( y ) \right\| _ { z } = d _ { \mathbb { P } } ( z , y ) . } \end{array}
$$

Its ordinary Euclidean norm generally differs from the hyperbolic distance because the tangent space metric is scaled by $\lambda _ { z }$ At the origin, $\lambda _ { 0 } = 2$ , and the maps reduce to

$$
\exp _ { 0 } ( \nu ) = \left\{ \begin{array} { l l } { \operatorname { t a n h } ( \left\| \nu \right\| ) \frac { \nu } { \left\| \nu \right\| } , } & { \nu \neq 0 , } \\ { 0 , } & { \nu = 0 , } \end{array} \right.
$$

and

$$
\log _ { 0 } ( z ) = { \left\{ \begin{array} { l l } { \operatorname { a r t a n h } ( \left\| z \right\| ) { \frac { z } { \left\| z \right\| } } , } & { z \neq 0 , } \\ { 0 , } & { z = 0 . } \end{array} \right. }
$$

These origin based expressions are used in the main text for the tangent space analysis of continuous descriptor ratings.

## A.4 Parallel transport

Optimization on a Riemannian manifold requires comparing tangent vectors attached to different points. Since the tangent spaces $T _ { z } \mathbb { P } ^ { n }$ and $T _ { \mathrm { y } } \mathbb { P } ^ { n }$ are distinct for $z \neq y _ { i }$ , tangent vectors cannot be directly added. The appropriate operation is parallel transport, which moves a tangent vector along a geodesic while preserving the Riemannian geometry.

For $z , y \in \mathbb { P } ^ { n }$ and $\nu \in T _ { z } \mathbb { P } ^ { n }$ , parallel transport along the geodesic from z to y is (Ganea et al., 2018; Becigneul & Ganea, 2019)

$$
P _ { z  y } ( \nu ) = \frac { \lambda _ { z } } { \lambda _ { y } } \mathrm { g y r } [ y , - z ] \nu .
$$

Here, the gyration operator associated with the gyrovector formalism of hyperbolic geometry is defined by (Ungar, 2009):

$$
\operatorname { g y r } [ a , b ] \nu = - \left( a \oplus b \right) \oplus \left[ a \oplus \left( b \oplus \nu \right) \right] .
$$

Parallel transport is used during optimization to transfer the first moment estimate between successive embedding positions.

## A.5 Embeddings initialization

In order to initialize our embeddings, we need to sample points on the Poincaré disk. The hyperbolic space admits several generalizations of the Gaussian distribution, which are routinely deployed in statistical modeling and machine learning. For our purposes, we consider the pseudo-hyperbolic Gaussian (Nagano et al., 2019). The latter is obtained by first sampling points v on the tangent space $T _ { 0 } \mathbb { P } ^ { 2 }$ according to a standard Gaussian distribution, and then projecting these points onto the hyperbolic disk: for each observation i, we sample $a _ { i } \sim \mathcal { N } ( 0 , \sigma _ { \mathrm { i n i t } } ^ { 2 } I _ { 2 } )$ in $T _ { 0 } \mathbb { P } ^ { 2 }$ , and set $z _ { i } ^ { ( 0 ) } = \exp _ { 0 } ( a _ { i } )$ . In other words, the pseudo-hyperbolic Gaussian is the push-forward via the exponential map of the standard Gaussian distribution over the tangent space at the origin. This procedure produces valid initial points inside the open unit disk. Independent samples are used for each random initialization. We set $\sigma _ { \mathrm { i n i t } } = 0 . 1$ in all experiments.

## A.6 Riemannian Adam optimization

To optimize the embedding coordinates, we employ a Riemannian Adam optimizer on the Poincaré disk, following the framework of Becigneul & Ganea (2019) and instantiating it with the closed-form Poincaré expressions for the Riemannian gradient, exponentia map, and parallel transport. Our implementation further uses the standard Adam bias correction (Kingma & Ba, 2015) and an explicit projection back into the open unit disk for numerical stability as performed by Kochurov et al. (2020).

Unlike the usual coordinate-wise form of Euclidean Adam, in our Riemannian implementation following Becigneul & Ganea (2019), each embedding point $z _ { i } \in \mathbb { P } ^ { 2 }$ is associated with a single scalar second-moment estimate $\nu _ { i } ^ { ( t ) }$ , rather than separate second-moment estimates for its two coordinates. Let $z _ { i } ^ { ( t ) } \in \mathbb { P } ^ { 2 }$ denote the embedding of observation i at iteration t, and let $\nabla \mathcal { L } ( z _ { i } ^ { ( t ) } ) \in \mathbb { R } ^ { 2 }$ be the corresponding Euclidean gradient of the loss L defined in the Section 2.3. The associated Riemannian gradient is

$$
g _ { i } ^ { ( t ) } = \nabla _ { \mathbb { P } } \mathcal { L } ( z _ { i } ^ { ( t ) } ) = \frac { ( 1 - \| z _ { i } ^ { ( t ) } \| ^ { 2 } ) ^ { 2 } } { 4 } \nabla \mathcal { L } ( z _ { i } ^ { ( t ) } ) .
$$

Let $\beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$ be hyperparameters controlling the exponential moving averages of the first-moment and second-moment estimates, respectively. We then maintain a first-moment estimate $m _ { i } ^ { ( t ) } \in T _ { z _ { i } ^ { ( t ) } } \mathbb { P } ^ { 2 }$ and a scalar second-moment estimate $\nu _ { i } ^ { ( t ) } \in \mathbb { R } _ { + }$ updated as

$$
\begin{array} { c } { { m _ { i } ^ { ( t ) } = \ S _ { 1 } { \cal P } _ { z _ { i } ^ { ( t - 1 ) }  z _ { i } ^ { ( t ) } } ( m _ { i } ^ { ( t - 1 ) } ) + ( 1 - \ S _ { 1 } ) g _ { i } ^ { ( t ) } , } } \\ { { \nu _ { i } ^ { ( t ) } = \ S _ { 2 } { \nu } _ { i } ^ { ( t - 1 ) } + ( 1 - \ S _ { 2 } ) \| g _ { i } ^ { ( t ) } \| _ { z _ { i } ^ { ( t ) } } ^ { 2 } , } } \end{array}
$$

where $P _ { z _ { i } ^ { ( t - 1 ) }  z _ { i } ^ { ( t ) } } ( m _ { i } ^ { ( t - 1 ) } )$ is the previous first moment transported to the current tangent space via parallel transport from $T _ { z _ { i } ^ { ( t - 1 ) } } \mathbb { P } ^ { 2 }$ to $T _ { z _ { i } ^ { ( t ) } } \mathbb { P } ^ { 2 }$ in order to preserve the momentum across iterations, and

$$
\| g _ { i } ^ { ( t ) } \| _ { z _ { i } ^ { ( t ) } } ^ { 2 } = g _ { z _ { i } ^ { ( t ) } } ( g _ { i } ^ { ( t ) } , g _ { i } ^ { ( t ) } ) .
$$

Following the standard Adam bias correction, we define

$$
\widehat { m } _ { i } ^ { ( t ) } = \frac { m _ { i } ^ { ( t ) } } { 1 - \beta _ { 1 } ^ { t } } , ~ \widehat { \nu } _ { i } ^ { ( t ) } = \frac { \nu _ { i } ^ { ( t ) } } { 1 - \beta _ { 2 } ^ { t } } .
$$

The tangent update direction is then

$$
h _ { i } ^ { ( t ) } = - \boldsymbol { \eta } \frac { \widehat { m } _ { i } ^ { ( t ) } } { \sqrt { \widehat { \nu } _ { i } ^ { ( t ) } } + \mathfrak { E } } ,
$$

where $\boldsymbol \eta > 0$ is the learning rate and $\varepsilon > 0$ is a numerical stability constant. The embedding is updated intrinsically on the manifold via the exponential map:

$$
\begin{array} { r } { z _ { i } ^ { ( t + 1 ) } = \exp _ { z _ { i } ^ { ( t ) } } \Big ( h _ { i } ^ { ( t ) } \Big ) . } \end{array}
$$

Finally, for numerical stability, we project points back into the open unit disk whenever needed. Specifically, after each update we apply

$$
\Pi _ { \mathbb { P } ^ { 2 } } ( u ) = \left\{ \begin{array} { l l } { u , } & { \mathrm { i f ~ } \| u \| < 1 - \varepsilon _ { \mathrm { p r o j } } , } \\ { \big ( 1 - \varepsilon _ { \mathrm { p r o j } } \big ) \frac { u } { \| u \| } , } & { \mathrm { o t h e r w i s e } . , } \end{array} \right.
$$

Implementation details. For all our experiments, we set $\varepsilon _ { \mathrm { p r o j } } = 1 0 ^ { - 5 } , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \varepsilon = 1 0 ^ { - 8 }$ , and $\eta = 0 . 1$ . For the Sagar dataset, optimization was performed in full batch. Because of the larger size of the GSLF dataset, its embedding was optimized using mini-batches of 195 molecules. All configurations were trained for 1000 epochs. In each case, the optimization loss had reached a stable plateau by the end of training, indicating convergence.

## B Additional results

<table><tr><td>Descriptor</td><td>Radial Pearson</td><td>Radial Spearman</td><td>pws</td><td>POB</td></tr><tr><td>Intensity</td><td> $0 . 4 2 \pm 0 . 0 5$ </td><td> $0 . 3 8 \pm 0 . 0 5$ </td><td> $< 0 . 0 0 1$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td>Pleasantness</td><td> $0 . 1 0 \pm 0 . 1 0$ </td><td> $0 . 0 7 \pm 0 . 1 0$ </td><td>0.835</td><td>0.873</td></tr><tr><td>Fishy</td><td> $0 . 0 2 \pm 0 . 0 7$ </td><td> $- 0 . 1 4 \pm 0 . 0 6$ </td><td>0.657</td><td>0.665</td></tr><tr><td>Burnt</td><td> $- 0 . 0 4 \pm 0 . 0 7$ </td><td> $- 0 . 1 9 \pm 0 . 0 7$ </td><td>0.356</td><td>0.317</td></tr><tr><td>Sour</td><td> $0 . 0 5 \pm 0 . 0 6$ </td><td> $- 0 . 0 8 \pm 0 . 0 5$ </td><td>0.220</td><td>0.207</td></tr><tr><td>Decayed</td><td> $0 . 0 8 \pm 0 . 0 7$ </td><td> $- 0 . 1 1 \pm 0 . 0 7$ </td><td>0.051</td><td>0.044</td></tr><tr><td>Musky</td><td> $0 . 0 4 \pm 0 . 0 9$ </td><td> $- 0 . 0 6 \pm 0 . 0 9$ </td><td>0.966</td><td>0.968</td></tr><tr><td>Fruity</td><td> $0 . 0 9 \pm 0 . 1 2$ </td><td> $- 0 . 1 6 \pm 0 . 1 1$ </td><td>0.049</td><td>0.052</td></tr><tr><td>Sweaty</td><td> $0 . 0 9 \pm 0 . 0 9$ </td><td> $- 0 . 0 8 \pm 0 . 0 9$ </td><td>0.727</td><td>0.755</td></tr><tr><td>Cool</td><td> $- 0 . 1 1 \pm 0 . 0 9$ </td><td> $- 0 . 2 2 \pm 0 . 0 9$ </td><td>0.032</td><td>0.029</td></tr><tr><td>Floral</td><td> $- 0 . 1 2 \pm 0 . 1 2$ </td><td> $- 0 . 2 4 \pm 0 . 1 1$ </td><td>0.005</td><td>0.003</td></tr><tr><td>Sweet</td><td> $0 . 1 0 \pm 0 . 1 3$ </td><td> $- 0 . 0 2 \pm 0 . 1 3$ </td><td>0.091</td><td>0.167</td></tr><tr><td>Warm</td><td> $0 . 0 1 \pm 0 . 1 0$ </td><td> $- 0 . 0 6 \pm 0 . 1 1$ </td><td>0.952</td><td>0.929</td></tr><tr><td>Bakery</td><td> $0 . 0 2 \pm 0 . 1 5$ </td><td> $- 0 . 0 9 \pm 0 . 1 5$ </td><td>0.813</td><td>0.769</td></tr><tr><td>Spicy</td><td> $- 0 . 1 7 \pm 0 . 0 3$ </td><td> $- 0 . 3 0 { \pm } 0 . 0 3$ </td><td>0.037</td><td>0.058</td></tr></table>

Table S1: Full radial descriptor results for the Sagar union embedding. Radial Pearson and Spearman correlations quantify the association between each descriptor and the hyperbolic radius. Values are reported as mean and standard deviation over 10 random seeds. The restricted permutation p-values correspond to the within-subject permutation, p<sub>WS</sub>, and the odorant-block permutation, p<sub>OB</sub>.

<table><tr><td>Descriptor</td><td> $\mathsf { D i r e c t i o n a l } R ^ { 2 }$ </td><td> $p _ { \mathrm { { W S } } }$ </td><td>POB</td></tr><tr><td>Intensity</td><td> $0 . 0 4 \pm 0 . 0 5$ </td><td>0.003</td><td>0.005</td></tr><tr><td>Pleasantness</td><td> $0 . 4 0 \pm 0 . 0 9$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Fishy</td><td> $0 . 2 3 \pm 0 . 0 4$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Burnt</td><td> $0 . 1 9 \pm 0 . 0 5$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Sour</td><td> $0 . 0 3 \pm 0 . 0 4$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Decayed</td><td> $0 . 3 4 \pm 0 . 0 6$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Musky</td><td> $0 . 5 1 \pm 0 . 0 6$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Fruity</td><td> $0 . 4 6 \pm 0 . 0 9$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Sweaty</td><td> $0 . 1 3 \pm 0 . 0 6$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Cool</td><td> $0 . 1 2 \pm 0 . 1 0$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Floral</td><td> $0 . 2 5 \pm 0 . 1 1$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Sweet</td><td> $0 . 5 1 \pm 0 . 1 2$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Warm</td><td> $0 . 2 6 \pm 0 . 1 2$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Bakery</td><td> $0 . 2 5 \pm 0 . 1 3$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Spicy</td><td> $0 . 0 5 \pm 0 . 0 2$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr></table>

Table S2: Full angular descriptor results for the Sagar union embedding. Directional $R ^ { 2 }$ quantifies how well each descriptor is explained by a global linear direction in the tangent-space representation of the Poincaré disk. Values are reported as mean and standard deviation over 10 random seeds. The restricted permutation p-values correspond to the within-subject permutation, p<sub>WS</sub>, and the odorant-block permutation, p<sub>OB</sub>.

![](images/aa09f7e6c7e23716e17fed6c3c08643f76dee7fd8eb4fd3000b91a92f5f82b49.jpg)  
Figure S1: Directional organization of the 15 continuous descriptors in a representative Sagar union embedding. Each panel shows the embedding colored by the corresponding descriptor rating. Arrows indicate the fitted tangent space directions of increasing descriptor values. Here, θ denotes the directional angle, and $R ^ { 2 }$ quantifies the strength of the fitted directional trend The reported values are specific to the random initialization (m = 5).  
Alt text: Fifteen Poincaré disk panels show directional organization for each Sagar descriptor.