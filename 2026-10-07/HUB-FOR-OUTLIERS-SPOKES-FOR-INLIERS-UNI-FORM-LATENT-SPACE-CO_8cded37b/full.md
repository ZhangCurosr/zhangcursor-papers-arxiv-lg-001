# HUB FOR OUTLIERS, SPOKES FOR INLIERS: UNI-FORM LATENT SPACE CONSTRUCTION FOR DUAL-MISMATCHED SEMI-SUPERVISED LEARNING

Li Yuan<sup>1\*</sup>, Yaxin Hou<sup>1\*</sup>, Jiawei Tang<sup>1</sup>, Yongbiao Gao<sup>2,3</sup>, Yuheng Jia<sup>1,4,5†</sup>

<sup>1</sup>School of Computer Science and Engineering, Southeast University, Nanjing, China

<sup>2</sup>Key Laboratory of Computing Power Network and Information Security, Ministry of Education, Shandong Computer Science Center (National Supercomputer Center in Jinan), Qilu University of Technology (Shandong Academy of Sciences), Jinan, China

<sup>3</sup>Shandong Provincial Key Laboratory of Computing Power Internet and Service Computing, Shandong Fundamental Research Center for Computer Science, Jinan, China

<sup>4</sup>Key Laboratory of New Generation Artificial Intelligence Technology and Its   
Interdisciplinary Applications (Southeast University), Ministry of Education, China   
<sup>5</sup>School of Computing Information Sciences, Saint Francis University, Hong Kong, China   
{yuan-li,yaxin,230259148,yhjia}@seu.edu.cn,   
gaoyb@qlu.edu.cn

## ABSTRACT

Semi-supervised learning typically assumes that labeled and unlabeled data share an identical class distribution and label space. However, this setting is often violated: unlabeled data may be imbalanced and contain unknown class samples, causing mismatches in both class distribution and label space. Such dual mismatch leads to majority classes dominating the latent space and unknown class samples being overconfidently misclassified, degrading feature discriminability and pseudolabel quality. To address this, we propose a hub-spoke latent geometry, where known classes are uniformly distributed around a central hub and each class forms compact clusters around its prototype, while the hub provides an anchor for a lowevidence region specifically designed for high-uncertainty unknown class samples. Integrated with an evidence-based classifier, this geometry ultimately enhances feature discriminability and uncertainty separation by mitigating majority-class domination through structured feature organization and guiding high-uncertainty unknown class samples toward the hub. Extensive experiments show that our method outperforms state-of-the-art methods, with a maximum improvement of 3.25% across various settings.

## 1 INTRODUCTION

Semi-supervised learning (SSL) effectively leverages large amounts of unlabeled data to improve model performance when labeled data is scarce. Representative SSL techniques, such as pseudolabeling (Chen et al., 2018; Cascante-Bonilla et al., 2021), consistency regularization (Miyato et al., 2019; Ke et al., 2019), or a combination of both (Cheng et al., 2025; Du et al., 2024) have demonstrated considerable success by utilizing the information within unlabeled data. However, these methods typically rely on the assumption that the labeled and unlabeled data share both a consistent class distribution (i.e., both are balanced) and an identical label space (i.e., no unknown class samples exist in the unlabeled data). In practice, however, the unlabeled data often exhibits mismatches in both class distribution and label space (i.e., the unlabeled data is imbalanced and contains unknown class samples), as illustrated in the rightmost panel of Fig. 1. Existing SSL methods lack an effective strategy to handle these realistic and challenging settings and therefore suffer from significant performance degradation.

![](images/38a8296345e589dcecb95d5b57a6400f6f3a21d1434724381892f1d9e99ea32d.jpg)  
Figure 1: A comparison of the class distributions in semi-supervised learning (SSL), open-set semisupervised learning (OSSL), long-tailed semi-supervised learning (LTSSL), and the dual-mismatched semi-supervised learning (DMSSL) scenarios. DMSSL considers a more realistic scenario where the labeled data is balanced, while the unlabeled data is imbalanced and contains samples from both known and unknown classes.

While existing long-tailed semi-supervised learning (LTSSL) methods tackle class distribution mismatch through approaches like auxiliary classifiers (Hou et al., 2025; Ma et al., 2024; Hou & Jia, 2025; Hou et al., 2026) and distribution alignment (Kim et al., 2020; Aimar et al., 2024), they often overlook the underlying feature distribution. We empirically observe that imbalanced class distribution causes majority classes (e.g., class 5) to dominate the feature representation space. This dominance compresses the representation space of minority classes (e.g., class 3 and class 7), leading to overlapping class boundaries (Fig. 2b left). This degradation limits the model’s discriminative capability and leads to low-quality pseudo-labels. Meanwhile, to handle label space mismatch, existing methods rely heavily on softmax-based confidence scores to filter out unknown class samples to prevent negative interference (Saito et al., 2021; Yu et al., 2020) or leverage them positively (Li et al., 2023; Bae et al., 2025). However, such scores lead to overconfident misclassifications of unknown class samples as known class ones, thereby degrading pseudo-label accuracy (Fig. 2c).

The aforementioned challenges posed by dual mismatch motivate us to explore a solution that jointly improves known class discriminability and handles unknown classes. To this end, we explicitly organize known and unknown classes in a structured latent space by constructing a uniform hubspoke geometry (Fig. 2a), where uniformly distributed spokes represent prototypes for known classes and a central hub provides the anchor for a low-evidence region designed for high-uncertainty unknown classes. By encouraging known classes to spread uniformly around the hub and samples to cluster around their corresponding prototypes (Fig. 2b right), the geometry promotes inter-class separation and intra-class compactness, mitigating representation bias caused by class imbalance. Building upon this geometry, we further introduce an evidence-based classifier to quantify predictive uncertainty. Specifically, unlabeled samples with high uncertainty which exhibit low similarity to known classes are guided toward the central hub. Our theoretical analysis shows that these highuncertainty candidate samples, combined with hub-centered clustering, induce a low-evidence region around the hub, providing theoretical support for this uncertainty-guided modeling (see Sec. 5 for more details). Consequently, this modeling improves the separation between known and unknown classes in the feature space and their uncertainty distributions, as illustrated in Fig. 2d.

## Our main contributions can be summarized as follows:

• We identify and tackle the realistic dual-mismatched semi-supervised learning scenario, where the unlabeled data simultaneously suffers from imbalanced class distribution and the presence of unknown classes.

• We propose the hub-spoke geometry which enhances feature discriminability under class imbalance and provides a low-evidence region for unknown class samples, with a theoretical characterization showing how high-uncertainty candidates and hub-centered clustering induce this region.

• Extensive experiments on CIFAR-10, CIFAR-100, SVHN, Food-101, and ImageNet-100 datasets under various dual-mismatched conditions demonstrate that our proposed method achieves new state-of-the-art (SOTA) performance gains by up to 3.25%.

![](images/ff835a43ffdac4e35d78ba4bc5dc71ce73eee1fcab8c0d84c828759799d02956.jpg)  
(a)

![](images/2f2e81d52954ac0e5aca90a6212b94513c329411fa3cac23bae125748a8ce742.jpg)

![](images/b1d25f12d9065bb90a7f2911d8717aaa9ad410d6a8aafd1caa6d458a6b292768.jpg)  
(b)

![](images/c40af3082eb96a2fc1ab00b6d06b78060278238c31362f86cb621cfd1ba9c7f1.jpg)  
(c)

![](images/c39315d669e5287d88eb2a39f001dc0d2eb88b9da79f7260fc12c70e39902976.jpg)

![](images/89e62da047a5a8961cadbc60aa1b38b6a3e59176680fd598139367232546a446.jpg)  
(d)  
Figure 2: (a) presents the proposed hub-spoke geometry; (b) presents the t-SNE visualizations of CIFAR-10 (Krizhevsky et al., 2009) test set for Meta-Expert (Hou & Jia, 2025) and our proposed method, respectively; (c) shows a comparison of the pseudo-label accuracy on CIFAR-10 among IOMatch (Li et al., 2023), CaliMatch (Bae et al., 2025), and our method across three settings, with 4 (lb4), 10 (lb10), and 25 (lb25) labeled samples per class, respectively; (d) presents the uncertainty distributions on CIFAR-10 for the cases without guiding and with guiding high-uncertainty unknown class samples toward the central hub in the hub-spoke geometry, respectively.

## 2 RELATED WORK

## 2.1 LONG-TAILED SEMI-SUPERVISED LEARNING

Existing long-tailed semi-supervised learning (LTSSL) methods primarily aim to mitigate model bias toward majority classes during the pseudo-labeling process, and these methods can be broadly categorized into two groups. The first focuses on pseudo-label refinement and distribution alignment. For instance, DARP (Kim et al., 2020) refines biased pseudo-labels via convex optimization, while DASO (Oh et al., 2022) leverages the complementary properties of linear and semantic classifiers. Moreover, ACR (Wei & Gan, 2023) dynamically calibrates pseudo-labels based on the distance between the estimated unlabeled data distribution and anchor distributions, and CReST (Wei et al., 2021) employs a progressive distribution alignment strategy to adjust its rebalancing intensity. The second involves structural or strategic adjustments to enforce balance. ABC (Lee et al., 2021) introduces an auxiliary classifier to alleviate model bias toward majority classes, and CDMAD (Lee & Kim, 2024) designs a debiasing mechanism by using pattern-free images. More recently, methods such as CPE (Ma et al., 2024) train multiple experts, each specialized for a specific unlabeled data distribution, to generate higher-quality pseudo-labels. Building on this, Meta-Expert (Hou & Jia, 2025) proposes dynamic expert assignment to better incorporate individual expert strengths. Despite their effectiveness in mitigating bias under class imbalance, these methods exhibit limited performance in realistic scenarios where unknown class samples exist in the unlabeled data.

## 2.2 OPEN-SET SEMI-SUPERVISED LEARNING

A parallel research direction, known as open-set semi-supervised learning (OSSL), explicitly tackles scenarios where unlabeled data contains unknown class samples. OpenMatch (Saito et al., 2021) leverages one-vs-all classifiers to isolate unknown class samples from known class ones. IOMatch (Li et al., 2023) unifies unknown class samples into a new class via a multi-binary classifier, facilitating joint training of closed-set and open-set classifiers. Se-FOSS (Wallin et al., 2024) utilizes selfsupervised learning techniques and energy-based scores to learn from all known and unknown class samples, while SCOMatch (Wang et al., 2024) explicitly selects reliable unknown class samples to form an additional class and refines the decision boundary through simultaneous closed-set and open-set self-training. Most recently, CaliMatch (Bae et al., 2025) addresses the critical issue of overconfidence by simultaneously calibrating the classifier and unknown class sample detector.

However, these methods assume a balanced class distribution and consequently perform poorly when unlabeled data is imbalanced.

Further discussion and analysis of related work is provided in Appendix A.

## 3 PRELIMINARIES

Problem formulation. This paper studies semi-supervised learning involving mismatches in both class distribution and label space between labeled and unlabeled data, which we term dualmismatched semi-supervised learning (DMSSL). In DMSSL, we consider both a labeled dataset $\mathcal { D } _ { l } = \{ ( x _ { i } ^ { l } , y _ { i } ^ { l } ) \} _ { i = 1 } ^ { N }$ and an unlabeled dataset $\mathcal { D } _ { u } = \{ ( x _ { j } ^ { u } , y _ { j } ^ { u } ) \} _ { j = 1 } ^ { M }$ . Here, $x _ { i } ^ { l }$ denotes the i-th labeled sample with its ground-truth label $y _ { i } ^ { l } \in \{ 1 , \ldots , C \}$ , while $x _ { j } ^ { u }$ represents the j-th unlabeled sample with its inaccessible ground-truth label $y _ { i } ^ { u } \in \{ 1 , \dots , C , \dots , C _ { I O } \}$ , where $C _ { I O } = C + C _ { O }$ is the total number of classes, including $C$ known and $C _ { O }$ unknown classes. $\mathcal { D } _ { u }$ can be decomposed as $\mathcal { D } _ { u } = \mathcal { D } _ { I } \cup \mathcal { D } _ { O }$ , where $\mathcal { D } _ { I }$ and $\mathcal { D } _ { O }$ contain known and unknown class samples, respectively. Further more, the labeled data $\mathcal { D } _ { l }$ follow a balanced class distribution, whereas the unlabeled data $\mathcal { D } _ { u }$ follow an imbalanced class distribution. Let $N _ { c }$ denote the number of labeled samples in class c, defining the labeled data imbalance ratio as $\begin{array} { r } { \gamma _ { l } = \frac { \operatorname* { m a x } _ { c } N _ { c } } { \operatorname* { m i n } _ { c } N _ { c } } = 1 } \end{array}$ . Theoretically, we can assume that $M _ { c }$ represents the number of samples for class c in $\mathcal { D } _ { u }$ , and define its imbalance ratio as $\begin{array} { r } { \gamma _ { u } = \frac { \operatorname* { m a x } _ { c } M _ { c } } { \operatorname* { m i n } _ { c } M _ { c } } > 1 } \end{array}$

Base semi-supervised learning framework. Common semi-supervised learning methods can be summarized as follows. Given a batch of $B _ { l }$ labeled samples $\mathcal { X } ^ { l } ~ = ~ \{ ( x _ { i } ^ { l } , y _ { i } ^ { \bar { l } } ) \} _ { i = 1 } ^ { B _ { l } }$ , a weak augmentation $\Omega _ { w } ( \cdot )$ is applied to each sample before being fed to an encoder $\Phi ( \cdot )$ that extracts feature representations $\boldsymbol { z } = \Phi ( \Omega _ { w } ( \boldsymbol { x } ) )$ , which are mapped to logits by a fully-connected layer $f ( \cdot )$ to produce class probability distributions $p = f ( z )$ . The model’s predictions on these labeled samples are supervised by the standard cross-entropy loss $\mathcal { H } ( \cdot )$ against their ground-truth labels:

$$
\mathcal { L } _ { s u p } = \frac { 1 } { B _ { l } } \sum _ { i = 1 } ^ { B _ { l } } \mathcal { H } ( p _ { i } , y _ { i } ) .\tag{1}
$$

Given a batch of $B _ { u }$ samples $\mathcal { X } ^ { u } = \{ x _ { j } ^ { u } \} _ { j = 1 } ^ { B _ { u } }$ , each sample is transformed by both a weak augmentation $\Omega _ { w } ( \cdot )$ and a strong augmentation $\Omega _ { s } ( \cdot )$ , and the resulting two views are then processed to yield class probability distributions $p ^ { w }$ and $p ^ { s } ,$ , respectively. The unsupervised loss is then defined as the cross-entropy loss $\mathcal { H } ( \cdot )$ between the prediction from the strong augmentation view and the pseudo-label generated from the weak augmentation view:

$$
\mathcal { L } _ { u n s u p } = \frac { 1 } { B _ { u } } \sum _ { j = 1 } ^ { B _ { u } } \mathbb { I } ( \operatorname* { m a x } ( p _ { j } ^ { w } ) > \tau ) \mathcal { H } ( p _ { j } ^ { s } , \hat { p } _ { j } ^ { w } ) ,\tag{2}
$$

where $\hat { p } _ { j } ^ { w }$ is the pseudo-label for the $j \cdot$ -th unlabeled sample in the current batch, $\tau$ is the confidence threshold, and $\mathbb { I } ( \cdot )$ denotes a binary sample mask that selects pseudo-labels with confidence larger than τ. This ensures that only high-confidence pseudo-labels contribute to the training process.

## 4 PROPOSED METHOD

We address the dual-mismatched semi-supervised learning scenario where unlabeled data suffers from both class imbalance and the presence of unknown class samples. Our method constructs a discriminative feature representation space via a hub-spoke geometry. Meanwhile, it identifies high-uncertainty unknown class samples by leveraging evidence-based uncertainty and models them by guiding their features toward the central hub.

## 4.1 HUB-SPOKE GEOMETRY CONSTRUCTION

As illustrated in Fig. 2b left, we observe that imbalanced class distribution leads to majority classes (e.g., class 5) dominating the feature representation space and compressing the feature representation space of minority classes $( \mathrm { e . g . }$ , class 3 and class 7), thereby causing overlapping class boundaries. This limits feature discriminability and leads to low-quality pseudo-labels. To counteract this, we propose both a dual regularizer that structures the feature representation space into a hub-spoke geometry from global and local perspectives, and a class-adaptive re-weighting strategy to dynamically focus on minority classes.

Global structure regularization. To prevent majority classes from dominating the feature representation space and amplifying model bias, we enforce a globally uniform feature distribution, where all known classes are uniformly distributed around a central hub. We achieve this through two losses designed to explicitly shape the global geometry.

First, an inter-class separation loss $\mathcal { L } _ { i q }$ is introduced to directly maximize the angular distances between labeled class prototypes. Formally, $\mathcal { L } _ { i g }$ is defined as:

$$
\mathcal { L } _ { i g } = \frac { 1 } { | C | ( | C | - 1 ) } \sum _ { i = 1 } ^ { | C | } \sum _ { j = 1 \atop j \neq i } ^ { | C | } \left( \frac { \boldsymbol { \mu } _ { i } ^ { l } \cdot \boldsymbol { \mu } _ { j } ^ { l } } { \| \boldsymbol { \mu } _ { i } ^ { l } \| _ { 2 } \| \boldsymbol { \mu } _ { j } ^ { l } \| _ { 2 } } + 1 \right) ^ { 2 } ,\tag{3}
$$

where $\mu _ { i } ^ { l }$ denotes the labeled class prototype for the i-th class, · denotes the dot product, and $\| \cdot \| _ { 2 }$ denotes the $L _ { 2 }$ norm.

Second, a prototype geometry loss $\mathcal { L } _ { o g }$ regulates the spatial arrangement of all labeled class prototypes relative to the global labeled class prototype $\begin{array} { r } { \bar { \mu } ^ { l } = \frac { 1 } { \vert C \vert } \sum _ { i = 1 } ^ { \vert C \vert } \mu _ { i } ^ { l } } \end{array}$ . Formally, $\mathcal { L } _ { o g }$ is defined as:

$$
\mathcal { L } _ { o g } = - \log \left( \frac { 1 } { | C | } \sum _ { i = 1 } ^ { | C | } d _ { i } \right) + \frac { 1 } { | C | } \sum _ { i = 1 } ^ { | C | } ( d _ { i } - \hat { d } ) ^ { 2 } ,\tag{4}
$$

where $d _ { i } = \lVert \boldsymbol { \mu } _ { i } ^ { l } - \bar { \boldsymbol { \mu } } ^ { l } \rVert _ { 2 }$ is the distance from the i-th labeled class prototype to the global labeled class prototype, and $\begin{array} { r } { \bar { d } = \frac { 1 } { | C | } \sum _ { i = 1 } ^ { | C | } } \end{array}$ d is the average distance.

Accordingly, the global structure loss $\mathcal { L } _ { g }$ integrates these two loss items and is formulated as:

$$
\begin{array} { r } { \mathcal { L } _ { g } = \mathcal { L } _ { i g } + \mathcal { L } _ { o g } . } \end{array}\tag{5}
$$

Local structure regularization. To enforce local compactness and uniformity, we ensure that samples cluster tightly and uniformly around their respective overall class prototypes (i.e., calculated on labeled and pseudo-labeled data). This is achieved through two losses.

The intra-local loss $\mathcal { L } _ { i l }$ maximizes the similarity between samples and their corresponding overall class prototypes while minimizing similarity to the overall class prototypes of all other classes, enhancing intra-class compactness. Formally, $\mathcal { L } _ { i l }$ is defined as:

$$
\mathcal { L } _ { i l } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp ( z _ { i } \cdot \mu _ { y _ { i } } / v ) } { \sum _ { j = 1 , j \neq y _ { i } } ^ { C } \exp ( z _ { i } \cdot \mu _ { j } / v ) } ,\tag{6}
$$

where $z _ { i }$ is the feature representation of $x _ { i }$ extracted by the encoder $\Phi ( \cdot ) , \mu _ { y _ { i } }$ is the overall class prototype for class $y _ { i }$ , v is the temperature parameter, and exp(·) is the exponential function.

The radial uniformity loss $\mathcal { L } _ { o l }$ promotes a uniform feature distribution around each labeled class prototype by minimizing pairwise cosine similarities between normalized feature representation directions within a class, preventing collapse into a narrow region. Formally, $\mathcal { L } _ { o l }$ is defined as:

$$
\mathcal { L } _ { o l } = \frac { 1 } { | C | } \sum _ { c = 1 } ^ { | C | } \bigg [ \frac { 1 } { N _ { c } ( N _ { c } - 1 ) } \sum _ { \substack { i = 1 } } ^ { N _ { c } } \sum _ { j = 1 , \atop j \neq i } ^ { N _ { c } } ( r _ { c , i } \cdot r _ { c , j } ) \bigg ] ^ { 2 } ,\tag{7}
$$

where $r _ { c , i } = ( z _ { c , i } - \mu _ { c } ^ { l } ) / \| z _ { c , i } - \mu _ { c } ^ { l } \| _ { 2 }$ is the normalized feature representation direction of sample $x _ { i }$ in class $c ,$ and $z _ { c , i }$ is the feature representation of sample $x _ { i }$ belonging to class c.

Accordingly, the local structure loss $\mathcal { L } _ { l }$ integrates these two loss items and is formulated as:

$$
\mathcal { L } _ { l } = \mathcal { L } _ { i l } + \mathcal { L } _ { o l } .\tag{8}
$$

Class-adaptive re-weighting. While the dual regularizer structures the feature representation space into the hub-spoke geometry and improves its discriminability, enhancing the feature discriminability for minority classes remains a critical challenge in long-tailed semi-supervised learning. Fortunately, our empirical analysis reveals that samples from minority classes exhibit a significantly larger deviation between the overall class prototype and the labeled class prototype, as shown in Fig. 3. This observation motivates us to leverage this deviation for a class-adaptive re-weighting strategy that dynamically prioritizes samples from minority classes. Specifically, for each sample, we compute the cosine similarity between the displacement from the overall class prototype to the sample feature representation $( { \mathrm { i . e . , } } q _ { i } = z _ { i } - \mu _ { y _ { i } } )$ and the overall deviation vector of the class $( \Delta _ { y _ { i } } = \mu _ { y _ { i } } - \mu _ { y _ { i } } ^ { l } )$ . This similarity score is then used to weight the unsupervised loss in Eq. 2. Formally, the weight $w _ { i }$ for sample $x _ { i }$ is defined as:

![](images/1c557cd977fd838e409d715fb28c38e89cd195fe65dab30230800ac21ef9bd30.jpg)  
Figure 3: A comparison of the mean deviation distance between the overall class prototype and the labeled class prototype.

$$
w _ { i } = 1 + \operatorname* { m a x } \left( 0 , \frac { q _ { i } \cdot \Delta _ { y _ { i } } } { \| q _ { i } \| _ { 2 } \cdot \| \Delta _ { y _ { i } } \| _ { 2 } } \right) ,\tag{9}
$$

where $\operatorname* { m a x } ( { \mathord { \cdot } } )$ denotes the maximum function. Since the deviation is inherently larger for minority classes, samples belonging to minority classes receive higher weights, thereby adaptively shifting the model’s focus toward these classes.

## 4.2 UNKNOWN CLASS IDENTIFICATION AND MODELING

To effectively identify high-uncertainty samples from unknown classes, we employ an evidence-based classifier $g ( \cdot )$ for uncertainty quantification. Given a sample $x ,$ the classifier produces a non-negative evidence vector $e ,$ where each component $e _ { c }$ indicates the support for classifying the sample into class $c .$ This vector parameterizes a Dirichlet distribution $\operatorname { D i r } ( \pi | \alpha )$ over the class probabilities, with the concentration parameters defined as $\alpha = e + 1$ . The predictive uncertainty is derived from the Dirichlet strength $\begin{array} { r } { S = \sum _ { c = 1 } ^ { C } \alpha _ { c } , } \end{array}$ , with the uncertainty defined as $u = C / S$ . A high value of u corresponds to low overall evidence, suggesting that the sample likely belongs to an unknown class.

Unknown class identification. During training, we leverage this uncertainty measure to identify potential high-uncertainty unknown class samples. For each batch, we rank samples by their uncertainty u in descending order and select the top-k most uncertain samples as candidates from unknown classes, where k is set to 10% of the current batch size (see Appendix D for the hyperparameter sensitivity analysis on $k )$ . Concretely, let $u _ { k }$ be the k-th largest uncertainty in the batch. We then form an evidence-based mask $\mathbb { I } ( u \geq u _ { k } )$ to select unknown class sample candidates, thus constructing the identified high-uncertainty unknown class sample set $\hat { D } _ { O , k }$ , which serves as the foundation for subsequent unknown class modeling.

To ensure the evidence-based classifier produces well-calibrated uncertainty estimates, we optimize it using an evidence loss $\mathcal { L } _ { e d l }$ that regularizes the evidence distribution to align with ground-truth labels while maintaining uncertainty awareness:

$$
\mathcal { L } _ { e d l } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left[ \sum _ { j = 1 } ^ { C } y _ { i j } \left( \psi ( S _ { i } ) - \psi ( \alpha _ { i j } ) \right) + \lambda _ { k l } \cdot \mathrm { K L } ( \mathrm { D i r } ( \pi _ { i } | \tilde { \alpha } _ { i } ) | | \mathrm { D i r } ( \pi _ { i } | \mathbf { 1 } ) ) \right] ,\tag{10}
$$

where $\psi ( \cdot )$ is the digamma function and is defined as the logarithmic derivative of the gamma function, $\alpha _ { i j }$ denotes the concentration parameter for class $j$ of sample $i ,$ and $\begin{array} { r } { \lambda _ { k l } = \operatorname* { m i n } ( 0 . 6 , \operatorname* { m a x } ( 0 , \frac { t _ { e p } } { 1 0 } ) ) } \end{array}$ where $t _ { e p }$ denotes the number of epochs since $\mathcal { L } _ { e d l }$ begins to be optimized during training. The adjusted parameters $\tilde { \alpha } _ { i } = y _ { i } + ( { \bf 1 } - y _ { i } ) \cdot \alpha _ { i }$ are obtained by removing the non-misleading evidence from the original parameters $\alpha _ { i }$

Unknown class modeling. Building upon the established hub-spoke geometry and the identified highuncertainty unknown class sample set $\hat { D } _ { O , k }$ , we introduce an unknown class sample clustering loss $\mathcal { L } _ { o c }$ to explicitly model their feature distribution. The global labeled class prototype $\bar { \mu } ^ { l }$ , which serves as the hub, is approximately equidistant from all known class prototypes and well separated from the corresponding spokes, making it a natural anchor for unknown class modeling. Without explicit unknown class modeling, unknown class samples may be dispersed among different known class spokes, causing feature and evidence overlap with known class samples. We therefore regularize the representations of the identified high-uncertainty unknown class samples by pulling them toward the hub. Since these samples exhibit low evidence by construction, guiding them toward the hub induces a low-evidence region around the hub, indicating the absence of known-class evidence (see Appendix 5 for theoretical analysis). As the hub remains geometrically separated from all known class spokes, this design not only preserves the discriminability of known classes but also improves the separation between known and unknown classes in the feature space and their uncertainty distributions, as illustrated in Fig. 2d. Formally, $\mathcal { L } _ { o c }$ is defined as:

$$
\mathcal { L } _ { o c } = \frac { 1 } { | \hat { D } _ { O , k } | } \sum _ { i = 1 } ^ { | \hat { D } _ { O , k } | } \Vert z _ { i } ^ { o } - \bar { \mu } ^ { l } \Vert _ { 2 } ^ { 2 } ,\tag{11}
$$

where $\hat { D } _ { O , k }$ denotes the set of identified high-uncertainty unknown class samples and $z _ { i } ^ { o }$ represents the feature representation of the i-th identified unknown class sample.

## 4.3 END-TO-END TRAINING PROCESS

Our training process is structured into three phases to ensure stable learning. First, a warm-up phase lasting sixty epochs focuses on learning stable initial feature representations using $\mathcal { L } _ { w a r m 1 } .$

$$
\begin{array} { r } { \mathcal { L } _ { w a r m 1 } = \mathcal { L } _ { l c o n } + \lambda \left( \mathcal { L } _ { g } + \mathcal { L } _ { e c o n } \right) , } \end{array}\tag{12}
$$

where $\mathcal { L } _ { l c o n }$ and $\mathcal { L } _ { e c o n }$ denote the contrastive losses between the weak and strong augmentation views for the classification head $f ( \cdot )$ and the evidence head $g ( \cdot )$ , respectively.

Subsequently, a second warm-up phase lasting twenty epochs integrates the supervised and unsupervised loss to establish baseline performance and stabilize pseudo-labeling, guided by $\mathcal { L } _ { w a r m 2 } \mathrm { : }$

$$
\mathcal { L } _ { w a r m 2 } = \mathcal { L } _ { s u p } + \mathcal { L } _ { u n s u p } + \lambda \cdot \mathcal { L } _ { e c o n } .\tag{13}
$$

Finally, the main training phase leverages the full objective function to jointly address mismatches in both class distribution and label space:

$$
\mathcal { L } _ { o v e r a l l } = \mathcal { L } _ { s u p } + \mathcal { L } _ { u n s u p } + \lambda \left( \mathcal { L } _ { e c o n } + \mathcal { L } _ { g } + \mathcal { L } _ { l } + \mathcal { L } _ { e d l } + \mathcal { L } _ { o c } \right) .\tag{14}
$$

Here, λ is a balancing hyperparameter set to 0.1 across all three phases. The class-adaptive weight w<sub>i</sub> from Eq. 9 is incorporated into the $\mathcal { L } _ { u n s u p }$ term to adaptively focus on minority classes. Additionally, the confidence-based mask in Eq. 2 is replaced by the combination of the confidence-based mask and the negated evidence-based mask. The pseudo-code is detailed in Algorithm 1 in Appendix B.

## 5 THEORETICAL ANALYSIS

We analyze how uncertainty-guided unknown class modeling induces a low-evidence region around the hub. Let $\begin{array} { r } { \bar { \mu } ^ { l } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \bar { \mu } _ { c } ^ { l } } \end{array}$ denote the hub. Following Sec. 4.2, $\alpha _ { c } ( z ) = 1 + g _ { c } ( z ) , S ( z ) =$ $\begin{array} { r } { C + \sum _ { c = 1 } ^ { C } g _ { c } ( z ) , u ( z ) = \frac { C } { S ( z ) } } \end{array}$ , where $g _ { c } ( \cdot )$ indicates $e _ { c }$

Lemma 1 (Low evidence of selected candidates). Let $\widehat { \mathcal { D } } _ { O , k }$ be the top-k most uncertain candidates in a training batch, and let $u _ { k }$ be the k-th largest uncertainty in the batch. Then,for every $z _ { i } ^ { o } \in \widehat { D } _ { O , k }$

$$
g _ { c } ( z _ { i } ^ { o } ) \leq \varepsilon _ { O , k } , \qquad \varepsilon _ { O , k } : = C \left( \frac { 1 } { u _ { k } } - 1 \right) , \quad c = 1 , \ldots , C .\tag{15}
$$

Proof. Since $u ( z _ { i } ^ { o } ) \geq u _ { k }$

$$
S ( z _ { i } ^ { o } ) = \frac { C } { u ( z _ { i } ^ { o } ) } \leq \frac { C } { u _ { k } } .\tag{16}
$$

Hence

$$
\sum _ { j = 1 } ^ { C } g _ { j } ( z _ { i } ^ { o } ) \leq C \left( \frac { 1 } { u _ { k } } - 1 \right) .\tag{17}
$$

Using $g _ { c } ( z _ { i } ^ { o } ) \geq 0$ gives Eq. 15.

Theorem 1 (Low-evidence region around the hub). Suppose that the evidencefunction $g _ { c } ( \cdot )$ $L _ { c ^ { - } }$ Lipschitz, i.e., $| g _ { c } ( z _ { i } ) - g _ { c } ( z _ { j } ) | \leq L _ { c } \| z _ { i } - z _ { j } \| _ { 2 }$ for any $z _ { i } , z _ { j }$ . For $\mathcal { H } ( r ) = \big \{ z : \| z - \bar { \mu } ^ { l } \| _ { 2 } \leq r \big \}$ we have

$$
\operatorname* { s u p } _ { z \in \mathcal { H } ( r ) } g _ { c } ( z ) \leq \varepsilon _ { O , k } + L _ { c } \left( r + \sqrt { L _ { o c } } \right) .\tag{18}
$$

Proof. Let $n _ { O , k } = | \widehat { \mathcal { D } } _ { O , k } |$ . From Eq. 11, we have $\begin{array} { r } { L _ { o c } = \frac { 1 } { n _ { O , k } } \sum _ { i = 1 } ^ { n _ { O , k } } \| z _ { i } ^ { o } - \bar { \mu } ^ { l } \| _ { 2 } ^ { 2 } } \end{array}$ . Since $L _ { o c }$ is the average squared distance to the hub, there exists $z _ { * } ^ { o } \in \widehat { \cal D } _ { O , k }$ such that $\| z _ { * } ^ { o } - \bar { \mu } ^ { l } \| _ { 2 } ^ { 2 } \leq L _ { o c } ,$ , and hence

$$
\| z _ { * } ^ { o } - \bar { \mu } ^ { l } \| _ { 2 } \leq \sqrt { L _ { o c } } .\tag{19}
$$

For any $z \in \mathcal { H } ( r )$

$$
\| z - z _ { * } ^ { o } \| _ { 2 } \leq \| z - \bar { \mu } ^ { l } \| _ { 2 } + \| z _ { * } ^ { o } - \bar { \mu } ^ { l } \| _ { 2 } \leq r + \sqrt { L _ { o c } } .\tag{20}
$$

Since $g _ { c }$ is $L _ { c } .$ -Lipschitz continuous, Lemma 1 gives

$$
g _ { c } ( z ) \leq g _ { c } ( z _ { * } ^ { o } ) + L _ { c } \| z - z _ { * } ^ { o } \| _ { 2 } \leq \varepsilon _ { O , k } + L _ { c } \left( r + \sqrt { L _ { o c } } \right) .\tag{21}
$$

Taking the supremum over $z \in \mathcal { H } ( r )$ yields Eq. 18.

Hence, high uncertainty among the selected candidates and strong hub-centered clustering yield a quantitatively low-evidence hub region and correspondingly high evidential uncertainty.

## 6 EXPERIMENTS

Datasets. We evaluate our method on five benchmarks (i.e., CIFAR-10 (Krizhevsky et al., 2009), CIFAR-100 (Krizhevsky et al., 2009), SVHN (Netzer et al., 2011), Food-101 (Bossard et al., 2014), and ImageNet-100 (Deng et al., 2009)). To simulate class distribution mismatch, we construct a balanced labeled dataset with different sizes (i.e., 10 and 25 samples per class), while the unlabeled dataset follows an imbalanced class distribution. For label space mismatch, we introduce unknown class samples into the unlabeled data. Concretely, we add 10 unknown classes from ImageNet-127 (Deng et al., 2009) to the CIFAR-10 and SVHN datasets, 50 from ImageNet-127 to the CIFAR-100 and Food-101 datasets, and 50 from CIFAR-100 to the ImageNet-100 dataset.

Baselines. We compare our method with three SSL algorithms, including FixMatch (Sohn et al., 2020), FreeMatch (Wang et al., 2023), and SoftMatch (Chen et al., 2023); four LTSSL algorithms, including SimPro (Du et al., 2024), CDMAD (Lee & Kim, 2024), Meta-Expert (Hou & Jia, 2025), and SC-SSL (Tian et al., 2026); and six OSSL algorithms, including OpenMatch (Saito et al., 2021), IOMatch (Li et al., 2023), SCOMatch (Wang et al., 2024), CaliMatch (Bae et al., 2025), ANEDL (Yu et al., 2024), and GGR (Chen et al., 2026). Moreover, we use the supervised learning (SL) setting as an upper-bound reference for performance. Implementation details are provided in Appendix C.

## 6.1 EXPERIMENTAL RESULTS AND ANALYSES

Main Results. Table 1 presents the experimental results on CIFAR-10, CIFAR-100, SVHN, Food-101, and ImageNet-100 datasets under varying known/unknown classes and labeled set sizes. As illustrated in Table 1, our method consistently outperforms all baseline methods and is robust to dual mismatch. On CIFAR-10, we outperform the recent method CaliMatch by 20.22% and 11.40% with 10 and 25 labeled samples per class, respectively. Moreover, the performance gain persists on CIFAR-100, where we achieve a 3.25% performance gain over the best-performing baseline FixMatch, with 10 labeled samples per class. On SVHN, while the performance gap narrows due to the dataset’s lower complexity and clearer known and unknown class distinction, our method still achieves competitive performance.

On the more challenging Food-101 dataset, our method substantially outperforms Meta-Expert by 2.08% and 2.55%, respectively, despite the overall low accuracy caused by its fine-grained and numerous categories. Finally, on the highly complex ImageNet-100, with 10 labeled samples per class, our method achieves 27.52%, surpassing CaliMatch by a substantial 12.35%; with 25 labeled samples per class, we outperform the previous best FreeMatch by 0.52%. Notably, our method exhibits a lower standard deviation across three runs compared to most baseline methods. Further experimental results and analyses are provided in Appendix D.

Table 1: Classification accuracy (%) on the test data of CIFAR-10, CIFAR-100, SVHN, Food-101, and ImageNet-100 with varying known/unknown classes and labeled set sizes. We report the mean with standard deviation over three runs with different random seeds.
<table><tr><td colspan="2">Dataset</td><td colspan="2">CIFAR-10</td><td colspan="2">CIFAR-100</td><td colspan="2">SVHN</td><td colspan="2">Food-101</td><td colspan="2">ImageNet-100</td></tr><tr><td colspan="2">Known / Unknown</td><td colspan="2">10 / 10</td><td colspan="2">100/ 50</td><td colspan="2">10 / 10</td><td colspan="2">101 / 50</td><td colspan="2">100 / 50</td></tr><tr><td colspan="2">Labels per class</td><td>10</td><td>25</td><td>10</td><td>25</td><td>10</td><td>25</td><td>10</td><td>25</td><td>10</td><td>25</td></tr><tr><td rowspan="2">SL</td><td>CE</td><td>78.69± 5.10</td><td>80.21± 4.85</td><td>51.01± 0.73 55.74± 0.57</td><td></td><td>88.93± 1.19</td><td>89.18± 1.05</td><td>39.79± 0.4842.56± 0.3844.63± 0.2746.61± 1.16</td><td></td><td></td><td></td></tr><tr><td>LA</td><td>85.45± 2.64</td><td>86.08± 2.47</td><td>56.20± 0.48</td><td>59.67± 0.45</td><td>92.31± 1.01</td><td>92.63± 1.19</td><td>45.48± 0.37</td><td>47.44± 0.20</td><td></td><td>49.61± 0.25 51.37± 0.86</td></tr><tr><td rowspan="4">TSS</td><td>FixMatch</td><td>69.57± 5.03</td><td>76.86± 4.69</td><td>39.40± 1.31</td><td>51.23± 1.29</td><td>89.71± 2.60</td><td>90.82± 1.02</td><td>15.95± 0.55</td><td>23.23± 0.25 27.44± 0.81 34.95± 0.76</td><td></td><td></td></tr><tr><td>FreeMatch</td><td>64.45± 6.47</td><td>75.72± 5.27</td><td>32.56±1.62</td><td>47.52± 1.21</td><td>89.10± 2.30</td><td>89.83± 2.54</td><td>19.64± 0.34</td><td></td><td></td><td>27.65± 0.20 27.39± 0.4536.31± 0.97</td></tr><tr><td>SoftMatch</td><td>67.97± 5.22</td><td>74.56± 2.94</td><td>34.22± 1.82</td><td>48.74± 0.46</td><td>86.34± 1.30</td><td>87.44± 2.05</td><td>19.68± 0.57</td><td></td><td></td><td>27.71± 1.07 27.13± 1.48 36.01± 1.01</td></tr><tr><td>SimPro</td><td>26.19±3.62</td><td></td><td></td><td></td><td>59.60± 5.27</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">TTSSST</td><td>CDMAD</td><td>58.77± 8.60</td><td></td><td>68.72± 4.87 35.10± 1.24 49.68± 1.29</td><td></td><td></td><td>83.89± 3.17</td><td>13.06± 1.23</td><td></td><td></td><td>25.18± 0.4526.19± 0.4636.14± 0.39</td></tr><tr><td>Meta-Expert</td><td>57.37±7.89</td><td></td><td>70.38± 4.41 24.47± 2.37 31.35± 3.08</td><td></td><td>81.35±3.20</td><td>82.54± 2.09</td><td>6.35± 1.25</td><td>8.31± 2.47</td><td></td><td>14.52± 2.4917.72± 1.43</td></tr><tr><td>SC-SSL</td><td>67.26± 7.13</td><td>69.83± 5.37</td><td>32.05± 1.54 47.54± 1.13</td><td></td><td>79.00± 2.66</td><td>86.38± 1.14</td><td>17.74± 0.39</td><td></td><td></td><td>25.32± 0.5927.08± 0.3734.40± 0.34</td></tr><tr><td></td><td></td><td></td><td>77.23± 4.41 35.23± 1.6349.03± 0.66</td><td></td><td>89.05± 3.71</td><td>89.63± 2.11</td><td>17.04± 0.23</td><td></td><td></td><td>23.95± 0.12 26.97± 1.18 35.37± 1.0</td></tr><tr><td>OpenMatch</td><td>47.21± 5.06</td><td></td><td>68.13± 3.54 32.81± 0.1947.31± 0.44</td><td></td><td>84.88± 5.73</td><td>90.47± 0.92</td><td>10.61± 0.92</td><td>16.00± 0.39</td><td></td><td>19.29± 0.83 25.74± 1.34</td></tr><tr><td rowspan="5">OSSSO</td><td>IOMatch SCOMatch</td><td>67.33± 5.06</td><td></td><td>77.23± 3.11 38.90± 1.50 50.62± 1.12</td><td></td><td>87.98± 2.87</td><td>90.80± 1.50</td><td>7.62± 0.37</td><td>14.46± 0.73</td><td>9.04± 6.76</td><td>21.92± 0.04</td></tr><tr><td></td><td>63.41± 6.06</td><td>74.07± 3.41</td><td>30.57± 2.9047.42± 1.81</td><td></td><td>88.52± 2.86</td><td>90.74± 0.94</td><td>13.14±2.65</td><td>16.03± 2.88</td><td></td><td>17.36± 2.36 22.04± 1.31</td></tr><tr><td>CaliMatch</td><td>51.33± 10.72</td><td></td><td>67.80± 0.16 35.99± 6.00 50.86± 5.36</td><td></td><td>55.55± 14.84</td><td>90.71± 1.66</td><td>7.22± 1.95</td><td></td><td></td><td>10.22± 5.44 15.17± 1.89 25.78± 3.46</td></tr><tr><td>ANEDL</td><td>27.58± 0.79</td><td>48.21± 4.01</td><td>22.87± 3.54 37.26± 3.59</td><td></td><td>56.30± 1.85</td><td>76.12± 0.64</td><td>8.77± 0.69</td><td></td><td>15.37± 1.8712.51± 1.56 20.23± 0.65</td><td></td></tr><tr><td>GGR Ours</td><td>61.24± 6.81</td><td>69.70± 6.69</td><td>25.69± 1.4341.67± 2.63</td><td></td><td>88.21± 0.96</td><td>88.67± 1.54</td><td>7.25± 0.66</td><td></td><td>13.47± 0.9813.20± 0.48 20.35± 0.91</td><td></td></tr></table>

## 6.2 ABLATION ANALYSES AND DISCUSSIONS

The effect of each module. We conduct comprehensive ablation studies on CIFAR-10 and CIFAR-100 to evaluate the contribution of each proposed module across three settings. As shown in Table 2, progressively integrating our modules leads to consistent performance gains.

The baseline model, devoid of any proposed modules, exhibits limited capability, especially in the lb4 setting, where it attains only 38.24% and 26.50% accuracy on CIFAR-10 and CIFAR-100, respectively. Incorporating the class-adaptive re-weighting (CAW) module brings consistent improvements, with gains of up to 2.10% on CIFAR-10 and 2.50% on CIFAR-100. Introducing the local structure regularization (LSR) module yields more substantial improvements, boosting accuracy on CIFAR-10 by 10.46% and CIFAR-100 by 2.35% in the lb4 setting.

Further adding the global structure regularization (GSR) enhances the local-only variant by an average of 0.65%. Finally, the unknown class identification and modeling (UCIM) module contributes additional gains, averaging 2.06% on CIFAR-10 and 0.45% on CIFAR-100. The full model achieves the best performance, confirming that each module addresses a unique aspect of the challenging dual-mismatch problem involved in the semi-supervised learning scenario and that their effects are complementary and cumulative.

Table 2: Classification accuracy (%) with and without the key components in the proposed method.
<table><tr><td colspan="3">Module</td><td colspan="3">CIFAR-10</td><td colspan="3">CIFAR-100</td></tr><tr><td>CAW LSR</td><td>GSR</td><td>UCIM</td><td>1b4</td><td>lb10</td><td>lb25</td><td>lb4</td><td>lb10</td><td>lb25</td></tr><tr><td></td><td></td><td></td><td>38.24</td><td>64.46</td><td>74.40</td><td>26.50</td><td>40.75</td><td>51.00</td></tr><tr><td>√</td><td></td><td></td><td>40.34</td><td>64.86</td><td>76.00</td><td>27.89</td><td>43.25</td><td>53.00</td></tr><tr><td>√</td><td>√</td><td></td><td>50.80</td><td>68.05</td><td>76.95</td><td>30.24</td><td>43.25</td><td>53.49</td></tr><tr><td>√</td><td>√</td><td>√</td><td>54.30</td><td>68.54</td><td>76.88</td><td>30.25</td><td>43.43</td><td>53.30</td></tr><tr><td></td><td>√</td><td>√</td><td></td><td>56.89</td><td>70.96 78.05</td><td>30.43</td><td></td><td>44.37 53.53</td></tr></table>

Moreover, we evaluate two key modules of our method. The hub-spoke geometry construction (HGC) module which combines LSR and GSR mitigates minority class performance degradation caused by class imbalance. As shown in Fig. 4a, HGC substantially improves minority class accuracy, e.g., from 33% to 64% in the lb4 setting on CIFAR-10, confirming its effectiveness in enhancing feature learning for minority classes under imbalanced distributions. In paral-

![](images/9f732b9c48933278a68bf02e24fd1de4b74bd89ead7369aae40d267c2c56274b.jpg)  
(a)

![](images/a3b6ed883482be3e1f1023f7061ba1f15d5a3f3599fe4c29c837b9243aabb672.jpg)  
(b)  
Figure 4: Ablation studies on HGC module for minority class accuracy and UCIM module for pseudo-label accuracy.

lel, the unknown class identification and modeling (UCIM) module boosts pseudo-label accuracy from 47% to 60% in the same lb4 setting on CIFAR-10 as illustrated in Fig. 4b, demonstrating that reliable unknown class handling enhances both pseudo-label quality and robustness against dual mismatch. Further ablation analyses and discussions are provided in Appendix E.

## 7 CONCLUSION

In this work, we addressed the challenging and realistic scenario of dual-mismatched semi-supervised learning, where unlabeled data exhibits an imbalanced class distribution and contains unknown class samples. Our proposed method integrates a structured hub-spoke geometry with evidence-based unknown class identification. The hub-spoke geometry ensures a discriminative and robust feature rep resentation space, while the evidence-based classifier provides well-calibrated uncertainty estimates. We further theoretically show that the combination of high-uncertainty samples and hub-centered clustering induces a low-evidence region around the hub, providing theoretical support for our unknown class modeling strategy. Extensive experiments demonstrate that our method consistently outperforms previous state-of-the-art methods under various dual mismatch conditions.

## AI USE STATEMENT

In this work, we used generative AI tools to assist in the writing of proofs and assist with translation. We have not used generative AI tools for other tasks requiring disclosure. Additionally, we used generative AI tools to edit software code and the research paper to improve readability. We have reviewed all AI-assisted work. AI-generated proof steps were manually checked for logical correctness. Translations were verified against the source text. AI-edited code was tested for correctness by two authors. All readability edits were approved by all authors. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

The research utilizes publicly available datasets containing no personally identifiable information. As a strictly technical contribution focused on algorithmic improvement, we foresee no ethical issues or potential for misuse.

## REPRODUCIBILITY STATEMENT

To ensure the reproducibility of our work, we provide a comprehensive description of the experimental setup in Sec. 6 and Appendix C. All datasets used in our experiments are publicly available, and the source code is included as supplementary material. These resources collectively enable faithful reproduction of our main experimental results.

## REFERENCES

Eduardo Aguilar, Bogdan Raducanu, Petia Radeva, and Joost van de Weijer. Continual evidential deep learning for out-of-distribution detection. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3444–3454, 2023.

Emanuel Sanchez Aimar, Nathaniel Helgesen, Yonghao Xu, Marco Kuhlmann, and Michael Felsberg. Flexible distribution alignment: Towards long-tailed semi-supervised learning with proper calibration. In European Conference on Computer Vision (ECCV), volume 15112, pp. 307–327, 2024.

Jinsoo Bae, Seoung Bum Kim, and Hyungrok Do. Calimatch: Adaptive calibration for improving safe semi-supervised learning. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 2867–2876, 2025.

Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 - mining discriminative components with random forests. In European Conference on Computer Vision (ECCV), volume 8694, pp. 446–461, 2014.

Paola Cascante-Bonilla, Fuwen Tan, Yanjun Qi, and Vicente Ordonez. Curriculum labeling: Revisiting pseudo-labeling for semi-supervised learning. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), volume 35, pp. 6912–6920, 2021.

Dongdong Chen, Wei Wang, Wei Gao, and Zhi-Hua Zhou. Tri-net for semi-supervised deep learning. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), pp. 2014– 2020, 2018.

Hao Chen, Ran Tao, Yue Fan, Yidong Wang, Jindong Wang, Bernt Schiele, Xing Xie, Bhiksha Raj, and Marios Savvides. Softmatch: Addressing the quantity-quality tradeoff in semi-supervised learning. In International Conference on Learning Representations (ICLR), 2023.

Jiahe Chen, Qian Shao, Qiyuan Chen, Jiaying He, Jintai Chen, Jian Wu, and Hongxia Xu. Geometric gradient rectification for safe open-set semi-supervised learning. In European Conference on Computer Vision (ECCV), 2026.

Bo Cheng, Jueqing Lu, Yuan Tian, Haifeng Zhao, Yi Chang, and Lan Du. Cgmatch: A different perspective of semi-supervised learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15381–15391, 2025.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 248–255, 2009.

Chaoqun Du, Yizeng Han, and Gao Huang. Simpro: A simple probabilistic framework towards realistic long-tailed semi-supervised learning. In International Conference on Machine Learning (ICML), pp. 11686–11703, 2024.

Yaxin Hou and Yuheng Jia. A square peg in a square hole: Meta-expert for long-tailed semi-supervised learning. In International Conference on Machine Learning (ICML), 2025.

Yaxin Hou, Bo Han, Yuheng Jia, Hui LIU, and Junhui Hou. Keep it on a leash: Controllable pseudo-label generation towards realistic long-tailed semi-supervised learning. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

Yaxin Hou, Jun Ma, Hanyang Li, Bo Han, Jie Yu, and Yuheng Jia. Beyond distribution estimation: Simplex anchored structural inference towards universal semi-supervised learning. In International Conference on Machine Learning (ICML), 2026.

Audun Jøsang. Subjective Logic. Springer, 2016.

Zhanghan Ke, Daoye Wang, Qiong Yan, Jimmy S. J. Ren, and Rynson W. H. Lau. Dual student: Breaking the limits of the teacher in semi-supervised learning. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6728–6736, 2019.

Jaehyung Kim, Youngbum Hur, Sejun Park, Eunho Yang, Sung Ju Hwang, and Jinwoo Shin. Distribution aligning refinery of pseudo-label for imbalanced semi-supervised learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pp. 14567–14579, 2020.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

Hyuck Lee and Heeyoung Kim. CDMAD: class-distribution-mismatch-aware debiasing for classimbalanced semi-supervised learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 23891–23900, 2024.

Hyuck Lee, Seungjae Shin, and Heeyoung Kim. ABC: auxiliary balanced classifier for classimbalanced semi-supervised learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, pp. 7082–7094, 2021.

Zekun Li, Lei Qi, Yinghuan Shi, and Yang Gao. Iomatch: Simplifying open-set semi-supervised learning with joint inliers and outliers utilization. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15824–15833, 2023.

Ilya Loshchilov and Frank Hutter. SGDR: stochastic gradient descent with warm restarts. In International Conference on Learning Representations (ICLR), 2017.

Chengcheng Ma, Ismail Elezi, Jiankang Deng, Weiming Dong, and Changsheng Xu. Three heads are better than one: Complementary experts for long-tailed semi-supervised learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI), volume 38, pp. 14229–14237, 2024.

Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. Virtual adversarial training: A regularization method for supervised and semi-supervised learning. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 41(8):1979–1993, 2019.

Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng. Reading digits in natural images with unsupervised feature learning. In Advances in Neural Information Processing Systems (NeurIPS), 2011.

Youngtaek Oh, Dong-Jin Kim, and In So Kweon. DASO: distribution-aware semantics-oriented pseudo-label for imbalanced semi-supervised learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9786–9796, 2022.

Jingen Qu, Yufei Chen, Xiaodong Yue, Wei Fu, and Qiguang Huang. Hyper-opinion evidential deep learning for out-of-distribution detection. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pp. 84645–84668, 2024.

Kuniaki Saito, Donghyun Kim, and Kate Saenko. Openmatch: Open-set consistency regularization for semi-supervised learning with outliers. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, pp. 25956–25967, 2021.

Kari Sentz and Scott Ferson. Combination of Evidence in Dempster-Shafer Theory. Sandia National Laboratories, 2002.

Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin Raffel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pp. 596–608, 2020.

Senmao Tian, Xiang Wei, and Shunli Zhang. Sampling control for imbalanced calibration in semisupervised learning. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), volume 40, pp. 25914–25922, 2026.

Erik Wallin, Lennart Svensson, Fredrik Kahl, and Lars Hammarstrand. Improving open-set semisupervised learning with self-supervision. In IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 2356–2365, 2024.

Yidong Wang, Hao Chen, Yue Fan, Wang Sun, Ran Tao, Wenxin Hou, Renjie Wang, Linyi Yang, Zhi Zhou, Lan-Zhe Guo, Heli Qi, Zhen Wu, Yufeng Li, Satoshi Nakamura, Wei Ye, Marios Savvides, Bhiksha Raj, Takahiro Shinozaki, Bernt Schiele, Jindong Wang, Xing Xie, and Yue Zhang. USB: A unified semi-supervised learning benchmark for classification. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Yidong Wang, Hao Chen, Qiang Heng, Wenxin Hou, Yue Fan, Zhen Wu, Jindong Wang, Marios Savvides, Takahiro Shinozaki, Bhiksha Raj, Bernt Schiele, and Xing Xie. Freematch: Selfadaptive thresholding for semi-supervised learning. In International Conference on Learning Representations (ICLR), 2023.

Zerun Wang, Liuyu Xiang, Lang Huang, Jiafeng Mao, Ling Xiao, and Toshihiko Yamasaki. Scomatch: Alleviating overtrusting in open-set semi-supervised learning. In European Conference on Computer Vision (ECCV), volume 15109, pp. 217–233, 2024.

Chen Wei, Kihyuk Sohn, Clayton Mellina, Alan L. Yuille, and Fan Yang. Crest: A class-rebalancing self-training framework for imbalanced semi-supervised learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10857–10866, 2021.

Tong Wei and Kai Gan. Towards realistic long-tailed semi-supervised learning: Consistency is all you need. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3469–3478, 2023.

Qing Yu, Daiki Ikami, Go Irie, and Kiyoharu Aizawa. Multi-task curriculum framework for open-set semi-supervised learning. In European Conference on Computer Vision (ECCV), volume 12357, pp. 438–454, 2020.

Yang Yu, Danruo Deng, Furui Liu, Qi Dou, Yueming Jin, Guangyong Chen, and Pheng-Ann Heng. ANEDL: adaptive negative evidential deep learning for open-set semi-supervised learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI), volume 38, pp. 16587–16595, 2024.

Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. In Proceedings ofthe British Machine Vision Conference (BMVC), 2016.

Shenkai Zhao, Xinao Zhang, Lipeng Pan, Xiaobin Xu, and Danilo Pelusi. Evidential deep active learning for semi-supervised classification. CoRR, abs/2505.20691, 2025.

## TABLE OF CONTENTS FOR THE APPENDIX

• A: More Related Work . . . . . . . . 15   
• B: Pseudo-code of the Proposed Method . . . . . 15   
• C: Implementation Details . . . . . . 15   
• D: Further Experimental Results and Analyses . . . . . . . . 16   
• E: Further Ablation Analyses and Discussions . . . . . 17   
• F: Compute Resources . . . . 18

## A MORE RELATED WORK

Evidential deep learning (EDL), grounded in the Dempster-Shafer theory (Sentz & Ferson, 2002) and subjective logic (Jøsang, 2016), has emerged as a prominent framework for uncertainty quantification and unknown class sample detection. For instance, HEDL (Qu et al., 2024) introduces hyper-domain evidence to capture both certain and ambiguous classes, enhancing uncertainty estimation via an opinion projection mechanism, while CEDL (Aguilar et al., 2023) integrates EDL into continual learning, leveraging its ability to distinguish unknown class samples from past known class ones. EDALSSC (Zhao et al., 2025) proposes a semi-supervised evidential model that dynamically balances evidence for active learning sample selection. Notably, ANEDL (Yu et al., 2024) is positioned as the first to adapt EDL for OSSL, utilizing an adaptive negative optimization strategy to improve unknown class sample detection and emphasize uncertain classes. However, it does not address the realistic scenario involving both class distribution and label space mismatches between labeled and unlabeled data.

## B PSEUDO-CODE OF THE PROPOSED METHOD

Algorithm 1 Training process of our proposed method   
Require: Labeled and unlabeled datasets $\mathcal { D } _ { l }$ and $\mathcal { D } _ { u } .$   
Ensure: Encoder Φ(·), classification head f(·), and evidence head $g ( \cdot )$   
1: Initialize the parameters of Φ(·), f(·), and g(·) randomly.   
2: for epoch = 1, 2, . . . do   
3: for batch = 1, 2, . . . do   
4: if epoch < 60 then   
5: Calculate $\mathcal { L } _ { w a r m 1 }$ by Eq. equation 12;   
6: end if   
7: if 60 ≤ epoch < 80 then   
8: Calculate $\mathcal { L } _ { w a r m 2 }$ by Eq. equation 13;   
9: end if   
10: if epoch $\geq 8 0$ then   
11: Construct hub-spoke geometry by Eq. equation 5 and Eq. equation 8;   
12: Compute class-adaptive weight w by Eq. equation 9;   
13: Calculate evidence loss $\mathcal { L } _ { e d l }$ by Eq. equation 10 for the evidence head;   
14: Pull identified unknown class samples toward the hub by Eq. equation 11;   
15: Obtain the overall loss $\mathcal { L } _ { o v e r a l l }$ by Eq. equation 14;   
16: end if   
17: Update network parameters via gradient descent;   
18: end for   
19: end for

## C IMPLEMENTATION DETAILS

We follow the default settings and hyperparameters in USB (Wang et al., 2022). Specifically, the batch size of labeled data $\bar { B _ { l } }$ is set to 64, while that of unlabeled data $B _ { u }$ is set to 7 times $\ddot { B _ { l } } ;$ the confidence threshold τ is set to 0.95, and the temperature parameter v is set to 0.1. Moreover, we use the WRN-28-2 (Zagoruyko & Komodakis, 2016) architecture, the SGD optimizer with momentum 0.9, and weight decay 5e-4 for training. We use a cosine learning rate decay scheme (Loshchilov & Hutter, 2017), setting the learning rate to η cos $\left( { \frac { 7 \pi t } { 1 6 T } } \right)$ . Here, $\eta = 0 . 0 3$ is the initial learning rate, t is the current training step, and $T = 2 ^ { 1 8 }$ is the total number of training steps. We repeat each experiment with three different random seeds (i.e., 0, 1, and 4) and report the mean and standard deviation of the performance. We conduct the experiments on a single NVIDIA RTX 4090 GPU using PyTorch v2.4.1.

![](images/bc0d993fec97bb98c2b7ad0cfb4e66a7232c3667abb0b2502da7cd7f48229536.jpg)  
(a)

![](images/c4b272d9b4854dc9bb2b5d02adf47c330988cc8374e920abf91f7488ce922169.jpg)  
(b)  
Figure 5: (a) and (b) respectively compare classification accuracy on the CIFAR-10 test data for $k \in \{ 0 \% , 1 0 \% , 2 0 \% \}$ and $v \in \{ 0 . 0 5 , 0 . \bar { 1 } , 0 . 2 \}$ across the three settings. We clearly see that $k = 1 0 \%$ and $k = 2 0 \%$ yield comparable results, and performance remains stable across different values of v.

## D FURTHER EXPERIMENTAL RESULTS AND ANALYSES

## D.1 HYPERPARAMETER SENSITIVITY ANALYSIS

We perform hyperparameter sensitivity analysis on three key hyperparameters: k, the percentage of high-uncertainty samples selected per batch (i.e., the number of identified unknown class samples per batch), v, the temperature scaling factor in Eq. 6, and λ, the balancing weight for losses in Eq. 14. As shown in Fig. 5a, our method is robust to the choice of k. On CIFAR-10, both $k = 1 0 \%$ and $k = 2 0 \%$ yield nearly identical performance and significantly outperform the $k = 0 \%$ baseline across all three settings. Specifically, $k = 1 0 \%$ improves over the baseline by +5.0%, +2.8%, and +1.0% for 4, 10, and 25 labeled samples per class, respectively. The comparable results for k = 20% (i.e., gains of +4.7%, +0.8%, and +0.1%) indicate our method’s relative insensitivity to this hyperparameter. Similarly, as shown in Fig. 5b, performance remains stable across different values of v, with an average standard deviation of merely 0.99%, confirming its robustness. For λ, we evaluate three values (0.05, 0.10, 0.20) on CIFAR-10, with results reported in Table 3. The default $\lambda = 0 . 1 0$ achieves the best or near-best accuracy across all labeled settings, e.g., 56.89% (lb4) and 70.96% (lb10), while $\lambda = 0 . 2 0$ yields a slightly higher result (78.24%) only for lb25. Overall, the performance variation across different λ values remains modest, demonstrating that our approach is also robust to this hyperparameter.

Table 3: Classification accuracy (%) with different λ values on CIFAR-10. The results show that λ = 0.1 achieves the best or near-best performance across all settings.
<table><tr><td>λ</td><td>1b4</td><td>lb10</td><td>lb25</td></tr><tr><td>0.05</td><td>52.74</td><td>69.18</td><td>77.43</td></tr><tr><td>0.10</td><td>56.89</td><td>70.96</td><td>78.05</td></tr><tr><td>0.20</td><td>55.34</td><td>68.43</td><td>78.24</td></tr></table>

## D.2 ROBUSTNESS TO UNKNOWN CLASS DIVERSITY AND EFFECTIVENESS OF UNKNOWN CLASS MODELING

To further evaluate the robustness and effectiveness of our proposed unknown class handling strategy, we examine the sensitivity of our method to the number of unknown classes by varying it from 50 to 10 on CIFAR-100 while keeping the total unknown class sample count fixed. As shown in Fig. 6a, our method maintains relatively stable performance across all settings despite substantial changes in the diversity of unknown classes. This robustness can be attributed to our hub-based unknown class modeling: instead of explicitly modeling each unknown category, we identify high-uncertainty samples based on their evidence and guide them toward a shared low-evidence region around the hub. Since the hub provides a common target for uncertain samples regardless of their specific unknown class, the same modeling mechanism can be applied as the number of unknown classes changes.

We further compare our unknown class modeling with a K-means clustering baseline on the identified high-uncertainty unknown class samples. For a fair comparison, we set the number of clusters K in K-means to the ground-truth number of unknown classes. As illustrated in Fig. 6b, our method consistently outperforms K-means across all three settings on CIFAR-10, achieving absolute gains of 0.5%, 4.8%, and 1.2% in the lb4, lb10, and lb25 settings, respectively. The superior performance, particularly in the lb10 setting, highlights the advantage of our hub-spoke geometry, which leverages evidence-based uncertainty to guide high-uncertainty unknown class samples toward a shared lowevidence hub, thereby enhancing known–unknown class discriminability and improving pseudo-label quality.

![](images/2e216d5681cdaf6ce57704d652360dfcd97307285370e77c7813b6f571cbdbc8.jpg)  
(a)

![](images/f3f5faf88fda107a1c949ee8bb7a1eefcb5e6046d93429e51cf94d597cbcb4d0.jpg)  
(b)  
Figure 6: (a) reports classification accuracy on CIFAR-100 under different numbers of unknown classes (50, 40, 30, 20, 10) across the three lb settings; (b) compares our method with K-means clustering on the identified unknown class samples on CIFAR-10.

## D.3 STATISTICAL SIGNIFICANCE

We also conduct pairwise t-tests (α = 0.05) against thirteen baselines under different label settings to evaluate the robustness of our approach and the full results are given in Table 4. Our method achieves significant improvements over all LTSSL methods in 75% of cases and over all OSSL methods in 76.7% of cases, confirming the consistent superiority of our method.

Table 4: Statistical significance of performance differences assessed with pairwise t-test at a 0.05 significance level, reported as win/tie/loss counts.
<table><tr><td>Method</td><td>lb10</td><td>lb25</td><td>Total</td></tr><tr><td>TSS</td><td>FixMatch 2/3/0 FreeMatch 1/4/0 SoftMatch 1/4/0</td><td>1/4/0 1/4/0 2/3/0</td><td>3/7/0 2/8/0 3/7/0 8/2/0</td></tr><tr><td>TTSST</td><td>SimPro 5/0/0 CDMAD 4/1/0 Meta-Expert 4/1/0 SC-SSL 2/3/0</td><td>3/2/0 5/0/0 4/1/0 3/2/0</td><td>9/1/0 8/2/0 5/5/0 9/1/0</td></tr><tr><td>TSSO</td><td>OpenMatch 4/1/0 IOMatch 3/2/0 SCOMatch 4/1/0 CaliMatch 2/3/0 ANEDL 5/0/0 GGR 4/1/0</td><td>5/0/0 4/1/0 3/2/0 3/2/0 5/0/0 4/1/0</td><td>7/3/0 7/3/0 5/5/0 10 /0 / 0 8/2/0</td></tr><tr><td>Total</td><td></td><td>41/24/0 43/22/0 84/46/0</td><td></td></tr></table>

## E FURTHER ABLATION ANALYSES AND DISCUSSIONS

To further validate the effectiveness of local structure regularization (LSR) and global structure regularization (GSR), we performed a fine-grained component-level analysis. As shown in Table 5, the integrated LSR module surpasses its individual component losses, namely $\mathcal { L } _ { i l }$ (LIL) and $\mathcal { L } _ { o l }$ (LOL), by average margins of 1.51% on CIFAR-10 and 0.83% on CIFAR-100. Similarly, the complete GSR module achieves average improvements of 1.73% on CIFAR-10 and 0.72% on CIFAR-100 over its individual component losses $\mathcal { L } _ { i g } \left( \mathrm { L I G } \right)$ and $\mathcal { L } _ { o g } \left( \mathrm { L O G } \right)$ . These fine-grained ablation results further substantiate the efficacy of both LSR and GSR modules while demonstrating the complementary relationships among their individual component losses.

Table 5: Classification accuracy (%) with and without the $\mathcal { L } _ { i l }$ (LIL), $\mathcal { L } _ { o l }$ (LOL), $\mathcal { L } _ { i g }$ (LIG), and $\mathcal { L } _ { o g } \left( \mathrm { L O G } \right)$ losses of the proposed method across three settings, with 4, 10, and 25 labeled samples per class. All ablation studies are conducted with the class-adaptive re-weighting (CAW) module integrated as the foundation. The datasets are CIFAR-10 and CIFAR-100, which comprise a balanced labeled dataset (imbalance ratio $\gamma _ { l } = 1 )$ of only known classes and an imbalanced unlabeled dataset (imbalance ratio $\gamma _ { u } = 1 0 0 )$ of both known and unknown classes.
<table><tr><td colspan="3">Module</td><td colspan="3">CIFAR-10</td><td colspan="3">CIFAR-100</td></tr><tr><td>LIL LOL</td><td>LIG</td><td>LOG</td><td>lb4</td><td>lb10</td><td>lb25</td><td>lb4</td><td>lb10</td><td>lb25</td></tr><tr><td>V</td><td></td><td></td><td>50.61</td><td>67.75</td><td>76.52</td><td>29.18</td><td>43.12</td><td>52.47</td></tr><tr><td></td><td>√</td><td></td><td>43.35</td><td>67.71</td><td>76.63</td><td>28.77</td><td>42.44</td><td>52.98</td></tr><tr><td>√</td><td>7</td><td></td><td>50.80</td><td>68.05</td><td>76.95</td><td>30.24</td><td>43.25</td><td>53.49</td></tr><tr><td>了</td><td>√</td><td>√</td><td>52.16</td><td>67.29</td><td>76.68</td><td>29.81</td><td>42.25</td><td>52.55</td></tr><tr><td>√</td><td>√</td><td></td><td>49.33</td><td>67.26</td><td>76.34</td><td>28.65</td><td>43.20</td><td>53.20</td></tr><tr><td></td><td>V</td><td>L</td><td></td><td>54.30 68.54</td><td>76.88</td><td>30.25</td><td>43.43</td><td>53.30</td></tr></table>

## F COMPUTE RESOURCES

• CPU: AMD EPYC 7642 48-Core Processor × 2

• GPU: NVIDIA GeForce RTX 4090 $2 4 \mathrm { G } \times 1$

• MEM: 500G

• Maximum total computing time: training + testing ≈ 20h