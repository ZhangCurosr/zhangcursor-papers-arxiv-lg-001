# Contrastive Representation-Guided Genetic Minority Oversampling for Imbalanced Time-Series Classification

Wenbin Pei, Yunrong Hao, Zhen Liu, Guan Wang, Bing Xue, Fellow, IEEE, Yiu-Ming Cheung, Fellow, IEEE, and Qiang Zhang, Senior Member, IEEE

Abstract—Real-world time-series classification tasks often exhibit class imbalance, which can be extremely severe in some applications. To avoid training biased classifiers on imbalanced data, sampling is one of the most popular data pre-processing techniques because of its classifier-agnostic nature. However, due to the complex temporal dependencies in original timeseries data and the scarcity of minority-class samples, existing sampling methods, including interpolation-based oversampling methods and deep learning-based generative models, usually suffer from limited generalization and poor diversity when generating new time-series samples. This paper proposes a Frequencydomain representation-guided Multi-tree Genetic Programmingbased oversampling approach (FreMGP) to imbalanced timeseries classification, where each individual represents a set of synthetic samples for the minority class. A frequency-domain class-discriminative representation module based on contrastive learning is also developed, guiding the evolutionary search toward high-quality synthetic time-series samples. Experiments on imbalanced time-series datasets demonstrate that FreMGP outperforms existing oversampling methods and consistently improves the performance of different classifiers, including both general machine learning and deep learning models.

This work was supported in part by the National Key Research and Development Program of China under grant 2024YFA1012700, the National Natural Science Foundation of China under grants 62206041, 12371516 and U21A20491, and the NSFC-Liaoning Province United Foundation under grant U1908214, the 111 Project under grant D23006, the Liaoning Revitalization Talents Program under grant XLYC2008017, and China University Industry-University-Research Innovation Fund under grants 2022IT174, Natural Science Foundation of Liaoning Province under grant 2023-BSBA-030, and an Open Fund of National Engineering Laboratory for Big Data System Computing Technology under grant SZU-BDSC-OF2024-09. (Corresponding author: Zhen Liu.)

Wenbin Pei and Yunrong Hao are with the School of Computer Science and Technology, Dalian University of Technology, Dalian 116024, China; Key Laboratory of Social Computing and Cognitive Intelligence (Dalian University of Technology), Ministry of Education, Dalian 116024, China (email: peiwenbin@dlut.edu.cn; haoyunrong@mail.dlut.edu.cn).

Zhen Liu is with the College of Computing and Data Science, Nanyang Technological University, Singapore (e-mail: zhen.liu@ntu.edu.sg).

Guan Wang is with the School of Mechanics and Aerospace Engineering, Dalian University of Technology, Dalian 116024, China (e-mail: guanwang@dlut.edu.cn).

Bing Xue is with the School of Engineering and Computer Science, Victoria University of Wellington, PO Box 600, Wellington 6140, New Zealand (emails: bing.xue@ecs.vuw.ac.nz).

Yiu-Ming Cheung is with the Department of Computer Science, Hong Kong Baptist University, Hong Kong, SAR, China (e-mail: ymc@comp.hkbu.edu.hk).

Qiang Zhang is with the School of Computer Science and Technology, Dalian University of Technology, Dalian 116024, China; Key Laboratory of Social Computing and Cognitive Intelligence (Dalian University of Technology), Ministry of Education, Dalian 116024, China; and also with the National and Local Joint Engineering Laboratory of Computer Aided Design, Dalian University, Dalian 116622, China (e-mail: zhangq@dlut.edu.cn).

Index Terms—Imbalanced Time-Series Classification, Sampling, Multi-Tree Strongly-typed Genetic Programming, Representation Learning.

## I. INTRODUCTION

Many real-world applications generate substantial amounts of time-series data. To make sense of such data, time-series classification (TSC) has emerged as an important machine learning task that aims to assign predefined labels to sequential data based on its temporal patterns [1]. To date, numerous classification algorithms, e.g., Hydra [2] and spectral-aware reservoir computing (SARC) [3], have been proposed for TSC, while most of them implicitly assume a balanced data distribution across different classes. However, in many applications, such as fault detection [4], medical diagnosis [5], and economics [6], time-series data is naturally imbalanced, posing a challenge to conventional TSC algorithms. Note that data imbalance can bias classifiers toward the majority class and weaken their ability to recognize minority-class samples that usually are of particular interest [7].

Data sampling [8] is one of the most popular techniques to address the class imbalance issue due to its classifier-agnostic nature. To rebalance data distribution, traditional interpolationbased oversampling methods, such as the synthetic minority over-sampling technique (SMOTE) [9] and its variants [8], [10], generate a number of synthetic samples by interpolating between minority-class samples and their neighbors. However, as these sampling methods were developed for tabular imbalanced data rather than time-series data, they are unable to capture the complex temporal patterns inherent in time series [11]. In the literature, to handle imbalanced time-series data, the temporal-oriented synthetic minority oversampling technique (T-SMOTE) [11] incorporates temporal characteristics during sample generation. Nevertheless, it still depends on local interpolation, which may limit the ability to preserve discriminative local variations [12] and frequency-domain characteristics in time series [13]. Furthermore, deep learning-based generative models, such as generative adversarial networks (GANs) [14], variational autoencoders (VAEs) [15], and diffusion-based models [16], have demonstrated their effectiveness in generating time-series data. Despite their strong data modeling capabilities, these methods typically demand a large number of training samples to accurately learn the complex temporal dependencies in time-series data [17]. Unfortunately, from a practical perspective, this is not always workable because time-series samples in the minority class are usually scarce, resulting in limited generalization and poor diversity when generating new data [17], [18].

Genetic programming (GP) offers a solution for generating data [19], [20]. Unlike black-box generative models, GP individuals are tree-structured, combining function and terminal nodes to form diverse mathematical expressions, each of which can be regarded as a time-series sample. More importantly, GP automatically learns complex transformations from the learned time-series data, allowing it to produce synthetic time-series samples that preserve the intricate temporal dependencies essential for time-series classification. However, it remains challenging for $\mathrm { G P }$ to generate high-quality, diverse timeseries data when only a limited number of minority-class samples are available in an imbalanced time-series classification (ITSC) task. Therefore, it is necessary to investigate the potential of GP for oversampling time series in ITSC. In this paper, we propose a Frequency-domain representationguided Multi-tree Genetic Programming-based oversampling approach (FreMGP), which learns a frequency-domain classdiscriminative representation space and uses it to guide the evolutionary oversampling process. Moreover, FreMGP adopts a multi-tree GP representation, where each individual generates a set of synthetic samples for one minority class rather than a single sample.

The main contributions of this paper are summarized as follows:

• We design a frequency-domain contrastive representation learning module. It uses prototype-guided contrastive learning to construct a class-discriminative representation space. This provides a reliable representation space for evaluating the quality of generated synthetic samples.

• We design a novel multi-tree GP structure, for which both the terminal set and the function set are specifically designed, to represent frequency-domain time-series samples for oversampling. This enables a set of synthetic minority-class samples to be generated within a single GP individual, making it easy to evaluate the overall diversity of the generated sample set.

• We develop a representation-guided two-stage fitness evaluation method and corresponding genetic operators. The fitness function evaluates generated samples in the learned representation space and considers both the similarity to existing minority-class samples and the diversity of the generated sample set. This fitness function guides the evolutionary search toward samples that are close to the minority-class region in the learned representation space while ensuring sufficient diversity of the generated samples.

## II. BACKGROUND

## A. Frequency-domain Contrastive Representation Learning

Representation learning aims to transform raw data into an embedding space, where samples can be represented by compact and informative feature vectors [21]. Given L as

![](images/f9a74eade9008d1a1b5a00a5c6a9b8a96901296782dd7d1ac3cc8188d5fcadc6.jpg)  
Fig. 1. An illustration of supervised contrastive learning in the embedding space. Samples from the same class as the anchor form positive pairs and are pulled closer, while samples from different classes form negative pairs and are pushed apart.

the sequence length, for a time-series sample $\mathbf { x } _ { i } ~ \in ~ \mathbb { R } ^ { L }$ , a representation encoder $f _ { \theta } ( \cdot )$ maps it to a feature vector:

$$
\mathbf { h } _ { i } = f _ { \theta } ( \mathbf { x } _ { i } ) ,\tag{1}
$$

where $\mathbf { h } _ { i }$ is the learned representation of $\mathbf { x } _ { i } .$ . For classification tasks, a desirable representation space is expected to preserve class-related information, so that samples from the same class are close to each other while samples from different classes are clearly separated.

Contrastive learning [22], [23] is a common representation learning paradigm that learns representations by comparing positive and negative sample pairs. Positive pairs are encouraged to be close in the embedding space, while negative pairs are pushed apart. When class labels are available, supervised contrastive learning can use label information to construct positive and negative pairs [24]. As illustrated in Fig. 1, samples from the same class as the anchor are treated as positive pairs, while samples from different classes are treated as negative pairs. A common supervised contrastive learning objective is defined as [24]:

$$
\mathcal { L } _ { \mathrm { c o m } } = \sum _ { i = 1 } ^ { B } \frac { - 1 } { | \mathcal { P } ( i ) | } \sum _ { p \in \mathcal { P } ( i ) } \log \frac { \exp ( \mathbf { z } _ { i } ^ { \top } \mathbf { z } _ { p } / \tau ) } { \sum _ { j = 1 , j \neq i } ^ { B } \exp ( \mathbf { z } _ { i } ^ { \top } \mathbf { z } _ { j } / \tau ) } ,\tag{2}
$$

where $B$ denotes the batch size, i denotes the index of an anchor sample, $\mathcal { P } ( i )$ denotes the set of positive sample indices that share the same class label as $\mathbf { x } _ { i }$ while excluding the anchor itself, $\mathbf { z } _ { i }$ is the normalized projection of the representation h<sub>i</sub>, and $\tau$ is a temperature parameter.

For time-series data, representations can be learned from either the time domain or the frequency domain [25], [26]. Compared with raw time-domain data, frequency-domain features provide a complementary view and help capture global patterns of time series. In particular, magnitude spectra are less sensitive to temporal shifts [27], which is useful for learning stable representations. Therefore, in this work, contrastive representation learning in the frequency domain is used to construct a representation space. The learned space provides compact feature representations for time-series samples and is later used to evaluate generated samples during evolutionary oversampling.

## B. Multi-Tree and Strongly-Typed Genetic Programming

Inspired by biological evolution, GP is an evolutionary algorithm that automatically evolves executable computer programs, typically represented as tree structures. In standard GP, an individual contains only one tree.

a) Multi-Tree Genetic Programming (MTGP): MTGP extends the standard GP by allowing multiple trees within an individual [28]. An individual in MTGP can be represented as $\mathcal { T } = \{ T _ { 1 } , T _ { 2 } , \dots , T _ { M } \}$ , where $T _ { m }$ denotes the m-th tree in the individual, and M is the number of trees in this individual.

b) Strongly-typed Genetic Programming (STGP): Different from standard GP, STGP enforces data type constraints on functions and terminals to construct strongly-typed trees [29]. In STGP, each terminal is assigned predefined data types, and correspondingly, each function has specified types for its arguments and the returned value. This restriction eliminates invalid trees, thereby reducing the search space.

In this work, MTGP is employed to enable each individual to represent a set of synthetic frequency-domain time-series samples, each of which is represented by a strongly-typed tree adhering to predefined type constraints.

## C. Related Works

a) Imbalanced Time-Series Classification: ITSC refers to the task of classifying time-series data into categories where the number of samples across different classes is imbalanced. Commonly used traditional classifiers mainly include multilayer perceptron (MLP) [30], Random Forest [31], and distance-based methods such as k-nearest neighbors with dynamic time warping [32]. To date, deep learning has significantly advanced the field [33], with numerous deep models proposed to automatically learn temporal dependencies, including long short-term memory (LSTM) networks [34], temporal convolutional networks (TCNs) [35], and attentionbased Transformers [36].

To address the class imbalance issue in ITSC, cost-sensitive learning has been used to improve performance on the minority class by assigning higher costs to the misclassification of minority-class instances [4], [35], [37]. However, these costsensitive methods require cost matrices that are often predesigned by humans. Sampling [8] is another popular family of techniques to address the class imbalance issue due to its classifier-agnostic nature. Sampling methods mainly include undersampling and oversampling [11], [17], [20]. Compared with undersampling, oversampling rebalances the class distribution by generating additional minority-class samples while retaining the original training data.

b) Existing Oversampling Methods: The simplest oversampling is random oversampling (ROS) [38], which simply duplicates some existing minority-class samples at random. Although ROS is simple to implement, it does not introduce new information and frequently results in overfitting. To introduce diversity, SMOTE [9] generates synthetic minority-class samples through linear interpolation between neighboring minority-class instances. Several SMOTE variants have been developed with different sampling and generation strategies, including Borderline-SMOTE [39], adaptive synthetic sampling approach (ADASYN) [40], and grouping-based SMOTE (GB-SMOTE) [41]. However, these methods were primarily designed for tabular imbalanced data and do not explicitly account for the temporal characteristics of time series [11]. For time-series data, T-SMOTE [11] incorporates temporal information into the oversampling process. To oversample imbalanced time-series data in ITSC, T-SMOTE [11] extends SMOTE by incorporating temporal characteristics during sample generation. However, T-SMOTE still depends on linear interpolation, which may limit the ability to preserve discriminative local variations [12] and frequency-domain characteristics in time series [13].

In addition to interpolation-based oversampling, deep learning-based generative methods have also been explored for time-series data augmentation [12], [13]. These methods aim to learn the underlying data distribution and synthesize new samples from the learned generative model. The boundaryfocused generative adversarial network (BFGAN) [42] introduces additional labels to reflect the importance of sample positions in the data space, so that the generator can better consider boundary information and multi-modal structures. The counterfactual augmentation minority generation (CFAMG) [7] generates counterfactual minority-class samples by identifying causal factors related to class differences and intervening on latent representations. ImagenFew [17] further studies how time series can be generated under datascarce conditions, using a pretrained diffusion framework with dynamic convolutions and dataset-token conditioning for domain-aware generation. Although deep generative models provide a powerful approach to capturing complex timeseries distribution, their use for ITSC still faces two major challenges [17], [18]. First, these deep models usually require sufficient training samples to learn reliable generative distributions. However, minority-class samples are often scarce in ITSC [17], [18]. Second, the generation process of deep models is usually implicit, making it difficult to precisely regulate structural properties of generated time series, such as local variations, temporal patterns, and spectral components [12], [13].

The evolutionary time-frequency domain-based synthetic minority oversampling approach (Evo-TFS) [20] is an evolutionary oversampling method for ITSC. It uses GP to synthesize time series from the minority class, and the evolutionary oversampling process is guided by the fitness function incorporating both time-domain and frequency-domain distances [20]. Evo-TFS demonstrates the potential of GP to generate highquality time-series samples in ITSC. However, Evo-TFS’s fitness function is based on direct time-frequency similarity and finds it hard to fully capture class-discriminative information that is important for classification. Moreover, Evo-TFS requires multiple independent runs to produce the required number of minority-class samples. This makes it difficult to guarantee the overall diversity of the generated samples explicitly. Therefore, in this work, we further investigate how GP can be used to synthesize a set of minority-class timeseries samples and how these samples can be evaluated.

![](images/ed4c1ce16d9a2761763fb02c2c53b320b1cf1f623ee92eb0db9032ba2d219e34.jpg)  
Fig. 2. The Overall Design of FreMGP.

## III. THE PROPOSED APPROACH

## A. The Overall Design of FreMGP

The goal of FreMGP is to generate high-quality and diverse minority-class time-series samples to rebalance the training data for ITSC. The overall design of FreMGP is described in Fig. 2.

In general, representations of time-series data can be learned from either the time domain or the frequency domain [25], [26]. Compared with time-domain data, frequency-domain features provide a complementary view and are particularly effective at capturing the global characteristics of time-series data. Moreover, compared to the time-series domain, magnitude spectra are less sensitive to temporal shifts [27], which benefits the learning of stable time-series representations. Therefore, the original training samples are first transformed into the frequency domain by the discrete Fourier transform (DFT) [43], as illustrated in Fig. 2. Let the training set be denoted as ${ \mathcal D } _ { \mathrm { t r a i n } } ~ = ~ \{ ( { \bf x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , where $\mathbf { x } _ { i } ~ \in ~ \mathbb { R } ^ { L }$ is a univariate time-series sample, L is the sequence length, and y<sub>i</sub> is the class label. Each time-series sample is transformed from the time domain into the frequency domain using DFT:

$$
X _ { i , f } = \sum _ { \ell = 0 } ^ { L - 1 } x _ { i , \ell } \exp \left( - \varsigma \frac { 2 \pi f \ell } { L } \right) , \quad f = 0 , 1 , \ldots , F - 1 ,\tag{3}
$$

where $X _ { i , f }$ denotes the complex frequency coefficient at frequency index $f , x _ { i , \ell }$ denotes the corresponding time-domain value at time index ℓ, ς is the imaginary unit satisfying $\varsigma ^ { 2 } =$ −1, and F is the number of retained frequency components. For real-valued time series, the one-sided spectrum can be used with $F = \lfloor L / 2 \rfloor + 1$ . The transformed frequency-domain sample is denoted as $\mathbf { X } _ { i } ~ \in ~ \mathbb { C } ^ { F }$ . Based on the frequencydomain data, FreMGP consists of two main modules:

1) Frequency-domain Representation Space Learning: In the first module, an encoder is trained through frequencydomain contrastive representation learning. This module aims to construct a class-discriminative representation space, where samples from the same class are encouraged to be close while samples from different classes are encouraged to be separated. After training, the encoder is fixed and later used in the fitness evaluation of the subsequent oversampling process.

2) MTGP-based Oversampling: In the second module, FreMGP performs evolutionary oversampling based on MTGP, where each individual contains multiple trees to represent a set of synthetic frequency-domain samples. The trained encoder is used to evaluate every individual in the learned representation space through a two-stage evolutionary search. In the first stage, generated samples are evaluated based on their similarity to their corresponding target minority-class samples. In the second stage, both the similarity and diversity are considered. The evolutionary learning process stops when the stopping criterion is satisfied.

After the evolutionary oversampling process, each generated frequency-domain sample $\widehat { \mathbf { X } } \in \widehat { \mathbb { C } } ^ { F }$ is transformed back into the time domain by the inverse discrete Fourier transform (IDFT) as $\widehat { \mathbf { x } } = \mathrm { I D F T } ( \widehat { \mathbf { X } } )$ , where $\widehat { \mathbf { x } } \in \mathbb { R } ^ { L }$ is the corresponding synthetic time-series sample. Afterwards, the synthetic samples are combined with the original training set to form a rebalanced training set for downstream classifiers. The details of the two modules in FreMGP are introduced below.

## B. Frequency-Domain Representation Space Learning

Directly evaluating generated samples in the original data space may mainly capture similarity at the instance level, often failing to reflect class-discriminative information. Therefore, in the first module of FreMGP, we use supervised contrastive learning to construct a class-discriminative frequency-domain representation space. The learned space is then used to evaluate generated samples during the subsequent evolutionary search.

## 1) Supervised Contrastive Loss

Specifically, for a frequency-domain sample $\mathbf { X } _ { i } \in \mathbb { C } ^ { F }$ , the encoder $E _ { \theta } ( \cdot )$ maps it into a hidden representation:

$$
\begin{array} { r } { \mathbf { h } _ { i } = E _ { \theta } ( \mathbf { X } _ { i } ) , } \end{array}\tag{4}
$$

where $\mathbf { h } _ { i } \in \mathbb { R } ^ { d _ { h } }$ is the learned hidden representation of $\mathbf { X } _ { i } .$

To learn the representation space, FreMGP follows the common encoder-projector design in contrastive learning [22], [24], where a projector $P _ { \phi } ( \cdot )$ maps $\mathbf { h } _ { i }$ into a normalized metric embedding:

$$
\mathbf { z } _ { i } = \frac { P _ { \phi } ( \mathbf { h } _ { i } ) } { \| P _ { \phi } ( \mathbf { h } _ { i } ) \| _ { 2 } } ,\tag{5}
$$

where $\mathbf { z } _ { i } \in \mathbb { R } ^ { d _ { z } }$ and $\| \mathbf { z } _ { i } \| _ { 2 } = 1$

Based on the normalized metric embeddings, FreMGP adopts a supervised contrastive loss [24] to learn a classdiscriminative embedding space. For each anchor sample $\mathbf { X } _ { i } ,$ samples with the same class label in a batch are treated as positive samples. The supervised contrastive loss is defined as follows:

$$
\mathcal { L } _ { s u p } ^ { ( i ) } = \frac { - 1 } { | \mathcal { P } ( i ) | } \sum _ { p \in \mathcal { P } ( i ) } \log \frac { \exp ( \mathbf { z } _ { i } ^ { T } \mathbf { z } _ { p } / \tau ) } { \sum _ { j \ne i } \exp ( \mathbf { z } _ { i } ^ { T } \mathbf { z } _ { j } / \tau ) } ,\tag{6}
$$

where ${ \mathcal { P } } ( i ) = \{ p \mid y _ { p } = y _ { i } , p \neq i \}$ is the positive index set of anchor sample $\mathbf { X } _ { i } ; \mathbf { z } _ { i } , \mathbf { z } _ { p } .$ , and $\mathbf { z } _ { j }$ denote the normalized metric embeddings of the anchor sample, a positive sample, and a non-anchor sample in the batch, respectively; and $\tau$ is the temperature parameter.

## 2) Prototype-Guided Loss

Although supervised contrastive learning encourages intraclass compactness and inter-class separation, class imbalance may still bias the learned representation space toward majority classes. This is because minority-class samples typically provide fewer same-class positives within each batch. To alleviate this issue, FreMGP introduces fixed class prototypes as classlevel anchors, providing each class with a stable class-level reference beyond sample-to-sample comparisons.

The fixed prototypes are constructed in three steps. First, an initial metric embedding space is trained using the normalized temperature-scaled cross-entropy (NT-Xent) loss [22], from which the normalized embeddings of majority-class samples are extracted. Second, a majority-class reference prototype is optimized on the unit hypersphere by minimizing its average Euclidean distance to these embeddings. Finally, FreMGP constructs a set of simplex equiangular tight frame (ETF) prototypes, where all class prototypes are uniformly separated with equal angular distances [44]. The majority-class simplex prototype is then aligned with the optimized majority-class reference prototype, and the same rotation is applied to all simplex prototypes to preserve their relative separation. These prototypes are kept fixed during representation training.

Based on these fixed prototypes, the prototype-guided loss for sample $\mathbf { X } _ { i }$ is defined as follows:

$$
\begin{array} { r } { \mathcal { L } _ { p } ^ { ( i ) } = - \log \frac { \exp ( \mathbf { z } _ { i } ^ { T } \mathbf { p } _ { y _ { i } } / \tau ) } { \exp ( \mathbf { z } _ { i } ^ { T } \mathbf { p } _ { y _ { i } } / \tau ) + \sum _ { j \ne i } \exp ( \mathbf { z } _ { i } ^ { T } \mathbf { z } _ { j } / \tau ) } , } \end{array}\tag{7}
$$

where $\mathbf { p } _ { y _ { i } }$ denotes the prototype corresponding to the label $y _ { i }$

## 3) The Overall Loss Function

Inspired by the prototype gating strategy for imbalanced supervised contrastive learning [45], FreMGP uses a prototype gate to decide whether the prototype-guided loss should be applied to each sample. This avoids imposing overly strict constraints on samples that are already close to their class prototypes. Specifically, the gate is defined as follows:

$$
g _ { i } = \left\{ { \begin{array} { l l } { 1 , } & { \mathbf { z } _ { i } ^ { T } \mathbf { p } _ { y _ { i } } < \gamma , } \\ { 0 , } & { \mathbf { z } _ { i } ^ { T } \mathbf { p } _ { y _ { i } } \geq \gamma , } \end{array} } \right.\tag{8}
$$

where $\gamma$ is a similarity threshold. When $g _ { i } = 1$ , the embedding $\mathbf { z } _ { i }$ is not close enough to its class prototype, and the prototypeguided loss is activated. When $g _ { i } = 0$ , the embedding $\mathbf { z } _ { i }$ is already close to its class prototype, so the additional prototype constraint is not applied.

Following prior work that selectively applies prototype constraints in supervised contrastive learning for imbalanced data [45], FreMGP uses the supervised contrastive loss as the basic representation learning objective and selectively adds the prototype-guided loss according to the prototype gate. The final representation learning objective is defined as follows:

$$
\mathcal { L } _ { r e p } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left( \mathcal { L } _ { s u p } ^ { ( i ) } + g _ { i } \mathcal { L } _ { p } ^ { ( i ) } \right) ,\tag{9}
$$

where B is the number of embeddings in a batch.

Due to the page limit, other implementation details, including the frequency-domain MLP encoder inspired by frequencydomain MLPs for time-series forecasting (FreTS) [46] and the projector architecture, are provided in Appendix B.1.

C. Frequency-domain Contrastive Representation-Guided MTGP Oversampling

## 1) Individual Representation

In FreMGP, each individual contains multiple trees, each of which represents one synthetic frequency-domain time-series sample. Let $N _ { c }$ denote the number of training samples in the minority class c, and let $N _ { \mathrm { m a x } }$ denote the number of training samples in the majority class. The number of synthetic samples required for class c is then given by $M _ { c } = N _ { \operatorname* { m a x } } - N _ { c }$ . Therefore, in FreMGP, each individual consists of $M _ { c }$ stronglytyped trees, denoted as $\mathcal { T } _ { c } = \{ T _ { 1 } , T _ { 2 } , \dots , T _ { M _ { c } } \}$ , where each tree represents one synthetic frequency-domain sample. As illustrated in Fig. 3, a tree consists of four layers: Input, Spectral Transform, Spectral Fusion, and Output. The nodes in the Input layer are selected from the terminal set, while the nodes in the Spectral Transform and Spectral Fusion layers are selected from the function set. The Output layer returns the final synthetic frequency-domain sample. The terminal set and function set are introduced as follows:

TABLE I  
THE TERMINAL SET AND FUNCTION SET USED IN FREMGP
<table><tr><td colspan="3">Terminal Set</td><td colspan="3">Function Set</td></tr><tr><td>Name</td><td>Type</td><td>Description</td><td>Name</td><td>Input Type</td><td>Output Type</td></tr><tr><td>Spectrum terminal</td><td>S</td><td>Segmented training spectrum</td><td>AS</td><td>[S, A]</td><td>S</td></tr><tr><td>ERC-AS</td><td>A</td><td>Amplitude scaling coefficient</td><td>PS</td><td>[S, B]</td><td>S</td></tr><tr><td>ERC-PS</td><td>B</td><td>Phase shifting coefficient</td><td>FW</td><td>[s, c]</td><td>S</td></tr><tr><td>ERC-FW</td><td>C</td><td>Frequency warping coefficient</td><td>SF</td><td>[S, S, S]</td><td>0</td></tr></table>

Note: S, A, B, C, and O denote the spectrum type, amplitude scaling coefficient type, phase shifting coefficient type, frequency warping coefficient type, and output type, respectively. ERC-AS, ERC-PS, and ERC-FW are sampled from (0, 2], [−π, π], and [0.5, 2.0], respectively.

![](images/2fc2317a7953ae73e06d42abcd3fcba5cef1ea40e970e9fb264556c0cef0d728.jpg)  
Fig. 3. Illustration of the STGP tree representation in FreMGP. (a) Layered structure of an STGP tree. (b) Example of an STGP tree.

a) Terminal Set: The terminal set contains spectrum terminals and ephemeral random constants (ERCs). The terminal set is summarized in Table I. The spectrum terminals are constructed from the frequency-domain samples of the training set.

To construct spectrum terminals, FreMGP decomposes each training spectrum into several frequency-band components. Specifically, each spectrum sample $\mathbf { X } _ { i }$ is divided into three consecutive frequency segments, corresponding to the lowfrequency, mid-frequency, and high-frequency segments. The first two segments have length $\lfloor F / 3 \rfloor$ , where F denotes the dimension of the frequency-domain data obtained by the DFT in equation (3). The last segment contains the remaining frequency coefficients. For each spectrum sample X , we construct its v-th spectrum terminal by retaining only the coefficients in the v-th frequency segment and setting all coefficients in the other segments to zero. In this way, the original training spectra are decomposed into local spectral components. These components serve as spectrum terminals and can be recombined by GP functions to generate new spectra. Each training spectrum produces three spectrum terminals, and the total number of spectrum terminals is $N \times 3 ,$ , where N denotes the total number of training samples across all classes. Each spectrum terminal is a complex-valued array of type S.

Besides spectrum terminals, FreMGP uses three types of ERCs as scalar parameters for spectral transformations. Specifically, ERC-AS, ERC-PS, and ERC-FW correspond to amplitude scaling, phase shifting, and frequency warping, respectively.

b) Function Set: FreMGP defines four spectral functions: amplitude scaling (AS), phase shifting (PS), frequency warping (FW), and spectral fusion (SF), summarized in Table I. The AS function adjusts the magnitude of a spectrum, the PS function modifies its phase, the FW function performs a non-linear transformation along the frequency axis, and the SF function fuses multiple transformed spectral branches into the final output spectrum. These functions are defined as follows:

$$
\begin{array} { c } { \displaystyle \mathrm { A S } ( \mathbf { S } , a ) = a \mathbf { S } , } \\ { \displaystyle \mathrm { P S } ( \mathbf { S } , b ) = \mathbf { S } \odot \exp ( \varsigma b ) , } \\ { \displaystyle \mathrm { F W } ( \mathbf { S } , c _ { w } ) _ { k } = \mathcal { T } _ { \mathbf { S } } \big ( \boldsymbol { \omega } _ { k } ^ { c _ { w } } \big ) , } \\ { \displaystyle \mathrm { S F } ( \mathbf { S } _ { 1 } , \mathbf { S } _ { 2 } , \mathbf { S } _ { 3 } ) = \sum _ { i = 1 } ^ { 3 } \mathbf { S } _ { i } , } \end{array}\tag{10}
$$

where $\mathbf { S } \in \mathbb { C } ^ { F }$ denotes a spectrum, $a \in \mathsf { A }$ is an amplitude scaling coefficient, $b \in { \mathsf { B } }$ is a phase shifting coefficient, $c _ { w } \in$ C is a frequency warping coefficient, ς is the imaginary unit satisfying $\varsigma ^ { 2 } = - 1$ , ⊙ denotes element-wise multiplication, ω<sub>k</sub> denotes the normalized frequency location at frequency index k, and $\mathcal { T } _ { \mathbf { S } } ( \cdot )$ denotes a linear interpolation on S for obtaining the spectral value at the warped frequency location. In each tree, SF serves as the root node and corresponds to the fusion layer in Fig. 3(a), where different spectral branches are fused. The other functions, including AS, PS, and FW, are used in the transform layer for constructing spectral branches.

2) Target Sample Assignment and Two-Stage Evolutionary Search

Before the evolutionary oversampling process, FreMGP assigns one target sample to each of the $M _ { c }$ trees in an individual. These target samples are selected from the original samples of the minority class c. Specifically, the original samples of minority class c are first ranked in ascending order of their distances to the class center. These ranked samples are then assigned to the trees until all the $M _ { c }$ trees in an individual have been assigned their corresponding target samples. Note that this work focuses only on the imbalance case where the imbalance ratio (IR) is greater than 2. In that case, $M _ { c }$ is greater than $N _ { c } ~ ( N _ { c }$ is the number of training samples in the minority class $c )$ , and all the $N _ { c }$ original minority-class samples must be used at least once. Therefore, the assignment process starts from the beginning of the ranked list and repeats cyclically until all $M _ { c }$ trees have their own target samples.

Each tree then uses its assigned target sample as a reference to generate one synthetic sample. Note that multiple trees in the same individual may share the same target minority-class sample for generation.

In FreMGP, the evolutionary oversampling process consists of two stages. Stage I is called Representation Proximity Search, which aims to drive generated samples toward the target minority-class region in the learned representation space. In this stage, the evolutionary search focuses mainly on improving the representation proximity between generated samples and their corresponding target minority-class samples. Stage II is called Local Distribution Expansion Search. This stage further improves the generated sample set by considering both similarity and diversity. Specifically, generated samples are encouraged to remain close to the target minority-class region while expanding in different directions around the target samples. Therefore, Stage II promotes diversity while keeping generated samples close to the target minority-class region.

FreMGP employs an adaptive criterion to determine when to transition from Stage I to Stage II. Let $\bar { \mathcal { F } } _ { \mathrm { I } } ^ { ( e ) }$ denote the average fitness values of the population at generation e in Stage I. The proximity threshold δ is adaptively determined according to the average fitness of the initial population in Stage I, defined as:

$$
\delta = \bar { \mathcal { F } } _ { \mathrm { I } } ^ { ( 0 ) } + \lambda \left( 1 - \bar { \mathcal { F } } _ { \mathrm { I } } ^ { ( 0 ) } \right) ,\tag{11}
$$

where $\bar { \mathcal { F } } _ { \mathrm { { I } } } ^ { ( 0 ) }$ is the average fitness of the initial population in Stage I, and λ controls how far the threshold moves from the initial fitness level toward the ideal fitness value of 1.

If the average fitness of Stage I remains above the threshold δ for five consecutive generations, the population is considered to have reached a reliable minority-class region in the learned representation space. The search then switches from Stage I to Stage II.

## 3) Fitness Functions in the Two-Stage Search

After the representation space is learned, the encoder $E _ { \theta } ( \cdot )$ is fixed and used to evaluate generated samples during evolutionary search. For the m-th generated frequency-domain sample $\mathbf { \hat { X } } _ { c } ^ { ( m ) }$ of class $c ,$ its representation is computed as:

$$
\widehat { \mathbf { h } } _ { c } ^ { ( m ) } = E _ { \theta } \left( \widehat { \mathbf { X } } _ { c } ^ { ( m ) } \right) .\tag{12}
$$

a) The fitness function in Stage I: This fitness function is designed to drive the Representation Proximity Search, guiding generated samples toward their assigned target samples in the learned representation space. For the m-th tree in the current individual, its representation proximity score is defined as:

$$
s _ { c } ^ { ( m ) } = \Phi \left( \left\| \widehat { \mathbf { h } } _ { c } ^ { ( m ) } - \mathbf { h } _ { c } ^ { ( m ) } \right\| _ { 2 } ; \rho \right) ,\tag{13}
$$

where $\mathbf { h } _ { c } ^ { ( m ) }$ is the original representation of the assigned target sample, $\widehat { \mathbf { h } } _ { c } ^ { ( m ) }$ is the representation of the generated sample produced by this tree, $\rho$ is an adaptive scale parameter computed as the median pairwise distance among target minority-class representations, and Φ $( d ; \rho )$ is the Gaussian mapping function defined as:

$$
\Phi ( d ; \rho ) = \exp \left( - \frac { d ^ { 2 } } { 2 \rho ^ { 2 } } \right)\tag{14}
$$

The major advantage of using Gaussian mapping is that its decay behavior is explicitly controlled by the scale parameter $\rho ,$ instead of using a fixed threshold to judge whether a generated sample is close enough.

Based on the tree-level representation proximity scores, the fitness function for Stage I is defined as:

$$
\mathcal { F } _ { \mathrm { I } } = \frac { 1 } { M _ { c } } \sum _ { m = 1 } ^ { M _ { c } } s _ { c } ^ { ( m ) } .\tag{15}
$$

According to this fitness function, when the representation of a generated sample produced by a tree is close to the representation of its assigned target sample, the obtained score $s _ { c } ^ { ( { \dot { m } } ) }$ approaches 1; otherwise, the score decreases as their distance increases. Thus, by averaging the proximity scores over all the $M _ { c }$ generated samples, a larger $\mathcal { F } _ { \mathrm { I } }$ score indicates that the samples generated by the trees are closer to their corresponding target minority-class samples in the learned representation space. Therefore, $\mathcal { F } _ { \mathrm { I } }$ encourages an individual to maintain representation proximity to the target minorityclass region.

b) The fitness function in Stage II: Although the fitness function in Stage I encourages generated samples to approach their assigned target samples, considering only the averaged representation proximity fitness may lead to a sample set with low diversity. This is because, based on the target sample assignment method, multiple trees in an individual may share the same target minority-class sample. To address the above issue, we categorize trees (i.e., generated samples) sharing the same target sample into a group, and evaluate the fitness of an individual in a group-wise manner in Stage II. Note that, within the same group, some generated samples may be very close to their target samples in the learned representation space while others remain relatively far away. This can still result in a high average fitness. Therefore, Stage II evaluates groups of generated samples by considering both radial consistency and angular diversity.

The radial consistency score is designed to measure whether generated samples within the same group are close to their shared target sample in the learned representation space while maintaining a stable expansion radius. Specifically, for the q-th group, we first compute the Euclidean distances between the representations of all generated samples in this group and the representation of their shared target sample. The group-level radial consistency score is defined as:

$$
S _ { \mathrm { r a d } } ^ { ( q ) } = \Phi \left( { { \bar { \Delta } } _ { q } } ; \boldsymbol { \rho } \right) \cdot \Phi \left( { \sigma _ { \Delta , q } } ; \boldsymbol { \rho } \right) ,\tag{16}
$$

where $\Phi ( \cdot ; \rho )$ is the Gaussian proximity mapping defined in equation (14), and $\rho$ is the adaptive scale parameter defined in Stage I. $\bar { \Delta } _ { q }$ and $\sigma _ { \Delta , q }$ denote the mean and standard deviation of the radial distances between the representations of the generated samples in the q-th group and the representation of their shared target sample:

$$
\bar { \boldsymbol { \Delta } } _ { q } = \frac { 1 } { K _ { q } } \sum _ { k = 1 } ^ { K _ { q } } \left\| \widehat { \mathbf { h } } _ { q , k } - \mathbf { h } _ { q } \right\| _ { 2 } ,\tag{17}
$$

![](images/8ef73b92b67f9ec2c9348bd5b74d1c4bee96c1310cf102dfa5f2f5642166a8a8.jpg)  
Fig. 4. Illustration of the stage-specific genetic operators in FreMGP. (a) Stage-I one-point subtree crossover with trees as the basic operation units. (b) Stage-I uniform subtree mutation with trees as the basic operation units. (c) Stage-II group-level repair-style crossover, where higher-scoring groups replace lower-scoring groups at the same group positions. (d) Stage-II group-level mutation, where participating groups are refined by mutating their priority trees.

$$
\sigma _ { \Delta , q } = \sqrt { \frac { 1 } { K _ { q } } \sum _ { k = 1 } ^ { K _ { q } } { \left( \left\| \widehat { \mathbf { h } } _ { q , k } - \mathbf { h } _ { q } \right\| _ { 2 } - \bar { \Delta } _ { q } \right) ^ { 2 } } } ,\tag{18}
$$

where $\widehat { \mathbf { h } } _ { q , k }$ denotes the representation of the k-th generated sample in the q-th group, $\mathbf { h } _ { q }$ denotes the shared target representation of this group, and $K _ { q }$ is the number of generated samples in this group.

This score $S _ { \mathrm { r a d } } ^ { ( q ) }$ measures radial consistency directly. The first factor evaluates the average proximity of generated samples to the shared target sample in the learned representation space, while the second evaluates the consistency of their radial distances. The product in $S _ { \mathrm { r a d } } ^ { ( q ) }$ assigns a high score only when both requirements are satisfied.

Although radial consistency encourages generated samples close to the shared target sample in the learned representation space with stable radial distances, it does not explicitly encourage them to move toward different directions. Therefore, FreMGP further introduces an angular diversity score. For each generated sample, we compute its direction of movement by normalizing the vector from the corresponding target representation to the representation of the generated sample:

$$
\pmb { \xi } _ { q , k } = \frac { \widehat { \mathbf { h } } _ { q , k } - \mathbf { h } _ { q } } { \left\| \widehat { \mathbf { h } } _ { q , k } - \mathbf { h } _ { q } \right\| _ { 2 } + \epsilon } ,\tag{19}
$$

where $\widehat { \mathbf { h } } _ { q , k }$ denotes the representation of the k-th generated sample in the q-th group, $\mathbf { h } _ { q }$ denotes the shared target representation of this group, and ϵ is a small constant for numerical stability.

Therefore, the group-level angular diversity score is defined as:

$$
S _ { \mathrm { a n g } } ^ { ( q ) } = \frac { 2 } { K _ { q } ( K _ { q } - 1 ) } \sum _ { 1 \le k < s \le K _ { q } } \frac { 1 - \xi _ { q , k } ^ { \top } \xi _ { q , s } } { 2 } ,\tag{20}
$$

where $K _ { q }$ is the number of generated samples in the q-th group. This score averages the pairwise directional differences among generated samples in the same group. A higher $S _ { \mathrm { a n g } } ^ { ( q ) }$ indicates that the generated samples move along more diverse directions around the shared target representation. Note that if a group contains only one generated sample, there is no pairwise directional difference to compute, and its angular diversity score is set to the radial consistency score.

The total score of group q is defined as:

$$
G _ { q } = \alpha S _ { \mathrm { r a d } } ^ { ( q ) } + ( 1 - \alpha ) S _ { \mathrm { a n g } } ^ { ( q ) } ,\tag{21}
$$

where α is a weight to balance between radial consistency and angular diversity.

Therefore, the fitness function in Stage II is defined as:

$$
\mathcal { F } _ { \mathrm { I I } } = \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } G _ { q } ,\tag{22}
$$

where $Q$ is the number of groups.

## 4) Genetic Operators

We design different genetic operators for evolution in different search stages. Specifically, genetic operations are conducted on individual trees in Stage I, but at the group level in Stage II.

a) The genetic operators in Stage I: To maintain consistency with the per-tree evaluation in Stage I, FreMGP conducts genetic operations at the tree level. Each tree in an individual has its own representation proximity score, calculated by equation (13), which can be used to calculate its participation probability for crossover or mutation. The participation probability of tree $T _ { m }$ is defined as:

$$
p _ { m } = \frac { s _ { \mathrm { m a x } } - s _ { c } ^ { ( m ) } } { \sum _ { j = 1 } ^ { M _ { c } } \left( s _ { \mathrm { m a x } } - s _ { c } ^ { ( j ) } \right) } ,\tag{23}
$$

where $s _ { \mathrm { m a x } } ~ = ~ \mathrm { m a x } _ { 1 \leq j \leq M _ { c } } ~ s _ { c } ^ { ( j ) }$ is the maximum tree-level representation proximity score within the current individual. According to this equation, lower-scoring trees receive higher probabilities.

Within each parent individual, half of the trees are selected probabilistically according to their participation probabilities to undergo crossover or mutation, while the other half remain unchanged. As shown in Fig. 4(a), the selected trees are then paired according to their participation order, and one-point sub-tree crossover is performed on each pair. As shown in Fig. 4(b), the mutation operator uses the same per-tree participation mechanism and applies uniform sub-tree mutation to breed.

b) The genetic operators in Stage II: In Stage II, each individual is evaluated based on a group of generated samples associated with the same target sample. Therefore, genetic operations at this stage are also performed at the group level.

As shown in Fig. 4(c), the crossover operator adopts a group-level strategy. For the same group position in two parent individuals, the group with the higher score (calculated by equation (21)) in one parent is copied to replace the lowerscoring group at the same position in the other parent. In Stage II, only 25% of the group positions are allowed to perform crossover. For two parent individuals, each group position is assigned a crossover participation probability according to the score difference between the two groups at the same position. The participation probability of the q-th group position is defined as:

$$
p _ { q } ^ { \mathrm { c x } } = \frac { \left| G _ { q } ^ { ( 1 ) } - G _ { q } ^ { ( 2 ) } \right| } { \sum _ { r = 1 } ^ { Q } \left| G _ { r } ^ { ( 1 ) } - G _ { r } ^ { ( 2 ) } \right| } ,\tag{24}
$$

where $G _ { q } ^ { ( 1 ) }$ and $G _ { q } ^ { ( 2 ) }$ denote the scores of the q-th group in the two parent individuals, respectively, and Q is the total number of groups. This probability is proportional to the score difference between the two parent groups at the same position. Therefore, group positions with larger quality gaps are more likely to participate in crossover.

As shown in Fig. 4(d), mutation in Stage II is also performed with groups. To give low-quality groups more opportunities for refinement, groups with lower scores are assigned higher mutation participation probabilities. The mutation participation probability of group q is defined as:

$$
p _ { q } ^ { \mathrm { m u t } } = \frac { G _ { \mathrm { m a x } } - G _ { q } } { \sum _ { r = 1 } ^ { Q } \left( G _ { \mathrm { m a x } } - G _ { r } \right) } ,\tag{25}
$$

where $G _ { q }$ denotes the score of the q-th group, Q is the total number of groups, and $G _ { \mathrm { m a x } } = \mathrm { m a x } _ { 1 \leq r \leq Q } G _ { \prime }$ is the maximum group score in the current individual. This probability is inversely related to the group score, so groups with lower $G _ { q }$ values are more likely to participate in mutation.

The same as the crossover operator in Stage II, only 25% of the groups in an individual are allowed to mutate according to their mutation participation probability. Within each mutation group, FreMGP further prioritizes a single tree to be mutated. The priority of the k-th tree in the q-th group is computed using a leave-one-out strategy, i.e., a tree is assigned a higher priority if its removal leads to a larger increase in the group score. This is defined as:

$$
\begin{array} { r } { \eta _ { q , k } = \alpha \left[ S _ { \mathrm { r a d } } ^ { ( q , - k ) } - S _ { \mathrm { r a d } } ^ { ( q ) } \right] _ { + } + \left( 1 - \alpha \right) \left[ S _ { \mathrm { a n g } } ^ { ( q , - k ) } - S _ { \mathrm { a n g } } ^ { ( q ) } \right] _ { + } } \end{array}\tag{26}
$$

where $S _ { \mathrm { r a d } } ^ { ( q , - k ) }$ and $S _ { \mathrm { a n g } } ^ { ( q , - k ) }$ denote the radial consistency and angular diversity scores after removing the k-th tree from the q-th group, and $[ \cdot ] _ { + }$ denotes the positive part. Here, α is the same weighting parameter as in equation (21). A larger $\eta _ { q , \ l }$ indicates that removing this tree improves the group quality more, so this tree is assigned a higher priority.

c) The elitism strategies: To retain high-quality trees, groups, and individuals during evolution, FreMGP uses elitism in both stages. In Stage I, the tree-level elitism is performed. At each tree position index, FreMGP compares the corresponding trees from different individuals and selects the one with the highest representation proximity score (equation (13)). This process continues until every tree position index has identified its best-performing tree. Afterwards, the best-performing trees identified at each tree position index are recombined to form new elite individuals. This tree-level elitism preserves the trees that have already been able to generate high-quality samples in Stage I.

TABLE II  
DATASETS IN THE EXPERIMENTS
<table><tr><td>Dataset</td><td>No.sample</td><td>Class</td><td>IR</td><td>Length</td></tr><tr><td>MixedShapesSmallTrain</td><td>60</td><td>5</td><td>2.22</td><td>1024</td></tr><tr><td>SonyAIBORobotSurface1</td><td>20</td><td>2</td><td>2.33</td><td>70</td></tr><tr><td>TwoLeadECG</td><td>17</td><td>2</td><td>2.4</td><td>82</td></tr><tr><td>Ham</td><td>71</td><td>2</td><td>4.07</td><td>431</td></tr><tr><td>ECG200</td><td>80</td><td>2</td><td>6.27</td><td>96</td></tr><tr><td>MiddlePhalanxOutlineCorrect</td><td>438</td><td>2</td><td>7.76</td><td>80</td></tr><tr><td>SyntheticControl</td><td>90</td><td>6</td><td>10</td><td>60</td></tr><tr><td>DistalPhalanxOutlineAgeGroup</td><td>377</td><td>3</td><td>10.71</td><td>80</td></tr><tr><td>SmoothSubspace</td><td>68</td><td>3</td><td>12.5</td><td>15</td></tr><tr><td>PowerCons</td><td>97</td><td>2</td><td>12.86</td><td>144</td></tr><tr><td>SwedishLeaf</td><td>140</td><td>15</td><td>14</td><td>128</td></tr><tr><td>Strawberry</td><td>414</td><td>2</td><td>19.7</td><td>235</td></tr><tr><td>StarLightCurves</td><td>612</td><td>3</td><td>40.93</td><td>1024</td></tr><tr><td>Wafer</td><td>917</td><td>2</td><td>64.42</td><td>152</td></tr></table>

Note: For multi-class datasets, IR is the ratio of the number of samples in the largest class to the number in the smallest class.

In Stage II, the group-level elitism is performed. At each group index, FreMGP compares scores of groups (calculated by the equation (21)) from different individuals and selects the better-performing group. This process continues until every group position index has identified its best-performing group. Afterwards, the best-performing groups identified at each group index are recombined to form new elite individuals. In addition, in Stage II, FreMGP also directly retains the individual with the highest fitness value. This allows preserving both locally well-performing groups and globally well-performing individuals.

## IV. EXPERIMENTAL DESIGN

## A. Datasets

In our experiments, we use 14 time-series datasets from the UCR Archive<sup>1</sup>. These datasets cover diverse domains, including healthcare, food analysis, industrial monitoring, robotics, and other real-world scenarios. Note that the UCR datasets provide predefined training/test splits. Following [11], we retain the original test sets and perform stratified sampling on the training sets to produce new imbalanced datasets that cover a wider range of IR values. Table II reports the number of training samples after stratified sampling, the number of classes, the IR, and the time-series length. The IR values range from 2.22 to 64.42.

## B. Baseline Methods

The baseline methods cover representative oversampling methods, including SMOTE [9], Borderline-SMOTE1 (abbreviated as BL-SMOTE) [39], ADASYN [40], SVM-SMOTE [47], and KMeans-SMOTE (abbreviated as KM-SMOTE) [48]. We also include time-series-specific oversampling methods, including T-SMOTE [11], INOS [49], and OHIT [50]. Furthermore, deep-learning-based generative methods, including CSMOTE [51], BFGAN [42], CFAMG [7], and ImagenFew [17], are included as baseline method. Finally, Evo-TFS [20] is also compared since it is the recent GPbased evolutionary oversampling method. Overall, these baseline methods span a wide range of oversampling strategies, allowing a comprehensive comparison with FreMGP. Detailed descriptions on these baseline methods are provided in $\mathsf { A p - }$ pendix C.

TABLE III  
PARAMETER SETTINGS
<table><tr><td>Module</td><td>Parameter</td><td>Value</td></tr><tr><td rowspan="4">learning</td><td>Encoder output dimension  $\overline { { d _ { h } } }$ </td><td>256</td></tr><tr><td>Representation Projection output dimension  $d _ { z }$ </td><td>128</td></tr><tr><td>Temperature τ</td><td>0.1</td></tr><tr><td>Prototype gate threshold γ</td><td>0.5</td></tr><tr><td rowspan="9">Evolutionary oversampling</td><td>Initialization method</td><td>Ramped half-and-half</td></tr><tr><td>Number of generations</td><td>100</td></tr><tr><td>Population size</td><td>128</td></tr><tr><td>Tournament size</td><td>3</td></tr><tr><td>Crossover rate</td><td>0.8</td></tr><tr><td>Mutation rate</td><td>0.2</td></tr><tr><td>Maximum tree depth</td><td>10</td></tr><tr><td>Stage transition coefficient λ</td><td>0.7</td></tr><tr><td>α</td><td>0.5</td></tr></table>

We employ three types of classifiers, namely LSTM networks [52], Transformer [53], and Mamba [54], to evaluate the quality of the generated time-series data. Each classifier is trained on the rebalanced training set generated by FreMGP or a baseline method, and then evaluated on the same original test set.

## C. Parameter Settings

Table III presents the parameter configurations of the proposed FreMGP method. Following common practices in contrastive representation learning [22], [24], the encoder output dimension $d _ { h }$ and the projection dimension $d _ { z }$ are set to 256 and 128, respectively, to provide a practical balance between representation capacity and computational cost. The temperature parameter τ is set to 0.1, which is a commonly used setting in supervised contrastive learning [24]. In addition, inspired by the prototype-based gating strategy for imbalanced supervised contrastive learning [45], the prototype gate threshold γ is set to 0.5 to selectively activate the prototypeguided loss. Other implementation details of the representation learning module, such as the optimizer and training epochs, are provided in the Appendix B.2.

For the evolutionary oversampling module, the population size is set to 128, and the maximum number of generations is set to 100 to ensure a sufficient number of evaluations. To prevent tree bloating during evolution, the maximum tree depth is set to 10. For the adaptive two-stage search, the stage transition coefficient λ is set to 0.7. These settings ensure that the search enters Stage II only after the population has reached a reliable representation proximity level in Stage I. The weight α is set to 0.5 to equally balance radial consistency and angular diversity in the Stage-II fitness evaluation. FreMGP employs stage-specific elitism strategies. At Stage I, three new elite individuals are reconstructed based on the tree-level elitism; at Stage II, one new elite individuals are reconstructed based on the group-level elitism and two top individuals are directly retained from the last generation.

The representation learning module was implemented using PyTorch, while the multi-tree genetic programming was implemented based on the DEAP package [55]. The conventional sampling baselines were implemented using the imbalancedlearn package [56], and the other baseline methods were implemented based on their publicly available source code.

## V. RESULTS AND ANALYSIS

On each dataset, FreMGP is independently executed 30 times using 30 different random seeds. Three classifiers, LSTM, Transformer and Mamba, are trained on the training sets rebalanced by different sampling methods, and then their classification performances are evaluated on the same original test set, using F1-Score [57], G-Mean [8], and AUC [58]. The Wilcoxon signed-rank test [59] with a significance level of 0.05 and the Holm–Bonferroni correction [60] for multiple comparisons have been conducted to assess whether the performance difference between FreMGP and a baseline method is statistically significant.

## A. Overall Performance and Analysis

Table IV reports the average classification performance of LSTM, Transformer, and Mamba classifiers using different sampling methods. The row “None” denotes the results obtained without using any oversampling method. The detailed results on each dataset are provided in the Appendix D.1. According to Table IV, compared with “None”, most oversampling methods can assist the three classifiers to improve the classification performance. This suggests that rebalancing the training set is necessary for imbalanced time-series classification. Among all the methods in the experiments, FreMGP achieves the best average results across all three classifiers and all three evaluation metrics. This demonstrates its effectiveness in rebalancing data to assist different classifiers to address the performance bias issue for ITSC tasks.

Table V reports the statistical comparison between FreMGP and each baseline method under different classifiers and evaluation metrics. Here, $" ^ { 6 } + " , " = " ,$ , and “−” denote that FreMGP is significantly better than, not significantly different from, and significantly worse than the corresponding baseline method, respectively. For F1-Score, FreMGP achieves significantly better or statistically comparable results than the baseline methods in 172, 176, and 174 out of 182 total cases for the LSTM, Transformer, and Mamba classifiers, respectively. Similar trends can also be observed for G-Mean, where FreMGP achieves significantly better or statistically comparable results than the baseline methods in 175, 176, and 176 out of 182 total cases for the three classifiers, respectively. For AUC, FreMGP also outperforms the baseline methods in most cases.

Table VI reports the average rankings of different oversampling methods over the 14 datasets under different classifiers and evaluation metrics, where a smaller ranking value indicates better performance. Overall, FreMGP achieves the best average ranking across the three classifiers under all performance measures. This indicates that FreMGP is agnostic to specific classifiers and performs effectively across different time-series classifiers. Some baseline methods also perform effectively in certain cases. For example, OHIT ranks second under LSTM, and INOS performs competitively under Transformer and Mamba.

TABLE IV  
OVERALL PERFORMANCE COMPARISON OF FREMGP AND BASELINE METHODS UNDER DIFFERENT CLASSIFIERS AND EVALUATION METRICS
<table><tr><td rowspan="2">Methods</td><td colspan="3">LSTM</td><td colspan="3">Transformer</td><td colspan="3">Mamba</td></tr><tr><td>F1-Score</td><td>G-Mean</td><td>AUC</td><td>F1-Score</td><td>G-Mean</td><td>AUC</td><td>F1-Score</td><td>G-Mean</td><td>AUC</td></tr><tr><td>None</td><td>0.4685</td><td>0.2254</td><td>0.7358</td><td>0.6042</td><td>0.4641</td><td>0.8505</td><td>0.6721</td><td>0.5614</td><td>0.8930</td></tr><tr><td>SMOTE</td><td>0.6899</td><td>0.6739</td><td>0.8231</td><td>0.7163</td><td>0.6882</td><td>0.8666</td><td>0.7540</td><td>0.7243</td><td>0.8952</td></tr><tr><td>BL-SMOTE</td><td>0.6691</td><td>0.6261</td><td>0.8218</td><td>0.7109</td><td>0.6767</td><td>0.8626</td><td>0.7404</td><td>0.7023</td><td>0.8787</td></tr><tr><td>ADASYN</td><td>0.6849</td><td>0.6671</td><td>0.8232</td><td>0.7292</td><td>0.7041</td><td>0.8769</td><td>0.7517</td><td>0.7206</td><td>0.8975</td></tr><tr><td>SVM-SMOTE</td><td>0.6668</td><td>0.6346</td><td>0.8133</td><td>0.7106</td><td>0.6731</td><td>0.8729</td><td>0.7496</td><td>0.7132</td><td>0.8889</td></tr><tr><td>KM-SMOTE</td><td>0.6830</td><td>0.6524</td><td>0.8226</td><td>0.7049</td><td>0.6616</td><td>0.8589</td><td>0.7434</td><td>0.7025</td><td>0.8844</td></tr><tr><td>T-SMOTE</td><td>0.5549</td><td>0.3984</td><td>0.7818</td><td>0.6663</td><td>0.5792</td><td>0.8436</td><td>0.7561</td><td>0.7196</td><td>0.8876</td></tr><tr><td>INOS</td><td>0.6791</td><td>0.6396</td><td>0.8284</td><td>0.7340</td><td>0.7090</td><td>0.8859</td><td>0.7734</td><td>0.7493</td><td>0.9094</td></tr><tr><td>OHIT</td><td>0.7043</td><td>0.6871</td><td>0.8309</td><td>0.7266</td><td>0.6973</td><td>0.8794</td><td>0.7512</td><td>0.7040</td><td>0.9011</td></tr><tr><td>CSMOTE</td><td>0.6696</td><td>0.6506</td><td>0.8278</td><td>0.7101</td><td>0.6802</td><td>0.8811</td><td>0.7389</td><td>0.6938</td><td>0.8998</td></tr><tr><td>BFGAN</td><td>0.5222</td><td>0.3544</td><td>0.7944</td><td>0.6306</td><td>0.4790</td><td>0.8443</td><td>0.6903</td><td>0.5996</td><td>0.8829</td></tr><tr><td>CFAMG</td><td>0.4184</td><td>0.2561</td><td>0.7038</td><td>0.4530</td><td>0.2726</td><td>0.7199</td><td>0.4036</td><td>0.1915</td><td>0.7341</td></tr><tr><td>ImagenFew</td><td>0.5274</td><td>0.3131</td><td>0.7590</td><td>0.6013</td><td>0.4648</td><td>0.8267</td><td>0.6916</td><td>0.5878</td><td>0.8439</td></tr><tr><td>Evo-TFS</td><td>0.6289</td><td>0.5515</td><td>0.7995</td><td>0.6774</td><td>0.5975</td><td>0.8647</td><td>0.7299</td><td>0.6729</td><td>0.8898</td></tr><tr><td>FreMGP</td><td>0.7176</td><td>0.7049</td><td>0.8461</td><td>0.7621</td><td>0.7454</td><td>0.8971</td><td>0.8042</td><td>0.7878</td><td>0.9169</td></tr></table>

TABLE V  
STATISTICAL WIN/TIE/LOSS COMPARISON OF FREMGP AGAINST BASELINE METHODS UNDER DIFFERENT CLASSIFIERS AND EVALUATION METRICS
<table><tr><td rowspan="2">Methods</td><td colspan="3">F1-Score</td><td colspan="3">G-Mean</td><td colspan="3">AUC</td></tr><tr><td>LSTM (+/=/-)</td><td>Transformer (+/=/-)</td><td>Mamba (+/=/-)</td><td>LSTM (+/=/-)</td><td>Transformer (+/=/-)</td><td>Mamba (+/=/-)</td><td>LSTM (+/=/-)</td><td>Transformer (+/=/-)</td><td>Mamba (+/=/-)</td></tr><tr><td>SMOTE</td><td>6/7/1</td><td>10/4/0</td><td>9/5/0</td><td>6/8/0</td><td>11/3/0</td><td>11/3/0</td><td>6/8/0</td><td>9/5/0</td><td>8/5/1</td></tr><tr><td>BL-SMOTE</td><td>8/6/0</td><td>12/2/0</td><td>11/2/1</td><td>8/6/0</td><td>12/2/0</td><td>12/1/1</td><td>6/8/0</td><td>11/2/1</td><td>9/4/1</td></tr><tr><td>ADASYN</td><td>6/7/1</td><td>10/2/2</td><td>9/5/0</td><td>8/6/0</td><td>10/2/2</td><td>10/4/0</td><td>6/8/0</td><td>7/6/1</td><td>6/5/3</td></tr><tr><td>SVM-SMOTE</td><td>8/6/0</td><td>11/3/0</td><td>10/4/0</td><td>9/5/0</td><td>13/1/0</td><td>10/4/0</td><td>7/7/0</td><td>9/3/2</td><td>7/6/1</td></tr><tr><td>KM-SMOTE</td><td>5/9/0</td><td>11/3/0</td><td>10/4/0</td><td>7/7/0</td><td>12/2/0</td><td>10/4/0</td><td>7/6/1</td><td>9/5/0</td><td>7/6/1</td></tr><tr><td>T-SMOTE</td><td>12/2/0</td><td>9/5/0</td><td>8/4/2</td><td>12/2/0</td><td>9/5/0</td><td>9/3/2</td><td>12/1/1</td><td>9/3/2</td><td>7/4/3</td></tr><tr><td>INOS</td><td>7/7/0</td><td>7/6/1</td><td>9/4/1</td><td>5/9/0</td><td>6/7/1</td><td>9/4/1</td><td>6/7/1</td><td>7/5/2</td><td>5/5/4</td></tr><tr><td>OHIT</td><td>4/7/3</td><td>7/7/0</td><td>7/6/1</td><td>3/8/3</td><td>6/8/0</td><td>9/4/1</td><td>3/8/3</td><td>7/5/2</td><td>7/3/4</td></tr><tr><td>CSMOTE</td><td>9/3/2</td><td>11/3/0</td><td>11/3/0</td><td>10/2/2</td><td>11/3/0</td><td>11/3/0</td><td>9/4/1</td><td>8/3/3</td><td>8/1/5</td></tr><tr><td>BFGAN</td><td>13/0/1</td><td>13/1/0</td><td>13/0/1</td><td>12/1/1</td><td>13/1/0</td><td>13/1/0</td><td>10/2/2</td><td>10/2/2</td><td>7/6/1</td></tr><tr><td>CFAMG</td><td>14/0/0</td><td>13/0/1</td><td>14/0/0</td><td>13/1/0</td><td>13/0/1</td><td>14/0/0</td><td>14/0/0</td><td>13/0/1</td><td>13/0/1</td></tr><tr><td>ImagenFew</td><td>12/0/2</td><td>11/1/2</td><td>11/2/1</td><td>12/1/1</td><td>11/1/2</td><td>11/2/1</td><td>12/2/0</td><td>11/1/2</td><td>9/4/1</td></tr><tr><td>Evo-TFS</td><td>12/2/0</td><td>10/4/0</td><td>10/4/0</td><td>11/3/0</td><td>12/2/0</td><td>11/3/0</td><td>9/5/0</td><td>10/3/1</td><td>8/5/1</td></tr><tr><td>Total</td><td>116/56/10</td><td>135/41/6</td><td>132/43/7</td><td>116/59/7</td><td>139/37/6</td><td>140/36/6</td><td>107/66/9</td><td>120/43/19</td><td>101/54/27</td></tr></table>

## B. Minority-Class Recognition Analysis Based on Confusion Matrices

To determine whether an oversampling method improves classification performance at the class level, we take the Strawberry dataset as an example, and confusion matrices are presented for qualitative analysis. Due to the page limit, only the confusion matrices obtained with the Mamba classifier are shown in the main paper, while the other results under the LSTM and Transformer classifiers are provided in Appendix D.3. As shown in Fig. 5, the confusion matrices on the Strawberry dataset show that most methods obtain high accuracy for Class 2, while their performance on Class 1 varies more clearly. Therefore, the main difference among the compared methods lies in whether they can improve the accuracy on Class 1 without largely sacrificing the accuracy on Class 2.

After using FreMGP to rebalance data, Mamba achieves an accuracy of 0.88 on Class 1 while maintaining an accuracy of 0.96 on Class 2. Using INOS, Mamba also performs well on Class 1, with an accuracy of 0.86. However, compared with INOS, FreMGP assists Mamba in obtaining slightly higher accuracies for both classes. Several sampling methods, such as T-SMOTE, BFGAN, ImagenFew, and Evo-TFS, are less effective in assisting Mamba to accurately recognize samples from Class 1, although their accuracies on Class 2 are high. These observations suggest that FreMGP can help Mamba improve the recognition accuracy on the minority class without degrading the recognition accuracy on the majority class.

## C. Performance Evaluation of FreMGP on State-of-the-Art (SOTA) Classifiers

To further examine whether FreMGP can be applied to SOTA time-series classifiers, we evaluate its performance with

![](images/cce1ccc1ed1b88e791dfe65beacfa0dcb2d0e916cc1aa06731a22b8fb7e1d318.jpg)

![](images/6d8dda250c868e04108ea8b335492769386cd01acddf0ec4186960742f48ccca.jpg)

![](images/c5eb17843704eab922784853311a08a697699f0ce70c193fb9e94d573c1b58e4.jpg)

![](images/bc34b2e467d140d8982f6b749792979685ff224dd164638a74c26583d3a14fc7.jpg)

![](images/ce58dcd78c747a57b0208d7349b15ae7b756cefe75b27a1478bd307d7a45041b.jpg)

TABLE VI  
AVERAGE RANKING COMPARISON OF DIFFERENT OVERSAMPLING METHODS UNDER DIFFERENT CLASSIFIERS
<table><tr><td rowspan="2">Methods</td><td colspan="3">LSTM</td><td colspan="3">Transformer</td><td colspan="3">Mamba</td></tr><tr><td>F1-Score</td><td>G-Mean</td><td>AUC</td><td>F1-Score</td><td>G-Mean</td><td>AUC</td><td>F1-Score</td><td>G-Mean</td><td>AUC</td></tr><tr><td>SMOTE</td><td>5.64</td><td>5.64</td><td>6.50</td><td>8.18</td><td>7.68</td><td>9.07</td><td>7.50</td><td>7.50</td><td>7.71</td></tr><tr><td>BL-SMOTE</td><td>7.21</td><td>7.21</td><td>8.54</td><td>7.89</td><td>7.39</td><td>8.46</td><td>8.21</td><td>8.18</td><td>9.25</td></tr><tr><td>ADASYN</td><td>6.36</td><td>6.07</td><td>7.54</td><td>6.07</td><td>6.07</td><td>6.50</td><td>6.79</td><td>7.00</td><td>7.07</td></tr><tr><td>SVM-SMOTE</td><td>7.39</td><td>7.21</td><td>7.68</td><td>8.04</td><td>8.07</td><td>6.36</td><td>7.18</td><td>7.04</td><td>7.50</td></tr><tr><td>KM-SMOTE</td><td>5.89</td><td>6.29</td><td>6.57</td><td>8.39</td><td>8.75</td><td>8.29</td><td>7.43</td><td>7.75</td><td>8.07</td></tr><tr><td>T-SMOTE</td><td>10.57</td><td>10.64</td><td>10.57</td><td>8.00</td><td>8.21</td><td>8.50</td><td>6.57</td><td>6.54</td><td>7.32</td></tr><tr><td>INOS</td><td>6.39</td><td>6.00</td><td>6.43</td><td>4.96</td><td>4.89</td><td>5.39</td><td>5.61</td><td>5.43</td><td>6.57</td></tr><tr><td>OHIT</td><td>3.68</td><td>3.75</td><td>5.68</td><td>5.64</td><td>5.64</td><td>5.57</td><td>6.39</td><td>6.68</td><td>5.57</td></tr><tr><td>CSMOTE</td><td>6.89</td><td>6.89</td><td>5.71</td><td>7.82</td><td>7.36</td><td>7.46</td><td>8.21</td><td>8.54</td><td>5.75</td></tr><tr><td>BFGAN</td><td>11.64</td><td>11.36</td><td>9.25</td><td>10.68</td><td>11.71</td><td>9.29</td><td>10.82</td><td>10.89</td><td>9.32</td></tr><tr><td>CFAMG</td><td>11.89</td><td>12.14</td><td>12.71</td><td>12.86</td><td>12.36</td><td>13.54</td><td>13.64</td><td>13.18</td><td>13.86</td></tr><tr><td>ImagenFew</td><td>11.00</td><td>11.57</td><td>10.79</td><td>10.64</td><td>9.96</td><td>10.57</td><td>10.36</td><td>10.07</td><td>10.50</td></tr><tr><td>Evo-TFS</td><td>8.96</td><td>8.96</td><td>9.14</td><td>8.18</td><td>8.61</td><td>8.57</td><td>8.46</td><td>8.32</td><td>9.21</td></tr><tr><td>FreMGP</td><td>3.61</td><td>3.00</td><td>4.00</td><td>2.43</td><td>2.29</td><td>3.82</td><td>2.79</td><td>2.57</td><td>4.96</td></tr></table>

![](images/746306d2d484bc72a09d9e5d1c5c010eb579426d566a3e179dfa0956acfd2f35.jpg)

Fig. 5. Comparison of confusion matrices for different oversampling methods on the Strawberry dataset with the Mamba classifier.  
![](images/f71ef223de98a85fb47b245c12074bce02bf9630d71275ef09bf66fc614548f3.jpg)  
Fig. 6. Performance comparison between the original imbalanced training data and FreMGP on additional classifiers.

Hydra [2] and SARC [3]. As shown in Fig. 6, FreMGP improves the performance of both classifiers over the original imbalanced training data across all the three evaluation metrics. For Hydra, FreMGP increases the F1-Score result from 0.809 to 0.847 and the G-Mean result from 0.774 to 0.828, while the AUC result shows a smaller improvement from 0.935 to 0.947. A similar trend can be observed for SARC, where FreMGP improves the F1-Score result from 0.782 to 0.826 and the G-Mean from 0.739 to 0.811, with a slight increase in AUC performance from 0.930 to 0.937. These results suggest that the samples generated by FreMGP can provide useful additional training information for different classifiers. The detailed results on each dataset are provided in Appendix D.4.

## D. Ablation Analysis

Table VII reports the ablation study results averaged over the 14 datasets under LSTM, Transformer, and Mamba classifiers. To examine the contribution of each component, we construct several variants of FreMGP. The variant “w/o Gen. Ops.” removes the proposed stage-specific genetic-operator design. The variant “w/o Proto. Loss” removes the prototype-guided loss from the representation learning module. The variants “w/o Stage I” and “w/o Stage II” remove the representation proximity search and the local distribution expansion search, respectively. The variants “w/o Diversity” and “w/o Similarity” remove the angular diversity term and the radial consistency term in the Stage-II fitness function, respectively. The detailed dataset-level results are provided in Appendix D.5.

TABLE VII  
ABLATION STUDY RESULTS UNDER DIFFERENT CLASSIFIERS ON 14 DATASETS
<table><tr><td>Classifier</td><td>Metric</td><td>w/o Gen. Ops.</td><td>w/o Proto. Loss</td><td>w/o Stage I</td><td>w/o Stage ⅡI</td><td>w/o Diversity</td><td>w/o Similarity</td><td>FreMGP</td></tr><tr><td rowspan="3">LSTM</td><td>F1-Score</td><td>0.592</td><td>0.678</td><td>0.617</td><td>0.684</td><td>0.695</td><td>0.678</td><td>0.718</td></tr><tr><td>G-Mean</td><td>0.509</td><td>0.646</td><td>0.536</td><td>0.659</td><td>0.671</td><td>0.647</td><td>0.705</td></tr><tr><td>AUC</td><td>0.770</td><td>0.822</td><td>0.785</td><td>0.829</td><td>0.834</td><td>0.822</td><td>0.846</td></tr><tr><td rowspan="3">Transformer</td><td>F1-Score</td><td>0.694</td><td>0.742</td><td>0.737</td><td>0.754</td><td>0.750</td><td>0.748</td><td>0.762</td></tr><tr><td>G-Mean</td><td>0.648</td><td>0.715</td><td>0.708</td><td>0.733</td><td>0.729</td><td>0.731</td><td>0.745</td></tr><tr><td>AUC</td><td>0.840</td><td>0.875</td><td>0.871</td><td>0.877</td><td>0.876</td><td>0.878</td><td>0.897</td></tr><tr><td rowspan="3">Mamba</td><td>F1-Score</td><td>0.719</td><td>0.766</td><td>0.748</td><td>0.783</td><td>0.792</td><td>0.783</td><td>0.804</td></tr><tr><td>G-Mean</td><td>0.656</td><td>0.724</td><td>0.704</td><td>0.759</td><td>0.773</td><td>0.758</td><td>0.788</td></tr><tr><td>AUC</td><td>0.879</td><td>0.900</td><td>0.897</td><td>0.898</td><td>0.905</td><td>0.904</td><td>0.917</td></tr></table>

![](images/e7a1ee1285491db93947205a6265c6271f498d480c5f16a462dc2715b2cc8631.jpg)  
(a) F1-Score

![](images/8387ed3d0c3c6ed7167dfb99bc7c489ca1226441981e69b6c798be989cf14e2f.jpg)  
(b) G-Mean

![](images/ebb80de4cc4592f6ca34ff3a75c7c87c453fb2fd9adbf472e318f04a723396bd.jpg)  
(c) AUC  
Fig. 7. Sensitivity analysis of the trade-off coefficient α in the Stage-II fitness function.

As shown in Table VII, FreMGP achieves the best average results among all its variants across the three classifiers and all three evaluation metrics. When the proposed genetic operator design is removed, performance drops significantly, particularly under the LSTM classifier. This suggests that the stagespecific tree-level and group-level operators play an important role in effectively guiding the evolutionary search. Removing the prototype-guided loss also reduces the performance, suggesting that the prototype constraint contributes to learning a more useful representation space for fitness evaluation.

The results of “w/o Stage I” and “w/o Stage II” show that both search stages are useful. Without Stage I, the performance drops noticeably, especially in terms of F1-Score and G-Mean. Without Stage II, the performance is also lower than that of the full FreMGP, indicating that local distribution expansion further improves the quality of generated samples once representation proximity has been attained. In addition, removing either the angular diversity term or the radial consistency term leads to lower average performance compared with FreMGP. This suggests that considering both similarity and diversity in the fitness function of Stage II is beneficial for generating high-quality minority-class samples.

## E. Parameter Sensitivity Analysis

To analyze the influence of the parameter α in the fitness function of Stage II, we vary α from 0.1 to 0.9 and conduct experiments under LSTM, Transformer, and Mamba classifiers. A larger α gives more importance to radial consistency, which encourages synthetic samples to stay close to the target minority samples. A smaller α gives more importance to angular diversity, which encourages synthetic samples to expand in different directions around the target samples.

Fig. 7 reports the averaged results over six representative datasets to show the overall trend of FreMGP under different values of α. Overall, FreMGP maintains stable performance when α varies from 0.1 to 0.9. This indicates that the parameter α is less sensitive. The detailed results on the six datasets are provided in Appendix D.6.

## VI. CONCLUSIONS

To generate high-quality and diverse minority-class timeseries samples to mitigate the issue of class imbalance and enhance classifier performance for imbalanced time-series classification, we have proposed FreMGP, frequency-domain time-series samples for oversampling. where a multi-tree GP structure has been carefully designed to represent frequencydomain time-series samples for oversampling. Furthermore, a frequency-domain contrastive representation learning module was designed, which provides a reliable representation space for evaluating synthetic samples. Based on this, a two-stage fitness evaluation strategy has been designed. The first stage drives generated samples toward the minority-class region in the learned representation space, while the second stage further encourages them to move in diverse directions by considering both similarity and diversity. This strategy effectively guides the evolutionary search toward samples that are close to minority-class samples in the learned representation space while ensuring sufficient diversity among the generated samples.

Experiments on imbalanced time-series datasets have demonstrated that FreMGP improves the classification performance of various classifiers across different evaluation metrics, including F1-Score, G-Mean, and AUC. Nevertheless, FreMGP has a higher computational cost than traditional non-EC sampling methods, such as SMOTE, mainly due to the evaluation of individuals over multiple generations. In future work, we will explore more efficient evolutionary strategies to reduce the computational burden.

## REFERENCES

[1] N. Mohammadi Foumani, L. Miller, C. W. Tan, G. I. Webb, G. Forestier, and M. Salehi, “Deep learning for time series classification and extrinsic regression: A current survey,” ACM Computing Surveys, vol. 56, no. 9, Apr. 2024.

[2] A. Dempster, D. F. Schmidt, and G. I. Webb, “Hydra: Competing convolutional kernels for fast and accurate time series classification,” Data Mining and Knowledge Discovery, vol. 37, no. 5, pp. 1779–1805, 2023. [Online]. Available: https://doi.org/10.1007/s10618-023-00939-3

[3] S. Liu, C. Wei, X. Zhou, and H. Chen, “Spectral-aware reservoir computing for fast and accurate time series classification,” in Proceedings of the 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 267. PMLR, 13–19 Jul 2025, pp. 39 774–39 788. [Online]. Available: https://proceedings.mlr.press/v267/liu25by.html

[4] X. Wang, Y. Zhang, N. Bai, Q. Yu, and Q. Wang, “Class-imbalanced time series anomaly detection method based on cost-sensitive hybrid network,” Expert Systems with Applications, vol. 238, p. 122192, 2024.

[5] J. Kwak and J. Jung, “Classification of imbalanced ECGs through segmentation models and augmented by conditional diffusion model,” PeerJ Computer Science, vol. 10, p. e2299, 2024.

[6] X. Li, W. Li, X. Yu, Z. Han, and Q. Jin, “Financial risk assessment of imbalanced data based on nonlinear causal time-series network,” Information Processing & Management, vol. 62, no. 3, p. 104025, 2025.

[7] L. Wang, S. Huang, C. Zheng, J. Liao, X. Zhu, H. Li, and L. Liu, “Mitigating data imbalance in time series classification based on counterfactual minority samples augmentation,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V2, 2025, pp. 2962–2973.

[8] W. Chen, K. Yang, Z. Yu, Y. Shi, and C. L. P. Chen, “A survey on imbalanced learning: latest research, applications and future directions,” Artificial Intelligence Review, vol. 57, no. 137, 2024.

[9] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic minority over-sampling technique,” Journal of Artificial Intelligence Research, vol. 16, pp. 321–357, 2002.

[10] A. Fernandez, S. Garc´ ´ıa, F. Herrera, and N. V. Chawla, “SMOTE for learning from imbalanced data: Progress and challenges, marking the 15-year anniversary,” Journal ofArtificial Intelligence Research, vol. 61, pp. 863–905, 2018.

[11] P. Zhao, C. Luo, B. Qiao, L. Wang, S. Rajmohan, Q. Lin, and D. Zhang, “T-SMOTE: Temporal-oriented synthetic minority oversampling technique for imbalanced time series classification,” in Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, 2022, pp. 2406–2412.

[12] G. Iglesias, E. Talavera, A. Gonz<sup>´</sup> alez-Prieto, A. Mozo, and S. G´ omez-´ Canaval, “Data augmentation techniques in time series domain: A survey and taxonomy,” Neural Computing and Applications, vol. 35, no. 14, pp. 10 123–10 145, 2023.

[13] Q. Wen, L. Sun, F. Yang, X. Song, J. Gao, X. Wang, and H. Xu, “Time series data augmentation for deep learning: A survey,” in Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, 2021, pp. 4653–4660.

[14] J. Jeon, J. Kim, H. Song, S. Cho, and N. Park, “GT-GAN: General purpose time series synthesis with generative adversarial networks,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 36 999–37 010.

[15] A. Desai, C. Freeman, Z. Wang, and I. Beaver, “TimeVAE: A variational auto-encoder for multivariate time series generation,” arXiv preprint arXiv:2111.08095, 2021.

[16] X. Yuan and Y. Qiao, “Diffusion-TS: Interpretable diffusion for general time series generation,” in International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/ forum?id=4h1apFjO99

[17] T. Gonen, I. Pemper, I. Naiman, N. Berman, and O. Azencot, “Time series generation under data scarcity: A unified generative modeling approach,” in Advances in Neural Information Processing Systems, vol. 38, 2025, pp. 172 751–172 790.

[18] M. Hayaeian Shirvan, M. H. Moattar, and M. Hosseinzadeh, “Deep generative approaches for oversampling in imbalanced data classification problems: A comprehensive review and comparative analysis,” Applied Soft Computing, vol. 170, p. 112677, 2025.

[19] W. Pei, Y. Cui, B. Xue, M. Zhang, J. Zhang, Y. Hou, G. Zou, and Z. Qiang, “DG-SMOTE: A distance-angle-based genetic synthetic minority over-sampling technique for unbalanced data learning,” IEEE Transactions on Evolutionary Computation, vol. 29, no. 6, pp. 2641 – 2655, 2025.

[20] W. Pei, R. Dai, B. Xue, M. Zhang, Q. Zhang, and Y.-M. Cheung, “Evo-TFS: Evolutionary time-frequency domain-based synthetic minority oversampling approach to imbalanced time series classification,” IEEE Transactions on Evolutionary Computation, pp. 1–1, 2026.

[21] Y. Bengio, A. Courville, and P. Vincent, “Representation learning: A review and new perspectives,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 35, no. 8, pp. 1798–1828, 2013.

[22] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in Proceedings of the 37th International Conference on Machine Learning, vol. 119. PMLR, 2020, pp. 1597–1607.

[23] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for unsupervised visual representation learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 9729–9738.

[24] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, and D. Krishnan, “Supervised contrastive learning,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 18 661–18 673.

[25] J. Dong, H. Wu, Y. Wang, Y.-Z. Qiu, L. Zhang, J. Wang, and M. Long, “TimeSiam: A pre-training framework for siamese time-series modeling,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235. PMLR, 21–27 Jul 2024, pp. 11 412–11 436.

[26] G. Woo, C. Liu, D. Sahoo, A. Kumar, and S. Hoi, “CoST: Contrastive learning of disentangled seasonal-trend representations for time series forecasting,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/ forum?id=PilZY3omXV2

[27] A. V. Oppenheim and R. W. Schafer, Discrete-Time Signal Processing, 3rd ed. Pearson, 2010.

[28] A. Lensen, B. Xue, and M. Zhang, “Generating redundant features with unsupervised multi-tree genetic programming,” in Genetic Programming, ser. Lecture Notes in Computer Science, vol. 10781. Springer, 2018, pp. 84–100.

[29] D. J. Montana, “Strongly typed genetic programming,” Evolutionary Computation, vol. 3, no. 2, pp. 199–230, 1995.

[30] F. A. Del Campo, M. C. G. Neri, O. O. V. Villegas, V. G. C. Sanchez,´ H. d. J. O. Dom´ınguez, and V. G. Jimenez, “Auto-adaptive multilayer´ perceptron for univariate time series classification,” Expert Systems with Applications, vol. 181, p. 115147, 2021.

[31] B. Goehry, H. Yan, Y. Goude, P. Massart, and J.-M. Poggi, “Random forests for time series,” 2021, working paper, HAL. [Online]. Available: https://hal.science/hal-03129751

[32] T. M. Tran, X.-M. T. Le, H. T. Nguyen, and V.-N. Huynh, “A novel non-parametric method for time series classification based on k-nearest neighbors and dynamic time warping barycenter averaging,” Engineering Applications of Artificial Intelligence, vol. 78, pp. 173–185, 2019.

[33] H. Ismail Fawaz, G. Forestier, J. Weber, L. Idoumghar, and P.-A. Muller, “Deep learning for time series classification: a review,” Data Mining and Knowledge Discovery, vol. 33, no. 4, pp. 917–963, 2019.

[34] F. Markovic, L. Jovanovic, P. Spalevic, J. Kaljevic, M. Zivkovic, V. Simic, H. Shaker, and N. Bacanin, “Parkinsons detection from gait

time series classification using modified metaheuristic optimized long short term memory,” Neural Processing Letters, vol. 57, no. 1, p. 14, 2025.

[35] X. Zhang, H. Peng, J. Zhang, and Y. Wang, “A cost-sensitive attention temporal convolutional network based on adaptive top-k differential evolution for imbalanced time-series classification,” Expert Systems with Applications, vol. 213, p. 119073, 2023.

[36] A. Katrompas, T. Ntakouris, and V. Metsis, “Recurrence and selfattention vs the Transformer for time-series classification: A comparative study,” in International Conference on Artificial Intelligence in Medicine. Springer, 2022, pp. 99–109.

[37] C. Elkan, “The foundations of cost-sensitive learning,” in Proceedings of the Seventeenth International Joint Conference on Artificial Intelligence, 2001, pp. 973–978.

[38] G. E. A. P. A. Batista, R. C. Prati, and M. C. Monard, “A study of the behavior of several methods for balancing machine learning training data,” ACM SIGKDD Explorations Newsletter, vol. 6, no. 1, pp. 20–29, 2004.

[39] H. Han, W.-Y. Wang, and B.-H. Mao, “Borderline-SMOTE: A new over-sampling method in imbalanced data sets learning,” in Advances in Intelligent Computing, ser. Lecture Notes in Computer Science, vol. 3644. Springer, 2005, pp. 878–887.

[40] H. He, Y. Bai, E. A. Garcia, and S. Li, “ADASYN: Adaptive synthetic sampling approach for imbalanced learning,” in Proceedings of the IEEE International Joint Conference on Neural Networks. IEEE, 2008, pp. 1322–1328.

[41] J. Ren, Y. Wang, Y. ming Cheung, X.-Z. Gao, and X. Guo, “Groupingbased oversampling in kernel space for imbalanced data classification,” Pattern Recognition, vol. 133, p. 108992, 2023. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0031320322004721

[42] H. K. Lee, J. Lee, and S. B. Kim, “Boundary-focused generative adversarial networks for imbalanced and multimodal time series,” IEEE Transactions on Knowledge and Data Engineering, vol. 34, no. 9, pp. 4102–4118, 2022.

[43] J. W. Cooley and J. W. Tukey, “An algorithm for the machine calculation of complex Fourier series,” Mathematics of Computation, vol. 19, no. 90, pp. 297–301, 1965.

[44] Y. Yang, H. Yuan, X. Li, Z. Lin, P. H. S. Torr, and D. Tao, “Neural collapse inspired feature-classifier alignment for few-shot class-incremental learning,” in International Conference on Learning Representations, 2023.

[45] D. Mildenberger, P. Hager, D. Rueckert, and M. J. Menten, “A tale of two classes: Adapting supervised contrastive learning to binary imbalanced datasets,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 10 305–10 314.

[46] K. Yi, Q. Zhang, W. Fan, S. Wang, P. Wang, H. He, N. An, D. Lian, L. Cao, and Z. Niu, “Frequency-domain MLPs are more effective learners in time series forecasting,” in Thirty-seventh Conference on Neural Information Processing Systems, vol. 36, 2023, pp. 76 656– 76 679.

[47] H. M. Nguyen, E. W. Cooper, and K. Kamei, “Borderline over-sampling for imbalanced data classification,” International Journal of Knowledge Engineering and Soft Data Paradigms, vol. 3, no. 1, pp. 4–21, 2011.

[48] G. Douzas, F. Bacao, and F. Last, “Improving imbalanced learning through a heuristic oversampling method based on k-means and SMOTE,” Information Sciences, vol. 465, pp. 1–20, 2018.

[49] H. Cao, X.-L. Li, D. Y.-K. Woon, and S.-K. Ng, “Integrated oversampling for imbalanced time series classification,” IEEE Transactions on Knowledge and Data Engineering, vol. 25, no. 12, pp. 2809–2822, 2013.

[50] T. Zhu, C. Luo, J. Li, S. Ren, and Z. Zhang, “Minority oversampling for imbalanced time series classification,” Knowledge-Based Systems, vol. 247, p. 108764, 2022.

[51] P. Liu, X. Guo, R. Wang, P. Chen, T. Wo, and X. Liu, “CSMOTE: Contrastive synthetic minority oversampling for imbalanced time series classification,” in Neural Information Processing, ser. Lecture Notes in Computer Science, vol. 13110. Springer, 2021, pp. 447–455.

[52] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.

[53] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 5998–6008.

[54] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.

[55] F.-A. Fortin, F.-M. De Rainville, M.-A. Gardner, M. Parizeau, and C. Gagne, “DEAP: Evolutionary algorithms made easy,” Journal of Machine Learning Research, vol. 13, no. 70, pp. 2171–2175, 2012.

[56] G. Lemaˆıtre, F. Nogueira, and C. K. Aridas, “Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning,” Journal of Machine Learning Research, vol. 18, no. 17, pp. 1–5, 2017.

[57] K. M. Sujon, R. Hassan, K. Choi, and M. A. Samad, “Accuracy, precision, recall, F1-score, or MCC? empirical evidence from advanced statistics, ML, and XAI for evaluating business predictive models,” Journal of Big Data, vol. 12, p. 268, 2025.

[58] E. Richardson, G. Treger, A. E. Brøndsted, A. D. Haue, M. Tulstrup, I. Bozic, K. Grønbæk, M. Sørensen, H. J. Ditzel, T. A. Kruse, P. Guldberg, D. A. Bell, V. N. Kristensen, L. Dyrskjøt, T. Reinert, K. Mouridsen, F. C. Nielsen, C. L. Andersen, O. Winther, L. J. Jensen, L. Dyrskjøt et al., “The receiver operating characteristic curve accurately assesses imbalanced datasets,” Patterns, vol. 5, no. 7, p. 100994, 2024.

[59] J. Demsar, “Statistical comparisons of classifiers over multiple data sets,”ˇ Journal of Machine Learning Research, vol. 7, pp. 1–30, 2006.

[60] S. Holm, “A simple sequentially rejective multiple test procedure,” Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65–70, 1979.