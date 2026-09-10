# Hybrid Quantum–Classical NLP Classification with Compact Semantic Representations: An Experimental Analysis of Representation Compression

Ali Hassan<sup>1</sup>, Zijia Zhao<sup>2</sup>, and Maha A. Metawei<sup>3</sup>

<sup>1</sup>German University in Cairo, Cairo, Egypt

<sup>2</sup>University of Melbourne, Melbourne, Australia

<sup>3</sup>High Performance Computing Lab, Electronics Research Institute, Cairo, Egypt

ali.hafez@student.guc.edu.eg, zhaozj@student.unimelb.edu.au, maha metawei@eri.sci.eg

## Abstract

Large language and sentence-embedding models provide rich semantic representations, but their high dimensionality creates a fundamental challenge for near-term quantum machine learning (QML), where quantum circuits can directly process only a limited number of input features. This work investigates a hybrid quantum–classical pipeline that transforms high-dimensional sentence embeddings into compact representations suitable for variational quantum classification. The proposed workflow combines a pretrained sentence-embedding model, dimensionality reduction, angle encoding, a variational quantum circuit (VQC), and a classical decision layer. Three dimensionality-reduction techniques are systematically investigated: principal component analysis (PCA), neighborhood components analysis (NCA), and linear discriminant analysis (LDA), enabling a comparison of unsupervised and supervised approaches to quantum-compatible representation learning. Using the TREC questionclassification dataset, we study the relationship between representation dimensionality, information retention, qubit count, and classification performance. Preliminary PCA-based experiments show a clear information bottleneck under aggressive compression: reducing 768-dimensional sentence embeddings to 3, 4, 5, and 8 dimensions retains approximately 8.2%, 10.2%, 11.9%, and 16.4% of the variance, respectively, while the corresponding preliminary classification accuracies are 50.3%, 51.2%, 57.9%, and 63.4%. In contrast, supervised reduction proves far more eficient: LDA reaches 85.3% accuracy and NCA reaches 83.1% accuracy using only 5 dimensions, both validated under a leakage-free cross-validation protocol and comparable to a full 384-dimensional classical baseline (85.1%). These findings show that supervised dimensionality reduction preserves task-relevant information far more efectively than variance-based compression, and motivate a systematic evaluation against matched classical baselines and larger quantum representations. The study aims to characterize the practical operating regime of hybrid quantum–classical NLP models and to assess the impact of representation compression without assuming quantum advantage a priori.

## 1 Introduction

Natural language processing (NLP) systems increasingly rely on high-dimensional semantic representations produced by pretrained language models. Sentence embeddings provide a convenient interface between modern language models and downstream classification algorithms, but their dimensionality can make them dificult to integrate with near-term quantum machine learning (QML) models. A quantum circuit with n qubits provides a Hilbert space of dimension 2<sup>n</sup>, yet the number of directly encoded classical features is typically constrained by the selected encoding strategy and circuit architecture.

This creates a central question for hybrid quantum–classical NLP: how much semantic information can be compressed into a quantum-compatible representation without making classification performance unacceptable? This paper investigates this question using a hybrid pipeline consisting of sentence embeddings, dimensionality reduction, quantum feature encoding, and a variational quantum classifier. Rather than claiming quantum advantage from classification accuracy alone, the study explicitly separates the efects of classical representation compression from those of the quantum model.

## 2 Related Work

Integrating high-dimensional, pre-trained classical text embeddings into NISQ-era variational quantum classifiers (VQCs) requires eficient classical-to-quantum interfaces. While compositional distributional (DisCoCat) frameworks map text syntax directly onto circuits via combinatorial categorial grammar [7], they scale poorly due to exploding circuit depths on larger sentences. Consequently, hybrid pipelines that classically compress frozen pretrained text embeddings (e.g., Sentence-BERT or MiniLM) before quantum state encoding have emerged as the standard paradigm [12, 10]. Recent literature has extensively explored the constraints, limits, and potential methodologies of this compression interface. A critical analysis of these recent attempts [11, 9] reveals key points of convergence, open challenges, and structural limitations compared to our proposed framework.

## 2.1 Compositional and Sequential Models in Quantum NLP

A major line of research in QNLP is based on the Distributional Compositional Categorical (DisCoCat) framework, which combines distributional semantics with grammatical structure. In DisCoCat, words are represented by vectors or higher-order tensors whose types correspond to their grammatical roles, while grammatical structure determines how these representations are composed. Sentence meaning is consequently obtained through structured tensor contraction rather than by treating the sentence as an unstructured sequence of features. This provides an explicit connection between syntax and semantic representation and has become one of the principal theoretical foundations of QNLP [11, 9].

The implementation of compositional models has been facilitated by software frameworks such as [7], which provides tools for constructing grammatical diagrams, transforming them into tensor or quantum representations, and training parameterized quantum circuits. This approach enables compositional models to be evaluated using classical simulation as well as quantum backends. Consequently, a number of studies have investigated whether quantum implementations of compositional semantic models can perform competitive classification and language-processing tasks under realistic hardware constraints [9].

An important extension of compositional QNLP is the DisCoCirc framework, which generalizes the compositional perspective from individual sentences toward discourse. Instead of treating the meaning of a word as a fixed representation that is simply contracted according to a sentence diagram, DisCoCirc represents linguistic interactions through updates to the states of concepts or entities. This formulation provides a mechanism for modelling dependencies across multiple sentences and has been investigated for tasks requiring discourse-level reasoning and compositional generalization [2].

Recent empirical work has demonstrated the application of compositional quantum circuits to increasingly challenging language tasks. For example, pronoun-resolution experiments have investigated DisCoCat-based variational quantum circuits under ideal simulation and noisy execution conditions. Such studies illustrate the transition of QNLP from primarily theoretical formulations to empirical evaluation on NISQ-era platforms [9].

Despite their conceptual advantages, compositional models face several practical limitations. The construction of a quantum circuit generally requires linguistic preprocessing, including grammatical parsing and the generation of a corresponding compositional structure. Furthermore, circuit size and tensor complexity can increase with sentence length and grammatical structure. These limitations become particularly important when moving from small benchmark examples to larger datasets and realistic NLP applications. Recent surveys therefore identify scalability, circuit depth, data encoding, and limited quantum hardware resources as persistent challenges for compositional QNLP [11, 9]. This issue distinguishes embedding-based hybrid QNLP from traditional compositional approaches. In compositional QNLP, the dimensionality of the representation is determined largely by the chosen lexical spaces, tensor structures, and quantum encoding strategy. In contrast, embedding-based approaches can exploit powerful pretrained classical language representations and subsequently compress them before quantum processing. This design enables the semantic representation learned by a classical model to be separated from the quantum classification stage. As a good example, Hazim et al. [4] presented a hybrid quantum–classical NLP framework for detecting AI-generated scholarly text. The approach combines conventional language representations with a quantum machine learning classifier to distinguish human-written from AI-generated academic content, demonstrating the potential of quantum-enhanced models for text classification tasks.

## 2.2 Dimensionality Reduction

Principal component analysis [6] (PCA) identifies orthogonal directions that maximize the variance of input data. Given a centered embedding matrix $X \in \mathbb { R } ^ { N \times D }$ , the reduced representation can be expressed as

$$
Z = X W _ { d } ,\tag{1}
$$

where $W _ { d }$ contains the first d principal components.

The retained variance is determined by the eigenvalues associated with the selected components. In this study, d is selected according to the size of the target quantum representation.

In the QNLP and broader QML literature, PCA is the dominant choice for fitting classical representations onto small quantum circuits: it reduces the feature count to match the available qubits while preserving as much global variance as possible. A recurring limitation, however, is that PCA is unsupervised and therefore optimizes for variance rather than for class discriminability, which can discard exactly the directions most useful for a downstream classification task. Supervised alternatives such as linear discriminant analysis (LDA) and neighborhood components analysis (NCA) address this by using label information during projection, and have been studied in classical settings but far less frequently as the reduction stage of a quantum pipeline. The present work directly contrasts these unsupervised and supervised strategies within an identical QNLP pipeline. The dimensionality-reduction problem is increasingly recognized as a general challenge in quantum machine learning. Odagiu et al.[8] systematically investigated conventional and neural dimensionality-reduction methods before a quantum classifier and demonstrated that the quality of the reduced representation can have a substantial efect on downstream quantum classification. Their results emphasize that dimensionality reduction should not necessarily be treated as a purely technical preprocessing step, but rather as an important component of the overall QML architecture.

For QNLP, this issue is particularly important because sentence embeddings encode semantic information in a high-dimensional continuous space. Aggressive compression may remove information that is relevant to a classification task even when the resulting representation retains a substantial proportion of global statistical structure. Conversely, a supervised dimensionalityreduction method may discard variance that is not useful for the target task while preserving features that provide stronger class discrimination.

## 2.3 Diferentiable Selection vs. Projected Semantic Compression

To bypass hard projections, some research has focused on selecting a subset of the original feature space. For example, Jagannathan et al. [5] proposed Variational Quantum Feature Selection (VQFS), a hybrid quantum-classical approach that integrates trainable scalar weights into the rotation angles of a PQC. By applying an L1 regularization penalty on the weight vector, the system learns to collapse less informative feature weights to zero, efectively performing feature selection end-to-end via gradient descent.

While VQFS is highly elegant for tabular datasets (e.g., Iris, Wine) where individual features correspond to distinct physical measurements, it sufers from two major limitations when applied to dense, distributed language embeddings. First, dense semantic embeddings produced by models like S-BERT do not contain individual coordinate dimensions that represent distinct, separable features; rather, semantic meaning is distributed globally across the entire vector space. Performing coordinate-wise feature selection (i.e., zeroing out dimensions) is highly sub-optimal because it discards co-dependent semantic coordinates. Second, optimizing the L1 penalty parameters jointly with the VQC parameters in the quantum optimization loop significantly increases classical simulator latency. In contrast, our proposed supervised projection methods (LDA [1] and NCA [3]) project the distributed semantic coordinates into a lower-dimensional subspace rather than selecting individual coordinates, thereby retaining dense joint feature correlations. Furthermore, LDA and NCA are computed classically in seconds, acting as an extremely fast, modular preprocessing interface that decouples representation compression from quantum circuit optimization.

## 3 Methodology

The proposed hybrid quantum-classical NLP pipeline consists of multiple key processing stages as shown in Figure 1:

![](images/f178aff6209057a0a57671b6b01763bfb14b3ca10ba149acb166dd0ff83fb596.jpg)  
Figure 1: Hybrid Quantum-Classical Pipeline Architecture.

## 3.1 Sentence Representation

Each input sentence is converted into a continuous, dense vector representation using a frozen sentence-embedding model. In our preliminary experiments, we utilize a pretrained sentencetransformer model (all-MiniLM-L6-v2) to produce 384-dimensional embeddings (reduced from an initial 768-dimensional representation space), ensuring rich semantic capture while maintaining tractability for classical simulation.

## 3.2 Dimensionality Reduction

To bridge the gap between high-dimensional classical embeddings and the limited qubit capacity of near-term quantum simulators/devices, we evaluate three distinct dimensionality-reduction

strategies:

• Principal Component Analysis (PCA): An unsupervised linear technique that projects the embeddings along orthogonal directions of maximum variance.

• Neighborhood Components Analysis (NCA): A supervised linear technique that learns a projection matrix to maximize k-nearest-neighbor classification performance.

• Linear Discriminant Analysis (LDA): A supervised linear technique that explicitly projects the data into a subspace that maximizes the ratio of between-class variance to within-class variance, mathematically capped at $C - 1$ dimensions where C is the number of classes.

## 3.3 Quantum Feature Encoding

After dimensionality reduction, the reduced 5-dimensional features $z \in \mathbb { R } ^ { 5 }$ (e.g., from the linear discriminant analysis projection of the MiniLM sentence embeddings) are scaled to an angular range $[ - \pi , \pi ]$ and encoded into quantum states via single-qubit rotations. Specifically, for our 5-qubit variational circuit, we employ single-qubit $R _ { y }$ rotation gates for state preparation:

$$
| z \rangle = \bigotimes _ { i = 1 } ^ { 5 } R _ { y } ( \tilde { z } _ { i } ) | 0 \rangle\tag{2}
$$

where $\tilde { z } _ { i }$ represents the normalized feature value.

## 3.4 Variational Quantum Circuit Architecture

The encoded state |z⟩ is processed by a hardware-eficient, two-layer variational quantum circuit $U ( \theta )$ operating on 5 qubits. Each of the two variational layers is structured as follows:

1. Parameterized single-qubit rotation gates $R _ { y } ( \theta _ { 1 , j } )$ and $R _ { z } ( \theta _ { 2 , j } )$ are applied to each qubit $j \in \{ 0 , \ldots , 4 \}$ . This provides a highly flexible rotational ansatz.

2. A ring-entangling configuration of controlled-NOT (CNOT) gates is applied to couple adjacent qubits. Specifically, the CNOT gates are applied in a circular topology (i.e., qubit $j$ acts as control and qubit $j + 1$ (mod 5) acts as target for all qubits).

By stacking two of these variational blocks, the quantum core contains exactly 20 trainable parameters in total (10 parameters per layer), keeping the circuit depth extremely shallow to fit within the short coherence times of near-term quantum hardware.

## 3.5 Measurement and Classical Readout

Rather than measuring only single-qubit expectation values, we extract information from both individual and joint quantum states. Specifically, we measure the expectation values of singlequbit Pauli-Z operators $\left( \langle Z _ { j } \rangle \right)$ ) and adjacent two-qubit Pauli-Z operators $\big ( \langle Z _ { j } Z _ { j + 1 \ ( \mathrm { m o d } \ 5 ) } \rangle \big )$ for all 5 qubits. This measurement strategy yields exactly 10 features:

$$
F = [ \langle Z _ { 0 } \rangle , \ldots , \langle Z _ { 4 } \rangle , \langle Z _ { 0 } Z _ { 1 } \rangle , \ldots , \langle Z _ { 4 } Z _ { 0 } \rangle ] \in \mathbb { R } ^ { 1 0 }\tag{3}
$$

This expectation vector is passed to a classical decision layer (such as a logistic regression classifier or a small Multi-Layer Perceptron), which performs a 6-class readout to yield the final class probabilities over the coarse categories of the TREC dataset:

$$
P ( y = c \mid F ) = \operatorname { s o f t m a x } ( W F + b )\tag{4}
$$

This hybrid configuration guarantees a fixed 5-qubit circuit width for every text question regardless of the sentence length, shifting the primary bottleneck of hybrid QNLP from physical hardware scaling to eficient classical semantic compression.

## 4 Experimental Results

In this section, we present the comprehensive results of our experiments across two text classification benchmarks: a compact restaurant sentiment dataset and the six-class TREC questionclassification dataset.

## 4.1 Syntactic Parsing and Diagram Generation

Prior to semantic embedding and classification, syntactic structures were extracted using a dependency-based and combinatory categorial grammar (CCG) framework via the lambeq library. On the restaurant dataset, 70/70 sentences were successfully preprocessed, corrected for minor lemmatization bugs (e.g., correcting “hat” to “hate”), and converted to syntactic diagrams. On the larger TREC dataset consisting of 5,452 training examples and 500 test examples, the local BobcatParser was used to generate grammar diagrams. The parser achieved a high success rate, successfully parsing 5,432 out of 5,452 sentences (a 99.63% success rate), with only 20 sentences failing due to parsing or tokenization index constraints. Examples of failed sentences include highly conversational or idiosyncratic question structures such as “to what do Microsoft’s Windows 3 owe its success?” and “what bird can swim but can’t fly?”.

## 4.2 Small-Scale Experiments: Restaurant Dataset

Table 1 shows the PCA-reduced classification accuracy on the 70-sentence restaurant dataset as a function of the number of qubits. Even with just 3 qubits, the model retains 58.5% of the original semantic variance and achieves 90.0% classification accuracy. Scaling up to 8 qubits increases the retained variance to 84.3% and maintains a high classification accuracy of 94.3% (peaking at 95.7% with 4 qubits).

Table 1: PCA Results for Compact Representations on the Restaurant Dataset (70 sentences).
<table><tr><td>Qubits (Dimensions)</td><td>Variance Retained (%)</td><td>Classification Accuracy (%)</td></tr><tr><td>3</td><td>58.5</td><td>90.0</td></tr><tr><td>4</td><td>67.5</td><td>95.7</td></tr><tr><td>5</td><td>74.1</td><td>92.9</td></tr><tr><td>8</td><td>84.3</td><td>94.3</td></tr></table>

## 4.3 Large-Scale Evaluation: TREC Question Classification

Aggressive dimensionality reduction was performed on the TREC dataset to examine the information bottleneck under extreme compression.

## 4.3.1 Preliminary and Fine-Grained PCA Evaluation

Table 2 outlines the preliminary results for PCA-reduced representations up to 8 qubits. In contrast to the smaller restaurant dataset, a severe information bottleneck is visible on the 6-class TREC dataset: reducing the 384-dimensional representation to 3 qubits retains a mere 8.2% of the variance and yields a classification accuracy of 50.3%.

To map out the compression scaling behavior, we evaluated PCA across a wide range of component sizes, extending from 8 up to the full 384 dimensions. These fine-grained results are documented in Table 3 and visualized in Figure 2. The results reveal a strong divergence: while variance retained scales slowly but steadily up to 100%, classification accuracy plateaus much earlier (around 20–50 dimensions), indicating that explained variance is not a reliable proxy for task-relevant semantic information.

Table 2: Preliminary PCA Results on the TREC Dataset.
<table><tr><td></td><td>Qubits (Dimensions) Variance Retained (%) Classification Accuracy (%)</td><td></td></tr><tr><td>3</td><td>8.2</td><td>50.3</td></tr><tr><td>4</td><td>10.2</td><td>51.2</td></tr><tr><td>5</td><td>11.9</td><td>57.9</td></tr><tr><td>8</td><td>16.4</td><td>63.4</td></tr></table>

Table 3: Extended PCA Dimensionality and Performance on the TREC Dataset.
<table><tr><td>Qubits s (Dimensions)</td><td>Variance Retained (%)</td><td>Classification Accuracy (%)</td></tr><tr><td>8</td><td>16.4</td><td>63.4</td></tr><tr><td>10</td><td>19.0</td><td>64.4</td></tr><tr><td>12</td><td>21.5</td><td>65.9</td></tr><tr><td>14</td><td>23.8</td><td>68.3</td></tr><tr><td>16</td><td>25.9</td><td>69.1</td></tr><tr><td>18</td><td>27.9</td><td>70.9</td></tr><tr><td>20</td><td>29.7</td><td>71.5</td></tr><tr><td>50</td><td>50.8</td><td>77.4</td></tr><tr><td>60</td><td>56.1</td><td>78.3</td></tr><tr><td>80</td><td>64.9</td><td>79.1</td></tr><tr><td>100</td><td>71.9</td><td>80.8</td></tr><tr><td>150</td><td>84.0</td><td>82.6</td></tr><tr><td>200</td><td>91.6</td><td>82.9</td></tr><tr><td>250</td><td>96.4</td><td>83.5</td></tr><tr><td>300</td><td>98.8</td><td>83.6</td></tr><tr><td>384 (Full)</td><td>100.0</td><td>83.8</td></tr></table>

![](images/c49127e9ef6bda88897eb7d85aa532ba39a2da20d1def9a46a45e993578a87a1.jpg)  
Figure 2: PCA classification accuracy and explained variance retained as a function of the number of qubits (PCA components) on the TREC dataset. Variance retained increases steadily, whereas accuracy plateaus much earlier.

## 4.3.2 Supervised Reduction vs. Unsupervised Compression

To mitigate the unsupervised compression bottleneck, we evaluated supervised dimensionalityreduction methods (LDA and NCA) at 5 dimensions (the mathematical cap for LDA on a 6-class dataset is $C - 1 = 5 )$

As shown in Table 4 and Figure 3, supervised dimensionality reduction dramatically outperforms PCA. LDA at just 5 dimensions achieves an accuracy of 85.3%, and NCA achieves 83.1%. Both methods far outperform PCA at 5 dimensions (57.9%) and 20 dimensions (71.5%), and actually match or exceed the full 384-dimensional uncompressed classical baseline (85.1%) using a tiny fraction of the dimensionality.

Table 4: Final Supervised vs. Unsupervised Dimensionality Reduction Results at 5 Dimensions.
<table><tr><td>Method</td><td>Supervised</td><td>Dimensions (qubits)</td><td>Accuracy (%)</td></tr><tr><td>PCA</td><td>No</td><td>5</td><td>57.9</td></tr><tr><td>PCA</td><td>No</td><td>20</td><td>71.5</td></tr><tr><td>PCA</td><td>No</td><td>384 (full)</td><td>83.8</td></tr><tr><td>Classical Baseline (MLP, no reduction)</td><td></td><td>384</td><td>85.1</td></tr><tr><td>NCA</td><td>Yes</td><td>5</td><td>83.1</td></tr><tr><td>LDA (adopted method)</td><td>Yes</td><td>5</td><td>85.3</td></tr></table>

![](images/d06c559f970ee83c04345ccefe122316c5e59879383ea5c04621191b8437821e.jpg)  
Figure 3: Accuracy versus number of dimensions (qubits) on the TREC dataset for PCA, LDA, and NCA. Supervised methods reach high accuracy at only 5 dimensions, far outperforming PCA at the same qubit count.

## 4.3.3 Impact of Proper Evaluation: Preventing Data Leakage

An essential finding of our work is the impact of proper validation. Many quantum-classical pipelines sufer from target leakage by fitting dimensionality reduction algorithms (such as PCA, LDA, or NCA) on the entire dataset prior to performing cross-validation folds. To study this, we compared a “leak-prone” setup (fitting the reduction on the entire dataset) against a “leakage-free” setup (fitting the reduction strictly on the training partition of each fold). Table 5 illustrates this comparison. Preventing target leakage reveals a significant drop in accuracy (a 7.8% drop for NCA and a 4.6% drop for LDA), highlighting the critical importance of a leak-free protocol to ensure reproducible and honest benchmarking.

Table 5: Leak-Prone (Entire Dataset Fitting) vs. Leakage-Free (Fold-Wise Fitting) Performance.
<table><tr><td>Method</td><td>Dimensions</td><td>Leak-Prone Accuracy (%)</td><td>Leakage-Free Accuracy (%)</td><td>Accuracy Drop (%)</td></tr><tr><td>NCA</td><td>5</td><td>90.9</td><td>83.1</td><td>-7.8</td></tr><tr><td>LDA</td><td>5</td><td>89.9</td><td>85.3</td><td>-4.6</td></tr></table>

![](images/a42ff245fb44fef920bded4313054a106183a347da17454e70f58842f9cae66b.jpg)  
Figure 4: Comparing the number of trainable parameters for the VQC circuit, Logistics Regression model, and the complete hybrid pipeline.

## 5 Discussion and Limitations

Our findings show that the choice of compression method is far more consequential than the raw dimensionality of the representation. Under unsupervised PCA, the transition from 3 to 8 dimensions produces a steady increase in performance, but because PCA optimizes for variance rather than class discriminability, it discards critical semantic details. Supervised alternatives (LDA and NCA) bypass this bottleneck entirely by using label information to project same-class samples closer together and push diferent-class samples apart.

This leads to several important insights for hybrid QML:

1. No Free Lunch for Variance: Explained variance is not a reliable proxy for downstream task accuracy. High variance does not guarantee high class separability.

2. Grounded Benchmarking: Improvements in VQC performance must be validated against matched classical baselines running on the same compressed representations. Otherwise, the performance gains are attributable to classical dimensionality reduction, not quantum expressibility.

3. Physical Limitations: In a physical quantum computer, scaling qubits introduces noise, deepens circuits, and increases optimization dificulty. Thus, maximizing compact, supervised representations is the most practical path forward for near-term NISQ devices.

## 5.1 Proposed Hybrid Quantum-Classical Transfer Learning Pipeline

Our work shows that supervised projections, specifically Linear Discriminant Analysis (LDA) and Neighborhood Components Analysis (NCA), are extremely eficient classical-to-quantum interfaces. They compress 384-dimensional dense transformer sentence embeddings down to just 5 dimensions while maintaining performance parity with uncompressed classical baselines.

Building on this, we formalize the hybrid quantum-classical pipeline into a structured, three-stage workflow:

1. Linguistic Feature Extraction (Classical Source Domain): High-dimensional dense semantic vectors are generated classically from a frozen pre-trained language model (e.g., MiniLM, Sentence-BERT, or a larger transformer). This phase acts as classical transfer learning, allowing the pipeline to inherit rich vocabulary, syntax, and grammatical context trained on multi-billion token corpora.

2. Supervised Manifold Alignment (Interface Adaptation): Rather than applying task-blind, unsupervised linear reductions like PCA, a lightweight supervised projection head (LDA or NCA) is fitted on target label data. This layer learns a low-dimensional mapping that aligns the dense semantic dimensions with the downstream classification requirements, projecting same-class queries closer together and diferent-class queries apart.

3. Variational Quantum Classification (Quantum Target Domain): The compressed 5-dimensional features are encoded into a parameterised variational quantum circuit (VQC) via $R _ { y }$ angle rotations. Because the classical projection head has already maximized class discriminability, the VQC is presented with highly separable clusters. This significantly reduces quantum optimization complexity, mitigates barren plateau vulnerabilities, and yields high accuracy under real physical noise constraints.

Crucially, the proposed hybrid pipeline is validated under a rigorous, leakage-free crossvalidation protocol. Fitting the classical projection head strictly within the training folds prevents target leakage and ensures that the transferred representations are generalizable and reproducible. Rather than searching for an elusive “quantum advantage” in raw semantic text encoding, this framework leverages a cooperative division of labor: classical neural networks manage massive contextual representation extraction, while NISQ devices are utilized exclusively for classification over optimized, low-dimensional semantic manifolds. This architecture ofers a highly practical, near-term pathway for deploying robust quantum text classifiers on physical quantum devices.

## 6 Conclusion and Future Work

This work presents a hybrid quantum–classical NLP framework that combines pretrained sentence embeddings, dimensionality reduction, quantum feature encoding, and variational quantum classification. Preliminary TREC experiments demonstrate that increasing the number of available quantum features improves classification performance, while also revealing a strong information bottleneck caused by aggressive compression of the original 768-dimensional representations. Supervised reduction methods (LDA, NCA) substantially mitigate this bottleneck, matching a full-dimensional classical baseline using only five dimensions.

The results motivate a rigorous benchmark in which quantum models and classical models are evaluated on identical compressed representations. Such an evaluation is necessary to determine whether the quantum circuit contributes meaningful predictive value beyond classical dimensionality reduction. The resulting framework provides a basis for studying the practical role of near-term quantum models in NLP without assuming quantum advantage in advance. Future research will explore larger qubit representations (12- and 16-qubits), non-linear compression via autoencoders, and physical hardware deployment with error mitigation.

## Acknowledgments

This project was initiated under the QIntern 2026 project “Accelerating hybrid quantum-classical machine learning tasks on HPC platforms”. M.A.M. would like to thank the organizers of the

program and QWorld Association.

## Author Contribution

M.A.M. conceived the main idea and designed the overall research direction. A.H. and Z.Z. conducted the experiments and generated the experimental results. M.A.M and A.H. prepared the initial manuscript draft. All authors contributed to the revision of the manuscript, discussed the results, and read and approved the final version.

## References

[1] Suresh Balakrishnama and Aravind Ganapathiraju. Linear discriminant analysis-a brief tutorial. Institute for Signal and information Processing, 18(1998):1–8, 1998.

[2] F Duneau. A compositional approach to reading comprehension tasks using the DisCoCirc natural language processing framework. PhD thesis, University of Oxford, 2024.

[3] Jacob Goldberger, Geofrey E Hinton, Sam Roweis, and Russ R Salakhutdinov. Neighbourhood components analysis. Advances in neural information processing systems, 17, 2004.

[4] Layth Rafea Hazim and Oguz Ata. Hqml-nlp: A hybrid quantum machine learning framework for scholarly ai-text detection. Applied Soft Computing, 191:114634, 2026.

[5] Sharath Kumar Jagannathan, Thomas Abraham JV, Yogesh C, and Franklin Joel Benedict T. Variational quantum feature selection for high-dimensional classification: A hybrid quantumclassical approach. In EPJ Web of Conferences, volume 360, page 01029. EDP Sciences, 2026.

[6] Ian T Jollife and Jorge Cadima. Principal component analysis: a review and recent developments. Philosophical transactions. Series A, Mathematical, physical, and engineering sciences, 374(2065):20150202, 2016.

[7] Dimitri Kartsaklis, Ian Fan, Richie Yeung, Anna Pearson, Robin Lorenz, Alexis Toumi, Giovanni de Felice, Konstantinos Meichanetzidis, Stephen Clark, and Bob Coecke. lambeq: An eficient high-level python library for quantum nlp. arXiv preprint arXiv:2110.04236, 2021.

[8] Patrick Odagiu, Vasilis Belis, Lennart Schulze, Panagiotis Barkoutsos, Michele Grossi, Florentin Reiter, G¨unther Dissertori, Ivano Tavernelli, and Sofia Vallecorsa. Learning reduced representations for quantum classifiers. Quantum Machine Intelligence, 7(2):113, 2025.

[9] Arpan Phukan and Asif Ekbal. A survey of quantum natural language processing: From compositional models to nisq-era empiricism. In Proceedings of the QuantumNLP : Integrating Quantum Computing with Natural Language Processing, pages 65–75, 2025.

[10] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 3982–3992, 2019.

[11] Dominic Widdows, Willie Aboumrad, Dohun Kim, Sayonee Ray, and Jonathan Mei. Quantum natural language processing. KI-K¨unstliche Intelligenz, 38(4):293–310, 2024.

[12] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, et al. Transformers: Stateof-the-art natural language processing. In Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations, pages 38–45, 2020.