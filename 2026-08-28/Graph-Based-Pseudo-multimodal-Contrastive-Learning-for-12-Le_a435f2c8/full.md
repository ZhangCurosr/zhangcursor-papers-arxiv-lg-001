# Graph-Based Pseudo-multimodal Contrastive Learning for 12-Lead ECG Representations

1<sup>st</sup> Mengyu Wang Graduate School of Engineering Science Yokohama National University Yokohama, Japan

4<sup>th</sup> Natsuko Jinba Technology and Innovation Department Fukuda Denshi Co.,Ltd Tokyo, Japan

2<sup>nd</sup> Kozo Okada Division of Cardiology Yokohama City University Medical Center Yokohama, Japan

5<sup>th</sup> Hiroki Yamaya Technology and Innovation Department Fukuda Denshi Co.,Ltd Tokyo, Japan

3<sup>rd</sup> Takafumi Goto Technology and Innovation Department Fukuda Denshi Co.,Ltd Tokyo, Japan

6<sup>th</sup> Kiyoshi Hibi Division of Cardiology Yokohama City University Medical Center Yokohama, Japan

7<sup>th</sup> Tomoki Hamagami   
Faculty of Engineering   
Yokohama National University   
Yokohama, Japan

Abstract—12-lead electrocardiogram (ECG) is a standard, non-invasive examination widely used for diagnosing coronary artery disease, where clinical interpretation relies on comparing waveform patterns across multiple leads. However, most existing ECG analysis methods focus on single-lead signals or treat each lead independently, and typically process ECG signals as onedimensional time-series data using CNNs or RNNs. While effective in modeling local waveform changes, such approaches have difficulty capturing inter-lead dependency and global waveform patterns essential for clinical diagnosis.

To address this limitation, we propose a graph-based pseudomultimodal contrastive learning framework called Graph-CMMC. ECG waveforms are transformed into Gramian Angular Difference Field (GADF) images to construct complementary representations of the same cardiac activity, enabling a pseudomultimodal learning setting. Using all 12 leads, Graph-CMMC aligns waveform and GADF representations in a self-supervised manner, while a graph-based relational module is employed to model inter-lead dependency and enforce structural consistency across leads during contrastive learning.

Experimental results on a multi-label coronary artery occlusion classification task demonstrate that the proposed framework achieves competitive performance compared to supervised learning methods. These results further suggest the effectiveness of using GADF as a complementary representation and incorporating explicit graph-based modeling of inter-lead dependency for learning robust 12-lead ECG representations.

Index Terms—12-lead ECG, Contrastive learning, Pseudomultimodal representation, Graph-based modeling, Selfsupervised learning

## I. INTRODUCTION

12-lead electrocardiogram (ECG) is a standard and noninvasive examination that records cardiac electrical activity from 12 different viewpoints and plays an essential role in the diagnosis of coronary artery disease. By comparing waveform patterns across multiple leads, clinicians can infer the location and severity of cardiac abnormalities. As a result, accurate ECG interpretation relies not only on temporal waveform morphology within each lead, but also on the relationships among leads.

In recent years, AI-based ECG analysis methods have been actively studied to support clinical diagnosis. Most existing approaches process ECG signals as one-dimensional timeseries data using convolutional neural networks (CNNs) or recurrent neural networks (RNNs). While these methods are effective in capturing local temporal patterns, they often focus on single-lead signals or treat each lead independently.

Moreover, a representation gap exists between current AIbased ECG analysis methods and actual clinical diagnostic practice. Physicians do not interpret ECG signals solely as sequential numerical values. Instead, they visually inspect global waveform shapes, rhythm patterns, and their consistency across leads. Time-series-based processing, which emphasizes point-wise numerical processing, may therefore be insufficient to capture the structural characteristics underlying multi-lead ECG signals. This mismatch can limit the robustness and reliability of learned representations, particularly in settings where labeled data are limited.

To address these challenges, we propose a graph-based pseudo-multimodal contrastive learning framework called Graph-CMMC, for representation learning from 12-lead ECG signals. In the proposed framework, ECG waveforms are transformed into Gramian Angular Difference Field (GADF) [1] images to construct complementary representations of the same cardiac activity. While waveform signals preserve finegrained temporal morphology, GADF representations emphasize global correlation structures over time. This pseudomultimodal formulation allows complementary representations to be obtained from the same ECG recordings, without requir-

ing any additional data.

Based on these representations, Graph-CMMC employs contrastive learning in a self-supervised manner to align waveform and GADF representations across all 12 leads. In addition, a graph-based relational module is introduced to explicitly model inter-lead dependency and enforce structural consistency during representation learning.

The proposed method is evaluated on a multi-label coronary artery occlusion classification task using real-world 12-lead ECG recordings to assess its effectiveness in learning robust representations.

In this work, we make the following contributions.

• We propose Graph-CMMC, a graph-based pseudomultimodal contrastive learning framework for representation learning from 12-lead ECG signals, which leverages complementary waveform and GADF representations during pretraining, rather than as multimodal inputs in the downstream task.

• We introduce a learnable graph structure to explicitly model inter-lead dependency and incorporate this dependency directly into the representation learning process.

• Through experiments and analyses, we show that the proposed framework not only enables the encoder to learn more stable representations, but also learns transferable inter-lead graph structures that contribute to improved downstream performance.

## II. RELATED WORKS

## A. Time-series-based ECG analysis

Deep learning has been widely applied to automated ECG analysis, particularly through time-series models such as CNNs and RNNs, including LSTM [2] and GRU variants [3]. These approaches model ECG signals as one-dimensional temporal sequences and have shown strong performance in detecting cardiac abnormalities such as arrhythmias and atrial fibrillation [4].

However, most time-series-based methods process ECG signals per-lead and treat each lead independently. Such designs make it difficult to explicitly capture inter-lead dependency, which plays an important role in the interpretation of 12-lead ECG recordings.

In addition to direct time-series modeling, several studies have explored time-series-to-image transformations for ECG analysis, such as Gramian Angular Field and its variant GADF. These approaches convert ECG waveforms into twodimensional images that emphasize waveform morphology and temporal correlation patterns, and have been applied to tasks such as myocardial infarction detection. Yousuf et al. [5] applied GAF-based representations to single-lead ECG signals and demonstrated their effectiveness for myocardial infarction classification. Similarly, Zhang et al. [6] employed GADFbased representations with a PCANet architecture, showing robustness to noise and stable feature extraction. However, similar to many time-series-based approaches, these studies mainly focused on single-lead ECG analysis, typically using only Lead II, which limits their applicability to multi-lead ECG analysis.

## B. Multi-lead ECG modeling

Recent studies have explored multi-lead ECG modeling to better capture the complementary information across different leads. For example, Zhou et al. [7] proposed a hierarchical convolutional framework with lead encoder attention to exploit multi-lead features for ECG classification. Such approaches explicitly consider the contributions of individual leads and have demonstrated improved performance over single-lead models.

These studies highlight the importance of multi-lead modeling in ECG analysis. However, most existing multi-lead ECG models are designed in a supervised learning setting and model inter-lead relationships within task-specific architectures.

## C. Self-supervised ECG representation learning

Self-supervised learning has gained increasing attention in ECG analysis due to its ability to learn meaningful representations from unlabeled signals. Existing self-supervised ECG representation learning methods can be broadly categorized into reconstruction-based approaches and contrastive learningbased approaches.

Reconstruction-based methods typically adopt masked modeling strategies to recover missing signal segments. A representative method is ST-MEM, proposed by Na et al. [8], which applies masked modeling to 1D ECG time-series signals. ST-MEM introduces lead-wise patch masking and reconstructs masked signals using a shared Transformer encoder with sinusoidal positional embeddings and lead embeddings. This design enables the model to capture intra-lead temporal structures while leveraging shared representations across multiple leads.

Despite its effectiveness, reconstruction-based methods such as ST-MEM have several limitations. Operating directly on raw 1D signals makes it difficult to capture global morphological patterns, such as ST-segment elevation or QRS deformation, which involve long-range temporal structures. In addition, reconstruction objectives focus on accurately recovering the input signal, making them sensitive to local fluctuations and baseline drift, which can limit their ability to learn robust and discriminative representations for downstream diagnostic tasks.

In contrast, contrastive learning-based methods aim to learn invariant representations by aligning positive pairs while separating negative samples. By directly optimizing representation similarity and separation, contrastive learning is well suited for learning transferable ECG representations.

CLOCS is a family of contrastive learning methods for ECG analysis, proposed by Kiyasseh et al. [9]. Among its components, Contrastive Multi-lead Coding (CMLC) focuses on exploiting inter-lead relationships. CMLC treats temporally aligned ECG signals from different leads as positive pairs, based on the observation that different leads correspond to distinct projections of the same underlying cardiac electrical activity. By contrasting lead-wise views of the same ECG recording, CMLC captures spatial invariance across leads and incorporates inter-lead information into the learned representations.

However, in existing contrastive ECG representation learning methods such as CMLC, inter-lead relationships are primarily used for defining positive pairs, rather than being explicitly modeled as structured relational constraints during representation learning. Moreover, these methods typically operate within a single representation space and do not explicitly ensure that inter-lead relationships remain consistent across different ECG representations.

## III. PROPOSED METHOD

To address these limitations, we propose Graph-CMMC (Graph-based Cross-Modal Matching Contrastive Learning), a self-supervised framework designed for 12-lead ECG signals. An overview of Graph-CMMC is illustrated in Fig. 1.

Given a 12-lead ECG recording, the framework first constructs a pseudo-multimodal representation by encoding the same cardiac activity in two forms: raw waveform signals and GADF images. These representations are processed by modality-specific encoders implemented as lightweight 2-layer CNNs, while inter-lead relationships are explicitly modeled using a shared graph structure. Finally, a contrastive learning objective is applied to align waveform and GADF representations in a self-supervised manner.

![](images/6591654a9b9ffeb43500d48e6c57783063ec7a2ad328100b1b090d56f70682d3.jpg)  
Fig. 1: An overview of Graph-CMMC

## A. Pseudo-multimodal

To bridge the representation gap between time-seriesbased ECG modeling and clinical diagnostic practice, Graph-CMMC adopts a pseudo-multimodal representation strategy. Here, pseudo-multimodal representation refers to constructing multiple complementary representations from the same 12- lead ECG recording, where GADF images provide a twodimensional view emphasizing global waveform structures that may be less explicit in raw waveform signals.

1) GADF Transform: The GADF transform maps temporal relationships in ECG signals into a spatial representation by encoding pairwise angular differences over time. Compared to raw waveform signals, this two-dimensional representation emphasizes global waveform morphology and temporal correlation patterns that are less explicit in time-series form and are closely related to the visual inspection process used by clinicians.

As shown in Fig. 2, the GADF transformation involves three steps. First, the ECG signal is normalized to the range [−1, 1].

$$
x _ { t } ^ { \prime } = \frac { x _ { t } - \operatorname* { m i n } ( \pmb { x } ) } { \operatorname* { m a x } ( \pmb { x } ) - \operatorname* { m i n } ( \pmb { x } ) } \times 2 - 1 .\tag{1}
$$

Second, the normalized signal is mapped to angular values.

$$
\phi _ { t } = \operatorname { a r c c o s } ( x _ { t } ^ { \prime } ) .\tag{2}
$$

Finally, the GADF image is constructed by computing the sine of pairwise angular differences.

$$
\mathbf { G A D F } = \left[ \begin{array} { c c c c } { \sin \left( \phi _ { 1 } - \phi _ { 1 } \right) } & { \cdot \cdot \cdot } & { \sin \left( \phi _ { 1 } - \phi _ { T } \right) } \\ { \sin \left( \phi _ { 2 } - \phi _ { 1 } \right) } & { \cdot \cdot } & { \sin \left( \phi _ { 2 } - \phi _ { T } \right) } \\ { \vdots } & { \ddots } & { \vdots } \\ { \sin \left( \phi _ { T } - \phi _ { 1 } \right) } & { \cdot \cdot } & { \sin \left( \phi _ { T } - \phi _ { T } \right) } \end{array} \right]\tag{3}
$$

where T denotes the length of the time series, and the generated GADF image is resized to 64 × 64 in this study.

![](images/61855b6251ffcbc0557aa046d53a2c4f870881a4b7025e8c0855cc579278ae9b.jpg)  
Fig. 2: An overview of GADF Transform

Through this process, a one-dimensional ECG waveform is converted into a structured, pattern-like two-dimensional image. In this study, GADF is introduced as a complementary representation during pretraining to encourage the model to capture such global structural information.

It should be noted that the pseudo-multimodal design in Graph-CMMC is introduced specifically for self-supervised pretraining, rather than for increasing the complexity of the downstream model. By aligning waveform-based and GADFbased representations during pretraining, the encoder is encouraged to capture both local temporal patterns and global structural characteristics. In the downstream task, only the waveform-based encoder is used in order to evaluate the quality and transferability of the learned representations under a simple and consistent setting.

## B. Graph-based Module

To explicitly capture inter-lead dependency in 12-lead ECG signals, Graph-CMMC incorporates a graph-based relational module. Each ECG lead is treated as a node in a graph, and lead-wise relationships are represented using an adjacency matrix that encodes inter-lead dependency.

The graph-based module operates on lead-level representations extracted by the modality-specific 2-layer CNN encoders. After feature extraction, a graph convolution layer is applied to these lead-level representations to enable relational message passing across leads. This design enables information exchange across leads and supports modeling inter-lead dependency in multi-lead ECG signals.

In the proposed framework, the adjacency matrix is learned through backpropagation in the waveform-based branch rather than the GADF-based branch. At each training step, the updated adjacency matrix is shared with the GADF-based branch and used for graph convolution. The GADF-based branch does not independently optimize the adjacency matrix, but instead utilizes the shared relational structure learned from waveform representations. This sharing refers to the use of a common inter-lead dependency structure, rather than sharing network parameters across modalities.

This design choice is motivated by the fact that waveform representations preserve direct temporal variations across leads, providing a more reliable basis for modeling lead-wise dependency. GADF representations are derived views that emphasize global structural patterns and are therefore used as complementary representations, rather than as the primary source for learning relational structure.

By modeling inter-lead relationships through a shared graph structure, Graph-CMMC encourages waveform-based and GADF-based representations to respect consistent leadwise dependency patterns during representation learning.

## C. Contrastive Learning

Graph-CMMC adopts a SimCLR-style [10] contrastive learning framework to align waveform-based and GADF-based representations. The objective is to encourage representations derived from the same ECG recording to be close in the embedding space, while separating representations from different recordings.

The resulting representations from the waveform and GADF branches are aggregated and projected into a shared embedding space using a lightweight MLP-based projection head. These projected representations are then used for contrastive learning.

For a given ECG recording, the waveform-based representation and the corresponding GADF-based representation are treated as a positive pair. Representations from different ECG recordings within the same mini-batch are treated as negative samples.

The contrastive objective is defined using the Normalized Temperature-scaled Cross-entropy (NT-Xent) loss. Given a positive pair $( \mathbf { z } _ { i } , \mathbf { z } _ { g } )$ , where $\mathbf { z } _ { i }$ and $\mathbf { z } _ { g }$ denote the projected representations from the waveform-based and GADF-based branches of the same ECG recording, respectively, the loss is defined as

$$
\mathcal { L } _ { i , g } = - \log \frac { \exp ( \sin ( \mathbf { z } _ { i } , \mathbf { z } _ { g } ) / \tau ) } { \sum _ { k \neq i } \exp ( \sin ( \mathbf { z } _ { i } , \mathbf { z } _ { k } ) / \tau ) }\tag{4}
$$

where sim(·, ·) denotes cosine similarity and τ is a temperature parameter. In practice, a symmetric formulation is adopted, where the contrastive loss is computed in both directions, i.e., waveform-to-GADF and GADF-to-waveform, and the results are averaged. This loss encourages alignment between corresponding waveform and GADF representations while separating non-corresponding pairs within the batch.

By optimizing this objective, Graph-CMMC aligns waveform and GADF representations while preserving inter-lead dependency through the shared graph structure.

## IV. EXPERIMENT

## A. Dataset

The 12-lead ECG dataset used in this study was collected from real clinical recordings provided by Yokohama City University Medical Center.

The dataset consists of ECG recordings from 1,068 patients. Each ECG sample is annotated by physicians with coronary artery occlusion information, where the culprit vessel was determined based on coronary angiography.

The original dataset contains 38,516 ECG segments, each with a temporal length of 10,000 points. For experimental evaluation, ECG segments were extracted using a fixed-length temporal window of 2,000 points, corresponding to approximately 4 cardiac cycles, to ensure consistent input length across samples. To maintain a balanced label distribution and reduce the impact of class imbalance, a total of 3,641 ECG samples were selected for this study, with approximately 1,000 samples for each coronary artery label.

The downstream task is formulated as a multi-label classification problem. Each ECG sample may be associated with one or multiple occluded coronary arteries. In this study, four major coronary arteries are considered: left anterior descending artery (LAD), left circumflex artery (LCX), right coronary artery (RCA), and left main trunk (LMT).

## B. Experimental Settings

The dataset is split into training, validation, and test sets with a ratio of 6:2:2. Patient-level separation is not enforced during data splitting, and as multiple ECG samples are available for some patients, ECG data from the same patient may appear across different subsets. To ensure fair comparison, the random seed is fixed so that all self-supervised methods are trained on the same data distribution.

The experiments are conducted in two stages: selfsupervised pretraining and downstream multi-label classification.

1) Self-supervised Pretraining: Self-supervised pretraining is performed using the entire training set without using any label information.

During pretraining, models are trained for 200 epochs with a batch size of 8. The Adam optimizer is used with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ . All models are trained using the same network architecture and optimization settings to ensure a fair comparison.

2) Downstream Classification: For downstream classification, the pretrained waveform-based encoder is extracted and used as a fixed feature extractor. A multi-layer perceptron (MLP) classifier is then attached to the encoder output for multi-label classification.

The MLP classifier consists of 2 fully connected layers with batch normalization, GELU activation, and dropout. During downstream training, the pretrained encoder is frozen and only the classification head is optimized.

The classification head is trained for 50 epochs with a batch size of 8 using the Adam optimizer. The learning rate is set to $1 \times 1 0 ^ { - 3 }$ . Early stopping based on validation performance is applied to prevent overfitting.

3) Evaluation Metrics: Model performance is evaluated using multiple metrics suitable for multi-label classification.

• Macro F1-score: The F1-score is computed independently for each label and then averaged across all labels. This metric reflects balanced performance among classes and is insensitive to label imbalance.

• Subset accuracy: A strict evaluation metric that counts a prediction as correct only when the predicted label set exactly matches the ground-truth label set for a given sample.

• Overall accuracy: Overall accuracy is computed as 1 − HammingLoss, where Hamming Loss measures the fraction of incorrectly predicted labels over all labels, providing a label-wise accuracy measure.

• Jaccard index: the Jaccard index (sample-based Intersection over Union) is used to evaluate the overlap between predicted and ground-truth label sets for each sample.

These metrics are computed on the test set. All experiments are repeated three times, and the results are reported as the average over these runs.

4) Comparison methods: To evaluate the effectiveness of the proposed Graph-CMMC, several self-supervised and supervised methods are selected for comparison. All comparison methods use the same dataset split, encoder architecture, and downstream classification head to ensure a fair evaluation.

• Supervised baseline: A fully supervised model with the same 2-layer 1D CNN encoder and MLP classifier as Graph-CMMC, trained end-to-end using labeled data.

• CMLC [9]: As shown in Fig. 3, CMLC performs contrastive learning by treating ECG signals from different leads of the same patient as positive pairs, while signals from other patients serve as negative samples.

![](images/6d4c95499754247fdd1c607a68e21ccf0770cf334fdae8bdf609d3999266583e.jpg)  
Fig. 3: An overview of CMLC

• CMMC: As shown in Fig. 4, CMMC performs contrastive learning by aligning waveform-based and GADFbased representations of the same ECG recording as positive pairs, while representations from different recordings are treated as negative samples. Unlike Graph-CMMC,

CMMC considers each lead independently and does not model inter-lead relationships using graph structures.

![](images/6562776b61f36b371708075027bb53ef9d41424cc617e35e73da5ecbf54b53e2.jpg)  
Fig. 4: An overview of CMMC

• CMMC+CMLC: The overview of CMMC+CMLC is shown in Fig. 5. This method combines CMMC and CMLC by jointly applying cross-modal contrastive learning and multi-lead contrastive learning.

![](images/f6b99e690409f0f3b11f775ff963dd58365c46e6ce091eeadc4cf6e2fafce530.jpg)  
Fig. 5: An overview of CMMC+CMLC

## C. Results

For the proposed Graph-CMMC, two downstream evaluation settings are considered to examine how the interlead dependency learned during self-supervised pretraining is transferred to the downstream task.

In Graph-CMMC (d1), the pretrained waveform-based encoder is followed by an MLP classifier for multi-label classification. In Graph-CMMC (d2), the inter-lead adjacency matrix learned during pretraining is incorporated into the downstream model to perform graph-based feature aggregation before classification. By comparing these two settings, we evaluate whether the inter-lead dependency learned in the selfsupervised stage remains effective after encoder transfer.

Table I summarizes the test performance of all methods in terms of Macro F1-score, Subset Accuracy, Overall Accuracy, and Jaccard index. The fully supervised baseline achieves the highest performance and serves as an upper-bound reference.

Among the self-supervised methods, Graph-CMMC achieves the best performance across all evaluation metrics. Graph-CMMC (d2) outperforms Graph-CMMC (d1), indicating that incorporating the learned inter-lead graph structure at the downstream stage provides additional gains beyond encoder-only transfer.

Regarding the comparison methods, CMLC shows lower performance across all metrics, suggesting that contrastive learning based only on inter-lead pairing is insufficient for this multi-label coronary artery occlusion task. CMMC improves over CMLC, indicating that aligning waveform and GADF representations leads to more informative features. However, its performance remains below Graph-CMMC, which models inter-lead dependency through a learnable graph structure. The combined method, CMLC+CMMC, does not outperform CMMC alone, suggesting that combining multi-lead and crossmodal contrastive objectives does not necessarily lead to additive improvements.

Table II reports class-wise F1-scores for each coronary artery. Graph-CMMC (d2) achieves higher F1-scores than Graph-CMMC (d1) for all four classes (LAD, LCX, LMT, and RCA), and outperforms the other self-supervised baselines across classes. These results indicate that the learned interlead modeling benefits multiple coronary regions rather than improving only a specific class.

TABLE I: Experimental results
<table><tr><td>Method</td><td>Macro F1</td><td>Subset Accuracy</td><td>Overall Accuracy</td><td>Jaccard</td></tr><tr><td>Supervised baseline</td><td>0.822</td><td>0.771</td><td>0.903</td><td>0.786</td></tr><tr><td>CMLC</td><td>0.563</td><td>0.432</td><td>0.797</td><td>0.515</td></tr><tr><td>CMLC+CMMC</td><td>0.703</td><td>0.593</td><td>0.846</td><td>0.628</td></tr><tr><td>CMMC</td><td>0.729</td><td>0.627</td><td>0.857</td><td>0.665</td></tr><tr><td>Graph-CMMC (d1)</td><td>0.746</td><td>0.670</td><td>0.855</td><td>0.715</td></tr><tr><td>Graph-CMMC (d2)</td><td>0.776</td><td>0.711</td><td>0.873</td><td>0.752</td></tr></table>

TABLE II: Per-class F1-score
<table><tr><td>Model</td><td>LAD</td><td>LCX</td><td>LMT</td><td>RCA</td></tr><tr><td>Supervised baseline</td><td>0.781</td><td>0.809</td><td>0.936</td><td>0.763</td></tr><tr><td>CMLC</td><td>0.328</td><td>0.526</td><td>0.769</td><td>0.629</td></tr><tr><td>CMLC+CMMC</td><td>0.680</td><td>0.602</td><td>0.834</td><td>0.696</td></tr><tr><td>CMMC</td><td>0.726</td><td>0.607</td><td>0.859</td><td>0.723</td></tr><tr><td>Graph-CMMC (d1)</td><td>0.739</td><td>0.697</td><td>0.861</td><td>0.686</td></tr><tr><td>Graph-CMMC (d2)</td><td>0.758</td><td>0.720</td><td>0.880</td><td>0.745</td></tr></table>

## D. Discussion

1) Feature space analysis of Inter-Lead Representations: To investigate where the performance differences among methods originate, we analyzed the inter-lead structure learned by each self-supervised approach. For each ECG sample, leadwise embeddings were extracted from the waveform-based encoder $h _ { i } \in \mathbb { R } ^ { D } , \ i = 1 , \dots , 1 2$ , and Euclidean distances were computed for all unique lead pairs, resulting in 66 pairwise distances for the 12-lead ECG. The distribution of these distances was summarized using four statistics: mean, standard deviation, minimum, and maximum.

Table III reports the inter-lead distance statistics for each method. CMLC exhibits a large standard deviation together with extreme minimum and maximum distances, indicating that both overly similar and overly separated lead pairs are present, resulting in an unstable inter-lead structure. CMMC shows a more balanced distance distribution, with moderate mean distance and reduced variance. However, when CMLC is combined with CMMC, the mean distance increases and the variance remains relatively large, suggesting that the combined objective does not improve the stability of the learned structure.

In contrast, Graph-CMMC achieves the smallest mean distance and the lowest standard deviation among all methods.

TABLE III: Inter-lead embedding distance statistics for different methods
<table><tr><td>Method</td><td>Mean (↓)</td><td>Std (↓)</td><td>Min</td><td>Max</td></tr><tr><td>CMMC</td><td>0.0659</td><td>0.0265</td><td>0.0188</td><td>0.1190</td></tr><tr><td>CMLC</td><td>0.0648</td><td>0.0335</td><td>0.0145</td><td>0.1735</td></tr><tr><td>CMMC+CMLC</td><td>0.0639</td><td>0.0264</td><td>0.0150</td><td>0.1381</td></tr><tr><td>Graph-CMMC</td><td>0.0589</td><td>0.0265</td><td>0.0119</td><td>0.1248</td></tr></table>

This indicates that the inter-lead structure is learned in a more compact and stable manner, without collapsing into trivial representations or excessively separating specific leads.

Fig. 6 provides a distribution-level view of inter-lead distances and supports the observations in Table III. Graph-CMMC shows a narrower interquartile range and fewer extreme values compared with the other methods, indicating more consistent inter-lead relationships across samples.

A possible interpretation is related to how each contrastive objective treats lead-specific information. CMLC encourages representations from different leads to become similar, which may suppress lead-specific characteristics. CMMC does not explicitly enforce such alignment, resulting in a more averaged distance distribution. When these objectives are combined, conflicting optimization pressures may arise and destabilize the feature space. In contrast, Graph-CMMC explicitly models inter-lead dependency through a graph structure, which is consistent with the observed compact and stable representations.

![](images/ed9da8b9688b10fe525fbaa9a4c0746e7ae503df56d657869450e012494b7c8b.jpg)  
Fig. 6: Boxplot comparison of inter-lead distance distributions for different methods

2) Visualization oflearned Inter-lead dependency in Graph-CMMC: To further examine what type of inter-lead dependency is captured by Graph-CMMC, we visualize the learned adjacency matrix and its community structure.

As shown in Fig. 7a, the adjacency matrix learned by Graph-CMMC is not symmetric. This indicates that the model captures directional dependencies, where features from one lead contribute differently to the representation of another lead. Such asymmetry is expected in 12-lead ECG signals, as each lead observes cardiac electrical activity from a different spatial orientation. Therefore, the learned asymmetric structure reflects directional information flow among leads rather than assuming mutual equivalence.

As an example of the learned dependency structure, Fig. 7b shows the inter-lead communities obtained by applying Louvain clustering [11] to the learned adjacency matrix. The 12 leads are partitioned into four communities: {I, aVF}, {aVL, V1, V3, V6}, {aVR, V4, V5}, and {II, III, V2}.

The resulting communities do not strictly correspond to anatomical groupings defined in clinical textbooks, and no clear consistency with clinical interpretations was observed. This reflects a limitation of data-driven approaches, which primarily extract statistical dependencies from training data rather than explicitly encoding clinically defined structures.

Despite this limitation, the adjacency matrix learned by Graph-CMMC captures inter-lead dependencies that are informative for the downstream task. Although the learned structure does not directly match predefined anatomical groupings, it emphasizes task-relevant relationships among leads that are beneficial for the downstream task.

![](images/1a91f9072754c8f3a49ca6789ffc7c526f95f7d965ad20e16968cfde07690257.jpg)  
(a) Heatmap

![](images/037f266c45967eaf31db2ebbea36388248ff91ac1b439de2d91d6771eebbe8c4.jpg)  
(b) Louvain clustering results  
Fig. 7: Visualization of the inter-lead dependency learned by Graph-CMMC. (a) Heatmap of the learned adjacency matrix. (b) Louvain clustering results based on the learned adjacency matrix.

## V. CONCLUSION

In this study, we proposed Graph-CMMC, a self-supervised graph-based pseudo-multimodal contrastive learning framework for representation learning from 12-lead ECG signals. The proposed method constructs complementary waveform and GADF representations from the same ECG recording and aligns them through contrastive learning, while explicitly modeling inter-lead dependency using a learnable graph structure.

By incorporating a graph-based relational module, Graph-CMMC enables structured information exchange across leads and enforces consistency of inter-lead relationships during representation learning. Compared with existing self-supervised methods, the proposed method achieves superior performance on a multi-label coronary artery occlusion classification task, approaching the performance of fully supervised learning. Furthermore, incorporating the learned inter-lead graph structure in the downstream stage provides additional performance gains, indicating that the dependency learned during selfsupervised pretraining remains effective after encoder transfer.

Additional analyses of the learned feature space demonstrate that Graph-CMMC produces more compact and stable interlead representations than comparison methods. Visualization of the learned adjacency matrix further reveals directional inter-lead dependencies that reflect task-relevant information flow among leads, although these learned structures do not strictly correspond to clinically predefined anatomical groupings.

This study highlights the effectiveness of combining pseudo-multimodal representation learning with explicit graph-based modeling for 12-lead ECG analysis. Graph-CMMC serves as a step toward bridging the gap between data-driven ECG representation learning and the multi-lead relational reasoning employed in clinical practice.

For future work, we plan to evaluate the proposed framework on larger and more diverse ECG datasets, including experiments with patient-wise data splitting. We will also investigate lightweight graph designs and conduct ablation studies on architectural choices such as encoder depth and the number of graph convolution layers. In addition, different downstream strategies and incorporating domain knowledge will be explored to better understand and interpret the learned inter-lead dependency structures.

## ACKNOWLEDGMENT

This study protocol was reviewed and approved by the Yokohama National University Ethics Committee for Medical and Biological Research Involving Human Subjects (Approval No. Hitoi-2023-28).

## REFERENCES

[1] Z. Wang, T. Oates, “Imaging time-series to improve classification and imputation,” in Proceedings of the 24th International Conference on Artificial Intelligence (IJCAI’15), pp. 3939-3945, AAAI Press, Buenos Aires, Argentina, 2015.

[2] S. Hochreiter, J. Schmidhuber, “Long short-term memory,” Neural Comput., 9, no. 8, pp. 1735-1780, 1997.

[3] R. Dey, F. M. Salem, “Gate-variants of gated recurrent unit (GRU) neural networks,” 2017 IEEE 60th international midwest symposium on circuits and systems (MWSCAS), IEEE, 2017.

[4] N. Strodthoff, P. Wagner, T. Schaeffter, W. Samek, “Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL,” IEEE Journal of Biomedical and Health Informatics, 25, pp. 1519-1528, 2021.

[5] A. Yousuf, R. Hafiz, S. Riaz, M. A. Farooq, K. Riaz, M. M. U. Rahman, “Inferior Myocardial Infarction Detection from Lead II of ECG: A Gramian Angular Field-Based 2D-CNN Approach,” IEEE Sensors Letters, 8, pp. 1-4, 2023.

[6] G. Zhang, Y. Si, D. Wang, W. Yang, Y. Sun, “Automated Detection of Myocardial Infarction Using a Gramian Angular Field and Principal Component Analysis Network,” IEEE Access, 7, pp. 171570-171583, 2019.

[7] F. Zhou, D. Fang, “Classification of multi-lead ECG based on multiple scales and hierarchical feature convolutional neural networks,” Scientific Reports, 15 (1), 16418, 2025.

[8] Y. Na, M. Park, Y. Tae, S. Joo, “Guiding Masked Representation Learning to Capture Spatio-Temporal Relationship of Electrocardiogram,” in Proceedings of the Twelfth International Conference on Learning Representations (ICLR), 2024.

[9] D. Kiyasseh, T. Zhu, D. A. Clifton, “Clocs: Contrastive learning of cardiac signals across space, time, and patients,” In International Conference on Machine Learning, PMLR, pp. 5606-5615, 2021.

[10] T. Chen, S. Kornblith,M. Norouzi, G. Hinton, “A simple framework for contrastive learning of visual representations,” In International conference on machine learning, PMLR, pp. 1597-1607, 2020.

[11] V. D. Blondel, J. L. Guillaume, R. Lambiotte, E. Lefebvre, “Fast unfolding of communities in large networks,” Journal of statistical mechanics: theory and experiment, 2008 (10), P10008, 2008.