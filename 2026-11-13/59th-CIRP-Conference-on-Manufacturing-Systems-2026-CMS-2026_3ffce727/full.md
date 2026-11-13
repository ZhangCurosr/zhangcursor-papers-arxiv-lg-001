# 59th CIRP Conference on Manufacturing Systems 2026 (CMS 2026)

# ScrewSense: Mapping the Invisible – Anomaly Detection and Error Classification in Screwdriving Processes

Tobias Lettner<sup>a,∗</sup>, Tatyana Ivanovska<sup>a</sup>, Anwar Al Assadi<sup>b</sup>, Christian Bergler<sup>a</sup>

<sup>a</sup>Department ofElectrical Engineering, Media and Computer Science, Ostbayerische Technische Hochschule Amberg-Weiden, Amberg, Germany <sup>b</sup>Fraunhofer Institute for Manufacturing Engineering and Automation, Stuttgart, Germany

\* Corresponding author: t.lettner@oth-aw.de

## Abstract

<sub>The</sub> <sub>growing</sub> <sub>volume</sub> <sub>of</sub> <sub>process</sub> <sub>data</sub> <sub>generated</sub> <sub>by</sub> <sub>screwdriving</sub> <sub>systems,</sub> <sub>combined</sub> <sub>with</sub> <sub>advancements</sub> <sub>in</sub> <sub>computing</sub> <sub>power,</sub> <sub>data</sub> <sub>storage,</sub> <sub>and</sub>N algorithms, is facilitating the application of artificial intelligence in this field. Accurately detecting and classifying screwdriving errors is crucial for ensuring product quality, avoiding costly rework in manufacturing, and supporting sustainability by facilitating the maintenance, reuse, and recycling of screw-connected products. In this context, machine learning approaches ofer a promising alternative to traditional monitoring techniques, such as window-based methods or envelope curves. This study examines the use of deep learning techniques for anomaly detection and error classification in screwdriving processes. The data corpus consists of multivariate time series capturing process parameters, such as torque and rotation angle. For anomaly detection, the embedding space of a vector quantized variational autoencoder was analyzed. In terms of error classification, four diferent model architectures were evaluated. The results demonstrate the efectiveness of anomaly detection and confirm that deep learning models can distinguish between multiple error categories beyond traditional binary classification. This extends the capabilities of<sup>s</sup> conventional monitoring approaches, achieving accuracies of up to 100 %. Additionally, real-world data material was examined in order to further<sup>c</sup> demonstrate model robustness. Overall the results indicate strong generalization capability and highlight the practical potential of the developed models for use in industrial screwdriving processes, eliminating the need for expert knowledge in error identification and significantly reducing manual efort and analysis time.v

© 2026 The Authors. This is the accepted manuscript of an article published in Procedia CIRP 146 (2026) 421–426, https://doi.org/10.1016/j.procir.2026.03.204.

This work is licensed under CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)

Peer-review under responsibility of the scientific committee of the 59th CIRP CMS 2026.

Keywords: Anomaly Detection; Error Classification; Deep Learning; Time Series Analysis; Industrial Manufacturing; Screwdriving Technology

## 1. Introduction and Motivation

Screwdriving systems have evolved into complex technologies that continuously collect process data, such as torque and<sup>r</sup> rotation angle [1]. This data is primarily used to evaluate the quality of screw joints by determining whether they meet the required specifications (OK versus NOK) [2]. Traditionally, this evaluation has relied on monitoring windows that define fixed upper and lower limits for key parameters [2, 3].

However, threshold-based methods only assess limited areas of the tightening process. Envelope curves [4] can be used to analyze the entire process, but they require extensive parameterization and only indicate whether a process is acceptable or not, without identifying the underlying cause of the deviation.

The trend toward lightweight construction, advanced materials, and higher automation in industrial production is making screwdriving processes increasingly complex. This necessitates adaptive and intelligent monitoring approaches to maintain consistent assembly quality and process reliability [5].

The ongoing rise of artificial intelligence, particularly deep learning, provides innovative solutions to these challenges. They enable flexible and comprehensive analysis of the entire screwdriving process by learning complex patterns and dependencies within the data. Machine learning reduces the need for manual parameter tuning next to expert- and domain-specific rules, supporting higher levels of automation, better quality control, and more reliable processes.

The paper is structured as follows: Related Work summarizes prior work, Data Situation describes the data and preprocessing, Methodology details the proposed approach, Experiments, Results, and Discussion presents the results, and Conclusion and Future Work summarizes the main findings and outlines future work.

<table><tr><td>Class</td><td>assm_a</td><td>dism_a</td><td>reasm_a</td><td>Class</td><td>dism_b</td><td>dism_c</td></tr><tr><td>OK</td><td>58</td><td>59</td><td>59</td><td>OK</td><td></td><td>50</td></tr><tr><td>BURSTING_TUBE</td><td>48</td><td>48</td><td>46</td><td>REFERENCE</td><td>485</td><td></td></tr><tr><td>TIGHT_SCREW</td><td>一</td><td>50</td><td>50</td><td>REFERENCE_LARGE</td><td>372</td><td></td></tr><tr><td>CROSS_THREADING</td><td></td><td>一</td><td>47</td><td>INCORRECT_DRIVE</td><td>480</td><td>101</td></tr><tr><td>DAMAGED_SCREW_DRIVE</td><td>一</td><td>51</td><td>一</td><td>TIGHT_SCREW</td><td>46</td><td>50</td></tr><tr><td>SCREW_EXTRACTION</td><td>51</td><td>53</td><td>1</td><td>DAMAGED_HEAD_LARGE</td><td>381</td><td></td></tr><tr><td>PLASTIFICATION_THREAD</td><td></td><td>50</td><td>49</td><td>DAMAGED_DRIVE</td><td></td><td>100</td></tr></table>

Table 1: Class distribution of all datasets used in this study, showing the available classes and the corresponding number of samples per dataset. The dataset identifiers follow the naming convention assm (assembly), dism (dismantling), and reasm (reassembly), with sufixes a, b, and c denoting diferent system configurations.

## 2. Related Work

Recent research has increasingly explored the use of machine learning, particularly modern and state-of-the-art deep learning techniques for monitoring and analyzing the screwdriving process.

Althoefer et al. [6] used artificial neural networks with radial basis functions to evaluate torque and insertion depth signatures to distinguish between five diferent error classes. The data originates from computer simulations and experiments involving a rotary torque sensor and an optical shaft encoder with an electric screwdriver. While their model achieved perfect accuracy, it was sensitive to changes in insertion depth.

Meiners et al. [7] evaluated several machine learning models including MLPs, SVMs, and 1D-CNNs, for classifying torque curve anomalies in individual screwdriving operations. The generated dataset contains 228 screwdriving process runs, covering both OK and several distinct failure classes. Using torque data as the only feature, the 1D-CNN achieved the best performance with an accuracy of 86.81 %, highlighting the potential of deep learning for process curve-based error classification.

Similarly, Al Assadi et al. [8] proposed a supervised learning approach to detect stripped and defect-free screw drives using angle-torque profiles from a dismantling process. The custom dataset of over 1,200 samples was used to evaluate a MLP and a 1D-CNN model. The 1D-CNN achieved accuracies between 96 % and 98 % on unseen test data.

Despite these successes, most approaches focus on predefined scenarios or rely on small, proprietary datasets. This limits their ability to generalize across complex production environments. In contrast, ScrewSense ofers a generalizable, deeplearning-based monitoring framework for screwdriving processes, which enables the detection of anomalies and the classification of errors across diverse industrial scenarios, independent of specific system configurations.

## 3. Data Situation

## 3.1. Dataset Material

This study utilizes a comprehensive collection of five diferent datasets on screwdriving processes for assembly, dismantling, and reassembly under various system configurations. The data material provides a diverse foundation for evaluating adaptive and intelligent deep learning driven monitoring approaches. All datasets are publicly available at https://github.com lettnert/ScrewSense.git.

The first three datasets, referred to as assm a, dism a, and reasm a, were generated using a DEPRAG 311ER27-0020 screwdriver with a vertical guidance system that was operated manually. Self-tapping Ejot Delta PT screws (3×10 mm) were driven into PA6.6-GF30 plastic specimens. These datasets represent the assembly, dismantling, and reassembly screwdriving operation of identical components. Two additional datasets, dism b and dism c, were recorded using diferent screwdriving systems. The data corpus dism b was generated with a DEPRAG 326E27-0080-H screwdriver mounted on a UR10e robotic arm, while dism c was created using a Bosch Rexroth Nexo screwdriving system with a manually operated gantry setup. Both datasets represent dismantling operations. The corresponding class distributions are summarized in Table 1.

Using the same configuration as for dism b, an additional set of 36 real-world data samples was acquired by loosening screws from lawn mower housings. This data corpus does not contain any class labels.

## 3.2. Data Preprocessing

The first step of the proposed data preprocessing pipeline describes a resampling step, followed by a min-max normalization. Resampling the input signals to a fixed length using linear interpolation [9] ensures consistent input dimensions across all samples. This process can also act as a form of compression or stretching, depending on the original length of the time series. Min-max normalization scales the data to a uniform range [10, 11]. This step stabilizes the training process and facilitates faster convergence by mitigating numerical instabilities. Each feature is normalized independently, preserving the relative diferences and distributions of the original data, which enhances the robustness and overall performance of the model.

Several data augmentations were applied [12] in order to further improve model generalization, comprising window warping, Gaussian noise injection, spike injection, step-like trend injection, and slope-like trend injection. Each transformation introduces controlled variations with respect to the input signals, enhancing robustness while preserving the underlying temporal characteristics. To avoid distorting the original signal characteristics, only a single augmentation is applied per sample, selected randomly during training.

In case of image-based models, time series data are converted into image representations using three methods: density transformation via two-dimensional histograms [13], Gramian Angular Field (GAF) transformation [14], and spectrogram transformation [15]. Before generating a spectrogram, the mean is subtracted from each sample to remove the DC component and align the signal around zero. The resulting amplitude values are then converted to the decibel scale to better capture signal dynamics [16]. Figure 1 visualizes the results of all three image conversion techniques.

![](images/5a1d71def6c7ef48aeb58d9ff7b9fcaed972040109649e675b16b956a45d82c5.jpg)  
Fig. 1: Example of image conversion results. From left to right: original input, density, GAF, and spectrogram representations of the torque signal. Each image ha a size of 224×224 pixel. The model uses three-channel inputs, with torque, rotation angle, and time as features. Density images use a 224×224 bin size; GAF is generated with the summation method; spectrogram parameters are nft = 512, hop length = 32, and a Hamming window.

## 4. Methodology

## 4.1. Anomaly Detection

In ScrewSense, anomaly detection is performed using a vector quantized variational autoencoder (VQVAE), originally introduced by Oord et al. [17]. In order to perform a task-specific model adaption, the VQVAE was pre-trained on a subset of the ImageNet dataset [18], which enabled it to learn general visual features and robust latent representations. Subsequently, the model was fine-tuned on the target corpus, where the time-series data were converted into two-dimensional images through GAF representations. Fine-tuning was conducted solely on samples labeled as OK or REFERENCE, representing the normal class. After training, each sample was processed through the encoder to obtain its embedding representation. To enable visualization and further analysis, the embeddings were reduced to two dimensions using t-SNE [19] (e.g. Figure 2). The resulting embeddings were plotted as a two-dimensional scatter plot, with each sample colored according to its class label. This provides an intuitive view of the clustering structure and potential anomalies.

## 4.2. Error Classification

In the error classification task, four distinct model architectures are utilized, each describing a diferent deep learning paradigm: MLP-, CNN-, transformer-, and vision-based models. The purpose of this comparison is to assess how efectively each architecture captures temporal dependencies and discriminates among error scenarios in screwdriving processes.

The first model, PatchTSMixer (PTSM) [20], is a lightweight architecture specifically designed for multivariate time series classification. It replaces the self-attention mechanism of transformers with a pure MLP-based design, thereby reducing memory and computational demands while maintaining strong performance on long sequences [21]. ConvTimeNet (CTN) [22] represents a CNN-based approach and employs deformable patch embeddings combined with hierarchical convolutional blocks to extract local patterns and capture multi-scale temporal dependencies within the data. ConvTran (CTR) [23], as a transformer-based model, combines convolutional feature extraction with advanced position encoding mechanisms to efectively capture temporal dependencies in time series data. Finally, the Vision-Transformer (ViT) [24] processes time series data that have been transformed into 2D-image representations. By leveraging the attention mechanism, originally developed for sequence modeling in natural language processing [25], the concept was transferred to the image domain by feeding subimage patches as sequential data to the model, allowing to learn robust and discriminative features for classification.

All models are trained supervised using the provided class labels and the Adam optimizer with exponential learning rate decay. Class imbalance is handled through weighted crossentropy loss. A 70/15/15 train/validation/test split with early stopping is employed, followed by 5-fold cross-validation, whose scores are reported in the experimental section.

## 5. Experiments, Results, and Discussion

## 5.1. Experiment-1: Anomaly Detection

The first experiment was conducted to analyze the embedding space generated from the datasets. Figure 2 visualizes the hidden embedding space of the data dism a corpus.

![](images/5d59b1dca4327ab0cb7ed68f1b373e1df4c902c36126f73f5fe97129a4ba1f95.jpg)  
Fig. 2: Embedding space of the dism a dataset. Each point represents a screwdriving sample described by torque, rotation angle, and time, with colors indicating class labels. Multiple anomalies occurred and are highlighted as A1 - A5.

While some classes are clearly distinct from others, five anomalies were observed and analyzed in more detail. These anomalies are denoted as A1 through A5. Samples belonging to anomaly A1 exhibit a screwdriving time approximately 140 ms longer than the other samples. Samples belonging to anomaly A2 show a breakaway torque that occurs about 50 ° later than in the normal. Anomalies A3 and A4 display an earlier breakaway torque of about 25 °, which also explains their close proximity within the embedding representation. Anomaly A5 represents an overlap between two classes, where the corresponding samples show a high degree of similarity. Due to the fact that all anomalies can occur in real-world scenarios, no samples were excluded for the adapted error classification experiment.

Figure 3 illustrates the embedding space of the most challenging dataset, reasm a. As an example, two anomaly samples from the OK class were selected, which are located within the CROSS THREADING cluster.

![](images/7108641a93d6a812a2a94ac3e8bb5c1704727b51cd8dec8996cad5527c24f5c6.jpg)  
Fig. 3: Embedding space of the reasm a dataset. Each point represents a screwdriving sample described by torque, rotation angle, and time, with colors indicating class labels. Multiple anomalies occurred. As an example, two OK samples are highlighted within the cluster dominated by CROSS THREADING instances.

Figure 4 visualizes the curve overlay of these samples compared to the normal class samples. These samples exhibit a cross-threading behavior, where the friction torque at the beginning of the screwdriving process is higher than average. This behavior is likely due to mislabeling during data collection. These samples were reassigned to the correct class for the adapted classification experiment.

![](images/59d8a85be6253709174b98660618dc151968568efca73232772c4d05ded8e4ae.jpg)  
Fig. 4: Curve overlay of the highlighted anomaly samples from Figure 3, compared to the normal class samples. The anomaly samples, displayed with higher opacity, exhibit typical cross-threading behavior, characterized by a higher friction torque during the initial rotations.

Similar patterns and anomalies were observed in the datasets assm a, dism b, and dism c. The same analysis procedure and methodology were applied to these three.

## 5.2. Experiment-2: Error Classification

The second experiment focused on using the proposed mod els to evaluate their classification performance on each dataset. For evaluation, balanced accuracy (BAC) and F1-score were used. The results are summarized in Table 2, with only the BAC reported for simplicity since the F1-score results were similar. All models demonstrate comparable overall performance. PatchTSMixer and ConvTimeNet perform well on simpler datasets. The ViT model with the GAF transformation outperforms the other models on more complex datasets.

<table><tr><td rowspan="2">Model</td><td colspan="5">Dataset performance - BAC</td><td rowspan="2">Mean</td></tr><tr><td>assm_a</td><td>dism_a</td><td>reasm_a</td><td>dism_b</td><td>dism_c</td></tr><tr><td>PTSM</td><td>1.0000</td><td>0.9157</td><td>0.7015</td><td>0.9912</td><td>0.9602</td><td>0.9137</td></tr><tr><td>CTN</td><td>1.0000</td><td>0.9425</td><td>0.5362</td><td>1.0000</td><td>0.9701</td><td>0.8898</td></tr><tr><td>CTR</td><td>0.9889</td><td>0.9257</td><td>0.6735</td><td>0.9867</td><td>0.9625</td><td>0.9074</td></tr><tr><td>ViT-Dens.</td><td>0.9944</td><td>0.9595</td><td>0.7391</td><td>0.9944</td><td>0.9700</td><td>0.9315</td></tr><tr><td>ViT-Spec.</td><td>0.9944</td><td>0.9276</td><td>0.6926</td><td>0.9532</td><td>0.9800</td><td>0.9096</td></tr><tr><td>ViT-GAF</td><td>0.9944</td><td>0.9757</td><td>0.7812</td><td>0.9934</td><td>0.9800</td><td>0.9450</td></tr></table>

Table 2: Model classification performance in terms of BAC across datasets, with the right column showing the averaged mean performance across all datasets. The best score for each dataset is highlighted in bold. ViT-GAF achieves the highest mean performance.

For a better comparison, the mean performance of the models over all datasets was calculated using unweighted averaging, treating each dataset as equally important. ViT-GAF achieves the best mean performance with a BAC-score of 0.9450 across all datasets. Therefore, the ViT-GAF will be used for Experiment-3/4. ViT-Density shows also strong average performance but does not outperform any other model on a single dataset. The other models show consistently lower average performance.

## 5.3. Experiment-3: Adapted Error Classification

An additional error classification was conducted using the ViT-GAF model on the adapted datasets obtained from Experiment-1. The comparison to the classification results on the original datasets from Experiment-2 are listed in Table 3.

<table><tr><td rowspan="2">Model: ViT-GAF Dataset</td><td colspan="2">Original dataset</td><td colspan="2">Adapted dataset</td></tr><tr><td>BAC</td><td>F1-Score</td><td>BAC</td><td>F1-Score</td></tr><tr><td>assm_a</td><td>0.9944</td><td>0.9937</td><td>1.0000</td><td>1.0000</td></tr><tr><td>dism_a</td><td>0.9757</td><td>0.9741</td><td>0.9757</td><td>0.9741</td></tr><tr><td>reasm_a</td><td>0.7812</td><td>0.7661</td><td>0.9234</td><td>0.9237</td></tr><tr><td>dism_b</td><td>0.9934</td><td>0.9982</td><td>0.9989</td><td>0.9989</td></tr><tr><td>dism_c</td><td>0.9800</td><td>0.9799</td><td>0.9900</td><td>0.9922</td></tr></table>

Table 3: Error classification results of the ViT-GAF model on adapted compared to original datasets, showing improvements for all datasets except dism a, which required no adaptation. Better value is highlighted in bold.

The classification performance improved across all datasets, except of dism a, because there was no adaption necessary.

There was a significant improvement for the most challenging dataset, reasm a, but some misclassifications remained. The confusion matrix and the embedding space (see Figure 5) reveal the reason for the misclassifications. Samples from the OK class were classified as PLASTIFICATION THREAD due to the similarity between these classes, caused by insuficient manipulation of the test specimens. The embedding space shows an overlap between these two classes.

![](images/556601fa245321c4d507fcec5d1f43d6c67f6c5b5f06a762716db495680cde3a.jpg)  
Fig. 5: Confusion matrix of the classification result and a subregion of the embedding space of the adapted reasm a dataset. Misclassifications mainly occur between OK and PLASTIFICATION THREAD class due to their similarity.

## 5.4. Experiment-4: Generalization to Real-World Data

The fourth experiment involved a generalization test using real-world data. The ViT-GAF model, trained solely on the dism b dataset, was applied unchanged to the real-world data. A total of 33 samples were classified as REFERENCE, and three as DAMAGED HEAD LARGE. Figure 6 illustrates the embedding space, which shows three distinct clusters for the REF-ERENCE samples and one DAMAGED HEAD LARGE sample that appears particularly far away from the other two.

![](images/01c781e851905eb611911eb3f0b0eddd0373eb88255daf3ca90486eddc29e49f.jpg)  
Fig. 6: Embedding space of the classified lawn mower data with highlighted anomalies. Sample colors indicate the predicted class labels.

The screwdriving curves, visualized in Figure 7, generally show accurate classification of all samples based on visual inspection. The DAMAGED HEAD LARGE samples exhibit the typical behavior of a drive slipping over the screw head, resulting in torque peaks during the screwdriving process.

The isolated sample, denoted as anomaly A1, exhibits a higher final angle compared to the two class samples, which explains its deviation in the embedding space. This deviation may be caused by an interruption during the screwdriving process. Anomaly A2 is characterized by a lower breakaway torque relative to the other samples, while anomaly A3 shows a higher friction torque during the loosening process. Both anomalies can occur in real applications. These anomalies occur within the REFERENCE class and can also occur in real applications. A visual inspection of the screwdriving curves shows that the screws are fully loosened and are classified correctly. Each curve follows the characteristic pattern of the loosening process: an initial increase in torque up to the breakaway point, followed by a decrease in torque and continued rotation.

![](images/8e49cc978ed4889fca5e54599b6f7b2d91a25a6abcd0002dcd33e400d6bafe3b.jpg)

![](images/8a04080c98c236e636cfeb9b3c518370f2d13410d35528eeb5f95c2381dda7eb.jpg)  
Fig. 7: Comparison of classified samples from lawn mower data. The upper subfigure shows the curve overlay of samples classified as REFERENCE, and the lower subfigure shows the curve overlay of samples classified as DAM-AGED HEAD LARGE.

## 6. Conclusion and Future Work

Deep learning techniques can detect anomalies and classify errors in the evaluation of screwdriving processes. The embedding space of the VQVAE model provides a clear visualization of diferent anomalies, ofering a promising approach for analyzing multiple screwdriving curves. The ViT-GAF performed well in error classification, and both models generalized well with respect to real-world production data. One limitation is the reliance on labeled data. Around 50 samples per class were suf ficient for good performance. However, diferences in systemor recording-specific signals may restrict model transferability across datasets. A direct comparison to existing approaches was not feasible, as no publicly evaluated benchmarks exist for the specific datasets and error types investigated in this work.

Future work includes generating synthetic data to augment training datasets, applying foundation models such as TOTEM [26], or leveraging meta-transfer learning techniques [27] to improve model generalization. Further advances may include integrating explainable artificial intelligence methods to increase transparency in anomaly detection. For example, anomalies can be visualized directly in time series using models like PredTrAD [28]. Another interesting area of exploration would be detecting errors during the screwdriving process by processing the time series sequentially rather than classifying only the final result. This would enable real-time anomaly detection and early intervention. Additionally, the PyScrew dataset [29], with over 34,000 screwdriving operations and 27 error types, ofers a valuable resource for model pre-training and validation. Leveraging this dataset enhances generalization and enable more robust evaluation across diverse screwdriving scenarios, and would also enable future comparisons with other approaches once common benchmarks become available.

## Acknowledgments

This work was supported by the German Federal Ministry for the Environment, Nature Conservation, Nuclear Safety and Consumer Protection under Desire4Electronics.

## References

[1] N. Rabbe, Messtechnik in der Schraubtechnik, in: R. Schmitt, E. Dietrich (Eds.), Handbuch Messtechnik in der industriellen Produktion: Valide Messergebnisse planen, erhalten, auswerten und verteilen, 1st Edition, Carl Hanser Verlag GmbH & Co. KG, Munchen, 2023, pp. 359–374.¨

[2] DEPRAG SCHULZ GMBH u. CO. KG, Technical Information: Screwdriving Technology and Quality Assurance, Tech. rep., DEPRAG SCHULZ GMBH u. CO. KG (2020). URL https://www.deprag.com/fileadmin/bilder\_content/ emedia/broschueren\_pics/Allgemein\_PDF/D0080/D0080en.pdf

[3] WEBER Schraubautomaten GmbH, Schraubtechnik Ratgeber (2025). URL https://www.weber-online.com/ schraubtechnik-ratgeber/

[4] W.-D. Schneider, Hullkurvenverfahren zur Prozeß¨ uberwachung, in: Flexi-¨ bel automatisiertes Taumelnieten, Springer Berlin Heidelberg, Berlin, Heidelberg, 1997, pp. 58–65. doi:10.1007/978-3-662-07247-9\_7.

[5] Nord-Lock Group, Benefits of Using Bolted Joints in the Industrial Sector (2021). URL https://www.nord-lock.com/learnings/bolting-tips/ 2021/industrial-bolted-joints/

[6] K. Althoefer, B. Lara, Y. Zweiri, L. D. Seneviratne, Automated failure classification for assembly with self-tapping threaded fastenings using artificial neural networks, Proceedings of the Institution of Mechanical Engineers, Part C: Journal of Mechanical Engineering Science 222 (6) (2008) 1081– 1095. doi:10.1243/09544062JMES546.

[7] M. Meiners, A. Mayr, J. Franke, Process curve analysis with machine learning on the example of screw fastening and press-in processes, Procedia CIRP 97 (2021) 166–171. doi:10.1016/j.procir.2020.05.220.

[8] Anwar Al Assadi and David Holtz and Frank Nagele and Christof Nitsche¨ and Werner Kraus and Marco F. Huber, Machine learning based screw drive state detection for unfastening screw connections, Journal of Manufacturing Systems 65 (2022) 19–32. doi:https://doi.org/10.1016/ j.jmsy.2022.07.013.

[9] pytorch.org, torch.nn.functional.interpolate. URL https://pytorch.org/docs/stable/generated/torch.nn. functional.interpolate.html

[10] A. Asesh, Normalization and Bias in Time Series Data, in: C. Biele, J. Kacprzyk, W. Kopec, J. W. Owsi´ nski, A. Romanowski, M. Sikorski´ (Eds.), Digital Interaction and Machine Intelligence, Springer International Publishing, Cham, 2022, pp. 88–97.

[11] Y. Wu, K. He, Group normalization, in: Proceedings of the European conference on computer vision (ECCV), 2018, pp. 3–19.

[12] Q. Wen, L. Sun, F. Yang, X. Song, J. Gao, X. Wang, H. Xu, Time Series Data Augmentation for Deep Learning: A Survey, in: Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-2021, International Joint Conferences on Artificial Intelligence Organization, 2021, p. 4653–4660. doi:10.24963/ijcai.2021/631.

[13] numpy.org, numpy.histogram2d. URL https://numpy.org/doc/stable/reference/generated/ numpy.histogram2d.html

[14] Z. Wang, T. Oates, Encoding Time Series as Images for Visual Inspection and Classification Using Tiled Convolutional Neural Networks, 2015.

[15] pytorch.org, torch.stft. URL https://pytorch.org/docs/stable/generated/torch. stft.html

[16] C. Bergler, Deep Learning Applied to Animal Linguistics, Phd thesis, Friedrich-Alexander-Universitat Erlangen-N¨ urnberg (FAU),¨ available at https://nbn-resolving.org/urn:nbn:de:bvb: 29-opus4-239268 (9 2023).

[17] A. van den Oord, O. Vinyals, K. Kavukcuoglu, Neural Discrete Representation Learning (2018). arXiv:1711.00937. URL https://arxiv.org/abs/1711.00937

[18] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, L. Fei-Fei, ImageNet Large Scale Visual Recognition Challenge, International Journal of Computer Vision (IJCV) 115 (3) (2015) 211–252. doi:10.1007/ s11263-015-0816-y.

[19] L. Van der Maaten, G. Hinton, Visualizing data using t-SNE., Journal of machine learning research 9 (11) (2008).

[20] V. Ekambaram, A. Jati, N. Nguyen, P. Sinthong, J. Kalagnanam, TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting, in: Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’23, ACM, 2023, p. 459–469. doi:10.1145/3580305.3599533.

[21] Q. Wen, T. Zhou, C. Zhang, W. Chen, Z. Ma, J. Yan, L. Sun, Transformers in Time Series: A Survey (2023). arXiv:2202.07125. URL https://arxiv.org/abs/2202.07125

[22] M. Cheng, J. Yang, T. Pan, Q. Liu, Z. Li, ConvTimeNet: A Deep Hierarchical Fully Convolutional Model for Multivariate Time Series Analysis (2024). arXiv:2403.01493. URL https://arxiv.org/abs/2403.01493

[23] S. Islam, H. Elmekki, A. Elsebai, J. Bentahar, N. Drawel, G. Rjoub, W. Pedrycz, A Comprehensive Survey on Applications of Transformers for Deep Learning Tasks (2023). arXiv:2306.07303. URL https://arxiv.org/abs/2306.07303

[24] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby, An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2021). arXiv:2010.11929. URL https://arxiv.org/abs/2010.11929

[25] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u. Kaiser, I. Polosukhin, Attention is All you Need, in: I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, R. Garnett (Eds.), Advances in Neural Information Processing Systems, Vol. 30, Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/ 2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

[26] S. Talukder, Y. Yue, G. Gkioxari, Totem: Tokenized time series embeddings for general time series analysis, arXiv preprint arXiv:2402.16412 (2024).

[27] L. Sun, J. Liang, C. Zhang, D. Wu, Y. Zhang, Meta-Transfer Metric Learning for Time Series Classification in 6G-Supported Intelligent Transportation Systems, IEEE Transactions on Intelligent Transportation Systems 25 (3) (2024) 2757–2767. doi:10.1109/TITS.2023.3250962.

[28] J. Schuster, A. Wolfel, F. Brunner, C. Bergler, PredTrAD – Prediction-¨ based Transformer for Anomaly Detection in Multivariate Time Series Data, in: Interspeech 2025, 2025, pp. 3873–3877. doi:10.21437/ Interspeech.2025-501.

[29] N. West, J. Deuse, PyScrew: A Comprehensive Dataset Collection from Industrial Screw Driving Experiments (2025). arXiv:2505.11925. URL https://arxiv.org/abs/2505.11925