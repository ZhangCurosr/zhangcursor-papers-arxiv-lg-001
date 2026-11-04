# Feature-Based and Deep Learning Methods for Astronomical Light Curve Classification

V. Baymakanov,<sup>1,</sup> <sup>\*</sup> Kh. Khamitov,<sup>1</sup> A. Raimova,<sup>1</sup> A. Isaeva,<sup>2</sup> R. Taran,

M. Zedan,<sup>1</sup> M. Kornilov,<sup>1,</sup> <sup>3</sup> M. Demianenko,<sup>4,</sup> <sup>5</sup> K. Malanchev,<sup>6</sup> and M. Hushchyn<sup>1</sup>

<sup>1</sup>HSE University

<sup>2</sup>Southern Federal University

<sup>3</sup>Lomonosov Moscow State University

<sup>4</sup>Max-Planck-Institut f¨ur Astronomie

<sup>5</sup>Heidelberg University

<sup>6</sup>Carnegie Mellon University

## Abstract

We compare feature-based machine learning and end-to-end deep learning for variable star classification using the Zwicky Transient Facility (ZTF) StarEmbed dataset, which contains about 40,000 objects across seven imbalanced classes. Classical models trained on physically motivated features consistently outperform deep networks trained directly on raw light curves. We find that limited labeled data contributes to the performance gap, while the multivariate, irregularly sampled, and morphologically diverse nature of light curves poses additional challenges for deep learning models. While deep learning holds promise for learning representations directly from observations, our results indicate that current models fail to generalize with the available training set size. These findings motivate further investigation of the benefits of scaling labeled datasets and self-supervised pretraining on large unlabeled surveys. This work establishes a quantitative baseline for future research and highlights the need for further innovations to unlock the potential of deep learning in astronomical time series analysis.

PACS numbers: 07.05.Mh, 95.75.Wx, 95.80.+p, 97.30.Jm, 97.30.Kn, 97.80.Hn, 97.30.Nr

Keywords: machine learning, deep learning, light curves, variable stars, ZTF

Submitted to Moscow University Physics Bulletin (special issue “Deep Learning in Computational Physics”, DLCP-2026).

## 1. INTRODUCTION

Modern astronomical sky surveys, such as the Zwicky Transient Facility (ZTF) [1] and the Legacy Survey of Space and Time (LSST) [2], collect enormous amounts of data. For example, the Vera C. Rubin Observatory [2], which recently began its operation, generates tens of terabytes of data per day, and this volume will only increase with the commissioning of newgeneration telescopes. Such data cannot be processed manually within a reasonable time frame, yet their processing is essential. Astronomers rely on it for a range of tasks, from classifying astronomical objects and detecting transient events to determining their astrophysical properties. Machine learning and deep learning methods enable the development of models that can eficiently handle these continuously growing data volumes.

The objects studied are usually divided into three broad categories: transients, such as supernova explosions [3, 4]; periodic objects, whose brightness variations repeat with a certain period [5]; and aperiodic or stochastic objects, whose brightness variations are noise-like [6, 7].

In this work, we address the multiclass classification of astronomical objects by their light curves, which is one of the most common and important tasks in modern astronomy. A light curve describes the variation of an object’s flux over time in one or more photometric filters, often referred to as bands. It can be viewed as a multivariate time series that typically exhibits an irregular observation cadence and missing data points due to observational constraints [8]. Moreover, each measurement comes with an associated uncertainty. These factors collectively make light-curve analysis challenging and should be taken into account for training machine learning models.

The classical approaches, developed over the past decades, build a separate specialized model for each task and dataset [9, 10]. For each light curve, several physically motivated features are extracted, and a machine-learning classifier is then fitted to these features. This approach is straightforward, transparent, and can yield good results, but models built this way transfer poorly to new surveys, with performance often degrading across datasets.

A more recent, promising alternative is based on foundation models [11, 12], trained in a selfsupervised regime to learn meaningful data representations, which are then fed into task-specific models. These models are universal and can be applied to any dataset and downstream task. However, their classification quality still underperforms compared to classical approaches. The quality gap between these two paradigms has not been fully investigated in previous works.

In this work, we study multiclass classification of astronomical objects by their light curves and compare two approaches: training classical machine learning models, such as Logistic Regression, CatBoost, and Multilayer Perceptron (MLP), on pre-extracted physically motivated features, and training deep neural networks with Transformer and Long Short-Term Memory (LSTM) architectures directly on raw observational data.

We test the hypothesis that deep learning networks can learn meaningful representations directly from raw light curves and are capable of solving the classification task. We also aim to quantify the performance gap between these two paradigms, identify the factors underlying their diferences, and evaluate whether the observed discrepancy stems from architectural limitations or from the intrinsic complexity of the data, including its irregular sampling, multivariate nature, and morphological diversity. In particular, we investigate the extent to which physically motivated augmentations, operating exclusively on real observations, can mitigate the data limitation of deep architectures and narrow the performance gap.

## 2. DATASET

We use the ZTF StarEmbed dataset as the benchmark throughout our study. This dataset is a subset of the ZTF astronomical survey [1], conducted at the Palomar Observatory, and contains light curves of stars belonging to seven distinct classes: EW and EA, which are eclipsing binaries; RRab, RRc, and RRd, which are subtypes of RR Lyrae pulsating stars; RS CVn, which are binary star systems; and LPV, which are long-period variable stars [13]. Originally introduced in [14], the dataset was designed as a benchmark for comparing diferent classifiers. It comprises roughly 40,000 objects and about 35 million individual observations, with a median light-curve length of approximately 770 data points.

The light curves are recorded in three ZTF photometric bands, namely �, �, and �. Each observation is represented as a tuple (�, mag, mag\_err, passband), where � denotes the observation time in Modified Julian Date, mag is the stellar magnitude, mag\_err is the associated measurement uncertainty, and passband specifies the filter used. The dataset is stored in the nested-pandas format, in which each light curve is organized as a nested table within a row of the main DataFrame. This structure enables eficient vectorized operations across the entire dataset. We split the data into training, validation, and test sets in a 7:1:2 ratio, while preserving the class proportions in each subset.

The dataset sufers from a strong class imbalance, with the number of objects per class detailed in Table 1. This imbalance poses additional challenges for model training and necessitates a careful selection of evaluation metrics. Representative light curves from each class are shown in Fig. 1.

Table 1. Number of objects per class in the ZTF StarEmbed dataset.
<table><tr><td>Class</td><td>EW</td><td>RRc</td><td>EA</td><td>RRab</td><td>RS CVn</td><td>LPV</td><td>RRd</td><td>Total</td></tr><tr><td>Number of objects</td><td>27,949</td><td>4,792</td><td>4,251</td><td>2,068</td><td>1,401</td><td>457</td><td>439</td><td>41,357</td></tr></table>

## 3. EVALUATION METRICS

To evaluate model performance and enable fair comparisons, we employ four metrics: accuracy, balanced accuracy, macro-averaged F1, and the macro-averaged area under the receiver operating characteristic curve (ROC-AUC).

The choice of these metrics is primarily driven by the class imbalance present in the dataset. The largest class, EW, contains roughly 64 times more objects than the smallest class, RRd. Under such conditions, plain accuracy becomes biased toward the majority classes, as a classifier that disregards rare classes can still achieve a high accuracy score. For this reason, we report accuracy only as a baseline reference. Our primary evaluation relies on macro-averaged metrics, which treat all classes equally regardless of their size. This ensures that performance on rare classes such as LPV and RRd contributes to the final score on the same footing as performance on the dominant EW class.

![](images/1786ce5fc56d51e00bb5e95bdfcf1ef51ecb8127ea817b391d642c199d8126f2.jpg)  
Figure 1. Examples of light curves from the ZTF StarEmbed dataset for the EW, EA, RRc, and LPV classes in the � and � passbands.

## 4. FEATURE-BASED APPROACH

We first explored an approach based on physically informative features extracted from light curves. For each photometric passband, we computed 25 features using the light-curve-python library [15]. These features were then concatenated and used to train three classifiers: Logistic Regression in the one-vs-rest regime, CatBoost [16], and a Multilayer Perceptron.

For each light curve, we independently computed a set of 25 statistical and periodogram-based features for each photometric band. The full list of features includes Amplitude, Anderson-DarlingNormal, BeyondNStd, Cusum, Eta, EtaE, ExcessVariance, InterPercentileRange, Kurtosis, MaximumSlope, Mean, MeanVariance, Median, MedianAbsoluteDeviation, MedianBufferRangePercentage, PercentAmplitude, PercentDiferenceMagnitudePercentile, ReducedChi2, Roms, Skew, StandardDeviation, StetsonK, WeightedMean, and Periodogram, which yields two outputs. Detailed descriptions of these features are available on the light-curve-python library website<sup>1</sup>. Figure 2 presents the nine features identified as the most important for classification by the CatBoost model.

![](images/c79dab5f85d22868d534c28a43e9f30bc6de0a29ef9ce76c869e67fcaaa18eca.jpg)  
Figure 2. Top-9 features with the highest importance scores, as determined by the CatBoost classifier trained on light-curve features.

## 5. DEEP LEARNING APPROACH

The second approach relies on training deep neural networks directly on raw light curves, omitting a separate feature extraction stage. Both architectures considered in this work share a common input representation. Each light curve is converted into a sequence of up to 512 observations, with each observation described by four features: the time in days since the first observation, the logarithm of the passband wavelength, the normalized flux, and the normalized flux uncertainty, the latter obtained by dividing the measurement uncertainty by the standard deviation of the flux. Light curves shorter than 512 observations are zero-padded, and a corresponding padding mask is passed to the network to exclude padded positions from computations.

Both networks are trained using the AdamW optimizer with a class-weighted cross-entropy loss, which helps mitigate the severe class imbalance in the dataset. Each of the configurations was additionally trained with each of the three light-curve augmentation strategies, described in detail in Section 5.3.

![](images/7636213826a95944f26ccc05615feae26adda637f2f462658ac917187ffdd9d8.jpg)  
Figure 3. The best-performing recurrent architecture: input projection, two bidirectional LSTM layers with hidden size 256, masked aggregation, and a classification head.

## 5.1. LSTM

For the recurrent approach, we explored both unidirectional and bidirectional networks with varying depths and classification heads. Bidirectional processing is appropriate in this setting because classification considers the entire light curve, so information from future time steps does not violate the problem formulation. The simplest configuration we tested consists of a single unidirectional LSTM [17] layer with a hidden size of 128, whose final hidden state is passed to a linear classifier. In deeper configurations, we aggregate the hidden states produced by the network using multiple pooling strategies: the mean, maximum, and standard deviation over all time steps, together with the final hidden state vectors. These aggregated representations are concatenated and fed into the classification head. Several head variants were considered, including a head with Gaussian Error Linear Unit (GELU) activation, a wide head, and a combination of both.

The best-performing architecture is illustrated in Figure 3. It comprises two bidirectional LSTM (BiLSTM) layers with a hidden size of 256, followed by masked aggregation combining the mean, maximum, standard deviation, and final state, and a classification head. The model contains approximately 3.7 million parameters. Training was regularized with early stopping on the validation set.

![](images/41a36f75e5e59ace6d3baaacf7a7745e83b67a79420de8b617193e1b858459d2.jpg)  
Figure 4. The deep Transformer architecture: input projection, sinusoidal positional encoding, four pre-norm encoder layers, masked aggregation, and an MLP classification head.

## 5.2. Transformer

We also examined a Transformer [18] encoder applied to the same input sequences. The encoder is configured with a model dimension of 128, four attention heads, a feed-forward dimension of 512, a pre-normalization layer arrangement, and fixed sinusoidal positional encodings added to the input projection. The simplest configuration consists of two encoder layers with mean pooling and a linear classification head, containing only about 0.27 million parameters. The deeper configuration, shown in Figure 4, employs four encoder layers and aggregates the encoder outputs via masked mean, maximum, and standard deviation pooling, followed by an expanding MLP head with GELU activation. During architecture selection, we compared narrow and wide head variants with both Rectified Linear Unit (ReLU) and GELU activations.

The Transformer was trained with a learning-rate warmup followed by a cosine decay schedule.

## 5.3. Light-Curve Augmentations

Since the input length is limited to 512 observations while the median light curve in the dataset contains about 770, most light curves cannot be fed to the networks in full, and some subset of observations must be selected in any case. This constraint makes augmentation a natural choice in our setting. Instead of fixing a single deterministic truncation, we randomly draw the subset at each training epoch. No artificial data points are generated. Every augmented light curve consists solely of real observations of the same object and preserves the irregular sampling inherent to the survey. The augmentations therefore do not distort the physical content of the data while substantially increasing its efective diversity. Three augmentation strategies were considered and are illustrated in Figure 5.

![](images/b512a4dde8784b36f90337ca6ef7d380efda6dcd83a291288fb6bebd4c2e7424.jpg)  
Figure 5. The three augmentations applied to the same light curve of 2,982 observations; the 512 selected observations are shown in blue, the rest in gray.

Random subsampling. 512 observations are drawn uniformly at random from the entire light curve and kept in chronological order. This preserves the full time span of the curve at the cost of a sparser cadence.

Random bins. 24-hour bins are selected at random, and all observations within each chosen bin are retained until 512 points are collected; any excess points from the final bin are discarded. This strategy preserves the dense intra-night structure of the observations.

Random window. 512 consecutive observations are kept starting from a random position, which emulates a shorter observing campaign.

## 6. MODEL COMPARISON

Table 2 presents a comprehensive comparison of all evaluated models on the held-out test set, encompassing both feature-based classical machine learning approaches and deep neural networks trained directly on raw light curve observations. The results reveal a clear and consistent superiority of the feature-based pipeline across virtually all evaluation metrics, underscoring the continued value of domain-specific feature engineering in astronomical classification tasks.

Among the feature-based models, CatBoost achieves the highest accuracy of 0.907 and, together with the MLP, the top macro-averaged ROC-AUC of 0.981. The MLP performs comparably with 0.903 accuracy, indicating that the extracted features form a rich and nearly linearly separable representation. Interestingly, logistic regression attains the highest balanced accuracy of 0.801, outperforming both CatBoost and the MLP. We attribute this to its linear nature, which reduces overfitting to majority classes and better preserves performance across the imbalanced class distribution.

Deep learning models trained from scratch on raw light curves exhibit substantially inferior performance relative to feature-based approaches. The best unaugmented configuration, the LSTM, achieves only 0.609 accuracy, 0.508 macro-averaged F1, and 0.914 macro-averaged ROC-AUC. This substantial performance gap underscores the inherent dificulty of learning robust representations directly from irregularly sampled, multivariate time series, corroborating our hypothesis that deep architectures are data-limited in this regime.

However, physically motivated augmentations considerably reduce this performance deficit. For the LSTM-like models, random windows and random daily bins emerge as the most efective augmentation strategies across the diferent configurations. In particular, the best-performing configuration raises accuracy from 0.609 to 0.763, macro-averaged F1 from 0.508 to 0.625, and macro-averaged ROC-AUC from 0.914 to 0.948.

For the Transformer-like models, random daily bins gives the best results in most configurations. Overall, the best-performing configuration improves accuracy from 0.588 to 0.692, macro-averaged F1 from 0.488 to 0.556, and macro-averaged ROC-AUC from 0.894 to 0.923.

Both random bins and random window improve balanced accuracy in nearly all configurations, whereas random subsampling is less stable and degrades performance in several cases. Which of the two strategies performs best varies from configuration to configuration without a consistent pattern, and without further investigation these diferences cannot be separated from run-to-run variability.

Table 2. Test-set results of all experiments (standard error as subscript). Best values in bold.
<table><tr><td>Architecture</td><td>Acc.</td><td>Bal. Acc.</td><td>Macro F1</td><td>ROC-AUC</td></tr><tr><td>Extracted Features (Logistic Regression)</td><td>0.787.005</td><td>0.801.011</td><td>0.661.008</td><td>0.955.005</td></tr><tr><td>Extracted Features (CatBoost)</td><td>0.907.004</td><td>0.742.009</td><td>0.771.009</td><td>0.981.003</td></tr><tr><td>Extracted Features (MLP)</td><td>0.903.003</td><td>0.762.010</td><td>0.779.009</td><td>0.981.004</td></tr><tr><td>LSTM (Linear)</td><td>0.609.005</td><td>0.653.011</td><td>0.508.008</td><td>0.914.003</td></tr><tr><td>LSTM (Linear), Rnd. subs.</td><td>0.437.006</td><td>0.505.009</td><td>0.336.006</td><td>0.833.005</td></tr><tr><td>LSTM (Linear), Rnd. bins</td><td>0.663.005</td><td>0.664.011</td><td>0.508.007</td><td>0.921.004</td></tr><tr><td>LSTM (Linear), Rnd. win.</td><td>0.667.005</td><td>0.692.010</td><td>0.544.007</td><td>0.929.003</td></tr><tr><td>Deep BiLSTM</td><td>0.521.005</td><td>0.611.011</td><td>0.416.007</td><td>0.901.003</td></tr><tr><td>Deep BiLSTM, Rnd. subs.</td><td>0.577.005</td><td>0.634.011</td><td>0.460.007</td><td>0.905.004</td></tr><tr><td>Deep BiLSTM, Rnd. bins</td><td>0.759.005</td><td>0.729.010</td><td>0.614.008</td><td>0.945.003</td></tr><tr><td>Deep BiLSTM, Rnd. win.</td><td>0.704.005</td><td>0.721.010</td><td>0.580.008</td><td>0.942.003</td></tr><tr><td>Deep BiLSTM (GELU)</td><td>0.538.005</td><td>0.579.011</td><td>0.438.007</td><td>0.884.004</td></tr><tr><td>Deep BiLSTM (GELU), Rnd. subs.</td><td>0.647.005</td><td>0.708.011</td><td>0.541.007</td><td>0.923.004</td></tr><tr><td>Deep BiLSTM (GELU), Rnd. bins</td><td>0.716.005</td><td>0.691.010</td><td>0.579.007</td><td>0.932.003</td></tr><tr><td>Deep BiLSTM (GELU), Rnd. win.</td><td>0.763.005</td><td>0.733.010</td><td>0.625.008</td><td>0.948.003</td></tr><tr><td>Deep BiLSTM (wide head)</td><td>0.587.005</td><td>0.615.009</td><td>0.474.007</td><td>0.900.004</td></tr><tr><td>Deep BiLSTM (wide head), Rnd. subs.</td><td>0.676.005</td><td>0.676.010</td><td>0.538.008</td><td>0.923.004</td></tr><tr><td>Deep BiLSTM (wide head), Rnd. bins</td><td>0.554.005</td><td>0.631.010</td><td>0.445.007</td><td>0.893.004</td></tr><tr><td>Deep BiLSTM (wide head), Rnd. win.</td><td>0.667.005</td><td>0.703.010</td><td>0.545.007</td><td>0.932.004</td></tr><tr><td>Deep BiLSTM (GELU, wide head)</td><td>0.557.005</td><td>0.604.009</td><td>0.450.007</td><td>0.893.004</td></tr><tr><td>Deep BiLSTM (GELU, wide), Rnd. subs.</td><td>0.644.005</td><td>0.652.011</td><td>0.503.008</td><td>0.913.004</td></tr><tr><td>Deep BiLSTM I (GELU, wide), Rnd. bins</td><td>0.740.005</td><td>0.724.011</td><td>0.601.008</td><td>0.938.004</td></tr><tr><td>Deep BiLSTM (GELU, wide), Rnd. win.</td><td>0.569.005</td><td>0.662.010</td><td>0.481.007</td><td>0.915.004</td></tr><tr><td>Transformer (Linear)</td><td>0.529.005</td><td>0.583.012</td><td>0.434.008</td><td>0.867.005</td></tr><tr><td>Transformer (Linear), Rnd. subs.</td><td>0.609.005</td><td>0.577.011</td><td>0.453.007</td><td>0.870.005</td></tr><tr><td>Transformer (Linear), Rnd. bins</td><td>0.666.005</td><td>0.621.010</td><td>0.504.008</td><td>0.902.004</td></tr><tr><td>Transformer (Linear), Rnd. win.</td><td>0.559.005</td><td>0.621.011</td><td>0.467.007</td><td>0.890.005</td></tr><tr><td>Deep Transf.</td><td>0.649.005</td><td>0.588.011</td><td>0.492.008</td><td>0.887.004</td></tr><tr><td>Deep Transf., Rnd. subs.</td><td>0.619.005</td><td>0.647.011</td><td>0.508.008</td><td>0.899.004</td></tr><tr><td>Deep Transf., Rnd. bins</td><td>0.668.005</td><td>0.661.011</td><td>0.539.008</td><td>0.914.004</td></tr><tr><td>Deep Transf., Rnd. win.</td><td>0.628.005</td><td>0.639.012</td><td>0.516.007</td><td>0.905.004</td></tr><tr><td>Deep Transf. (GELU)</td><td>0.641.005</td><td>0.646.011</td><td>0.507.008</td><td>0.904.004</td></tr><tr><td>Deep Transf. (GELU), Rnd. subs. Deep Transf.</td><td>0.630.005</td><td>0.636.011</td><td>0.515.008</td><td>0.907.004</td></tr><tr><td>. (GELU), Rnd. bins . (GELU), Rnd. win.</td><td>0.632.005</td><td>0.650.011</td><td>0.515.007</td><td>0.917.003</td></tr><tr><td>Deep Transf. Deep Transf.</td><td>0.684.005</td><td>0.674.011</td><td>0.543.007</td><td>0.923.003</td></tr><tr><td>. (wide head)</td><td>0.588.006</td><td>0.621.011</td><td>0.488.007</td><td>0.894.004</td></tr><tr><td>Deep Transf. (wide head), Rnd. subs.</td><td>0.606.005</td><td>0.676.011</td><td>0.510.007</td><td>0.907.004</td></tr><tr><td>Deep Transf. (wide head), Rnd. bins</td><td>0.692.005</td><td>0.678.011</td><td>0.556.008</td><td>0.917.004</td></tr><tr><td>Deep Transf. (wide head), Rnd. win.</td><td>0.676.005</td><td>0.676.011</td><td>0.531.007</td><td>0.914.004</td></tr><tr><td>Deep Transf. (GELU, wide)</td><td>0.629.005</td><td>0.622.011</td><td>0.485.007</td><td>0.894.005</td></tr><tr><td>Deep Transf. (GELU, wide), Rnd. subs.</td><td>0.607.005</td><td>0.655.011</td><td>0.495.007</td><td>0.904.004</td></tr><tr><td>Deep Transf. (GELU, wide), Rnd. bins</td><td>0.655.005</td><td>0.670.011</td><td>0.530.007</td><td>0.914.004</td></tr><tr><td>Deep Transf. (GELU, wide), Rnd. win.</td><td>0.661.005</td><td>0.647.010</td><td>0.514.009</td><td>0.912.004</td></tr></table>

## 7. CONCLUSIONS

This work provides the first comprehensive quantitative comparison of feature-based machine learning and end-to-end deep learning on the ZTF StarEmbed dataset. We rigorously measure the performance gap between these paradigms, establishing a clear baseline for future research.

We show that a range of LSTM and Transformer architectures trained on raw light curves consistently underperform models based on physics-motivated features. Data augmentation substantially improves their performance, suggesting that limited labeled data contributes to this gap, although it does not fully explain it. With about 40,000 objects, deep learning models still struggle to generalize, likely due in part to the multivariate, irregular, and morphologically diverse nature of light curves. These findings motivate further investigation of scaling labeled data and self-supervised pretraining on large unlabeled surveys.

We also introduce three novel augmentations, Random Window, Random Bins, and Random Subsampling, that use only real observations. Unlike synthetic methods, they preserve physical integrity while increasing diversity through the natural irregular sampling of astronomical surveys.

We systematically evaluate masked aggregation strategies combining mean, maximum, standard deviation, and final hidden states, instead of simple last-state pooling. This adaptation to irregular, multi-band light curves yields consistent performance gains across both architectures.

Finally, we provide evidence-based justification for moving from task-specific supervised learning to self-supervised foundation models pretrained on massive unlabeled surveys. Our results indicate that classical feature-based methods are approaching saturation, while deep learning remains constrained primarily by data availability, guiding future research directions in astro-informatics.

## ACKNOWLEDGMENTS

This research was supported in part through computational resources of high-performance computing (HPC) facilities at HSE University [19].

## FUNDING

The article was prepared within the framework of the project “Mirror Laboratories” HSE University. Support was provided by Schmidt Sciences, LLC, for K. Malanchev.

The authors of this work declare that they have no conflicts of interest.

[1] E. C. Bellm, S. R. Kulkarni, M. J. Graham, et al., Publ. Astron. Soc. Pac. 131, 018002 (2019). https://doi.org/10.1088/1538-3873/aaecbe

[2] Z. Ivezi´c, S. M. Kahn, J. A. Tyson, et al., Astrophys. J. <sup>ˇ</sup> 873, 111 (2019). https://doi.org/10.3847/1538-4357/ab042c

[3] Demianenko, M., Supernova Light Curves Approximation based on Neural Network Models, in Journal of Physics Conference Series, 2023, vol. 2438, no. 1, Art. no. 012128. doi:10.1088/1742-6596/2438/1/012128

[4] Semenikhin, T. A., Kornilov, M. V., Pruzhinskaya, M. V., Krushinsky, V. V., Malanchev, K. L., and Dodin, A. V., Supernova scores for active anomaly detection, Astronomy and Computing, vol. 56, Art. no. 101126, Elsevier, 2026. doi:10.1016/j.ascom.2026.101126

[5] Lavrukhina, A., Malanchev, K., and Kornilov, M. V., Automatic Detection of Plateau Phases in Light Curves of Variable Stars, Research Notes of the American Astronomical Society, vol. 7, no. 9, Art. no. 199, IOP, 2023. doi:10.3847/2515-5172/acfd2d

[6] Demianenko, M., Optical Variability of "Light-weight" Supermassive Black Holes at a Few Percent Level from ZTF Forced-Photometry Light Curves, in Astronomical Data Analysis Software and Systems XXXI, 2024, vol. 535, p. 283. doi:10.48550/arXiv.2201.03712

[7] Demianenko, M., Broad line region echo from highly accreting intermediate-mass black hole candidate SDSS J144850.08+160803.1: First probe of intra-night variability and reverberation mapping, Astronomy and Astrophysics, vol. 702, Art. no. A202, EDP, 2025. doi:10.1051/0004-6361/202555139

[8] Volkov, M., Demianenko, M., Matveev, D., Grishin, K. A., and Chilingarian, I. V., Website with interactive visualization of multivariate astronomical time series, Communications of the Byurakan Astrophysical Observatory, vol. 72, no. 2, pp. 334–340, 2025. doi:10.52526/25792776-25.72.2-334

[9] Demianenko, M., Malanchev, K., Samorodova, E., Sysak, M., Shiriaev, A., Derkach, D. & Hushchyn, M. Understanding of the properties of neural network approaches for transient light curve approximations. Astronomy & Astrophysics. 677 pp. A16 (2023)

[10] Dobryakov, S., Malanchev, K., Derkach, D., and Hushchyn, M., Photometric data-driven classification of Type Ia supernovae in the open Supernova Catalog, Astronomy and Computing, vol. 35, Art. no. 100451, Elsevier, (2021). doi:10.1016/j.ascom.2021.100451

[11] X. Zuo, Y. Tao, Y. Huang, et al., Astron. J. 171, 10 (2026). https://doi.org/10.3847/1538-3881/ae1467

[12] M. Goswami, K. Szafer, A. Choudhry, Y. Cai, S. Li, and A. Dubrawski, in Proc. 41st Int. Conf. on Machine Learning (ICML 2024), PMLR 235, 16115 (2024). https://doi.org/10.48550/arXiv.2402.03885

[13] X. Chen, S. Wang, L. Deng, et al., Astrophys. J. Suppl. Ser. 249, 18 (2020). https://doi.org/10.3847/1538-4365/ab9cae

[14] W. Li, H.-Y. Chen, N. Rehemtulla, et al., arXiv:2510.06200 [astro-ph.SR] (2025). https://doi.org/10.48550/arXiv.2510.06200

[15] K. L. Malanchev, M. V. Pruzhinskaya, V. S. Korolev, et al., Mon. Not. R. Astron. Soc. 502, 5147 (2021). https://doi.org/10.1093/mnras/stab316

[16] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, in Advances in Neural Information Processing Systems 31 (NeurIPS 2018) (Curran Associates, 2018), p. 6639. https://doi.org/10.48550/arXiv.1706.09516

[17] S. Hochreiter and J. Schmidhuber, Neural Comput. 9, 1735 (1997). https://doi.org/10.1162/neco.1997.9.8.1735

[18] A. Vaswani, N. Shazeer, N. Parmar, et al., in Advances in Neural Information Processing Systems 30 (NIPS 2017) (Curran Associates, 2017), p. 5998. https://doi.org/10.48550/arXiv.1706.03762

[19] Kostenetskiy, P., Kozyrev, V., Chulkevich, R., Salikova, M. & Mishenin, R. High-Performance Computing at HSE University. International Conference On Parallel Computational Technologies. pp. 17-29 (2025)